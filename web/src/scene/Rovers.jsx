import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '../globe/PlaneHtml.jsx'
import * as THREE from 'three'
import { appStore, simStore, useSim, roverPoses } from '../state/store.js'
import { placement } from '../lib/placement.js'
import { forEachTrailPoint, sampleGround, sampleGroundNear, TRAIL_POINTS } from '../state/simState.js'
import { FatPolyline } from './fatLine.js'
import { glowTexture, warningTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'
import { HAZARD, instrumentStyle, kindStyle } from '../lib/theme.js'
import { InstrumentIcon } from '../components/SampleBits.jsx'
import { hardwareMaterials, wreckMaterials } from './hardware.js'
import { MAST_TOP, WHEELS, WHEEL_R, roverParts } from './roverModel.js'
import { RoverDust, dustSources } from './RoverDust.jsx'

const MODEL_SCALE = 1.5 // a little larger than life so rovers read on a projector

const geo = {
  halo: new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
  idRing: new THREE.RingGeometry(1.22, 1.32, 64).rotateX(-Math.PI / 2),
  goalRing: new THREE.RingGeometry(0.72, 0.92, 48).rotateX(-Math.PI / 2),
  goalDot: new THREE.CircleGeometry(0.2, 20).rotateX(-Math.PI / 2),
  selRing: new THREE.RingGeometry(1.85, 1.98, 64).rotateX(-Math.PI / 2),
  faultRing: new THREE.RingGeometry(1.45, 1.9, 48).rotateX(-Math.PI / 2),
}
const selMat = new THREE.MeshBasicMaterial({ color: '#e8eef5', transparent: true, opacity: 0.8, depthWrite: false })
const faultMats = {
  ring: new THREE.MeshBasicMaterial({ color: HAZARD, transparent: true, depthWrite: false }),
  glow: new THREE.MeshBasicMaterial({ color: HAZARD, map: glowTexture(), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
  sign: new THREE.SpriteMaterial({ map: warningTexture(), transparent: true, depthWrite: false, depthTest: false }),
}

// Wheels, LiDAR head and beacon move; everything else is merged per material (roverModel.js).
// v9: a wreck is drawn with darkened, dull materials and no lights.
function RoverModel({ color, wheels, lidar, beacon, wrecked }) {
  const P = roverParts()
  const M = wrecked ? wreckMaterials() : hardwareMaterials()
  // The identity colour as an accent (stripes, mast band, nose), a little self-lit so it
  // reads on a washed-out projector.
  const accent = useMemo(() => {
    const dim = new THREE.Color(color).multiplyScalar(0.3)
    const m = wrecked ? new THREE.MeshStandardMaterial({ color: dim, roughness: 0.9, metalness: 0.1 }) : new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.55, roughness: 0.4, metalness: 0.1 })
    m.userData.realistic = true
    return m
  }, [color, wrecked])
  const laserMat = useMemo(() => new THREE.MeshBasicMaterial({ color }), [color])
  const beaconMat = useMemo(() => new THREE.SpriteMaterial({ map: glowTexture(), color, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: false }), [color])
  useEffect(() => () => (accent.dispose(), laserMat.dispose(), beaconMat.dispose()), [accent, laserMat, beaconMat])
  return (
    <group scale={MODEL_SCALE}>
      <mesh geometry={P.paint} material={M.paint} castShadow receiveShadow />
      <mesh geometry={P.gold} material={M.gold} castShadow />
      <mesh geometry={P.solar} material={M.solar} castShadow receiveShadow />
      <mesh geometry={P.dark} material={M.dark} castShadow />
      <mesh geometry={P.metal} material={M.metal} />
      <mesh geometry={P.glass} material={M.glass} />
      <mesh geometry={P.lamp} material={M.lamp} />
      <mesh geometry={P.accent} material={accent} />
      <group ref={lidar} position={P.lidarAt}>
        <mesh geometry={P.laser} material={laserMat} />
      </group>
      {WHEELS.map((w, i) => (
        <group key={i} position={w} ref={(el) => (wheels.current[i] = el)}>
          <mesh geometry={P.tyre} material={M.tyre} castShadow />
          <mesh geometry={P.hub} material={M.metal} />
        </group>
      ))}
      <sprite ref={beacon} material={beaconMat} position={[0.205, MAST_TOP, 0]} scale={0.011} renderOrder={5} visible={!wrecked} />
    </group>
  )
}

const _up = new THREE.Vector3()
const _fwd = new THREE.Vector3()
const _side = new THREE.Vector3()
const _basis = new THREE.Matrix4()
const _q = new THREE.Quaternion()
const _qa = new THREE.Quaternion()
const _e = new THREE.Euler()
const Y = new THREE.Vector3(0, 1, 0)
const D2R = Math.PI / 180

function groundNormal(T, x, y, out) {
  const d = 0.6
  const e = sampleGround(T, x + d, y)
  const w = sampleGround(T, x - d, y)
  const n = sampleGround(T, x, y + d)
  const s = sampleGround(T, x, y - d)
  const hx = e !== null && w !== null ? (e - w) / (2 * d) : 0
  const hy = n !== null && s !== null ? (n - s) / (2 * d) : 0
  return out.set(-hx, 1, hy).normalize()
}

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a))

// v9: rovers whose LiDAR head stops (no power to the mast, or parked for good).
const STILL = new Set(['fault', 'disabled', 'wrecked', 'towed', 'repairing'])
// v9: a chip on the name tag for the new states (the distress tag covers "disabled").
const TAG_CHIP = {
  rescuing: { label: 'Helping', kind: 'help' },
  towing: { label: 'Towing', kind: 'tow' },
  towed: { label: 'In tow', kind: 'tow' },
  repairing: { label: 'Repair', kind: 'repair' },
  wrecked: { label: 'Wreck', kind: 'wreck' },
}

function Rover({ meta, selected, onSelect }) {
  const outer = useRef()
  const inner = useRef()
  const pose = useRef(null)
  const fault = useRef()
  const sign = useRef()
  const sel = useRef()
  const wheels = useRef([])
  const lidar = useRef()
  const beacon = useRef()
  const state = useSim((s) => s.rovers[meta.id]?.state)
  const faulted = state === 'fault'
  // v9: a wreck is darkened; the name tag says what is going on.
  const wrecked = useSim((s) => s.rovers[meta.id]?.state === 'wrecked' || s.rovers[meta.id]?.health?.status === 'wrecked')
  const tagChip = TAG_CHIP[state] || null
  const asking = useSim((s) => s.requests.some((q) => q.rover === meta.id)) // v3: waiting for the operator
  // drei <Html> picks its DOM container once R3F has connected its events; mounting it
  // earlier makes it unmount and remount its own React root (a React error in dev).
  const labelsReady = useThree((s) => !!s.events.connected)
  const markMats = useMemo(
    () => ({
      halo: new THREE.MeshBasicMaterial({ color: meta.color, map: glowTexture(), transparent: true, opacity: 0.2, depthWrite: false, blending: THREE.AdditiveBlending }),
      ring: new THREE.MeshBasicMaterial({ color: meta.color, transparent: true, opacity: 0.8, depthWrite: false }),
    }),
    [meta.color],
  )
  useEffect(() => () => (markMats.halo.dispose(), markMats.ring.dispose()), [markMats])
  useEffect(() => () => (roverPoses.delete(meta.id), dustSources.delete(meta.id)), [meta.id])
  useEffect(() => () => (document.body.style.cursor = ''), [])

  // Click (not drag) on the rover or its label selects it: sensor panel and follow.
  // While a hazard is being placed the 3D view handles clicks (a motor fault picks the
  // rover under the cursor there), so the rover itself does nothing.
  const select = (e) => {
    e.stopPropagation()
    if (e.delta > 6 || appStore.get().placing) return
    onSelect?.(meta.id)
  }
  const tagClick = () => {
    const placing = appStore.get().placing
    if (placing?.mode === 'rover') placement.commit?.({ rover: meta.id })
    else if (!placing) onSelect?.(meta.id)
  }

  useFrame(({ clock, camera, size }, dt) => {
    const s = simStore.get()
    const r = s.rovers[meta.id]
    const o = outer.current
    if (!o) return
    if (!r) {
      o.visible = false
      roverPoses.delete(meta.id)
      dustSources.delete(meta.id)
      return
    }
    let p = pose.current
    if (!p || Math.hypot(r.x - p.x, r.y - p.y) > 8) {
      p = pose.current = { x: r.x, y: r.y, yaw: r.yaw, gz: null, spin: 0, v: 0, lidar: 0 }
    }
    // Ticks arrive at ~5 Hz; ease towards the latest one so motion stays smooth.
    const step = Math.min(dt, 0.1)
    const a = 1 - Math.exp(-step * 8)
    const ox = p.x
    const oy = p.y
    const oyaw = p.yaw
    p.x += (r.x - p.x) * a
    p.y += (r.y - p.y) * a
    p.yaw += wrap(r.yaw - p.yaw) * a
    const g = sampleGround(s.terrain, p.x, p.y)
    const gz = g ?? r.z
    p.gz = p.gz === null ? gz : p.gz + (gz - p.gz) * Math.min(1, a * 1.5)
    o.visible = true
    o.position.set(p.x, p.gz, -p.y)

    // Attitude: the IMU's pitch and roll when the sim sends them (nose up and right side
    // down positive), else the slope of the known ground.
    const sn = r.sensors
    if (sn && Number.isFinite(sn.pitch) && Number.isFinite(sn.roll)) {
      _q.setFromAxisAngle(Y, p.yaw)
      _qa.setFromEuler(_e.set(sn.roll * D2R, 0, sn.pitch * D2R, 'ZXY'))
      _q.multiply(_qa)
    } else {
      groundNormal(s.terrain, p.x, p.y, _up)
      _fwd.set(Math.cos(p.yaw), 0, -Math.sin(p.yaw))
      _fwd.addScaledVector(_up, -_fwd.dot(_up)).normalize()
      _side.crossVectors(_fwd, _up)
      _basis.makeBasis(_fwd, _up, _side)
      _q.setFromRotationMatrix(_basis)
    }
    if (inner.current) inner.current.quaternion.slerp(_q, Math.min(1, a * 1.2))
    roverPoses.set(meta.id, o.position)

    // Wheels: surface speed from the encoders when known (they spin faster than the rover
    // moves when it slips), signed by the direction of travel; skid steer turns spin the
    // two sides against each other.
    const moved = Math.hypot(p.x - ox, p.y - oy)
    const along = (p.x - ox) * Math.cos(p.yaw) + (p.y - oy) * Math.sin(p.yaw)
    const gs = moved / Math.max(step, 1e-3)
    const ws = sn && Number.isFinite(sn.wheel_speed) && r.state !== 'fault' && r.state !== 'towed' ? sn.wheel_speed : gs
    const dir = along < -1e-4 ? -1 : 1
    p.v += (ws * dir - p.v) * Math.min(1, step * 6)
    const turn = wrap(p.yaw - oyaw) / Math.max(step, 1e-3)
    const R = WHEEL_R * MODEL_SCALE
    const half = 0.4 * MODEL_SCALE
    if (!reducedMotion) {
      for (let i = 0; i < 4; i++) {
        const w = wheels.current[i]
        if (!w) continue
        const right = WHEELS[i][2] > 0
        const v = p.v + (right ? 1 : -1) * turn * half
        w.rotation.z -= (v / R) * step
      }
      if (lidar.current && !STILL.has(r.state) && !(r.health?.parts?.lidar < 3)) lidar.current.rotation.y += step * 9
    }
    // Dust from the rear wheels (RoverDust reads this).
    let src = dustSources.get(meta.id)
    if (!src) dustSources.set(meta.id, (src = { x: 0, y: 0, z: 0, yaw: 0, speed: 0, color: meta.color }))
    src.x = p.x
    src.y = p.y
    src.z = p.gz
    src.yaw = p.yaw
    src.speed = Math.max(Math.abs(p.v), gs)

    // Motor fault: a blinking crimson ring and a warning sign over the parked rover.
    const isFault = r.state === 'fault'
    const t = clock.elapsedTime
    const blink = reducedMotion ? 1 : 0.5 + 0.5 * Math.sign(Math.sin(t * 6.5))
    if (beacon.current) beacon.current.material.opacity = isFault || r.state === 'disabled' ? 0.25 + 0.5 * blink : 0.85
    if (fault.current) {
      fault.current.visible = isFault
      if (isFault) {
        faultMats.ring.opacity = 0.25 + 0.75 * blink
        faultMats.glow.opacity = 0.2 + 0.6 * blink
        const k = reducedMotion ? 1 : 1 + 0.12 * Math.sin(t * 3)
        fault.current.scale.set(k, 1, k)
      }
    }
    if (sign.current) {
      sign.current.visible = isFault
      if (isFault) {
        faultMats.sign.opacity = 0.35 + 0.65 * blink
        // Keep the sign about 26 px tall and just above the (DOM) name label at any zoom.
        const mPerPx = (2 * camera.position.distanceTo(o.position) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / Math.max(1, size.height)
        const k = Math.max(1.1, Math.min(4, 26 * mPerPx))
        sign.current.scale.setScalar(k)
        sign.current.position.y = 2.5 + 14 * mPerPx + k / 2 + 3 * mPerPx
      }
    }
    if (sel.current && sel.current.visible && !reducedMotion) {
      const k = 1 + 0.05 * Math.sin(t * 2.4)
      sel.current.scale.set(k, 1, k)
    }
  })

  return (
    <group
      ref={outer}
      visible={false}
      onClick={select}
      onPointerOver={(e) => {
        e.stopPropagation()
        if (!appStore.get().placing) document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => (document.body.style.cursor = '')}
    >
      <group ref={inner}>
        <RoverModel color={meta.color} wheels={wheels} lidar={lidar} beacon={beacon} wrecked={wrecked} />
        <mesh geometry={geo.halo} material={markMats.halo} scale={3.4} position={[0, 0.05, 0]} renderOrder={3} />
        <mesh geometry={geo.idRing} material={markMats.ring} position={[0, 0.07, 0]} renderOrder={3} />
      </group>
      <mesh ref={sel} geometry={geo.selRing} material={selMat} position={[0, 0.1, 0]} visible={!!selected} renderOrder={3} />
      <group ref={fault} visible={false}>
        <mesh geometry={geo.faultRing} material={faultMats.ring} position={[0, 0.12, 0]} renderOrder={4} />
        <mesh geometry={geo.halo} material={faultMats.glow} scale={6} position={[0, 0.08, 0]} renderOrder={4} />
      </group>
      <sprite ref={sign} material={faultMats.sign} position={[0, 3.55, 0]} scale={1.15} visible={false} renderOrder={10} />
      {labelsReady && (
        <Html position={[0, 2.5, 0]} center zIndexRange={[12, 0]} style={{ pointerEvents: 'none' }}>
          <button
            type="button"
            className="rover-tag"
            data-selected={selected || undefined}
            data-fault={faulted || undefined}
            data-alert={state === 'disabled' || undefined}
            data-wreck={state === 'wrecked' || undefined}
            data-ask={(asking && !faulted) || undefined}
            onClick={tagClick}
            title={`Show ${meta.name}'s sensors`}
          >
            <span className="dot" style={{ background: meta.color }} />
            {meta.name}
            <TagSampling roverId={meta.id} />
            {faulted && <span className="rover-tag-fault">Fault</span>}
            {tagChip && (
              <span className="rover-tag-chip" data-kind={tagChip.kind}>
                {tagChip.label}
              </span>
            )}
            {asking && !faulted && (
              <span className="rover-tag-ask" title="Waiting for your answer">
                ?
              </span>
            )}
          </button>
        </Html>
      )}
    </group>
  )
}

// While the rover samples: the instrument and whole-percent progress inside its own tag, in
// the target's colour. Inside drei's own React root, so it subscribes to the store itself; the
// selector string only changes on whole percents, so the tag does not re-render every tick.
function TagSampling({ roverId }) {
  const info = useSim((s) => {
    const sm = s.rovers[roverId]?.sampling
    if (!sm) return ''
    const kind = s.targets.find((t) => t.id === sm.target)?.kind ?? ''
    return `${sm.instrument ?? ''}|${Math.round((Number(sm.progress) || 0) * 100)}|${kind}`
  })
  if (!info) return null
  const [instrument, pct, kind] = info.split('|')
  return (
    <span className="rover-tag-samp" style={{ '--c': kindStyle(kind).color }}>
      <InstrumentIcon kind={instrument} />
      {instrumentStyle(instrument).label}
      <b>{pct}%</b>
    </span>
  )
}

const _c1 = new THREE.Color()
const _c2 = new THREE.Color()

function RoverTrail({ meta }) {
  const size = useThree((s) => s.size)
  const line = useMemo(() => new FatPolyline(TRAIL_POINTS + 1, { width: 2.5, opacity: 0.9, vertexColors: true }), [])
  const xyz = useMemo(() => new Float32Array((TRAIL_POINTS + 1) * 3), [])
  useEffect(() => () => line.dispose(), [line])
  useEffect(() => line.setResolution(size.width, size.height), [line, size])

  const colorAt = useMemo(() => {
    const head = new THREE.Color(meta.color)
    const tail = new THREE.Color(meta.color).multiplyScalar(0.28)
    return (q, n, out) => {
      const t = n > 1 ? q / (n - 1) : 1
      _c1.copy(tail).lerp(head, Math.pow(t, 1.4))
      out[0] = _c1.r
      out[1] = _c1.g
      out[2] = _c1.b
    }
  }, [meta.color])

  useFrame(() => {
    const s = simStore.get()
    const tr = s.trails[meta.id]
    const pose = roverPoses.get(meta.id)
    if (!tr || !tr.count || !pose) {
      line.object.visible = false
      return
    }
    // Every recorded point but the newest, then the displayed rover itself, so the
    // trail ends exactly under the (interpolated) rover.
    let q = 0
    forEachTrailPoint(tr, (x, y, z, idx, count) => {
      if (idx === count - 1) return
      xyz[q++] = x
      xyz[q++] = z + 0.14
      xyz[q++] = -y
    })
    xyz[q++] = pose.x
    xyz[q++] = pose.y + 0.14
    xyz[q++] = pose.z
    line.set(xyz, q / 3, colorAt)
  })

  return <primitive object={line.object} renderOrder={2} />
}

const PATH_MAX = 256

function RoverPath({ meta }) {
  const size = useThree((s) => s.size)
  const line = useMemo(() => new FatPolyline(PATH_MAX, { color: meta.color, width: 2, dashed: true, dashSize: 0.55, gapSize: 0.45 }), [meta.color])
  const xyz = useMemo(() => new Float32Array(PATH_MAX * 3), [])
  const goal = useRef()
  const last = useRef(null)
  const goalMat = useMemo(() => new THREE.MeshBasicMaterial({ color: meta.color, transparent: true, opacity: 0.95, depthWrite: false }), [meta.color])
  useEffect(() => () => line.dispose(), [line])
  useEffect(() => () => goalMat.dispose(), [goalMat])
  useEffect(() => line.setResolution(size.width, size.height), [line, size])

  useFrame(({ clock }) => {
    const s = simStore.get()
    const r = s.rovers[meta.id]
    const g = goal.current
    if (!r) {
      line.object.visible = false
      if (g) g.visible = false
      return
    }
    if (r !== last.current) {
      last.current = r
      const T = s.terrain
      const pts = r.path
      let n = 0
      let prevH = r.z
      const push = (x, y) => {
        if (n >= PATH_MAX) return
        const h = sampleGroundNear(T, x, y, 4)
        if (h !== null) prevH = h
        xyz[n * 3] = x
        xyz[n * 3 + 1] = prevH + 0.32
        xyz[n * 3 + 2] = -y
        n++
      }
      if (pts.length >= 2) {
        let total = 0
        for (let k = 1; k < pts.length; k++) total += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])
        const step = Math.max(0.7, total / (PATH_MAX - pts.length - 2))
        push(pts[0][0], pts[0][1])
        for (let k = 1; k < pts.length; k++) {
          const [x0, y0] = pts[k - 1]
          const [x1, y1] = pts[k]
          const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / step))
          for (let q = 1; q <= steps; q++) push(x0 + ((x1 - x0) * q) / steps, y0 + ((y1 - y0) * q) / steps)
        }
      }
      line.set(xyz, n)
      if (g) {
        if (r.goal) {
          const gh = sampleGroundNear(T, r.goal[0], r.goal[1], 6) ?? prevH
          g.position.set(r.goal[0], gh + 0.08, -r.goal[1])
          g.visible = true
        } else {
          g.visible = false
        }
      }
    }
    if (g && g.visible && !reducedMotion) {
      const k = 1 + 0.08 * Math.sin(clock.elapsedTime * 3)
      g.scale.set(k, 1, k)
    }
  })

  return (
    <group>
      <primitive object={line.object} renderOrder={2} />
      <group ref={goal} visible={false}>
        <mesh geometry={geo.goalRing} material={goalMat} renderOrder={3} />
        <mesh geometry={geo.goalDot} material={goalMat} renderOrder={3} />
      </group>
    </group>
  )
}

export function Rovers({ showPaths, selected, onSelect }) {
  const meta = useSim((s) => s.roverMeta)
  const gen = useSim((s) => s.gen)
  return (
    <group>
      {meta.map((m) => (
        <group key={`${gen}:${m.id}`}>
          <RoverTrail meta={m} />
          {showPaths && <RoverPath meta={m} />}
          <Rover meta={m} selected={selected === m.id} onSelect={onSelect} />
        </group>
      ))}
      <RoverDust />
    </group>
  )
}
