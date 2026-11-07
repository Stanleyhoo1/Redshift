import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { appStore, simStore, useSim, roverPoses } from '../state/store.js'
import { placement } from '../lib/placement.js'
import { forEachTrailPoint, sampleGround, sampleGroundNear, TRAIL_POINTS } from '../state/simState.js'
import { FatPolyline } from './fatLine.js'
import { glowTexture, warningTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'
import { HAZARD } from '../lib/theme.js'

const MODEL_SCALE = 1.5 // a little larger than life so rovers read on a projector

const geo = {
  body: new THREE.BoxGeometry(1.15, 0.3, 0.78),
  wheel: new THREE.CylinderGeometry(0.21, 0.21, 0.16, 14).rotateX(Math.PI / 2),
  panel: new THREE.BoxGeometry(0.56, 0.04, 0.86), // rear half only, so the colour shows from above
  mast: new THREE.BoxGeometry(0.06, 0.42, 0.06),
  head: new THREE.BoxGeometry(0.16, 0.13, 0.32),
  light: new THREE.BoxGeometry(0.04, 0.07, 0.56),
  halo: new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
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
const shared = {
  wheel: new THREE.MeshStandardMaterial({ color: '#1a1e26', roughness: 0.9, metalness: 0.1 }),
  panel: new THREE.MeshStandardMaterial({ color: '#1c2a45', roughness: 0.35, metalness: 0.6 }),
  metal: new THREE.MeshStandardMaterial({ color: '#9aa4b2', roughness: 0.5, metalness: 0.5 }),
  head: new THREE.MeshBasicMaterial({ color: '#e8f6ff' }),
  light: new THREE.MeshBasicMaterial({ color: '#ffffff' }),
}
const WHEELS = [
  [0.4, 0.21, 0.47],
  [-0.4, 0.21, 0.47],
  [0.4, 0.21, -0.47],
  [-0.4, 0.21, -0.47],
]

function RoverModel({ color }) {
  // A touch of emissive keeps the fleet colours readable on a washed-out projector.
  const body = useMemo(() => new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.15, emissive: color, emissiveIntensity: 0.22 }), [color])
  useEffect(() => () => body.dispose(), [body])
  return (
    <group scale={MODEL_SCALE}>
      <mesh geometry={geo.body} material={body} position={[0, 0.43, 0]} />
      {WHEELS.map((p, i) => (
        <mesh key={i} geometry={geo.wheel} material={shared.wheel} position={p} />
      ))}
      <mesh geometry={geo.panel} material={shared.panel} position={[-0.26, 0.6, 0]} />
      <mesh geometry={geo.mast} material={shared.metal} position={[0.38, 0.8, 0]} />
      <mesh geometry={geo.head} material={shared.head} position={[0.42, 1.04, 0]} />
      <mesh geometry={geo.light} material={shared.light} position={[0.585, 0.45, 0]} />
    </group>
  )
}

const _up = new THREE.Vector3()
const _fwd = new THREE.Vector3()
const _side = new THREE.Vector3()
const _basis = new THREE.Matrix4()
const _q = new THREE.Quaternion()

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

function Rover({ meta, selected, onSelect }) {
  const outer = useRef()
  const inner = useRef()
  const pose = useRef(null)
  const fault = useRef()
  const sign = useRef()
  const sel = useRef()
  const state = useSim((s) => s.rovers[meta.id]?.state)
  const faulted = state === 'fault'
  const asking = useSim((s) => s.requests.some((q) => q.rover === meta.id)) // v3: waiting for the operator
  // drei <Html> picks its DOM container once R3F has connected its events; mounting it
  // earlier makes it unmount and remount its own React root (a React error in dev).
  const labelsReady = useThree((s) => !!s.events.connected)
  const haloMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: meta.color, map: glowTexture(), transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }),
    [meta.color],
  )
  useEffect(() => () => haloMat.dispose(), [haloMat])
  useEffect(() => () => roverPoses.delete(meta.id), [meta.id])
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
      return
    }
    let p = pose.current
    if (!p || Math.hypot(r.x - p.x, r.y - p.y) > 8) {
      p = pose.current = { x: r.x, y: r.y, yaw: r.yaw, gz: null }
    }
    // Ticks arrive at ~5 Hz; ease towards the latest one so motion stays smooth.
    const a = 1 - Math.exp(-Math.min(dt, 0.1) * 8)
    p.x += (r.x - p.x) * a
    p.y += (r.y - p.y) * a
    p.yaw += wrap(r.yaw - p.yaw) * a
    const g = sampleGround(s.terrain, p.x, p.y)
    const gz = g ?? r.z
    p.gz = p.gz === null ? gz : p.gz + (gz - p.gz) * Math.min(1, a * 1.5)
    o.visible = true
    o.position.set(p.x, p.gz, -p.y)

    groundNormal(s.terrain, p.x, p.y, _up)
    _fwd.set(Math.cos(p.yaw), 0, -Math.sin(p.yaw))
    _fwd.addScaledVector(_up, -_fwd.dot(_up)).normalize()
    _side.crossVectors(_fwd, _up)
    _basis.makeBasis(_fwd, _up, _side)
    _q.setFromRotationMatrix(_basis)
    if (inner.current) inner.current.quaternion.slerp(_q, Math.min(1, a * 1.2))
    roverPoses.set(meta.id, o.position)

    // Motor fault: a blinking crimson ring and a warning sign over the parked rover.
    const isFault = r.state === 'fault'
    const t = clock.elapsedTime
    const blink = reducedMotion ? 1 : 0.5 + 0.5 * Math.sign(Math.sin(t * 6.5))
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
        <RoverModel color={meta.color} />
        <mesh geometry={geo.halo} material={haloMat} scale={4.2} position={[0, 0.06, 0]} renderOrder={3} />
      </group>
      <mesh ref={sel} geometry={geo.selRing} material={selMat} position={[0, 0.1, 0]} visible={!!selected} renderOrder={3} />
      <group ref={fault} visible={false}>
        <mesh geometry={geo.faultRing} material={faultMats.ring} position={[0, 0.12, 0]} renderOrder={4} />
        <mesh geometry={geo.halo} material={faultMats.glow} scale={6} position={[0, 0.08, 0]} renderOrder={4} />
      </group>
      <sprite ref={sign} material={faultMats.sign} position={[0, 3.55, 0]} scale={1.15} visible={false} renderOrder={10} />
      {labelsReady && (
        <Html position={[0, 2.5, 0]} center zIndexRange={[12, 0]} style={{ pointerEvents: 'none' }}>
          <button type="button" className="rover-tag" data-selected={selected || undefined} data-fault={faulted || undefined} data-ask={(asking && !faulted) || undefined} onClick={tagClick} title={`Show ${meta.name}'s sensors`}>
            <span className="dot" style={{ background: meta.color }} />
            {meta.name}
            {faulted && <span className="rover-tag-fault">Fault</span>}
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
    </group>
  )
}
