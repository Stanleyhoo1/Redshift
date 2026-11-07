import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { simStore, useSim, roverPoses } from '../state/store.js'
import { forEachTrailPoint, sampleGround, sampleGroundNear, TRAIL_POINTS } from '../state/simState.js'
import { FatPolyline } from './fatLine.js'
import { glowTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'

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

function Rover({ meta }) {
  const outer = useRef()
  const inner = useRef()
  const pose = useRef(null)
  // drei <Html> picks its DOM container once R3F has connected its events; mounting it
  // earlier makes it unmount and remount its own React root (a React error in dev).
  const labelsReady = useThree((s) => !!s.events.connected)
  const haloMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: meta.color, map: glowTexture(), transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }),
    [meta.color],
  )
  useEffect(() => () => haloMat.dispose(), [haloMat])
  useEffect(() => () => roverPoses.delete(meta.id), [meta.id])

  useFrame((_, dt) => {
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
  })

  return (
    <group ref={outer} visible={false}>
      <group ref={inner}>
        <RoverModel color={meta.color} />
        <mesh geometry={geo.halo} material={haloMat} scale={4.2} position={[0, 0.06, 0]} renderOrder={3} />
      </group>
      {labelsReady && (
        <Html position={[0, 2.5, 0]} center zIndexRange={[12, 0]} style={{ pointerEvents: 'none' }}>
          <div className="rover-tag">
            <span className="dot" style={{ background: meta.color }} />
            {meta.name}
          </div>
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

export function Rovers({ showPaths }) {
  const meta = useSim((s) => s.roverMeta)
  const gen = useSim((s) => s.gen)
  return (
    <group>
      {meta.map((m) => (
        <group key={`${gen}:${m.id}`}>
          <RoverTrail meta={m} />
          {showPaths && <RoverPath meta={m} />}
          <Rover meta={m} />
        </group>
      ))}
    </group>
  )
}
