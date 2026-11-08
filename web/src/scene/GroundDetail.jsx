import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { groundAt, knownAt, pointInPoly } from './ground.js'

// v7 ground detail (launch setting `ground_detail`, look-only: the sim sends no geoms for it).
// Scattered pebbles, dark crack lines and a few tiny craters, placed from the world seed so a
// world always looks the same. Instanced, four draw calls. A piece shows on ground the fleet
// knows, or everywhere with Reality on; none on water or the lander pad. Graphics Low draws a
// third of them.

const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0)
const UP = new THREE.Vector3(0, 1, 0)
const _m = new THREE.Matrix4()
const _p = new THREE.Vector3()
const _s = new THREE.Vector3()
const _q = new THREE.Quaternion()
const _q2 = new THREE.Quaternion()
const _n = new THREE.Vector3()
const _c = new THREE.Color()

export function mulberry32(a) {
  return function rand() {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// A thin jagged strip along +x (length 1, width 1), laid flat: one crack, varied per instance.
function crackGeometry() {
  const pts = [[-0.5, 0], [-0.32, 0.9], [-0.12, -0.7], [0.08, 0.6], [0.27, -0.9], [0.5, 0.2]]
  const pos = []
  const idx = []
  pts.forEach(([x, o], k) => {
    const w = 0.5 * (1 - Math.abs(x) * 1.4) + 0.12
    pos.push(x, 0, (o * 0.06) - w * 0.5, x, 0, (o * 0.06) + w * 0.5)
    if (k) {
      const a = (k - 1) * 2
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2)
    }
  })
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setIndex(idx)
  g.computeVertexNormals()
  return g
}

function layout(seed, T, lander, water) {
  const rand = mulberry32(((seed | 0) ^ 7108) >>> 0)
  const area = T.width * T.depth
  const counts = {
    pebble: Math.min(5000, Math.round(area * 0.03)),
    crack: Math.min(700, Math.round(area * 0.003)),
    crater: Math.min(90, Math.round(area * 0.0004)),
  }
  const bodies = Array.isArray(water) ? water.filter((b) => Array.isArray(b?.outline) && b.outline.length > 2) : []
  const padR = (lander?.radius ?? 2.5) * 1.04 + 1.2
  const ok = (x, y, r) => {
    if (lander && Math.hypot(x - lander.x, y - lander.y) < padR + r) return false
    for (const b of bodies) if (pointInPoly(x, y, b.outline)) return false
    return true
  }
  const pick = () => [T.ox + rand() * T.width, T.oy + rand() * T.depth]
  const out = { pebble: [], crack: [], crater: [] }
  // Pebbles gather in loose patches, with a thin scatter between them.
  for (let n = 0; n < counts.pebble; n++) {
    let x
    let y
    if (n % 4 && out.pebble.length) {
      const c = out.pebble[(rand() * out.pebble.length) | 0]
      const a = rand() * Math.PI * 2
      const d = 0.4 + rand() * rand() * 3.5
      x = c.x + d * Math.cos(a)
      y = c.y + d * Math.sin(a)
    } else [x, y] = pick()
    const s = 0.04 + rand() * rand() * 0.22
    if (!ok(x, y, s)) continue
    out.pebble.push({ x, y, s, sy: 0.45 + rand() * 0.45, yaw: rand() * Math.PI * 2, tilt: (rand() - 0.5) * 0.8, tone: rand() })
  }
  for (let n = 0; n < counts.crack; n++) {
    const [x, y] = pick()
    const len = 0.6 + rand() * 2.4
    if (!ok(x, y, len)) continue
    out.crack.push({ x, y, len, w: 0.05 + rand() * 0.07, yaw: rand() * Math.PI * 2 })
  }
  for (let n = 0; n < counts.crater; n++) {
    const [x, y] = pick()
    const r = 0.35 + rand() * rand() * 1.6
    if (!ok(x, y, r)) continue
    out.crater.push({ x, y, r })
  }
  return out
}

// Tilt to the local ground (normal from heights half a metre either side).
function groundTilt(x, y, q) {
  const e = 0.5
  const hx = (groundAt(x + e, y) - groundAt(x - e, y)) / (2 * e)
  const hy = (groundAt(x, y + e) - groundAt(x, y - e)) / (2 * e)
  _n.set(-hx, 1, hy).normalize()
  return q.setFromUnitVectors(UP, _n)
}

export function GroundDetail({ reality }) {
  const config = useSim((s) => s.config)
  const T = useSim((s) => s.terrain)
  const lander = useSim((s) => s.lander)
  const water = useSim((s) => s.features?.water)
  const graphics = useGraphics()
  const on = !!(config && config.ground_detail === true && T)
  const seed = Number.isFinite(config?.seed) ? config.seed : 0
  const items = useMemo(() => (on ? layout(seed, T, lander, water) : null), [on, seed, T, lander, water])
  const geos = useMemo(() => ({
    pebble: new THREE.DodecahedronGeometry(1, 0),
    crack: crackGeometry(),
    rim: new THREE.RingGeometry(0.72, 1.0, 20).rotateX(-Math.PI / 2),
    floor: new THREE.CircleGeometry(0.78, 20).rotateX(-Math.PI / 2),
  }), [])
  useEffect(() => () => Object.values(geos).forEach((g) => g.dispose()), [geos])
  const pebble = useRef()
  const crack = useRef()
  const rim = useRef()
  const floor = useRef()
  const next = useRef(0)

  useEffect(() => {
    next.current = 0
  }, [items, reality, graphics])

  useFrame(({ clock }) => {
    if (!items || !pebble.current || !crack.current || !rim.current || !floor.current) return
    const t = clock.elapsedTime
    if (t < next.current) return
    next.current = t + 0.8
    const every = graphics === 'low' ? 3 : 1
    const show = (k, x, y) => k % every === 0 && (reality || knownAt(x, y))
    items.pebble.forEach((p, k) => {
      if (!show(k, p.x, p.y)) return pebble.current.setMatrixAt(k, HIDDEN)
      const gz = groundAt(p.x, p.y)
      _q.setFromAxisAngle(UP, p.yaw)
      _q2.setFromAxisAngle(_n.set(1, 0, 0), p.tilt)
      _q.multiply(_q2)
      _m.compose(_p.set(p.x, gz + p.s * p.sy * 0.35, -p.y), _q, _s.set(p.s, p.s * p.sy, p.s * (0.8 + 0.3 * p.tone)))
      pebble.current.setMatrixAt(k, _m)
      pebble.current.setColorAt(k, _c.setScalar(0.72 + 0.4 * p.tone))
    })
    items.crack.forEach((c, k) => {
      if (!show(k, c.x, c.y)) return crack.current.setMatrixAt(k, HIDDEN)
      groundTilt(c.x, c.y, _q)
      _q2.setFromAxisAngle(UP, c.yaw)
      _q.multiply(_q2)
      _m.compose(_p.set(c.x, groundAt(c.x, c.y) + 0.025, -c.y), _q, _s.set(c.len, 1, c.w * 10))
      crack.current.setMatrixAt(k, _m)
    })
    items.crater.forEach((c, k) => {
      if (!show(k, c.x, c.y)) {
        rim.current.setMatrixAt(k, HIDDEN)
        floor.current.setMatrixAt(k, HIDDEN)
        return
      }
      groundTilt(c.x, c.y, _q)
      const gz = groundAt(c.x, c.y)
      _m.compose(_p.set(c.x, gz + 0.03, -c.y), _q, _s.set(c.r, 1, c.r))
      rim.current.setMatrixAt(k, _m)
      _m.compose(_p.set(c.x, gz + 0.025, -c.y), _q, _s.set(c.r, 1, c.r))
      floor.current.setMatrixAt(k, _m)
    })
    for (const r of [pebble, crack, rim, floor]) r.current.instanceMatrix.needsUpdate = true
    if (pebble.current.instanceColor) pebble.current.instanceColor.needsUpdate = true
  })

  if (!items) return null
  return (
    <group>
      <instancedMesh ref={pebble} args={[geos.pebble, undefined, Math.max(1, items.pebble.length)]} frustumCulled={false} castShadow receiveShadow>
        <meshStandardMaterial color="#6e5f53" roughness={0.9} flatShading />
      </instancedMesh>
      <instancedMesh ref={crack} args={[geos.crack, undefined, Math.max(1, items.crack.length)]} frustumCulled={false} renderOrder={1}>
        <meshStandardMaterial color="#1f1915" roughness={1} transparent opacity={0.72} depthWrite={false} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
      </instancedMesh>
      <instancedMesh ref={floor} args={[geos.floor, undefined, Math.max(1, items.crater.length)]} frustumCulled={false} renderOrder={1}>
        <meshStandardMaterial color="#3b2f27" roughness={1} transparent opacity={0.55} depthWrite={false} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
      </instancedMesh>
      <instancedMesh ref={rim} args={[geos.rim, undefined, Math.max(1, items.crater.length)]} frustumCulled={false} renderOrder={2}>
        <meshStandardMaterial color="#b49a82" roughness={1} transparent opacity={0.6} depthWrite={false} polygonOffset polygonOffsetFactor={-3} polygonOffsetUnits={-3} />
      </instancedMesh>
    </group>
  )
}
