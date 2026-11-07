import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { nowSeconds, roverPoses, simStore, trackPoses, useSim } from '../state/store.js'
import { sampleGround, sampleGroundNear } from '../state/simState.js'
import { LIFE } from '../lib/theme.js'
import { FatPolyline } from './fatLine.js'
import { glowTexture } from './textures.js'
import { groundAt, hashStr } from './ground.js'
import { reducedMotion } from './sceneConstants.js'

// Lifeforms (PROTOCOL.md version 4). `lifeforms` is the ground truth, `tracks` is what the
// fleet believes. A creature shows when a fleet track is near it (the rovers have noticed
// it), or faded with Reality on. Each track gets a dashed ring, a velocity arrow and its
// label in a small tag; lost tracks fade out. An observing rover gets a dashed line of
// sight to the track it watches.

const TRACK_NEAR = 3 // m: a creature counts as noticed when a live track is this close

// ---------------------------------------------------------------- creatures

// Stylised grazers: three body segments, six legs with a tripod gait, two antennae and
// glowing spots along the back. Every part of every creature is an instance, so a herd
// costs four draw calls (eight with the faded Reality-only set).
const MAXC = 40
const SEGS = [
  // x, y, half extents (x, y, z), all times the creature's size
  [-0.33, 0.4, 0.27, 0.17, 0.2],
  [0.0, 0.43, 0.18, 0.15, 0.17],
  [0.31, 0.45, 0.13, 0.11, 0.12],
]
const HIP_X = [0.12, -0.02, -0.18]
const FOOT_DX = [0.2, 0.03, -0.17]
const LIMBS = 14 // 12 leg parts and 2 antennae
const DOTS = 4

const sphereGeo = new THREE.IcosahedronGeometry(1, 1) // flat shaded: a faceted, stylised hide
const limbGeo = new THREE.CylinderGeometry(1, 0.8, 1, 5)
const dotGeo = new THREE.IcosahedronGeometry(1, 0)
const haloGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2)

function makeSet(ghost) {
  const o = ghost ? { transparent: true, opacity: 0.3, depthWrite: false } : {}
  const mats = {
    // A pale, faceted hide with some self-light (like the rovers), so a herd reads on
    // every slope colour; dark legs; glowing spots in the life colour.
    body: new THREE.MeshStandardMaterial({ color: '#e6dcc2', roughness: 0.65, metalness: 0.02, flatShading: true, emissive: '#8c8566', emissiveIntensity: 0.55, ...o }),
    limb: new THREE.MeshStandardMaterial({ color: '#6e6857', roughness: 0.8, metalness: 0.05, flatShading: true, emissive: '#2a2820', emissiveIntensity: 0.6, ...o }),
    dot: new THREE.MeshBasicMaterial({ color: LIFE, transparent: ghost, opacity: ghost ? 0.5 : 1, depthWrite: !ghost }),
    halo: new THREE.MeshBasicMaterial({ color: LIFE, map: glowTexture(), transparent: true, opacity: ghost ? 0.12 : 0.3, depthWrite: false, blending: THREE.AdditiveBlending }),
  }
  const mk = (g, m, n) => {
    const mesh = new THREE.InstancedMesh(g, m, n)
    mesh.count = 0
    mesh.frustumCulled = false
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    return mesh
  }
  const set = {
    body: mk(sphereGeo, mats.body, MAXC * SEGS.length),
    limb: mk(limbGeo, mats.limb, MAXC * LIMBS),
    dot: mk(dotGeo, mats.dot, MAXC * DOTS),
    halo: mk(haloGeo, mats.halo, MAXC),
    n: 0,
    mats,
  }
  set.halo.renderOrder = 3
  if (ghost) for (const k of ['body', 'limb', 'dot']) set[k].renderOrder = 5
  return set
}

const _G = new THREE.Matrix4()
const _L = new THREE.Matrix4()
const _M = new THREE.Matrix4()
const _q = new THREE.Quaternion()
const _qy = new THREE.Quaternion()
const _p = new THREE.Vector3()
const _s = new THREE.Vector3()
const _a = new THREE.Vector3()
const _b = new THREE.Vector3()
const _k = new THREE.Vector3()
const _d = new THREE.Vector3()
const UP = new THREE.Vector3(0, 1, 0)
const NOQ = new THREE.Quaternion()

function put(mesh, i, pos, quat, scale) {
  _L.compose(pos, quat, scale)
  _M.multiplyMatrices(_G, _L)
  mesh.setMatrixAt(i, _M)
}

// A limb from a to b (creature frame), `w` thick.
function limb(mesh, i, a, b, w) {
  _d.subVectors(b, a)
  const len = _d.length() || 1e-4
  _q.setFromUnitVectors(UP, _d.multiplyScalar(1 / len))
  _p.addVectors(a, b).multiplyScalar(0.5)
  put(mesh, i, _p, _q, _s.set(w, len, w))
}

const SIDES = [-1, 1]
const PARTS = ['body', 'limb', 'dot', 'halo']

function writeCreature(set, pose, S, t) {
  const c = set.n++
  _qy.setFromAxisAngle(UP, pose.yaw)
  _G.compose(_p.set(pose.x, pose.gz, -pose.y), _qy, _s.set(1, 1, 1))
  const move = Math.min(1, pose.spd / 0.22)
  const bob = Math.sin(pose.phase * 2) * 0.018 * S * move
  for (let k = 0; k < SEGS.length; k++) {
    const [x, y, sx, sy, sz] = SEGS[k]
    put(set.body, c * SEGS.length + k, _p.set(x * S, y * S + bob, 0), NOQ, _s.set(sx * S, sy * S, sz * S))
  }
  let li = c * LIMBS
  for (let i = 0; i < 3; i++) {
    for (const side of SIDES) {
      const ph = pose.phase + ((i + (side > 0 ? 1 : 0)) % 2) * Math.PI
      _a.set(HIP_X[i] * S, 0.38 * S + bob, side * 0.11 * S)
      const fx = (HIP_X[i] + FOOT_DX[i]) * S + 0.13 * S * move * Math.sin(ph)
      _b.set(fx, 0.09 * S * move * Math.max(0, Math.cos(ph)), side * 0.44 * S)
      _k.addVectors(_a, _b).multiplyScalar(0.5)
      _k.y += 0.21 * S
      _k.z += side * 0.06 * S
      limb(set.limb, li++, _a, _k, 0.034 * S)
      limb(set.limb, li++, _k, _b, 0.026 * S)
    }
  }
  const wig = reducedMotion ? 0 : Math.sin(t * 3 + pose.seed) * 0.04 * S
  for (const side of SIDES) {
    _a.set(0.41 * S, 0.5 * S + bob, side * 0.04 * S)
    _b.set(0.62 * S, 0.7 * S + bob + wig * side, side * 0.14 * S)
    limb(set.limb, li++, _a, _b, 0.014 * S)
  }
  // Bioluminescent spots: one on each segment and one at the tail, slowly pulsing.
  for (let k = 0; k < DOTS; k++) {
    const pulse = reducedMotion ? 1 : 0.8 + 0.35 * Math.sin(t * 2.2 + pose.seed + k * 1.3)
    if (k < SEGS.length) {
      const [x, y, , sy] = SEGS[k]
      _p.set(x * S, (y + sy * 0.92) * S + bob, 0)
    } else _p.set(-0.58 * S, 0.4 * S + bob, 0)
    const r = (k === DOTS - 1 ? 0.07 : 0.055) * S * pulse
    put(set.dot, c * DOTS + k, _p, NOQ, _s.set(r, r, r))
  }
  put(set.halo, c, _p.set(0, 0.04, 0), NOQ, _s.set(2 * S, 1, 2 * S))
}

function finish(set) {
  set.body.count = set.n * SEGS.length
  set.limb.count = set.n * LIMBS
  set.dot.count = set.n * DOTS
  set.halo.count = set.n
  for (const k of PARTS) if (set.n) set[k].instanceMatrix.needsUpdate = true
}

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a))

function Creatures({ reality }) {
  const sets = useMemo(() => ({ solid: makeSet(false), ghost: makeSet(true) }), [])
  const poses = useRef(new Map())
  const gen = useSim((s) => s.gen)
  useEffect(() => poses.current.clear(), [gen])
  useEffect(
    () => () => {
      for (const set of Object.values(sets)) {
        Object.values(set.mats).forEach((m) => m.dispose())
        for (const k of PARTS) set[k].dispose()
      }
    },
    [sets],
  )

  useFrame((_, dt) => {
    const s = simStore.get()
    const now = nowSeconds()
    const T = s.terrain
    sets.solid.n = 0
    sets.ghost.n = 0
    const a = 1 - Math.exp(-Math.min(dt, 0.1) * 5)
    for (const life of s.lifeforms) {
      let tracked = false
      for (const k of s.tracks) {
        if (k.status === 'tracking' && Math.abs(k.x - life.x) < TRACK_NEAR && Math.abs(k.y - life.y) < TRACK_NEAR && Math.hypot(k.x - life.x, k.y - life.y) < TRACK_NEAR) {
          tracked = true
          break
        }
      }
      const set = tracked ? sets.solid : reality ? sets.ghost : null
      let p = poses.current.get(life.id)
      if (!p || Math.hypot(life.x - p.x, life.y - p.y) > 6) {
        p = { x: life.x, y: life.y, yaw: life.yaw, gz: null, phase: 0, spd: 0, seed: (hashStr(life.id) % 628) / 100, at: now }
        poses.current.set(life.id, p)
      }
      const ox = p.x
      const oy = p.y
      p.x += (life.x - p.x) * a
      p.y += (life.y - p.y) * a
      p.yaw += wrap(life.yaw - p.yaw) * a
      const moved = Math.hypot(p.x - ox, p.y - oy)
      p.spd += (moved / Math.max(dt, 1e-3) - p.spd) * Math.min(1, dt * 4)
      p.phase += (moved / (0.4 * life.size)) * Math.PI
      p.at = now
      if (!set || set.n >= MAXC) continue
      const g = sampleGround(T, p.x, p.y) ?? (Number.isFinite(life.z) ? life.z - 0.25 * life.size : groundAt(p.x, p.y))
      p.gz = p.gz === null ? g : p.gz + (g - p.gz) * Math.min(1, a * 1.5)
      writeCreature(set, p, life.size, now)
    }
    // Forget creatures the sim no longer lists.
    if (poses.current.size > s.lifeforms.length + 8) for (const [id, p] of poses.current) if (now - p.at > 2) poses.current.delete(id)
    finish(sets.solid)
    finish(sets.ghost)
  })

  return (
    <group>
      {Object.values(sets).map((set, i) => (
        <group key={i}>
          <primitive object={set.halo} />
          <primitive object={set.body} />
          <primitive object={set.limb} />
          <primitive object={set.dot} />
        </group>
      ))}
    </group>
  )
}

// ---------------------------------------------------------------- tracks

const RING_SEGS = 48

// A herd makes several tracks close together: only the lead one (a watched track, else a
// live one, else the lowest id) carries a tag, with the number of others around it. The
// rest keep their ring and arrow, so the herd stays readable instead of a pile of tags.
const CLUSTER = 6 // m
const rank = (k) => (k.observer ? 0 : k.status === 'tracking' ? 1 : 2)
const byId = (a, b) => a.id.localeCompare(b.id, undefined, { numeric: true })
// -1 when another track nearby leads, else how many other tracks are around this one.
function clusterInfo(k, tracks) {
  let others = 0
  for (const o of tracks) {
    if (o === k || Math.abs(o.x - k.x) > CLUSTER || Math.abs(o.y - k.y) > CLUSTER || Math.hypot(o.x - k.x, o.y - k.y) > CLUSTER) continue
    if (rank(o) < rank(k) || (rank(o) === rank(k) && byId(o, k) < 0)) return -1
    others++
  }
  return others
}

function TrackTag({ id }) {
  // Only the text and status re-render (positions move in useFrame).
  const info = useSim((s) => {
    const k = s.tracks.find((q) => q.id === id)
    if (!k) return ''
    const more = clusterInfo(k, s.tracks)
    return more < 0 ? '' : `${k.status}\u0001${k.label ?? ''}\u0001${k.observer ?? ''}\u0001${more}`
  })
  const names = useSim((s) => s.roverMeta)
  if (!info) return null
  const [status, label, observer, more] = info.split('\u0001')
  const watcher = observer ? names.find((m) => m.id === observer) : null
  const n = Number(more)
  return (
    <div className="track-tag" data-lost={status === 'lost' || undefined} title={n ? `${n} more ${n === 1 ? 'track' : 'tracks'} close by` : undefined}>
      <span className="track-tag-id">{id}</span>
      <span className="track-tag-label">{label || 'moving object'}</span>
      {n > 0 && <span className="track-tag-more">+{n}</span>}
      {status === 'lost' ? <span className="track-tag-lost">lost</span> : watcher && <span className="track-tag-eye" style={{ '--c': watcher.color }} title={`${watcher.name} is watching`} />}
    </div>
  )
}

function TrackMarker({ id }) {
  const size = useThree((s) => s.size)
  const labelsReady = useThree((s) => !!s.events.connected)
  const group = useRef()
  const ring = useMemo(() => new FatPolyline(RING_SEGS + 1, { color: LIFE, width: 2.2, dashed: true, dashSize: 0.42, gapSize: 0.3, opacity: 0.95 }), [])
  const arrow = useMemo(() => new FatPolyline(5, { color: LIFE, width: 2.2, opacity: 0.95 }), [])
  const ringXyz = useMemo(() => new Float32Array((RING_SEGS + 1) * 3), [])
  const arrowXyz = useMemo(() => new Float32Array(5 * 3), [])
  const st = useRef({ x: null, y: null, gz: 0, lostAt: null })
  useEffect(() => () => (ring.dispose(), arrow.dispose(), trackPoses.delete(id)), [ring, arrow, id])
  useEffect(() => {
    ring.setResolution(size.width, size.height)
    arrow.setResolution(size.width, size.height)
  }, [ring, arrow, size])

  useFrame((_, dt) => {
    const s = simStore.get()
    let k = null
    for (const q of s.tracks) if (q.id === id) k = q
    const g = group.current
    if (!k || !g) {
      ring.object.visible = false
      arrow.object.visible = false
      if (g) g.visible = false
      return
    }
    const p = st.current
    const a = 1 - Math.exp(-Math.min(dt, 0.1) * 5)
    if (p.x === null || Math.hypot(k.x - p.x, k.y - p.y) > 6) {
      p.x = k.x
      p.y = k.y
      p.gz = sampleGroundNear(s.terrain, k.x, k.y, 6) ?? groundAt(k.x, k.y)
    }
    p.x += (k.x - p.x) * a
    p.y += (k.y - p.y) * a
    const gz = sampleGroundNear(s.terrain, p.x, p.y, 6) ?? p.gz
    p.gz += (gz - p.gz) * a
    g.visible = true
    g.position.set(p.x, p.gz, -p.y)
    let pose = trackPoses.get(id)
    if (!pose) trackPoses.set(id, (pose = { x: 0, y: 0, z: 0 }))
    pose.x = p.x
    pose.y = p.gz
    pose.z = -p.y

    // Lost: fade out over a few seconds.
    const now = nowSeconds()
    if (k.status === 'lost') p.lostAt ??= now
    else p.lostAt = null
    const fade = p.lostAt === null ? 1 : Math.max(0.12, 1 - (now - p.lostAt) / 6)
    ring.material.opacity = 0.95 * fade
    arrow.material.opacity = 0.95 * fade
    if (!reducedMotion && p.lostAt === null) ring.material.dashOffset -= dt * 0.5

    // Ring draped on the ground around the believed position.
    const R = 0.55 * k.size + 0.55
    for (let q = 0; q <= RING_SEGS; q++) {
      const ang = (q / RING_SEGS) * Math.PI * 2
      const x = p.x + Math.cos(ang) * R
      const y = p.y + Math.sin(ang) * R
      ringXyz[q * 3] = x
      ringXyz[q * 3 + 1] = (sampleGround(s.terrain, x, y) ?? p.gz) + 0.14
      ringXyz[q * 3 + 2] = -y
    }
    ring.set(ringXyz, RING_SEGS + 1)

    // Velocity arrow: where it will be in about 4 s, from the edge of the ring.
    const sp = Math.hypot(k.vx, k.vy)
    if (sp < 0.04 || k.status === 'lost') arrow.object.visible = false
    else {
      const ux = k.vx / sp
      const uy = k.vy / sp
      const len = Math.max(0.7, Math.min(4, sp * 4))
      const x0 = p.x + ux * R
      const y0 = p.y + uy * R
      const x1 = x0 + ux * len
      const y1 = y0 + uy * len
      const h = p.gz + 0.2
      const bl = 0.38
      const set = (q, x, y) => {
        arrowXyz[q * 3] = x
        arrowXyz[q * 3 + 1] = h
        arrowXyz[q * 3 + 2] = -y
      }
      set(0, x0, y0)
      set(1, x1, y1)
      set(2, x1 - (ux * 0.8 - uy * 0.55) * bl, y1 - (uy * 0.8 + ux * 0.55) * bl)
      set(3, x1, y1)
      set(4, x1 - (ux * 0.8 + uy * 0.55) * bl, y1 - (uy * 0.8 - ux * 0.55) * bl)
      arrow.set(arrowXyz, 5)
    }
  })

  return (
    <group>
      <primitive object={ring.object} renderOrder={4} />
      <primitive object={arrow.object} renderOrder={4} />
      <group ref={group} visible={false}>
        {labelsReady && (
          <Html position={[0, 1.9, 0]} center zIndexRange={[11, 0]} style={{ pointerEvents: 'none' }}>
            <TrackTag id={id} />
          </Html>
        )}
      </group>
    </group>
  )
}

function Tracks() {
  const ids = useSim((s) => s.tracks.map((k) => k.id).join(','))
  const gen = useSim((s) => s.gen)
  return (
    <group>
      {ids
        ? ids.split(',').map((id) => <TrackMarker key={`${gen}:${id}`} id={id} />)
        : null}
    </group>
  )
}

// ---------------------------------------------------------------- line of sight

function LineOfSight({ meta }) {
  const size = useThree((s) => s.size)
  const line = useMemo(() => new FatPolyline(2, { color: LIFE, width: 2.6, dashed: true, dashSize: 0.5, gapSize: 0.3, opacity: 0.95 }), [])
  const xyz = useMemo(() => new Float32Array(6), [])
  useEffect(() => () => line.dispose(), [line])
  useEffect(() => line.setResolution(size.width, size.height), [line, size])
  useFrame((_, dt) => {
    const r = simStore.get().rovers[meta.id]
    const rp = roverPoses.get(meta.id)
    const tp = r && r.state === 'observing' && r.track_id ? trackPoses.get(r.track_id) : null
    if (!rp || !tp) {
      line.object.visible = false
      return
    }
    xyz[0] = rp.x
    xyz[1] = rp.y + 1.75
    xyz[2] = rp.z
    xyz[3] = tp.x
    xyz[4] = tp.y + 0.75
    xyz[5] = tp.z
    line.set(xyz, 2)
    if (!reducedMotion) line.material.dashOffset -= dt * 1.4
  })
  return <primitive object={line.object} renderOrder={4} />
}

export function Life({ reality }) {
  const meta = useSim((s) => s.roverMeta)
  const gen = useSim((s) => s.gen)
  return (
    <group>
      <Creatures reality={reality} />
      <Tracks />
      {meta.map((m) => (
        <LineOfSight key={`${gen}:${m.id}`} meta={m} />
      ))}
    </group>
  )
}
