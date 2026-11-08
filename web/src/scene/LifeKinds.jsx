import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { nowSeconds, simStore, useSim } from '../state/store.js'
import { sampleGround } from '../state/simState.js'
import { useGraphics } from '../state/visualPrefs.js'
import { glowTexture } from './textures.js'
import { groundAt, hashStr } from './ground.js'
import { reducedMotion } from './sceneConstants.js'
import { KIND_LOOK, burrowLift, drifterPulse, skitterGait } from './lifeKinds.js'

// The lifeform kinds beyond grazers (PROTOCOL.md version 7, "More lifeforms"):
//   skitterer  small six-legged amber runner, quick jittery gait, in swarms
//   drifter    translucent glowing bell floating 1.5 to 3.5 m up, trailing tendrils, gentle bob
//   crawler    big dark domed shell with pale ridges, slow
//   burrower   segmented head rising from a small mound and sinking back (the mound stays)
// Like grazers, a creature shows solid when a live fleet track is near it, or faded with
// Reality on. Every part is an instance: a few draw calls per kind and set. Graphics Low
// drops the small parts (legs, tendrils, eyes, glows) and uses coarser spheres.

const MAXK = 48 // creatures per kind and set
const TRACK_NEAR = 3

const PARTS = {
  skitterer: { body: 1, head: 1, leg: 6, eye: 2 },
  drifter: { bell: 1, core: 1, tendril: 6, halo: 1 },
  crawler: { shell: 1, ridge: 4, head: 1, leg: 6, eye: 2 },
  burrower: { mound: 1, hole: 1, seg: 3, head: 1, eye: 2 },
}
const DETAIL = new Set(['leg', 'eye', 'tendril', 'halo']) // left out with Graphics Low

function makeGeos(high) {
  const seg = high ? [18, 12] : [10, 7]
  const sphere = new THREE.SphereGeometry(1, ...seg)
  const dome = new THREE.SphereGeometry(1, seg[0], seg[1], 0, Math.PI * 2, 0, Math.PI / 2)
  const limb = new THREE.CylinderGeometry(1, 0.7, 1, high ? 6 : 4)
  const mound = new THREE.SphereGeometry(1, high ? 16 : 9, 6, 0, Math.PI * 2, 0, Math.PI / 2)
  const hole = new THREE.CircleGeometry(1, high ? 16 : 9).rotateX(-Math.PI / 2)
  const halo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2)
  return { sphere, dome, limb, mound, hole, halo }
}

function makeMats(ghost) {
  const o = ghost ? { transparent: true, opacity: 0.32, depthWrite: false } : {}
  const std = (p) => {
    const m = new THREE.MeshStandardMaterial({ ...p, ...o, ...(p.transparent ? { opacity: ghost ? 0.2 : p.opacity, depthWrite: false } : {}) })
    m.userData.realistic = true
    return m
  }
  const glow = (c, op) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: ghost ? op * 0.45 : op, depthWrite: false })
  const L = KIND_LOOK
  return {
    skitBody: std({ color: L.skitterer.color, roughness: 0.42, emissive: '#7a3c08', emissiveIntensity: 0.45 }),
    skitDark: std({ color: '#5a3010', roughness: 0.5 }),
    skitEye: glow('#ffe9a8', 1),
    bell: std({ color: L.drifter.color, roughness: 0.15, metalness: 0, emissive: L.drifter.color, emissiveIntensity: 0.7, transparent: true, opacity: 0.5, side: THREE.DoubleSide }),
    core: glow(L.drifter.glow, 0.9),
    tendril: std({ color: L.drifter.glow, roughness: 0.3, emissive: L.drifter.glow, emissiveIntensity: 0.6, transparent: true, opacity: 0.55 }),
    halo: new THREE.MeshBasicMaterial({ color: L.drifter.color, map: glowTexture(), transparent: true, opacity: ghost ? 0.08 : 0.3, depthWrite: false, blending: THREE.AdditiveBlending }),
    shell: std({ color: L.crawler.shell, roughness: 0.35, metalness: 0.15, emissive: '#110f16', emissiveIntensity: 0.5 }),
    ridge: std({ color: L.crawler.color, roughness: 0.55, emissive: '#3a372c', emissiveIntensity: 0.4 }),
    crawlEye: glow('#f6f0c0', 1),
    mound: std({ color: L.burrower.mound, roughness: 0.95 }),
    hole: new THREE.MeshBasicMaterial({ color: '#17120f', transparent: ghost, opacity: ghost ? 0.3 : 1, depthWrite: !ghost }),
    burrow: std({ color: L.burrower.color, roughness: 0.48, emissive: '#4a1f38', emissiveIntensity: 0.45 }),
    burrowDark: std({ color: '#5c3550', roughness: 0.55 }),
    burrowEye: glow('#fff2c0', 1),
  }
}

const MAT_OF = {
  skitterer: { body: 'skitBody', head: 'skitDark', leg: 'skitDark', eye: 'skitEye' },
  drifter: { bell: 'bell', core: 'core', tendril: 'tendril', halo: 'halo' },
  crawler: { shell: 'shell', ridge: 'ridge', head: 'shell', leg: 'shell', eye: 'crawlEye' },
  burrower: { mound: 'mound', hole: 'hole', seg: 'burrowDark', head: 'burrow', eye: 'burrowEye' },
}
const GEO_OF = { body: 'sphere', head: 'sphere', leg: 'limb', eye: 'sphere', bell: 'dome', core: 'sphere', tendril: 'limb', halo: 'halo', shell: 'dome', ridge: 'dome', mound: 'mound', hole: 'hole', seg: 'sphere' }

function makeSet(geos, ghost) {
  const mats = makeMats(ghost)
  const kinds = {}
  for (const [kind, parts] of Object.entries(PARTS)) {
    const meshes = {}
    for (const [part, per] of Object.entries(parts)) {
      const mesh = new THREE.InstancedMesh(geos[GEO_OF[part]], mats[MAT_OF[kind][part]], MAXK * per)
      mesh.count = 0
      mesh.frustumCulled = false
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
      mesh.castShadow = !ghost && !DETAIL.has(part) && part !== 'hole' && part !== 'halo'
      mesh.receiveShadow = !ghost && part !== 'halo'
      if (ghost || part === 'bell' || part === 'tendril') mesh.renderOrder = 5
      if (part === 'halo') mesh.renderOrder = 3
      meshes[part] = { mesh, per, used: 0 }
    }
    kinds[kind] = { meshes, n: 0 }
  }
  return { kinds, mats }
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
const _d = new THREE.Vector3()
const UP = new THREE.Vector3(0, 1, 0)
const NOQ = new THREE.Quaternion()

function put(part, pos, quat, scale) {
  if (part.used >= MAXK * part.per) return
  _L.compose(pos, quat, scale)
  _M.multiplyMatrices(_G, _L)
  part.mesh.setMatrixAt(part.used++, _M)
}

function limb(part, a, b, w) {
  _d.subVectors(b, a)
  const len = _d.length() || 1e-4
  _q.setFromUnitVectors(UP, _d.multiplyScalar(1 / len))
  _p.addVectors(a, b).multiplyScalar(0.5)
  put(part, _p, _q, _s.set(w, len, w))
}

function frame(x, y, z, yaw) {
  _qy.setFromAxisAngle(UP, yaw)
  _G.compose(_p.set(x, z, -y), _qy, _s.set(1, 1, 1))
}

function writeSkitterer(m, p, S, t, high) {
  frame(p.x, p.y, p.gz, p.yaw)
  const move = Math.min(1, p.spd / 0.3)
  const g = skitterGait(p.phase, 0)
  const hz = 0.3 * S + g.jitter * S * move
  put(m.body, _p.set(-0.05 * S, hz, 0), NOQ, _s.set(0.5 * S, 0.22 * S, 0.32 * S))
  put(m.head, _p.set(0.42 * S, hz + 0.02 * S, 0), NOQ, _s.set(0.17 * S, 0.14 * S, 0.17 * S))
  if (!high) return
  for (let i = 0; i < 3; i++) {
    for (const side of [-1, 1]) {
      const sw = skitterGait(p.phase, i * 2.1 + (side > 0 ? Math.PI : 0)).swing * move
      _a.set((0.22 - i * 0.24) * S, hz, side * 0.18 * S)
      _b.set((0.3 - i * 0.26 + 0.12 * sw) * S, 0.01, side * 0.6 * S)
      limb(m.leg, _a, _b, 0.035 * S)
    }
  }
  for (const side of [-1, 1]) put(m.eye, _p.set(0.55 * S, hz + 0.07 * S, side * 0.07 * S), NOQ, _s.setScalar(0.045 * S))
}

function writeDrifter(m, p, S, t, high) {
  const pl = reducedMotion ? { squeeze: 1, bob: 0, sway: 0 } : drifterPulse(t, p.seed)
  const y0 = p.z + pl.bob
  frame(p.x, p.y, y0, p.yaw)
  const r = 0.5 * S
  put(m.bell, _p.set(0, -0.05 * r, 0), NOQ, _s.set(r * pl.squeeze, 0.75 * r * (2 - pl.squeeze), r * pl.squeeze))
  put(m.core, _p.set(0, 0.12 * r, 0), NOQ, _s.setScalar(0.32 * r))
  const hover = Math.max(0.6, y0 - p.gz)
  if (high) {
    for (let k = 0; k < 6; k++) {
      const a = (k / 6) * Math.PI * 2 + 0.3
      const len = hover * (0.5 + 0.08 * k)
      _a.set(0.6 * r * Math.cos(a), -0.05 * r, 0.6 * r * Math.sin(a))
      _b.set(0.8 * r * Math.cos(a) + pl.sway * Math.sin(t * 0.8 + k) * S, -len, 0.8 * r * Math.sin(a) + pl.sway * Math.cos(t * 0.6 + k) * S)
      limb(m.tendril, _a, _b, 0.025 * S)
    }
    // Soft light on the ground under it.
    frame(p.x, p.y, p.gz + 0.04, 0)
    put(m.halo, _p.set(0, 0, 0), NOQ, _s.set(2.6 * S, 1, 2.6 * S))
  }
}

function writeCrawler(m, p, S, t, high) {
  frame(p.x, p.y, p.gz, p.yaw)
  const move = Math.min(1, p.spd / 0.1)
  const sway = reducedMotion ? 0 : 0.01 * S * Math.sin(p.phase) * move
  put(m.shell, _p.set(0, 0.04 * S + sway, 0), NOQ, _s.set(0.5 * S, 0.38 * S, 0.37 * S))
  for (let k = 0; k < 4; k++) {
    const x = (-0.3 + k * 0.2) * S
    const w = Math.sqrt(Math.max(0.05, 1 - (x / (0.5 * S)) ** 2))
    put(m.ridge, _p.set(x, 0.045 * S + sway, 0), NOQ, _s.set(0.035 * S, 0.385 * S * w, 0.375 * S * w))
  }
  put(m.head, _p.set(0.5 * S, 0.02 * S, 0), NOQ, _s.set(0.13 * S, 0.1 * S, 0.15 * S))
  if (!high) return
  for (let i = 0; i < 3; i++) {
    for (const side of [-1, 1]) {
      const ph = p.phase * 0.5 + i * 1.7 + (side > 0 ? Math.PI : 0)
      _a.set((0.25 - i * 0.25) * S, 0.08 * S, side * 0.3 * S)
      _b.set((0.28 - i * 0.25 + 0.05 * Math.sin(ph) * move) * S, 0.0, side * 0.45 * S)
      limb(m.leg, _a, _b, 0.05 * S)
    }
  }
  for (const side of [-1, 1]) put(m.eye, _p.set(0.6 * S, 0.07 * S, side * 0.06 * S), NOQ, _s.setScalar(0.025 * S))
}

function writeBurrower(m, p, S, t, high) {
  frame(p.x, p.y, p.gz - 0.02, p.yaw)
  put(m.mound, _p.set(0, 0, 0), NOQ, _s.set(0.55 * S, 0.16 * S, 0.55 * S))
  put(m.hole, _p.set(0, 0.162 * S, 0), NOQ, _s.setScalar(0.22 * S))
  const lift = burrowLift(p.up)
  if (lift <= 0.02) return
  const h = 0.95 * S * lift
  const look = reducedMotion ? 0 : 0.12 * Math.sin(t * 1.7 + p.seed)
  for (let k = 0; k < 3; k++) {
    const y = 0.12 * S + (k + 0.5) * (h / 3.6)
    const r = (0.22 - 0.02 * k) * S
    put(m.seg, _p.set(Math.sin(look) * 0.03 * S * k, y, 0), NOQ, _s.set(r, 0.12 * S * lift + 0.01, r))
  }
  const hy = 0.12 * S + h
  _qy.setFromAxisAngle(UP, look)
  put(m.head, _p.set(0.03 * S, hy, 0), _qy, _s.set(0.25 * S, 0.22 * S, 0.24 * S))
  if (high) for (const side of [-1, 1]) put(m.eye, _p.set(0.2 * S * Math.cos(look) + 0.03 * S, hy + 0.06 * S, side * 0.1 * S - 0.2 * S * Math.sin(look)), NOQ, _s.setScalar(0.05 * S))
}

const WRITE = { skitterer: writeSkitterer, drifter: writeDrifter, crawler: writeCrawler, burrower: writeBurrower }
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a))

function begin(set) {
  for (const k of Object.values(set.kinds)) for (const part of Object.values(k.meshes)) part.used = 0
}
function finish(set) {
  for (const k of Object.values(set.kinds))
    for (const part of Object.values(k.meshes)) {
      part.mesh.count = part.used
      if (part.used) part.mesh.instanceMatrix.needsUpdate = true
    }
}

export function LifeKinds({ reality }) {
  const graphics = useGraphics()
  const high = graphics === 'high'
  const geos = useMemo(() => makeGeos(high), [high])
  const sets = useMemo(() => ({ solid: makeSet(geos, false), ghost: makeSet(geos, true) }), [geos])
  const poses = useRef(new Map())
  const gen = useSim((s) => s.gen)
  useEffect(() => poses.current.clear(), [gen])
  useEffect(
    () => () => {
      for (const set of Object.values(sets)) {
        Object.values(set.mats).forEach((m) => m.dispose())
        for (const k of Object.values(set.kinds)) for (const part of Object.values(k.meshes)) part.mesh.dispose()
      }
      Object.values(geos).forEach((g) => g.dispose())
    },
    [sets, geos],
  )

  useFrame((_, dt) => {
    const s = simStore.get()
    const now = nowSeconds()
    const T = s.terrain
    begin(sets.solid)
    begin(sets.ghost)
    const a = 1 - Math.exp(-Math.min(dt, 0.1) * 5)
    for (const life of s.lifeforms) {
      const write = WRITE[life.kind]
      if (!write) continue
      let tracked = false
      for (const k of s.tracks) {
        if (k.status === 'tracking' && Math.abs(k.x - life.x) < TRACK_NEAR && Math.abs(k.y - life.y) < TRACK_NEAR && Math.hypot(k.x - life.x, k.y - life.y) < TRACK_NEAR) {
          tracked = true
          break
        }
      }
      let p = poses.current.get(life.id)
      if (!p || Math.hypot(life.x - p.x, life.y - p.y) > 6) {
        p = { x: life.x, y: life.y, yaw: life.yaw, gz: null, z: life.z, up: life.up, phase: 0, spd: 0, seed: (hashStr(life.id) % 628) / 100, at: now }
        poses.current.set(life.id, p)
      }
      const ox = p.x
      const oy = p.y
      p.x += (life.x - p.x) * a
      p.y += (life.y - p.y) * a
      p.yaw += wrap(life.yaw - p.yaw) * a
      const moved = Math.hypot(p.x - ox, p.y - oy)
      p.spd += (moved / Math.max(dt, 1e-3) - p.spd) * Math.min(1, dt * 4)
      p.phase += (moved / (0.3 * life.size)) * Math.PI
      p.at = now
      const g = sampleGround(T, p.x, p.y) ?? groundAt(p.x, p.y)
      p.gz = p.gz === null ? g : p.gz + (g - p.gz) * Math.min(1, a * 1.5)
      if (Number.isFinite(life.z)) p.z = Number.isFinite(p.z) ? p.z + (life.z - p.z) * a : life.z
      else p.z = p.gz + 2
      const up = Number.isFinite(life.up) ? life.up : 1
      p.up = Number.isFinite(p.up) ? p.up + (up - p.up) * Math.min(1, a * 2) : up
      // A burrower that is down shows only its mound, and only with Reality on or a track near.
      const set = tracked ? sets.solid : reality ? sets.ghost : null
      if (!set) continue
      const k = set.kinds[life.kind]
      if (k.meshes[Object.keys(k.meshes)[0]].used >= MAXK) continue
      write(k.meshes, p, life.size, now, high)
    }
    if (poses.current.size > s.lifeforms.length + 8) for (const [id, p] of poses.current) if (now - p.at > 2) poses.current.delete(id)
    finish(sets.solid)
    finish(sets.ghost)
  })

  return (
    <group>
      {Object.entries(sets).map(([name, set]) => (
        <group key={name}>
          {Object.entries(set.kinds).flatMap(([kind, k]) => Object.entries(k.meshes).map(([part, { mesh }]) => <primitive key={`${kind}-${part}`} object={mesh} />))}
        </group>
      ))}
    </group>
  )
}
