import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { groundAt, knownAt } from './ground.js'
import { reducedMotion } from './sceneConstants.js'
import { ringFade } from './mapMarkers.js'
import { DrapedDisc, useDrape } from './drape.js'

// v7 geysers (PROTOCOL.md, `features.geysers` and `fleet.geyser_zones`). Each vent is a squat sinter
// mound: a pale ochre base, a white crust at the rim and a dark throat. While a vent erupts a column
// of soft steam puffs billows up to its `height` with its `radius`, and a soft dark wet patch darkens the
// ground (it dries more slowly than the steam fades). The steam grows in from the vent and collapses
// over about a second when `active` flips. Mounds and steam are instanced (two draw calls for every
// vent); each wet patch is a soft disc draped on the ground, drawn only while it is wet; plus the
// dashed keep-out rings the fleet learned. A vent shows once the fleet knows the ground
// there, or with Reality on. Low graphics draws fewer, larger puffs.

const MAX = 32
const PUFFS_HIGH = 26
const PUFFS_LOW = 9
const VENT_R = 1.25       // mound footprint (m), wider than the sim's 0.9 m vent geom
const VENT_H = 0.7        // mound height above the base (m), so it stands clear of the mapped vent bump
const WET_R = 2.0         // wet patch radius, in plume radii
const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0)
const _m = new THREE.Matrix4()
const _p = new THREE.Vector3()
const _s = new THREE.Vector3()
const _q = new THREE.Quaternion()
const _e = new THREE.Euler()
const _c = new THREE.Color()

const frac = (v) => v - Math.floor(v)
const hash = (n) => frac(Math.sin(n * 127.1 + 311.7) * 43758.5453)
const smooth = (t) => t * t * (3 - 2 * t)

// A mound with a crater, turned on a lathe, coloured by vertex: dark throat, white rim, ochre flanks.
function moundGeometry() {
  // Outer foot first, so the lathe faces point outwards.
  const prof = [[1.0, 0.0], [0.78, 0.42], [0.56, 0.82], [0.42, 1.0], [0.34, 0.95], [0.26, 0.62], [0.18, 0.32], [0, 0.3]]
  const geo = new THREE.LatheGeometry(prof.map(([r, h]) => new THREE.Vector2(r, h)), 20)
  const pos = geo.attributes.position
  const col = new Float32Array(pos.count * 3)
  const throat = new THREE.Color(0.16, 0.11, 0.09)
  const rim = new THREE.Color(0.95, 0.92, 0.84)
  const flank = new THREE.Color(0.8, 0.62, 0.4)
  for (let k = 0; k < pos.count; k++) {
    const x = pos.getX(k)
    const z = pos.getZ(k)
    const r = Math.hypot(x, z)
    if (r < 0.3) _c.copy(throat).lerp(rim, smooth(Math.max(0, (r - 0.2) / 0.1)))
    else _c.copy(rim).lerp(flank, smooth(Math.min(1, (r - 0.42) / 0.5)))
    col[k * 3] = _c.r
    col[k * 3 + 1] = _c.g
    col[k * 3 + 2] = _c.b
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
  geo.computeVertexNormals()
  return geo
}

// The ground under the vent from a ring of samples just outside it: the fleet's map sees the vent
// like a rock, so the centre sample can sit on top of it. The mean keeps the mound level on a slope.
function baseAt(x, y) {
  const z0 = groundAt(x, y)
  let z = 0
  for (let k = 0; k < 6; k++) {
    const a = (k / 6) * Math.PI * 2
    z += groundAt(x + 1.5 * Math.cos(a), y + 1.5 * Math.sin(a), z0)
  }
  return z / 6
}

let wetTexture = null
function wetMap() {
  if (wetTexture || typeof document === 'undefined') return wetTexture
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const g = c.getContext('2d')
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32)
  grad.addColorStop(0, 'rgba(255,255,255,1)')
  grad.addColorStop(0.6, 'rgba(255,255,255,0.8)')
  grad.addColorStop(1, 'rgba(255,255,255,0)')
  g.fillStyle = grad
  g.fillRect(0, 0, 64, 64)
  wetTexture = new THREE.CanvasTexture(c)
  return wetTexture
}

// The dark wet ground round one vent, draped on the terrain; its opacity follows the vent's wetness.
function WetPatch({ v, fades, reality }) {
  const disc = useMemo(() => new DrapedDisc(5, 32, { uvMode: 'disc' }), [])
  const mesh = useRef()
  const mat = useRef()
  const R = (Number.isFinite(v.radius) && v.radius > 0 ? v.radius : 1.5) * WET_R
  useDrape(() => disc.update(v.x, v.y, R, 0.06), [v.x, v.y, R])
  useFrame(() => {
    if (!mesh.current || !mat.current) return
    const f = fades.current.get(v.id)
    const w = f ? f.wet : 0
    mesh.current.visible = w > 0.01 && (reality || knownAt(v.x, v.y))
    mat.current.opacity = 0.5 * smooth(w)
  })
  return (
    <mesh ref={mesh} geometry={disc.geometry} frustumCulled={false} renderOrder={1} visible={false}>
      <meshBasicMaterial ref={mat} color="#1a100c" map={wetMap()} transparent opacity={0} depthWrite={false} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
    </mesh>
  )
}

// Drawn only when the camera is close (ringFade), so the overview stays calm.
function Ring({ x, y, r }) {
  const geo = useMemo(() => {
    const pts = []
    for (let k = 0; k <= 64; k++) {
      const a = (k / 64) * Math.PI * 2
      const px = x + r * Math.cos(a)
      const py = y + r * Math.sin(a)
      pts.push(new THREE.Vector3(px, groundAt(px, py) + 0.15, -py))
    }
    return new THREE.BufferGeometry().setFromPoints(pts)
  }, [x, y, r])
  const line = useRef()
  const mat = useRef()
  useFrame(({ camera }) => {
    if (!line.current || !mat.current) return
    const f = ringFade(Math.hypot(camera.position.x - x, camera.position.y - groundAt(x, y), camera.position.z + y))
    line.current.visible = f > 0.01
    mat.current.opacity = 0.85 * f
  })
  return (
    <line ref={line} geometry={geo} visible={false} onUpdate={(l) => l.computeLineDistances()}>
      <lineDashedMaterial ref={mat} color="#f6e7cf" dashSize={0.9} gapSize={0.6} transparent opacity={0} />
    </line>
  )
}

export function Geysers({ reality }) {
  const vents = useSim((s) => s.features?.geysers)
  const zones = useSim((s) => s.fleet?.geyser_zones)
  const graphics = useGraphics()
  const mound = useRef()
  const steam = useRef()
  const fades = useRef(new Map())
  const list = Array.isArray(vents) ? vents.filter((v) => v && Number.isFinite(v.x) && Number.isFinite(v.y)).slice(0, MAX) : []
  const geos = useMemo(() => ({
    mound: moundGeometry(),
    puff: new THREE.IcosahedronGeometry(1, 1),
  }), [])
  const puffs = graphics === 'low' ? PUFFS_LOW : PUFFS_HIGH

  useFrame(({ clock }, dt) => {
    if (!mound.current || !steam.current) return
    const t = reducedMotion ? 0 : clock.elapsedTime
    const step = Math.min(dt, 1) // wall time, so the fade takes about a second at any frame rate
    const F = fades.current
    let np = 0
    for (let k = 0; k < MAX; k++) {
      const v = list[k]
      const show = v && (reality || knownAt(v.x, v.y))
      if (!show) {
        mound.current.setMatrixAt(k, HIDDEN)
        continue
      }
      // Steam fades over about a second; the wet ground dries over about six.
      let f = F.get(v.id)
      if (!f) F.set(v.id, (f = { steam: 0, wet: 0 }))
      if (v.active) {
        f.steam = Math.min(1, f.steam + step)
        f.wet = Math.min(1, f.wet + step * 0.8)
      } else {
        f.steam = Math.max(0, f.steam - step)
        f.wet = Math.max(0, f.wet - step / 6)
      }
      const gz = baseAt(v.x, v.y)
      _m.compose(_p.set(v.x, gz - 0.12, -v.y), _q.identity(), _s.set(VENT_R, VENT_H, VENT_R))
      mound.current.setMatrixAt(k, _m)
      const R = Number.isFinite(v.radius) && v.radius > 0 ? v.radius : 1.5
      if (f.steam <= 0.001) continue
      // Puffs rise through the column and loop; each is hidden above the risen part while the
      // eruption builds, swells as it climbs and thins out near the top.
      const H = Number.isFinite(v.height) && v.height > 0 ? v.height : 8
      const fs = smooth(f.steam)
      const seed = (k + 1) * 17.3
      const size = graphics === 'low' ? 1.35 : 1
      for (let i = 0; i < puffs; i++) {
        const h1 = hash(seed + i)
        const h2 = hash(seed + i * 3.7 + 1)
        const u = frac(t / (2.2 + 0.8 * h2) + i / puffs + h1 * 0.3)
        if (u > fs) continue
        const rise = u * (H - VENT_H)
        const spread = R * (0.25 + 0.75 * u)
        const a = h1 * Math.PI * 2 + t * (0.4 + 0.3 * h2)
        const off = spread * 0.55 * h2
        const grow = (0.4 + 0.6 * Math.min(1, u / 0.15)) * (u > 0.8 ? 1 - (u - 0.8) / 0.2 : 1)
        const s = size * fs * grow * R * (0.45 + 0.75 * u)
        _e.set(h1 * 3, h2 * 3 + t * 0.2, 0)
        _m.compose(_p.set(v.x + off * Math.cos(a), gz + VENT_H * 0.8 + rise, -(v.y + off * Math.sin(a))), _q.setFromEuler(_e), _s.set(s, s * 0.85, s))
        steam.current.setMatrixAt(np, _m)
        const shade = 0.82 + 0.18 * h2 - 0.08 * u
        steam.current.setColorAt(np, _c.setRGB(shade, shade, shade * 0.98))
        np++
      }
    }
    for (let k = np; k < MAX * PUFFS_HIGH; k++) steam.current.setMatrixAt(k, HIDDEN)
    mound.current.instanceMatrix.needsUpdate = true
    steam.current.instanceMatrix.needsUpdate = true
    if (steam.current.instanceColor) steam.current.instanceColor.needsUpdate = true
  })

  if (!list.length) return null
  return (
    <group>
      <instancedMesh ref={mound} args={[geos.mound, undefined, MAX]} frustumCulled={false}>
        <meshStandardMaterial vertexColors roughness={0.95} />
      </instancedMesh>
      {list.map((v) => <WetPatch key={v.id} v={v} fades={fades} reality={reality} />)}
      <instancedMesh ref={steam} args={[geos.puff, undefined, MAX * PUFFS_HIGH]} frustumCulled={false} renderOrder={2}>
        <meshLambertMaterial color="#ffffff" emissive="#c9c4be" transparent opacity={0.26} depthWrite={false} />
      </instancedMesh>
      {Array.isArray(zones) && zones.filter((z) => Array.isArray(z) && z.length >= 3 && z.every(Number.isFinite)).map((z, i) => (
        <Ring key={i} x={z[0]} y={z[1]} r={z[2]} />
      ))}
    </group>
  )
}
