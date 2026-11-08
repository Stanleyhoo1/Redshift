import * as THREE from 'three'
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'
import { panelTexture, solarTexture } from './proceduralTextures.js'
import { skyEnvironment } from './sky.js'

// Shared materials and geometry helpers for the rovers and the lander: painted panels,
// anodised dark metal, bright machined metal, gold foil (thermal blankets), solar cells,
// glass optics and dusty tyres. All physically based and reflecting the hazy sky.

let mats = null

export function hardwareMaterials() {
  if (mats) return mats
  const env = skyEnvironment()
  const std = (o) => {
    const m = new THREE.MeshStandardMaterial({ envMap: env, ...o })
    m.userData.realistic = true
    return m
  }
  mats = {
    paint: std({ color: '#dcd8d0', map: panelTexture(), roughness: 0.55, metalness: 0.08, envMapIntensity: 0.7, vertexColors: true }),
    dark: std({ color: '#2a2c31', roughness: 0.45, metalness: 0.65, envMapIntensity: 0.9 }),
    metal: std({ color: '#a3a7ae', roughness: 0.3, metalness: 0.9, envMapIntensity: 1.0 }),
    gold: std({ color: '#d9a94e', roughness: 0.34, metalness: 1.0, envMapIntensity: 1.2 }),
    solar: std({ color: '#ffffff', map: solarTexture(), roughness: 0.2, metalness: 0.45, envMapIntensity: 1.4 }),
    glass: std({ color: '#0b0e14', roughness: 0.06, metalness: 0.3, envMapIntensity: 1.6 }),
    tyre: std({ color: '#3d362f', roughness: 0.95, metalness: 0.0, envMapIntensity: 0.3 }),
    lamp: new THREE.MeshBasicMaterial({ color: '#fff4e0' }),
  }
  return mats
}

// UVs in metres / scale from a box projection (dominant normal axis), so panel textures
// tile at the same size on every part.
export function boxUV(g, scale = 0.4) {
  const p = g.attributes.position
  const n = g.attributes.normal
  const uv = new Float32Array(p.count * 2)
  for (let i = 0; i < p.count; i++) {
    const ax = Math.abs(n.getX(i))
    const ay = Math.abs(n.getY(i))
    const az = Math.abs(n.getZ(i))
    let u
    let v
    if (ax >= ay && ax >= az) {
      u = p.getZ(i)
      v = p.getY(i)
    } else if (ay >= az) {
      u = p.getX(i)
      v = p.getZ(i)
    } else {
      u = p.getX(i)
      v = p.getY(i)
    }
    uv[i * 2] = u / scale + 0.5
    uv[i * 2 + 1] = v / scale + 0.5
  }
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  return g
}

// Vertex colours fading towards dust near the bottom (y below `low`, in final units).
export function grime(g, low, high, dust = '#9c8471', amount = 0.55) {
  const p = g.attributes.position
  const col = new Float32Array(p.count * 3)
  const d = new THREE.Color(dust)
  const c = new THREE.Color()
  for (let i = 0; i < p.count; i++) {
    const t = Math.min(1, Math.max(0, (high - p.getY(i)) / Math.max(1e-3, high - low)))
    c.setRGB(1, 1, 1).lerp(d, t * t * amount)
    col[i * 3] = c.r
    col[i * 3 + 1] = c.g
    col[i * 3 + 2] = c.b
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3))
  return g
}

// Keep only position, normal and uv (plus color when all have it), then merge.
export function merge(list) {
  const withColor = list.every((g) => g.attributes.color)
  const clean = list.map((g) => {
    const h = g.index ? g.toNonIndexed() : g.clone()
    for (const k of Object.keys(h.attributes)) if (k !== 'position' && k !== 'normal' && k !== 'uv' && !(withColor && k === 'color')) h.deleteAttribute(k)
    if (!h.attributes.uv) h.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(h.attributes.position.count * 2), 2))
    return h
  })
  const out = mergeGeometries(clean, false)
  clean.forEach((g) => g.dispose())
  out.computeBoundingSphere()
  return out
}

const _m = new THREE.Matrix4()
const _q = new THREE.Quaternion()
const _up = new THREE.Vector3(0, 1, 0)
const _d = new THREE.Vector3()

// A cylinder from a to b (Vector3), radius r (r2 at b).
export function strut(a, b, r, r2 = r, seg = 8) {
  _d.subVectors(b, a)
  const len = _d.length()
  const g = new THREE.CylinderGeometry(r2, r, len, seg)
  _q.setFromUnitVectors(_up, _d.normalize())
  _m.compose(new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5), _q, new THREE.Vector3(1, 1, 1))
  g.applyMatrix4(_m)
  return g
}

// Place a geometry (position, Euler rotation in radians).
export function place(g, x, y, z, rx = 0, ry = 0, rz = 0) {
  g.applyMatrix4(_m.compose(new THREE.Vector3(x, y, z), _q.setFromEuler(new THREE.Euler(rx, ry, rz)), new THREE.Vector3(1, 1, 1)))
  return g
}
