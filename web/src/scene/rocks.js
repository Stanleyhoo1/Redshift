import * as THREE from 'three'
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js'
import { REGOLITH } from '../lib/theme.js'
import { mulberry, regolithTextures } from './proceduralTextures.js'

// Irregular rocks: a subdivided icosahedron pushed out by 3D noise, cut by a few random
// planes (fractured faces), flattened underneath, with vertex colours for a dusty top,
// darker crevices and mineral variation. The material adds a triplanar detail bump in
// object space, so the grain scales with each rock.

function valueNoise3(seed) {
  const rand = mulberry(seed)
  const N = 256
  const perm = new Uint8Array(N * 2)
  const vals = new Float32Array(N)
  for (let k = 0; k < N; k++) {
    perm[k] = k
    vals[k] = rand() * 2 - 1
  }
  for (let k = N - 1; k > 0; k--) {
    const j = Math.floor(rand() * (k + 1))
    const t = perm[k]
    perm[k] = perm[j]
    perm[j] = t
  }
  for (let k = 0; k < N; k++) perm[N + k] = perm[k]
  const h = (x, y, z) => vals[perm[perm[perm[x & 255] + (y & 255)] + (z & 255)]]
  const f = (t) => t * t * (3 - 2 * t)
  return (x, y, z) => {
    const xi = Math.floor(x)
    const yi = Math.floor(y)
    const zi = Math.floor(z)
    const u = f(x - xi)
    const v = f(y - yi)
    const w = f(z - zi)
    const l = (a, b, t) => a + (b - a) * t
    return l(
      l(l(h(xi, yi, zi), h(xi + 1, yi, zi), u), l(h(xi, yi + 1, zi), h(xi + 1, yi + 1, zi), u), v),
      l(l(h(xi, yi, zi + 1), h(xi + 1, yi, zi + 1), u), l(h(xi, yi + 1, zi + 1), h(xi + 1, yi + 1, zi + 1), u), v),
      w,
    )
  }
}

const base = new THREE.Color(REGOLITH.rock)
const light = new THREE.Color(REGOLITH.rockLight)
const dusty = new THREE.Color(REGOLITH.dust)

// detail: icosahedron subdivision (1 for gravel, 3 for boulders). Unit size, base near y = -0.5.
export function rockGeometry(seed, detail = 3, { cuts = 4, squash = 0.75, tint = null } = {}) {
  let g = new THREE.IcosahedronGeometry(1, detail)
  g.deleteAttribute('normal')
  g.deleteAttribute('uv')
  g = mergeVertices(g)
  const rand = mulberry(seed * 7919 + 13)
  const n1 = valueNoise3(seed * 31 + 1)
  const n2 = valueNoise3(seed * 31 + 2)
  const planes = Array.from({ length: cuts }, () => {
    const v = new THREE.Vector3(rand() * 2 - 1, rand() * 1.4 - 0.4, rand() * 2 - 1).normalize()
    return { n: v, d: 0.62 + rand() * 0.3 }
  })
  const sx = 0.85 + rand() * 0.35
  const sz = 0.75 + rand() * 0.35
  const sy = squash * (0.85 + rand() * 0.3)
  const p = g.attributes.position
  const col = new Float32Array(p.count * 3)
  const v = new THREE.Vector3()
  const c = new THREE.Color()
  const tintC = tint ? new THREE.Color(tint) : null
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i)
    const o = 1.7
    let r = 1 + 0.26 * n1(v.x * o + 9, v.y * o + 3, v.z * o + 5) + 0.09 * n2(v.x * 4.1, v.y * 4.1, v.z * 4.1)
    for (const pl of planes) {
      const dp = v.dot(pl.n)
      if (dp > 1e-3) r = Math.min(r, pl.d / dp)
    }
    r = Math.max(r, 0.45)
    const bump = r
    v.multiplyScalar(r)
    v.x *= sx
    v.z *= sz
    v.y *= sy
    if (v.y < -0.42 * sy) v.y = -0.42 * sy + (v.y + 0.42 * sy) * 0.15
    p.setXYZ(i, v.x, v.y, v.z)
    // Colour: mineral variation, darker in hollows, a dusty coat on top.
    const m = 0.5 + 0.5 * n2(v.x * 2.3 + 4, v.y * 2.3, v.z * 2.3)
    c.copy(tintC || base).lerp(tintC ? tintC.clone().multiplyScalar(1.5) : light, m * 0.7)
    const hollow = Math.min(1, Math.max(0, (1.08 - bump) * 2.2))
    c.multiplyScalar(1 - hollow * 0.35)
    const top = Math.min(1, Math.max(0, (v.y / Math.max(0.2, sy) - 0.15) * 1.6))
    if (!tintC) c.lerp(dusty, top * 0.45)
    col[i * 3] = c.r
    col[i * 3 + 1] = c.g
    col[i * 3 + 2] = c.b
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3))
  g.computeVertexNormals()
  g.computeBoundingSphere()
  return g
}

const bumpCommonV = /* glsl */ `
varying vec3 vObjPos;
varying vec3 vObjN;
`
const bumpCommonF = /* glsl */ `
uniform sampler2D uRockDetail;
uniform float uBump;
uniform float uGrainScale;
varying vec3 vObjPos;
varying vec3 vObjN;
`
// Triplanar height from the regolith normal map (A = height) and a derivative bump.
const bumpFragment = /* glsl */ `
  {
    vec3 w = pow(abs(normalize(vObjN)), vec3(4.0));
    w /= (w.x + w.y + w.z + 1e-4);
    vec3 q = vObjPos * uGrainScale;
    float hx = texture2D(uRockDetail, q.yz).a;
    float hy = texture2D(uRockDetail, q.zx + 0.37).a;
    float hz = texture2D(uRockDetail, q.xy + 0.71).a;
    float hgt = hx * w.x + hy * w.y + hz * w.z;
    vec3 pos = -vViewPosition;
    vec3 dpx = dFdx(pos);
    vec3 dpy = dFdy(pos);
    float dhx = dFdx(hgt);
    float dhy = dFdy(hgt);
    vec3 r1 = cross(dpy, normal);
    vec3 r2 = cross(normal, dpx);
    float det = dot(dpx, r1);
    vec3 grad = sign(det) * (dhx * r1 + dhy * r2);
    normal = normalize(abs(det) * normal - grad * uBump);
    diffuseColor.rgb *= 0.82 + 0.36 * hgt;
  }
`

const materials = new Map()

// A shared rock material (vertex colours). opts: { color, roughness, metalness, bump, grain, envMap, envMapIntensity, emissive }
export function rockMaterial(key = 'rock', opts = {}) {
  if (materials.has(key)) return materials.get(key)
  const m = new THREE.MeshStandardMaterial({
    color: opts.color ?? '#ffffff',
    vertexColors: opts.vertexColors ?? true,
    roughness: opts.roughness ?? 0.9,
    metalness: opts.metalness ?? 0.02,
    envMap: opts.envMap ?? null,
    envMapIntensity: opts.envMapIntensity ?? 1,
    emissive: opts.emissive ?? '#000000',
    emissiveIntensity: opts.emissiveIntensity ?? 1,
  })
  m.userData.realistic = true
  const uniforms = {
    uRockDetail: { value: regolithTextures().normal },
    uBump: { value: opts.bump ?? 0.035 },
    uGrainScale: { value: opts.grain ?? 0.8 },
  }
  if (opts.bump !== 0) {
    m.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms)
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\n${bumpCommonV}`)
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vObjPos = position;\n  vObjN = normal;')
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\n${bumpCommonF}`)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\n${bumpFragment}`)
    }
    m.customProgramCacheKey = () => 'redshift-rock-1'
  }
  materials.set(key, m)
  return m
}

let variants = null
// Six boulder shapes, shared.
export function boulderVariants() {
  if (!variants) variants = Array.from({ length: 6 }, (_, v) => rockGeometry(101 + v * 17, 3))
  return variants
}
