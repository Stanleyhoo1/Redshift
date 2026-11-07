import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore, useSim, nowSeconds, roverPoses } from '../state/store.js'
import { SIGNAL, SLOPE, GROUND_RAMP } from '../lib/theme.js'
import { SUN_DIR, reducedMotion } from './sceneConstants.js'

// World (x, y, z-up) maps to three.js (x, z, -y) everywhere in the scene.
// The terrain is one indexed grid mesh with nx * ny vertices. Unknown samples are
// discarded in the fragment shader (the dark survey grid below shows through), and
// newly known samples rise into place with a short cyan flash.

const vertexShader = /* glsl */ `
attribute float aSlope;
attribute float aKnown;
attribute float aReveal;
uniform float uTime;
uniform float uMotion;
varying float vSlope;
varying float vKnown;
varying float vFlash;
varying float vH;
varying vec3 vNormal;
varying vec3 vPos;

void main() {
  float age = uTime - aReveal;
  float rise = aKnown > 0.5 ? smoothstep(0.0, 0.7, age) : 1.0;
  vec3 p = position;
  p.y -= (1.0 - rise) * 1.1 * uMotion;
  vSlope = aSlope;
  vKnown = aKnown;
  vFlash = aKnown > 0.5 ? exp(-max(age, 0.0) * 1.3) : 0.0;
  vH = position.y;
  vNormal = normal;
  vPos = p;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}
`

const fragmentShader = /* glsl */ `
uniform float uTime;
uniform float uLimit;
uniform float uMode;
uniform float uMinH;
uniform float uMaxH;
uniform float uMotion;
uniform vec3 uSunDir;
uniform vec3 uSignal;
uniform vec3 uOk;
uniform vec3 uCaution;
uniform vec3 uBad;
uniform vec3 uRamp[5];
uniform vec2 uRovers[8];
uniform int uRoverCount;
varying float vSlope;
varying float vKnown;
varying float vFlash;
varying float vH;
varying vec3 vNormal;
varying vec3 vPos;

vec3 ramp(float t) {
  t = clamp(t, 0.0, 1.0) * 4.0;
  if (t < 1.0) return mix(uRamp[0], uRamp[1], t);
  if (t < 2.0) return mix(uRamp[1], uRamp[2], t - 1.0);
  if (t < 3.0) return mix(uRamp[2], uRamp[3], t - 2.0);
  return mix(uRamp[3], uRamp[4], t - 3.0);
}

void main() {
  if (vKnown < 0.5) discard;
  vec3 n = normalize(vNormal);
  float ht = clamp((vH - uMinH) / max(uMaxH - uMinH, 0.1), 0.0, 1.0);

  // Slope relative to the fleet's current limit: comfortable below 60 %, caution
  // from 60 % up to the limit, impassable at or above it. The antialiasing band
  // sits just below each threshold, so a sample at the limit is always red.
  float r = vSlope / max(uLimit, 1.0);
  float aa = clamp(fwidth(r), 1e-4, 0.04);
  vec3 slopeCol = mix(uOk, uCaution, smoothstep(0.6 - aa, 0.6, r));
  slopeCol = mix(slopeCol, uBad, smoothstep(1.0 - aa, 1.0, r));
  vec3 base = uMode > 0.5 ? slopeCol * (0.84 + 0.3 * ht) : ramp(ht);

  float ndl = max(dot(n, uSunDir), 0.0);
  vec3 col = base * (0.42 + 0.68 * ndl);

  // Contour lines: thin every 1 m, stronger every 5 m, faded where they would crowd.
  float w = max(fwidth(vH), 1e-4);
  float d1 = abs(fract(vH + 0.5) - 0.5);
  float d5 = abs(fract(vH / 5.0 + 0.5) - 0.5) * 5.0;
  float crowd = 1.0 - smoothstep(0.1, 0.32, w);
  float lines = max((1.0 - smoothstep(0.0, 1.2 * w, d1)) * 0.2, (1.0 - smoothstep(0.0, 1.6 * w, d5)) * 0.38) * crowd;
  col *= 1.0 - lines;

  // Sensor pulses spreading from each rover.
  float pulse = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= uRoverCount) break;
    float dist = length(vPos.xz - uRovers[i]);
    float ph = fract(uTime * 0.4 + float(i) * 0.37);
    float rad = ph * 10.0;
    float fall = 1.0 - ph;
    pulse += exp(-pow((dist - rad) / 0.3, 2.0)) * fall * fall;
  }
  col = mix(col, uSignal, clamp(pulse * uMotion, 0.0, 1.0) * 0.4);

  // Reveal flash, and a bright rim along the edge of what the fleet knows.
  col = mix(col, uSignal, clamp(vFlash, 0.0, 1.0) * 0.75);
  float rim = 1.0 - smoothstep(0.5, 0.85, vKnown);
  col = mix(col, uSignal, rim * 0.9);

  gl_FragColor = vec4(col, 1.0);
}
`

// Linear working-space colour, like every other three.js material in the scene.
const toVec3 = (hex) => {
  const c = new THREE.Color(hex)
  return new THREE.Vector3(c.r, c.g, c.b)
}

// The shader writes its colour straight to the screen (no output encoding), so the
// palette goes in as sRGB values: the terrain then shows exactly the theme colours,
// the same ones the legend swatches use.
const display3 = (hex) => {
  const c = new THREE.Color(hex).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace)
  return new THREE.Vector3(c.r, c.g, c.b)
}

function createMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
    uniforms: {
      uTime: { value: 0 },
      uLimit: { value: 30 },
      uMode: { value: 1 },
      uMinH: { value: -3 },
      uMaxH: { value: 8 },
      uMotion: { value: 1 },
      uSunDir: { value: new THREE.Vector3(...SUN_DIR).normalize() },
      uSignal: { value: toVec3(SIGNAL) },
      uOk: { value: display3(SLOPE.ok) },
      uCaution: { value: display3(SLOPE.caution) },
      uBad: { value: display3(SLOPE.bad) },
      uRamp: { value: GROUND_RAMP.map(display3) },
      uRovers: { value: Array.from({ length: 8 }, () => new THREE.Vector2(1e5, 1e5)) },
      uRoverCount: { value: 0 },
    },
  })
}

// Owns the geometry for one terrain (one init) and keeps it in sync with the
// sample arrays, processing only the samples that changed.
class TerrainMesh {
  constructor(T) {
    this.T = T
    const { nx, ny, res, ox, oy } = T
    const n = nx * ny
    this.fogY = T.minH - 0.8
    this.pos = new Float32Array(n * 3)
    this.nor = new Float32Array(n * 3)
    this.slope = new Float32Array(n)
    this.known = new Float32Array(n)
    this.reveal = new Float32Array(n)
    this.stamp = new Uint32Array(n)
    this.touched = new Int32Array(n)
    this.frame = 0
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i
        this.pos[k * 3] = ox + i * res
        this.pos[k * 3 + 1] = this.fogY
        this.pos[k * 3 + 2] = -(oy + j * res)
        this.nor[k * 3 + 1] = 1
      }
    }
    const quads = (nx - 1) * (ny - 1)
    const index = n > 65535 ? new Uint32Array(quads * 6) : new Uint16Array(quads * 6)
    let q = 0
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = j * nx + i
        const b = a + 1
        const c = a + nx
        const d = c + 1
        index[q++] = a
        index[q++] = b
        index[q++] = c
        index[q++] = b
        index[q++] = d
        index[q++] = c
      }
    }
    const geo = new THREE.BufferGeometry()
    const dyn = (arr, size) => new THREE.BufferAttribute(arr, size).setUsage(THREE.DynamicDrawUsage)
    this.attrs = {
      position: dyn(this.pos, 3),
      normal: dyn(this.nor, 3),
      aSlope: dyn(this.slope, 1),
      aKnown: dyn(this.known, 1),
      aReveal: dyn(this.reveal, 1),
    }
    for (const [name, attr] of Object.entries(this.attrs)) geo.setAttribute(name, attr)
    geo.setIndex(new THREE.BufferAttribute(index, 1))
    const half = Math.hypot(T.width, T.depth) / 2
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(T.cx, (T.minH + T.maxH) / 2, -T.cy), half + 10)
    this.geometry = geo
    this.syncAll()
  }

  provisional(k) {
    // Height for an unknown vertex next to known ones: the mean of its known neighbours,
    // so triangles along the edge of the known area stay flat instead of forming walls.
    const { nx, ny, known, h } = this.T
    const i = k % nx
    const j = (k / nx) | 0
    let sum = 0
    let count = 0
    for (let dj = -1; dj <= 1; dj++) {
      const jj = j + dj
      if (jj < 0 || jj >= ny) continue
      for (let di = -1; di <= 1; di++) {
        const ii = i + di
        if (ii < 0 || ii >= nx || (!di && !dj)) continue
        const m = jj * nx + ii
        if (known[m]) {
          sum += h[m]
          count++
        }
      }
    }
    return count ? sum / count : this.fogY
  }

  normalAt(k) {
    const { nx, ny, res } = this.T
    const i = k % nx
    const j = (k / nx) | 0
    const il = i > 0 ? k - 1 : k
    const ir = i < nx - 1 ? k + 1 : k
    const jd = j > 0 ? k - nx : k
    const ju = j < ny - 1 ? k + nx : k
    const p = this.pos
    const hx = (p[ir * 3 + 1] - p[il * 3 + 1]) / (((ir - il) || 1) * res)
    const hy = (p[ju * 3 + 1] - p[jd * 3 + 1]) / ((((ju - jd) / nx) || 1) * res)
    // world normal (-hx, -hy, 1) in three.js coordinates is (-hx, 1, hy)
    const len = Math.hypot(hx, 1, hy)
    this.nor[k * 3] = -hx / len
    this.nor[k * 3 + 1] = 1 / len
    this.nor[k * 3 + 2] = hy / len
  }

  writeKnown(k) {
    const T = this.T
    this.pos[k * 3 + 1] = T.h[k]
    this.slope[k] = T.slope[k]
    this.known[k] = 1
    this.reveal[k] = T.revealAt[k]
  }

  flagAll() {
    for (const attr of Object.values(this.attrs)) attr.needsUpdate = true
  }

  syncAll() {
    const T = this.T
    const n = T.nx * T.ny
    for (let k = 0; k < n; k++) if (T.known[k]) this.writeKnown(k)
    for (let k = 0; k < n; k++) if (!T.known[k]) this.pos[k * 3 + 1] = this.provisional(k)
    for (let k = 0; k < n; k++) this.normalAt(k)
    for (let q = 0; q < T.dirtyCount; q++) T.dirtyFlag[T.dirty[q]] = 0
    T.dirtyCount = 0
    this.flagAll()
  }

  drain() {
    const T = this.T
    if (!T.dirtyCount) return false
    const { nx, ny } = T
    this.frame = (this.frame + 1) >>> 0 || 1
    const frame = this.frame
    let tc = 0
    for (let q = 0; q < T.dirtyCount; q++) {
      const k = T.dirty[q]
      T.dirtyFlag[k] = 0
      this.writeKnown(k)
      const i = k % nx
      const j = (k / nx) | 0
      for (let dj = -2; dj <= 2; dj++) {
        const jj = j + dj
        if (jj < 0 || jj >= ny) continue
        for (let di = -2; di <= 2; di++) {
          const ii = i + di
          if (ii < 0 || ii >= nx) continue
          const m = jj * nx + ii
          if (Math.abs(di) <= 1 && Math.abs(dj) <= 1 && !T.known[m]) this.pos[m * 3 + 1] = this.provisional(m)
          if (this.stamp[m] !== frame) {
            this.stamp[m] = frame
            this.touched[tc++] = m
          }
        }
      }
    }
    T.dirtyCount = 0
    for (let t = 0; t < tc; t++) this.normalAt(this.touched[t])
    this.flagAll()
    return true
  }

  dispose() {
    this.geometry.dispose()
  }
}

export function Terrain({ slopeColors }) {
  const terrain = useSim((s) => s.terrain)
  const roverMeta = useSim((s) => s.roverMeta)
  const material = useMemo(createMaterial, [])
  const mesh = useMemo(() => (terrain ? new TerrainMesh(terrain) : null), [terrain])

  useEffect(() => () => mesh?.dispose(), [mesh])
  useEffect(() => () => material.dispose(), [material])

  useEffect(() => {
    if (!terrain) return
    material.uniforms.uMinH.value = terrain.minH
    material.uniforms.uMaxH.value = terrain.maxH
  }, [terrain, material])

  useFrame(() => {
    if (!mesh) return
    const s = simStore.get()
    if (s.terrain !== mesh.T) return
    mesh.drain()
    const u = material.uniforms
    u.uTime.value = nowSeconds()
    u.uLimit.value = s.limit.current ?? 30
    u.uMode.value = slopeColors ? 1 : 0
    u.uMotion.value = reducedMotion ? 0 : 1
    let c = 0
    for (const m of roverMeta) {
      const p = roverPoses.get(m.id)
      if (!p || c >= 8) continue
      u.uRovers.value[c].set(p.x, p.z)
      c++
    }
    u.uRoverCount.value = c
  })

  if (!mesh) return null
  return <mesh geometry={mesh.geometry} material={material} frustumCulled={false} renderOrder={0} />
}
