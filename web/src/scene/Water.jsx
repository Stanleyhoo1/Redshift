import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { nowSeconds, simStore, useSim } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { useGraphics } from '../state/visualPrefs.js'
import { WATER } from '../lib/theme.js'
import { FatPolyline } from './fatLine.js'
import { reducedMotion } from './sceneConstants.js'
import { skyEnvironment } from './sky.js'
import { pointInPoly } from './ground.js'
import { waterShared } from './waterShared.js'

// Liquid water (PROTOCOL.md version 4). The true bodies come from `features.water` (flat
// surface at `level`, closed outline); the fleet's identified water from `fleet.water_zones`
// (discs). LiDAR gets no returns from liquid, so the terrain under water stays an unknown
// hole until a rover identifies it. Then the water surface fills the hole, clipped to the
// identified discs through a small mask texture (one texel per map sample). With Reality on
// the whole true surface shows, the part the fleet has not identified a little fainter,
// with a dashed outline.
//
// The surface is a physically based, nearly mirror-smooth liquid: it reflects the hazy sky
// (fresnel, so it brightens towards grazing angles), with travelling waves and fine ripples
// in the normal, deep and dark in the middle, lighter over the shallows near the shore,
// and the sun's glints. Around identified water the terrain is wet and darker (shore mask,
// shared with Terrain.jsx).

const vertexCommon = /* glsl */ `
varying vec3 vWorldW;
`
const fragmentCommon = /* glsl */ `
uniform float uTime;
uniform vec3 uDeep;
uniform vec3 uShallow;
uniform vec3 uGlint;
uniform sampler2D uMask;
uniform sampler2D uDepth;
uniform vec4 uMaskMap; // ox, oy, 1 / res, unused
uniform vec2 uMaskSize; // nx, ny
uniform float uClip; // 1: only where the fleet identified water
uniform float uUnknownAlpha;
uniform float uRipples;
varying vec3 vWorldW;

float whash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// A few travelling waves (world xz, metres): height in x, its gradient in yz. Each train
// fades out once a pixel (px metres) covers too much of its wavelength, so distant and
// grazing water does not alias into a pattern.
float waa(float f, float px) {
  return 1.0 - smoothstep(0.12, 0.4, f * px * 0.159);
}
vec3 waves(vec2 p, float t, float px) {
  vec3 w = vec3(0.0);
  vec2 d = vec2(0.928, 0.371);
  float ph = dot(d, p) * 1.6 + t * 1.25;
  w += vec3(sin(ph), d * 1.6 * cos(ph)) * 0.34 * waa(1.6, px);
  d = vec2(-0.514, 0.857);
  ph = dot(d, p) * 2.3 + t * 1.6;
  w += vec3(sin(ph), d * 2.3 * cos(ph)) * 0.26 * waa(2.3, px);
  d = vec2(0.196, -0.981);
  ph = dot(d, p) * 3.7 + t * 2.1;
  w += vec3(sin(ph), d * 3.7 * cos(ph)) * 0.18 * waa(3.7, px);
  d = vec2(-0.958, -0.287);
  ph = dot(d, p) * 5.9 + t * 2.9;
  w += vec3(sin(ph), d * 5.9 * cos(ph)) * 0.12 * waa(5.9, px);
  return w;
}

// Fine capillary ripples: short waves in many directions, patchy.
vec2 ripples(vec2 p, float t) {
  vec2 g = vec2(0.0);
  for (int k = 0; k < 5; k++) {
    float a = float(k) * 1.257 + 0.4;
    vec2 d = vec2(cos(a), sin(a));
    float f = 9.0 + float(k) * 3.7;
    float ph = dot(d, p) * f + t * (3.0 + float(k) * 0.7);
    g += d * f * cos(ph) * 0.004;
  }
  return g;
}
`

// Replaces <map_fragment>.
const surface = /* glsl */ `
  vec2 wxy = vec2(vWorldW.x, -vWorldW.z);
  vec2 muv = (vec2((wxy.x - uMaskMap.x) * uMaskMap.z, (wxy.y - uMaskMap.y) * uMaskMap.z) + 0.5) / uMaskSize;
  float mask = texture2D(uMask, muv).r;
  if (uClip > 0.5 && mask < 0.03) discard;
  float depth = texture2D(uDepth, muv).r;
  vec2 wpw = vWorldW.xz;
  // Fade the small waves out where a pixel covers more than they can show (no shimmer).
  vec2 fw = fwidth(wpw);
  float px = max(fw.x, fw.y);
  float near = 1.0 - smoothstep(0.04, 0.22, px);
  vec3 wv = waves(wpw, uTime, px);
  vec2 grad = wv.yz * 0.05;
  float patchy = 0.5 + 0.5 * sin(wpw.x * 0.41 - uTime * 0.21) * sin(wpw.y * 0.37 + wpw.x * 0.11 + uTime * 0.16);
  grad += ripples(wpw, uTime) * uRipples * (0.3 + 0.7 * patchy) * near * near;
  vec3 wN = normalize(vec3(-grad.x, 1.0, -grad.y));
  // Deep water is dark; the shallows (near the shore) show a lighter, greener body colour.
  vec3 body = mix(uShallow, uDeep, smoothstep(0.0, 0.85, depth));
  float swirl = 0.5 + 0.5 * sin(wpw.x * 0.23 + uTime * 0.17) * sin(wpw.y * 0.19 - uTime * 0.13);
  body *= 0.85 + 0.3 * swirl;
  diffuseColor.rgb = body;
  float known = uClip > 0.5 ? smoothstep(0.03, 0.5, mask) : mix(uUnknownAlpha, 1.0, smoothstep(0.03, 0.5, mask));
  diffuseColor.a = known * mix(0.82, 0.96, smoothstep(0.0, 0.3, depth));
  float wRim = uClip > 0.5 ? 1.0 - smoothstep(0.2, 0.75, mask) : 0.0;
`

const glints = /* glsl */ `
  {
    vec3 vdir = normalize(cameraPosition - vWorldW);
    vec2 q = vWorldW.xz * 2.2;
    vec2 id = floor(q);
    vec2 f = fract(q) - 0.5;
    float r = whash(id);
    vec2 off = vec2(whash(id + 3.1), whash(id + 7.7)) - 0.5;
    float tw = pow(max(0.0, sin(uTime * 2.4 + r * 40.0)), 18.0);
    float glint = step(0.88, r) * tw * (1.0 - smoothstep(0.02, 0.08, length(f - off * 0.6)));
    gl_FragColor.rgb += uGlint * glint * 0.4 * uRipples * (1.0 - smoothstep(0.05, 0.2, length(fwidth(vWorldW.xz))));
    // The edge of what the fleet has identified: a soft light rim.
    gl_FragColor.rgb = mix(gl_FragColor.rgb, uGlint, wRim * 0.18);
  }
`

function createMaterial(maskTex, depthTex, map) {
  const m = new THREE.MeshStandardMaterial({
    color: '#ffffff',
    roughness: 0.05,
    metalness: 0,
    transparent: true,
    depthWrite: true,
    envMap: skyEnvironment(),
    envMapIntensity: 2.2,
  })
  m.userData.realistic = true
  const display3 = (hex) => new THREE.Color(hex)
  const uniforms = {
    uTime: { value: 0 },
    uDeep: { value: display3('#0a2a3c') },
    uShallow: { value: display3('#21606a') },
    uGlint: { value: (() => {
      const c = new THREE.Color(WATER.glint).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace)
      return new THREE.Vector3(c.r, c.g, c.b)
    })() },
    uMask: { value: maskTex },
    uDepth: { value: depthTex },
    uMaskMap: { value: new THREE.Vector4(map.ox, map.oy, 1 / map.res, 0) },
    uMaskSize: { value: new THREE.Vector2(map.nx, map.ny) },
    uClip: { value: 1 },
    uUnknownAlpha: { value: 0.55 },
    uRipples: { value: 1 },
  }
  m.userData.uniforms = uniforms
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${vertexCommon}`)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vWorldW = (modelMatrix * vec4(transformed, 1.0)).xyz;')
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${fragmentCommon}`)
      .replace('#include <map_fragment>', surface)
      .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\n  normal = normalize((viewMatrix * vec4(wN, 0.0)).xyz);')
      .replace('#include <fog_fragment>', `#include <fog_fragment>\n${glints}`)
  }
  m.customProgramCacheKey = () => 'redshift-water-1'
  return m
}

// One texel per map sample: coverage of the identified discs (antialiased edges), and the
// wet shore around it for the terrain.
class ZoneMask {
  constructor(map) {
    this.map = map
    const n = map.nx * map.ny
    this.data = new Uint8Array(n)
    this.texture = new THREE.DataTexture(this.data, map.nx, map.ny, THREE.RedFormat, THREE.UnsignedByteType)
    this.texture.magFilter = THREE.LinearFilter
    this.texture.minFilter = THREE.LinearFilter
    this.texture.needsUpdate = true
    this.shoreData = new Uint8Array(n)
    this.tmp = new Float32Array(n)
    this.shore = new THREE.DataTexture(this.shoreData, map.nx, map.ny, THREE.RedFormat, THREE.UnsignedByteType)
    this.shore.magFilter = THREE.LinearFilter
    this.shore.minFilter = THREE.LinearFilter
    this.shore.needsUpdate = true
  }

  // zones: [[x, y, r]], radius(k) gives the (possibly animated) radius of zone k.
  draw(zones, radius) {
    const { nx, ny, res, ox, oy } = this.map
    const d = this.data
    d.fill(0)
    for (let k = 0; k < zones.length; k++) {
      const [x, y] = zones[k]
      const r = radius(k)
      if (!(r > 0)) continue
      const i0 = Math.max(0, Math.floor((x - r - ox) / res) - 1)
      const i1 = Math.min(nx - 1, Math.ceil((x + r - ox) / res) + 1)
      const j0 = Math.max(0, Math.floor((y - r - oy) / res) - 1)
      const j1 = Math.min(ny - 1, Math.ceil((y + r - oy) / res) + 1)
      for (let j = j0; j <= j1; j++) {
        const wy = oy + j * res - y
        for (let i = i0; i <= i1; i++) {
          const wx = ox + i * res - x
          const cov = Math.max(0, Math.min(1, 0.5 + (r - Math.sqrt(wx * wx + wy * wy)) / res))
          if (cov <= 0) continue
          const v = Math.round(cov * 255)
          const q = j * nx + i
          if (v > d[q]) d[q] = v
        }
      }
    }
    this.texture.needsUpdate = true
  }

  // Wet ground: the coverage spread out about 2.5 m with a falloff (separable max filter).
  drawShore() {
    const { nx, ny } = this.map
    const R = 5
    const src = this.data
    const tmp = this.tmp
    const out = this.shoreData
    for (let j = 0; j < ny; j++) {
      const row = j * nx
      for (let i = 0; i < nx; i++) {
        let m = 0
        const a = Math.max(0, i - R)
        const b = Math.min(nx - 1, i + R)
        for (let q = a; q <= b; q++) {
          const v = src[row + q]
          if (!v) continue
          const w = v * (1 - Math.abs(q - i) / (R + 1))
          if (w > m) m = w
        }
        tmp[row + i] = m
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        let m = 0
        const a = Math.max(0, j - R)
        const b = Math.min(ny - 1, j + R)
        for (let q = a; q <= b; q++) {
          const v = tmp[q * nx + i]
          if (!v) continue
          const w = v * (1 - Math.abs(q - j) / (R + 1))
          if (w > m) m = w
        }
        out[j * nx + i] = m
      }
    }
    this.shore.needsUpdate = true
  }

  dispose() {
    this.texture.dispose()
    this.shore.dispose()
  }
}

// How far inside each true body a sample is (0 at the shore, 1 at 6 m and more): the
// shallows are lighter. Ground truth, used only to colour water that is drawn anyway.
function depthTexture(map, bodies) {
  const { nx, ny, res, ox, oy } = map
  const data = new Uint8Array(nx * ny)
  for (const b of bodies) {
    const poly = b.outline
    let x0 = Infinity
    let x1 = -Infinity
    let y0 = Infinity
    let y1 = -Infinity
    for (const [x, y] of poly) {
      x0 = Math.min(x0, x)
      x1 = Math.max(x1, x)
      y0 = Math.min(y0, y)
      y1 = Math.max(y1, y)
    }
    const i0 = Math.max(0, Math.floor((x0 - ox) / res))
    const i1 = Math.min(nx - 1, Math.ceil((x1 - ox) / res))
    const j0 = Math.max(0, Math.floor((y0 - oy) / res))
    const j1 = Math.min(ny - 1, Math.ceil((y1 - oy) / res))
    for (let j = j0; j <= j1; j++) {
      const y = oy + j * res
      for (let i = i0; i <= i1; i++) {
        const x = ox + i * res
        if (!pointInPoly(x, y, poly)) continue
        let best = Infinity
        for (let a = 0, c = poly.length - 1; a < poly.length; c = a++) {
          const [ax, ay] = poly[c]
          const [bx, by] = poly[a]
          const dx = bx - ax
          const dy = by - ay
          const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)))
          const ex = ax + dx * t - x
          const ey = ay + dy * t - y
          best = Math.min(best, ex * ex + ey * ey)
        }
        data[j * nx + i] = Math.max(data[j * nx + i], Math.round(Math.min(1, Math.sqrt(best) / 6) * 255))
      }
    }
  }
  const tex = new THREE.DataTexture(data, nx, ny, THREE.RedFormat, THREE.UnsignedByteType)
  tex.magFilter = THREE.LinearFilter
  tex.minFilter = THREE.LinearFilter
  tex.needsUpdate = true
  return tex
}

function surfaceGeometry(body) {
  const shape = new THREE.Shape(body.outline.map(([x, y]) => new THREE.Vector2(x, y)))
  const g = new THREE.ShapeGeometry(shape)
  // XY plane to world: (x, y, 0) -> (x, level, -y)
  g.rotateX(-Math.PI / 2)
  g.translate(0, body.level, 0)
  g.computeBoundingSphere()
  return g
}

const okZone = (z) => Array.isArray(z) && z.length >= 3 && Number.isFinite(z[0]) && Number.isFinite(z[1]) && z[2] > 0
const GROW_S = 1.3 // seconds for a newly identified disc to spread out

function WaterOutline({ body }) {
  const size = useThree((s) => s.size)
  const line = useMemo(() => {
    const n = body.outline.length + 1
    const l = new FatPolyline(n, { color: WATER.line, width: 2, dashed: true, dashSize: 0.7, gapSize: 0.45, opacity: 0.95 })
    const xyz = new Float32Array(n * 3)
    for (let k = 0; k < n; k++) {
      const [x, y] = body.outline[k % body.outline.length]
      xyz[k * 3] = x
      xyz[k * 3 + 1] = body.level + 0.05
      xyz[k * 3 + 2] = -y
    }
    l.set(xyz, n)
    return l
  }, [body])
  useEffect(() => () => line.dispose(), [line])
  useEffect(() => line.setResolution(size.width, size.height), [line, size])
  return <primitive object={line.object} renderOrder={6} />
}

// An older sim may send water_zones without features: plain water discs at the height of
// the known ground around them.
function ZoneDiscs({ zones, material }) {
  const geo = useMemo(() => new THREE.CircleGeometry(1, 40).rotateX(-Math.PI / 2), [])
  useEffect(() => () => geo.dispose(), [geo])
  const T = simStore.get().terrain
  return zones.map((z, k) => {
    const h = sampleGroundNear(T, z[0], z[1] + z[2], 6) ?? sampleGroundNear(T, z[0], z[1], 12) ?? 0
    return <mesh key={k} geometry={geo} material={material} position={[z[0], h + 0.05, -z[1]]} scale={[z[2], 1, z[2]]} renderOrder={1} />
  })
}

export function Water({ reality }) {
  const map = useSim((s) => s.map)
  const gen = useSim((s) => s.gen)
  const bodies = useSim((s) => s.features.water)
  const zonesRaw = useSim((s) => s.fleet.water_zones)
  const graphics = useGraphics()
  const zones = useMemo(() => (Array.isArray(zonesRaw) ? zonesRaw.filter(okZone) : []), [zonesRaw])
  const keys = useMemo(() => zones.map((z) => `${z[0]},${z[1]},${z[2]}`), [zones])
  const mask = useMemo(() => (map ? new ZoneMask(map) : null), [map])
  const depth = useMemo(() => (map ? depthTexture(map, bodies) : null), [map, bodies])
  const material = useMemo(() => (mask && depth ? createMaterial(mask.texture, depth, map) : null), [mask, depth, map])
  const geos = useMemo(() => bodies.map(surfaceGeometry), [bodies])
  useEffect(() => () => mask?.dispose(), [mask])
  useEffect(() => () => depth?.dispose(), [depth])
  useEffect(() => () => material?.dispose(), [material])
  useEffect(() => () => geos.forEach((g) => g.dispose()), [geos])
  useEffect(() => {
    if (!mask) return
    waterShared.shore = mask.shore
    waterShared.map = map
    return () => {
      if (waterShared.shore === mask.shore) {
        waterShared.shore = null
        waterShared.map = null
      }
    }
  }, [mask, map])

  // When each disc first appeared (zones there when the view opened do not animate).
  const born = useRef({ gen: -1, at: new Map(), opened: 0 })
  const anim = useRef({ zones: null, animating: false, frame: 0 })
  if (born.current.gen !== gen) born.current = { gen, at: new Map(), opened: nowSeconds() }

  // Per disc, its grown radius right now (written into `radii`, read by the mask).
  const radii = useRef(new Float32Array(0))
  const radiusOf = useMemo(() => (k) => radii.current[k], [])

  useFrame(() => {
    if (!mask || !material) return
    const now = nowSeconds()
    const b = born.current
    const late = now - b.opened < 1.5 || reducedMotion
    let animating = false
    if (radii.current.length < zones.length) radii.current = new Float32Array(zones.length + 16)
    for (let k = 0; k < zones.length; k++) {
      let t = b.at.get(keys[k])
      if (t === undefined) {
        t = late ? -1e9 : now
        b.at.set(keys[k], t)
      }
      const age = (now - t) / GROW_S
      if (age < 1) animating = true
      radii.current[k] = age >= 1 ? zones[k][2] : zones[k][2] * (1 - Math.pow(1 - Math.max(0, age), 3))
    }
    // Redraw the mask only when the zones changed or a disc is still spreading; the shore
    // follows (a few times a second while spreading, then once when it settles).
    const a = anim.current
    const changed = a.zones !== zones
    if (changed || animating || a.animating) {
      mask.draw(zones, radiusOf)
      if (changed || !animating || a.frame++ % 6 === 0) mask.drawShore()
      a.zones = zones
      a.animating = animating
    }
    const u = material.userData.uniforms
    u.uTime.value = reducedMotion ? 0 : now
    u.uClip.value = reality ? 0 : 1
    u.uRipples.value = graphics === 'high' ? 1 : 0
  })

  if (!material) return null
  if (!bodies.length) return zones.length ? <ZoneDiscs zones={zones} material={material} /> : null
  return (
    <group>
      {bodies.map((b, k) => (
        <mesh key={`${gen}:${b.id}`} geometry={geos[k]} material={material} renderOrder={1} frustumCulled={false} receiveShadow />
      ))}
      {reality && bodies.map((b) => <WaterOutline key={`${gen}:${b.id}:line`} body={b} />)}
    </group>
  )
}
