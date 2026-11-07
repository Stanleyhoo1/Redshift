import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { nowSeconds, simStore, useSim } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { WATER } from '../lib/theme.js'
import { FatPolyline } from './fatLine.js'
import { SUN_DIR, reducedMotion } from './sceneConstants.js'

// Liquid water (PROTOCOL.md version 4). The true bodies come from `features.water` (flat
// surface at `level`, closed outline); the fleet's identified water from `fleet.water_zones`
// (discs). LiDAR gets no returns from liquid, so the terrain under water stays an unknown
// hole until a rover identifies it. Then the water surface fills the hole, clipped to the
// identified discs through a small mask texture (one texel per map sample). With Reality on
// the whole true surface shows, the part the fleet has not identified a little fainter,
// with a dashed outline.
//
// The surface is one cheap shader: no extra vertices, the waves are an animated normal
// (a few travelling sines), plus a sun highlight, sparse twinkling glints and fresnel.

const vertexShader = /* glsl */ `
varying vec3 vWorld;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`

const fragmentShader = /* glsl */ `
uniform float uTime;
uniform vec3 uDeep;
uniform vec3 uMid;
uniform vec3 uGlint;
uniform vec3 uSunDir;
uniform sampler2D uMask;
uniform vec4 uMaskMap; // ox, oy, 1 / res, unused
uniform vec2 uMaskSize; // nx, ny
uniform float uClip; // 1: only where the fleet identified water
uniform float uUnknownAlpha; // opacity factor for unidentified water when not clipped
uniform vec3 uFogColor;
uniform float uFogNear;
uniform float uFogFar;
varying vec3 vWorld;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// A few travelling waves (world xz, metres): height in x, its gradient in yz.
vec3 waves(vec2 p, float t) {
  vec3 w = vec3(0.0);
  vec2 d = vec2(0.928, 0.371);
  float ph = dot(d, p) * 1.6 + t * 1.25;
  w += vec3(sin(ph), d * 1.6 * cos(ph)) * 0.34;
  d = vec2(-0.514, 0.857);
  ph = dot(d, p) * 2.3 + t * 1.6;
  w += vec3(sin(ph), d * 2.3 * cos(ph)) * 0.26;
  d = vec2(0.196, -0.981);
  ph = dot(d, p) * 3.7 + t * 2.1;
  w += vec3(sin(ph), d * 3.7 * cos(ph)) * 0.18;
  d = vec2(-0.958, -0.287);
  ph = dot(d, p) * 5.9 + t * 2.9;
  w += vec3(sin(ph), d * 5.9 * cos(ph)) * 0.12;
  return w; // height about -0.9 to 0.9
}

void main() {
  vec2 wxy = vec2(vWorld.x, -vWorld.z);
  vec2 uv = (vec2((wxy.x - uMaskMap.x) * uMaskMap.z, (wxy.y - uMaskMap.y) * uMaskMap.z) + 0.5) / uMaskSize;
  float mask = texture2D(uMask, uv).r;
  if (uClip > 0.5 && mask < 0.03) discard;

  vec2 p = vWorld.xz;
  vec3 w = waves(p, uTime);
  float hgt = w.x;
  vec3 n = normalize(vec3(-w.y * 0.16, 1.0, -w.z * 0.16));
  vec3 v = normalize(cameraPosition - vWorld);
  vec3 h = normalize(uSunDir + v);
  float ndh = max(dot(n, h), 0.0);
  float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);

  // Slow swirls of deeper and lighter water, lighter towards grazing angles.
  float swirl = 0.5 + 0.5 * sin(p.x * 0.23 + uTime * 0.17) * sin(p.y * 0.19 - uTime * 0.13);
  vec3 col = mix(uDeep, uMid, 0.3 + 0.28 * swirl + 0.4 * fres);
  // The waves themselves, visible from above: lit crests, darker troughs.
  col *= 0.78 + 0.42 * max(dot(n, uSunDir), 0.0);
  // (patchy, so the crests do not line up into a regular lattice from afar)
  float patchy = 0.35 + 0.65 * (0.5 + 0.5 * sin(p.x * 0.41 - uTime * 0.21) * sin(p.y * 0.37 + p.x * 0.11 + uTime * 0.16));
  col = mix(col, uMid * 1.25, 0.28 * patchy * smoothstep(0.25, 0.95, hgt));
  col *= 1.0 - 0.18 * smoothstep(0.2, 0.9, -hgt);
  // Thin, wavy bright crest lines drifting across (one wave train bent by two slow ones).
  float bend = 1.2 * sin(dot(p, vec2(-0.5, 0.87)) * 1.3 + uTime * 0.5) + 0.8 * sin(dot(p, vec2(0.9, 0.4)) * 0.7 - uTime * 0.3);
  float rip = abs(sin(dot(p, vec2(0.8, 0.6)) * 2.4 + bend + uTime * 0.9));
  col += uGlint * 0.07 * (1.0 - smoothstep(0.0, 0.13, rip));

  // Sun highlight, and sparse glints that twinkle where the waves face the light.
  float spec = pow(ndh, 90.0) * 1.1 + pow(ndh, 18.0) * 0.12;
  vec2 q = p * 2.2;
  vec2 id = floor(q);
  vec2 f = fract(q) - 0.5;
  float r = hash(id);
  vec2 off = vec2(hash(id + 3.1), hash(id + 7.7)) - 0.5;
  float tw = pow(max(0.0, sin(uTime * 2.4 + r * 40.0)), 18.0);
  float glint = step(0.86, r) * tw * (1.0 - smoothstep(0.02, 0.09, length(f - off * 0.6))) * (0.35 + 0.65 * smoothstep(0.75, 0.98, ndh + 0.2));
  col += uGlint * (spec + glint * 0.9);

  // The edge of what the fleet has identified: a soft light rim.
  float rim = uClip > 0.5 ? 1.0 - smoothstep(0.2, 0.75, mask) : 0.0;
  col = mix(col, uGlint, rim * 0.28);

  float known = uClip > 0.5 ? smoothstep(0.03, 0.5, mask) : mix(uUnknownAlpha, 1.0, smoothstep(0.03, 0.5, mask));
  float alpha = (0.8 + 0.18 * fres) * known;

  float fogF = clamp((length(vWorld - cameraPosition) - uFogNear) / max(uFogFar - uFogNear, 1e-3), 0.0, 1.0);
  col = mix(col, uFogColor, fogF);
  gl_FragColor = vec4(col, alpha);
}
`

// The shader writes straight to the screen (flat, linear canvas): sRGB values in.
const display3 = (hex) => {
  const c = new THREE.Color(hex).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace)
  return new THREE.Vector3(c.r, c.g, c.b)
}

function createMaterial(maskTex, map) {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: true,
    side: THREE.DoubleSide,
    uniforms: {
      uTime: { value: 0 },
      uDeep: { value: display3(WATER.deep) },
      uMid: { value: display3(WATER.mid) },
      uGlint: { value: display3(WATER.glint) },
      uSunDir: { value: new THREE.Vector3(...SUN_DIR).normalize() },
      uMask: { value: maskTex },
      uMaskMap: { value: new THREE.Vector4(map.ox, map.oy, 1 / map.res, 0) },
      uMaskSize: { value: new THREE.Vector2(map.nx, map.ny) },
      uClip: { value: 1 },
      uUnknownAlpha: { value: 0.55 },
      uFogColor: { value: new THREE.Color() },
      uFogNear: { value: 1e5 },
      uFogFar: { value: 1e6 },
    },
  })
}

// One texel per map sample: coverage of the identified discs (antialiased edges).
class ZoneMask {
  constructor(map) {
    this.map = map
    this.data = new Uint8Array(map.nx * map.ny)
    this.texture = new THREE.DataTexture(this.data, map.nx, map.ny, THREE.RedFormat, THREE.UnsignedByteType)
    this.texture.magFilter = THREE.LinearFilter
    this.texture.minFilter = THREE.LinearFilter
    this.texture.needsUpdate = true
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

  dispose() {
    this.texture.dispose()
  }
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
  const scene = useThree((s) => s.scene)
  const zones = useMemo(() => (Array.isArray(zonesRaw) ? zonesRaw.filter(okZone) : []), [zonesRaw])
  const keys = useMemo(() => zones.map((z) => `${z[0]},${z[1]},${z[2]}`), [zones])
  const mask = useMemo(() => (map ? new ZoneMask(map) : null), [map])
  const material = useMemo(() => (mask ? createMaterial(mask.texture, map) : null), [mask, map])
  const geos = useMemo(() => bodies.map(surfaceGeometry), [bodies])
  useEffect(() => () => mask?.dispose(), [mask])
  useEffect(() => () => material?.dispose(), [material])
  useEffect(() => () => geos.forEach((g) => g.dispose()), [geos])

  // When each disc first appeared (zones there when the view opened do not animate).
  const born = useRef({ gen: -1, at: new Map(), opened: 0 })
  const anim = useRef({ zones: null, animating: false })
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
    // Redraw the mask only when the zones changed or a disc is still spreading.
    if (anim.current.zones !== zones || animating || anim.current.animating) {
      anim.current.zones = zones
      anim.current.animating = animating
      mask.draw(zones, radiusOf)
    }
    const u = material.uniforms
    u.uTime.value = reducedMotion ? 0 : now
    u.uClip.value = reality ? 0 : 1
    const fog = scene.fog
    if (fog && fog.isFog) {
      u.uFogColor.value.copy(fog.color)
      u.uFogNear.value = fog.near
      u.uFogFar.value = fog.far
    } else {
      u.uFogNear.value = 1e5
      u.uFogFar.value = 1e6
    }
  })

  if (!material) return null
  if (!bodies.length) return zones.length ? <ZoneDiscs zones={zones} material={material} /> : null
  return (
    <group>
      {bodies.map((b, k) => (
        <mesh key={`${gen}:${b.id}`} geometry={geos[k]} material={material} renderOrder={1} frustumCulled={false} />
      ))}
      {reality && bodies.map((b) => <WaterOutline key={`${gen}:${b.id}:line`} body={b} />)}
    </group>
  )
}
