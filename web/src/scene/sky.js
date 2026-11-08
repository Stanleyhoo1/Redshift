import * as THREE from 'three'
import { ATMOSPHERE } from '../lib/theme.js'
import { SUN_DIR } from './sceneConstants.js'

// The sky of a dense, hazy super-Earth: a gradient from peach haze at the horizon to a
// dusky mauve zenith, a large soft sun disk with a forward-scattering glow, and a dark haze
// below the horizon. The same model drives the visible sky dome (GLSL) and a small HDR
// equirectangular environment (CPU) that shiny materials reflect.

const lin = (hex) => new THREE.Color(hex) // linear working space

export const SKY = {
  zenith: lin(ATMOSPHERE.zenith),
  sky: lin(ATMOSPHERE.sky),
  horizon: lin(ATMOSPHERE.horizon),
  ground: lin(ATMOSPHERE.groundHaze),
  sun: lin(ATMOSPHERE.sun),
  glow: lin(ATMOSPHERE.sunGlow),
  dust: lin(ATMOSPHERE.dust),
  dustSky: lin(ATMOSPHERE.dustSky),
}
export const SUN = new THREE.Vector3(...SUN_DIR).normalize()

// GLSL: radiance of the sky in direction d (unit, y up). uDust 0..1 browns it out.
export const skyGLSL = /* glsl */ `
uniform vec3 uZenith;
uniform vec3 uSky;
uniform vec3 uHorizon;
uniform vec3 uGroundHaze;
uniform vec3 uSunCol;
uniform vec3 uGlowCol;
uniform vec3 uSunDir;
uniform vec3 uDustCol;
uniform vec3 uDustSky;
uniform float uDust;
uniform float uSunDisk;

vec3 skyRadiance(vec3 d) {
  float h = d.y;
  vec3 col = mix(uHorizon, uSky, smoothstep(0.0, 0.28, h));
  col = mix(col, uZenith, smoothstep(0.22, 0.95, h));
  // Below the horizon: the haze over the far plains, darkening downwards.
  col = mix(col, uGroundHaze, smoothstep(0.0, -0.22, h));
  float c = max(dot(d, uSunDir), 0.0);
  // Forward scattering: a broad warm glow and a tighter halo round the sun, brightest low down.
  float glow = pow(c, 6.0) * 0.32 + pow(c, 48.0) * 0.7 + pow(c, 600.0) * 2.2;
  col += uGlowCol * glow * (0.75 + 0.5 * (1.0 - smoothstep(0.0, 0.6, h)));
  // Horizon band brighter towards the sun's side.
  col += uGlowCol * 0.12 * pow(c, 2.0) * (1.0 - smoothstep(0.0, 0.3, abs(h)));
  // A large, soft red-dwarf disk (about 2.6 degrees across).
  float disk = smoothstep(0.99958, 0.99978, c);
  col += uSunCol * disk * 9.0 * uSunDisk;
  // Dust storm: everything goes brown and flat, the sun a dim smudge.
  vec3 dusty = mix(uDustCol, uDustSky, smoothstep(-0.1, 0.6, h)) + uDustCol * glow * 0.25;
  return mix(col, dusty, uDust);
}
`

export function skyUniforms() {
  return {
    uZenith: { value: SKY.zenith.clone() },
    uSky: { value: SKY.sky.clone() },
    uHorizon: { value: SKY.horizon.clone() },
    uGroundHaze: { value: SKY.ground.clone() },
    uSunCol: { value: SKY.sun.clone() },
    uGlowCol: { value: SKY.glow.clone() },
    uSunDir: { value: SUN.clone() },
    uDustCol: { value: SKY.dust.clone() },
    uDustSky: { value: SKY.dustSky.clone() },
    uDust: { value: 0 },
    uSunDisk: { value: 1 },
  }
}

// The visible sky: a big inverted sphere that travels with the camera, on the far plane.
export function createSkyMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: skyUniforms(),
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        vDir = position;
        vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_Position = p.xyww; // on the far plane
      }
    `,
    fragmentShader: /* glsl */ `
      ${skyGLSL}
      varying vec3 vDir;
      void main() {
        gl_FragColor = vec4(skyRadiance(normalize(vDir)), 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: true, // drawn after the opaque scene, on the far plane: only where nothing else is
    fog: false,
  })
}

// ---------------------------------------------------------------- environment map

const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

let env = null

// Small HDR equirectangular sky (no sun disk: the key light makes the sharp highlight).
// three.js turns it into a PMREM the first time a material uses it.
export function skyEnvironment() {
  if (env) return env
  const W = 128
  const H = 64
  const data = new Uint16Array(W * H * 4)
  const c = new THREE.Color()
  const t = new THREE.Color()
  const d = new THREE.Vector3()
  for (let j = 0; j < H; j++) {
    const v = (j + 0.5) / H
    const lat = (v - 0.5) * Math.PI
    for (let i = 0; i < W; i++) {
      const u = (i + 0.5) / W
      const lon = (u - 0.5) * Math.PI * 2
      d.set(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon))
      const h = d.y
      c.copy(SKY.horizon).lerp(SKY.sky, smooth(0, 0.28, h))
      c.lerp(SKY.zenith, smooth(0.22, 0.95, h))
      // Below the horizon the "ground" is the lit regolith, a little brighter than the far haze.
      t.copy(SKY.ground).multiplyScalar(1.15)
      c.lerp(t, smooth(0, -0.22, h))
      const cs = Math.max(0, d.dot(SUN))
      const glow = Math.pow(cs, 6) * 0.32 + Math.pow(cs, 48) * 0.7
      c.r += SKY.glow.r * glow
      c.g += SKY.glow.g * glow
      c.b += SKY.glow.b * glow
      const k = (j * W + i) * 4
      data[k] = THREE.DataUtils.toHalfFloat(c.r)
      data[k + 1] = THREE.DataUtils.toHalfFloat(c.g)
      data[k + 2] = THREE.DataUtils.toHalfFloat(c.b)
      data[k + 3] = THREE.DataUtils.toHalfFloat(1)
    }
  }
  env = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.HalfFloatType)
  env.mapping = THREE.EquirectangularReflectionMapping
  env.colorSpace = THREE.LinearSRGBColorSpace
  env.magFilter = THREE.LinearFilter
  env.minFilter = THREE.LinearFilter
  env.needsUpdate = true
  return env
}
