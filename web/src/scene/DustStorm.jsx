import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useGraphics } from '../state/visualPrefs.js'
import { ATMOSPHERE, DUST } from '../lib/theme.js'
import { atmo, reducedMotion } from './sceneConstants.js'

// Dust storm (an active "dust_storm" hazard). The haze itself (fog that thickens with
// distance, a brown sky, a dim sun) is in Atmosphere.jsx, driven by `atmo.dust`; this adds
// what moves: fine motes racing past in the wind and big, soft sheets of dust streaming
// low over the ground, both in a box that travels with the camera target. Particles are
// off with Low graphics (the haze stays).

const MOTES = 1400
const SHEETS = 70

const srgb = (hex) => {
  const c = new THREE.Color(hex).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace)
  return new THREE.Vector3(c.r, c.g, c.b)
}

const moteVertex = /* glsl */ `
uniform float uTime;
uniform float uAmount;
uniform float uBox;
uniform vec3 uCenter;
uniform float uScale;
attribute float aSeed;
varying float vAlpha;
void main() {
  // Wind drift, wrapped in a box that travels with the camera target.
  vec3 drift = vec3(uTime * (4.5 + aSeed * 3.0), sin(uTime * 0.9 + aSeed * 30.0) * 0.5, uTime * (1.4 + aSeed * 0.8));
  vec3 local = mod(position * uBox + drift - uCenter + 0.5 * uBox, uBox) - 0.5 * uBox;
  local.y = position.y * 8.0 - 1.0 + drift.y;
  vec3 world = uCenter + local;
  vec4 mv = modelViewMatrix * vec4(world, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp((0.08 + aSeed * 0.14) * uScale / -mv.z, 1.0, 5.0);
  float edge = 1.0 - smoothstep(0.3, 0.5, max(abs(local.x), abs(local.z)) / uBox);
  vAlpha = uAmount * edge * (0.35 + 0.5 * aSeed);
}
`

const moteFragment = /* glsl */ `
uniform vec3 uColor;
varying float vAlpha;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = dot(c, c);
  if (d > 0.25) discard;
  gl_FragColor = vec4(uColor, vAlpha * (1.0 - d * 4.0));
}
`

const sheetVertex = /* glsl */ `
uniform float uTime;
uniform float uAmount;
uniform float uBox;
uniform vec3 uCenter;
uniform float uScale;
attribute float aSeed;
varying float vAlpha;
varying float vSeed;
void main() {
  vec3 drift = vec3(uTime * (2.6 + aSeed * 1.6), 0.0, uTime * (0.9 + aSeed * 0.5));
  vec3 local = mod(position * uBox + drift - uCenter + 0.5 * uBox, uBox) - 0.5 * uBox;
  local.y = 0.4 + position.y * 4.5;
  vec3 world = uCenter + local;
  vec4 mv = modelViewMatrix * vec4(world, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp((5.0 + aSeed * 7.0) * uScale / -mv.z, 2.0, 900.0);
  float edge = 1.0 - smoothstep(0.25, 0.5, max(abs(local.x), abs(local.z)) / uBox);
  vAlpha = uAmount * edge * (0.08 + 0.1 * aSeed);
  vSeed = aSeed;
}
`

const sheetFragment = /* glsl */ `
uniform vec3 uColor;
varying float vAlpha;
varying float vSeed;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c) * 2.0;
  if (d > 1.0) discard;
  // Streaky, wind-stretched puffs.
  float streak = 0.75 + 0.25 * sin(c.x * 9.0 + vSeed * 31.0) * sin(c.y * 3.0 + vSeed * 17.0);
  gl_FragColor = vec4(uColor, vAlpha * smoothstep(1.0, 0.1, d) * streak);
}
`

function cloud(n, seed0) {
  const pos = new Float32Array(n * 3)
  const seed = new Float32Array(n)
  let a = seed0
  const rand = () => ((a = (a * 1103515245 + 12345) >>> 0) / 4294967296)
  for (let i = 0; i < n; i++) {
    pos[i * 3] = rand()
    pos[i * 3 + 1] = rand()
    pos[i * 3 + 2] = rand()
    seed[i] = rand()
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6)
  return g
}

function particleMaterial(vertexShader, fragmentShader, color) {
  const m = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    uniforms: {
      uTime: { value: 0 },
      uAmount: { value: 0 },
      uBox: { value: 60 },
      uCenter: { value: new THREE.Vector3() },
      uScale: { value: 400 },
      uColor: { value: srgb(color) },
    },
  })
  m.userData.realistic = true
  return m
}

function drive(obj, mat, on, amount, time, box, center, scale) {
  if (!obj) return
  obj.visible = on
  if (!on) return
  const u = mat.uniforms
  u.uAmount.value = amount
  u.uTime.value = time
  u.uBox.value = box
  if (center) u.uCenter.value.copy(center)
  u.uScale.value = scale
}

export function DustStorm() {
  const camera = useThree((s) => s.camera)
  const controls = useThree((s) => s.controls)
  const size = useThree((s) => s.size)
  const graphics = useGraphics()
  const motes = useRef()
  const sheets = useRef()
  const sys = useMemo(
    () => ({
      motesGeo: cloud(MOTES, 12345),
      sheetsGeo: cloud(SHEETS, 777),
      motesMat: particleMaterial(moteVertex, moteFragment, DUST),
      sheetsMat: particleMaterial(sheetVertex, sheetFragment, ATMOSPHERE.dust),
    }),
    [],
  )
  useEffect(() => () => Object.values(sys).forEach((o) => o.dispose()), [sys])

  useFrame(({ clock }) => {
    const a = atmo.dust
    const on = a > 0.01 && graphics === 'high'
    const center = controls?.target
    const D = center ? camera.position.distanceTo(center) : 60
    const scale = size.height * 0.5 * Math.min(1.5, window.devicePixelRatio || 1)
    const t = reducedMotion ? 0 : clock.elapsedTime
    drive(motes.current, sys.motesMat, on, a, t, Math.max(20, Math.min(90, D * 0.9)), center, scale)
    drive(sheets.current, sys.sheetsMat, on, a, t, Math.max(30, Math.min(140, D * 1.4)), center, scale)
  })

  return (
    <group>
      <points ref={sheets} geometry={sys.sheetsGeo} material={sys.sheetsMat} frustumCulled={false} renderOrder={7} visible={false} />
      <points ref={motes} geometry={sys.motesGeo} material={sys.motesMat} frustumCulled={false} renderOrder={8} visible={false} />
    </group>
  )
}
