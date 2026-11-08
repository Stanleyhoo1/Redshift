import { useEffect, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useGraphics } from '../state/visualPrefs.js'
import { REGOLITH } from '../lib/theme.js'
import { atmo, reducedMotion } from './sceneConstants.js'

// Dust kicked up behind moving rovers: one small particle system for the whole fleet.
// Each rover (Rovers.jsx) writes its displayed pose and speed into `dustSources`; puffs are
// emitted at the rear wheels in proportion to speed, billow out, drift downwind, settle and
// fade (in 2 g they do not rise far). Preallocated buffers, no per-frame allocations.
// Off with Low graphics.

export const dustSources = new Map() // rover id -> { x, y, z, yaw, speed }

const MAX = 480
const LIFE = 1.3
const SCALE = 1.5 // rover model scale
const WIND = [0.35, 0.12] // m/s, world x/y

const vertexShader = /* glsl */ `
attribute float aAge;
attribute float aSeed;
uniform float uScale;
varying float vAlpha;
varying float vSeed;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float t = clamp(aAge, 0.0, 1.0);
  float size = (0.16 + 0.55 * sqrt(t)) * (0.7 + 0.6 * aSeed);
  gl_PointSize = clamp(size * uScale / -mv.z, 0.0, 160.0);
  vAlpha = aAge < 0.0 ? 0.0 : (1.0 - t) * (1.0 - t) * smoothstep(0.0, 0.08, t);
  vSeed = aSeed;
}
`

const fragmentShader = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying float vAlpha;
varying float vSeed;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c) * 2.0;
  if (d > 1.0) discard;
  // Soft, slightly lumpy puff.
  float lump = 0.85 + 0.15 * sin(atan(c.y, c.x) * 5.0 + vSeed * 20.0);
  float a = smoothstep(1.0, 0.2 * lump, d) * vAlpha * uOpacity;
  gl_FragColor = vec4(uColor * (0.85 + 0.25 * (1.0 - d)), a);
}
`

export function RoverDust() {
  const graphics = useGraphics()
  const size = useThree((s) => s.size)
  const sys = useMemo(() => {
    const pos = new Float32Array(MAX * 3)
    const vel = new Float32Array(MAX * 3)
    const age = new Float32Array(MAX).fill(-1)
    const seed = new Float32Array(MAX)
    for (let i = 0; i < MAX; i++) {
      const v = Math.sin(i * 91.7) * 43758.5453
      seed[i] = v - Math.floor(v)
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('aAge', new THREE.BufferAttribute(age, 1).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6)
    // Lit dust, in display colours (unlit points keep their colour exactly).
    const c = new THREE.Color(REGOLITH.dustLight).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace)
    const m = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      uniforms: { uColor: { value: new THREE.Vector3(c.r * 0.85, c.g * 0.82, c.b * 0.78) }, uScale: { value: 400 }, uOpacity: { value: 0.4 } },
    })
    m.userData.realistic = true
    const sys = { g, m, pos, vel, age, seed, next: 0, acc: new Map(), step: 0, wasAny: false }
    // One rover's puffs this frame (called through Map.forEach, so no per-frame garbage).
    sys.emit = (s, id) => {
      if (s.speed < 0.08) return
      const step = sys.step
      let acc = (sys.acc.get(id) || 0) + step * Math.min(40, s.speed * 55)
      const c = Math.cos(s.yaw)
      const sn = Math.sin(s.yaw)
      while (acc >= 1) {
        acc -= 1
        const i = sys.next
        sys.next = (sys.next + 1) % MAX
        const side = (i % 2 ? 1 : -1) * 0.4 * SCALE
        const back = -0.22 * SCALE - 0.12
        // Rear wheel contact point in world (x east, y north): forward (c, sn), right (sn, -c).
        const wx = s.x + c * back + sn * side
        const wy = s.y + sn * back - c * side
        const r = seed[i]
        pos[i * 3] = wx + (r - 0.5) * 0.2
        pos[i * 3 + 1] = s.z + 0.08
        pos[i * 3 + 2] = -(wy + (seed[(i + 7) % MAX] - 0.5) * 0.2)
        // Thrown back and out, a little up.
        const kick = 0.25 + s.speed * 0.6
        vel[i * 3] = -c * kick + (r - 0.5) * 0.5 + WIND[0]
        vel[i * 3 + 1] = 0.35 + r * 0.35
        vel[i * 3 + 2] = -(-sn * kick + (seed[(i + 3) % MAX] - 0.5) * 0.5 + WIND[1])
        age[i] = 0
      }
      sys.acc.set(id, acc)
    }
    return sys
  }, [])
  useEffect(() => () => (sys.g.dispose(), sys.m.dispose()), [sys])
  useEffect(() => {
    sys.m.uniforms.uScale.value = size.height * 0.9 * Math.min(1.5, window.devicePixelRatio || 1)
  }, [size, sys])

  const high = graphics === 'high'
  useFrame((_, dt) => {
    const step = Math.min(dt, 0.1)
    const { pos, vel, age } = sys
    let any = false
    // Emit.
    if (high && !reducedMotion) {
      sys.step = step
      dustSources.forEach(sys.emit)
    }

    // Move and age.
    const drag = Math.exp(-step * 2.2)
    for (let i = 0; i < MAX; i++) {
      if (age[i] < 0) continue
      any = true
      age[i] += step / LIFE
      if (age[i] >= 1) {
        age[i] = -1
        continue
      }
      vel[i * 3] *= drag
      vel[i * 3 + 2] *= drag
      vel[i * 3 + 1] = vel[i * 3 + 1] * drag - 0.9 * step
      pos[i * 3] += vel[i * 3] * step
      pos[i * 3 + 1] = Math.max(pos[i * 3 + 1] + vel[i * 3 + 1] * step, pos[i * 3 + 1] - 0.02)
      pos[i * 3 + 2] += vel[i * 3 + 2] * step
    }
    sys.m.uniforms.uOpacity.value = 0.4 * (1 - 0.5 * atmo.dust)
    if (any || sys.wasAny) {
      sys.g.attributes.position.needsUpdate = true
      sys.g.attributes.aAge.needsUpdate = true
    }
    sys.wasAny = any
  })

  return <points geometry={sys.g} material={sys.m} frustumCulled={false} renderOrder={4} visible={high} />
}
