import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { DUST, VOID } from '../lib/theme.js'
import { reducedMotion } from './sceneConstants.js'

const MOTES = 900
const FOG_COLOR = new THREE.Color('#4a3424')
const SKY_CLEAR = new THREE.Color(VOID)
const SKY_DUST = new THREE.Color('#1d140e')

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
  vec3 drift = vec3(uTime * (2.6 + aSeed * 1.8), sin(uTime * 0.7 + aSeed * 30.0) * 0.6, uTime * (0.8 + aSeed * 0.6));
  vec3 local = mod(position * uBox + drift - uCenter + 0.5 * uBox, uBox) - 0.5 * uBox;
  local.y = position.y * 9.0 - 1.0 + drift.y;
  vec3 world = uCenter + local;
  vec4 mv = modelViewMatrix * vec4(world, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp((0.2 + aSeed * 0.3) * uScale / -mv.z, 1.5, 7.0);
  float edge = 1.0 - smoothstep(0.3, 0.5, max(abs(local.x), abs(local.z)) / uBox);
  vAlpha = uAmount * edge * (0.45 + 0.45 * aSeed);
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

// Dust storm (an active "dust_storm" hazard): scene fog that thickens with distance,
// a dusty sky and drifting motes. The fog stays attached to the scene at all times
// (only its range changes), so starting a storm never recompiles any shader.
export function DustStorm() {
  const active = useSim((s) => s.hazards.some((h) => h.type === 'dust_storm' && h.active))
  const scene = useThree((s) => s.scene)
  const camera = useThree((s) => s.camera)
  const controls = useThree((s) => s.controls)
  const size = useThree((s) => s.size)
  const amount = useRef(0)
  const points = useRef()
  const fog = useMemo(() => new THREE.Fog(FOG_COLOR.clone(), 1e5, 1e6), [])

  const { geometry, material } = useMemo(() => {
    const pos = new Float32Array(MOTES * 3)
    const seed = new Float32Array(MOTES)
    let a = 12345
    const rand = () => ((a = (a * 1103515245 + 12345) >>> 0) / 4294967296)
    for (let i = 0; i < MOTES; i++) {
      pos[i * 3] = rand()
      pos[i * 3 + 1] = rand()
      pos[i * 3 + 2] = rand()
      seed[i] = rand()
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6)
    const m = new THREE.ShaderMaterial({
      vertexShader: moteVertex,
      fragmentShader: moteFragment,
      transparent: true,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uAmount: { value: 0 },
        uBox: { value: 60 },
        uCenter: { value: new THREE.Vector3() },
        uScale: { value: 400 },
        uColor: { value: new THREE.Color(DUST) },
      },
    })
    return { geometry: g, material: m }
  }, [])

  // Layout effect: attached before the first frame, so materials compile with fog once.
  useLayoutEffect(() => {
    const prev = scene.fog
    scene.fog = fog
    return () => {
      scene.fog = prev
      if (scene.background?.isColor) scene.background.copy(SKY_CLEAR)
    }
  }, [scene, fog])
  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )

  useFrame(({ clock }, dt) => {
    const target = active ? 1 : 0
    const k = 1 - Math.exp(-Math.min(dt, 0.1) * (active ? 1.4 : 0.9))
    amount.current += (target - amount.current) * k
    if (Math.abs(target - amount.current) < 0.002) amount.current = target
    const a = amount.current
    if (scene.background?.isColor) scene.background.copy(SKY_CLEAR).lerp(SKY_DUST, a)
    const center = controls?.target
    const D = center ? camera.position.distanceTo(center) : 60
    if (a <= 0) {
      fog.near = 1e5
      fog.far = 1e6
    } else {
      // About 40 % haze at the point the camera looks at, thicker beyond, whatever the zoom.
      const near = D * 0.3
      fog.near = near
      fog.far = near + (D * 1.75) / a
    }
    const p = points.current
    if (p) {
      p.visible = a > 0.01
      const u = material.uniforms
      u.uAmount.value = a
      u.uTime.value = reducedMotion ? 0 : clock.elapsedTime
      u.uBox.value = Math.max(20, Math.min(90, D * 0.9))
      if (center) u.uCenter.value.copy(center)
      u.uScale.value = size.height * 0.5 * Math.min(2, window.devicePixelRatio || 1)
    }
  })

  return <points ref={points} geometry={geometry} material={material} frustumCulled={false} renderOrder={8} visible={false} />
}
