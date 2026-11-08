import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { appStore, simStore, useApp, useSim } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { instrumentStyle, kindStyle } from '../lib/theme.js'
import { InstrumentIcon } from '../components/SampleBits.jsx'
import { useGraphics } from '../state/visualPrefs.js'
import { FatPolyline } from './fatLine.js'
import { glowTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'
import { hashStr } from './ground.js'
import { crustTexture, filmTexture, iceTexture, mulberry } from './proceduralTextures.js'
import { rockGeometry, rockMaterial } from './rocks.js'
import { crystalGeo } from './Spires.jsx'
import { skyEnvironment } from './sky.js'
import { DrapedDisc, useDrape } from './drape.js'

const geo = {
  pole: new THREE.CylinderGeometry(0.028, 0.028, 2.5, 6).translate(0, 1.25, 0),
  head: new THREE.OctahedronGeometry(0.44, 0),
  ripple: new THREE.RingGeometry(0.85, 1.0, 48).rotateX(-Math.PI / 2),
  base: new THREE.RingGeometry(1.1, 1.36, 48).rotateX(-Math.PI / 2),
  track: new THREE.RingGeometry(1.5, 1.72, 64).rotateX(-Math.PI / 2),
}
const ARC = 64

// v4b: while a rover samples this target, a ring fills around it and a chip names the
// instrument. Progress comes from the rover (`sampling.progress`).
function SamplingRing({ targetId, roverId, color }) {
  const size = useThree((s) => s.size)
  const labelsReady = useThree((s) => !!s.events.connected)
  const arc = useMemo(() => new FatPolyline(ARC + 1, { color, width: 4, opacity: 0.98 }), [color])
  const xyz = useMemo(() => new Float32Array((ARC + 1) * 3), [])
  const trackMat = useMemo(() => new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.22, depthWrite: false }), [color])
  const shown = useRef(0)
  useEffect(() => () => (arc.dispose(), trackMat.dispose()), [arc, trackMat])
  useEffect(() => arc.setResolution(size.width, size.height), [arc, size])
  useFrame((_, dt) => {
    const sm = simStore.get().rovers[roverId]?.sampling
    const target = sm && sm.target === targetId ? sm.progress : 0
    // Ease between ticks so the ring fills smoothly.
    shown.current += (target - shown.current) * Math.min(1, dt * 6)
    const n = Math.max(2, Math.round(shown.current * ARC) + 1)
    for (let q = 0; q < n; q++) {
      const ang = Math.PI / 2 - (q / (n - 1)) * Math.PI * 2 * shown.current
      xyz[q * 3] = Math.cos(ang) * 1.61
      xyz[q * 3 + 1] = 0.12
      xyz[q * 3 + 2] = -Math.sin(ang) * 1.61
    }
    arc.set(xyz, n)
  })
  return (
    <group>
      <mesh geometry={geo.track} material={trackMat} position={[0, 0.1, 0]} renderOrder={3} />
      <primitive object={arc.object} renderOrder={4} />
      {labelsReady && (
        <Html position={[0, 5.2, 0]} center zIndexRange={[13, 0]} style={{ pointerEvents: 'none' }}>
          <SampleChip targetId={targetId} roverId={roverId} color={color} />
        </Html>
      )}
    </group>
  )
}

// Inside drei's own React root, so it subscribes to the store itself and stays in step.
function SampleChip({ targetId, roverId, color }) {
  const info = useSim((s) => {
    const sm = s.rovers[roverId]?.sampling
    return sm && sm.target === targetId ? `${sm.instrument ?? ''}|${Math.round(sm.progress * 100)}` : ''
  })
  if (!info) return null
  const [instrument, pct] = info.split('|')
  return (
    <div className="sample-chip" style={{ '--c': color }}>
      <InstrumentIcon kind={instrument} />
      {instrumentStyle(instrument).label}
      <b>{pct}%</b>
    </div>
  )
}

// v4b: an analysed sample's value, on hover or when picked (Samples tab or a click).
function ValueTag({ target, color }) {
  const v = Number.isFinite(target.value) ? target.value : null
  return (
    <div className="value-tag" style={{ '--c': color }}>
      <span className="value-tag-v">
        {v === null ? '...' : v}
        <small>/10</small>
      </span>
      <span className="value-tag-a">{target.analysis || 'Analysing on board'}</span>
    </div>
  )
}

// ---------------------------------------------------------------- physical features
// What a target looks like on the ground, under its marker (the marker keeps it findable
// from the overview): a mineral outcrop with crystal veins, a glossy ice patch, a vent
// mound with sulphur crust and rising gas, a film of organic traces.

const featureCache = new Map()
function featureParts(kind, id) {
  const key = `${kind}:${id}`
  if (featureCache.has(key)) return featureCache.get(key)
  const rand = mulberry(hashStr(id) + 77)
  let parts = null
  if (kind === 'mineral') {
    const rocks = Array.from({ length: 3 }, (_, k) => ({
      geo: rockGeometry(hashStr(id) + k * 13, 2, { tint: '#6e665f', cuts: 7, squash: 0.72 }),
      pos: [(rand() - 0.5) * 0.9, 0, (rand() - 0.5) * 0.9],
      rot: rand() * 6.28,
      s: k === 0 ? 0.55 + rand() * 0.15 : 0.28 + rand() * 0.18,
    }))
    rocks[0].pos = [0, 0, 0]
    // Crystal veins: small prisms poking out of the rocks.
    const veins = Array.from({ length: 22 }, (_, k) => {
      const r = rocks[k < 12 ? 0 : Math.floor(rand() * rocks.length)]
      const a = rand() * Math.PI * 2
      const d = r.s * (0.35 + rand() * 0.45)
      return { pos: [r.pos[0] + Math.cos(a) * d, r.s * (0.05 + rand() * 0.35), r.pos[2] + Math.sin(a) * d], tilt: [(rand() - 0.5) * 1.6, rand() * 6.28, (rand() - 0.5) * 1.6], s: [0.045 + rand() * 0.06, 0.16 + rand() * 0.26] }
    })
    parts = { rocks, veins }
  } else if (kind === 'ice') {
    parts = {
      chunks: Array.from({ length: 4 }, (_, k) => ({
        geo: rockGeometry(hashStr(id) + k * 29, 2, { cuts: 6, squash: 0.55, tint: '#9fc4e2' }),
        pos: [(rand() - 0.5) * 1.4, 0.02, (rand() - 0.5) * 1.4],
        rot: rand() * 6.28,
        s: 0.1 + rand() * 0.16,
      })),
    }
  } else if (kind === 'vent') {
    const pts = [
      [1.15, -0.05],
      [0.95, 0.06],
      [0.62, 0.2],
      [0.34, 0.33],
      [0.24, 0.32],
      [0.16, 0.2],
      [0.0, 0.14],
    ].map(([r, y]) => new THREE.Vector2(r, y))
    const g = new THREE.LatheGeometry(pts, 28)
    // Lumpy: push vertices in and out a little.
    const p = g.attributes.position
    const col = new Float32Array(p.count * 3)
    const c = new THREE.Color()
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i)
      const z = p.getZ(i)
      const r = Math.hypot(x, z)
      const a = Math.atan2(z, x)
      const k = 1 + 0.08 * Math.sin(a * 3 + 1.3) + 0.05 * Math.sin(a * 7 + 0.4)
      p.setX(i, x * k)
      p.setZ(i, z * k)
      // Dark rock on the flanks, yellow-orange crust round the mouth.
      const mouth = 1 - Math.min(1, Math.max(0, (r - 0.18) / 0.35))
      c.set('#4a3f38').lerp(new THREE.Color('#c99a3e'), mouth * 0.85)
      if (r < 0.17) c.set('#1c1714')
      col[i * 3] = c.r
      col[i * 3 + 1] = c.g
      col[i * 3 + 2] = c.b
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3))
    g.computeVertexNormals()
    parts = { mound: g }
  }
  featureCache.set(key, parts)
  if (featureCache.size > 200) featureCache.delete(featureCache.keys().next().value)
  return parts
}

const veinMat = (() => {
  const m = new THREE.MeshStandardMaterial({ color: '#7d4fd0', emissive: '#8b55ee', emissiveIntensity: 0.9, roughness: 0.1, metalness: 0.1, flatShading: true, envMap: skyEnvironment(), envMapIntensity: 1.8 })
  m.userData.realistic = true
  return m
})()
const iceChunkMat = rockMaterial('ice', { roughness: 0.15, metalness: 0.0, bump: 0.01, grain: 0.4, envMap: skyEnvironment(), envMapIntensity: 1.3 })
const outcropMat = rockMaterial('outcrop', { bump: 0.018, grain: 0.35 })
const ventMat = rockMaterial('vent', { bump: 0.012, grain: 0.3, roughness: 0.85 })
const VENT_GLOW = '#ff8a3d'

const _e = new THREE.Euler()
const _q = new THREE.Quaternion()
const _m = new THREE.Matrix4()
const _v = new THREE.Vector3()
const _s = new THREE.Vector3()

function Veins({ veins }) {
  const ref = useRef()
  useEffect(() => {
    const mesh = ref.current
    if (!mesh) return
    veins.forEach((v, i) => {
      _q.setFromEuler(_e.set(...v.tilt))
      _m.compose(_v.set(...v.pos), _q, _s.set(v.s[0], v.s[1], v.s[0]))
      mesh.setMatrixAt(i, _m)
    })
    mesh.instanceMatrix.needsUpdate = true
  }, [veins])
  return <instancedMesh ref={ref} args={[crystalGeo, veinMat, veins.length]} castShadow />
}

const GAS = 26
const gasVertex = /* glsl */ `
attribute float aSeed;
uniform float uTime;
uniform float uScale;
varying float vAlpha;
void main() {
  float t = fract(uTime * 0.22 + aSeed);
  float a = aSeed * 43.0;
  vec3 p = position;
  p.x += cos(a) * (0.08 + t * 0.35) + t * 0.45;
  p.z += sin(a) * (0.08 + t * 0.35) + t * 0.15;
  p.y += 0.3 + t * 2.4;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp((0.25 + t * 1.1) * uScale / -mv.z, 0.0, 200.0);
  vAlpha = smoothstep(0.0, 0.12, t) * (1.0 - t);
}
`
const gasFragment = /* glsl */ `
uniform vec3 uColor;
varying float vAlpha;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  if (d > 1.0) discard;
  gl_FragColor = vec4(uColor, smoothstep(1.0, 0.0, d) * vAlpha * 0.32);
}
`
function VentGas() {
  const size = useThree((s) => s.size)
  const sys = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(GAS * 3), 3))
    const seed = new Float32Array(GAS)
    for (let i = 0; i < GAS; i++) seed[i] = i / GAS + (Math.sin(i * 12.9) * 0.5 + 0.5) / GAS
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 1.5, 0), 4)
    const c = new THREE.Color('#efe2c4').getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace)
    const m = new THREE.ShaderMaterial({
      vertexShader: gasVertex,
      fragmentShader: gasFragment,
      transparent: true,
      depthWrite: false,
      uniforms: { uTime: { value: 0 }, uScale: { value: 400 }, uColor: { value: new THREE.Vector3(c.r, c.g, c.b) } },
    })
    return { g, m }
  }, [])
  useEffect(() => () => (sys.g.dispose(), sys.m.dispose()), [sys])
  useFrame(({ clock }) => {
    sys.m.uniforms.uTime.value = reducedMotion ? 0.4 : clock.elapsedTime
    sys.m.uniforms.uScale.value = size.height * 0.9
  })
  return <points geometry={sys.g} material={sys.m} renderOrder={4} />
}

// Inside the moving pin group (local, at the target's ground height).
function TargetFeature({ target }) {
  const graphics = useGraphics()
  const parts = useMemo(() => featureParts(target.kind, target.id), [target.kind, target.id])
  const glowMat = useMemo(() => new THREE.SpriteMaterial({ map: glowTexture(), color: VENT_GLOW, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending }), [])
  useEffect(() => () => glowMat.dispose(), [glowMat])
  if (!parts) return null
  if (target.kind === 'mineral') {
    return (
      <group>
        {parts.rocks.map((r, k) => (
          <mesh key={k} geometry={r.geo} material={outcropMat} position={[r.pos[0], r.s * 0.18, r.pos[2]]} rotation={[0, r.rot, 0]} scale={r.s} castShadow receiveShadow />
        ))}
        <Veins veins={parts.veins} />
      </group>
    )
  }
  if (target.kind === 'ice') {
    return (
      <group>
        {parts.chunks.map((r, k) => (
          <mesh key={k} geometry={r.geo} material={iceChunkMat} position={[r.pos[0], r.s * 0.2, r.pos[2]]} rotation={[0, r.rot, 0]} scale={r.s} castShadow />
        ))}
      </group>
    )
  }
  if (target.kind === 'vent') {
    return (
      <group>
        <mesh geometry={parts.mound} material={ventMat} castShadow receiveShadow />
        <sprite material={glowMat} position={[0, 0.3, 0]} scale={1.1} renderOrder={4} />
        {graphics === 'high' && <VentGas />}
      </group>
    )
  }
  return null
}

const iceMat = (() => {
  const m = new THREE.MeshStandardMaterial({ map: iceTexture(), color: '#e6f2fb', roughness: 0.07, metalness: 0.0, envMap: skyEnvironment(), envMapIntensity: 1.7, transparent: true, polygonOffset: true, polygonOffsetFactor: -2, depthWrite: false })
  m.userData.realistic = true
  return m
})()
const filmMat = (() => {
  const m = new THREE.MeshStandardMaterial({ map: filmTexture(), roughness: 0.42, metalness: 0.0, transparent: true, polygonOffset: true, polygonOffsetFactor: -2, depthWrite: false, envMap: skyEnvironment(), envMapIntensity: 0.6 })
  m.userData.realistic = true
  return m
})()
const crustMat = (() => {
  const m = new THREE.MeshStandardMaterial({ map: crustTexture(), roughness: 0.8, metalness: 0.0, transparent: true, polygonOffset: true, polygonOffsetFactor: -2, depthWrite: false })
  m.userData.realistic = true
  return m
})()

// Patches lying on the terrain (world coordinates, draped).
function TargetGround({ target }) {
  const kind = target.kind
  const spec = kind === 'ice' ? { r: 1.35, mat: iceMat, uv: 'disc' } : kind === 'trace' ? { r: 1.1, mat: filmMat, uv: 'disc' } : kind === 'vent' ? { r: 1.55, mat: crustMat, uv: 'disc' } : null
  const disc = useMemo(() => {
    if (!spec) return null
    const rand = mulberry(hashStr(target.id) + 5)
    const w = [rand() * 6, rand() * 6, rand() * 6]
    const shape = kind === 'vent' ? null : (a) => 0.82 + 0.12 * Math.sin(a * 2 + w[0]) + 0.08 * Math.sin(a * 3 + w[1]) + 0.05 * Math.sin(a * 5 + w[2])
    return new DrapedDisc(6, 48, { uvMode: spec.uv, shape, normals: true })
  }, [target.id, kind]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => disc?.dispose(), [disc])
  useDrape(() => disc?.update(target.x, target.y, spec.r, 0.03), [target.x, target.y, kind])
  if (!disc) return null
  return <mesh geometry={disc.geometry} material={spec.mat} renderOrder={1} receiveShadow />
}

// Discovered target: a beacon pin coloured by kind.
// found = pulsing, investigating = brighter and faster, sampled = steady with a ring.
function TargetPin({ target }) {
  const labelsReady = useThree((s) => !!s.events.connected)
  const [hover, setHover] = useState(false)
  const picked = useApp((a) => a.pickedTarget === target.id)
  const sampler = useSim((s) => {
    for (const r of Object.values(s.rovers)) if (r.sampling && r.sampling.target === target.id) return r.id
    return null
  })
  useEffect(() => () => (document.body.style.cursor = ''), [])
  const group = useRef()
  const head = useRef()
  const ripple = useRef()
  const glow = useRef()
  const color = kindStyle(target.kind).color
  const mats = useMemo(
    () => ({
      pole: new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 }),
      head: new THREE.MeshBasicMaterial({ color }),
      ripple: new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, depthWrite: false, blending: THREE.AdditiveBlending }),
      base: new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95, depthWrite: false }),
      glow: new THREE.SpriteMaterial({ map: glowTexture(), color, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.6 }),
    }),
    [color],
  )
  useEffect(() => () => Object.values(mats).forEach((m) => m.dispose()), [mats])

  const status = target.status
  useFrame(({ clock }) => {
    const g = group.current
    if (!g) return
    const h = sampleGroundNear(simStore.get().terrain, target.x, target.y, 6) ?? 0
    g.position.set(target.x, h, -target.y)
    const t = reducedMotion ? 0 : clock.elapsedTime
    const hd = head.current
    const rp = ripple.current
    const gl = glow.current
    if (status === 'sampled') {
      // Solid and steady: no pulse or bob, just a slow turn above the ring.
      mats.glow.opacity = 0.45
      if (gl) gl.scale.setScalar(2)
      if (rp) rp.visible = false
      if (hd) {
        hd.rotation.y = t * 0.4
        hd.position.y = 2.95
      }
      return
    }
    const fast = status === 'investigating'
    const pulse = 0.5 + 0.5 * Math.sin(t * (fast ? 6 : 3.2))
    mats.glow.opacity = fast ? 0.85 + 0.15 * pulse : 0.3 + 0.45 * pulse
    if (gl) gl.scale.setScalar(fast ? 2.6 : 1.8 + 0.5 * pulse)
    if (hd) {
      hd.rotation.y = t * (fast ? 2.2 : 0.8)
      hd.position.y = 2.95 + 0.12 * Math.sin(t * 2)
    }
    if (rp) {
      const ph = (t * (fast ? 1.1 : 0.6)) % 1
      rp.visible = true
      rp.scale.set(1 + ph * 1.8, 1, 1 + ph * 1.8)
      mats.ripple.opacity = 0.6 * (1 - ph)
    }
  })

  const analysed = status === 'sampled' && (target.value != null || target.analysis != null)
  return (
    <>
    <TargetGround target={target} />
    <group
      ref={group}
      onPointerOver={(e) => {
        e.stopPropagation()
        if (appStore.get().placing) return
        setHover(true)
        if (analysed) document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => {
        setHover(false)
        document.body.style.cursor = ''
      }}
      onClick={(e) => {
        e.stopPropagation()
        if (e.delta > 6 || appStore.get().placing || !analysed) return
        appStore.patch({ pickedTarget: picked ? null : target.id })
      }}
    >
      {sampler && <SamplingRing targetId={target.id} roverId={sampler} color={color} />}
      {labelsReady && analysed && (hover || picked) && (
        <Html position={[0, 4.15, 0]} center zIndexRange={[13, 0]} style={{ pointerEvents: 'none' }}>
          <ValueTag target={target} color={color} />
        </Html>
      )}
      <mesh geometry={geo.pole} material={mats.pole} />
      <mesh ref={head} geometry={geo.head} material={mats.head} position={[0, 2.95, 0]} />
      <sprite ref={glow} material={mats.glow} position={[0, 2.95, 0]} scale={2} renderOrder={4} />
      <mesh ref={ripple} geometry={geo.ripple} material={mats.ripple} position={[0, 0.08, 0]} renderOrder={3} />
      {status === 'sampled' && <mesh geometry={geo.base} material={mats.base} position={[0, 0.07, 0]} renderOrder={3} />}
      <TargetFeature target={target} />
    </group>
    </>
  )
}

export function Targets() {
  const targets = useSim((s) => s.targets)
  const gen = useSim((s) => s.gen)
  return (
    <group>
      {targets.map((t) => (
        <TargetPin key={`${gen}:${t.id}`} target={t} />
      ))}
    </group>
  )
}
