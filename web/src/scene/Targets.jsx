import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { appStore, simStore, useApp, useSim } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { instrumentStyle, kindStyle } from '../lib/theme.js'
import { InstrumentIcon } from '../components/SampleBits.jsx'
import { FatPolyline } from './fatLine.js'
import { glowTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'

const geo = {
  pole: new THREE.CylinderGeometry(0.05, 0.05, 2.5, 8).translate(0, 1.25, 0),
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
      pole: new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7 }),
      head: new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5, roughness: 0.35, flatShading: true }),
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
      mats.head.emissiveIntensity = 0.75
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
    mats.head.emissiveIntensity = fast ? 1.1 + 0.3 * pulse : 0.35 + 0.65 * pulse
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
      mats.ripple.opacity = 0.85 * (1 - ph)
    }
  })

  const analysed = status === 'sampled' && (target.value != null || target.analysis != null)
  return (
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
    </group>
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
