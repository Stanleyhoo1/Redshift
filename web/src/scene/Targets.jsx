import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore, useSim } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { kindStyle } from '../lib/theme.js'
import { glowTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'

const geo = {
  pole: new THREE.CylinderGeometry(0.05, 0.05, 2.5, 8).translate(0, 1.25, 0),
  head: new THREE.OctahedronGeometry(0.44, 0),
  ripple: new THREE.RingGeometry(0.85, 1.0, 48).rotateX(-Math.PI / 2),
  base: new THREE.RingGeometry(1.1, 1.36, 48).rotateX(-Math.PI / 2),
}

// Discovered target: a beacon pin coloured by kind.
// found = pulsing, investigating = brighter and faster, sampled = steady with a ring.
function TargetPin({ target }) {
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

  return (
    <group ref={group}>
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
