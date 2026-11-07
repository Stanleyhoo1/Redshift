import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { simStore, useSim } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { SIGNAL } from '../lib/theme.js'
import { glowTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'

// The lander: a hexagonal pad with a soft glow and a slow beacon.
export function Lander() {
  const lander = useSim((s) => s.lander)
  const labelsReady = useThree((s) => !!s.events.connected) // see Rovers.jsx
  const cached = useSim((s) => (Number.isFinite(s.fleet.samples_cached) ? s.fleet.samples_cached : null)) // v4b
  const group = useRef()
  const beacon = useRef()
  const h = useRef(null)
  const R = lander?.radius ?? 2.5

  const parts = useMemo(() => {
    return {
      pad: new THREE.CylinderGeometry(R, R * 1.05, 0.24, 6),
      deck: new THREE.RingGeometry(R * 0.7, R * 0.78, 6).rotateX(-Math.PI / 2),
      edge: new THREE.RingGeometry(R * 0.97, R * 1.02, 6).rotateX(-Math.PI / 2),
      glow: new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
      mast: new THREE.CylinderGeometry(0.05, 0.05, 1.6, 6).translate(0, 0.8, 0),
      bulb: new THREE.SphereGeometry(0.16, 12, 8),
      // Low metalness: there is no environment map, so a metallic pad would render near black.
      padMat: new THREE.MeshStandardMaterial({ color: '#5a6578', roughness: 0.6, metalness: 0.2, flatShading: true }),
      lineMat: new THREE.MeshBasicMaterial({ color: SIGNAL, transparent: true, opacity: 0.9 }),
      edgeMat: new THREE.MeshBasicMaterial({ color: '#7d8da3', transparent: true, opacity: 0.7 }),
      glowMat: new THREE.MeshBasicMaterial({ color: SIGNAL, map: glowTexture(), transparent: true, opacity: 0.32, depthWrite: false, blending: THREE.AdditiveBlending }),
      mastMat: new THREE.MeshStandardMaterial({ color: '#8e98a8', roughness: 0.5, metalness: 0.5 }),
      bulbMat: new THREE.MeshBasicMaterial({ color: '#dffbff', transparent: true }),
    }
  }, [R])
  useEffect(
    () => () => Object.values(parts).forEach((p) => p.dispose?.()),
    [parts],
  )

  useFrame(({ clock }) => {
    if (!lander || !group.current) return
    const g = sampleGroundNear(simStore.get().terrain, lander.x, lander.y, 10) ?? 0
    h.current = h.current === null ? g : h.current + (g - h.current) * 0.1
    group.current.position.set(lander.x, h.current, -lander.y)
    if (beacon.current && !reducedMotion) {
      const blink = Math.pow(0.5 + 0.5 * Math.sin(clock.elapsedTime * 2.4), 6)
      parts.bulbMat.opacity = 0.35 + 0.65 * blink
    }
  })

  if (!lander) return null
  return (
    <group ref={group}>
      <mesh geometry={parts.glow} material={parts.glowMat} scale={R * 4.2} position={[0, 0.05, 0]} renderOrder={3} />
      <mesh geometry={parts.pad} material={parts.padMat} position={[0, 0.12, 0]} />
      <mesh geometry={parts.deck} material={parts.lineMat} position={[0, 0.25, 0]} />
      <mesh geometry={parts.edge} material={parts.edgeMat} position={[0, 0.25, 0]} />
      <mesh geometry={parts.mast} material={parts.mastMat} position={[0, 0.24, 0]} />
      <mesh ref={beacon} geometry={parts.bulb} material={parts.bulbMat} position={[0, 1.9, 0]} />
      {/* Below the pad (on screen), so it does not collide with the tags of docked rovers above them. */}
      {labelsReady && (
        <Html position={[0, 0.3, R + 1.1]} center zIndexRange={[11, 0]} style={{ pointerEvents: 'none' }}>
          <div className="lander-tag">
            Lander
            {cached !== null && (
              <span className="lander-cache" data-some={cached > 0 || undefined} title="Samples in the lander's cache">
                <i aria-hidden="true" />
                {cached} cached
              </span>
            )}
          </div>
        </Html>
      )}
    </group>
  )
}
