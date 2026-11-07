import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { nowSeconds, simStore } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { HAZARD, SIGNAL } from '../lib/theme.js'
import { glowTexture } from './textures.js'

const LIFE = 4.5 // seconds: the fly-in takes about 1.5 s, then it keeps pulsing
const ringGeo = new THREE.RingGeometry(0.86, 1, 64).rotateX(-Math.PI / 2)
const beamGeo = new THREE.CylinderGeometry(0.06, 0.06, 1, 8, 1, true).translate(0, 0.5, 0)
const glowGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2)

// Where a clicked event happened: three rings spreading out and a thin beam, for a
// couple of seconds. Crimson for hazards, cyan for everything else.
export function FocusMarker({ focus }) {
  const group = useRef()
  const rings = useRef([])
  const beam = useRef()
  const glow = useRef()
  const born = useRef(-1e9)
  const color = focus?.hazard ? HAZARD : SIGNAL
  const mats = useMemo(
    () => ({
      ring: [0, 1, 2].map(() => new THREE.MeshBasicMaterial({ color, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending })),
      beam: new THREE.MeshBasicMaterial({ color, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
      glow: new THREE.MeshBasicMaterial({ color, map: glowTexture(), transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }),
    }),
    [color],
  )
  useEffect(() => () => [...mats.ring, mats.beam, mats.glow].forEach((m) => m.dispose()), [mats])
  useEffect(() => {
    if (focus) born.current = nowSeconds()
  }, [focus])

  useFrame(() => {
    const g = group.current
    if (!g) return
    const age = nowSeconds() - born.current
    if (!focus || age > LIFE) {
      g.visible = false
      return
    }
    g.visible = true
    const h = sampleGroundNear(simStore.get().terrain, focus.x, focus.y, 8) ?? 0
    g.position.set(focus.x, h + 0.1, -focus.y)
    const fade = 1 - Math.max(0, (age - LIFE + 0.8) / 0.8)
    for (let i = 0; i < 3; i++) {
      const r = rings.current[i]
      if (!r) continue
      const ph = (age * 0.9 + i / 3) % 1
      r.scale.setScalar(0.6 + ph * 4.2)
      mats.ring[i].opacity = (1 - ph) * 0.95 * fade
    }
    if (beam.current) {
      beam.current.scale.set(1, 6 + 1.5 * Math.sin(age * 6), 1)
      mats.beam.opacity = 0.75 * fade
    }
    if (glow.current) {
      glow.current.scale.setScalar(3.4 + 0.6 * Math.sin(age * 8))
      mats.glow.opacity = 0.8 * fade
    }
  })

  return (
    <group ref={group} visible={false}>
      {[0, 1, 2].map((i) => (
        <mesh key={i} ref={(el) => (rings.current[i] = el)} geometry={ringGeo} material={mats.ring[i]} renderOrder={9} />
      ))}
      <mesh ref={beam} geometry={beamGeo} material={mats.beam} renderOrder={9} />
      <mesh ref={glow} geometry={glowGeo} material={mats.glow} renderOrder={9} />
    </group>
  )
}
