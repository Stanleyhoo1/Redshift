import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { groundAt } from './ground.js'
import { reducedMotion } from './sceneConstants.js'

// Version 8 fleet `priority`: when a sim also sends where the prioritised places are
// (not in the protocol; `places`, `gaps` or `areas`), each gets a soft amber disc with a
// dashed-looking ring on the terrain and a small tag ("Gap 2"). Without positions nothing
// is drawn here (the header chip still says what the priority is). Materials are shared;
// the frame loop only writes opacities.

const AMBER = '#f3b24a'
const geo = {
  ring: new THREE.RingGeometry(0.9, 1, 72, 1).rotateX(-Math.PI / 2),
  disc: new THREE.CircleGeometry(1, 48).rotateX(-Math.PI / 2),
}

function Place({ p, n, kind, mats }) {
  const labelsReady = useThree((s) => !!s.events.connected)
  const group = useRef()
  const gz = useMemo(() => groundAt(p.x, p.y, 0), [p.x, p.y])
  const label = p.label || (kind === 'gaps' ? `Gap ${n}` : `Priority ${n}`)
  return (
    <group ref={group} position={[p.x, gz + 0.12, -p.y]}>
      <mesh geometry={geo.disc} material={mats.disc} scale={[p.r, 1, p.r]} renderOrder={3} />
      <mesh geometry={geo.ring} material={mats.ring} scale={[p.r, 1, p.r]} renderOrder={3} />
      {labelsReady && (
        <Html position={[0, 1.2, 0]} center zIndexRange={[9, 0]} style={{ pointerEvents: 'none' }}>
          <span className="prio-tag" title={`${label}: about ${Math.round(Math.PI * p.r * p.r)} m2`}>
            {label}
          </span>
        </Html>
      )}
    </group>
  )
}

export function PriorityPlaces() {
  const pr = useSim((s) => s.priority)
  const mats = useMemo(
    () => ({
      ring: new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.85, depthWrite: false, depthTest: false, toneMapped: false, fog: false }),
      disc: new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.12, depthWrite: false, depthTest: false, toneMapped: false, fog: false }),
    }),
    [],
  )
  useEffect(
    () => () => {
      mats.ring.dispose()
      mats.disc.dispose()
    },
    [mats],
  )
  const t = useRef(0)
  useFrame((_, dt) => {
    if (reducedMotion || !pr?.places.length) return
    t.current += dt
    const k = 0.5 + 0.5 * Math.sin(t.current * 2.2)
    mats.ring.opacity = 0.55 + 0.4 * k
    mats.disc.opacity = 0.08 + 0.08 * k
  })
  if (!pr?.places.length) return null
  return (
    <group>
      {pr.places.map((p, i) => (
        <Place key={`${i}:${p.x}:${p.y}`} p={p} n={i + 1} kind={pr.kind} mats={mats} />
      ))}
    </group>
  )
}
