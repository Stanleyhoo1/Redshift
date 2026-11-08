import { useMemo } from 'react'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { groundAt, knownAt } from './ground.js'

// v7 ice sheets (PROTOCOL.md `features.ice`): pale glossy blue discs on the level ground.
// Reality shows every sheet; otherwise a sheet shows once the fleet has mapped its centre.
export function IceSheets({ reality }) {
  const features = useSim((s) => s.features)
  const gen = useSim((s) => s.gen)
  const ice = (features && features.ice) || []
  const mat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#cfe8ff', roughness: 0.08, metalness: 0.1, transparent: true, opacity: 0.85,
    polygonOffset: true, polygonOffsetFactor: -2,
  }), [])
  return (
    <group>
      {ice.filter((s) => reality || knownAt(s.x, s.y)).map((s) => (
        <mesh key={`${gen}-${s.id}`} position={[s.x, groundAt(s.x, s.y) + 0.03, -s.y]} rotation={[-Math.PI / 2, 0, 0]}
          material={mat}>
          <circleGeometry args={[s.radius, 40]} />
        </mesh>
      ))}
    </group>
  )
}
