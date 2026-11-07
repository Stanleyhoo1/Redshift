import { useFrame } from '@react-three/fiber'
import { useMemo } from 'react'
import * as THREE from 'three'
import { ATMO_FRAGMENT, ATMO_VERTEX, PLANET_FRAGMENT, PLANET_VERTEX } from './sectorShader'

/** The unexplored planet: grey noise, graticule, scan sweep, and a reddish atmospheric rim. */
export function PlanetFog() {
  const planet = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: PLANET_VERTEX,
        fragmentShader: PLANET_FRAGMENT,
        uniforms: { uTime: { value: 0 } },
      }),
    [],
  )
  const rim = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: ATMO_VERTEX,
        fragmentShader: ATMO_FRAGMENT,
        uniforms: {
          uColor: { value: new THREE.Color('#ff6a55') },
          uPower: { value: 3.6 },
          uStrength: { value: 0.6 },
          uBack: { value: 0 },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  )
  const halo = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: ATMO_VERTEX,
        fragmentShader: ATMO_FRAGMENT,
        uniforms: {
          uColor: { value: new THREE.Color('#d2405a') },
          uPower: { value: 2.4 },
          uStrength: { value: 0.55 },
          uBack: { value: 1 },
        },
        side: THREE.BackSide,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  )

  useFrame(({ clock }) => {
    planet.uniforms.uTime.value = clock.elapsedTime
  })

  return (
    <group>
      <mesh material={planet} renderOrder={0}>
        <sphereGeometry args={[1, 128, 96]} />
      </mesh>
      <mesh material={rim} renderOrder={5}>
        <sphereGeometry args={[1.012, 96, 64]} />
      </mesh>
      <mesh material={halo} renderOrder={-1}>
        <sphereGeometry args={[1.2, 96, 64]} />
      </mesh>
    </group>
  )
}
