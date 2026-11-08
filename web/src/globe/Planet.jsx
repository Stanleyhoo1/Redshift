import { useEffect, useLayoutEffect, useMemo } from 'react'
import * as THREE from 'three'
import { AIR_R, applyLook, makeAirMaterial, makePlanetMaterial } from './planetMaterial.js'
import { globeRefs } from './globeDebug.js'

// The planet: a unit sphere whose surface is worked out per pixel from the direction (no
// texture, so no seams and no pole pinch), inside a shell of glowing air. Graphics Low
// compiles the same shader without GLOBE_HQ and uses fewer triangles.
export function Planet({ look, high }) {
  const geo = useMemo(() => new THREE.SphereGeometry(1, high ? 192 : 96, high ? 128 : 64), [high])
  const airGeo = useMemo(() => new THREE.SphereGeometry(AIR_R, high ? 96 : 64, high ? 64 : 32), [high])
  const mat = useMemo(() => makePlanetMaterial(high), [high])
  const air = useMemo(makeAirMaterial, [])
  // Before the next frame, so the first draw already has this world's look.
  useLayoutEffect(() => applyLook(mat, air, look), [mat, air, look])
  useEffect(() => {
    globeRefs.planet = mat
    return () => {
      if (globeRefs.planet === mat) globeRefs.planet = null
    }
  }, [mat])
  useEffect(() => () => geo.dispose(), [geo])
  useEffect(() => () => airGeo.dispose(), [airGeo])
  useEffect(() => () => mat.dispose(), [mat])
  useEffect(() => () => air.dispose(), [air])
  return (
    <>
      <mesh geometry={geo} material={mat} />
      <mesh geometry={airGeo} material={air} renderOrder={2} />
    </>
  )
}
