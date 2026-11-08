import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { reducedMotion } from '../scene/sceneConstants.js'
import { AIR_R, LOOK, applyLook, makeAirMaterial, makePlanetMaterial } from './planetMaterial.js'
import { viewMode } from './viewMode.js'
import { globeRefs } from './globeDebug.js'

// The planet: a unit sphere whose surface is worked out per pixel from the direction (no
// texture, so no seams and no pole pinch), inside a shell of glowing air. Graphics Low
// compiles the same shader without GLOBE_HQ and uses fewer triangles.
// Until Reality is on, the fleet has not surveyed the planet, so it is drawn blank (dark,
// with a faint graticule); Reality fades in the true surface over LOOK.revealS seconds
// (at once with reduced motion, or while the sector is drawn).
export function Planet({ look, high, reality = false }) {
  const geo = useMemo(() => new THREE.SphereGeometry(1, high ? 192 : 96, high ? 128 : 64), [high])
  const airGeo = useMemo(() => new THREE.SphereGeometry(AIR_R, high ? 96 : 64, high ? 64 : 32), [high])
  const mat = useMemo(() => makePlanetMaterial(high), [high])
  const air = useMemo(makeAirMaterial, [])
  // Before the next frame, so the first draw already has this world's look.
  useLayoutEffect(() => applyLook(mat, air, look), [mat, air, look])
  // A new material (Graphics) starts on the current look; a Reality flip fades below.
  const lastMat = useRef(null)
  useLayoutEffect(() => {
    const snap = lastMat.current !== mat || reducedMotion || viewMode.get().render !== 'globe'
    lastMat.current = mat
    if (snap) mat.uniforms.uReveal.value = reality ? 1 : 0
  }, [mat, reality])
  useFrame((_, dt) => {
    const u = mat.uniforms.uReveal
    const want = reality ? 1 : 0
    if (u.value === want) return
    if (reducedMotion || viewMode.get().render !== 'globe') u.value = want
    else u.value = THREE.MathUtils.clamp(u.value + Math.sign(want - u.value) * (Math.min(dt, 0.1) / LOOK.revealS), 0, 1)
  })
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
