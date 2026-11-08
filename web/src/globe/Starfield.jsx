import { useEffect, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { reducedMotion } from '../scene/sceneConstants.js'
import { GLOBE } from './globePalette.js'
import { makeStarMaterial } from './starMaterial.js'
import { starData } from './stars.js'

const TINTS = [GLOBE.starWarm, GLOBE.starWhite, GLOBE.starCool].map((h) => new THREE.Color(h))

// A fixed sky behind the planet. It sits outside the planet's spin, so the stars hold still
// while the planet turns. Graphics Low: fewer stars, no twinkle.
export function Starfield({ high }) {
  const dpr = useThree((s) => s.viewport.dpr)
  const geo = useMemo(() => {
    const d = starData(high ? 2600 : 1200)
    const tint = new Float32Array(d.kind.length * 3)
    d.kind.forEach((k, i) => TINTS[k].toArray(tint, i * 3))
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(d.position, 3))
    g.setAttribute('aMag', new THREE.BufferAttribute(d.mag, 1))
    g.setAttribute('aTint', new THREE.BufferAttribute(tint, 3))
    g.setAttribute('aPhase', new THREE.BufferAttribute(d.phase, 1))
    return g
  }, [high])
  const mat = useMemo(makeStarMaterial, [])
  useEffect(() => () => geo.dispose(), [geo])
  useEffect(() => () => mat.dispose(), [mat])
  useFrame((state) => {
    const u = mat.uniforms
    u.uPx.value = dpr
    u.uTwinkle.value = high && !reducedMotion ? 1 : 0
    u.uTime.value = state.clock.elapsedTime
  })
  return <points geometry={geo} material={mat} frustumCulled={false} renderOrder={-1} />
}
