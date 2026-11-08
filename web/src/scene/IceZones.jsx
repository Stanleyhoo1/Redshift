import { useEffect, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { ICE } from '../lib/terrainTheme.js'
import { FatPolyline } from './fatLine.js'
import { hatchTexture } from './textures.js'
import { DrapedDisc, useDrape } from './drape.js'

// v7 learned ice zones (PROTOCOL.md `fleet.ice_zones`, [[x, y, r]]): ice the fleet found through
// wheel slip. Drawn like learned soft ground, a hatched disc draped on the terrain, but pale blue
// with a dashed outline. Nothing is drawn when the field is missing.

const SEGS = 48

function IceZone({ x, y, r }) {
  const size = useThree((s) => s.size)
  const disc = useMemo(() => new DrapedDisc(6, SEGS), [])
  const ring = useMemo(() => new Float32Array((SEGS + 1) * 3), [])
  const outline = useMemo(() => new FatPolyline(SEGS + 1, { color: ICE.zone, width: 2.2, opacity: 0.95, dashed: true, dashSize: 0.6, gapSize: 0.4 }), [])
  const mat = useMemo(() => new THREE.MeshBasicMaterial({
    color: ICE.zone, map: hatchTexture(), transparent: true, opacity: 0.5, depthWrite: false, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -3,
  }), [])
  useEffect(() => () => {
    disc.dispose()
    outline.dispose()
    mat.dispose()
  }, [disc, outline, mat])
  useEffect(() => outline.setResolution(size.width, size.height), [outline, size])
  useDrape(() => {
    disc.update(x, y, r, 0.1, ring)
    outline.set(ring, SEGS + 1)
  }, [x, y, r])
  return (
    <group>
      <mesh geometry={disc.geometry} material={mat} renderOrder={3} />
      <primitive object={outline.object} renderOrder={3} />
    </group>
  )
}

const okZone = (z) => Array.isArray(z) && z.length >= 3 && Number.isFinite(z[0]) && Number.isFinite(z[1]) && Number.isFinite(z[2]) && z[2] > 0

export function IceZones() {
  const zones = useSim((s) => s.fleet?.ice_zones)
  const gen = useSim((s) => s.gen)
  if (!Array.isArray(zones)) return null
  return (
    <group>
      {zones.filter(okZone).map((z, i) => (
        <IceZone key={`${gen}:${i}`} x={z[0]} y={z[1]} r={z[2]} />
      ))}
    </group>
  )
}
