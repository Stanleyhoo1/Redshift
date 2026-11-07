import { useEffect, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import { Grid } from '@react-three/drei'
import { useSim } from '../state/store.js'
import { VOID } from '../lib/theme.js'
import { FatPolyline } from './fatLine.js'
import { fogYFor } from './sceneConstants.js'

// The uncharted planet: a dark floor with a faint survey grid, sitting just below
// the lowest terrain so revealed ground always rises out of it.
export function Ground() {
  const map = useSim((s) => s.map)
  const size = useThree((s) => s.size)
  const fogY = fogYFor(map)
  const cx = map ? map.cx : 0
  const cz = map ? -map.cy : 0

  const outline = useMemo(() => {
    if (!map) return null
    const line = new FatPolyline(5, { color: '#36506e', width: 1.5, opacity: 0.9 })
    const x0 = map.ox
    const x1 = map.ox + map.width
    const z0 = -map.oy
    const z1 = -(map.oy + map.depth)
    const y = fogY + 0.03
    line.set([x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1, x0, y, z0], 5)
    return line
  }, [map, fogY])

  useEffect(() => () => outline?.dispose(), [outline])
  useEffect(() => {
    outline?.setResolution(size.width, size.height)
  }, [outline, size])

  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position={[cx, fogY - 0.06, cz]} renderOrder={-2}>
        <planeGeometry args={[1200, 1200]} />
        <meshBasicMaterial color={VOID} />
      </mesh>
      <Grid
        position={[cx, fogY, cz]}
        args={[200, 200]}
        cellSize={1}
        cellThickness={0.6}
        cellColor="#131b27"
        sectionSize={5}
        sectionThickness={1.1}
        sectionColor="#1e2b3c"
        fadeDistance={170}
        fadeStrength={1.6}
        infiniteGrid
        renderOrder={-1}
      />
      {outline && <primitive object={outline.object} />}
    </group>
  )
}
