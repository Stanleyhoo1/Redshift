import { useEffect, useMemo, useRef } from 'react'
import { useSim } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { planetLook } from './globeMath.js'
import { GLOBE } from './globePalette.js'
import { globeRefs } from './globeDebug.js'
import { Starfield } from './Starfield.jsx'
import { GlobeRig } from './GlobeRig.jsx'

// Everything in the planet scene (an r3f portal with its own camera, see GlobeLayer.jsx).
export function GlobeWorld() {
  const world = useSim((s) => s.world)
  const high = useGraphics() === 'high'
  const look = useMemo(() => planetLook(world), [world])
  const spin = useRef(null)
  useEffect(() => {
    globeRefs.info = look
    globeRefs.spin = spin.current
    return () => {
      globeRefs.info = null
      globeRefs.spin = null
    }
  }, [look])
  return (
    <>
      <Starfield high={high} />
      <group ref={spin}>
        {/* Stand-in until Task 13 draws the planet. */}
        <mesh>
          <sphereGeometry args={[1, 64, 32]} />
          <meshBasicMaterial color={GLOBE.mare} />
        </mesh>
      </group>
      <GlobeRig look={look} spin={spin} />
    </>
  )
}
