import { useEffect, useMemo, useRef } from 'react'
import { useSim } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { planetLook } from './globeMath.js'
import { globeRefs } from './globeDebug.js'
import { Starfield } from './Starfield.jsx'
import { Planet } from './Planet.jsx'
import { GlobeRig } from './GlobeRig.jsx'

// Everything in the planet scene (an r3f portal with its own camera, see GlobeLayer.jsx).
// The planet turns inside `spin`; its sun is fixed in the planet's frame, so the site stays
// in daylight while it turns, and the stars stay still.
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
        <Planet look={look} high={high} />
      </group>
      <GlobeRig look={look} spin={spin} />
    </>
  )
}
