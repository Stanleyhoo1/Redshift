import { useEffect, useMemo, useRef } from 'react'
import { useSim } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { planetLook } from './globeMath.js'
import { globeRefs } from './globeDebug.js'
import { Starfield } from './Starfield.jsx'
import { Planet } from './Planet.jsx'
import { SiteMarker } from './SiteMarker.jsx'
import { SectorMap } from './SectorMap.jsx'
import { GlobeRig } from './GlobeRig.jsx'

// Everything in the planet scene (an r3f portal with its own camera, see GlobeLayer.jsx).
// The planet and its marker turn inside `spin`; the sun is fixed in the planet's frame, so
// the site stays in daylight while it turns, and the stars stay still. The planet is blank
// until Reality is on; the fleet's map grows on the landing sector either way.
export function GlobeWorld({ onSelect, reality = false }) {
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
        <Planet look={look} high={high} reality={reality} />
        <SectorMap look={look} high={high} />
        <SiteMarker look={look} onSelect={onSelect} />
      </group>
      <GlobeRig look={look} spin={spin} />
    </>
  )
}
