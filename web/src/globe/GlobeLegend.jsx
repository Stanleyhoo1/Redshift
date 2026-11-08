import { useMemo } from 'react'
import { useSim } from '../state/store.js'
import { fixed } from '../lib/format.js'
import { SIGNAL } from '../lib/theme.js'
import { GLOBE } from './globePalette.js'
import { formatKm, formatLatLon, planetLook } from './globeMath.js'
import '../styles/scene.css'
import '../styles/globe.css'

// The short legend on the planet view, in the map legend's place and style.
export function GlobeLegend() {
  const world = useSim((s) => s.world)
  const p = useMemo(() => planetLook(world), [world])
  return (
    <div className="hud hud-legend globe-legend" aria-label="Planet legend">
      <div className="legend-title">{p.name}</div>
      <p className="globe-facts">
        Radius {formatKm(p.radiusKm)}, Earth's density at {fixed(p.g, 1)} g
        <br />
        Landing site {formatLatLon(p.site)}
      </p>
      <ul className="legend-list">
        <li>
          <i className="sw sw-ring" style={{ '--c': SIGNAL }} />
          Landing sector <span>not to scale</span>
        </li>
        <li>
          <i className="sw" style={{ background: GLOBE.ice }} />
          Polar ice
        </li>
        <li>
          <i className="sw" style={{ background: GLOBE.mare }} />
          Dark plains
        </li>
      </ul>
      <div className="legend-hint">
        Drag to turn, scroll to zoom. <kbd>0</kbd> frames the site, <kbd>G</kbd> flies down.
      </div>
    </div>
  )
}
