import { useMemo } from 'react'
import { useSim } from '../state/store.js'
import { fixed } from '../lib/format.js'
import { SIGNAL } from '../lib/theme.js'
import { GLOBE } from './globePalette.js'
import { groundSwatch } from '../lib/biomes.js'
import { formatKm, formatLatLon, planetLook } from './globeMath.js'
import '../styles/scene.css'
import '../styles/globe.css'

// The short legend on the planet view, in the map legend's place and style. With Reality
// off the planet is unsurveyed (only the fleet's map shows); with it on, the true planet.
export function GlobeLegend({ reality = false }) {
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
          Landing sector <span className="gl-note">(not to scale)</span>
        </li>
        {reality ? (
          <>
            {p.palette.biome && (
              <li title="The planet's ground at the landing site">
                <i className="sw" style={{ background: p.palette.ground.dust }} />
                {p.palette.biome.name}
              </li>
            )}
            <li>
              <i className="sw" style={{ background: GLOBE.ice }} />
              Polar ice
            </li>
            <li>
              <i className="sw" style={{ background: p.palette.biome ? p.palette.ground.soil : GLOBE.mare }} />
              Dark plains
            </li>
          </>
        ) : (
          <>
            <li title="The ground the rovers have mapped, live">
              <i className="sw sw-ground" style={p.palette.biome ? { background: groundSwatch(world) } : undefined} />
              Mapped ground
            </li>
            <li title="Not surveyed yet">
              <i className="sw" style={{ background: GLOBE.blank, border: '1px solid #3a3e45' }} />
              Unsurveyed
            </li>
          </>
        )}
      </ul>
      <div className="legend-hint">
        Drag to turn, scroll to zoom. <kbd>0</kbd> frames the site, <kbd>G</kbd> flies down.
        {!reality && (
          <>
            {' '}
            <kbd>R</kbd> shows the true planet.
          </>
        )}
      </div>
    </div>
  )
}
