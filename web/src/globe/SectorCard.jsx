import { useSim } from '../state/store.js'
import { fixed, mmss } from '../lib/format.js'
import { stateStyle } from '../lib/theme.js'
import { gravityG } from './globeMath.js'
import { globeSettled, leaveGlobe, useViewMode } from './viewMode.js'
import '../styles/globe.css'

// The landing sector's live card on the planet. drei <Html> renders it into its own React
// root next to the canvas, so it reads the stores directly. A rover row selects that rover,
// which flies back down first (App.jsx). Pointer presses stop here, so a click on the card
// never starts an orbit drag.
export function SectorCard({ cardRef, onSelect }) {
  const map = useSim((s) => s.map)
  const world = useSim((s) => s.world)
  const explored = useSim((s) => s.fleet.explored_pct)
  const t = useSim((s) => s.t)
  const meta = useSim((s) => s.roverMeta)
  const rovers = useSim((s) => s.rovers)
  const settled = useViewMode(globeSettled)
  const size = map ? `${Math.round(map.width)} x ${Math.round(map.depth)} m` : 'Size unknown'
  return (
    <section ref={cardRef} className="globe-card" data-settled={settled || undefined} aria-label="Landing sector" onPointerDown={(e) => e.stopPropagation()}>
      <h2 className="gc-title">Landing sector</h2>
      <p className="gc-sub">{size}, not to scale</p>
      <dl className="gc-stats">
        <div>
          <dt>Explored</dt>
          <dd>{fixed(explored, 1)} %</dd>
        </div>
        <div>
          <dt>Gravity</dt>
          <dd>{fixed(gravityG(world), 1)} g</dd>
        </div>
        <div>
          <dt>Sim time</dt>
          <dd>{mmss(t)}</dd>
        </div>
      </dl>
      <ul className="gc-rovers" aria-label="Rovers">
        {meta.map((m) => {
          const s = stateStyle(rovers[m.id]?.state)
          return (
            <li key={m.id}>
              <button type="button" className="gc-rover" onClick={() => onSelect?.(m.id)} title={`Fly down and follow ${m.name}`}>
                <span className="gc-dot" style={{ background: m.color }} />
                <span className="gc-name">{m.name}</span>
                <span className="gc-state" style={{ color: s.color }}>
                  {s.label}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      <button type="button" className="btn gc-down" onClick={() => leaveGlobe()}>
        Fly down to the sector
      </button>
    </section>
  )
}
