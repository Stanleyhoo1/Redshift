import { useSim } from '../state/store.js'
import { degrees, fixed } from '../lib/format.js'
import { GROUND_RAMP, KIND_STYLE, LIFE, SIGNAL, SLOPE, SPIRE, WATER } from '../lib/theme.js'

export function Legend({ slopeColors, frontiers, life = true }) {
  const limit = useSim((s) => s.limit)
  const map = useSim((s) => s.map)
  const v4 = useSim((s) => s.v4)
  const hasTrace = useSim((s) => s.v4 || s.targets.some((t) => t.kind === 'trace'))
  const lim = Number.isFinite(limit.current) ? limit.current : 30
  const caution = lim * 0.6
  return (
    <div className="hud hud-legend" aria-label="Map legend">
      {slopeColors ? (
        <>
          <div className="legend-title" key={limit.changedAt ?? 'base'} data-flash={limit.changedAt !== null || undefined}>
            Ground slope, limit {degrees(lim)} deg
          </div>
          <ul className="legend-list">
            <li>
              <i className="sw" style={{ background: SLOPE.ok }} />
              Comfortable <span>under {degrees(Math.round(caution * 10) / 10)}</span>
            </li>
            <li>
              <i className="sw" style={{ background: SLOPE.caution }} />
              Caution{' '}
              <span>
                {degrees(Math.round(caution * 10) / 10)} to {degrees(lim)}
              </span>
            </li>
            <li>
              <i className="sw" style={{ background: SLOPE.bad }} />
              Impassable <span>{degrees(lim)} and up</span>
            </li>
          </ul>
        </>
      ) : (
        <>
          <div className="legend-title">Ground height</div>
          <div className="ramp" style={{ background: `linear-gradient(90deg, ${GROUND_RAMP.join(', ')})` }} />
          <div className="ramp-ends">
            <span>{map ? `${fixed(map.minH, 1)} m` : '--'}</span>
            <span>{map ? `${fixed(map.maxH, 1)} m` : '--'}</span>
          </div>
        </>
      )}
      <ul className="legend-list legend-row">
        <li>
          <i className="sw sw-void" />
          Unexplored
        </li>
        {frontiers && (
          <li>
            <i className="sw sw-ring" style={{ '--c': SIGNAL }} />
            Frontier
          </li>
        )}
      </ul>
      {v4 && (
        <ul className="legend-list legend-row">
          <li title="Liquid water the fleet has identified (LiDAR gets no returns from it)">
            <i className="sw sw-water" style={{ '--c': WATER.mid, '--c2': WATER.deep }} />
            Water
          </li>
          <li title="Alien crystal spires: LiDAR obstacles">
            <i className="sw sw-spire" style={{ '--c': SPIRE.body, '--g': SPIRE.glow }} />
            Spire
          </li>
          {life && (
            <>
              <li title="A creature the fleet is tracking">
                <i className="sw sw-life" style={{ '--c': LIFE }} />
                Lifeform
              </li>
              <li title="A fleet track: what the rovers believe is moving there">
                <i className="sw sw-track" style={{ '--c': LIFE }} />
                Track
              </li>
            </>
          )}
        </ul>
      )}
      <ul className="legend-list legend-row">
        {Object.entries(KIND_STYLE)
          .filter(([k]) => k !== 'trace' || hasTrace)
          .map(([k, v]) => (
            <li key={k}>
              <i className="sw sw-diamond" style={{ '--c': v.color }} />
              {v.label}
            </li>
          ))}
      </ul>
    </div>
  )
}
