import { useSim } from '../state/store.js'
import { degrees, fixed } from '../lib/format.js'
import { KIND_STYLE, LIFE, SIGNAL, SLOPE, SPIRE, WATER } from '../lib/theme.js'
import '../styles/scene.css'
import { TerrainLegend } from './TerrainLegend.jsx'
import { biomeOf, groundSwatch } from '../lib/biomes.js'

export function Legend({ slopeColors, frontiers, life = true }) {
  const limit = useSim((s) => s.limit)
  const map = useSim((s) => s.map)
  const world = useSim((s) => s.world)
  const biome = biomeOf(world)
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
          <div className="legend-title" key={limit.changedAt ?? 'base'} data-flash={limit.changedAt !== null || undefined}>
            Slope limit {degrees(lim)} deg
          </div>
          <ul className="legend-list">
            <li title={biome ? `The ground as the rovers mapped it: ${biome.name.toLowerCase()}, rock on steep ground` : 'The ground as the rovers mapped it: dust on the flats, rock on steep ground'}>
              <i className="sw sw-ground" style={biome ? { background: groundSwatch(world) } : undefined} />
              Mapped ground <span>{map ? `${fixed(map.minH, 1)} to ${fixed(map.maxH, 1)} m` : '--'}</span>
            </li>
          </ul>
          <div className="legend-hint">
            Press <kbd>S</kbd> for slope colours
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
      <TerrainLegend />
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
