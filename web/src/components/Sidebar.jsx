import { useSim } from '../state/store.js'
import { fixed } from '../lib/format.js'
import { FleetStats } from './FleetStats.jsx'
import { RoverList } from './RoverList.jsx'
import { EventLog } from './EventLog.jsx'

const WORD = 'Redshift'
const INK = [232, 238, 245]
const RED = [255, 92, 70]

// The name set as its own idea: each letter a little wider and a little redder,
// like light stretched on its way to us.
function Wordmark() {
  const n = WORD.length
  return (
    <h1 className="wordmark" aria-label={WORD}>
      {[...WORD].map((ch, i) => {
        const t = i / (n - 1)
        const k = Math.pow(t, 1.7)
        const rgb = INK.map((c, j) => Math.round(c + (RED[j] - c) * k))
        return (
          <span key={i} aria-hidden="true" style={{ fontStretch: `${84 + t * 41}%`, color: `rgb(${rgb.join(',')})` }}>
            {ch}
          </span>
        )
      })}
    </h1>
  )
}

function WorldFacts() {
  const world = useSim((s) => s.world)
  const g = Number.isFinite(world?.gravity_g) ? world.gravity_g : Number.isFinite(world?.gravity) ? world.gravity / 9.81 : null
  return (
    <dl className="facts">
      <div>
        <dt>World</dt>
        <dd>{world?.name || '--'}</dd>
      </div>
      <div>
        <dt>Gravity</dt>
        <dd>{g === null ? '--' : `${fixed(g, 1)} g`}</dd>
      </div>
      <div>
        <dt>Seed</dt>
        <dd>{world?.seed ?? '--'}</dd>
      </div>
    </dl>
  )
}

export function Sidebar({ follow, onFollow }) {
  return (
    <aside className="side">
      <header className="side-head">
        <Wordmark />
        <p className="subtitle">Autonomous super-Earth survey</p>
        <WorldFacts />
      </header>
      <FleetStats />
      <RoverList follow={follow} onFollow={onFollow} />
      <EventLog />
    </aside>
  )
}
