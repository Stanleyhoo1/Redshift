import { memo, useEffect, useState } from 'react'
import { useSim } from '../state/store.js'
import { fixed } from '../lib/format.js'
import { kindStyle, stateStyle } from '../lib/theme.js'

function Battery({ value, charging }) {
  const pct = Number.isFinite(value) ? Math.round(value * 100) : null
  const level = pct === null ? 'none' : pct < 20 ? 'low' : pct < 45 ? 'mid' : 'ok'
  return (
    <div className="battery" data-level={level} data-charging={charging || undefined} title="Battery">
      <div className="battery-track" aria-hidden="true">
        <i style={{ width: `${pct ?? 0}%` }} />
      </div>
      <span className="battery-pct">{pct === null ? '--' : `${pct}%`}</span>
    </div>
  )
}

function goalText(rover, targets, lander) {
  if (!rover) return 'Waiting for telemetry'
  const dist = (p) => Math.hypot(p[0] - rover.x, p[1] - rover.y)
  if (rover.state === 'charging') return 'Charging at the lander'
  // Kept short: the state pill already says what kind of goal this is.
  if (rover.target_id) {
    const t = targets.find((x) => x.id === rover.target_id)
    if (!t) return `Target ${rover.target_id}`
    const what = String(kindStyle(t.kind).long || 'target')
    return `${what[0].toUpperCase()}${what.slice(1)} ${t.id}, ${Math.round(dist([t.x, t.y]))} m away`
  }
  if (rover.goal) {
    const d = Math.round(dist(rover.goal))
    if (rover.state === 'returning' && lander && Math.hypot(rover.goal[0] - lander.x, rover.goal[1] - lander.y) < lander.radius + 3) {
      return `To the lander, ${d} m away`
    }
    return `Goal (${fixed(rover.goal[0], 1)}, ${fixed(rover.goal[1], 1)}), ${d} m away`
  }
  if (rover.state === 'recovering') return 'Backing off, replanning'
  if (rover.state === 'returning') return 'Heading back to the lander'
  if (rover.state === 'idle') return 'Waiting for a new goal'
  return 'No goal'
}

const RoverCard = memo(function RoverCard({ meta, rover, targets, lander, following, onFollow }) {
  const st = stateStyle(rover?.state)
  const toggle = () => onFollow(following ? null : meta.id)
  return (
    <li>
      <div
        className="rover"
        role="button"
        tabIndex={0}
        aria-pressed={following}
        data-following={following || undefined}
        title={following ? 'Stop following' : `Follow ${meta.name}`}
        style={{ '--rover': meta.color }}
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            toggle()
          }
        }}
      >
        <div className="rover-top">
          <span className="rover-dot" aria-hidden="true" />
          <span className="rover-name">
            {meta.name}
            {following && <span className="rover-following">Following</span>}
          </span>
          <span className="pill" style={{ '--pill': st.color }}>
            {rover ? st.label : 'No data'}
          </span>
        </div>
        <div className="rover-mid">
          <Battery value={rover?.battery} charging={rover?.state === 'charging'} />
          <span className="rover-speed">{rover ? `${fixed(rover.speed, 1)} m/s` : '--'}</span>
        </div>
        <div className="rover-goal">{goalText(rover, targets, lander)}</div>
      </div>
    </li>
  )
})

function useMediaQuery(query) {
  const [match, setMatch] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const mq = window.matchMedia(query)
    const update = () => setMatch(mq.matches)
    update()
    mq.addEventListener('change', update)
    return () => mq.removeEventListener('change', update)
  }, [query])
  return match
}

export function RoverList({ follow, onFollow }) {
  const meta = useSim((s) => s.roverMeta)
  const rovers = useSim((s) => s.rovers)
  const targets = useSim((s) => s.targets)
  const lander = useSim((s) => s.lander)
  // Two-row cards on shorter screens or bigger fleets, so the event log keeps its room.
  const short = useMediaQuery('(max-height: 960px)')
  const compact = short || meta.length > 3
  return (
    <section className="rovers" aria-label="Rovers">
      <ul className="rover-list" data-count={meta.length} data-compact={compact || undefined}>
        {meta.map((m) => (
          <RoverCard key={m.id} meta={m} rover={rovers[m.id]} targets={targets} lander={lander} following={follow === m.id} onFollow={onFollow} />
        ))}
        {!meta.length && <li className="empty-note">No rovers yet.</li>}
      </ul>
    </section>
  )
}
