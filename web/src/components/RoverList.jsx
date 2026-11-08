import { memo, useEffect, useState } from 'react'
import { useSim } from '../state/store.js'
import { fixed } from '../lib/format.js'
import { kindStyle } from '../lib/theme.js'
import { focusStatus, focusTitle } from '../lib/focus.js'
import { Battery, BrainChip, StatePill } from './RoverBits.jsx'
import { SamplingLine, Tubes } from './SampleBits.jsx'
import { HealthRow } from './HealthBits.jsx'
import { V9_STATES, rescueLine } from '../lib/rescue.js'

// v7: the card shows the focus line unless something more specific is going on.
const FOCUS_HIDDEN = new Set(['observing', 'returning', 'recovering', 'charging', 'fault', ...V9_STATES])
const showsFocus = (rover) => !!rover?.focus && !rover.sampling && !FOCUS_HIDDEN.has(rover.state)

function goalText(rover, targets, lander, tracks, askingLife, v9) {
  if (!rover) return 'Waiting for telemetry'
  const dist = (p) => Math.hypot(p[0] - rover.x, p[1] - rover.y)
  // v9: in trouble, helping, towing, under repair or left as a wreck.
  if (v9 && V9_STATES.has(rover.state)) return rescueLine(rover, v9.rescues, v9.name, v9.dist, v9.asking) || 'Waiting for telemetry'
  if (rover.state === 'charging') return 'Charging at the lander'
  if (rover.state === 'fault') return 'Motors off, rebooting'
  // v4: watching a lifeform track. v7: it follows at a safe distance while it asks you.
  if (rover.state === 'observing') {
    const k = rover.track_id ? tracks.find((q) => q.id === rover.track_id) : null
    if (askingLife) return k ? `Following ${k.id} at a safe distance, waiting for you` : 'Following at a safe distance, waiting for you'
    if (!k) return 'Watching something move'
    return `Watching ${k.id}, ${k.label || 'a moving object'}, ${Math.round(dist([k.x, k.y]))} m`
  }
  // v7: concentrating on an area (the operator's, or its own curiosity).
  if (showsFocus(rover)) return focusStatus(rover.focus)
  if (rover.state === 'returning' && Number.isFinite(rover.capacity) && rover.samples?.length >= rover.capacity) return 'Tubes full, taking the samples to the lander'
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
    return `Heading to its goal, ${d} m away`
  }
  if (rover.state === 'recovering') return 'Backing off, replanning'
  if (rover.state === 'returning') return 'Heading back to the lander'
  if (rover.state === 'idle') return 'Waiting for a new goal'
  return 'No goal'
}

const RoverCard = memo(function RoverCard({ meta, rover, targets, lander, hazards, budget, tracks, askingLife, v9, selected, following, onSelect }) {
  const toggle = () => onSelect(meta.id)
  const lit = selected || following
  const trouble = rover?.state === 'disabled' ? 'alert' : rover?.state === 'wrecked' ? 'wreck' : undefined
  return (
    <li>
      <div
        className="rover"
        role="button"
        tabIndex={0}
        aria-pressed={selected}
        data-following={lit || undefined}
        data-fault={rover?.state === 'fault' || undefined}
        data-trouble={trouble}
        data-health={rover?.health ? true : undefined}
        title={[
          selected ? 'Close the sensor panel' : `Show ${meta.name}'s sensors and follow it`,
          rover?.goal ? `Goal (${fixed(rover.goal[0], 1)}, ${fixed(rover.goal[1], 1)})` : null,
          Number.isFinite(rover?.brain?.avg_ms) ? `Brain ${fixed(rover.brain.avg_ms, 1)} ms average` : null,
        ]
          .filter(Boolean)
          .join('\n')}
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
          {rover?.capacity && <Tubes samples={rover.samples} capacity={rover.capacity} targets={targets} />}
          <StatePill rover={rover} hazards={hazards} />
        </div>
        <div className="rover-mid">
          <Battery value={rover?.battery} charging={rover?.state === 'charging'} />
          <span className="rover-speed">
            <BrainChip brain={rover?.brain} budget={budget} />
            <span>{rover ? `${fixed(rover.speed, 1)} m/s` : '--'}</span>
          </span>
        </div>
        <div className="rover-goal" data-focus={showsFocus(rover) ? rover.focus.reason : undefined} data-v9={(v9 && V9_STATES.has(rover?.state) && rover.state) || undefined} title={showsFocus(rover) ? focusTitle(rover.focus) : undefined}>
          {rover?.sampling && !V9_STATES.has(rover.state) ? <SamplingLine sampling={rover.sampling} target={targets.find((t) => t.id === rover.sampling.target)} /> : goalText(rover, targets, lander, tracks, askingLife, v9)}
        </div>
        {rover?.health && <HealthRow health={rover.health} />}
      </div>
    </li>
  )
})

// What the card's rescue line needs: names, distances to the other rover, the rescues. Kept
// as stable as the inputs so the memoised cards only re-render when something changed.
const ctxCache = new Map()
function rescueCtx(id, rovers, meta, rescues, asking) {
  const me = rovers[id]
  const key = `${id}`
  const prev = ctxCache.get(key)
  const askingMe = asking.split(',').includes(id)
  // Distances move every tick, but only matter while a helper is on its way: round them.
  const other = rescues.find((r) => (r.rover === id || r.helper === id) && r.phase !== 'done')
  const otherId = other ? (other.rover === id ? other.helper : other.rover) : null
  const o = otherId ? rovers[otherId] : null
  const d = me && o ? Math.round(Math.hypot(me.x - o.x, me.y - o.y)) : null
  if (prev && prev.rescues === rescues && prev.meta === meta && prev.d === d && prev.asking === askingMe) return prev
  const names = Object.fromEntries(meta.map((m) => [m.id, m.name]))
  const next = { rescues, meta, d, asking: askingMe, name: (x) => names[x] || x, dist: (x) => (x === otherId ? d : null) }
  ctxCache.set(key, next)
  return next
}

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

export function RoverList({ follow, selected, onSelect }) {
  const meta = useSim((s) => s.roverMeta)
  const rovers = useSim((s) => s.rovers)
  const targets = useSim((s) => s.targets)
  const lander = useSim((s) => s.lander)
  const hazards = useSim((s) => s.hazards)
  const budget = useSim((s) => s.fleet.budget_ms)
  const tracks = useSim((s) => s.tracks)
  // v7: rovers with an open lifeform question (they follow the herd while they wait).
  const askingLife = useSim((s) => s.requests.filter((q) => q.kind === 'lifeform').map((q) => q.rover).join(','))
  // v9: rescues in progress, and rovers waiting for a rescue decision (helper or subject).
  const rescues = useSim((s) => s.rescues)
  const askingRescue = useSim((s) => s.requests.filter((q) => q.kind === 'rescue').flatMap((q) => [q.rover, q.about]).join(','))
  const v9 = useSim((s) => s.v9)
  // Two-row cards on shorter screens or bigger fleets, so the event log keeps its room.
  const short = useMediaQuery('(max-height: 960px)')
  const compact = short || meta.length > 3
  return (
    <section className="rovers" aria-label="Rovers">
      <ul className="rover-list" data-count={meta.length} data-compact={compact || undefined}>
        {meta.map((m) => (
          <RoverCard
            key={m.id}
            meta={m}
            rover={rovers[m.id]}
            targets={targets}
            lander={lander}
            hazards={hazards}
            budget={budget}
            tracks={tracks}
            askingLife={askingLife.split(',').includes(m.id)}
            v9={v9 ? rescueCtx(m.id, rovers, meta, rescues, askingRescue) : null}
            selected={selected === m.id}
            following={follow === m.id}
            onSelect={onSelect}
          />
        ))}
        {!meta.length && <li className="empty-note">No rovers yet.</li>}
      </ul>
    </section>
  )
}
