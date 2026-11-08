import { memo, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useSim } from '../state/store.js'
import { cleanMsg, mmss, shortName } from '../lib/format.js'
import { GemmaStar, SourceTag } from './Planner.jsx'
import { LifeIcon } from './FleetStats.jsx'
import { PartIcon } from './HealthBits.jsx'
import { causeText, partLabel, partLevel } from '../lib/theme.js'

// A herd brought in from the hazard bar arrives as a "hazard" event (an environment
// change), but it is a find, not a danger: shown as such.
const LIFE_RE = /\b(herd|creatures?|lifeforms?|grazers?)\b/i

const PAGE = 120 // rows rendered at first; "Show older" adds more (long logs stay fast)

// "proceed_slowly" -> "Proceed slowly", "reroute" -> "Reroute"; labels with spaces stay.
export function optionWord(s) {
  if (!s) return ''
  if (/\s/.test(s)) return s
  const t = s.replace(/[_-]+/g, ' ').trim()
  return t ? t[0].toUpperCase() + t.slice(1) : s
}

const confLevel = (c) => (!Number.isFinite(c) ? 'none' : c >= 0.75 ? 'high' : c >= 0.5 ? 'mid' : 'low')
const BY_LABEL = { gemma: 'Gemma', rules: 'Rules', operator: 'Operator' }

// v8: a decision the rover made on its own (event `decision`): who decided and how sure,
// the choice, and on expand the question, every option considered and why.
function Decision({ d }) {
  const [open, setOpen] = useState(false)
  const by = d.by || 'rules'
  const level = confLevel(d.confidence)
  const choice = optionWord(d.choice)
  const n = d.options.length
  return (
    <div className="dec" data-open={open || undefined}>
      <button type="button" className="dec-sum" aria-expanded={open} onClick={() => setOpen((v) => !v)} title={open ? 'Hide the details' : 'Show the options it weighed and why'}>
        <span className="dec-by" data-by={by}>
          {by === 'gemma' && <GemmaStar className="src-star" />}
          {BY_LABEL[by] || optionWord(by)}
          {Number.isFinite(d.confidence) && (
            <b className="dec-conf" data-level={level} title={level === 'low' ? 'Below 0.5: the rover takes the safest option' : 'Confidence'}>
              {d.confidence.toFixed(2)}
            </b>
          )}
        </span>
        <span className="dec-verb">chose</span>
        <span className="dec-choice">{choice || '--'}</span>
        {n > 1 && <span className="dec-of">of {n}</span>}
        <svg className="dec-chev" viewBox="0 0 10 10" aria-hidden="true">
          <path d="M3 2l3.5 3L3 8" />
        </svg>
      </button>
      {open && (
        <div className="dec-body">
          {d.question && (
            <p className="dec-q">
              <span>Question</span>
              {d.question}
            </p>
          )}
          {n > 0 && (
            <ul className="dec-opts" aria-label="Options considered">
              {d.options.map((o) => (
                <li key={o} data-chosen={o === d.choice || undefined}>
                  {o === d.choice && (
                    <svg viewBox="0 0 12 12" aria-hidden="true">
                      <path d="M2.2 6.4 4.8 9 9.8 3.2" />
                    </svg>
                  )}
                  {optionWord(o)}
                </li>
              ))}
            </ul>
          )}
          {d.why && (
            <p className="dec-q">
              <span>Why</span>
              {d.why}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

// v9: a hit (event `damage`): the part, before and after on a small bar, cause and shock.
function DamageLine({ d }) {
  const level = partLevel(d.after)
  const b = Number.isFinite(d.before) ? d.before : null
  const a = Number.isFinite(d.after) ? d.after : null
  return (
    <span className="dmg" data-level={level} title={`${partLabel(d.part)} ${b ?? '--'}% to ${a ?? '--'}%${d.cause ? `, ${causeText(d.cause)}` : ''}${Number.isFinite(d.shock_g) && d.shock_g > 0 ? `, ${d.shock_g.toFixed(1)} g` : ''}`}>
      <PartIcon part={d.part} className="dmg-icon" />
      <span className="dmg-part">{partLabel(d.part)}</span>
      <span className="dmg-bar" aria-hidden="true">
        {b !== null && <u style={{ width: `${b}%` }} />}
        {a !== null && <i style={{ width: `${a}%` }} />}
      </span>
      <span className="dmg-nums">
        {b ?? '--'}
        <svg viewBox="0 0 10 8" aria-hidden="true">
          <path d="M1 4h7M5.5 1.5 8 4 5.5 6.5" />
        </svg>
        <b>{a ?? '--'}</b>
      </span>
      {d.cause && <span className="dmg-cause">{causeText(d.cause)}</span>}
      {Number.isFinite(d.shock_g) && d.shock_g > 0 && <span className="dmg-g">{d.shock_g.toFixed(1)} g</span>}
    </span>
  )
}

const EventRow = memo(function EventRow({ e, rover, live, byId, onFocus }) {
  const life = e.level === 'hazard' && LIFE_RE.test(e.msg)
  const hazard = e.level === 'hazard' && !life
  const level = life ? 'life' : e.damage ? 'damage' : e.level
  const content = (
    <>
      <span className="ev-time">{mmss(e.t)}</span>
      {life ? (
        <span className="ev-icon ev-icon-life">
          <LifeIcon />
        </span>
      ) : hazard ? (
        <svg className="ev-icon" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M6 1.2 11 10.4H1z" />
          <path d="M6 4.6v2.6M6 8.6v.1" />
        </svg>
      ) : (
        <span className="ev-dot" style={rover ? { background: rover.color } : undefined} data-fleet={!rover || undefined} aria-hidden="true" />
      )}
      <span className="ev-text">
        <span className="ev-who">{life ? 'Lifeform' : hazard ? 'Hazard' : rover ? rover.name : e.rover || 'Fleet'}</span>{' '}
        {e.source !== 'rules' && !e.decision && <SourceTag source={e.source} />}
        <EventMsg text={cleanMsg(e.msg, e.rover, byId)} />
        {e.damage && <DamageLine d={e.damage} />}
        {e.at && (
          <span className="ev-goto" aria-hidden="true">
            <svg viewBox="0 0 12 12">
              <circle cx="6" cy="6" r="3.6" />
              <path d="M6 .8v2M6 9.2v2M.8 6h2M9.2 6h2" />
            </svg>
          </span>
        )}
      </span>
    </>
  )
  const decision = e.decision ? <Decision d={e.decision} /> : null
  // Events with a location fly the camera there (the decision details sit outside that
  // button, so expanding them never moves the camera).
  if (e.at && onFocus) {
    return (
      <li className="ev ev-link" data-level={level} data-live={live || undefined} data-decision={!!decision || undefined}>
        <button type="button" className="ev-btn" onClick={() => onFocus({ ...e, level: hazard ? 'hazard' : e.level === 'hazard' ? 'discovery' : e.level })} title={`Show where this happened (${e.at[0].toFixed(0)}, ${e.at[1].toFixed(0)})`}>
          {content}
        </button>
        {decision}
      </li>
    )
  }
  return (
    <li className="ev" data-level={level} data-live={live || undefined} data-decision={!!decision || undefined}>
      {content}
      {decision}
    </li>
  )
})

// The filter chips: All, one per rover (in its colour), Fleet (events without a rover) and
// Decisions (v8 events with a `decision`), each with its count.
function Filters({ filter, setFilter, counts, meta, hasDecisions, hasDamage }) {
  const chip = (id, label, n, extra = {}) => (
    <button key={id} type="button" className="evf" aria-pressed={filter === id} onClick={() => setFilter(id)} {...extra}>
      {label}
      <span className="evf-n">{n}</span>
    </button>
  )
  return (
    <div className="evf-row" role="toolbar" aria-label="Filter the event log">
      {chip('key', 'Highlights', counts.key, { title: 'Decisions, discoveries, hazards, warnings and damage (no routine lines)' })}
      {chip('all', 'All', counts.all, { title: 'Every event, including routine planning lines' })}
      {meta.map((m) =>
        chip(
          m.id,
          <>
            <i className="evf-dot" aria-hidden="true" />
            {shortName(m.name, m.id)}
          </>,
          counts[m.id] || 0,
          { style: { '--c': m.color }, 'data-rover': true, title: `${m.name} only`, 'aria-label': `${m.name}: ${counts[m.id] || 0} events` },
        ),
      )}
      {chip('fleet', 'Fleet', counts.fleet, { title: 'Fleet-wide events and hazards (no single rover)' })}
      {(hasDamage || filter === 'damage') &&
        chip(
          'damage',
          <>
            <svg className="evf-dmg" viewBox="0 0 12 12" aria-hidden="true">
              <path d="M6.6 1 3.4 6.6h2.4L5 11l3.6-6H6.2z" />
            </svg>
            Damage
          </>,
          counts.damage,
          { title: 'Hits and the parts they damaged (each rover filter shows its own too)', 'data-dmg': true },
        )}
      {(hasDecisions || filter === 'decisions') &&
        chip(
          'decisions',
          <>
            <GemmaStar className="evf-star" />
            Decisions
          </>,
          counts.decisions,
          { title: 'Choices the rovers made on their own: who decided, the options and why', 'data-dec': true },
        )}
    </div>
  )
}

// Highlights: what a judge should see (decisions, damage, hazards, warnings, discoveries), without the routine
// info lines (frontier switches, headings). The default view.
const isKey = (e) => !!(e.decision || e.damage || (e.level && e.level !== 'info'))

const EMPTY = {
  key: 'Decisions, discoveries, hazards and damage will appear here. "All" shows every routine line too.',
  all: 'Rover decisions and discoveries will appear here.',
  fleet: 'No fleet-wide events yet.',
  decisions: 'No decisions logged yet. In autonomous mode every choice a rover makes on its own (and why) appears here.',
  damage: 'No damage so far. Every hit a rover takes (part, before and after, cause, shock) appears here.',
}

// Newest first. Rendered under the sidebar's "Events" tab (the tab bar is the heading).
export function EventLog({ onFocusEvent, hidden }) {
  const events = useSim((s) => s.events)
  const meta = useSim((s) => s.roverMeta)
  const initSeq = useSim((s) => s.initSeq)
  const gen = useSim((s) => s.gen)
  const v8 = useSim((s) => typeof s.fleet.mode === 'string')
  const [filter, setFilter] = useState('key')
  const [limit, setLimit] = useState(PAGE)
  const list = useRef(null)
  const atTop = useRef(true)
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])

  const counts = useMemo(() => {
    const c = { all: events.length, key: 0, fleet: 0, decisions: 0, damage: 0 }
    for (const e of events) {
      if (isKey(e)) c.key++
      if (e.rover) c[e.rover] = (c[e.rover] || 0) + 1
      else c.fleet++
      if (e.decision) c.decisions++
      if (e.damage) c.damage++
    }
    return c
  }, [events])

  const shown = useMemo(() => {
    if (filter === 'all') return events
    if (filter === 'key') return events.filter(isKey)
    if (filter === 'fleet') return events.filter((e) => !e.rover)
    if (filter === 'decisions') return events.filter((e) => e.decision)
    if (filter === 'damage') return events.filter((e) => e.damage)
    return events.filter((e) => e.rover === filter)
  }, [events, filter])

  // A rover that left (sim reset) cannot stay the filter.
  if (filter !== 'all' && filter !== 'key' && filter !== 'fleet' && filter !== 'decisions' && filter !== 'damage' && meta.length && !byId[filter]) setFilter('all')

  const pick = (id) => {
    setFilter(id)
    setLimit(PAGE)
    atTop.current = true
    if (list.current) list.current.scrollTop = 0
  }

  // A new init (sim reset or reconnect) starts a fresh log, shown from the top.
  useLayoutEffect(() => {
    atTop.current = true
    setLimit(PAGE)
  }, [gen])

  // Newest first. Stay pinned to the top unless someone scrolled down to read.
  useLayoutEffect(() => {
    if (atTop.current && list.current) list.current.scrollTop = 0
  }, [events, gen, hidden, filter])

  const rover = byId[filter]
  const empty = rover ? `No events from ${rover.name} yet.` : EMPTY[filter] || EMPTY.all
  const rows = shown.length > limit ? shown.slice(0, limit) : shown
  return (
    <div className="log" id="panel-events" role="tabpanel" aria-labelledby="tab-events" hidden={hidden}>
      {events.length > 0 && <Filters filter={filter} setFilter={pick} counts={counts} meta={meta} hasDecisions={v8 || counts.decisions > 0} hasDamage={counts.damage > 0} />}
      <ol
        className="log-list"
        ref={list}
        aria-live={hidden ? 'off' : 'polite'}
        onScroll={(ev) => {
          atTop.current = ev.currentTarget.scrollTop < 24
        }}
      >
        {rows.map((e) => (
          <EventRow key={e.key} e={e} rover={e.rover ? byId[e.rover] : null} live={e.key > initSeq} byId={byId} onFocus={onFocusEvent} />
        ))}
        {!shown.length && <li className="empty-note">{empty}</li>}
        {shown.length > limit && (
          <li className="log-more">
            <button type="button" onClick={() => setLimit((n) => n + PAGE * 2)}>
              Show older ({shown.length - limit} more)
            </button>
          </li>
        )}
      </ol>
    </div>
  )
}

// A Gemma decision quotes its reasoning after ": '": that part goes on its own dimmer line,
// clamped to two lines (the full text in the tooltip), so one event no longer fills the column.
// Any other message renders as before.
function EventMsg({ text }) {
  const msg = typeof text === 'string' ? text : String(text ?? '')
  const cut = msg.indexOf(": '")
  if (cut <= 0) return <span className="ev-msg">{msg}</span>
  const why = msg.slice(cut + 2)
  return (
    <span className="ev-msg">
      {msg.slice(0, cut + 1)}
      <span className="ev-why" title={why}>
        {why}
      </span>
    </span>
  )
}
