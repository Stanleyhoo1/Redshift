import { memo, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useSim } from '../state/store.js'
import { cleanMsg, mmss, shortName } from '../lib/format.js'
import { GemmaStar, SourceTag } from './Planner.jsx'
import { LifeIcon } from './FleetStats.jsx'

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

const EventRow = memo(function EventRow({ e, rover, live, byId, onFocus }) {
  const life = e.level === 'hazard' && LIFE_RE.test(e.msg)
  const hazard = e.level === 'hazard' && !life
  const level = life ? 'life' : e.level
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
        <span className="ev-msg">{cleanMsg(e.msg, e.rover, byId)}</span>
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
function Filters({ filter, setFilter, counts, meta, hasDecisions }) {
  const chip = (id, label, n, extra = {}) => (
    <button key={id} type="button" className="evf" aria-pressed={filter === id} onClick={() => setFilter(id)} {...extra}>
      {label}
      <span className="evf-n">{n}</span>
    </button>
  )
  return (
    <div className="evf-row" role="toolbar" aria-label="Filter the event log">
      {chip('all', 'All', counts.all, { title: 'Every event' })}
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

const EMPTY = {
  all: 'Rover decisions and discoveries will appear here.',
  fleet: 'No fleet-wide events yet.',
  decisions: 'No decisions logged yet. In autonomous mode every choice a rover makes on its own (and why) appears here.',
}

// Newest first. Rendered under the sidebar's "Events" tab (the tab bar is the heading).
export function EventLog({ onFocusEvent, hidden }) {
  const events = useSim((s) => s.events)
  const meta = useSim((s) => s.roverMeta)
  const initSeq = useSim((s) => s.initSeq)
  const gen = useSim((s) => s.gen)
  const v8 = useSim((s) => typeof s.fleet.mode === 'string')
  const [filter, setFilter] = useState('all')
  const [limit, setLimit] = useState(PAGE)
  const list = useRef(null)
  const atTop = useRef(true)
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])

  const counts = useMemo(() => {
    const c = { all: events.length, fleet: 0, decisions: 0 }
    for (const e of events) {
      if (e.rover) c[e.rover] = (c[e.rover] || 0) + 1
      else c.fleet++
      if (e.decision) c.decisions++
    }
    return c
  }, [events])

  const shown = useMemo(() => {
    if (filter === 'all') return events
    if (filter === 'fleet') return events.filter((e) => !e.rover)
    if (filter === 'decisions') return events.filter((e) => e.decision)
    return events.filter((e) => e.rover === filter)
  }, [events, filter])

  // A rover that left (sim reset) cannot stay the filter.
  if (filter !== 'all' && filter !== 'fleet' && filter !== 'decisions' && meta.length && !byId[filter]) setFilter('all')

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
      {events.length > 0 && <Filters filter={filter} setFilter={pick} counts={counts} meta={meta} hasDecisions={v8 || counts.decisions > 0} />}
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
