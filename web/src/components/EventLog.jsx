import { memo, useLayoutEffect, useMemo, useRef } from 'react'
import { useSim } from '../state/store.js'
import { mmss } from '../lib/format.js'

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// The sim writes messages like "r2 sampled ice deposit t7 (found by r3)". The row already shows
// the rover's name, so drop the leading id and swap any other rover ids for their names.
function cleanMsg(msg, roverId, byId) {
  let out = msg
  if (roverId && out.startsWith(roverId + ' ')) out = out.slice(roverId.length + 1)
  const ids = Object.keys(byId)
  if (!ids.length) return out
  const re = new RegExp(`\\b(${ids.sort((a, b) => b.length - a.length).map(escapeRe).join('|')})\\b`, 'g')
  return out.replace(re, (id) => byId[id]?.name || id)
}

const EventRow = memo(function EventRow({ e, rover, live, byId, onFocus }) {
  const hazard = e.level === 'hazard'
  const content = (
    <>
      <span className="ev-time">{mmss(e.t)}</span>
      {hazard ? (
        <svg className="ev-icon" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M6 1.2 11 10.4H1z" />
          <path d="M6 4.6v2.6M6 8.6v.1" />
        </svg>
      ) : (
        <span className="ev-dot" style={rover ? { background: rover.color } : undefined} data-fleet={!rover || undefined} aria-hidden="true" />
      )}
      <span className="ev-text">
        <span className="ev-who">{hazard ? 'Hazard' : rover ? rover.name : e.rover || 'Fleet'}</span>{' '}
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
  // Events with a location fly the camera there.
  if (e.at && onFocus) {
    return (
      <li className="ev ev-link" data-level={e.level} data-live={live || undefined}>
        <button type="button" className="ev-btn" onClick={() => onFocus(e)} title={`Show where this happened (${e.at[0].toFixed(0)}, ${e.at[1].toFixed(0)})`}>
          {content}
        </button>
      </li>
    )
  }
  return (
    <li className="ev" data-level={e.level} data-live={live || undefined}>
      {content}
    </li>
  )
})

export function EventLog({ onFocusEvent }) {
  const events = useSim((s) => s.events)
  const meta = useSim((s) => s.roverMeta)
  const initSeq = useSim((s) => s.initSeq)
  const gen = useSim((s) => s.gen)
  const list = useRef(null)
  const atTop = useRef(true)
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])

  // A new init (sim reset or reconnect) starts a fresh log, shown from the top.
  useLayoutEffect(() => {
    atTop.current = true
  }, [gen])

  // Newest first. Stay pinned to the top unless someone scrolled down to read.
  useLayoutEffect(() => {
    if (atTop.current && list.current) list.current.scrollTop = 0
  }, [events, gen])

  return (
    <section className="log" aria-label="Event log">
      <div className="log-head">
        <h2>Event log</h2>
        <span className="log-count">{events.length ? `${events.length} events` : ''}</span>
      </div>
      <ol
        className="log-list"
        ref={list}
        aria-live="polite"
        onScroll={(ev) => {
          atTop.current = ev.currentTarget.scrollTop < 24
        }}
      >
        {events.map((e) => (
          <EventRow key={e.key} e={e} rover={e.rover ? byId[e.rover] : null} live={e.key > initSeq} byId={byId} onFocus={onFocusEvent} />
        ))}
        {!events.length && <li className="empty-note">Rover decisions and discoveries will appear here.</li>}
      </ol>
    </section>
  )
}
