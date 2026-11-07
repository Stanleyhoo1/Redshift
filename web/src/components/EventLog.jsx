import { memo, useLayoutEffect, useMemo, useRef } from 'react'
import { useSim } from '../state/store.js'
import { cleanMsg, mmss } from '../lib/format.js'
import { SourceTag } from './Planner.jsx'
import { LifeIcon } from './FleetStats.jsx'

// A herd brought in from the hazard bar arrives as a "hazard" event (an environment
// change), but it is a find, not a danger: shown as such.
const LIFE_RE = /\b(herd|creatures?|lifeforms?|grazers?)\b/i

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
        {e.source !== 'rules' && <SourceTag source={e.source} />}
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
      <li className="ev ev-link" data-level={level} data-live={live || undefined}>
        <button type="button" className="ev-btn" onClick={() => onFocus({ ...e, level: hazard ? 'hazard' : e.level === 'hazard' ? 'discovery' : e.level })} title={`Show where this happened (${e.at[0].toFixed(0)}, ${e.at[1].toFixed(0)})`}>
          {content}
        </button>
      </li>
    )
  }
  return (
    <li className="ev" data-level={level} data-live={live || undefined}>
      {content}
    </li>
  )
})

// Newest first. Rendered under the sidebar's "Events" tab (the tab bar is the heading).
export function EventLog({ onFocusEvent, hidden }) {
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
  }, [events, gen, hidden])

  return (
    <ol
      className="log-list"
      id="panel-events"
      role="tabpanel"
      aria-labelledby="tab-events"
      hidden={hidden}
      ref={list}
      aria-live={hidden ? 'off' : 'polite'}
      onScroll={(ev) => {
        atTop.current = ev.currentTarget.scrollTop < 24
      }}
    >
      {events.map((e) => (
        <EventRow key={e.key} e={e} rover={e.rover ? byId[e.rover] : null} live={e.key > initSeq} byId={byId} onFocus={onFocusEvent} />
      ))}
      {!events.length && <li className="empty-note">Rover decisions and discoveries will appear here.</li>}
    </ol>
  )
}
