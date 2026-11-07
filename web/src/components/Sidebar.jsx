import { useEffect, useRef, useState } from 'react'
import { useSim } from '../state/store.js'
import { fixed } from '../lib/format.js'
import { FleetStats } from './FleetStats.jsx'
import { RoverList } from './RoverList.jsx'
import { EventLog } from './EventLog.jsx'
import { CommsPanel } from './Comms.jsx'
import { PlannerPill } from './Planner.jsx'

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

// Unread rover and fleet messages that arrived live while the Comms tab was closed.
function useUnread(tab) {
  const chat = useSim((s) => s.chat)
  const initSeq = useSim((s) => s.chatInitSeq)
  const [seen, setSeen] = useState(0)
  const last = chat.length ? chat[chat.length - 1].key : 0
  useEffect(() => {
    if (tab === 'comms') setSeen(last)
  }, [tab, last])
  if (tab === 'comms') return 0
  const from = Math.max(seen, initSeq)
  let n = 0
  for (let k = chat.length - 1; k >= 0 && chat[k].key > from; k--) if (chat[k].from !== 'operator') n++
  return n
}

// The lower part of the sidebar: the event log and the operator comms, as two tabs.
function LowerTabs({ tab, setTab, onFocusEvent }) {
  const events = useSim((s) => s.events.length)
  const asking = useSim((s) => s.requests.length)
  const unread = useUnread(tab)
  const input = useRef(null)

  // "/" jumps to the message box, like many chat tools.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.target instanceof Element && e.target.closest('input, textarea, select, [contenteditable="true"]')) return
      e.preventDefault()
      setTab('comms')
      requestAnimationFrame(() => input.current?.focus())
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setTab])

  const tabKeys = (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault()
      const next = tab === 'events' ? 'comms' : 'events'
      setTab(next)
      document.getElementById(`tab-${next}`)?.focus()
    }
  }

  return (
    <section className="lower" data-tab={tab} aria-label="Event log and operator comms">
      <div className="tabs" role="tablist" aria-label="Sidebar" onKeyDown={tabKeys}>
        <button type="button" role="tab" id="tab-events" className="tab" aria-selected={tab === 'events'} aria-controls="panel-events" tabIndex={tab === 'events' ? 0 : -1} onClick={() => setTab('events')}>
          Events
          {events > 0 && <span className="tab-n">{events}</span>}
        </button>
        <button
          type="button"
          role="tab"
          id="tab-comms"
          className="tab"
          aria-selected={tab === 'comms'}
          aria-controls="panel-comms"
          tabIndex={tab === 'comms' ? 0 : -1}
          data-ask={(asking > 0 && tab !== 'comms') || undefined}
          onClick={() => setTab('comms')}
          title="Talk to the rovers (/)"
        >
          Comms
          {asking > 0 ? (
            <span className="tab-ask">{asking === 1 ? 'Question' : `${asking} questions`}</span>
          ) : (
            unread > 0 && (
              <span className="tab-badge" aria-label={`${unread} unread`}>
                {unread > 9 ? '9+' : unread}
              </span>
            )
          )}
        </button>
      </div>
      <EventLog onFocusEvent={onFocusEvent} hidden={tab !== 'events'} />
      <CommsPanel hidden={tab !== 'comms'} inputRef={input} />
    </section>
  )
}

export function Sidebar({ follow, selected, onSelect, onFocusEvent }) {
  // With Comms open the fleet stats fold into one row, so the conversation has room.
  const [tab, setTab] = useState('events')
  return (
    <aside className="side" data-tab={tab}>
      <header className="side-head">
        <div className="side-head-top">
          <Wordmark />
          <PlannerPill />
        </div>
        <p className="subtitle">Autonomous super-Earth survey</p>
        <WorldFacts />
      </header>
      <FleetStats />
      <RoverList follow={follow} selected={selected} onSelect={onSelect} />
      <LowerTabs tab={tab} setTab={setTab} onFocusEvent={onFocusEvent} />
    </aside>
  )
}
