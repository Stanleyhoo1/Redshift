import { useEffect, useRef, useState } from 'react'
import { appStore, useSim } from '../state/store.js'
import { fixed, mmss } from '../lib/format.js'
import { FleetStats } from './FleetStats.jsx'
import { RoverList } from './RoverList.jsx'
import { EventLog } from './EventLog.jsx'
import { CommsPanel } from './Comms.jsx'
import { PlannerPill } from './Planner.jsx'
import { SamplesPanel } from './Samples.jsx'
import { ModeSwitch, PriorityChip } from './Autonomy.jsx'

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

// v8: with autonomy modes the subtitle's place holds the Autonomous / Supervised switch
// (in supervised mode "autonomous survey" would not be true).
function Subtitle() {
  const v8 = useSim((s) => typeof s.fleet.mode === 'string')
  return v8 ? <ModeSwitch /> : <p className="subtitle">Autonomous super-Earth survey</p>
}

// With sample science (v4b) the stats grid shows the score, so sim time moves up here.
function SimClock() {
  const t = useSim((s) => s.t)
  const speed = useSim((s) => s.fleet.sim_speed)
  return (
    <div className="facts-time" title="Sim time and speed">
      <dt>Sim time</dt>
      <dd>
        {mmss(t)}
        {Number.isFinite(speed) && <span className="facts-dim"> {fixed(speed, 0)}x</span>}
      </dd>
    </div>
  )
}

function WorldFacts() {
  const world = useSim((s) => s.world)
  const science = useSim((s) => Number.isFinite(s.fleet.science_score))
  const g = Number.isFinite(world?.gravity_g) ? world.gravity_g : Number.isFinite(world?.gravity) ? world.gravity / 9.81 : null
  return (
    <dl className="facts">
      <div>
        <dt>World</dt>
        <dd>{world?.name || '--'}</dd>
      </div>
      <div className="facts-g">
        <dt>Gravity</dt>
        <dd>{g === null ? '--' : `${fixed(g, 1)} g`}</dd>
      </div>
      <div>
        <dt>Seed</dt>
        <dd>{world?.seed ?? '--'}</dd>
      </div>
      {science && <SimClock />}
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
function LowerTabs({ tab, setTab, onFocusEvent, selected }) {
  const events = useSim((s) => s.events.length)
  const asking = useSim((s) => s.requests.length)
  // v4b: the Samples tab, when the sim collects samples.
  const hasSamples = useSim((s) => Number.isFinite(s.fleet.science_score) || Object.values(s.rovers).some((r) => r.capacity))
  const nSamples = useSim((s) => s.targets.filter((t) => t.status === 'sampled' || t.cached || t.carried_by).length)
  const tabs = hasSamples ? ['events', 'comms', 'samples'] : ['events', 'comms']
  useEffect(() => {
    if (tab === 'samples' && !hasSamples) setTab('events')
  }, [tab, hasSamples, setTab])
  const unread = useUnread(tab)
  const input = useRef(null)

  // "/" jumps to the message box, like many chat tools.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey || appStore.get().modal) return
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
      const i = tabs.indexOf(tab)
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length]
      setTab(next)
      document.getElementById(`tab-${next}`)?.focus()
    }
  }

  return (
    <section className="lower" data-tab={tab} aria-label="Event log, operator comms and samples">
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
        {hasSamples && (
          <button type="button" role="tab" id="tab-samples" className="tab" aria-selected={tab === 'samples'} aria-controls="panel-samples" tabIndex={tab === 'samples' ? 0 : -1} onClick={() => setTab('samples')} title="Samples taken, best first">
            Samples
            {nSamples > 0 && <span className="tab-n">{nSamples}</span>}
          </button>
        )}
      </div>
      <EventLog onFocusEvent={onFocusEvent} hidden={tab !== 'events'} />
      <CommsPanel hidden={tab !== 'comms'} inputRef={input} selected={selected} />
      {hasSamples && <SamplesPanel hidden={tab !== 'samples'} onFocusEvent={onFocusEvent} />}
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
        <Subtitle />
        <PriorityChip />
        <WorldFacts />
      </header>
      <FleetStats />
      <RoverList follow={follow} selected={selected} onSelect={onSelect} />
      <LowerTabs tab={tab} setTab={setTab} onFocusEvent={onFocusEvent} selected={selected} />
    </aside>
  )
}
