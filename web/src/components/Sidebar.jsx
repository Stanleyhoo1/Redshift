import { useEffect, useRef, useState } from 'react'
import { appStore, useSim } from '../state/store.js'
import { fixed, mmss } from '../lib/format.js'
import { FleetStats } from './FleetStats.jsx'
import { RoverList } from './RoverList.jsx'
import { EventLog } from './EventLog.jsx'
import { CommsPanel } from './Comms.jsx'
import { PlannerPill } from './Planner.jsx'
import { SamplesPanel } from './Samples.jsx'
import { BIOMES, biomeOf } from '../lib/biomes.js'
import { FoldButton, FoldSection, setFold, useFold } from './Fold.jsx'
import '../styles/declutter-side.css'

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
  const biome = biomeOf(world)
  return (
    <dl className="facts">
      <div>
        <dt>World</dt>
        <dd>
          {world?.name || '--'}
          {biome && (
            <span className="facts-biome" title={biome.secondary ? `Mostly ${biome.name.toLowerCase()}, with patches of ${BIOMES[biome.secondary].name.toLowerCase()}` : biome.name}>
              <i style={{ background: BIOMES[biome.id].map[1] }} />
              {biome.name}
            </span>
          )}
        </dd>
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
function LowerTabs({ tab, setTab, onFocusEvent, selected, input }) {
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
            <span className="tab-ask" aria-label={`${asking} ${asking === 1 ? 'question' : 'questions'} waiting`} title={`${asking} ${asking === 1 ? 'question' : 'questions'} waiting`}>
              {asking}
            </span>
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

// Short folded summaries, so a folded section still says what is in it.
function useSummaries() {
  const explored = useSim((s) => s.fleet.explored_pct)
  const found = useSim((s) => (Number.isFinite(s.fleet.targets_found) ? s.fleet.targets_found : s.targets.length))
  const total = useSim((s) => s.fleet.targets_total)
  const score = useSim((s) => s.fleet.science_score)
  const nRovers = useSim((s) => s.roverMeta.length)
  const states = useSim((s) => s.roverMeta.map((m) => s.rovers[m.id]?.state || '').join(','))
  const events = useSim((s) => s.events.length)
  const asking = useSim((s) => s.requests.length)
  const mission = [
    Number.isFinite(explored) ? `${fixed(explored, 1)} % explored` : null,
    `${found} of ${Number.isFinite(total) ? total : '--'} targets`,
    Number.isFinite(score) ? `${fixed(score, 0)} pts` : null,
  ]
    .filter(Boolean)
    .join(', ')
  const list = states.split(',')
  const trouble = ['fault', 'recovering', 'charging'].map((k) => [k, list.filter((x) => x === k).length]).filter(([, n]) => n > 0)
  const rovers = [`${nRovers} ${nRovers === 1 ? 'rover' : 'rovers'}`, ...trouble.map(([k, n]) => `${n} ${k}`)].join(', ')
  const log = [`${events} ${events === 1 ? 'event' : 'events'}`, asking ? `${asking} ${asking === 1 ? 'question' : 'questions'} waiting` : null].filter(Boolean).join(', ')
  return { mission, rovers, log, asking }
}

// The whole sidebar folded to a narrow rail: each rover's colour and state, and the questions waiting.
function SideRail({ onOpen, onSelect, selected }) {
  const meta = useSim((s) => s.roverMeta)
  const rovers = useSim((s) => s.rovers)
  const asking = useSim((s) => s.requests.length)
  return (
    <aside className="side side-rail" aria-label="Sidebar, folded">
      <FoldButton open={false} onToggle={onOpen} label="the sidebar" className="rail-open" />
      <ul className="rail-rovers">
        {meta.map((m) => {
          const st = rovers[m.id]?.state || 'unknown'
          return (
            <li key={m.id}>
              <button
                type="button"
                className="rail-rover"
                data-state={st}
                aria-pressed={selected === m.id}
                style={{ '--rover': m.color }}
                title={`${m.name}: ${st}`}
                aria-label={`${m.name}, ${st}`}
                onClick={() => onSelect(m.id)}
              >
                <i aria-hidden="true" />
                <span>{st.slice(0, 4)}</span>
              </button>
            </li>
          )
        })}
      </ul>
      {asking > 0 && (
        <button type="button" className="rail-ask" onClick={onOpen} title={`${asking} ${asking === 1 ? 'question' : 'questions'} waiting`} aria-label={`${asking} ${asking === 1 ? 'question' : 'questions'} waiting, open the sidebar`}>
          {asking}
          <small>ask</small>
        </button>
      )}
    </aside>
  )
}

export function Sidebar({ follow, selected, onSelect, onFocusEvent, inert = false }) {
  // With Comms open the fleet stats fold into one row, so the conversation has room.
  const [tab, setTab] = useState('events')
  const [panelOpen, setPanelOpen] = useFold('side.panel', true)
  const sum = useSummaries()
  const input = useRef(null)

  // The rail is set on the root, so --side-w and the map grid follow it.
  useEffect(() => {
    const root = document.documentElement
    if (panelOpen) delete root.dataset.side
    else root.dataset.side = 'rail'
    window.dispatchEvent(new Event('resize'))
  }, [panelOpen])

  // "/" jumps to the message box, like many chat tools. It opens the sidebar and the log first.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey || appStore.get().modal) return
      if (e.target instanceof Element && e.target.closest('input, textarea, select, [contenteditable="true"]')) return
      e.preventDefault()
      setFold('side.panel', true)
      setFold('side.log', true)
      setTab('comms')
      requestAnimationFrame(() => requestAnimationFrame(() => input.current?.focus()))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  if (!panelOpen) return <SideRail onOpen={() => setPanelOpen(true)} onSelect={onSelect} selected={selected} />

  return (
    <aside className="side" data-tab={tab} data-ask={sum.asking > 0 || undefined} inert={inert}>
      <header className="side-head">
        <div className="side-head-top">
          <Wordmark />
          <PlannerPill />
          <FoldButton open onToggle={() => setPanelOpen(false)} label="the sidebar" className="side-collapse" />
        </div>
        <p className="subtitle">Autonomous super-Earth survey</p>
        <WorldFacts />
      </header>
      <FoldSection id="side.mission" title="Mission" summary={sum.mission} className="side-fold" defaultOpen={typeof window === 'undefined' || window.innerHeight > 800}>
        <FleetStats />
      </FoldSection>
      <FoldSection id="side.rovers" title="Rovers" summary={sum.rovers} className="side-fold">
        <RoverList follow={follow} selected={selected} onSelect={onSelect} />
      </FoldSection>
      <FoldSection id="side.log" title="Log" summary={sum.log} className="side-fold side-fold-log">
        <LowerTabs tab={tab} setTab={setTab} onFocusEvent={onFocusEvent} selected={selected} input={input} />
      </FoldSection>
    </aside>
  )
}
