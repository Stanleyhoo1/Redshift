import { useEffect, useRef, useState } from 'react'
import { useSim } from '../state/store.js'

// The local Gemma planner (PROTOCOL.md version 3, `planner`). It advises the rule-based
// rovers in the background; when it is not ready they run on rules alone.

const num = (v) => typeof v === 'number' && Number.isFinite(v)

// "gemma4:e2b" -> "Gemma 4 E2B", "gemma3:4b" -> "Gemma 3 4B", "gemma3n:e2b" -> "Gemma 3n E2B".
export function prettyModel(id) {
  if (!id) return 'Gemma'
  const m = /^gemma[\s-]?(\d+(?:\.\d+)?n?)(?::|[\s-])?(.*)$/i.exec(id)
  if (!m) return id
  const tag = m[2].split(/[-_]/)[0].trim().toUpperCase()
  return `Gemma ${m[1]}${tag ? ` ${tag}` : ''}`
}

// Pill: "0.4 s" (one decimal is plenty at a glance); below 100 ms in ms.
export function fmtLatency(ms) {
  if (!num(ms)) return '--'
  if (ms < 100) return `${Math.round(ms)} ms`
  return ms < 10000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms / 1000)} s`
}

// Details: "410 ms", "3.0 s".
export function fmtMsLong(ms) {
  if (!num(ms)) return '--'
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

const KINDS = {
  ready: { label: 'Ready' },
  warming: { label: 'Warming up' },
  offline: { label: 'Offline' },
  disabled: { label: 'Off' },
}

// What to show for a planner status: kind (colour), short label, pill second line.
export function plannerView(p) {
  let kind = p.state
  if (!KINDS[kind]) kind = /warm|start|load|pull/i.test(kind) ? 'warming' : p.online ? 'ready' : 'offline'
  const hasLatency = num(p.avg_ms) && p.avg_ms > 0 && (p.calls ?? 1) > 0
  let line2
  if (kind === 'ready') line2 = hasLatency ? `Ready · ${fmtLatency(p.avg_ms)}` : 'Ready'
  else if (kind === 'warming') line2 = 'Warming up'
  else line2 = `${KINDS[kind].label} · rules only`
  return { kind, label: KINDS[kind].label, line2, hasLatency }
}

export function plannerLevel(p) {
  const budget = p.budget_ms
  if (!num(budget) || budget <= 0 || !num(p.avg_ms)) return 'none'
  if (num(p.p95_ms) && p.p95_ms > budget) return 'over'
  if ((p.over_budget ?? 0) > 0 || p.avg_ms > budget * 0.5 || (p.p95_ms ?? 0) > budget * 0.8) return 'high'
  return 'ok'
}

export function GemmaStar({ className = 'gm-star' }) {
  return (
    <svg className={className} viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 .8C8.6 4.6 11.4 7.4 15.2 8 11.4 8.6 8.6 11.4 8 15.2 7.4 11.4 4.6 8.6.8 8 4.6 7.4 7.4 4.6 8 .8z" />
    </svg>
  )
}

const SOURCE_TAGS = {
  gemma: { label: 'Gemma', title: 'Interpreted or decided by the local Gemma model' },
  rules: { label: 'Rules', title: 'Decided by the rule-based fallback' },
  operator: { label: 'Operator', title: 'Done because the operator asked' },
}

// Who produced a message or decision: a small tag after the name.
export function SourceTag({ source }) {
  const s = SOURCE_TAGS[source]
  if (!s) return null
  return (
    <span className="src-tag" data-src={source} title={s.title}>
      {source === 'gemma' && <GemmaStar className="src-star" />}
      {s.label}
    </span>
  )
}

// avg filled, p95 and last as ticks, the budget as a white line (same language as the brain bar).
function LatencyBar({ p }) {
  const budget = num(p.budget_ms) && p.budget_ms > 0 ? p.budget_ms : null
  const max = Math.max(budget ? budget * 1.2 : 0, (p.p95_ms || 0) * 1.1, (p.last_ms || 0) * 1.05, 1)
  const pct = (v) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`
  return (
    <span className="sp-brain-bar" aria-hidden="true">
      <i className="sp-brain-avg" style={{ width: pct(p.avg_ms || 0) }} />
      {num(p.p95_ms) && p.p95_ms > 0 && <i className="sp-brain-peak" style={{ left: pct(p.p95_ms) }} />}
      {num(p.last_ms) && p.last_ms > 0 && <i className="sp-brain-last" style={{ left: pct(p.last_ms) }} />}
      {budget && <i className="sp-brain-budget" style={{ left: pct(budget) }} />}
    </span>
  )
}

const NOT_READY = {
  warming: 'Warming up. The rovers run on rules until it is ready.',
  offline: 'Offline. The rovers run on rules alone.',
  disabled: 'Turned off. The rovers run on rules alone.',
}

// Second row of the sensor panel's compute section: Gemma latency against its budget.
export function PlannerRow({ planner: p }) {
  const v = plannerView(p)
  if (v.kind !== 'ready' || !v.hasLatency) {
    return (
      <div className="sp-brain sp-planner" data-level="none" data-state={v.kind}>
        <span className="sp-k">Gemma</span>
        <span className="sp-planner-note">
          <i className="gm-dot" aria-hidden="true" />
          {v.kind === 'ready' ? `Ready, no calls yet. Budget ${fmtMsLong(p.budget_ms)} per call.` : NOT_READY[v.kind]}
        </span>
      </div>
    )
  }
  const level = plannerLevel(p)
  const over = p.over_budget ?? 0
  const budget = num(p.budget_ms) && p.budget_ms > 0 ? p.budget_ms : null
  return (
    <div className="sp-brain sp-planner" data-level={level} data-state={v.kind} title={`${prettyModel(p.model)} on this machine, shared by the fleet`}>
      <span className="sp-k">Gemma</span>
      <span className="sp-brain-nums">
        <b>{fmtMsLong(p.avg_ms)}</b> avg, <b>{fmtMsLong(p.p95_ms)}</b> p95
      </span>
      <LatencyBar p={p} />
      <span className="sp-brain-budget-text">{budget ? `budget ${fmtMsLong(budget)} per call, fleet-wide` : 'no stated budget'}</span>
      {budget && (
        <span className="sp-brain-verdict" data-over={level === 'over' || undefined} data-some={(over > 0 && level !== 'over') || undefined} title={over > 0 ? `${over} calls took longer than the ${fmtMsLong(budget)} budget (the rules decided instead)` : undefined}>
          {over > 0 ? `${over} over` : 'Within budget'}
        </span>
      )}
    </div>
  )
}

// Compact status in the sidebar header: model, state, average latency. Hover or click
// for the numbers behind it.
export function PlannerPill() {
  const p = useSim((s) => s.planner)
  const [open, setOpen] = useState(false) // false | 'hover' | 'pinned'
  const root = useRef(null)
  const timer = useRef(0)

  useEffect(() => () => clearTimeout(timer.current), [])
  useEffect(() => {
    if (!open) return
    const onDown = (e) => {
      if (!root.current?.contains(e.target)) setOpen(false)
    }
    // Capture phase, so Esc closes this first instead of the sensor panel behind it.
    const onKey = (e) => {
      if (e.key !== 'Escape') return
      e.stopImmediatePropagation()
      setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      window.removeEventListener('keydown', onKey, true)
    }
  }, [open])

  if (!p) return null
  const v = plannerView(p)
  const name = prettyModel(p.model)
  const hover = (on) => {
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setOpen((o) => (on ? o || 'hover' : o === 'hover' ? false : o)), on ? 120 : 200)
  }
  const level = plannerLevel(p)
  const over = p.over_budget ?? 0
  return (
    <div className="gm" ref={root} onMouseEnter={() => hover(true)} onMouseLeave={() => hover(false)}>
      <button
        type="button"
        className="gm-pill"
        data-state={v.kind}
        aria-expanded={!!open}
        aria-label={`${name}: ${v.line2}. Show planner details`}
        onClick={() => setOpen((o) => (o === 'pinned' ? false : 'pinned'))}
      >
        <span className="gm-l1">
          <i className="gm-dot" aria-hidden="true" />
          {name}
        </span>
        <span className="gm-l2">{v.line2}</span>
      </button>
      {open && (
        <div className="gm-pop" role="dialog" aria-label="Gemma planner details" data-state={v.kind}>
          <div className="gm-pop-head">
            <GemmaStar />
            <b>{name}</b>
            <span className="gm-pop-state">{v.label}</span>
          </div>
          <p className="gm-pop-sub">
            {p.model ? <code>{p.model}</code> : 'Local model'}, running on this machine. It advises; the rules drive.
          </p>
          <dl className="gm-dl">
            <div>
              <dt>Calls</dt>
              <dd>
                {p.calls ?? '--'}
                {num(p.calls) && (
                  <span className="gm-dim">
                    {' '}
                    ({p.ok ?? 0} answered, {p.fallbacks ?? 0} fell back to rules)
                  </span>
                )}
              </dd>
            </div>
            <div>
              <dt>Latency</dt>
              <dd>
                {v.hasLatency ? (
                  <>
                    {fmtMsLong(p.avg_ms)} avg <span className="gm-dim">·</span> {fmtMsLong(p.p95_ms)} p95
                  </>
                ) : (
                  <span className="gm-dim">No answers yet</span>
                )}
              </dd>
            </div>
            {v.hasLatency && num(p.last_ms) && (
              <div>
                <dt>Last call</dt>
                <dd>{fmtMsLong(p.last_ms)}</dd>
              </div>
            )}
            <div>
              <dt>Budget</dt>
              <dd>
                {num(p.budget_ms) ? `${fmtMsLong(p.budget_ms)} per call` : '--'}
                {num(p.budget_ms) && (
                  <span className="gm-dim" data-over={over > 0 || undefined}>
                    {' '}
                    · {over > 0 ? `${over} over` : 'none over'}
                  </span>
                )}
              </dd>
            </div>
            {num(p.asked_operator) && (
              <div>
                <dt>Asked you</dt>
                <dd>
                  {p.asked_operator} {p.asked_operator === 1 ? 'time' : 'times'}
                </dd>
              </div>
            )}
          </dl>
          {v.hasLatency && (
            <div className="gm-pop-bar sp-brain" data-level={level}>
              <LatencyBar p={p} />
            </div>
          )}
          <p className="gm-pop-note">
            {v.kind === 'ready'
              ? 'The rules act 5 times a second and never wait for Gemma. A late or invalid answer falls back to them.'
              : NOT_READY[v.kind]}
            {p.detail && v.kind !== 'ready' && <span className="gm-dim"> ({p.detail})</span>}
          </p>
        </div>
      )}
    </div>
  )
}
