import { stateStyle } from '../lib/theme.js'

export function Battery({ value, charging }) {
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

// Seconds left on the rover's motor fault (an active "rover_fault" hazard), or null.
export function faultRemaining(hazards, roverId) {
  for (const h of hazards) {
    if (h.type === 'rover_fault' && h.active && h.rover === roverId && Number.isFinite(h.remaining)) return Math.max(0, h.remaining)
  }
  return null
}

export function StatePill({ rover, hazards }) {
  const st = stateStyle(rover?.state)
  if (rover?.state === 'fault') {
    const left = faultRemaining(hazards, rover.id)
    return (
      <span className="pill pill-fault" role="status">
        <i className="pill-blink" aria-hidden="true" />
        {st.label}
        {left !== null && <span className="pill-count">{Math.ceil(left)} s</span>}
      </span>
    )
  }
  // v9: a rover that cannot drive stands out (solid, blinking); a wreck is greyed out.
  if (rover && st.alert) {
    return (
      <span className="pill pill-alert" role="status" style={{ '--pill': st.color }} title="Immobile: it cannot drive and has asked for help">
        <i className="pill-blink" aria-hidden="true" />
        {st.label}
      </span>
    )
  }
  return (
    <span className="pill" data-wreck={(rover && st.wreck) || undefined} style={{ '--pill': st.color }}>
      {rover ? st.label : 'No data'}
    </span>
  )
}

// "5.1" below 10 ms, "18" above: enough precision to compare with the budget.
export function fmtMs(v) {
  if (!Number.isFinite(v)) return '--'
  return v < 10 ? v.toFixed(1) : String(Math.round(v))
}

// How heavy the rover's decision making runs against the stated budget.
export function brainLevel(brain, budget) {
  if (!brain || !Number.isFinite(budget) || budget <= 0) return 'none'
  if ((brain.over ?? 0) > 0 || brain.peak_ms > budget) return 'over'
  if (brain.avg_ms > budget * 0.5 || brain.peak_ms > budget * 0.8) return 'high'
  return 'ok'
}

// Tiny brain-time readout for the rover cards: average ms with a micro bar of the budget.
export function BrainChip({ brain, budget }) {
  if (!brain || !Number.isFinite(brain.avg_ms)) return null
  const level = brainLevel(brain, budget)
  const frac = Number.isFinite(budget) && budget > 0 ? Math.min(1, brain.avg_ms / budget) : 0
  const title = `Brain compute: ${fmtMs(brain.avg_ms)} ms average, ${fmtMs(brain.peak_ms)} ms peak${Number.isFinite(budget) ? `, budget ${fmtMs(budget)} ms per cycle` : ''}`
  return (
    <span className="brain-chip" data-level={level} title={title} aria-label={title}>
      <i className="brain-chip-bar" aria-hidden="true">
        <b style={{ width: `${Math.max(6, frac * 100)}%` }} />
      </i>
      {fmtMs(brain.avg_ms)} ms
    </span>
  )
}
