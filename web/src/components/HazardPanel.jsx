import { useEffect, useMemo, useRef, useState } from 'react'
import { useSim } from '../state/store.js'
import { postHazard } from '../net/connection.js'
import { fixed, shortName } from '../lib/format.js'
import { HAZARD_TYPES } from '../lib/theme.js'
import { useHudRect } from '../lib/useHudRect.js'

const COOLDOWN_OK = 4000
const COOLDOWN_ERR = 1200
const HINT = 'Rovers are never told. They only find out through their own sensors.'

const ICONS = {
  rockfall: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M2.5 13.5 4.6 9l3.1-1 2 2.2-.9 3.3z" />
      <path d="M9.6 13.5 11 10.6l2.4.4.9 2.5z" />
      <path d="M8.6 2.2v3M6.4 3.2l1 1.6M10.8 3.2l-1 1.6" />
    </svg>
  ),
  sand: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M1.5 6.5c1.6-1.4 3.2-1.4 4.8 0s3.2 1.4 4.8 0 2.4-1 3.4-.6" />
      <path d="M1.5 10.5c1.6-1.4 3.2-1.4 4.8 0s3.2 1.4 4.8 0 2.4-1 3.4-.6" />
    </svg>
  ),
  dust_storm: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M1.5 5.5h8.2a2 2 0 1 0-2-2" />
      <path d="M1.5 9h11.3a2 2 0 1 1-2 2" />
      <path d="M1.5 12.5h4.5" />
    </svg>
  ),
  rover_fault: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M9.2 1.5 4 9h4l-1.2 5.5L12 7H8z" />
    </svg>
  ),
}

function describe(h, type, byId) {
  const name = (id) => byId[id]?.name || id
  if (type === 'rockfall') return h?.rover ? `Rockfall on ${name(h.rover)}'s route. Watch its LiDAR pick up the boulders.` : 'Rockfall triggered.'
  if (type === 'sand') return h?.rover ? `Soft sand hidden on ${name(h.rover)}'s route. Watch its wheel slip.` : 'Soft sand placed.'
  if (type === 'dust_storm') return `Dust storm started${Number.isFinite(h?.remaining) ? `, ${Math.round(h.remaining)} s` : ''}. Watch the LiDAR range shrink.`
  if (type === 'rover_fault') return h?.rover ? `Motor fault on ${name(h.rover)}. Watch the fleet route around it.` : 'Motor fault triggered.'
  return 'Hazard triggered.'
}

function useCooldowns() {
  const [cool, setCool] = useState({}) // type -> { ms, key }
  const timers = useRef({})
  useEffect(() => () => Object.values(timers.current).forEach(clearTimeout), [])
  const start = (type, ms) => {
    clearTimeout(timers.current[type])
    setCool((c) => ({ ...c, [type]: { ms, key: Date.now() } }))
    timers.current[type] = setTimeout(() => setCool((c) => ({ ...c, [type]: null })), ms)
  }
  return [cool, start]
}

// Inject environment changes into the running sim (POST /api/hazard). The rovers are
// not told: the point is to watch them notice through their sensors and re-plan.
export function HazardPanel() {
  const meta = useSim((s) => s.roverMeta)
  const hazards = useSim((s) => s.hazards)
  const ready = useSim((s) => s.ready)
  const [target, setTarget] = useState(null) // null = let the sim pick
  const [pending, setPending] = useState(null)
  const [status, setStatus] = useState(null) // { kind: 'ok' | 'error', text, key }
  const [cool, startCool] = useCooldowns()
  const ref = useHudRect('hazards')
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])

  useEffect(() => {
    if (target && !byId[target]) setTarget(null)
  }, [byId, target])
  useEffect(() => {
    if (!status) return
    const id = setTimeout(() => setStatus(null), status.kind === 'error' ? 7000 : 5000)
    return () => clearTimeout(id)
  }, [status])

  const fire = async (type) => {
    if (pending || cool[type]) return
    setPending(type)
    try {
      const h = await postHazard(type, target)
      setStatus({ kind: 'ok', text: describe(h, type, byId), key: Date.now() })
      startCool(type, COOLDOWN_OK)
    } catch (err) {
      setStatus({ kind: 'error', text: err?.message || 'Hazard refused', key: Date.now() })
      startCool(type, COOLDOWN_ERR)
    } finally {
      setPending(null)
    }
  }

  // Live badges: storm time left, rovers with a fault.
  const storm = hazards.find((h) => h.type === 'dust_storm' && h.active)
  const faults = hazards.filter((h) => h.type === 'rover_fault' && h.active)
  const badge = {
    dust_storm: storm && Number.isFinite(storm.remaining) ? `${Math.ceil(storm.remaining)} s` : storm ? 'On' : null,
    rover_fault: faults.length > 1 ? `${faults.length} on` : faults.length ? (Number.isFinite(faults[0].remaining) ? `${Math.ceil(faults[0].remaining)} s` : 'On') : null,
    rockfall: null,
    sand: null,
  }

  return (
    <div className="hud hud-hazards" ref={ref} aria-label="Hazards">
      <div className="hz-row">
        <div className="hz-title">
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 1.8 15 14H1z" />
            <path d="M8 6.2v3.6M8 11.6v.2" />
          </svg>
          Hazards
        </div>
        <div className="hz-buttons">
          {HAZARD_TYPES.map(({ type, label, hint }) => {
            const c = cool[type]
            return (
              <button
                key={type}
                type="button"
                className="hz-btn"
                data-type={type}
                data-cooling={c ? true : undefined}
                data-active={badge[type] ? true : undefined}
                disabled={!ready || pending === type || !!c}
                aria-busy={pending === type || undefined}
                onClick={() => fire(type)}
                title={hint}
              >
                {ICONS[type]}
                <span>{label}</span>
                {badge[type] && <span className="hz-badge">{badge[type]}</span>}
                {c && <i key={c.key} className="hz-cd" style={{ '--cd': `${c.ms}ms` }} aria-hidden="true" />}
              </button>
            )
          })}
        </div>
      </div>
      <div className="hz-row hz-row-2">
        <div className="hz-target">
          <span className="hz-label" id="hz-target-label">
            Target
          </span>
          <div className="seg seg-sm" role="group" aria-labelledby="hz-target-label">
            <button type="button" aria-pressed={!target} onClick={() => setTarget(null)} title="Let the sim pick a rover">
              Auto
            </button>
            {meta.map((m) => (
              <button key={m.id} type="button" aria-pressed={target === m.id} onClick={() => setTarget(m.id)} title={`Aim at ${m.name}`}>
                <span className="dot" style={{ background: m.color }} />
                {shortName(m.name, m.id)}
              </button>
            ))}
          </div>
        </div>
        <p className="hz-status" data-kind={status?.kind} key={status?.key ?? 'hint'} role={status?.kind === 'error' ? 'alert' : 'status'}>
          {status ? status.text : HINT}
        </p>
      </div>
    </div>
  )
}

// "Dust storm: LiDAR range 3 m, 45 s left", under the hazard bar while a storm blows.
export function DustBanner() {
  const storm = useSim((s) => s.hazards.find((h) => h.type === 'dust_storm' && h.active) || null)
  const range = useSim((s) => {
    let min = Infinity
    for (const r of Object.values(s.rovers)) if (Number.isFinite(r.sensors?.lidar_range)) min = Math.min(min, r.sensors.lidar_range)
    return Number.isFinite(min) ? min : null
  })
  if (!storm) return null
  const left = Number.isFinite(storm.remaining) ? `${Math.ceil(storm.remaining)} s left` : 'blowing'
  const rangeText = range === null ? 'LiDAR range reduced' : `LiDAR range ${range % 1 ? fixed(range, 1) : range} m`
  return (
    <div className="banner banner-dust" role="status">
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path d="M1.5 5.5h8.2a2 2 0 1 0-2-2" />
        <path d="M1.5 9h11.3a2 2 0 1 1-2 2" />
        <path d="M1.5 12.5h4.5" />
      </svg>
      <span>
        <b>Dust storm:</b> {rangeText}, {left}
      </span>
    </div>
  )
}

// A dusty veil over the 3D view while a storm blows (on top of the scene fog).
export function DustHaze() {
  const on = useSim((s) => s.hazards.some((h) => h.type === 'dust_storm' && h.active))
  return <div className="dust-haze" data-on={on || undefined} aria-hidden="true" />
}
