import { useEffect, useMemo, useRef, useState } from 'react'
import { appStore, useApp, useSim } from '../state/store.js'
import { postHazard } from '../net/connection.js'
import { fixed, shortName } from '../lib/format.js'
import { HAZARD_TYPES } from '../lib/theme.js'
import { PLACE_TEXT, placement, startPlacing, stopPlacing } from '../lib/placement.js'
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
  spire_fall: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3.2 13.6 5 7.6l1.6-5 1.4 5.2-1 5.8z" />
      <path d="M9.5 13.6h5" />
      <path d="M10.2 5.2c1.6.5 2.8 1.7 3.3 3.3M12.4 8.6l1.1-.1.2-1.1" />
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
  lifeform: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <ellipse cx="6.6" cy="8.4" rx="3.6" ry="2.2" />
      <circle cx="11.6" cy="7.4" r="1.6" />
      <path d="M4.4 10.4 3.4 13M6.6 10.6v2.6M8.8 10.4l1 2.6M12.6 6l1-2M11.2 5.9l-.3-2.1" />
    </svg>
  ),
}

const at2 = (h) => (Number.isFinite(h?.x) && Number.isFinite(h?.y) ? ` at (${fixed(h.x, 0)}, ${fixed(h.y, 0)})` : '')

function describe(h, type, byId) {
  const name = (id) => byId[id]?.name || id
  if (type === 'rockfall') return h?.rover ? `Rockfall on ${name(h.rover)}'s route${at2(h)}. Watch its LiDAR pick up the boulders.` : `Rockfall${at2(h)}. Watch the rovers' LiDAR pick up the boulders.`
  if (type === 'sand') return h?.rover ? `Soft sand hidden on ${name(h.rover)}'s route${at2(h)}. Watch its wheel slip.` : `Soft sand hidden${at2(h)}. Only wheel slip will find it.`
  if (type === 'dust_storm') return `Dust storm started${Number.isFinite(h?.remaining) ? `, ${Math.round(h.remaining)} s` : ''}. Watch the LiDAR range shrink.`
  if (type === 'rover_fault') return h?.rover ? `Motor fault on ${name(h.rover)}. Watch the fleet route around it.` : 'Motor fault triggered.'
  if (type === 'spire_fall') return `Spire ${h?.spire ?? ''} topples${h?.rover ? ` across ${name(h.rover)}'s route` : ''}. Watch the LiDAR pick it up.`.replace('  ', ' ')
  if (type === 'lifeform') return `A small herd wanders in${h?.rover ? ` near ${name(h.rover)}` : ''}. The rovers still have to notice it.`
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
// With a version 4 sim, Rockfall, Soft sand, Falling spire and Lifeform are placed by
// clicking the map, and Motor fault by clicking a rover (or aimed with Target).
export function HazardPanel() {
  const meta = useSim((s) => s.roverMeta)
  const hazards = useSim((s) => s.hazards)
  const ready = useSim((s) => s.ready)
  const v4 = useSim((s) => s.v4)
  const gen = useSim((s) => s.gen)
  const placing = useApp((a) => a.placing)
  const [target, setTarget] = useState(null) // null = let the sim pick
  const [pending, setPending] = useState(null)
  const [status, setStatus] = useState(null) // { kind: 'ok' | 'error', text, key }
  const [cool, startCool] = useCooldowns()
  const ref = useHudRect('hazards')
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])
  const types = HAZARD_TYPES.filter((h) => !h.v4 || v4)

  useEffect(() => {
    if (target && !byId[target]) setTarget(null)
  }, [byId, target])
  useEffect(() => {
    if (!status) return
    const id = setTimeout(() => setStatus(null), status.kind === 'error' ? 7000 : 5000)
    return () => clearTimeout(id)
  }, [status])
  // A reset (or an older sim) ends placement.
  useEffect(() => stopPlacing, [gen])
  useEffect(() => {
    if (!v4 || !ready) stopPlacing()
  }, [v4, ready])

  // One request. `where`: { x, y } | { rover } | { auto: true } | null. Resolves { ok, error }.
  const fire = async (type, where = null) => {
    if (pending) return { ok: false }
    setPending(type)
    try {
      const rover = where?.rover ?? target
      const h = await postHazard(type, rover, Number.isFinite(where?.x) ? [where.x, where.y] : null)
      setStatus({ kind: 'ok', text: describe(h, type, byId), key: Date.now() })
      startCool(type, COOLDOWN_OK)
      stopPlacing()
      return { ok: true }
    } catch (err) {
      setStatus({ kind: 'error', text: err?.message || 'Hazard refused', key: Date.now() })
      // In placement mode the user can simply click somewhere else.
      if (!appStore.get().placing) startCool(type, COOLDOWN_ERR)
      return { ok: false, error: err?.message }
    } finally {
      setPending(null)
    }
  }

  // The 3D view calls this on a click while placing.
  const fireRef = useRef(fire)
  fireRef.current = fire
  useEffect(() => {
    placement.commit = (where) => {
      const p = appStore.get().placing
      return p ? fireRef.current(p.type, where) : Promise.resolve({ ok: false })
    }
    return () => {
      placement.commit = null
    }
  }, [])

  const press = (h) => {
    if (placing?.type === h.type) return stopPlacing() // second click cancels
    if (!v4 || !h.place) {
      stopPlacing()
      return fire(h.type)
    }
    if (h.place === 'rover' && target) {
      stopPlacing()
      return fire(h.type, { rover: target })
    }
    setStatus(null)
    startPlacing(h.type, h.place)
  }

  // Picking a rover with Target while choosing one for a motor fault fires it.
  const pickTarget = (id) => {
    setTarget(id)
    if (id && placing?.mode === 'rover') fire(placing.type, { rover: id })
  }

  // Live badges: storm time left, rovers with a fault.
  const storm = hazards.find((h) => h.type === 'dust_storm' && h.active)
  const faults = hazards.filter((h) => h.type === 'rover_fault' && h.active)
  const badge = {
    dust_storm: storm && Number.isFinite(storm.remaining) ? `${Math.ceil(storm.remaining)} s` : storm ? 'On' : null,
    rover_fault: faults.length > 1 ? `${faults.length} on` : faults.length ? (Number.isFinite(faults[0].remaining) ? `${Math.ceil(faults[0].remaining)} s` : 'On') : null,
  }

  const button = (h) => {
    const c = cool[h.type]
    const on = placing?.type === h.type
    return (
      <button
        key={h.type}
        type="button"
        className="hz-btn"
        data-type={h.type}
        data-discovery={h.discovery || undefined}
        data-cooling={c ? true : undefined}
        data-active={badge[h.type] ? true : undefined}
        data-placing={on || undefined}
        aria-pressed={v4 && h.place ? on : undefined}
        disabled={!ready || pending === h.type || !!c}
        aria-busy={pending === h.type || undefined}
        onClick={() => press(h)}
        title={on ? 'Cancel (Esc)' : h.hint}
      >
        {ICONS[h.type]}
        <span>{h.label}</span>
        {badge[h.type] && <span className="hz-badge">{badge[h.type]}</span>}
        {c && <i key={c.key} className="hz-cd" style={{ '--cd': `${c.ms}ms` }} aria-hidden="true" />}
      </button>
    )
  }

  const hazardTypes = types.filter((h) => !h.discovery)
  const finds = types.filter((h) => h.discovery)
  return (
    <div className="hud hud-hazards" ref={ref} aria-label="Hazards" data-v4={v4 || undefined}>
      <div className="hz-row">
        <div className="hz-title">
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 1.8 15 14H1z" />
            <path d="M8 6.2v3.6M8 11.6v.2" />
          </svg>
          <span>Hazards</span>
        </div>
        <div className="hz-buttons">
          {hazardTypes.map(button)}
          {finds.length > 0 && <i className="hz-sep" aria-hidden="true" />}
          {finds.map(button)}
        </div>
      </div>
      <div className="hz-row hz-row-2">
        <div className="hz-target">
          <span className="hz-label" id="hz-target-label">
            Target
          </span>
          <div className="seg seg-sm" role="group" aria-labelledby="hz-target-label">
            <button type="button" aria-pressed={!target} onClick={() => pickTarget(null)} title="Let the sim pick a rover">
              Auto
            </button>
            {meta.map((m) => (
              <button key={m.id} type="button" aria-pressed={target === m.id} onClick={() => pickTarget(m.id)} title={`Aim at ${m.name}`}>
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

// While placing: what to do, over the map under the hazard bar.
export function PlacementHint() {
  const placing = useApp((a) => a.placing)
  if (!placing) return null
  const t = PLACE_TEXT[placing.type] || { what: 'it' }
  const discovery = placing.type === 'lifeform'
  return (
    <div className="banner banner-place" role="status" data-discovery={discovery || undefined}>
      <span className="banner-place-icon">{ICONS[placing.type]}</span>
      {placing.mode === 'rover' ? (
        <span>
          Click a rover to give it {t.what}. <kbd>Esc</kbd> to cancel. <kbd>Right-click</kbd> to let the sim pick.
        </span>
      ) : placing.type === 'spire_fall' ? (
        <span>
          Click near a spire to topple it that way, or anywhere to drop a new one. <kbd>Esc</kbd> to cancel. <kbd>Right-click</kbd> for automatic placement.
        </span>
      ) : (
        <span>
          Click the map to place {t.what}. <kbd>Esc</kbd> to cancel. <kbd>Right-click</kbd> for automatic placement.
        </span>
      )}
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
