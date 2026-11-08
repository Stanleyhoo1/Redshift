import { useEffect, useMemo, useRef, useState } from 'react'
import { appStore, useApp, useSim } from '../state/store.js'
import { postHazard } from '../net/connection.js'
import { fixed, shortName } from '../lib/format.js'
import { HAZARD_TYPES, PARTS, SEVERITIES, partLabel } from '../lib/theme.js'
import { PLACE_TEXT, placement, startPlacing, stopPlacing } from '../lib/placement.js'
import { useHudRect } from '../lib/useHudRect.js'
import { PartIcon } from './HealthBits.jsx'

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
  damage: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 1.6 9.3 5.4l3.6-1.6-1.6 3.6 3.6 1.3-3.6 1.3 1.6 3.6-3.6-1.6L8 15.6l-1.3-3.6-3.6 1.6 1.6-3.6-3.6-1.3 3.6-1.3-1.6-3.6 3.6 1.6z" />
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
  if (type === 'damage') {
    const sev = h?.severity ? `${h.severity[0].toUpperCase()}${h.severity.slice(1)} damage` : 'Damage'
    return `${sev} on ${h?.rover ? name(h.rover) : 'a rover'}${h?.part ? ` (${partLabel(h.part).toLowerCase()})` : ''}. Watch its health and diagnosis${h?.severity === 'disable' || h?.severity === 'wreck' ? ', and the rescue' : ''}.`
  }
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
  const v9 = useSim((s) => s.v9)
  const gen = useSim((s) => s.gen)
  const placing = useApp((a) => a.placing)
  const [target, setTarget] = useState(null) // null = let the sim pick
  const [pending, setPending] = useState(null)
  const [status, setStatus] = useState(null) // { kind: 'ok' | 'error', text, key }
  const [cool, startCool] = useCooldowns()
  const ref = useHudRect('hazards')
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])
  const types = HAZARD_TYPES.filter((h) => (!h.v4 || v4) && (!h.v9 || v9))
  // v9: the Damage popover, open for one rover: { rover }; the part picked (null = any);
  // the sim's refusal, shown inside it.
  const [dmg, setDmg] = useState(null)
  const [dmgPart, setDmgPart] = useState(null)
  const [dmgErr, setDmgErr] = useState(null)
  const openDamage = (rover) => {
    stopPlacing()
    setDmgErr(null)
    setDmg({ rover })
  }
  const closeDamage = () => {
    setDmg(null)
    setDmgErr(null)
  }

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
    setDmg(null)
    setDmgErr(null)
  }, [gen])
  if (dmg && meta.length && !byId[dmg.rover]) setDmg(null)
  // Esc closes the Damage popover before anything else (capture: ahead of the app's keys).
  useEffect(() => {
    if (!dmg) return
    const onKey = (e) => {
      if (e.key !== 'Escape') return
      e.stopImmediatePropagation()
      setDmg(null)
      setDmgErr(null)
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [dmg])
  useEffect(() => {
    if (!v4 || !ready) stopPlacing()
  }, [v4, ready])

  // One request. `where`: { x, y } | { rover } | { auto: true } | null. Resolves { ok, error }.
  // `extra`: more body fields (v9 damage: severity, part).
  const fire = async (type, where = null, extra = null) => {
    if (pending) return { ok: false }
    // v9: picking a rover for Damage opens the severity popover instead of firing.
    if (type === 'damage' && !extra) {
      const rover = where?.rover ?? target
      if (rover) openDamage(rover)
      else if (where?.auto) setStatus({ kind: 'error', text: 'Pick a rover for the damage: click it, or choose it under Target.', key: Date.now() })
      return { ok: !!rover }
    }
    setPending(type)
    try {
      const rover = where?.rover ?? target
      const h = await postHazard(type, rover, Number.isFinite(where?.x) ? [where.x, where.y] : null, extra)
      setStatus({ kind: 'ok', text: describe(h || { rover, ...extra }, type, byId), key: Date.now() })
      startCool(type, type === 'damage' ? 1500 : COOLDOWN_OK)
      stopPlacing()
      if (type === 'damage') closeDamage()
      return { ok: true }
    } catch (err) {
      if (type === 'damage') {
        // Shown in the popover, which stays open so another choice is one click away.
        setDmgErr(err?.message || 'Damage refused')
        return { ok: false, error: err?.message }
      }
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
    if (h.type === 'damage' && dmg) return closeDamage()
    if (h.type === 'damage' && target) return openDamage(target)
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
    else if (id && dmg) openDamage(id)
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
    const on = placing?.type === h.type || (h.type === 'damage' && !!dmg)
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
    <div className="hud hud-hazards" ref={ref} aria-label="Hazards" data-v4={v4 || undefined} data-v9={v9 || undefined}>
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
      {dmg && byId[dmg.rover] && (
        <DamagePopover
          meta={byId[dmg.rover]}
          part={dmgPart}
          setPart={setDmgPart}
          error={dmgErr}
          busy={pending === 'damage'}
          onPick={(severity) => {
            setDmgErr(null)
            fire('damage', { rover: dmg.rover }, { severity, part: dmgPart })
          }}
          onClose={closeDamage}
        />
      )}
    </div>
  )
}

// v9: after picking a rover for Damage: which part (optional) and how bad. One click on a
// severity sends it; the sim's refusal shows here.
function DamagePopover({ meta, part, setPart, error, busy, onPick, onClose }) {
  const health = useSim((s) => s.rovers[meta.id]?.health || null)
  const state = useSim((s) => s.rovers[meta.id]?.state || null)
  return (
    <div className="hz-dmg" role="group" aria-label={`Damage ${meta.name}`} style={{ '--rover': meta.color }}>
      <div className="hz-dmg-head">
        <span className="dot" style={{ background: meta.color }} aria-hidden="true" />
        <b>Damage {meta.name}</b>
        {health && (
          <span className="hz-dmg-hp">
            health {Number.isFinite(health.hp) ? `${health.hp}%` : '--'}
            {state && state !== 'exploring' ? `, ${state}` : ''}
          </span>
        )}
        <button type="button" className="hz-dmg-x" onClick={onClose} title="Close (Esc)" aria-label="Close">
          <svg viewBox="0 0 12 12" aria-hidden="true">
            <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" />
          </svg>
        </button>
      </div>
      <div className="hz-dmg-row">
        <span className="hz-label" id="hz-dmg-part">
          Part
        </span>
        <div className="seg seg-sm hz-dmg-parts" role="group" aria-labelledby="hz-dmg-part">
          <button type="button" aria-pressed={!part} onClick={() => setPart(null)} title="Let the sim pick the part">
            Any
          </button>
          {PARTS.map((p) => {
            const v = health?.parts?.[p.key]
            return (
              <button key={p.key} type="button" aria-pressed={part === p.key} onClick={() => setPart(p.key)} title={`${p.label}${Number.isFinite(v) ? ` (now ${v}%)` : ''}`}>
                <PartIcon part={p.key} className="hz-dmg-pi" />
                {p.short}
              </button>
            )
          })}
        </div>
      </div>
      <div className="hz-dmg-row">
        <span className="hz-label">How bad</span>
        <div className="hz-dmg-sev">
          {SEVERITIES.map((s) => (
            <button key={s.key} type="button" className="hz-sev" data-sev={s.key} disabled={busy} aria-busy={busy || undefined} onClick={() => onPick(s.key)} title={s.hint}>
              {s.label}
            </button>
          ))}
        </div>
      </div>
      {error && (
        <p className="hz-dmg-err" role="alert">
          {error}
        </p>
      )}
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
      {placing.type === 'damage' ? (
        <span>
          Click a rover to damage it, then pick how bad. <kbd>Esc</kbd> to cancel.
        </span>
      ) : placing.mode === 'rover' ? (
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
