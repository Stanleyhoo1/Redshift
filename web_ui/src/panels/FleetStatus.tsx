import { useRobotIds } from '../lib/hooks'
import { fmt, robotName, robotShort } from '../lib/format'
import { batteryTone, healthTone } from '../lib/sensors'
import { useStore } from '../lib/store'
import { C, robotColor, stateColor, TONE_COLOR } from '../lib/theme'
import { FleetGlyph } from '../screens/prelaunch/ConfigFields'
import { StatePill } from '../ui/Panel'

/** One chip per robot: callsign, state, health and battery. Critical units flash red. Click to select. */
export function FleetStatus() {
  const ids = useRobotIds()
  return (
    <div className="flex min-w-0 items-center gap-1.5" role="list" aria-label="Fleet">
      {ids.map((id) => (
        <FleetChip key={id} id={id} />
      ))}
    </div>
  )
}

function FleetChip({ id }: { id: string }) {
  const r = useStore((s) => s.telemetry?.robots[id])
  const selected = useStore((s) => s.selectedRobot === id)
  if (!r) return null
  const isScout = r.type === 'scout'
  const h = r.health
  const ht = healthTone(h?.pct, h?.critical)
  const crit = ht === 'crit' || r.state === 'AWAIT_RESCUE'
  const name = robotName(id, r.name)
  return (
    <button
      role="listitem"
      onClick={() => useStore.getState().selectRobot(id)}
      aria-pressed={selected}
      title={`${name}${h ? `: health ${fmt(h.pct, 0)} %, battery ${fmt(h.battery_pct, 0)} %` : ''}`}
      className={`flex h-10 min-w-0 items-center gap-2 rounded-[3px] border px-2 text-left transition-colors ${
        crit ? 'animate-crit-flash border-ha bg-ha/20' : selected ? 'border-rule-2 bg-bay-2' : 'border-transparent hover:border-rule'
      }`}
    >
      <FleetGlyph kind={isScout ? 'scout' : 'rover'} on size={13} color={crit ? C.ha : robotColor(id)} />
      <span className="min-w-0">
        <span className="flex items-center gap-1.5">
          <span className="font-cond text-[13.5px] leading-none font-[800] tracking-[0.03em] text-bone">{robotShort(id)}</span>
          <StatePill state={r.state} color={stateColor(r.state)} />
        </span>
        <span className="num mt-[3px] flex items-center gap-2 text-[11px] leading-none text-dim">
          {h ? (
            <>
              <Gauge label="health" pct={h.pct} color={TONE_COLOR[ht]} />
              <Gauge label="battery" pct={h.battery_pct} color={batteryTone(h.battery_pct) === 'ok' ? '#f2dc6b' : TONE_COLOR[batteryTone(h.battery_pct)]} battery />
            </>
          ) : (
            <span>{isScout ? `${(r.altitude_agl ?? 0).toFixed(0)} m up` : `${r.speed.toFixed(2)} m/s`}</span>
          )}
        </span>
      </span>
    </button>
  )
}

/** Tiny icon + percentage: a heart-rate tick for health, a cell for battery. */
function Gauge({ label, pct, color, battery = false }: { label: string; pct: number | null | undefined; color: string; battery?: boolean }) {
  const v = Math.max(0, Math.min(100, Number.isFinite(pct) ? (pct as number) : 0))
  return (
    <span className="flex items-center gap-1" aria-label={`${label} ${v.toFixed(0)} %`}>
      {battery ? (
        <svg width="13" height="8" viewBox="0 0 13 8" aria-hidden>
          <rect x="0.5" y="0.5" width="10.5" height="7" rx="1" fill="none" stroke={C.dim} />
          <rect x="11.5" y="2.5" width="1.5" height="3" fill={C.dim} />
          <rect x="1.8" y="1.8" width={(8 * v) / 100} height="4.4" fill={color} />
        </svg>
      ) : (
        <svg width="11" height="9" viewBox="0 0 11 9" aria-hidden>
          <path d="M0.5 5 H3 L4.3 1.5 L6.4 8 L7.6 5 H10.5" fill="none" stroke={color} strokeWidth="1.3" strokeLinejoin="round" />
        </svg>
      )}
      <span className="font-[650]" style={{ color }}>
        {Number.isFinite(pct) ? `${v.toFixed(0)}%` : '—'}
      </span>
    </span>
  )
}
