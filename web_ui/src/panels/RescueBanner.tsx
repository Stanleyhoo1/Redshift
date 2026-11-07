import { useRescue, useRobotIds } from '../lib/hooks'
import { faultLabel, faultSummary, fmt, fmtElapsed, robotName } from '../lib/format'
import { distressElapsed, phaseIndex, RESCUE_PHASES, type RescueView } from '../lib/rescue'
import { healthTone } from '../lib/sensors'
import { useStore } from '../lib/store'
import { C, TONE_COLOR } from '../lib/theme'

/** SOS + autonomous rescue status over the globe, from the distress call to the repaired rover. */
export function RescueBanner() {
  const v = useRescue()
  const ids = useRobotIds()
  if (!v) return null
  const since = distressElapsed(v.id, performance.now())
  const rovers = ids.filter((i) => i.startsWith('rover'))
  const done = v.resolved || v.aborted

  return (
    <section
      key={v.id}
      aria-live="assertive"
      aria-label="Rescue status"
      className={`animate-banner-in pointer-events-auto overflow-hidden rounded-[3px] border bg-void/90 shadow-[0_18px_50px_-20px_rgba(0,0,0,0.85)] backdrop-blur-md ${done ? 'border-rescue/70' : 'border-ha'}`}
    >
      {done ? <Wrapup v={v} /> : <SosLine v={v} since={since} />}
      {!done && <RescueLine v={v} lone={rovers.length < 2} />}
    </section>
  )
}

function Name({ id, className = '' }: { id: string; className?: string }) {
  const name = useStore((s) => s.telemetry?.robots[id]?.name)
  return (
    <button type="button" onClick={() => useStore.getState().selectRobot(id)} title={`Select ${robotName(id, name)}`} className={`font-semi-cond font-[780] hover:underline ${className}`}>
      {robotName(id, name)}
    </button>
  )
}

function SosLine({ v, since }: { v: RescueView; since: number }) {
  const tone = healthTone(v.health_pct, (v.health_pct ?? 100) < 15)
  const summary = faultSummary(v.faults)
  return (
    <div className="flex items-center gap-3 bg-[linear-gradient(90deg,rgba(255,74,92,0.22),rgba(255,74,92,0.05))] px-3 py-2">
      <span className="font-cond animate-pulse-ring flex h-8 shrink-0 items-center gap-1.5 rounded-[2px] bg-ha px-2 text-[19px] leading-none font-[900] tracking-[0.06em] text-void">
        <span className="h-2 w-2 rounded-full bg-void" />
        SOS
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-baseline gap-x-3">
          <Name id={v.target} className="truncate text-[18px] leading-tight text-bone" />
          <span className="num shrink-0 text-[13px] text-haze">
            health{' '}
            <span className="font-cond text-[20px] leading-none font-[850]" style={{ color: TONE_COLOR[tone] }}>
              {v.health_pct != null ? `${v.health_pct.toFixed(0)}%` : '—'}
            </span>
          </span>
          {summary && <span className="shrink-0 text-[13px] font-[650] text-ha">{summary}</span>}
        </div>
        {v.faults.length > 0 && <div className="truncate text-[11.5px] text-haze">{v.faults.map(faultLabel).join(', ')}</div>}
      </div>
      <span className="num shrink-0 text-right text-[11.5px] leading-tight text-dim">
        distress call
        <span className="block text-[15px] font-[650] text-bone">{fmtElapsed(since)}</span>
      </span>
    </div>
  )
}

function RescueLine({ v, lone }: { v: RescueView; lone: boolean }) {
  if (!v.rescuer) {
    return (
      <div className="border-t border-rule px-3 py-2 text-[13px] text-haze">
        {lone ? (
          <>
            <span className="font-[650] text-nad">No recovery rover in this fleet.</span> The damaged rover holds position and keeps calling for rescue.
          </>
        ) : (
          <>Assigning a rescuer…</>
        )}
      </div>
    )
  }
  const cur = phaseIndex(v.phase)
  return (
    <div className="border-t border-rescue/40 px-3 pt-2 pb-2.5">
      <div className="mb-2 flex items-center gap-2.5">
        <Name id={v.rescuer} className="text-[15px] text-bone" />
        <span className="font-cond rounded-[2px] border border-rescue/70 px-1.5 py-[3px] text-[12.5px] leading-none font-[800] tracking-[0.05em] text-rescue">AUTONOMOUS RESCUE MODE</span>
        <LockState v={v} />
      </div>
      <ol className="grid grid-cols-4 gap-1" aria-label="Rescue sequence">
        {RESCUE_PHASES.map((p, k) => {
          const state = k < cur ? 'done' : k === cur ? 'now' : 'next'
          const detail =
            p === 'TRANSIT' && state === 'now' && v.range_m != null
              ? `${fmt(v.range_m, 1)} m to go`
              : p === 'DOCKING' && state === 'now' && v.range_m != null
                ? `${fmt(v.range_m, 2)} m gap`
                : p === 'REPAIRING' && state === 'now'
                  ? 'restoring drivetrain'
                  : state === 'done'
                    ? 'complete'
                    : p === 'TRANSIT'
                      ? 'A* to stand-off'
                      : p === 'DOCKING'
                        ? 'camera-guided'
                        : p === 'REPAIRING'
                          ? 'field repair'
                          : 'both resume'
          return (
            <li
              key={p}
              aria-current={state === 'now' ? 'step' : undefined}
              className={`relative flex min-w-0 flex-col rounded-[2px] px-2 py-1.5 ${state === 'now' ? 'bg-rescue text-void' : state === 'done' ? 'bg-rescue/12 text-rescue' : 'bg-bay text-dim'}`}
            >
              <span className="font-cond flex items-center gap-1.5 text-[15px] leading-none font-[850] tracking-[0.04em]">
                <span className="num text-[11px] font-[700] opacity-70">{k + 1}</span>
                {p}
                {state === 'done' && (
                  <svg width="11" height="9" viewBox="0 0 11 9" aria-hidden className="ml-auto">
                    <path d="M1 4.5 L4 7.5 L10 1" fill="none" stroke="currentColor" strokeWidth="1.8" />
                  </svg>
                )}
              </span>
              <span className={`num mt-1 truncate text-[11.5px] leading-none ${state === 'now' ? 'font-[700]' : ''}`}>{detail}</span>
              {state === 'now' && <span className="absolute inset-x-0 -bottom-[3px] h-[2px] animate-pulse bg-rescue" aria-hidden />}
            </li>
          )
        })}
      </ol>
    </div>
  )
}

/** Perception lock on the damaged teammate (§8.4 `locked`), the gate for the final creep in. */
function LockState({ v }: { v: RescueView }) {
  const docking = v.phase === 'DOCKING'
  const repairing = v.phase === 'REPAIRING'
  const label = repairing ? 'DOCKED' : v.locked ? 'LOCKED ON' : docking ? 'ACQUIRING LOCK' : 'MAP-GUIDED'
  const color = v.locked || repairing ? C.rescue : docking ? C.nad : C.haze
  return (
    <span className="ml-auto flex shrink-0 items-center gap-1.5 text-[12px]" title="Camera, depth and LiDAR lock on the damaged rover">
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden className={docking && !v.locked ? 'animate-spin [animation-duration:3s]' : ''}>
        <circle cx="9" cy="9" r="5.5" fill="none" stroke={color} strokeWidth="1.4" strokeDasharray={v.locked || repairing ? undefined : '3 2.5'} />
        <path d="M9 0.5 V4 M9 14 V17.5 M0.5 9 H4 M14 9 H17.5" stroke={color} strokeWidth="1.4" />
        {(v.locked || repairing) && <circle cx="9" cy="9" r="1.8" fill={color} />}
      </svg>
      <span className="text-dim">perception</span>
      <span className="font-cond text-[14px] leading-none font-[850] tracking-[0.04em]" style={{ color }}>
        {label}
      </span>
    </span>
  )
}

function Wrapup({ v }: { v: RescueView }) {
  const tone = healthTone(v.health_pct)
  return (
    <div className="flex items-center gap-3 bg-[linear-gradient(90deg,rgba(198,255,61,0.16),rgba(198,255,61,0.03))] px-3 py-2.5">
      <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden className="shrink-0">
        <circle cx="13" cy="13" r="12" fill="none" stroke={v.aborted ? C.nad : C.rescue} strokeWidth="1.6" />
        {v.aborted ? <path d="M8 8 L18 18 M18 8 L8 18" stroke={C.nad} strokeWidth="2" /> : <path d="M7 13.5 L11.5 18 L19.5 8.5" fill="none" stroke={C.rescue} strokeWidth="2.2" />}
      </svg>
      <div className="min-w-0 flex-1 text-[14px] leading-snug text-haze">
        {v.aborted ? (
          <>
            <span className="font-[750] text-nad">Rescue aborted.</span> <Name id={v.target} className="text-bone" /> is still waiting for help.
          </>
        ) : (
          <>
            <Name id={v.target} className="text-bone" /> repaired
            {v.rescuer ? (
              <>
                {' '}
                by <Name id={v.rescuer} className="text-bone" />
              </>
            ) : null}
            . Health back to{' '}
            <span className="font-cond num text-[18px] font-[850]" style={{ color: TONE_COLOR[tone] }}>
              {v.health_pct != null ? `${v.health_pct.toFixed(0)}%` : '—'}
            </span>
            , both rovers resuming exploration.
          </>
        )}
      </div>
    </div>
  )
}
