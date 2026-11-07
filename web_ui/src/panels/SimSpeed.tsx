import { useState } from 'react'
import { api } from '../lib/api'
import { DEFAULT_OPTIONS } from '../lib/defaults'
import { fmtSpeed } from '../lib/format'
import { isLiveStatus, useStore } from '../lib/store'
import { C } from '../lib/theme'

/** Live simulation-speed switch + the achieved real-time factor (simulated s per wall s). */
export function SimSpeed() {
  const mission = useStore((s) => s.telemetry?.mission)
  const choices = useStore((s) => s.options.sim_speed?.choices) ?? DEFAULT_OPTIONS.sim_speed.choices
  const [pending, setPending] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const live = isLiveStatus(mission?.status)
  const target = pending ?? mission?.sim_speed ?? mission?.config?.sim_speed ?? 1
  const rtf = mission?.rtf ?? null
  // the simulator can't keep up with the target (rendering-bound): say so instead of pretending
  const lagging = rtf != null && rtf < 0.85 * target
  const stalled = rtf === 0

  const choose = async (x: number) => {
    if (pending != null || Math.abs(x - target) < 1e-6) return
    setPending(x)
    setError(null)
    try {
      const r = await api.setSpeed(x)
      if (!r.ok) setError(r.detail ?? 'Rejected by backend')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setPending(null)
    }
  }

  const rtfTitle = stalled
    ? 'Simulator clock has stopped'
    : lagging
      ? `The simulator can't keep up: achieving ${rtf?.toFixed(2)}× of the ${fmtSpeed(target)} target`
      : 'Achieved real-time factor: simulated seconds per real second'

  return (
    <div className="flex items-center gap-2.5 border-l border-rule pl-5">
      <div
        role="radiogroup"
        aria-label="Simulation speed"
        title={error ?? 'Simulation speed'}
        className={`flex rounded-[3px] border bg-bay p-0.5 ${error ? 'border-ha' : 'border-rule'}`}
      >
        {choices.map((x) => {
          const on = Math.abs(x - target) < 1e-6
          return (
            <button
              key={x}
              type="button"
              role="radio"
              aria-checked={on}
              disabled={!live || pending != null}
              onClick={() => choose(x)}
              className={`num rounded-[2px] px-1.5 py-0.5 text-[12.5px] font-[650] transition-colors ${on ? 'bg-bay-2 text-bone shadow-[inset_0_0_0_1px_var(--color-hb)]' : 'text-haze hover:text-bone'} disabled:cursor-default`}
            >
              {fmtSpeed(x)}
            </button>
          )
        })}
      </div>
      <span className="num text-[12.5px] whitespace-nowrap text-haze" title={rtfTitle}>
        RTF{' '}
        <span style={{ color: stalled ? C.ha : lagging ? C.nad : C.bone }}>{rtf == null ? '—' : rtf.toFixed(2)}</span>
      </span>
    </div>
  )
}
