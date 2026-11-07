import { memo, useMemo, useState } from 'react'
import { useRobotIds } from '../lib/hooks'
import { fmt, fmtClock, kindLabel, robotName } from '../lib/format'
import { useStore } from '../lib/store'
import { SEVERITY_COLOR, stateColor } from '../lib/theme'
import type { TraceEntry } from '../lib/types'
import { FleetGlyph } from '../screens/prelaunch/ConfigFields'
import { Panel, SensorChips, StatePill } from '../ui/Panel'

export function DecisionTrace() {
  const traces = useStore((s) => s.traces)
  const ids = useRobotIds()
  const [filter, setFilter] = useState<string>('all')
  const shown = useMemo(() => (filter === 'all' ? traces : traces.filter((t) => t.robot === filter)), [traces, filter])
  const counts = useMemo(() => {
    const c = { warn: 0, critical: 0 }
    for (const t of traces) if (t.severity !== 'info') c[t.severity]++
    return c
  }, [traces])

  return (
    <Panel
      title="Decision trace"
      right={
        <span className="num text-[11.5px] text-dim">
          {traces.length} decisions
          {counts.critical > 0 && <span className="ml-2 text-ha">{counts.critical} critical</span>}
        </span>
      }
      className="h-full"
      bodyClass="flex flex-col"
    >
      <div className="flex shrink-0 items-center gap-1 border-b border-rule/70 px-3 py-1.5" role="tablist" aria-label="Filter by robot">
        {['all', ...ids].map((id) => (
          <button
            key={id}
            role="tab"
            aria-selected={filter === id}
            onClick={() => setFilter(id)}
            className={`rounded-[2px] px-2 py-0.5 text-[12px] font-[600] ${filter === id ? 'bg-bay-2 text-bone shadow-[inset_0_0_0_1px_var(--color-rule-2)]' : 'text-dim hover:text-haze'}`}
          >
            {id === 'all' ? 'Whole fleet' : robotName(id)}
          </button>
        ))}
        <span className="ml-auto flex items-center gap-2.5 text-[11px] text-dim">
          <Legend color={SEVERITY_COLOR.info} label="info" />
          <Legend color={SEVERITY_COLOR.warn} label="warn" />
          <Legend color={SEVERITY_COLOR.critical} label="critical" />
        </span>
      </div>
      <ol className="scroll-thin min-h-0 flex-1 overflow-y-auto" aria-live="polite" aria-relevant="additions">
        {shown.length === 0 && (
          <li className="px-4 py-8 text-center text-[13px] text-dim">Decisions appear here the moment an agent senses, reasons and acts.</li>
        )}
        {shown.map((t) => (
          <TraceRow key={`${t.robot}:${t.id}:${t.t}`} t={t} />
        ))}
      </ol>
    </Panel>
  )
}

const Legend = ({ color, label }: { color: string; label: string }) => (
  <span className="flex items-center gap-1">
    <span className="h-2 w-[3px] rounded-[1px]" style={{ backgroundColor: color }} />
    {label}
  </span>
)

const TraceRow = memo(function TraceRow({ t }: { t: TraceEntry }) {
  const sev = SEVERITY_COLOR[t.severity] ?? SEVERITY_COLOR.info
  const conf = Math.max(0, Math.min(1, t.confidence ?? 0))
  return (
    <li
      className="animate-row-in relative border-b border-rule/60 py-2.5 pr-3 pl-4"
      style={{ ['--row-tint' as string]: sev, backgroundColor: t.severity === 'critical' ? `${sev}10` : undefined }}
    >
      <span className="absolute top-0 bottom-0 left-0 w-[3px]" style={{ backgroundColor: sev }} aria-hidden />
      <div className="flex items-center gap-2">
        <time className="num shrink-0 font-mono text-[11px] text-dim [font-stretch:80%]">{fmtClock(t.t)}</time>
        <span className="flex shrink-0 items-center gap-1 text-[12.5px] font-[650] text-bone">
          <FleetGlyph kind={t.robot.startsWith('scout') ? 'scout' : 'rover'} on size={10} />
          {robotName(t.robot)}
        </span>
        <StatePill state={t.state} color={stateColor(t.state)} />
        <span className="font-semi-cond min-w-0 truncate text-[13.5px] font-[750]" style={{ color: t.severity === 'info' ? undefined : sev }}>
          {kindLabel(t.kind)}
        </span>
        <span className="ml-auto flex shrink-0 items-center gap-1.5" title={`confidence ${conf.toFixed(2)}`}>
          <span className="relative hidden h-1.5 w-10 rounded-full bg-rule @min-[430px]:block">
            <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${conf * 100}%`, backgroundColor: sev }} />
          </span>
          <span className="num w-7 text-right text-[11.5px] text-haze">{conf.toFixed(2)}</span>
        </span>
      </div>

      <div className="relative mt-2 grid grid-cols-[52px_minmax(0,1fr)] gap-x-2 gap-y-1.5">
        <span className="absolute top-[9px] bottom-[9px] left-[4px] w-px bg-rule-2" aria-hidden />
        <Step label="Sense" node="dot" color="#a39cc8">
          <span className="mr-1.5 inline-flex translate-y-[-1px] align-middle">
            <SensorChips tags={t.sensors} />
          </span>
          <span className="font-mono text-[11.5px] leading-[1.55] text-bone/90 [font-stretch:82%]">{t.input}</span>
        </Step>
        <Step label="Reason" node="dot" color="#a39cc8">
          <span className="text-[13px] leading-snug text-haze">{t.reasoning}</span>
        </Step>
        <Step label="Act" node="arrow" color={sev}>
          <span className="text-[13px] leading-snug font-[620] text-bone">{t.action}</span>
          {t.latency_ms !== undefined && <span className="num ml-2 text-[11px] whitespace-nowrap text-dim">{fmt(t.latency_ms, 0)} ms</span>}
        </Step>
      </div>
    </li>
  )
})

function Step({ label, node, color, children }: { label: string; node: 'dot' | 'arrow'; color: string; children: React.ReactNode }) {
  return (
    <>
      <span className="relative flex items-start gap-1.5 pt-[1px] text-[11.5px] text-dim">
        {node === 'dot' ? (
          <span className="relative z-10 mt-[4px] h-[9px] w-[9px] shrink-0 rounded-full border bg-bay" style={{ borderColor: color }} />
        ) : (
          <svg width="9" height="11" viewBox="0 0 9 11" className="relative z-10 mt-[3px] shrink-0" aria-hidden>
            <path d="M0 0 L9 5.5 L0 11 Z" fill={color} />
          </svg>
        )}
        {label}
      </span>
      <div className="min-w-0">{children}</div>
    </>
  )
}
