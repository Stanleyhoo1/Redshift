import type { ReactNode } from 'react'
import { useSelected } from '../lib/hooks'
import { fmt, robotShort } from '../lib/format'
import { useStore } from '../lib/store'
import { C } from '../lib/theme'
import { Panel } from '../ui/Panel'

const CAUTION_AT = 0.45
const SAFE_HOLD_AT = 0.8

export function EdgeHud() {
  const [sel, robot] = useSelected()
  const profiles = useStore((s) => s.options.edge_profiles)
  const e = robot?.edge
  const prof = profiles.find((p) => p.id === e?.profile)
  const host = e ? e.host_ms.perception + e.host_ms.planner + e.host_ms.mapping : 0
  const unc = robot?.uncertainty.score ?? e?.uncertainty ?? 0
  const throttle = e?.throttle_level ?? 0

  return (
    <Panel
      title={`Edge compute${sel ? `, ${robotShort(sel)}` : ''}`}
      sensors={['EDGE']}
      right={<span className="font-semi-cond truncate text-[12.5px] font-[700] text-bone">{prof?.label ?? e?.profile ?? '—'}</span>}
      bodyClass="flex flex-col gap-2.5 px-3 py-2"
    >
      {/* latency: measured on the host vs scaled to the target device */}
      <div>
        <div className="mb-1 flex items-baseline justify-between gap-2 text-[12px]">
          <span className="text-haze">
            Loop latency<span className="num text-dim">{e ? `, ${fmt(e.loop_hz, 1)} Hz` : ''}</span>
          </span>
          <span className="flex gap-2.5 text-[10.5px] text-dim">
            <Key c="#8ec5ff">perception</Key>
            <Key c="#d6a6ff">planner</Key>
            <Key c="#e3c48c">mapping</Key>
          </span>
        </div>
        <LatencyBars hostMs={host} deviceMs={e?.device_ms ?? 0} parts={e?.host_ms} device={prof?.device ?? 'device'} />
      </div>

      <div className="grid grid-cols-3 gap-2">
        <Metric label="Load" value={`${fmt(e?.load_pct, 0)}%`} tone={(e?.load_pct ?? 0) > 90 ? C.ha : (e?.load_pct ?? 0) > 75 ? C.nad : C.bone}>
          <Meter value={(e?.load_pct ?? 0) / 100} marks={[0.6, 0.9]} color={(e?.load_pct ?? 0) > 90 ? C.ha : C.hb} />
        </Metric>
        <Metric label="Power" value={`${fmt(e?.power_w, 1)} W`}>
          <Meter
            value={prof && e ? (e.power_w - prof.watts_idle) / (prof.watts_max - prof.watts_idle || 1) : 0}
            color={C.o3}
            caption={prof ? `${prof.watts_idle}–${prof.watts_max} W` : undefined}
          />
        </Metric>
        <Metric label="Perception" value={`${fmt(e?.perception_hz, 1)} Hz`}>
          <Meter value={prof && e ? e.perception_hz / prof.perception_hz_max : 0} color={C.sci} caption={prof ? `max ${prof.perception_hz_max} Hz` : undefined} />
        </Metric>
      </div>

      <div className="flex items-center justify-between gap-3 text-[12px]">
        <span className="text-haze">Thermal / load throttle</span>
        <span className="flex items-center gap-1" aria-label={`Throttle level ${throttle}`}>
          {[1, 2, 3].map((k) => (
            <span key={k} className="h-2.5 w-5 rounded-[1px]" style={{ backgroundColor: k <= throttle ? C.nad : C.rule }} />
          ))}
          <span className="num ml-1.5 w-[64px] text-right text-bone">{throttle === 0 ? 'none' : `−${throttle * 25}% rate`}</span>
        </span>
      </div>

      <UncertaintyGauge value={unc} />
    </Panel>
  )
}

function Metric({ label, value, tone = C.bone, children }: { label: string; value: string; tone?: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11.5px] text-haze">{label}</div>
      <div className="font-semi-cond num text-[19px] leading-tight font-[750]" style={{ color: tone }}>
        {value}
      </div>
      {children}
    </div>
  )
}

function Meter({ value, color, marks = [], caption }: { value: number; color: string; marks?: number[]; caption?: string }) {
  const v = Math.max(0, Math.min(1, value || 0))
  return (
    <div className="mt-1">
      <div className="relative h-1.5 rounded-full bg-rule">
        <div className="h-1.5 rounded-full transition-[width] duration-300" style={{ width: `${v * 100}%`, backgroundColor: color }} />
        {marks.map((m) => (
          <span key={m} className="absolute -top-0.5 h-2.5 w-px bg-bone/60" style={{ left: `${m * 100}%` }} />
        ))}
      </div>
      {caption && <div className="num mt-0.5 text-[10.5px] text-dim">{caption}</div>}
    </div>
  )
}

function LatencyBars({ hostMs, deviceMs, parts, device }: { hostMs: number; deviceMs: number; parts?: { perception: number; planner: number; mapping: number }; device: string }) {
  const scale = Math.max(100, Math.ceil(Math.max(hostMs, deviceMs) / 50) * 50)
  const seg = (ms: number) => `${(ms / scale) * 100}%`
  return (
    <div className="grid grid-cols-[64px_minmax(0,1fr)_56px] items-center gap-x-2 gap-y-1.5 text-[12px]" title={`Bars scaled to ${scale} ms`}>
      <span className="text-dim">This host</span>
      <div className="flex h-2.5 overflow-hidden rounded-[1px] bg-rule/60">
        {parts && (
          <>
            <span style={{ width: seg(parts.perception), backgroundColor: '#8ec5ff' }} title="perception" />
            <span style={{ width: seg(parts.planner), backgroundColor: '#d6a6ff' }} title="planner" />
            <span style={{ width: seg(parts.mapping), backgroundColor: '#e3c48c' }} title="mapping" />
          </>
        )}
      </div>
      <span className="num text-right text-bone">{fmt(hostMs, 1)} ms</span>
      <span className="truncate text-dim" title={device}>
        On device
      </span>
      <div className="h-2.5 overflow-hidden rounded-[1px] bg-rule/60">
        <div className="h-2.5 transition-[width] duration-300" style={{ width: seg(deviceMs), backgroundColor: C.nad }} />
      </div>
      <span className="num text-right font-[700] text-nad">{fmt(deviceMs, 1)} ms</span>
    </div>
  )
}

const Key = ({ c, children }: { c: string; children: ReactNode }) => (
  <span className="flex items-center gap-1">
    <span className="h-1.5 w-1.5 rounded-[1px]" style={{ backgroundColor: c }} />
    {children}
  </span>
)

function UncertaintyGauge({ value }: { value: number }) {
  const v = Math.max(0, Math.min(1, value))
  const tone = v >= SAFE_HOLD_AT ? C.ha : v >= CAUTION_AT ? C.nad : C.o3
  const verdict = v >= SAFE_HOLD_AT ? 'Safe-hold threshold' : v >= CAUTION_AT ? 'Caution: slow, LiDAR-only' : 'Sensors trusted'
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-[12px] text-haze">Perception uncertainty</span>
        <span className="flex items-baseline gap-2">
          <span className="text-[11.5px]" style={{ color: tone }}>
            {verdict}
          </span>
          <span className="font-semi-cond num text-[19px] leading-none font-[800]" style={{ color: tone }}>
            {v.toFixed(2)}
          </span>
        </span>
      </div>
      <div className="relative h-3.5">
        <div className="absolute inset-y-0 left-0 rounded-l-[2px] bg-o3/15" style={{ width: `${CAUTION_AT * 100}%` }} />
        <div className="absolute inset-y-0 bg-nad/20" style={{ left: `${CAUTION_AT * 100}%`, width: `${(SAFE_HOLD_AT - CAUTION_AT) * 100}%` }} />
        <div className="absolute inset-y-0 right-0 rounded-r-[2px] bg-ha/25" style={{ left: `${SAFE_HOLD_AT * 100}%` }} />
        <div className="absolute inset-y-[3px] left-0 rounded-[1px] transition-[width] duration-300" style={{ width: `${v * 100}%`, backgroundColor: tone }} />
        {[CAUTION_AT, SAFE_HOLD_AT].map((m) => (
          <span key={m} className="absolute -top-0.5 -bottom-0.5 w-px bg-bone" style={{ left: `${m * 100}%` }} />
        ))}
      </div>
      <div className="num relative mt-0.5 h-3.5 text-[10.5px] text-dim">
        <span className="absolute left-0">0</span>
        <span className="absolute -translate-x-1/2" style={{ left: `${CAUTION_AT * 100}%` }}>
          0.45 caution
        </span>
        <span className="absolute -translate-x-1/2" style={{ left: `${SAFE_HOLD_AT * 100}%` }}>
          0.8 hold
        </span>
        <span className="absolute right-0">1</span>
      </div>
    </div>
  )
}
