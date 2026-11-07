import type { ReactNode } from 'react'
import type { BackendOption, EdgeProfileOption, RangeOption, SectorOption } from '../../lib/types'

export function Field({ title, hint, children }: { title: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <fieldset className="min-w-0">
      <legend className="mb-2.5 flex w-full items-baseline justify-between gap-4">
        <span className="text-[15px] font-medium text-haze">{title}</span>
        {hint && <span className="text-[12px] text-dim">{hint}</span>}
      </legend>
      {children}
    </fieldset>
  )
}

const choiceBase =
  'group relative flex min-w-0 flex-col items-start gap-1 rounded-[3px] border px-3.5 py-3 text-left transition-colors'
const choiceOn = 'border-hb bg-bay-2 shadow-[inset_0_0_0_1px_var(--color-hb)]'
const choiceOff = 'border-rule bg-bay hover:border-rule-2 hover:bg-bay-2/60'

export function SectorPicker({ sectors, value, onChange }: { sectors: SectorOption[]; value: string; onChange: (id: SectorOption['id']) => void }) {
  return (
    <div className="grid grid-cols-3 gap-2.5" role="radiogroup" aria-label="Sector size">
      {sectors.map((s) => {
        const on = s.id === value
        return (
          <button key={s.id} type="button" role="radio" aria-checked={on} onClick={() => onChange(s.id)} className={`${choiceBase} ${on ? choiceOn : choiceOff}`}>
            <span className="flex w-full items-start justify-between">
              <span className="text-[13px] text-haze">{s.label}</span>
              <GridGlyph cells={s.cells} on={on} />
            </span>
            <span className="font-cond num text-[34px] leading-none font-[800] text-bone">
              {s.size_m}
              <span className="ml-0.5 text-[18px] font-[600] text-haze">m</span>
            </span>
            <span className="num text-[12px] text-dim">
              {s.res_m} m cells, {s.cells}×{s.cells} grid
            </span>
          </button>
        )
      })}
    </div>
  )
}

/** Tiny square whose line density reflects the grid resolution. */
function GridGlyph({ cells, on }: { cells: number; on: boolean }) {
  const n = Math.max(4, Math.round(cells / 20))
  const lines = Array.from({ length: n - 1 }, (_, k) => ((k + 1) * 24) / n)
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden className={on ? 'text-hb' : 'text-rule-2'}>
      <rect x="0.5" y="0.5" width="23" height="23" fill="none" stroke="currentColor" />
      {lines.map((p) => (
        <g key={p} stroke="currentColor" strokeWidth="0.5" opacity="0.7">
          <line x1={p} y1="0" x2={p} y2="24" />
          <line x1="0" y1={p} x2="24" y2={p} />
        </g>
      ))}
    </svg>
  )
}

export function Stepper({
  label,
  range,
  value,
  onChange,
  glyph,
  unit,
}: {
  label: string
  range: RangeOption
  value: number
  onChange: (v: number) => void
  glyph: 'rover' | 'scout'
  unit: [string, string]
}) {
  const btn =
    'grid h-9 w-9 place-items-center rounded-[3px] border border-rule bg-bay text-[18px] text-bone hover:border-rule-2 disabled:opacity-30'
  return (
    <div className="flex items-center justify-between gap-3 rounded-[3px] border border-rule bg-bay px-3.5 py-2.5">
      <div className="min-w-0">
        <div className="text-[13px] text-haze">{label}</div>
        <div className="mt-1 flex h-4 items-center gap-1.5">
          {Array.from({ length: range.max }, (_, k) => (
            <FleetGlyph key={k} kind={glyph} on={k < value} />
          ))}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <button type="button" className={btn} onClick={() => onChange(value - 1)} disabled={value <= range.min} aria-label={`Fewer ${label.toLowerCase()}`}>
          −
        </button>
        <output className="font-cond num w-14 text-center text-[15px] font-[700] text-bone" aria-live="polite">
          {value} {value === 1 ? unit[0] : unit[1]}
        </output>
        <button type="button" className={btn} onClick={() => onChange(value + 1)} disabled={value >= range.max} aria-label={`More ${label.toLowerCase()}`}>
          +
        </button>
      </div>
    </div>
  )
}

export function FleetGlyph({ kind, on, size = 14, color }: { kind: 'rover' | 'scout'; on: boolean; size?: number; color?: string }) {
  const c = color ?? (on ? '#ece7f5' : '#4a4282')
  return kind === 'rover' ? (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden>
      <path d="M7 1 L13 12 L7 9 L1 12 Z" fill={on ? c : 'none'} stroke={c} strokeWidth="1.2" strokeLinejoin="round" />
    </svg>
  ) : (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden>
      <path d="M7 1 L12 7 L7 13 L2 7 Z" fill={on ? c : 'none'} stroke={c} strokeWidth="1.2" strokeLinejoin="round" />
    </svg>
  )
}

export function EdgePicker({ profiles, value, onChange }: { profiles: EdgeProfileOption[]; value: string; onChange: (id: EdgeProfileOption['id']) => void }) {
  return (
    <div className="grid grid-cols-2 gap-2.5" role="radiogroup" aria-label="Edge compute profile">
      {profiles.map((p) => {
        const on = p.id === value
        return (
          <button key={p.id} type="button" role="radio" aria-checked={on} onClick={() => onChange(p.id)} className={`${choiceBase} ${on ? choiceOn : choiceOff}`}>
            <span className="flex w-full items-baseline justify-between gap-2">
              <span className="font-semi-cond truncate text-[16px] font-[700] text-bone">{p.label}</span>
              <span className="num shrink-0 text-[12px] text-haze">
                {p.watts_idle}–{p.watts_max} W
              </span>
            </span>
            <span className="truncate text-[12px] text-dim">{p.device}</span>
            <span className="num mt-1 flex w-full justify-between gap-2 text-[12px] text-haze">
              <span>Perception ≤ {p.perception_hz_max} Hz</span>
              <span>{Math.round(p.max_expansions / 1000)}k A* nodes</span>
            </span>
            <PowerBar idle={p.watts_idle} max={p.watts_max} on={on} />
          </button>
        )
      })}
    </div>
  )
}

function PowerBar({ idle, max, on }: { idle: number; max: number; on: boolean }) {
  const scale = 60
  return (
    <span className="mt-1 block h-1 w-full rounded-full bg-rule/70" aria-hidden>
      <span
        className={`block h-1 rounded-full ${on ? 'bg-hb' : 'bg-rule-2'}`}
        style={{ marginLeft: `${(idle / scale) * 100}%`, width: `${((max - idle) / scale) * 100}%` }}
      />
    </span>
  )
}

export function BackendPicker({ backends, value, onChange }: { backends: BackendOption[]; value: string; onChange: (id: BackendOption['id']) => void }) {
  return (
    <div className="grid grid-cols-2 rounded-[3px] border border-rule bg-bay p-1" role="radiogroup" aria-label="Simulation backend">
      {backends.map((b) => {
        const on = b.id === value
        return (
          <button
            key={b.id}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={!b.available}
            onClick={() => onChange(b.id)}
            className={`rounded-[2px] px-3 py-2 text-left transition-colors ${on ? 'bg-bay-2 text-bone shadow-[inset_0_0_0_1px_var(--color-hb)]' : 'text-haze hover:text-bone'} disabled:opacity-40`}
          >
            <span className="font-semi-cond block text-[15px] font-[700]">{b.label}</span>
            <span className="block text-[12px] text-dim">
              {!b.available ? 'Not installed' : b.id === 'gazebo' ? 'Full physics and rendered sensors' : 'Lightweight kinematic fallback'}
            </span>
          </button>
        )
      })}
    </div>
  )
}
