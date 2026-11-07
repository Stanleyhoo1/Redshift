import type { ReactNode } from 'react'
import { sensorColor } from '../lib/theme'

/** Instrument bay: flat fill, hairline edge, title + the sensors that feed it. */
export function Panel({
  title,
  sensors,
  right,
  children,
  className = '',
  bodyClass = '',
}: {
  title: string
  sensors?: string[]
  right?: ReactNode
  children: ReactNode
  className?: string
  bodyClass?: string
}) {
  return (
    <section className={`flex min-h-0 min-w-0 flex-col rounded-[3px] border border-rule bg-bay ${className}`} aria-label={title}>
      <header className="flex h-8 shrink-0 items-center justify-between gap-2 border-b border-rule/80 px-3">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="truncate text-[13px] font-[560] text-haze">{title}</h2>
          {sensors && <SensorChips tags={sensors} />}
        </div>
        {right && <div className="flex shrink-0 items-center gap-1.5">{right}</div>}
      </header>
      <div className={`min-h-0 flex-1 ${bodyClass}`}>{children}</div>
    </section>
  )
}

export function SensorChips({ tags, size = 'sm' }: { tags: string[]; size?: 'sm' | 'md' }) {
  return (
    <span className="inline-flex items-center gap-1">
      {tags.map((t) => (
        <SensorChip key={t} tag={t} size={size} />
      ))}
    </span>
  )
}

export function SensorChip({ tag, size = 'sm' }: { tag: string; size?: 'sm' | 'md' }) {
  const c = sensorColor(tag)
  return (
    <span
      className={`font-cond inline-flex items-center rounded-[2px] border font-[700] leading-none tracking-[0.02em] ${size === 'md' ? 'px-1.5 py-[3px] text-[11.5px]' : 'px-1 py-[2px] text-[10.5px]'}`}
      style={{ color: c, borderColor: `${c}66`, backgroundColor: `${c}14` }}
    >
      {tag}
    </span>
  )
}

export function Badge({ children, color, solid = false, pulse = false }: { children: ReactNode; color: string; solid?: boolean; pulse?: boolean }) {
  return (
    <span
      className={`font-cond inline-flex items-center gap-1 rounded-[2px] px-1.5 py-[3px] text-[11.5px] leading-none font-[750] tracking-[0.02em] ${pulse ? 'animate-pulse-ring' : ''}`}
      style={solid ? { backgroundColor: color, color: '#16132b' } : { color, boxShadow: `inset 0 0 0 1px ${color}88`, backgroundColor: `${color}1a` }}
    >
      {children}
    </span>
  )
}

export function StatePill({ state, color }: { state: string; color: string }) {
  return (
    <span
      className="font-cond inline-flex items-center rounded-[2px] px-1.5 py-[3px] text-[11.5px] leading-none font-[750] tracking-[0.03em] whitespace-nowrap"
      style={{ color, backgroundColor: `${color}1f`, boxShadow: `inset 0 0 0 1px ${color}55` }}
    >
      {state.replace(/_/g, ' ')}
    </span>
  )
}
