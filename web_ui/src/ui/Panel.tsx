import type { ReactNode } from 'react'
import { sensorColor } from '../lib/theme'

// Sensor chips drop out of the header when the panel is too narrow to hold them beside the title.
const CHIPS_FROM = {
  0: 'inline-flex',
  185: 'hidden @min-[185px]:inline-flex',
  205: 'hidden @min-[205px]:inline-flex',
  380: 'hidden @min-[380px]:inline-flex',
} as const

/** Instrument bay: flat fill, hairline edge, title + the sensors that feed it. */
export function Panel({
  title,
  sensors,
  right,
  children,
  className = '',
  bodyClass = '',
  chipsFrom = 0,
}: {
  title: string
  sensors?: string[]
  right?: ReactNode
  children: ReactNode
  className?: string
  bodyClass?: string
  chipsFrom?: keyof typeof CHIPS_FROM
}) {
  return (
    <section className={`@container flex min-h-0 min-w-0 flex-col rounded-[3px] border border-rule bg-bay ${className}`} aria-label={title}>
      <header className="flex h-8 shrink-0 items-center justify-between gap-2 border-b border-rule/80 px-3">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="truncate text-[13px] font-[560] text-haze">{title}</h2>
          {sensors && <SensorChips tags={sensors} className={CHIPS_FROM[chipsFrom]} />}
        </div>
        {right && <div className="flex shrink-0 items-center gap-1.5">{right}</div>}
      </header>
      <div className={`min-h-0 flex-1 ${bodyClass}`}>{children}</div>
    </section>
  )
}

export function SensorChips({ tags, size = 'sm', className = 'inline-flex' }: { tags: string[]; size?: 'sm' | 'md'; className?: string }) {
  return (
    <span className={`${className} items-center gap-1`}>
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
      style={solid ? { backgroundColor: color, color: '#16132b' } : { color, boxShadow: `inset 0 0 0 1px ${color}88`, backgroundColor: 'rgba(22,19,43,0.78)' }}
    >
      {children}
    </span>
  )
}

export function StatePill({ state, color }: { state: string; color: string }) {
  return (
    <span
      className="font-cond inline-flex items-center rounded-[2px] px-1.5 py-[3px] text-[11.5px] leading-none font-[750] tracking-[0.03em] whitespace-nowrap"
      style={{ color, background: `linear-gradient(${color}24, ${color}24), #1e1a38`, boxShadow: `inset 0 0 0 1px ${color}55` }}
    >
      {state.replace(/_/g, ' ')}
    </span>
  )
}
