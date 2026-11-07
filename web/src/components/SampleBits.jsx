import { instrumentStyle, kindStyle } from '../lib/theme.js'

// Small pieces for sample collection (PROTOCOL.md version 4b), shared by the rover cards,
// the sensor panel, the Samples tab and the 3D tags.

const PATHS = {
  drill: ['M8 1.6v2.6', 'M5.4 4.2h5.2', 'M6.4 4.2 8 14.4l1.6-10.2', 'M6.9 7.4h2.2', 'M7.3 10.4h1.4'],
  scoop: ['M2.6 7.2h7.6l3-3.4', 'M2.6 7.2c0 2.9 2 4.9 4.6 4.9s4.3-2 4.3-4.9'],
  sniffer: ['M4.6 14.2v-3.6h6.8v3.6', 'M6 8.4c0-1.3 1.1-1.3 1.1-2.6S6 4.5 6 3.2', 'M9.9 8.4c0-1.3 1.1-1.3 1.1-2.6S9.9 4.5 9.9 3.2'],
  swab: ['M2.8 13.2 9.4 6.6', 'M9.4 6.6c-.9-.9-.9-2.6.4-3.9s3-1.3 3.9-.4.9 2.6-.4 3.9-3 1.3-3.9.4z'],
}

export function InstrumentIcon({ kind, className = 'instr-icon' }) {
  const d = PATHS[kind] || PATHS.drill
  return (
    <svg className={className} viewBox="0 0 16 16" aria-hidden="true">
      {d.map((p, i) => (
        <path key={i} d={p} />
      ))}
    </svg>
  )
}

// Tube slots: filled ones in the colour of what they hold.
export function Tubes({ samples, capacity, targets, title }) {
  if (!Number.isFinite(capacity) || capacity <= 0) return null
  const ids = Array.isArray(samples) ? samples : []
  const slots = []
  for (let k = 0; k < Math.min(capacity, 6); k++) {
    const id = ids[k]
    const t = id ? targets.find((x) => x.id === id) : null
    slots.push(<i key={k} className="tube" data-full={id ? true : undefined} style={id ? { '--c': kindStyle(t?.kind).color } : undefined} />)
  }
  const label = title ?? `${ids.length} of ${capacity} sample tubes full`
  return (
    <span className="tubes" title={label} aria-label={label} data-full={ids.length >= capacity || undefined}>
      {slots}
    </span>
  )
}

export function ValueBar({ value }) {
  const v = Number.isFinite(value) ? Math.max(0, Math.min(10, value)) : null
  return (
    <span className="vbar" data-none={v === null || undefined} aria-hidden="true">
      <i style={{ width: `${v === null ? 0 : v * 10}%` }} />
    </span>
  )
}

// "Drill 40%" with a thin progress bar.
export function SamplingLine({ sampling, target }) {
  const ins = instrumentStyle(sampling.instrument)
  const pct = Math.round(sampling.progress * 100)
  const ks = target ? kindStyle(target.kind) : null
  return (
    <span className="sampling" title={`${ins.verb}${target ? ` at the ${ks.long} ${target.id}` : ''}: ${pct}%`}>
      <InstrumentIcon kind={sampling.instrument} />
      <b>{ins.label}</b>
      <span className="sampling-bar" aria-hidden="true">
        <i style={{ width: `${pct}%` }} />
      </span>
      <span className="sampling-pct">{pct}%</span>
    </span>
  )
}
