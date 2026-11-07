export function mmss(t) {
  if (!Number.isFinite(t) || t < 0) t = 0
  const total = Math.floor(t)
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function fixed(v, digits = 1, fallback = '--') {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : fallback
}

// "Rover 2" stays "Rover 2"; short labels for tight spots ("2").
export function shortName(name, id) {
  const m = /^rover\s+(.+)$/i.exec(name || '')
  return m ? m[1] : name || id
}

export function degrees(v) {
  if (!Number.isFinite(v)) return '--'
  return Math.abs(v - Math.round(v)) < 0.05 ? String(Math.round(v)) : v.toFixed(1)
}
