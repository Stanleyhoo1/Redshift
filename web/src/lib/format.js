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

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// The sim writes messages like "r2 sampled ice deposit t7 (found by r3)". The row already shows
// the rover's name, so drop the leading id and swap any other rover ids for their names.
export function cleanMsg(msg, roverId, byId) {
  let out = msg
  if (roverId && out.startsWith(roverId + ' ')) out = out.slice(roverId.length + 1)
  const ids = Object.keys(byId)
  if (!ids.length) return out
  const re = new RegExp(`\\b(${ids.sort((a, b) => b.length - a.length).map(escapeRe).join('|')})\\b`, 'g')
  return out.replace(re, (id) => byId[id]?.name || id)
}
