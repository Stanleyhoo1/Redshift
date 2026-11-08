// Pure helpers for the setup screen (v7 launch settings, see PROTOCOL.md "Version 7 additions").
// No React or DOM in here, so scripts/setup-unit.mjs can check them in Node.

export const GROUP_TITLE = {
  planet: 'Planet',
  fleet: 'Fleet',
  sector: 'Sector',
  simulation: 'Simulation',
  features: 'Features',
  wear: 'Wear and repair',
}

// An older sim (no GET /api/config) can still be given a seed, through POST /api/reset.
export const SEED_FIELD = { key: 'seed', type: 'int', label: 'Seed', group: 'planet', help: 'Same seed, same terrain, targets and life.', min: 0, max: 2147483647, step: 1, unit: '' }

const G = 9.81
const INK = [232, 238, 245] // --ink
const RED = [255, 92, 70] // the red end of the Redshift wordmark

const trim = (v) => String(Number(Number(v).toFixed(2)))
const decimals = (n) => {
  const t = String(n)
  const i = t.indexOf('.')
  return i < 0 ? 0 : t.length - i - 1
}

// The server's fields by group, in the server's order. A group this client does not know lands in
// Simulation, so a setting added to the sim later still shows up.
export function fieldsByGroup(fields) {
  const out = {}
  for (const f of Array.isArray(fields) ? fields : []) {
    if (!f || typeof f.key !== 'string') continue
    const g = GROUP_TITLE[f.group] ? f.group : 'simulation'
    ;(out[g] ||= []).push(f)
  }
  return out
}

// Only the settings that differ from `current`: the launch body (keys left out stay as they are).
export function diffSettings(values, current) {
  const out = {}
  for (const [k, v] of Object.entries(values || {})) {
    const c = current?.[k]
    const same = typeof v === 'number' && typeof c === 'number' ? Math.abs(v - c) < 1e-9 : v === c
    if (!same) out[k] = v
  }
  return out
}

// A slider value on the field's step grid and inside its range, without float noise.
export function snap(field, v) {
  const lo = Number(field.min)
  const hi = Number(field.max)
  const step = Number(field.step) || 0
  let x = Math.min(hi, Math.max(lo, Number(v)))
  if (!Number.isFinite(x)) x = lo
  if (step > 0) {
    x = Math.min(hi, lo + Math.round((x - lo) / step) * step)
    x = Number(x.toFixed(Math.min(6, Math.max(decimals(step), decimals(lo)))))
  }
  return field.type === 'int' ? Math.round(x) : x
}

// The physics row (GET /api/config `physics`) nearest this gravity, or null.
export function physicsAt(rows, g) {
  let best = null
  for (const r of Array.isArray(rows) ? rows : []) {
    if (!Number.isFinite(r?.gravity_g)) continue
    if (!best || Math.abs(r.gravity_g - g) < Math.abs(best.gravity_g - g)) best = r
  }
  return best
}

export function presetName(presets, g) {
  const p = (Array.isArray(presets) ? presets : []).find((x) => Math.abs(x.value - g) < 1e-6)
  return p ? p.label : ''
}

export const metersPerSecond2 = (g) => (g * G).toFixed(1)

// Heavier gravity, heavier numeral: Archivo's width narrows, its weight grows and the ink shifts toward
// red, the gravitational redshift the planet is named after. t: 0 at the lightest gravity, 1 at the heaviest.
export function gravityLook(g, lo = 0.5, hi = 3) {
  const t = Math.min(1, Math.max(0, (g - lo) / (hi - lo)))
  const k = (0.55 * Math.max(0, t - 0.3)) / 0.7
  const rgb = INK.map((c, i) => Math.round(c + (RED[i] - c) * k))
  return { stretch: Math.round(122 - t * 50), weight: Math.round(380 + t * 420), color: `rgb(${rgb.join(', ')})` }
}

export function formatValue(field, v) {
  if (!Number.isFinite(v)) return ''
  if (field?.key === 'gravity_g') return `${v.toFixed(1)} g`
  const unit = field?.unit || ''
  if (unit === 'x') return `${trim(v)}x real time`
  return unit ? `${trim(v)} ${unit}` : trim(v)
}

// A seed typed by hand: digits only, inside the field's range; null when it is not a valid seed.
export function parseSeed(text, field = SEED_FIELD) {
  const s = String(text ?? '').trim()
  if (!/^\d{1,10}$/.test(s)) return null
  const n = Number(s)
  return n >= field.min && n <= field.max ? n : null
}

export const randomSeed = () => Math.floor(Math.random() * 100000)

// One sentence under the form: what lands where.
export function summary(values, fields) {
  if (!values) return ''
  const list = Array.isArray(fields) ? fields : []
  if (!list.some((f) => f.key === 'gravity_g')) return Number.isFinite(values.seed) ? `Seed ${values.seed}.` : ''
  const n = Number(values.num_rovers) || 0
  const parts = [`${n} rover${n === 1 ? ' lands' : 's land'} on a ${trim(values.map_size)} m sector at ${Number(values.gravity_g).toFixed(1)} g.`]
  const feats = list.filter((f) => f.group === 'features' && f.type === 'bool')
  const tail = []
  if (feats.length) tail.push(`${feats.filter((f) => values[f.key]).length} of ${feats.length} features on`)
  if (typeof values.wear === 'string') tail.push(values.wear === 'off' ? 'no wear' : `${values.wear} wear`)
  if (Number.isFinite(values.sim_speed)) tail.push(`${trim(values.sim_speed)}x real time`)
  if (tail.length) {
    const s = tail.join(', ')
    parts.push(`${s[0].toUpperCase()}${s.slice(1)}.`)
  }
  return parts.join(' ')
}
