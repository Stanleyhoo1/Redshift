import { simTimeNow } from '../state/simState.js'

// Version 7 rover `focus`: the area a rover concentrates on for a while, because the
// operator asked ("explore around the lake") or out of its own curiosity after a find.

// Sim seconds left right now, counted down smoothly between ticks (null: no time given).
export function focusLeft(s, focus, now) {
  if (!focus || !Number.isFinite(focus.ends_in)) return null
  return Math.max(0, focus.ends_in - Math.max(0, simTimeNow(s, now) - s.t))
}

// Status line for the rover card and the sensor panel.
export function focusStatus(focus, left = focus?.ends_in) {
  const tail = Number.isFinite(left) ? ` (${Math.ceil(left)} s left)` : ''
  return focus?.reason === 'interest' ? `Curious: sticking around${tail}` : `Exploring an area${tail}`
}

// Longer description, for tooltips: "Exploring the pond's shore, within 15 m".
export function focusTitle(focus) {
  if (!focus) return ''
  const what = focus.reason === 'interest' ? `Curious about ${focus.label || 'this area'}, sticking around to look for more` : `Exploring ${focus.label || 'this area'}, as you asked`
  return `${what}: unknown ground within ${Math.round(focus.r)} m of (${Math.round(focus.x)}, ${Math.round(focus.y)}) first`
}

// The tag on the map, without the countdown: "Exploring the pond's shore".
export function focusMapLabel(focus) {
  if (!focus) return ''
  if (focus.reason === 'interest') return focus.label ? `Curious about ${focus.label}` : 'Curious: sticking around'
  return focus.label ? `Exploring ${focus.label}` : 'Exploring this area'
}
