import { appStore, simStore } from '../state/store.js'
import { sampleGround, sampleGroundNear } from '../state/simState.js'

// Ground height for things drawn on the terrain: what the fleet measured, else the
// ground truth (when loaded), else the nearest known ground, else `fallback`.
export function groundAt(x, y, fallback = 0) {
  const T = simStore.get().terrain
  const g = sampleGround(T, x, y)
  if (g !== null) return g
  const tr = truthHeight(x, y)
  if (tr !== null) return tr
  return sampleGroundNear(T, x, y, 8) ?? fallback
}

// The true height at (x, y), or null when /api/truth has not loaded (or failed).
export function truthHeight(x, y) {
  const tr = appStore.get().truth
  if (!tr || !tr.heights || tr.gen !== simStore.get().gen) return null
  const i = Math.round((x - tr.ox) / tr.res)
  const j = Math.round((y - tr.oy) / tr.res)
  if (i < 0 || j < 0 || i >= tr.nx || j >= tr.ny) return null
  return tr.heights[j * tr.nx + i]
}

// Is the map sample nearest to (x, y) known to the fleet?
export function knownAt(x, y) {
  const T = simStore.get().terrain
  if (!T) return false
  const i = Math.round((x - T.ox) / T.res)
  const j = Math.round((y - T.oy) / T.res)
  if (i < 0 || j < 0 || i >= T.nx || j >= T.ny) return false
  return !!T.known[j * T.nx + i]
}

export function hashStr(s) {
  let h = 7
  for (const ch of String(s)) h = (Math.imul(h, 31) + ch.charCodeAt(0)) >>> 0
  return h
}

export function pointInPoly(x, y, poly) {
  let inside = false
  for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
    const xa = poly[a][0]
    const ya = poly[a][1]
    const xb = poly[b][0]
    const yb = poly[b][1]
    if (ya > y !== yb > y && x < ((xb - xa) * (y - ya)) / (yb - ya) + xa) inside = !inside
  }
  return inside
}
