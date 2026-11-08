import { appStore, useApp } from '../state/store.js'
import { postAnswer } from '../net/connection.js'

// Answers this client sends to clarification requests (POST /api/answer). The card and
// the question in the comms thread share this, so answering in one updates the other.

function setAnswer(id, value) {
  appStore.patch({ answers: { ...appStore.get().answers, [id]: value } })
}

export async function sendAnswer(id, option) {
  const cur = appStore.get().answers[id]
  if (cur && (cur.state === 'pending' || cur.state === 'sent')) return
  setAnswer(id, { state: 'pending', option })
  try {
    await postAnswer(id, option)
    setAnswer(id, { state: 'sent', option, at: performance.now() / 1000 })
  } catch (err) {
    setAnswer(id, { state: 'error', option, error: err?.message || 'Answer refused', status: err?.status ?? 0 })
  }
}

export const useAnswer = (id) => useApp((a) => (id ? a.answers[id] || null : null))
export const useAnswers = () => useApp((a) => a.answers)

export const optionLabel = (q, id) => q?.options?.find((o) => o.id === id)?.label || id || ''

// ---------------------------------------------------------------- version 6

// A request that waits for the operator (no countdown, nothing taken automatically).
export const isWaiting = (q) => !!q && q.deadline === null
// The option the rover takes on its own when time runs out: only with a countdown.
export const autoDefault = (q) => (q && q.deadline !== null && q.default ? q.default : null)

// Options that name a place on the map, numbered 1, 2, 3 in the sim's order. Other
// options ("Never mind") get no number.
export function placeNumbers(q) {
  const out = {}
  let n = 0
  for (const o of q?.options || []) if (Number.isFinite(o.x) && Number.isFinite(o.y)) out[o.id] = ++n
  return out
}
export const hasPlaces = (q) => !!q?.options?.some((o) => Number.isFinite(o.x) && Number.isFinite(o.y))

// The first part of a long option label, for a map badge: "Lake 14 m across, 22 m
// north-east of me" -> "Lake 14 m across".
export function shortPlace(label) {
  const s = String(label || '')
  const cut = s.split(/,\s|;\s| - /)[0]
  return cut.length > 30 ? `${cut.slice(0, 28).trimEnd()}...` : cut
}

// Hover link between a map marker, the request card chip and the chat chip.
export const hoverKey = (qid, oid) => `${qid}:${oid}`
export function setPlaceHover(key) {
  if (appStore.get().placeHover !== key) appStore.patch({ placeHover: key })
}
export function clearPlaceHover(key) {
  if (appStore.get().placeHover === key) appStore.patch({ placeHover: null })
}
export const usePlaceHover = () => useApp((a) => a.placeHover)

// Props for an option chip that names a place: hovering it lights its map marker.
export function placeChipProps(q, o, hover) {
  if (!Number.isFinite(o.x) || !Number.isFinite(o.y)) return {}
  const key = hoverKey(q.id, o.id)
  return {
    'data-hl': hover === key || undefined,
    onMouseEnter: () => setPlaceHover(key),
    onMouseLeave: () => clearPlaceHover(key),
    onFocus: () => setPlaceHover(key),
    onBlur: () => clearPlaceHover(key),
  }
}
