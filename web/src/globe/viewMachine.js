// The switch between the sector view ('plane') and the planet view ('globe') as a small
// state machine, pure so Node can test it. Times are seconds on any one clock.
//   view    the view asked for
//   render  the scene being drawn
//   phase   idle | out (leaving the drawn scene: the sector camera rises, or the planet
//           camera dives) | fade (a short fade to black, then render switches) | in (the
//           new camera eases in)
//   since   when the phase began; seq counts accepted requests; home counts the 0 key
//   via     how the drawn scene was reached: 'start', 'fade', 'turn' (turned round
//           half-way, so nothing was faded) or 'instant' (reduced motion)

export const DUR = { out: 0.75, fade: 0.2, in: 1.5 }
const EPS = 1e-6
const VIEWS = new Set(['plane', 'globe'])

export const initialView = () => ({ view: 'plane', render: 'plane', phase: 'idle', since: 0, seq: 0, home: 0, via: 'start' })

export function requestView(s, want, now, instant = false) {
  if (!VIEWS.has(want)) return s
  if (instant) {
    if (s.view === want && s.render === want && s.phase === 'idle') return s
    return { ...s, view: want, render: want, phase: 'idle', since: now, seq: s.seq + 1, via: 'instant' }
  }
  if (s.view === want) return s
  // Still drawing the scene asked for (turned round during the rise, the dive or the fade):
  // ease back in from wherever the camera is. Otherwise leave the drawn scene first.
  if (s.render === want) return { ...s, view: want, phase: 'in', since: now, seq: s.seq + 1, via: 'turn' }
  return { ...s, view: want, phase: 'out', since: now, seq: s.seq + 1 }
}

// Moves to the next phase when the current one has run its time; the same object otherwise.
export function advance(s, now) {
  const dt = now - s.since + EPS
  if (s.phase === 'out' && dt >= DUR.out) return { ...s, phase: 'fade', since: now }
  if (s.phase === 'fade' && dt >= DUR.fade) return { ...s, render: s.view, phase: 'in', since: now, via: 'fade' }
  if (s.phase === 'in' && dt >= DUR.in) return { ...s, phase: 'idle', since: now }
  return s
}

export const frameHome = (s) => ({ ...s, home: s.home + 1 })

// How far through the current phase, 0 to 1 (1 when idle).
export function progress(s, now) {
  const d = DUR[s.phase]
  return d ? Math.min(1, Math.max(0, (now - s.since) / d)) : 1
}

// Selectors. They return primitives, so they are safe for useSyncExternalStore.
export const globeShown = (s) => s.render === 'globe'
export const planeLabelsShown = (s) => s.view === 'plane' && s.render === 'plane'
export const fading = (s) => s.phase === 'fade'
export const globeSettled = (s) => s.render === 'globe' && s.view === 'globe' && s.phase === 'idle'
export const globeCardShown = (s) => s.render === 'globe' && s.view === 'globe'
