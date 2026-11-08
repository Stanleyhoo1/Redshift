import { useSyncExternalStore } from 'react'
import { stopPlacing } from '../lib/placement.js'
import { reducedMotion } from '../scene/sceneConstants.js'
import { advance, frameHome as bumpHome, initialView, requestView } from './viewMachine.js'

// Which 3D view is up, the landing sector ('plane') or the whole planet ('globe'), and
// where the switch between them is (see viewMachine.js). Client-side only: the 3D rigs
// read it every frame, React reads it through useViewMode.

export { fading, globeCardShown, globeSettled, globeShown, planeLabelsShown } from './viewMachine.js'

const clock = () => performance.now() / 1000
let state = initialView()
const subs = new Set()

function set(next) {
  if (next === state) return
  state = next
  subs.forEach((fn) => fn())
}

export const viewMode = {
  get: () => state,
  // Ask for a view. Hazard placement never survives the trip to the planet. With
  // prefers-reduced-motion the switch is instant.
  request(want, { instant = reducedMotion } = {}) {
    if (want === 'globe') stopPlacing()
    set(requestView(state, want, clock(), instant))
  },
  toggle(opts) {
    viewMode.request(state.view === 'globe' ? 'plane' : 'globe', opts)
  },
  // The 0 key on the planet: glide back to the home view.
  frameHome() {
    if (state.render === 'globe') set(bumpHome(state))
  },
  // Called once per frame (GlobeLayer's ViewClock) to move the switch along.
  tick(now = clock()) {
    set(advance(state, now))
  },
  subscribe(fn) {
    subs.add(fn)
    return () => subs.delete(fn)
  },
}

export const useViewMode = (selector) => useSyncExternalStore(viewMode.subscribe, () => selector(state))

// Anything that acts on the sector calls this first. On the planet, or on the way to it,
// it starts the flight back down and returns true.
export function leaveGlobe() {
  if (state.view !== 'globe' && state.render !== 'globe') return false
  viewMode.request('plane')
  return true
}
