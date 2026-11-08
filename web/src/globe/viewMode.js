import { useSyncExternalStore } from 'react'
import { appStore, simStore } from '../state/store.js'
import { stopPlacing } from '../lib/placement.js'
import { hasPlaces } from '../lib/answers.js'
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

// Brings the sector back for things that do not go through App: a rover asking the
// operator to pick a place (each such question once; one already open when the planet was
// asked for does not bounce it), and hazard placement started from anywhere. Returns the
// uninstall function. GlobeLayer installs it.
export function installViewGuards() {
  const seen = new Set()
  // Every place question in this state counts as seen, so none of them bounces the view.
  const noteAll = (s) => {
    seen.clear()
    for (const q of s.requests) if (hasPlaces(q)) seen.add(q.id)
  }
  let gen = simStore.get().gen
  noteAll(simStore.get())
  const offSim = simStore.subscribe(() => {
    const s = simStore.get()
    // A new init: Reset sim, or a reconnect, which re-sends the open questions with their
    // old ids (applyInit bumps gen either way). What it carries counts as seen; only place
    // questions that arrive after it bring the sector back.
    if (s.gen !== gen) {
      gen = s.gen
      noteAll(s)
      return
    }
    for (const q of s.requests) {
      if (!hasPlaces(q) || seen.has(q.id)) continue
      seen.add(q.id)
      leaveGlobe()
    }
  })
  const offApp = appStore.subscribe(() => {
    if (appStore.get().placing) leaveGlobe()
  })
  return () => {
    offSim()
    offApp()
  }
}
