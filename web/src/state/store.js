import { useSyncExternalStore } from 'react'
import { emptyState, reduce } from './simState.js'

export const nowSeconds = () => performance.now() / 1000

function createStore(initial) {
  let state = initial
  const subs = new Set()
  return {
    get: () => state,
    set(next) {
      if (next === state) return
      state = next
      subs.forEach((fn) => fn())
    },
    patch(partial) {
      this.set({ ...state, ...partial })
    },
    subscribe(fn) {
      subs.add(fn)
      return () => subs.delete(fn)
    },
  }
}

// Everything that comes from the sim.
export const simStore = createStore(emptyState())

// Connection, ground truth and other client-side facts.
export const appStore = createStore({
  conn: 'connecting', // connecting | open | reconnecting
  truth: null, // { gen, nx, ny, res, ox, oy, heights: Float32Array, targets } or { gen, error }
  // v3: answers this client sent to clarification requests, by request id:
  // { state: 'pending' | 'sent' | 'error', option, error, status, at }
  answers: {},
})

export function dispatchSim(msg) {
  simStore.set(reduce(simStore.get(), msg, nowSeconds()))
}

// Selectors must return values that keep their identity until they change.
export const useSim = (selector) => useSyncExternalStore(simStore.subscribe, () => selector(simStore.get()))
export const useApp = (selector) => useSyncExternalStore(appStore.subscribe, () => selector(appStore.get()))

// Displayed (interpolated) rover poses, written by the 3D rovers every frame and read
// by the camera rig for follow mode. Plain mutable map, no React involved.
export const roverPoses = new Map()

// Where the HUD panels sit over the 3D view, in px relative to the view's top left:
// { left, top, right, bottom } per panel, or null when the panel is not shown. Written
// by the panels (useHudRect), read by the camera rig every frame to keep the picture
// centred in the space the panels leave free.
export const hudLayout = { controls: null, hazards: null, sensors: null, version: 0 }
