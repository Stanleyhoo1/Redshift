import { useSyncExternalStore } from 'react'
import { emptyState, reduce } from './simState.js'
import { recordTelemetry } from './telemetry.js'

export const nowSeconds = () => performance.now() / 1000

function createStore(initial) {
  let state = initial
  const subs = new Set()
  // Deferred notification (see setSoon): one macrotask later, so a burst of messages
  // that queued up while the page was busy costs one React render instead of one each.
  let due = false
  let channel = null
  const notify = () => {
    due = false
    subs.forEach((fn) => fn())
  }
  return {
    get: () => state,
    set(next) {
      if (next === state) return
      state = next
      due = false
      subs.forEach((fn) => fn())
    },
    // The new state is readable at once (the 3D view reads it every frame); React
    // subscribers hear about it right after the current batch of queued events.
    setSoon(next) {
      if (next === state) return
      state = next
      if (due) return
      due = true
      if (typeof MessageChannel === 'undefined') return void setTimeout(() => due && notify(), 0)
      if (!channel) {
        channel = new MessageChannel()
        channel.port1.onmessage = () => due && notify()
      }
      channel.port2.postMessage(0)
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
  // v4 point-and-click hazards: { type, mode: 'point' | 'rover' } while the user places one.
  placing: null,
  // v4b: a target the user picked (Samples tab or its pin): its value tag stays up.
  pickedTarget: null,
  // The user panned away from the followed rover: the camera is free, the rover stays
  // selected, and "Back to" resumes. Set by the camera rig, cleared by App.
  followPaused: false,
  // A large viewer over everything: { kind: 'camera', rover, view } or
  // { kind: 'sensors', rover, focus: 'lidar' | 'motion' | 'compute' }, or null.
  modal: null,
  // v6: the place candidate under the pointer, `${requestId}:${optionId}`, shared by the
  // map markers, the request card and the chat chips so each lights the others.
  placeHover: null,
  // v8: draw the detector's boxes over the camera pictures (remembered in localStorage).
  boxes: readPref('redshift.boxes', 'on') !== 'off',
})

function readPref(key, fallback) {
  try {
    return localStorage.getItem(key) ?? fallback
  } catch {
    return fallback
  }
}

export function setShowBoxes(on) {
  appStore.patch({ boxes: !!on })
  try {
    localStorage.setItem('redshift.boxes', on ? 'on' : 'off')
  } catch {
    // private mode or storage disabled: this session only
  }
}

export function dispatchSim(msg) {
  const now = nowSeconds()
  const next = reduce(simStore.get(), msg, now)
  if (msg && (msg.type === 'tick' || msg.type === 'init')) recordTelemetry(next, msg, now)
  simStore.setSoon(next)
}

// Selectors must return values that keep their identity until they change.
export const useSim = (selector) => useSyncExternalStore(simStore.subscribe, () => selector(simStore.get()))
export const useApp = (selector) => useSyncExternalStore(appStore.subscribe, () => selector(appStore.get()))

// Displayed (interpolated) rover poses, written by the 3D rovers every frame and read
// by the camera rig for follow mode. Plain mutable map, no React involved.
export const roverPoses = new Map()

// v4: displayed (interpolated) track positions, { x, y, z } in three.js coordinates,
// written by the track markers and read by the observing rovers' lines of sight.
export const trackPoses = new Map()

// Where the HUD panels sit over the 3D view, in px relative to the view's top left:
// { left, top, right, bottom } per panel, or null when the panel is not shown. Written
// by the panels (useHudRect), read by the camera rig every frame to keep the picture
// centred in the space the panels leave free.
export const hudLayout = { controls: null, hazards: null, sensors: null, requests: null, version: 0 }
