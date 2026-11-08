import { useSyncExternalStore } from 'react'

// Client-side look preferences, owned by the 3D view (not the sim). Kept out of store.js.
//   graphics: 'high' (shadows, gravel, particles, detailed materials) or 'low'.
// Remembered in localStorage when it is available.

const KEY = 'redshift.graphics'

function read() {
  try {
    return localStorage.getItem(KEY) === 'low' ? 'low' : 'high'
  } catch {
    return 'high'
  }
}

let graphics = read()
const subs = new Set()

export const visualPrefs = {
  get graphics() {
    return graphics
  },
  setGraphics(value) {
    const v = value === 'low' ? 'low' : 'high'
    if (v === graphics) return
    graphics = v
    try {
      localStorage.setItem(KEY, v)
    } catch {
      // private mode or storage disabled: keep it for this session only
    }
    subs.forEach((fn) => fn())
  },
  subscribe(fn) {
    subs.add(fn)
    return () => subs.delete(fn)
  },
}

export const useGraphics = () => useSyncExternalStore(visualPrefs.subscribe, () => graphics)
export const highGraphics = () => graphics === 'high'
