import { appStore } from '../state/store.js'

// Point-and-click hazards (PROTOCOL.md version 4). The hazard bar starts and stops
// placement; the 3D view tracks the cursor and calls `commit` on a click. The hazard bar
// owns the request (target rover, status line, cooldowns), so it registers `commit`:
// commit({ x, y }) | commit({ rover }) | commit({ auto: true }) -> Promise<{ ok, error? }>.
export const placement = { commit: null }

export function startPlacing(type, mode) {
  appStore.patch({ placing: { type, mode } })
}

export function stopPlacing() {
  if (appStore.get().placing) appStore.patch({ placing: null })
}

export const isPlacing = () => !!appStore.get().placing

export const PLACE_TEXT = {
  rockfall: { what: 'a rockfall', label: 'Rockfall' },
  sand: { what: 'soft sand', label: 'Soft sand' },
  spire_fall: { what: 'a falling spire', label: 'Falling spire' },
  lifeform: { what: 'a herd of grazers', label: 'Herd of grazers' },
  rover_fault: { what: 'a motor fault', label: 'Motor fault' },
}
