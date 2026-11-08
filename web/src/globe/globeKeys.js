import { appStore } from '../state/store.js'
import { viewMode } from './viewMode.js'

// G switches between the sector and the planet; on the planet 0 frames the site again
// instead of App's overview. Capture phase on window, like Planner.jsx's Esc: it runs
// before App's own key handler, which therefore never sees a 0 pressed on the planet and
// needs no change. Same guards as App. Returns the uninstall function.
export function installGlobeKeys() {
  const onKey = (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return
    if (appStore.get().modal) return
    if (e.target instanceof Element && e.target.closest('input, textarea, select, [contenteditable="true"]')) return
    const k = e.key.toLowerCase()
    if (k === 'g') {
      e.stopImmediatePropagation()
      viewMode.toggle()
    } else if (k === '0' && viewMode.get().render === 'globe') {
      e.stopImmediatePropagation()
      viewMode.frameHome()
    }
  }
  window.addEventListener('keydown', onKey, true)
  return () => window.removeEventListener('keydown', onKey, true)
}
