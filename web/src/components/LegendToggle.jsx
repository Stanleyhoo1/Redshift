import { useSyncExternalStore } from 'react'
import { degrees } from '../lib/format.js'

// The map key folds to a one-line pill, so on short screens it does not cover the controls
// panel or the question strips. Remembered per browser (localStorage when it is available);
// the first time, it starts folded on screens 820 px tall or less.

const KEY = 'redshift.legend'

function read() {
  try {
    const v = localStorage.getItem(KEY)
    if (v === 'open') return true
    if (v === 'closed') return false
  } catch {
    // private mode or storage disabled: fall through to the screen size
  }
  try {
    return !window.matchMedia('(max-height: 820px)').matches
  } catch {
    return true
  }
}

let open = typeof window === 'undefined' ? true : read()
const subs = new Set()

export function setLegendOpen(value) {
  const v = !!value
  if (v === open) return
  open = v
  try {
    localStorage.setItem(KEY, v ? 'open' : 'closed')
  } catch {
    // keep it for this session only
  }
  subs.forEach((fn) => fn())
}

const subscribe = (fn) => {
  subs.add(fn)
  return () => subs.delete(fn)
}

export const useLegendOpen = () => useSyncExternalStore(subscribe, () => open, () => true)

// The fold button. Open, it is a small chevron in the legend's top-right corner; folded, it is
// the whole legend: "Map key" and the slope limit, which flashes when the limit changes.
export function LegendToggle({ open, lim, changedAt }) {
  return (
    <button
      type="button"
      className="legend-toggle"
      aria-expanded={open}
      aria-label={open ? 'Hide the map key' : 'Show the map key'}
      title={open ? 'Hide the map key' : 'Show the map key'}
      onClick={() => setLegendOpen(!open)}
    >
      {!open && (
        <span className="legend-toggle-text">
          Map key{' '}
          <span className="legend-toggle-lim" key={changedAt ?? 'base'} data-flash={changedAt != null || undefined}>
            slope limit {degrees(lim)} deg
          </span>
        </span>
      )}
      <svg className="legend-toggle-chev" viewBox="0 0 12 12" aria-hidden="true">
        <path d={open ? 'M3 4.5l3 3 3-3' : 'M3 7.5l3-3 3 3'} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  )
}
