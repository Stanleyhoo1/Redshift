import { useCallback, useEffect, useRef } from 'react'
import { hudLayout } from '../state/store.js'

// Reports where a HUD panel sits over the 3D view (hudLayout[key]), so the camera rig
// can keep the picture centred in the space the panels leave free. Returns a ref.
export function useHudRect(key) {
  const el = useRef(null)
  const ro = useRef(null)

  const measure = useCallback(() => {
    const node = el.current
    const view = node?.closest('.view')
    if (!node || !view) return
    const a = node.getBoundingClientRect()
    const b = view.getBoundingClientRect()
    const next = { left: a.left - b.left, top: a.top - b.top, right: a.right - b.left, bottom: a.bottom - b.top }
    const prev = hudLayout[key]
    if (prev && ['left', 'top', 'right', 'bottom'].every((k) => Math.abs(prev[k] - next[k]) < 0.5)) return
    // Hidden panels (display: none) measure as an empty box: treat them as absent.
    hudLayout[key] = a.width > 0 && a.height > 0 ? next : null
    hudLayout.version++
  }, [key])

  const ref = useCallback(
    (node) => {
      ro.current?.disconnect()
      el.current = node
      if (!node) {
        if (hudLayout[key]) {
          hudLayout[key] = null
          hudLayout.version++
        }
        return
      }
      ro.current = new ResizeObserver(measure)
      ro.current.observe(node)
      const view = node.closest('.view')
      if (view) ro.current.observe(view)
      measure()
    },
    [key, measure],
  )

  useEffect(() => {
    window.addEventListener('resize', measure)
    return () => {
      window.removeEventListener('resize', measure)
      ro.current?.disconnect()
    }
  }, [measure])

  return ref
}
