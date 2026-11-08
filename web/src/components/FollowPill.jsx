import { useEffect, useRef, useState } from 'react'
import { hudLayout, useSim } from '../state/store.js'
import '../styles/interaction.css'

// Where the pill goes: centred over the free part of the view, just above the sensor
// panel when it is open (otherwise near the bottom edge). Reads the measured HUD panel
// rectangles (hudLayout), polled lightly while the pill is up.
function usePlacement(ref) {
  const [pos, setPos] = useState(null)
  useEffect(() => {
    let seen = -1
    let size = ''
    const update = () => {
      const view = ref.current?.closest('.view')
      if (!view) return
      const W = view.clientWidth
      const H = view.clientHeight
      const key = `${W}x${H}`
      if (hudLayout.version === seen && key === size) return
      seen = hudLayout.version
      size = key
      const c = hudLayout.controls
      const sp = hudLayout.sensors
      const left = c && W > 640 ? c.right : 0
      const x = (left + W) / 2
      const bottom = sp && sp.top > 120 ? H - sp.top + 12 : 22
      setPos((p) => (p && p.x === x && p.bottom === bottom ? p : { x, bottom }))
    }
    update()
    const id = setInterval(update, 200)
    return () => clearInterval(id)
  }, [ref])
  return pos
}

// Shown while the user has panned away from the followed rover. Click (or F) glides
// the camera back and follows again; the rover stayed selected all along.
export function FollowPill({ roverId, onResume }) {
  const meta = useSim((s) => s.roverMeta.find((m) => m.id === roverId) || null)
  const ref = useRef(null)
  const pos = usePlacement(ref)
  if (!meta) return null
  return (
    <div className="ix-follow" ref={ref} role="status" style={pos ? { left: pos.x, bottom: pos.bottom } : { visibility: 'hidden' }}>
      <span className="ix-follow-note">
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M8 1.5v13M1.5 8h13M8 1.5 6 3.5M8 1.5l2 2M8 14.5l-2-2M8 14.5l2-2M1.5 8l2-2M1.5 8l2 2M14.5 8l-2-2M14.5 8l-2 2" />
        </svg>
        Free camera
      </span>
      <button type="button" className="ix-follow-btn" onClick={onResume} title={`Follow ${meta.name} again (F)`} style={{ '--rover': meta.color }}>
        <span className="dot" aria-hidden="true" />
        Back to {meta.name}
        <kbd>F</kbd>
      </button>
    </div>
  )
}
