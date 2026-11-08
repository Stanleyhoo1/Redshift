import { useEffect, useRef, useState } from 'react'
import { useSim } from '../state/store.js'
import '../styles/interaction.css'

// A large viewer over the whole page: Esc or a click outside the box closes it, focus
// moves into it and back to where it was when it closes. While it is open App ignores
// its own shortcuts (see App.jsx), so keys only reach the viewer.
export function Modal({ label, onClose, className = '', children, onKey }) {
  const box = useRef(null)
  const downOutside = useRef(false)
  const close = useRef(onClose)
  close.current = onClose
  const keyFn = useRef(onKey)
  keyFn.current = onKey
  // Where focus was when the viewer opened, read while rendering: by the time an effect runs, App has made
  // the page behind inert and the browser may have moved focus off that element already.
  const [before] = useState(() => (typeof document === 'undefined' ? null : document.activeElement))

  useEffect(() => {
    box.current?.focus({ preventScroll: true })
    const onKeyDown = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'Escape') {
        e.preventDefault()
        close.current()
        return
      }
      if (e.target instanceof Element && e.target.closest('input, textarea, select')) return
      keyFn.current?.(e)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      if (before instanceof HTMLElement && before.isConnected) before.focus({ preventScroll: true })
    }
  }, [before])

  return (
    <div
      className="ix-backdrop"
      onPointerDown={(e) => {
        downOutside.current = e.target === e.currentTarget
      }}
      onClick={(e) => {
        if (downOutside.current && e.target === e.currentTarget) onClose()
        downOutside.current = false
      }}
    >
      <section className={`ix-modal ${className}`} role="dialog" aria-modal="true" aria-label={label} tabIndex={-1} ref={box}>
        {children}
      </section>
    </div>
  )
}

export function CloseButton({ onClick }) {
  return (
    <button type="button" className="ix-close" onClick={onClick} title="Close (Esc)" aria-label="Close">
      <svg viewBox="0 0 12 12" aria-hidden="true">
        <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" />
      </svg>
    </button>
  )
}

// One button per rover, in its colour. Number keys pick them too (handled by the viewer).
export function RoverSwitch({ value, onChange }) {
  const meta = useSim((s) => s.roverMeta)
  return (
    <div className="seg ix-rovers" role="group" aria-label="Rover">
      {meta.map((m, i) => (
        <button key={m.id} type="button" aria-pressed={value === m.id} onClick={() => onChange(m.id)} title={`${m.name} (${i + 1})`}>
          <span className="dot" style={{ background: m.color }} />
          {m.name}
        </button>
      ))}
    </div>
  )
}

// Number keys 1-9 pick a rover inside a viewer.
export function roverFromKey(e, meta) {
  if (!/^[1-9]$/.test(e.key)) return null
  return meta[Number(e.key) - 1]?.id ?? null
}
