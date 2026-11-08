import { useId, useSyncExternalStore } from 'react'
import '../styles/fold.css'

// Collapsible dashboard areas. One shared pattern so every panel folds the same way:
//   const [open, setOpen] = useFold('sidebar.events', true)
//   <FoldButton open={open} onToggle={() => setOpen(!open)} label="events" />
//   <FoldSection id="sidebar.rovers" title="Rovers" summary="3 rovers, 1 low">...</FoldSection>
// The open or folded state is remembered per browser (localStorage, when it is available) under
// redshift.fold.<id>, and every component using the same id stays in step.

const PREFIX = 'redshift.fold.'
const state = new Map()
const subs = new Set()

function read(id, fallback) {
  if (state.has(id)) return state.get(id)
  let v = fallback
  try {
    const s = localStorage.getItem(PREFIX + id)
    if (s === 'open') v = true
    else if (s === 'closed') v = false
  } catch {
    // private mode or storage disabled: use the default
  }
  state.set(id, v)
  return v
}

export function setFold(id, value) {
  const v = !!value
  if (state.get(id) === v) return
  state.set(id, v)
  try {
    localStorage.setItem(PREFIX + id, v ? 'open' : 'closed')
  } catch {
    // keep it for this session only
  }
  subs.forEach((fn) => fn())
}

const subscribe = (fn) => {
  subs.add(fn)
  return () => subs.delete(fn)
}

// [open, setOpen] for one foldable area. defaultOpen is used until the user folds or opens it.
export function useFold(id, defaultOpen = true) {
  const open = useSyncExternalStore(
    subscribe,
    () => read(id, defaultOpen),
    () => defaultOpen,
  )
  return [open, (v) => setFold(id, v)]
}

// The chevron button. Points down while open (click to fold), right while folded.
export function FoldButton({ open, onToggle, label, controls, className = '' }) {
  const what = label ? ` ${label}` : ''
  return (
    <button
      type="button"
      className={`fold-btn ${className}`.trim()}
      aria-expanded={open}
      aria-controls={controls}
      aria-label={open ? `Hide${what}` : `Show${what}`}
      title={open ? `Hide${what}` : `Show${what}`}
      onClick={onToggle}
    >
      <svg className="fold-chev" viewBox="0 0 12 12" aria-hidden="true" data-open={open || undefined}>
        <path d="M4.5 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  )
}

// A section with a clickable header. Folded, only the header shows, with the optional summary
// (a short line such as "3 rovers, 1 low battery") so the user still sees the gist.
// `actions` render on the right of the header and do not toggle the fold.
export function FoldSection({ id, title, summary, defaultOpen = true, actions, className = '', children }) {
  const [open, setOpen] = useFold(id, defaultOpen)
  const bodyId = useId()
  return (
    <section className={`fold ${className}`.trim()} data-folded={!open || undefined}>
      <div className="fold-head">
        <button
          type="button"
          className="fold-head-btn"
          aria-expanded={open}
          aria-controls={bodyId}
          title={open ? `Hide ${title}` : `Show ${title}`}
          onClick={() => setOpen(!open)}
        >
          <svg className="fold-chev" viewBox="0 0 12 12" aria-hidden="true" data-open={open || undefined}>
            <path d="M4.5 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="fold-title">{title}</span>
          {!open && summary ? <span className="fold-summary">{summary}</span> : null}
        </button>
        {actions ? <div className="fold-actions">{actions}</div> : null}
      </div>
      <div className="fold-body" id={bodyId} hidden={!open}>
        {open ? children : null}
      </div>
    </section>
  )
}
