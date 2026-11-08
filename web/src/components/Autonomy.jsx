import { useEffect, useState } from 'react'
import { useSim } from '../state/store.js'
import { postChat, postMode } from '../net/connection.js'
import { GemmaStar } from './Planner.jsx'

// Version 8: who makes the rovers' decisions (fleet `mode`, POST /api/mode), and the
// fleet `priority` the operator set by chat ("prioritize the gaps").

const MODES = {
  autonomous: {
    label: 'Autonomous',
    note: 'Gemma decides',
    tip: 'Gemma decides on its own (the rules when it is offline) and logs every choice with its reasons. Your orders still win.',
  },
  supervised: {
    label: 'Supervised',
    note: 'You decide',
    tip: "Every decision comes to you as a question, with Gemma's suggestion marked. The rovers hold until you answer.",
  },
}

const PersonIcon = () => (
  <svg className="mode-ico" viewBox="0 0 16 16" aria-hidden="true">
    <circle cx="8" cy="5.2" r="2.6" />
    <path d="M3 14c.4-2.9 2.4-4.4 5-4.4s4.6 1.5 5 4.4" />
  </svg>
)

// Compact Autonomous / Supervised switch. Shows the sim's `fleet.mode`; a click posts the
// other mode and shows it as pending until the sim confirms it in a tick. Hidden with an
// older sim (no `mode`).
export function ModeSwitch() {
  const mode = useSim((s) => (typeof s.fleet.mode === 'string' ? s.fleet.mode : null))
  const [pending, setPending] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (pending && mode === pending) setPending(null)
  }, [mode, pending])
  useEffect(() => {
    if (!pending) return
    const id = setTimeout(() => setPending(null), 6000) // the sim never confirmed: show what it says
    return () => clearTimeout(id)
  }, [pending])
  useEffect(() => {
    if (!error) return
    const id = setTimeout(() => setError(null), 6000)
    return () => clearTimeout(id)
  }, [error])

  if (!mode) return null
  const shown = pending || (MODES[mode] ? mode : 'autonomous')
  const set = async (m) => {
    if (m === shown) return
    setError(null)
    setPending(m)
    try {
      await postMode(m)
    } catch (err) {
      setPending(null)
      setError(err?.message || 'The sim refused the change')
    }
  }
  return (
    <div className="mode-row" data-mode={shown}>
      <div className="mode-seg" role="radiogroup" aria-label="Who makes the rovers' decisions">
        {Object.entries(MODES).map(([id, m]) => (
          <button
            key={id}
            type="button"
            role="radio"
            aria-checked={shown === id}
            data-mode={id}
            data-pending={(pending === id) || undefined}
            data-tip={m.tip}
            aria-label={`${m.label}: ${m.tip}`}
            onClick={() => set(id)}
          >
            {id === 'autonomous' ? <GemmaStar className="mode-ico mode-star" /> : <PersonIcon />}
            {m.label}
          </button>
        ))}
      </div>
      <span className="mode-note" data-error={!!error || undefined} role={error ? 'alert' : undefined}>
        {error || (pending ? 'Switching...' : MODES[shown].note)}
      </span>
    </div>
  )
}

// "Priority: the unexplored gaps (4 left)" with a cancel button that tells the fleet "back to
// normal" over chat (the protocol's way to clear a priority).
function PriorityInner({ p }) {
  const [state, setState] = useState(null) // null | 'sending' | { error }
  const label = p.label || (p.kind ? `the ${p.kind}` : 'set by the operator')
  const left = p.count !== null ? ` (${p.count} left)` : ''
  const cancel = async () => {
    setState('sending')
    try {
      await postChat('back to normal')
    } catch (err) {
      setState({ error: err?.message || 'Could not reach the sim' })
    }
  }
  const text = `Priority: ${label}${left}`
  return (
    <div className="prio" data-kind={p.kind || undefined} role="status">
      <svg className="prio-ico" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M4 14V2.5M4 3h7.5l-1.8 2.6L11.5 8H4" />
      </svg>
      <span className="prio-text" title={text}>
        Priority: <b>{label}</b>
        {left}
      </span>
      {state === 'sending' ? (
        <span className="prio-busy">Cancelling...</span>
      ) : (
        <button type="button" className="prio-x" onClick={cancel} title='Back to normal: sends "back to normal" to the fleet' aria-label="Cancel the priority">
          <svg viewBox="0 0 12 12" aria-hidden="true">
            <path d="M3 3l6 6M9 3l-6 6" />
          </svg>
        </button>
      )}
      {state?.error && <span className="prio-err">{state.error}</span>}
    </div>
  )
}

export function PriorityChip() {
  const p = useSim((s) => s.priority)
  if (!p) return null
  return <PriorityInner key={`${p.kind}:${p.since}:${p.label}`} p={p} />
}
