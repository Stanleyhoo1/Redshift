import { useEffect, useRef, useState } from 'react'
import { appStore, nowSeconds, simStore, useApp, useSim } from '../state/store.js'
import { resetSim } from '../net/connection.js'
import { shortName } from '../lib/format.js'
import { useHudRect } from '../lib/useHudRect.js'
import { useGraphics, visualPrefs } from '../state/visualPrefs.js'
import { ViewModeControl } from '../globe/ViewModeControl.jsx'
import '../styles/scene.css'

// The realistic ground is the first impression: slope colours start off (once per page
// load; the S key and the toggle still switch them).
let slopeDefaultApplied = false

// Graphics quality: High (shadows, gravel, particles, detailed materials) or Low.
function GraphicsControl() {
  const graphics = useGraphics()
  return (
    <div className="hud-group">
      {/* v7 planet view: View, Sector | Planet, laid out like the Graphics row below it */}
      <ViewModeControl />
      <div className="gfx-row">
        <div className="hud-label" id="gfx-label">
          Graphics
        </div>
        <div className="seg gfx-seg" role="group" aria-labelledby="gfx-label">
          <button type="button" aria-pressed={graphics === 'high'} onClick={() => visualPrefs.setGraphics('high')} title="Shadows, gravel, dust and detailed materials">
            High
          </button>
          <button type="button" aria-pressed={graphics === 'low'} onClick={() => visualPrefs.setGraphics('low')} title="Faster: no shadows, gravel or particles, simpler materials">
            Low
          </button>
        </div>
      </div>
    </div>
  )
}

function useStale() {
  const [stale, setStale] = useState(false)
  useEffect(() => {
    const id = setInterval(() => {
      const s = simStore.get()
      const open = appStore.get().conn === 'open'
      setStale(open && s.ready && nowSeconds() - s.lastMsgAt > 3)
    }, 1000)
    return () => clearInterval(id)
  }, [])
  return stale
}

function ConnStatus() {
  const conn = useApp((a) => a.conn)
  const ready = useSim((s) => s.ready)
  const stale = useStale()
  let state = conn
  let label = 'Connecting'
  if (conn === 'open') {
    state = stale ? 'stale' : ready ? 'open' : 'connecting'
    label = stale ? 'No data' : ready ? 'Live' : 'Waiting for sim'
  } else if (conn === 'reconnecting') label = 'Reconnecting'
  return (
    <div className="conn" data-state={state} role="status">
      <span className="conn-dot" aria-hidden="true" />
      {label}
    </div>
  )
}

function Toggle({ label, on, onChange, hint, keyHint }) {
  return (
    <button type="button" className="toggle" aria-pressed={on} onClick={() => onChange(!on)} title={keyHint ? `${label} (${keyHint})` : label}>
      <span className="toggle-label">{label}</span>
      {hint && <span className="toggle-hint">{hint}</span>}
      <span className="switch" aria-hidden="true" />
    </button>
  )
}

function ResetControl() {
  const gen = useSim((s) => s.gen)
  const [phase, setPhase] = useState('idle') // idle | confirm | pending | error
  const [error, setError] = useState('')
  const startGen = useRef(0)

  useEffect(() => {
    if (phase === 'pending' && gen !== startGen.current) setPhase('idle')
  }, [gen, phase])
  useEffect(() => {
    if (phase !== 'error') return
    const id = setTimeout(() => setPhase('idle'), 6000)
    return () => clearTimeout(id)
  }, [phase])
  useEffect(() => {
    if (phase !== 'pending') return
    // The sim answers the POST and then broadcasts init; give up waiting after a while.
    const id = setTimeout(() => setPhase('idle'), 15000)
    return () => clearTimeout(id)
  }, [phase])

  const confirm = async () => {
    startGen.current = gen
    setPhase('pending')
    try {
      await resetSim(Math.floor(Math.random() * 100000))
    } catch (err) {
      setError(err?.message || 'Reset failed')
      setPhase('error')
    }
  }

  if (phase === 'confirm') {
    return (
      <div className="reset reset-confirm">
        <span>Start over with a new random world?</span>
        <div className="reset-actions">
          <button type="button" className="btn btn-danger" onClick={confirm} autoFocus>
            Reset
          </button>
          <button type="button" className="btn" onClick={() => setPhase('idle')}>
            Cancel
          </button>
        </div>
      </div>
    )
  }
  return (
    <div className="reset">
      <button type="button" className="btn btn-wide" disabled={phase === 'pending'} onClick={() => setPhase('confirm')}>
        {phase === 'pending' ? 'Resetting...' : 'Reset sim'}
      </button>
      {phase === 'error' && <p className="reset-error">{error}. Check that the sim is running.</p>}
    </div>
  )
}

export function ViewControls({ view, setView, follow, onSelect, onOverview }) {
  const meta = useSim((s) => s.roverMeta)
  const gen = useSim((s) => s.gen)
  const truth = useApp((a) => a.truth)
  // An older sim sends no scans: say so instead of showing an empty layer.
  const noScan = useSim((s) => s.ticks > 3 && !Object.values(s.rovers).some((r) => r.scan))
  const v4 = useSim((s) => s.v4)
  const ref = useHudRect('controls')
  const set = (key) => (value) => setView((v) => ({ ...v, [key]: value }))
  useEffect(() => {
    if (slopeDefaultApplied) return
    slopeDefaultApplied = true
    setView((v) => (v.slope ? { ...v, slope: false } : v))
  }, [setView])
  let realityHint = ''
  if (view.reality) {
    if (!truth || truth.gen !== gen) realityHint = 'Loading'
    else if (truth.error) realityHint = 'Unavailable'
  }
  return (
    <div className="hud hud-controls" ref={ref}>
      <ConnStatus />
      <div className="hud-group">
        <Toggle label="Reality" on={view.reality} onChange={set('reality')} hint={realityHint} keyHint="R" />
        <Toggle label="Frontiers" on={view.frontiers} onChange={set('frontiers')} keyHint="F" />
        <Toggle label="Paths" on={view.paths} onChange={set('paths')} keyHint="P" />
        <Toggle label="Slope colours" on={view.slope} onChange={set('slope')} keyHint="S" />
        <Toggle label="LiDAR" on={view.lidar} onChange={set('lidar')} hint={view.lidar && noScan ? 'No data' : ''} keyHint="L" />
        {v4 && <Toggle label="Life" on={view.life} onChange={set('life')} keyHint="B" />}
      </div>
      <div className="hud-group">
        <div className="hud-label" id="follow-label">
          Follow
        </div>
        <div className="seg" role="group" aria-labelledby="follow-label">
          <button type="button" aria-pressed={!follow} onClick={onOverview} title="Overview (0)">
            Off
          </button>
          {meta.map((m, i) => (
            <button key={m.id} type="button" aria-pressed={follow === m.id} onClick={() => onSelect(m.id)} title={`Follow ${m.name} and show its sensors (${i + 1})`}>
              <span className="dot" style={{ background: m.color }} />
              {shortName(m.name, m.id)}
            </button>
          ))}
        </div>
      </div>
      <GraphicsControl />
      <ResetControl />
    </div>
  )
}
