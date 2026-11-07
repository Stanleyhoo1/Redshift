import { useCallback, useEffect, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Scene } from './scene/Scene.jsx'
import { Sidebar } from './components/Sidebar.jsx'
import { ViewControls } from './components/ViewControls.jsx'
import { Legend } from './components/Legend.jsx'
import { startConnection, wsUrl } from './net/connection.js'
import { simStore, useApp, useSim } from './state/store.js'

function Waiting() {
  const ready = useSim((s) => s.ready)
  const conn = useApp((a) => a.conn)
  if (ready) {
    // Keep showing the last known state, but say plainly that it is not live.
    if (conn === 'open') return null
    return (
      <div className="banner" role="status">
        Lost the connection to the sim. Reconnecting...
      </div>
    )
  }
  return (
    <div className="waiting" role="status">
      <div className="waiting-card">
        <p className="waiting-title">{conn === 'open' ? 'Connected. Waiting for the first map update.' : 'Waiting for the simulation'}</p>
        <p className="waiting-sub">
          Looking for the sim at <code>{wsUrl()}</code>. Start the sim server, or run <code>npm run mock</code> for test data.
        </p>
      </div>
    </div>
  )
}

const TOGGLE_KEYS = { r: 'reality', f: 'frontiers', p: 'paths', s: 'slope' }

export default function App() {
  const [view, setView] = useState({ reality: false, frontiers: true, paths: true, slope: true })
  const [follow, setFollow] = useState(null)
  const [overview, setOverview] = useState(0) // bump to glide the camera back to the overview
  const roverMeta = useSim((s) => s.roverMeta)
  const goOverview = useCallback(() => {
    setFollow(null)
    setOverview((n) => n + 1)
  }, [])

  useEffect(() => startConnection(), [])

  // After a reset the followed rover may no longer exist.
  useEffect(() => {
    if (follow && !roverMeta.some((m) => m.id === follow)) setFollow(null)
  }, [roverMeta, follow])

  // Keyboard shortcuts for the demo: R F P S toggle layers, 1-9 follow a rover, 0 or Esc for overview.
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return
      if (e.target instanceof Element && e.target.closest('input, textarea, select, [contenteditable="true"]')) return
      const k = e.key.toLowerCase()
      if (TOGGLE_KEYS[k]) {
        const key = TOGGLE_KEYS[k]
        setView((v) => ({ ...v, [key]: !v[key] }))
      } else if (k === '0' || k === 'escape') {
        goOverview()
      } else if (/^[1-9]$/.test(k)) {
        const m = simStore.get().roverMeta[Number(k) - 1]
        if (m) setFollow(m.id)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [goOverview])

  return (
    <div className="app">
      <main className="view">
        {/* Pixel ratio capped at 1.5: crisp with MSAA, and spares laptop GPUs on retina screens. */}
        <Canvas flat linear dpr={[1, 1.5]} camera={{ position: [0, 68, 46], fov: 42, near: 0.5, far: 2000 }} gl={{ antialias: true, powerPreference: 'high-performance' }}>
          <Scene view={view} follow={follow} overview={overview} />
        </Canvas>
        <div className="vignette" aria-hidden="true" />
        <ViewControls view={view} setView={setView} follow={follow} setFollow={setFollow} onOverview={goOverview} />
        <Legend slopeColors={view.slope} frontiers={view.frontiers} />
        <Waiting />
      </main>
      <Sidebar follow={follow} onFollow={setFollow} />
    </div>
  )
}
