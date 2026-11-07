import { useCallback, useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Scene } from './scene/Scene.jsx'
import { Sidebar } from './components/Sidebar.jsx'
import { ViewControls } from './components/ViewControls.jsx'
import { Legend } from './components/Legend.jsx'
import { SensorPanel } from './components/SensorPanel.jsx'
import { DustBanner, DustHaze, HazardPanel } from './components/HazardPanel.jsx'
import { startConnection, wsUrl } from './net/connection.js'
import { simStore, useApp, useSim } from './state/store.js'

function Waiting() {
  const ready = useSim((s) => s.ready)
  const conn = useApp((a) => a.conn)
  if (ready) return null
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

// Keep showing the last known state, but say plainly that it is not live.
function ConnBanner() {
  const ready = useSim((s) => s.ready)
  const conn = useApp((a) => a.conn)
  if (!ready || conn === 'open') return null
  return (
    <div className="banner" role="status">
      Lost the connection to the sim. Reconnecting...
    </div>
  )
}

const TOGGLE_KEYS = { r: 'reality', f: 'frontiers', p: 'paths', s: 'slope', l: 'lidar' }

export default function App() {
  const [view, setView] = useState({ reality: false, frontiers: true, paths: true, slope: true, lidar: true })
  const [follow, setFollow] = useState(null)
  const [selected, setSelected] = useState(null) // rover whose sensor panel is open
  const [overview, setOverview] = useState(0) // bump to glide the camera back to the overview
  const [focus, setFocus] = useState(null) // { x, y, key, hazard }: a clicked event's location
  const roverMeta = useSim((s) => s.roverMeta)
  const selectedRef = useRef(selected)
  selectedRef.current = selected

  const goOverview = useCallback(() => {
    setFollow(null)
    setSelected(null)
    setOverview((n) => n + 1)
  }, [])

  // Selecting a rover (card, 3D model or label, Follow button, number key) opens its
  // sensor panel and follows it. Selecting it again closes the panel and returns.
  const select = useCallback(
    (id) => {
      if (id && selectedRef.current === id) {
        goOverview()
        return
      }
      setSelected(id)
      setFollow(id)
    },
    [goOverview],
  )
  // Follow buttons, 3D clicks and number keys only ever open (never toggle closed).
  const show = useCallback((id) => {
    setSelected(id)
    setFollow(id)
  }, [])
  const closePanel = useCallback(() => setSelected(null), [])

  const focusEvent = useCallback((e) => {
    if (!e?.at) return
    setFollow(null)
    setFocus({ x: e.at[0], y: e.at[1], key: `${e.key}:${performance.now()}`, hazard: e.level === 'hazard' })
  }, [])

  useEffect(() => startConnection(), [])

  // After a reset the followed or selected rover may no longer exist.
  useEffect(() => {
    if (follow && !roverMeta.some((m) => m.id === follow)) setFollow(null)
    if (selected && !roverMeta.some((m) => m.id === selected)) setSelected(null)
  }, [roverMeta, follow, selected])

  // Keyboard shortcuts for the demo: R F P S L toggle layers, 1-9 select a rover,
  // Esc closes the sensor panel (then returns to the overview), 0 for the overview.
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return
      if (e.target instanceof Element && e.target.closest('input, textarea, select, [contenteditable="true"]')) return
      const k = e.key.toLowerCase()
      if (TOGGLE_KEYS[k]) {
        const key = TOGGLE_KEYS[k]
        setView((v) => ({ ...v, [key]: !v[key] }))
      } else if (k === 'escape') {
        if (selectedRef.current) setSelected(null)
        else goOverview()
      } else if (k === '0') {
        goOverview()
      } else if (/^[1-9]$/.test(k)) {
        const m = simStore.get().roverMeta[Number(k) - 1]
        if (m) show(m.id)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [goOverview, show])

  return (
    <div className="app">
      <main className="view">
        {/* Pixel ratio capped at 1.5: crisp with MSAA, and spares laptop GPUs on retina screens. */}
        <Canvas flat linear dpr={[1, 1.5]} camera={{ position: [0, 68, 46], fov: 42, near: 0.5, far: 2000 }} gl={{ antialias: true, powerPreference: 'high-performance' }}>
          <Scene view={view} follow={follow} overview={overview} selected={selected} onSelect={show} focus={focus} />
        </Canvas>
        <DustHaze />
        <div className="vignette" aria-hidden="true" />
        <ViewControls view={view} setView={setView} follow={follow} onSelect={show} onOverview={goOverview} />
        <div className="hud-top">
          <HazardPanel />
          <div className="banners">
            <ConnBanner />
            <DustBanner />
          </div>
        </div>
        {selected ? <SensorPanel roverId={selected} onClose={closePanel} /> : <Legend slopeColors={view.slope} frontiers={view.frontiers} />}
        <Waiting />
      </main>
      <Sidebar follow={follow} selected={selected} onSelect={select} onFocusEvent={focusEvent} />
    </div>
  )
}
