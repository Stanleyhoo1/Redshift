import { useCallback, useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Scene } from './scene/Scene.jsx'
import { Sidebar } from './components/Sidebar.jsx'
import { ViewControls } from './components/ViewControls.jsx'
import { Legend } from './components/Legend.jsx'
import { SensorPanel } from './components/SensorPanel.jsx'
import { DustBanner, DustHaze, HazardPanel, PlacementHint } from './components/HazardPanel.jsx'
import { RequestCards } from './components/RequestCards.jsx'
import { FollowPill } from './components/FollowPill.jsx'
import { CameraViewer } from './components/CameraViewer.jsx'
import { SensorDetail } from './components/SensorDetail.jsx'
import { startConnection, wsUrl } from './net/connection.js'
import { appStore, simStore, useApp, useSim } from './state/store.js'
import { stopPlacing } from './lib/placement.js'
// v7 planet view (src/globe)
import { GlobeFade } from './globe/GlobeFade.jsx'

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

const TOGGLE_KEYS = { r: 'reality', f: 'frontiers', p: 'paths', s: 'slope', l: 'lidar', b: 'life' }

export default function App() {
  const [view, setView] = useState({ reality: false, frontiers: true, paths: true, slope: true, lidar: true, life: true })
  const [follow, setFollow] = useState(null)
  const [selected, setSelected] = useState(null) // rover whose sensor panel is open
  const [overview, setOverview] = useState(0) // bump to glide the camera back to the overview
  const [focus, setFocus] = useState(null) // { x, y, key, hazard }: a clicked event's location
  const roverMeta = useSim((s) => s.roverMeta)
  const followPaused = useApp((a) => a.followPaused)
  const modal = useApp((a) => a.modal)
  const selectedRef = useRef(selected)
  selectedRef.current = selected
  const followRef = useRef(follow)
  followRef.current = follow

  // Panning away pauses the follow (CameraRig sets appStore.followPaused); this resumes.
  const resumeFollow = useCallback(() => appStore.patch({ followPaused: false }), [])

  const goOverview = useCallback(() => {
    setFollow(null)
    setSelected(null)
    appStore.patch({ followPaused: false })
    setOverview((n) => n + 1)
  }, [])

  // Selecting a rover (card, 3D model or label, Follow button, number key) opens its
  // sensor panel and follows it. Selecting it again closes the panel and returns
  // (while the follow is paused, it resumes instead).
  const select = useCallback(
    (id) => {
      if (id && selectedRef.current === id) {
        if (appStore.get().followPaused && followRef.current === id) resumeFollow()
        else goOverview()
        return
      }
      setSelected(id)
      setFollow(id)
      appStore.patch({ followPaused: false })
    },
    [goOverview, resumeFollow],
  )
  // Follow buttons, 3D clicks and number keys only ever open (never toggle closed); on
  // the paused rover they resume following it.
  const show = useCallback((id) => {
    setSelected(id)
    setFollow(id)
    appStore.patch({ followPaused: false })
  }, [])
  const closePanel = useCallback(() => setSelected(null), [])

  // An event with a location flies the camera there. While following, that pauses the
  // follow (the rover stays selected) instead of dropping it.
  const focusEvent = useCallback((e) => {
    if (!e?.at) return
    if (followRef.current) appStore.patch({ followPaused: true })
    setFocus({ x: e.at[0], y: e.at[1], key: `${e.key}:${performance.now()}`, hazard: e.level === 'hazard' })
  }, [])

  // A new world: forget the picked sample.
  const gen = useSim((s) => s.gen)
  useEffect(() => {
    if (appStore.get().pickedTarget) appStore.patch({ pickedTarget: null })
  }, [gen])

  useEffect(() => startConnection(), [])

  // After a reset the followed or selected rover may no longer exist.
  useEffect(() => {
    if (follow && !roverMeta.some((m) => m.id === follow)) {
      setFollow(null)
      appStore.patch({ followPaused: false })
    }
    if (selected && !roverMeta.some((m) => m.id === selected)) setSelected(null)
  }, [roverMeta, follow, selected])

  // Keyboard shortcuts for the demo: R F P S L toggle layers, 1-9 select a rover,
  // Esc closes the sensor panel (then returns to the overview), 0 for the overview.
  // F resumes a paused follow (otherwise it toggles Frontiers). A large viewer handles
  // its own keys while it is open.
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return
      if (appStore.get().modal) return
      if (e.target instanceof Element && e.target.closest('input, textarea, select, [contenteditable="true"]')) return
      const k = e.key.toLowerCase()
      if (k === 'f' && followRef.current && appStore.get().followPaused) {
        resumeFollow()
      } else if (TOGGLE_KEYS[k]) {
        const key = TOGGLE_KEYS[k]
        setView((v) => ({ ...v, [key]: !v[key] }))
      } else if (k === 'escape') {
        // Esc first cancels hazard placement, then closes the panel, then the overview.
        if (appStore.get().placing) stopPlacing()
        else if (appStore.get().pickedTarget) appStore.patch({ pickedTarget: null })
        else if (selectedRef.current) setSelected(null)
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
  }, [goOverview, show, resumeFollow])

  const paused = !!follow && followPaused
  return (
    <div className="app">
      <main className="view" data-follow-paused={paused || undefined}>
        {/* Pixel ratio capped at 1.5: crisp with MSAA, and spares laptop GPUs on retina screens. */}
        <Canvas flat linear dpr={[1, 1.5]} camera={{ position: [0, 68, 46], fov: 42, near: 0.5, far: 2000 }} gl={{ antialias: true, powerPreference: 'high-performance' }}>
          <Scene view={view} follow={follow} overview={overview} selected={selected} onSelect={show} focus={focus} />
        </Canvas>
        {/* v7: the fade between the sector and the planet */}
        <GlobeFade />
        <DustHaze />
        <div className="vignette" aria-hidden="true" />
        <ViewControls view={view} setView={setView} follow={follow} onSelect={show} onOverview={goOverview} />
        <div className="hud-top">
          <HazardPanel />
          <PlacementHint />
          <RequestCards onSelect={show} />
          <div className="banners">
            <ConnBanner />
            <DustBanner />
          </div>
        </div>
        {selected ? <SensorPanel roverId={selected} onClose={closePanel} /> : <Legend slopeColors={view.slope} frontiers={view.frontiers} life={view.life} />}
        {paused && <FollowPill roverId={follow} onResume={resumeFollow} />}
        <Waiting />
      </main>
      <Sidebar follow={follow} selected={selected} onSelect={select} onFocusEvent={focusEvent} />
      {modal?.kind === 'camera' && <CameraViewer key={`cam:${modal.rover}`} roverId={modal.rover} view={modal.view} />}
      {modal?.kind === 'sensors' && <SensorDetail key={`sd:${modal.key}`} roverId={modal.rover} focus={modal.focus} />}
    </div>
  )
}
