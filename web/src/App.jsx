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
import { SetupScreen } from './components/SetupScreen.jsx'
import { startConnection, wsUrl } from './net/connection.js'
import { appStore, simStore, useApp, useSim } from './state/store.js'
import { stopPlacing } from './lib/placement.js'
// v7 planet view (src/globe)
import { GlobeFade } from './globe/GlobeFade.jsx'
import { GlobeLegend } from './globe/GlobeLegend.jsx'
import { globeShown, leaveGlobe, useViewMode } from './globe/viewMode.js'

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
  const globe = useViewMode(globeShown) // v7: the planet is drawn instead of the sector
  const placing = useApp((a) => !!a.placing) // a hazard is being placed: the question cards fold to strips
  const selectedRef = useRef(selected)
  selectedRef.current = selected
  const followRef = useRef(follow)
  followRef.current = follow

  // Panning away pauses the follow (CameraRig sets appStore.followPaused); this resumes.
  const resumeFollow = useCallback(() => appStore.patch({ followPaused: false }), [])

  // A folded question strip was opened: stop placing a hazard, or fly back down from the planet.
  const expandRequests = useCallback(() => {
    stopPlacing()
    leaveGlobe()
  }, [])

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
      // v7: picked while on the planet: fly back down and open it (never a toggle-close).
      const fromGlobe = id ? leaveGlobe() : false
      if (id && selectedRef.current === id && !fromGlobe) {
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
    leaveGlobe() // v7: rovers live on the sector; from the planet this flies back down first
    setSelected(id)
    setFollow(id)
    appStore.patch({ followPaused: false })
  }, [])
  const closePanel = useCallback(() => setSelected(null), [])

  // An event with a location flies the camera there. While following, that pauses the
  // follow (the rover stays selected) instead of dropping it.
  const focusEvent = useCallback((e) => {
    if (!e?.at) return
    leaveGlobe() // v7: events happen on the sector
    if (followRef.current) appStore.patch({ followPaused: true })
    setFocus({ x: e.at[0], y: e.at[1], key: `${e.key}:${performance.now()}`, hazard: e.level === 'hazard' })
  }, [])

  // A new world: forget the picked sample.
  const gen = useSim((s) => s.gen)
  useEffect(() => {
    if (appStore.get().pickedTarget) appStore.patch({ pickedTarget: null })
  }, [gen])

  useEffect(() => startConnection(), [])

  // v7: while the sim waits for a launch, the setup screen is open and cannot be dismissed. The launched
  // world's init (phase "running") closes it. An older sim never sends phase "setup". The phase decides,
  // not how the screen was opened: whatever replaces or closes the forced screen during setup (another
  // viewer, New mission, Esc) is put back at once.
  const phase = useSim((s) => s.phase)
  const setupForced = phase === 'setup'
  useEffect(() => {
    const forcedOpen = modal?.kind === 'setup' && !!modal.forced
    if (phase === 'setup' && !forcedOpen) appStore.patch({ modal: { kind: 'setup', forced: true } })
    else if (phase !== 'setup' && forcedOpen) appStore.patch({ modal: null })
  }, [phase, modal])

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
  // Under a viewer or the setup screen the dashboard is inert: Tab cannot reach it (nor Reset sim or New
  // mission behind the forced setup screen), and screen readers skip it.
  const behind = !!modal || setupForced
  return (
    <div className="app">
      <main className="view" data-follow-paused={paused || undefined} inert={behind}>
        {/* Pixel ratio capped at 1.5: crisp with MSAA, and spares laptop GPUs on retina screens. */}
        <Canvas flat linear dpr={[1, 1.5]} camera={{ position: [0, 68, 46], fov: 42, near: 0.5, far: 2000 }} gl={{ antialias: true, powerPreference: 'high-performance' }}>
          <Scene view={view} follow={follow} overview={overview} selected={selected} onSelect={show} focus={focus} />
        </Canvas>
        {/* v7: the fade between the sector and the planet; no dust veil over the planet */}
        <GlobeFade />
        {!globe && <DustHaze />}
        <div className="vignette" aria-hidden="true" />
        <ViewControls view={view} setView={setView} follow={follow} onSelect={show} onOverview={goOverview} />
        <div className="hud-top">
          <HazardPanel />
          <PlacementHint />
          <RequestCards onSelect={show} collapsed={globe || placing} onExpand={expandRequests} />
          <div className="banners">
            <ConnBanner />
            <DustBanner />
          </div>
        </div>
        {/* v7: on the planet its own legend shows; an open sensor panel stays mounted but
            hidden, so it comes back as it was; no follow pill over the planet */}
        {globe && <GlobeLegend reality={view.reality} />}
        {selected ? (
          <div className="globe-keep" hidden={globe}>
            <SensorPanel roverId={selected} onClose={closePanel} />
          </div>
        ) : (
          !globe && <Legend slopeColors={view.slope} frontiers={view.frontiers} life={view.life} reality={view.reality} />
        )}
        {paused && !globe && <FollowPill roverId={follow} onResume={resumeFollow} />}
        <Waiting />
      </main>
      <Sidebar follow={follow} selected={selected} onSelect={select} onFocusEvent={focusEvent} inert={behind} />
      {!setupForced && modal?.kind === 'camera' && <CameraViewer key={`cam:${modal.rover}`} roverId={modal.rover} view={modal.view} />}
      {!setupForced && modal?.kind === 'sensors' && <SensorDetail key={`sd:${modal.key}`} roverId={modal.rover} focus={modal.focus} />}
      {(setupForced || modal?.kind === 'setup') && <SetupScreen key={setupForced ? 'setup-forced' : 'setup'} forced={setupForced} />}
    </div>
  )
}
