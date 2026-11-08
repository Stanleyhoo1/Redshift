import { useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore } from '../state/store.js'
import { groundAt } from './ground.js'
import { Terrain } from './Terrain.jsx'
import { Ground } from './Ground.jsx'
import { Lander } from './Lander.jsx'
import { Rovers } from './Rovers.jsx'
import { Lidar } from './Lidar.jsx'
import { Targets } from './Targets.jsx'
import { Frontiers } from './Frontiers.jsx'
import { Reality } from './Reality.jsx'
import { HazardVisuals } from './HazardVisuals.jsx'
import { DustStorm } from './DustStorm.jsx'
import { FocusMarker } from './FocusMarker.jsx'
import { Water } from './Water.jsx'
import { Spires, fallDebug } from './Spires.jsx'
import { IceSheets } from './IceSheets.jsx'
import { Flora } from './Flora.jsx'
import { GroundDetail } from './GroundDetail.jsx'
import { Life } from './Life.jsx'
import { Placement } from './Placement.jsx'
import { PlaceCandidates } from './PlaceCandidates.jsx'
import { CameraRig } from './CameraRig.jsx'
import { Atmosphere } from './Atmosphere.jsx'
import { Gravel } from './Gravel.jsx'
import { cameraDebug } from './debugCamera.js'
import { GlobeLayer } from '../globe/GlobeLayer.jsx'
import { globeDebug } from '../globe/globeDebug.js'
import { viewMode } from '../globe/viewMode.js'

// Test aid, only with ?debug in the URL: lets scripts find where a world point is on
// screen and read a slim copy of the sim state (see scripts/v4-shots.mjs).
const DEBUG = typeof location !== 'undefined' && /[?&]debug\b/.test(location.search)

function DebugHook() {
  const camera = useThree((s) => s.camera)
  const gl = useThree((s) => s.gl)
  const get = useThree((s) => s.get)
  useEffect(() => {
    window.__redshift = {
      ...cameraDebug(camera, () => get().controls),
      ...globeDebug(),
      screenOf(x, y, lift = 0) {
        const v = new THREE.Vector3(x, groundAt(x, y) + lift, -y).project(camera)
        const r = gl.domElement.getBoundingClientRect()
        return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height, visible: v.z < 1 }
      },
      // The measured slope (degrees) at (x, y), or null where the fleet knows nothing.
      slopeAt(x, y) {
        const T = simStore.get().terrain
        const i = Math.round((x - T.ox) / T.res)
        const j = Math.round((y - T.oy) / T.res)
        if (i < 0 || j < 0 || i >= T.nx || j >= T.ny || !T.known[j * T.nx + i]) return null
        return T.slope[j * T.nx + i]
      },
      // Hold every spire fall this client saw at t seconds after it began (null: let go).
      holdFalls(t) {
        fallDebug.hold = Number.isFinite(t) ? t : null
      },
      state() {
        const s = simStore.get()
        const T = s.terrain
        const known = (x, y) => {
          const i = Math.round((x - T.ox) / T.res)
          const j = Math.round((y - T.oy) / T.res)
          return i >= 0 && j >= 0 && i < T.nx && j < T.ny && !!T.known[j * T.nx + i]
        }
        return {
          t: s.t,
          rovers: s.rovers,
          features: s.features,
          spiresKnown: s.features.spires.filter((p) => known(p.x, p.y)).map((p) => p.id),
          targets: s.targets,
          tracks: s.tracks,
          lifeforms: s.lifeforms,
          fleet: s.fleet,
          hazards: s.hazards,
          requests: s.requests,
          lander: s.lander,
          phase: s.phase,
          config: s.config,
          world: s.world,
          gen: s.gen,
          roverMeta: s.roverMeta,
          view: viewMode.get().view,
          viewRender: viewMode.get().render,
          viewPhase: viewMode.get().phase,
        }
      },
    }
    return () => {
      fallDebug.hold = null
      delete window.__redshift
    }
  }, [camera, gl, get])
  return null
}

export function Scene({ view, follow, overview, selected, onSelect, focus }) {
  return (
    <>
      <Atmosphere />
      <DustStorm />
      <Ground />
      <Terrain slopeColors={view.slope} />
      <Gravel />
      <Water reality={view.reality} />
      <HazardVisuals reality={view.reality} />
      <Spires reality={view.reality} />
      <IceSheets reality={view.reality} />
      <Flora reality={view.reality} />
      <GroundDetail reality={view.reality} />
      <Lander />
      {view.frontiers && <Frontiers />}
      <Targets />
      <Rovers showPaths={view.paths} selected={selected} onSelect={onSelect} />
      {view.lidar && <Lidar />}
      {view.life && <Life reality={view.reality} />}
      {view.reality && <Reality />}
      <FocusMarker focus={focus} />
      <PlaceCandidates />
      <Placement reality={view.reality} />
      <CameraRig follow={follow} overview={overview} focus={focus} />
      <GlobeLayer onSelect={onSelect} />
      {DEBUG && <DebugHook />}
    </>
  )
}
