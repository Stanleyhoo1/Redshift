import { useEffect, useMemo } from 'react'
import { createPortal, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { VOID } from '../lib/theme.js'
import { GLOBE_FOV } from './globeMath.js'
import { installViewGuards, viewMode } from './viewMode.js'
import { globeRefs } from './globeDebug.js'
import { GlobeWorld } from './GlobeWorld.jsx'
import { installGlobeKeys } from './globeKeys.js'

const rootOf = (o) => {
  while (o.parent) o = o.parent
  return o
}

// Moves the view switch along once per frame, before any rig or the render reads it.
function ViewClock() {
  useFrame(() => viewMode.tick(), -2)
  return null
}

// Draws the scene that is up. A positive priority takes the render over from r3f.
function RenderSwitch({ scene, camera }) {
  useFrame((state) => {
    if (viewMode.get().render === 'globe') state.gl.render(scene, camera)
    else state.gl.render(state.scene, state.camera)
  }, 1)
  return null
}

// v7 planet view (spec section 5). One canvas: the sector scene stays mounted, and keeps
// following the sim, while the planet scene and its own camera are drawn instead. The
// planet scene is an r3f portal, so its objects get pointer events and its label the page.
export function GlobeLayer({ onSelect, reality = false }) {
  const gl = useThree((s) => s.gl)
  const rootScene = useThree((s) => s.scene)
  const get = useThree((s) => s.get)
  const setEvents = useThree((s) => s.setEvents)
  const scene = useMemo(() => {
    const s = new THREE.Scene()
    s.background = new THREE.Color(VOID)
    return s
  }, [])
  const camera = useMemo(() => new THREE.PerspectiveCamera(GLOBE_FOV, 1, 0.01, 400), [])

  // three.js raycasts hidden objects too: only the scene being drawn takes pointer events.
  useEffect(() => {
    const prev = get().events.filter
    setEvents({
      filter: (hits, state) => {
        const want = viewMode.get().render === 'globe' ? scene : rootScene
        const mine = hits.filter((h) => rootOf(h.object) === want)
        return prev ? prev(mine, state) : mine
      },
    })
    return () => setEvents({ filter: prev })
  }, [get, setEvents, scene, rootScene])

  // G, and 0 on the planet (see globeKeys.js).
  useEffect(() => installGlobeKeys(), [])

  // Place questions and placement started outside App bring the sector back.
  useEffect(() => installViewGuards(), [])

  // Compile the planet's shaders when it is first asked for (during the rise), so the
  // first frame after the fade does not stall. A session that never opens it pays nothing.
  useEffect(() => {
    let done = false
    return viewMode.subscribe(() => {
      if (done || viewMode.get().view !== 'globe') return
      done = true
      // compile() only starts the driver's work (three checks the result on first use);
      // compileAsync() warns where KHR_parallel_shader_compile is missing.
      gl.compile(scene, camera)
    })
  }, [gl, scene, camera])

  useEffect(() => {
    globeRefs.camera = camera
    globeRefs.gl = gl
    return () => {
      globeRefs.camera = null
      globeRefs.gl = null
    }
  }, [camera, gl])

  return (
    <>
      <ViewClock />
      <RenderSwitch scene={scene} camera={camera} />
      {createPortal(<GlobeWorld onSelect={onSelect} reality={reality} />, scene, { camera })}
    </>
  )
}
