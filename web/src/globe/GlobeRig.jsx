import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { reducedMotion } from '../scene/sceneConstants.js'
import { applyShift, freeRect } from '../scene/CameraRig.jsx'
import { GLOBE_FOV, MAX_DIST, MIN_DIST, START_ALT, homeDir, homeDistance, idleSway, lerpPose, smoother } from './globeMath.js'
import { progress } from './viewMachine.js'
import { settleControls } from './planeTransition.js'
import { viewMode } from './viewMode.js'
import { globeRefs } from './globeDebug.js'

const SPIN_AFTER = 8 // seconds without input before the planet starts to turn
const SWAY_PERIOD = 120 // seconds for one idle sway: out to IDLE_SWAY (globeMath.js) and back
const HOME_GLIDE = 1.2 // seconds, the 0 key and a new world

// The planet camera. It orbits the centre with north up, framed in the part of the view
// the HUD leaves free (the sector camera's view offset). Arriving from the sector it
// starts low over the site, where the sector camera left off looking down, and eases out
// to the home view; leaving, it dives back to that spot before the fade. Left alone for a
// while the planet sways slowly about 20 degrees towards the sun and back (its sun turns
// with it, so the site stays lit, and the sway never turns the site away); a drag pauses
// that, 0 glides back home, and so does a new world (Reset sim on the planet). With
// reduced motion every move is instant and nothing turns by itself.
export function GlobeRig({ look, spin }) {
  const camera = useThree((s) => s.camera)
  const size = useThree((s) => s.size)
  const controls = useRef(null)
  const st = useRef({ shift: null, phaseKey: '', phase: '', drawn: false, from: null, home: viewMode.get().home, seed: null, glide: null, lastInput: 0, dragging: false, sway: 0 })

  useEffect(() => {
    globeRefs.rig = st.current
    return () => {
      globeRefs.rig = null
    }
  }, [])

  const freeFrac = () => {
    const f = freeRect(size)
    return (f.y1 - f.y0) / Math.max(1, size.height)
  }
  const homePose = () => ({ dir: homeDir(look.frame), dist: homeDistance(GLOBE_FOV, freeFrac()), spin: 0 })
  const closePose = () => ({ dir: look.frame.up, dist: 1 + START_ALT, spin: 0 })
  const pose = () => {
    const d = camera.position.length() || 1
    return { dir: [camera.position.x / d, camera.position.y / d, camera.position.z / d], dist: d, spin: spin.current ? spin.current.rotation.y : 0 }
  }
  const put = (p) => {
    camera.position.set(p.dir[0] * p.dist, p.dir[1] * p.dist, p.dir[2] * p.dist)
    camera.up.set(0, 1, 0)
    camera.lookAt(0, 0, 0)
    if (spin.current) spin.current.rotation.y = p.spin
    // Every scripted move ends at spin 0 (the home and close poses), so the next idle sway
    // starts from its beginning, at rest.
    st.current.sway = 0
  }

  useFrame((_, dt) => {
    const c = controls.current
    const v = viewMode.get()
    const s = st.current
    const drawn = v.render === 'globe'
    if (c) c.enabled = drawn && v.view === 'globe' && v.phase === 'idle' && !s.glide
    if (!drawn) {
      s.drawn = false
      return
    }
    const aspect = size.width / Math.max(1, size.height)
    if (camera.aspect !== aspect) {
      camera.aspect = aspect
      camera.updateProjectionMatrix()
    }
    applyShift(s, camera, size, freeRect(size), dt)
    const now = performance.now() / 1000
    const key = `${v.seq}:${v.phase}`
    if (key !== s.phaseKey || !s.drawn) {
      // A new phase, or the first frame drawn: note where its motion starts.
      const arriving = !s.drawn
      const landed = s.phase === 'in' && v.phase === 'idle'
      s.phaseKey = key
      s.phase = v.phase
      s.drawn = true
      s.home = v.home
      s.seed = look.seed // flights aim at this world's poses every frame already
      s.glide = null
      s.lastInput = now
      if (c) settleControls(c)
      if (arriving) put(v.phase === 'in' ? closePose() : homePose())
      // Frames do not line up with the end of a flight, so its last frame can stop a little
      // short: land exactly on the home view.
      else if (landed) put(homePose())
      s.from = pose()
    }
    // While drawn, 'in' always heads for the planet and 'out' and 'fade' for the sector.
    if (v.phase === 'in') put(lerpPose(s.from, homePose(), smoother(progress(v, now))))
    else if (v.phase === 'out') put(lerpPose(s.from, closePose(), smoother(progress(v, now))))
    else if (v.phase === 'fade') put(closePose())
    else idle(c, s, now, dt, v)
  })

  function idle(c, s, now, dt, v) {
    // The 0 key, or a new world (Reset sim stays in the controls on the planet, and the new
    // seed moves the site): glide to the home view of the world that is up now.
    if (v.home !== s.home || look.seed !== s.seed) {
      s.home = v.home
      s.seed = look.seed
      if (c) settleControls(c)
      if (reducedMotion) put(homePose())
      else s.glide = { from: pose(), t0: now }
    }
    if (s.glide) {
      const t = (now - s.glide.t0) / HOME_GLIDE
      put(lerpPose(s.glide.from, homePose(), smoother(t)))
      if (t >= 1) s.glide = null
      s.lastInput = now
      return
    }
    if (reducedMotion || s.dragging || now - s.lastInput < SPIN_AFTER || !spin.current) return
    // A slow sway out and back (idleSway), not an endless turn, so the site, its card and the
    // day side stay in view however long the planet is left. A drag pauses it where it is,
    // and it carries on from there. The step is capped at 0.25 s, so a long frame (or a tab
    // coming back) cannot jump it far, while a slow software renderer still turns it.
    s.sway = (s.sway + ((Math.PI * 2) / SWAY_PERIOD) * Math.min(dt, 0.25)) % (Math.PI * 2)
    spin.current.rotation.y = idleSway(s.sway)
  }

  return (
    <OrbitControls
      ref={controls}
      enablePan={false}
      enableDamping
      dampingFactor={0.07}
      rotateSpeed={0.45}
      zoomSpeed={0.7}
      minDistance={MIN_DIST}
      maxDistance={MAX_DIST}
      minPolarAngle={0.25}
      maxPolarAngle={Math.PI - 0.25}
      onStart={() => {
        const s = st.current
        s.dragging = true
        s.lastInput = performance.now() / 1000
      }}
      onEnd={() => {
        const s = st.current
        s.dragging = false
        s.lastInput = performance.now() / 1000
      }}
    />
  )
}
