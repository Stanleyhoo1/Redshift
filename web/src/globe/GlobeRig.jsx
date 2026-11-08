import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { applyShift, freeRect } from '../scene/CameraRig.jsx'
import { GLOBE_FOV, MAX_DIST, MIN_DIST, homeDir, homeDistance } from './globeMath.js'
import { settleControls } from './planeTransition.js'
import { viewMode } from './viewMode.js'
import { globeRefs } from './globeDebug.js'

// The planet camera: orbits the centre with north up, framed in the part of the view the
// HUD leaves free (the same view offset as the sector camera). First version: it shows
// the home view when the planet is first drawn and again on the 0 key.
export function GlobeRig({ look, spin }) {
  const camera = useThree((s) => s.camera)
  const size = useThree((s) => s.size)
  const controls = useRef(null)
  const st = useRef({ shift: null, drawn: false, home: viewMode.get().home, glide: null, dragging: false, lastInput: 0 })

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
  const put = (p) => {
    camera.position.set(p.dir[0] * p.dist, p.dir[1] * p.dist, p.dir[2] * p.dist)
    camera.up.set(0, 1, 0)
    camera.lookAt(0, 0, 0)
    if (spin.current) spin.current.rotation.y = p.spin
  }

  useFrame((_, dt) => {
    const c = controls.current
    const v = viewMode.get()
    const s = st.current
    if (c) c.enabled = v.render === 'globe' && v.view === 'globe' && v.phase === 'idle'
    if (v.render !== 'globe') {
      s.drawn = false
      return
    }
    const aspect = size.width / Math.max(1, size.height)
    if (camera.aspect !== aspect) {
      camera.aspect = aspect
      camera.updateProjectionMatrix()
    }
    applyShift(s, camera, size, freeRect(size), dt)
    if (!s.drawn || v.home !== s.home) {
      s.drawn = true
      s.home = v.home
      if (c) settleControls(c)
      put(homePose())
    }
  })

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
