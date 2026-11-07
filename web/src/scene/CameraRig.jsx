import { useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { simStore, useSim, roverPoses, hudLayout } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'

const _d = new THREE.Vector3()
const _o = new THREE.Vector3()
const _p = new THREE.Vector3()
const FOLLOW_DISTANCE = 28
const FLY_DISTANCE = 30
const ELEVATION = 0.98 // overview camera angle above the horizon, radians

// The part of the view the HUD panels leave free, in px: { x0, x1, y0, y1 }.
// The control panel covers the left edge and the hazard bar the top, so the picture is
// shifted to be centred in what is left. With the sensor panel open (a rover selected)
// the free space is the band above it, where the followed rover then sits.
function freeRect(size) {
  const { width: W, height: H } = size
  if (W <= 640) return { x0: 0, x1: W, y0: H * 0.06, y1: H * 0.94 }
  const c = hudLayout.controls
  const hz = hudLayout.hazards
  const sp = hudLayout.sensors
  // Without a measurement (first frames) fall back to the old fixed left shift.
  const x0 = c ? Math.min(W * 0.4, c.right) : 2 * Math.min(124, W * 0.12)
  const top = Math.max(H * 0.06, hz ? hz.bottom + 10 : 0)
  if (sp && sp.top > top + 80) return { x0: x0 + 8, x1: W - 8, y0: top, y1: sp.top - 8 }
  return { x0, x1: W, y0: top, y1: H * 0.94 }
}

// The area the overview keeps in frame: what the fleet knows so far plus a margin,
// never smaller than about 40 % of the map, and never outside it. It grows into the
// full map as exploration goes on.
function overviewBox(map, T) {
  const span = Math.max(map.width, map.depth)
  let x0 = map.ox
  let x1 = map.ox + map.width
  let y0 = map.oy
  let y1 = map.oy + map.depth
  if (T && T.bi1 >= T.bi0) {
    const pad = Math.max(4, span * 0.06)
    const min = Math.min(span, Math.max(32, span * 0.4))
    const grow = (a, b, lo, hi) => {
      a -= pad
      b += pad
      if (b - a < min) {
        const c = (a + b) / 2
        a = c - min / 2
        b = c + min / 2
      }
      if (a < lo) [a, b] = [lo, b + (lo - a)]
      if (b > hi) [a, b] = [a - (b - hi), hi]
      return [Math.max(a, lo), b]
    }
    ;[x0, x1] = grow(T.ox + T.bi0 * T.res, T.ox + T.bi1 * T.res, x0, x1)
    ;[y0, y1] = grow(T.oy + T.bj0 * T.res, T.oy + T.bj1 * T.res, y0, y1)
  }
  return { x0, x1, y0, y1, h: (map.minH + map.maxH) / 2 }
}

// Camera pose that fits a box into the free part of the view. The view offset already
// centres the free rect on screen, so the fit only needs its half extents in NDC.
function overviewPose(box, size, fov, free) {
  const cam = new THREE.PerspectiveCamera(fov, size.width / Math.max(1, size.height), 0.1, 10000)
  const dir = new THREE.Vector3(0, Math.sin(ELEVATION), Math.cos(ELEVATION))
  const target = new THREE.Vector3((box.x0 + box.x1) / 2, box.h, -(box.y0 + box.y1) / 2)
  const corners = []
  for (const x of [box.x0, box.x1]) for (const y of [box.y0, box.y1]) corners.push(new THREE.Vector3(x, box.h, -y))
  const xMax = ((free.x1 - free.x0) / Math.max(1, size.width)) * 0.97
  const yMax = (free.y1 - free.y0) / Math.max(1, size.height)
  const span = Math.max(box.x1 - box.x0, box.y1 - box.y0, 1)
  const place = (d) => {
    cam.position.copy(target).addScaledVector(dir, d)
    cam.lookAt(target)
    cam.updateMatrixWorld()
  }
  const fit = () => {
    let lo = span * 0.2
    let hi = span * 8
    for (let it = 0; it < 28; it++) {
      const d = (lo + hi) / 2
      place(d)
      const ok = corners.every((c) => {
        _p.copy(c).project(cam)
        return _p.z < 1 && Math.abs(_p.x) <= xMax && Math.abs(_p.y) <= yMax
      })
      if (ok) hi = d
      else lo = d
    }
    place(hi)
    return hi
  }
  // Perspective makes the near edge look bigger; nudge the target towards the camera
  // so the box sits in the middle of the screen, then fit again.
  let d = fit()
  let yLo = Infinity
  let yHi = -Infinity
  for (const c of corners) {
    _p.copy(c).project(cam)
    yLo = Math.min(yLo, _p.y)
    yHi = Math.max(yHi, _p.y)
  }
  target.z -= (((yLo + yHi) / 2) * d * Math.tan(THREE.MathUtils.degToRad(fov / 2))) / Math.sin(ELEVATION)
  d = fit()
  return { target, position: target.clone().addScaledVector(dir, d) }
}

// Keeps the free part of the view centred on screen with a view offset, easing when a
// panel opens or closes so the picture glides instead of jumping.
function applyShift(s, camera, size, free, dt) {
  const W = size.width
  const H = size.height
  const wx = (free.x0 + free.x1) / 2 - W / 2
  const wy = (free.y0 + free.y1) / 2 - H / 2
  const resized = !s.shift || s.shift.W !== W || s.shift.H !== H
  if (resized) s.shift = { x: wx, y: wy, W, H }
  else {
    const k = 1 - Math.exp(-Math.min(dt, 0.1) * 5)
    const nx = s.shift.x + (wx - s.shift.x) * k
    const ny = s.shift.y + (wy - s.shift.y) * k
    if (Math.abs(nx - s.shift.x) < 0.05 && Math.abs(ny - s.shift.y) < 0.05) return
    s.shift.x = Math.abs(wx - nx) < 0.3 ? wx : nx
    s.shift.y = Math.abs(wy - ny) < 0.3 ? wy : ny
  }
  if (Math.abs(s.shift.x) < 0.5 && Math.abs(s.shift.y) < 0.5) camera.clearViewOffset()
  else camera.setViewOffset(W, H, -s.shift.x, -s.shift.y, W, H)
}

// Orbit camera with four modes:
//   auto   - overview that keeps the explored area in frame, easing out as it grows
//   user   - someone moved the camera: hands off until "Off" (0 / Esc) or a sim reset
//   follow - glides along with one rover; you can still orbit around it
//   fly    - glides to a spot (an event with a location), then hands over as "user"
export function CameraRig({ follow, overview = 0, focus = null }) {
  const map = useSim((s) => s.map)
  const gen = useSim((s) => s.gen)
  const camera = useThree((s) => s.camera)
  const size = useThree((s) => s.size)
  const controls = useRef()
  const st = useRef({ mode: 'auto', snapped: false, key: '', checkedAt: 0, goal: null, followId: null, zooming: false, overview, gen, focusKey: focus?.key ?? null, fly: null, shift: null })

  useFrame((_, dt) => {
    const c = controls.current
    if (!c || !map) return
    const s = st.current
    const free = freeRect(size)
    applyShift(s, camera, size, free, dt)
    const wantOverview = overview !== s.overview
    const newRun = gen !== s.gen
    s.overview = overview
    s.gen = gen
    if (follow !== s.followId) {
      s.followId = follow
      s.zooming = !!follow
      s.mode = follow ? 'follow' : 'auto'
    }
    if (!follow && (wantOverview || newRun)) s.mode = 'auto'

    // A clicked event with a location: fly there, keeping the current viewing angle.
    if (focus && focus.key !== s.focusKey) {
      s.focusKey = focus.key
      if (!newRun && !follow) {
        const h = sampleGroundNear(simStore.get().terrain, focus.x, focus.y, 8) ?? (map.minH + map.maxH) / 2
        const target = new THREE.Vector3(focus.x, h, -focus.y)
        _o.copy(camera.position).sub(c.target)
        if (_o.lengthSq() < 1e-6) _o.set(0, Math.sin(ELEVATION), Math.cos(ELEVATION))
        // Never flatter than about 35 degrees, so the spot is not hidden behind a hill.
        _o.normalize()
        if (_o.y < 0.57) {
          const flat = Math.hypot(_o.x, _o.z) || 1
          _o.set((_o.x / flat) * 0.82, 0.57, (_o.z / flat) * 0.82)
        }
        s.fly = { target, position: target.clone().addScaledVector(_o, FLY_DISTANCE) }
        s.mode = 'fly'
      }
    }

    if (s.mode === 'fly') {
      const a = 1 - Math.exp(-Math.min(dt, 0.1) * 2.6)
      c.target.lerp(s.fly.target, a)
      camera.position.lerp(s.fly.position, a)
      camera.lookAt(c.target)
      if (camera.position.distanceTo(s.fly.position) < 0.2 && c.target.distanceTo(s.fly.target) < 0.05) s.mode = 'user'
      return
    }

    if (s.mode === 'auto') {
      const T = simStore.get().terrain
      const fr = `${Math.round(free.x0)},${Math.round(free.x1)},${Math.round(free.y0)},${Math.round(free.y1)}`
      const key = `${gen}:${size.width}x${size.height}:${fr}:${T ? `${T.bi0},${T.bi1},${T.bj0},${T.bj1}` : ''}`
      const now = performance.now()
      if (key !== s.key && (!s.goal || newRun || now - s.checkedAt > 250)) {
        s.key = key
        s.checkedAt = now
        s.goal = overviewPose(overviewBox(map, T), size, camera.fov, free)
      }
      if (!s.snapped) {
        s.snapped = true
        c.target.copy(s.goal.target)
        camera.position.copy(s.goal.position)
      } else {
        const k = 1 - Math.exp(-Math.min(dt, 0.1) * 1.8)
        c.target.lerp(s.goal.target, k)
        camera.position.lerp(s.goal.position, k)
      }
      camera.lookAt(c.target)
      return
    }

    if (s.mode !== 'follow') return
    const p = roverPoses.get(follow)
    if (!p) return
    const a = 1 - Math.exp(-Math.min(dt, 0.1) * 3.5)
    _d.copy(p).sub(c.target).multiplyScalar(a)
    c.target.add(_d)
    camera.position.add(_d)
    if (s.zooming) {
      _o.copy(camera.position).sub(c.target)
      const len = _o.length()
      if (Math.abs(len - FOLLOW_DISTANCE) < 0.6) s.zooming = false
      else {
        _o.setLength(len + (FOLLOW_DISTANCE - len) * a)
        camera.position.copy(c.target).add(_o)
      }
    }
  })

  const span = map ? Math.max(map.width, map.depth) : 80
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={1.42}
      minDistance={5}
      maxDistance={Math.max(240, span * 3)}
      zoomSpeed={0.8}
      onStart={() => {
        // Someone grabbed the camera: stop auto framing, a fly-to (or the follow zoom).
        const s = st.current
        if (s.mode === 'auto' || s.mode === 'fly') s.mode = 'user'
        s.zooming = false
      }}
    />
  )
}
