import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { appStore, simStore, useSim, roverPoses, hudLayout } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { hasPlaces } from '../lib/answers.js'
import { groundAt } from './ground.js'
import { viewMode } from '../globe/viewMode.js'

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
export function freeRect(size) {
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

// ---------------------------------------------------------------- v6: fit place candidates

const FIT_QUIET_MS = 1500 // wait this long after the user last moved the camera
const FIT_GIVE_UP_MS = 9000 // and drop the fit if they keep moving it this long
const _f = new THREE.Vector3()
const _r = new THREE.Vector3()
const _u = new THREE.Vector3()

// Where a fitted view may put things: the free rect, minus the request card (top right),
// either below it or beside it, whichever leaves more room. Pixels, inset by a margin.
function fitRect(size) {
  const W = size.width
  const H = size.height
  const f = freeRect(size)
  let { x0, x1, y0, y1 } = f
  const rq = hudLayout.requests
  if (rq && rq.right > x0 && rq.left < x1 && rq.bottom > y0) {
    const below = { x0, x1, y0: Math.max(y0, rq.bottom + 12), y1 }
    const beside = { x0, x1: Math.min(x1, rq.left - 12), y0, y1 }
    const area = (r) => Math.max(0, r.x1 - r.x0) * Math.max(0, r.y1 - r.y0)
    const pick = area(below) >= area(beside) ? below : beside
    if (pick.x1 - pick.x0 > W * 0.25 && pick.y1 - pick.y0 > H * 0.25) ({ x0, x1, y0, y1 } = pick)
  }
  const m = Math.min(36, W * 0.03)
  return { x0: x0 + m, x1: x1 - m, y0: y0 + m, y1: y1 - m }
}

// Project a world point (x east, y north, h up) to view pixels; false when behind.
function toPx(cam, size, x, y, h, out) {
  _p.set(x, h, -y).project(cam)
  out.x = ((_p.x + 1) / 2) * size.width
  out.y = ((1 - _p.y) / 2) * size.height
  return _p.z < 1 && _p.z > -1
}

function allInView(cam, size, pts, rect) {
  const o = { x: 0, y: 0 }
  for (const [x, y, h] of pts) {
    if (!toPx(cam, size, x, y, h, o)) return false
    if (o.x < rect.x0 || o.x > rect.x1 || o.y < rect.y0 || o.y > rect.y1) return false
  }
  return true
}

// A camera pose that keeps the current heading (elevation kept between about 40 and 70
// degrees so hills do not hide anything) and shows every point inside `rect`.
function fitPose(camera, target0, size, pts, rect, maxDist) {
  const cam = camera.clone()
  cam.updateProjectionMatrix()
  const dir = new THREE.Vector3().copy(camera.position).sub(target0)
  if (dir.lengthSq() < 1e-6) dir.set(0, Math.sin(ELEVATION), Math.cos(ELEVATION))
  dir.normalize()
  const ey = THREE.MathUtils.clamp(dir.y, Math.sin(0.7), Math.sin(1.22))
  if (ey !== dir.y) {
    const flat = Math.hypot(dir.x, dir.z) || 1
    const c = Math.cos(Math.asin(ey))
    dir.set(dir.x === 0 && dir.z === 0 ? 0 : (dir.x / flat) * c, ey, dir.x === 0 && dir.z === 0 ? c : (dir.z / flat) * c)
  }
  const target = new THREE.Vector3()
  for (const [x, y, h] of pts) target.add(_p.set(x, h, -y))
  target.multiplyScalar(1 / pts.length)
  const place = (d) => {
    cam.position.copy(target).addScaledVector(dir, d)
    cam.lookAt(target)
    cam.updateMatrixWorld(true)
  }
  const o = { x: 0, y: 0 }
  const fits = () => {
    for (const [x, y, h] of pts) {
      if (!toPx(cam, size, x, y, h, o)) return false
      if (o.x < rect.x0 || o.x > rect.x1 || o.y < rect.y0 || o.y > rect.y1) return false
    }
    return true
  }
  let d = maxDist
  for (let pass = 0; pass < 3; pass++) {
    let lo = 6
    let hi = maxDist
    for (let it = 0; it < 24; it++) {
      const mid = (lo + hi) / 2
      place(mid)
      if (fits()) hi = mid
      else lo = mid
    }
    d = hi
    place(d)
    // Centre the points in the rect: slide the target along the view plane.
    let px0 = Infinity
    let px1 = -Infinity
    let py0 = Infinity
    let py1 = -Infinity
    for (const [x, y, h] of pts) {
      toPx(cam, size, x, y, h, o)
      px0 = Math.min(px0, o.x)
      px1 = Math.max(px1, o.x)
      py0 = Math.min(py0, o.y)
      py1 = Math.max(py1, o.y)
    }
    const dx = (px0 + px1) / 2 - (rect.x0 + rect.x1) / 2
    const dy = (py0 + py1) / 2 - (rect.y0 + rect.y1) / 2
    if (Math.abs(dx) < 2 && Math.abs(dy) < 2) break
    const mPerPx = (2 * d * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2))) / Math.max(1, size.height)
    _r.setFromMatrixColumn(cam.matrixWorld, 0)
    _u.setFromMatrixColumn(cam.matrixWorld, 1)
    target.addScaledVector(_r, dx * mPerPx).addScaledVector(_u, -dy * mPerPx)
  }
  place(d)
  return { target, position: cam.position.clone() }
}

// Keeps the free part of the view centred on screen with a view offset, easing when a
// panel opens or closes so the picture glides instead of jumping.
export function applyShift(s, camera, size, free, dt) {
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
//   follow - glides along with one rover; you can still orbit and zoom around it
//   fly    - glides to a spot (an event with a location), then hands over as "user"
// Version 6: when a request with place candidates opens, the camera glides once to show
// the asking rover and every candidate (only if they are not in view already, and never
// while the user is moving the camera). A followed rover's follow pauses for it ("Back
// to" shows). When the request closes and nobody touched the camera meanwhile, it goes
// back to the overview or the follow it left.
// Panning while following (anything that moves the orbit target), or clicking an event
// that flies elsewhere, pauses the follow: the camera goes free where it is, the rover
// stays selected, and appStore.followPaused tells App to show "Back to". Clearing that
// flag glides back to the rover from wherever the camera is.
export function CameraRig({ follow, overview = 0, focus = null }) {
  const map = useSim((s) => s.map)
  const gen = useSim((s) => s.gen)
  const camera = useThree((s) => s.camera)
  const size = useThree((s) => s.size)
  const dom = useThree((s) => s.gl.domElement)
  const controls = useRef()
  const st = useRef({ mode: 'auto', snapped: false, key: '', checkedAt: 0, goal: null, followId: null, zooming: false, overview, gen, focusKey: focus?.key ?? null, fly: null, shift: null, last: new THREE.Vector3(), hasLast: false, panned: 0, panIntent: false, dragging: false, lastGesture: -1e9, fit: null, fitCarry: null, fitPending: null, fitDone: new Set() })
  // v6: the newest open request with place candidates; its rover and candidates are
  // framed once when it opens (see placeFit).
  const requests = useSim((s) => s.requests)
  const placeReq = useMemo(() => {
    for (let k = requests.length - 1; k >= 0; k--) if (hasPlaces(requests[k])) return requests[k]
    return null
  }, [requests])
  const placeRef = useRef(placeReq)
  placeRef.current = placeReq

  // Which gesture is under way, the way OrbitControls maps them: right button (or a
  // modifier with the left one) or two fingers pan. Only target motion from a pan
  // gesture (including its damping tail) can detach the follow; a rotate or zoom
  // never does, even while an earlier pan is still coasting.
  useEffect(() => {
    const touches = new Set()
    const down = (e) => {
      const s = st.current
      if (e.pointerType === 'touch') {
        touches.add(e.pointerId)
        s.panIntent = touches.size >= 2
      } else s.panIntent = e.button === 2 || (e.button === 0 && (e.ctrlKey || e.metaKey || e.shiftKey))
    }
    const up = (e) => touches.delete(e.pointerId)
    dom.addEventListener('pointerdown', down)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', up)
    return () => {
      dom.removeEventListener('pointerdown', down)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', up)
    }
  }, [dom])

  useFrame((_, dt) => {
    const c = controls.current
    if (!c || !map) return
    const s = st.current
    // v7 planet view (src/globe): the sector camera only moves while the sector is wanted
    // and drawn; meanwhile its controls ignore the mouse.
    const v = viewMode.get()
    c.enabled = v.view === 'plane' && v.render === 'plane'
    if (v.view === 'globe' || v.render === 'globe') {
      s.hasLast = false
      return
    }
    // Anything that moved the orbit target since our last frame was the user panning
    // (OrbitControls updates before this frame; rotate and zoom leave the target alone).
    const moved = s.hasLast ? c.target.distanceTo(s.last) : 0
    step(c, s, dt, moved)
    s.last.copy(c.target)
    s.hasLast = true
  })

  function step(c, s, dt, moved) {
    const free = freeRect(size)
    applyShift(s, camera, size, free, dt)
    const wantOverview = overview !== s.overview
    const newRun = gen !== s.gen
    s.overview = overview
    s.gen = gen
    const paused = !!follow && appStore.get().followPaused
    if (newRun) {
      s.fit = null
      s.fitCarry = null
      s.fitPending = null
      s.fitDone.clear()
    }
    if (s.fit && (follow !== s.followId || wantOverview)) s.fit.touched = true
    if (follow !== s.followId) {
      s.followId = follow
      s.zooming = !!follow && !paused
      s.mode = follow ? (paused ? 'user' : 'follow') : 'auto'
      s.panned = 0
      s.panIntent = false
    }
    if (!follow && (wantOverview || newRun)) s.mode = 'auto'
    if (follow) {
      if (paused && s.mode === 'follow') {
        // Paused from outside (an event click): stay put.
        s.mode = 'user'
        s.zooming = false
      } else if (!paused && (s.mode === 'user' || s.mode === 'fly' || s.mode === 'auto')) {
        // "Back to": glide to the rover from here, keeping the angle. Only a camera
        // left far out (or very close) also eases back to the usual distance.
        const len = camera.position.distanceTo(c.target)
        s.mode = 'follow'
        s.panned = 0
        s.panIntent = false
        s.zooming = len > FOLLOW_DISTANCE * 2.2 || len < 8
        // Drop any pan still coasting (damping), so it does not drift against the glide.
        if (c._panOffset?.isVector3) c._panOffset.set(0, 0, 0)
      } else if (s.mode === 'follow' && s.panIntent && moved > 1e-4) {
        // A pan while following: past a small dead zone it detaches, right where the
        // user dragged to (no pull back). Damping keeps the motion smooth.
        s.panned += moved
        const len = camera.position.distanceTo(c.target)
        if (s.panned > Math.max(0.25, len * 0.01)) {
          s.mode = 'user'
          s.zooming = false
          appStore.patch({ followPaused: true })
          return
        }
      }
    }

    // A clicked event with a location: fly there, keeping the current viewing angle.
    if (focus && focus.key !== s.focusKey) {
      s.focusKey = focus.key
      if (!newRun && (!follow || paused)) {
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
        s.fly = { target, position: target.clone().addScaledVector(_o, FLY_DISTANCE), rate: 2.6, kind: 'event' }
        s.mode = 'fly'
        if (s.fit) s.fit.touched = true
      }
    }

    placeFit(c, s, paused)

    if (s.mode === 'fly') {
      const a = 1 - Math.exp(-Math.min(dt, 0.1) * (s.fly.rate || 2.6))
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
  }

  // v6: frame the asking rover and its place candidates once per request.
  function placeFit(c, s, paused) {
    const q = placeRef.current
    const now = performance.now()
    if (s.fit && (!q || q.id !== s.fit.id)) {
      const f = s.fit
      s.fit = null
      if (!f.touched) {
        // Another request is about to be framed: it inherits where to go back to.
        if (q && !s.fitDone.has(q.id)) s.fitCarry = f
        else if (f.back === 'auto' && !follow && (s.mode === 'user' || s.mode === 'fly')) s.mode = 'auto'
        else if (f.back === 'follow' && follow && follow === f.followId && paused) appStore.patch({ followPaused: false })
      }
    }
    if (!q || s.fitDone.has(q.id)) return
    if (!s.fitPending || s.fitPending.id !== q.id) s.fitPending = { id: q.id, since: now }
    const busy = s.dragging || now - s.lastGesture < FIT_QUIET_MS || (s.mode === 'fly' && s.fly?.kind !== 'fit')
    if (busy) {
      if (now - s.fitPending.since > FIT_GIVE_UP_MS) {
        s.fitDone.add(q.id)
        s.fitPending = null
        s.fitCarry = null
      }
      return
    }
    s.fitDone.add(q.id)
    s.fitPending = null
    const carry = s.fitCarry
    s.fitCarry = null
    // Rover and candidates, with room around each for the ring and the badge above it.
    const key = []
    const pad = []
    const rp = roverPoses.get(q.rover)
    if (rp) {
      key.push([rp.x, -rp.z, rp.y])
      for (const [dx, dy] of [[-2, -2], [2, 2]]) pad.push([rp.x + dx, -rp.z + dy, rp.y])
      pad.push([rp.x, -rp.z, rp.y + 3])
    }
    for (const o of q.options) {
      if (!Number.isFinite(o.x) || !Number.isFinite(o.y)) continue
      const h = groundAt(o.x, o.y, (map.minH + map.maxH) / 2)
      key.push([o.x, o.y, h])
      for (const [dx, dy] of [[-3.5, -3.5], [3.5, -3.5], [-3.5, 3.5], [3.5, 3.5]]) pad.push([o.x + dx, o.y + dy, h])
      pad.push([o.x, o.y, h + 6])
    }
    if (!key.length) return
    const rect = fitRect(size)
    const back = carry ? carry.back : s.mode === 'auto' ? 'auto' : s.mode === 'follow' && !paused ? 'follow' : null
    s.fit = { id: q.id, back, followId: carry ? carry.followId : follow, touched: false }
    camera.updateMatrixWorld()
    if (allInView(camera, size, key, rect)) return
    const span = Math.max(map.width, map.depth)
    const pose = fitPose(camera, c.target, size, [...key, ...pad], rect, Math.max(240, span * 3) * 0.95)
    if (s.mode === 'follow') {
      appStore.patch({ followPaused: true })
      s.zooming = false
    }
    if (c._panOffset?.isVector3) c._panOffset.set(0, 0, 0)
    s.fly = { target: pose.target, position: pose.position, rate: 1.6, kind: 'fit' }
    s.mode = 'fly'
  }

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
        // While following, orbit and zoom keep following; a pan detaches (see above).
        const s = st.current
        if (s.mode === 'auto' || s.mode === 'fly') s.mode = 'user'
        s.zooming = false
        s.panned = 0
        s.dragging = true
        s.lastGesture = performance.now()
        if (s.fit) s.fit.touched = true
      }}
      onEnd={() => {
        const s = st.current
        s.dragging = false
        s.lastGesture = performance.now()
      }}
    />
  )
}
