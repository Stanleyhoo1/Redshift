import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { nowSeconds, simStore, useSim } from '../state/store.js'
import { focusLeft, focusMapLabel } from '../lib/focus.js'
import { FatPolyline } from './fatLine.js'
import { groundAt } from './ground.js'
import { reducedMotion } from './sceneConstants.js'

// Version 7 rover `focus`: the area a rover concentrates on, as a dashed circle draped
// on the terrain in the rover's colour (a little stronger when the operator asked for
// it, softer when it is the rover's own curiosity), with a small tag on the edge facing
// the camera: "Exploring the pond's shore · 64 s", counting down between ticks. Fades in
// (growing slightly into place) and out. One always-mounted marker per rover: buffers
// are made once, and the frame loop allocates nothing.

const SEGS = 120 // points around the circle
const LIFT = 0.14 // m above the ground
const TAG_OUT = 0.7 // m outside the circle
const TAG_UP = 0.55 // m above the ground
const FADE_IN = 0.6 // s
const FADE_OUT = 0.9 // s
const STYLE = {
  // dashes: how many around the circle; fill: dash share of each period
  operator: { width: 2.3, opacity: 0.88, dashes: 44, fill: 0.62, crawl: 0.35 },
  interest: { width: 1.7, opacity: 0.55, dashes: 72, fill: 0.4, crawl: 0.18 },
}
const TWO_PI = Math.PI * 2
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a))

// Bounding boxes of the water outlines, so the drape can lift the circle onto a lake's
// surface without testing every polygon for every point. Cached per features object.
const boxCache = new WeakMap()
function waterBoxes(water) {
  let boxes = boxCache.get(water)
  if (boxes) return boxes
  boxes = water
    .filter((w) => w.outline.length > 2)
    .map((w) => {
      let x0 = Infinity
      let x1 = -Infinity
      let y0 = Infinity
      let y1 = -Infinity
      for (const [x, y] of w.outline) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
      return { x0, x1, y0, y1, level: w.level, outline: w.outline }
    })
  boxCache.set(water, boxes)
  return boxes
}

function inPoly(x, y, poly) {
  let inside = false
  for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
    const ya = poly[a][1]
    const yb = poly[b][1]
    if (ya > y !== yb > y && x < ((poly[b][0] - poly[a][0]) * (y - ya)) / (yb - ya) + poly[a][0]) inside = !inside
  }
  return inside
}

function FocusArea({ meta }) {
  const size = useThree((s) => s.size)
  // drei <Html> picks its DOM container once R3F has connected its events (see Rovers.jsx).
  const labelsReady = useThree((s) => !!s.events.connected)
  const line = useMemo(() => new FatPolyline(SEGS + 1, { color: meta.color, width: STYLE.operator.width, opacity: 0.5, dashed: true }), [meta.color])
  const xyz = useMemo(() => new Float32Array((SEGS + 1) * 3), [])
  const anchor = useRef()
  const tag = useRef()
  const text = useRef()
  const st = useRef({
    alpha: 0, // shown amount, 0..1
    grow: 1, // 0..1 while a new focus settles into place
    x: 0,
    y: 0,
    r: 0,
    reason: '',
    label: null,
    // what the circle was last draped for
    dx: NaN,
    dy: NaN,
    dr: NaN,
    dReason: '',
    version: -1,
    water: null,
    at: 0,
    angle: NaN, // where the tag sits on the edge (radians, world frame)
    sec: -2, // seconds shown on the tag
    tagAlpha: -1,
    tagText: '',
  })
  useEffect(() => () => line.dispose(), [line])
  useEffect(() => line.setResolution(size.width, size.height), [line, size])

  useFrame(({ camera }, dt) => {
    const p = st.current
    const s = simStore.get()
    const f = s.rovers[meta.id]?.focus || null
    if (f) {
      const fresh = p.alpha <= 0 || Math.hypot(f.x - p.x, f.y - p.y) > Math.max(2, p.r * 0.5)
      if (fresh) {
        p.grow = reducedMotion ? 1 : 0
        p.angle = NaN
        p.sec = -2
      }
      p.x = f.x
      p.y = f.y
      p.r = f.r
      p.reason = f.reason
      if (p.label !== f.label) {
        p.label = f.label
        p.sec = -2 // rewrite the tag
      }
    }
    const step = reducedMotion ? 1 : dt
    p.alpha = f ? Math.min(1, p.alpha + step / FADE_IN) : Math.max(0, p.alpha - step / FADE_OUT)
    if (p.alpha <= 0) {
      line.object.visible = false
      if (tag.current && p.tagAlpha !== 0) {
        tag.current.style.opacity = '0'
        p.tagAlpha = 0
      }
      return
    }
    p.grow = Math.min(1, p.grow + step / FADE_IN)
    const look = STYLE[p.reason] || STYLE.operator
    const ease = 1 - (1 - p.grow) * (1 - p.grow)
    const R = p.r * (0.9 + 0.1 * ease)

    // Drape: when the circle moves or grows, and at most twice a second for new ground.
    const T = s.terrain
    const version = T ? T.version : 0
    const water = s.features.water
    const now = performance.now()
    const moved = p.dx !== p.x || p.dy !== p.y || Math.abs(p.dr - R) > 0.005 || p.dReason !== p.reason
    if (moved || ((version !== p.version || water !== p.water) && now - p.at > 500)) {
      p.dx = p.x
      p.dy = p.y
      p.dr = R
      p.dReason = p.reason
      p.version = version
      p.water = water
      p.at = now
      const boxes = waterBoxes(water)
      let h = groundAt(p.x, p.y, 0)
      let len = 0
      for (let i = 0; i <= SEGS; i++) {
        const a = ((i % SEGS) / SEGS) * TWO_PI
        const x = p.x + Math.cos(a) * R
        const y = p.y + Math.sin(a) * R
        h = groundAt(x, y, h)
        for (let b = 0; b < boxes.length; b++) {
          const w = boxes[b]
          if (w.level > h && x >= w.x0 && x <= w.x1 && y >= w.y0 && y <= w.y1 && inPoly(x, y, w.outline)) h = w.level
        }
        const o = i * 3
        xyz[o] = x
        xyz[o + 1] = h + LIFT
        xyz[o + 2] = -y
        if (i) len += Math.hypot(x - xyz[o - 3], xyz[o + 1] - xyz[o - 2], xyz[o + 2] - xyz[o - 1])
      }
      line.set(xyz, SEGS + 1)
      // Whole dashes around the draped length, so the seam does not show.
      const period = len / look.dashes
      line.material.dashSize = period * look.fill
      line.material.gapSize = period * (1 - look.fill)
      line.material.linewidth = look.width
    }
    line.object.visible = true
    line.material.opacity = look.opacity * p.alpha
    if (!reducedMotion) line.material.dashOffset -= dt * look.crawl

    // The tag sits on the edge facing the camera, and glides round as the view orbits
    // (it stays put when the camera looks straight down on the circle).
    const cx = camera.position.x - p.x
    const cy = -camera.position.z - p.y
    if (Math.hypot(cx, cy) > R * 0.3) {
      const want = Math.atan2(cy, cx)
      p.angle = Number.isNaN(p.angle) ? want : wrap(p.angle + wrap(want - p.angle) * Math.min(1, dt * 3))
    } else if (Number.isNaN(p.angle)) p.angle = -Math.PI / 2
    if (anchor.current) {
      const k = Math.round(((p.angle + TWO_PI) % TWO_PI) / TWO_PI * SEGS) % SEGS
      const gh = xyz[k * 3 + 1] - LIFT
      anchor.current.position.set(p.x + Math.cos(p.angle) * (R + TAG_OUT), gh + TAG_UP, -(p.y + Math.sin(p.angle) * (R + TAG_OUT)))
    }

    const el = tag.current
    if (!el) return
    // Text: rewritten only when the whole seconds change. Between ticks the countdown
    // runs on the estimated sim time; it never ticks back up by a second from jitter.
    if (f && text.current) {
      const left = focusLeft(s, f, nowSeconds())
      let sec = left === null ? -1 : Math.ceil(left)
      if (p.sec >= 0 && sec === p.sec + 1) sec = p.sec
      if (sec !== p.sec) {
        p.sec = sec
        text.current.textContent = sec >= 0 ? `${focusMapLabel(f)} · ${sec} s` : focusMapLabel(f)
      }
      if (el.dataset.reason !== p.reason) el.dataset.reason = p.reason
    }
    // Opacity in 1/40 steps, so the style is only written while fading.
    const ta = Math.round(p.alpha * 40) / 40
    if (ta !== p.tagAlpha) {
      p.tagAlpha = ta
      el.style.opacity = String(ta)
    }
  })

  return (
    <group>
      <primitive object={line.object} renderOrder={4} />
      <group ref={anchor}>
        {labelsReady && (
          <Html center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
            <div ref={tag} className="focus-tag" style={{ '--c': meta.color, opacity: 0 }}>
              <i className="focus-tag-ring" aria-hidden="true" />
              <span ref={text} />
            </div>
          </Html>
        )}
      </group>
    </group>
  )
}

export function FocusAreas() {
  const meta = useSim((s) => s.roverMeta)
  const gen = useSim((s) => s.gen)
  return (
    <group>
      {meta.map((m) => (
        <FocusArea key={`${gen}:${m.id}`} meta={m} />
      ))}
    </group>
  )
}
