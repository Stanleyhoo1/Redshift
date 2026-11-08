import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '../globe/PlaneHtml.jsx'
import * as THREE from 'three'
import { appStore, roverPoses, simStore, useSim } from '../state/store.js'
import { clearPlaceHover, hasPlaces, hoverKey, placeNumbers, sendAnswer, setPlaceHover, shortPlace, useAnswer, usePlaceHover } from '../lib/answers.js'
import { SIGNAL } from '../lib/theme.js'
import { FatPolyline } from './fatLine.js'
import { glowTexture } from './textures.js'
import { groundAt, pointInPoly } from './ground.js'
import { reducedMotion } from './sceneConstants.js'

// Version 6 `choose_place` requests: every candidate place (an option with x, y) of an
// open request gets a numbered marker on the terrain in the asking rover's colour: a
// pulsing ring, a beam up to a badge with its number and a short label, and a faint
// dashed line from the rover. Clicking the marker or its badge answers the request
// (POST /api/answer with that option). Hovering a marker lights the matching chip in the
// request card and the chat, and the other way round (appStore.placeHover). Markers
// leave with the request.

const LINE_N = 28 // points along the draped rover-to-candidate line
const BADGE_UP = 3.6 // m above the ground at the closest zoom

const geo = {
  ring: new THREE.RingGeometry(0.8, 1, 64).rotateX(-Math.PI / 2),
  ripple: new THREE.RingGeometry(0.93, 1, 64).rotateX(-Math.PI / 2),
  disc: new THREE.CircleGeometry(1, 40).rotateX(-Math.PI / 2),
  glow: new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
  beam: new THREE.CylinderGeometry(0.05, 0.05, 1, 8, 1, true).translate(0, 0.5, 0),
}

function PlaceBadge({ q, o, n, color, onPick }) {
  const key = hoverKey(q.id, o.id)
  const hover = usePlaceHover()
  const answer = useAnswer(q.id)
  const busy = answer?.state === 'pending' || answer?.state === 'sent'
  const chosen = busy && answer.option === o.id
  return (
    <button
      type="button"
      className="place-mark"
      style={{ '--c': color }}
      data-hl={hover === key || undefined}
      data-chosen={chosen || undefined}
      data-dim={(busy && !chosen) || undefined}
      disabled={busy}
      onClick={onPick}
      onMouseEnter={() => setPlaceHover(key)}
      onMouseLeave={() => clearPlaceHover(key)}
      title={`${o.label}. Click to choose it.`}
    >
      <b className="place-num">{n}</b>
      <span className="place-mark-label">{shortPlace(o.label)}</span>
    </button>
  )
}

function PlaceMarker({ q, o, n, color }) {
  const size = useThree((s) => s.size)
  const labelsReady = useThree((s) => !!s.events.connected)
  const features = useSim((s) => s.features)
  const group = useRef()
  const scaled = useRef()
  const ring = useRef()
  const ripple = useRef()
  const glow = useRef()
  const beam = useRef()
  const badge = useRef()
  const key = hoverKey(q.id, o.id)
  const st = useRef({ gz: null, rx: NaN, ry: NaN, lineAt: -1, hovered: false })
  const mats = useMemo(() => {
    const basic = (p) => new THREE.MeshBasicMaterial({ color, transparent: true, depthWrite: false, depthTest: false, toneMapped: false, fog: false, ...p })
    return {
      ring: basic({ opacity: 0.95 }),
      ripple: basic({ opacity: 0.6, blending: THREE.AdditiveBlending }),
      disc: basic({ opacity: 0.14 }),
      glow: basic({ map: glowTexture(), opacity: 0.55, blending: THREE.AdditiveBlending }),
      beam: basic({ opacity: 0.7, blending: THREE.AdditiveBlending }),
    }
  }, [color])
  const line = useMemo(() => new FatPolyline(LINE_N, { color, width: 1.8, opacity: 0.5, dashed: true, dashSize: 0.7, gapSize: 0.55 }), [color])
  const xyz = useMemo(() => new Float32Array(LINE_N * 3), [])
  useEffect(() => () => (Object.values(mats).forEach((m) => m.dispose()), line.dispose()), [mats, line])
  useEffect(() => line.setResolution(size.width, size.height), [line, size])
  useEffect(() => () => clearPlaceHover(key), [key])
  useEffect(() => () => st.current.hovered && (document.body.style.cursor = ''), [])

  // A candidate in water sits on the water surface, not the bottom of the LiDAR hole.
  const waterLevel = useMemo(() => {
    for (const w of features.water || []) if (w.outline?.length > 2 && pointInPoly(o.x, o.y, w.outline)) return w.level
    return null
  }, [features, o.x, o.y])

  const pick = () => {
    if (appStore.get().placing) return
    sendAnswer(q.id, o.id)
  }

  useFrame(({ clock, camera }, dt) => {
    const g = group.current
    if (!g) return
    const p = st.current
    let h = groundAt(o.x, o.y, 0)
    if (waterLevel !== null && waterLevel > h) h = waterLevel
    p.gz = p.gz === null ? h : p.gz + (h - p.gz) * Math.min(1, dt * 4)
    g.position.set(o.x, p.gz, -o.y)

    const a = appStore.get()
    const hl = a.placeHover === key
    const ans = a.answers[q.id]
    const busy = ans && (ans.state === 'pending' || ans.state === 'sent')
    const chosen = busy && ans.option === o.id
    const dim = busy && !chosen

    // Grow with distance so the marker reads from the overview as well as up close.
    const dist = camera.position.distanceTo(g.position)
    const k = Math.max(1, Math.min(3.4, dist / 34))
    const t = clock.elapsedTime
    if (scaled.current) scaled.current.scale.setScalar(k * (hl || chosen ? 1.22 : 1))
    const pulse = reducedMotion ? 1 : 1 + 0.07 * Math.sin(t * 3.2 + n)
    if (ring.current) ring.current.scale.set(1.9 * pulse, 1, 1.9 * pulse)
    mats.ring.opacity = dim ? 0.3 : 0.95
    mats.disc.opacity = dim ? 0.05 : hl || chosen ? 0.3 : 0.14
    if (ripple.current) {
      const ph = reducedMotion ? 0.4 : (t * (hl ? 1.1 : 0.62) + n * 0.31) % 1
      ripple.current.scale.set(1.9 + ph * 2.6, 1, 1.9 + ph * 2.6)
      mats.ripple.opacity = (1 - ph) * (dim ? 0.15 : hl ? 0.95 : 0.6)
    }
    if (glow.current) glow.current.scale.setScalar(hl || chosen ? 7.5 : 5.5)
    mats.glow.opacity = dim ? 0.15 : hl || chosen ? 0.85 : 0.5
    const up = BADGE_UP * k
    if (beam.current) beam.current.scale.set(k, up, k)
    mats.beam.opacity = dim ? 0.2 : 0.7
    if (badge.current) badge.current.position.y = up

    // The faint line from the asking rover, draped on the ground. Redrawn when the rover
    // moves, and now and then for newly mapped ground.
    const rp = roverPoses.get(q.rover)
    if (!rp) {
      line.object.visible = false
      return
    }
    const rx = rp.x
    const ry = -rp.z
    const now = t
    if (Math.abs(rx - p.rx) > 0.04 || Math.abs(ry - p.ry) > 0.04 || now - p.lineAt > 0.6) {
      p.rx = rx
      p.ry = ry
      p.lineAt = now
      for (let i = 0; i < LINE_N; i++) {
        const f = i / (LINE_N - 1)
        const x = rx + (o.x - rx) * f
        const y = ry + (o.y - ry) * f
        let gh = i === 0 ? rp.y : groundAt(x, y, rp.y)
        if (i === LINE_N - 1) gh = p.gz
        xyz[i * 3] = x
        xyz[i * 3 + 1] = gh + 0.35
        xyz[i * 3 + 2] = -y
      }
      line.set(xyz, LINE_N)
    }
    line.material.opacity = dim ? 0.15 : hl || chosen ? 0.9 : 0.45
    if (!reducedMotion) line.material.dashOffset -= dt * 0.9
  })

  const over = (e) => {
    e.stopPropagation()
    if (appStore.get().placing) return
    st.current.hovered = true
    document.body.style.cursor = 'pointer'
    setPlaceHover(key)
  }
  const out = () => {
    if (st.current.hovered) document.body.style.cursor = ''
    st.current.hovered = false
    clearPlaceHover(key)
  }

  return (
    <group>
      <primitive object={line.object} renderOrder={8} />
      <group ref={group}>
        <group ref={scaled}>
          <mesh ref={glow} geometry={geo.glow} material={mats.glow} position={[0, 0.06, 0]} renderOrder={8} />
          <mesh geometry={geo.disc} material={mats.disc} scale={[1.9, 1, 1.9]} position={[0, 0.08, 0]} renderOrder={9} onClick={(e) => (e.stopPropagation(), e.delta <= 6 && pick())} onPointerOver={over} onPointerOut={out} />
          <mesh ref={ripple} geometry={geo.ripple} material={mats.ripple} position={[0, 0.1, 0]} renderOrder={9} />
          <mesh ref={ring} geometry={geo.ring} material={mats.ring} position={[0, 0.12, 0]} renderOrder={10} />
        </group>
        <mesh ref={beam} geometry={geo.beam} material={mats.beam} renderOrder={9} />
        <group ref={badge} position={[0, BADGE_UP, 0]}>
          {labelsReady && (
            <Html center zIndexRange={[16, 0]} style={{ pointerEvents: 'none' }}>
              <PlaceBadge q={q} o={o} n={n} color={color} onPick={pick} />
            </Html>
          )}
        </group>
      </group>
    </group>
  )
}

export function PlaceCandidates() {
  const requests = useSim((s) => s.requests)
  const meta = useSim((s) => s.roverMeta)
  const gen = useSim((s) => s.gen)
  const open = useMemo(() => requests.filter(hasPlaces), [requests])
  if (!open.length) return null
  return (
    <group>
      {open.map((q) => {
        const nums = placeNumbers(q)
        const color = meta.find((m) => m.id === q.rover)?.color || SIGNAL
        return q.options.filter((o) => nums[o.id]).map((o) => <PlaceMarker key={`${gen}:${q.id}:${o.id}`} q={q} o={o} n={nums[o.id]} color={color} />)
      })}
    </group>
  )
}
