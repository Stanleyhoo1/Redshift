import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { nowSeconds, roverPoses, simStore, useApp } from '../state/store.js'
import { sampleGround } from '../state/simState.js'
import { HAZARD, HAZARD_TYPES, LIFE, SAND, SPIRE } from '../lib/theme.js'
import { PLACE_TEXT, placement } from '../lib/placement.js'
import { FatPolyline } from './fatLine.js'
import { glowTexture } from './textures.js'
import { knownAt, pointInPoly } from './ground.js'
import { crystalGeo } from './Spires.jsx'
import { fogYFor } from './sceneConstants.js'
import { pickGround } from './pick.js'

// Point-and-click hazards (PROTOCOL.md version 4). While the hazard bar is in placement
// mode, a ghost marker follows the cursor on the terrain: the hazard's radius, tinted red
// where the sim would obviously refuse (off the map, on the lander, too close to a rover,
// in identified water, sand on steep ground). A click without a drag places it (orbit
// drags never do); a right click lets the sim pick the spot. For a motor fault the
// nearest rover under the cursor lights up instead, and a click picks it.

const SEGS = 64
const BAD = '#ff5a4f'
const COLORS = { rockfall: '#e8dcc8', sand: SAND, spire_fall: SPIRE.line, lifeform: LIFE, rover_fault: HAZARD }
const ROVER_PICK_PX = 44
const SPIRE_REACH = 6 // m: PROTOCOL.md, a spire fall topples the nearest standing spire within about 6 m

const r1 = (v) => Math.round(v * 10) / 10
const META = Object.fromEntries(HAZARD_TYPES.map((h) => [h.type, h]))
const EAST = [1, 0]

// What a click right here would do, from what this client can see. Writes into one
// shared object (this runs every frame while placing).
const _cur = { x: 0, y: 0, h: 0, valid: true, reason: '', spire: null, dir: null, rover: null }
const _dir = [1, 0]
const bad = (reason) => ((_cur.valid = false), (_cur.reason = reason), _cur)

function assess(type, hit, s, reality) {
  const out = _cur
  out.x = hit.x
  out.y = hit.y
  out.h = hit.h
  out.valid = true
  out.reason = ''
  out.spire = null
  out.dir = null
  out.rover = null
  const map = s.map
  const outside = (v, o, w) => v < o || v > o + w
  if (!hit.onMap || outside(hit.x, map.ox, map.width) || outside(hit.y, map.oy, map.depth)) return bad('Off the map')
  const L = s.lander
  if (L && Math.hypot(hit.x - L.x, hit.y - L.y) < L.radius + 0.5) return bad('On the lander')
  for (const m of s.roverMeta) {
    const p = roverPoses.get(m.id)
    if (!p) continue
    if (Math.hypot(p.x - hit.x, -p.z - hit.y) < 2) return bad(`Too close to ${m.name}`)
  }
  let wet = false
  const zones = s.fleet.water_zones
  if (Array.isArray(zones)) for (const z of zones) if (Array.isArray(z) && Math.hypot(z[0] - hit.x, z[1] - hit.y) < z[2]) wet = true
  if (!wet && reality) for (const w of s.features.water) if (pointInPoly(hit.x, hit.y, w.outline)) wet = true
  if (type === 'spire_fall') {
    // The spire that would topple: the nearest standing one the user can see.
    let best = null
    let bd = SPIRE_REACH
    for (const sp of s.features.spires) {
      if (!sp.standing || s.fallen[sp.id]) continue
      if (!reality && !knownAt(sp.x, sp.y)) continue
      const d = Math.hypot(sp.x - hit.x, sp.y - hit.y)
      if (d <= bd) {
        bd = d
        best = sp
      }
    }
    if (best) {
      const d = Math.max(1e-3, Math.hypot(hit.x - best.x, hit.y - best.y))
      out.spire = best
      if (d > 0.3) {
        _dir[0] = (hit.x - best.x) / d
        _dir[1] = (hit.y - best.y) / d
        out.dir = _dir
      }
      return out
    }
    return wet ? bad('In the water') : out
  }
  if (wet) return bad('In the water')
  if (type === 'sand') {
    const T = s.terrain
    const i = Math.round((hit.x - T.ox) / T.res)
    const j = Math.round((hit.y - T.oy) / T.res)
    const k = j * T.nx + i
    if (T.known[k] && T.slope[k] > 20) return bad(`Too steep for sand (${Math.round(T.slope[k])} deg)`)
  }
  return out
}

const _v = new THREE.Vector3()

// The rover under the cursor (screen space, so the small models are easy to hit).
function roverUnder(ndc, camera, size, s) {
  let best = null
  let bd = ROVER_PICK_PX
  for (const m of s.roverMeta) {
    const p = roverPoses.get(m.id)
    if (!p) continue
    _v.set(p.x, p.y + 0.8, p.z).project(camera)
    if (_v.z > 1) continue
    const dx = ((_v.x - ndc.x) * size.width) / 2
    const dy = ((_v.y - ndc.y) * size.height) / 2
    const d = Math.hypot(dx, dy)
    if (d < bd) {
      bd = d
      best = m
    }
  }
  return best
}

const ringGeo = new THREE.RingGeometry(0.86, 1, 56).rotateX(-Math.PI / 2)
const beamGeo = new THREE.CylinderGeometry(0.04, 0.04, 1, 6, 1, true).translate(0, 0.5, 0)
const glowGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2)

// Bottom-centre on its anchor (CSS), so it never covers the marker itself; over a rover
// it clears the rover's own name tag.
function GhostLabel({ info, rover = false }) {
  if (!info) return null
  return (
    <div className="ghost-tag" data-bad={!info.valid || undefined} data-rover={rover || undefined} style={{ '--c': info.color }}>
      <b>{info.title}</b>
      {info.sub && <span>{info.sub}</span>}
    </div>
  )
}

export function Placement({ reality }) {
  const placing = useApp((a) => a.placing)
  const gl = useThree((s) => s.gl)
  const camera = useThree((s) => s.camera)
  const size = useThree((s) => s.size)
  const labelsReady = useThree((s) => !!s.events.connected)
  const st = useRef({ ndc: null, busy: false, flash: null, info: '', cur: null })
  const [info, setInfo] = useState(null)
  const group = useRef()
  const disc = useRef()
  const beam = useRef()
  const glow = useRef()
  const ghostSpire = useRef()
  const roverRing = useRef()
  const flash = useRef()
  const outline = useMemo(() => new FatPolyline(SEGS + 1, { color: '#ffffff', width: 2.4, dashed: true, dashSize: 0.5, gapSize: 0.32, opacity: 0.98 }), [])
  const fallLine = useMemo(() => new FatPolyline(2, { color: SPIRE.line, width: 3, dashed: true, dashSize: 0.4, gapSize: 0.25, opacity: 0.95 }), [])
  const ringXyz = useMemo(() => new Float32Array((SEGS + 1) * 3), [])
  const lineXyz = useMemo(() => new Float32Array(6), [])
  const mats = useMemo(
    () => ({
      disc: new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.16, depthWrite: false, depthTest: false, side: THREE.DoubleSide }),
      beam: new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.8, depthWrite: false, blending: THREE.AdditiveBlending }),
      glow: new THREE.MeshBasicMaterial({ color: '#ffffff', map: glowTexture(), transparent: true, opacity: 0.7, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }),
      spire: new THREE.MeshBasicMaterial({ color: SPIRE.line, transparent: true, opacity: 0.45, depthWrite: false }),
      rover: new THREE.MeshBasicMaterial({ color: HAZARD, transparent: true, opacity: 0.95, depthWrite: false, depthTest: false }),
      flash: new THREE.MeshBasicMaterial({ color: BAD, transparent: true, opacity: 0, depthWrite: false, depthTest: false }),
    }),
    [],
  )
  const discGeo = useMemo(() => {
    const g = new THREE.CircleGeometry(1, SEGS).rotateX(-Math.PI / 2)
    g.attributes.position.setUsage(THREE.DynamicDrawUsage)
    return g
  }, [])
  useEffect(
    () => () => {
      outline.dispose()
      fallLine.dispose()
      discGeo.dispose()
      Object.values(mats).forEach((m) => m.dispose())
    },
    [outline, fallLine, discGeo, mats],
  )
  useEffect(() => {
    outline.material.depthTest = false
    fallLine.material.depthTest = false
  }, [outline, fallLine])
  useEffect(() => {
    outline.setResolution(size.width, size.height)
    fallLine.setResolution(size.width, size.height)
  }, [outline, fallLine, size])

  // Pointer handling on the canvas while placing.
  useEffect(() => {
    if (!placing) return
    const el = gl.domElement
    const s = st.current
    el.style.cursor = 'crosshair'
    let down = null
    const ndc = new THREE.Vector2()
    const ndcOf = (e) => {
      const r = el.getBoundingClientRect()
      return ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    }
    const onMove = (e) => {
      s.ndc = ndcOf(e)
    }
    const onLeave = () => {
      s.ndc = null
    }
    const onDown = (e) => {
      down = { x: e.clientX, y: e.clientY, b: e.button }
    }
    const send = async (where, at) => {
      if (s.busy || !placement.commit) return
      s.busy = true
      try {
        const res = await placement.commit(where)
        if (res && !res.ok && at) s.flash = { ...at, t: nowSeconds() }
      } finally {
        s.busy = false
      }
    }
    const onUp = (e) => {
      if (!down || e.button !== down.b) return
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y)
      down = null
      if (moved > 5) return // that was an orbit drag, not a click
      if (e.button === 2) return send({ auto: true }, null)
      if (e.button !== 0) return
      s.ndc = ndcOf(e)
      const cur = s.compute?.()
      if (!cur) return
      if (placing.mode === 'rover') {
        if (cur.rover) send({ rover: cur.rover.id }, null)
        return
      }
      send({ x: r1(cur.x), y: r1(cur.y) }, { x: cur.x, y: cur.y, h: cur.h })
    }
    const onCtx = (e) => e.preventDefault()
    el.addEventListener('pointermove', onMove)
    el.addEventListener('pointerleave', onLeave)
    el.addEventListener('pointerdown', onDown)
    el.addEventListener('pointerup', onUp)
    el.addEventListener('contextmenu', onCtx)
    return () => {
      el.removeEventListener('pointermove', onMove)
      el.removeEventListener('pointerleave', onLeave)
      el.removeEventListener('pointerdown', onDown)
      el.removeEventListener('pointerup', onUp)
      el.removeEventListener('contextmenu', onCtx)
      el.style.cursor = ''
      s.ndc = null
      s.cur = null
      setInfo(null)
    }
  }, [placing, gl])

  const raycaster = useMemo(() => new THREE.Raycaster(), [])
  // What is under the cursor right now (used every frame and on a click). Reads the
  // latest props from a ref, so it is created once.
  const live = useRef(null)
  live.current = { placing, reality, camera, size }
  const compute = useMemo(
    () => () => {
      const { placing: pl, reality: real, camera: cam, size: sz } = live.current
      const s = st.current
      const sim = simStore.get()
      if (!pl || !s.ndc || !sim.terrain) return null
      if (pl.mode === 'rover') {
        _cur.rover = roverUnder(s.ndc, cam, sz, sim)
        return _cur
      }
      raycaster.setFromCamera(s.ndc, cam)
      const hit = pickGround(raycaster.ray, sim.terrain, fogYFor(sim.map))
      return hit ? assess(pl.type, hit, sim, real) : null
    },
    [raycaster],
  )
  st.current.compute = compute

  useFrame(() => {
    const s = st.current
    const g = group.current
    const sim = simStore.get()
    const now = nowSeconds()
    // A refused click: a red ring bursts where it was.
    if (flash.current) {
      const f = s.flash
      const age = f ? now - f.t : 9
      flash.current.visible = age < 1.1
      if (age < 1.1) {
        flash.current.position.set(f.x, f.h + 0.12, -f.y)
        flash.current.scale.setScalar(0.6 + age * 3.2)
        mats.flash.opacity = 0.95 * (1 - age / 1.1)
      }
    }
    const cur = compute()
    s.cur = cur
    if (!g) return
    if (!cur) {
      g.visible = false
      if (roverRing.current) roverRing.current.visible = false
      outline.object.visible = false
      fallLine.object.visible = false
      if (s.info) {
        s.info = ''
        setInfo(null)
      }
      return
    }
    const type = placing.type
    const meta = META[type]
    const label = PLACE_TEXT[type]?.label || type

    // Motor fault: light up the rover under the cursor.
    if (placing.mode === 'rover') {
      g.visible = false
      outline.object.visible = false
      fallLine.object.visible = false
      const rr = roverRing.current
      const m = cur.rover
      const p = m && roverPoses.get(m.id)
      const faulted = m && sim.rovers[m.id]?.state === 'fault'
      if (rr) {
        rr.visible = !!p
        if (p) {
          rr.position.set(p.x, p.y + 0.15, p.z)
          rr.scale.setScalar(2.3 + 0.12 * Math.sin(now * 6))
          mats.rover.color.set(faulted ? '#8d99aa' : HAZARD)
        }
      }
      const key = m ? `${m.id}|${faulted}` : ''
      if (key !== s.info) {
        s.info = key
        setInfo(m ? { title: faulted ? `${m.name} already has a fault` : `Motor fault on ${m.name}`, sub: faulted ? '' : 'Click to trigger', valid: !faulted, color: HAZARD } : null)
      }
      return
    }
    if (roverRing.current) roverRing.current.visible = false

    const color = cur.valid ? COLORS[type] || '#ffffff' : BAD
    g.visible = true
    g.position.set(cur.x, cur.h, -cur.y)
    mats.disc.color.set(color)
    mats.beam.color.set(color)
    mats.glow.color.set(color)
    outline.material.color.set(color)

    // Radius ring and disc draped on what the fleet knows (the dark floor elsewhere).
    let R = meta?.radius ?? 2
    if (type === 'spire_fall') R = cur.spire ? 0.6 : 0.8
    const T = sim.terrain
    const pos = discGeo.attributes.position
    pos.setXYZ(0, 0, 0.1, 0)
    for (let q = 0; q <= SEGS; q++) {
      const a = (q / SEGS) * Math.PI * 2
      const x = cur.x + Math.cos(a) * R
      const y = cur.y + Math.sin(a) * R
      const h = (sampleGround(T, x, y) ?? cur.h) + 0.1
      ringXyz[q * 3] = x
      ringXyz[q * 3 + 1] = h
      ringXyz[q * 3 + 2] = -y
      pos.setXYZ(q + 1, Math.cos(a) * R, h - cur.h, -Math.sin(a) * R)
    }
    pos.needsUpdate = true
    discGeo.computeBoundingSphere()
    outline.set(ringXyz, SEGS + 1)
    outline.material.dashOffset -= 0.01
    if (beam.current) beam.current.scale.set(1, 2.4 + 0.25 * Math.sin(now * 5), 1)
    if (glow.current) glow.current.scale.setScalar(R * 0.9 + 0.6)

    // Spire fall: the spire that would topple and where it would land, or a new spire.
    let title = label
    let sub = `(${r1(cur.x)}, ${r1(cur.y)})`
    if (type === 'spire_fall') {
      if (ghostSpire.current) ghostSpire.current.visible = !cur.spire && cur.valid
      if (cur.spire) {
        const sp = cur.spire
        const dir = cur.dir || EAST
        const gz = sampleGround(T, sp.x, sp.y) ?? cur.h
        lineXyz[0] = sp.x
        lineXyz[1] = gz + 0.25
        lineXyz[2] = -sp.y
        const tx = sp.x + dir[0] * sp.height
        const ty = sp.y + dir[1] * sp.height
        lineXyz[3] = tx
        lineXyz[4] = (sampleGround(T, tx, ty) ?? gz) + 0.25
        lineXyz[5] = -ty
        fallLine.set(lineXyz, 2)
        fallLine.material.color.set(cur.valid ? SPIRE.line : BAD)
        title = `Topple spire ${sp.id}`
        sub = `falls ${r1(sp.height)} m this way`
      } else {
        fallLine.object.visible = false
        title = 'Drop a new spire'
      }
    } else {
      fallLine.object.visible = false
      if (ghostSpire.current) ghostSpire.current.visible = false
    }
    if (!cur.valid) {
      title = cur.reason
      sub = 'The sim will refuse this spot'
    }
    const key = `${title}|${sub}|${cur.valid}`
    if (key !== s.info) {
      s.info = key
      setInfo({ title, sub, valid: cur.valid, color })
    }
  })

  return (
    <group>
      <group ref={group} visible={false}>
        <mesh geometry={discGeo} material={mats.disc} renderOrder={12} />
        <mesh ref={glow} geometry={glowGeo} material={mats.glow} position={[0, 0.12, 0]} renderOrder={12} />
        <mesh ref={beam} geometry={beamGeo} material={mats.beam} renderOrder={12} />
        <mesh ref={ghostSpire} geometry={crystalGeo} material={mats.spire} scale={[0.36, 3.2, 0.36]} visible={false} renderOrder={12} />
        {labelsReady && info && placing?.mode === 'point' && (
          <Html position={[0, 2.75, 0]} zIndexRange={[14, 0]} style={{ pointerEvents: 'none' }}>
            <GhostLabel info={info} />
          </Html>
        )}
      </group>
      <primitive object={outline.object} renderOrder={12} />
      <primitive object={fallLine.object} renderOrder={12} />
      <mesh ref={roverRing} geometry={ringGeo} material={mats.rover} visible={false} renderOrder={12}>
        {labelsReady && info && placing?.mode === 'rover' && (
          <Html position={[0, 2.5, 0]} zIndexRange={[14, 0]} style={{ pointerEvents: 'none' }}>
            <GhostLabel info={info} rover />
          </Html>
        )}
      </mesh>
      <mesh ref={flash} geometry={ringGeo} material={mats.flash} visible={false} renderOrder={12} />
    </group>
  )
}
