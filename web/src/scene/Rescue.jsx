import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { roverPoses, simStore, useSim } from '../state/store.js'
import { FatPolyline } from './fatLine.js'
import { groundAt } from './ground.js'
import { dotTexture, glowTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'
import { distressTag } from '../lib/rescue.js'

// Version 9 on the map: a flash and sparks where a rover is hit; a pulsing distress beacon
// (expanding red and amber rings, a light pillar, "Rover 2 needs help") over a rover that
// cannot drive; a dashed line from its helper while it is on the way; a status tag with
// progress while it is diagnosed, fixed or salvaged; a tow line between a helper and the
// rover it tows; a marker under each wreck. One always-mounted set per rover: buffers and
// materials are made once, and the frame loops allocate nothing.

const RED = '#ff4d4d'
const AMBER = '#ffb547'
const SPARKS = 22
const HIT_S = 1.2 // s a hit effect lasts
const RINGS = 3
const RING_PERIOD = 2.4 // s per ring
const LINE_SEGS = 40
const TOW_SEGS = 10

// Loops instead of Array.find: these run every frame.
function rescueOfId(rescues, id) {
  for (let i = 0; i < rescues.length; i++) if (rescues[i].rover === id && rescues[i].phase !== 'done') return rescues[i]
  return null
}
function helpingId(rescues, id) {
  for (let i = 0; i < rescues.length; i++) if (rescues[i].helper === id && rescues[i].phase !== 'done') return rescues[i]
  return null
}

const geo = {
  ring: new THREE.RingGeometry(0.9, 1, 64).rotateX(-Math.PI / 2),
  shock: new THREE.RingGeometry(0.78, 1, 48).rotateX(-Math.PI / 2),
  pillar: new THREE.CylinderGeometry(0.07, 0.07, 1, 8, 1, true).translate(0, 0.5, 0),
  disc: new THREE.CircleGeometry(1, 40).rotateX(-Math.PI / 2),
  cross: new THREE.PlaneGeometry(0.16, 2.6).rotateX(-Math.PI / 2),
}

// ---------------------------------------------------------------- hits

function HitFx({ meta }) {
  const group = useRef()
  const flash = useRef()
  const shock = useRef()
  const points = useRef()
  const st = useRef({ t0: -99, k: 1, dust: false, lastHitT: null, seenHit: false, lastKey: 0, at: -99 })
  const mats = useMemo(
    () => ({
      flash: new THREE.SpriteMaterial({ map: glowTexture(), color: '#ffd9a0', transparent: true, opacity: 0, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }),
      shock: new THREE.MeshBasicMaterial({ color: '#ffb36b', transparent: true, opacity: 0, depthWrite: false, depthTest: false }),
      sparks: new THREE.PointsMaterial({ color: '#ffd28a', map: dotTexture(), size: 6, sizeAttenuation: false, transparent: true, opacity: 0, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }),
    }),
    [],
  )
  const spark = useMemo(() => {
    const pos = new Float32Array(SPARKS * 3)
    const vel = new Float32Array(SPARKS * 3)
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 50)
    return { pos, vel, g }
  }, [])
  useEffect(() => () => (Object.values(mats).forEach((m) => m.dispose()), spark.g.dispose()), [mats, spark])

  useFrame(({ clock }) => {
    const p = st.current
    const s = simStore.get()
    const r = s.rovers[meta.id]
    const now = clock.elapsedTime
    // New hits: the rover's last_hit changed, or a live damage event names it.
    let trigger = null
    const lh = r?.health?.last_hit
    if (lh && Number.isFinite(lh.t) && lh.t !== p.lastHitT) {
      // Only a fresh hit plays (not one from before this client joined).
      if (p.seenHit || s.t - lh.t < 3) trigger = { shock: lh.shock_g, dust: lh.cause === 'dust' }
      p.lastHitT = lh.t
    }
    p.seenHit = true
    const top = s.events.length ? s.events[0].key : 0
    if (top > p.lastKey) {
      for (let i = 0; i < s.events.length && s.events[i].key > p.lastKey; i++) {
        const e = s.events[i]
        if (e.damage && e.rover === meta.id && e.key > s.initSeq && !trigger) trigger = { shock: e.damage.shock_g, dust: e.damage.cause === 'dust' }
      }
      p.lastKey = top
    }
    if (trigger && now - p.at > 0.5) {
      p.at = now
      p.t0 = now
      const shockG = Number.isFinite(trigger.shock) ? trigger.shock : 5
      p.dust = trigger.dust || shockG < 0.5
      p.k = p.dust ? 0.6 : Math.max(0.6, Math.min(1.7, shockG / 7))
      const sp = 2.2 + 2.6 * p.k
      for (let i = 0; i < SPARKS; i++) {
        const a = Math.random() * Math.PI * 2
        const up = 0.35 + Math.random() * 0.9
        const v = sp * (0.4 + Math.random() * 0.6)
        spark.vel[i * 3] = Math.cos(a) * v * (1 - up * 0.4)
        spark.vel[i * 3 + 1] = up * v
        spark.vel[i * 3 + 2] = Math.sin(a) * v * (1 - up * 0.4)
      }
      mats.sparks.color.set(p.dust ? '#c9a27a' : '#ffd28a')
      mats.shock.color.set(p.dust ? '#b98a5c' : '#ffb36b')
    }
    const g = group.current
    if (!g) return
    const age = now - p.t0
    const pose = roverPoses.get(meta.id)
    if (age > HIT_S || !pose || reducedMotion) {
      if (g.visible) g.visible = false
      return
    }
    g.visible = true
    g.position.copy(pose)
    const u = age / HIT_S
    // Flash: a bright burst at the chassis, gone in a third of a second.
    const f = Math.max(0, 1 - age / 0.35)
    mats.flash.opacity = p.dust ? f * 0.35 : f
    if (flash.current) flash.current.scale.setScalar((1.2 + 3.4 * (1 - f)) * p.k)
    // Shock ring on the ground.
    const q = Math.min(1, age / 0.75)
    mats.shock.opacity = 0.9 * (1 - q)
    if (shock.current) shock.current.scale.setScalar((0.6 + 3.2 * q) * p.k)
    // Sparks (dust: a few slow grains).
    const sp = points.current
    if (sp) {
      sp.visible = true
      const pos = spark.pos
      const grav = p.dust ? -2 : -9
      for (let i = 0; i < SPARKS; i++) {
        const o = i * 3
        pos[o] = spark.vel[o] * age
        pos[o + 1] = 0.9 + spark.vel[o + 1] * age + 0.5 * grav * age * age
        pos[o + 2] = spark.vel[o + 2] * age
        if (pos[o + 1] < 0.05) pos[o + 1] = 0.05
      }
      spark.g.attributes.position.needsUpdate = true
      mats.sparks.opacity = (1 - u) * (p.dust ? 0.6 : 1)
    }
  })

  return (
    <group ref={group} visible={false}>
      <sprite ref={flash} material={mats.flash} position={[0, 0.9, 0]} renderOrder={14} />
      <mesh ref={shock} geometry={geo.shock} material={mats.shock} position={[0, 0.12, 0]} renderOrder={14} />
      <points ref={points} geometry={spark.g} material={mats.sparks} renderOrder={15} frustumCulled={false} />
    </group>
  )
}

// ---------------------------------------------------------------- distress and rescue tag

function Distress({ meta }) {
  const group = useRef()
  const rings = useRef([])
  const pillar = useRef()
  const tagAnchor = useRef()
  const tag = useRef()
  const title = useRef()
  const sub = useRef()
  const bar = useRef()
  const labelsReady = useThree((s) => !!s.events.connected)
  const st = useRef({ alpha: 0, calm: 0, key: '', prog: -1, kind: '', shown: -1, state: null, info: null })
  // Name and distance lookups for the tag, made once (they read the frame's state from ctx).
  const ctx = useMemo(() => {
    const c = { s: null, pose: null }
    c.name = (id) => {
      const m = c.s.roverMeta
      for (let i = 0; i < m.length; i++) if (m[i].id === id) return m[i].name
      return id
    }
    c.dist = (id) => {
      const o = roverPoses.get(id)
      return o && c.pose ? Math.hypot(o.x - c.pose.x, o.z - c.pose.z) : null
    }
    return c
  }, [])
  const mats = useMemo(
    () => ({
      rings: Array.from({ length: RINGS }, (_, i) => new THREE.MeshBasicMaterial({ color: i % 2 ? AMBER : RED, transparent: true, opacity: 0, depthWrite: false, depthTest: false, side: THREE.DoubleSide })),
      pillar: new THREE.MeshBasicMaterial({ color: RED, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }),
    }),
    [],
  )
  useEffect(() => () => (mats.rings.forEach((m) => m.dispose()), mats.pillar.dispose()), [mats])

  useFrame(({ clock, camera }, dt) => {
    const p = st.current
    const s = simStore.get()
    const r = s.rovers[meta.id]
    const pose = roverPoses.get(meta.id)
    const g = group.current
    if (!g) return
    const beacon = !!r && !!pose && r.state === 'disabled'
    const rs = r ? rescueOfId(s.rescues, meta.id) : null
    // Calmer once help has arrived (the helper is diagnosing or working on it).
    const arrived = !!rs?.helper && rs.phase !== 'en_route'
    const step = reducedMotion ? 1 : Math.min(dt, 0.1)
    p.alpha = beacon ? Math.min(1, p.alpha + step * 2) : Math.max(0, p.alpha - step * 1.5)
    p.calm += ((arrived ? 1 : 0) - p.calm) * Math.min(1, step * 2)
    if (pose) g.position.copy(pose)

    // Beacon rings and light pillar, sized so they read from the overview.
    if (p.alpha > 0 && pose) {
      g.visible = true
      const d = camera.position.distanceTo(pose)
      const rMax = Math.max(5, Math.min(16, d * 0.11)) * (1 - 0.55 * p.calm)
      const t = clock.elapsedTime
      for (let i = 0; i < RINGS; i++) {
        const m = rings.current[i]
        if (!m) continue
        const u = reducedMotion ? 0.5 + i * 0.15 : ((t / (RING_PERIOD * (1 + p.calm)) + i / RINGS) % 1)
        m.scale.setScalar(1.2 + (rMax - 1.2) * u)
        mats.rings[i].opacity = p.alpha * (1 - u) * (1 - u) * (0.95 - 0.35 * p.calm)
        m.position.y = 0.16 + i * 0.01
      }
      if (pillar.current) {
        const pulse = reducedMotion ? 1 : 0.65 + 0.35 * Math.sin(t * 4)
        mats.pillar.opacity = p.alpha * (1 - p.calm) * 0.55 * pulse
        pillar.current.scale.set(1 + d * 0.004, Math.max(6, Math.min(16, d * 0.12)), 1 + d * 0.004)
        pillar.current.visible = p.calm < 0.98
      }
    } else if (g.visible) g.visible = false // the tag (DOM) shows on its own

    // The tag: "Rover 2 needs help" / helper on the way / diagnosing with progress.
    const el = tag.current
    if (!el || !title.current || !sub.current || !bar.current) return
    // Worked out once per sim update (about 5 a second), not every frame.
    if (s !== p.state) {
      p.state = s
      p.info = null
      if (r && pose) {
        ctx.s = s
        ctx.pose = pose
        let asking = false
        for (let i = 0; i < s.requests.length; i++) if (s.requests[i].kind === 'rescue' && (s.requests[i].about === meta.id || (rs && s.requests[i].rover === rs.helper))) asking = true
        p.info = distressTag(r, s.rescues, ctx.name, ctx.dist, asking)
      }
    }
    const info = p.info
    const shown = info ? 1 : 0
    if (shown !== p.shown) {
      p.shown = shown
      el.style.display = shown ? '' : 'none'
      if (tagAnchor.current) tagAnchor.current.visible = !!shown
    }
    if (!info) return
    const key = `${info.title}|${info.sub || ''}`
    if (key !== p.key) {
      p.key = key
      title.current.textContent = info.title
      sub.current.textContent = info.sub || ''
      sub.current.style.display = info.sub ? '' : 'none'
    }
    if (info.kind !== p.kind) {
      p.kind = info.kind
      el.dataset.kind = info.kind
    }
    const prog = Number.isFinite(info.progress) ? Math.round(info.progress * 50) / 50 : info.busy ? -2 : -1
    if (prog !== p.prog) {
      p.prog = prog
      bar.current.parentElement.style.display = prog === -1 ? 'none' : ''
      bar.current.parentElement.dataset.busy = prog === -2 ? 'true' : ''
      bar.current.style.width = prog >= 0 ? `${Math.max(4, prog * 100)}%` : '40%'
    }
  })

  return (
    <group ref={group} visible={false}>
      {Array.from({ length: RINGS }, (_, i) => (
        <mesh key={i} ref={(el) => (rings.current[i] = el)} geometry={geo.ring} material={mats.rings[i]} renderOrder={6} />
      ))}
      <mesh ref={pillar} geometry={geo.pillar} material={mats.pillar} renderOrder={6} />
      {/* At the name tag's height; CSS lifts the tag just above it at any zoom. */}
      <group ref={tagAnchor} position={[0, 2.5, 0]} visible={false}>
        {labelsReady && (
          <Html center zIndexRange={[13, 0]} style={{ pointerEvents: 'none' }}>
            <div ref={tag} className="distress-tag" style={{ display: 'none' }}>
              <span className="distress-title">
                <i className="distress-dot" aria-hidden="true" />
                <b ref={title} />
              </span>
              <span ref={sub} className="distress-sub" />
              <span className="distress-prog" style={{ display: 'none' }}>
                <i ref={bar} />
              </span>
            </div>
          </Html>
        )}
      </group>
    </group>
  )
}

// ---------------------------------------------------------------- helper and tow lines

function RescueLines({ meta }) {
  const size = useThree((s) => s.size)
  const helper = useMemo(() => new FatPolyline(LINE_SEGS + 1, { color: meta.color, width: 2.4, opacity: 0.95, dashed: true, dashSize: 0.7, gapSize: 0.45 }), [meta.color])
  const tow = useMemo(() => new FatPolyline(TOW_SEGS + 1, { color: '#ffd38a', width: 3.2, opacity: 1 }), [])
  const hx = useMemo(() => new Float32Array((LINE_SEGS + 1) * 3), [])
  const tx = useMemo(() => new Float32Array((TOW_SEGS + 1) * 3), [])
  useEffect(() => () => (helper.dispose(), tow.dispose()), [helper, tow])
  useEffect(() => {
    helper.setResolution(size.width, size.height)
    tow.setResolution(size.width, size.height)
  }, [helper, tow, size])
  useEffect(() => {
    helper.material.depthTest = false
  }, [helper])

  useFrame((_, dt) => {
    const s = simStore.get()
    const rs = helpingId(s.rescues, meta.id)
    const me = roverPoses.get(meta.id)
    const other = rs ? roverPoses.get(rs.rover) : null
    const r = s.rovers[meta.id]
    // Helper on the way: a dashed line in its colour, draped on the ground, crawling towards
    // the rover in trouble.
    if (rs && rs.phase === 'en_route' && me && other) {
      const ax = me.x
      const ay = -me.z
      const bx = other.x
      const by = -other.z
      let h = me.y
      for (let i = 0; i <= LINE_SEGS; i++) {
        const u = i / LINE_SEGS
        const x = ax + (bx - ax) * u
        const y = ay + (by - ay) * u
        h = groundAt(x, y, h)
        hx[i * 3] = x
        hx[i * 3 + 1] = h + 0.45
        hx[i * 3 + 2] = -y
      }
      helper.set(hx, LINE_SEGS + 1)
      if (!reducedMotion) helper.material.dashOffset -= dt * 1.6
    } else helper.object.visible = false
    // Towing: a sagging tow line between the two rovers at hitch height.
    const towing = (r?.state === 'towing' || rs?.phase === 'towing') && me && other
    if (towing) {
      for (let i = 0; i <= TOW_SEGS; i++) {
        const u = i / TOW_SEGS
        tx[i * 3] = me.x + (other.x - me.x) * u
        tx[i * 3 + 1] = me.y + (other.y - me.y) * u + 0.42 - 0.18 * Math.sin(Math.PI * u)
        tx[i * 3 + 2] = me.z + (other.z - me.z) * u
      }
      tow.set(tx, TOW_SEGS + 1)
    } else tow.object.visible = false
  })

  return (
    <group>
      <primitive object={helper.object} renderOrder={7} />
      <primitive object={tow.object} renderOrder={7} />
    </group>
  )
}

// ---------------------------------------------------------------- wrecks

const wreckMats = {
  disc: new THREE.MeshBasicMaterial({ color: '#0b0d12', transparent: true, opacity: 0.45, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }),
  ring: new THREE.MeshBasicMaterial({ color: '#9aa4b4', transparent: true, opacity: 0.8, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 }),
  cross: new THREE.MeshBasicMaterial({ color: '#9aa4b4', transparent: true, opacity: 0.6, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 }),
}

function WreckMark({ wreck, name, showLabel }) {
  const group = useRef()
  const labelsReady = useThree((s) => !!s.events.connected)
  useFrame(() => {
    const g = group.current
    if (!g) return
    // Under the rover while it is there, else where the sim says the wreck is.
    const pose = wreck.rover ? roverPoses.get(wreck.rover) : null
    const near = pose && Math.hypot(pose.x - wreck.x, -pose.z - wreck.y) < 3
    if (near) g.position.set(pose.x, pose.y + 0.12, pose.z)
    else g.position.set(wreck.x, groundAt(wreck.x, wreck.y, 0) + 0.12, -wreck.y)
  })
  return (
    <group ref={group}>
      <mesh geometry={geo.disc} material={wreckMats.disc} scale={1.9} renderOrder={3} />
      <mesh geometry={geo.ring} material={wreckMats.ring} scale={1.9} renderOrder={3} />
      <mesh geometry={geo.cross} material={wreckMats.cross} rotation={[0, Math.PI / 4, 0]} renderOrder={3} />
      <mesh geometry={geo.cross} material={wreckMats.cross} rotation={[0, -Math.PI / 4, 0]} renderOrder={3} />
      {labelsReady && showLabel && (
        <Html position={[0, 1.6, 0]} center zIndexRange={[11, 0]} style={{ pointerEvents: 'none' }}>
          <div className="wreck-tag">Wreck{name ? ` · ${name}` : ''}</div>
        </Html>
      )}
    </group>
  )
}

function Wrecks() {
  const wrecks = useSim((s) => s.wrecks)
  const meta = useSim((s) => s.roverMeta)
  // A rover still drawn at its wreck carries the label on its own name tag ("Wreck").
  const present = useSim((s) => s.wrecks.map((w) => (w.rover && s.rovers[w.rover] ? 1 : 0)).join(''))
  return (
    <group>
      {wrecks.map((w, i) => (
        <WreckMark key={`${w.rover ?? 'w'}:${i}`} wreck={w} name={meta.find((m) => m.id === w.rover)?.name ?? w.rover} showLabel={present[i] !== '1'} />
      ))}
    </group>
  )
}

export function Rescue() {
  const meta = useSim((s) => s.roverMeta)
  const gen = useSim((s) => s.gen)
  const v9 = useSim((s) => s.v9)
  if (!v9) return null
  return (
    <group>
      {meta.map((m) => (
        <group key={`${gen}:${m.id}`}>
          <HitFx meta={m} />
          <Distress meta={m} />
          <RescueLines meta={m} />
        </group>
      ))}
      <Wrecks key={gen} />
    </group>
  )
}
