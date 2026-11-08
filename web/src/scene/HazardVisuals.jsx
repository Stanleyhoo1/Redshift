import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { nowSeconds, simStore, useSim } from '../state/store.js'
import { simTimeNow } from '../state/simState.js'
import { DUST, SAND } from '../lib/theme.js'
import { FatPolyline } from './fatLine.js'
import { glowTexture, hatchTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'
import { mulberry } from './proceduralTextures.js'
import { boulderVariants, rockMaterial } from './rocks.js'
import { DrapedDisc, useDrape } from './drape.js'

// ---------------------------------------------------------------- shared helpers

const hashStr = (s) => [...String(s)].reduce((h, ch) => (Math.imul(h, 31) + ch.charCodeAt(0)) >>> 0, 7)

// ---------------------------------------------------------------- rockfall

const ROCK_VARIANTS = boulderVariants()
const rockMat = rockMaterial('boulder', { bump: 0.02, grain: 0.4 })
const ringGeo = new THREE.RingGeometry(0.82, 1, 40).rotateX(-Math.PI / 2)
const DROP_S = 0.9 // wall seconds for one boulder to fall
const DROP_H = 16 // metres it falls from

// True boulder positions from the hazard (PROTOCOL.md: [x, y, z_centre, radius]).
// A fresh rockfall drops in when the UI first sees it, with a dust puff on landing;
// one that started long before (a late join, a reconnect) is simply there.
function Rockfall({ hazard }) {
  const rocks = hazard.rocks
  const meshes = useRef([])
  const puffs = useRef([])
  const rings = useRef([])
  const start = useRef(null)
  // The hazard list is rebuilt on every tick: key the per-rock objects on id and count.
  const count = rocks.length
  const spin = useMemo(() => {
    const rand = mulberry(hashStr(hazard.id))
    return Array.from({ length: count }, () => [rand() * 6.28, rand() * 6.28, rand() * 6.28, Math.floor(rand() * ROCK_VARIANTS.length), 0.9 + rand() * 0.25])
  }, [hazard.id, count])
  const mats = useMemo(
    () =>
      Array.from({ length: count }, () => ({
        puff: new THREE.SpriteMaterial({ map: glowTexture(), color: DUST, transparent: true, depthWrite: false, opacity: 0 }),
        ring: new THREE.MeshBasicMaterial({ color: DUST, transparent: true, depthWrite: false, opacity: 0 }),
      })),
    [count],
  )
  useEffect(() => () => mats.forEach((m) => (m.puff.dispose(), m.ring.dispose())), [mats])

  useFrame(() => {
    const now = nowSeconds()
    if (!start.current) {
      const s = simStore.get()
      const fresh = simTimeNow(s, now) - hazard.t < 3 && !reducedMotion
      start.current = fresh ? now : -1e9
    }
    const age = now - start.current
    for (let i = 0; i < rocks.length; i++) {
      const m = meshes.current[i]
      if (!m) continue
      const [x, y, zc, r] = rocks[i]
      const delay = (i * 0.13) % 0.5
      const p = Math.max(0, Math.min(1, (age - delay) / DROP_S))
      const fall = 1 - p * p // accelerating, like a real fall
      const [rx, ry, rz, , squash] = spin[i]
      m.visible = age >= delay
      m.position.set(x, zc + fall * DROP_H, -y)
      m.rotation.set(rx + fall * 3, ry, rz + fall * 2)
      m.scale.set(r, r * squash, r)
      const pa = age - delay - DROP_S
      const live = pa > 0 && pa < 1.5
      const pf = puffs.current[i]
      const rg = rings.current[i]
      if (pf) {
        pf.visible = live
        if (live) {
          const k = pa / 1.5
          pf.position.set(x, zc + 0.2 + k * 0.8, -y)
          pf.scale.setScalar(r * (2 + k * 5))
          mats[i].puff.opacity = 0.75 * Math.pow(1 - k, 1.6)
        }
      }
      if (rg) {
        rg.visible = live
        if (live) {
          const k = pa / 1.5
          rg.position.set(x, zc - r * 0.45 + 0.08, -y)
          rg.scale.setScalar(r * (1.2 + k * 4))
          mats[i].ring.opacity = 0.7 * (1 - k)
        }
      }
    }
  })

  return (
    <group>
      {rocks.map((rk, i) => (
        <group key={i}>
          <mesh ref={(el) => (meshes.current[i] = el)} geometry={ROCK_VARIANTS[spin[i][3]]} material={rockMat} visible={false} castShadow receiveShadow />
          <sprite ref={(el) => (puffs.current[i] = el)} material={mats[i].puff} visible={false} renderOrder={6} />
          <mesh ref={(el) => (rings.current[i] = el)} geometry={ringGeo} material={mats[i].ring} visible={false} renderOrder={6} />
        </group>
      ))}
    </group>
  )
}

// ---------------------------------------------------------------- soft ground

const ZONE_SEGS = 56

// Soft ground the fleet believes in (fleet.soft_zones: found through wheel slip):
// a hatched sandy disc draped on the terrain, with an outline. Pulses when new.
function SoftZone({ zone, since }) {
  const [x, y, r] = zone
  const size = useThree((s) => s.size)
  const disc = useMemo(() => new DrapedDisc(7, ZONE_SEGS), [])
  const ring = useMemo(() => new Float32Array((ZONE_SEGS + 1) * 3), [])
  const outline = useMemo(() => new FatPolyline(ZONE_SEGS + 1, { color: SAND, width: 2.2, opacity: 0.95 }), [])
  const pulse = useMemo(() => new FatPolyline(ZONE_SEGS + 1, { color: SAND, width: 2, opacity: 0.9 }), [])
  const pulseXyz = useMemo(() => new Float32Array((ZONE_SEGS + 1) * 3), [])
  const mat = useMemo(() => {
    const tex = hatchTexture()
    return new THREE.MeshBasicMaterial({ color: SAND, map: tex, transparent: true, opacity: 0.62, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2 })
  }, [])
  // Zones already there when the view opened (a late join) do not pulse.
  const born = useRef(nowSeconds() - since > 1.5 ? nowSeconds() : -1e9)
  useEffect(
    () => () => {
      disc.dispose()
      outline.dispose()
      pulse.dispose()
      mat.dispose()
    },
    [disc, outline, pulse, mat],
  )
  useEffect(() => {
    outline.setResolution(size.width, size.height)
    pulse.setResolution(size.width, size.height)
  }, [outline, pulse, size])

  useDrape(() => {
    disc.update(x, y, r, 0.08, ring)
    outline.set(ring, ZONE_SEGS + 1)
  }, [x, y, r])

  useFrame(() => {
    const age = nowSeconds() - born.current
    if (age > 3 || reducedMotion) {
      pulse.object.visible = false
      return
    }
    // Two rings spreading out from the zone edge for the first few seconds.
    const k = (age % 1.5) / 1.5
    const grow = 1 + k * 0.9
    for (let s = 0; s <= ZONE_SEGS; s++) {
      pulseXyz[s * 3] = x + (ring[s * 3] - x) * grow
      pulseXyz[s * 3 + 1] = ring[s * 3 + 1] + 0.05
      pulseXyz[s * 3 + 2] = -y + (ring[s * 3 + 2] + y) * grow
    }
    pulse.set(pulseXyz, ZONE_SEGS + 1)
    pulse.material.opacity = 0.9 * (1 - k)
  })

  return (
    <group>
      <mesh geometry={disc.geometry} material={mat} renderOrder={3} />
      <primitive object={outline.object} renderOrder={3} />
      <primitive object={pulse.object} renderOrder={3} />
    </group>
  )
}

// The real sand patch (hidden from the rovers), drawn only with Reality on.
function SandTruth({ hazard }) {
  const size = useThree((s) => s.size)
  const disc = useMemo(() => new DrapedDisc(5, ZONE_SEGS), [])
  const ring = useMemo(() => new Float32Array((ZONE_SEGS + 1) * 3), [])
  const outline = useMemo(() => new FatPolyline(ZONE_SEGS + 1, { color: SAND, width: 2.2, dashed: true, dashSize: 0.45, gapSize: 0.35 }), [])
  const mat = useMemo(() => new THREE.MeshBasicMaterial({ color: SAND, transparent: true, opacity: 0.14, depthWrite: false, side: THREE.DoubleSide }), [])
  useEffect(
    () => () => {
      disc.dispose()
      outline.dispose()
      mat.dispose()
    },
    [disc, outline, mat],
  )
  useEffect(() => outline.setResolution(size.width, size.height), [outline, size])
  useDrape(() => {
    disc.update(hazard.x, hazard.y, hazard.radius, 0.12, ring)
    outline.set(ring, ZONE_SEGS + 1)
  }, [hazard.x, hazard.y, hazard.radius])
  return (
    <group>
      <mesh geometry={disc.geometry} material={mat} renderOrder={5} />
      <primitive object={outline.object} renderOrder={5} />
    </group>
  )
}

const okZone = (z) => Array.isArray(z) && z.length >= 3 && z.every((v, i) => i > 2 || Number.isFinite(v)) && z[2] > 0

export function HazardVisuals({ reality }) {
  const hazards = useSim((s) => s.hazards)
  const zones = useSim((s) => s.fleet.soft_zones)
  const gen = useSim((s) => s.gen)
  const since = useRef(nowSeconds()).current
  const list = Array.isArray(zones) ? zones.filter(okZone) : []
  return (
    <group>
      {hazards.map((h) => {
        if (h.type === 'rockfall' && h.rocks.length) return <Rockfall key={`${gen}:${h.id}`} hazard={h} />
        if (h.type === 'sand' && reality && Number.isFinite(h.x) && Number.isFinite(h.y) && h.radius > 0) return <SandTruth key={`${gen}:${h.id}`} hazard={h} />
        return null
      })}
      {list.map((z, i) => (
        <SoftZone key={`${gen}:${i}`} zone={z} since={since} />
      ))}
    </group>
  )
}
