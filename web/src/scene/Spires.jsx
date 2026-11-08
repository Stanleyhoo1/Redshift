import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { appStore, nowSeconds, simStore, useSim } from '../state/store.js'
import { simTimeNow } from '../state/simState.js'
import { SPIRE } from '../lib/theme.js'
import { glowTexture } from './textures.js'
import { groundAt, hashStr, knownAt } from './ground.js'
import { mulberry } from './proceduralTextures.js'
import { reducedMotion } from './sceneConstants.js'
import { skyEnvironment } from './sky.js'

// Alien crystal spires (PROTOCOL.md version 4, `features.spires`). One instanced crystal
// per spire plus a cluster of shards at its foot, a soft glow at the tip and a pool of its
// light on the ground, so the whole field is a handful of draw calls. The crystal is
// glassy: it reflects the hazy sky on its facets (brighter at grazing angles, fresnel) and
// glows from inside, strongest towards the tip. A spire shows once the fleet knows the
// ground under it (it grows out of the ground as the terrain reveals) or with Reality on.
// A `spire_fall` hazard tips it over from standing to lying between its base and tip in
// about 1.5 s, with a dust puff along its length when it lands.

const MAX = 96
const SHARDS = 6
const FALL_S = 1.5
const GROW_S = 0.8

// A hexagonal crystal from y = 0 to 1, radius 1 at its foot: a slightly tapering prism
// with a pointed termination.
function crystalGeometry() {
  const pts = [
    [0.0, 0],
    [0.92, 0],
    [1.0, 0.03],
    [0.86, 0.74],
    [0.8, 0.79],
    [0.0, 1],
  ].map(([r, y]) => new THREE.Vector2(r, y))
  const g = new THREE.LatheGeometry(pts, 6)
  g.computeVertexNormals()
  return g
}

export const crystalGeo = crystalGeometry()
const geo = crystalGeo

const crystalV = /* glsl */ `
varying float vLocalY;
`
const crystalF = /* glsl */ `
uniform vec3 uGlow;
uniform vec3 uRim;
uniform float uGlowAmt;
varying float vLocalY;
`
// Inner light (seen most where a facet faces you) and a fresnel rim, added as emission.
const crystalEmit = /* glsl */ `
  {
    vec3 vd = normalize(vViewPosition);
    float nv = clamp(abs(dot(normal, vd)), 0.0, 1.0);
    // Light caught inside differs facet by facet, like a gem.
    vec3 rf = reflect(-vd, normal);
    float facet = 0.55 + 0.45 * sin(dot(rf, vec3(6.1, 3.7, 4.9)) + vLocalY * 3.0);
    float inner = (0.12 + 0.88 * pow(clamp(vLocalY, 0.0, 1.0), 1.8)) * (0.25 + 0.75 * nv) * facet;
    float rim = pow(1.0 - nv, 3.0);
#if defined( USE_INSTANCING_COLOR ) || defined( USE_COLOR )
    vec3 tint = vColor.rgb;
#else
    vec3 tint = vec3(1.0);
#endif
    totalEmissiveRadiance += uGlow * inner * uGlowAmt * tint + uRim * rim * 0.55;
  }
`

function crystalMaterial(glowAmt) {
  const m = new THREE.MeshStandardMaterial({
    color: '#1d1238',
    roughness: 0.08,
    metalness: 0.15,
    flatShading: true,
    envMap: skyEnvironment(),
    envMapIntensity: 2.4,
    vertexColors: false,
  })
  m.userData.realistic = true
  const uniforms = {
    uGlow: { value: new THREE.Color(SPIRE.glow).multiplyScalar(0.9) },
    uRim: { value: new THREE.Color(SPIRE.body) },
    uGlowAmt: { value: glowAmt },
  }
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${crystalV}`)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vLocalY = position.y;')
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${crystalF}`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n${crystalEmit}`)
  }
  m.customProgramCacheKey = () => 'redshift-crystal-1'
  return m
}

// vColor is the instance colour (three defines it for instanced meshes with colours).
const bodyMat = crystalMaterial(1)
const shardMat = crystalMaterial(0.75)
const poolGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2)
const poolMat = new THREE.MeshBasicMaterial({ color: SPIRE.glow, map: glowTexture(), transparent: true, opacity: 0.38, depthWrite: false, blending: THREE.AdditiveBlending })

const _m = new THREE.Matrix4()
const _q = new THREE.Quaternion()
const _q2 = new THREE.Quaternion()
const _qy = new THREE.Quaternion()
const _p = new THREE.Vector3()
const _s = new THREE.Vector3()
const _d = new THREE.Vector3()
const UP = new THREE.Vector3(0, 1, 0)
const ID = new THREE.Quaternion()
const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0)

// When this client first saw each fall (shared with the dust puffs), keyed by gen:spire.
const fallStart = new Map()

// Test aid (?debug only, see Scene.jsx): hold every fall this client saw at `hold`
// seconds after it started, so a screenshot can catch it mid-fall on a slow machine.
export const fallDebug = { hold: null }
const fallAge = (start, now) => (fallDebug.hold !== null && start > -1e8 ? fallDebug.hold : now - start)

function fallStartFor(gen, key, fall) {
  const k = `${gen}:${key}`
  let t = fallStart.get(k)
  if (t === undefined) {
    const now = nowSeconds()
    const fresh = !reducedMotion && simTimeNow(simStore.get(), now) - fall.t < 3
    t = fresh ? now : -1e9
    fallStart.set(k, t)
    if (fallStart.size > 400) fallStart.delete(fallStart.keys().next().value)
  }
  return t
}

// Every spire to draw: the features, plus spires that only appear in a spire_fall hazard.
function spireList(features, fallen) {
  const out = features.spires.map((s) => ({ ...s, fall: fallen[s.id] || null }))
  for (const [id, f] of Object.entries(fallen)) {
    if (out.some((s) => s.id === id)) continue
    out.push({ id, x: f.x0, y: f.y0, height: Math.hypot(f.x1 - f.x0, f.y1 - f.y0), radius: 0.36, standing: false, fall: f })
  }
  return out.slice(0, MAX)
}

// Where a spire lies once it has fallen: the hazard's tip, or (no hazard seen, e.g. after a
// late join) a direction that stays the same for this spire.
function tipOf(s) {
  if (s.fall) return [s.fall.x1, s.fall.y1]
  const a = (hashStr(s.id) % 628) / 100
  return [s.x + Math.cos(a) * s.height, s.y + Math.sin(a) * s.height]
}

export function Spires({ reality }) {
  const features = useSim((s) => s.features)
  const fallen = useSim((s) => s.fallen)
  const gen = useSim((s) => s.gen)
  const list = useMemo(() => spireList(features, fallen), [features, fallen])
  const body = useRef()
  const shards = useRef()
  const pools = useRef()
  const glowPos = useMemo(() => new Float32Array(MAX * 3), [])
  const glow = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(glowPos, 3).setUsage(THREE.DynamicDrawUsage))
    g.setDrawRange(0, 0)
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6)
    return g
  }, [glowPos])
  const glowMat = useMemo(
    () => new THREE.PointsMaterial({ map: glowTexture(), color: SPIRE.glow, size: 1.7, sizeAttenuation: true, transparent: true, opacity: 0.7, depthWrite: false, blending: THREE.AdditiveBlending }),
    [],
  )
  useEffect(() => () => (glow.dispose(), glowMat.dispose()), [glow, glowMat])
  // Per spire: when it became visible, its yaw and shard layout. Reset on a new list.
  const st = useRef({ list: null, shown: new Map(), version: -1, reality: null, busy: true, openedAt: nowSeconds() })
  // A cluster at the foot: shards of varied size leaning out at varied angles.
  const shardLayout = useMemo(
    () =>
      list.map((s) => {
        const rand = mulberry(hashStr(s.id))
        const a0 = rand() * Math.PI * 2
        return Array.from({ length: SHARDS }, (_, k) => {
          const big = k < 2
          return {
            a: a0 + (k / SHARDS) * Math.PI * 2 + (rand() - 0.5) * 0.8,
            dist: big ? 1.05 + rand() * 0.3 : 1.2 + rand() * 0.9,
            tilt: big ? 0.18 + rand() * 0.25 : 0.35 + rand() * 0.5,
            scale: big ? 0.32 + rand() * 0.2 : 0.1 + rand() * 0.2,
            width: big ? 0.55 + rand() * 0.15 : 0.3 + rand() * 0.25,
          }
        })
      }),
    [list],
  )
  // Instance colours (tint of the inner glow): set before the first frame, so the material
  // compiles with them once.
  useLayoutEffect(() => {
    const c = new THREE.Color()
    const rand = mulberry(5)
    for (let k = 0; k < MAX; k++) {
      body.current?.setColorAt(k, c.setHSL(0.72 + (rand() - 0.5) * 0.06, 0.6, 0.45 + rand() * 0.15).multiplyScalar(2))
      pools.current?.setColorAt(k, c.set(1, 1, 1))
    }
    for (let k = 0; k < MAX * SHARDS; k++) shards.current?.setColorAt(k, c.setHSL(0.73 + (rand() - 0.5) * 0.08, 0.6, 0.38 + rand() * 0.2).multiplyScalar(2))
    for (const m of [body.current, shards.current, pools.current]) if (m?.instanceColor) m.instanceColor.needsUpdate = true
  }, [])

  useFrame(() => {
    const b = body.current
    const sh = shards.current
    const pl = pools.current
    if (!b || !sh || !pl) return
    const now = nowSeconds()
    const s = st.current
    const T = simStore.get().terrain
    const version = T ? T.version : 0
    const truthReady = !!appStore.get().truth?.heights
    if (s.list !== list) {
      s.list = list
      s.busy = true
      if (s.gen !== gen) {
        s.shown = new Map()
        s.openedAt = now
      }
      s.gen = gen
    }
    // Nothing moving and nothing new to reveal: skip the matrix work.
    if (!s.busy && version === s.version && reality === s.reality && truthReady === s.truthReady) return
    s.version = version
    s.reality = reality
    s.truthReady = truthReady
    let busy = false
    let glows = 0
    for (let k = 0; k < list.length; k++) {
      const sp = list[k]
      const lying = !sp.standing || !!sp.fall
      const [tx, ty] = lying ? tipOf(sp) : [sp.x, sp.y]
      const visible = reality || knownAt(sp.x, sp.y) || (lying && (knownAt(tx, ty) || knownAt((sp.x + tx) / 2, (sp.y + ty) / 2)))
      if (!visible) {
        b.setMatrixAt(k, HIDDEN)
        pl.setMatrixAt(k, HIDDEN)
        for (let q = 0; q < SHARDS; q++) sh.setMatrixAt(k * SHARDS + q, HIDDEN)
        continue
      }
      let shownAt = s.shown.get(sp.id)
      if (shownAt === undefined) {
        // Already known when the view opened: no growing.
        shownAt = reducedMotion || now - s.openedAt < 1.2 ? -1e9 : now
        s.shown.set(sp.id, shownAt)
      }
      const grow = Math.min(1, (now - shownAt) / GROW_S)
      const g = 1 - Math.pow(1 - grow, 3)
      if (grow < 1) busy = true
      const gz = groundAt(sp.x, sp.y)
      _qy.setFromAxisAngle(UP, (hashStr(sp.id) % 314) / 100)
      // Fall: from upright to the direction of the tip, accelerating, with a small bounce.
      if (lying) {
        const start = sp.fall ? fallStartFor(gen, sp.id, sp.fall) : -1e9
        const age = fallAge(start, now)
        const p = Math.min(1, Math.max(0, age / FALL_S))
        const tz = groundAt(tx, ty, gz) + sp.radius * 0.8
        _d.set(tx - sp.x, tz - gz, -(ty - sp.y)).normalize()
        _q2.setFromUnitVectors(UP, _d)
        let e = p * p
        if (age > FALL_S && age < FALL_S + 0.45) e = 1 - 0.05 * Math.sin(((age - FALL_S) / 0.45) * Math.PI)
        if (age < FALL_S + 0.5 || fallDebug.hold !== null) busy = true
        _q.copy(ID).slerp(_q2, e)
      } else {
        _q.copy(ID)
      }
      _q.multiply(_qy)
      _p.set(sp.x, gz - 0.05, -sp.y)
      _s.set(sp.radius * (0.4 + 0.6 * g), sp.height * g, sp.radius * (0.4 + 0.6 * g))
      _m.compose(_p, _q, _s)
      b.setMatrixAt(k, _m)
      // Glowing tip while standing, and the crystal's light pooled on the ground.
      if (!lying) {
        glowPos[glows * 3] = sp.x
        glowPos[glows * 3 + 1] = gz + sp.height * g * 0.9
        glowPos[glows * 3 + 2] = -sp.y
        glows++
      }
      const pr = (sp.radius * 7 + sp.height * 0.6) * g * (lying ? 0.6 : 1)
      _m.compose(_p.set(sp.x, gz + 0.06, -sp.y), ID, _s.set(pr, 1, pr))
      pl.setMatrixAt(k, _m)
      // Shards at the foot stay put when the spire falls.
      const lay = shardLayout[k]
      for (let q = 0; q < SHARDS; q++) {
        const L = lay[q]
        _d.set(Math.cos(L.a), 0, Math.sin(L.a))
        _q2.setFromAxisAngle(_s.set(_d.z, 0, -_d.x), L.tilt) // lean outwards
        _qy.setFromAxisAngle(UP, L.a * 3.1)
        _q2.multiply(_qy)
        const rx = sp.x + _d.x * sp.radius * L.dist
        const ry = sp.y - _d.z * sp.radius * L.dist
        _p.set(rx, groundAt(rx, ry, gz) - 0.06, -ry)
        const hs = sp.height * L.scale * g
        const ws = sp.radius * L.width * g
        _m.compose(_p, _q2, _s.set(ws, hs, ws))
        sh.setMatrixAt(k * SHARDS + q, _m)
      }
    }
    b.count = list.length
    sh.count = list.length * SHARDS
    pl.count = list.length
    b.instanceMatrix.needsUpdate = true
    sh.instanceMatrix.needsUpdate = true
    pl.instanceMatrix.needsUpdate = true
    glow.setDrawRange(0, glows)
    glow.attributes.position.needsUpdate = true
    s.busy = busy
  })

  return (
    <group>
      <instancedMesh ref={pools} args={[poolGeo, poolMat, MAX]} frustumCulled={false} count={0} renderOrder={2} />
      <instancedMesh ref={body} args={[geo, bodyMat, MAX]} frustumCulled={false} count={0} castShadow receiveShadow />
      <instancedMesh ref={shards} args={[geo, shardMat, MAX * SHARDS]} frustumCulled={false} count={0} castShadow receiveShadow />
      <points geometry={glow} material={glowMat} frustumCulled={false} renderOrder={4} />
      {Object.entries(fallen).map(([id, f]) => (
        <FallDust key={`${gen}:${id}:${f.hazard}`} id={id} fall={f} gen={gen} />
      ))}
    </group>
  )
}

const PUFFS = 8
const PUFF_S = 2.2 // seconds the dust hangs in the air
const PUFF_COLOR = '#d8bf98' // pale dust, so it reads on every slope colour

// Dust thrown up along the spire where it lands (only for falls this client saw happen).
function FallDust({ id, fall, gen }) {
  const sprites = useRef([])
  const ring = useRef()
  const mats = useMemo(
    () => ({
      puff: Array.from({ length: PUFFS }, () => new THREE.SpriteMaterial({ map: glowTexture(), color: PUFF_COLOR, transparent: true, depthWrite: false, opacity: 0 })),
      ring: new THREE.MeshBasicMaterial({ color: PUFF_COLOR, transparent: true, depthWrite: false, opacity: 0 }),
    }),
    [],
  )
  const ringGeo = useMemo(() => new THREE.RingGeometry(0.8, 1, 40).rotateX(-Math.PI / 2), [])
  useEffect(() => () => (mats.puff.forEach((m) => m.dispose()), mats.ring.dispose(), ringGeo.dispose()), [mats, ringGeo])

  useFrame(() => {
    const start = fallStartFor(gen, id, fall)
    const pa = fallAge(start, nowSeconds()) - FALL_S * 0.85
    const live = pa > 0 && pa < PUFF_S
    const k = live ? pa / PUFF_S : 0
    const ease = 1 - Math.pow(1 - k, 2.2) // billows out fast, then hangs
    for (let q = 0; q < PUFFS; q++) {
      const sp = sprites.current[q]
      if (!sp) continue
      sp.visible = live
      if (!live) continue
      // Spread along the spire, the tip end (which lands hardest) a little bigger.
      const f = (q + 0.5) / PUFFS
      const side = (q % 2 ? 1 : -1) * 0.35 * ease
      const dx = fall.x1 - fall.x0
      const dy = fall.y1 - fall.y0
      const len = Math.hypot(dx, dy) || 1
      const x = fall.x0 + dx * f - (dy / len) * side
      const y = fall.y0 + dy * f + (dx / len) * side
      sp.position.set(x, groundAt(x, y) + 0.3 + ease * (0.6 + 0.5 * f), -y)
      sp.scale.setScalar((1.1 + f * 0.9) * (1 + ease * 2.4))
      mats.puff[q].opacity = 0.9 * Math.pow(1 - k, 1.3) * (0.65 + 0.35 * f)
    }
    const r = ring.current
    if (r) {
      r.visible = live
      if (live) {
        r.position.set(fall.x1, groundAt(fall.x1, fall.y1) + 0.08, -fall.y1)
        r.scale.setScalar(0.8 + ease * 3.6)
        mats.ring.opacity = 0.75 * (1 - k)
      }
    }
  })

  return (
    <group>
      {mats.puff.map((m, q) => (
        <sprite key={q} ref={(el) => (sprites.current[q] = el)} material={m} visible={false} renderOrder={6} />
      ))}
      <mesh ref={ring} geometry={ringGeo} material={mats.ring} visible={false} renderOrder={6} />
    </group>
  )
}
