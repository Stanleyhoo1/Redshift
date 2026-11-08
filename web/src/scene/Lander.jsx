import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '../globe/PlaneHtml.jsx'
import * as THREE from 'three'
import { simStore, useSim } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { SIGNAL } from '../lib/theme.js'
import { glowTexture } from './textures.js'
import { reducedMotion } from './sceneConstants.js'
import { boxUV, grime, hardwareMaterials, merge, place, strut } from './hardware.js'
import { padTexture } from './proceduralTextures.js'
import { DrapedDisc, useDrape } from './drape.js'

// The lander: a compact descent stage on splayed legs in the middle of its charging pad.
// The rovers park on the pad round it (the sim spreads their docks evenly on a circle), so
// the legs and solar wings stand in the gaps between the docks: a tripod for three rovers,
// four legs otherwise. Gold thermal blankets, white panels, a dish antenna, a slow beacon,
// and a draped pad with charging markings. The cache count stays in its label.

const V = (x, y, z) => new THREE.Vector3(x, y, z)

// Directions (three.js x, z) between the sim's dock spots for n rovers.
function legAngles(n) {
  const k = n === 3 || n === 4 ? n : 4
  const off = n === 3 || n === 4 ? Math.PI / k : Math.PI / 4
  return Array.from({ length: k }, (_, i) => off + (i / k) * Math.PI * 2)
}

function buildLander(n) {
  const paint = []
  const gold = []
  const dark = []
  const metal = []
  const solar = []
  const dir = (a, r, y) => V(Math.cos(a) * r, y, -Math.sin(a) * r)
  // Descent stage: octagonal body, gold blanket below, engine bell underneath.
  paint.push(boxUV(new THREE.CylinderGeometry(0.82, 0.86, 0.62, 8).translate(0, 1.27, 0)))
  gold.push(new THREE.CylinderGeometry(0.88, 0.8, 0.42, 8).translate(0, 0.75, 0))
  dark.push(new THREE.CylinderGeometry(0.2, 0.34, 0.3, 18, 1, true).translate(0, 0.42, 0))
  dark.push(new THREE.CylinderGeometry(0.96, 0.92, 0.06, 8).translate(0, 1.61, 0))
  // Deck instruments.
  paint.push(place(boxUV(new THREE.BoxGeometry(0.34, 0.22, 0.26)), -0.32, 1.75, 0.28, 0, 0.3, 0))
  paint.push(place(boxUV(new THREE.BoxGeometry(0.22, 0.14, 0.2)), 0.18, 1.71, -0.4, 0, -0.2, 0))
  // Dish antenna on a short mast, tipped up towards the sky.
  const dish = new THREE.LatheGeometry(
    Array.from({ length: 9 }, (_, k) => new THREE.Vector2((k / 8) * 0.38, ((k / 8) * (k / 8)) * 0.12)),
    28,
  )
  const dishM = new THREE.Matrix4().compose(V(0.36, 2.14, 0.18), new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.55, 0.9, 0.35)), V(1, 1, 1))
  paint.push(boxUV(dish.clone().applyMatrix4(dishM)))
  dark.push(strut(V(0.36, 1.62, 0.18), V(0.36, 2.12, 0.18), 0.035))
  dark.push(strut(V(0, 0.02, 0).applyMatrix4(dishM), V(0, 0.3, 0).applyMatrix4(dishM), 0.01, 0.008, 6))
  metal.push(new THREE.CylinderGeometry(0.03, 0.03, 0.07, 10).applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0.31, 0)).applyMatrix4(dishM))
  // Beacon mast.
  dark.push(strut(V(-0.42, 1.62, -0.3), V(-0.42, 2.5, -0.3), 0.018, 0.012, 6))
  // Legs (A-frame plus shock strut) and footpads, solar wings above them.
  for (const a of legAngles(n)) {
    const foot = dir(a, 1.62, 0.06)
    const t = V(-Math.sin(a), 0, -Math.cos(a)) // tangent
    const hi = dir(a, 0.84, 1.0)
    dark.push(strut(hi, foot, 0.045, 0.04))
    metal.push(strut(hi.clone().lerp(foot, 0.35), hi.clone().lerp(foot, 0.62), 0.058, 0.058, 10))
    for (const s of [1, -1]) dark.push(strut(dir(a, 0.84, 0.58).addScaledVector(t, s * 0.34), foot.clone().add(V(0, 0.04, 0)), 0.025))
    metal.push(new THREE.CylinderGeometry(0.2, 0.25, 0.06, 18).translate(foot.x, 0.03, foot.z))
    // Wing: frame and cells, drooping a little at the tip.
    const wing = new THREE.Group()
    const mid = dir(a, 1.62, 1.86)
    const cells = boxUV(new THREE.BoxGeometry(1.36, 0.022, 0.62))
    const frame = new THREE.BoxGeometry(1.42, 0.03, 0.68)
    const m = new THREE.Matrix4().compose(mid.clone().add(V(0, -0.012, 0)), new THREE.Quaternion().setFromEuler(new THREE.Euler(0, a, -0.14)), V(1, 1, 1))
    wing.updateMatrix()
    solar.push(cells.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0.012, 0)).applyMatrix4(m))
    dark.push(frame.applyMatrix4(m))
    dark.push(strut(dir(a, 0.8, 1.6), dir(a, 1.25, 1.84), 0.03))
  }
  for (const g of paint) grime(g, 0.6, 1.4, '#9c8471', 0.4)
  return {
    paint: merge(paint),
    gold: merge(gold),
    dark: merge(dark),
    metal: merge(metal),
    solar: merge(solar),
  }
}

export function Lander() {
  const lander = useSim((s) => s.lander)
  const roverCount = useSim((s) => s.roverMeta.length)
  const labelsReady = useThree((s) => !!s.events.connected) // see Rovers.jsx
  const cached = useSim((s) => (Number.isFinite(s.fleet.samples_cached) ? s.fleet.samples_cached : null)) // v4b
  const group = useRef()
  const h = useRef(null)
  const R = lander?.radius ?? 2.5
  const M = hardwareMaterials()
  const parts = useMemo(() => buildLander(roverCount), [roverCount])
  useEffect(() => () => Object.values(parts).forEach((g) => g.dispose()), [parts])

  const fx = useMemo(
    () => ({
      edge: new THREE.RingGeometry(R * 0.985, R * 1.02, 96).rotateX(-Math.PI / 2),
      edgeMat: new THREE.MeshBasicMaterial({ color: SIGNAL, transparent: true, opacity: 0.55, depthWrite: false }),
      bulb: new THREE.SphereGeometry(0.06, 12, 8),
      bulbMat: new THREE.MeshBasicMaterial({ color: '#ff5a4a', transparent: true }),
      beamMat: new THREE.SpriteMaterial({ map: glowTexture(), color: '#ff6a50', transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
    }),
    [R],
  )
  useEffect(() => () => Object.values(fx).forEach((p) => p.dispose?.()), [fx])

  // The charging pad, draped on the ground.
  const pad = useMemo(() => new DrapedDisc(10, 72, { uvMode: 'disc', normals: true }), [])
  const padMat = useMemo(() => {
    const m = new THREE.MeshStandardMaterial({ map: padTexture(), transparent: true, roughness: 0.82, metalness: 0.05, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, depthWrite: false })
    m.userData.realistic = true
    return m
  }, [])
  useEffect(() => () => (pad.dispose(), padMat.dispose()), [pad, padMat])
  useDrape(() => {
    if (lander) pad.update(lander.x, lander.y, R * 1.04, 0.035)
  }, [lander?.x, lander?.y, R])

  const bulb = useRef()
  const beam = useRef()
  const tag = useRef()
  const tick = useRef(0)
  useFrame(({ clock }) => {
    if (!lander || !group.current) return
    const g = sampleGroundNear(simStore.get().terrain, lander.x, lander.y, 10) ?? 0
    h.current = h.current === null ? g : h.current + (g - h.current) * 0.1
    group.current.position.set(lander.x, h.current, -lander.y)
    const blink = reducedMotion ? 1 : Math.pow(0.5 + 0.5 * Math.sin(clock.elapsedTime * 2.4), 6)
    fx.bulbMat.opacity = 0.35 + 0.65 * blink
    fx.beamMat.opacity = 0.15 + 0.75 * blink
    if (beam.current) beam.current.scale.setScalar(0.9 + 0.5 * blink)
    // Docked rovers' tags land on the lander tag on screen: slide it down out from under
    // them (checked a few times a second, it reads layout).
    const el = tag.current
    if (el && el.parentElement && clock.elapsedTime - tick.current > 0.25) {
      tick.current = clock.elapsedTime
      const tr = el.parentElement.getBoundingClientRect()
      let push = 0
      for (const r of document.querySelectorAll('.rover-tag')) {
        const b = r.getBoundingClientRect()
        if (b.width && b.right > tr.left && b.left < tr.right && b.bottom > tr.top && b.top < tr.bottom) push = Math.max(push, Math.round(b.bottom - tr.top + 2))
      }
      if (push !== (Number(el.dataset.push) || 0)) {
        el.dataset.push = String(push)
        el.style.transform = push ? `translateY(${push}px)` : ''
      }
    }
  })

  if (!lander) return null
  return (
    <group>
      <mesh geometry={pad.geometry} material={padMat} renderOrder={1} receiveShadow />
      <group ref={group}>
        <mesh geometry={fx.edge} material={fx.edgeMat} position={[0, 0.09, 0]} renderOrder={3} />
        <mesh geometry={parts.paint} material={M.paint} castShadow receiveShadow />
        <mesh geometry={parts.gold} material={M.gold} castShadow />
        <mesh geometry={parts.dark} material={M.dark} castShadow receiveShadow />
        <mesh geometry={parts.metal} material={M.metal} castShadow />
        <mesh geometry={parts.solar} material={M.solar} castShadow receiveShadow />
        <mesh ref={bulb} geometry={fx.bulb} material={fx.bulbMat} position={[-0.42, 2.53, -0.3]} />
        <sprite ref={beam} material={fx.beamMat} position={[-0.42, 2.53, -0.3]} scale={1} renderOrder={5} />
        {/* Below the pad (on screen), so it does not collide with the tags of docked rovers above them. */}
        {labelsReady && (
          <Html position={[0, 0.3, R + 1.1]} center zIndexRange={[11, 0]} style={{ pointerEvents: 'none' }}>
            <div className="lander-tag" ref={tag}>
              Lander
              {/* An empty cache shows just "Lander" (an empty tube next to a 0 read as "0 0"); the
                  sidebar's science score already says "Cache empty". */}
              {cached !== null && cached > 0 && (
                <span className="lander-cache" data-some={cached > 0 || undefined} title="Samples in the lander's cache">
                  <i aria-hidden="true" />
                  {cached}
                  <span className="lander-cache-word"> cached</span>
                </span>
              )}
            </div>
          </Html>
        )}
      </group>
    </group>
  )
}
