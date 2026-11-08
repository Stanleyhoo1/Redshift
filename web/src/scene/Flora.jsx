import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { groundAt, knownAt } from './ground.js'
import { reducedMotion } from './sceneConstants.js'

// v7 alien flora (PROTOCOL.md, `features.flora` and `fleet.flora_zones`). Fungal trees: a thin pale
// stalk under a wide violet or teal cap with a soft glowing rim. Bushes: a few glowing lime bulbs
// that pulse gently. Instanced, so a few draw calls in all. A plant shows once the fleet knows the
// ground under it, or with Reality on. Low graphics draws every other bush. Known plants get a
// dashed keep-out ring.

const MAX = 128
const BULBS = 3
const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0)
const VIOLET = new THREE.Color(0.56, 0.24, 0.86)
const TEAL = new THREE.Color(0.12, 0.74, 0.7)
const _m = new THREE.Matrix4()
const _p = new THREE.Vector3()
const _s = new THREE.Vector3()
const _q = new THREE.Quaternion()
const _c = new THREE.Color()

function Ring({ x, y, r }) {
  const geo = useMemo(() => {
    const pts = []
    for (let k = 0; k <= 64; k++) {
      const a = (k / 64) * Math.PI * 2
      const px = x + r * Math.cos(a)
      const py = y + r * Math.sin(a)
      pts.push(new THREE.Vector3(px, groundAt(px, py) + 0.15, -py))
    }
    return new THREE.BufferGeometry().setFromPoints(pts)
  }, [x, y, r])
  const ref = useRef()
  return (
    <line ref={ref} geometry={geo} onUpdate={(l) => l.computeLineDistances()}>
      <lineDashedMaterial color="#b6ff5a" dashSize={0.8} gapSize={0.6} transparent opacity={0.8} />
    </line>
  )
}

export function Flora({ reality }) {
  const plants = useSim((s) => s.features?.flora)
  const zones = useSim((s) => s.fleet?.flora_zones)
  const graphics = useGraphics()
  const stalk = useRef()
  const cap = useRef()
  const bulb = useRef()
  const bulbMat = useRef()
  const list = Array.isArray(plants) ? plants : []
  const geos = useMemo(() => ({
    stalk: new THREE.CylinderGeometry(0.12, 0.2, 1, 7),
    cap: new THREE.SphereGeometry(1, 14, 8, 0, Math.PI * 2, 0, Math.PI / 2),
    bulb: new THREE.SphereGeometry(1, 10, 8),
  }), [])

  useFrame(({ clock }) => {
    if (!stalk.current || !cap.current || !bulb.current) return
    let nb = 0
    for (let k = 0; k < MAX; k++) {
      const p = list[k]
      const show = p && (reality || knownAt(p.x, p.y)) && !(graphics === 'low' && p.kind === 'bush' && k % 2)
      if (!show) {
        stalk.current.setMatrixAt(k, HIDDEN)
        cap.current.setMatrixAt(k, HIDDEN)
        continue
      }
      const gz = groundAt(p.x, p.y)
      if (p.kind === 'tree') {
        _m.compose(_p.set(p.x, gz + p.height / 2, -p.y), _q.identity(), _s.set(1, p.height, 1))
        stalk.current.setMatrixAt(k, _m)
        _m.compose(_p.set(p.x, gz + p.height - 0.1, -p.y), _q.identity(), _s.set(p.radius, 0.42 * p.radius, p.radius))
        cap.current.setMatrixAt(k, _m)
        cap.current.setColorAt(k, _c.copy(VIOLET).lerp(TEAL, p.hue))
      } else {
        stalk.current.setMatrixAt(k, HIDDEN)
        cap.current.setMatrixAt(k, HIDDEN)
        for (let b = 0; b < BULBS && nb < MAX * BULBS; b++) {
          const a = p.hue * 6.28 + b * 2.1
          const s = 0.14 + 0.1 * p.radius
          _m.compose(_p.set(p.x + 0.6 * p.radius * Math.cos(a), gz + 0.6 * p.height + 0.2 * b * s, -(p.y + 0.6 * p.radius * Math.sin(a))), _q.identity(), _s.set(s, s, s))
          bulb.current.setMatrixAt(nb++, _m)
        }
      }
    }
    for (let k = nb; k < MAX * BULBS; k++) bulb.current.setMatrixAt(k, HIDDEN)
    stalk.current.instanceMatrix.needsUpdate = true
    cap.current.instanceMatrix.needsUpdate = true
    if (cap.current.instanceColor) cap.current.instanceColor.needsUpdate = true
    bulb.current.instanceMatrix.needsUpdate = true
    if (bulbMat.current) bulbMat.current.emissiveIntensity = reducedMotion ? 1.2 : 1.0 + 0.5 * Math.sin(clock.elapsedTime * 1.6)
  })

  if (!list.length) return null
  return (
    <group>
      <instancedMesh ref={stalk} args={[geos.stalk, undefined, MAX]} frustumCulled={false}>
        <meshStandardMaterial color="#d8d2c4" roughness={0.8} />
      </instancedMesh>
      <instancedMesh ref={cap} args={[geos.cap, undefined, MAX]} frustumCulled={false}>
        <meshStandardMaterial color="#ffffff" emissive="#5a3a8a" emissiveIntensity={0.35} roughness={0.6} side={THREE.DoubleSide} />
      </instancedMesh>
      <instancedMesh ref={bulb} args={[geos.bulb, undefined, MAX * BULBS]} frustumCulled={false}>
        <meshStandardMaterial ref={bulbMat} color="#b8ff48" emissive="#9cff2a" emissiveIntensity={1.2} roughness={0.4} />
      </instancedMesh>
      {Array.isArray(zones) && zones.filter((z) => Array.isArray(z) && z.length >= 3 && z.every(Number.isFinite)).map((z, i) => (
        <Ring key={i} x={z[0]} y={z[1]} r={z[2]} />
      ))}
    </group>
  )
}
