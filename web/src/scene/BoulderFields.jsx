import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { groundAt, hashStr, knownAt } from './ground.js'
import { boulderVariants, rockMaterial } from './rocks.js'
import { mulberry } from './proceduralTextures.js'

// v7 boulder fields (PROTOCOL.md, `features.boulder_fields[].rocks`: [x, y, size]). Each boulder
// is one of the shared rock shapes in the rockfall look, instanced by shape (six draw calls in
// all), set on the ground round it, so it sits over the bump the fleet's map has there. A boulder
// shows once the fleet knows the ground under it, or with Reality on.

const VARIANTS = boulderVariants()
const rockMat = rockMaterial('boulder', { bump: 0.02, grain: 0.4 })
const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0)
const REFRESH_S = 0.4
const _m = new THREE.Matrix4()
const _p = new THREE.Vector3()
const _s = new THREE.Vector3()
const _q = new THREE.Quaternion()
const _e = new THREE.Euler()

// The ground round a boulder: the lowest of a ring of samples just outside it (the map has the
// boulder itself as a bump, so the centre sample sits on top of it).
function baseAt(x, y, r) {
  const z0 = groundAt(x, y)
  let z = Infinity
  for (let k = 0; k < 6; k++) {
    const a = (k / 6) * Math.PI * 2
    z = Math.min(z, groundAt(x + (r + 0.45) * Math.cos(a), y + (r + 0.45) * Math.sin(a), z0))
  }
  return Number.isFinite(z) ? z : z0
}

export function BoulderFields({ reality }) {
  const fields = useSim((s) => s.features?.boulder_fields)
  const meshes = useRef([])
  const last = useRef({ at: -1e9, reality: null, rocks: null, meshes: [] })
  // Every boulder with its shape, spin and a little size jitter, stable per field id.
  const rocks = useMemo(() => {
    const out = VARIANTS.map(() => [])
    if (!Array.isArray(fields)) return out
    for (const f of fields) {
      if (!f || !Array.isArray(f.rocks)) continue
      const rand = mulberry(hashStr(f.id))
      for (const r of f.rocks) {
        if (!Array.isArray(r) || r.length < 3 || !r.every(Number.isFinite)) continue
        const v = Math.floor(rand() * VARIANTS.length)
        out[v].push({ x: r[0], y: r[1], s: Math.max(0.12, r[2]), yaw: rand() * Math.PI * 2, tilt: (rand() - 0.5) * 0.3, k: 0.95 + rand() * 0.2 })
      }
    }
    return out
  }, [fields])
  const total = rocks.reduce((a, l) => a + l.length, 0)

  useFrame(({ clock }) => {
    const now = clock.elapsedTime
    const L = last.current
    // New meshes start with identity matrices and a new world moves the boulders: place them at
    // once, otherwise every REFRESH_S (the known ground grows slowly).
    const fresh = L.rocks !== rocks || meshes.current.some((m, v) => m && m !== L.meshes[v])
    if (!fresh && now - L.at < REFRESH_S && L.reality === reality) return
    L.at = now
    L.reality = reality
    L.rocks = rocks
    L.meshes = meshes.current.slice()
    for (let v = 0; v < VARIANTS.length; v++) {
      const mesh = meshes.current[v]
      const list = rocks[v]
      if (!mesh) continue
      for (let k = 0; k < list.length; k++) {
        const b = list[k]
        if (!(reality || knownAt(b.x, b.y))) {
          mesh.setMatrixAt(k, HIDDEN)
          continue
        }
        const s = b.s * b.k
        const gz = baseAt(b.x, b.y, b.s)
        _e.set(b.tilt, b.yaw, b.tilt * 0.6)
        _m.compose(_p.set(b.x, gz + 0.3 * s, -b.y), _q.setFromEuler(_e), _s.set(s * 1.05, s * 1.1, s * 1.05))
        mesh.setMatrixAt(k, _m)
      }
      mesh.instanceMatrix.needsUpdate = true
    }
  })

  if (!total) return null
  return (
    <group>
      {rocks.map((list, v) =>
        list.length ? (
          <instancedMesh key={`${v}:${list.length}`} ref={(el) => (meshes.current[v] = el)} args={[VARIANTS[v], rockMat, list.length]} frustumCulled={false} castShadow receiveShadow />
        ) : null,
      )}
    </group>
  )
}
