import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore, useSim } from '../state/store.js'
import { dotTexture } from './textures.js'

const CAP = 300 // PROTOCOL.md: at most 300 points per sweep
const _c = new THREE.Color()

// Each rover's latest LiDAR sweep (rover.scan, PROTOCOL.md version 2) as points in its
// colour. One preallocated buffer per rover, written in place only when a new sweep
// arrives (about 5 times a second), so the per-frame cost is nil. Points fade with
// distance from the rover and keep a small on-screen size, so the map stays readable.
function RoverLidar({ meta }) {
  const last = useRef(undefined)
  const { geometry, material, pos, col } = useMemo(() => {
    const pos = new Float32Array(CAP * 3)
    const col = new Float32Array(CAP * 4)
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('color', new THREE.BufferAttribute(col, 4).setUsage(THREE.DynamicDrawUsage))
    g.setDrawRange(0, 0)
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6)
    const m = new THREE.PointsMaterial({
      size: 0.42,
      sizeAttenuation: true,
      vertexColors: true,
      map: dotTexture(),
      transparent: true,
      depthWrite: false,
      alphaTest: 0.02,
    })
    // Keep points between 2.5 and 7 device pixels: visible in the overview, not blobs up close.
    m.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace('#include <fog_vertex>', '#include <fog_vertex>\n  gl_PointSize = clamp(gl_PointSize, 2.5, 7.0);')
    }
    return { geometry: g, material: m, pos, col }
  }, [])
  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )

  useFrame(() => {
    const r = simStore.get().rovers[meta.id]
    const scan = r ? r.scan : null
    if (scan === last.current) return
    last.current = scan
    if (!scan) {
      geometry.setDrawRange(0, 0)
      return
    }
    const range = Number.isFinite(r.sensors?.lidar_range) && r.sensors.lidar_range > 0 ? r.sensors.lidar_range : 8
    _c.set(meta.color)
    const n = Math.min(CAP, Math.floor(scan.length / 3))
    let q = 0
    for (let p = 0; p < n; p++) {
      const x = scan[p * 3]
      const y = scan[p * 3 + 1]
      const z = scan[p * 3 + 2]
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue
      pos[q * 3] = x
      pos[q * 3 + 1] = z + 0.07
      pos[q * 3 + 2] = -y
      const d = Math.min(1, Math.hypot(x - r.x, y - r.y) / Math.max(1, range))
      const lift = 0.75 + 0.25 * (1 - d)
      col[q * 4] = _c.r * lift + (1 - lift) * 0.2
      col[q * 4 + 1] = _c.g * lift + (1 - lift) * 0.2
      col[q * 4 + 2] = _c.b * lift + (1 - lift) * 0.2
      col[q * 4 + 3] = 0.95 - 0.45 * d
      q++
    }
    geometry.setDrawRange(0, q)
    geometry.attributes.position.clearUpdateRanges()
    geometry.attributes.position.addUpdateRange(0, q * 3)
    geometry.attributes.position.needsUpdate = true
    geometry.attributes.color.clearUpdateRanges()
    geometry.attributes.color.addUpdateRange(0, q * 4)
    geometry.attributes.color.needsUpdate = true
  })

  return <points geometry={geometry} material={material} frustumCulled={false} renderOrder={4} />
}

export function Lidar() {
  const meta = useSim((s) => s.roverMeta)
  const gen = useSim((s) => s.gen)
  return (
    <group>
      {meta.map((m) => (
        <RoverLidar key={`${gen}:${m.id}`} meta={m} />
      ))}
    </group>
  )
}
