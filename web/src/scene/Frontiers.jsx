import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore, useSim } from '../state/store.js'
import { sampleGroundNear } from '../state/simState.js'
import { SIGNAL } from '../lib/theme.js'
import { glowTexture } from './textures.js'
import { fogYFor, reducedMotion } from './sceneConstants.js'

const MAX = 256
const glowGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2)
const ringGeo = new THREE.RingGeometry(0.4, 0.46, 40).rotateX(-Math.PI / 2)
const coreGeo = new THREE.CircleGeometry(0.16, 16).rotateX(-Math.PI / 2)
const _o = new THREE.Object3D()
const _c = new THREE.Color()
const signal = new THREE.Color(SIGNAL)

// Frontier cluster centres: soft cyan glows sized by cluster size, gently pulsing.
export function Frontiers() {
  const map = useSim((s) => s.map)
  const glow = useRef()
  const ring = useRef()
  const core = useRef()
  const cache = useRef({ list: null, pts: [] })
  const mats = useMemo(
    () => ({
      glow: new THREE.MeshBasicMaterial({ map: glowTexture(), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
      ring: new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
      core: new THREE.MeshBasicMaterial({ color: SIGNAL, depthWrite: false }),
    }),
    [],
  )
  useEffect(() => () => Object.values(mats).forEach((m) => m.dispose()), [mats])

  useFrame(({ clock }) => {
    const s = simStore.get()
    const c = cache.current
    if (s.frontiers !== c.list) {
      c.list = s.frontiers
      const fogY = fogYFor(map)
      c.pts = s.frontiers.slice(0, MAX).map((f, i) => {
        const size = Number.isFinite(f[2]) ? f[2] : 10
        return {
          x: f[0],
          y: f[1],
          h: sampleGroundNear(s.terrain, f[0], f[1], 8) ?? fogY,
          scale: 1.1 + Math.sqrt(Math.max(1, size)) * 0.3,
          phase: (i * 2.399) % (Math.PI * 2),
        }
      })
    }
    const t = reducedMotion ? 0 : clock.elapsedTime
    const pts = c.pts
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i]
      const wave = Math.sin(t * 2 + p.phase)
      const k = p.scale * (1 + 0.1 * wave)
      _o.position.set(p.x, p.h + 0.16, -p.y)
      _o.scale.set(k * 2.2, 1, k * 2.2)
      _o.updateMatrix()
      glow.current.setMatrixAt(i, _o.matrix)
      _c.copy(signal).multiplyScalar(0.45 + 0.2 * wave)
      glow.current.setColorAt(i, _c)
      _o.scale.set(k, 1, k)
      _o.position.y = p.h + 0.18
      _o.updateMatrix()
      ring.current.setMatrixAt(i, _o.matrix)
      _c.copy(signal).multiplyScalar(0.75 + 0.25 * wave)
      ring.current.setColorAt(i, _c)
      _o.scale.set(1, 1, 1)
      _o.position.y = p.h + 0.2
      _o.updateMatrix()
      core.current.setMatrixAt(i, _o.matrix)
    }
    for (const m of [glow.current, ring.current, core.current]) {
      m.count = pts.length
      m.instanceMatrix.needsUpdate = true
      if (m.instanceColor) m.instanceColor.needsUpdate = true
    }
  })

  return (
    <group>
      <instancedMesh ref={glow} args={[glowGeo, mats.glow, MAX]} frustumCulled={false} renderOrder={3} />
      <instancedMesh ref={ring} args={[ringGeo, mats.ring, MAX]} frustumCulled={false} renderOrder={3} />
      <instancedMesh ref={core} args={[coreGeo, mats.core, MAX]} frustumCulled={false} renderOrder={3} />
    </group>
  )
}
