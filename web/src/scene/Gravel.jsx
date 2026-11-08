import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore, useSim } from '../state/store.js'
import { sampleGround } from '../state/simState.js'
import { useGraphics } from '../state/visualPrefs.js'
import { rockGeometry, rockMaterial } from './rocks.js'

// Small loose stones on the known ground near the point the camera looks at, so close-ups
// read as real regolith. Deterministic per map cell (the same stones every time you come
// back), more on steep ground, none on the lander pad. Tiny (2 to 9 cm) so nobody mistakes
// them for obstacles. One instanced mesh, rebuilt only when the view moves a couple of
// metres or the map changes; hidden when the camera is far out. Off with Low graphics.

const MAX = 7000
const RADIUS = 17 // m around the orbit target
const FAR = 62 // m: no gravel when the camera is further than this from its target

const _m = new THREE.Matrix4()
const _q = new THREE.Quaternion()
const _p = new THREE.Vector3()
const _s = new THREE.Vector3()
const _e = new THREE.Euler()
const _c = new THREE.Color()

// Cheap integer hash of a cell (and a salt) to [0, 1).
function h01(i, j, salt) {
  let h = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(salt, 2246822519)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

export function Gravel() {
  const graphics = useGraphics()
  const lander = useSim((s) => s.lander)
  const mesh = useRef()
  const st = useRef({ cx: 1e9, cz: 1e9, version: -1, at: 0, T: null })
  const geo = useMemo(() => rockGeometry(9, 1, { cuts: 3, squash: 0.6 }), [])
  const mat = useMemo(() => rockMaterial('gravel', { bump: 0, roughness: 0.92 }), [])
  useEffect(() => () => geo.dispose(), [geo])
  const high = graphics === 'high'

  useFrame(({ camera, controls }) => {
    const g = mesh.current
    if (!g) return
    const T = simStore.get().terrain
    const c = controls?.target
    const far = !c || camera.position.distanceTo(c) > FAR
    g.visible = high && !far && !!T
    if (!g.visible) return
    const s = st.current
    const now = performance.now()
    const moved = Math.hypot(c.x - s.cx, c.z - s.cz) > 2.2
    if (!(moved || T.version !== s.version || T !== s.T) || now - s.at < 250) return
    s.cx = c.x
    s.cz = c.z
    s.version = T.version
    s.T = T
    s.at = now
    const { nx, ny, res, ox, oy, known, slope } = T
    const cy = -c.z
    const i0 = Math.max(0, Math.floor((c.x - RADIUS - ox) / res))
    const i1 = Math.min(nx - 2, Math.ceil((c.x + RADIUS - ox) / res))
    const j0 = Math.max(0, Math.floor((cy - RADIUS - oy) / res))
    const j1 = Math.min(ny - 2, Math.ceil((cy + RADIUS - oy) / res))
    let n = 0
    for (let j = j0; j <= j1 && n < MAX; j++) {
      for (let i = i0; i <= i1 && n < MAX; i++) {
        const k = j * nx + i
        if (!known[k] || !known[k + 1] || !known[k + nx] || !known[k + nx + 1]) continue
        const x0 = ox + i * res
        const y0 = oy + j * res
        const d = Math.hypot(x0 - c.x, y0 - cy)
        if (d > RADIUS) continue
        if (lander && Math.hypot(x0 - lander.x, y0 - lander.y) < lander.radius + 0.3) continue
        // More stones on steeper ground, clumped by a coarse hash.
        const sl = slope[k] || 0
        const clump = h01(i >> 3, j >> 3, 5)
        const count = Math.floor(h01(i, j, 1) * (0.6 + Math.min(1.6, sl / 18) + clump * 1.2))
        for (let q = 0; q < count && n < MAX; q++) {
          const x = x0 + h01(i, j, 10 + q) * res
          const y = y0 + h01(i, j, 20 + q) * res
          const z = sampleGround(T, x, y)
          if (z === null) continue
          const big = h01(i, j, 30 + q)
          const size = (0.02 + big * big * big * 0.07) * Math.min(1, (RADIUS - d) / 4)
          if (size < 0.012) continue
          _e.set(h01(i, j, 40 + q) * 6.28, h01(i, j, 50 + q) * 6.28, h01(i, j, 60 + q) * 6.28)
          _q.setFromEuler(_e)
          _m.compose(_p.set(x, z + size * 0.15, -y), _q, _s.set(size * (0.8 + big * 0.5), size, size * (0.7 + h01(i, j, 70 + q) * 0.6)))
          g.setMatrixAt(n, _m)
          const tone = 0.62 + h01(i, j, 80 + q) * 0.5
          g.setColorAt(n, _c.setRGB(tone, tone * 0.96, tone * 0.92))
          n++
        }
      }
    }
    g.count = n
    g.instanceMatrix.needsUpdate = true
    if (g.instanceColor) g.instanceColor.needsUpdate = true
  })

  return <instancedMesh ref={mesh} args={[geo, mat, MAX]} count={0} frustumCulled={false} receiveShadow visible={false} />
}
