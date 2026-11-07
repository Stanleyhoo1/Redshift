import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { useApp, useSim } from '../state/store.js'
import { kindStyle } from '../lib/theme.js'

// Ground truth from /api/truth: a faint lattice of the real terrain and every real
// target, so you can compare what the fleet knows with what is actually there.
function buildLattice(truth) {
  const { nx, ny, res, ox, oy, heights } = truth
  const stride = Math.max(1, Math.round(2 / res)) // a line every ~2 m
  const lift = 0.06
  const segs = []
  const at = (i, j) => [ox + i * res, heights[j * nx + i] + lift, -(oy + j * res)]
  for (let j = 0; j < ny; j += stride) {
    for (let i = 0; i < nx - 1; i++) segs.push(...at(i, j), ...at(i + 1, j))
  }
  for (let i = 0; i < nx; i += stride) {
    for (let j = 0; j < ny - 1; j++) segs.push(...at(i, j), ...at(i, j + 1))
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(segs, 3))
  geo.computeBoundingSphere()
  return geo
}

function truthHeight(truth, x, y) {
  const i = Math.round((x - truth.ox) / truth.res)
  const j = Math.round((y - truth.oy) / truth.res)
  if (i < 0 || j < 0 || i >= truth.nx || j >= truth.ny) return 0
  return truth.heights[j * truth.nx + i]
}

const markerGeo = new THREE.OctahedronGeometry(0.7, 0)
const stemGeo = new THREE.CylinderGeometry(0.025, 0.025, 1.6, 6).translate(0, 0.8, 0)
const ringGeo = new THREE.RingGeometry(0.9, 1.0, 32).rotateX(-Math.PI / 2)

function TruthTarget({ target, truth }) {
  const color = kindStyle(target.kind).color
  const mats = useMemo(
    () => ({
      wire: new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.8, depthWrite: false }),
      faint: new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.35, depthWrite: false }),
    }),
    [color],
  )
  useEffect(() => () => Object.values(mats).forEach((m) => m.dispose()), [mats])
  const h = truthHeight(truth, target.x, target.y)
  return (
    <group position={[target.x, h, -target.y]}>
      <mesh geometry={stemGeo} material={mats.faint} />
      <mesh geometry={markerGeo} material={mats.wire} position={[0, 2.0, 0]} />
      <mesh geometry={ringGeo} material={mats.faint} position={[0, 0.07, 0]} />
    </group>
  )
}

export function Reality() {
  const truth = useApp((a) => a.truth)
  const gen = useSim((s) => s.gen)
  const valid = truth && !truth.error && truth.gen === gen ? truth : null
  const lattice = useMemo(() => (valid ? buildLattice(valid) : null), [valid])
  const mat = useMemo(() => new THREE.LineBasicMaterial({ color: '#cfeeff', transparent: true, opacity: 0.32, depthWrite: false }), [])
  useEffect(() => () => lattice?.dispose(), [lattice])
  useEffect(() => () => mat.dispose(), [mat])
  if (!valid) return null
  return (
    <group>
      <lineSegments geometry={lattice} material={mat} renderOrder={5} />
      {valid.targets.map((t) => (
        <TruthTarget key={t.id ?? `${t.x},${t.y}`} target={t} truth={valid} />
      ))}
    </group>
  )
}
