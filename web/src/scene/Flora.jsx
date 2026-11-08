import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { groundAt, knownAt } from './ground.js'
import { reducedMotion } from './sceneConstants.js'
import { FLORA } from '../lib/terrainTheme.js'

// v7 alien flora (PROTOCOL.md, `features.flora` and `fleet.flora_zones`). Fungal trees: a thin pale
// stalk under a wide violet or teal cap with a soft glowing rim. Bushes: a few glowing lime bulbs
// that pulse gently. Instanced, so a few draw calls in all. A plant shows once the fleet knows the
// ground under it, or with Reality on. Low graphics draws every other bush. Known plants get a
// dashed keep-out ring.

const MAX = 128
const BULBS = 3
const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0)
// Hex colours are sRGB, so the caps keep their saturation once converted to linear.
const VIOLET = new THREE.Color(FLORA.violet)
const TEAL = new THREE.Color(FLORA.teal)
const _m = new THREE.Matrix4()
const _p = new THREE.Vector3()
const _s = new THREE.Vector3()
const _q = new THREE.Quaternion()
const _c = new THREE.Color()

// The caps glow in their own colour (the instance colour squared, so the glow stays saturated),
// brighter at the rim, so a violet or teal cap stands out on dark ground from afar but still
// shades as a dome up close.
function glowCaps(shader) {
  shader.fragmentShader = shader.fragmentShader.replace(
    '#include <emissivemap_fragment>',
    `#include <emissivemap_fragment>
  {
    float rim = 1.0 - abs(dot(normalize(normal), normalize(vViewPosition)));
#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
    totalEmissiveRadiance = vColor.rgb * vColor.rgb * (0.7 + 1.2 * rim * rim * rim);
#else
    totalEmissiveRadiance = vec3(0.0);
#endif
  }`,
  )
}

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
          // A tight clump sitting on the ground, each bulb a little higher than the last, so
          // none floats in the air with nothing under it.
          const off = 0.45 * s * (b + 1)
          const bx = p.x + off * Math.cos(a)
          const by = p.y + off * Math.sin(a)
          _m.compose(_p.set(bx, groundAt(bx, by) + s * (0.75 + 0.35 * b), -by), _q.identity(), _s.set(s, s, s))
          bulb.current.setMatrixAt(nb++, _m)
        }
      }
    }
    for (let k = nb; k < MAX * BULBS; k++) bulb.current.setMatrixAt(k, HIDDEN)
    stalk.current.instanceMatrix.needsUpdate = true
    cap.current.instanceMatrix.needsUpdate = true
    if (cap.current.instanceColor) cap.current.instanceColor.needsUpdate = true
    bulb.current.instanceMatrix.needsUpdate = true
    if (bulbMat.current) bulbMat.current.emissiveIntensity = reducedMotion ? 1.5 : 1.35 + 0.45 * Math.sin(clock.elapsedTime * 1.6)
  })

  if (!list.length) return null
  return (
    <group>
      <instancedMesh ref={stalk} args={[geos.stalk, undefined, MAX]} frustumCulled={false}>
        <meshStandardMaterial color="#d8d2c4" roughness={0.8} />
      </instancedMesh>
      <instancedMesh ref={cap} args={[geos.cap, undefined, MAX]} frustumCulled={false}>
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" roughness={0.6} side={THREE.DoubleSide} onBeforeCompile={glowCaps} customProgramCacheKey={() => 'redshift-flora-cap-1'} />
      </instancedMesh>
      <instancedMesh ref={bulb} args={[geos.bulb, undefined, MAX * BULBS]} frustumCulled={false}>
        <meshStandardMaterial ref={bulbMat} color="#b8ff48" emissive="#5cff1e" emissiveIntensity={1.5} roughness={0.4} />
      </instancedMesh>
      {Array.isArray(zones) && zones.filter((z) => Array.isArray(z) && z.length >= 3 && z.every(Number.isFinite)).map((z, i) => (
        <Ring key={i} x={z[0]} y={z[1]} r={z[2]} />
      ))}
    </group>
  )
}
