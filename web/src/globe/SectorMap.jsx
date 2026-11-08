import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore, useSim } from '../state/store.js'
import { paintSectorMap, patchGeometry, sectorLook } from './sectorMap.js'
import { makeSectorMapMaterial, makeSectorTexture } from './sectorMapMaterial.js'
import { viewMode } from './viewMode.js'
import { globeRefs } from './globeDebug.js'

const ZONE_KEYS = ['water_zones', 'flora_zones', 'ice_zones', 'soft_zones']
const zoneSig = (fleet) => JSON.stringify(ZONE_KEYS.map((k) => (Array.isArray(fleet?.[k]) ? fleet[k] : null)))

// The fleet's own map of the landing sector, live on the planet: an enlarged square patch at
// the site (sectorMap.js) showing only the ground the rovers have mapped, so it starts empty
// and grows as they explore. It is repainted from the store's terrain while the planet is
// drawn, at most twice a second (once a second on Graphics Low), and only when the map or
// the fleet's zones changed; a new terrain after a reset is painted at once.
export function SectorMap({ look, high }) {
  const map = useSim((s) => s.map)
  const mat = useMemo(makeSectorMapMaterial, [])
  const geo = useMemo(() => {
    if (!map) return null
    const p = patchGeometry(look.frame, map)
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(p.positions, 3))
    g.setAttribute('uv', new THREE.BufferAttribute(p.uvs, 2))
    g.setAttribute('aLocal', new THREE.BufferAttribute(p.local, 2))
    g.setIndex(new THREE.BufferAttribute(p.indices, 1))
    g.computeBoundingSphere()
    return g
  }, [look, map])
  const st = useRef({ T: null, version: -1, zones: '', next: 0, tex: null, data: null, opaque: 0 })
  const mesh = useRef(null)

  useLayoutEffect(() => {
    mat.uniforms.uSun.value.set(look.sun[0], look.sun[1], look.sun[2]).normalize()
  }, [mat, look])
  useEffect(() => () => geo?.dispose(), [geo])
  useEffect(
    () => () => {
      mat.dispose()
      st.current.tex?.dispose()
    },
    [mat],
  )
  useEffect(() => {
    globeRefs.sectorMap = st.current
    return () => {
      globeRefs.sectorMap = null
    }
  }, [])

  useFrame((state) => {
    if (viewMode.get().render !== 'globe') return
    const s = st.current
    if (mesh.current) mesh.current.visible = !!s.tex // nothing to draw before the first paint
    const sim = simStore.get()
    const T = sim.terrain
    if (!T) return
    const zones = zoneSig(sim.fleet)
    const fresh = T !== s.T
    if (!fresh && T.version === s.version && zones === s.zones) return
    const now = state.clock.elapsedTime
    if (!fresh && now < s.next) return
    s.next = now + (high ? 0.5 : 1)
    s.T = T
    s.version = T.version
    s.zones = zones
    if (s.lookWorld !== sim.world || s.lookFeatures !== sim.features) {
      s.look = sectorLook(sim.world, sim.features)
      s.lookWorld = sim.world
      s.lookFeatures = sim.features
    }
    const data = paintSectorMap(T, sim.fleet, s.data, s.look)
    if (!s.tex || s.tex.image.width !== T.nx || s.tex.image.height !== T.ny || data !== s.data) {
      s.tex?.dispose()
      s.tex = makeSectorTexture(data, T.nx, T.ny)
      mat.uniforms.uMap.value = s.tex
    } else {
      s.tex.needsUpdate = true
    }
    s.data = data
    let opaque = 0
    for (let k = 3; k < data.length; k += 4) if (data[k] === 255) opaque++
    s.opaque = opaque
    if (mesh.current) mesh.current.visible = true
  })

  if (!geo) return null
  return <mesh ref={mesh} geometry={geo} material={mat} renderOrder={1} visible={false} />
}
