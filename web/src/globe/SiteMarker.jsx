import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { reducedMotion } from '../scene/sceneConstants.js'
import { useSim } from '../state/store.js'
import { GLOBE_FOV, MARKER_R, homeDistance } from './globeMath.js'
import { SITE_LIFT, ringMinScale } from './sectorMap.js'
import { makeMarkerMaterial } from './markerMaterial.js'
import { globeCardShown, leaveGlobe, useViewMode, viewMode } from './viewMode.js'
import { globeRefs } from './globeDebug.js'
import { SectorCard } from './SectorCard.jsx'

const HOME_SURF = homeDistance(GLOBE_FOV, 0.88) - 1 // camera height over the ground at home
const _p = new THREE.Vector3()
const _c = new THREE.Vector3()
const _s = new THREE.Vector3()
const _e = new THREE.Vector3()
const _n = new THREE.Vector3()
const CARD_GAP = 30 // px from the ring's right edge to the card (its leader line is 26 px)

// A point of the ring, draped on the ground like the ring itself, in screen pixels.
function ringPoint(g, camera, size, x, y, out) {
  g.localToWorld(out.set(x, y, 0))
  out.normalize().multiplyScalar(SITE_LIFT).project(camera)
  out.set((out.x * size.width) / 2, (out.y * size.height) / 2, 0)
  return out
}

// The landing sector on the planet: a cyan ring lying on the ground at the site, about the
// same size on screen at any zoom (a symbol: the sector itself is far too small to see) but
// never smaller than the enlarged sector map it frames (SectorMap.jsx), and the live card
// beside it, just clear of the ring. Clicking the ring flies back down. Its hit disc is invisible
// but still raycast, and the planet has no handlers, so a click only counts when the site
// faces the camera.
export function SiteMarker({ look, onSelect }) {
  const group = useRef(null)
  const card = useRef(null)
  const st = useRef({ facing: 1, over: false, hover: 0 })
  const camera = useThree((s) => s.camera)
  const size = useThree((s) => s.size)
  const map = useSim((s) => s.map)
  const minScale = useMemo(() => ringMinScale(map), [map])
  const cardShown = useViewMode(globeCardShown)
  const mat = useMemo(() => {
    const m = makeMarkerMaterial()
    m.uniforms.uInner.value = 0 // the sector map fills the ring
    m.uniforms.uDrape.value = SITE_LIFT
    return m
  }, [])
  const geo = useMemo(() => new THREE.PlaneGeometry(MARKER_R * 4, MARKER_R * 4, 24, 24), [])
  const hitGeo = useMemo(() => new THREE.CircleGeometry(MARKER_R * 1.3, 40), [])
  const hitMat = useMemo(() => new THREE.MeshBasicMaterial(), [])
  useEffect(
    () => () => {
      mat.dispose()
      geo.dispose()
      hitGeo.dispose()
      hitMat.dispose()
    },
    [mat, geo, hitGeo, hitMat],
  )
  useEffect(
    () => () => {
      document.body.style.cursor = ''
    },
    [],
  )
  useEffect(() => {
    globeRefs.site = group.current
    return () => {
      globeRefs.site = null
    }
  }, [])

  // Flat on the ground at the site: local x east, y north, z up.
  useLayoutEffect(() => {
    const g = group.current
    const { up, east, north } = look.frame
    const basis = new THREE.Matrix4().makeBasis(new THREE.Vector3(...east), new THREE.Vector3(...north), new THREE.Vector3(...up))
    g.quaternion.setFromRotationMatrix(basis)
    g.position.set(up[0], up[1], up[2]).multiplyScalar(SITE_LIFT)
  }, [look])

  useFrame((state, dt) => {
    const g = group.current
    const s = st.current
    if (!g) return
    if (viewMode.get().render !== 'globe') {
      if (s.over) {
        s.over = false
        document.body.style.cursor = ''
      }
      return
    }
    g.getWorldPosition(_p)
    s.facing = _c.copy(camera.position).sub(_p).normalize().dot(_p.normalize())
    const zoom = THREE.MathUtils.clamp((camera.position.length() - 1) / HOME_SURF, 0.35, 1.8)
    const scale = Math.max(minScale, zoom)
    g.scale.setScalar(scale)
    s.hover += ((s.over ? 1 : 0) - s.hover) * (1 - Math.exp(-Math.min(dt, 0.1) * 10))
    const u = mat.uniforms
    u.uFade.value = THREE.MathUtils.smoothstep(s.facing, 0.02, 0.25)
    u.uHover.value = s.hover
    u.uPulse.value = reducedMotion ? 0 : 1
    u.uThin.value = Math.min(1, (1.2 * zoom) / scale) // the line keeps its usual width on screen
    u.uTime.value = state.clock.elapsedTime
    // Imperative, so React re-renders of the card never reset it.
    if (card.current) {
      card.current.toggleAttribute('data-away', s.facing < 0.3)
      // How far the ring reaches to the right of the site on screen (an ellipse there).
      g.updateWorldMatrix(true, false)
      ringPoint(g, camera, size, 0, 0, _s)
      const ex = ringPoint(g, camera, size, MARKER_R, 0, _e).x - _s.x
      const nx = ringPoint(g, camera, size, 0, MARKER_R, _n).x - _s.x
      card.current.style.setProperty('--ring-gap', `${Math.round(Math.hypot(ex, nx) + CARD_GAP)}px`)
    }
  })

  const facingUs = () => st.current.facing > 0.2
  const click = (e) => {
    e.stopPropagation()
    if (e.delta > 4 || !facingUs()) return
    leaveGlobe()
  }
  const over = (e) => {
    if (!facingUs()) return
    e.stopPropagation()
    st.current.over = true
    document.body.style.cursor = 'pointer'
  }
  const out = () => {
    st.current.over = false
    document.body.style.cursor = ''
  }

  return (
    <group ref={group}>
      <mesh geometry={geo} material={mat} renderOrder={3} />
      <mesh geometry={hitGeo} material={hitMat} visible={false} onClick={click} onPointerOver={over} onPointerOut={out} />
      {cardShown && (
        <Html zIndexRange={[18, 0]} wrapperClass="globe-card-wrap">
          <SectorCard cardRef={card} onSelect={onSelect} />
        </Html>
      )}
    </group>
  )
}
