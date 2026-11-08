import { useEffect, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useSim } from '../state/store.js'
import { groundAt, knownAt } from './ground.js'
import { reducedMotion } from './sceneConstants.js'
import { GLYPH_COLOR, KIND, MARKER_PX, QUAD_MARGIN, floraGlyphColor, makeMapMarkerGeometry, makeMapMarkerMaterial, ventPulse } from './mapMarkers.js'

// Map markers (see mapMarkers.js): a glyph over every plant, geyser vent and boulder field the
// fleet has mapped (or every one with Reality on), the same rule as the models. Constant size on
// screen, so the overview reads; they fade out up close where the real models take over.

const MAX = 256

export function MapMarkers({ reality }) {
  const flora = useSim((s) => s.features?.flora)
  const vents = useSim((s) => s.features?.geysers)
  const fields = useSim((s) => s.features?.boulder_fields)
  const size = useThree((s) => s.size)
  const geo = useMemo(() => makeMapMarkerGeometry(MAX), [])
  const mat = useMemo(() => makeMapMarkerMaterial(), [])
  useEffect(() => () => (geo.dispose(), mat.dispose()), [geo, mat])
  useEffect(() => {
    mat.uniforms.uViewport.value.set(Math.max(1, size.width), Math.max(1, size.height))
  }, [mat, size])

  useFrame(({ clock }) => {
    const t = reducedMotion ? 0 : clock.elapsedTime
    mat.uniforms.uTime.value = t
    const C = geo.attributes.aCenter
    const K = geo.attributes.aColor
    const D = geo.attributes.aData
    let n = 0
    const put = (x, y, z, rgb, kind, px, pulse = 0) => {
      if (n >= MAX) return
      C.setXYZ(n, x, z, -y)
      K.setXYZ(n, rgb[0], rgb[1], rgb[2])
      D.setXYZW(n, kind, px * QUAD_MARGIN, 1, pulse)
      n++
    }
    const seen = (x, y) => Number.isFinite(x) && Number.isFinite(y) && (reality || knownAt(x, y))
    if (Array.isArray(flora)) {
      for (const p of flora) {
        if (!p || !seen(p.x, p.y)) continue
        const gz = groundAt(p.x, p.y)
        if (p.kind === 'tree') put(p.x, p.y, gz + (Number.isFinite(p.height) ? p.height : 3) + 0.3, floraGlyphColor(p.hue), KIND.tree, MARKER_PX.tree)
        else put(p.x, p.y, gz + 0.6, GLYPH_COLOR.bush, KIND.bush, MARKER_PX.bush)
      }
    }
    if (Array.isArray(vents)) {
      for (const v of vents) {
        if (!v || !seen(v.x, v.y)) continue
        put(v.x, v.y, groundAt(v.x, v.y) + 0.9, GLYPH_COLOR.geyser, KIND.geyser, MARKER_PX.geyser, ventPulse(v.active, t, reducedMotion))
      }
    }
    if (Array.isArray(fields)) {
      for (const f of fields) {
        if (!f || !seen(f.x, f.y)) continue
        put(f.x, f.y, groundAt(f.x, f.y) + 1.2, GLYPH_COLOR.boulders, KIND.boulders, MARKER_PX.boulders)
      }
    }
    geo.instanceCount = n
    C.needsUpdate = true
    K.needsUpdate = true
    D.needsUpdate = true
  })

  return <mesh geometry={geo} material={mat} frustumCulled={false} renderOrder={20} />
}
