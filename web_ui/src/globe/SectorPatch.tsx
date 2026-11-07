import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { ELEV_EXAGGERATION } from '../lib/geo'
import { useStore } from '../lib/store'
import { PATCH_LIFT } from './globeUtil'
import { PATCH_FRAGMENT, PATCH_VERTEX } from './sectorShader'

const SEGMENTS = 256

function makeTexture(image?: ImageBitmap): THREE.Texture {
  const t = new THREE.Texture(image as unknown as HTMLImageElement)
  t.magFilter = THREE.NearestFilter
  t.minFilter = THREE.NearestFilter
  t.generateMipmaps = false
  t.flipY = false // PNG row 0 (south) → v = 0
  t.colorSpace = THREE.NoColorSpace
  t.wrapS = THREE.ClampToEdgeWrapping
  t.wrapT = THREE.ClampToEdgeWrapping
  t.premultiplyAlpha = false
  if (image) t.needsUpdate = true
  return t
}

// 1×1 black placeholder so the sampler is always valid before the first map arrives
function placeholder(): THREE.DataTexture {
  const t = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1, THREE.RGBAFormat)
  t.needsUpdate = true
  return t
}

/** The mission sector: displaced, classified terrain where mapped, fog with a cell grid elsewhere. */
export function SectorPatch() {
  const texRef = useRef<THREE.Texture | null>(null)
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: PATCH_VERTEX,
        fragmentShader: PATCH_FRAGMENT,
        uniforms: {
          uMap: { value: placeholder() },
          uHasMap: { value: 0 },
          uExag: { value: ELEV_EXAGGERATION },
          uLift: { value: PATCH_LIFT },
          uTime: { value: 0 },
          uCells: { value: new THREE.Vector2(200, 200) },
          uElevRange: { value: 5 },
          uRes: { value: 0.5 },
        },
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      }),
    [],
  )

  useEffect(() => {
    const apply = (map: ReturnType<typeof useStore.getState>['map']) => {
      if (!map) return
      let tex = texRef.current
      const img = tex?.image as ImageBitmap | undefined
      if (!tex || !img || img.width !== map.width || img.height !== map.height) {
        tex?.dispose()
        tex = makeTexture(map.bitmap)
        texRef.current = tex
      } else {
        tex.image = map.bitmap
        tex.needsUpdate = true
      }
      const u = material.uniforms
      u.uMap.value = tex
      u.uHasMap.value = 1
      u.uCells.value.set(map.width, map.height)
      u.uElevRange.value = Math.max(0.1, map.meta.elev_max - map.meta.elev_min)
      u.uRes.value = map.meta.res
    }
    apply(useStore.getState().map)
    return useStore.subscribe((s, prev) => {
      if (s.map !== prev.map) apply(s.map)
    })
  }, [material])

  useEffect(
    () => () => {
      texRef.current?.dispose()
      material.dispose()
    },
    [material],
  )

  useFrame(({ clock }) => {
    material.uniforms.uTime.value = clock.elapsedTime
  })

  return (
    <mesh material={material} renderOrder={1} frustumCulled={false}>
      <planeGeometry args={[1, 1, SEGMENTS, SEGMENTS]} />
    </mesh>
  )
}
