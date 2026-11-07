import { useLayoutEffect, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { Line2 } from 'three/examples/jsm/lines/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { ELEV_EXAGGERATION, elevToNorm, localFrame, sectorToSphereR } from '../lib/geo'
import { elevNormAt, type DecodedMap } from '../lib/mapDecode'
import type { XY } from '../lib/types'

export const PATCH_LIFT = 0.0012 // patch floats just above the fog sphere
export const LINE_LIFT = 0.007 // paths / trails above the displaced terrain

/** Radius of the displayed terrain at (x, y): known map elevation, else the robot's own z. */
export function surfaceRadius(map: DecodedMap | null, x: number, y: number, fallbackZ?: number): number {
  let e = elevNormAt(map, x, y)
  if (e === null && fallbackZ !== undefined && map) e = elevToNorm(fallbackZ, map.meta.elev_min, map.meta.elev_max)
  return 1 + PATCH_LIFT + (e ?? 0) * ELEV_EXAGGERATION
}

export function surfacePoint(map: DecodedMap | null, x: number, y: number, L: number, lift = 0, fallbackZ?: number, out = new THREE.Vector3()) {
  const p = sectorToSphereR(x, y, surfaceRadius(map, x, y, fallbackZ) + lift, L)
  return out.set(p[0], p[1], p[2])
}

/** Flat Float32 positions for a world polyline draped on the terrain. */
export function drape(points: readonly XY[], map: DecodedMap | null, L: number, lift = LINE_LIFT, fallbackZ?: number): Float32Array {
  const out = new Float32Array(points.length * 3)
  const v = new THREE.Vector3()
  points.forEach(([x, y], k) => {
    surfacePoint(map, x, y, L, lift, fallbackZ, v)
    out[k * 3] = v.x
    out[k * 3 + 1] = v.y
    out[k * 3 + 2] = v.z
  })
  return out
}

const _m = new THREE.Matrix4()
const _x = new THREE.Vector3()
const _y = new THREE.Vector3()
const _z = new THREE.Vector3()
/** Orientation with +Y = surface up and +Z = world heading `yaw` (CCW from east). */
export function surfaceQuaternion(x: number, y: number, yaw: number, L: number, out = new THREE.Quaternion()) {
  const { east, north, up } = localFrame(x, y, L)
  const c = Math.cos(yaw)
  const s = Math.sin(yaw)
  _y.set(up[0], up[1], up[2])
  _z.set(c * east[0] + s * north[0], c * east[1] + s * north[1], c * east[2] + s * north[2])
  _x.crossVectors(_y, _z).normalize()
  _m.makeBasis(_x, _y, _z)
  return out.setFromRotationMatrix(_m)
}

/** Glow sprite texture (radial falloff), generated once. */
let glowTex: THREE.Texture | null = null
export function glowTexture(): THREE.Texture {
  if (glowTex) return glowTex
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const g = c.getContext('2d')!
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32)
  grd.addColorStop(0, 'rgba(255,255,255,1)')
  grd.addColorStop(0.25, 'rgba(255,255,255,0.45)')
  grd.addColorStop(1, 'rgba(255,255,255,0)')
  g.fillStyle = grd
  g.fillRect(0, 0, 64, 64)
  glowTex = new THREE.CanvasTexture(c)
  return glowTex
}

export interface FatLineOpts {
  color: string
  width: number
  dashed?: boolean
  dashSize?: number
  gapSize?: number
  opacity?: number
  additive?: boolean
  depthTest?: boolean
}

/**
 * Persistent Line2 whose geometry is swapped (old one disposed) when the positions change, while
 * the material — and its compiled shader — lives as long as the component.
 */
export function useFatLine(positions: Float32Array | null, colors: Float32Array | null, opts: FatLineOpts): Line2 {
  const size = useThree((s) => s.size)
  const line = useMemo(() => {
    const mat = new LineMaterial({
      color: new THREE.Color(colors ? '#ffffff' : opts.color),
      linewidth: opts.width,
      dashed: !!opts.dashed,
      dashSize: opts.dashSize ?? 0.02,
      gapSize: opts.gapSize ?? 0.012,
      transparent: true,
      opacity: opts.opacity ?? 1,
      vertexColors: !!colors,
      depthWrite: false,
      depthTest: opts.depthTest ?? true,
      blending: opts.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    })
    if (opts.dashed) mat.defines.USE_DASH = ''
    const l = new Line2(new LineGeometry(), mat)
    l.frustumCulled = false
    return l
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useLayoutEffect(() => {
    line.material.resolution.set(size.width, size.height)
  }, [line, size.width, size.height])

  useLayoutEffect(() => {
    const old = line.geometry
    const g = new LineGeometry()
    if (positions && positions.length >= 6) {
      g.setPositions(positions)
      if (colors) g.setColors(colors)
      line.visible = true
    } else {
      line.visible = false
    }
    line.geometry = g
    if (line.visible) line.computeLineDistances()
    old.dispose()
  }, [line, positions, colors])

  useLayoutEffect(() => {
    if (!colors) line.material.color.set(opts.color)
  }, [line, opts.color, colors])

  useLayoutEffect(
    () => () => {
      line.geometry.dispose()
      line.material.dispose()
    },
    [line],
  )
  return line
}
