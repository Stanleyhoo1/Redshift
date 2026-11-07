// Decodes WS `map` messages (protocol §5). PNG row 0 = southmost grid row (y = −L/2); we keep
// that order everywhere (texture v = 0 ↔ row 0, pixel index i·w + j ↔ grid row i, col j).
import { CLS, type MapMeta, type MapStats } from './types'

export interface DecodedMap {
  meta: MapMeta
  stats: MapStats
  bitmap: ImageBitmap
  pixels: Uint8ClampedArray // RGBA, row 0 = south
  width: number
  height: number
  costmap: HTMLCanvasElement // class-coloured, canvas row 0 = grid row 0 (south)
  version: number
  receivedAt: number
}

export const decodeClass = (g: number): number => Math.round(g / 40)

// Class → RGBA for the radar's local costmap crop.
const CLASS_RGBA: Record<number, [number, number, number, number]> = {
  [CLS.UNKNOWN]: [0, 0, 0, 0],
  [CLS.FREE]: [120, 84, 92, 120],
  [CLS.CAUTION]: [255, 178, 56, 170],
  [CLS.HAZARD]: [255, 74, 92, 200],
  [CLS.PERMANENT]: [178, 24, 48, 235],
  [CLS.SCIENCE]: [92, 242, 230, 230],
  [CLS.OBSTACLE]: [150, 96, 104, 235],
}

function b64ToBytes(b64: string): Uint8Array<ArrayBuffer> {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

let version = 0
let scratch: HTMLCanvasElement | null = null

export async function decodeMap(png_b64: string, meta: MapMeta, stats: MapStats): Promise<DecodedMap> {
  const blob = new Blob([b64ToBytes(png_b64)], { type: 'image/png' })
  const bitmap = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' })
  const w = bitmap.width
  const h = bitmap.height
  if (!scratch) scratch = document.createElement('canvas')
  scratch.width = w
  scratch.height = h
  const sctx = scratch.getContext('2d', { willReadFrequently: true })!
  sctx.clearRect(0, 0, w, h)
  sctx.drawImage(bitmap, 0, 0)
  const pixels = sctx.getImageData(0, 0, w, h).data

  const costmap = document.createElement('canvas')
  costmap.width = w
  costmap.height = h
  const cctx = costmap.getContext('2d')!
  const img = cctx.createImageData(w, h)
  const d = img.data
  for (let p = 0; p < w * h; p++) {
    const o = p * 4
    const cls = decodeClass(pixels[o + 1])
    const c = CLASS_RGBA[cls] ?? CLASS_RGBA[CLS.FREE]
    // modulate free terrain by elevation so relief is readable on the radar
    const k = cls === CLS.FREE ? 0.6 + 0.4 * (pixels[o] / 255) : 1
    d[o] = c[0] * k
    d[o + 1] = c[1] * k
    d[o + 2] = c[2] * k
    d[o + 3] = c[3]
  }
  cctx.putImageData(img, 0, 0)

  return {
    meta,
    stats,
    bitmap,
    pixels,
    width: w,
    height: h,
    costmap,
    version: ++version,
    receivedAt: performance.now(),
  }
}

/** Grid cell of a world point, or null outside the sector. */
export function cellAt(map: Pick<DecodedMap, 'meta' | 'width' | 'height'>, x: number, y: number): [number, number] | null {
  const { origin, res } = map.meta
  const j = Math.floor((x - origin[0]) / res)
  const i = Math.floor((y - origin[1]) / res)
  if (i < 0 || j < 0 || i >= map.height || j >= map.width) return null
  return [i, j]
}

/** Normalised elevation (R/255) of a known cell, or null when unknown / outside. */
export function elevNormAt(map: DecodedMap | null, x: number, y: number): number | null {
  if (!map) return null
  const c = cellAt(map, x, y)
  if (!c) return null
  const o = (c[0] * map.width + c[1]) * 4
  if (decodeClass(map.pixels[o + 1]) === CLS.UNKNOWN) return null
  return map.pixels[o] / 255
}
