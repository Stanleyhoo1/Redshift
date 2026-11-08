import { REGOLITH, SAND, WATER } from '../lib/theme.js'
import { FLORA, ICE } from '../lib/terrainTheme.js'
import { SUN_DIR } from '../scene/sceneConstants.js'
import { MARKER_R } from './globeMath.js'

// The fleet's map on the planet (SectorMap.jsx), as plain numbers so Node can test it: the
// patch it is drawn on and the RGBA pixels it shows. The real sector (a few hundred metres)
// would be far too small to see from orbit, so the patch is enlarged to PATCH_HALF_DEG
// either side of the site. It is square like the sector, map north along the planet's
// north and map east along its east.

export const PATCH_HALF_DEG = 7 // angular half-size of the longer side, degrees (not to scale)
export const PATCH_LIFT = 0.002 // above the ground, planet radii, so it never z-fights
export const PATCH_SEGS = 32 // grid steps each way: enough to follow the curve
export const SITE_LIFT = 1.003 // the landing ring lies this far from the centre
const RING_MARGIN = 1.1 // the ring clears the patch corners by this factor

const RAD = Math.PI / 180

// Half-sizes in radians east-west and north-south, the longer side PATCH_HALF_DEG.
export function patchHalfAngles(map, halfDeg = PATCH_HALF_DEG) {
  const w = map?.width > 0 ? map.width : 1
  const d = map?.depth > 0 ? map.depth : 1
  const m = Math.max(w, d)
  return { east: halfDeg * RAD * (w / m), north: halfDeg * RAD * (d / m) }
}

// Direction from the planet's centre for a point a, b (radians) east and north of the site
// on the patch: a gnomonic square, so the middle of each edge is exactly the half-size away.
export function patchDir(frame, a, b) {
  const { up, east, north } = frame
  const ta = Math.tan(a)
  const tb = Math.tan(b)
  const v = [up[0] + ta * east[0] + tb * north[0], up[1] + ta * east[1] + tb * north[1], up[2] + ta * east[2] + tb * north[2]]
  const l = Math.hypot(v[0], v[1], v[2])
  return [v[0] / l, v[1] / l, v[2] / l]
}

// The patch mesh: (segs + 1)^2 vertices, u east and v north. Texel (i, j) of the map lies at
// grid sample (i, j), so the uvs run from the first texel's centre to the last one's.
export function patchGeometry(frame, map, { halfDeg = PATCH_HALF_DEG, segs = PATCH_SEGS, lift = PATCH_LIFT } = {}) {
  const half = patchHalfAngles(map, halfDeg)
  const nx = Math.max(2, map?.nx | 0 || 2)
  const ny = Math.max(2, map?.ny | 0 || 2)
  const n = segs + 1
  const positions = new Float32Array(n * n * 3)
  const uvs = new Float32Array(n * n * 2)
  const local = new Float32Array(n * n * 2) // 0 to 1 across the patch, for its frame
  const indices = new Uint16Array(segs * segs * 6)
  const r = 1 + lift
  for (let q = 0; q < n; q++) {
    for (let p = 0; p < n; p++) {
      const u = p / segs
      const v = q / segs
      const d = patchDir(frame, (2 * u - 1) * half.east, (2 * v - 1) * half.north)
      const k = q * n + p
      positions[k * 3] = d[0] * r
      positions[k * 3 + 1] = d[1] * r
      positions[k * 3 + 2] = d[2] * r
      uvs[k * 2] = (0.5 + u * (nx - 1)) / nx
      uvs[k * 2 + 1] = (0.5 + v * (ny - 1)) / ny
      local[k * 2] = u
      local[k * 2 + 1] = v
    }
  }
  let o = 0
  for (let q = 0; q < segs; q++) {
    for (let p = 0; p < segs; p++) {
      const a = q * n + p
      const b = a + 1
      const c = a + n
      const e = c + 1
      indices[o++] = a
      indices[o++] = b
      indices[o++] = e
      indices[o++] = a
      indices[o++] = e
      indices[o++] = c
    }
  }
  return { positions, uvs, local, indices, half }
}

// The smallest landing ring scale (SiteMarker.jsx, ring radius MARKER_R at scale 1) that
// frames the patch: the ring lies on the ground and clears its corners.
export function ringMinScale(map, halfDeg = PATCH_HALF_DEG) {
  const h = patchHalfAngles(map, halfDeg)
  const corner = Math.atan(Math.hypot(Math.tan(h.east), Math.tan(h.north)))
  return (SITE_LIFT * Math.tan(corner) * RING_MARGIN) / MARKER_R
}

// Colours (sRGB bytes). The ground ramp is the map legend's "Mapped ground" swatch, dark
// low ground to pale dust high up; steep ground turns to rock as in the sector.
const rgb = (hex) => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)]
export const MAP_RAMP = ['#4b4440', '#8f7766', '#b19885'].map(rgb)
const ROCK = rgb(REGOLITH.rock)
// Fleet zones, painted over the ground in this order (water last, on top).
export const ZONE_STYLE = [
  ['soft_zones', rgb(SAND), 0.55],
  ['ice_zones', rgb(ICE.zone), 0.6],
  ['flora_zones', rgb(FLORA.ring), 0.5],
  ['water_zones', rgb(WATER.mid), 1],
]

// Shaded relief light: the sector's sun azimuth (from the west-north-west), raised to 45
// degrees so the shading stays gentle. World axes: x east, y north, z up.
const LA = Math.hypot(SUN_DIR[0], SUN_DIR[2])
const LIGHT = [(SUN_DIR[0] / LA) * Math.SQRT1_2, (-SUN_DIR[2] / LA) * Math.SQRT1_2, Math.SQRT1_2]
const RELIEF = 1.2 // vertical exaggeration of the shading (the patch is shrunk a lot)

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v)
const smooth = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1)
  return t * t * (3 - 2 * t)
}

const zoneOk = (z) => Array.isArray(z) && z.length >= 3 && Number.isFinite(z[0]) && Number.isFinite(z[1]) && Number.isFinite(z[2]) && z[2] > 0

// Paints the fleet's map T (simState.js createTerrain) and its zones into RGBA bytes, nx by
// ny, row j = grid row j (south first). Unknown ground is fully transparent black; known
// ground and the zones are opaque (so filtered texels come out premultiplied). Reuses `out` when it has the right size (a reset may change nx, ny). Pure: the
// same map gives the same bytes.
export function paintSectorMap(T, fleet, out) {
  const nx = T?.nx | 0
  const ny = T?.ny | 0
  const n = nx * ny * 4
  if (!out || out.length !== n) out = new Uint8Array(n)
  else out.fill(0)
  if (!T || !nx || !ny) return out
  const { h, slope, known, res } = T
  const lo = Number.isFinite(T.minH) ? T.minH : -3
  const span = Math.max(0.1, (Number.isFinite(T.maxH) ? T.maxH : 8) - lo)
  const step = res > 0 ? res : 0.5
  if (T.knownCount !== 0) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i
        if (!known[k]) continue
        const h0 = h[k]
        // Height gradient from the known neighbours (an unknown one counts as level).
        const hw = i > 0 && known[k - 1] ? h[k - 1] : h0
        const he = i < nx - 1 && known[k + 1] ? h[k + 1] : h0
        const hs = j > 0 && known[k - nx] ? h[k - nx] : h0
        const hn = j < ny - 1 && known[k + nx] ? h[k + nx] : h0
        const gx = ((he - hw) / (2 * step)) * RELIEF
        const gy = ((hn - hs) / (2 * step)) * RELIEF
        const shade = clamp((-gx * LIGHT[0] - gy * LIGHT[1] + LIGHT[2]) / Math.sqrt(gx * gx + gy * gy + 1) / LIGHT[2], 0.55, 1.3)
        // Ramp by height, rock on steep ground.
        const t = clamp((h0 - lo) / span, 0, 1) * (MAP_RAMP.length - 1)
        const r0 = Math.min(MAP_RAMP.length - 2, Math.floor(t))
        const f = t - r0
        const A = MAP_RAMP[r0]
        const B = MAP_RAMP[r0 + 1]
        const rock = smooth(20, 40, Number.isFinite(slope[k]) ? slope[k] : 0) * 0.6
        const o = k * 4
        for (let c = 0; c < 3; c++) {
          const base = A[c] + (B[c] - A[c]) * f
          out[o + c] = clamp(Math.round((base + (ROCK[c] - base) * rock) * shade), 0, 255)
        }
        out[o + 3] = 255
      }
    }
  }
  // The fleet's zones: discs [x, y, r] in world metres.
  for (const [key, col, mix] of ZONE_STYLE) {
    const zones = fleet?.[key]
    if (!Array.isArray(zones)) continue
    for (const z of zones) {
      if (!zoneOk(z)) continue
      const fi = (z[0] - T.ox) / step
      const fj = (z[1] - T.oy) / step
      const rr = z[2] / step
      const i0 = Math.max(0, Math.floor(fi - rr))
      const i1 = Math.min(nx - 1, Math.ceil(fi + rr))
      const j0 = Math.max(0, Math.floor(fj - rr))
      const j1 = Math.min(ny - 1, Math.ceil(fj + rr))
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          if ((i - fi) * (i - fi) + (j - fj) * (j - fj) > rr * rr) continue
          const o = (j * nx + i) * 4
          if (out[o + 3] === 255) {
            for (let c = 0; c < 3; c++) out[o + c] = Math.round(out[o + c] + (col[c] - out[o + c]) * mix)
          } else {
            // Over ground the LiDAR has not seen (water gives no returns): the zone alone.
            out[o] = col[0]
            out[o + 1] = col[1]
            out[o + 2] = col[2]
            out[o + 3] = 255
          }
        }
      }
    }
  }
  return out
}
