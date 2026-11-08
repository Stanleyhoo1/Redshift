import { mulberry } from '../scene/proceduralTextures.js'
import { SUN_DIR } from '../scene/sceneConstants.js'
import { worldPalette } from '../lib/biomes.js'

// Plain numbers for the planet view (no React, no three.js), so Node can test them.
// Directions are [x, y, z] arrays in the planet's own frame: y is the spin axis (north up),
// longitude 0 faces +z and east is +x there. Angles are degrees unless a name says not.

export const EARTH_RADIUS_KM = 6371
export const GLOBE_FOV = 35 // the planet camera's vertical field of view, degrees
export const START_ALT = 0.3 // the planet camera arrives this high over the site, planet radii
export const MIN_DIST = 1.35 // zoom limits, planet radii from the centre
export const MAX_DIST = 9
export const MARKER_R = 0.085 // landing ring radius, planet radii (a symbol, not to scale)
export const HOME_FILL = 0.72 // at home the planet fills this share of the free view height
const HOME_EAST = 0.3 // the home view leans east of the site, towards the terminator,
const HOME_SOUTH = 0.1 // and a little south, so the site sits above the middle
const SITE_MAX_LAT = 32 // the caps stay in view and the site never hides behind a pole

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const comb = (terms) => terms.reduce((o, [k, v]) => [o[0] + k * v[0], o[1] + k * v[1], o[2] + k * v[2]], [0, 0, 0])
const round1 = (v) => Math.round(v * 10) / 10

export function normalize(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

// Same density as Earth: surface gravity grows with the radius (g = 4/3 pi G rho R).
export const planetRadiusKm = (g) => Math.round((Number.isFinite(g) && g > 0 ? g : 1) * EARTH_RADIUS_KM)
export const formatKm = (km) => `${String(Math.round(km)).replace(/\B(?=(\d{3})+(?!\d))/g, ',')} km`

export function gravityG(world) {
  if (Number.isFinite(world?.gravity_g) && world.gravity_g > 0) return world.gravity_g
  if (Number.isFinite(world?.gravity) && world.gravity > 0) return world.gravity / 9.81
  return 2
}

export const planetName = (world) => (typeof world?.name === 'string' && world.name.trim() ? world.name.trim() : 'Unnamed planet')

// FNV-1a over the seed's text, so numbers and strings both give a well-mixed 32-bit value.
export function hashSeed(seed) {
  const s = String(seed ?? 0)
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

// The landing site, fixed per seed (tenths of a degree, so labels and drawing agree).
export function siteFromSeed(seed) {
  const r = mulberry(hashSeed(seed) ^ 0x51e5)
  const lat = round1((r() * 2 - 1) * SITE_MAX_LAT)
  const lon = round1(r() * 360 - 180)
  return { lat, lon: lon >= 180 ? -180 : lon }
}

export function formatLatLon({ lat, lon }) {
  const ns = lat >= 0 ? 'N' : 'S'
  const ew = lon >= 0 ? 'E' : 'W'
  return `${Math.abs(lat).toFixed(1)}° ${ns}, ${Math.abs(lon).toFixed(1)}° ${ew}`
}

export function latLonToDir(lat, lon) {
  const a = (lat * Math.PI) / 180
  const b = (lon * Math.PI) / 180
  return [Math.cos(a) * Math.sin(b), Math.sin(a), Math.cos(a) * Math.cos(b)]
}

// Local axes at a point on the planet: up (out of the ground), east and north.
export function siteFrame(lat, lon) {
  const up = latLonToDir(lat, lon)
  const east = normalize(cross([0, 1, 0], up))
  const north = cross(up, east)
  return { up, east, north }
}

// The sector's sun (SUN_DIR: x east, y up, z south) placed over the site: the same low sun
// from the west-north-west, so the site is in daylight and the shading matches the sector.
export const sunDirFor = (f) => normalize(comb([[SUN_DIR[0], f.east], [SUN_DIR[1], f.up], [-SUN_DIR[2], f.north]]))

// The home view direction (from the planet's centre to the camera).
export const homeDir = (f) => normalize(comb([[1, f.up], [HOME_EAST, f.east], [-HOME_SOUTH, f.north]]))

// Distance from the centre at which the planet fills `fill` of a view `frac` of the canvas
// height tall (the part the HUD leaves free): its angular radius a has sin a = 1 / d.
export function homeDistance(fovDeg, frac = 1, fill = HOME_FILL) {
  const t = Math.tan((fovDeg * Math.PI) / 360) * fill * clamp(frac, 0.2, 1)
  return clamp(1 / Math.sin(Math.atan(t)), MIN_DIST + 0.4, MAX_DIST)
}

export function smoother(t) {
  const x = clamp(t, 0, 1)
  return x * x * x * (x * (x * 6 - 15) + 10)
}

export const wrapAngle = (a) => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI))

// Spherical interpolation between unit vectors.
export function slerp(a, b, t) {
  const w = Math.acos(clamp(dot(a, b), -1, 1))
  if (w < 1e-6) return normalize(comb([[1 - t, a], [t, b]]))
  if (Math.PI - w < 1e-6) {
    // Opposite directions: turn through any perpendicular.
    const p = normalize(Math.abs(a[1]) < 0.9 ? cross(a, [0, 1, 0]) : cross(a, [1, 0, 0]))
    return normalize(comb([[Math.cos(Math.PI * t), a], [Math.sin(Math.PI * t), p]]))
  }
  const s = Math.sin(w)
  return comb([[Math.sin((1 - t) * w) / s, a], [Math.sin(t * w) / s, b]])
}

// Planet camera poses: { dir (camera direction from the centre), dist (planet radii), spin
// (the planet's turn about its axis, radians) }. Distance eases in log space, so equal steps
// feel equal at any height, and the spin takes the short way round.
export function lerpPose(a, b, t) {
  const k = clamp(t, 0, 1)
  return {
    dir: slerp(a.dir, b.dir, k),
    dist: Math.exp(Math.log(a.dist) + (Math.log(b.dist) - Math.log(a.dist)) * k),
    spin: a.spin + wrapAngle(b.spin - a.spin) * k,
  }
}

// A turn by `a` radians about the spin axis, the same as three.js `rotation.y = a`. So
// turnY(v, -spin) is a world direction seen from the turned planet.
export const turnY = (v, a) => [v[0] * Math.cos(a) + v[2] * Math.sin(a), v[1], v[2] * Math.cos(a) - v[0] * Math.sin(a)]

// The idle motion (GlobeRig.jsx): not an endless spin, which would carry the site to the
// night side within two minutes, but a slow sway from home out to IDLE_SWAY and back, at
// rest at both ends. A positive turn brings the camera's side of the planet towards the
// sun, so the middle of the disc only gets brighter and the site stays facing the camera.
export const IDLE_SWAY = 0.35 // radians, about 20 degrees
export const idleSway = (phase) => (IDLE_SWAY * (1 - Math.cos(phase))) / 2

// Per-world surface knobs for the planet shader (planetMaterial.js).
export function seedLook(seed) {
  const r = mulberry(hashSeed(seed) ^ 0x2545f491)
  const span = () => r() * 97 - 48.5
  return {
    offset: [span(), span(), span()], // where in noise space this world's surface comes from
    maria: 0.06 + r() * 0.12, // noise level above which the dark plains begin
    craters: 0.45 + r() * 0.35, // share of lattice cells that hold a crater
    capN: 0.86 + r() * 0.06, // sine of the latitude where the northern ice starts
    capS: 0.83 + r() * 0.08, // and the southern
    rust: 0.25 + r() * 0.5, // how much rusty stain the highlands get
  }
}

// Everything the planet view needs to know about a world.
export function planetLook(world) {
  const seed = world?.seed ?? 0
  const site = siteFromSeed(seed)
  const frame = siteFrame(site.lat, site.lon)
  const g = gravityG(world)
  return { seed, site, frame, sun: sunDirFor(frame), surface: seedLook(seed), name: planetName(world), g, radiusKm: planetRadiusKm(g), palette: worldPalette(world) }
}
