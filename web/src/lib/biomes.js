// v7 seeded biomes: the colours of each world's ground, sky and planet (sRGB hex). The sim picks a
// primary biome per seed and a secondary one in a few soft regions (init world.biome, features.biome_map);
// the dashboard draws both and blends them with the secondary share from biome_map. An older sim, the
// mock, or biomes off send neither, and everything keeps today's rust look (the `rust` palette below is
// exactly today's theme colours). Pure data and small helpers: no three.js, so node tests can import it.
import { ATMOSPHERE, REGOLITH } from './theme.js'

const RUST_GROUND = {
  dustLight: REGOLITH.dustLight,
  dust: REGOLITH.dust,
  soil: REGOLITH.soil,
  rock: REGOLITH.rock,
  rockLight: REGOLITH.rockLight,
  strataA: REGOLITH.strataA,
  strataB: REGOLITH.strataB,
  pebble: REGOLITH.pebble,
  dune: '#c99a6c',
}
const RUST_AIR = {
  zenith: ATMOSPHERE.zenith,
  sky: ATMOSPHERE.sky,
  horizon: ATMOSPHERE.horizon,
  haze: ATMOSPHERE.haze,
  groundHaze: ATMOSPHERE.groundHaze,
  sunLight: ATMOSPHERE.sunLight,
  sunGlow: ATMOSPHERE.sunGlow,
  fillSky: ATMOSPHERE.fillSky,
  fillGround: ATMOSPHERE.fillGround,
}

// ground: the terrain shader's palette (Terrain.jsx); air: sky dome, haze and light (Atmosphere.jsx);
// map: the sector map ramp, dark low ground to pale high ground (planet view and the legend swatch).
// Chosen to sit well with the violet and teal fungal trees, the lime bushes and the cyan HUD: cool,
// muted grounds, no ochre, each world with its own clear character.
export const BIOMES = {
  basalt: {
    name: 'Basalt fields',
    ground: { dustLight: '#7f8996', dust: '#59626e', soil: '#2c3139', rock: '#2a2e35', rockLight: '#5f6874', strataA: '#525b68', strataB: '#1d2127', pebble: '#8c95a0', dune: '#6c7582' },
    air: { zenith: '#262c3a', sky: '#5f6878', horizon: '#a6adb8', haze: '#8a929e', groundHaze: '#3a4048', sunLight: '#f2f0ea', sunGlow: '#d9c7b0', fillSky: '#9aa3b2', fillGround: '#2c3036' },
    map: ['#30353d', '#5c6672', '#8f99a5'],
  },
  verdigris: {
    name: 'Verdigris lowlands',
    ground: { dustLight: '#95a796', dust: '#6a7f6d', soil: '#38483d', rock: '#3c4740', rockLight: '#718276', strataA: '#5e7566', strataB: '#2b3530', pebble: '#a6b4a6', dune: '#aab48c' },
    air: { zenith: '#22323a', sky: '#5b7672', horizon: '#a9bfae', haze: '#8aa294', groundHaze: '#34443b', sunLight: '#f6f2df', sunGlow: '#e0d39c', fillSky: '#9fb5aa', fillGround: '#2f3a31' },
    map: ['#33413a', '#64796a', '#98ab9a'],
  },
  frost: {
    name: 'Frost plains',
    ground: { dustLight: '#cdd6df', dust: '#a2b0be', soil: '#748392', rock: '#58636f', rockLight: '#909dab', strataA: '#a2b3c4', strataB: '#46515e', pebble: '#f3f6f9', dune: '#cbd7e2' },
    air: { zenith: '#2d3a52', sky: '#7d90aa', horizon: '#c9d4df', haze: '#a9b7c6', groundHaze: '#56616e', sunLight: '#f4f6fb', sunGlow: '#dfe4f2', fillSky: '#b4c2d4', fillGround: '#4c5560' },
    map: ['#5a6572', '#97a6b5', '#d3dce5'],
  },
  ash: {
    name: 'Ash wastes',
    ground: { dustLight: '#a8a1b3', dust: '#7e778b', soil: '#4a4556', rock: '#433f4c', rockLight: '#746d80', strataA: '#6e637c', strataB: '#2f2b37', pebble: '#b7b0c0', dune: '#9c94aa' },
    air: { zenith: '#2b2638', sky: '#6c6279', horizon: '#b5abbb', haze: '#968c9f', groundHaze: '#433d4a', sunLight: '#f6efe9', sunGlow: '#e2c4b8', fillSky: '#aaa1b3', fillGround: '#36313c' },
    map: ['#3d3947', '#6f6880', '#a69fb2'],
  },
  salt: {
    name: 'Salt flats',
    ground: { dustLight: '#e9e2cf', dust: '#cbc1aa', soil: '#a69d88', rock: '#4d5662', rockLight: '#78828d', strataA: '#8f99a4', strataB: '#3a424c', pebble: '#f4f0e4', dune: '#e3d8b9' },
    air: { zenith: '#34405a', sky: '#8c97a8', horizon: '#e0d8c8', haze: '#c4bcac', groundHaze: '#6c675d', sunLight: '#fff8ea', sunGlow: '#f1ddb5', fillSky: '#c6c8cc', fillGround: '#5e5a50' },
    map: ['#6f6b62', '#b2ab98', '#ece6d6'],
  },
  obsidian: {
    name: 'Obsidian badlands',
    ground: { dustLight: '#4d4659', dust: '#332d3d', soil: '#1b1822', rock: '#18151d', rockLight: '#433a54', strataA: '#4f4266', strataB: '#110f16', pebble: '#625775', dune: '#3c3549' },
    air: { zenith: '#1c1628', sky: '#4a3d5e', horizon: '#8f80a0', haze: '#6f6380', groundHaze: '#26202e', sunLight: '#f3ecf6', sunGlow: '#c9a8de', fillSky: '#8d82a0', fillGround: '#1e1a24' },
    map: ['#1f1b26', '#3d354a', '#6d6182'],
  },
  rust: {
    name: 'Rust plains',
    ground: RUST_GROUND,
    air: RUST_AIR,
    map: ['#4b4440', '#8f7766', '#b19885'],
  },
}
export const BIOME_IDS = Object.keys(BIOMES)

// init world.biome, checked: { id, name, secondary } with known ids, else null (old sims, biomes off).
export function biomeOf(world) {
  const b = world?.biome
  if (!b || typeof b !== 'object' || !BIOMES[b.id]) return null
  const secondary = BIOMES[b.secondary] && b.secondary !== b.id ? b.secondary : null
  const name = typeof b.name === 'string' && b.name ? b.name : BIOMES[b.id].name
  return { id: b.id, name, secondary }
}

// features.biome_map, checked: { nx, ny, ox, oy, res, w: Float32Array } or null.
export function biomeMapOf(f) {
  const m = f?.biome_map
  if (!m || typeof m !== 'object') return null
  const nx = m.nx | 0
  const ny = m.ny | 0
  const res = Number(m.res)
  if (nx < 2 || ny < 2 || !(res > 0) || !Array.isArray(m.w) || m.w.length !== nx * ny) return null
  const ox = Number(m.origin?.[0])
  const oy = Number(m.origin?.[1])
  if (!Number.isFinite(ox) || !Number.isFinite(oy)) return null
  const w = new Float32Array(nx * ny)
  for (let k = 0; k < w.length; k++) {
    const v = Number(m.w[k])
    w[k] = Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0
  }
  return { nx, ny, ox, oy, res, w }
}

// The secondary share at world (x, y): bilinear over biome_map, 0 outside it or without one.
export function biomeWeightAt(map, x, y) {
  if (!map) return 0
  const fx = (x - map.ox) / map.res
  const fy = (y - map.oy) / map.res
  if (!(fx >= 0 && fy >= 0 && fx <= map.nx - 1 && fy <= map.ny - 1)) return 0
  const i = Math.min(map.nx - 2, Math.floor(fx))
  const j = Math.min(map.ny - 2, Math.floor(fy))
  const tx = fx - i
  const ty = fy - j
  const w = map.w
  const k = j * map.nx + i
  const a = w[k] + (w[k + 1] - w[k]) * tx
  const b = w[k + map.nx] + (w[k + map.nx + 1] - w[k + map.nx]) * tx
  return a + (b - a) * ty
}

const hex2 = (v) => Math.round(Math.min(255, Math.max(0, v))).toString(16).padStart(2, '0')
export const hexToRgb = (hex) => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)]

// Mix two sRGB hex colours (t = 0 gives a, 1 gives b), as hex.
export function mixHex(a, b, t) {
  const A = hexToRgb(a)
  const B = hexToRgb(b)
  const k = Math.min(1, Math.max(0, t))
  return '#' + A.map((c, n) => hex2(c + (B[n] - c) * k)).join('')
}

// Mix every colour of two palettes of the same shape (objects of hex, or arrays of hex).
export function mixPalette(a, b, t) {
  if (Array.isArray(a)) return a.map((c, n) => mixHex(c, b[n], t))
  const out = {}
  for (const k of Object.keys(a)) out[k] = mixHex(a[k], b[k] ?? a[k], t)
  return out
}

// The palettes for a world: { ground, air, map, ground2, map2, biome } where ground2 / map2 belong to the
// secondary biome (the primary's again when there is none). Without a biome: today's rust look.
export function worldPalette(world) {
  const b = biomeOf(world)
  const p = BIOMES[b?.id] ?? BIOMES.rust
  const s = BIOMES[b?.secondary] ?? p
  return { biome: b, ground: p.ground, air: p.air, map: p.map, ground2: s.ground, map2: s.map }
}

// The legend's "Mapped ground" swatch: the map ramp as a gradient, pale to dark; with a secondary biome,
// a sliver of its ground at the end.
export function groundSwatch(world) {
  const pal = worldPalette(world)
  const [lo, mid, hi] = pal.map
  if (!pal.biome) return `linear-gradient(135deg, ${hi} 0%, ${mid} 45%, ${lo} 100%)`
  const s = pal.map2
  return `linear-gradient(135deg, ${hi} 0%, ${mid} 40%, ${lo} 72%, ${s[1]} 74%, ${s[2]} 100%)`
}
