// Redshift mock sim.
// A protocol-compliant fake of the simulation server (see ../../PROTOCOL.md) so the
// web UI can be developed and demoed without MuJoCo. It has no real physics or
// autonomy, it only produces realistic-looking data:
//   procedural terrain, rovers driving to frontiers along A* paths, sensing within
//   a 10 m radius, target discoveries, battery drain and charging, slope-limit learning.
// Version 2 additions: per-rover sensors (a ray-marched 360 degree LiDAR over the
// heightmap, IMU, wheel slip, motor load, spectrometer), brain timings against a
// compute budget, and injectable hazards (POST /api/hazard): rockfall, soft sand,
// dust storm and motor fault. Rovers only react to what their sensors pick up.
// Version 3 additions: a fake local Gemma planner (warming for the first ~10 s, then
// 300 to 900 ms per call, stats in `planner`), Gemma-sourced events that quote sensor
// numbers, operator chat (POST /api/chat, keyword parsing with refusals and questions)
// and clarification requests (`requests`, POST /api/answer) after rockfalls, soft sand
// and once about 40 s after the start, with a wall-clock countdown to the safe default.
// Version 4 additions: lakes and a river carved into the terrain (LiDAR gets no returns
// from liquid, so they stay holes until a rover identifies them; `fleet.water_zones`
// grows as rovers look at more of the water), groves of crystal spires (LiDAR obstacles,
// `spire_fall` hazard), wandering herds of grazers (`lifeforms`), fleet `tracks` that start
// when a rover's LiDAR sees something move, a first-sighting question, rovers that switch
// to "observing" and follow at about 6 m, and point-and-click placement (x, y) for every
// hazard type with the 400 answers from the protocol.
// Version 4b additions: sample tubes (3 per rover), instruments by target kind (drill,
// scoop, sniffer, swab) with progress, on-board analyses with a science value, organic
// "trace" targets near herds, and deliveries to the lander's sample cache.
//
// Run: node mock/mock-server.mjs
// Env: PORT, SEED, NUM_ROVERS, SIM_SPEED, MAP_SIZE, GRAVITY_G, BRAIN_BUDGET_MS,
//      GEMMA_ENABLED, GEMMA_MODEL, GEMMA_BUDGET_MS, OPERATOR_TIMEOUT_S,
//      MOCK_GEMMA (offline), MOCK_GEMMA_WARMUP_S, MOCK_FIRST_QUESTION_S, MOCK_SAMPLE_SCALE,
//      MOCK_PROTOCOL (2 or 3 = an older sim without the later fields)
// Mock-only test hooks (not part of the protocol): POST /api/_mock/planner {"state"},
// POST /api/_mock/ask {"rover"?}, POST /api/_mock/battery {"rover", "battery"},
// POST /api/_mock/teleport {"rover", "x", "y", "hold"?}, POST /api/_mock/rearm {"armed"?} (first-sighting question).

import http from 'node:http'
import { WebSocketServer } from 'ws'

const PORT = Number(process.env.PORT || 8000)
const TICK_HZ = 5
const SIM_SPEED = Number(process.env.SIM_SPEED || 3)
const NUM_ROVERS = Math.max(1, Math.min(8, Number(process.env.NUM_ROVERS || 3) | 0))
const MAP_SIZE = Number(process.env.MAP_SIZE || 120)
const GRAVITY_G = Number(process.env.GRAVITY_G || 2)
const RES = 0.5
const SENSOR_RADIUS = 10
const NUM_TARGETS = 8 // plus one organic trace per herd (v4b)
const INITIAL_SLOPE_LIMIT = 30
const MIN_SLOPE_LIMIT = 18
const LANDER = { x: 0, y: 0, radius: 2.5 }
const ROVER_COLORS = ['#ff7a45', '#ffd23f', '#ff5fa2', '#7cf29a', '#b39dff', '#4dd0e1', '#f4f4f4', '#ffa8a8']
const KIND_LABEL = { mineral: 'mineral deposit', ice: 'ice patch', vent: 'thermal vent', trace: 'organic trace' }

// Version 2: sensors, edge budget and hazards.
const LIDAR_RANGE = 8
const DUST_LIDAR_RANGE = 3
const LIDAR_AZ = 72
const LIDAR_EL = [-40, -31, -24, -19, -15, -12, -9.5, -7.5, -6, -4.7, -3.6, -2.5, -1, 2].map((d) => (d * Math.PI) / 180)
const MAST_H = 0.75
const SCAN_MAX = 300
const SPECTRO_RANGE = 8
const BUDGET_MS = Number(process.env.BRAIN_BUDGET_MS || 40)
const CYCLE_MS = 200
const BRAIN_WINDOW_S = 10
const DUST_S = 90
const FAULT_S = 90

// Version 3: local Gemma planner, operator chat and clarification requests.
const GEMMA_MODEL = process.env.GEMMA_MODEL || 'gemma4:e2b'
const GEMMA_BUDGET_MS = Number(process.env.GEMMA_BUDGET_MS || 3000)
const GEMMA_WARMUP_S = Number(process.env.MOCK_GEMMA_WARMUP_S || 10)
const OPERATOR_TIMEOUT_S = Number(process.env.OPERATOR_TIMEOUT_S || 20)
const FIRST_QUESTION_S = Number(process.env.MOCK_FIRST_QUESTION_S || 40)
const PROTOCOL = Number(process.env.MOCK_PROTOCOL || 4) // 2 or 3 = behave like an older sim (no later fields or endpoints)
const HAZARD_TYPES = ['rockfall', 'sand', 'dust_storm', 'rover_fault', ...(PROTOCOL >= 4 ? ['spire_fall', 'lifeform'] : [])]

// Version 4: water, spires, lifeforms and tracks.
const IDENTIFY_RANGE = 4.5 // m from the water's edge at which a rover works out it is liquid
const COVER_SPACING = 3.4 // water_zones are discs on this grid over each body of water
const COVER_RADIUS = 3.1
const TRACK_LOST_S = 6 // sim s without a LiDAR sighting before a track is lost
const TRACK_DROP_S = 30 // and then dropped
const OBSERVE_DIST = 6
const OBSERVE_S = 45
const MAX_LIFEFORMS = 30

// Version 4b: sample collection.
const TUBES = 3
const INSTRUMENT_FOR = { mineral: 'drill', ice: 'scoop', vent: 'sniffer', trace: 'swab' }
const SAMPLE_SCALE = Math.max(0.1, Number(process.env.MOCK_SAMPLE_SCALE || 1)) // mock only: longer sampling for screenshots on slow machines
const SAMPLE_S = Object.fromEntries(Object.entries({ drill: 16, scoop: 9, sniffer: 11, swab: 7 }).map(([k, v]) => [k, Math.round(v * SAMPLE_SCALE)]))
const ANALYSES = {
  mineral: [
    [8, 'Olivine-rich basalt with water-bearing clays'],
    [6, 'Iron oxide crust over fine-grained basalt'],
    [9, 'Layered sulfates: long-lived standing water'],
    [4, 'Plain basalt, heavily weathered'],
  ],
  ice: [
    [7, 'Water ice with 3% perchlorate salts'],
    [9, 'Clean water ice, trapped gas bubbles'],
    [5, 'Frost over dusty regolith'],
  ],
  vent: [
    [8, 'Methane and hydrogen sulfide, warm outflow'],
    [6, 'Sulfur dioxide plume, 41 C at the mouth'],
    [3, 'Mostly carbon dioxide, faint heat'],
  ],
  trace: [
    [10, 'Complex organics: amino-acid-like chains'],
    [9, 'Mucus-like film with cell-sized structures'],
    [7, 'Organic-rich droppings, high nitrogen'],
  ],
}
const ENERGY_PER_M = 0.0064 // battery fraction per metre driven (0.0032 per s at about 0.5 m/s)
const BATTERY_RESERVE = 0.2
const wallNow = () => performance.now() / 1000
const DIRS = {
  north: [0, 1],
  south: [0, -1],
  east: [1, 0],
  west: [-1, 0],
  'north-east': [Math.SQRT1_2, Math.SQRT1_2],
  'north-west': [-Math.SQRT1_2, Math.SQRT1_2],
  'south-east': [Math.SQRT1_2, -Math.SQRT1_2],
  'south-west': [-Math.SQRT1_2, -Math.SQRT1_2],
}

// ---------------------------------------------------------------------------
// Small helpers

function mulberry32(seed) {
  let a = seed >>> 0
  return function rand() {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
const lerp = (a, b, t) => a + (b - a) * t
const smoothstep = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1)
  return t * t * (3 - 2 * t)
}
const r1 = (v) => Math.round(v * 10) / 10
const r2 = (v) => Math.round(v * 100) / 100
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a))

function makePerlin(rand) {
  const perm = Array.from({ length: 256 }, (_, i) => i)
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[perm[i], perm[j]] = [perm[j], perm[i]]
  }
  const p = new Uint8Array(512)
  for (let i = 0; i < 512; i++) p[i] = perm[i & 255]
  const gx = [1, -1, 1, -1, 1, -1, 0, 0]
  const gy = [1, 1, -1, -1, 0, 0, 1, -1]
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10)
  return function noise(x, y) {
    const xi = Math.floor(x)
    const yi = Math.floor(y)
    const xf = x - xi
    const yf = y - yi
    const X = xi & 255
    const Y = yi & 255
    const g = (h, dx, dy) => gx[h & 7] * dx + gy[h & 7] * dy
    const aa = p[p[X] + Y]
    const ab = p[p[X] + Y + 1]
    const ba = p[p[X + 1] + Y]
    const bb = p[p[X + 1] + Y + 1]
    const u = fade(xf)
    const v = fade(yf)
    return lerp(lerp(g(aa, xf, yf), g(ba, xf - 1, yf), u), lerp(g(ab, xf, yf - 1), g(bb, xf - 1, yf - 1), u), v)
  }
}

function fbm(noise, x, y, octaves, gain) {
  let sum = 0
  let amp = 1
  let freq = 1
  let norm = 0
  for (let o = 0; o < octaves; o++) {
    sum += amp * noise(x * freq, y * freq)
    norm += amp
    amp *= gain
    freq *= 2
  }
  return sum / norm
}

// Plane geometry for water outlines and fallen spires.
function pointInPoly(x, y, poly) {
  let inside = false
  for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
    const [xa, ya] = poly[a]
    const [xb, yb] = poly[b]
    if (ya > y !== yb > y && x < ((xb - xa) * (y - ya)) / (yb - ya) + xa) inside = !inside
  }
  return inside
}
function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax
  const dy = by - ay
  const len2 = dx * dx + dy * dy || 1e-9
  const t = clamp(((px - ax) * dx + (py - ay) * dy) / len2, 0, 1)
  return Math.hypot(ax + dx * t - px, ay + dy * t - py)
}
function distToPoly(x, y, poly) {
  let d = Infinity
  for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) d = Math.min(d, segDist(x, y, poly[b][0], poly[b][1], poly[a][0], poly[a][1]))
  return d
}
const compass = (dx, dy) => ['east', 'north-east', 'north', 'north-west', 'west', 'south-west', 'south', 'south-east'][((Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8]

// ---------------------------------------------------------------------------
// World generation

function generateWorld(seed) {
  const rand = mulberry32(seed * 7919 + 17)
  const n = Math.round(MAP_SIZE / RES) + 1
  const ox = -MAP_SIZE / 2
  const oy = -MAP_SIZE / 2
  const hills = makePerlin(rand)
  const ridges = makePerlin(rand)
  const detail = makePerlin(rand)
  const offx = rand() * 200
  const offy = rand() * 200
  const phase1 = rand() * Math.PI * 2
  const phase2 = rand() * Math.PI * 2

  const half = MAP_SIZE / 2
  const awayFromLander = (minR) => {
    for (;;) {
      const x = (rand() * 2 - 1) * (half - 6)
      const y = (rand() * 2 - 1) * (half - 6)
      if (Math.hypot(x - LANDER.x, y - LANDER.y) > minR) return [x, y]
    }
  }
  const craters = []
  const craterCount = 5 + Math.floor(rand() * 4)
  for (let c = 0; c < craterCount; c++) {
    const [x, y] = awayFromLander(13)
    const r = 3 + rand() * 6
    craters.push({ x, y, r, depth: r * (0.28 + rand() * 0.18), rim: r * (0.12 + rand() * 0.1) })
  }
  const mesas = []
  const mesaCount = 1 + Math.floor(rand() * 2)
  for (let m = 0; m < mesaCount; m++) {
    const [x, y] = awayFromLander(16)
    mesas.push({ x, y, r: 4 + rand() * 4, h: 2.2 + rand() * 1.8, edge: 1.6 + rand() * 0.8 })
  }

  const heights = new Float32Array(n * n)
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = ox + i * RES
      const y = oy + j * RES
      let h = 2.4 * fbm(hills, (x + offx) / 24, (y + offy) / 24, 4, 0.45)
      const rr = 1 - Math.abs(ridges((x - offy) / 17, (y + offx) / 17))
      const ridgeMask = smoothstep(-0.1, 0.35, hills((x + 31) / 40, (y - 17) / 40))
      h += 2.6 * rr * rr * rr * ridgeMask
      h += 0.7 * Math.sin(x * 0.085 + phase1) * Math.cos(y * 0.07 + phase2)
      h += 0.12 * detail(x / 2.2, y / 2.2)
      for (const c of craters) {
        const d = Math.hypot(x - c.x, y - c.y) / c.r
        if (d < 1.8) {
          if (d < 1) h -= c.depth * (1 - d * d)
          h += c.rim * Math.exp(-(((d - 1) / 0.22) ** 2))
        }
      }
      for (const m of mesas) {
        const d = Math.hypot(x - m.x, y - m.y)
        h += m.h * smoothstep(m.r + m.edge, m.r, d)
      }
      heights[j * n + i] = h
    }
  }

  // Flatten a landing zone around the lander and make the lander sit at h = 0.
  const li = Math.round((LANDER.x - ox) / RES)
  const lj = Math.round((LANDER.y - oy) / RES)
  const h0 = heights[lj * n + li]
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = ox + i * RES
      const y = oy + j * RES
      const w = smoothstep(4.5, 10, Math.hypot(x - LANDER.x, y - LANDER.y))
      const k = j * n + i
      heights[k] = lerp(h0, heights[k], w) - h0
    }
  }

  // v4: two lakes and a short river, carved into the terrain at a flat water level.
  const water = PROTOCOL >= 4 ? carveWater(rand, heights, n, ox, oy) : []
  const waterId = new Int8Array(n * n).fill(-1)
  water.forEach((w, wi) => {
    const xs = w.outline.map((p) => p[0])
    const ys = w.outline.map((p) => p[1])
    for (let j = Math.max(0, Math.floor((Math.min(...ys) - oy) / RES)); j <= Math.min(n - 1, Math.ceil((Math.max(...ys) - oy) / RES)); j++) {
      for (let i = Math.max(0, Math.floor((Math.min(...xs) - ox) / RES)); i <= Math.min(n - 1, Math.ceil((Math.max(...xs) - ox) / RES)); i++) {
        if (pointInPoly(ox + i * RES, oy + j * RES, w.outline)) waterId[j * n + i] = wi
      }
    }
  })
  const nearWater = (x, y, d) => water.some((w) => pointInPoly(x, y, w.outline) || distToPoly(x, y, w.outline) < d)

  const slopes = computeSlopes(heights, n, RES)
  let minH = Infinity
  let maxH = -Infinity
  for (const h of heights) {
    if (h < minH) minH = h
    if (h > maxH) maxH = h
  }

  // Targets on gentle ground, away from the lander and from each other.
  const kinds = []
  for (let k = 0; k < NUM_TARGETS; k++) kinds.push(['mineral', 'ice', 'vent'][k % 3])
  for (let k = kinds.length - 1; k > 0; k--) {
    const s = Math.floor(rand() * (k + 1))
    ;[kinds[k], kinds[s]] = [kinds[s], kinds[k]]
  }
  const targets = []
  let guard = 0
  while (targets.length < NUM_TARGETS && guard++ < 5000) {
    const [x, y] = awayFromLander(9)
    const i = Math.round((x - ox) / RES)
    const j = Math.round((y - oy) / RES)
    if (slopes[j * n + i] > 16) continue
    if (targets.some((t) => Math.hypot(t.x - x, t.y - y) < 9)) continue
    if (nearWater(x, y, 2.5)) continue
    targets.push({ id: `t${targets.length + 1}`, x: r1(x), y: r1(y), kind: kinds[targets.length] })
  }

  const slopeAtXY = (x, y) => slopes[clamp(Math.round((y - oy) / RES), 0, n - 1) * n + clamp(Math.round((x - ox) / RES), 0, n - 1)]
  const dry = (x, y, minLander, maxSlope) =>
    Math.abs(x) < half - 5 && Math.abs(y) < half - 5 && Math.hypot(x - LANDER.x, y - LANDER.y) > minLander && slopeAtXY(x, y) <= maxSlope && !nearWater(x, y, 2)

  // v4: groves of crystal spires (the first one close to the lander, so it shows early).
  const spires = []
  if (PROTOCOL >= 4) {
    for (let g = 0; g < 4; g++) {
      let c = null
      for (let tries = 0; tries < 200 && !c; tries++) {
        const a = rand() * Math.PI * 2
        const d = g === 0 ? 11 + rand() * 4 : 18 + rand() * 30
        const x = Math.cos(a) * d
        const y = Math.sin(a) * d
        if (dry(x, y, 9, 20) && targets.every((t) => Math.hypot(t.x - x, t.y - y) > 5)) c = [x, y]
      }
      if (!c) continue
      const count = 3 + Math.floor(rand() * 3)
      for (let k = 0, tries = 0; k < count && tries < 80; tries++) {
        const a = rand() * Math.PI * 2
        const d = k === 0 ? 0 : 1.6 + rand() * 3.6
        const x = c[0] + Math.cos(a) * d
        const y = c[1] + Math.sin(a) * d
        if (!dry(x, y, 8, 22) || spires.some((q) => Math.hypot(q.x - x, q.y - y) < 2.2) || targets.some((t) => Math.hypot(t.x - x, t.y - y) < 2.5)) continue
        spires.push({ id: `s${spires.length + 1}`, x: r1(x), y: r1(y), height: r1(2.2 + rand() * 2.2), radius: r2(0.28 + rand() * 0.17), standing: true })
        k++
      }
    }
  }

  // v4: herds of grazers, and (v4b) the organic traces they leave behind.
  const herds = []
  if (PROTOCOL >= 4) {
    for (let h = 0; h < 3; h++) {
      let c = null
      for (let tries = 0; tries < 300 && !c; tries++) {
        const a = rand() * Math.PI * 2
        const d = 26 + rand() * 20
        const x = Math.cos(a) * d
        const y = Math.sin(a) * d
        if (dry(x, y, 14, 16) && !nearWater(x, y, 5) && herds.every((q) => Math.hypot(q.cx - x, q.cy - y) > 16)) c = [x, y]
      }
      if (!c) continue
      herds.push({ cx: c[0], cy: c[1], count: 3 + Math.floor(rand() * 3) })
      for (let tries = 0; tries < 60; tries++) {
        const a = rand() * Math.PI * 2
        const d = 2 + rand() * 3
        const x = c[0] + Math.cos(a) * d
        const y = c[1] + Math.sin(a) * d
        if (!dry(x, y, 12, 16) || targets.some((t) => Math.hypot(t.x - x, t.y - y) < 4)) continue
        targets.push({ id: `t${targets.length + 1}`, x: r1(x), y: r1(y), kind: 'trace' })
        break
      }
    }
  }

  // Discs over each body of water: what water_zones is built from as rovers identify it.
  const cover = water.map((w) => {
    const out = []
    const xs = w.outline.map((p) => p[0])
    const ys = w.outline.map((p) => p[1])
    for (let y = Math.min(...ys); y <= Math.max(...ys); y += COVER_SPACING) {
      for (let x = Math.min(...xs); x <= Math.max(...xs); x += COVER_SPACING) {
        if (pointInPoly(x, y, w.outline)) out.push({ x: r1(x), y: r1(y), r: COVER_RADIUS, seen: false })
      }
    }
    let acc = 99
    for (let k = 0; k < w.outline.length; k++) {
      const [ax, ay] = w.outline[k]
      const [bx, by] = w.outline[(k + 1) % w.outline.length]
      acc += Math.hypot(bx - ax, by - ay)
      if (acc < 2.8) continue
      acc = 0
      if (out.some((c) => Math.hypot(c.x - ax, c.y - ay) < 1.6)) continue
      out.push({ x: r1(ax), y: r1(ay), r: r1(COVER_RADIUS * 0.75), seen: false })
    }
    return out
  })

  return { seed, n, ox, oy, heights, slopes, minH, maxH, targets, water, waterId, cover, spires, herds }
}

// Carves the lakes and the river; returns [{ id, kind, level, outline }].
function carveWater(rand, heights, n, ox, oy) {
  const half = MAP_SIZE / 2
  const orig = Float32Array.from(heights)
  const hAt = (x, y) => orig[clamp(Math.round((y - oy) / RES), 0, n - 1) * n + clamp(Math.round((x - ox) / RES), 0, n - 1)]
  const out = []
  const used = []
  const pickDir = () => {
    for (let tries = 0; tries < 100; tries++) {
      const a = rand() * Math.PI * 2
      if (used.every((b) => Math.abs(wrapAngle(a - b)) > 1.4)) {
        used.push(a)
        return a
      }
    }
    return rand() * Math.PI * 2
  }
  const each = (x0, x1, y0, y1, fn) => {
    for (let j = Math.max(0, Math.floor((y0 - oy) / RES)); j <= Math.min(n - 1, Math.ceil((y1 - oy) / RES)); j++) {
      for (let i = Math.max(0, Math.floor((x0 - ox) / RES)); i <= Math.min(n - 1, Math.ceil((x1 - ox) / RES)); i++) fn(i, j, ox + i * RES, oy + j * RES, j * n + i)
    }
  }
  // Lakes: a wobbly outline, a bowl under it and a beach that rises back to the terrain.
  const lakes = [
    [19 + rand() * 4, 6 + rand() * 2],
    [33 + rand() * 8, 6.5 + rand() * 3],
  ]
  for (const [dist, R] of lakes) {
    const a = pickDir()
    const lim = half - R * 1.5 - 6
    const cx = clamp(Math.cos(a) * dist, -lim, lim)
    const cy = clamp(Math.sin(a) * dist, -lim, lim)
    const p1 = rand() * 6.28
    const p2 = rand() * 6.28
    const a1 = 0.14 + rand() * 0.1
    const a2 = 0.06 + rand() * 0.06
    const rAt = (th) => R * (1 + a1 * Math.sin(3 * th + p1) + a2 * Math.sin(5 * th + p2))
    const outline = []
    let ring = 0
    for (let k = 0; k < 44; k++) {
      const th = (k / 44) * Math.PI * 2
      const r = rAt(th)
      outline.push([r1(cx + Math.cos(th) * r), r1(cy + Math.sin(th) * r)])
      ring += hAt(cx + Math.cos(th) * r * 1.2, cy + Math.sin(th) * r * 1.2)
    }
    const level = r2(ring / 44 - 0.1)
    const ext = R * 1.4 * 1.7
    each(cx - ext, cx + ext, cy - ext, cy + ext, (i, j, x, y, k) => {
      const q = Math.hypot(x - cx, y - cy) / rAt(Math.atan2(y - cy, x - cx))
      if (pointInPoly(x, y, outline)) heights[k] = Math.min(heights[k], level - 0.2 - 1.5 * (1 - Math.min(1, q) ** 2))
      else if (q < 1.6) {
        const qq = Math.max(1, q)
        heights[k] = Math.max(level + 0.05, lerp(level + 0.08 + 0.5 * (qq - 1), heights[k], smoothstep(1, 1.6, qq)))
      }
    })
    out.push({ id: `w${out.length + 1}`, kind: 'lake', level, outline })
  }
  // A short river: a bending channel with banks.
  {
    const a = pickDir()
    const d = 30 + rand() * 8
    let x = Math.cos(a) * d
    let y = Math.sin(a) * d
    let heading = a + Math.PI / 2 + (rand() - 0.5)
    const pts = [[x, y]]
    for (let k = 0; k < 5; k++) {
      heading += (rand() - 0.5) * 0.7
      x = clamp(x + Math.cos(heading) * 6.5, -half + 8, half - 8)
      y = clamp(y + Math.sin(heading) * 6.5, -half + 8, half - 8)
      pts.push([x, y])
    }
    // Resample the centre line every metre, with a width that tapers at both ends.
    const line = []
    for (let k = 0; k < pts.length - 1; k++) {
      const steps = Math.ceil(Math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]))
      for (let q = 0; q < steps; q++) line.push([lerp(pts[k][0], pts[k + 1][0], q / steps), lerp(pts[k][1], pts[k + 1][1], q / steps)])
    }
    line.push(pts[pts.length - 1])
    const W = 1.5 + rand() * 0.4
    const width = (t) => W * (0.55 + 0.45 * Math.sin(Math.PI * clamp(t, 0, 1)))
    const left = []
    const right = []
    let sum = 0
    line.forEach(([px, py], k) => {
      const [ax, ay] = line[Math.max(0, k - 1)]
      const [bx, by] = line[Math.min(line.length - 1, k + 1)]
      const len = Math.hypot(bx - ax, by - ay) || 1
      const nx = -(by - ay) / len
      const ny = (bx - ax) / len
      const w = width(k / (line.length - 1))
      if (k % 2 === 0 || k === line.length - 1) {
        left.push([r1(px + nx * w), r1(py + ny * w)])
        right.push([r1(px - nx * w), r1(py - ny * w)])
      }
      sum += hAt(px, py)
    })
    const outline = [...left, ...right.reverse()]
    const level = r2(sum / line.length - 0.15)
    const xs = line.map((p) => p[0])
    const ys = line.map((p) => p[1])
    each(Math.min(...xs) - W - 3, Math.max(...xs) + W + 3, Math.min(...ys) - W - 3, Math.max(...ys) + W + 3, (i, j, px, py, k) => {
      let best = Infinity
      let bt = 0
      for (let q = 0; q < line.length - 1; q++) {
        const dd = segDist(px, py, line[q][0], line[q][1], line[q + 1][0], line[q + 1][1])
        if (dd < best) {
          best = dd
          bt = q / (line.length - 1)
        }
      }
      const w = width(bt)
      if (pointInPoly(px, py, outline)) heights[k] = Math.min(heights[k], level - 0.2 - 0.6 * (1 - Math.min(1, best / w) ** 2))
      else if (best < w + 2.5) heights[k] = Math.max(level + 0.05, lerp(level + 0.08 + 0.4 * (best - w), heights[k], smoothstep(w, w + 2.5, best)))
    })
    out.push({ id: `w${out.length + 1}`, kind: 'river', level, outline })
  }
  return out
}

function computeSlopes(heights, n, res) {
  const slopes = new Float32Array(n * n)
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const il = Math.max(0, i - 1)
      const ir = Math.min(n - 1, i + 1)
      const jd = Math.max(0, j - 1)
      const ju = Math.min(n - 1, j + 1)
      const hx = (heights[j * n + ir] - heights[j * n + il]) / ((ir - il) * res)
      const hy = (heights[ju * n + i] - heights[jd * n + i]) / ((ju - jd) * res)
      slopes[j * n + i] = (Math.atan(Math.hypot(hx, hy)) * 180) / Math.PI
    }
  }
  return slopes
}

// ---------------------------------------------------------------------------
// Fake local Gemma (v3). It lives outside the sim: a reset does not unload the model.

const planner = {
  mode: process.env.GEMMA_ENABLED === '0' || PROTOCOL < 3 ? 'disabled' : process.env.MOCK_GEMMA === 'offline' ? 'offline' : 'auto',
  readyAt: wallNow() + GEMMA_WARMUP_S,
  calls: 0,
  ok: 0,
  fallbacks: 0,
  asked: 0,
  over: 0,
  last: 0,
  lat: [], // wall ms of the last 200 answered calls
}

function plannerState() {
  if (planner.mode !== 'auto') return planner.mode
  return wallNow() < planner.readyAt ? 'warming' : 'ready'
}
const gemmaReady = () => plannerState() === 'ready'

// One model call. Returns { ok, ms }: ok false means the rules decide (fallback).
// Image prompts (anomalies seen by the camera) take a little longer than text ones.
function gemmaCall(kind) {
  const state = plannerState()
  if (state === 'disabled') return { ok: false, ms: 0 }
  planner.calls++
  if (state !== 'ready') {
    planner.fallbacks++
    return { ok: false, ms: 0 }
  }
  const rnd = Math.random
  if (rnd() < 0.008) {
    // No answer within GEMMA_TIMEOUT_S (4 s): the rules' own choice stands.
    planner.fallbacks++
    planner.over++
    return { ok: false, ms: 0 }
  }
  let ms = 300 + rnd() * 420 + (rnd() < 0.3 ? rnd() * 200 : 0)
  if (kind === 'vision') ms += 90 + rnd() * 160
  if (rnd() < 0.04) ms = 1050 + rnd() * 600
  ms = Math.round(ms)
  planner.ok++
  planner.last = ms
  planner.lat.push(ms)
  if (planner.lat.length > 200) planner.lat.shift()
  if (ms > GEMMA_BUDGET_MS) planner.over++
  return { ok: true, ms }
}

function plannerJson() {
  const state = plannerState()
  const lat = [...planner.lat].sort((a, b) => a - b)
  const avg = lat.length ? lat.reduce((s, v) => s + v, 0) / lat.length : 0
  const p95 = lat.length ? lat[Math.min(lat.length - 1, Math.ceil(lat.length * 0.95) - 1)] : 0
  return {
    model: GEMMA_MODEL,
    state,
    online: state === 'ready',
    calls: planner.calls,
    ok: planner.ok,
    fallbacks: planner.fallbacks,
    avg_ms: r1(avg),
    p95_ms: r1(p95),
    last_ms: r1(planner.last),
    budget_ms: GEMMA_BUDGET_MS,
    over_budget: planner.over,
    asked_operator: planner.asked,
  }
}

// ---------------------------------------------------------------------------
// Simulation state

let world
let sim

// Run fn after a wall-clock delay, unless the sim was reset in the meantime.
function later(ms, fn) {
  const owner = sim
  setTimeout(() => {
    if (sim === owner) fn()
  }, Math.max(0, ms))
}

function newSim(seed) {
  world = generateWorld(seed)
  const rand = mulberry32(seed * 104729 + 3)
  const n = world.n
  sim = {
    rand,
    t: 0,
    known: new Uint8Array(n * n),
    measured: new Float32Array(n * n),
    mslope: new Float32Array(n * n), // measured slope (boulders raise it)
    knownCount: 0,
    pendingCells: [],
    rovers: [],
    discovered: new Map(),
    events: [],
    pendingEvents: [],
    frontiers: [],
    slopeLimit: INITIAL_SLOPE_LIMIT,
    nextStallT: 40 + rand() * 20,
    nextSlipT: 70 + rand() * 40,
    milestones: [10, 25, 50, 75, 90],
    hazards: [],
    nextHazard: 1,
    rocks: [], // { x, y, z, r, landT, seen, hazard }
    softZones: [], // what the fleet believes: [x, y, r]
    // v3
    chat: [], // last 50, for init
    pendingChat: [],
    nextChat: 1,
    requests: [], // open clarification questions
    closedRequests: new Map(), // id -> 'answered' | 'expired' | 'cancelled'
    nextRequest: 1,
    firstQuestionAt: wallNow() + FIRST_QUESTION_S,
    firstAsked: false,
    // v4
    waterSeen: new Uint8Array(n * n), // water cells inside identified zones (count as explored)
    waterSeenCount: 0,
    waterZones: [], // [x, y, r] the fleet has identified
    identified: new Set(), // water body indices
    spireSeen: new Set(),
    lifeforms: [],
    herds: [],
    nextLife: 1,
    tracks: [],
    nextTrack: 1,
    lifeConfirmed: new Set(), // lifeform ids the fleet has confirmed
    firstLifeAsked: false,
    nextSpire: world.spires.length + 1,
    featuresVersion: 0, // bumped when a spire falls or appears: the next tick resends `features`
    featuresSent: 0,
  }
  for (const h of world.herds) spawnHerd(h.cx, h.cy, h.count)
  for (let r = 0; r < NUM_ROVERS; r++) {
    const a = (r / NUM_ROVERS) * Math.PI * 2 + Math.PI / 6
    const x = LANDER.x + Math.cos(a) * 3.4
    const y = LANDER.y + Math.sin(a) * 3.4
    sim.rovers.push({
      id: `r${r + 1}`,
      name: `Rover ${r + 1}`,
      color: ROVER_COLORS[r % ROVER_COLORS.length],
      dockAngle: a,
      x,
      y,
      yaw: a,
      speed: 0,
      battery: clamp(1 - r * 0.13, 0.4, 1),
      state: 'idle',
      goal: null,
      path: [],
      targetId: null,
      timer: 1 + r * 0.8,
      replanT: 0,
      lastGoalEventT: -999,
      // v2
      replannedAt: -999,
      reactedAt: -999,
      sandT: 0,
      slipBoost: null,
      scan: [],
      hits: 0,
      sensors: null,
      cycles: [], // [t, ms] over the last BRAIN_WINDOW_S
      faultHazard: null,
      // v3
      request: null, // its open clarification question
      hold: false, // told to hold (operator, or "wait" as an answer)
      holdUntil: null,
      slowUntil: -999,
      // v4
      trackId: null,
      observeMode: null, // 'follow' | 'watch'
      observeUntil: 0,
      observeReplanT: 0,
      // v4b
      samples: [],
      sampling: null, // { target, instrument, progress, total }
    })
  }
  addEvent(null, 'info', `Survey started. ${NUM_ROVERS} rover${NUM_ROVERS === 1 ? '' : 's'} deployed from the lander.`)
  for (const r of sim.rovers) sense(r)
  updateSensors(CYCLE_MS / 1000)
  computeFrontiers()
}

function addEvent(roverId, level, msg, at = null, source = null) {
  const e = { t: r1(sim.t), rover: roverId, level, msg }
  if (at) e.at = [r1(at[0]), r1(at[1])]
  if (source) e.source = source
  sim.pendingEvents.push(e)
  sim.events.push(e)
  if (sim.events.length > 50) sim.events.splice(0, sim.events.length - 50)
}

const idx = (i, j) => j * world.n + i
const toI = (x) => Math.round((x - world.ox) / RES)
const toJ = (y) => Math.round((y - world.oy) / RES)
const inMap = (i, j) => i >= 0 && j >= 0 && i < world.n && j < world.n

function groundAt(x, y) {
  const i = clamp(toI(x), 0, world.n - 1)
  const j = clamp(toJ(y), 0, world.n - 1)
  return world.heights[idx(i, j)]
}
function slopeAt(x, y) {
  const i = clamp(toI(x), 0, world.n - 1)
  const j = clamp(toJ(y), 0, world.n - 1)
  return world.slopes[idx(i, j)]
}

// Bilinear height of the true terrain (for the LiDAR rays and the IMU).
function groundSmooth(x, y) {
  const fx = clamp((x - world.ox) / RES, 0, world.n - 1.001)
  const fy = clamp((y - world.oy) / RES, 0, world.n - 1.001)
  const i = Math.floor(fx)
  const j = Math.floor(fy)
  const tx = fx - i
  const ty = fy - j
  const h = world.heights
  const k = j * world.n + i
  return lerp(lerp(h[k], h[k + 1], tx), lerp(h[k + world.n], h[k + world.n + 1], tx), ty)
}

function gaussian(rand) {
  const u = Math.max(1e-9, rand())
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
}

const dustActive = () => sim.hazards.some((h) => h.type === 'dust_storm' && h.active)
const lidarRange = () => (dustActive() ? DUST_LIDAR_RANGE : LIDAR_RANGE)

function sense(rover) {
  // Mapping follows the LiDAR: a dust storm shrinks what each rover can add to the map.
  const R = Math.round((dustActive() ? DUST_LIDAR_RANGE + 0.5 : SENSOR_RADIUS) / RES)
  const ci = toI(rover.x)
  const cj = toJ(rover.y)
  for (let dj = -R; dj <= R; dj++) {
    for (let di = -R; di <= R; di++) {
      if (di * di + dj * dj > R * R) continue
      const i = ci + di
      const j = cj + dj
      if (!inMap(i, j)) continue
      const k = idx(i, j)
      if (sim.known[k]) continue
      if (world.waterId[k] >= 0) continue // v4: no LiDAR returns from liquid
      sim.known[k] = 1
      sim.knownCount++
      const h = r2(world.heights[k] + gaussian(sim.rand) * 0.015)
      sim.measured[k] = h
      sim.mslope[k] = world.slopes[k]
      sim.pendingCells.push([i, j, h, r1(world.slopes[k])])
    }
  }
  if (PROTOCOL >= 4) {
    senseSpires(rover)
    senseWater(rover)
  }
  for (const t of world.targets) {
    if (sim.discovered.has(t.id)) continue
    if (Math.hypot(t.x - rover.x, t.y - rover.y) > (dustActive() ? DUST_LIDAR_RANGE + 2 : SENSOR_RADIUS - 1)) continue
    sim.discovered.set(t.id, { ...t, status: 'found', found_by: rover.id, found_t: r1(sim.t), assigned: null })
    const label = KIND_LABEL[t.kind]
    addEvent(rover.id, 'discovery', `Found ${/^[aeiou]/.test(label) ? 'an' : 'a'} ${label} at (${t.x.toFixed(1)}, ${t.y.toFixed(1)}).`)
  }
}

// v4: spires are LiDAR obstacles. Their footprint goes into the shared map as steep
// ground (the height stays the ground's, so the crystal is not buried in a terrain spike).
function markFootprint(cells, x0, y0, x1, y1, r) {
  const R = Math.ceil((Math.hypot(x1 - x0, y1 - y0) / 2 + r + 0.4) / RES)
  const ci = toI((x0 + x1) / 2)
  const cj = toJ((y0 + y1) / 2)
  for (let dj = -R; dj <= R; dj++) {
    for (let di = -R; di <= R; di++) {
      const i = ci + di
      const j = cj + dj
      if (!inMap(i, j)) continue
      const d = segDist(world.ox + i * RES, world.oy + j * RES, x0, y0, x1, y1)
      if (d > r + 0.15) continue
      const k = idx(i, j)
      if (world.waterId[k] >= 0) continue
      if (!sim.known[k]) {
        sim.known[k] = 1
        sim.knownCount++
        sim.measured[k] = r2(world.heights[k])
      }
      sim.mslope[k] = Math.max(sim.mslope[k], 46 - d * 6)
      cells.push([i, j, sim.measured[k], r1(sim.mslope[k])])
    }
  }
}

function spireSegment(sp) {
  if (sp.standing) return [sp.x, sp.y, sp.x, sp.y]
  return [sp.x0 ?? sp.x, sp.y0 ?? sp.y, sp.x1 ?? sp.x, sp.y1 ?? sp.y]
}

function senseSpires(rover) {
  const range = lidarRange()
  for (const sp of world.spires) {
    if (sp.fallT && sim.t < sp.fallT) continue // still falling
    const [x0, y0, x1, y1] = spireSegment(sp)
    const key = `${sp.id}:${sp.standing ? 'up' : 'down'}`
    if (sim.spireSeen.has(key) || segDist(rover.x, rover.y, x0, y0, x1, y1) > range) continue
    sim.spireSeen.add(key)
    markFootprint(sim.pendingCells, x0, y0, x1, y1, sp.radius)
    if (!sp.standing) {
      const dist = Math.round(segDist(rover.x, rover.y, x0, y0, x1, y1))
      const blocking = rover.path.length && pathNear(rover, (x0 + x1) / 2, (y0 + y1) / 2, Math.hypot(x1 - x0, y1 - y0) / 2 + 1)
      const changed = replanAround((x0 + x1) / 2, (y0 + y1) / 2, Math.hypot(x1 - x0, y1 - y0) / 2 + 1.2)
      rover.reactedAt = sim.t
      addEvent(
        rover.id,
        'warn',
        blocking || changed.includes(rover)
          ? `LiDAR: a ${r1(Math.hypot(x1 - x0, y1 - y0))} m spire now lies across the route ${dist} m ahead. Marked it impassable and replanned.`
          : `LiDAR: a fallen spire ${dist} m away, ${r1(Math.hypot(x1 - x0, y1 - y0))} m long. Marked it on the shared map.`,
        [(x0 + x1) / 2, (y0 + y1) / 2],
      )
    }
  }
}

// v4: a rover close to a LiDAR hole looks at it (camera, or the rules: flat edges all at
// one height) and works out it is liquid. The identified area grows as rovers see more.
function senseWater(rover) {
  world.water.forEach((w, wi) => {
    const cover = world.cover[wi]
    if (!sim.identified.has(wi)) {
      if (distToPoly(rover.x, rover.y, w.outline) > IDENTIFY_RANGE && !pointInPoly(rover.x, rover.y, w.outline)) return
      sim.identified.add(wi)
      revealCover(wi, rover, SENSOR_RADIUS)
      const xs = w.outline.map((p) => p[0])
      const ys = w.outline.map((p) => p[1])
      const across = Math.round(Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)))
      const near = cover.filter((c) => c.seen).sort((a, b) => Math.hypot(a.x - rover.x, a.y - rover.y) - Math.hypot(b.x - rover.x, b.y - rover.y))[0]
      const at = near ? [near.x, near.y] : [rover.x, rover.y]
      const what = w.kind === 'river' ? 'a stream' : 'a lake'
      const g = gemmaCall('vision')
      rover.reactedAt = sim.t
      const changed = replanAround(at[0], at[1], 6)
      if (g.ok) {
        later(g.ms, () =>
          addEvent(rover.id, 'discovery', `No LiDAR returns over a patch about ${across} m across, and the camera shows a flat, rippling surface: liquid water, ${what}. Marked impassable for the fleet.`, at, 'gemma'),
        )
      } else {
        addEvent(rover.id, 'discovery', `No LiDAR returns over a patch about ${across} m across, edges all at the same height: treating it as liquid (${what}). Marked impassable.`, at)
      }
      if (changed.some((r) => r !== rover)) addEvent(null, 'info', `${changed.filter((r) => r !== rover).map((r) => r.id).join(' and ')} replanned around the water.`, at)
      return
    }
    revealCover(wi, rover, SENSOR_RADIUS)
  })
}

function revealCover(wi, rover, range) {
  const w = world.water[wi]
  for (const c of world.cover[wi]) {
    if (c.seen || Math.hypot(c.x - rover.x, c.y - rover.y) > range) continue
    c.seen = true
    sim.waterZones.push([c.x, c.y, c.r])
    // The water cells under the disc now count as explored.
    const R = Math.ceil(c.r / RES)
    for (let dj = -R; dj <= R; dj++) {
      for (let di = -R; di <= R; di++) {
        const i = toI(c.x) + di
        const j = toJ(c.y) + dj
        if (!inMap(i, j) || di * di + dj * dj > R * R) continue
        const k = idx(i, j)
        if (world.waterId[k] === wi && !sim.waterSeen[k]) {
          sim.waterSeen[k] = 1
          sim.waterSeenCount++
        }
      }
    }
  }
  void w
}

const waterAt = (x, y) => {
  const i = toI(x)
  const j = toJ(y)
  return inMap(i, j) ? world.waterId[idx(i, j)] : -1
}

function computeFrontiers() {
  const n = world.n
  const bucket = 6
  const buckets = new Map()
  // Unknown but identified water is not unexplored: no frontier along a lake shore.
  const open = (k) => !sim.known[k] && !sim.waterSeen[k]
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = j * n + i
      if (!sim.known[k]) continue
      const edge =
        (i > 0 && open(k - 1)) ||
        (i < n - 1 && open(k + 1)) ||
        (j > 0 && open(k - n)) ||
        (j < n - 1 && open(k + n))
      if (!edge) continue
      const x = world.ox + i * RES
      const y = world.oy + j * RES
      const key = Math.floor((x - world.ox) / bucket) * 1000 + Math.floor((y - world.oy) / bucket)
      let b = buckets.get(key)
      if (!b) buckets.set(key, (b = { sx: 0, sy: 0, c: 0 }))
      b.sx += x
      b.sy += y
      b.c++
    }
  }
  sim.frontiers = [...buckets.values()]
    .filter((b) => b.c >= 4)
    .map((b) => [r1(b.sx / b.c), r1(b.sy / b.c), b.c])
    .sort((a, b) => b[2] - a[2])
    .slice(0, 80)
}

// ---------------------------------------------------------------------------
// Path planning: A* on a 1 m grid over what the fleet knows (unknown is optimistic).

// What the fleet knows it must keep clear of: boulders its LiDAR has seen, soft ground
// found through wheel slip, and rovers parked with a fault. Never the hidden truth.
function knownObstacles(self) {
  const out = []
  for (const rk of sim.rocks) if (rk.seen) out.push([rk.x, rk.y, rk.r + 0.8])
  for (const z of sim.softZones) out.push([z[0], z[1], z[2] + 0.6])
  for (const r of sim.rovers) if (r !== self && r.state === 'fault') out.push([r.x, r.y, 1.7])
  // v4: identified water, and spires the LiDAR has seen (fallen ones as a row of discs).
  for (const z of sim.waterZones) out.push([z[0], z[1], z[2] + 0.3])
  for (const sp of world.spires) {
    if (!sim.spireSeen.has(`${sp.id}:${sp.standing ? 'up' : 'down'}`)) continue
    const [x0, y0, x1, y1] = spireSegment(sp)
    const len = Math.hypot(x1 - x0, y1 - y0)
    const steps = Math.max(1, Math.ceil(len / 0.8))
    for (let q = 0; q <= steps; q++) out.push([lerp(x0, x1, q / steps), lerp(y0, y1, q / steps), sp.radius + 0.7])
  }
  return out
}

function planPath(sx, sy, gx, gy, self = null) {
  // Discs that contain the start are only expensive (so a rover can drive out of one).
  const obstacles = []
  const soft = []
  for (const o of knownObstacles(self)) (Math.hypot(sx - o[0], sy - o[1]) < o[2] ? soft : obstacles).push(o)
  const inDisc = (list, x, y) => {
    for (const o of list) {
      const dx = x - o[0]
      const dy = y - o[1]
      if (dx * dx + dy * dy < o[2] * o[2]) return true
    }
    return false
  }
  const step = 2
  const m = Math.floor((world.n - 1) / step) + 1
  const toNode = (x, y) => [clamp(Math.round(toI(x) / step), 0, m - 1), clamp(Math.round(toJ(y) / step), 0, m - 1)]
  const [si, sj] = toNode(sx, sy)
  const [gi, gj] = toNode(gx, gy)
  const N = m * m
  const g = new Float32Array(N).fill(Infinity)
  const came = new Int32Array(N).fill(-1)
  const closed = new Uint8Array(N)
  const heap = []
  const push = (node, f) => {
    heap.push([f, node])
    let c = heap.length - 1
    while (c > 0) {
      const p = (c - 1) >> 1
      if (heap[p][0] <= heap[c][0]) break
      ;[heap[p], heap[c]] = [heap[c], heap[p]]
      c = p
    }
  }
  const pop = () => {
    const top = heap[0]
    const last = heap.pop()
    if (heap.length) {
      heap[0] = last
      let c = 0
      for (;;) {
        const l = 2 * c + 1
        const r = l + 1
        let s = c
        if (l < heap.length && heap[l][0] < heap[s][0]) s = l
        if (r < heap.length && heap[r][0] < heap[s][0]) s = r
        if (s === c) break
        ;[heap[s], heap[c]] = [heap[c], heap[s]]
        c = s
      }
    }
    return top
  }
  const start = sj * m + si
  const goal = gj * m + gi
  g[start] = 0
  push(start, Math.hypot(gi - si, gj - sj))
  const limit = sim.slopeLimit
  let found = false
  let expanded = 0
  while (heap.length && expanded < 20000) {
    const [, cur] = pop()
    if (closed[cur]) continue
    closed[cur] = 1
    expanded++
    if (cur === goal) {
      found = true
      break
    }
    const ci = cur % m
    const cj = (cur / m) | 0
    for (let dj = -1; dj <= 1; dj++) {
      for (let di = -1; di <= 1; di++) {
        if (!di && !dj) continue
        const ni = ci + di
        const nj = cj + dj
        if (ni < 0 || nj < 0 || ni >= m || nj >= m) continue
        const nb = nj * m + ni
        if (closed[nb]) continue
        const k = idx(ni * step, nj * step)
        let cost = Math.hypot(di, dj)
        const wx = world.ox + ni * step * RES
        const wy = world.oy + nj * step * RES
        if (obstacles.length && nb !== goal && inDisc(obstacles, wx, wy)) continue
        if (soft.length && inDisc(soft, wx, wy)) cost *= 3
        if (sim.known[k]) {
          const s = sim.mslope[k]
          if (s >= limit && nb !== goal) continue
          cost *= 1 + (Math.min(s, limit) / limit) ** 2 * 2.5
        } else {
          cost *= 1.25
        }
        const ng = g[cur] + cost
        if (ng < g[nb]) {
          g[nb] = ng
          came[nb] = cur
          push(nb, ng + Math.hypot(gi - ni, gj - nj))
        }
      }
    }
  }
  if (!found) return null
  const nodes = []
  for (let c = goal; c !== -1; c = came[c]) nodes.push(c)
  nodes.reverse()
  const pts = nodes.map((c) => [world.ox + (c % m) * step * RES, world.oy + ((c / m) | 0) * step * RES])
  // Keep it under the protocol limit of 30 points (the first one is the rover itself).
  const maxPts = 28
  const out = []
  const stride = Math.max(1, Math.ceil(pts.length / maxPts))
  for (let p = stride; p < pts.length - 1; p += stride) out.push(pts[p])
  out.push([gx, gy])
  return out
}

// ---------------------------------------------------------------------------
// Rover behaviour

function setGoal(rover, gx, gy, state, why, source = null) {
  const path = planPath(rover.x, rover.y, gx, gy, rover)
  if (!path) return false
  rover.state = state
  rover.goal = [r1(gx), r1(gy)]
  rover.path = path
  rover.replanT = sim.t + 6
  rover.replannedAt = sim.t
  if (why && sim.t - rover.lastGoalEventT > 25) {
    addEvent(rover.id, 'info', why, null, source)
    rover.lastGoalEventT = sim.t
  }
  return true
}

// Steepest known slope along the rover's current route, in whole degrees.
function routeMaxSlope(rover) {
  let max = 0
  let px = rover.x
  let py = rover.y
  for (const [x, y] of rover.path) {
    const n = Math.max(1, Math.ceil(Math.hypot(x - px, y - py) / RES))
    for (let s = 1; s <= n; s++) {
      const i = toI(px + ((x - px) * s) / n)
      const j = toJ(py + ((y - py) * s) / n)
      if (inMap(i, j) && sim.known[idx(i, j)]) max = Math.max(max, sim.mslope[idx(i, j)])
    }
    px = x
    py = y
  }
  return Math.round(max)
}

function chooseExplorationGoal(rover) {
  // An unclaimed discovered target nearby wins over plain exploration.
  const free = PROTOCOL >= 4 ? TUBES - rover.samples.length : 1
  if (free <= 0 && rover.samples.length) {
    startReturn(rover, `All ${TUBES} tubes full. Heading to the lander to cache the samples.`)
    return true
  }
  // v4b: traces (highest expected value) first, then the closest.
  const open = [...sim.discovered.values()].filter((t) => t.status !== 'sampled' && !t.assigned && Math.hypot(t.x - rover.x, t.y - rover.y) < 30)
  open.sort((a, b) => (b.kind === 'trace') - (a.kind === 'trace') || Math.hypot(a.x - rover.x, a.y - rover.y) - Math.hypot(b.x - rover.x, b.y - rover.y))
  if (PROTOCOL >= 4 && open.length > 1 && rover.battery > 0.45 && free > 0 && sim.t - rover.lastGoalEventT > 20) {
    const [a, b] = open
    const g = gemmaCall('choice')
    if (g.ok) {
      const da = Math.round(Math.hypot(a.x - rover.x, a.y - rover.y))
      const db = Math.round(Math.hypot(b.x - rover.x, b.y - rover.y))
      later(g.ms, () =>
        addEvent(rover.id, 'info', `Second opinion: sample the ${KIND_LABEL[a.kind]} ${a.id} (${da} m) before the ${KIND_LABEL[b.kind]} ${b.id} (${db} m). ${free} free tube${free === 1 ? '' : 's'}, battery ${Math.round(rover.battery * 100)}%.`, null, 'gemma'),
      )
    }
  }
  for (const t of open) {
    if (rover.battery > 0.45 && free > 0) {
      if (setGoal(rover, t.x, t.y, 'investigating', `Heading to the ${KIND_LABEL[t.kind]} found by ${t.found_by === rover.id ? 'this rover' : nameOf(t.found_by)}.`)) {
        t.assigned = rover.id
        rover.targetId = t.id
        rover.lastGoalEventT = sim.t
        return true
      }
    }
  }
  const others = sim.rovers.filter((r) => r !== rover && r.goal)
  const avoid = knownObstacles(rover)
  let best = null
  let bestScore = -Infinity
  for (const f of sim.frontiers) {
    const d = Math.hypot(f[0] - rover.x, f[1] - rover.y)
    if (d < 4) continue
    if (avoid.some((o) => Math.hypot(f[0] - o[0], f[1] - o[1]) < o[2] + 0.5)) continue
    let score = Math.pow(f[2], 0.6) / (d + 6)
    for (const o of others) {
      const od = Math.hypot(o.goal[0] - f[0], o.goal[1] - f[1])
      if (od < 14) score *= 0.25 + (od / 14) * 0.5
    }
    score *= 0.85 + sim.rand() * 0.3
    if (score > bestScore) {
      bestScore = score
      best = f
    }
  }
  if (!best) return false
  rover.targetId = null
  // Gemma picks among the rule-made candidates about half the time it is asked; its
  // reason must quote numbers it was given (cells, distance, battery, slope).
  const eventDue = sim.t - rover.lastGoalEventT > 25
  if (eventDue && sim.rand() < 0.6) {
    const g = gemmaCall('choice')
    if (g.ok && setGoal(rover, best[0], best[1], 'exploring')) {
      const d = Math.round(Math.hypot(best[0] - rover.x, best[1] - rover.y))
      const slope = routeMaxSlope(rover)
      later(g.ms, () =>
        addEvent(
          rover.id,
          'info',
          `Chose the frontier at (${best[0].toFixed(0)}, ${best[1].toFixed(0)}): ${best[2]} unknown cells ${d} m away, max slope ${slope} deg on the way, battery ${Math.round(rover.battery * 100)}%.`,
          null,
          'gemma',
        ),
      )
      rover.lastGoalEventT = sim.t
      return true
    }
  }
  return setGoal(rover, best[0], best[1], 'exploring', `Heading to a frontier at (${best[0].toFixed(0)}, ${best[1].toFixed(0)}).`)
}

function nameOf(id) {
  return sim.rovers.find((r) => r.id === id)?.name ?? id
}

function dockPoint(rover) {
  return [LANDER.x + Math.cos(rover.dockAngle) * (LANDER.radius + 0.9), LANDER.y + Math.sin(rover.dockAngle) * (LANDER.radius + 0.9)]
}

function releaseTarget(rover) {
  rover.sampling = null
  if (!rover.targetId) return
  const t = sim.discovered.get(rover.targetId)
  if (t && t.status !== 'sampled') {
    t.assigned = null
    if (t.status === 'investigating') t.status = 'found'
  }
  rover.targetId = null
}

function startReturn(rover, msg = null, source = null) {
  releaseTarget(rover)
  const [dx, dy] = dockPoint(rover)
  if (!setGoal(rover, dx, dy, 'returning')) {
    rover.state = 'returning'
    rover.goal = [r1(dx), r1(dy)]
    rover.path = [[dx, dy]]
  }
  addEvent(rover.id, 'info', msg || `Battery at ${Math.round(rover.battery * 100)}%. Returning to the lander to charge.`, null, source)
  rover.lastGoalEventT = sim.t
}

function trueSandAt(x, y) {
  return sim.hazards.find((h) => h.type === 'sand' && Math.hypot(x - h.x, y - h.y) < h.radius) || null
}

function drive(rover, dt, maxSpeed) {
  if (dustActive()) maxSpeed *= 0.5 // unsure what is ahead: slow down
  if (trueSandAt(rover.x, rover.y)) maxSpeed = Math.min(maxSpeed, 0.12) // bogged down
  // v3: asked the operator (crawl or hold until answered), answered "approach slowly",
  // or told to hold.
  if (rover.request) maxSpeed = Math.min(maxSpeed, rover.request.hold ? 0 : 0.15)
  if (rover.slowUntil > sim.t) maxSpeed = Math.min(maxSpeed, 0.2)
  if (rover.hold) {
    rover.speed = Math.max(0, rover.speed - dt * 1.5)
    return false
  }
  if (!rover.path.length) {
    rover.speed = Math.max(0, rover.speed - dt * 1.5)
    return true
  }
  let [wx, wy] = rover.path[0]
  let d = Math.hypot(wx - rover.x, wy - rover.y)
  while (d < 0.7 && rover.path.length) {
    rover.path.shift()
    if (!rover.path.length) break
    ;[wx, wy] = rover.path[0]
    d = Math.hypot(wx - rover.x, wy - rover.y)
  }
  if (!rover.path.length) {
    rover.speed = Math.max(0, rover.speed - dt * 1.5)
    return true
  }
  const want = Math.atan2(wy - rover.y, wx - rover.x)
  const diff = wrapAngle(want - rover.yaw)
  rover.yaw = wrapAngle(rover.yaw + clamp(diff, -1.1 * dt, 1.1 * dt))
  const slope = slopeAt(rover.x, rover.y)
  const target = maxSpeed * (1 - 0.45 * clamp(slope / sim.slopeLimit, 0, 1)) * (Math.abs(diff) > 0.9 ? 0.25 : 1)
  rover.speed += clamp(target - rover.speed, -dt * 1.2, dt * 0.6)
  const nx = rover.x + Math.cos(rover.yaw) * rover.speed * dt
  const ny = rover.y + Math.sin(rover.yaw) * rover.speed * dt
  // Bumper: a boulder the LiDAR has not reported yet still stops the rover.
  const rock = sim.rocks.find((rk) => sim.t >= rk.landT && Math.hypot(nx - rk.x, ny - rk.y) < rk.r + 0.45)
  if (rock) {
    rover.speed = 0
    if (!rock.seen) markRocksSeen(rover, [rock])
    return false
  }
  if (PROTOCOL >= 4) {
    // Liquid stops the wheels at the shore: reveal what is in view and go around.
    const wi = waterAt(nx, ny)
    if (wi >= 0) {
      rover.speed = 0
      revealCover(wi, rover, SENSOR_RADIUS)
      bumpReplan(rover)
      return false
    }
    for (const sp of world.spires) {
      if (sp.fallT && sim.t < sp.fallT) continue
      const [x0, y0, x1, y1] = spireSegment(sp)
      if (segDist(nx, ny, x0, y0, x1, y1) < sp.radius + 0.5) {
        rover.speed = 0
        sim.spireSeen.add(`${sp.id}:${sp.standing ? 'up' : 'down'}`)
        bumpReplan(rover)
        return false
      }
    }
  }
  const lim = MAP_SIZE / 2 - 1
  rover.x = clamp(nx, -lim, lim)
  rover.y = clamp(ny, -lim, lim)
  return false
}

// Blocked by something the map did not show: plan again, or give the goal up.
function bumpReplan(rover) {
  if (sim.t - (rover.bumpT ?? -99) < 1.5) return
  rover.bumpT = sim.t
  if (rover.state === 'observing') {
    rover.observeReplanT = 0
    return
  }
  if (rover.goal && setGoal(rover, rover.goal[0], rover.goal[1], rover.state)) return
  releaseTarget(rover)
  Object.assign(rover, { goal: null, path: [], state: 'idle', timer: 0.5 })
}

function faceToward(rover, x, y, dt) {
  const diff = wrapAngle(Math.atan2(y - rover.y, x - rover.x) - rover.yaw)
  rover.yaw = wrapAngle(rover.yaw + clamp(diff, -1.1 * dt, 1.1 * dt))
}

// ---------------------------------------------------------------- v4: observing a lifeform

function startObserve(rover, track, mode) {
  releaseTarget(rover)
  if (rover.trackId && rover.trackId !== track.id) {
    const old = sim.tracks.find((q) => q.id === rover.trackId)
    if (old && old.observer === rover.id) old.observer = null
  }
  Object.assign(rover, { state: 'observing', trackId: track.id, observeMode: mode, observeUntil: sim.t + OBSERVE_S, observeReplanT: 0, path: [], goal: null })
  track.observer = rover.id
}

function endObserve(rover, why) {
  const k = sim.tracks.find((q) => q.id === rover.trackId)
  if (k && k.observer === rover.id) k.observer = null
  const id = rover.trackId
  Object.assign(rover, { trackId: null, observeMode: null, state: 'idle', timer: 1, path: [], goal: null })
  rover.lastGoalEventT = sim.t
  if (why === 'lost') addEvent(rover.id, 'info', `Lost sight of ${id}: no moving returns for ${TRACK_LOST_S} s. Back to exploring.`)
  else if (why === 'battery') addEvent(rover.id, 'info', `Battery at ${Math.round(rover.battery * 100)}%. Stopped observing ${id} to save power.`)
  else if (why === 'done') addEvent(rover.id, 'info', `Observed ${id} for ${OBSERVE_S} s: gait, size and grazing logged. Back to exploring.`)
}

function stepObserve(rover, dt) {
  const k = sim.tracks.find((q) => q.id === rover.trackId)
  const life = k && sim.lifeforms.find((l) => l.id === k.lifeId)
  if (!k || !life || k.status === 'lost') return endObserve(rover, 'lost')
  if (lowBattery(rover, 0.3)) {
    endObserve(rover, 'battery')
    return startReturn(rover)
  }
  if (sim.t > rover.observeUntil && !rover.request) return endObserve(rover, 'done')
  const dx = rover.x - life.x
  const dy = rover.y - life.y
  const d = Math.hypot(dx, dy) || 1
  const follow = rover.observeMode === 'follow' && !rover.request && !rover.hold
  if (follow && (d > OBSERVE_DIST + 1.3 || d < OBSERVE_DIST - 1.6)) {
    const tx = life.x + (dx / d) * OBSERVE_DIST
    const ty = life.y + (dy / d) * OBSERVE_DIST
    if (sim.t > rover.observeReplanT || !rover.path.length) {
      rover.observeReplanT = sim.t + 2.5
      const path = planPath(rover.x, rover.y, tx, ty, rover)
      if (path) {
        rover.path = path
        rover.goal = [r1(tx), r1(ty)]
        rover.replannedAt = sim.t
      }
    }
    drive(rover, dt, 0.45)
  } else {
    rover.path = []
    rover.goal = null
    rover.speed = Math.max(0, rover.speed - dt * 1.5)
    faceToward(rover, life.x, life.y, dt)
  }
}

// ---------------------------------------------------------------- v4b: samples

function scienceScore() {
  let sum = 0
  for (const t of sim.discovered.values()) if (t.cached && Number.isFinite(t.value)) sum += t.value
  return sum
}

function finishSample(rover, t) {
  const instrument = rover.sampling.instrument
  rover.sampling = null
  t.status = 'sampled'
  t.instrument = instrument
  t.carried_by = rover.id
  t.cached = false
  rover.samples.push(t.id)
  const list = ANALYSES[t.kind] || ANALYSES.mineral
  const [value, analysis] = list[Math.floor(sim.rand() * list.length)]
  const g = gemmaCall('vision')
  const tubes = rover.samples.length
  later(g.ok ? g.ms : 250, () => {
    t.value = value
    t.analysis = analysis
    addEvent(rover.id, 'discovery', `Analysed the ${INSTRUMENT_FOR[t.kind] || 'drill'} sample from the ${KIND_LABEL[t.kind]}: ${analysis}. Science value ${value}/10. Tube ${tubes} of ${TUBES}.`, [t.x, t.y], g.ok ? 'gemma' : null)
  })
  rover.targetId = null
  rover.goal = null
  rover.state = 'idle'
  rover.timer = 1
  if (rover.samples.length >= TUBES) startReturn(rover, `All ${TUBES} tubes full. Heading to the lander to cache the samples.`)
}

function deliverSamples(rover) {
  if (!rover.samples.length) return 0
  const n = rover.samples.length
  for (const id of rover.samples) {
    const t = sim.discovered.get(id)
    if (t) {
      t.cached = true
      t.carried_by = null
    }
  }
  rover.samples = []
  const cached = [...sim.discovered.values()].filter((t) => t.cached).length
  addEvent(rover.id, 'discovery', `Cached ${n} sample${n === 1 ? '' : 's'} at the lander (${cached} in the cache). Fleet science score ${scienceScore()}.`, [LANDER.x, LANDER.y])
  return n
}

function stepRover(rover, dt) {
  const moving = rover.speed > 0.05
  rover.battery = clamp(rover.battery - dt * (moving ? 0.0032 : 0.0008), 0, 1)

  // Wheel slip is the only way to feel soft sand: the LiDAR sees flat ground. The
  // wheels dig in over a couple of seconds (slip climbs), then the rover reacts.
  const driving = rover.state === 'exploring' || rover.state === 'investigating' || rover.state === 'returning'
  const inSand = !!trueSandAt(rover.x, rover.y)
  rover.sinkT = inSand && rover.state !== 'fault' && rover.state !== 'charging' ? (rover.sinkT || 0) + dt : 0
  if (driving && inSand) {
    rover.sandT += dt
    if (rover.sandT > 2.4) detectSoftGround(rover)
  } else if (rover.state !== 'recovering') {
    rover.sandT = 0
  }

  // "Wait here" as an answer holds for a minute of sim time at most.
  if (rover.hold && rover.holdUntil !== null && sim.t > rover.holdUntil) {
    rover.hold = false
    rover.holdUntil = null
    addChat(rover.id, 'operator', 'Waited 60 s. Carrying on exploring.', 'info', 'rules')
  }
  // Safety beats orders: a held rover with a nearly flat battery goes to charge.
  if (rover.hold && rover.battery < 0.12 && rover.state !== 'charging' && rover.state !== 'fault') {
    rover.hold = false
    rover.holdUntil = null
    addChat(rover.id, 'operator', `Battery at ${Math.round(rover.battery * 100)}%, so I can't hold any longer. Heading back to charge.`, 'info', 'rules')
    if (rover.state !== 'returning') startReturn(rover)
  }

  switch (rover.state) {
    case 'fault': {
      rover.speed = 0
      rover.path = []
      rover.goal = null
      const h = rover.faultHazard
      if (!h || h.remaining <= 0) {
        rover.faultHazard = null
        rover.state = 'idle'
        rover.timer = 1
        addEvent(rover.id, 'info', 'Reboot complete. Motors back online, resuming exploration.', [rover.x, rover.y])
        rover.lastGoalEventT = sim.t
      }
      break
    }
    case 'idle': {
      rover.speed = 0
      if (rover.hold) break
      rover.timer -= dt
      if (lowBattery(rover, 0.3)) {
        startReturn(rover) // nothing left to explore nearby, so do not idle on a flat battery
        break
      }
      if (rover.timer <= 0 && !chooseExplorationGoal(rover)) rover.timer = 3
      break
    }
    case 'exploring': {
      if (lowBattery(rover, 0.28)) {
        startReturn(rover)
        break
      }
      const arrived = drive(rover, dt, 0.55)
      if (sim.t > rover.replanT && rover.goal) {
        const before = rover.path.length
        const ok = setGoal(rover, rover.goal[0], rover.goal[1], 'exploring')
        if (!ok) chooseExplorationGoal(rover)
        else if (Math.abs(rover.path.length - before) > 4 && sim.rand() < 0.35) addEvent(rover.id, 'info', 'New terrain ahead. Replanned the route around steep ground.')
      }
      if (arrived) {
        rover.goal = null
        if (!chooseExplorationGoal(rover)) {
          rover.state = 'idle'
          rover.timer = 2
        }
      }
      break
    }
    case 'investigating': {
      const t = sim.discovered.get(rover.targetId)
      if (!t) {
        rover.state = 'idle'
        rover.timer = 0.5
        break
      }
      if (lowBattery(rover, 0.22)) {
        startReturn(rover)
        break
      }
      const dist = Math.hypot(t.x - rover.x, t.y - rover.y)
      if (dist > 1.6) {
        if (drive(rover, dt, 0.5) && dist > 2.5) setGoal(rover, t.x, t.y, 'investigating')
        break
      }
      rover.path = []
      rover.speed = Math.max(0, rover.speed - dt * 2)
      if (PROTOCOL >= 4) {
        if (t.status === 'found' || !rover.sampling) {
          t.status = 'investigating'
          const instrument = INSTRUMENT_FOR[t.kind] || 'drill'
          rover.sampling = { target: t.id, instrument, progress: 0, total: SAMPLE_S[instrument], elapsed: 0 }
          const verb = { drill: 'Drilling a core', scoop: 'Scooping ice', sniffer: 'Sampling the gas', swab: 'Swabbing the trace' }[instrument]
          addEvent(rover.id, 'info', `Arrived at the ${KIND_LABEL[t.kind]}. ${verb} (${SAMPLE_S[instrument]} s), tube ${rover.samples.length + 1} of ${TUBES}.`)
        }
        rover.sampling.elapsed += dt
        rover.sampling.progress = clamp(rover.sampling.elapsed / rover.sampling.total, 0, 1)
        if (rover.sampling.progress >= 1) finishSample(rover, t)
        break
      }
      if (t.status === 'found') {
        t.status = 'investigating'
        rover.timer = 12
        addEvent(rover.id, 'info', `Arrived at the ${KIND_LABEL[t.kind]}. Collecting a sample.`)
      }
      rover.timer -= dt
      if (rover.timer <= 0) {
        t.status = 'sampled'
        const sampled = [...sim.discovered.values()].filter((d) => d.status === 'sampled').length
        addEvent(rover.id, 'discovery', `Sample collected from the ${KIND_LABEL[t.kind]}. ${sampled} of ${world.targets.length} targets sampled.`)
        rover.targetId = null
        rover.goal = null
        rover.state = 'idle'
        rover.timer = 1
      }
      break
    }
    case 'returning': {
      const [dx, dy] = dockPoint(rover)
      if (Math.hypot(dx - rover.x, dy - rover.y) < 1.1) {
        rover.path = []
        rover.goal = null
        const delivered = deliverSamples(rover)
        if (delivered && rover.battery >= 0.7) {
          rover.state = 'idle'
          rover.timer = 1.5
          addEvent(rover.id, 'info', `Tubes empty, battery ${Math.round(rover.battery * 100)}%. Back to exploring.`)
          break
        }
        rover.state = 'charging'
        addEvent(rover.id, 'info', 'Docked at the lander. Charging.')
        break
      }
      if (drive(rover, dt, 0.6)) setGoal(rover, dx, dy, 'returning')
      break
    }
    case 'charging': {
      rover.speed = 0
      rover.battery = clamp(rover.battery + dt * 0.03, 0, 1)
      if (rover.battery >= 0.96) {
        addEvent(rover.id, 'info', `Charged to ${Math.round(rover.battery * 100)}%. Back to exploring.`)
        rover.lastGoalEventT = sim.t
        rover.state = 'idle'
        rover.timer = 0.5
      }
      break
    }
    case 'observing': {
      stepObserve(rover, dt)
      break
    }
    case 'recovering': {
      rover.timer -= dt
      rover.speed = 0.25
      rover.x -= Math.cos(rover.yaw) * 0.25 * dt
      rover.y -= Math.sin(rover.yaw) * 0.25 * dt
      if (rover.timer <= 0) {
        rover.speed = 0
        rover.state = 'idle'
        rover.timer = 0.5
        if (rover.recoverWhy === 'sand') addEvent(rover.id, 'info', 'Out of the soft ground. Replanning around the soft zone.')
        else addEvent(rover.id, 'info', `Backed off the slope. Replanning with the ${sim.slopeLimit} deg limit.`)
        rover.recoverWhy = null
        rover.sandT = 0
      }
      break
    }
  }
}

// ---------------------------------------------------------------------------
// Lifeforms (v4): herds of grazers wander, graze, avoid water and keep clear of rovers.

function lifeOk(x, y) {
  const lim = MAP_SIZE / 2 - 4
  if (Math.abs(x) > lim || Math.abs(y) > lim) return false
  if (Math.hypot(x - LANDER.x, y - LANDER.y) < 9) return false
  if (waterAt(x, y) >= 0) return false
  return slopeAt(x, y) < 24
}

function spawnHerd(cx, cy, count) {
  const herd = { id: sim.herds.length + 1, cx, cy, heading: sim.rand() * Math.PI * 2, speed: 0.14 + sim.rand() * 0.1, pause: 0, seenT: -999, confirmT: -999 }
  sim.herds.push(herd)
  for (let k = 0; k < count && sim.lifeforms.length < MAX_LIFEFORMS; k++) {
    const a = (k / count) * Math.PI * 2 + sim.rand()
    const d = 0.8 + sim.rand() * 1.8
    sim.lifeforms.push({
      id: `l${sim.nextLife++}`,
      kind: 'grazer',
      x: cx + Math.cos(a) * d,
      y: cy + Math.sin(a) * d,
      yaw: sim.rand() * Math.PI * 2,
      vx: 0,
      vy: 0,
      size: r2(0.95 + sim.rand() * 0.4),
      herd,
      oa: a,
      od: d,
    })
  }
  return herd
}

function moveHerd(herd, cx, cy) {
  const dx = cx - herd.cx
  const dy = cy - herd.cy
  herd.cx = cx
  herd.cy = cy
  for (const l of sim.lifeforms) {
    if (l.herd !== herd) continue
    l.x += dx
    l.y += dy
  }
}

function stepLifeforms(dt) {
  // A herd being watched mostly grazes (so a watching rover keeps it in view).
  const watched = new Set()
  for (const r of sim.rovers) {
    if (r.state !== 'observing') continue
    const k = sim.tracks.find((q) => q.id === r.trackId)
    const l = k && sim.lifeforms.find((q) => q.id === k.lifeId)
    if (l) watched.add(l.herd)
  }
  for (const h of sim.herds) {
    if (h.pause > 0) {
      h.pause -= dt
      continue
    }
    const calm = watched.has(h)
    if (sim.rand() < dt / (calm ? 12 : 35)) h.pause = 5 + sim.rand() * 8 // stop to graze
    h.heading += gaussian(sim.rand) * 0.35 * dt
    const sp = h.speed * (calm ? 0.35 : 1)
    const nx = h.cx + Math.cos(h.heading) * sp * dt
    const ny = h.cy + Math.sin(h.heading) * sp * dt
    const ahead = [h.cx + Math.cos(h.heading) * 3, h.cy + Math.sin(h.heading) * 3]
    if (!lifeOk(nx, ny) || !lifeOk(ahead[0], ahead[1])) h.heading += Math.PI * (0.4 + sim.rand() * 0.4)
    else {
      h.cx = nx
      h.cy = ny
    }
  }
  for (const l of sim.lifeforms) {
    const h = l.herd
    l.oa += dt * 0.05
    let tx = h.cx + Math.cos(l.oa) * l.od
    let ty = h.cy + Math.sin(l.oa) * l.od
    // Keep clear of rovers (they are big and noisy), and of each other.
    let fx = 0
    let fy = 0
    for (const r of sim.rovers) {
      const d = Math.hypot(l.x - r.x, l.y - r.y)
      if (d < 3.5 && d > 0.01) {
        fx += ((l.x - r.x) / d) * (3.5 - d) * 1.6
        fy += ((l.y - r.y) / d) * (3.5 - d) * 1.6
      }
    }
    for (const o of sim.lifeforms) {
      if (o === l) continue
      const d = Math.hypot(l.x - o.x, l.y - o.y)
      if (d < 1.3 && d > 0.01) {
        fx += ((l.x - o.x) / d) * (1.3 - d)
        fy += ((l.y - o.y) / d) * (1.3 - d)
      }
    }
    let vx = (tx - l.x) * 0.35 + fx
    let vy = (ty - l.y) * 0.35 + fy
    const grazing = h.pause > 0 && Math.hypot(tx - l.x, ty - l.y) < 0.7 && !fx && !fy
    const sp = Math.hypot(vx, vy)
    const max = grazing ? 0 : 0.5
    if (sp > max) {
      vx = sp > 0 ? (vx / sp) * max : 0
      vy = sp > 0 ? (vy / sp) * max : 0
    }
    const k = Math.min(1, dt * 1.5)
    l.vx += (vx - l.vx) * k
    l.vy += (vy - l.vy) * k
    const nx = l.x + l.vx * dt
    const ny = l.y + l.vy * dt
    if (lifeOk(nx, ny) || !lifeOk(l.x, l.y)) {
      l.x = nx
      l.y = ny
    } else {
      l.vx *= -0.3
      l.vy *= -0.3
    }
    if (Math.hypot(l.vx, l.vy) > 0.05) {
      const diff = wrapAngle(Math.atan2(l.vy, l.vx) - l.yaw)
      l.yaw = wrapAngle(l.yaw + clamp(diff, -1.5 * dt, 1.5 * dt))
    }
    void tx
    void ty
  }
}

// Tracks: a rover's LiDAR sees returns move between scans. The camera (Gemma) labels them.
function stepTracks() {
  const range = lidarRange()
  for (const life of sim.lifeforms) {
    let seer = null
    let best = Infinity
    for (const r of sim.rovers) {
      if (r.state === 'charging') continue
      const d = Math.hypot(life.x - r.x, life.y - r.y)
      if (d <= range && d < best) {
        best = d
        seer = r
      }
    }
    if (!seer) continue
    const noise = () => gaussian(sim.rand) * 0.06
    let k = sim.tracks.find((q) => q.lifeId === life.id)
    if (k) {
      Object.assign(k, { x: life.x + noise(), y: life.y + noise(), vx: life.vx, vy: life.vy, last_seen: sim.t, status: 'tracking' })
      continue
    }
    k = { id: `k${sim.nextTrack++}`, lifeId: life.id, x: life.x, y: life.y, vx: life.vx, vy: life.vy, size: r1(life.size * (0.92 + sim.rand() * 0.12)), last_seen: sim.t, observer: null, label: 'moving object', status: 'tracking', confirmed: false }
    sim.tracks.push(k)
    const herd = life.herd
    const dist = Math.round(best)
    const dir = compass(life.x - seer.x, life.y - seer.y)
    if (sim.t - herd.seenT > 25) {
      herd.seenT = sim.t
      addEvent(seer.id, 'warn', `LiDAR returns moving between scans ${dist} m ${dir}: something is moving on its own. Tracking it as ${k.id}.`, [life.x, life.y])
    }
    // The camera takes a look; without Gemma the rules settle on a gait after a while.
    const g = gemmaCall('vision')
    later(g.ok ? g.ms : 8000 / SIM_SPEED, () => {
      k.label = g.ok ? `possible lifeform, six legs, about ${r1(life.size)} m` : `moving object, legged gait, about ${r1(life.size)} m`
      k.confirmed = true
      sim.lifeConfirmed.add(life.id)
      if (sim.t - herd.confirmT > 30) {
        herd.confirmT = sim.t
        const n = sim.lifeforms.filter((l) => l.herd === herd).length
        addEvent(
          seer.id,
          'discovery',
          g.ok
            ? `Camera: possible lifeform, six legs, about ${r1(life.size)} m long, grazing${n > 1 ? `, with ${n - 1} more like it nearby` : ''}. Tracking ${k.id}.`
            : `Moving object with a legged gait, about ${r1(life.size)} m, ${n > 1 ? `${n} of them` : 'alone'}. Tracking ${k.id}.`,
          [life.x, life.y],
          g.ok ? 'gemma' : null,
        )
      }
    })
    // Should this rover stop and watch? Not if it is busy, low, or someone already watches the herd.
    // (Asking is fine for a rover the operator put on hold; following on its own is not.)
    const free = ['exploring', 'idle', 'investigating'].includes(seer.state) && !seer.sampling && !seer.request && seer.battery > 0.4
    const watched = sim.rovers.some((r) => r.state === 'observing' && sim.tracks.some((q) => q.id === r.trackId && sim.lifeforms.find((l) => l.id === q.lifeId)?.herd === herd))
    if (!free || watched || (seer.hold && (sim.firstLifeAsked || PROTOCOL < 3))) continue
    if (!sim.firstLifeAsked && PROTOCOL >= 3) {
      startObserve(seer, k, 'watch')
      sim.firstLifeAsked = askAbout(seer, 'life', { dist, dir, at: [life.x, life.y] })
    } else {
      startObserve(seer, k, 'follow')
      const g2 = gemmaCall('choice')
      const at = [life.x, life.y]
      const say = () =>
        addEvent(seer.id, 'info', `Observing ${k.id} from ${OBSERVE_DIST} m for ${OBSERVE_S} s: it moves about ${r1(Math.max(0.1, Math.hypot(life.vx, life.vy)))} m/s and keeps to its herd. Following without getting closer.`, at, g2.ok ? 'gemma' : null)
      if (g2.ok) later(g2.ms, say)
      else say()
    }
  }
  for (const k of sim.tracks) {
    if (k.status === 'tracking' && sim.t - k.last_seen > TRACK_LOST_S) {
      k.status = 'lost'
      k.observer = null
    }
  }
  sim.tracks = sim.tracks.filter((k) => k.status !== 'lost' || sim.t - k.last_seen < TRACK_DROP_S)
}

// ---------------------------------------------------------------------------
// Hazard reactions (driven only by what the rovers sense)

function pathNear(rover, x, y, r) {
  let px = rover.x
  let py = rover.y
  for (const [qx, qy] of rover.path) {
    const dx = qx - px
    const dy = qy - py
    const len2 = dx * dx + dy * dy || 1e-9
    const t = clamp(((x - px) * dx + (y - py) * dy) / len2, 0, 1)
    if (Math.hypot(px + dx * t - x, py + dy * t - y) < r) return true
    px = qx
    py = qy
  }
  return false
}

// Replan every rover whose route crosses (x, y, r). Returns the rovers that changed course.
function replanAround(x, y, r, except = null) {
  const changed = []
  for (const rover of sim.rovers) {
    if (rover === except || !rover.goal || !rover.path.length) continue
    if (rover.state === 'fault' || rover.state === 'charging' || rover.state === 'recovering') continue
    if (!pathNear(rover, x, y, r)) continue
    if (setGoal(rover, rover.goal[0], rover.goal[1], rover.state)) changed.push(rover)
    else {
      releaseTarget(rover)
      rover.goal = null
      rover.path = []
      rover.state = 'idle'
      rover.timer = 0.5
      changed.push(rover)
    }
  }
  return changed
}

// The LiDAR (or the bumper) found boulders: write them into the shared map as steep
// ground and steer around them.
function markRocksSeen(rover, rocks) {
  const fresh = rocks.filter((rk) => !rk.seen)
  if (!fresh.length) return
  for (const rk of fresh) {
    rk.seen = true
    const R = Math.ceil((rk.r + 0.3) / RES)
    const ci = toI(rk.x)
    const cj = toJ(rk.y)
    for (let dj = -R; dj <= R; dj++) {
      for (let di = -R; di <= R; di++) {
        const i = ci + di
        const j = cj + dj
        if (!inMap(i, j)) continue
        const d = Math.hypot(world.ox + i * RES - rk.x, world.oy + j * RES - rk.y)
        if (d > rk.r + 0.25) continue
        const k = idx(i, j)
        const top = rk.z + Math.sqrt(Math.max(0, rk.r * rk.r - Math.min(d, rk.r) ** 2))
        const h = r2(Math.max(world.heights[k], top))
        if (!sim.known[k]) {
          sim.known[k] = 1
          sim.knownCount++
        }
        sim.measured[k] = h
        sim.mslope[k] = Math.max(sim.mslope[k], 38 + d * 10)
        sim.pendingCells.push([i, j, h, r1(sim.mslope[k])])
      }
    }
  }
  const byHazard = new Map()
  for (const rk of fresh) byHazard.set(rk.hazard, [...(byHazard.get(rk.hazard) || []), rk])
  for (const [hz, list] of byHazard) {
    const cx = list.reduce((s, rk) => s + rk.x, 0) / list.length
    const cy = list.reduce((s, rk) => s + rk.y, 0) / list.length
    const dist = Math.round(Math.hypot(cx - rover.x, cy - rover.y))
    const blocking = rover.path.length && list.some((rk) => pathNear(rover, rk.x, rk.y, rk.r + 0.8))
    const changed = replanAround(cx, cy, (hz?.radius ?? 3) + 1)
    rover.reactedAt = sim.t
    if (!hz.reported) {
      hz.reported = true
      const n = hz.rocks.length
      addEvent(
        rover.id,
        'warn',
        blocking || changed.includes(rover)
          ? `LiDAR: ${n} boulders ${dist} m ahead, blocking the route. Marked them impassable and replanned around them.`
          : `LiDAR: ${n} new boulders ${dist} m away. Marked them impassable on the shared map.`,
        [cx, cy],
      )
      const others = changed.filter((r) => r !== rover)
      if (others.length) addEvent(null, 'info', `${others.map((r) => r.id).join(' and ')} replanned around the boulders using the shared map.`, [cx, cy])
      // v3: new objects it cannot classify: crawl and ask the operator.
      if (hz.type === 'rockfall') {
        const tall = Math.max(...list.map((rk) => rk.z + rk.r - groundSmooth(rk.x, rk.y)))
        askAbout(rover, 'rockfall', { n, tall: r1(tall), dist, at: [cx, cy] })
      }
    }
  }
}

function detectSoftGround(rover) {
  const slip = Math.round((rover.sensors?.slip ?? 0.7) * 100)
  const ex = rover.x + Math.cos(rover.yaw) * 1.0
  const ey = rover.y + Math.sin(rover.yaw) * 1.0
  const near = sim.softZones.find((z) => Math.hypot(z[0] - ex, z[1] - ey) < z[2] + 1.5)
  if (near) {
    near[2] = r1(Math.min(4.5, Math.max(near[2], Math.hypot(near[0] - ex, near[1] - ey) + 1.2)))
  } else {
    sim.softZones.push([r1(ex), r1(ey), 2.2])
  }
  const truth = trueSandAt(rover.x, rover.y)
  if (truth) truth.known = true
  rover.sandT = -999 // one detection per visit
  rover.reactedAt = sim.t
  const changed = replanAround(ex, ey, 3.2, rover)
  // v3: it stops and asks what to do (without an answer it backs out, the safe default).
  // Once per patch and minute: a rover that slips again soon after just backs out.
  const asking = sim.t - (rover.sandAskT ?? -999) > 90 && askAbout(rover, 'sand', { slip, flat: r2(0.04 + sim.rand() * 0.06), at: [ex, ey] })
  if (asking) rover.sandAskT = sim.t
  addEvent(rover.id, 'warn', `Wheel slip ${slip}%, wheels turning but barely moving: soft ground. ${asking ? 'Stopped' : 'Backing out'} and marking a soft zone.`, [ex, ey])
  if (!asking) backOut(rover)
  addEvent(null, 'info', changed.length ? `Soft zone shared with the fleet. ${changed.map((r) => r.id).join(' and ')} replanned to avoid it.` : 'Soft zone shared with the fleet. Every route now avoids it.')
}

function backOut(rover) {
  releaseTarget(rover)
  rover.hold = false
  rover.state = 'recovering'
  rover.recoverWhy = 'sand'
  rover.timer = 3.5
  rover.path = []
  rover.goal = null
}

function fleetEvents() {
  if (sim.t >= sim.nextStallT && sim.slopeLimit > MIN_SLOPE_LIMIT) {
    const candidates = sim.rovers.filter((r) => r.state === 'exploring' && r.speed > 0.1)
    if (candidates.length) {
      const rover = candidates[Math.floor(sim.rand() * candidates.length)]
      const stallSlope = Math.max(MIN_SLOPE_LIMIT + 2, Math.round(sim.slopeLimit - 2 - sim.rand() * 5))
      const newLimit = Math.max(MIN_SLOPE_LIMIT, stallSlope - 2)
      sim.slopeLimit = newLimit
      releaseTarget(rover)
      rover.state = 'recovering'
      rover.timer = 5
      rover.path = []
      rover.goal = null
      addEvent(rover.id, 'warn', `Stalled on a ${stallSlope} deg slope. Lowering fleet slope limit to ${newLimit} deg.`)
      sim.nextStallT = sim.t + 80 + sim.rand() * 60
    } else {
      sim.nextStallT = sim.t + 5
    }
  }
  if (sim.t >= sim.nextSlipT) {
    const movers = sim.rovers.filter((r) => r.speed > 0.1)
    if (movers.length) {
      const rover = movers[Math.floor(sim.rand() * movers.length)]
      rover.slipBoost = { until: sim.t + 2.5, value: 0.34 + sim.rand() * 0.1 }
      const slip = Math.round(rover.slipBoost.value * 100)
      const g = gemmaCall('choice')
      if (g.ok) {
        const load = Math.round((rover.sensors?.motor_load ?? 0.5) * 100)
        const at = [rover.x, rover.y]
        later(g.ms, () => addEvent(rover.id, 'warn', `Loose ground, not soft sand: slip ${slip}% with motor load ${load}%, under the 45% soft-ground mark. Slowing down.`, at, 'gemma'))
      } else {
        addEvent(rover.id, 'warn', `Wheel slip ${slip}% on loose ground. Slowing down.`, [rover.x, rover.y])
      }
      rover.speed *= 0.3
    }
    sim.nextSlipT = sim.t + 60 + sim.rand() * 60
  }
  const pct = ((sim.knownCount + sim.waterSeenCount) / sim.known.length) * 100
  while (sim.milestones.length && pct >= sim.milestones[0]) {
    addEvent(null, 'info', `${sim.milestones.shift()}% of the map explored.`)
  }
}

// ---------------------------------------------------------------------------
// Sensors (v2)

// 360 degree LiDAR: LIDAR_AZ x LIDAR_EL rays from the mast, marched over the true
// heightmap and the landed boulders. Hills and boulders cast shadows. Returns the hit
// count and up to SCAN_MAX points; boulders it hits are reported to the fleet map.
function lidarScan(rover, range) {
  const gz = groundSmooth(rover.x, rover.y)
  const oz = gz + MAST_H
  const near = sim.rocks.filter((rk) => sim.t >= rk.landT && Math.hypot(rk.x - rover.x, rk.y - rover.y) < range + rk.r)
  const spires = []
  if (PROTOCOL >= 4) {
    for (const sp of world.spires) {
      if (sp.fallT && sim.t < sp.fallT) continue
      const seg = spireSegment(sp)
      if (segDist(rover.x, rover.y, ...seg) > range + 0.5) continue
      if (sp.g0 == null) sp.g0 = groundSmooth(sp.x, sp.y)
      spires.push([sp, seg])
    }
  }
  const seen = new Set()
  const pts = []
  const step = 0.2
  const top = world.maxH + 1
  for (let a = 0; a < LIDAR_AZ; a++) {
    const az = rover.yaw + (a / LIDAR_AZ) * Math.PI * 2
    const ca = Math.cos(az)
    const sa = Math.sin(az)
    for (const el of LIDAR_EL) {
      const ce = Math.cos(el)
      const se = Math.sin(el)
      for (let d = step; d <= range; d += step) {
        const x = rover.x + ca * ce * d
        const y = rover.y + sa * ce * d
        const z = oz + se * d
        if (z > top) break
        let rock = null
        for (const rk of near) {
          const dx = x - rk.x
          const dy = y - rk.y
          const dz = z - rk.z
          if (dx * dx + dy * dy + dz * dz < rk.r * rk.r) {
            rock = rk
            break
          }
        }
        if (rock) {
          seen.add(rock)
          pts.push(x, y, z)
          break
        }
        let spire = false
        for (const [sp, seg] of spires) {
          if (sp.standing) {
            const hz = z - sp.g0
            if (hz < 0 || hz > sp.height) continue
            const rr = sp.radius * (1 - (hz / sp.height) * 0.85)
            if ((x - sp.x) ** 2 + (y - sp.y) ** 2 < rr * rr) spire = true
          } else if (segDist(x, y, ...seg) < sp.radius && z < groundSmooth(x, y) + sp.radius * 2) spire = true
          if (spire) break
        }
        if (spire) {
          pts.push(x, y, z)
          break
        }
        const g = groundSmooth(x, y)
        if (PROTOCOL >= 4) {
          const wi = waterAt(x, y)
          if (wi >= 0 && z <= world.water[wi].level) break // liquid: the beam never comes back
        }
        if (z <= g) {
          pts.push(x, y, g)
          break
        }
      }
    }
  }
  const hits = pts.length / 3
  const stride = Math.max(1, Math.ceil(hits / SCAN_MAX))
  const scan = []
  for (let p = 0; p < hits; p += stride) scan.push(r1(pts[p * 3]), r1(pts[p * 3 + 1]), r1(pts[p * 3 + 2]))
  return { hits, scan, seen: [...seen] }
}

function readSensors(rover, range, hits) {
  // IMU from the terrain gradient along and across the heading.
  const d = 0.6
  const hx = (groundSmooth(rover.x + d, rover.y) - groundSmooth(rover.x - d, rover.y)) / (2 * d)
  const hy = (groundSmooth(rover.x, rover.y + d) - groundSmooth(rover.x, rover.y - d)) / (2 * d)
  const c = Math.cos(rover.yaw)
  const s = Math.sin(rover.yaw)
  const deg = (v) => (Math.atan(v) * 180) / Math.PI
  const shake = rover.speed > 0.05 ? 0.5 : 0.08
  const pitch = deg(hx * c + hy * s) + gaussian(sim.rand) * shake
  const roll = deg(-(hx * s - hy * c)) + gaussian(sim.rand) * shake

  // Wheels: ground speed is what the rover really does; slip is the difference.
  const slope = slopeAt(rover.x, rover.y)
  let slip = 0.03 + 0.12 * clamp(slope / sim.slopeLimit, 0, 1) + Math.abs(gaussian(sim.rand)) * 0.015
  const inSand = trueSandAt(rover.x, rover.y)
  const powered = rover.state !== 'fault' && rover.state !== 'charging' && rover.state !== 'idle'
  if (inSand && powered) slip = 0.3 + 0.46 * clamp((rover.sinkT || 0) / 1.8, 0, 1) + sim.rand() * 0.05
  if (rover.slipBoost && sim.t < rover.slipBoost.until) slip = Math.max(slip, rover.slipBoost.value + gaussian(sim.rand) * 0.03)
  slip = clamp(slip, 0, 0.95)
  const ground = Math.abs(rover.speed)
  let wheel = powered ? ground / Math.max(0.05, 1 - slip) : 0
  if (inSand && powered) wheel = Math.max(wheel, 0.42 + sim.rand() * 0.06)
  if (!powered) slip = 0
  const load = powered
    ? clamp(0.16 + 0.55 * clamp(slope / 40, 0, 1) + 0.5 * slip + 0.25 * ground + gaussian(sim.rand) * 0.02, 0.03, 1)
    : 0

  let spectro = null
  for (const t of sim.discovered.values()) {
    const dist = Math.hypot(t.x - rover.x, t.y - rover.y)
    if (dist <= SPECTRO_RANGE && (!spectro || dist < spectro.dist)) spectro = { id: t.id, dist: r1(dist) }
  }
  return {
    lidar_range: range,
    lidar_hits: hits,
    pitch: r1(pitch),
    roll: r1(roll),
    wheel_speed: r2(wheel),
    ground_speed: r2(ground),
    slip: r2(slip),
    motor_load: r2(load),
    spectro,
  }
}

// Decision-making compute per 200 ms cycle: steady perception and mapping work, with
// spikes when A* replans or a hazard forces a reaction.
function brainCycles(rover, dtTotal) {
  const n = Math.max(1, Math.round((dtTotal * 1000) / CYCLE_MS))
  for (let c = 0; c < n; c++) {
    const t = sim.t - ((n - 1 - c) * CYCLE_MS) / 1000
    let ms = 2.6 + sim.rand() * 1.3 + rover.hits * 0.0016
    if (rover.state === 'fault') ms = 1.2 + sim.rand() * 0.5
    if (c === n - 1 && sim.t - rover.replannedAt < dtTotal) ms += 7 + sim.rand() * 8
    if (c === n - 1 && sim.t - rover.reactedAt < dtTotal) ms += 9 + sim.rand() * 6
    rover.cycles.push([t, ms])
  }
  while (rover.cycles.length && rover.cycles[0][0] < sim.t - BRAIN_WINDOW_S) rover.cycles.shift()
}

function brainSummary(rover) {
  const list = rover.cycles
  if (!list.length) return null
  let sum = 0
  let peak = 0
  let over = 0
  for (const [, ms] of list) {
    sum += ms
    peak = Math.max(peak, ms)
    if (ms > BUDGET_MS) over++
  }
  return { last_ms: r1(list[list.length - 1][1]), avg_ms: r1(sum / list.length), peak_ms: r1(peak), over }
}

function updateSensors(dtTotal) {
  const range = lidarRange()
  for (const rover of sim.rovers) {
    const { hits, scan, seen } = lidarScan(rover, range)
    rover.hits = hits
    rover.scan = scan
    if (seen.some((rk) => !rk.seen)) markRocksSeen(rover, seen)
    rover.sensors = readSensors(rover, range, hits)
    brainCycles(rover, dtTotal)
  }
}

// ---------------------------------------------------------------------------
// Hazards (v2): POST /api/hazard

function publicHazard(h) {
  const out = { id: h.id, type: h.type, t: r1(h.t), rover: h.rover, active: h.active }
  if (h.type === 'rockfall') Object.assign(out, { x: r1(h.x), y: r1(h.y), radius: h.radius, rocks: h.rocks.map((rk) => [r1(rk.x), r1(rk.y), r2(rk.z), r2(rk.r)]) })
  if (h.type === 'sand') Object.assign(out, { x: r1(h.x), y: r1(h.y), radius: h.radius, known: !!h.known })
  if (h.type === 'dust_storm' || h.type === 'rover_fault') out.remaining = r1(Math.max(0, h.remaining))
  if (h.type === 'rover_fault') {
    const r = sim.rovers.find((q) => q.id === h.rover)
    if (r) Object.assign(out, { x: r1(r.x), y: r1(r.y) })
  }
  if (h.type === 'spire_fall') Object.assign(out, { spire: h.spire, x0: h.x0, y0: h.y0, x1: h.x1, y1: h.y1 })
  if (h.type === 'lifeform') Object.assign(out, { x: h.x, y: h.y })
  return out
}

class HazardError extends Error {}

function remainingPath(rover) {
  let len = 0
  let px = rover.x
  let py = rover.y
  for (const [x, y] of rover.path) {
    len += Math.hypot(x - px, y - py)
    px = x
    py = y
  }
  return len
}

// The point `dist` metres along the rover's route (or straight ahead without one).
function pointAhead(rover, dist) {
  let px = rover.x
  let py = rover.y
  let left = dist
  for (const [x, y] of rover.path) {
    const seg = Math.hypot(x - px, y - py)
    if (seg >= left) return [px + ((x - px) * left) / seg, py + ((y - py) * left) / seg]
    left -= seg
    px = x
    py = y
  }
  if (rover.path.length && dist - left > 2.5) return [px, py]
  return [rover.x + Math.cos(rover.yaw) * dist, rover.y + Math.sin(rover.yaw) * dist]
}

function validSpot(x, y, r) {
  const lim = MAP_SIZE / 2 - r - 1
  if (Math.abs(x) > lim || Math.abs(y) > lim) return false
  if (Math.hypot(x - LANDER.x, y - LANDER.y) < LANDER.radius + r + 0.8) return false
  if (PROTOCOL >= 4 && world.water.some((w) => pointInPoly(x, y, w.outline) || distToPoly(x, y, w.outline) < r)) return false
  return !sim.hazards.some((h) => (h.type === 'rockfall' || h.type === 'sand') && Math.hypot(h.x - x, h.y - y) < h.radius + r)
}

function pickRover(requested) {
  const r = sim.rovers.find((q) => q.id === String(requested))
  if (!r) throw new HazardError(`No rover called ${requested}.`)
  return r
}

// Rovers to try for a hazard on a route, best first: long routes ahead, then the rest.
function routeCandidates(needRoute) {
  const ok = sim.rovers.filter((r) => r.state !== 'fault' && r.state !== 'charging')
  const drivers = ok.filter((r) => r.path.length && remainingPath(r) > (needRoute ? 4.4 : 6)).sort((a, b) => remainingPath(b) - remainingPath(a))
  if (drivers.length > 1 && sim.rand() < 0.4) [drivers[0], drivers[1]] = [drivers[1], drivers[0]]
  if (needRoute) return drivers
  return [...drivers, ...ok.filter((r) => !drivers.includes(r))]
}

// v4: where the user clicked. Throws the 400 answers from the protocol.
function checkPoint(type, [x, y]) {
  const half = MAP_SIZE / 2
  const at = `(${r1(x)}, ${r1(y)})`
  if (Math.abs(x) > half || Math.abs(y) > half) throw new HazardError(`${at} is off the map: it spans ${-half} to ${half} m on both axes.`)
  if (Math.hypot(x - LANDER.x, y - LANDER.y) < LANDER.radius + 0.5) throw new HazardError(`${at} is on the lander. Pick a spot at least ${LANDER.radius + 0.5} m from its centre.`)
  for (const r of sim.rovers) {
    const d = Math.hypot(r.x - x, r.y - y)
    if (d < 2) throw new HazardError(`${at} is ${r1(d)} m from ${r.name}. Pick a spot at least 2 m from any rover.`)
  }
  const wi = waterAt(x, y)
  if (wi >= 0 && type !== 'spire_fall') throw new HazardError(`${at} is in the ${world.water[wi].kind === 'river' ? 'stream' : 'lake'}. Pick dry ground.`)
  const slope = Math.round(slopeAt(x, y))
  if (type === 'sand' && slope > 20) throw new HazardError(`The ground at ${at} slopes ${slope} deg: too steep for soft sand to collect. Pick flatter ground (under 20 deg).`)
  if (type === 'lifeform' && slope > 24) throw new HazardError(`The ground at ${at} slopes ${slope} deg: too steep for the creatures to graze there.`)
  if (type === 'rockfall' || type === 'sand') {
    const r = type === 'rockfall' ? 2.6 : 2.2
    const clash = sim.hazards.find((h) => (h.type === 'rockfall' || h.type === 'sand') && Math.hypot(h.x - x, h.y - y) < h.radius + r)
    if (clash) throw new HazardError(`${at} overlaps the ${clash.type === 'rockfall' ? 'rockfall' : 'soft sand'} already there. Pick a spot a few metres away.`)
  }
  return [x, y]
}

// The rover whose route passes closest to (x, y), within `within` metres: [rover, point on route].
function routeNear(x, y, within, prefer = null) {
  let best = null
  for (const r of sim.rovers) {
    if (r.state === 'fault' || !r.path.length || (prefer && r !== prefer)) continue
    let px = r.x
    let py = r.y
    for (const [qx, qy] of r.path) {
      const dx = qx - px
      const dy = qy - py
      const len2 = dx * dx + dy * dy || 1e-9
      const t = clamp(((x - px) * dx + (y - py) * dy) / len2, 0, 1)
      const cx = px + dx * t
      const cy = py + dy * t
      const d = Math.hypot(cx - x, cy - y)
      if (d < within && (!best || d < best[2]) && Math.hypot(cx - r.x, cy - r.y) > 3) best = [r, [cx, cy], d]
      px = qx
      py = qy
    }
  }
  return best
}

function makeRocks(h, spot, radius) {
  h.rocks = []
  const count = 5 + Math.floor(sim.rand() * 3)
  const lim = MAP_SIZE / 2 - 0.8
  for (let k = 0; k < count && h.rocks.length < count; k++) {
    for (let tries = 0; tries < 20; tries++) {
      const a = sim.rand() * Math.PI * 2
      const d = k === 0 ? sim.rand() * 0.4 : 0.6 + sim.rand() * (radius - 0.9)
      const r = 0.36 + sim.rand() * 0.42 + (k === 0 ? 0.2 : 0)
      const x = spot[0] + Math.cos(a) * d
      const y = spot[1] + Math.sin(a) * d
      if (Math.abs(x) > lim || Math.abs(y) > lim) continue
      if (h.rocks.some((q) => Math.hypot(q.x - x, q.y - y) < q.r + r - 0.1)) continue
      if (sim.rovers.some((q) => Math.hypot(q.x - x, q.y - y) < r + 0.9)) continue
      if (PROTOCOL >= 4 && waterAt(x, y) >= 0) continue
      h.rocks.push({ x, y, z: groundSmooth(x, y) + r * 0.55, r, landT: sim.t + 0.6 + sim.rand() * 0.4, seen: false, hazard: h })
      break
    }
  }
  sim.rocks.push(...h.rocks)
}

// Rockfall or soft sand where the user clicked (snapped onto a rover route nearby).
function placeAt(base, type, [x, y], requested) {
  const radius = type === 'rockfall' ? 2.6 : r1(2.0 + sim.rand() * 0.4)
  const prefer = requested != null ? pickRover(requested) : null
  let spot = [x, y]
  let rover = null
  const near = routeNear(x, y, type === 'rockfall' ? 4 : radius + 1, prefer)
  if (near) {
    rover = near[0]
    // Boulders go across the route itself, if that spot is free.
    if (type === 'rockfall' && sim.rovers.every((r) => Math.hypot(r.x - near[1][0], r.y - near[1][1]) > radius + 0.8) && waterAt(near[1][0], near[1][1]) < 0) spot = near[1]
  }
  const h = { ...base, rover: rover?.id ?? prefer?.id ?? null, x: spot[0], y: spot[1], radius }
  if (type === 'rockfall') {
    makeRocks(h, spot, radius)
    if (!h.rocks.length) throw new HazardError(`No room for boulders at (${r1(x)}, ${r1(y)}). Pick a more open spot.`)
    addEvent(null, 'hazard', rover ? `Rockfall on ${rover.id}'s route.` : `Rockfall at (${r1(spot[0])}, ${r1(spot[1])}).`, spot)
  } else {
    h.known = false
    addEvent(null, 'hazard', rover ? `Soft sand on ${rover.id}'s route. Its LiDAR cannot see it.` : `Soft sand at (${r1(spot[0])}, ${r1(spot[1])}). LiDAR cannot see it.`, spot)
  }
  sim.hazards.push(h)
  sim.nextHazard++
  return h
}

// v4: a spire topples (toward the clicked point, or across a rover's route).
function createSpireFall(base, requested, at) {
  const prefer = requested != null ? pickRover(requested) : null
  if (prefer && prefer.state === 'fault' && !at) throw new HazardError(`${prefer.name} is parked with a motor fault. Pick another rover.`)
  const okFall = (x0, y0, x1, y1, r) => {
    const lim = MAP_SIZE / 2 - 0.5
    if (Math.abs(x1) > lim || Math.abs(y1) > lim) return 'The spire would fall off the map.'
    if (segDist(LANDER.x, LANDER.y, x0, y0, x1, y1) < LANDER.radius + 0.4) return 'The spire would fall onto the lander.'
    const hit = sim.rovers.find((q) => segDist(q.x, q.y, x0, y0, x1, y1) < r + 1.1)
    if (hit) return `The spire would fall onto ${hit.name}. Pick another spot.`
    return null
  }
  let sp = null
  let dir = null
  let rover = prefer
  if (at) {
    const [x, y] = at
    sp = world.spires.filter((q) => q.standing && Math.hypot(q.x - x, q.y - y) <= 6).sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y))[0] || null
    if (sp) {
      const d = Math.hypot(x - sp.x, y - sp.y)
      const a = d > 0.3 ? Math.atan2(y - sp.y, x - sp.x) : sim.rand() * Math.PI * 2
      dir = [Math.cos(a), Math.sin(a)]
    } else {
      if (waterAt(x, y) >= 0) throw new HazardError(`(${r1(x)}, ${r1(y)}) is in the water and there is no spire within 6 m to topple. Pick dry ground.`)
      sp = { id: `s${sim.nextSpire}`, x: r1(x), y: r1(y), height: r1(2.6 + sim.rand() * 1.4), radius: r2(0.3 + sim.rand() * 0.12), standing: true, fresh: true }
      // Fall across the nearest rover's route if there is one in reach, else any free way.
      const near = routeNear(x, y, sp.height + 1, prefer)
      const tries = near ? [Math.atan2(near[1][1] - y, near[1][0] - x)] : []
      for (let k = 0; k < 12; k++) tries.push((k / 12) * Math.PI * 2)
      for (const a of tries) {
        const tx = x + Math.cos(a) * sp.height
        const ty = y + Math.sin(a) * sp.height
        if (!okFall(x, y, tx, ty, sp.radius)) {
          dir = [Math.cos(a), Math.sin(a)]
          break
        }
      }
      if (!dir) throw new HazardError(`No room for a spire to fall at (${r1(x)}, ${r1(y)}). Pick a more open spot.`)
      if (near) rover = near[0]
    }
  } else {
    const cands = prefer ? [prefer] : routeCandidates(true)
    for (const cand of cands) {
      for (const d of [5.5, 7, 8.5, 4.5]) {
        if (d > remainingPath(cand) - 0.5) continue
        const P = pointAhead(cand, d)
        const Q = pointAhead(cand, d + 0.5)
        const along = Math.atan2(Q[1] - P[1], Q[0] - P[0])
        const stand = world.spires.find((q) => q.standing && Math.hypot(q.x - P[0], q.y - P[1]) < q.height - 0.6 && Math.hypot(q.x - P[0], q.y - P[1]) > 1.5)
        if (stand) {
          const a = Math.atan2(P[1] - stand.y, P[0] - stand.x)
          if (!okFall(stand.x, stand.y, stand.x + Math.cos(a) * stand.height, stand.y + Math.sin(a) * stand.height, stand.radius)) {
            sp = stand
            dir = [Math.cos(a), Math.sin(a)]
          }
        } else {
          const h = r1(2.8 + sim.rand() * 1.2)
          const side = sim.rand() < 0.5 ? 1 : -1
          const bx = P[0] + Math.cos(along + (side * Math.PI) / 2) * h * 0.55
          const by = P[1] + Math.sin(along + (side * Math.PI) / 2) * h * 0.55
          const a = along - (side * Math.PI) / 2
          if (lifeOk(bx, by) && !okFall(bx, by, bx + Math.cos(a) * h, by + Math.sin(a) * h, 0.36)) {
            sp = { id: `s${sim.nextSpire}`, x: r1(bx), y: r1(by), height: h, radius: r2(0.3 + sim.rand() * 0.12), standing: true, fresh: true }
            dir = [Math.cos(a), Math.sin(a)]
          }
        }
        if (sp) {
          rover = cand
          break
        }
      }
      if (sp) break
    }
    if (!sp) throw new HazardError(prefer ? `No spot near ${prefer.name}'s route for a spire to fall across. Try again in a few seconds.` : 'No rover is driving a long enough route for a spire to fall across. Try again in a few seconds.')
  }
  const x1 = r1(sp.x + dir[0] * sp.height)
  const y1 = r1(sp.y + dir[1] * sp.height)
  const bad = okFall(sp.x, sp.y, x1, y1, sp.radius)
  if (bad) throw new HazardError(bad)
  if (sp.fresh) {
    delete sp.fresh
    world.spires.push(sp)
    sim.nextSpire++
  }
  Object.assign(sp, { standing: false, x0: sp.x, y0: sp.y, x1, y1, fallT: sim.t + 1.5 })
  sim.featuresVersion++ // like the real sim: the changed spires go out in the next tick
  if (!rover) rover = sim.rovers.filter((r) => r.path.length && pathNear(r, (sp.x + x1) / 2, (sp.y + y1) / 2, sp.height / 2 + 1))[0] || null
  const h = { ...base, type: 'spire_fall', rover: rover?.id ?? null, spire: sp.id, x0: sp.x, y0: sp.y, x1, y1 }
  sim.hazards.push(h)
  sim.nextHazard++
  addEvent(null, 'hazard', rover ? `Spire ${sp.id} toppled across ${rover.id}'s route.` : `Spire ${sp.id} toppled.`, [(sp.x + x1) / 2, (sp.y + y1) / 2])
  return h
}

// v4: a small herd wanders into view of a rover (a demo convenience: the rovers still
// have to notice it themselves).
function createLifeformHazard(base, requested, at) {
  let rover = requested != null ? pickRover(requested) : null
  if (!rover) {
    if (at) rover = [...sim.rovers].sort((a, b) => Math.hypot(a.x - at[0], a.y - at[1]) - Math.hypot(b.x - at[0], b.y - at[1]))[0]
    else rover = routeCandidates(false)[0] || sim.rovers[0]
  }
  let spot = at
  if (!spot) {
    for (const d of [10, 11, 9, 12, 8.5, 13]) {
      for (const side of [0, 3, -3, 6, -6]) {
        const P = pointAhead(rover, d)
        const a = rover.yaw + Math.PI / 2
        const x = P[0] + Math.cos(a) * side
        const y = P[1] + Math.sin(a) * side
        if (lifeOk(x, y) && sim.rovers.every((r) => Math.hypot(r.x - x, r.y - y) > 4)) {
          spot = [x, y]
          break
        }
      }
      if (spot) break
    }
    if (!spot) throw new HazardError(`No room for a herd near ${rover.name} right now (water, steep ground or the map edge). Try another rover.`)
  }
  let herd
  if (sim.herds.length < 6 && sim.lifeforms.length + 4 <= MAX_LIFEFORMS) herd = spawnHerd(spot[0], spot[1], 3 + Math.floor(sim.rand() * 2))
  else {
    // Move the herd nobody is watching that is farthest from every rover.
    const watched = new Set(sim.tracks.filter((k) => k.observer).map((k) => sim.lifeforms.find((l) => l.id === k.lifeId)?.herd))
    herd = [...sim.herds].filter((h) => !watched.has(h)).sort((a, b) => Math.min(...sim.rovers.map((r) => Math.hypot(r.x - b.cx, r.y - b.cy))) - Math.min(...sim.rovers.map((r) => Math.hypot(r.x - a.cx, r.y - a.cy))))[0] || sim.herds[0]
    moveHerd(herd, spot[0], spot[1])
  }
  herd.heading = Math.atan2(rover.y - spot[1], rover.x - spot[0]) + gaussian(sim.rand) * 0.4
  herd.pause = 0
  const n = sim.lifeforms.filter((l) => l.herd === herd).length
  const h = { ...base, type: 'lifeform', rover: rover.id, x: r1(spot[0]), y: r1(spot[1]), endT: sim.t + 40 }
  sim.hazards.push(h)
  sim.nextHazard++
  addEvent(null, 'hazard', `A herd of ${n} creatures wandered in ${Math.round(Math.hypot(rover.x - spot[0], rover.y - spot[1]))} m ${compass(spot[0] - rover.x, spot[1] - rover.y)} of ${rover.id}.`, spot)
  return h
}

function createHazard(type, requested, point = null) {
  if (!HAZARD_TYPES.includes(type)) throw new HazardError(`Unknown hazard type "${type}". Use ${HAZARD_TYPES.slice(0, -1).join(', ')} or ${HAZARD_TYPES[HAZARD_TYPES.length - 1]}.`)
  const base = { id: `h${sim.nextHazard}`, type, t: sim.t, rover: null, active: true }

  if (type === 'dust_storm') {
    const live = sim.hazards.find((h) => h.type === 'dust_storm' && h.active)
    if (live) throw new HazardError(`A dust storm is already blowing (${Math.ceil(live.remaining)} s left).`)
    const h = { ...base, remaining: DUST_S }
    sim.hazards.push(h)
    sim.nextHazard++
    addEvent(null, 'hazard', 'Dust storm rolling in. Visibility dropping fast.')
    for (const r of sim.rovers) r.reactedAt = sim.t
    const g = gemmaCall('vision')
    if (g.ok) {
      const before = Math.round(sim.rovers.reduce((s, r) => s + r.hits, 0) / sim.rovers.length)
      later(g.ms, () =>
        addEvent(
          null,
          'warn',
          `Camera and LiDAR agree it is dust, not an obstacle: returns down from ${before} to about ${Math.round(before * 0.16)} per sweep, range ${DUST_LIDAR_RANGE} m. Halving speed and mapping only what is close.`,
          null,
          'gemma',
        ),
      )
    } else {
      addEvent(null, 'warn', `LiDAR returns collapsing on every rover: range down to ${DUST_LIDAR_RANGE} m. Halving speed and mapping only what is close.`)
    }
    return h
  }

  if (type === 'rover_fault') {
    const rover = requested != null ? pickRover(requested) : null
    if (rover && rover.state === 'fault') throw new HazardError(`${rover.name} already has a motor fault (${Math.ceil(rover.faultHazard?.remaining ?? 0)} s left).`)
    const pool = sim.rovers.filter((r) => r.state !== 'fault')
    if (!pool.length) throw new HazardError('Every rover already has a motor fault.')
    const moving = pool.filter((r) => r.speed > 0.1 && r.state !== 'charging')
    const victim = rover || (moving.length ? moving : pool)[Math.floor(sim.rand() * (moving.length || pool.length))]
    const h = { ...base, rover: victim.id, remaining: FAULT_S }
    sim.hazards.push(h)
    sim.nextHazard++
    const hadTarget = victim.targetId ? sim.discovered.get(victim.targetId) : null
    releaseTarget(victim)
    victim.state = 'fault'
    victim.faultHazard = h
    victim.speed = 0
    victim.path = []
    victim.goal = null
    addEvent(null, 'hazard', `Motor fault on ${victim.id}.`, [victim.x, victim.y])
    addEvent(victim.id, 'warn', `Motors not responding. Parked, rebooting (${FAULT_S} s). Fleet told to treat it as an obstacle.`, [victim.x, victim.y])
    const changed = replanAround(victim.x, victim.y, 2.2, victim)
    if (changed.length) addEvent(null, 'info', `${changed.map((r) => r.id).join(' and ')} replanned around the parked ${victim.id}.`, [victim.x, victim.y])
    if (hadTarget && hadTarget.status !== 'sampled') {
      const helper = sim.rovers
        .filter((r) => r !== victim && r.state !== 'fault' && r.state !== 'charging' && r.battery > 0.4 && !r.targetId)
        .sort((a, b) => Math.hypot(a.x - hadTarget.x, a.y - hadTarget.y) - Math.hypot(b.x - hadTarget.x, b.y - hadTarget.y))[0]
      if (helper && setGoal(helper, hadTarget.x, hadTarget.y, 'investigating')) {
        hadTarget.assigned = helper.id
        helper.targetId = hadTarget.id
        helper.reactedAt = sim.t
        addEvent(helper.id, 'info', `Taking over the ${KIND_LABEL[hadTarget.kind]} that ${victim.id} was heading to.`, [hadTarget.x, hadTarget.y])
      }
    }
    return h
  }

  // v4: point and click (older sims ignore x and y). Dust and faults never use the point.
  const at = point && PROTOCOL >= 4 ? checkPoint(type, point) : null
  if (type === 'spire_fall') return createSpireFall(base, requested, at)
  if (type === 'lifeform') return createLifeformHazard(base, requested, at)
  if (at) return placeAt(base, type, at, requested)

  // rockfall and sand go on a rover's route
  let candidates
  if (requested != null) {
    const r = pickRover(requested)
    if (r.state === 'fault') throw new HazardError(`${r.name} is parked with a motor fault. Pick another rover.`)
    if (type === 'sand' && (!r.path.length || remainingPath(r) < 4)) throw new HazardError(`${r.name} is not driving a route right now, so soft sand would never be found. Try another rover.`)
    candidates = [r]
  } else {
    candidates = routeCandidates(type === 'sand')
    if (!candidates.length) throw new HazardError(type === 'sand' ? 'No rover is driving a long enough route right now. Try again in a few seconds.' : 'Every rover has a motor fault.')
  }
  const radius = type === 'rockfall' ? 2.6 : r1(2.0 + sim.rand() * 0.5)
  const dists = type === 'rockfall' ? [6.5, 7.5, 5.5, 8.5, 9.5, 4.8] : [4.2, 5, 3.6, 6, 7, 3]
  let spot = null
  let rover = candidates[0]
  for (const cand of candidates) {
    for (const d of dists) {
      if (type === 'sand' && d > remainingPath(cand) - 0.8) continue
      const [x, y] = pointAhead(cand, d)
      if (validSpot(x, y, radius) && sim.rovers.every((r) => Math.hypot(r.x - x, r.y - y) > radius + 1.2)) {
        spot = [x, y]
        break
      }
    }
    if (spot) {
      rover = cand
      break
    }
  }
  if (!spot) {
    const who = candidates.length === 1 ? `${rover.name}'s route` : 'any rover route'
    throw new HazardError(`No valid spot on ${who} (too close to the lander, the map edge or another hazard). Try again in a few seconds.`)
  }
  const h = { ...base, rover: rover.id, x: spot[0], y: spot[1], radius }
  if (type === 'rockfall') {
    makeRocks(h, spot, radius)
    addEvent(null, 'hazard', `Rockfall on ${rover.id}'s route.`, spot)
  } else {
    h.known = false
    addEvent(null, 'hazard', `Soft sand on ${rover.id}'s route. Its LiDAR cannot see it.`, spot)
  }
  sim.hazards.push(h)
  sim.nextHazard++
  return h
}

function stepHazards(dt) {
  for (const h of sim.hazards) {
    if (h.type === 'lifeform' && h.active && sim.t > h.endT) {
      h.active = false
      h.endedAt = sim.t
    }
    if (!h.active || h.remaining == null) continue
    h.remaining -= dt
    if (h.remaining > 0) continue
    h.remaining = 0
    h.active = false
    h.endedAt = sim.t
    if (h.type === 'dust_storm') addEvent(null, 'info', `Dust storm over. LiDAR back to ${LIDAR_RANGE} m, full speed again.`)
  }
  // Finished storms and faults stay listed (inactive) for a little while, then go.
  sim.hazards = sim.hazards.filter((h) => h.active || sim.t - h.endedAt < 10)
}

// ---------------------------------------------------------------------------
// Operator chat and clarification requests (v3)

function addChat(from, to, text, kind, source = null, requestId = null) {
  const m = { id: `c${sim.nextChat++}`, t: r1(sim.t), from, to, text, kind, source, request_id: requestId }
  sim.pendingChat.push(m)
  sim.chat.push(m)
  if (sim.chat.length > 50) sim.chat.splice(0, sim.chat.length - 50)
  return m
}

const OPTION_SETS = {
  rockfall: [
    { id: 'reroute', label: 'Go around' },
    { id: 'proceed_slowly', label: 'Approach slowly' },
    { id: 'wait', label: 'Wait here' },
  ],
  unknown: [
    { id: 'reroute', label: 'Go around' },
    { id: 'proceed_slowly', label: 'Approach slowly' },
    { id: 'wait', label: 'Wait here' },
  ],
  sand: [
    { id: 'back_out', label: 'Back out' },
    { id: 'proceed_slowly', label: 'Push through slowly' },
    { id: 'wait', label: 'Wait here' },
  ],
}
OPTION_SETS.life = [
  { id: 'follow', label: 'Follow it' },
  { id: 'observe_here', label: 'Observe from here' },
  { id: 'keep_exploring', label: 'Keep exploring' },
]
const SAFE_DEFAULT = { rockfall: 'reroute', unknown: 'reroute', sand: 'back_out', life: 'observe_here' }

// The rover (or Gemma looking at its camera) cannot interpret what it senses: it slows
// (or holds, on soft ground) and asks. Returns false when it already has a question open.
function askAbout(rover, kind, info) {
  if (PROTOCOL < 3 || rover.request || rover.state === 'fault') return false
  const hold = kind === 'sand' || kind === 'life'
  rover.request = { pending: true, hold } // act cautiously while the model looks at the image
  const g = gemmaCall('vision')
  later(g.ok ? g.ms : 120, () => {
    if (!rover.request?.pending) return
    if (rover.state === 'fault') {
      rover.request = null
      return
    }
    const conf = 34 + Math.floor(Math.random() * 18)
    const source = g.ok ? 'gemma' : 'rules'
    let question
    let note
    if (kind === 'rockfall') {
      question = `LiDAR shows ${info.n} objects up to ${info.tall} m tall, ${info.dist} m ahead, that were not on my map. ${g.ok ? `From the camera I can't tell rocks from a slope edge (${conf}% confidence).` : 'My rules cannot tell rocks from a slope edge.'} What should I do?`
      note = `Unsure about ${info.n} new objects ${info.dist} m ahead${g.ok ? ` (camera ${conf}% confident)` : ''}. Slowing to 0.15 m/s and asking the operator.`
    } else if (kind === 'life') {
      question = `Possible lifeform ${info.dist} m ${info.dir}. Follow it or keep exploring?`
      note = `Something is moving ${info.dist} m ${info.dir}${g.ok ? `: the camera sees legs (${conf + 30}% confident)` : ''}. Watching it from here and asking the operator.`
    } else if (kind === 'sand') {
      question = `Wheel slip ${info.slip}% but LiDAR shows flat ground (height change under ${info.flat} m).${g.ok ? ` The camera can't tell what the surface is (${conf}% confidence).` : ''} Back out or push through slowly?`
      note = `Slip ${info.slip}% on ground that looks flat${g.ok ? ` (camera ${conf}% confident)` : ''}. Holding and asking the operator.`
    } else {
      question = `Something I can't identify ${info.dist} m ahead: LiDAR says ${info.tall} m tall${g.ok ? `, but the camera is only ${conf}% sure it is a rock` : ', and my rules have no class for it'}. What should I do?`
      note = `Unidentified object ${info.dist} m ahead, ${info.tall} m tall${g.ok ? `, camera ${conf}% confident` : ''}. Slowing to 0.15 m/s and asking the operator.`
    }
    const q = {
      id: `q${sim.nextRequest++}`,
      rover: rover.id,
      t: sim.t,
      question,
      options: OPTION_SETS[kind],
      default: SAFE_DEFAULT[kind],
      deadline: wallNow() + OPERATOR_TIMEOUT_S,
      kind,
      hold,
      at: info.at || [rover.x, rover.y],
    }
    rover.request = q
    sim.requests.push(q)
    planner.asked++
    addChat(rover.id, 'operator', question, 'question', source, q.id)
    addEvent(rover.id, 'warn', note, q.at, source)
  })
  return true
}

function publicRequest(q) {
  return {
    id: q.id,
    rover: q.rover,
    t: r1(q.t),
    question: q.question,
    options: q.options,
    default: q.default,
    expires_in: r1(Math.max(0, q.deadline - wallNow())),
    image: `/api/camera/${q.rover}.jpg?view=fpv`,
  }
}

// Carry out an answer (or the safe default). Returns what the rover says it does.
function applyOption(rover, q, option) {
  if (!rover) return ''
  if (rover.state === 'fault') return 'Motors are down, so I will decide once they are back.'
  if (q.kind === 'life') {
    const k = sim.tracks.find((x) => x.id === rover.trackId)
    if (option === 'keep_exploring' || !k || rover.state !== 'observing') {
      if (rover.state === 'observing') endObserve(rover, null)
      return option === 'keep_exploring' ? 'Leaving it be and carrying on exploring. The track stays on the shared map.' : 'It moved out of sight, so I am carrying on exploring.'
    }
    rover.observeMode = option === 'follow' ? 'follow' : 'watch'
    rover.observeUntil = sim.t + OBSERVE_S
    rover.hold = false // the operator just said what to do
    rover.holdUntil = null
    rover.observeReplanT = 0
    return option === 'follow' ? `Following ${k.id} at ${OBSERVE_DIST} m, never closer, for up to ${OBSERVE_S} s.` : `Watching ${k.id} from here for up to ${OBSERVE_S} s.`
  }
  if (option === 'reroute') {
    rover.hold = false
    const state = rover.state === 'investigating' ? 'investigating' : 'exploring'
    if (rover.goal && setGoal(rover, rover.goal[0], rover.goal[1], state)) {
      return `Going around: new route ${Math.round(remainingPath(rover))} m, keeping at least 1.5 m clear.`
    }
    releaseTarget(rover)
    Object.assign(rover, { goal: null, path: [], state: 'idle', timer: 0.5 })
    return 'Going around: picking a new goal away from it.'
  }
  if (option === 'proceed_slowly') {
    rover.hold = false
    rover.slowUntil = sim.t + 20
    return q.kind === 'sand' ? 'Pushing through at 0.12 m/s. I will back out if slip passes 85%.' : 'Approaching at 0.2 m/s. I will stop 1 m short and look again.'
  }
  if (option === 'wait') {
    rover.hold = true
    rover.holdUntil = sim.t + 60
    return 'Holding here for up to 60 s. Say "resume" to carry on.'
  }
  if (option === 'back_out') {
    backOut(rover)
    return 'Backing out the way I came. The soft zone is on the shared map.'
  }
  return ''
}

function resolveRequest(q, how, option) {
  sim.requests = sim.requests.filter((x) => x !== q)
  sim.closedRequests.set(q.id, how)
  const rover = sim.rovers.find((r) => r.id === q.rover)
  if (rover && rover.request === q) rover.request = null
  const label = q.options.find((o) => o.id === option)?.label || option
  if (how === 'answered') {
    addChat('operator', q.rover, label, 'answer', null, q.id)
    later(300 + Math.random() * 400, () => {
      const did = applyOption(rover, q, option)
      addChat(q.rover, 'operator', did || `OK: ${label.toLowerCase()}.`, 'ack', 'rules', q.id)
      addEvent(q.rover, 'info', `Operator chose "${label}". ${did}`.trim(), q.at, 'operator')
    })
  } else {
    const did = applyOption(rover, q, option)
    addChat(q.rover, 'operator', `No answer in ${OPERATOR_TIMEOUT_S} s, so I took the safe default: ${label.toLowerCase()}. ${did}`.trim(), 'info', 'rules', q.id)
    addEvent(q.rover, 'info', `No operator answer in ${OPERATOR_TIMEOUT_S} s. Took the safe default: ${label.toLowerCase()}.`, q.at)
  }
}

function stepRequests() {
  const now = wallNow()
  for (const q of [...sim.requests]) if (now >= q.deadline) resolveRequest(q, 'expired', q.default)
  if (PROTOCOL >= 3 && !sim.firstAsked && now >= sim.firstQuestionAt) {
    const driving = sim.rovers.filter((r) => r.state === 'exploring' && !r.request && !r.hold && r.speed > 0.1 && remainingPath(r) > 5)
    if (driving.length) {
      sim.firstAsked = true
      askUnknown(driving[Math.floor(sim.rand() * driving.length)])
    } else {
      sim.firstQuestionAt = now + 1
    }
  }
}

function askUnknown(rover) {
  const dist = 4 + Math.floor(sim.rand() * 3)
  return askAbout(rover, 'unknown', { dist, tall: r1(0.6 + sim.rand() * 0.5), at: pointAhead(rover, dist) })
}

class HttpError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

function answerRequest(id, option) {
  const key = String(id ?? '')
  const q = sim.requests.find((x) => x.id === key)
  if (!q) {
    const how = sim.closedRequests.get(key)
    if (how === 'expired') throw new HttpError(409, 'Too late: that question already expired and the rover took the safe default.')
    if (how) throw new HttpError(409, 'That question was already answered.')
    throw new HttpError(404, `No open question with id "${key}".`)
  }
  if (!q.options.some((o) => o.id === option)) throw new HttpError(400, `Unknown option "${option}". Use ${q.options.map((o) => o.id).join(', ')}.`)
  resolveRequest(q, 'answered', option)
}

// --- operator commands (what Gemma or the regex fallback would extract)

function parseCommand(lower) {
  let m
  if ((m = /\b(?:go|drive|head|move|navigate|travel)\s+(?:to|towards)\s*\(?\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*|\s+)(-?\d+(?:\.\d+)?)/.exec(lower))) {
    return { cmd: 'go_to', x: Number(m[1]), y: Number(m[2]) }
  }
  const dir = /\b(?:explore|go|head|drive|move|scout|survey|search)\s+(?:to\s+the\s+|the\s+|towards?\s+(?:the\s+)?)?(north[- ]?east|north[- ]?west|south[- ]?east|south[- ]?west|north|south|east|west)\b/.exec(lower)
  if (dir) return { cmd: 'explore', dir: dir[1].replace(/^(north|south)[- ]?(east|west)$/, '$1-$2') }
  if (/\b(return|come back|come home|go home|head home|go back|back to (?:the )?(?:lander|base)|dock|recall)\b/.test(lower)) return { cmd: 'return' }
  if (/\b(resume|continue|carry on|keep going|unpause|go on)\b/.test(lower)) return { cmd: 'resume' }
  if (/\b(hold|stop|halt|freeze|wait|pause|stay)\b/.test(lower)) return { cmd: 'hold' }
  if (/\b(explore|survey|scout)\b/.test(lower)) return { cmd: 'resume' }
  return { cmd: 'unclear' }
}

// Time to head home: under the fixed floor, or not much more than the trip back needs
// (on the 120 m map a far rover would otherwise run flat on the way).
function lowBattery(rover, floor) {
  const home = Math.hypot(rover.x - LANDER.x, rover.y - LANDER.y) * 1.3 * ENERGY_PER_M
  return rover.battery < Math.max(floor, home + 0.08)
}

function batteryNeed(rover, gx, gy) {
  const there = Math.hypot(gx - rover.x, gy - rover.y) * 1.15 // routes bend
  const back = Math.hypot(gx - LANDER.x, gy - LANDER.y) * 1.15
  return { there: Math.round(there), back: Math.round(back), need: (there + back) * ENERGY_PER_M }
}

function lowBatteryRefusal(rover, gx, gy, what) {
  const { there, back, need } = batteryNeed(rover, gx, gy)
  if (rover.battery - need >= BATTERY_RESERVE) return null
  return `Battery ${Math.round(rover.battery * 100)}%: ${what} is ${there} m, plus ${back} m back to the lander. That uses about ${Math.round(need * 100)}% and would leave under ${Math.round(BATTERY_RESERVE * 100)}%, so I'd rather charge first.`
}

function unavailable(rover) {
  if (rover.state === 'fault') return `Motors are rebooting (${Math.ceil(rover.faultHazard?.remaining ?? 0)} s left). I can't move until they are back.`
  if (rover.state === 'charging' && rover.battery < 0.6) return `Charging at the lander, ${Math.round(rover.battery * 100)}%. I'll go once I'm above 60%.`
  return null
}

function goTo(rover, x, y, state, eventMsg, source) {
  releaseTarget(rover)
  rover.hold = false
  if (!setGoal(rover, x, y, state)) return null
  addEvent(rover.id, 'info', eventMsg(Math.round(remainingPath(rover))), [x, y], source)
  rover.lastGoalEventT = sim.t
  return Math.round(remainingPath(rover))
}

function cmdGoTo(rover, x, y) {
  const busy = unavailable(rover)
  if (busy) return ['refuse', busy]
  const lim = MAP_SIZE / 2 - 1
  if (Math.abs(x) > lim || Math.abs(y) > lim) return ['refuse', `(${x}, ${y}) is outside the map: it spans ${-MAP_SIZE / 2} to ${MAP_SIZE / 2} m on both axes.`]
  const i = toI(x)
  const j = toJ(y)
  if (sim.known[idx(i, j)] && sim.mslope[idx(i, j)] >= sim.slopeLimit) {
    return ['refuse', `The ground at (${x}, ${y}) slopes ${Math.round(sim.mslope[idx(i, j)])} deg, over the ${sim.slopeLimit} deg limit.`]
  }
  const zone = sim.softZones.find((z) => Math.hypot(z[0] - x, z[1] - y) < z[2] + 0.5)
  if (zone) return ['refuse', `(${x}, ${y}) is inside the soft zone at (${zone[0].toFixed(0)}, ${zone[1].toFixed(0)}) that the fleet marked. Pick a spot ${Math.ceil(zone[2] + 1)} m or more from it.`]
  const low = lowBatteryRefusal(rover, x, y, `(${x}, ${y})`)
  if (low) return ['refuse', low]
  const len = goTo(rover, x, y, 'exploring', (l) => `Operator: go to (${x}, ${y}). Route ${l} m.`, 'operator')
  if (len === null) return ['refuse', `No safe route to (${x}, ${y}) on what I know: the way is blocked by slopes over ${sim.slopeLimit} deg.`]
  return ['ack', `Heading to (${x}, ${y}), ${len} m. Max slope on the way ${routeMaxSlope(rover)} deg.`]
}

function cmdExplore(rover, dir) {
  const busy = unavailable(rover)
  if (busy) return ['refuse', busy]
  const [ux, uy] = DIRS[dir]
  let best = null
  let bestScore = -Infinity
  for (const f of sim.frontiers) {
    const dx = f[0] - rover.x
    const dy = f[1] - rover.y
    const d = Math.hypot(dx, dy)
    if (d < 5) continue
    const along = (dx * ux + dy * uy) / d
    if (along < 0.75) continue
    const score = along * Math.pow(f[2], 0.5) / (d + 8)
    if (score > bestScore) {
      bestScore = score
      best = f
    }
  }
  const pct = Math.round((sim.knownCount / sim.known.length) * 100)
  if (!best) return ['refuse', `There is no unexplored ground ${dir} of me that I can reach (${pct}% of the map is explored).`]
  const [fx, fy] = [Math.round(best[0]), Math.round(best[1])]
  const low = lowBatteryRefusal(rover, fx, fy, `the frontier ${dir} at (${fx}, ${fy})`)
  if (low) return ['refuse', low]
  const len = goTo(rover, best[0], best[1], 'exploring', (l) => `Operator: explore ${dir}. Frontier at (${fx}, ${fy}), ${l} m.`, 'operator')
  if (len === null) return ['refuse', `No safe route ${dir} on what I know: slopes over the ${sim.slopeLimit} deg limit block the way.`]
  return ['ack', `Heading ${dir} to the frontier at (${fx}, ${fy}), ${len} m. ${best[2]} unknown cells there, max slope on the way ${routeMaxSlope(rover)} deg.`]
}

function cmdReturn(rover) {
  if (rover.state === 'fault') return ['refuse', `Motors are rebooting (${Math.ceil(rover.faultHazard?.remaining ?? 0)} s left). I'll head back once they are up.`]
  if (rover.state === 'charging') return ['ack', `Already docked and charging, ${Math.round(rover.battery * 100)}%.`]
  rover.hold = false
  const [dx, dy] = dockPoint(rover)
  const d = Math.round(Math.hypot(dx - rover.x, dy - rover.y) * 1.1)
  if (rover.state === 'returning') return ['ack', `Already on my way back, about ${d} m to go.`]
  startReturn(rover, `Operator asked me to return. Heading to the lander, about ${d} m.`, 'operator')
  return ['ack', `Returning to the lander, about ${d} m. Battery ${Math.round(rover.battery * 100)}%.`]
}

function handleChat(text, roverParam) {
  const raw = String(text ?? '').trim()
  if (!raw) throw new HttpError(400, 'Type a message first.')
  if (raw.length > 500) throw new HttpError(400, `That message is ${raw.length} characters long. Keep it under 500.`)
  let targets
  let to
  if (roverParam != null && roverParam !== '' && roverParam !== 'all') {
    const r = sim.rovers.find((q) => q.id === String(roverParam))
    if (!r) throw new HttpError(400, `No rover called "${roverParam}".`)
    targets = [r]
    to = r.id
  } else {
    const lower = raw.toLowerCase()
    const named = [...new Set([...lower.matchAll(/\b(?:rover\s*|r)(\d+)\b/g)].map((m) => `r${m[1]}`))]
      .map((id) => sim.rovers.find((r) => r.id === id))
      .filter(Boolean)
    targets = named.length ? named : sim.rovers
    to = named.length === 1 ? named[0].id : 'all'
  }
  const msg = addChat('operator', to, raw, 'command')
  // Gemma turns the text into a command; when it is not ready the regex parser does.
  const g = gemmaCall('command')
  const source = g.ok ? 'gemma' : 'rules'
  later(g.ok ? g.ms : 120 + Math.random() * 120, () => respond(parseCommand(raw.toLowerCase()), targets, to, raw, source))
  return msg.id
}

function respond(c, targets, to, raw, source) {
  const many = to === 'all' && targets.length > 1
  const reply = (rover, [kind, text]) => addChat(rover.id, 'operator', text, kind, source)
  if (c.cmd === 'hold' || c.cmd === 'resume') {
    const hold = c.cmd === 'hold'
    const changed = []
    for (const r of targets) {
      if (r.hold !== hold) changed.push(r)
      r.hold = hold
      r.holdUntil = null
    }
    if (many) {
      addChat('fleet', 'operator', hold ? `Holding all ${targets.length} rovers where they are. Say "resume all" to carry on.` : `Resuming: ${targets.length} rovers back to exploring.`, 'ack', source)
      addEvent(null, 'info', hold ? 'Operator put the whole fleet on hold.' : 'Operator resumed the fleet.', null, 'operator')
      return
    }
    for (const r of targets) {
      if (hold) reply(r, ['ack', `Holding at (${r.x.toFixed(0)}, ${r.y.toFixed(0)}). Say "resume" when you want me to carry on.`])
      else reply(r, ['ack', changed.includes(r) ? 'Resuming exploration.' : 'I was not on hold. Carrying on exploring.'])
      addEvent(r.id, 'info', hold ? 'Operator: hold. Stopped where I am.' : 'Operator: resume. Back to exploring.', null, 'operator')
    }
    return
  }
  if (c.cmd === 'return') {
    for (const r of targets) reply(r, cmdReturn(r))
    return
  }
  if (c.cmd === 'explore') {
    for (const r of targets) reply(r, cmdExplore(r, c.dir))
    return
  }
  if (c.cmd === 'go_to') {
    let rover = targets[0]
    if (many) {
      // One rover is enough for a point: send the closest one that can go.
      const able = targets.filter((r) => !unavailable(r)).sort((a, b) => Math.hypot(a.x - c.x, a.y - c.y) - Math.hypot(b.x - c.x, b.y - c.y))
      rover = able[0] || targets[0]
      addChat('fleet', 'operator', `${rover.name} is the closest free rover (${Math.round(Math.hypot(rover.x - c.x, rover.y - c.y))} m), sending it.`, 'info', source)
    }
    reply(rover, cmdGoTo(rover, c.x, c.y))
    return
  }
  const short = raw.length > 40 ? `${raw.slice(0, 38)}...` : raw
  const text =
    source === 'gemma'
      ? `I'm not sure what you mean by "${short}". Should I return, hold, resume, explore a direction (like "explore north") or go to a point such as "go to 10, -5"?`
      : `I could not parse "${short}". Try: return, hold, resume, explore north, or go to 10, -5.`
  if (many) addChat('fleet', 'operator', text, 'question', source)
  else reply(targets[0], ['question', text])
}

// ---------------------------------------------------------------------------
// Messages

function fleetSummary() {
  return {
    explored_pct: r1((sim.knownCount / sim.known.length) * 100),
    slope_limit_deg: sim.slopeLimit,
    slope_limit_initial_deg: INITIAL_SLOPE_LIMIT,
    targets_found: sim.discovered.size,
    targets_total: world.targets.length,
    sim_speed: SIM_SPEED,
    budget_ms: BUDGET_MS,
    cycle_ms: CYCLE_MS,
    soft_zones: sim.softZones.map((z) => [r1(z[0]), r1(z[1]), r1(z[2])]),
    ...(PROTOCOL >= 4 && {
      explored_pct: r1(((sim.knownCount + sim.waterSeenCount) / sim.known.length) * 100),
      water_zones: sim.waterZones,
      lifeforms_seen: sim.lifeConfirmed.size,
      samples_cached: [...sim.discovered.values()].filter((t) => t.cached).length,
      samples_carried: sim.rovers.reduce((n, r) => n + r.samples.length, 0),
      science_score: scienceScore(),
    }),
  }
}

function publicLifeform(l) {
  return { id: l.id, kind: l.kind, x: r2(l.x), y: r2(l.y), z: r2(groundSmooth(l.x, l.y) + 0.25 * l.size), yaw: r2(l.yaw), speed: r2(Math.hypot(l.vx, l.vy)), size: l.size }
}

function publicTrack(k) {
  return { id: k.id, x: r2(k.x), y: r2(k.y), vx: r2(k.vx), vy: r2(k.vy), size: k.size, last_seen: r1(k.last_seen), observer: k.observer, label: k.label, status: k.status }
}

function featuresJson() {
  return {
    water: world.water.map((w) => ({ id: w.id, kind: w.kind, level: w.level, outline: w.outline })),
    spires: world.spires.map((sp) => ({ id: sp.id, x: sp.x, y: sp.y, height: sp.height, radius: sp.radius, standing: sp.standing })),
  }
}

function targetList() {
  return [...sim.discovered.values()].map((t) => ({
    id: t.id,
    x: t.x,
    y: t.y,
    kind: t.kind,
    status: t.status,
    found_by: t.found_by,
    found_t: t.found_t,
    ...(PROTOCOL >= 4 && {
      instrument: t.instrument ?? null,
      value: t.value ?? null,
      analysis: t.analysis ?? null,
      carried_by: t.carried_by ?? null,
      cached: !!t.cached,
    }),
  }))
}

function roverStates() {
  return sim.rovers.map((r) => ({
    id: r.id,
    x: r2(r.x),
    y: r2(r.y),
    z: r2(groundAt(r.x, r.y) + 0.25),
    yaw: r2(r.yaw),
    speed: r2(Math.abs(r.speed)),
    battery: r2(r.battery),
    state: r.state,
    goal: r.goal,
    path: r.path.length ? [[r2(r.x), r2(r.y)], ...r.path.slice(0, 29).map(([x, y]) => [r2(x), r2(y)])] : [],
    target_id: r.targetId,
    sensors: r.sensors,
    scan: r.scan,
    brain: brainSummary(r),
    ...(PROTOCOL >= 4 && {
      track_id: r.trackId,
      samples: [...r.samples],
      capacity: TUBES,
      ...(r.sampling && { sampling: { target: r.sampling.target, instrument: r.sampling.instrument, progress: r2(r.sampling.progress) } }),
    }),
  }))
}

function initMessage() {
  const cells = []
  const n = world.n
  for (let k = 0; k < sim.known.length; k++) {
    if (sim.known[k]) cells.push([k % n, (k / n) | 0, sim.measured[k], r1(sim.mslope[k])])
  }
  return {
    type: 'init',
    world: { name: 'Redshift-b', gravity: r2(GRAVITY_G * 9.81), gravity_g: GRAVITY_G, seed: world.seed },
    map: { nx: n, ny: n, res: RES, origin: [world.ox, world.oy], min_h: r2(world.minH), max_h: r2(world.maxH) },
    lander: { ...LANDER },
    rovers: sim.rovers.map((r) => ({ id: r.id, name: r.name, color: r.color })),
    known: { cells },
    targets: targetList(),
    events: sim.events.slice(-50),
    fleet: fleetSummary(),
    hazards: sim.hazards.map(publicHazard),
    t: r1(sim.t),
    ...(PROTOCOL >= 3 && { planner: plannerJson(), chat: sim.chat.slice(-50), requests: sim.requests.map(publicRequest) }),
    ...(PROTOCOL >= 4 && { features: featuresJson(), lifeforms: sim.lifeforms.map(publicLifeform), tracks: sim.tracks.map(publicTrack) }),
  }
}

let truthCache = null
function truthJson() {
  if (!truthCache) {
    truthCache = JSON.stringify({
      nx: world.n,
      ny: world.n,
      res: RES,
      origin: [world.ox, world.oy],
      heights: Array.from(world.heights, r2),
      targets: world.targets.map((t) => ({ id: t.id, x: t.x, y: t.y, kind: t.kind })),
    })
  }
  return truthCache
}

function tick() {
  const dtTotal = SIM_SPEED / TICK_HZ
  const sub = 3
  for (let s = 0; s < sub; s++) {
    const dt = dtTotal / sub
    sim.t += dt
    stepHazards(dt)
    for (const r of sim.rovers) stepRover(r, dt)
    if (PROTOCOL >= 4) stepLifeforms(dt)
  }
  for (const r of sim.rovers) sense(r)
  updateSensors(dtTotal)
  if (PROTOCOL >= 4) stepTracks()
  computeFrontiers()
  fleetEvents()
  stepRequests()
  const msg = {
    type: 'tick',
    t: r1(sim.t),
    rovers: roverStates(),
    cells: sim.pendingCells,
    frontiers: sim.frontiers,
    targets: targetList(),
    events: sim.pendingEvents,
    fleet: fleetSummary(),
    hazards: sim.hazards.map(publicHazard),
    ...(PROTOCOL >= 3 && { planner: plannerJson(), chat: sim.pendingChat, requests: sim.requests.map(publicRequest) }),
    ...(PROTOCOL >= 4 && { lifeforms: sim.lifeforms.map(publicLifeform), tracks: sim.tracks.map(publicTrack) }),
    ...(PROTOCOL >= 4 && sim.featuresSent !== sim.featuresVersion && { features: featuresJson() }),
  }
  sim.featuresSent = sim.featuresVersion
  sim.pendingCells = []
  sim.pendingEvents = []
  sim.pendingChat = []
  broadcast(JSON.stringify(msg))
}

// ---------------------------------------------------------------------------
// Servers

const wss = new WebSocketServer({ noServer: true })

function broadcast(text) {
  for (const client of wss.clients) {
    if (client.readyState === 1) client.send(text)
  }
}

function reset(seed) {
  truthCache = null
  newSim(seed)
  broadcast(JSON.stringify(initMessage()))
  // Every client just got all of this in init: do not repeat it in the next tick.
  sim.pendingEvents = []
  sim.pendingCells = []
  sim.pendingChat = []
  console.log(`[mock] reset, seed ${seed}`)
}

function sendJson(res, status, body) {
  const text = typeof body === 'string' ? body : JSON.stringify(body)
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Cache-Control': 'no-store',
  })
  res.end(text)
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost')
  if (req.method === 'OPTIONS') return sendJson(res, 204, '')
  if (req.method === 'GET' && url.pathname === '/api/health') return sendJson(res, 200, { ok: true })
  if (req.method === 'GET' && url.pathname === '/api/truth') return sendJson(res, 200, truthJson())
  if (req.method === 'POST' && url.pathname === '/api/reset') {
    let body = ''
    req.on('data', (chunk) => {
      body += chunk
      if (body.length > 1e5) req.destroy()
    })
    req.on('end', () => {
      let seed = world.seed
      try {
        const parsed = body ? JSON.parse(body) : {}
        if (Number.isFinite(parsed?.seed)) seed = Math.trunc(parsed.seed)
      } catch {
        return sendJson(res, 400, { ok: false, error: 'Body must be JSON like {"seed": 7}' })
      }
      reset(seed)
      sendJson(res, 200, { ok: true, seed })
    })
    return
  }
  if (req.method === 'POST' && url.pathname === '/api/hazard') {
    let body = ''
    req.on('data', (chunk) => {
      body += chunk
      if (body.length > 1e5) req.destroy()
    })
    req.on('end', () => {
      let parsed
      try {
        parsed = body ? JSON.parse(body) : {}
      } catch {
        return sendJson(res, 400, { ok: false, error: 'Body must be JSON like {"type": "rockfall", "rover": "r2"}' })
      }
      try {
        const hasX = parsed?.x != null
        const hasY = parsed?.y != null
        if ((hasX || hasY) && !(Number.isFinite(parsed.x) && Number.isFinite(parsed.y))) throw new HazardError('Give both x and y as numbers (world metres), or neither.')
        const point = hasX && hasY ? [parsed.x, parsed.y] : null
        const h = createHazard(String(parsed?.type ?? ''), parsed?.rover == null || parsed.rover === '' ? null : parsed.rover, point)
        console.log(`[mock] hazard ${h.id} ${h.type}${h.rover ? ` on ${h.rover}` : ''}`)
        sendJson(res, 200, { ok: true, hazard: publicHazard(h) })
      } catch (err) {
        if (!(err instanceof HazardError)) console.error(err)
        sendJson(res, 400, { ok: false, error: err instanceof HazardError ? err.message : 'Could not create the hazard.' })
      }
    })
    return
  }
  // v3: operator chat, answers to clarification requests, and mock-only test hooks.
  const v3 = {
    '/api/chat': (b) => ({ ok: true, id: handleChat(b?.text, b?.rover) }),
    '/api/answer': (b) => {
      answerRequest(b?.id, b?.option == null ? '' : String(b.option))
      return { ok: true }
    },
    '/api/_mock/planner': (b) => {
      const state = String(b?.state || '')
      if (!['ready', 'warming', 'offline', 'disabled'].includes(state)) throw new HttpError(400, 'state must be ready, warming, offline or disabled')
      planner.mode = state === 'ready' || state === 'warming' ? 'auto' : state
      planner.readyAt = state === 'warming' ? wallNow() + GEMMA_WARMUP_S : state === 'ready' ? wallNow() : planner.readyAt
      return { ok: true, planner: plannerJson() }
    },
    '/api/_mock/ask': (b) => {
      const pool = sim.rovers.filter((r) => (b?.rover ? r.id === b.rover : true) && !r.request && r.state !== 'fault')
      const rover = pool.find((r) => r.path.length) || pool[0]
      if (!rover || !askUnknown(rover)) throw new HttpError(400, 'No rover free to ask a question.')
      return { ok: true, rover: rover.id }
    },
    '/api/_mock/teleport': (b) => {
      const rover = sim.rovers.find((r) => r.id === b?.rover)
      if (!rover || !Number.isFinite(b?.x) || !Number.isFinite(b?.y)) throw new HttpError(400, 'Body: {"rover": "r1", "x": 10, "y": -5, "hold": false}')
      if (Math.abs(b.x) > MAP_SIZE / 2 - 1 || Math.abs(b.y) > MAP_SIZE / 2 - 1 || waterAt(b.x, b.y) >= 0) throw new HttpError(400, 'That spot is off the map or in the water.')
      if (rover.state === 'observing') endObserve(rover, null)
      releaseTarget(rover)
      Object.assign(rover, { x: b.x, y: b.y, path: [], goal: null, speed: 0, hold: !!b.hold, holdUntil: null, timer: 1.5 })
      if (rover.state !== 'fault') rover.state = 'idle'
      if (Number.isFinite(b.yaw)) rover.yaw = b.yaw
      sense(rover)
      return { ok: true }
    },
    '/api/_mock/rearm': (b) => {
      // {"armed": true} (default): the next lifeform sighting asks the operator again.
      sim.firstLifeAsked = b?.armed === false
      return { ok: true, armed: !sim.firstLifeAsked }
    },
    '/api/_mock/battery': (b) => {
      const rover = sim.rovers.find((r) => r.id === b?.rover)
      if (!rover || !Number.isFinite(b?.battery)) throw new HttpError(400, 'Body: {"rover": "r1", "battery": 0.3}')
      rover.battery = clamp(b.battery, 0, 1)
      return { ok: true }
    },
  }
  if (req.method === 'POST' && v3[url.pathname] && (PROTOCOL >= 3 || url.pathname.startsWith('/api/_mock/'))) {
    let body = ''
    req.on('data', (chunk) => {
      body += chunk
      if (body.length > 1e5) req.destroy()
    })
    req.on('end', () => {
      let parsed
      try {
        parsed = body ? JSON.parse(body) : {}
      } catch {
        return sendJson(res, 400, { ok: false, error: 'Body must be JSON.' })
      }
      try {
        sendJson(res, 200, v3[url.pathname](parsed))
      } catch (err) {
        if (!(err instanceof HttpError)) console.error(err)
        sendJson(res, err instanceof HttpError ? err.status : 500, { ok: false, error: err instanceof HttpError ? err.message : 'Something went wrong in the mock.' })
      }
    })
    return
  }
  const cam = /^\/api\/camera\/([^/]+)\.jpg$/.exec(url.pathname)
  if (req.method === 'GET' && cam) {
    const id = decodeURIComponent(cam[1])
    if (!sim.rovers.some((r) => r.id === id)) return sendJson(res, 404, { ok: false, error: `Unknown rover ${id}` })
    // No renderer in the mock: the UI must show its "Camera unavailable" state.
    return sendJson(res, 503, { ok: false, error: 'Camera rendering is not available in the mock sim' })
  }
  if (req.method === 'GET' && url.pathname === '/') {
    res.writeHead(200, { 'Content-Type': 'text/plain' })
    return res.end('Redshift mock sim. WebSocket on /ws, HTTP on /api/health, /api/truth, /api/reset, /api/hazard, /api/chat, /api/answer, /api/camera/{id}.jpg.\n')
  }
  sendJson(res, 404, { ok: false, error: 'Not found' })
})

server.on('upgrade', (req, socket, head) => {
  const { pathname } = new URL(req.url, 'http://localhost')
  if (pathname !== '/ws') {
    socket.destroy()
    return
  }
  wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req))
})

wss.on('connection', (ws) => {
  ws.on('message', () => {}) // the protocol says the server ignores client text
  ws.on('error', () => {})
  ws.send(JSON.stringify(initMessage()))
})

newSim(Number(process.env.SEED || 42) | 0)
setInterval(tick, 1000 / TICK_HZ)
server.listen(PORT, () => {
  console.log(`[mock] Redshift mock sim on http://localhost:${PORT} (WebSocket /ws), seed ${world.seed}, ${NUM_ROVERS} rovers, ${SIM_SPEED}x, Gemma ${plannerState()}`)
})
