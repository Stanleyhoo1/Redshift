// Redshift mock sim.
// A protocol-compliant fake of the simulation server (see ../../PROTOCOL.md) so the
// web UI can be developed and demoed without MuJoCo. It has no real physics or
// autonomy, it only produces realistic-looking data:
//   procedural terrain, rovers driving to frontiers along A* paths, sensing within
//   a 10 m radius, target discoveries, battery drain and charging, slope-limit learning.
//
// Run: node mock/mock-server.mjs   (env: PORT, SEED, NUM_ROVERS, SIM_SPEED, MAP_SIZE, GRAVITY_G)

import http from 'node:http'
import { WebSocketServer } from 'ws'

const PORT = Number(process.env.PORT || 8000)
const TICK_HZ = 5
const SIM_SPEED = Number(process.env.SIM_SPEED || 3)
const NUM_ROVERS = Math.max(1, Math.min(8, Number(process.env.NUM_ROVERS || 3) | 0))
const MAP_SIZE = Number(process.env.MAP_SIZE || 80)
const GRAVITY_G = Number(process.env.GRAVITY_G || 2)
const RES = 0.5
const SENSOR_RADIUS = 10
const NUM_TARGETS = 6
const INITIAL_SLOPE_LIMIT = 30
const MIN_SLOPE_LIMIT = 18
const LANDER = { x: 0, y: 0, radius: 2.5 }
const ROVER_COLORS = ['#ff7a45', '#ffd23f', '#ff5fa2', '#7cf29a', '#b39dff', '#4dd0e1', '#f4f4f4', '#ffa8a8']
const KIND_LABEL = { mineral: 'mineral deposit', ice: 'ice patch', vent: 'thermal vent' }

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
    targets.push({ id: `t${targets.length + 1}`, x: r1(x), y: r1(y), kind: kinds[targets.length] })
  }

  return { seed, n, ox, oy, heights, slopes, minH, maxH, targets }
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
// Simulation state

let world
let sim

function newSim(seed) {
  world = generateWorld(seed)
  const rand = mulberry32(seed * 104729 + 3)
  const n = world.n
  sim = {
    rand,
    t: 0,
    known: new Uint8Array(n * n),
    measured: new Float32Array(n * n),
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
  }
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
    })
  }
  addEvent(null, 'info', `Survey started. ${NUM_ROVERS} rover${NUM_ROVERS === 1 ? '' : 's'} deployed from the lander.`)
  for (const r of sim.rovers) sense(r)
  computeFrontiers()
}

function addEvent(roverId, level, msg) {
  const e = { t: r1(sim.t), rover: roverId, level, msg }
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

function gaussian(rand) {
  const u = Math.max(1e-9, rand())
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
}

function sense(rover) {
  const R = Math.round(SENSOR_RADIUS / RES)
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
      sim.known[k] = 1
      sim.knownCount++
      const h = r2(world.heights[k] + gaussian(sim.rand) * 0.015)
      sim.measured[k] = h
      sim.pendingCells.push([i, j, h, r1(world.slopes[k])])
    }
  }
  for (const t of world.targets) {
    if (sim.discovered.has(t.id)) continue
    if (Math.hypot(t.x - rover.x, t.y - rover.y) > SENSOR_RADIUS - 1) continue
    sim.discovered.set(t.id, { ...t, status: 'found', found_by: rover.id, found_t: r1(sim.t), assigned: null })
    const label = KIND_LABEL[t.kind]
    addEvent(rover.id, 'discovery', `Found ${/^[aeiou]/.test(label) ? 'an' : 'a'} ${label} at (${t.x.toFixed(1)}, ${t.y.toFixed(1)}).`)
  }
}

function computeFrontiers() {
  const n = world.n
  const bucket = 6
  const buckets = new Map()
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = j * n + i
      if (!sim.known[k]) continue
      const edge =
        (i > 0 && !sim.known[k - 1]) ||
        (i < n - 1 && !sim.known[k + 1]) ||
        (j > 0 && !sim.known[k - n]) ||
        (j < n - 1 && !sim.known[k + n])
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

function planPath(sx, sy, gx, gy) {
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
        if (sim.known[k]) {
          const s = world.slopes[k]
          if (s >= limit && nb !== goal) continue
          cost *= 1 + (s / limit) ** 2 * 2.5
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

function setGoal(rover, gx, gy, state, why) {
  const path = planPath(rover.x, rover.y, gx, gy)
  if (!path) return false
  rover.state = state
  rover.goal = [r1(gx), r1(gy)]
  rover.path = path
  rover.replanT = sim.t + 6
  if (why && sim.t - rover.lastGoalEventT > 25) {
    addEvent(rover.id, 'info', why)
    rover.lastGoalEventT = sim.t
  }
  return true
}

function chooseExplorationGoal(rover) {
  // An unclaimed discovered target nearby wins over plain exploration.
  for (const t of sim.discovered.values()) {
    if (t.status === 'sampled' || t.assigned) continue
    if (Math.hypot(t.x - rover.x, t.y - rover.y) < 30 && rover.battery > 0.45) {
      if (setGoal(rover, t.x, t.y, 'investigating', `Heading to the ${KIND_LABEL[t.kind]} found by ${t.found_by === rover.id ? 'this rover' : nameOf(t.found_by)}.`)) {
        t.assigned = rover.id
        rover.targetId = t.id
        rover.lastGoalEventT = sim.t
        return true
      }
    }
  }
  const others = sim.rovers.filter((r) => r !== rover && r.goal)
  let best = null
  let bestScore = -Infinity
  for (const f of sim.frontiers) {
    const d = Math.hypot(f[0] - rover.x, f[1] - rover.y)
    if (d < 4) continue
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
  return setGoal(rover, best[0], best[1], 'exploring', `Heading to a frontier at (${best[0].toFixed(0)}, ${best[1].toFixed(0)}).`)
}

function nameOf(id) {
  return sim.rovers.find((r) => r.id === id)?.name ?? id
}

function dockPoint(rover) {
  return [LANDER.x + Math.cos(rover.dockAngle) * (LANDER.radius + 0.9), LANDER.y + Math.sin(rover.dockAngle) * (LANDER.radius + 0.9)]
}

function releaseTarget(rover) {
  if (!rover.targetId) return
  const t = sim.discovered.get(rover.targetId)
  if (t && t.status !== 'sampled') {
    t.assigned = null
    if (t.status === 'investigating') t.status = 'found'
  }
  rover.targetId = null
}

function startReturn(rover) {
  releaseTarget(rover)
  const [dx, dy] = dockPoint(rover)
  if (!setGoal(rover, dx, dy, 'returning')) {
    rover.state = 'returning'
    rover.goal = [r1(dx), r1(dy)]
    rover.path = [[dx, dy]]
  }
  addEvent(rover.id, 'info', `Battery at ${Math.round(rover.battery * 100)}%. Returning to the lander to charge.`)
  rover.lastGoalEventT = sim.t
}

function drive(rover, dt, maxSpeed) {
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
  rover.x += Math.cos(rover.yaw) * rover.speed * dt
  rover.y += Math.sin(rover.yaw) * rover.speed * dt
  const lim = MAP_SIZE / 2 - 1
  rover.x = clamp(rover.x, -lim, lim)
  rover.y = clamp(rover.y, -lim, lim)
  return false
}

function stepRover(rover, dt) {
  const moving = rover.speed > 0.05
  rover.battery = clamp(rover.battery - dt * (moving ? 0.0032 : 0.0008), 0, 1)

  switch (rover.state) {
    case 'idle': {
      rover.speed = 0
      rover.timer -= dt
      if (rover.battery < 0.3) {
        startReturn(rover) // nothing left to explore nearby, so do not idle on a flat battery
        break
      }
      if (rover.timer <= 0 && !chooseExplorationGoal(rover)) rover.timer = 3
      break
    }
    case 'exploring': {
      if (rover.battery < 0.28) {
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
      if (rover.battery < 0.22) {
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
        rover.state = 'charging'
        rover.path = []
        rover.goal = null
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
    case 'recovering': {
      rover.timer -= dt
      rover.speed = 0.25
      rover.x -= Math.cos(rover.yaw) * 0.25 * dt
      rover.y -= Math.sin(rover.yaw) * 0.25 * dt
      if (rover.timer <= 0) {
        rover.speed = 0
        rover.state = 'idle'
        rover.timer = 0.5
        addEvent(rover.id, 'info', `Backed off the slope. Replanning with the ${sim.slopeLimit} deg limit.`)
      }
      break
    }
  }
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
      addEvent(rover.id, 'warn', 'Wheel slip on loose ground. Slowing down.')
      rover.speed *= 0.3
    }
    sim.nextSlipT = sim.t + 60 + sim.rand() * 60
  }
  const pct = (sim.knownCount / sim.known.length) * 100
  while (sim.milestones.length && pct >= sim.milestones[0]) {
    addEvent(null, 'info', `${sim.milestones.shift()}% of the map explored.`)
  }
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
  }))
}

function initMessage() {
  const cells = []
  const n = world.n
  for (let k = 0; k < sim.known.length; k++) {
    if (sim.known[k]) cells.push([k % n, (k / n) | 0, sim.measured[k], r1(world.slopes[k])])
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
    for (const r of sim.rovers) stepRover(r, dt)
  }
  for (const r of sim.rovers) sense(r)
  computeFrontiers()
  fleetEvents()
  const msg = {
    type: 'tick',
    t: r1(sim.t),
    rovers: roverStates(),
    cells: sim.pendingCells,
    frontiers: sim.frontiers,
    targets: targetList(),
    events: sim.pendingEvents,
    fleet: fleetSummary(),
  }
  sim.pendingCells = []
  sim.pendingEvents = []
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
  if (req.method === 'GET' && url.pathname === '/') {
    res.writeHead(200, { 'Content-Type': 'text/plain' })
    return res.end('Redshift mock sim. WebSocket on /ws, HTTP on /api/health, /api/truth, /api/reset.\n')
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
  console.log(`[mock] Redshift mock sim on http://localhost:${PORT} (WebSocket /ws), seed ${world.seed}, ${NUM_ROVERS} rovers, ${SIM_SPEED}x`)
})
