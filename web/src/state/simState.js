// Pure, framework-free handling of sim messages (see PROTOCOL.md). No React or
// three.js in here, so the message handling can also be exercised in Node.
//
// The terrain arrays are mutated in place for speed. Everything else is replaced
// immutably, so React selectors see a new object when (and only when) it changed.

export const MAX_EVENTS = 200
export const TRAIL_POINTS = 200
export const FAR_PAST = -1e6
export const DEFAULT_ROVER_COLORS = ['#ff7a45', '#ffd23f', '#ff5fa2', '#7cf29a', '#b39dff', '#4dd0e1', '#f4f4f4', '#ffa8a8']

export function emptyState() {
  return {
    gen: 0, // increments on every init, so views can reset
    ready: false,
    world: null,
    map: null,
    lander: null,
    roverMeta: [], // [{ id, name, color }] in display order
    rovers: {}, // id -> latest rover state from a tick
    trails: {}, // id -> { buf, head, count, version }
    targets: [],
    frontiers: [],
    events: [], // newest first, each with a client-side `key`
    fleet: {},
    hazards: [], // v2: full list from the latest tick (may stay empty with older sims)
    limit: { current: null, initial: null, prev: null, changedAt: null },
    t: 0,
    terrain: null,
    eventSeq: 0,
    initSeq: 0, // events with a key above this arrived live (after the latest init)
    ticks: 0,
    lastMsgAt: 0,
  }
}

const num = (v, fallback) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback)

export function normalizeMap(map = {}) {
  const nx = Math.max(2, Math.round(num(map.nx, 161)))
  const ny = Math.max(2, Math.round(num(map.ny, 161)))
  const res = num(map.res, 0.5) > 0 ? num(map.res, 0.5) : 0.5
  const origin = Array.isArray(map.origin) ? map.origin : [-((nx - 1) * res) / 2, -((ny - 1) * res) / 2]
  const ox = num(origin[0], -((nx - 1) * res) / 2)
  const oy = num(origin[1], -((ny - 1) * res) / 2)
  let minH = num(map.min_h, -3)
  let maxH = num(map.max_h, 8)
  if (maxH - minH < 0.1) maxH = minH + 0.1
  return {
    nx,
    ny,
    res,
    ox,
    oy,
    minH,
    maxH,
    width: (nx - 1) * res,
    depth: (ny - 1) * res,
    cx: ox + ((nx - 1) * res) / 2,
    cy: oy + ((ny - 1) * res) / 2,
  }
}

export function createTerrain(map) {
  const n = map.nx * map.ny
  return {
    ...map,
    h: new Float32Array(n),
    slope: new Float32Array(n),
    known: new Uint8Array(n),
    revealAt: new Float32Array(n).fill(FAR_PAST), // wall clock seconds when first seen
    knownCount: 0,
    // Bounding box of known samples (grid indices), for the auto-framing camera.
    bi0: map.nx,
    bi1: -1,
    bj0: map.ny,
    bj1: -1,
    // Queue of changed sample indices, drained by the renderer.
    dirty: new Int32Array(n),
    dirtyFlag: new Uint8Array(n),
    dirtyCount: 0,
    version: 0,
  }
}

// cells: [[i, j, h, slope_deg], ...]. Returns how many were applied.
export function applyCells(T, cells, now) {
  if (!T || !Array.isArray(cells)) return 0
  let applied = 0
  for (let c = 0; c < cells.length; c++) {
    const cell = cells[c]
    if (!cell) continue
    const i = cell[0] | 0
    const j = cell[1] | 0
    if (i < 0 || j < 0 || i >= T.nx || j >= T.ny) continue
    const h = cell[2]
    if (typeof h !== 'number' || !Number.isFinite(h)) continue // a null height is not a measurement
    const s = typeof cell[3] === 'number' ? cell[3] : NaN
    const k = j * T.nx + i
    if (!T.known[k]) {
      T.known[k] = 1
      T.knownCount++
      T.revealAt[k] = now
      if (i < T.bi0) T.bi0 = i
      if (i > T.bi1) T.bi1 = i
      if (j < T.bj0) T.bj0 = j
      if (j > T.bj1) T.bj1 = j
    }
    T.h[k] = h
    if (Number.isFinite(s)) T.slope[k] = s // otherwise keep the last value (0 for a new sample)
    if (!T.dirtyFlag[k]) {
      T.dirtyFlag[k] = 1
      T.dirty[T.dirtyCount++] = k
    }
    applied++
  }
  if (applied) T.version++
  return applied
}

// Height of the known ground at world (x, y): bilinear over the known corners,
// or null when none of the four surrounding samples is known.
export function sampleGround(T, x, y) {
  if (!T) return null
  const fx = (x - T.ox) / T.res
  const fy = (y - T.oy) / T.res
  if (fx < 0 || fy < 0 || fx > T.nx - 1 || fy > T.ny - 1) return null
  const i0 = Math.min(T.nx - 2, Math.floor(fx))
  const j0 = Math.min(T.ny - 2, Math.floor(fy))
  const tx = fx - i0
  const ty = fy - j0
  const k00 = j0 * T.nx + i0
  const ks = [k00, k00 + 1, k00 + T.nx, k00 + T.nx + 1]
  const ws = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
  let sum = 0
  let wsum = 0
  for (let q = 0; q < 4; q++) {
    if (T.known[ks[q]]) {
      sum += T.h[ks[q]] * ws[q]
      wsum += ws[q]
    }
  }
  if (wsum < 1e-6) return null
  return sum / wsum
}

// Like sampleGround, but searches outwards (up to `radius` samples) when the point is unknown.
export function sampleGroundNear(T, x, y, radius = 6) {
  const direct = sampleGround(T, x, y)
  if (direct !== null || !T) return direct
  const ci = Math.round((x - T.ox) / T.res)
  const cj = Math.round((y - T.oy) / T.res)
  for (let r = 1; r <= radius; r++) {
    let sum = 0
    let count = 0
    for (let dj = -r; dj <= r; dj++) {
      for (let di = -r; di <= r; di++) {
        if (Math.max(Math.abs(di), Math.abs(dj)) !== r) continue
        const i = ci + di
        const j = cj + dj
        if (i < 0 || j < 0 || i >= T.nx || j >= T.ny) continue
        const k = j * T.nx + i
        if (T.known[k]) {
          sum += T.h[k]
          count++
        }
      }
    }
    if (count) return sum / count
  }
  return null
}

function makeTrail() {
  return { buf: new Float32Array(TRAIL_POINTS * 3), head: 0, count: 0, version: 0 }
}

function pushTrail(trail, x, y, z) {
  if (trail.count) {
    const last = ((trail.head - 1 + TRAIL_POINTS) % TRAIL_POINTS) * 3
    const dx = trail.buf[last] - x
    const dy = trail.buf[last + 1] - y
    if (dx * dx + dy * dy < 0.15 * 0.15) return false
  }
  const o = trail.head * 3
  trail.buf[o] = x
  trail.buf[o + 1] = y
  trail.buf[o + 2] = z
  trail.head = (trail.head + 1) % TRAIL_POINTS
  trail.count = Math.min(TRAIL_POINTS, trail.count + 1)
  trail.version++
  return true
}

// Oldest to newest, as world [x, y, z] triples. Calls fn(x, y, z, index, count).
export function forEachTrailPoint(trail, fn) {
  const start = (trail.head - trail.count + TRAIL_POINTS) % TRAIL_POINTS
  for (let q = 0; q < trail.count; q++) {
    const o = ((start + q) % TRAIL_POINTS) * 3
    fn(trail.buf[o], trail.buf[o + 1], trail.buf[o + 2], q, trail.count)
  }
}

function roverMetaFrom(list) {
  return (Array.isArray(list) ? list : [])
    .filter((r) => r && r.id != null)
    .map((r, i) => ({
      id: String(r.id),
      name: r.name ? String(r.name) : String(r.id),
      color: typeof r.color === 'string' && r.color ? r.color : DEFAULT_ROVER_COLORS[i % DEFAULT_ROVER_COLORS.length],
    }))
}

const EVENT_LEVELS = new Set(['info', 'warn', 'discovery', 'hazard'])

function keyEvents(list, seq) {
  const out = []
  for (const e of Array.isArray(list) ? list : []) {
    if (!e || typeof e !== 'object') continue
    seq++
    out.push({
      key: seq,
      t: num(e.t, 0),
      rover: e.rover == null ? null : String(e.rover),
      level: EVENT_LEVELS.has(e.level) ? e.level : 'info',
      msg: e.msg == null ? '' : String(e.msg),
      at: isXY(e.at) ? [e.at[0], e.at[1]] : null,
    })
  }
  return { events: out, seq }
}

function limitFromFleet(fleet, prevLimit, now, detectChange) {
  const current = num(fleet.slope_limit_deg, prevLimit.current)
  const initial = num(fleet.slope_limit_initial_deg, prevLimit.initial ?? current)
  if (current === prevLimit.current && initial === prevLimit.initial) return prevLimit
  const changed = detectChange && prevLimit.current != null && current != null && Math.abs(current - prevLimit.current) >= 0.05
  return {
    current,
    initial,
    prev: changed ? prevLimit.current : prevLimit.prev,
    changedAt: changed ? now : prevLimit.changedAt,
  }
}

function isXY(p) {
  return Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1])
}
const validTarget = (t) => t && typeof t === 'object' && t.id != null && Number.isFinite(t.x) && Number.isFinite(t.y)
const isObj = (v) => v && typeof v === 'object' && !Array.isArray(v)

// v2 scan: flat [x, y, z, ...] in world metres. Kept as is (no copy) when it is usable.
function sanitizeScan(scan) {
  if (!Array.isArray(scan) || scan.length < 3) return null
  return scan
}

// v2 hazards: keep the shape from PROTOCOL.md, drop anything unusable.
export function sanitizeHazards(list) {
  if (!Array.isArray(list)) return []
  const out = []
  for (const h of list) {
    if (!isObj(h) || typeof h.type !== 'string') continue
    out.push({
      id: h.id == null ? `${h.type}:${h.t}` : String(h.id),
      type: h.type,
      t: num(h.t, 0),
      x: num(h.x, null),
      y: num(h.y, null),
      radius: num(h.radius, null),
      rover: h.rover == null ? null : String(h.rover),
      rocks: Array.isArray(h.rocks) ? h.rocks.filter((r) => isXY(r)) : [],
      known: !!h.known,
      remaining: num(h.remaining, null),
      active: h.active !== false,
    })
  }
  return out
}

function sanitizeRover(r, prev) {
  return {
    id: String(r.id),
    x: num(r.x, prev?.x ?? 0),
    y: num(r.y, prev?.y ?? 0),
    z: num(r.z, prev?.z ?? 0),
    yaw: num(r.yaw, prev?.yaw ?? 0),
    speed: num(r.speed, 0),
    battery: Number.isFinite(r.battery) ? Math.max(0, Math.min(1, r.battery)) : (prev?.battery ?? null), // null shows as "--"
    state: typeof r.state === 'string' ? r.state : 'idle',
    goal: isXY(r.goal) ? [r.goal[0], r.goal[1]] : null,
    path: Array.isArray(r.path) ? r.path.filter(isXY) : [],
    target_id: r.target_id == null ? null : String(r.target_id),
    // v2 (null when the sim does not send them)
    sensors: isObj(r.sensors) ? r.sensors : null,
    scan: sanitizeScan(r.scan),
    brain: isObj(r.brain) ? r.brain : null,
  }
}

// Sim time right now, estimated between ticks (ticks carry `t` about 5 times a second).
export function simTimeNow(s, now) {
  const speed = Number.isFinite(s.fleet?.sim_speed) && s.fleet.sim_speed > 0 ? s.fleet.sim_speed : 1
  return s.t + Math.max(0, Math.min(0.5, now - s.lastMsgAt)) * speed
}

export function applyInit(prev, msg, now) {
  const s = emptyState()
  s.gen = (prev?.gen ?? 0) + 1
  s.ready = true
  s.lastMsgAt = now
  s.world = msg.world && typeof msg.world === 'object' ? msg.world : {}
  s.map = normalizeMap(msg.map)
  const l = msg.lander || {}
  s.lander = { x: num(l.x, 0), y: num(l.y, 0), radius: num(l.radius, 2.5) }
  s.roverMeta = roverMetaFrom(msg.rovers)
  s.terrain = createTerrain(s.map)
  // Cells known before we joined appear immediately, without the reveal animation.
  applyCells(s.terrain, msg.known?.cells, FAR_PAST)
  for (const r of s.roverMeta) s.trails[r.id] = makeTrail()
  // Some servers may include positions in init; accept them if present.
  for (const r of Array.isArray(msg.rovers) ? msg.rovers : []) {
    if (r && r.id != null && Number.isFinite(r.x) && Number.isFinite(r.y)) s.rovers[String(r.id)] = sanitizeRover(r)
  }
  s.targets = Array.isArray(msg.targets) ? msg.targets.filter(validTarget) : []
  const keyed = keyEvents((Array.isArray(msg.events) ? msg.events : []).slice(-50), prev?.eventSeq ?? 0)
  s.events = keyed.events.reverse()
  s.eventSeq = keyed.seq
  s.initSeq = keyed.seq
  s.fleet = msg.fleet && typeof msg.fleet === 'object' ? { ...msg.fleet } : {}
  s.limit = limitFromFleet(s.fleet, s.limit, now, false)
  s.hazards = sanitizeHazards(msg.hazards)
  s.t = num(msg.t, s.events[0]?.t ?? 0)
  return s
}

export function applyTick(prev, msg, now) {
  if (!prev.ready) return prev // a tick before any init cannot be placed on a map
  const s = { ...prev, ticks: prev.ticks + 1, lastMsgAt: now }
  s.t = num(msg.t, prev.t)

  // Cells first, so rover trails can sit on freshly revealed ground.
  applyCells(s.terrain, msg.cells, now)

  if (Array.isArray(msg.rovers)) {
    const rovers = { ...prev.rovers }
    let meta = prev.roverMeta
    let trails = prev.trails
    for (const raw of msg.rovers) {
      if (!raw || raw.id == null) continue
      const id = String(raw.id)
      const r = sanitizeRover(raw, prev.rovers[id])
      rovers[id] = r
      if (!meta.some((m) => m.id === id)) {
        // A rover we were not told about in init: show it anyway.
        meta = [...meta, { id, name: id, color: DEFAULT_ROVER_COLORS[meta.length % DEFAULT_ROVER_COLORS.length] }]
      }
      if (!trails[id]) trails = { ...trails, [id]: makeTrail() }
      const gz = sampleGround(s.terrain, r.x, r.y)
      pushTrail(trails[id], r.x, r.y, gz ?? r.z)
    }
    s.rovers = rovers
    s.roverMeta = meta
    s.trails = trails
  }

  if (Array.isArray(msg.frontiers)) s.frontiers = msg.frontiers.filter(isXY)
  if (Array.isArray(msg.targets)) s.targets = msg.targets.filter(validTarget)

  if (Array.isArray(msg.events) && msg.events.length) {
    const keyed = keyEvents(msg.events, prev.eventSeq)
    // A client that joins between two ticks can get an event both in init and in the
    // next tick. Skip exact repeats of recent events.
    const recent = prev.events.slice(0, 50)
    const fresh = keyed.events.filter((e) => !recent.some((r) => r.t === e.t && r.rover === e.rover && r.level === e.level && r.msg === e.msg))
    s.eventSeq = keyed.seq
    if (fresh.length) s.events = fresh.reverse().concat(prev.events).slice(0, MAX_EVENTS)
  }

  if (msg.fleet && typeof msg.fleet === 'object') {
    s.fleet = { ...prev.fleet, ...msg.fleet }
    s.limit = limitFromFleet(s.fleet, prev.limit, now, true)
  }
  if (Array.isArray(msg.hazards)) {
    // Sent every tick; keep the old array when nothing changed so selectors stay quiet.
    const next = sanitizeHazards(msg.hazards)
    s.hazards = JSON.stringify(next) === JSON.stringify(prev.hazards) ? prev.hazards : next
  }
  return s
}

export function reduce(state, msg, now) {
  if (!msg || typeof msg !== 'object') return state
  if (msg.type === 'init') return applyInit(state, msg, now)
  if (msg.type === 'tick') return applyTick(state, msg, now)
  return state
}
