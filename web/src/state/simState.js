// Pure, framework-free handling of sim messages (see PROTOCOL.md). No React or
// three.js in here, so the message handling can also be exercised in Node.
//
// The terrain arrays are mutated in place for speed. Everything else is replaced
// immutably, so React selectors see a new object when (and only when) it changed.

export const MAX_EVENTS = 1000 // v8: per-rover log filters need some depth; the list renders a capped window
export const MAX_CHAT = 200
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
    // v3 (stay empty or null with older sims)
    v3: false, // the sim has sent planner, chat or requests at least once
    planner: null, // local Gemma status
    chat: [], // oldest first, each with a client-side `key`
    chatSeq: 0,
    chatInitSeq: 0, // messages with a key above this arrived live
    requests: [], // open clarification questions, each with a wall-clock `deadline`
    // v4 (stay empty with older sims)
    v4: false, // the sim has sent features, lifeforms or tracks at least once
    features: EMPTY_FEATURES, // { water: [{ id, kind, level, outline }], spires: [{ id, x, y, height, radius, standing }] }
    lifeforms: [], // ground truth creatures
    tracks: [], // what the fleet believes it is tracking
    fallen: {}, // spire id -> { x0, y0, x1, y1, t, hazard }: every spire seen falling since the last init
    limit: { current: null, initial: null, prev: null, changedAt: null },
    // v8 (null with older sims): fleet.priority, sanitized, same object while unchanged
    priority: null,
    t: 0,
    terrain: null,
    eventSeq: 0,
    initSeq: 0, // events with a key above this arrived live (after the latest init)
    ticks: 0,
    lastMsgAt: 0,
  }
}

const EMPTY_FEATURES = { water: [], spires: [] }

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
  // Called many times per frame by the 3D view: no temporary arrays.
  const K = T.known
  const H = T.h
  const k = j0 * T.nx + i0
  let sum = 0
  let wsum = 0
  let w = (1 - tx) * (1 - ty)
  if (K[k]) (sum += H[k] * w), (wsum += w)
  w = tx * (1 - ty)
  if (K[k + 1]) (sum += H[k + 1] * w), (wsum += w)
  w = (1 - tx) * ty
  if (K[k + T.nx]) (sum += H[k + T.nx] * w), (wsum += w)
  w = tx * ty
  if (K[k + T.nx + 1]) (sum += H[k + T.nx + 1] * w), (wsum += w)
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
    // A rover never drives 6 m in one tick: it was moved (a reset, a test hook). Start over
    // rather than draw a straight line across the map.
    if (dx * dx + dy * dy > 36) trail.count = 0
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
      source: EVENT_SOURCES.has(e.source) ? e.source : 'rules', // v3: missing means rules
      decision: sanitizeDecision(e.decision), // v8: null when the event records no decision
    })
  }
  return { events: out, seq }
}

const EVENT_SOURCES = new Set(['rules', 'gemma', 'operator'])

// v8 event `decision`: { question, options: [string], choice, by, confidence, why }. Options
// may be ids ("proceed_slowly") or labels; objects with an id or label are accepted too.
export function sanitizeDecision(d) {
  if (!isObj(d)) return null
  const word = (v) => (v == null ? null : isObj(v) ? String(v.label ?? v.id ?? '') || null : String(v) || null)
  const options = (Array.isArray(d.options) ? d.options : []).map(word).filter(Boolean).slice(0, 12)
  const choice = word(d.choice)
  if (!choice && !options.length) return null
  const conf = num(d.confidence, null)
  return {
    question: typeof d.question === 'string' && d.question.trim() ? d.question.trim() : null,
    options,
    choice,
    by: typeof d.by === 'string' && d.by ? d.by.toLowerCase() : null,
    confidence: conf === null ? null : Math.max(0, Math.min(1, conf > 1 ? conf / 100 : conf)),
    why: typeof d.why === 'string' && d.why.trim() ? d.why.trim() : null,
  }
}

// v8 fleet `priority`: { kind, label, count, since } while set, else null. Positions are not
// in the protocol; if a sim sends them (`places`, `gaps` or `areas`: [x, y, r] or {x, y, r,
// label}) the map draws them. Same object while unchanged.
export function sanitizePriority(p, prev) {
  if (!isObj(p)) return null
  const raw = [p.places, p.gaps, p.areas].find(Array.isArray) || []
  const places = []
  for (const q of raw) {
    const x = Array.isArray(q) ? q[0] : q?.x
    const y = Array.isArray(q) ? q[1] : q?.y
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    const r = Array.isArray(q) ? q[2] : (q.r ?? q.radius)
    places.push({ x, y, r: Number.isFinite(r) && r > 0.2 ? r : 3, label: !Array.isArray(q) && typeof q.label === 'string' ? q.label : null })
    if (places.length >= 24) break
  }
  const next = {
    kind: typeof p.kind === 'string' && p.kind ? p.kind : null,
    label: typeof p.label === 'string' && p.label.trim() ? p.label.trim() : null,
    count: Number.isFinite(p.count) ? Math.max(0, Math.round(p.count)) : null,
    since: num(p.since, null),
    places,
  }
  return prev && JSON.stringify(prev) === JSON.stringify(next) ? prev : next
}

// v8 detections (rover `vision.objects`, camera header X-Detections): label, conf 0..1,
// box [x0, y0, x1, y1] normalised (top left origin), dist m, bearing deg (left positive).
export function sanitizeDetections(list) {
  if (!Array.isArray(list)) return null
  const out = []
  for (const o of list) {
    if (!isObj(o) || !Array.isArray(o.box) || o.box.length < 4) continue
    let [x0, y0, x1, y1] = o.box
    if (![x0, y0, x1, y1].every(Number.isFinite)) continue
    if (x1 < x0) [x0, x1] = [x1, x0]
    if (y1 < y0) [y0, y1] = [y1, y0]
    const c = num(o.conf, null)
    out.push({
      label: typeof o.label === 'string' && o.label ? o.label.toLowerCase() : 'object',
      conf: c === null ? null : Math.max(0, Math.min(1, c > 1 ? c / 100 : c)),
      box: [Math.max(0, Math.min(1, x0)), Math.max(0, Math.min(1, y0)), Math.max(0, Math.min(1, x1)), Math.max(0, Math.min(1, y1))],
      dist: num(o.dist, null),
      bearing: num(o.bearing, null),
    })
    if (out.length >= 48) break
  }
  return out
}

function sanitizeVision(v, prev) {
  if (!isObj(v)) return null
  const objects = sanitizeDetections(v.objects) || []
  const next = { frame: num(v.frame, null), t: num(v.t, null), view: typeof v.view === 'string' ? v.view : null, ms: num(v.ms, null), objects }
  if (prev && prev.frame === next.frame && prev.t === next.t && next.frame !== null) return prev
  return next
}
const CHAT_KINDS = new Set(['command', 'ack', 'refuse', 'question', 'answer', 'info', 'reply'])

// v3 chat messages, keyed like events. Unknown kinds read as info.
function keyChat(list, seq) {
  const out = []
  for (const m of Array.isArray(list) ? list : []) {
    if (!isObj(m) || m.text == null || m.text === '') continue
    seq++
    out.push({
      key: seq,
      id: m.id == null ? null : String(m.id),
      t: num(m.t, 0),
      from: m.from == null ? 'fleet' : String(m.from),
      to: m.to == null ? null : String(m.to),
      text: String(m.text),
      kind: CHAT_KINDS.has(m.kind) ? m.kind : 'info',
      source: m.source === 'gemma' || m.source === 'rules' ? m.source : null,
      request_id: m.request_id == null ? null : String(m.request_id),
      // v5: the operator message this answers (on reply, ack and refuse). Missing: null.
      reply_to: m.reply_to == null || m.reply_to === '' ? null : String(m.reply_to),
    })
  }
  return { chat: out, seq }
}

const sameChat = (a, b) => (a.id !== null && b.id !== null ? a.id === b.id : a.t === b.t && a.from === b.from && a.text === b.text)

// v3 planner status: the protocol's fields, numbers or null. Same object when unchanged.
function sanitizePlanner(p, prev) {
  if (!isObj(p)) return null
  const n = (v) => num(v, null)
  const next = {
    model: p.model == null ? null : String(p.model),
    state: typeof p.state === 'string' && p.state ? p.state : p.online ? 'ready' : 'offline',
    online: !!p.online,
    calls: n(p.calls),
    ok: n(p.ok),
    fallbacks: n(p.fallbacks),
    avg_ms: n(p.avg_ms),
    p95_ms: n(p.p95_ms),
    last_ms: n(p.last_ms),
    budget_ms: n(p.budget_ms),
    over_budget: n(p.over_budget),
    asked_operator: n(p.asked_operator), // not in the protocol; shown when a sim sends it
    detail: typeof p.detail === 'string' && p.detail ? p.detail : null,
  }
  return prev && JSON.stringify(prev) === JSON.stringify(next) ? prev : next
}

// v3 open requests. `expires_in` is wall seconds left when the message was sent; it is
// turned into a client-side deadline (performance.now() seconds). Network and main-thread
// delays only ever make a fresh estimate late, so the earliest one is kept (a jump of
// more than 2 s later is taken as the sim extending the time).
// `total` estimates the full countdown, so the bar starts full even for a late joiner.
export function sanitizeRequests(list, prev, now, simT, speed) {
  if (!Array.isArray(list)) return prev
  const old = new Map(prev.map((q) => [q.id, q]))
  const out = []
  for (const q of list) {
    if (!isObj(q) || q.id == null) continue
    const id = String(q.id)
    const was = old.get(id)
    // v6: an option may name a place on the map (`x`, `y`), e.g. a `choose_place` candidate.
    const options = (Array.isArray(q.options) ? q.options : [])
      .filter((o) => (isObj(o) && o.id != null) || typeof o === 'string')
      .map((o) => {
        if (typeof o === 'string') return { id: o, label: o }
        const out = { id: String(o.id), label: o.label == null ? String(o.id) : String(o.label) }
        if (o.suggested === true) out.suggested = true // v8: Gemma's pick in supervised mode
        if (Number.isFinite(o.x) && Number.isFinite(o.y)) {
          out.x = o.x
          out.y = o.y
        }
        return out
      })
    const left = num(q.expires_in, null)
    let deadline = left === null ? null : now + Math.max(0, left)
    if (was && was.deadline !== null && deadline !== null && deadline > was.deadline - 0.05 && deadline < was.deadline + 2) deadline = was.deadline
    const t = num(q.t, simT)
    const elapsed = speed > 0 ? Math.max(0, (simT - t) / speed) : 0
    const total = was?.total ?? (left === null ? null : Math.max(1, Math.min(left + elapsed, left + 120)))
    const next = {
      id,
      rover: q.rover == null ? null : String(q.rover),
      // v6: "anomaly" | "lifeform" | "choose_place" (null from older sims)
      kind: typeof q.kind === 'string' && q.kind ? q.kind : null,
      t,
      question: q.question == null ? '' : String(q.question),
      options,
      default: q.default == null ? null : String(q.default),
      image: typeof q.image === 'string' && q.image ? q.image : null,
      deadline,
      total,
    }
    out.push(was && JSON.stringify(was) === JSON.stringify(next) ? was : next)
  }
  return out.length === prev.length && out.every((q, k) => q === prev[k]) ? prev : out
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
      // v4 spire_fall: the spire and its base and tip once it lies down
      spire: h.spire == null ? null : String(h.spire),
      x0: num(h.x0, null),
      y0: num(h.y0, null),
      x1: num(h.x1, null),
      y1: num(h.y1, null),
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
    // v4 and v4b (null when the sim does not send them)
    track_id: r.track_id == null ? null : String(r.track_id),
    samples: Array.isArray(r.samples) ? r.samples.filter((id) => id != null).map(String) : null,
    capacity: Number.isFinite(r.capacity) && r.capacity > 0 ? Math.round(r.capacity) : null,
    sampling: isObj(r.sampling) && r.sampling.target != null ? { target: String(r.sampling.target), instrument: r.sampling.instrument == null ? null : String(r.sampling.instrument), progress: Math.max(0, Math.min(1, num(r.sampling.progress, 0))) } : null,
    // v5 running totals (null when the sim does not send them)
    stats: sanitizeStats(r.stats, prev?.stats),
    // v7: the area the rover concentrates on (present only while it does), else null
    focus: sanitizeFocus(r.focus),
    // v8: the onboard detector's latest result, else null
    vision: sanitizeVision(r.vision, prev?.vision),
  }
}

// v7 rover focus: { x, y, r, reason: 'operator' | 'interest', label, ends_in } or null.
// `ends_in` is sim seconds left as of the tick (null when the sim gives none).
function sanitizeFocus(f) {
  if (!isObj(f) || !Number.isFinite(f.x) || !Number.isFinite(f.y)) return null
  const r = num(f.r, 0)
  return {
    x: f.x,
    y: f.y,
    r: r > 0.5 ? r : 10,
    reason: f.reason === 'interest' ? 'interest' : 'operator',
    label: typeof f.label === 'string' && f.label.trim() ? f.label.trim() : null,
    ends_in: Number.isFinite(f.ends_in) ? Math.max(0, f.ends_in) : null,
    // v8: "explore" (unknown ground first) or "resurvey" (a loop through mapped ground)
    mode: f.mode === 'resurvey' ? 'resurvey' : f.mode === 'explore' ? 'explore' : null,
  }
}

const STAT_KEYS = ['distance_m', 'mapped_m2', 'targets_found', 'samples_taken', 'hazards_met', 'lifeforms_seen', 'replans']

// v5 rover stats: numbers or null per field. Same object while unchanged.
function sanitizeStats(st, prev) {
  if (!isObj(st)) return null
  const next = {}
  for (const k of STAT_KEYS) next[k] = num(st[k], null)
  if (prev && STAT_KEYS.every((k) => prev[k] === next[k])) return prev
  return next
}

// v4 static features (init, or a tick that resends them). Same object when unchanged.
export function sanitizeFeatures(f, prev) {
  if (!isObj(f)) return prev ?? EMPTY_FEATURES
  const water = (Array.isArray(f.water) ? f.water : [])
    .filter((w) => isObj(w) && Number.isFinite(w.level) && Array.isArray(w.outline))
    .map((w, i) => ({ id: w.id == null ? `w${i + 1}` : String(w.id), kind: w.kind === 'river' ? 'river' : 'lake', level: w.level, outline: w.outline.filter(isXY).map((p) => [p[0], p[1]]) }))
    .filter((w) => w.outline.length >= 3)
  const spires = (Array.isArray(f.spires) ? f.spires : [])
    .filter((p) => isObj(p) && Number.isFinite(p.x) && Number.isFinite(p.y))
    .map((p, i) => ({ id: p.id == null ? `s${i + 1}` : String(p.id), x: p.x, y: p.y, height: num(p.height, 3) > 0.2 ? num(p.height, 3) : 3, radius: num(p.radius, 0.35) > 0.02 ? num(p.radius, 0.35) : 0.35, standing: p.standing !== false }))
  // A resend after a spire fall changes only the spires: keep the water (and its meshes).
  const sameWater = prev && JSON.stringify(prev.water) === JSON.stringify(water)
  const sameSpires = prev && JSON.stringify(prev.spires) === JSON.stringify(spires)
  if (sameWater && sameSpires) return prev
  return { water: sameWater ? prev.water : water, spires: sameSpires ? prev.spires : spires }
}

const LIFE_KINDS = new Set(['grazer'])
function sanitizeLifeforms(list) {
  if (!Array.isArray(list)) return null
  return list
    .filter((l) => isObj(l) && l.id != null && Number.isFinite(l.x) && Number.isFinite(l.y))
    .map((l) => ({ id: String(l.id), kind: LIFE_KINDS.has(l.kind) ? l.kind : String(l.kind || 'grazer'), x: l.x, y: l.y, z: num(l.z, null), yaw: num(l.yaw, 0), speed: num(l.speed, 0), size: num(l.size, 1) > 0.1 ? num(l.size, 1) : 1 }))
}

function sanitizeTracks(list) {
  if (!Array.isArray(list)) return null
  return list
    .filter((k) => isObj(k) && k.id != null && Number.isFinite(k.x) && Number.isFinite(k.y))
    .map((k) => ({
      id: String(k.id),
      x: k.x,
      y: k.y,
      vx: num(k.vx, 0),
      vy: num(k.vy, 0),
      size: num(k.size, 1) > 0.1 ? num(k.size, 1) : 1,
      last_seen: num(k.last_seen, null),
      observer: k.observer == null ? null : String(k.observer),
      label: k.label == null || k.label === '' ? null : String(k.label),
      status: k.status === 'lost' ? 'lost' : 'tracking',
    }))
}

// Spires seen falling (spire_fall hazards), remembered until the next init even if the
// sim later drops the hazard from its list.
function updateFallen(prev, hazards) {
  let out = prev
  for (const h of hazards) {
    if (h.type !== 'spire_fall' || ![h.x0, h.y0, h.x1, h.y1].every(Number.isFinite)) continue
    const key = h.spire ?? h.id
    const was = out[key]
    if (was && was.hazard === h.id && was.x1 === h.x1 && was.y1 === h.y1) continue
    if (out === prev) out = { ...prev }
    out[key] = { x0: h.x0, y0: h.y0, x1: h.x1, y1: h.y1, t: h.t, hazard: h.id }
  }
  return out
}

// v4 fields, shared by init and tick. Each one may be missing (older sims).
function applyV4(s, prev, msg) {
  if ('features' in msg || 'lifeforms' in msg || 'tracks' in msg) s.v4 = true
  if ('features' in msg) s.features = sanitizeFeatures(msg.features, prev.features)
  const life = sanitizeLifeforms(msg.lifeforms)
  if (life) s.lifeforms = life
  const tracks = sanitizeTracks(msg.tracks)
  if (tracks) s.tracks = JSON.stringify(tracks) === JSON.stringify(prev.tracks) ? prev.tracks : tracks
  s.fallen = updateFallen(prev.fallen, s.hazards)
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
  s.priority = sanitizePriority(s.fleet.priority, null)
  s.hazards = sanitizeHazards(msg.hazards)
  s.t = num(msg.t, s.events[0]?.t ?? 0)
  s.chatSeq = prev?.chatSeq ?? 0 // keys keep growing across inits
  s.chatInitSeq = s.chatSeq
  applyV3(s, s, msg, now, true)
  applyV4(s, s, msg)
  return s
}

// v3 fields, shared by init and tick. Each one may be missing (older sims).
function applyV3(s, prev, msg, now, isInit) {
  if ('planner' in msg || 'chat' in msg || 'requests' in msg) s.v3 = true
  if ('planner' in msg) s.planner = sanitizePlanner(msg.planner, prev.planner)
  if (Array.isArray(msg.chat)) {
    if (isInit) {
      const keyed = keyChat(msg.chat.slice(-50), prev.chatSeq)
      s.chat = keyed.chat
      s.chatSeq = keyed.seq
      s.chatInitSeq = keyed.seq
    } else if (msg.chat.length) {
      const keyed = keyChat(msg.chat, prev.chatSeq)
      // A client that joins between two ticks can get a message both in init and in the next tick.
      const recent = prev.chat.slice(-60)
      const fresh = keyed.chat.filter((m) => !recent.some((r) => sameChat(r, m)))
      s.chatSeq = keyed.seq
      if (fresh.length) s.chat = prev.chat.concat(fresh).slice(-MAX_CHAT)
    }
  }
  const speed = Number.isFinite(s.fleet?.sim_speed) && s.fleet.sim_speed > 0 ? s.fleet.sim_speed : 1
  if (Array.isArray(msg.requests)) s.requests = sanitizeRequests(msg.requests, prev.requests, now, s.t, speed)
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
    // v8: `priority` is present only while set, so a tick without it clears it.
    if (!('priority' in msg.fleet)) delete s.fleet.priority
    s.priority = sanitizePriority(s.fleet.priority, prev.priority)
    s.limit = limitFromFleet(s.fleet, prev.limit, now, true)
    // v4: the identified water is resent every tick; keep the old array while it is the
    // same, so the water mask is only redrawn when the fleet identifies more.
    const wz = s.fleet.water_zones
    if (Array.isArray(wz) && Array.isArray(prev.fleet.water_zones) && wz.length === prev.fleet.water_zones.length && JSON.stringify(wz) === JSON.stringify(prev.fleet.water_zones)) {
      s.fleet.water_zones = prev.fleet.water_zones
    }
  }
  if (Array.isArray(msg.hazards)) {
    // Sent every tick; keep the old array when nothing changed so selectors stay quiet.
    const next = sanitizeHazards(msg.hazards)
    s.hazards = JSON.stringify(next) === JSON.stringify(prev.hazards) ? prev.hazards : next
  }
  applyV3(s, prev, msg, now, false)
  applyV4(s, prev, msg)
  return s
}

export function reduce(state, msg, now) {
  if (!msg || typeof msg !== 'object') return state
  if (msg.type === 'init') return applyInit(state, msg, now)
  if (msg.type === 'tick') return applyTick(state, msg, now)
  return state
}
