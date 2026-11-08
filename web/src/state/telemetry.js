// Client-side history of every rover's readings (and the fleet-wide Gemma latency), for
// the sensor detail charts. Each sample keeps the sim time `t` (what the charts plot) and
// the wall-clock arrival time `w`. Filled from every tick for every rover, whether or not a
// panel is open, in fixed-size ring buffers: memory stays bounded however long the sim
// runs. No React in here: the charts read the buffers when they draw.

export const TELEMETRY_CAP = 1800 // samples per rover: 6 minutes at 5 ticks a second

export const ROVER_SERIES = ['speed', 'wheel', 'slip', 'load', 'pitch', 'roll', 'battery', 'brain', 'brainAvg', 'lidar']
export const FLEET_SERIES = ['gemma', 'gemmaAvg', 'gemmaP95']

function makeBuf(keys) {
  const series = {}
  for (const k of keys) series[k] = new Float32Array(TELEMETRY_CAP)
  return { head: 0, count: 0, t: new Float64Array(TELEMETRY_CAP), w: new Float64Array(TELEMETRY_CAP), series, version: 0 }
}

export const telemetry = {
  gen: -1,
  rovers: new Map(), // id -> buffer
  fleet: makeBuf(FLEET_SERIES),
  version: 0,
}

const val = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : NaN)

function push(buf, t, w, values) {
  const i = buf.head
  buf.t[i] = t
  buf.w[i] = w
  for (const k in values) buf.series[k][i] = values[k]
  buf.head = (i + 1) % TELEMETRY_CAP
  if (buf.count < TELEMETRY_CAP) buf.count++
  buf.version++
}

function clear(buf) {
  buf.head = 0
  buf.count = 0
  buf.version++
}

// Reused for every sample, so recording allocates nothing per tick.
const _r = {}
const _f = {}

// Called with the reduced state after every init and tick (see store.js).
export function recordTelemetry(state, msg, now) {
  if (state.gen !== telemetry.gen) {
    telemetry.gen = state.gen
    for (const b of telemetry.rovers.values()) clear(b)
    clear(telemetry.fleet)
  }
  if (!msg || msg.type !== 'tick' || !Array.isArray(msg.rovers)) return
  const t = state.t
  for (const raw of msg.rovers) {
    if (!raw || raw.id == null) continue
    const id = String(raw.id)
    const r = state.rovers[id]
    if (!r) continue
    let buf = telemetry.rovers.get(id)
    if (!buf) {
      buf = makeBuf(ROVER_SERIES)
      telemetry.rovers.set(id, buf)
    }
    // A repeated sim time (a paused sim) adds nothing new.
    if (buf.count && buf.t[(buf.head - 1 + TELEMETRY_CAP) % TELEMETRY_CAP] === t) continue
    const s = r.sensors
    _r.speed = val(s?.ground_speed ?? r.speed)
    _r.wheel = val(s?.wheel_speed)
    _r.slip = val(s?.slip)
    _r.load = val(s?.motor_load)
    _r.pitch = val(s?.pitch)
    _r.roll = val(s?.roll)
    _r.battery = val(r.battery)
    _r.brain = val(r.brain?.last_ms)
    _r.brainAvg = val(r.brain?.avg_ms)
    _r.lidar = val(s?.lidar_range)
    push(buf, t, now, _r)
  }
  const p = state.planner
  const fb = telemetry.fleet
  if (p && !(fb.count && fb.t[(fb.head - 1 + TELEMETRY_CAP) % TELEMETRY_CAP] === t)) {
    const ready = p.state === 'ready'
    _f.gemma = ready ? val(p.last_ms) : NaN
    _f.gemmaAvg = ready ? val(p.avg_ms) : NaN
    _f.gemmaP95 = ready ? val(p.p95_ms) : NaN
    push(fb, t, now, _f)
  }
  telemetry.version++
}

// Index of the oldest sample, so callers can walk `count` samples from there.
export const oldestIndex = (buf) => (buf.head - buf.count + TELEMETRY_CAP) % TELEMETRY_CAP
