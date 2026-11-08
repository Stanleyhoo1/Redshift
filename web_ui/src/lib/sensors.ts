// Live instrument readings (docs/protocol.md §8.7 `sensors` + `health`), with fallbacks for older
// backends that only send the v1 fields, plus the pure maths behind the sparklines.
import type { RobotTelemetry } from './types'

export const G0 = 9.80665

/** One flattened sample per telemetry frame; null = channel not reported. */
export interface SensorSample {
  t: number
  accel_g: number | null
  gyro_dps: number | null
  roll: number | null
  pitch: number | null
  lidar_nearest_m: number | null
  lidar_returns_pct: number | null
  camera_conf: number | null
  v_cmd: number | null
  v_wheel: number | null
  v_ground: number | null
  slip: number | null
  altitude_agl: number | null
  /** drone only: its own wind estimate from drift (§10.3) */
  wind_mps: number | null
  battery_pct: number | null
  solar_w: number | null
  load_w: number | null
  net_w: number | null
  health_pct: number | null
}
export type SensorKey = Exclude<keyof SensorSample, 't'>

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** Magnitude of a 3-vector in rad/s, as °/s. */
export function gyroDps(g: readonly (number | null)[] | null | undefined): number | null {
  if (!g || g.length < 3) return null
  const [x, y, z] = g.map(num)
  if (x === null || y === null || z === null) return null
  return (Math.hypot(x, y, z) * 180) / Math.PI
}

/** Flatten a robot's telemetry into one sample, falling back to the v1 fields where §8 is missing. */
export function sampleOf(r: RobotTelemetry, t: number): SensorSample {
  const s = r.sensors ?? null
  const h = r.health ?? null
  const scout = r.type === 'scout'
  const accel = s?.accel ?? null
  const accelG = num(s?.accel_g) ?? (accel ? Math.hypot(accel[0], accel[1], accel[2]) / G0 : null)
  return {
    t,
    accel_g: accelG,
    gyro_dps: gyroDps(s?.gyro ?? r.imu?.gyro),
    roll: num(s?.roll_deg) ?? num(r.imu?.roll_deg),
    pitch: num(s?.pitch_deg) ?? num(r.imu?.pitch_deg),
    lidar_nearest_m: scout ? null : num(s?.lidar_nearest_m),
    lidar_returns_pct: scout ? null : num(s?.lidar_returns_pct),
    camera_conf: num(s?.camera_conf) ?? num(r.uncertainty?.camera_conf),
    v_cmd: num(s?.v_cmd),
    v_wheel: scout ? null : num(s?.v_wheel),
    v_ground: num(s?.v_ground) ?? num(r.speed),
    slip: scout ? null : (num(s?.slip) ?? num(r.slip)),
    altitude_agl: num(s?.altitude_agl) ?? num(r.altitude_agl),
    wind_mps: scout ? num(r.wind_mps) : null,
    battery_pct: num(h?.battery_pct),
    solar_w: num(h?.solar_w),
    load_w: num(h?.load_w),
    net_w: num(h?.net_w),
    health_pct: num(h?.pct),
  }
}

/** Hours until the battery is empty at the current net power; null when charging / level / unknown. */
export function timeToEmptyH(batteryWh: number | null | undefined, netW: number | null | undefined): number | null {
  if (batteryWh == null || netW == null || !Number.isFinite(batteryWh) || !Number.isFinite(netW)) return null
  if (netW >= -0.5) return null
  return Math.max(0, batteryWh) / -netW
}

export type Tone = 'ok' | 'warn' | 'crit'

/** Health %: critical below 15 (§8.7), degraded below 60 (the rover will not resume under it). */
export function healthTone(pct: number | null | undefined, critical?: boolean | null): Tone {
  if (critical) return 'crit'
  if (pct == null) return 'ok'
  return pct < 15 ? 'crit' : pct < 60 ? 'warn' : 'ok'
}

export function batteryTone(pct: number | null | undefined): Tone {
  if (pct == null) return 'ok'
  return pct < 10 ? 'crit' : pct < 25 ? 'warn' : 'ok'
}

// ---- sparkline maths ----------------------------------------------------------------------

export interface Range {
  lo: number
  hi: number
}

/**
 * Value range for a sparkline: the data's own min/max, widened to at least `minSpan` around its
 * middle (so noise on a steady channel stays a flat line), always including `include` values
 * (reference lines such as gravity or zero), then clamped to `clamp`.
 */
export function autoRange(values: readonly (number | null)[], minSpan: number, include: readonly number[] = [], clamp?: Range): Range | null {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (lo === Infinity) return null
  for (const v of include) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  if (hi - lo < minSpan) {
    const mid = (hi + lo) / 2
    lo = mid - minSpan / 2
    hi = mid + minSpan / 2
  }
  if (clamp) {
    // shift the window inside the clamp before cutting it, so a pinned channel keeps its span
    if (lo < clamp.lo) {
      hi = Math.min(clamp.hi, hi + (clamp.lo - lo))
      lo = clamp.lo
    }
    if (hi > clamp.hi) {
      lo = Math.max(clamp.lo, lo - (hi - clamp.hi))
      hi = clamp.hi
    }
  }
  return { lo, hi }
}

/**
 * SVG polyline point strings for one channel over the window [tEnd − span, tEnd], mapped onto a
 * W×H box (y down). A null sample or a gap longer than `maxGap` seconds breaks the line.
 */
export function sparkSegments(
  ts: readonly number[],
  values: readonly (number | null)[],
  tEnd: number,
  span: number,
  W: number,
  H: number,
  range: Range,
  maxGap = 1.5,
): string[] {
  const out: string[] = []
  let cur: string[] = []
  let prevT = -Infinity
  const k = range.hi - range.lo || 1
  for (let i = 0; i < ts.length; i++) {
    const t = ts[i]
    const v = values[i]
    if (t < tEnd - span) continue
    if (v === null || !Number.isFinite(v) || t - prevT > maxGap) {
      if (cur.length > 1) out.push(cur.join(' '))
      cur = []
    }
    prevT = t
    if (v === null || !Number.isFinite(v)) continue
    const x = W - ((tEnd - t) / span) * W
    const y = H - ((Math.min(range.hi, Math.max(range.lo, v)) - range.lo) / k) * H
    cur.push(`${x.toFixed(1)},${y.toFixed(1)}`)
  }
  if (cur.length > 1) out.push(cur.join(' '))
  return out
}

/** y coordinate of a reference value in the same box (clamped). */
export function sparkY(v: number, H: number, range: Range): number {
  const k = range.hi - range.lo || 1
  return H - ((Math.min(range.hi, Math.max(range.lo, v)) - range.lo) / k) * H
}
