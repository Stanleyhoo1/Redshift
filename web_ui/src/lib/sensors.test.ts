import { describe, expect, it } from 'vitest'
import { pushWindowed } from './history'
import { autoRange, gyroDps, healthTone, sampleOf, sparkSegments, sparkY, timeToEmptyH } from './sensors'
import type { RobotTelemetry } from './types'

const base: RobotTelemetry = {
  type: 'rover',
  pose: { x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0 },
  speed: 0.6,
  state: 'EXPLORING',
  mode: 'FUSED',
  goal: null,
  v_limit: 0.83,
  slip: 0.05,
  tip_alarm_deg: 28.8,
  max_slope_deg: 16.8,
  altitude_agl: null,
  imu: { roll_deg: 1.5, pitch_deg: -2, gyro: [0, 0, Math.PI / 18] },
  uncertainty: { score: 0.1, camera_conf: 0.9, lidar_conf: 0.95, imu_vib: 0.05 },
  edge: { robot: 'rover_0', profile: 'orin_nano_15w', host_ms: { perception: 1, planner: 1, mapping: 1 }, device_ms: 5, loop_hz: 10, load_pct: 50, power_w: 10, perception_hz: 10, throttle_level: 0, uncertainty: 0.1 },
  path: [],
  trail: [],
  last_safe: null,
}

describe('sampleOf', () => {
  it('falls back to v1 fields when §8 blocks are missing', () => {
    const s = sampleOf(base, 10)
    expect(s.roll).toBe(1.5)
    expect(s.camera_conf).toBe(0.9)
    expect(s.v_ground).toBe(0.6)
    expect(s.slip).toBe(0.05)
    expect(s.gyro_dps).toBeCloseTo(10, 6)
    expect(s.health_pct).toBeNull()
    expect(s.lidar_nearest_m).toBeNull()
  })
  it('prefers the sensors block and derives accel_g from the vector', () => {
    const s = sampleOf(
      {
        ...base,
        sensors: { accel: [0, 0, 2 * 9.80665], accel_g: null, gyro: [0, 0, 0], roll_deg: 3, pitch_deg: 4, lidar_nearest_m: 4.2, lidar_returns_pct: 78, camera_conf: 0.97, v_cmd: 0.8, v_wheel: 0.79, v_ground: 0.76, slip: 0.04, altitude_agl: null },
        health: { pct: 92, structure_pct: 96, drive_pct: 96, battery_pct: 81.4, battery_wh: 488.4, capacity_wh: 600, solar_w: 182.4, load_w: 131, net_w: 51.4, faults: [], critical: false, shocks: 2, peak_g: 1.9 },
      },
      11,
    )
    expect(s.accel_g).toBeCloseTo(2, 9)
    expect(s.roll).toBe(3)
    expect(s.lidar_nearest_m).toBe(4.2)
    expect(s.v_cmd).toBe(0.8)
    expect(s.health_pct).toBe(92)
    expect(s.net_w).toBe(51.4)
  })
  it('never reports LiDAR or wheel channels for the drone', () => {
    const s = sampleOf({ ...base, type: 'scout', altitude_agl: 12, sensors: { accel: null, accel_g: 1, gyro: null, roll_deg: 0, pitch_deg: 0, lidar_nearest_m: 3, lidar_returns_pct: 50, camera_conf: 1, v_cmd: 3, v_wheel: 3, v_ground: 3, slip: 0, altitude_agl: 12 } }, 0)
    expect(s.lidar_nearest_m).toBeNull()
    expect(s.v_wheel).toBeNull()
    expect(s.slip).toBeNull()
    expect(s.altitude_agl).toBe(12)
  })
  it('measures gyro magnitude in °/s', () => expect(gyroDps([0, 0, Math.PI])).toBeCloseTo(180, 9))
})

describe('time to empty', () => {
  it('divides remaining energy by the drain', () => expect(timeToEmptyH(488.4, -69.8)).toBeCloseTo(7.0, 1))
  it('is null while charging or level', () => {
    expect(timeToEmptyH(488, 51)).toBeNull()
    expect(timeToEmptyH(488, -0.2)).toBeNull()
    expect(timeToEmptyH(null, -50)).toBeNull()
  })
})

describe('health tone', () => {
  it('uses the §8.7 critical threshold', () => {
    expect(healthTone(7)).toBe('crit')
    expect(healthTone(40)).toBe('warn')
    expect(healthTone(92)).toBe('ok')
    expect(healthTone(92, true)).toBe('crit')
  })
})

describe('sparkline maths', () => {
  it('widens a flat channel to the minimum span', () => {
    expect(autoRange([2.2, 2.21, 2.19], 1)).toEqual({ lo: expect.closeTo(1.7, 9), hi: expect.closeTo(2.7, 9) })
  })
  it('includes reference values and respects the clamp', () => {
    expect(autoRange([5, 6], 0.1, [0])).toEqual({ lo: 0, hi: 6 })
    const r = autoRange([99.5, 99.9], 10, [], { lo: 0, hi: 100 })!
    expect(r.hi).toBe(100)
    expect(r.hi - r.lo).toBeCloseTo(10, 9)
  })
  it('returns null without data', () => expect(autoRange([null, null], 1)).toBeNull())
  it('maps samples into the box and breaks on gaps / nulls', () => {
    const ts = [0, 1, 2, 3, 10, 11]
    const vs = [0, 1, null, 1, 0, 1]
    const segs = sparkSegments(ts, vs, 11, 20, 200, 40, { lo: 0, hi: 1 })
    expect(segs).toEqual(['90.0,40.0 100.0,0.0', '190.0,40.0 200.0,0.0'])
  })
  it('drops samples older than the window', () => {
    expect(sparkSegments([0, 5, 29, 30], [1, 1, 1, 1], 30, 20, 100, 10, { lo: 0, hi: 2 })).toEqual(['95.0,5.0 100.0,5.0'])
  })
  it('places reference lines', () => expect(sparkY(1, 40, { lo: 0, hi: 2 })).toBe(20))
})

describe('pushWindowed', () => {
  it('keeps a rolling window', () => {
    const buf: { t: number }[] = []
    for (let k = 0; k <= 40; k++) pushWindowed(buf, { t: k * 0.5 }, 5)
    expect(buf[0].t).toBe(15)
    expect(buf[buf.length - 1].t).toBe(20)
  })
  it('restarts when the clock jumps backwards', () => {
    const buf = [{ t: 100 }, { t: 101 }]
    pushWindowed(buf, { t: 3 }, 20)
    expect(buf).toEqual([{ t: 3 }])
  })
})
