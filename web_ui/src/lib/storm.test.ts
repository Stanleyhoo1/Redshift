import { describe, expect, it } from 'vitest'
import { detectionLabel, envLine, isDustStorm, pillarSighting, powerSave, stormTintTarget, windReading } from './storm'
import type { RobotTelemetry } from './types'

const rover = (patch: Partial<RobotTelemetry> = {}): RobotTelemetry =>
  ({
    type: 'rover',
    health: { pct: 92, structure_pct: 96, drive_pct: 96, battery_pct: 78, battery_wh: 468, capacity_wh: 600, solar_w: 31.2, load_w: 120, net_w: -88.8, faults: [], critical: false, shocks: 0, peak_g: 0 },
    ...patch,
  }) as RobotTelemetry
const drone = (patch: Partial<RobotTelemetry> = {}): RobotTelemetry => ({ type: 'scout', altitude_agl: 12, ...patch }) as RobotTelemetry

describe('dust storm env', () => {
  it('recognises the storm kind only', () => {
    expect(isDustStorm({ dust_level: 1, seismic: false, kind: 'dust_storm' })).toBe(true)
    expect(isDustStorm({ dust_level: 0.85, seismic: true, kind: 'dust' })).toBe(false)
    expect(isDustStorm({ dust_level: 0.85, seismic: true })).toBe(false)
    expect(isDustStorm(null)).toBe(false)
  })
  it('tints orange-red in a storm, brown in plain dust', () => {
    expect(stormTintTarget({ dust_level: 1, seismic: false, kind: 'dust_storm' }, 0)).toBe(1)
    expect(stormTintTarget({ dust_level: 0.85, seismic: false, kind: 'dust' }, 1)).toBe(0)
  })
  it('keeps the tint while lingering dust reports no kind, and clears it once the air is clear', () => {
    expect(stormTintTarget({ dust_level: 0.4, seismic: false, kind: null }, 1)).toBe(1)
    expect(stormTintTarget({ dust_level: 0.4, seismic: false }, 0)).toBe(0)
    expect(stormTintTarget({ dust_level: 0, seismic: false, kind: null }, 1)).toBe(0)
    expect(stormTintTarget(undefined, 1)).toBe(0)
  })
  it('summarises the environment for the mission overlay', () => {
    expect(envLine({ dust_level: 0, seismic: false })).toBeNull()
    expect(envLine({ dust_level: 0.85, seismic: true, kind: 'dust' })).toEqual({ text: 'Dust haze, opacity 0.85, seismic tremor', storm: false })
    expect(envLine({ dust_level: 1, seismic: false, kind: 'dust_storm', wind_mps: 13.6 })).toEqual({ text: 'Dust storm, opacity 1.00, wind 14 m/s', storm: true })
    expect(envLine({ dust_level: 0, seismic: true })).toEqual({ text: 'Seismic tremor', storm: false })
  })
})

describe('power save (rover battery conservation)', () => {
  it('is null unless the rover reports it', () => {
    expect(powerSave(rover())).toBeNull()
    expect(powerSave(rover({ power_save: false }))).toBeNull()
    expect(powerSave(drone({ power_save: true }))).toBeNull()
  })
  it('carries the measured solar power', () => {
    expect(powerSave(rover({ power_save: true }))).toEqual({ solar_w: 31.2 })
    expect(powerSave(rover({ power_save: true, health: null }))).toEqual({ solar_w: null })
  })
})

describe('drone wind', () => {
  it('is null for rovers and for drones that report nothing', () => {
    expect(windReading(rover({ wind_mps: 3 }))).toBeNull()
    expect(windReading(drone())).toBeNull()
  })
  it('reports the estimate, the descent and when the fleet chip shows it', () => {
    expect(windReading(drone({ wind_mps: 13.2, wind_descent: true }))).toEqual({ mps: 13.2, descent: true, show: true })
    expect(windReading(drone({ wind_mps: 1.1, wind_descent: false }))).toEqual({ mps: 1.1, descent: false, show: false })
    expect(windReading(drone({ wind_mps: null, wind_descent: true }))).toEqual({ mps: null, descent: true, show: true })
    expect(windReading(drone({ wind_mps: Number.NaN }))).toEqual({ mps: null, descent: false, show: false })
  })
})

describe('pillar detections', () => {
  it('labels PILLAR the way the annotated feed draws it', () => {
    expect(detectionLabel('PILLAR')).toBe('FALLEN PILLAR')
    expect(detectionLabel('ROCK')).toBe('ROCK')
  })
  it('picks the nearest pillar the robot itself reports', () => {
    const r = rover({
      detections: [
        { cls: 'ROCK', conf: 0.9, range_m: 3, world_xy: [1, 1] },
        { cls: 'PILLAR', conf: 0.7, range_m: 9.5, world_xy: [5, 5] },
        { cls: 'PILLAR', conf: 0.8, range_m: 7.4, world_xy: [4, 4] },
      ],
    })
    expect(pillarSighting(r)?.range_m).toBe(7.4)
    expect(pillarSighting(rover({ detections: [{ cls: 'ROCK', conf: 0.9, range_m: 3, world_xy: null }] }))).toBeNull()
    expect(pillarSighting(rover({ detections: null }))).toBeNull()
    expect(pillarSighting(rover())).toBeNull()
  })
})
