import { describe, expect, it } from 'vitest'
import { G_EARTH, parsePillarEvent, pillarFallAngle, pillarLandS, pillarRestAngle } from './pillar'

const ev = (patch: Record<string, unknown> = {}) => ({
  op: 'pillar_collapse',
  ok: true,
  robot: 'rover_0',
  base: [10, 2],
  tip: [10, 8],
  cross: [10, 4.8],
  height_m: 6.0,
  radius_m: 0.35,
  waited_s: 0,
  t: 1762550000,
  ...patch,
})

describe('pillar fall physics (mirrors scene_events.pillar_fall_angle)', () => {
  const opts = { height_m: 6, radius_m: 0.35, g: G_EARTH * 2.2, tilt0_deg: 4 }
  it('starts at the initial tilt', () => expect(pillarFallAngle(0, opts)).toBeCloseTo((4 * Math.PI) / 180, 9))
  it('is monotonic and clamps at the resting angle', () => {
    let prev = -1
    for (let t = 0; t <= 4; t += 0.05) {
      const th = pillarFallAngle(t, opts)
      expect(th).toBeGreaterThanOrEqual(prev)
      prev = th
    }
    expect(prev).toBeCloseTo(pillarRestAngle(6, 0.35), 9)
    expect(pillarRestAngle(6, 0.35)).toBeLessThan(Math.PI / 2)
    expect(pillarRestAngle(6, 0.35)).toBeGreaterThan(1.45)
  })
  it('lands in about 1.2–2.0 s at 2.2 g', () => {
    const land = pillarLandS(opts)
    expect(land).toBeGreaterThan(1.2)
    expect(land).toBeLessThan(2.0)
    expect(pillarFallAngle(land - 0.1, opts)).toBeLessThan(pillarRestAngle(6, 0.35))
  })
  it('falls faster under stronger gravity', () => expect(pillarLandS({ ...opts, g: G_EARTH * 3 })).toBeLessThan(pillarLandS(opts)))
  it('is deterministic', () => expect(pillarFallAngle(0.9, opts)).toBe(pillarFallAngle(0.9, opts)))
  it('treats negative time as the release instant', () => expect(pillarFallAngle(-1, opts)).toBe(pillarFallAngle(0, opts)))
})

describe('parsePillarEvent', () => {
  it('derives the fall direction and keeps the footprint', () => {
    const p = parsePillarEvent(ev())!
    expect(p.robot).toBe('rover_0')
    expect(p.base).toEqual([10, 2])
    expect(p.tip).toEqual([10, 8])
    expect(p.dir[0]).toBeCloseTo(0, 9)
    expect(p.dir[1]).toBeCloseTo(1, 9)
    expect(p.height_m).toBe(6)
    expect(p.cross).toEqual([10, 4.8])
  })
  it('falls back to the footprint length and the default radius', () => {
    const p = parsePillarEvent(ev({ height_m: undefined, radius_m: undefined, cross: undefined }))!
    expect(p.height_m).toBeCloseTo(6, 9)
    expect(p.radius_m).toBe(0.35)
    expect(p.cross).toBeNull()
  })
  it('rejects failures, other ops and malformed footprints', () => {
    expect(parsePillarEvent(ev({ ok: false, detail: 'no rover' }))).toBeNull()
    expect(parsePillarEvent(ev({ op: 'spawn_rockslide' }))).toBeNull()
    expect(parsePillarEvent(ev({ base: [1] }))).toBeNull()
    expect(parsePillarEvent(ev({ tip: [10, Number.NaN] }))).toBeNull()
    expect(parsePillarEvent(ev({ tip: [10, 2.01] }))).toBeNull() // zero-length footprint has no direction
  })
})
