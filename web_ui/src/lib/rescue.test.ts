import { beforeEach, describe, expect, it } from 'vitest'
import { clearDistressClock, deriveRescue, distressElapsed, phaseIndex, RESOLVED_SHOW_MS } from './rescue'
import { useStore } from './store'
import type { RescueEvent, RobotHealth, RobotTelemetry, SosEvent } from './types'

const health = (pct: number, faults: string[] = []): RobotHealth => ({
  pct,
  structure_pct: 74,
  drive_pct: 10,
  battery_pct: 80,
  battery_wh: 480,
  capacity_wh: 600,
  solar_w: 100,
  load_w: 60,
  net_w: 40,
  faults,
  critical: pct < 15,
  shocks: 3,
  peak_g: 5.1,
})
const robot = (patch: Partial<RobotTelemetry>): RobotTelemetry => ({ type: 'rover', state: 'EXPLORING', ...patch }) as RobotTelemetry

const sosEv = (status: 'active' | 'resolved'): SosEvent => ({ op: 'sos', id: 'sos-rover_0-1', robot: 'rover_0', xy: [1, 2], health_pct: 7.5, faults: ['DRIVE_FL_OVERCURRENT'], status, t: 0 })
const rescueEv = (phase: string, locked = false): RescueEvent => ({ op: 'rescue', id: 'sos-rover_0-1', rescuer: 'rover_1', target: 'rover_0', phase, range_m: 9.5, locked, t: 0 })

describe('deriveRescue', () => {
  it('is null when nobody is in trouble', () => {
    expect(deriveRescue({ rover_0: robot({}), rover_1: robot({}) }, {}, {}, 1000)).toBeNull()
  })

  it('shows a bare SOS for a rover awaiting rescue, with health from telemetry', () => {
    const v = deriveRescue({ rover_0: robot({ state: 'AWAIT_RESCUE', health: health(7.4, ['DRIVE_FL_OVERCURRENT']) }) }, {}, {}, 1000)!
    expect(v.target).toBe('rover_0')
    expect(v.rescuer).toBeNull()
    expect(v.phase).toBe('AWAITING')
    expect(v.health_pct).toBe(7.4)
    expect(v.faults).toEqual(['DRIVE_FL_OVERCURRENT'])
  })

  it('takes phase, range and lock from the rescuer telemetry block', () => {
    const v = deriveRescue(
      {
        rover_0: robot({ state: 'AWAIT_RESCUE', health: health(7.4), rescue: { id: 'sos-rover_0-1', role: 'target', peer: 'rover_1', phase: 'TRANSIT', range_m: 20, locked: false } }),
        rover_1: robot({ state: 'RESCUE_DOCKING', rescue: { id: 'sos-rover_0-1', role: 'rescuer', peer: 'rover_0', phase: 'DOCKING', range_m: 2.4, locked: true } }),
      },
      { a: { ev: rescueEv('TRANSIT'), receivedAt: 0 } } as never,
      {},
      1000,
    )!
    expect(v).toMatchObject({ target: 'rover_0', rescuer: 'rover_1', phase: 'DOCKING', range_m: 2.4, locked: true, resolved: false })
  })

  it('falls back to fresh events when telemetry has no §8 blocks', () => {
    const v = deriveRescue({ rover_0: robot({}), rover_1: robot({}) }, { s: { ev: sosEv('active'), receivedAt: 900 } }, { r: { ev: rescueEv('TRANSIT'), receivedAt: 950 } }, 1000)!
    expect(v).toMatchObject({ target: 'rover_0', rescuer: 'rover_1', phase: 'TRANSIT', range_m: 9.5, health_pct: 7.5 })
  })

  it('ignores stale events', () => {
    expect(deriveRescue({}, { s: { ev: sosEv('active'), receivedAt: 0 } }, {}, 60000)).toBeNull()
  })

  it('shows a resolved wrap-up for a while after DONE, then clears', () => {
    const rescues = { r: { ev: rescueEv('DONE'), receivedAt: 1000 } }
    const v = deriveRescue({ rover_0: robot({ state: 'RESUMING', health: health(74) }) }, {}, rescues, 2000)!
    expect(v).toMatchObject({ resolved: true, phase: 'DONE', health_pct: 74 })
    expect(deriveRescue({}, {}, rescues, 1000 + RESOLVED_SHOW_MS + 1)).toBeNull()
  })

  it('treats a DONE telemetry block as resolved, not active', () => {
    const v = deriveRescue({ rover_1: robot({ rescue: { id: 'x', role: 'rescuer', peer: 'rover_0', phase: 'DONE', range_m: 1.3, locked: true } }) }, {}, {}, 0)!
    expect(v.resolved).toBe(true)
    expect(v.locked).toBe(false)
  })

  it('ends the rescue when the SOS resolves, while the rescuer still backs off at REPAIRING', () => {
    // repair done: the target left AWAIT_RESCUE (block null) and says resolved; the rescuer's block and 1 Hz
    // heartbeat still read REPAIRING during its 2 m back-off (§8.6)
    const robots = {
      rover_0: robot({ state: 'RESUMING', health: health(74), rescue: null }),
      rover_1: robot({ state: 'REPAIRING', rescue: { id: 'sos-rover_0-1', role: 'rescuer', peer: 'rover_0', phase: 'REPAIRING', range_m: 1.4, locked: true } }),
    }
    const sos = { 'sos-rover_0-1': { ev: sosEv('resolved'), receivedAt: 1000 } }
    const rescues = { 'sos-rover_0-1': { ev: rescueEv('REPAIRING', true), receivedAt: 1500 } }
    const v = deriveRescue(robots, sos, rescues, 2000)!
    expect(v).toMatchObject({ id: 'sos-rover_0-1', target: 'rover_0', rescuer: 'rover_1', phase: 'DONE', resolved: true, aborted: false, locked: false, health_pct: 74 })
    // once the wrap-up has run its course a lingering REPAIRING block does not bring the SOS back
    expect(deriveRescue(robots, sos, {}, 1000 + RESOLVED_SHOW_MS + 1)).toBeNull()
  })

  it('keeps the real SOS id from the distress call through the rescuer assignment', () => {
    // the damaged rover's own block carries the SOS id before any rescuer is assigned (peer null)
    const awaiting = { id: 'sos-rover_0-1', role: 'target', peer: null, phase: 'AWAITING', range_m: null, locked: false }
    const before = deriveRescue({ rover_0: robot({ state: 'AWAIT_RESCUE', health: health(7.4), rescue: awaiting }), rover_1: robot({}) }, {}, {}, 1000)!
    expect(before).toMatchObject({ id: 'sos-rover_0-1', target: 'rover_0', rescuer: null, phase: 'AWAITING', resolved: false })
    const after = deriveRescue(
      {
        rover_0: robot({ state: 'AWAIT_RESCUE', health: health(7.4), rescue: { ...awaiting, peer: 'rover_1', phase: 'TRANSIT' } }),
        rover_1: robot({ state: 'RESCUE_TRANSIT', rescue: { id: 'sos-rover_0-1', role: 'rescuer', peer: 'rover_0', phase: 'TRANSIT', range_m: 20, locked: false } }),
      },
      {},
      {},
      2000,
    )!
    expect(after).toMatchObject({ id: 'sos-rover_0-1', rescuer: 'rover_1', phase: 'TRANSIT' })
  })

  it('lets a fresh SOS event replace the id of a rover seen only in AWAIT_RESCUE', () => {
    const robots = { rover_0: robot({ state: 'AWAIT_RESCUE', health: health(7.4) }) }
    expect(deriveRescue(robots, {}, {}, 1000)!.id).toBe('sos-rover_0') // nothing names the SOS yet
    expect(deriveRescue(robots, { s: { ev: sosEv('active'), receivedAt: 900 } }, {}, 1000)!.id).toBe('sos-rover_0-1')
  })

  it('orders phases along the track', () => {
    expect(phaseIndex('TRANSIT')).toBeLessThan(phaseIndex('DOCKING'))
    expect(phaseIndex('REPAIRING')).toBeLessThan(phaseIndex('DONE'))
    expect(phaseIndex('AWAITING')).toBe(-1)
  })
})

describe('distressElapsed', () => {
  beforeEach(() => clearDistressClock())

  it('counts from the first time an SOS id is shown', () => {
    expect(distressElapsed('sos-rover_0-1', 1000)).toBe(0)
    expect(distressElapsed('sos-rover_0-1', 4500)).toBe(3.5)
  })

  it('restarts every mission, where SOS ids repeat (sos-rover_0-1 …)', () => {
    distressElapsed('sos-rover_0-1', 1000)
    useStore.getState().resetMission()
    expect(distressElapsed('sos-rover_0-1', 90_000)).toBe(0)
  })
})
