import { describe, expect, it } from 'vitest'
import { checklistLabel, faultLabel, faultSummary, fmtHours, fmtSigned, robotName, robotRole, robotShort } from './format'

describe('robot names (§8.1)', () => {
  it('maps the fleet table', () => {
    expect(robotName('rover_0')).toBe('Scout Rover 1')
    expect(robotName('rover_1')).toBe('Heavy Recovery Rover 2')
    expect(robotName('scout_0')).toBe('Survey Drone')
    expect(robotShort('rover_0')).toBe('ROVER_1')
    expect(robotShort('rover_1')).toBe('ROVER_2')
    expect(robotShort('scout_0')).toBe('DRONE')
  })
  it('prefers the telemetry name', () => expect(robotName('rover_0', 'Pathfinder')).toBe('Pathfinder'))
  it('falls back to 1-based names for unknown ids', () => {
    expect(robotName('rover_3')).toBe('Rover 4')
    expect(robotShort('rover_3')).toBe('ROVER_4')
    expect(robotName('scout_1')).toBe('Drone 2')
    expect(robotShort('scout_1')).toBe('DRONE_2')
    expect(robotName('weird')).toBe('weird')
  })
  it('derives roles from the id when telemetry has none', () => {
    expect(robotRole('rover_0')).toBe('scout')
    expect(robotRole('rover_5')).toBe('recovery')
    expect(robotRole('scout_0')).toBe('drone')
    expect(robotRole('rover_5', 'scout')).toBe('scout')
  })
  it('labels robot checklist items by display name', () => expect(checklistLabel('rover_1')).toBe('Heavy Recovery Rover 2 online'))
})

describe('fault codes', () => {
  it('reads drive actuator faults', () => {
    expect(faultLabel('DRIVE_FL_OVERCURRENT')).toBe('Front-left motor overcurrent')
    expect(faultLabel('DRIVE_RL_ENCODER_LOST')).toBe('Rear-left encoder lost')
    expect(faultLabel('BATTERY_LOW')).toBe('Battery low')
  })
  it('summarises a fault list', () => {
    expect(faultSummary(['DRIVE_FL_OVERCURRENT', 'DRIVE_RL_ENCODER_LOST'])).toBe('drivetrain fault')
    expect(faultSummary(['IMU_SATURATED'])).toBe('imu saturated')
    expect(faultSummary([])).toBeNull()
    expect(faultSummary(undefined)).toBeNull()
  })
})

describe('number formatting', () => {
  it('formats durations in hours', () => {
    expect(fmtHours(7.04)).toBe('7.0 h')
    expect(fmtHours(0.5)).toBe('30 min')
    expect(fmtHours(0.001)).toBe('1 min')
    expect(fmtHours(150)).toBe('> 99 h')
    expect(fmtHours(null)).toBe('—')
  })
  it('signs values with a true minus', () => {
    expect(fmtSigned(51.44)).toBe('+51.4')
    expect(fmtSigned(-12)).toBe('−12.0')
    expect(fmtSigned(0.01)).toBe('0.0')
    expect(fmtSigned(undefined)).toBe('—')
  })
})
