// Human-readable formatting for telemetry values.

// docs/protocol.md §8.1: fleet roles, display names and short labels.
export const ROBOT_NAMES: Record<string, { name: string; short: string; role: string }> = {
  rover_0: { name: 'Scout Rover 1', short: 'ROVER_1', role: 'scout' },
  rover_1: { name: 'Heavy Recovery Rover 2', short: 'ROVER_2', role: 'recovery' },
  scout_0: { name: 'Survey Drone', short: 'DRONE', role: 'drone' },
}

const parseId = (id: string) => /^([a-z]+)_(\d+)$/.exec(id)

/**
 * Human display name. Prefers the name the backend sends in telemetry (`robots.<r>.name`), then the
 * §8.1 table, then a generated 1-based name ("rover_3" → "Rover 4").
 */
export function robotName(id: string, telemetryName?: string | null): string {
  if (telemetryName) return telemetryName
  const known = ROBOT_NAMES[id]
  if (known) return known.name
  const m = parseId(id)
  if (!m) return id
  const kind = m[1] === 'scout' ? 'Drone' : `${m[1][0].toUpperCase()}${m[1].slice(1)}`
  return `${kind} ${Number(m[2]) + 1}`
}

/** Callsign-style short label (§8.1): ROVER_1, ROVER_2, DRONE. */
export function robotShort(id: string): string {
  const known = ROBOT_NAMES[id]
  if (known) return known.short
  const m = parseId(id)
  if (!m) return id.toUpperCase()
  return `${m[1] === 'scout' ? 'DRONE' : m[1].toUpperCase()}_${Number(m[2]) + 1}`
}

/** Role from telemetry, else from the id (rover_0 = scout, other rovers = recovery, scouts = drone). */
export function robotRole(id: string, telemetryRole?: string | null): string {
  if (telemetryRole) return telemetryRole
  if (ROBOT_NAMES[id]) return ROBOT_NAMES[id].role
  if (id.startsWith('scout')) return 'drone'
  return id === 'rover_0' ? 'scout' : 'recovery'
}

const WHEEL: Record<string, string> = { FL: 'front-left', FR: 'front-right', RL: 'rear-left', RR: 'rear-right' }

/** "DRIVE_FL_OVERCURRENT" → "Front-left motor overcurrent"; unknown codes are de-underscored. */
export function faultLabel(code: string): string {
  const m = /^DRIVE_(FL|FR|RL|RR)_(.+)$/.exec(code)
  if (m) {
    const what = m[2].toLowerCase().replace(/_/g, ' ')
    const wheel = WHEEL[m[1]]
    return `${wheel[0].toUpperCase()}${wheel.slice(1)} ${what === 'overcurrent' ? 'motor overcurrent' : what}`
  }
  const s = code.toLowerCase().replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

/** One phrase for a fault list: "drivetrain fault" when any drive actuator reports one. */
export function faultSummary(faults: readonly string[] | null | undefined): string | null {
  if (!faults || faults.length === 0) return null
  if (faults.some((f) => f.startsWith('DRIVE'))) return 'drivetrain fault'
  return faultLabel(faults[0]).toLowerCase()
}

/** Hours → "7.0 h", "42 min", "> 99 h". */
export function fmtHours(h: number | null | undefined): string {
  if (h === null || h === undefined || !Number.isFinite(h) || h < 0) return '—'
  if (h >= 99) return '> 99 h'
  if (h < 1) return `${Math.max(1, Math.round(h * 60))} min`
  return `${h.toFixed(1)} h`
}

/** Signed value: +51.4 / −12.0 (true minus sign). */
export function fmtSigned(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—'
  const v = Math.abs(n).toFixed(digits)
  if (Number(v) === 0) return v
  return `${n < 0 ? '−' : '+'}${v}`
}

export function stateLabel(state: string | undefined): string {
  return (state ?? '—').replace(/_/g, ' ')
}

export function kindLabel(kind: string): string {
  const s = kind.replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function fmtClock(tSec: number): string {
  const d = new Date(tSec * 1000)
  if (!Number.isFinite(d.getTime()) || tSec < 1e8) return fmtElapsed(tSec) // sim-time fallback
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

export function fmtElapsed(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

export function fmt(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—'
  return n.toFixed(digits)
}

export const DEG = 180 / Math.PI

export const FACTOR_LABEL: Record<string, string> = {
  slip: 'Soil traction (slip)',
  drive_torque: 'Wheel torque',
  tip: 'Tip-over',
}

export const CHECKLIST_LABEL: Record<string, string> = {
  gazebo: 'Gazebo world',
  mock_world: 'Mock world',
  world: 'World simulation',
  bridge: 'ROS ↔ Gazebo bridge',
  sensor_effects: 'Sensor effects layer',
  perception: 'Perception',
  mapping: 'Mapping coordinator',
  planner: 'Agent planners',
  backend: 'Backend link',
}

export function checklistLabel(key: string): string {
  if (CHECKLIST_LABEL[key]) return CHECKLIST_LABEL[key]
  if (/^(rover|scout)_\d+$/.test(key)) return `${robotName(key)} online`
  return kindLabel(key)
}

/** 0.5 → "0.5×", 1 → "1×" */
export function fmtSpeed(x: number): string {
  return `${Number(x.toFixed(2))}×`
}
