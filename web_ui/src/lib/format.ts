// Human-readable formatting for telemetry values.

export function robotName(id: string): string {
  const m = /^([a-z]+)_(\d+)$/.exec(id)
  if (!m) return id
  return `${m[1][0].toUpperCase()}${m[1].slice(1)} ${m[2]}`
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
