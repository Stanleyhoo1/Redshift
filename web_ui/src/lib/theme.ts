// Colour tokens shared by canvas / three.js code. CSS uses the same values via @theme in index.css.
// Semantic colours sit on real spectral emission lines: Hβ 486 nm (info), O III 501 nm (nominal),
// Na D 589 nm (caution), Hα 656 nm (critical). The planet itself lives at the red end.
// Rescue sits at 555 nm, the peak of human photopic sensitivity: the hi-vis yellow-green of rescue gear.
import type { Tone } from './sensors'
import type { SensorTag, Severity } from './types'

export const C = {
  void: '#16132b',
  bay: '#1e1a38',
  bay2: '#26214a',
  rule: '#362f63',
  rule2: '#4a4282',
  bone: '#ece7f5',
  haze: '#a39cc8',
  dim: '#6e6797',
  hb: '#4cc3ff',
  o3: '#3df5a8',
  nad: '#ffb238',
  ha: '#ff4a5c',
  sci: '#5cf2e6',
  rescue: '#c6ff3d',
  rust: '#b5482a',
  ochre: '#d39a4a',
  basalt: '#2a2326',
} as const

export const SEVERITY_COLOR: Record<Severity, string> = {
  info: C.hb,
  warn: C.nad,
  critical: C.ha,
}

export const SENSOR_COLOR: Record<SensorTag, string> = {
  CAM: '#8ec5ff',
  DEPTH: '#b3a4ff',
  LIDAR: '#5eebc0',
  IMU: '#f59bd8',
  ODOM: '#cfc8ea',
  MAP: '#e3c48c',
  FLEET: '#9be564',
  EDGE: '#d6a6ff',
  DIAG: '#ff9f7a',
}

/** Panel-only channel groups that are not trace sensor tags. */
const GROUP_COLOR: Record<string, string> = {
  PWR: '#f2dc6b',
  ALT: '#c9b8ff',
}

export function sensorColor(tag: string): string {
  return (SENSOR_COLOR as Record<string, string>)[tag] ?? GROUP_COLOR[tag] ?? C.haze
}

export const TONE_COLOR: Record<Tone, string> = { ok: C.o3, warn: C.nad, crit: C.ha }

/** Rescue-sequence trace kinds (§8.6 + rescue_dock) get the rescue colour in the trace. */
export const RESCUE_KINDS = new Set(['rescue_start', 'rescue_dock', 'rescue_lock', 'docked', 'repair_start', 'repair_complete', 'repaired'])

/** State → colour family. */
export function stateColor(state: string | undefined): string {
  switch (state) {
    case 'EXPLORING':
    case 'SURVEY':
    case 'TAKEOFF':
    case 'RESUMING':
      return C.o3
    case 'REPLANNING':
      return C.hb
    case 'CAUTION':
      return C.nad
    case 'SAFE_HOLD':
    case 'RECOVERY_REVERSE':
    case 'AWAIT_RESCUE':
      return C.ha
    case 'RESCUE_TRANSIT':
    case 'RESCUE_DOCKING':
    case 'REPAIRING':
      return C.rescue
    case 'AWAIT_SCOUT':
    case 'FLY_TO_SCAN':
    case 'SCANNING':
      return C.sci
    case 'MISSION_COMPLETE':
      return C.bone
    default:
      return C.dim
  }
}

/** States in which a rover's marker pulses as an alarm on the globe. */
export const ALARM_STATES = new Set(['SAFE_HOLD', 'RECOVERY_REVERSE', 'AWAIT_RESCUE'])

/** Robot marker colour on the globe and in chips. */
export function robotColor(id: string): string {
  if (id.startsWith('scout')) return '#c9b8ff'
  if (id === 'rover_1') return '#ffd08a'
  return '#ffe9c7'
}
