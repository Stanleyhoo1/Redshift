// Dust storm, battery conservation and drone wind (docs/protocol.md §10). The world's env (`mission.env`) is the
// operator's view of the weather; power_save / wind_* are what each robot inferred from its own sensors.
import type { CompactDetection, MissionEnv, RobotTelemetry } from './types'

/** sensor_effects STORM_RGB: the storm's orange-red haze (the globe's dust shell tints towards it) */
export const STORM_RGB = [205, 88, 48] as const
/** fleet chips show the drone's wind from this estimate up, so calm-air estimator noise stays off the top bar */
export const WIND_SHOW_MPS = 4.0
/** wind.WindDescent.ENTER_MPS: the wind reading turns amber where the drone's own descent rule starts counting */
export const WIND_ENTER_MPS = 8.0
/** below this dust level the air counts as clear (the tint lets go) */
const CLEAR_DUST = 0.02

type Env = Partial<MissionEnv> | null | undefined

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

export const isDustStorm = (env: Env): boolean => env?.kind === 'dust_storm'

/**
 * Target of the globe dust shell's storm tint (0 = brown dust haze, 1 = orange-red storm). A backend that reports no
 * kind while dust still lingers keeps the previous tint, so a storm does not turn brown as it fades out.
 */
export function stormTintTarget(env: Env, prev: number): number {
  if (env?.kind === 'dust_storm') return 1
  if (env?.kind) return 0
  return (num(env?.dust_level) ?? 0) > CLEAR_DUST ? prev : 0
}

/** One line for the mission overlay ("Dust storm, opacity 1.00, wind 14 m/s"); null when the air is clear. */
export function envLine(env: Env): { text: string; storm: boolean } | null {
  const dust = num(env?.dust_level) ?? 0
  const seismic = !!env?.seismic
  if (dust <= 0 && !seismic) return null
  const storm = isDustStorm(env) && dust > 0
  const parts: string[] = []
  if (dust > 0) {
    const wind = num(env?.wind_mps)
    parts.push(`${storm ? 'Dust storm' : 'Dust haze'}, opacity ${dust.toFixed(2)}`)
    if (storm && wind !== null && wind > 0) parts.push(`wind ${wind.toFixed(0)} m/s`)
  }
  if (seismic) parts.push(parts.length ? 'seismic tremor' : 'Seismic tremor')
  return { text: parts.join(', '), storm }
}

/** A rover's battery-conservation badge: null unless its agent_state reports power save. */
export function powerSave(r: RobotTelemetry | undefined | null): { solar_w: number | null } | null {
  if (!r || r.type !== 'rover' || r.power_save !== true) return null
  return { solar_w: num(r.health?.solar_w) }
}

/** The drone's own wind estimate and descent; null for rovers, and for a drone that reports neither field. */
export function windReading(r: RobotTelemetry | undefined | null): { mps: number | null; descent: boolean; show: boolean } | null {
  if (!r || r.type !== 'scout' || (r.wind_mps === undefined && r.wind_descent === undefined)) return null
  const mps = num(r.wind_mps)
  const descent = r.wind_descent === true
  return { mps, descent, show: descent || (mps !== null && mps >= WIND_SHOW_MPS) }
}

/** §10.2: how the annotated feed names a class ("FALLEN PILLAR"); other classes read as their code. */
export function detectionLabel(cls: string): string {
  return cls === 'PILLAR' ? 'FALLEN PILLAR' : cls
}

/** The nearest PILLAR the robot's own perception reports right now, or null. */
export function pillarSighting(r: RobotTelemetry | undefined | null): CompactDetection | null {
  let best: CompactDetection | null = null
  for (const d of r?.detections ?? []) {
    if (d?.cls !== 'PILLAR') continue
    const rng = num(d.range_m) ?? Infinity
    if (!best || rng < (num(best.range_m) ?? Infinity)) best = d
  }
  return best
}
