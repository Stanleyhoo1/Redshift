// Pillar collapse (docs/protocol.md §10.1): the falling-rod physics behind the globe's fall effect. It mirrors
// redshift_sim.scene_events.pillar_fall_angle, so the globe topples the pillar in step with the world.
import type { WorldEvent, XY } from './types'

/** g used by scene_events (m/s² per mission g) */
export const G_EARTH = 9.81
/** scene_events.PILLAR defaults, used when the event leaves a field out */
export const PILLAR_DEFAULT = { height_m: 6.0, radius_m: 0.35, tilt0_deg: 4.0 } as const
const STEP_S = 0.001 // fixed integration step (s)
const MAX_FALL_S = 10 // a fall that never lands (degenerate inputs) stops here
const MIN_FOOTPRINT_M = 0.5 // a shorter base → tip footprint has no usable fall direction

export interface FallOpts {
  height_m?: number
  radius_m?: number
  /** m/s² (mission gravity_g × G_EARTH) */
  g?: number
  tilt0_deg?: number
}

/** Resting tilt from vertical (rad) once the trunk's side lies on flat ground. */
export function pillarRestAngle(height_m: number, radius_m: number): number {
  return Math.PI / 2 - Math.asin(Math.min(1, Math.max(0, radius_m / height_m)))
}

function fallParams(o: FallOpts) {
  const L = o.height_m ?? PILLAR_DEFAULT.height_m
  return {
    k: (3 * (o.g ?? G_EARTH * 2.2)) / (2 * L), // θ'' = (3 g / 2 L) sin θ
    th0: ((o.tilt0_deg ?? PILLAR_DEFAULT.tilt0_deg) * Math.PI) / 180,
    rest: pillarRestAngle(L, o.radius_m ?? PILLAR_DEFAULT.radius_m),
  }
}

/**
 * Tilt from vertical (rad) t_s seconds after release: a rigid rod pivoting on its base from the initial tilt,
 * integrated with a fixed step (semi-implicit Euler) and clamped at the resting angle. Monotonic, deterministic.
 */
export function pillarFallAngle(t_s: number, o: FallOpts = {}): number {
  const { k, th0, rest } = fallParams(o)
  const n = Math.round(Math.min(MAX_FALL_S, Math.max(0, t_s)) / STEP_S)
  let th = th0
  let w = 0
  for (let i = 0; i < n && th < rest; i++) {
    w += k * Math.sin(th) * STEP_S
    th += w * STEP_S
  }
  return Math.min(th, rest)
}

/** Seconds from release until the trunk lies on the ground. */
export function pillarLandS(o: FallOpts = {}): number {
  const { k, th0, rest } = fallParams(o)
  let th = th0
  let w = 0
  let i = 0
  while (th < rest && i * STEP_S < MAX_FALL_S) {
    w += k * Math.sin(th) * STEP_S
    th += w * STEP_S
    i++
  }
  return i * STEP_S
}

/** A validated pillar_collapse event, with the fall direction derived from its footprint. */
export interface PillarFall {
  robot: string
  base: XY
  tip: XY
  /** unit vector base → tip (world x east, y north) */
  dir: XY
  height_m: number
  radius_m: number
  cross: XY | null
}

const isXY = (v: unknown): v is XY => Array.isArray(v) && v.length >= 2 && Number.isFinite(v[0]) && Number.isFinite(v[1])
const posNum = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null)

/** null unless `ev` is a successful pillar_collapse with a usable base → tip footprint. */
export function parsePillarEvent(ev: WorldEvent | Record<string, unknown>): PillarFall | null {
  const e = ev as Record<string, unknown>
  if (e.op !== 'pillar_collapse' || e.ok === false || !isXY(e.base) || !isXY(e.tip)) return null
  const base: XY = [e.base[0], e.base[1]]
  const tip: XY = [e.tip[0], e.tip[1]]
  const len = Math.hypot(tip[0] - base[0], tip[1] - base[1])
  if (len < MIN_FOOTPRINT_M) return null
  return {
    robot: typeof e.robot === 'string' ? e.robot : '',
    base,
    tip,
    dir: [(tip[0] - base[0]) / len, (tip[1] - base[1]) / len],
    height_m: posNum(e.height_m) ?? len,
    radius_m: posNum(e.radius_m) ?? PILLAR_DEFAULT.radius_m,
    cross: isXY(e.cross) ? [e.cross[0], e.cross[1]] : null,
  }
}
