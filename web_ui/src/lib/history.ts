// Non-reactive rolling buffers fed by the store's telemetry ingest. Components read them during
// renders they already do (telemetry re-renders at 15 Hz), so no extra state churn.
import type { XY } from './types'

export interface ImuSample {
  t: number
  roll: number
  pitch: number
}

const IMU_WINDOW_S = 20
const imu = new Map<string, ImuSample[]>()

export function pushImu(robot: string, s: ImuSample) {
  let buf = imu.get(robot)
  if (!buf) imu.set(robot, (buf = []))
  buf.push(s)
  const cutoff = s.t - IMU_WINDOW_S
  let drop = 0
  while (drop < buf.length && buf[drop].t < cutoff) drop++
  if (drop) buf.splice(0, drop)
}

export const imuHistory = (robot: string): readonly ImuSample[] => imu.get(robot) ?? []

// ---- camera frame-rate estimate ------------------------------------------------------------
const arrivals = new Map<string, number[]>()

export function noteFrame(robot: string, now: number): number {
  let a = arrivals.get(robot)
  if (!a) arrivals.set(robot, (a = []))
  a.push(now)
  while (a.length && a[0] < now - 2000) a.shift()
  if (a.length < 2) return 0
  return ((a.length - 1) * 1000) / (a[a.length - 1] - a[0])
}

// ---- replan detection ----------------------------------------------------------------------
function distToSegment(p: XY, a: XY, b: XY): number {
  const vx = b[0] - a[0]
  const vy = b[1] - a[1]
  const wx = p[0] - a[0]
  const wy = p[1] - a[1]
  const l2 = vx * vx + vy * vy
  const t = l2 > 0 ? Math.max(0, Math.min(1, (wx * vx + wy * vy) / l2)) : 0
  return Math.hypot(wx - t * vx, wy - t * vy)
}

export function distToPolyline(p: XY, line: readonly XY[]): number {
  if (line.length === 0) return Infinity
  if (line.length === 1) return Math.hypot(p[0] - line[0][0], p[1] - line[0][1])
  let best = Infinity
  for (let k = 0; k < line.length - 1; k++) best = Math.min(best, distToSegment(p, line[k], line[k + 1]))
  return best
}

export function polylineLength(line: readonly XY[]): number {
  let s = 0
  for (let k = 1; k < line.length; k++) s += Math.hypot(line[k][0] - line[k - 1][0], line[k][1] - line[k - 1][1])
  return s
}

/** The part of `path` still ahead of the robot (from the closest vertex onward). */
export function remainingPath(path: readonly XY[], robot: XY): XY[] {
  if (path.length === 0) return []
  let bi = 0
  let bd = Infinity
  for (let k = 0; k < path.length; k++) {
    const d = Math.hypot(path[k][0] - robot[0], path[k][1] - robot[1])
    if (d < bd) {
      bd = d
      bi = k
    }
  }
  return path.slice(bi)
}

/**
 * True when the new plan abandons a meaningful part of the old one: the remaining old path is
 * ≥ minLength metres and some of it lies more than `threshold` metres from the new path.
 */
export function isDrasticReplan(oldPath: readonly XY[], newPath: readonly XY[], robot: XY, threshold = 2.5, minLength = 4): boolean {
  if (oldPath.length < 2 || newPath.length < 2) return false
  const rest = remainingPath(oldPath, robot)
  if (polylineLength(rest) < minLength) return false
  const step = Math.max(1, Math.floor(rest.length / 24))
  for (let k = 0; k < rest.length; k += step) {
    if (distToPolyline(rest[k], newPath) > threshold) return true
  }
  return false
}

// ---- abandoned-plan ghosts (shown fading red on the globe after a replan) -----------------------
export interface PathGhost {
  robot: string
  points: XY[]
  born: number // performance.now() ms
}
export const GHOST_LIFETIME_MS = 3500
export const ghosts: PathGhost[] = []
const lastGhostAt = new Map<string, number>()

export function notePathUpdate(robot: string, oldPath: readonly XY[] | undefined, newPath: readonly XY[], robotXY: XY, now: number) {
  if (!oldPath || oldPath === newPath) return
  if ((lastGhostAt.get(robot) ?? -Infinity) > now - 1500) return
  if (!isDrasticReplan(oldPath, newPath, robotXY)) return
  lastGhostAt.set(robot, now)
  ghosts.push({ robot, points: remainingPath(oldPath, robotXY), born: now })
  while (ghosts.length && ghosts[0].born < now - GHOST_LIFETIME_MS) ghosts.shift()
  if (ghosts.length > 6) ghosts.shift()
}
