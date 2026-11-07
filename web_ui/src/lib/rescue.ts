// One view of the fleet's SOS / rescue situation (docs/protocol.md §8.4, §8.6, §8.7), merged from
// the 15 Hz telemetry blocks (preferred: fresh) and the 1 Hz `sos` / `rescue` events (fallback, and
// the only source of the "resolved" moment).
import type { RescueEvent, RobotTelemetry, SosEvent } from './types'

export const RESCUE_PHASES = ['TRANSIT', 'DOCKING', 'REPAIRING', 'DONE'] as const

export interface Received<E> {
  ev: E
  receivedAt: number // performance.now() ms
}

export interface RescueView {
  id: string
  target: string
  rescuer: string | null
  /** AWAITING = SOS out, no rescuer assigned (yet, or none in the fleet) */
  phase: string
  range_m: number | null
  locked: boolean
  health_pct: number | null
  faults: string[]
  /** SOS resolved / rescue DONE: shown briefly as a wrap-up */
  resolved: boolean
  aborted: boolean
}

/** Events older than this are stale (both topics repeat at 1 Hz while active). */
export const EVENT_FRESH_MS = 4000
/** How long the "resolved" wrap-up stays on screen. */
export const RESOLVED_SHOW_MS = 9000

const fresh = (r: Received<unknown> | undefined, now: number, ms = EVENT_FRESH_MS) => !!r && now - r.receivedAt <= ms
/** Placeholder id for a rover seen only in AWAIT_RESCUE; the real one is `sos-<robot>-<n>`. */
const syntheticId = (robot: string) => `sos-${robot}`

// "distress call" clock: when each SOS id was first shown (performance.now() ms). SOS ids restart every mission
// (sos-rover_0-1 …), so the store's resetMission clears it.
const firstSeen = new Map<string, number>()

/** Seconds since the SOS `id` was first shown. */
export function distressElapsed(id: string, now: number): number {
  let t0 = firstSeen.get(id)
  if (t0 == null) firstSeen.set(id, (t0 = now))
  return (now - t0) / 1000
}

export function clearDistressClock() {
  firstSeen.clear()
}

/** Map a phase onto the four-step track (ABORTED → -1, AWAITING → -1). */
export function phaseIndex(phase: string): number {
  return (RESCUE_PHASES as readonly string[]).indexOf(phase)
}

export function deriveRescue(
  robots: Record<string, RobotTelemetry> | undefined,
  sos: Record<string, Received<SosEvent>>,
  rescues: Record<string, Received<RescueEvent>>,
  now: number,
): RescueView | null {
  const rs = robots ?? {}
  const views = new Map<string, RescueView>() // by target robot
  const touch = (target: string, id: string): RescueView => {
    let v = views.get(target)
    if (!v) {
      const h = rs[target]?.health
      v = {
        id,
        target,
        rescuer: null,
        phase: 'AWAITING',
        range_m: null,
        locked: false,
        health_pct: h?.pct ?? null,
        faults: h?.faults ?? [],
        resolved: false,
        aborted: false,
      }
      views.set(target, v)
    }
    return v
  }

  // 1. live telemetry: the rescuer's block is authoritative for phase / range / lock; the damaged rover's own
  //    block names its SOS id from the distress call on (peer null until a rescuer is assigned)
  for (const [id, r] of Object.entries(rs)) {
    const b = r.rescue
    if (!b || (b.role === 'rescuer' && !b.peer)) continue
    const done = b.phase === 'DONE' || b.phase === 'ABORTED'
    const ends = { resolved: b.phase === 'DONE', aborted: b.phase === 'ABORTED' }
    if (b.role === 'rescuer' && b.peer) {
      const v = touch(b.peer, b.id)
      Object.assign(v, { id: b.id, rescuer: id, phase: b.phase, range_m: b.range_m ?? null, locked: !!b.locked && !done, ...ends })
    } else {
      const v = touch(id, b.id)
      if (!v.rescuer) Object.assign(v, { id: b.id, rescuer: b.peer || null, phase: b.phase, range_m: b.range_m ?? v.range_m, locked: !!b.locked && !done, ...ends })
    }
  }
  // a damaged rover holding for rescue before any block / event names a rescuer
  for (const [id, r] of Object.entries(rs)) {
    if (r.state === 'AWAIT_RESCUE') touch(id, syntheticId(id))
  }

  // 2. events fill what telemetry does not carry
  for (const rec of Object.values(sos)) {
    const e = rec.ev
    if (e.status !== 'active' || !fresh(rec, now)) continue
    const v = touch(e.robot, e.id)
    if (v.id === syntheticId(e.robot)) v.id = e.id
    if (v.health_pct == null) v.health_pct = e.health_pct ?? null
    if (!v.faults.length && e.faults) v.faults = e.faults
  }
  for (const rec of Object.values(rescues)) {
    const e = rec.ev
    if (!fresh(rec, now) || e.phase === 'DONE' || e.phase === 'ABORTED') continue
    const v = views.get(e.target) ?? (Object.values(sos).some((s) => s.ev.id === e.id && s.ev.status === 'resolved') ? undefined : touch(e.target, e.id))
    if (!v || v.rescuer) continue
    Object.assign(v, { id: e.id, rescuer: e.rescuer, phase: e.phase, range_m: e.range_m ?? null, locked: !!e.locked })
  }

  // the SOS resolves first: the rescuer's block and 1 Hz heartbeat still say REPAIRING while it backs off 2 m
  // (§8.6), so a resolved record for the same id ends the rescue (wrap-up while fresh, gone after that)
  for (const [target, v] of views) {
    const rec = Object.values(sos).find((s) => s.ev.id === v.id && s.ev.status === 'resolved')
    if (!rec) continue
    if (fresh(rec, now, RESOLVED_SHOW_MS)) Object.assign(v, { phase: 'DONE', locked: false, resolved: true, aborted: false })
    else views.delete(target)
  }

  const all = [...views.values()]
  const active = all.filter((v) => !v.resolved && !v.aborted)
  if (active.length) {
    // most advanced rescue first (an assigned rescuer beats a bare SOS)
    active.sort((a, b) => phaseIndex(b.phase) - phaseIndex(a.phase))
    return active[0]
  }
  if (all.length) return all[0]

  // 3. nothing active: show the most recent wrap-up (rescue DONE / ABORTED, or SOS resolved)
  let best: RescueView | null = null
  let bestAt = -Infinity
  for (const rec of Object.values(rescues)) {
    const e = rec.ev
    if ((e.phase !== 'DONE' && e.phase !== 'ABORTED') || !fresh(rec, now, RESOLVED_SHOW_MS) || rec.receivedAt <= bestAt) continue
    bestAt = rec.receivedAt
    const h = rs[e.target]?.health
    best = {
      id: e.id,
      target: e.target,
      rescuer: e.rescuer,
      phase: e.phase,
      range_m: e.range_m ?? null,
      locked: !!e.locked,
      health_pct: h?.pct ?? null,
      faults: h?.faults ?? [],
      resolved: e.phase === 'DONE',
      aborted: e.phase === 'ABORTED',
    }
  }
  if (best) return best
  for (const rec of Object.values(sos)) {
    const e = rec.ev
    if (e.status !== 'resolved' || !fresh(rec, now, RESOLVED_SHOW_MS) || rec.receivedAt <= bestAt) continue
    bestAt = rec.receivedAt
    const h = rs[e.robot]?.health
    best = {
      id: e.id,
      target: e.robot,
      rescuer: null,
      phase: 'DONE',
      range_m: null,
      locked: false,
      health_pct: h?.pct ?? e.health_pct ?? null,
      faults: h?.faults ?? [],
      resolved: true,
      aborted: false,
    }
  }
  return best
}
