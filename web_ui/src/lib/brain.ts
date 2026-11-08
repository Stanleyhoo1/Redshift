// Pure helpers for the advisory Gemma brain: validating /brain/decision frames, matching decisions to operator
// commands, and the HUD's wording. Honesty rules (docs/protocol.md): every number shown comes from a message field —
// the MEASURED cloud round-trip, real token counts, the model's own name — and the on-device figure is only ever the
// module's labelled projection (projected_note, verbatim). Nothing here invents a latency.
import type { BrainAgl, BrainCap, BrainDecision, BrainOption, BrainStats, BrainStatus, OperatorAssessment, RobotTelemetry } from './types'

/** decisions kept in the store (newest first) */
export const BRAIN_CAP = 60
/** Decision.model when the deterministic on-board rules chose the action */
export const LOCAL_MODEL = 'local-rules'
/** an operator command with no decision after this long is reported as unanswered */
export const COMMAND_TIMEOUT_MS = 20000

const str = (v: unknown, dflt = ''): string => (typeof v === 'string' ? v : v == null ? dflt : String(v))
const strOrNull = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

function parseLocal(v: unknown): OperatorAssessment | null {
  if (!isObj(v)) return null
  return {
    intent: str(v.intent, 'unclear'),
    requested_speed_mps: num(v.requested_speed_mps),
    requested_altitude_m: num(v.requested_altitude_m),
    hazard: strOrNull(v.hazard),
    exceeds_limit: v.exceeds_limit === true,
    unsafe: v.unsafe === true,
    reasons: Array.isArray(v.reasons) ? v.reasons.map((r) => str(r)) : [],
  }
}

/**
 * A renderable decision, or null when the frame is not one. The backend forwards the brain node's JSON verbatim, so
 * every field is type-checked here once and the panels never meet a malformed value.
 */
export function parseDecision(x: unknown, nowS = Date.now() / 1000): BrainDecision | null {
  if (!isObj(x) || typeof x.action !== 'string' || !x.action || typeof x.event !== 'string') return null
  return {
    action: x.action,
    target_speed_mps: num(x.target_speed_mps),
    target_altitude_m: num(x.target_altitude_m),
    reasoning_trace: str(x.reasoning_trace),
    source: str(x.source, 'local-fallback'),
    model: str(x.model, LOCAL_MODEL),
    event: x.event,
    latency_ms: num(x.latency_ms),
    tokens_in: num(x.tokens_in),
    tokens_out: num(x.tokens_out),
    projected_edge_ms: num(x.projected_edge_ms),
    projected_device: str(x.projected_device),
    projected_fits: x.projected_fits === true,
    projected_note: str(x.projected_note),
    error: strOrNull(x.error),
    adjusted: Array.isArray(x.adjusted) ? x.adjusted.map((a) => str(a)) : [],
    target_robot: strOrNull(x.target_robot),
    robot: strOrNull(x.robot),
    request_id: strOrNull(x.request_id),
    stale: x.stale === true,
    age_ms: num(x.age_ms),
    t: num(x.t) ?? nowS,
    trigger: str(x.trigger),
    trigger_input: str(x.trigger_input),
    command_id: strOrNull(x.command_id),
    command_text: strOrNull(x.command_text),
    local: parseLocal(x.local),
    immediate: x.immediate === true,
  }
}

const decisionKey = (d: BrainDecision) => `${d.request_id ?? ''}|${d.command_id ?? ''}|${d.immediate ? 'i' : ''}|${d.t}|${d.action}`

/** Newest first, capped; an exact repeat (same request, command, kind and time) is dropped (e.g. after a WS replay). */
export function pushDecision(list: readonly BrainDecision[], d: BrainDecision, cap = BRAIN_CAP): BrainDecision[] {
  const k = decisionKey(d)
  if (list.some((x) => decisionKey(x) === k)) return list as BrainDecision[]
  const next = [d, ...list]
  if (next.length > cap) next.length = cap
  return next
}

export interface CommandOutcome {
  /** the instant local decision on an operator STOP */
  immediate: BrainDecision | null
  /** the brain's answer (Gemma or the local fallback); replaces `immediate` */
  final: BrainDecision | null
}

/** The decisions answering operator command `id` (`decisions` newest first). */
export function commandOutcome(decisions: readonly BrainDecision[], id: string): CommandOutcome {
  let immediate: BrainDecision | null = null
  let final: BrainDecision | null = null
  for (const d of decisions) {
    if (d.command_id !== id) continue
    if (d.immediate) immediate ??= d
    else final ??= d
    if (immediate && final) break
  }
  return { immediate, final }
}

/** Latest decision about `robot` (as subject or as the robot it deploys / moves); any robot when null. */
export function latestDecision(decisions: readonly BrainDecision[], robot: string | null = null): BrainDecision | null {
  if (!robot) return decisions[0] ?? null
  return decisions.find((d) => d.robot === robot || d.target_robot === robot) ?? null
}

export type Tone = 'ok' | 'info' | 'warn' | 'crit'

export interface Verdict {
  label: string
  tone: Tone
}

/**
 * One-line result of a decision as seen by the operator. `effect` is what the target robot's telemetry confirms
 * (effectOf): a stop, hold or cap is only reported as done when the robot says it applies it; otherwise the label
 * says it was sent / advised. Defaults to 'none' so a caller without telemetry never claims anything.
 */
export function verdictOf(d: BrainDecision, effect: Effect = 'none'): Verdict {
  const cap = d.target_speed_mps
  const hold = isHold(cap)
  if (d.immediate) {
    return effect === 'applied' ? { label: 'STOP applied at once (local)', tone: 'crit' } : { label: 'Local stop sent, not confirmed', tone: 'warn' }
  }
  // a hazard-free "faster" command lifts the operator's own stop / cap and sets none (protocol §9.6): whatever speed
  // the answer carries, the rover is back on its local limits, so never call that a cap
  if (d.event === 'operator_command' && d.local?.intent === 'faster' && !d.local?.hazard) {
    return { label: 'Operator cap lifted: local limits apply', tone: 'ok' }
  }
  switch (d.action) {
    case 'REJECT_UNSAFE_COMMAND':
      return { label: 'REJECTED', tone: 'warn' }
    case 'REQUEST_CLARIFICATION':
      return { label: hold && effect === 'applied' ? 'Hold: clarification requested' : 'Clarification requested', tone: 'warn' }
    case 'ADJUST_DRONE_ALTITUDE':
      return { label: d.target_altitude_m != null ? `Drone to ${fmtNum(d.target_altitude_m, 0)} m AGL` : 'Drone altitude', tone: 'ok' }
    case 'DEPLOY_RECOVERY_ROVER':
      return { label: `Deploy ${d.target_robot ?? 'recovery rover'}`, tone: 'info' }
    default: {
      if (cap == null) return { label: actionText(d.action), tone: 'info' }
      const what = hold ? `cap ${fmtSpeed(cap)} m/s` : `${fmtNum(cap, 2)} m/s`
      if (effect === 'applied') return hold ? { label: `Hold: ${what}`, tone: 'crit' } : { label: `Cap ${what}`, tone: 'ok' }
      if (effect === 'pending') return { label: hold ? `Hold pending (${what})` : `Cap ${what} pending`, tone: 'info' }
      return { label: hold ? `Hold advised (${what})` : `Cap ${what} advised`, tone: 'info' }
    }
  }
}

export type CommandPhase = 'pending' | 'answered' | 'unanswered'

/** Where an operator command stands: waiting on the brain, answered (immediate or final), or timed out. */
export function commandPhase(o: CommandOutcome, sentAtMs: number, nowMs: number, timeoutMs = COMMAND_TIMEOUT_MS): CommandPhase {
  if (o.final) return 'answered'
  if (nowMs - sentAtMs > timeoutMs) return o.immediate ? 'answered' : 'unanswered'
  return 'pending'
}

/** A cap below this is a hold: the rover is meant to stand still (the smoke test's "stopped" threshold, too). */
export const HOLD_BELOW_MPS = 0.05
export const isHold = (mps: number | null | undefined): boolean => mps != null && Number.isFinite(mps) && mps < HOLD_BELOW_MPS

/**
 * What the robot's own telemetry confirms about decision `d`:
 * - 'applied': it reports the decision's cap (or cruise AGL) with `applied: true`, i.e. it is being honoured now;
 * - 'pending': it reports the cap / AGL but not as applied (a motion the cap does not govern, a scout not surveying);
 * - 'none': nothing matching (not received yet, expired, replaced, or a decision that changes no motion).
 * A backend that does not send `applied` never confirms. An immediate stop needs a rover reporting an operator hold.
 */
export type Effect = 'applied' | 'pending' | 'none'
export type RobotBrainView = Partial<Pick<RobotTelemetry, 'type' | 'brain_cap' | 'brain_agl'>>

export function effectOf(robot: RobotBrainView | null | undefined, d: BrainDecision): Effect {
  if (!robot) return 'none'
  if (d.action === 'ADJUST_DRONE_ALTITUDE' && !d.immediate) {
    const a = robot.brain_agl
    const tgt = d.target_altitude_m
    if (!a || tgt == null || Math.abs(a.agl_m - tgt) > 0.05) return 'none'
    return a.applied === true ? 'applied' : 'pending'
  }
  const c = robot.brain_cap
  const sp = d.target_speed_mps
  if (!c) return 'none'
  const operator = d.event === 'operator_command'
  if (d.immediate) {
    if (robot.type !== 'rover' || !isHold(c.speed_mps) || c.operator !== true) return 'none'
  } else {
    if (sp == null || c.operator !== operator) return 'none'
    // the planner sets no cap for a deployment or a rejection, except an operator stop riding on one
    if ((d.action === 'DEPLOY_RECOVERY_ROVER' || d.action === 'REJECT_UNSAFE_COMMAND') && !(operator && sp === 0)) return 'none'
    if (!(Math.abs(c.speed_mps - sp) < 5e-3 || (isHold(c.speed_mps) && isHold(sp)))) return 'none'
  }
  return c.applied === true ? 'applied' : 'pending'
}

/** True only when the robot reports the decision's cap (or AGL) as applied right now: effectOf(...) === 'applied'. */
export function capInEffect(robot: RobotBrainView | null | undefined, d: BrainDecision): boolean {
  return effectOf(robot, d) === 'applied'
}

/** The robot whose telemetry confirms `d`: the drone an altitude change moves, else the decision's subject. */
export function effectRobot(d: BrainDecision, fallback: string | null = null): string | null {
  if (d.action === 'ADJUST_DRONE_ALTITUDE') return d.target_robot ?? d.robot ?? fallback
  return d.robot ?? fallback
}

// ---------------------------------------------------------------------------------------------------- wording

/** "SLOW_AND_REPLAN" → "SLOW AND REPLAN" (shown in the condensed caps face, like FSM states). */
export const actionText = (a: string) => a.replace(/_/g, ' ')

const EVENT_LABEL: Record<string, string> = {
  novel_hazard: 'Novel hazard',
  low_confidence: 'Low confidence',
  critical_damage: 'Critical damage',
  peer_sos: 'Peer SOS',
  operator_command: 'Operator command',
}
export function eventLabel(e: string): string {
  if (EVENT_LABEL[e]) return EVENT_LABEL[e]
  const s = e.replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

/** Sensor-chip tag for the brain that decided. */
export const sourceTag = (source: string): 'GEMMA' | 'LOCAL' => (source === 'gemma' ? 'GEMMA' : 'LOCAL')

function fmtNum(n: number, digits: number): string {
  return n.toFixed(digits)
}
/** a cap's speed: "0" for an exact hold, else two decimals */
const fmtSpeed = (mps: number) => (mps === 0 ? '0' : fmtNum(mps, 2))

/** "cap 0.20 m/s", "hold (cap 0 m/s)", "AGL 8 m", or null when the decision changes no number. */
export function fmtTarget(d: Pick<BrainDecision, 'target_speed_mps' | 'target_altitude_m'>): string | null {
  const parts: string[] = []
  if (d.target_speed_mps != null) parts.push(isHold(d.target_speed_mps) ? `hold (cap ${fmtSpeed(d.target_speed_mps)} m/s)` : `cap ${fmtNum(d.target_speed_mps, 2)} m/s`)
  if (d.target_altitude_m != null) parts.push(`AGL ${fmtNum(d.target_altitude_m, 0)} m`)
  return parts.length ? parts.join(' · ') : null
}

/** Measured milliseconds, whole: 842.4 → "842 ms", 1840 → "1840 ms"; null → "—". */
export function fmtMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—'
  return `${Math.round(ms)} ms`
}

/** "512 in / 64 out"; "—" when the call reported no usage. */
export function fmtTokens(tin: number | null | undefined, tout: number | null | undefined): string {
  if (tin == null && tout == null) return '—'
  const f = (n: number | null | undefined) => (n == null ? '—' : String(Math.round(n)))
  return `${f(tin)} in / ${f(tout)} out`
}

const fmtRemaining = (s: number) => `${Math.max(0, Math.ceil(Number.isFinite(s) ? s : 0))} s`

/** Rover cap: "cap 0.20 m/s · operator · 42 s" (remaining simulated seconds); a hold is any cap below 0.05 m/s. */
export function fmtBrainCap(c: BrainCap): string {
  const who = c.operator ? 'operator' : c.source === 'gemma' ? 'Gemma' : 'local rules'
  const what = isHold(c.speed_mps) ? `hold ${fmtSpeed(c.speed_mps)} m/s` : `cap ${fmtNum(c.speed_mps, 2)} m/s`
  return `${what} · ${who} · ${fmtRemaining(c.remaining_s)}`
}

/** Scout cruise altitude: "AGL 8 m · 42 s". */
export function fmtBrainAgl(a: BrainAgl): string {
  return `AGL ${fmtNum(a.agl_m, 0)} m · ${fmtRemaining(a.remaining_s)}`
}

/** 'hold' / 'cap' / 'agl' = applied (the robot honours it now); 'pending' = reported but not applied (muted). */
export type BrainChipTone = 'hold' | 'cap' | 'agl' | 'pending'
export interface BrainChip {
  text: string
  tone: BrainChipTone
}

/** A speed cap as a chip: honoured only when the rover reports `applied: true`, else "pending · …". */
export function capChip(c: BrainCap): BrainChip {
  if (c.applied !== true) return { text: `pending · ${fmtBrainCap(c)}`, tone: 'pending' }
  return { text: fmtBrainCap(c), tone: isHold(c.speed_mps) ? 'hold' : 'cap' }
}

/** A cruise altitude as a chip: honoured only when the scout reports `applied: true` (it is surveying at it). */
export function aglChip(a: BrainAgl): BrainChip {
  return a.applied === true ? { text: fmtBrainAgl(a), tone: 'agl' } : { text: `pending · ${fmtBrainAgl(a)}`, tone: 'pending' }
}

/** The robot's brain chips: its speed cap, then its cruise altitude (each only when reported). */
export function brainChips(r: Partial<Pick<RobotTelemetry, 'brain_cap' | 'brain_agl'>> | null | undefined): BrainChip[] {
  const out: BrainChip[] = []
  if (r?.brain_cap) out.push(capChip(r.brain_cap))
  if (r?.brain_agl) out.push(aglChip(r.brain_agl))
  return out
}

export type AdvisorMode = 'cloud' | 'no-key' | 'disabled' | 'not-reporting' | 'stale'

export interface Advisor {
  mode: AdvisorMode
  /** configured cloud model (null when unknown) */
  model: string | null
  text: string
}

/**
 * Who advises right now. /brain/status is the truth while a mission runs; before the brain node reports, the mission
 * config (ai_brain) and /api/options (key presence) tell what it will be. A status the backend marks `stale` (the
 * node stopped reporting) says nothing about now.
 */
export function advisorOf(status: BrainStatus | null | undefined, option?: BrainOption, enabledCfg?: boolean): Advisor {
  if (status?.stale === true) return { mode: 'stale', model: status.model || null, text: 'brain node not reporting' }
  if (status) {
    if (!status.enabled) return { mode: 'disabled', model: status.model || null, text: 'disabled' }
    if (!status.key_present) return { mode: 'no-key', model: status.model || null, text: 'local rules — no API key' }
    return { mode: 'cloud', model: status.model || null, text: `${status.model} · Google AI cloud` }
  }
  if (enabledCfg === false) return { mode: 'disabled', model: option?.model ?? null, text: 'disabled' }
  if (option && !option.key_present) return { mode: 'no-key', model: option.model, text: 'local rules — no API key' }
  return { mode: 'not-reporting', model: option?.model ?? null, text: 'brain node not reporting yet' }
}

export interface Counter {
  label: string
  value: number | null
  /** last values of a brain node that stopped reporting: shown greyed */
  muted: boolean
}

/** The four HUD counters, straight from AgentBrain.stats() (null = not reported); `stale` greys them. */
export function brainCounters(stats: Partial<BrainStats> | null | undefined, stale = false): Counter[] {
  const v = (k: keyof BrainStats) => num(stats?.[k])
  return [
    { label: 'cloud calls', value: v('cloud_calls'), muted: stale },
    { label: 'Gemma answers', value: v('gemma'), muted: stale },
    { label: 'local fallbacks', value: v('local'), muted: stale },
    { label: 'rate-limited', value: v('rate_limited'), muted: stale },
  ]
}

export interface AdvisorPlan {
  /** the toggle's title: names Gemma only when the backend reports a Gemma model */
  title: string
  /** the model /api/options reports, or null when the backend did not report one */
  model: string | null
  /** the backend reports no API key: the local rules answer every event */
  noKey: boolean
  /** the deploy summary's advisor sentence ('' when the advisor is off) */
  summary: string
}

/** Text for the advisor before launch: nothing about the model or Gemma advising unless /api/options says so. */
export function advisorPlan(option: BrainOption | null | undefined, on: boolean): AdvisorPlan {
  const model = (typeof option?.model === 'string' && option.model.trim()) || null
  const noKey = !!option && !option.key_present
  const title = model && /gemma/i.test(model) ? 'Gemma advisory brain (cloud)' : 'Advisory brain (cloud)'
  let summary = ''
  if (on) {
    if (!model) summary = 'Advisor model unknown (backend not reporting); the on-board rules keep authority.'
    else if (noKey) summary = 'Advisor on, but no API key on the backend: local rules only.'
    else summary = `${model} advises from the Google AI cloud; the on-board rules keep authority.`
  }
  return { title, model, noKey, summary }
}
