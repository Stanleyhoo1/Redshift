import { beforeEach, describe, expect, it } from 'vitest'
import {
  advisorOf,
  advisorPlan,
  aglChip,
  brainChips,
  brainCounters,
  capChip,
  capInEffect,
  commandOutcome,
  commandPhase,
  COMMAND_TIMEOUT_MS,
  effectOf,
  effectRobot,
  eventLabel,
  fmtBrainAgl,
  fmtBrainCap,
  fmtMs,
  fmtTarget,
  fmtTokens,
  isHold,
  latestDecision,
  parseDecision,
  pushDecision,
  sourceTag,
  verdictOf,
} from './brain'
import { useStore } from './store'
import type { BrainCap, BrainDecision, BrainStatus } from './types'

/** A /brain/decision as the node publishes it (Decision.as_dict() + routing fields). */
const raw = (patch: Record<string, unknown> = {}): Record<string, unknown> => ({
  action: 'SLOW_AND_REPLAN',
  target_speed_mps: 0.2,
  target_altitude_m: null,
  reasoning_trace: 'Operator slow-down accepted.',
  source: 'gemma',
  model: 'gemma-4-26b-a4b-it',
  event: 'operator_command',
  latency_ms: 842.4,
  tokens_in: 512,
  tokens_out: 64,
  projected_edge_ms: null,
  projected_device: 'orin_nano_15w',
  projected_fits: false,
  projected_note: 'gemma-4-26b-a4b-it needs ~14.5 GB > 6.4 GB usable on Jetson Orin Nano 8 GB: on this device the local rules are the on-device brain',
  error: null,
  adjusted: [],
  target_robot: null,
  robot: 'rover_0',
  request_id: 'r-7',
  stale: false,
  age_ms: 910,
  t: 1_700_000_000,
  trigger: 'operator',
  trigger_input: 'slow down to 0.2 m/s',
  command_id: 'cmd-2',
  command_text: 'slow down to 0.2 m/s',
  local: { intent: 'set_speed', requested_speed_mps: 0.2, requested_altitude_m: null, hazard: null, exceeds_limit: false, unsafe: false, reasons: [] },
  immediate: false,
  ...patch,
})
const dec = (patch: Record<string, unknown> = {}) => parseDecision(raw(patch))!

describe('parseDecision', () => {
  it('keeps every field of a well-formed decision', () => {
    const d = dec()
    expect(d).toMatchObject({ action: 'SLOW_AND_REPLAN', target_speed_mps: 0.2, source: 'gemma', model: 'gemma-4-26b-a4b-it', latency_ms: 842.4 })
    expect(d.local?.intent).toBe('set_speed')
    expect(d.command_id).toBe('cmd-2')
  })

  it('rejects frames that are not decisions', () => {
    for (const bad of [null, 'x', 3, [], {}, { action: 7, event: 'novel_hazard' }, { action: 'SLOW_AND_REPLAN' }, { action: '', event: 'x' }]) {
      expect(parseDecision(bad)).toBeNull()
    }
  })

  it('coerces wrong field types to safe defaults instead of passing them to the panels', () => {
    const d = parseDecision({ action: 'REJECT_UNSAFE_COMMAND', event: 'operator_command', latency_ms: 'fast', tokens_in: NaN, adjusted: 'x', local: 'y', immediate: 'yes' }, 42)!
    expect(d.latency_ms).toBeNull()
    expect(d.tokens_in).toBeNull()
    expect(d.adjusted).toEqual([])
    expect(d.local).toBeNull()
    expect(d.immediate).toBe(false)
    expect(d.source).toBe('local-fallback')
    expect(d.model).toBe('local-rules')
    expect(d.t).toBe(42)
  })

  it('reads an unbounded requested speed (∞ → null by the backend) as null', () => {
    const d = dec({ local: { intent: 'faster', requested_speed_mps: null, unsafe: true, reasons: ['asks for more than the local limit'] } })
    expect(d.local).toMatchObject({ intent: 'faster', requested_speed_mps: null, unsafe: true, exceeds_limit: false })
  })
})

describe('pushDecision', () => {
  it('prepends, caps, and drops exact repeats', () => {
    const a = dec({ request_id: 'a', t: 1 })
    const b = dec({ request_id: 'b', t: 2 })
    let list = pushDecision([], a)
    list = pushDecision(list, b)
    expect(list.map((d) => d.request_id)).toEqual(['b', 'a'])
    expect(pushDecision(list, dec({ request_id: 'b', t: 2 }))).toBe(list)
    expect(pushDecision(list, dec({ request_id: 'c', t: 3 }), 2).map((d) => d.request_id)).toEqual(['c', 'b'])
  })

  it('keeps the immediate stop and the later answer for the same command apart', () => {
    const imm = dec({ request_id: null, immediate: true, source: 'local-fallback', model: 'local-rules', t: 5 })
    const fin = dec({ request_id: null, immediate: false, t: 5 })
    expect(pushDecision([imm], fin)).toHaveLength(2)
  })
})

describe('commandOutcome / commandPhase', () => {
  const stopImm = dec({ command_id: 'cmd-3', immediate: true, source: 'local-fallback', model: 'local-rules', target_speed_mps: 0, t: 10 })
  const stopFinal = dec({ command_id: 'cmd-3', immediate: false, target_speed_mps: 0, t: 11 })
  const other = dec({ command_id: 'cmd-9', t: 12 })

  it('finds the immediate and the final decision for a command id', () => {
    expect(commandOutcome([other, stopFinal, stopImm], 'cmd-3')).toEqual({ immediate: stopImm, final: stopFinal })
    expect(commandOutcome([stopImm], 'cmd-3')).toEqual({ immediate: stopImm, final: null })
    expect(commandOutcome([other], 'cmd-3')).toEqual({ immediate: null, final: null })
  })

  it('takes the newest decision when a command was answered twice', () => {
    const newer = dec({ command_id: 'cmd-3', t: 20, action: 'REQUEST_CLARIFICATION' })
    expect(commandOutcome([newer, stopFinal], 'cmd-3').final).toBe(newer)
  })

  it('is pending until answered, unanswered after the timeout', () => {
    const none = { immediate: null, final: null }
    expect(commandPhase(none, 0, 1000)).toBe('pending')
    expect(commandPhase(none, 0, COMMAND_TIMEOUT_MS + 1)).toBe('unanswered')
    expect(commandPhase({ immediate: stopImm, final: null }, 0, 1000)).toBe('pending')
    expect(commandPhase({ immediate: stopImm, final: null }, 0, COMMAND_TIMEOUT_MS + 1)).toBe('answered')
    expect(commandPhase({ immediate: null, final: stopFinal }, 0, 10)).toBe('answered')
  })
})

describe('verdictOf', () => {
  it('reads a rejection in the warn tone', () => expect(verdictOf(dec({ action: 'REJECT_UNSAFE_COMMAND', target_speed_mps: null }))).toEqual({ label: 'REJECTED', tone: 'warn' }))
  it('reports an applied cap', () => expect(verdictOf(dec(), 'applied')).toEqual({ label: 'Cap 0.20 m/s', tone: 'ok' }))
  it('reports a hold', () => expect(verdictOf(dec({ target_speed_mps: 0 }), 'applied').label).toBe('Hold: cap 0 m/s'))
  it('never calls a hazard-free "faster" command a cap: it lifts the operator cap', () => {
    const local = { intent: 'faster', requested_speed_mps: 0.83, requested_altitude_m: null, hazard: null, exceeds_limit: false, unsafe: false, reasons: [] }
    expect(verdictOf(dec({ event: 'operator_command', target_speed_mps: 0.83, local }))).toEqual({ label: 'Operator cap lifted: local limits apply', tone: 'ok' })
    expect(verdictOf(dec({ event: 'operator_command', action: 'REJECT_UNSAFE_COMMAND', target_speed_mps: null, local: { ...local, hazard: 'rockslide', unsafe: true } })).label).toBe('REJECTED')
  })
  it('reports the immediate local stop', () => expect(verdictOf(dec({ immediate: true, target_speed_mps: 0 }), 'applied').label).toBe('STOP applied at once (local)'))
  it('reports drone altitude and deployment', () => {
    expect(verdictOf(dec({ action: 'ADJUST_DRONE_ALTITUDE', target_speed_mps: null, target_altitude_m: 8 })).label).toBe('Drone to 8 m AGL')
    expect(verdictOf(dec({ action: 'DEPLOY_RECOVERY_ROVER', target_robot: 'rover_1', target_speed_mps: null })).label).toBe('Deploy rover_1')
  })
  it('flags a clarification hold', () => expect(verdictOf(dec({ action: 'REQUEST_CLARIFICATION', target_speed_mps: 0 }), 'applied')).toEqual({ label: 'Hold: clarification requested', tone: 'warn' }))

  it('never claims a stop, hold or cap the robot has not confirmed', () => {
    // no telemetry confirmation (the default): sent / advised, not done
    expect(verdictOf(dec({ immediate: true, target_speed_mps: 0 }))).toEqual({ label: 'Local stop sent, not confirmed', tone: 'warn' })
    expect(verdictOf(dec({ immediate: true, target_speed_mps: 0 }), 'pending').label).toBe('Local stop sent, not confirmed')
    expect(verdictOf(dec())).toEqual({ label: 'Cap 0.20 m/s advised', tone: 'info' })
    expect(verdictOf(dec({ target_speed_mps: 0 }))).toEqual({ label: 'Hold advised (cap 0 m/s)', tone: 'info' })
    // set on the rover but not applied to its drive
    expect(verdictOf(dec(), 'pending')).toEqual({ label: 'Cap 0.20 m/s pending', tone: 'info' })
    expect(verdictOf(dec({ target_speed_mps: 0.03 }), 'pending').label).toBe('Hold pending (cap 0.03 m/s)')
    expect(verdictOf(dec({ target_speed_mps: 0.03 }), 'applied')).toEqual({ label: 'Hold: cap 0.03 m/s', tone: 'crit' })
  })

  it('renders a drone clarification as a clarification request, not a hold', () => {
    const d = dec({ robot: 'scout_0', action: 'REQUEST_CLARIFICATION', target_speed_mps: 0, reasoning_trace: 'Drones take altitude commands only.' })
    expect(verdictOf(d, effectOf({ type: 'scout', brain_cap: null, brain_agl: null }, d))).toEqual({ label: 'Clarification requested', tone: 'warn' })
    expect(verdictOf(dec({ action: 'REQUEST_CLARIFICATION', target_speed_mps: null }), 'applied').label).toBe('Clarification requested')
  })
})

describe('latestDecision / capInEffect', () => {
  const a = dec({ robot: 'rover_0', t: 1 })
  const b = dec({ robot: 'rover_1', action: 'DEPLOY_RECOVERY_ROVER', target_robot: 'rover_1', t: 2, request_id: 'x' })
  it('selects by subject or deployed robot', () => {
    expect(latestDecision([b, a])).toBe(b)
    expect(latestDecision([b, a], 'rover_0')).toBe(a)
    expect(latestDecision([b, a], 'scout_0')).toBeNull()
    expect(latestDecision([dec({ robot: 'rover_0', target_robot: 'rover_1', request_id: 'y' })], 'rover_1')).not.toBeNull()
  })
  it('confirms a cap only when the rover reports the same value', () => {
    const cap = { speed_mps: 0.2, action: 'SLOW_AND_REPLAN', source: 'gemma', operator: true, remaining_s: 40, applied: true }
    expect(capInEffect({ brain_cap: cap }, a)).toBe(true)
    expect(capInEffect({ brain_cap: { ...cap, speed_mps: 0.3 } }, a)).toBe(false)
    expect(capInEffect({ brain_cap: null }, a)).toBe(false)
    expect(capInEffect(undefined, a)).toBe(false)
    // reported but not applied, or an older backend without `applied`: not in effect
    expect(capInEffect({ brain_cap: { ...cap, applied: false } }, a)).toBe(false)
    const { applied: _omit, ...legacy } = cap
    expect(capInEffect({ brain_cap: legacy }, a)).toBe(false)
  })
})

describe('effectOf / effectRobot', () => {
  const cap = (patch: Partial<BrainCap> = {}): BrainCap => ({ speed_mps: 0.2, action: 'SLOW_AND_REPLAN', source: 'gemma', operator: true, remaining_s: 40, applied: true, ...patch })
  const rover = (c: BrainCap | null) => ({ type: 'rover' as const, brain_cap: c, brain_agl: null })

  it('is applied, pending or none from the rover\'s own brain_cap', () => {
    const d = dec()
    expect(effectOf(rover(cap()), d)).toBe('applied')
    expect(effectOf(rover(cap({ applied: false })), d)).toBe('pending')
    expect(effectOf(rover(cap({ applied: undefined })), d)).toBe('pending')
    expect(effectOf(rover(cap({ speed_mps: 0.35 })), d)).toBe('none')
    expect(effectOf(rover(null), d)).toBe('none')
    expect(effectOf(undefined, d)).toBe('none')
  })

  it('does not credit an operator decision with an event cap of the same value (nor the reverse)', () => {
    expect(effectOf(rover(cap({ operator: false })), dec())).toBe('none')
    expect(effectOf(rover(cap({ operator: true })), dec({ event: 'novel_hazard', command_id: null }))).toBe('none')
    expect(effectOf(rover(cap({ operator: false })), dec({ event: 'novel_hazard', command_id: null }))).toBe('applied')
  })

  it('treats any cap below 0.05 m/s as a hold when matching', () => {
    expect(effectOf(rover(cap({ speed_mps: 0.03 })), dec({ target_speed_mps: 0 }))).toBe('applied')
    expect(effectOf(rover(cap({ speed_mps: 0.05 })), dec({ target_speed_mps: 0 }))).toBe('none')
  })

  it('confirms an immediate stop only on a rover reporting an applied operator hold', () => {
    const imm = dec({ immediate: true, target_speed_mps: 0, source: 'local-fallback', model: 'local-rules' })
    expect(effectOf(rover(cap({ speed_mps: 0 })), imm)).toBe('applied')
    expect(effectOf(rover(cap({ speed_mps: 0, applied: false })), imm)).toBe('pending')
    expect(effectOf(rover(cap({ speed_mps: 0, operator: false })), imm)).toBe('none')
    expect(effectOf(rover(cap({ speed_mps: 0.2 })), imm)).toBe('none')
    expect(effectOf({ type: 'scout', brain_cap: cap({ speed_mps: 0 }), brain_agl: null }, imm)).toBe('none')
    expect(effectOf({ brain_cap: cap({ speed_mps: 0 }) }, imm)).toBe('none')
  })

  it('credits no cap to a rejection or deployment, except an operator stop riding on a rejection', () => {
    expect(effectOf(rover(cap()), dec({ action: 'REJECT_UNSAFE_COMMAND' }))).toBe('none')
    expect(effectOf(rover(cap({ operator: false })), dec({ action: 'DEPLOY_RECOVERY_ROVER', event: 'peer_sos' }))).toBe('none')
    expect(effectOf(rover(cap({ speed_mps: 0 })), dec({ action: 'REJECT_UNSAFE_COMMAND', target_speed_mps: 0 }))).toBe('applied')
    expect(effectOf(rover(cap()), dec({ target_speed_mps: null }))).toBe('none')
  })

  it('reads a drone altitude change from the scout\'s brain_agl', () => {
    const alt = dec({ action: 'ADJUST_DRONE_ALTITUDE', target_speed_mps: null, target_altitude_m: 8, target_robot: 'scout_0' })
    const scout = (agl: number, applied?: boolean) => ({ type: 'scout' as const, brain_cap: null, brain_agl: { agl_m: agl, remaining_s: 50, applied } })
    expect(effectOf(scout(8, true), alt)).toBe('applied')
    expect(effectOf(scout(8, false), alt)).toBe('pending')
    expect(effectOf(scout(8), alt)).toBe('pending')
    expect(effectOf(scout(12, true), alt)).toBe('none')
    expect(effectOf({ type: 'scout', brain_cap: null, brain_agl: null }, alt)).toBe('none')
  })

  it('names the robot whose telemetry confirms the decision', () => {
    expect(effectRobot(dec({ robot: 'rover_1' }))).toBe('rover_1')
    expect(effectRobot(dec({ action: 'ADJUST_DRONE_ALTITUDE', robot: 'rover_0', target_robot: 'scout_0' }))).toBe('scout_0')
    expect(effectRobot(dec({ action: 'ADJUST_DRONE_ALTITUDE', robot: 'scout_0', target_robot: null }))).toBe('scout_0')
    expect(effectRobot(dec({ robot: null }), 'rover_0')).toBe('rover_0')
    expect(effectRobot(dec({ robot: null }))).toBeNull()
  })
})

describe('wording', () => {
  it('formats targets', () => {
    expect(fmtTarget({ target_speed_mps: 0.2, target_altitude_m: null })).toBe('cap 0.20 m/s')
    expect(fmtTarget({ target_speed_mps: 0, target_altitude_m: null })).toBe('hold (cap 0 m/s)')
    expect(fmtTarget({ target_speed_mps: null, target_altitude_m: 8 })).toBe('AGL 8 m')
    expect(fmtTarget({ target_speed_mps: null, target_altitude_m: null })).toBeNull()
    expect(fmtTarget({ target_speed_mps: 0.03, target_altitude_m: null })).toBe('hold (cap 0.03 m/s)')
  })
  it('formats measured latency and tokens without inventing values', () => {
    expect(fmtMs(842.4)).toBe('842 ms')
    expect(fmtMs(1840)).toBe('1840 ms')
    expect(fmtMs(null)).toBe('—')
    expect(fmtTokens(512, 64)).toBe('512 in / 64 out')
    expect(fmtTokens(null, null)).toBe('—')
    expect(fmtTokens(512, null)).toBe('512 in / — out')
  })
  it('formats rover caps and scout altitude', () => {
    expect(fmtBrainCap({ speed_mps: 0.2, action: 'SLOW_AND_REPLAN', source: 'gemma', operator: true, remaining_s: 41.2 })).toBe('cap 0.20 m/s · operator · 42 s')
    expect(fmtBrainCap({ speed_mps: 0, action: 'REQUEST_CLARIFICATION', source: 'gemma', operator: false, remaining_s: 3.1 })).toBe('hold 0 m/s · Gemma · 4 s')
    expect(fmtBrainCap({ speed_mps: 0.3, action: 'SLOW_AND_REPLAN', source: 'local-fallback', operator: false, remaining_s: -1 })).toBe('cap 0.30 m/s · local rules · 0 s')
    expect(fmtBrainAgl({ agl_m: 8, remaining_s: 59.5 })).toBe('AGL 8 m · 60 s')
    expect(fmtBrainCap({ speed_mps: 0.04, action: 'SLOW_AND_REPLAN', source: 'gemma', operator: false, remaining_s: 10 })).toBe('hold 0.04 m/s · Gemma · 10 s')
  })
  it('holds are any speed below 0.05 m/s', () => {
    expect([0, 0.02, 0.049].every(isHold)).toBe(true)
    expect([0.05, 0.2].some(isHold)).toBe(false)
    expect(isHold(null)).toBe(false)
    expect(isHold(Number.NaN)).toBe(false)
  })
  it('shows a cap or altitude the robot does not report as applied as pending (muted), not honoured', () => {
    const c: BrainCap = { speed_mps: 0.2, action: 'SLOW_AND_REPLAN', source: 'gemma', operator: true, remaining_s: 41.2 }
    expect(capChip({ ...c, applied: true })).toEqual({ text: 'cap 0.20 m/s · operator · 42 s', tone: 'cap' })
    expect(capChip({ ...c, speed_mps: 0.02, applied: true }).tone).toBe('hold')
    expect(capChip({ ...c, applied: false })).toEqual({ text: 'pending · cap 0.20 m/s · operator · 42 s', tone: 'pending' })
    expect(capChip(c).tone).toBe('pending')
    expect(aglChip({ agl_m: 8, remaining_s: 59.5, applied: true })).toEqual({ text: 'AGL 8 m · 60 s', tone: 'agl' })
    expect(aglChip({ agl_m: 8, remaining_s: 59.5, applied: false })).toEqual({ text: 'pending · AGL 8 m · 60 s', tone: 'pending' })
    expect(brainChips({ brain_cap: { ...c, applied: true }, brain_agl: { agl_m: 8, remaining_s: 5 } }).map((x) => x.tone)).toEqual(['cap', 'pending'])
    expect(brainChips({ brain_cap: null, brain_agl: null })).toEqual([])
    expect(brainChips(undefined)).toEqual([])
  })
  it('labels events and sources', () => {
    expect(eventLabel('peer_sos')).toBe('Peer SOS')
    expect(eventLabel('some_new_event')).toBe('Some new event')
    expect(sourceTag('gemma')).toBe('GEMMA')
    expect(sourceTag('local-fallback')).toBe('LOCAL')
  })
})

describe('advisorOf / brainCounters', () => {
  const status = (patch: Partial<BrainStatus> = {}): BrainStatus => ({
    enabled: true,
    key_present: true,
    model: 'gemma-4-26b-a4b-it',
    device: 'orin_nano_15w',
    stats: { cloud_calls: 3, gemma: 2, local: 1, rate_limited: 0 },
    backoff_s: 0,
    last_error: null,
    t: 0,
    ...patch,
  })
  it('names the cloud model only when it really advises', () => {
    expect(advisorOf(status())).toMatchObject({ mode: 'cloud', model: 'gemma-4-26b-a4b-it', text: 'gemma-4-26b-a4b-it · Google AI cloud' })
    expect(advisorOf(status({ key_present: false })).text).toBe('local rules — no API key')
    expect(advisorOf(status({ enabled: false })).text).toBe('disabled')
  })
  it('falls back to the config and options before the brain node reports', () => {
    const opt = { model: 'gemma-4-26b-a4b-it', key_present: false, default_enabled: true }
    expect(advisorOf(null, opt, true).mode).toBe('no-key')
    expect(advisorOf(null, opt, false).mode).toBe('disabled')
    expect(advisorOf(null, { ...opt, key_present: true }, true).mode).toBe('not-reporting')
    expect(advisorOf(undefined).mode).toBe('not-reporting')
  })
  it('reads the four counters straight from stats', () => {
    expect(brainCounters(status().stats).map((c) => c.value)).toEqual([3, 2, 1, 0])
    expect(brainCounters(null).every((c) => c.value === null)).toBe(true)
  })
  it('says the brain node is not reporting when the backend marks its status stale, and greys the counters', () => {
    expect(advisorOf(status({ stale: true }))).toMatchObject({ mode: 'stale', text: 'brain node not reporting' })
    expect(advisorOf(status({ stale: true, key_present: false })).mode).toBe('stale')
    expect(advisorOf(status({ stale: false })).mode).toBe('cloud')
    const stale = brainCounters(status().stats, true)
    expect(stale.map((c) => c.value)).toEqual([3, 2, 1, 0])
    expect(stale.every((c) => c.muted)).toBe(true)
    expect(brainCounters(status().stats).some((c) => c.muted)).toBe(false)
  })
})

describe('advisorPlan (pre-launch)', () => {
  const opt = { model: 'gemma-4-26b-a4b-it', key_present: true, default_enabled: true }
  it('names no model and claims no Gemma advice when /api/options has no brain block', () => {
    const p = advisorPlan(undefined, true)
    expect(p).toMatchObject({ model: null, noKey: false, title: 'Advisory brain (cloud)' })
    expect(p.summary).toBe('Advisor model unknown (backend not reporting); the on-board rules keep authority.')
    expect(p.summary).not.toMatch(/gemma/i)
    expect(advisorPlan({ ...opt, model: '  ' }, true).model).toBeNull()
  })
  it('names the reported model, and says so when there is no key', () => {
    expect(advisorPlan(opt, true)).toEqual({
      title: 'Gemma advisory brain (cloud)',
      model: 'gemma-4-26b-a4b-it',
      noKey: false,
      summary: 'gemma-4-26b-a4b-it advises from the Google AI cloud; the on-board rules keep authority.',
    })
    expect(advisorPlan({ ...opt, key_present: false }, true)).toMatchObject({ noKey: true, summary: 'Advisor on, but no API key on the backend: local rules only.' })
    expect(advisorPlan({ ...opt, model: 'other-model' }, true).title).toBe('Advisory brain (cloud)')
    expect(advisorPlan(opt, false).summary).toBe('')
  })
})

describe('store: brain messages', () => {
  beforeEach(() => useStore.getState().resetMission())

  it('ingests valid decisions newest first and ignores malformed ones', () => {
    const { ingest } = useStore.getState()
    ingest({ type: 'brain', decision: raw({ request_id: 'a', t: 1 }) as unknown as BrainDecision })
    ingest({ type: 'brain', decision: { nope: true } as unknown as BrainDecision })
    ingest({ type: 'brain', decision: raw({ request_id: 'b', t: 2 }) as unknown as BrainDecision })
    expect(useStore.getState().brainDecisions.map((d) => d.request_id)).toEqual(['b', 'a'])
  })

  it('remembers sent commands and clears both on a new mission', () => {
    useStore.getState().noteCommand({ id: 'cmd-1', text: 'stop', robot: 'rover_0', sentAt: 0 })
    useStore.getState().ingest({ type: 'brain', decision: raw() as unknown as BrainDecision })
    expect(useStore.getState().commands).toHaveLength(1)
    useStore.getState().resetMission()
    expect(useStore.getState().commands).toHaveLength(0)
    expect(useStore.getState().brainDecisions).toHaveLength(0)
  })
})
