import { CFG } from './config';
import { Belief } from './grid';
import type { Readings } from './perception';
import type { Predictions } from './prediction';
import type { PathResult } from './planner';
import { TOOL_NAMES, ToolCall, Goal } from './utility';

export interface RuleResult { ok: boolean; rule: string; detail: string; goal?: Goal }
export interface RuleCtx {
  b: Belief; rd: Readings; P: Predictions; home: { x: number; y: number }; probes: { x: number; y: number }[];
  plan: (x: number, y: number) => PathResult | null;
}

const num = (v: unknown) => (typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN);
const reject = (rule: string, detail: string): RuleResult => ({ ok: false, rule, detail });

/**
 * RULE SET (guardrails). Every tool call from the LLM is checked against the brain's SENSED map
 * before it is allowed to influence the rover. Mirrored in disaster_brain/rules.py.
 *
 * Rules, in order:
 *   unknown_tool, bad_args, out_of_bounds, target_in_water, predicted_hazard, unreachable,
 *   battery_reserve, no_probes_left, probe_spacing, unsafe_hold
 */
export function validateToolCall(call: ToolCall, c: RuleCtx): RuleResult {
  if (!call || typeof call.name !== 'string' || !TOOL_NAMES.includes(call.name)) return reject('unknown_tool', `tool "${call?.name}" is not in the allowed set`);
  const a = call.args ?? {};
  const reason = typeof a.reason === 'string' ? a.reason.slice(0, 160) : '';
  const t = c.rd.t;
  const mk = (x: number, y: number, radius: number, until: number, text: string): RuleResult =>
    ({ ok: true, rule: 'ok', detail: 'all rules passed', goal: { call: { name: call.name, args: a }, t, until, x, y, radius, text } });

  if (call.name === 'hold_position') {
    const s = num(a.seconds);
    if (!isFinite(s) || s < 1 || s > 60) return reject('bad_args', `seconds must be 1..60, got ${a.seconds}`);
    const i = c.b.idx(c.rd.x, c.rd.y);
    if (c.b.risk[i] > 0.4 || c.b.predicted[i] > 0.5) return reject('unsafe_hold', `rover position is inside a predicted hazard zone (risk ${c.b.predicted[i].toFixed(2)})`);
    return mk(c.rd.x, c.rd.y, 0, t + s, `hold_position ${s.toFixed(0)} s: "${reason}"`);
  }
  if (call.name === 'retreat_to_high_ground') return mk(c.rd.x, c.rd.y, 0, t + 45, `retreat_to_high_ground: "${reason}"`);

  const x = num(a.x), y = num(a.y);
  if (!isFinite(x) || !isFinite(y)) return reject('bad_args', `x/y must be numbers, got ${JSON.stringify([a.x, a.y])}`);
  let radius = 0;
  if (call.name === 'survey_area') {
    radius = num(a.radius);
    if (!isFinite(radius) || radius < 5 || radius > 60) return reject('bad_args', `radius must be 5..60 m, got ${a.radius}`);
  }
  if (!c.b.inside(x, y) || Math.abs(x) > CFG.HALF - 3 || Math.abs(y) > CFG.HALF - 3) return reject('out_of_bounds', `(${x.toFixed(0)},${y.toFixed(0)}) is outside the mission area`);
  const i = c.b.idx(x, y);
  if (c.b.water[i] > 0.05 || c.b.predWater[i] > 0.3) return reject('target_in_water', `(${x.toFixed(0)},${y.toFixed(0)}) is water or predicted to flood (sensed depth ${Math.max(0, c.b.water[i]).toFixed(2)} m)`);
  if (c.b.predicted[i] > 0.4 || c.b.risk[i] > 0.4) return reject('predicted_hazard', `(${x.toFixed(0)},${y.toFixed(0)}) is inside a predicted hazard zone (risk ${Math.max(c.b.predicted[i], c.b.risk[i]).toFixed(2)})`);
  if (call.name === 'place_probe_at') {
    if (c.rd.status.probes_left <= 0) return reject('no_probes_left', 'no probes left');
    const near = c.probes.find(p => Math.hypot(p.x - x, p.y - y) < CFG.probeRuleSpacing);
    if (near) return reject('probe_spacing', `within ${CFG.probeRuleSpacing} m of the probe at (${near.x.toFixed(0)},${near.y.toFixed(0)})`);
  }
  const p = c.plan(x, y);
  if (!p) return reject('unreachable', `no safe path to (${x.toFixed(0)},${y.toFixed(0)}) on the sensed map`);
  const back = Math.hypot(x - c.home.x, y - c.home.y) * 1.3;
  const need = (p.cost + back) * CFG.batteryPerM + CFG.batteryReserve;
  if (need > c.rd.status.battery) return reject('battery_reserve', `needs ~${need.toFixed(0)}% battery to go and return with reserve, have ${c.rd.status.battery.toFixed(0)}%`);
  return mk(x, y, radius || 12, t + 60, `${call.name}(${x.toFixed(0)},${y.toFixed(0)}${radius ? `,r=${radius.toFixed(0)}` : ''}): "${reason}"`);
}
