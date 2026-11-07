/**
 * Rule-set test: feed deliberately bad (and a few good) tool calls to the brain's guardrail and check the verdicts.
 *   npm run test:rules
 * Runs a flood scenario for 60 s first so the brain has sensed water, a rising stage and placed probes.
 */
import { Sim } from '../sim/sim';
import { LocalBrain } from '../brain/brain';
import { disasterConfig } from './args';
import type { ToolCall } from '../brain/utility';

const sim = new Sim({ seed: 3, duration: 75, disasters: disasterConfig('flood,fire', 0.6, false) });
const b = new LocalBrain();
while (!sim.done) { if (sim.steps % 2 === 0) sim.apply(b.tick(sim.frame())); sim.step(); }
const rd = b.rd!, B = b.b;
// find concrete cells from the BRAIN's belief (not ground truth): a water cell, a hazard cell, a safe known cell
let water = -1, hazard = -1, safe = -1, unknown = -1, wall = -1;
for (let i = 0; i < B.risk.length; i++) {
  const d = Math.hypot(B.cx(i) - rd.x, B.cy(i) - rd.y);
  if (water < 0 && B.water[i] > 0.3) water = i;
  if (hazard < 0 && B.predicted[i] > 0.6 && B.water[i] <= 0.05 && B.predWater[i] < 0.3) hazard = i;
  if (safe < 0 && d > 8 && d < 25 && B.lastSeen[i] >= 0 && B.predicted[i] < 0.1 && B.water[i] === 0 && !B.blocked(i) && B.channels[4][i] < 0.2) safe = i;
  if (wall < 0 && B.obst[i] > 0.9 && B.risk[i] < 0.1 && B.water[i] <= 0.05 && B.predWater[i] < 0.3 && B.predicted[i] < 0.3) wall = i;
  if (unknown < 0 && B.lastSeen[i] < 0 && Math.abs(B.cx(i)) < 90 && Math.abs(B.cy(i)) < 90) unknown = i;
}
const P = (i: number) => ({ x: +B.cx(i).toFixed(1), y: +B.cy(i).toFixed(1) });
const probes = rd.status.probes;
// an enclosed spot: a cell whose neighbours are all blocked/water is unreachable; fake it by asking for a known obstacle cell's neighbour ring
const cases: { name: string; call: any; expect: string; battery?: number }[] = [
  { name: 'unknown tool', call: { name: 'self_destruct', args: { reason: 'test' } }, expect: 'unknown_tool' },
  { name: 'missing args', call: { name: 'go_to_region', args: { x: 'north', reason: 'test' } }, expect: 'bad_args' },
  { name: 'hold too long', call: { name: 'hold_position', args: { seconds: 600, reason: 'test' } }, expect: 'bad_args' },
  { name: 'survey radius 500', call: { name: 'survey_area', args: { ...P(safe), radius: 500, reason: 'test' } }, expect: 'bad_args' },
  { name: 'off the map', call: { name: 'go_to_region', args: { x: 150, y: -20, reason: 'test' } }, expect: 'out_of_bounds' },
  { name: 'drive into the river', call: { name: 'go_to_region', args: { ...P(water), reason: 'test' } }, expect: 'target_in_water' },
  { name: 'probe in the river', call: { name: 'place_probe_at', args: { ...P(water), reason: 'test' } }, expect: 'target_in_water' },
  { name: 'into predicted hazard', call: { name: 'go_to_region', args: { ...P(hazard), reason: 'test' } }, expect: 'predicted_hazard' },
  { name: 'probe next to a probe', call: { name: 'place_probe_at', args: { x: (probes[0]?.x ?? 0) + 6, y: (probes[0]?.y ?? 0) + 4, reason: 'test' } }, expect: 'probe_spacing' },
  { name: 'into a wall (obstacle cell)', call: { name: 'go_to_region', args: { ...P(Math.max(wall, 0)), reason: 'test' } }, expect: 'unreachable' },
  { name: 'hold inside hazard zone', call: { name: 'hold_position', args: { seconds: 20, reason: 'test' } }, expect: 'unsafe_hold' },
  { name: 'beyond battery reserve', call: { name: 'go_to_region', args: { ...P(safe), reason: 'test' } }, expect: 'battery_reserve', battery: 5 },
  { name: 'GOOD: survey known safe area', call: { name: 'survey_area', args: { ...P(safe), radius: 20, reason: 'test' } }, expect: 'ok' },
  { name: 'GOOD: retreat to high ground', call: { name: 'retreat_to_high_ground', args: { reason: 'test' } }, expect: 'ok' },
  { name: 'GOOD: hold 10 s', call: { name: 'hold_position', args: { seconds: 10, reason: 'test' } }, expect: 'ok' },
];
let pass = 0;
console.log(`brain state at t=${rd.t}: rover (${rd.x.toFixed(0)},${rd.y.toFixed(0)}), probes ${probes.length}, water cell ${JSON.stringify(P(water))}, hazard cell ${hazard >= 0 ? JSON.stringify(P(hazard)) : 'none'}, safe cell ${JSON.stringify(P(safe))}\n`);
for (const c of cases) {
  if (c.expect === 'predicted_hazard' && hazard < 0) { console.log(`SKIP  ${c.name} (no predicted hazard cell sensed)`); continue; }
  if (c.expect === 'probe_spacing' && !probes.length) { console.log(`SKIP  ${c.name} (no probe placed)`); continue; }
  if (c.expect === 'unreachable' && wall < 0) { console.log(`SKIP  ${c.name} (no obstacle cell sensed)`); continue; }
  const saved = rd.status.battery, ri = B.idx(rd.x, rd.y), savedPred = B.predicted[ri];
  if (c.battery !== undefined) rd.status.battery = c.battery;
  if (c.expect === 'unsafe_hold') B.predicted[ri] = 0.9; // pretend the rover's cell is inside a predicted zone
  const r = b.proposeGoal(c.call as ToolCall);
  rd.status.battery = saved; B.predicted[ri] = savedPred;
  b.goal = null;
  const ok = r.rule === c.expect;
  pass += +ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${c.name.padEnd(30)} → ${r.ok ? 'ACCEPTED' : `REJECTED (${r.rule})`}  ${r.ok ? '' : r.detail}`);
}
console.log(`\n${pass}/${cases.length} rule-set checks passed`);
process.exit(pass === cases.length ? 0 : 1);
