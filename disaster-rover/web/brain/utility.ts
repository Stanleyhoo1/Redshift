import { CFG, CH, Behaviour, BEHAVIOURS, clamp01 } from './config';
import { Belief } from './grid';
import type { Readings } from './perception';
import type { Predictions } from './prediction';

export type ToolName = 'go_to_region' | 'place_probe_at' | 'retreat_to_high_ground' | 'hold_position' | 'survey_area';
export const TOOL_NAMES: ToolName[] = ['go_to_region', 'place_probe_at', 'retreat_to_high_ground', 'hold_position', 'survey_area'];
export interface ToolCall { name: ToolName; args: Record<string, unknown> }
/** An accepted agent tool call = the current mission goal. */
export interface Goal { call: ToolCall; t: number; until: number; x: number; y: number; radius: number; text: string }

export interface Spot { x: number; y: number; value: number; cost: number; why: string }

/** Situation summary computed from the belief, used by utility scoring. Sensor-derived only. */
export interface Ctx {
  dangerHere: number; dangerNear: number; dynNear: number; predHere: number; predWaterNear: number;
  hazDist: number; structNear: boolean; aboveWater: number;
  unexplored: number; battery: number; health: number; probesLeft: number;
  probe: Spot | null; safe: Spot | null; high: Spot | null; frontier: Spot | null;
}

export type Scores = Record<Behaviour, number>;

export function buildCtx(b: Belief, rd: Readings, P: Predictions, reach: Float32Array, probes: { x: number; y: number }[], home: { x: number; y: number }, goal: Goal | null): Ctx {
  const G = CFG.G, n = G * G;
  const ri = b.idx(rd.x, rd.y);
  let dangerNear = 0, dynNear = 0, predWaterNear = 0;
  const R = 6; // 12 m
  const x0 = ri % G, y0 = (ri / G) | 0;
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const x = x0 + dx, y = y0 + dy; if (x < 0 || y < 0 || x >= G || y >= G || dx * dx + dy * dy > R * R) continue;
    const i = y * G + x, fall = 1 - Math.sqrt(dx * dx + dy * dy) / (R + 2);
    const dyn = Math.max(b.channels[CH.heat][i], b.channels[CH.rubble][i], b.channels[CH.wind][i]);
    dangerNear = Math.max(dangerNear, b.risk[i] * fall);
    dynNear = Math.max(dynNear, Math.max(dyn, b.predWater[i] > 0.05 ? 0 : b.predicted[i] * 0.8) * fall);
    // water that is predicted to arrive (not the river that is already there)
    if (dx * dx + dy * dy <= 9) predWaterNear = Math.max(predWaterNear, (b.predWater[i] - b.channels[CH.water][i]) * fall);
  }
  const e = b.elev[ri];
  const aboveWater = P.flood.active && !isNaN(e) && !isNaN(P.flood.stage) ? e - P.flood.stage : 99;

  // bearings of nearby dynamic hazards: a refuge on the far side of us from them beats one beside them
  const away: number[] = [];
  if (P.fire.active && P.fire.nearest < 50) away.push(P.fire.bearing);
  if (P.tornado.active && P.tornado.dist < 50) away.push(Math.atan2(P.tornado.y - rd.y, P.tornado.x - rd.x));
  if (P.meteor.active && Math.hypot(P.meteor.cx - rd.x, P.meteor.cy - rd.y) < 50) away.push(Math.atan2(P.meteor.cy - rd.y, P.meteor.cx - rd.x));
  // ---- candidate spots from one Dijkstra (reach) ----
  let safe: Spot | null = null, high: Spot | null = null, probe: Spot | null = null, frontier: Spot | null = null;
  const placed = probes.length;
  for (let i = 0; i < n; i++) {
    const c = reach[i]; if (c === Infinity) continue;
    const x = b.cx(i), y = b.cy(i), pr = b.predicted[i];
    // safest spot: far from dynamic hazards and predicted water, cheap to reach
    if (pr < 0.3 && b.predWater[i] < 0.15) {
      let toward = 0;
      for (const a of away) toward = Math.max(toward, Math.cos(Math.atan2(y - rd.y, x - rd.x) - a));
      const v = Math.min(b.hazDist[i], 45) - 0.2 * c - 40 * pr - 15 * Math.max(0, toward) + (P.flood.active && !isNaN(b.elev[i]) ? Math.min(4, b.elev[i] - P.flood.predStage) * 2 : 0);
      if (!safe || v > safe.value) safe = { x, y, value: v, cost: c, why: `${Math.min(b.hazDist[i], 99).toFixed(0)} m from hazards` };
    }
    // high ground: elevation above the predicted flood stage
    if (!isNaN(b.elev[i]) && !b.struct[i] && pr < 0.35 && b.predWater[i] < 0.05) {
      const above = b.elev[i] - (isNaN(P.flood.predStage) ? 0 : P.flood.predStage);
      const v = Math.min(above, 6) - 0.035 * c - 15 * pr - (b.channels[CH.slope][i] > 0.5 ? 2 : 0);
      if (!high || v > high.value) high = { x, y, value: v, cost: c, why: `+${above.toFixed(1)} m above water` };
    }
  }
  // probe spots (sub-sampled)
  if (rd.status.probes_left > 0) {
    for (let i = 0; i < n; i += 1) {
      if ((i % G) % 2 || ((i / G) | 0) % 2) continue;
      const c = reach[i]; if (c === Infinity || c > 70) continue;
      if (b.lastSeen[i] < 0 || b.blocked(i) || b.predicted[i] > 0.2 || b.water[i] > 0.02 || b.predWater[i] > 0.1) continue;
      const x = b.cx(i), y = b.cy(i);
      if (probes.some(p => Math.hypot(p.x - x, p.y - y) < CFG.probeSpacing)) continue;
      const sp = probeValue(b, i, P, probes);
      const v = sp.value - 0.004 * c;
      if (!probe || v > probe.value) probe = { x, y, value: v, cost: c, why: sp.why };
    }
  }
  // exploration frontier: known & safe cells next to unknown ones
  const goalSurvey = goal && (goal.call.name === 'survey_area' || goal.call.name === 'go_to_region') ? goal : null;
  for (let i = 0; i < n; i++) {
    const c = reach[i]; if (c === Infinity || c < 6) continue;
    if (b.planRisk(i) > 0.3) continue;
    const x = i % G, y = (i / G) | 0;
    let unk = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const X = x + dx, Y = y + dy; if (X >= 0 && Y >= 0 && X < G && Y < G && b.lastSeen[Y * G + X] < 0) unk++; }
    if (!unk) continue;
    const px = b.cx(i), py = b.cy(i);
    const head = Math.cos(Math.atan2(py - rd.y, px - rd.x) - rd.yaw);
    let v = unk * 0.4 - 0.05 * c + 1.5 * head - 8 * b.predicted[i];
    // the map edge is a poor frontier
    if (Math.abs(px) > 92 || Math.abs(py) > 92) v -= 3;
    if (goalSurvey) { const dg = Math.hypot(px - goalSurvey.x, py - goalSurvey.y); v += dg < goalSurvey.radius ? 6 : -0.05 * dg; }
    if (!frontier || v > frontier.value) frontier = { x: px, y: py, value: v, cost: c, why: `${unk} unknown cells` };
  }
  void home; void placed;
  return {
    dangerHere: b.risk[ri], dangerNear, dynNear, predHere: b.predicted[ri], predWaterNear, hazDist: b.hazDist[ri],
    structNear: b.structDist[ri] < 10, aboveWater, unexplored: 1 - b.knownFrac(),
    battery: rd.status.battery, health: rd.status.health, probesLeft: rd.status.probes_left, probe, safe, high, frontier,
  };
}

/**
 * Probe spot value = expected coverage (new ground inside its sensing radius) + warning value
 * (sits between a sensed hazard and where it is heading, or is a flood gauge next to water, or watches the town).
 */
export function probeValue(b: Belief, i: number, P: Predictions, probes: { x: number; y: number }[]): { value: number; why: string } {
  const x = b.cx(i), y = b.cy(i);
  let fresh = 0, tot = 0;
  for (const r of [0, 11, 20]) for (let k = 0; k < (r ? 12 : 1); k++) {
    const px = x + r * Math.cos(k * Math.PI / 6), py = y + r * Math.sin(k * Math.PI / 6);
    tot++;
    if (!b.inside(px, py)) continue;
    if (!probes.some(p => Math.hypot(p.x - px, p.y - py) < CFG.probeRadius)) fresh++;
  }
  const cov = fresh / tot;
  let warn = 0.2, why = 'coverage';
  const hd = b.hazDist[i];
  const anyDyn = P.fire.active || P.tornado.active || P.meteor.active || P.quake.recent;
  if (anyDyn && hd > 10 && hd < 50) { const w = clamp01(1 - Math.abs(hd - 24) / 26); if (w > warn) { warn = w; why = `early warning ${hd.toFixed(0)} m ahead of hazard`; } }
  const wd = b.waterDist[i];
  if (wd < 30 && !isNaN(b.elev[i])) {
    const st = isNaN(P.flood.stage) ? 0 : P.flood.stage;
    const above = b.elev[i] - st;
    if (above > 0.4 && above < 3) { const w = (P.flood.active ? 0.9 : 0.6) * (1 - wd / 40); if (w > warn) { warn = w; why = `flood gauge ${wd.toFixed(0)} m from water, +${above.toFixed(1)} m`; } }
  }
  const sd = b.structDist[i];
  if (sd > 5 && sd < 25) { const w = 0.5; if (w > warn) { warn = w; why = `watches structures ${sd.toFixed(0)} m away`; } }
  return { value: 0.4 * cov + 0.6 * warn, why };
}

/** UTILITY: score every behaviour from the sensed state + the agent's goal. */
export function score(ctx: Ctx, P: Predictions, rd: Readings, goal: Goal | null, placed: number): Scores {
  const s: Scores = { explore: 0, flee: 0, high_ground: 0, hold_still: 0, place_probe: 0, return_to_safety: 0 };
  const lowBatt = ctx.battery < CFG.batteryLow;
  s.explore = (ctx.frontier ? 0.5 : 0.1) * (1 - ctx.dangerNear) * Math.min(1, ctx.unexplored * 2.5) * (lowBatt ? 0.3 : 1);
  // flee: dynamic hazards nearby / predicted to arrive soon
  let f = ctx.dynNear;
  if (P.fire.active && P.fire.eta < 45) f = Math.max(f, 0.35 + 0.55 * (1 - P.fire.eta / 45));
  if (P.fire.active && P.fire.nearest < 35) f = Math.max(f, 0.8 + 0.15 * (1 - P.fire.nearest / 35));
  if (P.fire.active && P.fire.eta < 30) f = Math.max(f, 0.7);
  if (P.tornado.active && P.tornado.cpa < 32 && P.tornado.tcpa < 35) f = Math.max(f, 0.5 + 0.45 * (1 - P.tornado.cpa / 32));
  if (P.tornado.active && P.tornado.dist < 35) f = Math.max(f, 0.6 + 0.35 * (1 - P.tornado.dist / 35));
  if (P.meteor.active) { const d = Math.hypot(P.meteor.cx - rd.x, P.meteor.cy - rd.y); if (d < P.meteor.spread * 1.6) f = Math.max(f, 0.55 + 0.4 * (1 - d / (P.meteor.spread * 1.6))); }
  if (P.quake.recent && ctx.structNear) f = Math.max(f, P.quake.active ? 0.85 : 0.45);
  s.flee = clamp01(f) * (ctx.safe ? 1 : 0.5);
  // high ground: water rising and we're low / predicted water close
  if (P.flood.active) s.high_ground = clamp01(0.3 + 0.7 * ctx.predWaterNear + (ctx.aboveWater < 2 ? 0.35 * (2 - Math.max(0, ctx.aboveWater)) / 2 : 0));
  else s.high_ground = 0.5 * ctx.predWaterNear;
  if (!ctx.high) s.high_ground *= 0.5;
  // hold still while the ground shakes in the open (moving risks driving into a fresh crack)
  s.hold_still = P.quake.active && !ctx.structNear ? 0.75 : 0;
  // probes: value of the best spot, paced over the mission
  if (ctx.probesLeft > 0 && ctx.probe && rd.t > 8) {
    const behind = placed < Math.min(6, rd.t / 24) ? 0.12 : 0;
    s.place_probe = clamp01(ctx.probe.value * (1 - ctx.dangerNear) * (ctx.probe.cost < 45 ? 1 : 0.7) * 0.9 + behind);
  }
  if (lowBatt) s.return_to_safety = 0.6 + (CFG.batteryLow - ctx.battery) / 50;
  if (ctx.health < 35) s.return_to_safety = Math.max(s.return_to_safety, 0.55);

  // ---- the agent's mission goal biases the scores (it never bypasses the safety layer) ----
  if (goal) {
    const g = goal.call.name;
    if (g === 'go_to_region' || g === 'survey_area') s.explore += 0.35;
    if (g === 'place_probe_at' && ctx.probesLeft > 0) s.place_probe = Math.max(s.place_probe, 0.62);
    if (g === 'retreat_to_high_ground') s.high_ground = Math.max(s.high_ground + 0.4, 0.65);
    if (g === 'hold_position') s.hold_still = Math.max(s.hold_still, 0.62);
  }
  for (const k of BEHAVIOURS) s[k] = +clamp01(s[k]).toFixed(2);
  return s;
}

/** Highest score wins, with hysteresis + minimum dwell so the choice doesn't flicker. */
export function select(s: Scores, current: Behaviour, since: number, t: number): Behaviour {
  let best: Behaviour = current, bestV = -1;
  for (const k of BEHAVIOURS) { const v = s[k] + (k === current ? CFG.hysteresis : 0); if (v > bestV) { bestV = v; best = k; } }
  if (best !== current && t - since < CFG.minDwell && s[best] < s[current] + 0.25) return current;
  return best;
}
