import { CFG, Behaviour, compass, wrap, CH } from './config';
import { Belief } from './grid';
import { perceive, PerceptionState, Readings } from './perception';
import { Predictor } from './prediction';
import { astar, reachMap, pathRisk, PathResult } from './planner';
import { buildCtx, score, select, Ctx, Scores, Goal, ToolCall, Spot } from './utility';
import { Safety, Reflex } from './safety';
import { validateToolCall, RuleResult } from './rules';
import type { Brain } from './types';
import type { SensorFrame, BrainOutput, Twist } from './msgs';
import { zeroTwist } from './msgs';

const tw = (v: number, w: number): Twist => ({ linear: { x: v, y: 0, z: 0 }, angular: { x: 0, y: 0, z: w } });

export interface BrainEvent { kind: string; text: string }

/**
 * The local TypeScript brain. Same pipeline as the rclpy brain:
 * PERCEPTION -> PREDICTION -> UTILITY SELECTION -> PLANNER (A*) -> control, with SAFETY REFLEXES overriding.
 * Input is one SensorFrame; it never sees the sim's world state.
 */
export class LocalBrain implements Brain {
  readonly name: string = 'local-ts';
  b = new Belief(); ps = new PerceptionState(); pred = new Predictor(); safety = new Safety();
  rd: Readings | null = null;
  behaviour: Behaviour = 'explore'; since = 0;
  scores: Scores = { explore: 0, flee: 0, high_ground: 0, hold_still: 0, place_probe: 0, return_to_safety: 0 };
  ctx: Ctx | null = null; reach: Float32Array | null = null;
  target: (Spot & { kind: Behaviour; set: number }) | null = null;
  path: PathResult | null = null; lastPlan = -1;
  home: { x: number; y: number } | null = null;
  goal: Goal | null = null;
  reflex: Reflex | null = null;
  confidence = 0.5;
  badUntil = new Float32Array(CFG.G * CFG.G);
  /** events for the agent (new hazard, hazard on path, low battery, probe placed) */
  events: BrainEvent[] = [];
  private reflexPos = { x: 0, y: 0, t: 0 };
  private lastMap = -1; private lastLog = -1; private lastLogKey = '';
  private stuckSince = -1; private recoverUntil = -1; private recoverW = 1;
  private lastProbeDrop = -1e9; private battEvent = false; private pathEvent = -1e9;

  tick(f: SensorFrame): BrainOutput {
    const b = this.b, P = this.pred.p;
    const rd = perceive(b, f, this.ps); this.rd = rd;
    const t = rd.t;
    if (!this.home) this.home = { x: rd.x, y: rd.y };
    this.pred.observe(rd, this.ps);
    if (!rd.status.alive) return { cmd_vel: zeroTwist(), drop_probe: null, decision_log: null };

    // ---- map + predictions + candidate spots, 2 Hz ----
    if (this.lastMap < 0 || t - this.lastMap >= CFG.mapPeriod) {
      this.lastMap = t;
      this.pred.mapUpdate(b, rd, this.ps);
      for (const k of this.pred.newlySensed) this.events.push({ kind: 'new_hazard', text: `new hazard type sensed: ${k}` });
      this.reach = reachMap(b, rd.x, rd.y, 75);
      const reach = this.reach;
      for (let i = 0; i < reach.length; i++) if (this.badUntil[i] > t) reach[i] = Infinity;
      this.ctx = buildCtx(b, rd, P, reach, this.probesPlaced(rd), this.home, this.goal);
      if (!this.battEvent && rd.status.battery < CFG.batteryAgentEvent) { this.battEvent = true; this.events.push({ kind: 'battery', text: `battery ${rd.status.battery.toFixed(0)}%` }); }
    }
    const ctx = this.ctx!;
    this.expireGoal(rd);

    // ---- utility selection ----
    const placed = 6 - rd.status.probes_left;
    this.scores = score(ctx, P, rd, this.goal, placed);
    const prev = this.behaviour;
    const nb = select(this.scores, this.behaviour, this.since, t);
    if (nb !== this.behaviour) { this.behaviour = nb; this.since = t; this.target = null; this.path = null; }
    const sorted = Object.values(this.scores).sort((a, c) => c - a);
    this.confidence = Math.max(0.05, Math.min(0.99, 0.45 + (sorted[0] - sorted[1]) + 0.35 * b.conf[b.idx(rd.x, rd.y)]));

    // ---- target for the chosen behaviour ----
    this.chooseTarget(rd, ctx);

    // ---- plan (A* over the predicted risk map), re-plan when the map changes under the path ----
    let drop = false;
    let cmd = tw(0, 0);
    if (this.target) {
      const tgt = this.target;
      const dist = Math.hypot(tgt.x - rd.x, tgt.y - rd.y);
      const risky = this.path ? pathRisk(b, this.path.cells, 40).risk > 0.6 : false;
      if (!this.path || t - this.lastPlan > 2.0 || risky) {
        if (risky && t - this.pathEvent > 15) { this.pathEvent = t; this.events.push({ kind: 'hazard_on_path', text: 'predicted hazard entered the planned path' }); }
        this.lastPlan = t;
        const urgent = this.behaviour === 'flee' || this.behaviour === 'high_ground' || this.behaviour === 'return_to_safety';
        this.path = astar(b, rd.x, rd.y, tgt.x, tgt.y, false) ?? (urgent ? astar(b, rd.x, rd.y, tgt.x, tgt.y, true) : null);
        if (!this.path) { this.blacklist(tgt.x, tgt.y, t); this.target = null; }
      }
      if (this.target && dist < 2.5) {
        if (this.behaviour === 'place_probe' && rd.status.probes_left > 0 && t - this.lastProbeDrop > 3) {
          drop = true; this.lastProbeDrop = t;
          this.events.push({ kind: 'probe_placed', text: `probe placed at (${rd.x.toFixed(0)}, ${rd.y.toFixed(0)}): ${tgt.why}` });
          if (this.goal?.call.name === 'place_probe_at') this.goal = null;
          this.since = t - CFG.minDwell; // allow immediate re-selection
        }
        if (this.behaviour === 'explore' && this.goal && (this.goal.call.name === 'go_to_region') && Math.hypot(this.goal.x - rd.x, this.goal.y - rd.y) < 5) this.goal = null;
        this.target = null; this.path = null;
      } else if (this.path) cmd = this.follow(rd, this.path);
    } else if (this.behaviour === 'explore' || this.behaviour === 'flee') {
      cmd = tw(0, 0.5); // nothing reachable: turn in place to sense more
    }

    // ---- stuck recovery ----
    if (t < this.recoverUntil) cmd = tw(-1.0, this.recoverW);
    else if (Math.abs(cmd.linear.x) > 0.6 && Math.abs(rd.v) < 0.12) {
      if (this.stuckSince < 0) this.stuckSince = t;
      if (t - this.stuckSince > 1.5) {
        this.recoverUntil = t + 1.2; this.recoverW = Math.floor(t) % 2 ? 1.2 : -1.2; this.stuckSince = -1;
        const ax = rd.x + 2 * Math.cos(rd.yaw), ay = rd.y + 2 * Math.sin(rd.yaw);
        if (b.inside(ax, ay)) b.obst[b.idx(ax, ay)] = 1;
        this.path = null;
      }
    } else this.stuckSince = -1;

    // ---- SAFETY REFLEXES override everything ----
    const reflex = this.safety.check(rd, b, P);
    const reflexStarted = reflex && (!this.reflex || this.reflex.name !== reflex.name);
    this.reflex = reflex;
    if (reflex) {
      cmd = reflex.cmd;
      // a reflex that produces no motion (wedged against something) gets a wiggle: turn, then push forward
      if (!reflexStarted && Math.hypot(rd.x - this.reflexPos.x, rd.y - this.reflexPos.y) < 0.4 && t - this.reflexPos.t > 1.5) {
        const ph = (t - this.reflexPos.t) % 3;
        cmd = ph < 1.5 ? tw(0, 1.4) : tw(1.5, 0);
        reflex.text += ' [stalled: wiggle]';
      } else if (reflexStarted || Math.hypot(rd.x - this.reflexPos.x, rd.y - this.reflexPos.y) >= 0.4) this.reflexPos = { x: rd.x, y: rd.y, t };
    }
    // never drive forward into something LiDAR sees right in front
    if (cmd.linear.x > 0 && rd.lidarFront < CFG.reflexFront) { cmd = tw(0, cmd.angular.z || 0.9); }

    // ---- decision log ----
    let log: string | null = null;
    const key = `${this.behaviour}|${reflex?.name ?? ''}`;
    if (reflexStarted) log = `REFLEX ${reflex!.text} (overrides ${this.behaviour})`;
    else if (key !== this.lastLogKey || t - this.lastLog > 4 || drop) log = this.logLine(rd, prev, drop);
    if (log) { this.lastLog = t; this.lastLogKey = key; }
    return { cmd_vel: cmd, drop_probe: drop ? { x: rd.x, y: rd.y, z: 0 } : null, decision_log: log };
  }

  private probesPlaced(rd: Readings) { return rd.status.probes.map(p => ({ x: p.x, y: p.y })); }

  private blacklist(x: number, y: number, t: number) {
    const i = this.b.idx(x, y), G = CFG.G;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const k = i + dy * G + dx; if (k >= 0 && k < this.badUntil.length) this.badUntil[k] = t + 20; }
  }

  private chooseTarget(rd: Readings, ctx: Ctx) {
    const t = rd.t, b = this.b, cur = this.target, beh = this.behaviour;
    let want: Spot | null = null;
    const g = this.goal;
    switch (beh) {
      case 'explore':
        if (g && g.call.name === 'go_to_region') want = { x: g.x, y: g.y, value: 1, cost: 0, why: `agent goal: ${g.text}` };
        else want = ctx.frontier; break;
      case 'flee': want = ctx.safe; break;
      case 'high_ground': want = ctx.high; break;
      case 'place_probe':
        if (g && g.call.name === 'place_probe_at') want = { x: g.x, y: g.y, value: 1, cost: 0, why: `agent goal: ${g.text}` };
        else want = ctx.probe; break;
      case 'return_to_safety': {
        const h = this.home!; const hi = b.idx(h.x, h.y);
        want = b.planRisk(hi) < 0.3 ? { x: h.x, y: h.y, value: 1, cost: 0, why: 'home base' } : ctx.safe; break;
      }
      case 'hold_still': want = null; break;
    }
    if (!want) { this.target = null; return; }
    // keep the current target unless it became risky, is stale, or the new one is clearly better/elsewhere
    if (cur && cur.kind === beh) {
      const ci = b.idx(cur.x, cur.y);
      const stale = beh === 'explore' ? t - cur.set > 15 : t - cur.set > 6;
      const bad = b.planRisk(ci) > 0.45 || b.blocked(ci) || (beh !== 'place_probe' && beh !== 'return_to_safety' && b.lastSeen[ci] >= 0 && beh === 'explore' && this.frontierGone(ci));
      const moved = Math.hypot(want.x - cur.x, want.y - cur.y) > 8;
      if (!bad && !(stale && moved) && !(moved && (beh === 'flee' || beh === 'high_ground') && want.value > cur.value + 3)) return;
    }
    this.target = { ...want, kind: beh, set: t };
    this.path = null;
  }
  private frontierGone(i: number) {
    const G = CFG.G, x = i % G, y = (i / G) | 0;
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) { const X = x + dx, Y = y + dy; if (X >= 0 && Y >= 0 && X < G && Y < G && this.b.lastSeen[Y * G + X] < 0) return false; }
    return true;
  }

  /** pure pursuit on the A* path + LiDAR-informed speed */
  private follow(rd: Readings, p: PathResult): Twist {
    const pts = p.path;
    let k = 0, best = Infinity;
    for (let i = 0; i < pts.length; i++) { const d = Math.hypot(pts[i].x - rd.x, pts[i].y - rd.y); if (d < best) { best = d; k = i; } }
    let tx = pts[pts.length - 1].x, ty = pts[pts.length - 1].y;
    for (let i = k; i < pts.length; i++) if (Math.hypot(pts[i].x - rd.x, pts[i].y - rd.y) > 3.5) { tx = pts[i].x; ty = pts[i].y; break; }
    const err = wrap(Math.atan2(ty - rd.y, tx - rd.x) - rd.yaw);
    const urgent = this.behaviour === 'flee' || this.behaviour === 'high_ground';
    let v = (urgent ? CFG.maxV : CFG.maxV * 0.8) * Math.max(0, Math.cos(err)) ** 2;
    if (Math.abs(err) > 1.2) v = 0;
    if (rd.lidarFront < 4) v = Math.min(v, 0.6 + 0.4 * rd.lidarFront);
    const w = Math.max(-CFG.maxW, Math.min(CFG.maxW, 2.2 * err));
    return tw(v, w);
  }

  private expireGoal(rd: Readings) {
    if (this.goal && rd.t > this.goal.until) this.goal = null;
  }

  private logLine(rd: Readings, prev: Behaviour, drop: boolean): string {
    const P = this.pred.p, s = this.scores, beh = this.behaviour;
    // most salient sensor reading
    let trig = `LiDAR ${(rd.returnFrac * 100).toFixed(0)}% returns, map ${(100 * (1 - (this.ctx?.unexplored ?? 1))).toFixed(0)}% known`;
    if (rd.thermalMax.temp > 80) trig = `Thermal ${rd.thermalMax.temp.toFixed(0)}°C ${rd.thermalMax.dist.toFixed(0)} m ${compass(rd.thermalMax.bearing)}`;
    if (P.flood.active || rd.depth > 0.05) trig = `Water ${rd.depth > 0.02 ? `depth ${rd.depth.toFixed(2)} m` : rd.waterNear ? `edge ${rd.waterNear.dist.toFixed(0)} m ${compass(rd.waterNear.bearing)}` : 'not in view'}`;
    if (rd.windSpeed > 11 || rd.funnel) trig = `Wind ${rd.windSpeed.toFixed(0)} m/s, ΔP −${Math.max(0, rd.pressureDrop).toFixed(0)} Pa${rd.funnel ? `, funnel ${rd.funnel.dist.toFixed(0)} m ${compass(rd.funnel.bearing)}` : ''}`;
    if (rd.shake > CFG.shakeThresh) trig = `IMU shake ${rd.shake.toFixed(1)} m/s²`;
    if (rd.flashes.length) trig = `Impact flash ${rd.flashes[0].dist.toFixed(0)} m ${compass(rd.flashes[0].bearing)}`;
    if (beh === 'place_probe' || drop) trig = `Probe spot: ${this.target?.why ?? this.ctx?.probe?.why ?? '-'}`;
    const preds = [P.fire.text, P.flood.text, P.tornado.text, P.meteor.text, P.quake.text].filter(Boolean);
    const sc = (Object.entries(s) as [string, number][]).sort((a, c) => c[1] - a[1]).slice(0, 3).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(' ');
    const tg = this.target;
    const via = tg ? ` → ${Math.hypot(tg.x - rd.x, tg.y - rd.y).toFixed(0)} m ${compass(Math.atan2(tg.y - rd.y, tg.x - rd.x))} (${tg.why})` : beh === 'hold_still' ? ' (stay put)' : '';
    const action = drop ? 'DROP PROBE' : beh.toUpperCase();
    const goal = this.goal ? ` [goal: ${this.goal.call.name}]` : '';
    return `${trig} | ${preds.length ? preds.join('; ') : 'no hazard predicted'} | ${sc} → ${action}${via} (${this.confidence.toFixed(2)})${prev !== beh ? ` [was ${prev}]` : ''}${goal}`;
  }

  // ================= agent interface =================
  /** Situation summary for the LLM, built ONLY from the brain's sensed map/predictions/status. */
  summary(): string {
    const rd = this.rd, ctx = this.ctx, P = this.pred.p, b = this.b;
    if (!rd || !ctx) return 'no data yet';
    const dirs: string[] = [];
    // unexplored directions
    for (let k = 0; k < 8; k++) {
      const a = k * Math.PI / 4; let unk = 0, tot = 0;
      for (const d of [30, 50, 70]) { const x = rd.x + d * Math.cos(a), y = rd.y + d * Math.sin(a); if (!b.inside(x, y)) continue; tot++; if (b.lastSeen[b.idx(x, y)] < 0) unk++; }
      if (tot && unk / tot > 0.6) dirs.push(compass(a));
    }
    const hz = [P.fire.text, P.flood.text, P.tornado.text, P.meteor.text, P.quake.text].filter(Boolean);
    const probes = rd.status.probes.map(p => `(${p.x.toFixed(0)},${p.y.toFixed(0)}${p.alive ? '' : ' lost'})`).join(' ') || 'none';
    const f = (s: Spot | null) => s ? `(${s.x.toFixed(0)},${s.y.toFixed(0)})` : 'none';
    return [
      `t=${rd.t.toFixed(0)}s rover at (${rd.x.toFixed(0)},${rd.y.toFixed(0)}) battery ${rd.status.battery.toFixed(0)}% health ${rd.status.health.toFixed(0)}% probes_left ${rd.status.probes_left} behaviour ${this.behaviour}.`,
      `Map ${(100 * (1 - ctx.unexplored)).toFixed(0)}% explored; unexplored directions: ${dirs.join(',') || 'few'}.`,
      `Hazards sensed: ${hz.length ? hz.join('; ') : 'none'}.`,
      `Probes placed: ${probes}. Best probe spot: ${f(ctx.probe)} (${ctx.probe?.why ?? '-'}). Safest reachable spot: ${f(ctx.safe)}. Highest safe ground: ${f(ctx.high)}${ctx.high ? ` (${ctx.high.why})` : ''}.`,
      `Current goal: ${this.goal ? this.goal.text : 'none'}.`,
    ].join('\n');
  }

  /** RULE SET: validate an agent tool call against the sensed map. Accepted calls become the mission goal. */
  proposeGoal(call: ToolCall): RuleResult {
    if (!this.rd || !this.ctx) return { ok: false, rule: 'no_state', detail: 'brain has no sensor data yet' };
    const res = validateToolCall(call, {
      b: this.b, rd: this.rd, P: this.pred.p, home: this.home!, probes: this.probesPlaced(this.rd),
      plan: (x, y) => astar(this.b, this.rd!.x, this.rd!.y, x, y, false),
    });
    if (res.ok) this.goal = res.goal!;
    return res;
  }

  /** for the UI */
  info() {
    return {
      behaviour: this.behaviour, reflex: this.reflex?.text ?? null, scores: this.scores, confidence: this.confidence,
      goal: this.goal ? this.goal.text : 'none', goalPt: this.goal && this.goal.call.name !== 'hold_position' && this.goal.call.name !== 'retreat_to_high_ground' ? { x: this.goal.x, y: this.goal.y } : null,
      path: this.path?.path ?? null, view: this.b,
    };
  }
}
export { CH };
