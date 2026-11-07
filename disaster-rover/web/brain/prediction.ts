import { CFG, CH, clamp01, compass } from './config';
import { Belief } from './grid';
import type { Readings, PerceptionState } from './perception';

interface TP { t: number; x: number; y: number; w?: number }

/** least-squares slope of v over t */
function slope(ts: number[], vs: number[]): number {
  const n = ts.length; if (n < 2) return 0;
  let mt = 0, mv = 0; for (let i = 0; i < n; i++) { mt += ts[i]; mv += vs[i]; } mt /= n; mv /= n;
  let num = 0, den = 0; for (let i = 0; i < n; i++) { num += (ts[i] - mt) * (vs[i] - mv); den += (ts[i] - mt) ** 2; }
  return den > 1e-6 ? num / den : 0;
}

export interface Predictions {
  fire: { active: boolean; vx: number; vy: number; speed: number; growth: number; nearest: number; eta: number; bearing: number; text: string };
  flood: { active: boolean; stage: number; rate: number; predStage: number; text: string };
  tornado: { active: boolean; x: number; y: number; vx: number; vy: number; R: number; dist: number; cpa: number; tcpa: number; text: string };
  meteor: { active: boolean; cx: number; cy: number; vx: number; vy: number; spread: number; count: number; text: string };
  quake: { active: boolean; recent: boolean; level: number; text: string };
}

const emptyPred = (): Predictions => ({
  fire: { active: false, vx: 0, vy: 0, speed: 0, growth: 0, nearest: 999, eta: 999, bearing: 0, text: '' },
  flood: { active: false, stage: NaN, rate: 0, predStage: NaN, text: '' },
  tornado: { active: false, x: 0, y: 0, vx: 0, vy: 0, R: 6, dist: 999, cpa: 999, tcpa: 999, text: '' },
  meteor: { active: false, cx: 0, cy: 0, vx: 0, vy: 0, spread: 15, count: 0, text: '' },
  quake: { active: false, recent: false, level: 0, text: '' },
});

/**
 * PREDICTION: estimates each hazard's spread direction and speed from the rover's own sensor history,
 * and paints a hazard map for `CFG.horizon` seconds ahead.
 */
export class Predictor {
  p: Predictions = emptyPred();
  fireHist: { t: number; x: number; y: number; r: number }[] = [];
  stageHist: { t: number; s: number }[] = [];
  torHist: TP[] = [];
  impacts: (TP & { r: number })[] = [];
  lastShakeT = -1e9; shakeLevel = 0;
  /** hazard types ever sensed (for "new hazard type" agent events) */
  sensed = new Set<string>();
  newlySensed: string[] = [];
  private tmp = new Float32Array(CFG.G * CFG.G);

  /** fast per-frame observations */
  observe(rd: Readings, ps: PerceptionState) {
    const t = rd.t;
    if (!isNaN(rd.stageObs)) { this.stageHist.push({ t, s: rd.stageObs }); while (this.stageHist.length && this.stageHist[0].t < t - 40) this.stageHist.shift(); }
    // tornado: funnel seen by camera, or inferred from wind + pressure (center is 90° left of the residual wind for a CCW vortex)
    if (rd.funnel) this.torHist.push({ t, x: rd.funnel.x, y: rd.funnel.y, w: 1 });
    else if (rd.windSpeed > CFG.tornadoWind || rd.pressureDrop > CFG.tornadoPressure) {
      const rx = rd.windX - ps.ambX, ry = rd.windY - ps.ambY;
      const a = Math.atan2(ry, rx) + Math.PI / 2;
      const r = Math.max(8, Math.min(60, 6 * Math.sqrt(0.6 * 55 * 55 / Math.max(10, rd.pressureDrop))));
      this.torHist.push({ t, x: rd.x + r * Math.cos(a), y: rd.y + r * Math.sin(a), w: 0.3 });
    }
    while (this.torHist.length && this.torHist[0].t < t - 12) this.torHist.shift();
    for (const f of rd.flashes) {
      if (this.impacts.some(i => Math.abs(i.t - t) < 0.4 && Math.hypot(i.x - f.x, i.y - f.y) < 4)) continue;
      this.impacts.push({ t, x: f.x, y: f.y, r: f.size });
    }
    while (this.impacts.length && this.impacts[0].t < t - 30) this.impacts.shift();
    if (rd.shake > CFG.shakeThresh) { this.lastShakeT = t; this.shakeLevel = rd.shake; }
    else this.shakeLevel *= 0.9;
  }

  /** Recompute risk channels, predicted map and hazard distance field. */
  mapUpdate(b: Belief, rd: Readings, ps: PerceptionState) {
    const t = rd.t, G = CFG.G, n = G * G, T = CFG.horizon;
    b.updateConf(t);
    const [cw, ch, cr, cwi, cs] = [b.channels[CH.water], b.channels[CH.heat], b.channels[CH.rubble], b.channels[CH.wind], b.channels[CH.slope]];
    const pred = b.predicted, tmp = this.tmp;
    cw.fill(0); ch.fill(0); cr.fill(0); cwi.fill(0); cs.fill(0); pred.fill(0); tmp.fill(0); b.predWater.fill(0);
    this.newlySensed = [];
    const mark = (k: string) => { if (!this.sensed.has(k)) { this.sensed.add(k); this.newlySensed.push(k); } };

    // ---- water now ----
    for (let i = 0; i < n; i++) if (b.water[i] > 0) cw[i] = clamp01(b.water[i] / 0.35 + 0.15);

    // ---- heat now (observed temperature, aged, then dilated: radiant heat reaches ~8 m) ----
    const hot: number[] = [];
    for (let i = 0; i < n; i++) {
      const h = b.heat[i]; if (isNaN(h) || h <= 60) continue;
      const age = t - b.heatT[i];
      const Te = CFG.ambient + (h - CFG.ambient) * Math.exp(-Math.max(0, age - 15) / 40);
      const r0 = clamp01((Te - 60) / 250);
      if (r0 > tmp[i]) tmp[i] = r0;
      if (Te > CFG.hotTemp && age < 25) hot.push(i);
    }
    for (let i = 0; i < n; i++) {
      const r0 = tmp[i]; if (r0 < 0.08) continue;
      paint(ch, i, 4, (d) => r0 * (1 - d / 5));
    }
    if (hot.length) mark('fire');

    // ---- rubble: quake debris near structures, recent impacts ----
    const sinceShake = t - this.lastShakeT;
    if (sinceShake < 60) {
      mark('earthquake');
      const q = sinceShake < 2 ? 0.9 : 0.55 * (1 - sinceShake / 60) + 0.15;
      for (let i = 0; i < n; i++) if (b.struct[i] && b.obst[i] > 0.3) paint(cr, i, 3, d => q * (1 - d / 4));
    }
    for (const im of this.impacts) {
      const age = t - im.t, v = 0.85 * (1 - age / 30);
      paint(cr, b.idx(im.x, im.y), Math.ceil((im.r + 4) / CFG.RES), d => v * (1 - d / ((im.r + 6) / CFG.RES)));
    }
    if (this.impacts.length) mark('meteor');

    // ---- slope from elevation ----
    tmp.fill(0);
    for (let i = 0; i < n; i++) {
      const e0 = b.elev[i]; if (isNaN(e0) || b.struct[i]) continue;
      const x = i % G, y = (i / G) | 0; let m = 0;
      if (x + 1 < G && !isNaN(b.elev[i + 1]) && !b.struct[i + 1]) m = Math.max(m, Math.abs(b.elev[i + 1] - e0));
      if (x > 0 && !isNaN(b.elev[i - 1]) && !b.struct[i - 1]) m = Math.max(m, Math.abs(b.elev[i - 1] - e0));
      if (y + 1 < G && !isNaN(b.elev[i + G]) && !b.struct[i + G]) m = Math.max(m, Math.abs(b.elev[i + G] - e0));
      if (y > 0 && !isNaN(b.elev[i - G]) && !b.struct[i - G]) m = Math.max(m, Math.abs(b.elev[i - G] - e0));
      const deg = Math.atan(m / CFG.RES) * 180 / Math.PI;
      tmp[i] = clamp01((deg - 15) / 15);
    }
    // conservative: a cell is as steep as its steepest neighbour (2 m cells hide sub-cell relief)
    for (let i = 0; i < n; i++) {
      const x = i % G, y = (i / G) | 0; let m = tmp[i];
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy; if (X >= 0 && Y >= 0 && X < G && Y < G && tmp[Y * G + X] > m) m = tmp[Y * G + X]; }
      cs[i] = m;
    }

    // ================= PREDICTIONS =================
    const P = this.p;
    // ---- fire: front velocity from the history of the hot region (centroid drift + growth), wind as prior ----
    P.fire.active = hot.length > 0;
    if (hot.length) {
      let sx = 0, sy = 0; for (const i of hot) { sx += b.cx(i); sy += b.cy(i); }
      const cx = sx / hot.length, cy = sy / hot.length, rad = Math.sqrt(hot.length * CFG.RES * CFG.RES / Math.PI);
      this.fireHist.push({ t, x: cx, y: cy, r: rad });
      while (this.fireHist.length && this.fireHist[0].t < t - 20) this.fireHist.shift();
      const ts = this.fireHist.map(h => h.t);
      const span = ts.length ? ts[ts.length - 1] - ts[0] : 0;
      const growth = Math.max(0, Math.min(1.5, slope(ts, this.fireHist.map(h => h.r))));
      const ambS = Math.hypot(ps.ambX, ps.ambY) || 1;
      const wdx = ps.ambX / ambS, wdy = ps.ambY / ambS;
      let vx = wdx * 0.35, vy = wdy * 0.35;
      if (span > 6) {
        const mvx = slope(ts, this.fireHist.map(h => h.x)), mvy = slope(ts, this.fireHist.map(h => h.y));
        const sp = Math.max(growth, 0.15);
        vx = 0.55 * mvx + 0.45 * wdx * sp; vy = 0.55 * mvy + 0.45 * wdy * sp;
      }
      const sp = Math.hypot(vx, vy); if (sp > 2) { vx *= 2 / sp; vy *= 2 / sp; }
      // paint the front swept 30 s ahead: front cells = hot cells with a non-hot neighbour
      const isHot = new Uint8Array(n); for (const i of hot) isHot[i] = 1;
      let front = hot.filter(i => { const x = i % G, y = (i / G) | 0; return (x > 0 && !isHot[i - 1]) || (x < G - 1 && !isHot[i + 1]) || (y > 0 && !isHot[i - G]) || (y < G - 1 && !isHot[i + G]); });
      if (front.length > 160) { const s = front.length / 160; front = front.filter((_, k) => Math.floor(k % s) === 0); }
      const rr = Math.min(7, Math.ceil((3 + growth * T * 0.4) / CFG.RES));
      let nearest = 999, nearestI = -1;
      for (const i of hot) { const d = Math.hypot(b.cx(i) - rd.x, b.cy(i) - rd.y); if (d < nearest) { nearest = d; nearestI = i; } }
      for (const i of front) for (let k = 0; k <= 3; k++) {
        const px = b.cx(i) + vx * T * k / 3, py = b.cy(i) + vy * T * k / 3; if (!b.inside(px, py)) continue;
        paint(pred, b.idx(px, py), rr, d => 0.9 * (1 - d / (rr + 1)));
      }
      // ETA of the front to the rover: closing speed along the line rover<-fire
      const bx = nearestI >= 0 ? rd.x - b.cx(nearestI) : 0, by = nearestI >= 0 ? rd.y - b.cy(nearestI) : 0;
      const closing = nearest > 0 ? (vx * bx + vy * by) / Math.max(nearest, 1) + growth * 0.5 : 0;
      const eta = closing > 0.02 ? Math.max(0, nearest - 6) / closing : 999;
      const bearing = nearestI >= 0 ? Math.atan2(b.cy(nearestI) - rd.y, b.cx(nearestI) - rd.x) : 0;
      Object.assign(P.fire, { vx, vy, speed: Math.hypot(vx, vy), growth, nearest, eta, bearing });
      P.fire.text = `fire front ${nearest.toFixed(0)} m ${compass(bearing)} moving ${compass(Math.atan2(vy, vx))} ${Math.hypot(vx, vy).toFixed(2)} m/s${eta < 120 ? `, reaches rover in ${eta.toFixed(0)} s` : ''}`;
    } else { this.fireHist = []; P.fire.text = ''; }

    // ---- flood: stage (water surface elevation) and its rise rate from the camera + depth camera ----
    if (this.stageHist.length >= 3) {
      const recent = this.stageHist.slice(-10).map(s => s.s).sort((a, c) => a - c);
      const stage = recent[recent.length >> 1];
      const rate = slope(this.stageHist.map(s => s.t), this.stageHist.map(s => s.s));
      const span = this.stageHist[this.stageHist.length - 1].t - this.stageHist[0].t;
      const old = this.stageHist.filter(h => h.t < this.stageHist[0].t + 6).map(h => h.s).sort((a, c) => a - c);
      const rise = stage - old[old.length >> 1];
      const rising = rate > 0.004 && span > 10 && rise > 0.1;
      const predStage = stage + (rising ? rate * T + 0.25 : 0.1);
      Object.assign(P.flood, { active: rising, stage, rate, predStage });
      if (rising) mark('flood');
      // flood fill from known water through dry ground lower than the predicted stage
      const q: number[] = []; const seen = new Uint8Array(n);
      for (let i = 0; i < n; i++) if (b.water[i] > 0.05) { q.push(i); seen[i] = 1; }
      for (let h = 0; h < q.length; h++) {
        const i = q[h], x = i % G, y = (i / G) | 0;
        for (const j of [x > 0 ? i - 1 : -1, x < G - 1 ? i + 1 : -1, y > 0 ? i - G : -1, y < G - 1 ? i + G : -1]) {
          if (j < 0 || seen[j]) continue;
          const e = !isNaN(b.dryElev[j]) ? b.dryElev[j] : (!isNaN(b.elev[j]) && !b.struct[j] ? b.elev[j] : NaN);
          if (isNaN(e) || e >= predStage) continue;
          seen[j] = 1; q.push(j);
          const v = clamp01((predStage - e) / 0.35 + 0.2);
          if (v > pred[j]) pred[j] = v;
          if (v > b.predWater[j]) b.predWater[j] = v;
        }
      }
      const above = (isNaN(b.elev[b.idx(rd.x, rd.y)]) ? NaN : b.elev[b.idx(rd.x, rd.y)]) - stage;
      P.flood.text = rising ? `water stage ${stage.toFixed(2)} m rising ${(rate * 100).toFixed(1)} cm/s (+${(rate * T).toFixed(2)} m in ${T} s)${!isNaN(above) ? `, rover ${above.toFixed(1)} m above water` : ''}` : '';
    }

    // ---- tornado: track from funnel sightings / wind inference ----
    const th = this.torHist;
    P.tornado.active = th.length > 0 && t - th[th.length - 1].t < 6;
    if (P.tornado.active) {
      mark('tornado');
      const good = th.filter(h => (h.w ?? 1) > 0.5).length >= 3 ? th.filter(h => (h.w ?? 1) > 0.5) : th;
      const last = good.slice(-5);
      const x = last.reduce((a, h) => a + h.x, 0) / last.length, y = last.reduce((a, h) => a + h.y, 0) / last.length;
      let vx = 0, vy = 0;
      if (good.length >= 4 && good[good.length - 1].t - good[0].t > 2) { vx = slope(good.map(h => h.t), good.map(h => h.x)); vy = slope(good.map(h => h.t), good.map(h => h.y)); }
      const sp = Math.hypot(vx, vy); if (sp > 7) { vx *= 7 / sp; vy *= 7 / sp; }
      const R = 7;
      const dist = Math.hypot(x - rd.x, y - rd.y);
      // closest point of approach to the (stationary) rover
      const rx = rd.x - x, ry = rd.y - y, vv = vx * vx + vy * vy;
      const tcpa = vv > 0.01 ? Math.max(0, (rx * vx + ry * vy) / vv) : 0;
      const cpa = Math.hypot(rx - vx * tcpa, ry - vy * tcpa);
      Object.assign(P.tornado, { x, y, vx, vy, R, dist, cpa, tcpa });
      const now = (px: number, py: number, v: number) => { if (b.inside(px, py)) paint(cwi, b.idx(px, py), Math.ceil(26 / CFG.RES), d => v * clamp01(1.15 - d * CFG.RES / 26)); };
      now(x, y, 1);
      for (let s = 0; s <= T; s += 3) { const px = x + vx * s, py = y + vy * s; if (b.inside(px, py)) paint(pred, b.idx(px, py), Math.ceil(24 / CFG.RES), d => 0.95 * clamp01(1.1 - d * CFG.RES / 24)); }
      P.tornado.text = `vortex ${dist.toFixed(0)} m ${compass(Math.atan2(y - rd.y, x - rd.x))} moving ${compass(Math.atan2(vy, vx))} ${Math.hypot(vx, vy).toFixed(1)} m/s, passes ${cpa.toFixed(0)} m from rover in ${tcpa.toFixed(0)} s`;
    } else P.tornado.text = '';

    // ---- meteors: impact zone centroid + drift from observed flashes ----
    const im = this.impacts.filter(i => t - i.t < 25);
    P.meteor.active = im.length >= 1;
    if (im.length >= 1) {
      let sw = 0, sx = 0, sy = 0;
      for (const i of im) { const w = 1 - (t - i.t) / 30; sw += w; sx += i.x * w; sy += i.y * w; }
      const cx = sx / sw, cy = sy / sw;
      let sp2 = 0; for (const i of im) sp2 += (i.x - cx) ** 2 + (i.y - cy) ** 2;
      const spread = Math.max(14, Math.min(45, Math.sqrt(sp2 / im.length) * 1.4));
      let vx = 0, vy = 0;
      if (im.length >= 5) { vx = Math.max(-2, Math.min(2, slope(im.map(i => i.t), im.map(i => i.x)))); vy = Math.max(-2, Math.min(2, slope(im.map(i => i.t), im.map(i => i.y)))); }
      Object.assign(P.meteor, { cx, cy, vx, vy, spread, count: im.length });
      const level = Math.min(0.75, 0.25 + im.length * 0.06);
      for (let s = 0; s <= 15; s += 5) { const px = cx + vx * s, py = cy + vy * s; if (b.inside(px, py)) paint(pred, b.idx(px, py), Math.ceil(spread / CFG.RES), d => level * clamp01(1.2 - d * CFG.RES / spread)); }
      const d = Math.hypot(cx - rd.x, cy - rd.y);
      P.meteor.text = `${im.length} impacts in 25 s, zone centre ${d.toFixed(0)} m ${compass(Math.atan2(cy - rd.y, cx - rd.x))} (radius ${spread.toFixed(0)} m)${Math.hypot(vx, vy) > 0.3 ? ` drifting ${compass(Math.atan2(vy, vx))}` : ''}`;
    } else P.meteor.text = '';

    // ---- quake ----
    P.quake.active = t - this.lastShakeT < 1.5; P.quake.recent = sinceShake < 60; P.quake.level = this.shakeLevel;
    P.quake.text = P.quake.active ? `ground shaking ${this.shakeLevel.toFixed(1)} m/s², aftershocks likely near structures` : P.quake.recent ? `aftershock risk near structures (${sinceShake.toFixed(0)} s since shaking)` : '';

    // ---- combine ----
    for (let i = 0; i < n; i++) {
      let r = cw[i]; if (ch[i] > r) r = ch[i]; if (cr[i] > r) r = cr[i]; if (cwi[i] > r) r = cwi[i];
      b.risk[i] = r;
      if (r > pred[i]) pred[i] = r;
    }
    // distance to the nearest dynamic hazard (heat / rubble / wind, current or predicted), chamfer transform
    const hd = b.hazDist;
    for (let i = 0; i < n; i++) hd[i] = (ch[i] > 0.4 || cr[i] > 0.4 || cwi[i] > 0.4 || (pred[i] > 0.45 && cw[i] < 0.05)) ? 0 : 1e6;
    chamfer(hd, G);
    for (let i = 0; i < n; i++) { if (cw[i] > b.predWater[i]) b.predWater[i] = cw[i]; b.waterDist[i] = b.water[i] > 0.05 ? 0 : 1e6; b.structDist[i] = b.struct[i] ? 0 : 1e6; }
    chamfer(b.waterDist, G); chamfer(b.structDist, G);
  }
}

/** paint a disk of radius R cells around cell i with value f(distance in cells), keeping the max */
export function paint(arr: Float32Array, i: number, R: number, f: (d: number) => number) {
  const G = CFG.G, x0 = i % G, y0 = (i / G) | 0;
  for (let dy = -R; dy <= R; dy++) {
    const y = y0 + dy; if (y < 0 || y >= G) continue;
    for (let dx = -R; dx <= R; dx++) {
      const x = x0 + dx; if (x < 0 || x >= G) continue;
      const d = Math.sqrt(dx * dx + dy * dy); if (d > R) continue;
      const v = f(d), k = y * G + x; if (v > arr[k]) arr[k] = v;
    }
  }
}

/** two-pass chamfer distance transform (cells -> metres), in place */
export function chamfer(d: Float32Array, G: number) {
  const a = CFG.RES, b = CFG.RES * 1.4142;
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
    const i = y * G + x; let v = d[i];
    if (x > 0 && d[i - 1] + a < v) v = d[i - 1] + a;
    if (y > 0) { if (d[i - G] + a < v) v = d[i - G] + a; if (x > 0 && d[i - G - 1] + b < v) v = d[i - G - 1] + b; if (x < G - 1 && d[i - G + 1] + b < v) v = d[i - G + 1] + b; }
    d[i] = v;
  }
  for (let y = G - 1; y >= 0; y--) for (let x = G - 1; x >= 0; x--) {
    const i = y * G + x; let v = d[i];
    if (x < G - 1 && d[i + 1] + a < v) v = d[i + 1] + a;
    if (y < G - 1) { if (d[i + G] + a < v) v = d[i + G] + a; if (x < G - 1 && d[i + G + 1] + b < v) v = d[i + G + 1] + b; if (x > 0 && d[i + G - 1] + b < v) v = d[i + G - 1] + b; }
    d[i] = v;
  }
}
