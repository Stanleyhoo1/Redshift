import { Rng } from './rng';
import { World, N, CELL, HALF, Occ, Land } from './world';

export type DisasterKind = 'flood' | 'fire' | 'quake' | 'tornado' | 'meteor';
export const DISASTER_KINDS: DisasterKind[] = ['flood', 'fire', 'quake', 'tornado', 'meteor'];
export interface DisasterSetting { enabled: boolean; intensity: number }
export type DisasterConfig = Record<DisasterKind, DisasterSetting> & { random: boolean };

export const defaultDisasterConfig = (): DisasterConfig => ({
  flood: { enabled: false, intensity: 0.6 }, fire: { enabled: false, intensity: 0.6 }, quake: { enabled: false, intensity: 0.6 },
  tornado: { enabled: false, intensity: 0.6 }, meteor: { enabled: false, intensity: 0.6 }, random: false,
});

export interface SimEvent { t: number; kind: DisasterKind | 'sim'; text: string }
export interface Meteor { x: number; y: number; tImpact: number; r: number; done: boolean }
export interface Flash { x: number; y: number; t: number; r: number }
export interface Tornado { x: number; y: number; vx: number; vy: number; R: number; vmax: number; t0: number; life: number; active: boolean; trail: { x: number; y: number }[] }
export interface Debris { x: number; y: number; z: number; vx: number; vy: number; vz: number }

const AMBIENT = 22;
const P0 = 101325;

/**
 * Ground-truth disaster engine. Everything here is hidden from the brain;
 * the brain only ever receives what sensors.ts derives from it.
 */
export class Disasters {
  cfg: DisasterConfig;
  events: SimEvent[] = [];
  /** ambient wind (m/s), slowly varying, sometimes shifts direction */
  windDir: number; windSpeed: number; windShiftAt: number;
  flood = { active: false, t0: 0, rate: 0, maxStage: 0, acc: 0 };
  fireActive = false; fireAcc = 0; fireT0 = 0;
  /** time when each burning cell ignited */
  burnStart = new Float32Array(N * N);
  burnOut = new Float32Array(N * N);
  quake = { active: false, t0: 0, dur: 0, amp: 0, aftershocks: [] as number[], crackBudget: 0 };
  tornadoes: Tornado[] = [];
  debris: Debris[] = [];
  meteor = { active: false, t0: 0, dur: 0, cx: 0, cy: 0, vx: 0, vy: 0, spread: 0, rate: 0, acc: 0 };
  meteors: Meteor[] = [];
  flashes: Flash[] = [];
  /** scheduled start times (s) for each enabled disaster */
  schedule: { kind: DisasterKind; t: number }[] = [];
  private rng: Rng;
  private smokeTmp = new Float32Array(N * N);
  private floodFront: number[] = [];
  private wet = new Uint8Array(N * N);

  constructor(private world: World, rng: Rng, cfg: DisasterConfig, duration = 180) {
    this.rng = rng;
    this.cfg = cfg;
    this.windDir = rng.range(-Math.PI, Math.PI);
    this.windSpeed = rng.range(3, 7);
    this.windShiftAt = rng.range(70, 110);
    const enabled = (Object.keys(cfg) as (keyof DisasterConfig)[]).filter(k => k !== 'random' && (cfg[k] as DisasterSetting).enabled) as DisasterKind[];
    if (cfg.random) {
      for (const k of enabled) this.schedule.push({ kind: k, t: rng.range(8, Math.max(20, duration * 0.55)) });
      // quakes and meteor showers may recur in random mode
      for (const k of enabled) if ((k === 'quake' || k === 'meteor' || k === 'tornado') && rng.next() < 0.5) this.schedule.push({ kind: k, t: rng.range(duration * 0.5, duration * 0.85) });
    } else {
      enabled.forEach((k, i) => this.schedule.push({ kind: k, t: 12 + i * 22 }));
    }
    this.schedule.sort((a, b) => a.t - b.t);
    for (let i = 0; i < N * N; i++) if (world.river[i]) this.wet[i] = 1;
  }

  log(t: number, kind: SimEvent['kind'], text: string) { this.events.push({ t, kind, text }); if (this.events.length > 200) this.events.shift(); }
  intensity(k: DisasterKind) { return Math.max(0.05, Math.min(1, this.cfg[k].intensity)); }

  get windX() { return this.windSpeed * Math.cos(this.windDir); }
  get windY() { return this.windSpeed * Math.sin(this.windDir); }

  /** Start a disaster now (scheduler, Random mode, or a chaos-panel button). */
  trigger(kind: DisasterKind, t: number, roverX: number, roverY: number) {
    const I = this.intensity(kind), rng = this.rng, w = this.world;
    if (kind === 'flood') {
      if (!this.flood.active) { this.flood = { active: true, t0: t, rate: 0.006 + 0.018 * I, maxStage: 1.2 + 3.2 * I, acc: 0 }; this.log(t, kind, 'Flood: river is rising'); }
      else { this.flood.rate *= 1.6; this.flood.maxStage += 1; this.log(t, kind, 'Flood surge: river rising faster'); }
    } else if (kind === 'fire') {
      // ignite a fuel-rich cell some distance from the rover
      for (let k = 0; k < 400; k++) {
        const ang = rng.range(-Math.PI, Math.PI), d = rng.range(35, 75);
        const x = roverX + d * Math.cos(ang), y = roverY + d * Math.sin(ang);
        if (!w.inBounds(x, y)) continue;
        const i = w.cellIndexAt(x, y);
        if (w.fuel[i] < (k < 200 ? 0.9 : 0.4) || w.burnt[i] || w.burning[i] > 0) continue;
        this.ignite(i, t);
        for (const n of this.neighbors(i)) if (w.fuel[n] > 0.2) this.ignite(n, t);
        if (!this.fireActive) this.fireT0 = t;
        this.fireActive = true;
        this.log(t, kind, `Wildfire ignited ${Math.round(d)} m from rover, wind ${this.windSpeed.toFixed(1)} m/s`);
        break;
      }
    } else if (kind === 'quake') {
      this.quake = { active: true, t0: t, dur: 8 + 10 * I, amp: 2 + 6 * I, aftershocks: [t + rng.range(30, 50)], crackBudget: 2 + Math.round(4 * I) };
      this.log(t, kind, `Earthquake M${(5 + 2.5 * I).toFixed(1)}`);
    } else if (kind === 'tornado') {
      // spawn at a map edge region and head roughly across the map, passing near-ish the rover
      // touch down 60-90 m away, inside the map (clamping a far point to the edge could put it right next to the rover)
      let x = 0, y = 0;
      for (let k = 0; k < 60; k++) {
        const ang = rng.range(-Math.PI, Math.PI), d = rng.range(60, 90);
        x = roverX + d * Math.cos(ang); y = roverY + d * Math.sin(ang);
        if (Math.abs(x) < 92 && Math.abs(y) < 92) break;
      }
      const aim = Math.atan2(roverY - y, roverX - x) + rng.range(-0.35, 0.35);
      const sp = 2.2 + 2 * I;
      this.tornadoes.push({ x, y, vx: sp * Math.cos(aim), vy: sp * Math.sin(aim), R: 5 + 3 * I, vmax: 40 + 30 * I, t0: t, life: 70 + 30 * I, active: true, trail: [] });
      this.log(t, kind, 'Tornado touched down');
    } else if (kind === 'meteor') {
      const ang = rng.range(-Math.PI, Math.PI), d = rng.range(20, 60);
      this.meteor = { active: true, t0: t, dur: 45 + 30 * I, cx: roverX + d * Math.cos(ang), cy: roverY + d * Math.sin(ang), vx: rng.range(-1.5, 1.5), vy: rng.range(-1.5, 1.5), spread: 22, rate: 0.4 + 1.3 * I, acc: 0 };
      this.log(t, kind, 'Meteor shower incoming');
    }
  }

  private neighbors(i: number): number[] {
    const cx = i % N, cy = (i / N) | 0, out: number[] = [];
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue; const x = cx + dx, y = cy + dy;
      if (x >= 0 && y >= 0 && x < N && y < N) out.push(y * N + x);
    }
    return out;
  }
  private ignite(i: number, t: number) {
    const w = this.world;
    if (w.burning[i] > 0 || w.burnt[i] || w.water[i] > 0.05 || w.fuel[i] <= 0) return;
    w.burning[i] = 0.2; this.burnStart[i] = t;
  }

  step(dt: number, t: number, roverX: number, roverY: number, hit: (dmg: number, cause: string, fatal?: boolean) => void) {
    const w = this.world, rng = this.rng;
    while (this.schedule.length && this.schedule[0].t <= t) this.trigger(this.schedule.shift()!.kind, t, roverX, roverY);

    // ambient wind drifts, and shifts once (adaptation test for fire prediction)
    this.windDir += rng.gauss(0.01) * dt;
    if (t >= this.windShiftAt) { this.windDir += rng.range(0.9, 1.8) * (rng.next() < 0.5 ? -1 : 1); this.windShiftAt = 1e9; if (this.fireActive) this.log(t, 'fire', 'Wind shifted direction'); }

    // ---------------- flood ----------------
    if (this.flood.active) {
      w.stage = Math.min(this.flood.maxStage, w.stage + this.flood.rate * dt);
      this.flood.acc += dt;
      if (this.flood.acc >= 0.8) { this.flood.acc = 0; this.spreadFlood(); }
    }

    // ---------------- fire ----------------
    if (this.fireActive) {
      this.fireAcc += dt;
      if (this.fireAcc >= 0.25) { this.stepFire(this.fireAcc, t); this.fireAcc = 0; }
    }

    // ---------------- quake ----------------
    const q = this.quake;
    if (q.active) {
      const e = t - q.t0;
      if (e < q.dur) {
        const env = Math.min(1, e / 2) * Math.max(0, 1 - Math.max(0, e - q.dur * 0.6) / (q.dur * 0.4));
        const I = this.intensity('quake');
        for (const b of w.buildings) {
          const before = b.collapse;
          if (b.collapse < 0.85 && rng.next() < dt * env * (0.04 + 0.12 * I) * (b.h / 8)) b.collapse = Math.min(0.9, b.collapse + rng.range(0.3, 0.6));
          if (b.collapse >= 0.3 && before < 0.3) {
            const r = Math.max(b.w, b.d) / 2 + 2.5;
            this.addRubble(b.x, b.y, r);
            const d = Math.hypot(roverX - b.x, roverY - b.y);
            if (d < r + 1.5) hit(55 + 40 * I, `crushed by a collapsing building`, b.collapse > 0.6);
            this.log(t, 'quake', `Building ${b.id} collapsed`);
          }
        }
        if (q.crackBudget > 0 && rng.next() < dt * 0.5) { this.addCrack(roverX, roverY); q.crackBudget--; }
      } else if (q.aftershocks.length && t >= q.aftershocks[0]) {
        q.aftershocks.shift(); q.t0 = t; q.dur = 5 + 4 * this.intensity('quake'); q.amp *= 0.6; q.crackBudget = 1;
        this.log(t, 'quake', 'Aftershock');
      } else if (!q.aftershocks.length) q.active = false;
    }

    // ---------------- tornado ----------------
    for (const tor of this.tornadoes) {
      if (!tor.active) continue;
      tor.vx += rng.gauss(0.4) * dt; tor.vy += rng.gauss(0.4) * dt;
      tor.x += tor.vx * dt; tor.y += tor.vy * dt;
      if (tor.trail.length === 0 || Math.hypot(tor.x - tor.trail[tor.trail.length - 1].x, tor.y - tor.trail[tor.trail.length - 1].y) > 2) tor.trail.push({ x: tor.x, y: tor.y });
      if (t - tor.t0 > tor.life || Math.abs(tor.x) > HALF + 10 || Math.abs(tor.y) > HALF + 10) { tor.active = false; this.log(t, 'tornado', 'Tornado dissipated'); continue; }
      // strip vegetation / leave debris along the path
      const i = w.cellIndexAt(tor.x, tor.y);
      if (w.inBounds(tor.x, tor.y) && rng.next() < dt * 2) { w.rubble[i] = Math.max(w.rubble[i], 0.4); w.fuel[i] *= 0.5; }
      for (const tr of w.trees) if (!tr.burnt && Math.hypot(tr.x - tor.x, tr.y - tor.y) < tor.R * 0.8) { tr.burnt = true; w.setOccDisk(tr.x, tr.y, tr.r + 0.1, Occ.Free); }
      // thrown debris
      if (rng.next() < dt * 3) {
        const a = rng.range(-Math.PI, Math.PI), r = tor.R * rng.range(0.8, 2);
        const px = tor.x + r * Math.cos(a), py = tor.y + r * Math.sin(a);
        const sp = rng.range(10, 20);
        this.debris.push({ x: px, y: py, z: rng.range(3, 12), vx: -Math.sin(a) * sp + tor.vx, vy: Math.cos(a) * sp + tor.vy, vz: rng.range(-2, 4) });
      }
    }
    for (const d of this.debris) {
      d.vz -= 9.81 * dt; d.x += d.vx * dt; d.y += d.vy * dt; d.z += d.vz * dt;
      if (Math.hypot(d.x - roverX, d.y - roverY) < 1.4 && d.z < 1.5) { hit(18 + 14 * this.intensity('tornado'), 'struck by tornado debris'); d.z = -1; }
    }
    this.debris = this.debris.filter(d => d.z > 0 && w.inBounds(d.x, d.y));

    // ---------------- meteors ----------------
    const m = this.meteor;
    if (m.active) {
      if (t - m.t0 > m.dur) { m.active = false; this.log(t, 'meteor', 'Meteor shower ended'); }
      else {
        m.cx += m.vx * dt; m.cy += m.vy * dt;
        m.acc += dt * m.rate;
        while (m.acc >= 1) {
          m.acc -= 1;
          const x = m.cx + rng.gauss(m.spread), y = m.cy + rng.gauss(m.spread);
          if (w.inBounds(x, y)) this.meteors.push({ x, y, tImpact: t + rng.range(2.5, 3.5), r: 2.5 + 2 * this.intensity('meteor'), done: false });
        }
      }
    }
    for (const mt of this.meteors) {
      if (mt.done || t < mt.tImpact) continue;
      mt.done = true;
      this.flashes.push({ x: mt.x, y: mt.y, t, r: mt.r });
      this.addCrater(mt.x, mt.y, mt.r * 0.7, t);
      const d = Math.hypot(roverX - mt.x, roverY - mt.y);
      if (d < mt.r) hit(200, 'hit by a meteor', true);
      else if (d < mt.r * 2.2) hit(30 * (1 - (d - mt.r) / (mt.r * 1.2)) + 5, 'meteor blast');
      const ci = w.cellIndexAt(mt.x, mt.y);
      if (w.fuel[ci] > 0.6 && rng.next() < 0.1) { this.ignite(ci, t); if (!this.fireActive) { this.fireActive = true; this.fireT0 = t; this.log(t, 'fire', 'Meteor impact started a fire'); } }
    }
    this.meteors = this.meteors.filter(mt => !mt.done);
    this.flashes = this.flashes.filter(f => t - f.t < 1.0);
  }

  private spreadFlood() {
    const w = this.world;
    // grow one ring of cells per update from currently wet cells (water flows, it doesn't teleport)
    const next: number[] = [];
    for (let i = 0; i < N * N; i++) {
      if (!this.wet[i]) continue;
      const cx = i % N, cy = (i / N) | 0;
      const nb = [cx > 0 ? i - 1 : -1, cx < N - 1 ? i + 1 : -1, cy > 0 ? i - N : -1, cy < N - 1 ? i + N : -1];
      for (const n of nb) if (n >= 0 && !this.wet[n] && w.height[n] < w.stage) next.push(n);
    }
    for (const n of next) this.wet[n] = 1;
    for (let i = 0; i < N * N; i++) {
      if (!this.wet[i]) continue;
      const d = Math.max(0, w.stage - w.height[i]);
      w.water[i] = Math.max(w.water[i], d);
      if (d > 0.05) { if (w.burning[i] > 0) { w.burning[i] = 0; w.burnt[i] = 1; } w.fuel[i] = 0; }
    }
  }

  private stepFire(dt: number, t: number) {
    const w = this.world, rng = this.rng;
    const I = this.intensity('fire');
    const wx = this.windX, wy = this.windY, ws = this.windSpeed;
    const base = 0.05 + 0.22 * I;
    const newly: number[] = [];
    let anyBurning = false;
    for (let i = 0; i < N * N; i++) {
      const b = w.burning[i];
      if (b <= 0) continue;
      anyBurning = true;
      const age = t - this.burnStart[i];
      const life = 14 + 22 * w.fuel[i];
      w.burning[i] = age < 4 ? Math.min(1, 0.2 + age / 4) : age > life ? Math.max(0, 1 - (age - life) / 6) : 1;
      if (age > life + 6) { w.burning[i] = 0; w.burnt[i] = 1; w.fuel[i] = 0; this.burnOut[i] = t; continue; }
      const cx = i % N, cy = (i / N) | 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const x = cx + dx, y = cy + dy; if (x < 0 || y < 0 || x >= N || y >= N) continue;
        const n = y * N + x;
        if (w.fuel[n] <= 0 || w.burning[n] > 0 || w.burnt[n] || w.water[n] > 0.05) continue;
        const dl = Math.hypot(dx, dy);
        const align = (dx * wx + dy * wy) / (dl * Math.max(ws, 0.1));
        const windF = Math.exp(1.1 * align * ws / 5);
        // fire climbs slopes faster
        const up = (w.height[n] - w.height[i]) / (CELL * dl);
        const slopeF = Math.exp(Math.max(-1, Math.min(1.5, up * 2.5)));
        const p = base * w.fuel[n] * windF * slopeF * w.burning[i] * dt / dl;
        if (rng.next() < p) newly.push(n);
      }
    }
    for (const n of newly) this.ignite(n, t);
    // trees in burning cells char; trunks remain as obstacles
    for (const tr of w.trees) if (!tr.burnt && w.burning[w.cellIndexAt(tr.x, tr.y)] > 0.8) tr.burnt = true;
    // smoke: emit, advect downwind (semi-Lagrangian), diffuse, decay
    const sm = w.smoke, tmp = this.smokeTmp;
    const sx = wx * dt / CELL, sy = wy * dt / CELL;
    for (let cy = 0; cy < N; cy++) for (let cx = 0; cx < N; cx++) {
      const fx = Math.max(0, Math.min(N - 1.001, cx - sx)), fy = Math.max(0, Math.min(N - 1.001, cy - sy));
      const ix = Math.floor(fx), iy = Math.floor(fy), ax = fx - ix, ay = fy - iy;
      const v = sm[iy * N + ix] * (1 - ax) * (1 - ay) + sm[iy * N + ix + 1] * ax * (1 - ay) + sm[(iy + 1) * N + ix] * (1 - ax) * ay + sm[(iy + 1) * N + ix + 1] * ax * ay;
      const i = cy * N + cx;
      tmp[i] = Math.min(1, v * Math.pow(0.93, dt) + w.burning[i] * 0.5 * dt);
    }
    sm.set(tmp);
    if (!anyBurning && !newly.length) { this.fireActive = false; }
  }

  private addRubble(x: number, y: number, r: number) {
    const w = this.world;
    for (let k = 0; k < 14; k++) {
      const a = this.rng.range(-Math.PI, Math.PI), d = this.rng.range(0, r);
      w.setOccDisk(x + d * Math.cos(a), y + d * Math.sin(a), this.rng.range(0.6, 1.3), Occ.Rubble, true);
    }
    for (let dy = -r; dy <= r; dy += CELL) for (let dx = -r; dx <= r; dx += CELL) {
      if (dx * dx + dy * dy > r * r || !w.inBounds(x + dx, y + dy)) continue;
      const i = w.cellIndexAt(x + dx, y + dy); w.rubble[i] = Math.min(1, w.rubble[i] + 0.7);
    }
  }
  private addCrack(rx: number, ry: number) {
    const w = this.world, rng = this.rng;
    for (let k = 0; k < 30; k++) {
      const x0 = rng.range(-90, 90), y0 = rng.range(-90, 90), a = rng.range(0, Math.PI), L = rng.range(15, 40);
      const x1 = x0 + L * Math.cos(a), y1 = y0 + L * Math.sin(a);
      // not right under the rover (keep it fair), but anywhere else
      const t = Math.max(0, Math.min(1, ((rx - x0) * (x1 - x0) + (ry - y0) * (y1 - y0)) / (L * L)));
      if (Math.hypot(rx - (x0 + t * (x1 - x0)), ry - (y0 + t * (y1 - y0))) < 5) continue;
      const wd = rng.range(0.6, 1.4);
      for (let s = 0; s <= L; s += 0.4) {
        const jitter = Math.sin(s * 0.7) * 0.8;
        w.setOccDisk(x0 + s * Math.cos(a) - jitter * Math.sin(a), y0 + s * Math.sin(a) + jitter * Math.cos(a), wd / 2, Occ.Crack);
      }
      w.cracks.push({ x0, y0, x1, y1, w: wd });
      this.log(this.quake.t0, 'quake', `Ground crack opened (${L.toFixed(0)} m)`);
      return;
    }
  }
  private addCrater(x: number, y: number, r: number, t: number) {
    const w = this.world;
    w.craters.push({ x, y, r, t });
    w.setOccDisk(x, y, r, Occ.Crater);
    for (let k = 0; k < 5; k++) { const a = this.rng.range(-Math.PI, Math.PI); w.setOccDisk(x + (r + 0.8) * Math.cos(a), y + (r + 0.8) * Math.sin(a), 0.6, Occ.Rock, true); }
    for (let dy = -r - 2; dy <= r + 2; dy += CELL) for (let dx = -r - 2; dx <= r + 2; dx += CELL) {
      const d = Math.hypot(dx, dy); if (d > r + 2 || !w.inBounds(x + dx, y + dy)) continue;
      const i = w.cellIndexAt(x + dx, y + dy);
      w.height[i] -= d < r ? 1.2 * (1 - d / r) : 0;
    }
    void Land;
  }

  // ---------------- fields queried by sensors and rover ----------------
  /** apparent surface temperature of a cell (what a thermal camera sees) */
  cellTemp(i: number, t: number): number {
    const w = this.world;
    if (w.burning[i] > 0) return AMBIENT + 780 * w.burning[i];
    if (w.burnt[i] && this.burnOut[i] > 0) return AMBIENT + 90 * Math.exp(-(t - this.burnOut[i]) / 25);
    return AMBIENT;
  }
  /** air temperature at a point (what the rover body experiences) */
  airTemp(x: number, y: number): number {
    const w = this.world;
    if (!this.fireActive) return AMBIENT;
    const [cx, cy] = World.cellOf(x, y);
    let s = 0;
    for (let dy = -7; dy <= 7; dy++) for (let dx = -7; dx <= 7; dx++) {
      const X = cx + dx, Y = cy + dy; if (X < 0 || Y < 0 || X >= N || Y >= N) continue;
      const b = w.burning[Y * N + X]; if (b <= 0) continue;
      const [px, py] = World.cellCenter(X, Y);
      const d = Math.hypot(px - x, py - y);
      s += 95 * b * Math.exp(-d / 2.6);
    }
    return AMBIENT + Math.min(900, s);
  }
  windAt(x: number, y: number): [number, number] {
    let wx = this.windX, wy = this.windY;
    for (const tor of this.tornadoes) {
      if (!tor.active) continue;
      const dx = x - tor.x, dy = y - tor.y, r = Math.max(0.5, Math.hypot(dx, dy));
      const vt = r < tor.R ? tor.vmax * r / tor.R : tor.vmax * tor.R / r;
      // counter-clockwise rotation + slight inflow
      wx += vt * (-dy / r) - 0.25 * vt * dx / r + tor.vx * Math.min(1, tor.R * 2 / r);
      wy += vt * (dx / r) - 0.25 * vt * dy / r + tor.vy * Math.min(1, tor.R * 2 / r);
    }
    return [wx, wy];
  }
  pressureAt(x: number, y: number): number {
    let p = P0;
    for (const tor of this.tornadoes) {
      if (!tor.active) continue;
      const r = Math.max(0.5, Math.hypot(x - tor.x, y - tor.y));
      const k = r < tor.R ? 2 - (r / tor.R) ** 2 : (tor.R / r) ** 2;
      p -= 0.6 * tor.vmax * tor.vmax * k;
    }
    return p;
  }
  shakeAmp(t: number): number {
    const q = this.quake;
    if (!q.active) return 0;
    const e = t - q.t0;
    if (e < 0 || e > q.dur) return 0;
    return q.amp * Math.min(1, e / 2) * Math.max(0, 1 - Math.max(0, e - q.dur * 0.6) / (q.dur * 0.4));
  }
  anyActive(): boolean {
    return this.flood.active || this.fireActive || this.quake.active || this.tornadoes.some(t => t.active) || this.meteor.active || this.meteors.length > 0;
  }
}
