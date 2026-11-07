import { Rng } from './rng';
import { World, N, CELL, HALF } from './world';
import { Rover, ROVER } from './rover';
import { Disasters, DisasterConfig, DisasterKind, defaultDisasterConfig } from './disasters';
import { Sensors, SENSOR, Probe } from './sensors';
import type { BrainOutput, SensorFrame, Twist, ProbeReading } from '../brain/msgs';
import { zeroTwist } from '../brain/msgs';

export interface SimOptions { seed: number; disasters?: DisasterConfig; duration?: number; flat?: boolean }
export interface SimMetrics {
  survived: boolean; cause: string; t: number; health: number; battery: number;
  probesPlaced: number; probesAlive: number; surveyedPct: number;
  /** % of hazard-affected ground that ended up inside some probe's sensing radius */
  hazardCoveragePct: number;
}

export class Sim {
  static DT = 0.05;
  /** brain/sensor rate is every BRAIN_EVERY physics steps (10 Hz) */
  static BRAIN_EVERY = 2;
  readonly seed: number;
  readonly duration: number;
  world: World; rover: Rover; dis: Disasters; sensors: Sensors;
  t = 0; seq = 0; steps = 0;
  probes: Probe[] = [];
  probeReadings: ProbeReading[] = [];
  /** fog of war: cells ever within sensor range of the rover or a probe */
  revealed = new Uint8Array(N * N);
  /** cells ever touched by a hazard (for the coverage metric; ground truth) */
  affected = new Uint8Array(N * N);
  cmd: Twist = zeroTwist();
  lastFrame: SensorFrame | null = null;
  private rng: Rng; private sensorRng: Rng;
  private lastProbeSample = -1;

  constructor(opts: SimOptions) {
    this.seed = opts.seed;
    this.duration = opts.duration ?? 180;
    this.world = new World({ seed: opts.seed, flat: opts.flat });
    this.rng = new Rng(opts.seed * 7 + 1);
    this.sensorRng = new Rng(opts.seed * 13 + 5);
    this.dis = new Disasters(this.world, this.rng, opts.disasters ?? defaultDisasterConfig(), this.duration);
    const s = this.world.start;
    this.rover = new Rover(s.x, s.y, s.yaw);
    this.rover.z = this.world.heightAt(s.x, s.y);
    this.sensors = new Sensors(this.world, this.dis, this.sensorRng);
    this.reveal(s.x, s.y, SENSOR.revealRover);
  }

  get done() { return !this.rover.alive || this.t >= this.duration; }

  trigger(kind: DisasterKind) { this.dis.trigger(kind, this.t, this.rover.x, this.rover.y); }

  /** Apply a brain output: velocity command and optional probe drop. */
  apply(out: BrainOutput) {
    this.cmd = out.cmd_vel;
    if (out.drop_probe) this.dropProbe();
  }

  dropProbe(): boolean {
    const r = this.rover;
    if (!r.alive || r.probesLeft <= 0) return false;
    r.probesLeft--;
    const p: Probe = { id: this.probes.length, x: r.x, y: r.y, alive: true, t: this.t };
    this.probes.push(p);
    this.lastProbeSample = -1;
    this.dis.log(this.t, 'sim', `Probe ${p.id} placed at (${p.x.toFixed(0)}, ${p.y.toFixed(0)})`);
    return true;
  }

  /** Advance one physics step with the current command. */
  step() {
    if (this.done) return;
    const dt = Sim.DT, r = this.rover, w = this.world;
    this.dis.step(dt, this.t, r.x, r.y, (dmg, cause, fatal) => { if (fatal) r.kill(cause); else r.damage(dmg, cause); });
    const [wx, wy] = this.dis.windAt(r.x, r.y);
    const shake = this.dis.shakeAmp(this.t);
    r.step(dt, this.cmd, w, { windX: wx, windY: wy, tempC: this.dis.airTemp(r.x, r.y), shake, joltX: shake * this.rng.gauss(0.15), joltY: shake * this.rng.gauss(0.15) });
    // tornado flip check: wind beyond what the chassis can hold
    const ws = Math.hypot(wx, wy);
    if (ws > 46) r.kill(`flipped by tornado (${ws.toFixed(0)} m/s wind)`);
    // probes can be destroyed too
    for (const p of this.probes) {
      if (!p.alive) continue;
      const i = w.cellIndexAt(p.x, p.y);
      if (w.water[i] > 1.2) { p.alive = false; p.cause = 'submerged'; }
      else if (this.dis.airTemp(p.x, p.y) > 450) { p.alive = false; p.cause = 'burned'; }
      else if (Math.hypot(...this.dis.windAt(p.x, p.y)) > 55) { p.alive = false; p.cause = 'tornado'; }
      else if (this.dis.flashes.some(f => this.t - f.t < Sim.DT * 1.5 && Math.hypot(f.x - p.x, f.y - p.y) < f.r)) { p.alive = false; p.cause = 'meteor'; }
      if (!p.alive) this.dis.log(this.t, 'sim', `Probe ${p.id} lost (${p.cause})`);
    }
    this.t += dt; this.steps++;
    if (this.steps % 10 === 0) {
      this.reveal(r.x, r.y, SENSOR.revealRover);
      for (const p of this.probes) if (p.alive) this.reveal(p.x, p.y, SENSOR.revealProbe);
      for (let i = 0; i < N * N; i++) if ((w.water[i] > 0.05 && !w.river[i]) || w.burning[i] > 0 || w.rubble[i] > 0.2) this.affected[i] = 1;
      for (const c of w.craters) this.affected[w.cellIndexAt(c.x, c.y)] = 1;
    }
  }

  /** Produce the sensor frame the brain is allowed to see. */
  frame(): SensorFrame {
    this.seq++;
    const sampleProbes = this.t - this.lastProbeSample >= 1.0;
    if (sampleProbes) this.lastProbeSample = this.t;
    const { frame, probeReadings } = this.sensors.frame(this.t, this.seq, this.rover, this.probes, sampleProbes, this.probeReadings);
    this.probeReadings = probeReadings;
    this.lastFrame = frame;
    return frame;
  }

  reveal(x: number, y: number, rad: number) {
    const [cx, cy] = World.cellOf(x, y), R = Math.ceil(rad / CELL);
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const X = cx + dx, Y = cy + dy;
      if (X < 0 || Y < 0 || X >= N || Y >= N || (dx * dx + dy * dy) * CELL * CELL > rad * rad) continue;
      this.revealed[Y * N + X] = 1;
    }
  }

  metrics(): SimMetrics {
    const r = this.rover;
    let rev = 0; for (let i = 0; i < N * N; i++) rev += this.revealed[i];
    let aff = 0, cov = 0;
    for (let i = 0; i < N * N; i++) {
      if (!this.affected[i]) continue; aff++;
      const x = -HALF + ((i % N) + 0.5) * CELL, y = -HALF + (((i / N) | 0) + 0.5) * CELL;
      if (this.probes.some(p => Math.hypot(p.x - x, p.y - y) < SENSOR.probeRadius)) cov++;
    }
    return {
      survived: r.alive, cause: r.alive ? '' : r.cause, t: +this.t.toFixed(1), health: +r.health.toFixed(0), battery: +r.battery.toFixed(0),
      probesPlaced: ROVER.probes - r.probesLeft, probesAlive: this.probes.filter(p => p.alive).length,
      surveyedPct: +(100 * rev / (N * N)).toFixed(1), hazardCoveragePct: aff ? +(100 * cov / aff).toFixed(1) : 0,
    };
  }
}
