import { World, Occ } from './world';
import type { Twist } from '../brain/msgs';

export const ROVER = {
  radius: 0.8,
  maxV: 3.0,
  maxW: 1.6,
  accel: 3.0,
  alphaW: 5.0,
  /** water deeper than this kills the rover (electronics flooded) */
  waterLimit: 0.5,
  /** water deeper than this slows it down */
  waterSlow: 0.2,
  /** air temperature that starts damaging the rover */
  tempDamage: 70,
  /** air temperature that destroys it outright */
  tempLimit: 160,
  /** roll/pitch beyond this tips it over */
  maxTiltDeg: 34,
  probes: 6,
};

export interface EnvAtRover { windX: number; windY: number; tempC: number; shake: number; joltX: number; joltY: number }

export class Rover {
  x: number; y: number; yaw: number; z = 0;
  v = 0; w = 0;
  /** externally imposed velocity (tornado push) */
  pushX = 0; pushY = 0;
  roll = 0; pitch = 0;
  /** last body-frame acceleration, for the IMU */
  ax = 0; ay = 0; az = 9.81;
  health = 100; battery = 100; probesLeft = ROVER.probes;
  alive = true; cause = '';
  odoDist = 0;
  stuckTime = 0;

  constructor(x: number, y: number, yaw: number) { this.x = x; this.y = y; this.yaw = yaw; }

  kill(cause: string) { if (this.alive) { this.alive = false; this.cause = cause; this.v = 0; this.w = 0; } }
  damage(amount: number, cause: string) {
    if (!this.alive) return;
    this.health -= amount;
    if (this.health <= 0) { this.health = 0; this.kill(cause); }
  }

  step(dt: number, cmd: Twist, world: World, env: EnvAtRover) {
    if (!this.alive) return;
    const depth = world.waterAt(this.x, this.y);
    let vCmd = Math.max(-ROVER.maxV * 0.5, Math.min(ROVER.maxV, cmd.linear.x || 0));
    const wCmd = Math.max(-ROVER.maxW, Math.min(ROVER.maxW, cmd.angular.z || 0));
    if (this.battery <= 0) vCmd = 0;
    if (depth > ROVER.waterSlow) vCmd *= 0.45;
    if (world.rubble[world.cellIndexAt(this.x, this.y)] > 0.3) vCmd *= 0.6;
    const v0 = this.v;
    const dv = Math.max(-ROVER.accel * dt, Math.min(ROVER.accel * dt, vCmd - this.v));
    this.v += dv;
    const dw = Math.max(-ROVER.alphaW * dt, Math.min(ROVER.alphaW * dt, wCmd - this.w));
    this.w += dw;

    // slope: rover slows climbing, speeds a bit going down
    const [gx, gy] = world.gradAt(this.x, this.y);
    const along = gx * Math.cos(this.yaw) + gy * Math.sin(this.yaw);
    const effV = this.v * (1 - Math.max(-0.2, Math.min(0.6, along * 0.8)));

    // wind push (tornado). Ordinary wind is too weak to matter.
    const ws = Math.hypot(env.windX, env.windY);
    const pushK = ws > 14 ? (ws - 14) * 0.09 : 0;
    this.pushX += (env.windX / Math.max(ws, 1e-6) * pushK - this.pushX) * Math.min(1, dt * 2);
    this.pushY += (env.windY / Math.max(ws, 1e-6) * pushK - this.pushY) * Math.min(1, dt * 2);

    // seismic jolt
    const jx = env.joltX, jy = env.joltY;

    const nyaw = this.yaw + this.w * dt;
    const nx = this.x + (effV * Math.cos(nyaw) + this.pushX + jx) * dt;
    const ny = this.y + (effV * Math.sin(nyaw) + this.pushY + jy) * dt;
    this.yaw = Math.atan2(Math.sin(nyaw), Math.cos(nyaw));

    // (if already overlapping something, e.g. rubble fell on us, allow moving out)
    if (this.collides(world, nx, ny, Math.atan2(ny - this.y, nx - this.x))) {
      const speed = Math.hypot(effV + this.pushX, this.pushY);
      if (speed > 4) this.damage((speed - 4) * 8, 'smashed into an obstacle');
      this.v = 0; this.pushX *= -0.3; this.pushY *= -0.3;
      this.stuckTime += dt;
    } else {
      this.odoDist += Math.hypot(nx - this.x, ny - this.y);
      this.x = nx; this.y = ny; this.stuckTime = 0;
    }
    this.z = world.heightAt(this.x, this.y);
    const [g2x, g2y] = world.gradAt(this.x, this.y);
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
    this.pitch = Math.atan(g2x * c + g2y * s);
    this.roll = Math.atan(-g2x * s + g2y * c);

    // body accelerations for the IMU
    this.ax = (this.v - v0) / dt + 9.81 * Math.sin(this.pitch);
    this.ay = this.v * this.w - 9.81 * Math.sin(this.roll);
    this.az = 9.81 * Math.cos(this.pitch) * Math.cos(this.roll);

    // ---- failure conditions (ground truth, judged by the sim, never seen by the brain) ----
    const d2 = world.waterAt(this.x, this.y);
    if (d2 > ROVER.waterLimit) this.kill(`drowned in ${d2.toFixed(2)} m of water`);
    if (env.tempC > ROVER.tempLimit) this.kill(`overheated (${env.tempC.toFixed(0)}°C air)`);
    else if (env.tempC > ROVER.tempDamage) this.damage((env.tempC - ROVER.tempDamage) * 0.12 * dt, `overheated (${env.tempC.toFixed(0)}°C air)`);
    const tilt = Math.max(Math.abs(this.roll), Math.abs(this.pitch)) * 180 / Math.PI;
    if (tilt > ROVER.maxTiltDeg) this.kill(`rolled over on a ${tilt.toFixed(0)}° slope`);
    if (!world.inBounds(this.x, this.y)) this.kill('left the mission area');

    this.battery = Math.max(0, this.battery - dt * (0.10 + 0.22 * Math.abs(this.v) / ROVER.maxV + (d2 > 0.05 ? 0.1 : 0)));
  }

  /**
   * Blocking contact test: a footprint point that lands in an obstacle only blocks the move if the
   * move brings the rover centre closer to that point (driving into it). Sliding past or backing away is allowed.
   */
  collides(world: World, x: number, y: number, _dir = 0): boolean {
    for (let a = 0; a < 16; a++) {
      const ang = a * Math.PI / 8;
      const px = x + ROVER.radius * Math.cos(ang), py = y + ROVER.radius * Math.sin(ang);
      const o = world.occAt(px, py);
      if (o === Occ.Free || o === Occ.Rubble) continue;
      if (Math.hypot(px - x, py - y) < Math.hypot(px - this.x, py - this.y) - 1e-4) return true;
    }
    return false;
  }
}
