import { CFG, CH, compass, wrap } from './config';
import { Belief } from './grid';
import type { Readings } from './perception';
import type { Predictions } from './prediction';
import type { Twist } from './msgs';

export interface Reflex { name: string; cmd: Twist; text: string }

const tw = (v: number, w: number): Twist => ({ linear: { x: v, y: 0, z: 0 }, angular: { x: 0, y: 0, z: w } });

/** Drive toward a world heading as fast as possible, using reverse if the heading is behind us. */
export function escapeHeading(rd: Readings, heading: number, speed = CFG.maxV): Twist {
  const err = wrap(heading - rd.yaw);
  if (Math.abs(err) > 2.0) {
    // behind: back out while swinging the tail toward the heading
    const errBack = wrap(heading - (rd.yaw + Math.PI));
    return tw(-speed * 0.5, Math.max(-CFG.maxW, Math.min(CFG.maxW, 2.5 * errBack)));
  }
  const v = speed * Math.max(0.25, Math.cos(err));
  return tw(v, Math.max(-CFG.maxW, Math.min(CFG.maxW, 2.5 * err)));
}

/** LiDAR clearance (m) along a world heading: min range of beams within ±halfWidth. */
export function clearance(rd: Readings, heading: number, halfWidth = 0.3): number {
  let m = Infinity;
  for (let k = 0; k < rd.scanR.length; k++) {
    const a = rd.scanA0 + k * rd.scanInc;
    if (Math.abs(wrap(a - heading)) > halfWidth) continue;
    const r = rd.scanR[k] > rd.scanMax ? rd.scanMax : rd.scanR[k];
    if (r < m) m = r;
  }
  return m;
}

/**
 * Local steering: if the heading we're about to drive along is blocked within `need` metres,
 * re-aim at the nearest heading that is clear (VFH-style). Applied to every forward command.
 */
export function steerClear(rd: Readings, cmd: Twist, need = 2.6): Twist {
  if (cmd.linear.x <= 0.05) return cmd;
  const want = rd.yaw + cmd.angular.z * 0.6;
  if (clearance(rd, want) >= need) return cmd;
  let best = NaN, bestCost = Infinity;
  for (let k = -12; k <= 12; k++) {
    const h = want + k * Math.PI / 18;
    const c = clearance(rd, h);
    if (c < need) continue;
    const cost = Math.abs(k) - Math.min(c, 10) * 0.05;
    if (cost < bestCost) { bestCost = cost; best = h; }
  }
  if (isNaN(best)) return { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: cmd.angular.z >= 0 ? 1.2 : -1.2 } };
  const err = wrap(best - rd.yaw);
  return { linear: { x: Math.abs(err) > 1.0 ? 0 : Math.min(cmd.linear.x, CFG.maxV * Math.cos(err)), y: 0, z: 0 }, angular: { x: 0, y: 0, z: Math.max(-CFG.maxW, Math.min(CFG.maxW, 2.5 * err)) } };
}

export class Safety {
  lowConfSince = -1;
  lastRescan = -1e9;
  lastReflex: string | null = null;

  /**
   * SAFETY REFLEXES: hard limits that override every behaviour, instantly.
   * Ordered by how fast each hazard kills.
   */
  check(rd: Readings, b: Belief, P: Predictions): Reflex | null {
    // 1. incoming meteor whose impact point is near us
    let worst: { d: number; ttl: number; x: number; y: number; size: number } | null = null;
    for (const s of rd.streaks) {
      const d = Math.hypot(s.x - rd.x, s.y - rd.y);
      if (d < s.size + 4 && (!worst || d - s.size < worst.d - worst.size)) worst = { d, ttl: s.ttl, x: s.x, y: s.y, size: s.size };
    }
    if (worst && rd.depth <= 0.35) {
      const away = this.dodgeHeading(rd, b, worst.d > 0.5 ? Math.atan2(rd.y - worst.y, rd.x - worst.x) : rd.yaw);
      return { name: 'meteor_dodge', cmd: escapeHeading(rd, away), text: `Camera: meteor streak, impact ${worst.d.toFixed(1)} m ${compass(Math.atan2(worst.y - rd.y, worst.x - rd.x))} in ${worst.ttl.toFixed(1)} s → DODGE ${compass(away)}` };
    }
    // 2. standing in water deeper than the reflex limit (hard limit 0.5 m)
    // with fire close, shallow water (well under the 0.5 m limit) is an acceptable refuge
    const heatNear = rd.tempC > 40 || rd.hotSectors.some(s => s.dist < 14);
    if (rd.depth > (heatNear ? 0.36 : CFG.reflexWater)) {
      const h = heatNear ? this.escapeSearch(rd, b, 0.3) : this.bestDryHeading(rd, b);
      return { name: 'water', cmd: escapeHeading(rd, h, 2.2), text: `Water depth ${rd.depth.toFixed(2)} m > ${CFG.reflexWater} m limit → climb out ${compass(h)}` };
    }
    // 3. heat: onboard thermometer or a >100 °C hotspot very close
    const close = rd.hotSectors.filter(s => s.dist < 9);
    if (rd.tempC > CFG.reflexTemp || close.length) {
      const h = this.escapeSearch(rd, b, 0.38);
      const near = close.length ? close.reduce((a, c) => (c.dist < a.dist ? c : a)) : null;
      return { name: 'heat', cmd: escapeHeading(rd, h), text: `${rd.tempC > CFG.reflexTemp ? `Body temp ${rd.tempC.toFixed(0)}°C > ${CFG.reflexTemp}°C` : `Thermal ${near!.temp.toFixed(0)}°C at ${near!.dist.toFixed(1)} m ${compass(near!.bearing)}`} → ESCAPE ${compass(h)}` };
    }
    // 4. tornado wind
    if (rd.windSpeed > CFG.reflexWind) {
      const center = P.tornado.active ? Math.atan2(P.tornado.y - rd.y, P.tornado.x - rd.x) : Math.atan2(rd.windY, rd.windX) + Math.PI / 2;
      // run away from the centre and away from its track
      let h = center + Math.PI;
      if (P.tornado.active && Math.hypot(P.tornado.vx, P.tornado.vy) > 0.5) {
        const side = Math.sign((rd.x - P.tornado.x) * -P.tornado.vy + (rd.y - P.tornado.y) * P.tornado.vx) || 1;
        const perp = Math.atan2(P.tornado.vx * side, -P.tornado.vy * side);
        h = Math.atan2(Math.sin(h) + Math.sin(perp), Math.cos(h) + Math.cos(perp));
      }
      h = this.avoidWater(rd, b, h);
      return { name: 'wind', cmd: escapeHeading(rd, h), text: `Wind ${rd.windSpeed.toFixed(0)} m/s > ${CFG.reflexWind}, pressure −${rd.pressureDrop.toFixed(0)} Pa → RUN ${compass(h)} away from vortex` };
    }
    // 5. tipping
    if (rd.tiltDeg > CFG.reflexTilt) {
      // back straight down the way we came (we arrived from gentler ground) and remember this slope
      const ax = rd.x + 2.5 * Math.cos(rd.yaw), ay = rd.y + 2.5 * Math.sin(rd.yaw);
      if (b.inside(ax, ay)) b.obst[b.idx(ax, ay)] = 1;
      let back = Math.abs(rd.pitch) > Math.abs(rd.roll) ? (rd.pitch > 0 ? -1.0 : 1.0) : -1.0;
      // never back blindly into water
      const bx = rd.x + Math.sign(back) * 3 * Math.cos(rd.yaw), by = rd.y + Math.sign(back) * 3 * Math.sin(rd.yaw);
      if (b.inside(bx, by) && b.water[b.idx(bx, by)] > 0.05) back = -back * 0.8;
      return { name: 'tilt', cmd: { linear: { x: back, y: 0, z: 0 }, angular: { x: 0, y: 0, z: Math.abs(rd.roll) > Math.abs(rd.pitch) ? 0.8 * Math.sign(rd.roll) : 0 } }, text: `IMU tilt ${rd.tiltDeg.toFixed(0)}° > ${CFG.reflexTilt}° → back off the slope, mark it impassable` };
    }
    // 6. stale data: we don't know what's around us any more
    if (rd.dt > CFG.staleFrame) return { name: 'stale', cmd: tw(0, 0), text: `Sensor frame ${rd.dt.toFixed(2)} s old → STOP` };
    // 7. low perception confidence (smoke-blinded LiDAR): stop and re-scan for a moment, then creep.
    //    Not while a known fire is close (standing still in smoke next to a fire is the worse option).
    const fireClose = P.fire.active && P.fire.nearest < 30;
    if (rd.returnFrac < CFG.lowConfReturnFrac && rd.smoke > 0.25 && !fireClose && rd.t - this.lastRescan > 10) {
      if (this.lowConfSince < 0) this.lowConfSince = rd.t;
      if (rd.t - this.lowConfSince >= 2.5) { this.lastRescan = rd.t; this.lowConfSince = -1; }
      else return { name: 'rescan', cmd: tw(0, 0.6), text: `LiDAR returns ${(rd.returnFrac * 100).toFixed(0)}%, smoke ${rd.smoke.toFixed(2)} → low confidence: STOP & RE-SCAN` };
    } else this.lowConfSince = -1;
    return null;
  }

  /**
   * Escape heading that trades off heat and water: away from hot sectors (weighted T/d²),
   * not into water deeper than `maxDepth`, not into obstacles.
   */
  /** away from the impact point, but not into water / obstacles / steep ground */
  private dodgeHeading(rd: Readings, b: Belief, away: number): number {
    let best = away, bestS = -Infinity;
    for (let k = 0; k < 16; k++) {
      const a = away + (k - 8) * Math.PI / 16;
      let s = 2 * Math.cos(a - away) + Math.min(clearance(rd, a), 10) / 10;
      for (const d of [3, 6]) {
        const px = rd.x + d * Math.cos(a), py = rd.y + d * Math.sin(a);
        if (!b.inside(px, py)) { s -= 10; continue; }
        const i = b.idx(px, py);
        if (b.water[i] > 0.15) s -= 10;
        if (b.blocked(i)) s -= 4;
        if (b.channels[CH.slope][i] >= 0.6) s -= 6;
      }
      if (s > bestS) { bestS = s; best = a; }
    }
    return best;
  }

  escapeSearch(rd: Readings, b: Belief, maxDepth: number): number {
    let best = rd.yaw + Math.PI, bestS = -Infinity;
    for (let k = 0; k < 24; k++) {
      const a = k * Math.PI / 12;
      let s = 0;
      for (const hs of rd.hotSectors) s -= (hs.temp / Math.max(4, hs.dist * hs.dist)) * Math.max(0, Math.cos(a - hs.bearing)) * 3;
      for (const d of [3, 6, 10]) {
        const px = rd.x + d * Math.cos(a), py = rd.y + d * Math.sin(a);
        if (!b.inside(px, py)) { s -= 30; continue; }
        const i = b.idx(px, py);
        if (b.blocked(i)) s -= 8;
        const wd = b.water[i];
        if (wd > maxDepth) s -= 25 * (wd - maxDepth + 0.1);
        s -= 6 * b.channels[CH.heat][i];
        // steep ground (rollover) and banks that drop toward known water
        if (b.channels[CH.slope][i] >= 0.6) s -= 10;
        if (b.waterDist[i] < 4 && b.channels[CH.slope][i] > 0.3) s -= 8;
      }
      s += 0.5 * clearance(rd, a) / 30;
      if (s > bestS) { bestS = s; best = a; }
    }
    return best;
  }

  private bestDryHeading(rd: Readings, b: Belief): number {
    let best = rd.yaw + Math.PI, bestS = -Infinity;
    for (let k = 0; k < 16; k++) {
      const a = k * Math.PI / 8; let s = 0;
      for (const d of [3, 6, 9]) {
        const px = rd.x + d * Math.cos(a), py = rd.y + d * Math.sin(a);
        if (!b.inside(px, py)) { s -= 50; continue; }
        const i = b.idx(px, py);
        const e = !isNaN(b.elev[i]) ? b.elev[i] : 0;
        s += e - (b.water[i] > 0.05 ? 3 * b.water[i] : 0) - (b.blocked(i) ? 20 : 0) + (b.water[i] === 0 ? 0.5 : 0);
      }
      s -= 0.3 * Math.abs(wrap(a - rd.yaw)) * 0; // pure elevation choice
      if (s > bestS) { bestS = s; best = a; }
    }
    return best;
  }
  private avoidWater(rd: Readings, b: Belief, h: number): number {
    for (const off of [0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6]) {
      const a = h + off; let ok = true;
      for (const d of [3, 6, 10]) { const px = rd.x + d * Math.cos(a), py = rd.y + d * Math.sin(a); if (!b.inside(px, py)) { ok = false; break; } const i = b.idx(px, py); if (b.water[i] > 0.15 || b.blocked(i)) { ok = false; break; } }
      if (ok) return a;
    }
    return h;
  }
}
