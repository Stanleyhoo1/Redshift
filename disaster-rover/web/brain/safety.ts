import { CFG, compass, wrap } from './config';
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

export class Safety {
  lowConfSince = -1;
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
    if (worst) {
      const away = worst.d > 0.5 ? Math.atan2(rd.y - worst.y, rd.x - worst.x) : rd.yaw;
      return { name: 'meteor_dodge', cmd: escapeHeading(rd, away), text: `Camera: meteor streak, impact ${worst.d.toFixed(1)} m ${compass(Math.atan2(worst.y - rd.y, worst.x - rd.x))} in ${worst.ttl.toFixed(1)} s → DODGE ${compass(away)}` };
    }
    // 2. standing in water deeper than the reflex limit (hard limit 0.5 m)
    if (rd.depth > CFG.reflexWater) {
      const h = this.bestDryHeading(rd, b);
      return { name: 'water', cmd: escapeHeading(rd, h, 2.2), text: `Water depth ${rd.depth.toFixed(2)} m > ${CFG.reflexWater} m limit → climb out ${compass(h)}` };
    }
    // 3. heat: onboard thermometer or a >100 °C hotspot very close
    const close = rd.hotSectors.filter(s => s.dist < 9);
    if (rd.tempC > CFG.reflexTemp || close.length) {
      let rx = 0, ry = 0;
      for (const s of rd.hotSectors) { const w = s.temp / Math.max(1, s.dist * s.dist); rx -= w * Math.cos(s.bearing); ry -= w * Math.sin(s.bearing); }
      // don't flee into water
      let h = (rx || ry) ? Math.atan2(ry, rx) : rd.yaw + Math.PI;
      h = this.avoidWater(rd, b, h);
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
      const back = Math.abs(rd.pitch) > Math.abs(rd.roll) ? (rd.pitch > 0 ? -1.0 : 1.0) : -1.0;
      return { name: 'tilt', cmd: { linear: { x: back, y: 0, z: 0 }, angular: { x: 0, y: 0, z: Math.abs(rd.roll) > Math.abs(rd.pitch) ? 0.8 * Math.sign(rd.roll) : 0 } }, text: `IMU tilt ${rd.tiltDeg.toFixed(0)}° > ${CFG.reflexTilt}° → back off the slope, mark it impassable` };
    }
    // 6. stale data: we don't know what's around us any more
    if (rd.dt > CFG.staleFrame) return { name: 'stale', cmd: tw(0, 0), text: `Sensor frame ${rd.dt.toFixed(2)} s old → STOP` };
    // 7. low perception confidence (smoke-blinded LiDAR): stop and re-scan for a moment, then creep
    if (rd.returnFrac < CFG.lowConfReturnFrac && rd.smoke > 0.25) {
      if (this.lowConfSince < 0) this.lowConfSince = rd.t;
      if (rd.t - this.lowConfSince < 2.5) return { name: 'rescan', cmd: tw(0, 0.6), text: `LiDAR returns ${(rd.returnFrac * 100).toFixed(0)}%, smoke ${rd.smoke.toFixed(2)} → low confidence: STOP & RE-SCAN` };
    } else this.lowConfSince = -1;
    return null;
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
