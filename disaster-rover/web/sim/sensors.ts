import { World, Occ, N, CELL } from './world';
import { Rover } from './rover';
import { Disasters } from './disasters';
import { Rng } from './rng';
import type { SensorFrame, RoverStatus, ProbeReading, Float32MultiArray } from '../brain/msgs';
import { header, yawToQuat } from '../brain/msgs';

export const SENSOR = {
  lidarBeams: 72,
  lidarRange: 30,
  lidarNoise: 0.05,
  thermalSectors: 36,
  thermalRange: 45,
  cameraRange: 28,
  detectRange: 90,
  elevRadius: 20,
  elevRes: 2,
  /** fog-of-war reveal radius around the rover and around each probe */
  revealRover: 28,
  revealProbe: 22,
  probeRadius: 22,
};
export const NO_DATA = -9999;

export interface Probe { id: number; x: number; y: number; alive: boolean; t: number; cause?: string }

const fma = (labels: [string, number][], data: number[]): Float32MultiArray => {
  const dim: { label: string; size: number; stride: number }[] = [];
  let stride = labels.reduce((a, l) => a * l[1], 1);
  for (const [label, size] of labels) { dim.push({ label, size, stride }); stride = stride / size; }
  return { layout: { dim, data_offset: 0 }, data };
};

/** Simulated, range-limited, noisy sensors. Reads ground truth; outputs only what hardware could measure. */
export class Sensors {
  constructor(private world: World, private dis: Disasters, private rng: Rng) {}

  /** mean smoke between rover and a point, sampled */
  private smokeAlong(x0: number, y0: number, ang: number, dist: number): number {
    let s = 0, n = 0;
    for (let d = 0; d < dist; d += 2) { s += this.world.smoke[this.world.cellIndexAt(x0 + d * Math.cos(ang), y0 + d * Math.sin(ang))]; n++; }
    return n ? s / n : 0;
  }

  frame(t: number, seq: number, r: Rover, probes: Probe[], sampleProbes: boolean, lastProbeReadings: ProbeReading[]): { frame: SensorFrame; probeReadings: ProbeReading[] } {
    const w = this.world, dis = this.dis, rng = this.rng;
    const hdr = header(t);
    const localSmoke = w.smoke[w.cellIndexAt(r.x, r.y)];

    // ---- LiDAR: 2D 360° raycast through the occupancy grid; smoke attenuates range ----
    const ranges: number[] = [], intens: number[] = [];
    const inc = 2 * Math.PI / SENSOR.lidarBeams;
    const sensorZ = r.z + 0.6;
    for (let k = 0; k < SENSOR.lidarBeams; k++) {
      const a = r.yaw - Math.PI + k * inc;
      const ca = Math.cos(a), sa = Math.sin(a);
      let hit = SENSOR.lidarRange + 1;
      let smokeAcc = 0;
      for (let d = 0.6; d <= SENSOR.lidarRange; d += 0.25) {
        const px = r.x + d * ca, py = r.y + d * sa;
        const o = w.occAt(px, py);
        if (o !== Occ.Free && o !== Occ.Crater && o !== Occ.Crack) { hit = d; break; }
        if (o === Occ.Crater || o === Occ.Crack) { hit = d; break; } // negative obstacles: modelled as a tilted-down beam return
        if (w.heightAt(px, py) > sensorZ + 0.9) { hit = d; break; }
        smokeAcc += w.smoke[w.cellIndexAt(px, py)] * 0.25;
        if (smokeAcc > 2.5) { hit = SENSOR.lidarRange + 1; break; } // fully attenuated: no return
      }
      if (hit <= SENSOR.lidarRange) { ranges.push(Math.max(0.1, hit + rng.gauss(SENSOR.lidarNoise))); intens.push(100 * (1 - Math.min(1, smokeAcc / 2.5))); }
      else { ranges.push(SENSOR.lidarRange + 1); intens.push(0); }
    }

    // ---- thermal camera: 36 sectors, max apparent temperature and its distance (IR sees through smoke) ----
    const ns = SENSOR.thermalSectors;
    const tMax = new Array(ns).fill(22), tDist = new Array(ns).fill(-1), hotNear = new Array(ns).fill(-1);
    const [rcx, rcy] = World.cellOf(r.x, r.y);
    const R = Math.ceil(SENSOR.thermalRange / CELL);
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const X = rcx + dx, Y = rcy + dy; if (X < 0 || Y < 0 || X >= N || Y >= N) continue;
      const i = Y * N + X;
      if (w.burning[i] <= 0 && !w.burnt[i]) continue;
      const [px, py] = World.cellCenter(X, Y);
      const d = Math.hypot(px - r.x, py - r.y); if (d > SENSOR.thermalRange) continue;
      const T = dis.cellTemp(i, t) * (1 - 0.25 * d / SENSOR.thermalRange);
      let s = Math.floor(((Math.atan2(py - r.y, px - r.x) + Math.PI) / (2 * Math.PI)) * ns) % ns; if (s < 0) s += ns;
      if (T > tMax[s]) { tMax[s] = T; tDist[s] = d; }
      if (T > 100 && (hotNear[s] < 0 || d < hotNear[s])) hotNear[s] = d;
    }
    const thermal: number[] = [];
    for (let s = 0; s < ns; s++) thermal.push(tMax[s] + rng.gauss(2), tDist[s] < 0 ? -1 : tDist[s] + rng.gauss(0.5), hotNear[s] < 0 ? -1 : hotNear[s] + rng.gauss(0.4));

    // ---- camera: water detector per sector, range cut by smoke ----
    const water: number[] = [];
    for (let s = 0; s < ns; s++) {
      const a = -Math.PI + (s + 0.5) * 2 * Math.PI / ns;
      const vis = SENSOR.cameraRange * Math.max(0.15, 1 - 1.6 * Math.max(localSmoke, this.smokeAlong(r.x, r.y, a, 15)));
      let found = -1;
      for (let d = 1; d <= vis; d += 1) {
        const px = r.x + d * Math.cos(a), py = r.y + d * Math.sin(a);
        if (!w.inBounds(px, py)) break;
        if (w.waterAt(px, py) > 0.05) { found = d; break; }
        const o = w.occAt(px, py); if (o === Occ.Building || o === Occ.Tree) break;
      }
      water.push(found < 0 ? -1 : found + rng.gauss(0.3));
    }

    // ---- camera detections: impact flashes, tornado funnel, incoming meteor streaks, smoke ----
    const det: number[] = [];
    const visFar = SENSOR.detectRange * Math.max(0.2, 1 - 1.4 * localSmoke);
    for (const f of dis.flashes) {
      const d = Math.hypot(f.x - r.x, f.y - r.y);
      if (d < visFar && t - f.t < 0.15) det.push(1, Math.atan2(f.y - r.y, f.x - r.x) + rng.gauss(0.02), d * (1 + rng.gauss(0.04)), f.r, 0);
    }
    for (const tor of dis.tornadoes) {
      if (!tor.active) continue;
      const d = Math.hypot(tor.x - r.x, tor.y - r.y);
      if (d < visFar) det.push(2, Math.atan2(tor.y - r.y, tor.x - r.x) + rng.gauss(0.03), d * (1 + rng.gauss(0.06)), tor.R * (1 + rng.gauss(0.1)), 0);
    }
    for (const m of dis.meteors) {
      const d = Math.hypot(m.x - r.x, m.y - r.y), ttl = m.tImpact - t;
      if (d < 60 && ttl < 3.2 && ttl > 0) det.push(3, Math.atan2(m.y - r.y, m.x - r.x) + rng.gauss(0.02), d * (1 + rng.gauss(0.05)), m.r * 1.1, ttl);
    }
    if (localSmoke > 0.05) det.push(4, 0, 0, localSmoke, 0);

    // ---- depth camera elevation patch (DSM, includes structures) ----
    const n = 2 * Math.round(SENSOR.elevRadius / SENSOR.elevRes) + 1;
    // samples on 2 m cell centres (odd coordinates) so each one lands squarely in a map cell
    const ox = Math.round((r.x - 1) / SENSOR.elevRes) * SENSOR.elevRes + 1 - SENSOR.elevRadius, oy = Math.round((r.y - 1) / SENSOR.elevRes) * SENSOR.elevRes + 1 - SENSOR.elevRadius;
    const elev: number[] = [ox, oy, SENSOR.elevRes, n];
    const elevVis = SENSOR.elevRadius * Math.max(0.4, 1 - 1.2 * localSmoke);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const px = ox + i * SENSOR.elevRes, py = oy + j * SENSOR.elevRes;
      if (Math.hypot(px - r.x, py - r.y) > elevVis || !w.inBounds(px, py)) { elev.push(NO_DATA); continue; }
      elev.push(w.surfaceAt(px, py) + rng.gauss(0.05));
    }

    // ---- wind / pressure / IMU / odom / water ----
    const [wx, wy] = dis.windAt(r.x, r.y);
    const shake = dis.shakeAmp(t);
    const sh = () => shake * rng.gauss(1);

    // ---- probes: each one is a small static sensor station ----
    let probeReadings = lastProbeReadings;
    if (sampleProbes) {
      probeReadings = probes.map(p => {
        if (!p.alive) return { id: p.id, x: p.x, y: p.y, alive: false, water: 0, temp: 0, wind: 0, shake: 0, samples: [] };
        const samples: number[] = [];
        const step = 5;
        for (let dy = -SENSOR.probeRadius; dy <= SENSOR.probeRadius; dy += step) for (let dx = -SENSOR.probeRadius; dx <= SENSOR.probeRadius; dx += step) {
          if (dx * dx + dy * dy > SENSOR.probeRadius ** 2) continue;
          const px = p.x + dx, py = p.y + dy; if (!w.inBounds(px, py)) continue;
          const i = w.cellIndexAt(px, py);
          samples.push(+px.toFixed(1), +py.toFixed(1), +(w.water[i] + Math.abs(rng.gauss(0.01))).toFixed(2), Math.round(dis.cellTemp(i, t) + rng.gauss(2)), +(w.surfaceAt(px, py) + rng.gauss(0.05)).toFixed(2));
        }
        const [pwx, pwy] = dis.windAt(p.x, p.y);
        return { id: p.id, x: p.x, y: p.y, alive: true, water: +w.waterAt(p.x, p.y).toFixed(2), temp: Math.round(dis.airTemp(p.x, p.y)), wind: +Math.hypot(pwx, pwy).toFixed(1), shake: +dis.shakeAmp(t).toFixed(2), samples };
      });
    }

    const status: RoverStatus = { seq, t: +t.toFixed(2), health: +r.health.toFixed(1), battery: +r.battery.toFixed(1), probes_left: r.probesLeft, alive: r.alive, temp_c: +(dis.airTemp(r.x, r.y) + rng.gauss(0.5)).toFixed(1), probes: probeReadings };
    const frame: SensorFrame = {
      scan: { header: { ...hdr, frame_id: 'laser' }, angle_min: -Math.PI, angle_max: Math.PI - inc, angle_increment: inc, time_increment: 0, scan_time: 0.1, range_min: 0.1, range_max: SENSOR.lidarRange, ranges, intensities: intens },
      odom: {
        header: { ...hdr, frame_id: 'odom' }, child_frame_id: 'base_link',
        pose: { pose: { position: { x: r.x + rng.gauss(0.05), y: r.y + rng.gauss(0.05), z: r.z }, orientation: yawToQuat(r.yaw + rng.gauss(0.005)) }, covariance: new Array(36).fill(0) },
        twist: { twist: { linear: { x: r.v + rng.gauss(0.02), y: 0, z: 0 }, angular: { x: 0, y: 0, z: r.w + rng.gauss(0.01) } }, covariance: new Array(36).fill(0) },
      },
      imu: {
        header: { ...hdr, frame_id: 'imu' },
        orientation: yawToQuat(r.yaw, r.roll + rng.gauss(0.003), r.pitch + rng.gauss(0.003)), orientation_covariance: new Array(9).fill(0),
        angular_velocity: { x: sh() * 0.05, y: sh() * 0.05, z: r.w + rng.gauss(0.005) }, angular_velocity_covariance: new Array(9).fill(0),
        linear_acceleration: { x: r.ax + sh() + rng.gauss(0.05), y: r.ay + sh() + rng.gauss(0.05), z: r.az + sh() * 0.5 + rng.gauss(0.05) }, linear_acceleration_covariance: new Array(9).fill(0),
      },
      thermal: fma([['sector', ns], ['field', 3]], thermal),
      water_depth: { data: Math.max(0, w.waterAt(r.x, r.y) + rng.gauss(0.01)) },
      wind: { header: { ...hdr, frame_id: 'map' }, vector: { x: wx + rng.gauss(0.3), y: wy + rng.gauss(0.3), z: 0 } },
      pressure: { header: hdr, fluid_pressure: dis.pressureAt(r.x, r.y) + rng.gauss(8), variance: 64 },
      elevation: fma([['meta', 4 + n * n]], elev),
      camera_water: fma([['sector', ns]], water),
      camera_detections: fma([['detection', det.length / 5], ['field', 5]], det),
      rover_status: { data: JSON.stringify(status) },
    };
    return { frame, probeReadings };
  }
}
