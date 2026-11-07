import { CFG, clamp01, wrap } from './config';
import { Belief } from './grid';
import type { SensorFrame, RoverStatus } from './msgs';
import { quatToYaw, quatToRollPitch } from './msgs';

export const NO_DATA = -9999;

export interface Detection { cls: number; bearing: number; dist: number; size: number; ttl: number; x: number; y: number }
/** Everything derived from ONE sensor frame. No ground truth anywhere in here. */
export interface Readings {
  t: number; dt: number;
  x: number; y: number; yaw: number; v: number; w: number; roll: number; pitch: number; tiltDeg: number;
  status: RoverStatus;
  depth: number; tempC: number;
  windX: number; windY: number; windSpeed: number; pressure: number; pressureDrop: number;
  shake: number;
  lidarFront: number; returnFrac: number; smoke: number;
  /** hottest thermal sector */
  thermalMax: { temp: number; dist: number; bearing: number };
  /** sectors with a >100 °C pixel: bearing (world), distance to it, sector max temp */
  hotSectors: { bearing: number; dist: number; temp: number }[];
  streaks: Detection[]; flashes: Detection[]; funnel: Detection | null;
  /** nearest visible water (camera) */
  waterNear: { dist: number; bearing: number } | null;
  /** brain-side estimate of the water surface elevation (flood stage), NaN if no water seen */
  stageObs: number;
}

/** Slow state carried between frames by perception. */
export class PerceptionState {
  lastT = -1;
  accDiffs: number[] = [];
  lastV = NaN;
  pressureBase = NaN;
  ambX = 0; ambY = 0; ambN = 0;
}

function markHot(b: Belief, px: number, py: number, T: number, t: number) {
  if (!b.inside(px, py)) return;
  const i = b.idx(px, py);
  b.heat[i] = b.heatT[i] === t && !isNaN(b.heat[i]) ? Math.max(b.heat[i], T) : T;
  b.heatT[i] = t; b.see(i, t);
}

const median = (a: number[]) => { if (!a.length) return 0; const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };

export function perceive(b: Belief, f: SensorFrame, ps: PerceptionState): Readings {
  const status: RoverStatus = JSON.parse(f.rover_status.data);
  // frame time = /rover_status time (it closes the frame; topics can arrive out of order over rosbridge)
  const t = status.t;
  const dt = ps.lastT < 0 ? 0.1 : t - ps.lastT; ps.lastT = t;
  const p = f.odom.pose.pose.position, yaw = quatToYaw(f.odom.pose.pose.orientation);
  const [roll, pitch] = quatToRollPitch(f.imu.orientation);
  const x = p.x, y = p.y;

  // ---------- LiDAR: free space along beams, obstacle evidence at hits ----------
  const sc = f.scan;
  let returns = 0, front = 99;
  const smokeDet = f.camera_detections.data;
  let smoke = 0;
  for (let k = 0; k + 4 < smokeDet.length; k += 5) if (smokeDet[k] === 4) smoke = smokeDet[k + 3];
  const freeLimitNoReturn = smoke > 0.15 ? 6 : sc.range_max;
  for (let k = 0; k < sc.ranges.length; k++) {
    const r = sc.ranges[k], a = yaw + sc.angle_min + k * sc.angle_increment;
    const hit = r <= sc.range_max && r >= sc.range_min;
    if (hit) returns++;
    const rel = wrap(sc.angle_min + k * sc.angle_increment);
    if (Math.abs(rel) < 0.35 && hit) front = Math.min(front, r);
    const freeTo = hit ? r - 0.6 : freeLimitNoReturn;
    const ca = Math.cos(a), sa = Math.sin(a);
    let lastI = -1;
    for (let d = 1.0; d < freeTo; d += 1.0) {
      const px = x + d * ca, py = y + d * sa; if (!b.inside(px, py)) break;
      const i = b.idx(px, py); if (i === lastI) continue; lastI = i;
      b.obst[i] = Math.max(0, b.obst[i] - 0.12); b.see(i, t);
    }
    if (hit) {
      const hx = x + r * ca, hy = y + r * sa;
      if (b.inside(hx, hy)) { const i = b.idx(hx, hy); b.obst[i] = Math.min(1, b.obst[i] + 0.45); b.see(i, t); }
    }
  }
  const returnFrac = returns / sc.ranges.length;

  // ---------- depth camera: elevation patch (surface incl. structures & water) ----------
  const e = f.elevation.data;
  const ox = e[0], oy = e[1], res = e[2], n = e[3];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const v = e[4 + j * n + i]; if (v === NO_DATA || v === null) continue;
    const px = ox + i * res, py = oy + j * res; if (!b.inside(px, py)) continue;
    const c = b.idx(px, py);
    b.elev[c] = isNaN(b.elev[c]) ? v : b.elev[c] * 0.5 + v * 0.5;
    b.see(c, t);
  }
  // structures: cells standing well above their neighbours (buildings, trees)
  for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) {
    const v = e[4 + j * n + i]; if (v === NO_DATA || v === null) continue;
    let lo = v;
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const u = e[4 + (j + dj) * n + i + di]; if (u !== NO_DATA && u !== null && u < lo) lo = u; }
    const px = ox + i * res, py = oy + j * res; if (!b.inside(px, py)) continue;
    b.struct[b.idx(px, py)] = v - lo > 2.5 ? 1 : 0;
  }

  // ---------- water: depth gauge under the rover + camera water edges ----------
  const depth = f.water_depth.data;
  const here = b.idx(x, y);
  b.water[here] = depth; b.waterT[here] = t;
  if (depth < 0.03 && !isNaN(b.elev[here])) b.dryElev[here] = b.elev[here];
  const cw = f.camera_water.data, ns = cw.length;
  let waterNear: Readings['waterNear'] = null;
  const stageSamples: number[] = [];
  const vis = CFG.cameraRange * Math.max(0.15, 1 - 1.6 * smoke);
  for (let s = 0; s < ns; s++) {
    const a = -Math.PI + (s + 0.5) * 2 * Math.PI / ns, d = cw[s];
    const ca = Math.cos(a), sa = Math.sin(a);
    const dryTo = d < 0 ? vis : d - 1;
    let lastI = -1;
    for (let r = 1; r < dryTo; r += 1) {
      const px = x + r * ca, py = y + r * sa; if (!b.inside(px, py)) break;
      const i = b.idx(px, py); if (i === lastI) continue; lastI = i;
      b.water[i] = 0; b.waterT[i] = t; b.see(i, t);
      if (!isNaN(b.elev[i]) && !b.struct[i]) b.dryElev[i] = b.elev[i];
    }
    if (d >= 0) {
      if (!waterNear || d < waterNear.dist) waterNear = { dist: d, bearing: a };
      // water seen from d out to a few metres beyond
      for (let r = d; r < Math.min(d + 6, vis + 2); r += 1) {
        const px = x + r * ca, py = y + r * sa; if (!b.inside(px, py)) break;
        const i = b.idx(px, py);
        b.waterT[i] = t; b.see(i, t);
        if (b.water[i] < 0.3) b.water[i] = 0.3;
        // only where the depth camera is looking right now (fresh surface elevation)
        if (r === d && d < 19 && !isNaN(b.elev[i])) stageSamples.push(b.elev[i]);
      }
    }
  }
  // depth estimate for seen water cells: surface (stage) minus the dry ground we saw earlier
  const stageObs = stageSamples.length ? median(stageSamples) : (depth > 0.05 && !isNaN(b.elev[here]) ? b.elev[here] : NaN);
  if (!isNaN(stageObs)) {
    for (let i = 0; i < b.water.length; i++) if (b.waterT[i] === t && b.water[i] > 0 && !isNaN(b.dryElev[i])) b.water[i] = Math.max(b.water[i], stageObs - b.dryElev[i]);
  }

  // ---------- thermal camera: 36 sectors [maxTemp, dist] ----------
  const th = f.thermal.data, nsec = th.length / 3;
  let thermalMax = { temp: CFG.ambient, dist: -1, bearing: 0 };
  const hotSectors: Readings['hotSectors'] = [];
  for (let s = 0; s < nsec; s++) {
    const T = th[3 * s], d = th[3 * s + 1], hot = th[3 * s + 2];
    const a = -Math.PI + (s + 0.5) * 2 * Math.PI / nsec;
    const ca = Math.cos(a), sa = Math.sin(a);
    if (T > thermalMax.temp) thermalMax = { temp: T, dist: d, bearing: a };
    // everything nearer than the first >100 °C pixel is below 100 °C: treat as cool
    const coolTo = hot > 0 ? hot - 2 : CFG.thermalRange;
    let lastI = -1;
    for (let r = 2; r < coolTo; r += 2) {
      const px = x + r * ca, py = y + r * sa; if (!b.inside(px, py)) break;
      const i = b.idx(px, py); if (i === lastI) continue; lastI = i;
      b.heat[i] = T < 60 ? CFG.ambient : Math.min(isNaN(b.heat[i]) ? 60 : b.heat[i], 60); b.heatT[i] = t;
    }
    if (hot > 0) hotSectors.push({ bearing: a, dist: hot, temp: T });
    if (hot > 0) markHot(b, x + hot * ca, y + hot * sa, Math.max(150, Math.min(T, 400)), t);
    if (T > 60 && d > 0) markHot(b, x + d * ca, y + d * sa, T, t);
  }
  const tempC = status.temp_c;
  if (tempC > 45) { b.heat[here] = Math.max(isNaN(b.heat[here]) ? 0 : b.heat[here], tempC); b.heatT[here] = t; }

  // ---------- camera detections ----------
  const streaks: Detection[] = [], flashes: Detection[] = [];
  let funnel: Detection | null = null;
  for (let k = 0; k + 4 < smokeDet.length; k += 5) {
    const cls = smokeDet[k]; if (cls === 4) continue;
    const det: Detection = { cls, bearing: smokeDet[k + 1], dist: smokeDet[k + 2], size: smokeDet[k + 3], ttl: smokeDet[k + 4], x: 0, y: 0 };
    det.x = x + det.dist * Math.cos(det.bearing); det.y = y + det.dist * Math.sin(det.bearing);
    if (cls === 1) flashes.push(det); else if (cls === 2) funnel = det; else if (cls === 3) streaks.push(det);
  }

  // ---------- probes: each one is a remote sensor station ----------
  for (const pr of status.probes) {
    if (!pr.alive) continue;
    const sm = pr.samples;
    for (let k = 0; k + 4 < sm.length; k += 5) {
      const px = sm[k], py = sm[k + 1]; if (!b.inside(px, py)) continue;
      const i = b.idx(px, py);
      b.see(i, t);
      b.water[i] = sm[k + 2]; b.waterT[i] = t;
      b.heat[i] = sm[k + 3]; b.heatT[i] = t;
      b.elev[i] = sm[k + 4];
      if (sm[k + 2] < 0.03) b.dryElev[i] = sm[k + 4];
    }
  }

  // ---------- IMU shake: median of sample-to-sample accel jumps (rover motion is smooth, seismic is not) ----------
  // residual = measured accel minus what our own motion + gravity explains; seismic shaking is what's left
  const la = f.imu.linear_acceleration, v = f.odom.twist.twist.linear.x, w = f.odom.twist.twist.angular.z;
  if (!isNaN(ps.lastV) && dt > 0.02) {
    const ex = (v - ps.lastV) / dt + 9.81 * Math.sin(pitch), ey = v * w - 9.81 * Math.sin(roll), ez = 9.81 * Math.cos(pitch) * Math.cos(roll);
    ps.accDiffs.push(Math.hypot(la.x - ex, la.y - ey, la.z - ez)); if (ps.accDiffs.length > 10) ps.accDiffs.shift();
  }
  ps.lastV = v;
  const shake = median(ps.accDiffs);

  // ---------- wind & pressure ----------
  const wx = f.wind.vector.x, wy = f.wind.vector.y, ws = Math.hypot(wx, wy);
  const pr = f.pressure.fluid_pressure;
  if (isNaN(ps.pressureBase)) ps.pressureBase = pr;
  if (ws < CFG.tornadoWind * 0.7) {
    ps.pressureBase = ps.pressureBase * 0.97 + pr * 0.03;
    ps.ambX = ps.ambX * 0.95 + wx * 0.05; ps.ambY = ps.ambY * 0.95 + wy * 0.05; ps.ambN++;
  }

  return {
    t, dt, x, y, yaw, v: f.odom.twist.twist.linear.x, w: f.odom.twist.twist.angular.z, roll, pitch,
    tiltDeg: Math.max(Math.abs(roll), Math.abs(pitch)) * 180 / Math.PI,
    status, depth, tempC, windX: wx, windY: wy, windSpeed: ws, pressure: pr, pressureDrop: ps.pressureBase - pr, shake,
    lidarFront: front, returnFrac, smoke, thermalMax, hotSectors, streaks, flashes, funnel, waterNear, stageObs,
  };
}
export { clamp01 };
