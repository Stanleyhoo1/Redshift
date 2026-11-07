"""PERCEPTION: one sensor frame -> belief grid updates + a Readings summary. Mirror of web/brain/perception.ts.

The frame is a dict with ROS message shapes (same field names as the .msg files), so it can be
built from rclpy messages (brain_node.py) or from rosbridge JSON / the TS sim (tests).
"""
import json
import math

import numpy as np

from .config import CFG, NO_DATA, wrap


def _quat_to_yaw(q):
    return math.atan2(2 * (q['w'] * q['z'] + q['x'] * q['y']), 1 - 2 * (q['y'] * q['y'] + q['z'] * q['z']))


def _quat_to_roll_pitch(q):
    roll = math.atan2(2 * (q['w'] * q['x'] + q['y'] * q['z']), 1 - 2 * (q['x'] * q['x'] + q['y'] * q['y']))
    s = 2 * (q['w'] * q['y'] - q['z'] * q['x'])
    pitch = math.copysign(math.pi / 2, s) if abs(s) >= 1 else math.asin(s)
    return roll, pitch


def _median(a):
    if not a:
        return 0.0
    s = sorted(a)
    return s[len(s) >> 1]


class PerceptionState:
    def __init__(self):
        self.last_t = -1.0
        self.acc_diffs = []
        self.last_v = float('nan')
        self.pressure_base = float('nan')
        self.amb_x = 0.0
        self.amb_y = 0.0


class Readings:
    """Everything derived from ONE sensor frame. No ground truth anywhere in here."""
    __slots__ = ('t', 'dt', 'x', 'y', 'yaw', 'v', 'w', 'roll', 'pitch', 'tilt_deg', 'status', 'depth', 'temp_c',
                 'wind_x', 'wind_y', 'wind_speed', 'pressure', 'pressure_drop', 'shake', 'lidar_front', 'return_frac',
                 'smoke', 'scan_r', 'scan_a0', 'scan_inc', 'scan_max', 'thermal_max', 'hot_sectors', 'streaks', 'flashes', 'funnel', 'water_near', 'stage_obs', 'probe_alerts')


def _mark_hot(b, px, py, T, t):
    if not b.inside(px, py):
        return
    i = b.idx(px, py)
    if b.heat_t[i] == t and not math.isnan(b.heat[i]):
        b.heat[i] = max(b.heat[i], T)
    else:
        b.heat[i] = T
    b.heat_t[i] = t
    b.last_seen[i] = t


def perceive(b, f, ps):
    status = json.loads(f['rover_status']['data'])
    # frame time = /rover_status time (it closes the frame; topics can arrive out of order over rosbridge)
    t = float(status['t'])
    dt = 0.1 if ps.last_t < 0 else t - ps.last_t
    ps.last_t = t
    pose = f['odom']['pose']['pose']
    x, y = pose['position']['x'], pose['position']['y']
    yaw = _quat_to_yaw(pose['orientation'])
    roll, pitch = _quat_to_roll_pitch(f['imu']['orientation'])

    # ---------- LiDAR ----------
    sc = f['scan']
    rmax, rmin = sc['range_max'], sc['range_min']
    det = f['camera_detections']['data']
    smoke = 0.0
    for k in range(0, len(det) - 4, 5):
        if det[k] == 4:
            smoke = det[k + 3]
    free_no_ret = 6.0 if smoke > 0.15 else rmax
    returns = 0
    front = 99.0
    obst, last_seen = b.obst, b.last_seen
    for k, r in enumerate(sc['ranges']):
        if r is None:
            r = rmax + 1
        rel = sc['angle_min'] + k * sc['angle_increment']
        a = yaw + rel
        hit = rmin <= r <= rmax
        if hit:
            returns += 1
            if abs(wrap(rel)) < 0.35:
                front = min(front, r)
        free_to = r - 0.6 if hit else free_no_ret
        ca, sa = math.cos(a), math.sin(a)
        last_i = -1
        d = 1.0
        while d < free_to:
            px, py = x + d * ca, y + d * sa
            if not b.inside(px, py):
                break
            i = b.idx(px, py)
            if i != last_i:
                last_i = i
                o = obst[i] - 0.12
                obst[i] = o if o > 0 else 0.0
                last_seen[i] = t
            d += 1.0
        if hit:
            hx, hy = x + r * ca, y + r * sa
            if b.inside(hx, hy):
                i = b.idx(hx, hy)
                obst[i] = min(1.0, obst[i] + 0.45)
                last_seen[i] = t
    return_frac = returns / max(1, len(sc['ranges']))

    # ---------- depth camera elevation patch ----------
    e = f['elevation']['data']
    ox, oy, res, n = e[0], e[1], e[2], int(e[3])

    def ev(i, j):
        v = e[4 + j * n + i]
        return None if (v is None or v == NO_DATA) else v

    for j in range(n):
        for i in range(n):
            v = ev(i, j)
            if v is None:
                continue
            px, py = ox + i * res, oy + j * res
            if not b.inside(px, py):
                continue
            c = b.idx(px, py)
            b.elev[c] = v if math.isnan(b.elev[c]) else b.elev[c] * 0.5 + v * 0.5
            last_seen[c] = t
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            v = ev(i, j)
            if v is None:
                continue
            lo = v
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u = ev(i + di, j + dj)
                if u is not None and u < lo:
                    lo = u
            px, py = ox + i * res, oy + j * res
            if not b.inside(px, py):
                continue
            b.struct[b.idx(px, py)] = (v - lo) > 2.5

    # ---------- water ----------
    depth = f['water_depth']['data']
    here = b.idx(x, y)
    b.water[here] = depth
    b.water_t[here] = t
    if depth < 0.03 and not math.isnan(b.elev[here]):
        b.dry_elev[here] = b.elev[here]
    cw = f['camera_water']['data']
    ns = len(cw)
    water_near = None
    stage_samples = []
    vis = CFG['cameraRange'] * max(0.15, 1 - 1.6 * smoke)
    for s in range(ns):
        a = -math.pi + (s + 0.5) * 2 * math.pi / ns
        d = cw[s] if cw[s] is not None else -1
        ca, sa = math.cos(a), math.sin(a)
        dry_to = vis if d < 0 else d - 1
        last_i = -1
        r = 1.0
        while r < dry_to:
            px, py = x + r * ca, y + r * sa
            if not b.inside(px, py):
                break
            i = b.idx(px, py)
            if i != last_i:
                last_i = i
                b.water[i] = 0.0
                b.water_t[i] = t
                last_seen[i] = t
                if not math.isnan(b.elev[i]) and not b.struct[i]:
                    b.dry_elev[i] = b.elev[i]
            r += 1.0
        if d >= 0:
            if water_near is None or d < water_near[0]:
                water_near = (d, a)
            r = d
            while r < min(d + 6, vis + 2):
                px, py = x + r * ca, y + r * sa
                if not b.inside(px, py):
                    break
                i = b.idx(px, py)
                b.water_t[i] = t
                last_seen[i] = t
                if b.water[i] < 0.3:
                    b.water[i] = 0.3
                if r == d and d < 19 and not math.isnan(b.elev[i]):
                    stage_samples.append(b.elev[i])
                r += 1.0
    if stage_samples:
        stage_obs = _median(stage_samples)
    elif depth > 0.05 and not math.isnan(b.elev[here]):
        stage_obs = b.elev[here]
    else:
        stage_obs = float('nan')
    if not math.isnan(stage_obs):
        m = (b.water_t == t) & (b.water > 0) & ~np.isnan(b.dry_elev)
        b.water[m] = np.maximum(b.water[m], stage_obs - b.dry_elev[m])

    # ---------- thermal: 36 sectors x [maxTemp, distOfMax, distOfNearest>100C] ----------
    th = f['thermal']['data']
    nsec = len(th) // 3
    thermal_max = (CFG['ambient'], -1.0, 0.0)
    hot_sectors = []
    amb = CFG['ambient']
    for s in range(nsec):
        T, d, hot = th[3 * s], th[3 * s + 1], th[3 * s + 2]
        a = -math.pi + (s + 0.5) * 2 * math.pi / nsec
        ca, sa = math.cos(a), math.sin(a)
        if T > thermal_max[0]:
            thermal_max = (T, d, a)
        cool_to = hot - 2 if hot > 0 else CFG['thermalRange']
        last_i = -1
        r = 2.0
        while r < cool_to:
            px, py = x + r * ca, y + r * sa
            if not b.inside(px, py):
                break
            i = b.idx(px, py)
            if i != last_i:
                last_i = i
                b.heat[i] = amb if T < 60 else min(60.0 if math.isnan(b.heat[i]) else b.heat[i], 60.0)
                b.heat_t[i] = t
            r += 2.0
        if hot > 0:
            hot_sectors.append((a, hot, T))
            _mark_hot(b, x + hot * ca, y + hot * sa, max(150.0, min(T, 400.0)), t)
        if T > 60 and d > 0:
            _mark_hot(b, x + d * ca, y + d * sa, T, t)
    temp_c = status['temp_c']
    if temp_c > 45:
        b.heat[here] = max(0.0 if math.isnan(b.heat[here]) else b.heat[here], temp_c)
        b.heat_t[here] = t

    # ---------- camera detections ----------
    streaks, flashes, funnel = [], [], None
    for k in range(0, len(det) - 4, 5):
        cls = det[k]
        if cls == 4:
            continue
        d = dict(cls=cls, bearing=det[k + 1], dist=det[k + 2], size=det[k + 3], ttl=det[k + 4])
        d['x'] = x + d['dist'] * math.cos(d['bearing'])
        d['y'] = y + d['dist'] * math.sin(d['bearing'])
        if cls == 1:
            flashes.append(d)
        elif cls == 2:
            funnel = d
        elif cls == 3:
            streaks.append(d)

    # ---------- probes ----------
    probe_alerts, probe_stage = [], []
    for pr in status['probes']:
        if not pr['alive']:
            continue
        sm = pr['samples']
        for k in range(0, len(sm) - 4, 5):
            px, py = sm[k], sm[k + 1]
            if not b.inside(px, py):
                continue
            i = b.idx(px, py)
            was_dry = b.water[i] == 0
            was_cool = (not math.isnan(b.heat[i])) and b.heat[i] < 60
            if sm[k + 2] > 0.05 and was_dry:
                probe_alerts.append(dict(id=pr['id'], kind='water', x=px, y=py, value=sm[k + 2]))
            if sm[k + 3] > 100 and was_cool:
                probe_alerts.append(dict(id=pr['id'], kind='heat', x=px, y=py, value=sm[k + 3]))
            if sm[k + 2] > 0.05:
                probe_stage.append(sm[k + 4])
            last_seen[i] = t
            b.water[i] = sm[k + 2]
            b.water_t[i] = t
            b.heat[i] = sm[k + 3]
            b.heat_t[i] = t
            b.elev[i] = sm[k + 4]
            if sm[k + 2] < 0.03:
                b.dry_elev[i] = sm[k + 4]

    if math.isnan(stage_obs) and probe_stage:
        stage_obs = _median(probe_stage)

    # ---------- IMU shake residual ----------
    la = f['imu']['linear_acceleration']
    tw = f['odom']['twist']['twist']
    v, w = tw['linear']['x'], tw['angular']['z']
    if not math.isnan(ps.last_v) and dt > 0.02:
        ex = (v - ps.last_v) / dt + 9.81 * math.sin(pitch)
        ey = v * w - 9.81 * math.sin(roll)
        ez = 9.81 * math.cos(pitch) * math.cos(roll)
        ps.acc_diffs.append(math.sqrt((la['x'] - ex) ** 2 + (la['y'] - ey) ** 2 + (la['z'] - ez) ** 2))
        if len(ps.acc_diffs) > 10:
            ps.acc_diffs.pop(0)
    ps.last_v = v
    shake = _median(ps.acc_diffs)

    # ---------- wind & pressure ----------
    wx, wy = f['wind']['vector']['x'], f['wind']['vector']['y']
    ws = math.hypot(wx, wy)
    pr = f['pressure']['fluid_pressure']
    if math.isnan(ps.pressure_base):
        ps.pressure_base = pr
    if ws < CFG['tornadoWind'] * 0.7:
        ps.pressure_base = ps.pressure_base * 0.97 + pr * 0.03
        ps.amb_x = ps.amb_x * 0.95 + wx * 0.05
        ps.amb_y = ps.amb_y * 0.95 + wy * 0.05

    rd = Readings()
    rd.t, rd.dt, rd.x, rd.y, rd.yaw, rd.v, rd.w, rd.roll, rd.pitch = t, dt, x, y, yaw, v, w, roll, pitch
    rd.tilt_deg = max(abs(roll), abs(pitch)) * 180 / math.pi
    rd.status, rd.depth, rd.temp_c = status, depth, temp_c
    rd.wind_x, rd.wind_y, rd.wind_speed, rd.pressure, rd.pressure_drop = wx, wy, ws, pr, ps.pressure_base - pr
    rd.shake, rd.lidar_front, rd.return_frac, rd.smoke = shake, front, return_frac, smoke
    rd.scan_r = [rmax + 1 if r is None else r for r in sc['ranges']]
    rd.scan_a0, rd.scan_inc, rd.scan_max = yaw + sc['angle_min'], sc['angle_increment'], rmax
    rd.thermal_max, rd.hot_sectors = thermal_max, hot_sectors
    rd.streaks, rd.flashes, rd.funnel, rd.water_near, rd.stage_obs = streaks, flashes, funnel, water_near, stage_obs
    rd.probe_alerts = probe_alerts
    return rd
