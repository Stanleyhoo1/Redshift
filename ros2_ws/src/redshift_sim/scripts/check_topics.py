#!/usr/bin/env python3
"""Topic-contract parity check between the two world backends (Gazebo + ros_gz bridge vs. mock world).

Listens to every *raw* world topic of the given robots for a few seconds and prints a signature per topic
(type, rate, frame ids, image encodings/shapes, camera intrinsics, point-cloud layout and ring order, IMU
gravity, no-return encoding). Save one backend's signature and compare the other against it:

    python3 check_topics.py --robots rover_0 scout_0 --save gz_signature.json      # with Gazebo running
    python3 check_topics.py --robots rover_0 scout_0 --compare gz_signature.json   # with the mock running

`gz_signature.json` next to this script is the reference captured from Gazebo Harmonic 8.15 (medium sector,
1 rover + 1 scout, robots at rest).

Exit code 0 = every topic present and structurally identical, rates within ±30 %.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2

RATE_TOL = 0.30
G0 = 9.80665


def topics_for(robot: str) -> dict:
    t = {'odom': Odometry, 'imu_raw': Imu, 'camera/image_raw': Image, 'camera/depth_raw': Image,
         'camera/camera_info': CameraInfo}
    if robot.startswith('rover'):
        t.update({'wheel_odom': Odometry, 'lidar/points_raw': PointCloud2})
    return {f'/{robot}/{k}': v for k, v in t.items()}


def _rpy(q):
    r = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    p = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    y = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return r, p, y


def _nonfinite(a: np.ndarray) -> dict:
    n = max(a.size, 1)
    return {'finite_frac': round(float(np.isfinite(a).sum()) / n, 3),
            'posinf_frac': round(float(np.isposinf(a).sum()) / n, 3),
            'neginf_frac': round(float(np.isneginf(a).sum()) / n, 3),
            'nan_frac': round(float(np.isnan(a).sum()) / n, 3)}


def describe(msg, odom_yaw: float | None) -> dict:
    """Structural signature of one message (keys starting with '~' are informative, not compared)."""
    d = {'frame_id': msg.header.frame_id}
    if isinstance(msg, Image):
        d.update(encoding=msg.encoding, width=msg.width, height=msg.height, step=msg.step,
                 is_bigendian=msg.is_bigendian)
        if msg.encoding == '32FC1':
            a = np.frombuffer(bytes(msg.data), np.float32)
            d['~depth'] = _nonfinite(a)
            fin = a[np.isfinite(a)]
            d['~depth_range'] = [round(float(fin.min()), 2), round(float(fin.max()), 2)] if fin.size else None
            d['no_return'] = ('+inf' if np.isposinf(a).any() else '') + ('nan' if np.isnan(a).any() else '') or 'none'
        else:
            a = np.frombuffer(bytes(msg.data), np.uint8)
            d['~mean_rgb'] = [round(float(v), 1) for v in a.reshape(-1, 3).mean(axis=0)] if a.size % 3 == 0 else None
    elif isinstance(msg, CameraInfo):
        d.update(width=msg.width, height=msg.height, k=[round(float(v), 1) for v in msg.k],
                 p=[round(float(v), 1) for v in msg.p], distortion_model=msg.distortion_model)
    elif isinstance(msg, PointCloud2):
        d.update(width=msg.width, height=msg.height, point_step=msg.point_step, is_dense=bool(msg.is_dense),
                 fields=[[f.name, f.offset, f.datatype, f.count] for f in msg.fields])
        offs = {f.name: f.offset for f in msg.fields}
        raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
        xyz = np.stack([raw[:, offs[k]:offs[k] + 4].copy().view(np.float32)[:, 0] for k in 'xyz'], axis=1)
        d['~points'] = _nonfinite(xyz[:, 0])
        d['no_return'] = ('+inf' if np.isposinf(xyz).any() else '') + ('-inf' if np.isneginf(xyz).any() else '') + \
                         ('nan' if np.isnan(xyz).any() else '') or 'none'
        g = xyz.reshape(msg.height, msg.width, 3).astype(np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            elev = np.degrees(np.arctan2(g[..., 2], np.hypot(g[..., 0], g[..., 1])))
            az = np.degrees(np.arctan2(g[..., 1], g[..., 0]))
        row_el = np.nanmedian(elev, axis=1)
        col_az = np.nanmedian(az, axis=0)
        d['ring0_elev_deg'] = round(float(row_el[0]), 0) if np.isfinite(row_el[0]) else None
        d['ring_order'] = 'ascending' if np.nanmean(np.diff(row_el)) > 0 else 'descending'
        fin = np.isfinite(col_az)
        if fin.sum() > 10:
            idx = np.flatnonzero(fin)
            a = np.radians(col_az[idx])
            step = np.diff(np.unwrap(a))
            ccw = np.median(step) > 0
            d['col_order'] = 'ccw' if ccw else 'cw'
            # circular mean of (azimuth − column·Δ): the azimuth of column 0, robust to the ±180° wrap
            off = a - (1 if ccw else -1) * idx * 2 * math.pi / msg.width
            az0 = round(math.degrees(math.atan2(np.sin(off).mean(), np.cos(off).mean())), 0)
            d['col0_az_deg'] = (az0 + 180.0) % 360.0 - 180.0          # ±180 → −180
    elif isinstance(msg, Imu):
        a = msg.linear_acceleration
        d['~accel_g'] = round(math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z) / G0, 2)
        r, p, y = _rpy(msg.orientation)
        d['~roll_pitch_deg'] = [round(math.degrees(r), 1), round(math.degrees(p), 1)]
        if odom_yaw is not None:
            dy = math.degrees((y - odom_yaw + math.pi) % (2 * math.pi) - math.pi)
            d['orientation_ref'] = 'world' if abs(dy) < 5 else f'offset {dy:.0f} deg'
    elif isinstance(msg, Odometry):
        d['child_frame_id'] = msg.child_frame_id
        d['~z'] = round(msg.pose.pose.position.z, 2)
    return d


class Collector:
    def __init__(self, node, robots):
        self.node = node
        self.samples: dict[str, list] = {}
        self.first: dict[str, dict] = {}
        self.last: dict[str, dict] = {}
        self.odom_yaw: dict[str, float] = {}
        self.types: dict[str, str] = {}
        for r in robots:
            for name, typ in topics_for(r).items():
                self.samples[name] = []
                self.types[name] = f'{typ.__module__.split(".")[0]}/{typ.__name__}'
                node.create_subscription(typ, name, lambda m, n=name, r=r: self.on_msg(n, r, m), qos_profile_sensor_data)

    def on_msg(self, name, robot, msg):
        self.samples[name].append(time.monotonic())
        if name.endswith('/odom'):
            self.odom_yaw[robot] = _rpy(msg.pose.pose.orientation)[2]
        if name not in self.first or len(self.samples[name]) == 5:   # 5th message: past any start-up transients
            d = describe(msg, self.odom_yaw.get(robot))
            self.first.setdefault(name, d)
            self.last[name] = d

    def signature(self, duration: float) -> dict:
        out = {}
        for name, ts in self.samples.items():
            ts = ts[len(ts) // 10:]
            hz = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 0.0
            out[name] = dict(type=self.types[name], hz=round(hz, 1), count=len(self.samples[name]),
                             **(self.last.get(name) or {}))
        return out


def compare(ref: dict, cur: dict) -> list[str]:
    errors = []
    for name, r in ref.items():
        c = cur.get(name)
        if not c or not c.get('count'):
            errors.append(f'{name}: missing (reference has {r.get("hz")} Hz)')
            continue
        if r['hz'] and abs(c['hz'] - r['hz']) > RATE_TOL * r['hz']:
            errors.append(f'{name}: rate {c["hz"]} Hz vs reference {r["hz"]} Hz')
        for k, v in r.items():
            if k.startswith('~') or k in ('hz', 'count'):
                continue
            if c.get(k) != v:
                errors.append(f'{name}: {k} = {c.get(k)!r}, reference {v!r}')
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robots', nargs='+', default=['rover_0', 'scout_0'])
    ap.add_argument('--duration', type=float, default=6.0)
    ap.add_argument('--save')
    ap.add_argument('--compare')
    a = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node('check_topics')
    col = Collector(node, a.robots)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.duration:
        rclpy.spin_once(node, timeout_sec=0.05)
    sig = col.signature(a.duration)
    node.destroy_node()
    rclpy.try_shutdown()
    for name, s in sig.items():
        print(f'{name:32s} {s["hz"]:6.1f} Hz  ' + json.dumps({k: v for k, v in s.items() if k not in ('hz', 'type', 'count')}))
    if a.save:
        with open(a.save, 'w') as f:
            json.dump(sig, f, indent=1)
        print(f'saved signature to {a.save}')
    if a.compare:
        with open(a.compare) as f:
            ref = json.load(f)
        errors = compare(ref, sig)
        for e in errors:
            print('MISMATCH', e)
        print('PARITY OK' if not errors else f'{len(errors)} mismatches')
        sys.exit(1 if errors else 0)
    sys.exit(0 if all(s['count'] for s in sig.values()) else 1)


if __name__ == '__main__':
    main()
