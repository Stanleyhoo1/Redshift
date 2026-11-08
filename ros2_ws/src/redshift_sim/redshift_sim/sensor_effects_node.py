"""Atmospheric / seismic environment model applied to raw sensor streams (both world backends).

Clear conditions: zero-copy passthrough  *_raw → clean topic names.
Dust anomaly (level L ramps in/out over ~1.5 s):
  RGB   haze toward dust colour (0.65·L), blur σ = 1 + 3L, sensor noise σ = 25L, drifting particles
  depth visibility clip at 4 + 36(1 − L) m, 40·L % dropout, range noise
  LiDAR 25·L % dropout, range noise σ = 0.08·L m, 2·L % near-field dust backscatter returns
Dust storm (anomaly kind `dust_storm`, protocol §10.1): the same model at the storm's level, except the RGB haze goes
  toward orange-red STORM_RGB at 0.75·L; its wind (`wind_mps`, ramped like the dust level) is published on
  /world/env_state for the world backends, which push the drone with it.
Seismic: accelerometer/gyro vibration on the IMU.
The autonomy stack never reads the dust flag — it must infer degradation from the data itself.
"""
from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import String

DUST_RGB = np.array([170, 120, 80], np.float32)   # RGB order used on rgb8 images below
STORM_RGB = np.array([205, 88, 48], np.float32)   # dust storm: orange-red
STORM_KIND = 'dust_storm'
HAZE = {'dust': (DUST_RGB, 0.65), STORM_KIND: (STORM_RGB, 0.75)}   # kind → (haze colour, strength per unit level)
RAMP_S = 1.5
PARTICLES = 60                                     # drifting dust streaks at full level
PARTICLE_RGB = (235, 205, 165)
# command defaults: a plain anomaly as before; a storm as the backend's /api/inject/storm sends it
DUST_DEFAULTS = {'dust_level': 0.85, 'duration_s': 25.0, 'seismic_s': 4.0, 'wind_mps': 0.0}
STORM_DEFAULTS = {'dust_level': 1.0, 'duration_s': 45.0, 'seismic_s': 0.0, 'wind_mps': 14.0}
WIND_MAX_MPS = 40.0                                # clamp on a commanded storm wind


def _num(v, default: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return float(np.clip(x, lo, hi)) if math.isfinite(x) else default


def anomaly_params(cmd: dict) -> dict:
    """Validated parameters of an `anomaly` command: {kind, dust_level, duration_s, seismic_s, wind_mps}. Kind is
    `dust_storm` or `dust` (no or any other kind: the plain anomaly, which never carries wind). A missing or bad
    value takes its kind's default."""
    kind = STORM_KIND if cmd.get('kind') == STORM_KIND else 'dust'
    d = STORM_DEFAULTS if kind == STORM_KIND else DUST_DEFAULTS
    return {'kind': kind,
            'dust_level': _num(cmd.get('dust_level', d['dust_level']), d['dust_level'], 0.0, 1.0),
            'duration_s': _num(cmd.get('duration_s', d['duration_s']), d['duration_s'], 0.0, math.inf),
            'seismic_s': _num(cmd.get('seismic_s', d['seismic_s']), d['seismic_s'], 0.0, math.inf),
            'wind_mps': _num(cmd.get('wind_mps', d['wind_mps']), d['wind_mps'], 0.0, WIND_MAX_MPS)
            if kind == STORM_KIND else 0.0}


def haze_tint(img: np.ndarray, level: float, kind: str | None = None, bgr: bool = False) -> np.ndarray:
    """Blend a float HxWx3 frame toward the haze colour of `kind`: brown DUST_RGB at 0.65·L for plain dust (or no
    kind), orange-red STORM_RGB at 0.75·L for a dust storm. `bgr` = the frame is in BGR order."""
    rgb, k = HAZE.get(kind, HAZE['dust'])
    a = k * float(level)
    return img * (1 - a) + (rgb[::-1] if bgr else rgb) * a


def degrade_rgb(img: np.ndarray, level: float, kind: str | None, rng: np.random.Generator, particles: np.ndarray,
                bgr: bool = False) -> np.ndarray:
    """One uint8 HxWx3 camera frame through the dust: haze_tint, blur σ = 1 + 3L, sensor noise σ = 25L and drifting
    particles (`particles`: PARTICLES×4 state in [0, 1), advanced in place). Returns uint8."""
    out = haze_tint(img.astype(np.float32), level, kind, bgr)
    out = cv2.GaussianBlur(out, (0, 0), 1.0 + 3.0 * level)
    out += rng.normal(0, 25.0 * level, out.shape).astype(np.float32)
    h, w = out.shape[:2]
    p = particles
    p[:, 0] = (p[:, 0] + 0.02 + 0.03 * p[:, 2]) % 1.0
    p[:, 1] = (p[:, 1] + 0.004 * (p[:, 3] - 0.5)) % 1.0
    for k in range(int(PARTICLES * level)):
        x0, y0 = int(p[k, 0] * w), int(p[k, 1] * h)
        cv2.line(out, (x0, y0), (x0 + int(4 + 8 * p[k, 2]), y0 + 1), PARTICLE_RGB, 1, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ramp(value: float, goal: float, step: float) -> float:
    return min(goal, value + step) if goal > value else max(goal, value - step)


class Atmosphere:
    """The anomaly state behind every stream (simulated time): dust level L and storm wind, both ramped over RAMP_S,
    the seismic window, and the anomaly kind, which stays set until the dust has fully settled again."""

    def __init__(self):
        self.target = 0.0
        self.level = 0.0
        self.until = 0.0
        self.seismic_until = 0.0
        self.kind: str | None = None
        self.wind_peak = 0.0          # the storm's wind_mps (0 for plain dust)
        self.wind = 0.0
        self._wind_rate = 0.0         # m/s per s: the wind reaches its peak (or calm) in RAMP_S, like the dust level

    def command(self, cmd: dict, now: float) -> dict:
        """Start an anomaly at `now`; returns the /world/events echo (the node adds `t`)."""
        p = anomaly_params(cmd)
        self.kind = p['kind']
        self.target = p['dust_level']
        self.until = now + p['duration_s']
        self.seismic_until = now + p['seismic_s']
        self.wind_peak = p['wind_mps']
        self._wind_rate = max(self.wind_peak, self.wind) / RAMP_S
        return {'op': 'anomaly', 'ok': True, 'dust_level': self.target, 'duration_s': p['duration_s'],
                'kind': self.kind, 'wind_mps': self.wind_peak}

    def clear(self, now: float) -> None:
        self.until = now
        self.seismic_until = now

    def seismic(self, now: float) -> bool:
        return now < self.seismic_until

    def step(self, now: float, dt: float) -> None:
        dt = max(0.0, dt)
        active = now < self.until
        self.level = _ramp(self.level, self.target if active else 0.0, dt / RAMP_S)
        self.wind = _ramp(self.wind, self.wind_peak if active else 0.0, dt * self._wind_rate)
        if not active and self.level == 0.0 and self.wind == 0.0:
            self.kind = None

    def env_state(self, now: float) -> dict:
        """The /world/env_state payload (protocol §10.1): read by the world backends, never by the autonomy."""
        return {'dust_level': round(self.level, 3), 'seismic': self.seismic(now),
                'remaining_s': round(self.until - now, 1) if now < self.until else None,
                'kind': self.kind, 'wind_mps': round(self.wind, 2)}


class SensorEffects(Node):
    def __init__(self):
        super().__init__('sensor_effects')
        self.declare_parameter('robot', 'rover_0')
        self.declare_parameter('robot_type', 'rover')
        self.declare_parameter('publish_env', False)
        self.robot = self.get_parameter('robot').value
        rtype = self.get_parameter('robot_type').value
        self.publish_env = bool(self.get_parameter('publish_env').value)
        self.rng = np.random.default_rng(abs(hash(self.robot)) % 2 ** 32)
        r = self.robot
        self.atm = Atmosphere()
        self._last_t = self.now_s()
        self.pub_img = self.create_publisher(Image, f'/{r}/camera/image', qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, f'/{r}/camera/depth', qos_profile_sensor_data)
        self.pub_imu = self.create_publisher(Imu, f'/{r}/imu', qos_profile_sensor_data)
        self.create_subscription(Image, f'/{r}/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(Image, f'/{r}/camera/depth_raw', self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Imu, f'/{r}/imu_raw', self.on_imu, qos_profile_sensor_data)
        if rtype == 'rover':
            self.pub_pts = self.create_publisher(PointCloud2, f'/{r}/lidar/points', qos_profile_sensor_data)
            self.create_subscription(PointCloud2, f'/{r}/lidar/points_raw', self.on_points, qos_profile_sensor_data)
        self.create_subscription(String, '/world/command', self.on_cmd, 10)
        self.env_pub = self.create_publisher(String, '/world/env_state', 10)
        self.events = self.create_publisher(String, '/world/events', 10)
        self.create_timer(0.1, self.tick)
        self._particles = self.rng.random((PARTICLES, 4)).astype(np.float32)

    def now_s(self) -> float:
        """Simulated time (use_sim_time): durations and ramps follow the simulation speed."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ control
    def on_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(cmd, dict):
            return
        now = self.now_s()
        if cmd.get('op') == 'anomaly':
            ev = self.atm.command(cmd, now)
            if self.publish_env:
                self.events.publish(String(data=json.dumps(dict(ev, t=time.time()))))
        elif cmd.get('op') == 'clear_anomaly':
            self.atm.clear(now)

    def tick(self):
        now = self.now_s()
        dt = max(0.0, now - self._last_t)
        self._last_t = now
        self.atm.step(now, dt)
        if self.publish_env:
            self.env_pub.publish(String(data=json.dumps(self.atm.env_state(now))))

    # ------------------------------------------------------------------ streams
    def on_image(self, msg: Image):
        L = self.atm.level
        if L < 0.02 or msg.encoding not in ('rgb8', 'bgr8'):
            self.pub_img.publish(msg)
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        out = Image(header=msg.header, height=msg.height, width=msg.width, encoding=msg.encoding,
                    is_bigendian=msg.is_bigendian, step=msg.step)
        out.data = degrade_rgb(img, L, self.atm.kind, self.rng, self._particles, bgr=msg.encoding == 'bgr8').tobytes()
        self.pub_img.publish(out)

    def on_depth(self, msg: Image):
        L = self.atm.level
        if L < 0.02 or msg.encoding != '32FC1':
            self.pub_depth.publish(msg)
            return
        d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
        vis = 4.0 + 36.0 * (1 - L)
        d[d > vis] = np.nan
        d[self.rng.random(d.shape) < 0.4 * L] = np.nan
        d += (self.rng.normal(0, 0.02 * L, d.shape) * np.nan_to_num(d, nan=0.0)).astype(np.float32)
        out = Image(header=msg.header, height=msg.height, width=msg.width, encoding=msg.encoding,
                    is_bigendian=msg.is_bigendian, step=msg.step)
        out.data = d.astype(np.float32).tobytes()
        self.pub_depth.publish(out)

    def on_points(self, msg: PointCloud2):
        L = self.atm.level
        if L < 0.02:
            self.pub_pts.publish(msg)
            return
        offs = {f.name: f.offset for f in msg.fields}
        stride = msg.point_step // 4
        arr = np.frombuffer(msg.data, np.float32).reshape(-1, stride).copy()
        xi, yi, zi = offs['x'] // 4, offs['y'] // 4, offs['z'] // 4
        xyz = arr[:, [xi, yi, zi]]
        r = np.linalg.norm(xyz, axis=1)
        fin = np.isfinite(r) & (r > 0)
        scale = np.ones_like(r)
        scale[fin] = (r[fin] + self.rng.normal(0, 0.08 * L, int(fin.sum()))) / r[fin]
        n = len(r)
        back = self.rng.random(n) < 0.02 * L
        dirs = np.where(fin[:, None], xyz / np.where(fin, r, 1.0)[:, None], 0.0)
        xyz = xyz * scale[:, None]
        if back.any():
            rr = self.rng.uniform(0.4, 2.0, int(back.sum()))
            # rays without a return still have a direction implied by their organized index; use a random one
            d_back = dirs[back]
            empty = ~np.any(d_back, axis=1)
            if empty.any():
                az = self.rng.uniform(-math.pi, math.pi, int(empty.sum()))
                d_back[empty] = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)
            xyz[back] = d_back * rr[:, None]
        drop = self.rng.random(n) < 0.25 * L
        xyz[drop] = np.inf
        arr[:, [xi, yi, zi]] = xyz
        out = PointCloud2(header=msg.header, height=msg.height, width=msg.width, fields=msg.fields,
                          is_bigendian=msg.is_bigendian, point_step=msg.point_step, row_step=msg.row_step,
                          is_dense=False)
        out.data = arr.tobytes()
        self.pub_pts.publish(out)

    def on_imu(self, msg: Imu):
        t = self.now_s()
        if not self.atm.seismic(t):
            self.pub_imu.publish(msg)
            return
        a = msg.linear_acceleration
        w = msg.angular_velocity
        n = self.rng.normal(0, 1.0, 6)
        s = math.sin(2 * math.pi * 6.0 * t)
        a.x += 3.0 * n[0] + 4.0 * s
        a.y += 3.0 * n[1] + 3.0 * math.cos(2 * math.pi * 7.3 * t)
        a.z += 3.0 * n[2]
        w.x += 0.3 * n[3]
        w.y += 0.3 * n[4]
        w.z += 0.15 * n[5]
        self.pub_imu.publish(msg)


def main():
    rclpy.init()
    node = SensorEffects()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
