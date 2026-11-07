"""Pure helpers turning cached ROS data into the WebSocket payloads of docs/protocol.md §7."""
from __future__ import annotations

import base64
import collections
import math
import struct
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import _paths  # noqa: F401
from redshift_autonomy.agent_fsm import display_name, role_of


def png_b64(rgb: np.ndarray) -> str:
    """RGB uint8 image → base64 PNG (row 0 stays row 0 = south edge of the sector)."""
    ok, buf = cv2.imencode('.png', cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError('PNG encode failed')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def lidar_points_body(points: np.ndarray, lidar_xyz, max_points: int = 1500) -> list:
    """Sensor-frame cloud (any shape (..., 3), NaN/inf = no return) → ≤ max_points robot-frame [x, y, z]."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) > max_points:
        p = p[np.linspace(0, len(p) - 1, max_points).astype(int)]
    p = p + np.asarray(lidar_xyz, dtype=np.float64)
    return np.round(p, 2).tolist()


def downsample_path(xy, max_n: int = 150) -> list:
    if xy is None or len(xy) == 0:
        return []
    p = np.asarray(xy, dtype=np.float64)
    if len(p) > max_n:
        p = p[np.unique(np.r_[np.linspace(0, len(p) - 1, max_n - 1).astype(int), len(p) - 1])]
    return np.round(p, 2).tolist()


class Trail:
    def __init__(self, spacing_m: float = 0.5, cap: int = 300):
        self.spacing = spacing_m
        self.cap = cap
        self._pts: list = []

    def add(self, x: float, y: float) -> None:
        if not self._pts or math.hypot(x - self._pts[-1][0], y - self._pts[-1][1]) >= self.spacing - 1e-9:
            self._pts.append([round(x, 2), round(y, 2)])
            if len(self._pts) > self.cap:
                del self._pts[:len(self._pts) - self.cap]

    def points(self) -> list:
        return list(self._pts)

    def clear(self) -> None:
        self._pts.clear()


@dataclass
class RobotCache:
    name: str
    pose: dict | None = None
    imu: dict | None = None
    agent_state: dict | None = None
    edge: dict | None = None
    detections: dict | None = None
    path: list = field(default_factory=list)
    trail: Trail = field(default_factory=Trail)
    jpeg_b64: str | None = None
    jpeg_t: float = 0.0
    depth_b64: str | None = None
    depth_t: float = 0.0
    cloud: np.ndarray | None = None
    cloud_t: float = 0.0
    first: dict = field(default_factory=dict)          # topic kind -> first-arrival time (readiness)

    @property
    def rtype(self) -> str:
        return 'rover' if self.name.startswith('rover') else 'scout'


_ZERO_POSE = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}


def robot_telemetry(c: RobotCache) -> dict:
    a = c.agent_state or {}
    pose = c.pose or _ZERO_POSE
    imu = c.imu or {'roll_deg': round(math.degrees(pose['roll']), 2), 'pitch_deg': round(math.degrees(pose['pitch']), 2),
                    'gyro': [0.0, 0.0, 0.0]}
    unc = (c.detections or {}).get('uncertainty') or {}
    path = c.path
    if c.rtype == 'scout' and not path and a.get('goal') and c.pose:
        path = [[round(pose['x'], 2), round(pose['y'], 2)], [round(v, 2) for v in a['goal']]]
    edge = dict(c.edge or {})
    edge.setdefault('robot', c.name)
    for k, v in (('profile', ''), ('host_ms', {'perception': 0.0, 'planner': 0.0, 'mapping': 0.0}), ('device_ms', 0.0),
                 ('loop_hz', 0.0), ('load_pct', 0.0), ('power_w', 0.0), ('perception_hz', 0.0),
                 ('throttle_level', 0)):
        edge.setdefault(k, v)
    if edge.get('uncertainty') is None:
        edge['uncertainty'] = float(unc.get('score', 0.0))
    return {
        'type': c.rtype,
        'pose': {k: round(float(pose[k]), 4) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')},
        'speed': float(a.get('speed') or 0.0),
        'state': a.get('state', 'BOOT'),
        'mode': a.get('mode', 'FUSED'),
        'goal': a.get('goal'),
        'v_limit': float(a.get('v_limit') or 0.0),
        'slip': float(a.get('slip') or 0.0),
        'tip_alarm_deg': float(a.get('tip_alarm_deg') or 0.0),
        'max_slope_deg': float(a.get('max_slope_deg') or 0.0),
        'altitude_agl': a.get('altitude_agl'),
        'imu': imu,
        'uncertainty': {'score': float(unc.get('score', 0.0)), 'camera_conf': float(unc.get('camera_conf', 1.0)),
                        'lidar_conf': float(unc.get('lidar_conf', 1.0)), 'imu_vib': float(unc.get('imu_vib', 0.0))},
        'edge': edge,
        'path': path,
        'trail': c.trail.points(),
        'last_safe': a.get('last_safe'),
        'role': a.get('role') or role_of(c.name),
        'name': a.get('name') or display_name(c.name),
        'health': a.get('health'),
        'rescue': a.get('rescue'),
        'sensors': sensor_readings(c, imu, unc),
    }


def sensor_readings(c: RobotCache, imu: dict, unc: dict) -> dict:
    """Live instrument values for the sensor panel (docs/protocol.md §8.7)."""
    a = c.agent_state or {}
    lid = (c.detections or {}).get('lidar') or {}
    acc = imu.get('accel')
    rover = c.rtype == 'rover'
    return {
        'accel': acc, 'accel_g': None if acc is None else round(math.sqrt(sum(v * v for v in acc)) / 9.80665, 3),
        'gyro': imu.get('gyro'), 'roll_deg': imu.get('roll_deg'), 'pitch_deg': imu.get('pitch_deg'),
        'lidar_nearest_m': lid.get('nearest_m') if rover else None,
        'lidar_returns_pct': lid.get('returns_pct') if rover else None,
        'camera_conf': float(unc.get('camera_conf', 1.0)),
        'v_cmd': a.get('v_cmd'), 'v_wheel': a.get('v_wheel'), 'v_ground': a.get('v_ground', a.get('speed')),
        'slip': a.get('slip'), 'altitude_agl': a.get('altitude_agl'),
    }


def decode_clock_cdr(raw: bytes) -> float | None:
    """Seconds from a serialized rosgraph_msgs/Clock (CDR: 4-byte encapsulation header, int32 sec, uint32 nanosec).
    The backend subscribes to /clock raw (≈100–200 Hz), so it skips rclpy deserialisation."""
    if len(raw) < 12:
        return None
    endian = '<' if raw[1] == 1 else '>'
    sec, nsec = struct.unpack_from(endian + 'iI', raw, 4)
    return sec + nsec * 1e-9


class RtfMeter:
    """Achieved real-time factor: simulated seconds per wall second over a sliding window."""

    def __init__(self, window_s: float = 2.0, sample_every_s: float = 0.05):
        self.window_s, self.every = window_s, sample_every_s
        self._samples: collections.deque = collections.deque()
        self._latest: tuple[float, float] | None = None

    @property
    def sim_t(self) -> float | None:
        return self._latest[1] if self._latest else None

    def add(self, wall: float, sim: float):
        if self._latest and sim + 1e-6 < self._latest[1]:
            self._samples.clear()                  # clock went backwards: a new world started
        self._latest = (wall, sim)
        if not self._samples or wall - self._samples[-1][0] >= self.every:
            self._samples.append((wall, sim))
        while self._samples and wall - self._samples[0][0] > self.window_s:
            self._samples.popleft()

    def rtf(self, now_wall: float) -> float | None:
        if not self._latest or not self._samples:
            return None
        w1, s1 = self._latest
        if now_wall - w1 > 1.0:
            return 0.0                             # no clock for a second: stalled or paused
        w0, s0 = self._samples[0]
        if w1 - w0 < 0.5:
            return None
        return (s1 - s0) / (w1 - w0)
