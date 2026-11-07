"""Small ROS <-> numpy helpers shared by the autonomy nodes."""
from __future__ import annotations

import math

import numpy as np


def quat_to_rpy(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_rp(roll: float, pitch: float) -> np.ndarray:
    """Rotation that maps body vectors into a gravity-levelled (yaw-free) frame."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


def image_to_numpy(msg) -> np.ndarray:
    if msg.encoding in ('rgb8', 'bgr8'):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        return a if msg.encoding == 'rgb8' else a[..., ::-1]
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    raise ValueError(f'unsupported encoding {msg.encoding}')


def cloud_to_numpy(msg) -> np.ndarray:
    """PointCloud2 (float32 x, y, z at any offsets) → (height, width, 3) float32 with NaN for no-return."""
    offs = {f.name: f.offset for f in msg.fields}
    stride = msg.point_step // 4
    arr = np.frombuffer(msg.data, np.float32).reshape(-1, stride)
    xyz = arr[:, [offs['x'] // 4, offs['y'] // 4, offs['z'] // 4]].astype(np.float32)
    xyz[~np.isfinite(xyz).all(axis=1)] = np.nan
    h = max(1, msg.height)
    return xyz.reshape(h, -1, 3)
