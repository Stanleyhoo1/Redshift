"""Unscripted scene-change geometry shared by the Gazebo world interface and the mock world."""
from __future__ import annotations

import math

import numpy as np

# Lateral "ground subsidence" jolt applied to the rover base link for `duration_s`
# (body frame: +y = left). Tuned in Gazebo so peak roll lands in [32°, 45°] at 2.2 g and the rover
# self-rights (see plan T12). The mock world uses `mock_roll_impulse_deg_s` instead.
JOLT = {
    'force': [0.0, 1400.0, 0.0],
    'torque': [700.0, 0.0, 0.0],
    'duration_s': 0.35,
    'mock_roll_impulse_deg_s': 260.0,
}


def jolt_wrench_world(yaw: float, g_mult: float = 1.0):
    """Rotate the body-frame jolt into the world frame; scales with weight so tilt is comparable."""
    c, s = math.cos(yaw), math.sin(yaw)
    k = max(1.0, g_mult) / 2.2
    f, t = JOLT['force'], JOLT['torque']
    force = [k * (c * f[0] - s * f[1]), k * (s * f[0] + c * f[1]), k * f[2]]
    torque = [k * (c * t[0] - s * t[1]), k * (s * t[0] + c * t[1]), k * t[2]]
    return force, torque


def _point_along(path: np.ndarray, pose, ahead_m: float):
    """Project the robot onto the polyline, walk `ahead_m` forward; returns (point, unit tangent)."""
    x, y, yaw = pose
    if path is None or len(path) < 2:
        t = np.array([math.cos(yaw), math.sin(yaw)])
        return np.array([x, y]) + ahead_m * t, t
    p = np.asarray(path, dtype=np.float64)
    seg = p[1:] - p[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    best, best_d, best_t = 0, math.inf, 0.0
    for k in range(len(seg)):
        if seg_len[k] < 1e-9:
            continue
        tt = float(np.clip(np.dot([x, y] - p[k], seg[k]) / seg_len[k] ** 2, 0.0, 1.0))
        q = p[k] + tt * seg[k]
        d = math.hypot(q[0] - x, q[1] - y)
        if d < best_d:
            best, best_d, best_t = k, d, tt
    remaining = ahead_m
    k, tt = best, best_t
    while True:
        if seg_len[k] < 1e-9:
            k += 1
            if k >= len(seg):
                break
            continue
        left = (1.0 - tt) * seg_len[k]
        tangent = seg[k] / seg_len[k]
        if remaining <= left or k == len(seg) - 1:
            point = p[k] + tt * seg[k] + remaining * tangent
            return point, tangent
        remaining -= left
        k, tt = k + 1, 0.0
    tangent = seg[-1] / max(seg_len[-1], 1e-9)
    return p[-1] + remaining * tangent, tangent


def rockslide_layout(path_xy, pose, ahead_m: float = 7.0, width_m: float = 7.0, n: int = 6, seed=None):
    """A line of boulders across the robot's planned path `ahead_m` metres ahead.

    Returns (rocks [(x, y, radius)], impact (x, y)).
    """
    rng = np.random.default_rng(seed)
    path = None if path_xy is None else np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    point, tangent = _point_along(path, pose, ahead_m)
    normal = np.array([-tangent[1], tangent[0]])
    rocks = []
    for off in np.linspace(-width_m / 2, width_m / 2, n):
        o = off + rng.uniform(-0.2, 0.2)
        a = rng.uniform(-0.4, 0.4)
        c = point + o * normal + a * tangent
        rocks.append((round(float(c[0]), 3), round(float(c[1]), 3), round(float(rng.uniform(0.5, 1.0)), 3)))
    return rocks, (round(float(point[0]), 3), round(float(point[1]), 3))
