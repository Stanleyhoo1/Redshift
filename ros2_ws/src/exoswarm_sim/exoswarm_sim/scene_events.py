"""Unscripted scene-change geometry shared by the Gazebo world interface and the mock world."""
from __future__ import annotations

import math

import numpy as np

# "Ground subsidence" jolt: a roll torque on the rover base link, released by the *world* (not the
# agent) once the true roll reaches a cut-off angle. Tuned in Gazebo at 2.2 g (plan T12):
# peak roll ≈ 2·cut − 9°, rover self-rights for peaks up to ~50°. Torque scales with g so the
# motion is gravity-invariant; the cut-off targets a peak `peak_over_alarm_deg` beyond the tip alarm.
JOLT = {
    'torque_nm_at_2_2g': 750.0,
    'force': [0.0, 0.0, 0.0],
    'peak_over_alarm_deg': 5.0,
    'max_duration_s': 1.0,
}


def jolt_cut_angle_deg(tip_alarm_deg: float) -> float:
    target_peak = tip_alarm_deg + JOLT['peak_over_alarm_deg']
    return (target_peak + 9.0) / 2.0


def jolt_wrench_world(yaw: float, g_mult: float = 2.2):
    """Body-frame roll torque rotated into the world frame, scaled with weight."""
    tq = JOLT['torque_nm_at_2_2g'] * g_mult / 2.2
    c, s = math.cos(yaw), math.sin(yaw)
    return [0.0, 0.0, 0.0], [c * tq, s * tq, 0.0]


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
