"""Unscripted scene-change geometry shared by the Gazebo world interface and the mock world."""
from __future__ import annotations

import math
from functools import lru_cache

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


def _project(p: np.ndarray, x: float, y: float):
    """Closest point of the polyline to (x, y): (segment index, fraction along it, segment vectors, lengths)."""
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
    return best, best_t, seg, seg_len


def remaining_m(path, pose) -> float:
    """Route length still ahead of the robot (0 without a usable path)."""
    if path is None or len(path) < 2:
        return 0.0
    k, tt, _, seg_len = _project(np.asarray(path, dtype=np.float64), pose[0], pose[1])
    return float((1.0 - tt) * seg_len[k] + seg_len[k + 1:].sum())


def _point_along(path: np.ndarray, pose, ahead_m: float):
    """Project the robot onto the polyline, walk `ahead_m` forward; returns (point, unit tangent)."""
    x, y, yaw = pose
    if path is None or len(path) < 2:
        t = np.array([math.cos(yaw), math.sin(yaw)])
        return np.array([x, y]) + ahead_m * t, t
    p = np.asarray(path, dtype=np.float64)
    best, best_t, seg, seg_len = _project(p, x, y)
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


ROUTE_FRACTION = 0.6     # on a short route the line falls at most this far along what remains (room before the goal)
MIN_AHEAD_M = 3.0         # never closer than this to the robot
DEFER_BELOW_M = 4.0       # less route left: the rover is about to reach its goal, so the slide waits for its next route
DEFER_MAX_S = 8.0         # ... for at most this long (simulated seconds), then falls along the heading


def rockslide_due(path_xy, pose, waited_s: float) -> bool:
    """Drop an armed rockslide now? Not while under DEFER_BELOW_M of route remains: the rover is about to reach its
    goal and choose a new one, and a slide placed now would fall past the goal, off every route it will drive. It
    waits (DEFER_MAX_S at most) for the next planned route and then lands across that."""
    path = None if path_xy is None else np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    return remaining_m(path, pose) >= DEFER_BELOW_M or waited_s >= DEFER_MAX_S


def rockslide_layout(path_xy, pose, ahead_m: float = 7.0, width_m: float = 7.0, n: int = 6, seed=None):
    """A line of boulders across the robot's planned path `ahead_m` metres ahead.

    If less route than that remains (the robot is about to reach its goal), the line moves back onto the route,
    ROUTE_FRACTION of the way along what is left but at least MIN_AHEAD_M ahead, so it still cuts the path rather
    than landing beyond its end. With no path, or under MIN_AHEAD_M of it left, it falls `ahead_m` along the heading /
    last segment. Returns (rocks [(x, y, radius)], impact (x, y)).
    """
    rng = np.random.default_rng(seed)
    path = None if path_xy is None else np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    left = remaining_m(path, pose)
    if left > MIN_AHEAD_M:
        ahead_m = min(ahead_m, max(MIN_AHEAD_M, ROUTE_FRACTION * left))
    point, tangent = _point_along(path, pose, ahead_m)
    normal = np.array([-tangent[1], tangent[0]])
    rocks = []
    for off in np.linspace(-width_m / 2, width_m / 2, n):
        o = off + rng.uniform(-0.2, 0.2)
        a = rng.uniform(-0.4, 0.4)
        c = point + o * normal + a * tangent
        rocks.append((round(float(c[0]), 3), round(float(c[1]), 3), round(float(rng.uniform(0.5, 1.0)), 3)))
    return rocks, (round(float(point[0]), 3), round(float(point[1]), 3))


# "Falling pillar": a tall basalt column beside the route topples across it (spec §1.1). Gazebo spawns one dynamic
# cylinder and lets gravity do it; the mock world moves n overlapping near-spheres along the analytic fall.
PILLAR = {'height_m': 6.0, 'radius_m': 0.35, 'side_m': 2.8, 'ahead_m': 8.0, 'mass_kg': 1500.0,
          'tilt0_deg': 4.0, 'albedo_rgb': (20, 19, 18)}          # fresh basalt, albedo ≈ 0.08
PILLAR_CHUNK_SCALE = 1.15     # mock chunk radius / pillar radius: the spheres' waists still meet the cylinder wall
PILLAR_REST_RAD = math.pi / 2  # lying on its side (the chunks are lifted by r·sin θ, so the trunk rests flat)
FALL_DT_S = 1e-3               # fixed RK4 step of the toppling rod (deterministic, ≈ 1700 steps at 2.2 g)
FALL_MAX_S = 60.0              # integration cap (a pillar with no tilt or no gravity never lands)
MIN_TILT_RAD = 1e-3            # a perfectly upright rod is an equilibrium: it always starts a little tilted


def pillar_layout(path_xy, pose, ahead_m: float = PILLAR['ahead_m'], side_m: float = PILLAR['side_m'],
                  seed=None) -> dict:
    """Base of a pillar standing `side_m` to the left or right (seeded) of the route, at the point the rockslide logic
    picks (ROUTE_FRACTION / MIN_AHEAD_M via _point_along). It falls toward and across the route: fall_dir is the unit
    vector from the base through the route point, so the trunk spans side_m before the route and the rest past it.
    Returns {'base': (x, y), 'fall_dir': (ux, uy), 'height_m', 'radius_m', 'cross': (x, y), 'side': +1 | -1}."""
    rng = np.random.default_rng(seed)
    path = None if path_xy is None else np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    left = remaining_m(path, pose)
    if left > MIN_AHEAD_M:
        ahead_m = min(ahead_m, max(MIN_AHEAD_M, ROUTE_FRACTION * left))
    point, tangent = _point_along(path, pose, ahead_m)
    normal = np.array([-tangent[1], tangent[0]])                 # left of the direction of travel
    side = 1 if rng.random() < 0.5 else -1
    base = point + side * side_m * normal
    fall = -side * normal
    return {'base': (round(float(base[0]), 3), round(float(base[1]), 3)),
            'fall_dir': (round(float(fall[0]), 6), round(float(fall[1]), 6)),
            'height_m': float(PILLAR['height_m']), 'radius_m': float(PILLAR['radius_m']),
            'cross': (round(float(point[0]), 3), round(float(point[1]), 3)), 'side': side}


@lru_cache(maxsize=32)
def _fall_table(height_m: float, g: float, tilt0_deg: float) -> np.ndarray:
    """θ at every FALL_DT_S from release until it reaches PILLAR_REST_RAD (RK4 on θ'' = (3 g / 2L) sin θ)."""
    th = max(math.radians(tilt0_deg), MIN_TILT_RAD)
    if height_m <= 0.0 or g <= 0.0:
        return np.array([th])
    k = 3.0 * g / (2.0 * height_m)
    w, h = 0.0, FALL_DT_S
    out = [th]
    for _ in range(int(FALL_MAX_S / h)):
        a1, b1 = w, k * math.sin(th)
        a2, b2 = w + 0.5 * h * b1, k * math.sin(th + 0.5 * h * a1)
        a3, b3 = w + 0.5 * h * b2, k * math.sin(th + 0.5 * h * a2)
        a4, b4 = w + h * b3, k * math.sin(th + h * a3)
        th += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
        w += h * (b1 + 2 * b2 + 2 * b3 + b4) / 6.0
        out.append(min(th, PILLAR_REST_RAD))
        if th >= PILLAR_REST_RAD:
            break
    return np.array(out)


def pillar_fall_angle(t_s: float, height_m: float = 6.0, g: float = 9.81 * 2.2, tilt0_deg: float = 4.0) -> float:
    """Tilt from vertical (rad) t_s seconds after release: a rigid rod pivoting on its base, θ'' = (3 g / 2L) sin θ
    from rest at θ0 = tilt0, integrated with a fixed small step and clamped at PILLAR_REST_RAD. Monotonic and
    deterministic; at 2.2 g a 6 m pillar lands after ≈ 1.66 s (pillar_fall_time)."""
    tab = _fall_table(float(height_m), float(g), float(tilt0_deg))
    if t_s <= 0.0:
        return float(tab[0])
    k = t_s / FALL_DT_S
    i = int(k)
    if i >= len(tab) - 1:
        return float(tab[-1])
    return float(tab[i] + (tab[i + 1] - tab[i]) * (k - i))


def pillar_fall_time(height_m: float = 6.0, g: float = 9.81 * 2.2, tilt0_deg: float = 4.0) -> float:
    """Seconds from release until pillar_fall_angle reaches PILLAR_REST_RAD (inf if it never does)."""
    tab = _fall_table(float(height_m), float(g), float(tilt0_deg))
    return (len(tab) - 1) * FALL_DT_S if tab[-1] >= PILLAR_REST_RAD else math.inf


def pillar_segments(layout: dict, theta: float, n: int = 9) -> list[tuple[float, float, float, float]]:
    """(x, y, z_above_base, r) for n near-spherical chunks along the pillar axis tilted by theta toward fall_dir,
    r = radius_m·PILLAR_CHUNK_SCALE. Centres sit evenly from r to height − r along the axis, so the chunks cover the
    footprint from base to tip and neighbours overlap (n ≥ 8 for the default pillar). The axis rotates about the base
    and is lifted by r·sin θ, as a column pivoting on its edge: upright the bottom chunk stands on the ground, lying
    down (θ = π/2) every chunk does."""
    L = float(layout['height_m'])
    r = float(layout['radius_m']) * PILLAR_CHUNK_SCALE
    bx, by = layout['base']
    ux, uy = layout['fall_dir']
    s = np.linspace(r, L - r, n) if n > 1 else np.array([L / 2.0])
    st, ct = math.sin(theta), math.cos(theta)
    reach = s * st
    z = s * ct + r * st
    return [(float(bx + h * ux), float(by + h * uy), float(zz), r) for h, zz in zip(reach, z)]


def pillar_footprint(layout: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """(base_xy, tip_xy) of the fallen pillar on the ground: tip = base + height·fall_dir."""
    bx, by = layout['base']
    ux, uy = layout['fall_dir']
    L = float(layout['height_m'])
    return (float(bx), float(by)), (round(float(bx + L * ux), 3), round(float(by + L * uy), 3))


def pillar_drape(segments, theta: float, base_z: float, ground_z) -> np.ndarray:
    """World chunk centres (n, 3) on uneven ground: the base's ground height + z_above_base, blended by sin θ toward
    each chunk's own ground height, so the landed trunk lies on the terrain under it instead of floating over a dip
    or sinking into a rise (`ground_z`: ground height at each chunk's x, y)."""
    seg = np.asarray(segments, np.float64).reshape(-1, 4)
    w = math.sin(theta)
    z = base_z + seg[:, 2] + w * (np.asarray(ground_z, np.float64) - base_z)
    return np.column_stack([seg[:, 0], seg[:, 1], z])


class PillarFall:
    """A released pillar's chunks over time on the real terrain (the mock world's stand-in for the rigid body):
    angle from pillar_fall_angle, chunks from pillar_segments, heights from pillar_drape. `ground_at(x, y)` is the
    terrain height (vectorised, e.g. Scene.height_at)."""

    def __init__(self, layout: dict, t_release: float, ground_at, g: float = 9.81 * 2.2, n: int = 9,
                 tilt0_deg: float = PILLAR['tilt0_deg']):
        self.layout, self.t0, self.ground_at, self.n = layout, float(t_release), ground_at, int(n)
        self.g, self.tilt0_deg = float(g), float(tilt0_deg)
        self.base_z = float(ground_at(*layout['base']))
        self.t_land = pillar_fall_time(layout['height_m'], self.g, self.tilt0_deg)
        self.landed = False

    def centres(self, t: float) -> tuple[np.ndarray, float]:
        """(chunk centres (n, 3), chunk radius) at time t; sets `landed` once the trunk lies on the ground."""
        theta = pillar_fall_angle(t - self.t0, self.layout['height_m'], self.g, self.tilt0_deg)
        segs = pillar_segments(self.layout, theta, self.n)
        xy = np.asarray([s[:2] for s in segs], np.float64)
        c = pillar_drape(segs, theta, self.base_z, self.ground_at(xy[:, 0], xy[:, 1]))
        self.landed = theta >= PILLAR_REST_RAD
        return c, segs[0][3]
