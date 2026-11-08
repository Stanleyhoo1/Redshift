"""Vectorized numpy ray casting for the mock world: LiDAR, depth and RGB on the same heightfield, boulders and
science crystals that Gazebo renders, published in Gazebo's exact sensor conventions.

* Heightfield: Lipschitz-safe ray marching (step = height-above-terrain / max closing rate, with a range-
  proportional minimum step) and two secant refinements on the crossing.
* Objects: boulders and crystals as yawed ellipsoids, intersected analytically after a bounding-sphere cull.
  Rockslide boulders fall from `drop_m` and are added at runtime; the falling pillar's chunks are obstacles the mock
  world moves itself every physics step (`add_objects` / `set_object_centres`).
* Robots: every robot is a few yawed ellipsoids (kind ROBOT, `robot_body`) moved every physics step, so each robot's
  camera and LiDAR see the others; a sensor excludes its own body (`exclude=` object ids). Robots are not obstacles
  to `Scene.blocked` (no robot–robot collision in the mock).
* Conventions (checked against Gazebo by scripts/check_topics.py): camera link x forward, image u → −y,
  v → −z, ray parameter = planar depth, no return = +inf; LiDAR (rings, columns) = (−15°…+10° ascending,
  azimuth −180° + c·1° ccw) in the sensor frame, no return = ±inf along the beam.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

LIDAR_SHAPE = (16, 360)
SKY_RGB = np.array([0.70, 0.44, 0.30], np.float32)            # world.sdf <background>
# Lights: world.sdf's warm ambient/sun, desaturated toward grey. ogre2's PBR keeps grey boulders grey under
# that light; plain Lambert with the raw tint would push them past vision.py's low-saturation rock gate.
AMBIENT = np.array([0.46, 0.42, 0.39], np.float32)
SUN_RGB = np.array([1.0, 0.94, 0.88], np.float32)
SUN_DIR = np.array([-0.55, 0.35, -0.75], np.float32)          # direction the light travels
REGOLITH_RGB = np.array([156, 84, 52], np.float32) / 255.0    # world_builder regolith texture base colour
CRYSTAL_ALBEDO = np.array([0.15, 0.85, 0.9], np.float32)
CRYSTAL_EMISSIVE = np.array([0.05, 0.55, 0.65], np.float32)
EXPOSURE = 0.78
HAZE_M = 160.0
MAX_ITERS = 160
# robot bodies (body-frame offsets from base_link, radii, albedo, emissive), approximating robot_models' SDF visuals
TYRE_RGB = (0.08, 0.08, 0.09)
DECK_RGB = (0.10, 0.12, 0.22)                                 # solar panel
MAST_RGB = (0.30, 0.30, 0.32)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """World-from-body rotation Rz(yaw)·Ry(pitch)·Rx(roll) (ROS / Gazebo convention)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def camera_rotation(R_body: np.ndarray, pitch: float) -> np.ndarray:
    """World-from-camera-link rotation for a camera pitched down by `pitch` about the body y axis."""
    c, s = math.cos(pitch), math.sin(pitch)
    return np.asarray(R_body) @ np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def camera_intrinsics(w: int, h: int, hfov: float) -> tuple[float, float, float]:
    """(fx = fy, cx, cy) exactly as Gazebo's CameraInfo reports them."""
    return w / (2.0 * math.tan(hfov / 2.0)), w / 2.0, h / 2.0


@lru_cache(maxsize=16)
def _camera_rays(w: int, h: int, hfov: float, scale: float) -> np.ndarray:
    """Camera-link ray directions with x = 1 (so t = planar depth), for a (h·scale, w·scale) render whose
    nearest-neighbour upsample lands each full-resolution pixel on its own ray."""
    fx, cx, cy = camera_intrinsics(w, h, hfov)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    k = 1.0 / scale
    us = (np.arange(rw) + 0.5) * k - 0.5          # full-res pixel coordinate at each render pixel centre
    vs = (np.arange(rh) + 0.5) * k - 0.5
    U, V = np.meshgrid(us, vs)
    d = np.stack([np.ones_like(U), -(U - cx) / fx, -(V - cy) / fx], axis=-1)
    return d.reshape(-1, 3)


@lru_cache(maxsize=4)
def _lidar_dirs(rings: int, samples: int, vmin: float, vmax: float) -> np.ndarray:
    el = vmin + np.arange(rings) * (vmax - vmin) / max(rings - 1, 1)
    az = -math.pi + np.arange(samples) * 2.0 * math.pi / samples
    E, A = np.meshgrid(el, az, indexing='ij')
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)


def _default_texture(n: int = 128, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pad = 12                                          # wrap-pad so the tile stays seamless after blurring
    big = np.pad(rng.random((n, n)).astype(np.float32), pad, mode='wrap')
    noise = cv2.GaussianBlur(big, (0, 0), 3.0)[pad:-pad, pad:-pad]
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-6)
    return REGOLITH_RGB[None, None, :] * (0.72 + 0.5 * noise[..., None]) / 0.97


def robot_body(name: str) -> list[tuple]:
    """Ellipsoid parts [(offset (3,), radii (3,), albedo (3,), emissive (3,))] of a robot's body, in its base_link
    frame. Rovers: chassis in the livery colour (robot_models.rover_livery: rover_0 white with an orange nose stripe,
    the recovery rovers amber with a dark one), a dark solar deck, a tyre pair per side and the LiDAR mast.
    Scout: one small flat disc."""
    from .robot_models import rover_livery
    z3 = (0.0, 0.0, 0.0)
    if not name.startswith('rover'):
        return [((0.0, 0.0, 0.0), (0.30, 0.30, 0.05), (0.80, 0.80, 0.83), (0.0, 0.12, 0.15))]
    lv = rover_livery(name)
    return [((0.0, 0.0, 0.0), (0.55, 0.36, 0.25), lv['chassis'], z3),
            ((-0.05, 0.0, 0.20), (0.42, 0.29, 0.06), DECK_RGB, z3),
            ((0.0, 0.43, -0.08), (0.50, 0.08, 0.20), TYRE_RGB, z3),
            ((0.0, -0.43, -0.08), (0.50, 0.08, 0.20), TYRE_RGB, z3),
            ((0.50, 0.0, 0.0), (0.06, 0.26, 0.05), lv['stripe'], lv['stripe_emissive']),
            ((0.25, 0.0, 0.45), (0.06, 0.06, 0.20), MAST_RGB, z3)]


class Scene:
    """Heightfield (rows = y, cols = x, origin at the sector centre) plus ellipsoid objects."""
    NONE, GROUND, ROCK, CRYSTAL, ROBOT = 0, 1, 2, 3, 4

    def __init__(self, heights: np.ndarray, size_m: float, rocks=(), crystals=(), texture: np.ndarray | None = None,
                 tex_size_m: float = 14.0, g: float = 9.81):
        self.h = np.ascontiguousarray(heights, dtype=np.float32)
        self.n = int(self.h.shape[0])
        self.size_m = float(size_m)
        self.res = self.size_m / (self.n - 1)
        self.half = self.size_m / 2.0
        self.g = float(g)
        self.t = 0.0
        dx = np.abs(np.diff(self.h, axis=1)).max(initial=0.0) / self.res
        dy = np.abs(np.diff(self.h, axis=0)).max(initial=0.0) / self.res
        self.lip = float(max(math.hypot(dx, dy), 0.05)) * 1.05
        gy, gx = np.gradient(self.h.astype(np.float64), self.res)
        nrm = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        self.normals = (nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)).astype(np.float32)
        self.texture = (_default_texture() if texture is None else np.asarray(texture, np.float32))
        self.tex_size_m = float(tex_size_m)
        # objects: centre, radii, yaw, albedo, emissive, kind, rest z, drop, spawn time
        self.c = np.zeros((0, 3))
        self.radii = np.zeros((0, 3))
        self.yaw = np.zeros(0)
        self.albedo = np.zeros((0, 3), np.float32)
        self.emissive = np.zeros((0, 3), np.float32)
        self.kind = np.zeros(0, np.int8)
        self.rest_z = np.zeros(0)
        self.drop = np.zeros(0)
        self.t_spawn = np.zeros(0)
        if len(rocks):
            r = np.asarray(rocks, np.float64).reshape(-1, 8)
            shade = r[:, 7:8]
            self._add(r[:, 0:3], r[:, 3:6], r[:, 6], np.hstack([shade, 0.98 * shade, 0.96 * shade]),
                      np.zeros((len(r), 3)), self.ROCK)
        if len(crystals):
            xy = np.asarray(crystals, np.float64).reshape(-1, 2)
            gz = self.height_at(xy[:, 0], xy[:, 1])
            self._add(np.column_stack([xy, gz + 0.2]), np.tile([0.14, 0.14, 0.22], (len(xy), 1)), np.zeros(len(xy)),
                      np.tile(CRYSTAL_ALBEDO, (len(xy), 1)), np.tile(CRYSTAL_EMISSIVE, (len(xy), 1)), self.CRYSTAL)

    @classmethod
    def from_terrain(cls, terr, texture: np.ndarray | None = None, g: float = 9.81) -> 'Scene':
        """The same boulders / crystals world_builder writes into world.sdf."""
        rocks = []
        for k, r in enumerate(terr.rocks):
            c = r.height / 2.0
            rocks.append((r.x, r.y, float(terr.height_at(r.x, r.y)) + 0.7 * c, r.radius, 0.8 * r.radius, c, r.yaw,
                          0.38 + 0.08 * ((k * 37) % 5) / 4))
        return cls(terr.heights, terr.size_m, rocks=rocks, crystals=list(terr.science), texture=texture, g=g)

    # ------------------------------------------------------------------ objects
    def _add(self, centre, radii, yaw, albedo, emissive, kind, drop=None, t_spawn=None) -> list[int]:
        # `yaw` is published last: a render running concurrently in another thread iterates over
        # range(len(self.yaw)), so every other per-object array must already be long enough.
        k0, m = len(self.yaw), len(yaw)
        rest_z = np.concatenate([self.rest_z, np.asarray(centre)[:, 2]])
        drop = np.concatenate([self.drop, np.zeros(m) if drop is None else np.full(m, drop)])
        t_spawn = np.concatenate([self.t_spawn, np.zeros(m) if t_spawn is None else np.full(m, t_spawn)])
        c = np.vstack([self.c, centre])
        dt = np.maximum(0.0, self.t - t_spawn)
        c[:, 2] = rest_z + np.maximum(0.0, drop - 0.5 * self.g * dt * dt)
        self.radii = np.vstack([self.radii, radii])
        self.albedo = np.vstack([self.albedo, np.asarray(albedo, np.float32)])
        self.emissive = np.vstack([self.emissive, np.asarray(emissive, np.float32)])
        self.kind = np.concatenate([self.kind, np.full(m, kind, np.int8)])
        self.rest_z, self.drop, self.t_spawn, self.c = rest_z, drop, t_spawn, c
        self.yaw = np.concatenate([self.yaw, yaw])
        return list(range(k0, k0 + m))

    def add_rocks(self, rocks, t: float, drop_m: float = 4.0) -> list[int]:
        """Rockslide boulders [(x, y, r)] (the Gazebo rock_model_sdf shape: radii r, 0.85r, 0.65r) resting on the
        ground, released `drop_m` above it at time `t`."""
        r = np.asarray(rocks, np.float64).reshape(-1, 3)
        if not len(r):
            return []
        gz = self.height_at(r[:, 0], r[:, 1])
        radii = np.column_stack([r[:, 2], 0.85 * r[:, 2], 0.65 * r[:, 2]])
        shade = 0.40
        return self._add(np.column_stack([r[:, :2], gz + radii[:, 2]]), radii, np.arange(len(r)) * 0.9,
                         np.tile([shade, 0.98 * shade, 0.96 * shade], (len(r), 1)), np.zeros((len(r), 3)), self.ROCK,
                         drop=drop_m, t_spawn=t)

    def add_objects(self, centres, radii, albedo, kind: int | None = None) -> list[int]:
        """Obstacles the caller moves itself (the falling pillar's chunks): kind ROCK unless given, so they collide
        (`blocked`) and show in LiDAR, depth and RGB like boulders. `radii`: (n,) spheres or (n, 3) ellipsoid radii;
        `albedo`: (3,) or (n, 3). Move them with set_object_centres."""
        c = np.asarray(centres, np.float64).reshape(-1, 3)
        m = len(c)
        if not m:
            return []
        rad = np.asarray(radii, np.float64)
        rad = np.repeat(rad.reshape(-1, 1), 3, axis=1) if rad.ndim < 2 else rad.reshape(-1, 3)
        alb = np.broadcast_to(np.asarray(albedo, np.float32).reshape(-1, 3), (m, 3))
        return self._add(c, np.broadcast_to(rad, (m, 3)), np.zeros(m), alb, np.zeros((m, 3), np.float32),
                         self.ROCK if kind is None else kind)

    def set_object_centres(self, ids, centres) -> None:
        """Move objects to new world centres (n, 3): raycast, shading and `blocked` see them there from now on, and
        nothing of them stays behind. Physics thread only, like set_body_pose: renderers work on scene_objects
        snapshots taken on that thread, so a frame sees every chunk at one instant."""
        ids = np.asarray(ids, np.int64)
        c = np.asarray(centres, np.float64).reshape(-1, 3)
        self.rest_z[ids] = c[:, 2]                     # update() rewrites c[:, 2] from rest_z
        self.c[ids] = c

    def add_body(self, parts) -> list[int]:
        """Add a robot body (robot_body parts) as ROBOT objects at the origin; place it with set_body_pose."""
        off = np.array([p[0] for p in parts], np.float64).reshape(-1, 3)
        return self._add(off, np.array([p[1] for p in parts], np.float64), np.zeros(len(off)),
                         np.array([p[2] for p in parts], np.float32), np.array([p[3] for p in parts], np.float32),
                         self.ROBOT)

    def set_body_pose(self, ids, parts, pos, R: np.ndarray, yaw: float) -> None:
        """Move a robot body (`parts` from robot_body, or their (n, 3) offsets): part centres follow the full
        attitude R, the ellipsoids themselves only the yaw. Call it from the thread that calls _add (the physics
        thread): renderers read concurrently and may see a part one step stale, which is harmless for a frame."""
        off = parts if isinstance(parts, np.ndarray) else np.array([p[0] for p in parts], np.float64).reshape(-1, 3)
        c = np.asarray(pos, np.float64) + off @ np.asarray(R, np.float64).T
        ids = np.asarray(ids, np.int64)
        self.rest_z[ids] = c[:, 2]                     # update() rewrites c[:, 2] from rest_z
        self.c[ids] = c
        self.yaw[ids] = yaw

    def update(self, t: float) -> None:
        """Advance falling boulders to time t."""
        self.t = float(t)
        if len(self.rest_z) and (self.drop > 0).any():
            dt = np.maximum(0.0, self.t - self.t_spawn)
            self.c[:, 2] = self.rest_z + np.maximum(0.0, self.drop - 0.5 * self.g * dt * dt)

    def blocked(self, x: float, y: float, radius: float) -> bool:
        """Does a robot disc at (x, y) overlap any landed object's footprint? Robot bodies never block (the mock
        has no robot–robot collision)."""
        if not len(self.yaw):
            return False
        dx, dy = x - self.c[:, 0], y - self.c[:, 1]
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        lx, ly = cy * dx + sy * dy, -sy * dx + cy * dy
        landed = self.c[:, 2] - self.radii[:, 2] < self.height_at(self.c[:, 0], self.c[:, 1]) + 0.3
        inside = (lx / (self.radii[:, 0] + radius)) ** 2 + (ly / (self.radii[:, 1] + radius)) ** 2 < 1.0
        return bool((inside & landed & (self.kind != self.ROBOT)).any())

    # ------------------------------------------------------------------ terrain
    def height_at(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        fx = np.clip((x + self.half) / self.res, 0.0, self.n - 1.000001)
        fy = np.clip((y + self.half) / self.res, 0.0, self.n - 1.000001)
        j, i = fx.astype(np.int64), fy.astype(np.int64)
        ax, ay = fx - j, fy - i
        hf = self.h.ravel()
        flat = i * self.n + j
        h00, h01, h10, h11 = hf[flat], hf[flat + 1], hf[flat + self.n], hf[flat + self.n + 1]
        top = h00 + (h01 - h00) * ax
        return top + (h10 + (h11 - h10) * ax - top) * ay

    def _inside(self, x, y):
        return (np.abs(x) <= self.half) & (np.abs(y) <= self.half)

    def _march(self, o: np.ndarray, d: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
        """First heightfield crossing along each ray (inf if none). Rays starting below the surface see
        nothing (back faces), like Gazebo's renderer."""
        n = len(d)
        t_hit = np.full(n, np.inf)
        closing = self.lip * np.hypot(d[:, 0], d[:, 1]) - d[:, 2]
        dlen = np.linalg.norm(d, axis=1)
        t = np.full(n, float(t_min))
        p = o + d * t[:, None]
        dz = p[:, 2] - self.height_at(p[:, 0], p[:, 1])
        idx = np.flatnonzero((dz > 0) & self._inside(p[:, 0], p[:, 1]) & (closing > 0))
        tp, dzp = t[idx].copy(), dz[idx].copy()
        for _ in range(MAX_ITERS):
            if not len(idx):
                break
            cl = closing[idx]
            step = np.maximum(dzp / cl, (0.03 + 0.012 * tp * dlen[idx]) / dlen[idx])
            ti = tp + step
            oi, di = o[idx], d[idx]
            p = oi + di * ti[:, None]
            dz = p[:, 2] - self.height_at(p[:, 0], p[:, 1])
            hit = (dz <= 0) & (ti <= t_max)
            if hit.any():
                a, b = tp[hit], ti[hit]
                fa, fb = dzp[hit], dz[hit]
                oh, dh = oi[hit], di[hit]
                for _r in range(2):                       # secant (regula falsi) refinement
                    tm = a + (b - a) * fa / np.maximum(fa - fb, 1e-9)
                    q = oh + dh * tm[:, None]
                    fm = q[:, 2] - self.height_at(q[:, 0], q[:, 1])
                    above = fm > 0
                    a, fa = np.where(above, tm, a), np.where(above, fm, fa)
                    b, fb = np.where(above, b, tm), np.where(above, fb, fm)
                t_hit[idx[hit]] = a + (b - a) * fa / np.maximum(fa - fb, 1e-9)
            keep = ~hit & (ti <= t_max) & self._inside(p[:, 0], p[:, 1])
            idx, tp, dzp = idx[keep], ti[keep], dz[keep]
        return t_hit

    # ------------------------------------------------------------------ ray casting
    def raycast(self, o, d, t_min: float, t_max: float, exclude=None):
        """Rays o + t·d (o: (3,) or (N, 3)). Returns t (inf = miss), kind, unit normal (N, 3), object index.
        `exclude`: object ids the rays pass through (a sensor's own robot body)."""
        d = np.asarray(d, np.float64).reshape(-1, 3)
        o_arr = np.asarray(o, np.float64)
        single = o_arr.ndim == 1
        o = np.broadcast_to(o_arr, d.shape) if single else o_arr.reshape(-1, 3)
        t = self._march(o, d, t_min, t_max)
        kind = np.where(np.isfinite(t), self.GROUND, self.NONE).astype(np.int8)
        obj = np.full(len(d), -1, np.int64)
        normal = np.zeros((len(d), 3), np.float32)
        dn = d / np.linalg.norm(d, axis=1, keepdims=True)
        dlen_max = float(np.linalg.norm(d, axis=1).max()) if len(d) else 1.0
        skip = frozenset(exclude) if exclude else ()
        # view-cone cull (single origin): every ray lies within `cone` of `axis`, so an object whose bounding sphere
        # is further off-axis than cone + its angular radius cannot be hit. Off for near-omnidirectional fans (LiDAR).
        cone = None
        if single and len(d):
            axis = dn.sum(axis=0)
            na = float(np.linalg.norm(axis))
            if na > 1e-9:
                axis /= na
                cone = math.acos(max(-1.0, min(1.0, float((dn @ axis).min()))))
                if cone > math.radians(80.0):
                    cone = None
        for k in range(len(self.yaw)):
            if k in skip:
                continue
            rho = float(self.radii[k].max())
            w = self.c[k] - (o_arr if single else o)
            if single:
                dist = float(np.linalg.norm(w))
                if dist > t_max * dlen_max + rho:
                    continue
                if cone is not None and dist > rho:
                    off = math.acos(max(-1.0, min(1.0, float(w @ axis) / dist)))
                    if off > cone + math.asin(rho / dist):
                        continue
                along = dn @ w
                cand = np.flatnonzero((dist * dist - along * along < rho * rho) & (along > -rho))
            else:
                along = np.einsum('ij,ij->i', dn, w)
                cand = np.flatnonzero((np.einsum('ij,ij->i', w, w) - along * along < rho * rho) & (along > -rho))
            if not len(cand):
                continue
            cy, sy = math.cos(self.yaw[k]), math.sin(self.yaw[k])
            Rt = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
            pl = ((o[cand] - self.c[k]) @ Rt.T) / self.radii[k]
            ql = (d[cand] @ Rt.T) / self.radii[k]
            A = np.einsum('ij,ij->i', ql, ql)
            B = 2.0 * np.einsum('ij,ij->i', pl, ql)
            C = np.einsum('ij,ij->i', pl, pl) - 1.0
            disc = B * B - 4 * A * C
            ok = (disc >= 0) & (C > 0)
            th = np.where(ok, (-B - np.sqrt(np.maximum(disc, 0.0))) / (2 * A), np.inf)
            better = ok & (th >= t_min) & (th <= t_max) & (th < t[cand])
            if not better.any():
                continue
            ci = cand[better]
            t[ci] = th[better]
            kind[ci] = self.kind[k]
            obj[ci] = k
            s = pl[better] + ql[better] * th[better][:, None]
            nl = s / self.radii[k]
            nw = nl @ Rt                                   # back to world (Rt is orthonormal)
            normal[ci] = (nw / np.linalg.norm(nw, axis=1, keepdims=True)).astype(np.float32)
        g = kind == self.GROUND
        if g.any():
            p = o[g] + d[g] * t[g][:, None]
            fx = np.clip(np.rint((p[:, 0] + self.half) / self.res), 0, self.n - 1).astype(np.int64)
            fy = np.clip(np.rint((p[:, 1] + self.half) / self.res), 0, self.n - 1).astype(np.int64)
            normal[g] = self.normals[fy, fx]
        return t, kind, normal, obj

    # ------------------------------------------------------------------ shading
    def shade(self, o, d, t, kind, normal, obj) -> np.ndarray:
        """Lambert + ambient + emissive with distance haze; sky gradient for misses. RGB float in [0, 1]."""
        n = len(t)
        rgb = np.empty((n, 3), np.float32)
        dn = d / np.linalg.norm(d, axis=1, keepdims=True)
        sky = kind == self.NONE
        if sky.any():
            up = np.clip(dn[sky, 2], -0.2, 1.0)[:, None]
            rgb[sky] = SKY_RGB * (1.0 + 0.25 * np.exp(-6.0 * np.abs(up))) - 0.12 * np.maximum(up, 0.0)
        hit = ~sky
        if hit.any():
            L = -SUN_DIR / np.linalg.norm(SUN_DIR)
            lam = np.clip(normal[hit] @ L, 0.0, 1.0)[:, None]
            alb = np.empty((int(hit.sum()), 3), np.float32)
            emi = np.zeros_like(alb)
            kh = kind[hit]
            g = kh == self.GROUND
            if g.any():
                p = (o[hit][g] if o.ndim == 2 else o) + d[hit][g] * t[hit][g][:, None]
                T = self.texture.shape[0]
                tu = ((p[:, 0] / self.tex_size_m) % 1.0 * T).astype(np.int64) % T
                tv = ((p[:, 1] / self.tex_size_m) % 1.0 * T).astype(np.int64) % T
                alb[g] = self.texture[tv, tu]
            ob = ~g
            if ob.any():
                ids = obj[hit][ob]
                alb[ob] = self.albedo[ids]
                emi[ob] = self.emissive[ids]
            col = alb * (AMBIENT + SUN_RGB * lam) * EXPOSURE + emi
            dist = t[hit] * np.linalg.norm(d[hit], axis=1)
            fog = (1.0 - np.exp(-dist / HAZE_M))[:, None].astype(np.float32)
            rgb[hit] = col * (1.0 - fog) + SKY_RGB * fog
        return np.clip(rgb, 0.0, 1.0)


def camera_frame(scene: Scene, origin, R_cam: np.ndarray, w: int, h: int, hfov: float, far: float,
                 scale: float = 0.5, near: float = 0.1, exclude=None) -> tuple[np.ndarray, np.ndarray]:
    """RGB (h, w, 3) uint8 and planar depth (h, w) float32 (+inf = no return) from a camera link at `origin`
    with world-from-camera rotation `R_cam`, rendered at `scale` and upsampled to (w, h). `exclude`: object ids
    invisible to this camera (its own robot body)."""
    rays = _camera_rays(w, h, float(hfov), float(scale))
    d = rays @ np.asarray(R_cam, np.float64).T
    o = np.asarray(origin, np.float64)
    t, kind, normal, obj = scene.raycast(o, d, near, far, exclude=exclude)
    rgb = scene.shade(o, d, t, kind, normal, obj)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    depth = np.where(np.isfinite(t) & (t <= far), t, np.inf).astype(np.float32).reshape(rh, rw)
    img = (rgb.reshape(rh, rw, 3) * 255.0 + 0.5).astype(np.uint8)
    if (rw, rh) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return img, depth


def lidar_scan(scene: Scene, origin, R_body: np.ndarray, spec: dict, exclude=None) -> np.ndarray:
    """Organized (rings, samples, 3) float32 points in the sensor frame; no return = inf along the beam.
    `exclude`: object ids the beams pass through (its own robot body)."""
    rings, samples = int(spec.get('lidar_rings', 16)), int(spec.get('lidar_samples', 360))
    vmin, vmax = spec.get('lidar_vfov', [-0.2618, 0.1745])
    rmin, rmax = spec.get('lidar_range', [0.3, 25.0])
    ds = _lidar_dirs(rings, samples, float(vmin), float(vmax))
    t, _, _, _ = scene.raycast(np.asarray(origin, np.float64), ds @ np.asarray(R_body, np.float64).T, rmin, rmax,
                               exclude=exclude)
    # no return: Gazebo writes range = inf along the beam; its beam components are never exactly 0, so the
    # result is ±inf per axis (never inf·0 = NaN)
    away = np.where(ds == 0.0, 1e-9, ds) * np.inf
    pts = np.where(np.isfinite(t)[:, None], ds * np.where(np.isfinite(t), t, 0.0)[:, None], away)
    return pts.astype(np.float32).reshape(rings, samples, 3)
