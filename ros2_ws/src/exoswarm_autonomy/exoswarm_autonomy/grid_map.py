"""Shared 2.5-D occupancy + elevation grid fused from every robot's range sensors.

Per cell we keep min/max point height, hit counts and "high" hits (points > STEP_M above the cell's
ground estimate). Organized LiDAR scans are additionally triangulated between vertically adjacent
rings so the surface between ring hits becomes known (as a real mesh-based mapper would).

Classes (docs/protocol.md §5): 0 unknown, 1 free, 2 caution, 3 hazard (slope), 4 permanent hazard,
5 science target, 6 obstacle (step / rock).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

UNKNOWN, FREE, CAUTION, HAZARD, PERMANENT, SCIENCE, OBSTACLE = range(7)
STEP_M = 0.30
RANGE_MARGIN = 0.02     # m of step margin per m of range beyond 5 m
HIGH_HITS = 2
REVEAL_S = 6.0


def slope_from_elevation(elev: np.ndarray, res: float) -> np.ndarray:
    """Slope (deg) of an elevation grid with NaN = unknown (normalized-convolution gap fill, 3x3 Sobel)."""
    valid = np.isfinite(elev)
    gz = np.where(valid, elev, 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    num = cv2.blur(gz, (3, 3))
    den = cv2.blur(w, (3, 3))
    filled = np.where(valid, elev, np.where(den > 0, num / np.maximum(den, 1e-6), 0.0)).astype(np.float32)
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * res)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * res)
    return np.where(valid, np.degrees(np.arctan(np.hypot(gx, gy))), 0.0).astype(np.float32)


class GlobalMap:
    def __init__(self, size_m: float, res: float):
        self.size_m = float(size_m)
        self.res = float(res)
        self.n = int(round(size_m / res))
        self.origin = (-size_m / 2.0, -size_m / 2.0)
        shape = (self.n, self.n)
        self.zmin = np.full(shape, np.inf, np.float32)
        self.zmax = np.full(shape, -np.inf, np.float32)
        self.zfill = np.full(shape, np.nan, np.float32)
        self.hits = np.zeros(shape, np.int32)
        self.high = np.zeros(shape, np.int32)
        self.known = np.zeros(shape, bool)
        self.first_seen = np.full(shape, np.nan, np.float64)
        self.permanent = np.zeros(shape, bool)
        self.science = np.zeros(shape, bool)
        self.slope = np.zeros(shape, np.float32)
        self.cls = np.zeros(shape, np.uint8)
        self.lethal = np.zeros(shape, bool)
        self.caution = np.zeros(shape, bool)
        self.science_xy: list = []
        self.permanent_marks: list = []
        self._elev_lo = math.inf
        self._elev_hi = -math.inf

    # ------------------------------------------------------------------ geometry
    def cell_of(self, x: float, y: float):
        j = int(math.floor((x - self.origin[0]) / self.res))
        i = int(math.floor((y - self.origin[1]) / self.res))
        return min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1)

    def _flat_idx(self, xy: np.ndarray):
        j = np.floor((xy[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        i = np.floor((xy[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        return i * self.n + j, ok

    # ------------------------------------------------------------------ fusion
    def integrate(self, pts: np.ndarray, t: float, sensor_xyz=None, min_range: float = 0.0,
                  max_range: float = math.inf) -> int:
        """Fuse world-frame points (N, 3). Returns number of points used."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        rng = None
        if sensor_xyz is not None and len(pts):
            rng = np.linalg.norm(pts - np.asarray(sensor_xyz, dtype=np.float64), axis=1)
            keep = (rng >= min_range) & (rng <= max_range)
            pts, rng = pts[keep], rng[keep]
        if not len(pts):
            return 0
        flat, ok = self._flat_idx(pts)
        if not ok.any():
            return 0
        flat, z = flat[ok], pts[ok, 2].astype(np.float32)
        zmin, zmax = self.zmin.ravel(), self.zmax.ravel()
        # obstacle evidence: a vertical step inside this scan, or a step above the cell's previously mapped
        # ground with a margin growing with range (1 deg of pose/pitch error at 20 m is 0.35 m of height)
        scan_min = np.full(self.n * self.n, np.inf, np.float32)
        np.minimum.at(scan_min, flat, z)
        margin = STEP_M if rng is None else STEP_M + RANGE_MARGIN * np.maximum(0.0, rng[ok] - 5.0)
        high = (z > scan_min[flat] + STEP_M) | (z > zmin[flat] + margin)
        np.minimum.at(zmin, flat, z)
        np.maximum.at(zmax, flat, z)
        np.add.at(self.hits.ravel(), flat, 1)
        if high.any():
            np.add.at(self.high.ravel(), flat[high], 1)
        self._mark_known(flat, t)
        self._elev_lo = min(self._elev_lo, float(z.min()))
        self._elev_hi = max(self._elev_hi, float(np.percentile(z, 99)))
        return int(len(flat))

    def integrate_organized(self, grid: np.ndarray, sensor_xyz, t: float, min_range: float = 0.0,
                            max_gap_m: float = 6.0, max_grade_deg: float = 40.0) -> int:
        """Fuse an organized scan (V rings, H columns, 3) with NaN for no-return, then fill the surface
        between vertically adjacent ground hits of the same column."""
        grid = np.asarray(grid, dtype=np.float64)
        used = self.integrate(grid.reshape(-1, 3), t, sensor_xyz=sensor_xyz, min_range=min_range)
        if grid.shape[0] < 2:
            return used
        a, b = grid[:-1].reshape(-1, 3), grid[1:].reshape(-1, 3)
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[ok], b[ok]
        if not len(a):
            return used
        dxy = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
        good = (dxy < max_gap_m) & (np.abs(b[:, 2] - a[:, 2]) < dxy * math.tan(math.radians(max_grade_deg)) + 0.05)
        a, b, dxy = a[good], b[good], dxy[good]
        if not len(a):
            return used
        steps = np.maximum(2, np.ceil(dxy / (0.5 * self.res)).astype(int))
        reps = np.repeat(np.arange(len(a)), steps)
        starts = np.cumsum(steps) - steps
        frac = (np.arange(int(steps.sum())) - np.repeat(starts, steps)) / np.repeat(steps, steps)
        fill = a[reps] + (b[reps] - a[reps]) * frac[:, None]
        flat, inside = self._flat_idx(fill)
        flat, z = flat[inside], fill[inside, 2].astype(np.float32)
        zf = self.zfill.ravel()
        # keep the lowest interpolated surface per cell (conservative ground estimate)
        tmp = np.full(self.n * self.n, np.inf, np.float32)
        np.minimum.at(tmp, flat, z)
        upd = np.unique(flat)
        zf[upd] = np.minimum(np.where(np.isnan(zf[upd]), np.inf, zf[upd]), tmp[upd])
        self._mark_known(flat, t)
        return used + int(len(flat))

    def _mark_known(self, flat: np.ndarray, t: float) -> None:
        kn = self.known.ravel()
        new = flat[~kn[flat]]
        if len(new):
            self.first_seen.ravel()[new] = t
            kn[new] = True

    # ------------------------------------------------------------------ annotations
    def mark_permanent(self, xy, radius_m: float) -> None:
        x, y = xy
        ii, jj = np.mgrid[0:self.n, 0:self.n]
        cx = self.origin[0] + (jj + 0.5) * self.res
        cy = self.origin[1] + (ii + 0.5) * self.res
        self.permanent |= np.hypot(cx - x, cy - y) <= radius_m
        self.permanent_marks.append([round(float(x), 2), round(float(y), 2), round(float(radius_m), 2)])

    def mark_science(self, xy, dedup_m: float = 1.5) -> bool:
        x, y = xy
        if any(math.hypot(x - a, y - b) < dedup_m for a, b in self.science_xy):
            return False
        self.science_xy.append((round(float(x), 2), round(float(y), 2)))
        self.science[self.cell_of(x, y)] = True
        return True

    # ------------------------------------------------------------------ derived layers
    def ground(self) -> np.ndarray:
        g = np.where(self.hits > 0, self.zmin, self.zfill)
        return np.where(self.known, g, np.nan).astype(np.float32)

    def recompute(self, limits) -> None:
        g = self.ground()
        valid = np.isfinite(g)
        self.slope = slope_from_elevation(g, self.res)
        theta = limits.max_slope_deg
        obstacle = valid & (self.high >= HIGH_HITS)
        slope_bad = valid & (self.slope > theta)
        caution = valid & (self.slope > 0.7 * theta) & ~slope_bad
        cls = np.zeros_like(self.cls)
        cls[self.known] = FREE
        cls[caution] = CAUTION
        cls[slope_bad] = HAZARD
        cls[self.science & self.known] = SCIENCE
        cls[obstacle] = OBSTACLE
        cls[self.permanent] = PERMANENT
        self.cls = cls
        self.caution = caution
        self.lethal = slope_bad | obstacle | self.permanent

    # ------------------------------------------------------------------ outputs
    def percent_known(self) -> float:
        return 100.0 * float(self.known.mean())

    def elev_range(self):
        if not math.isfinite(self._elev_lo):
            return 0.0, 1.0
        lo, hi = self._elev_lo - 0.5, self._elev_hi + 0.5
        return lo, max(hi, lo + 1.0)

    def layers_rgb(self, t: float) -> np.ndarray:
        lo, hi = self.elev_range()
        g = self.ground()
        r = np.where(self.known, np.clip((np.nan_to_num(g, nan=lo) - lo) / (hi - lo), 0, 1) * 255, 0)
        age = np.where(np.isnan(self.first_seen), np.inf, t - self.first_seen)
        b = np.clip(1.0 - age / REVEAL_S, 0, 1) * 255
        img = np.stack([np.round(r), self.cls.astype(np.float64) * 40, np.round(b)], axis=-1)
        return img.astype(np.uint8)

    def to_occupancy(self) -> np.ndarray:
        occ = np.full((self.n, self.n), -1, np.int8)
        occ[self.known] = 0
        occ[self.caution] = 50
        occ[self.lethal] = 100
        return occ

    def stats(self) -> dict:
        lo, hi = self.elev_range()
        return {
            'percent': round(self.percent_known(), 2),
            'known_cells': int(self.known.sum()),
            'total_cells': int(self.n * self.n),
            'hazard_cells': int(self.lethal.sum()),
            'science': [list(s) for s in self.science_xy],
            'permanent_hazards': list(self.permanent_marks),
            'elev_min': round(lo, 2),
            'elev_max': round(hi, 2),
        }
