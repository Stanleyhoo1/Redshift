"""Procedural super-Earth terrain: fractal hills, craters, ridges, rocks and science targets.

Everything is seeded and deterministic. Heights live on the protocol grid: heights[j, i] is the
sample at x = origin_x + i * res, y = origin_y + j * res.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config

KINDS = ("mineral", "ice", "vent")
BREACH_STRETCH = 1.5   # crater bowl is (1 + this) times longer in the breach direction
BREACH_WIDTH = 0.7     # angular half width of the breach (rad)
TARGET_PATH_SLOPE = 15.0   # every target can be reached from the lander without crossing steeper ground
KIND_LABEL = {"mineral": "mineral outcrop", "ice": "ice deposit", "vent": "thermal vent"}


@dataclass
class Rock:
    x: float
    y: float
    z: float
    sx: float
    sy: float
    sz: float
    yaw: float


@dataclass
class TargetSpec:
    id: str
    x: float
    y: float
    z: float
    kind: str


def _value_noise(rng: np.random.Generator, nx: int, ny: int, res: float, wavelength: float) -> np.ndarray:
    ext = max(nx, ny) * res
    nl = int(math.ceil(ext / wavelength)) + 3
    lat = rng.uniform(-1.0, 1.0, (nl, nl))
    off = rng.uniform(0.0, 1.0, 2)
    u = np.arange(nx) * res / wavelength + off[0]
    v = np.arange(ny) * res / wavelength + off[1]
    iu = np.floor(u).astype(int)
    iv = np.floor(v).astype(int)
    fu = u - iu
    fv = v - iv
    su = fu ** 3 * (fu * (fu * 6 - 15) + 10)
    sv = fv ** 3 * (fv * (fv * 6 - 15) + 10)
    a = lat[:, iu] * (1 - su) + lat[:, iu + 1] * su
    return a[iv, :] * (1 - sv)[:, None] + a[iv + 1, :] * sv[:, None]


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    """rng.uniform that tolerates an empty range (small maps)."""
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _seg_dist(X, Y, p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L2 = dx * dx + dy * dy
    t = np.clip(((X - p0[0]) * dx + (Y - p0[1]) * dy) / L2, 0.0, 1.0)
    px = p0[0] + t * dx
    py = p0[1] + t * dy
    return np.hypot(X - px, Y - py), t


def slope_deg(h: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(h, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


class Terrain:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.res = cfg.res
        self.nx, self.ny = cfg.nx, cfg.ny
        self.origin = cfg.origin
        rng = np.random.default_rng(cfg.seed)
        xs = self.origin[0] + np.arange(self.nx) * self.res
        ys = self.origin[1] + np.arange(self.ny) * self.res
        self.X, self.Y = np.meshgrid(xs, ys)
        self.features = []
        self.heights = self._make_heights(rng)
        self.min_h = float(self.heights.min())
        self.max_h = float(self.heights.max())
        self.slope = slope_deg(self.heights, self.res)
        self.rocks = self._make_rocks(rng)
        self.truth_heights = self._with_rocks()
        self.reach = self.reachable(TARGET_PATH_SLOPE)
        self.targets = self._make_targets(rng)

    # ------------------------------------------------------------------ heights
    def _make_heights(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        X, Y = self.X, self.Y
        R = np.hypot(X, Y)
        half = cfg.map_size / 2.0
        scale = cfg.map_size / 80.0
        near = min(scale, 1.0)     # shrink "distance from the lander" ranges on small maps

        h = np.zeros_like(X)
        for wl, amp in ((60.0, 3.0), (30.0, 1.4), (15.0, 0.55), (7.5, 0.2), (3.5, 0.06), (1.8, 0.02)):
            h += amp * _value_noise(rng, self.nx, self.ny, self.res, wl)
        # the lander sits in a shallow basin; highlands rise toward the map edge
        h += 5.0 * near * _smoothstep(10.0 * scale, 50.0 * scale, R) * (0.75 + 0.35 * _value_noise(rng, self.nx, self.ny, self.res, 40.0))

        placed = []

        # mounds: rounded hills whose flanks (about 22 to 28 deg) look fine to an optimistic planner but
        # are too steep for the rovers at 2 g. One sits close to the lander so the fleet meets it early.
        n_mounds = int(rng.integers(2, 4))
        for k in range(n_mounds):
            for _ in range(300):
                w = rng.uniform(4.5, 6.0) * near
                dist = rng.uniform(10.0, 15.0) * scale if k == 0 else _u(rng, 14.0 * near, half - 8.0)
                ang = rng.uniform(0, 2 * math.pi)
                cx, cy = dist * math.cos(ang), dist * math.sin(ang)
                if abs(cx) > half - w - 3 or abs(cy) > half - w - 3:
                    continue
                if near < 1.0 and dist < cfg.lander_flat_radius + 1.2 * w + 2.0:
                    continue   # small maps: keep the landing site clear
                if all(math.hypot(cx - px, cy - py) > 1.6 * w + pr + 3 for px, py, pr in placed):
                    break
            else:
                continue
            steep = math.tan(math.radians(rng.uniform(24.0, 28.0)))
            amp = steep * w / 0.858
            r2 = ((X - cx) ** 2 + (Y - cy) ** 2) / (w * w)
            h += amp * np.exp(-r2)
            placed.append((cx, cy, 1.6 * w))
            self.features.append(("mound", round(cx, 1), round(cy, 1), round(w, 1)))

        # ridges: long raised spines with steep flanks
        n_ridges = int(rng.integers(1, 3))
        placed_before_ridges = len(placed)
        for k in range(n_ridges):
            for _ in range(300):
                L = rng.uniform(18.0, 30.0) * scale
                ang = rng.uniform(0, math.pi)
                dist = _u(rng, 17.0 * near, half - 10.0)
                pa = rng.uniform(0, 2 * math.pi)
                mx, my = dist * math.cos(pa), dist * math.sin(pa)
                p0 = (mx - 0.5 * L * math.cos(ang), my - 0.5 * L * math.sin(ang))
                p1 = (mx + 0.5 * L * math.cos(ang), my + 0.5 * L * math.sin(ang))
                d_lander, _ = _seg_dist(np.array([0.0]), np.array([0.0]), p0, p1)
                inb = all(abs(c) < half - 6 for c in (*p0, *p1))
                clear = all(float(_seg_dist(np.array([px]), np.array([py]), p0, p1)[0][0]) > pr + 6
                            for px, py, pr in placed)
                if d_lander[0] > 15.0 * near and inb and clear:
                    break
            else:
                continue
            amp = rng.uniform(3.0, 4.2)
            width = rng.uniform(5.5, 7.0)
            d, t = _seg_dist(X, Y, p0, p1)
            taper = np.sin(np.pi * np.clip(t, 0, 1)) ** 0.6
            wobble = 1.0 + 0.25 * _value_noise(rng, self.nx, self.ny, self.res, 6.0)
            s = np.clip(d / width, 0, 1)
            h += amp * taper * wobble * (1 - s * s) ** 2
            placed.append(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, L / 2))
            self.features.append(("ridge", tuple(round(c, 1) for c in p0), tuple(round(c, 1) for c in p1)))

        # craters: bowl with a raised rim. On one side (the breach) the bowl is stretched into a long
        # gentle ramp and the rim is lowered, so a rover that drives in can always drive back out.
        # (Stretching the radius keeps the floor smooth; scaling the depth by angle would leave a
        # steep pinwheel crease at the centre.)
        n_craters = int(rng.integers(3, 5))
        for k in range(n_craters):
            for _ in range(300):
                rad = rng.uniform(4.5, 8.0) * near
                dist = rng.uniform(13.0, 20.0) * scale if k == 0 else _u(rng, 14.0 * near, half - 8.0)
                ang = rng.uniform(0, 2 * math.pi)
                cx, cy = dist * math.cos(ang), dist * math.sin(ang)
                if abs(cx) > half - rad - 3 or abs(cy) > half - rad - 3:
                    continue
                if near < 1.0 and dist < cfg.lander_flat_radius + 1.6 * rad + 2.0:
                    continue   # small maps: keep the landing site clear
                if all(math.hypot(cx - px, cy - py) > 1.6 * rad + pr + 3 for px, py, pr in placed):
                    break
            else:
                continue
            placed.append((cx, cy, 1.6 * rad))
            depth = rad * rng.uniform(0.22, 0.3)
            rim = depth * rng.uniform(0.35, 0.55)
            breach_dir = math.atan2(-cy, -cx) + rng.uniform(-0.6, 0.6)   # opens roughly toward the lander
            dang = np.angle(np.exp(1j * (np.arctan2(Y - cy, X - cx) - breach_dir)))
            b = np.exp(-(dang / BREACH_WIDTH) ** 2)
            r = np.hypot(X - cx, Y - cy) / (1.0 + BREACH_STRETCH * b)
            rim_b = rim * (1.0 - 0.75 * b)
            inside = rim_b - (depth + rim_b) * (1 - (r / rad) ** 2)
            outside = rim_b * np.exp(-(((r - rad) / (0.55 * rad)) ** 2))
            h += np.where(r < rad, inside, outside)
            self.features.append(("crater", round(cx, 1), round(cy, 1), round(rad, 1)))

        # flatten the landing site
        h0 = float(h[self.ny // 2, self.nx // 2])
        blend = _smoothstep(cfg.lander_flat_radius, cfg.lander_flat_radius + 4.0, R)
        h = h0 + (h - h0) * blend
        # gentle bias so the far terrain spans roughly -3 .. 8 m
        h = h - h0
        lo, hi = float(h.min()), float(h.max())
        if hi > 8.5:
            pos = h > 0
            h[pos] = h[pos] * (8.5 / hi)
        if lo < -3.0:
            neg = h < 0
            h[neg] = h[neg] * (3.0 / -lo)
        return h

    # ------------------------------------------------------------------ helpers
    def height_at(self, x: float, y: float) -> float:
        fx = (x - self.origin[0]) / self.res
        fy = (y - self.origin[1]) / self.res
        i0 = int(np.clip(math.floor(fx), 0, self.nx - 2))
        j0 = int(np.clip(math.floor(fy), 0, self.ny - 2))
        tx = min(max(fx - i0, 0.0), 1.0)
        ty = min(max(fy - j0, 0.0), 1.0)
        H = self.heights
        return float(
            H[j0, i0] * (1 - tx) * (1 - ty) + H[j0, i0 + 1] * tx * (1 - ty)
            + H[j0 + 1, i0] * (1 - tx) * ty + H[j0 + 1, i0 + 1] * tx * ty
        )

    def ij(self, x: float, y: float):
        i = int(round((x - self.origin[0]) / self.res))
        j = int(round((y - self.origin[1]) / self.res))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def reachable(self, max_slope: float) -> np.ndarray:
        """Samples connected to the lander through ground no steeper than max_slope (rocks included)."""
        ok = slope_deg(self.truth_heights, self.res) <= max_slope
        reach = np.zeros_like(ok)
        i, j = self.ij(0.0, 0.0)
        reach[j, i] = ok[j, i]
        while True:
            grown = reach.copy()
            grown[1:, :] |= reach[:-1, :]
            grown[:-1, :] |= reach[1:, :]
            grown[:, 1:] |= reach[:, :-1]
            grown[:, :-1] |= reach[:, 1:]
            grown &= ok
            if (grown == reach).all():
                return reach
            reach = grown

    def max_slope_near(self, x: float, y: float, radius: float) -> float:
        i, j = self.ij(x, y)
        k = max(1, int(math.ceil(radius / self.res)))
        win = self.slope[max(0, j - k):j + k + 1, max(0, i - k):i + k + 1]
        return float(win.max())

    # ------------------------------------------------------------------ rocks
    def _make_rocks(self, rng: np.random.Generator) -> list:
        cfg = self.cfg
        half = cfg.map_size / 2.0
        rocks = []
        n = int(cfg.num_rocks * (cfg.map_size / 80.0) ** 2)
        tries = 0
        while len(rocks) < n and tries < n * 50:
            tries += 1
            x, y = rng.uniform(-half + 2, half - 2, 2)
            if math.hypot(x, y) < cfg.lander_flat_radius + 3.0:
                continue
            size = rng.uniform(0.3, 1.0)
            sx = size / 2 * rng.uniform(0.8, 1.2)
            sy = size / 2 * rng.uniform(0.7, 1.1)
            sz = size / 2 * rng.uniform(0.6, 0.9)
            if any(math.hypot(x - r.x, y - r.y) < 1.5 for r in rocks):
                continue
            z = self.height_at(x, y) + sz * 0.45
            rocks.append(Rock(float(x), float(y), float(z), float(sx), float(sy), float(sz), float(rng.uniform(0, math.pi))))
        return rocks

    # ------------------------------------------------------------------ targets
    def _make_targets(self, rng: np.random.Generator) -> list:
        cfg = self.cfg
        half = cfg.map_size / 2.0
        scale = cfg.map_size / 80.0
        n_clusters = int(rng.integers(2, 4))
        kinds = list(rng.permutation(KINDS))[:n_clusters]
        total = int(rng.integers(6, 10))
        counts = [total // n_clusters] * n_clusters
        for k in range(total - sum(counts)):
            counts[k] += 1

        def ok_spot(x, y, others, min_sep):
            if abs(x) > half - 5 or abs(y) > half - 5:
                return False
            if self.max_slope_near(x, y, 1.5) > 12.0:
                return False
            ti, tj = self.ij(x, y)
            if not self.reach[tj, ti]:
                return False
            if any(math.hypot(x - r.x, y - r.y) < r.sx + r.sy + 1.5 for r in self.rocks):
                return False
            return all(math.hypot(x - ox, y - oy) >= min_sep for ox, oy in others)

        centres = []
        for c in range(n_clusters):
            for _ in range(500):
                dist = _u(rng, 12.0 * min(scale, 1.0), min(34.0 * scale, half - 6))
                ang = rng.uniform(0, 2 * math.pi)
                cx, cy = dist * math.cos(ang), dist * math.sin(ang)
                if ok_spot(cx, cy, centres, 20.0 * scale):
                    centres.append((cx, cy))
                    break
        targets = []
        pts = []
        tid = 1
        for c, (cx, cy) in enumerate(centres):
            placed = 0
            for _ in range(800):
                if placed >= counts[c]:
                    break
                if placed == 0:
                    x, y = cx, cy
                else:
                    rr = rng.uniform(2.5, 6.5)
                    aa = rng.uniform(0, 2 * math.pi)
                    x, y = cx + rr * math.cos(aa), cy + rr * math.sin(aa)
                if not ok_spot(x, y, pts, 2.5):
                    continue
                kind = str(kinds[c]) if rng.uniform() > 0.15 else str(rng.choice(KINDS))
                targets.append(TargetSpec(f"t{tid}", float(x), float(y), self.height_at(x, y), kind))
                pts.append((x, y))
                tid += 1
                placed += 1
        return targets

    def _with_rocks(self) -> np.ndarray:
        H = self.heights.copy()
        for r in self.rocks:
            ext = max(r.sx, r.sy)
            i0, j0 = self.ij(r.x - ext, r.y - ext)
            i1, j1 = self.ij(r.x + ext, r.y + ext)
            sub_x = self.X[j0:j1 + 1, i0:i1 + 1] - r.x
            sub_y = self.Y[j0:j1 + 1, i0:i1 + 1] - r.y
            c, s = math.cos(-r.yaw), math.sin(-r.yaw)
            lx = c * sub_x - s * sub_y
            ly = s * sub_x + c * sub_y
            q = (lx / r.sx) ** 2 + (ly / r.sy) ** 2
            top = r.z + r.sz * np.sqrt(np.clip(1 - q, 0, 1))
            top = np.where(q < 1, top, -np.inf)
            H[j0:j1 + 1, i0:i1 + 1] = np.maximum(H[j0:j1 + 1, i0:i1 + 1], top)
        return H
