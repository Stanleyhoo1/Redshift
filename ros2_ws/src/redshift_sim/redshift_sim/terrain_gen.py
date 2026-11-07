"""Seeded procedural Super-Earth terrain shared by the Gazebo world builder and the mock world.

The heightfield is a vertex grid: ``heights[i, j]`` is the elevation at
x = −L/2 + j·res, y = −L/2 + i·res (row index grows with +y, matching docs/protocol.md).
Boulders and science crystals are separate objects (Gazebo models / mock primitives), never baked
into the heightfield, so the robots must *perceive* them.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

# heightmap vertex counts must be 2^n + 1 for Gazebo
SECTOR_GRID_N = {50.0: 129, 100.0: 257, 200.0: 257}
RELIEF_M = {50.0: 2.0, 100.0: 3.5, 200.0: 6.0}
ROCK_COUNT = {50.0: 28, 100.0: 48, 200.0: 80}
SCIENCE_COUNT = {50.0: 4, 100.0: 6, 200.0: 8}
CRATER_COUNT = {50.0: 1, 100.0: 3, 200.0: 5}


@dataclass(frozen=True)
class Rock:
    x: float
    y: float
    radius: float   # horizontal semi-axis (m)
    height: float   # vertical extent above ground (m)
    yaw: float


@dataclass(frozen=True)
class Crater:
    x: float
    y: float
    radius: float
    depth: float


@dataclass
class Terrain:
    heights: np.ndarray
    size_m: float
    seed: int
    rocks: list = field(default_factory=list)
    craters: list = field(default_factory=list)
    science: list = field(default_factory=list)      # [(x, y)]
    ridges: list = field(default_factory=list)       # [((x0, y0), (x1, y1), height, half_width)]
    spawn_xy: list = field(default_factory=list)     # rover spawn points [(x, y), (x, y)]
    scout_spawn_xy: tuple = (0.0, 0.0)

    @property
    def n(self) -> int:
        return int(self.heights.shape[0])

    @property
    def res(self) -> float:
        return self.size_m / (self.n - 1)

    @property
    def half(self) -> float:
        return self.size_m / 2.0

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        j = int(round((x + self.half) / self.res))
        i = int(round((y + self.half) / self.res))
        return min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1)

    def height_at(self, x, y):
        """Bilinear elevation; accepts scalars or numpy arrays."""
        fx = np.clip((np.asarray(x, dtype=np.float64) + self.half) / self.res, 0, self.n - 1.000001)
        fy = np.clip((np.asarray(y, dtype=np.float64) + self.half) / self.res, 0, self.n - 1.000001)
        j0 = np.floor(fx).astype(int)
        i0 = np.floor(fy).astype(int)
        tx, ty = fx - j0, fy - i0
        h = self.heights
        v = (h[i0, j0] * (1 - tx) * (1 - ty) + h[i0, j0 + 1] * tx * (1 - ty)
             + h[i0 + 1, j0] * (1 - tx) * ty + h[i0 + 1, j0 + 1] * tx * ty)
        return float(v) if np.ndim(v) == 0 else v

    def slope_deg(self) -> np.ndarray:
        gy, gx = np.gradient(self.heights.astype(np.float64), self.res)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def normal_at(self, x: float, y: float, d: float = 0.3) -> np.ndarray:
        dzdx = (self.height_at(x + d, y) - self.height_at(x - d, y)) / (2 * d)
        dzdy = (self.height_at(x, y + d) - self.height_at(x, y - d)) / (2 * d)
        nrm = np.array([-dzdx, -dzdy, 1.0])
        return nrm / np.linalg.norm(nrm)

    # ---- persistence (runtime/terrain.npz is read by the mock world and world interface) ----
    def save(self, path) -> None:
        meta = {
            'size_m': self.size_m, 'seed': self.seed,
            'rocks': [asdict(r) for r in self.rocks], 'craters': [asdict(c) for c in self.craters],
            'science': [list(s) for s in self.science], 'ridges': self.ridges,
            'spawn_xy': [list(s) for s in self.spawn_xy], 'scout_spawn_xy': list(self.scout_spawn_xy),
        }
        np.savez_compressed(path, heights=self.heights, meta=json.dumps(meta))

    @classmethod
    def load(cls, path) -> 'Terrain':
        with np.load(Path(path)) as z:
            heights = z['heights']
            meta = json.loads(str(z['meta']))
        return cls(heights=heights, size_m=meta['size_m'], seed=meta['seed'],
                   rocks=[Rock(**r) for r in meta['rocks']], craters=[Crater(**c) for c in meta['craters']],
                   science=[tuple(s) for s in meta['science']],
                   ridges=[tuple(tuple(p) if isinstance(p, list) else p for p in r) for r in meta['ridges']],
                   spawn_xy=[tuple(s) for s in meta['spawn_xy']], scout_spawn_xy=tuple(meta['scout_spawn_xy']))


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _fbm(rng: np.random.Generator, n: int, octaves: int = 5, base: int = 4) -> np.ndarray:
    out = np.zeros((n, n))
    amp = 1.0
    for k in range(octaves):
        cells = base * 2 ** k + 1
        g = rng.standard_normal((cells, cells))
        out += amp * ndimage.zoom(g, n / cells, order=3, mode='nearest')[:n, :n]
        amp *= 0.45
    out -= out.mean()
    return out / (np.abs(out).max() + 1e-9)


def _segment_distance(px, py, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0.0, 1.0)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy)), t


def _thermal_erosion(h: np.ndarray, res: float, talus_deg: float = 34.0, iters: int = 60) -> np.ndarray:
    """Relax any neighbour slope steeper than the talus angle (keeps features overlapping sane)."""
    h = h.copy()
    max_dh = math.tan(math.radians(talus_deg)) * res
    for _ in range(iters):
        moved = False
        for axis in (0, 1):
            d = np.diff(h, axis=axis)
            excess = np.where(np.abs(d) > max_dh, (np.abs(d) - max_dh) * np.sign(d) * 0.25, 0.0)
            if not excess.any():
                continue
            moved = True
            if axis == 0:
                h[:-1, :] += excess
                h[1:, :] -= excess
            else:
                h[:, :-1] += excess
                h[:, 1:] -= excess
        if not moved:
            break
    return h


def generate_terrain(size_m: float, seed: int = 42) -> Terrain:
    size_m = float(size_m)
    if size_m not in SECTOR_GRID_N:
        raise ValueError(f'unsupported sector size {size_m}; choose one of {sorted(SECTOR_GRID_N)}')
    rng = np.random.default_rng(seed)
    n = SECTOR_GRID_N[size_m]
    half = size_m / 2
    res = size_m / (n - 1)
    xs = -half + np.arange(n) * res
    X, Y = np.meshgrid(xs, xs)  # X[i, j] = x_j, Y[i, j] = y_i

    h = _fbm(rng, n) * RELIEF_M[size_m] / 2

    # keep the rolling plains gentle: rescale so the 75th-percentile slope is ~7 deg
    def p75_slope(arr):
        gy, gx = np.gradient(arr, res)
        return np.percentile(np.degrees(np.arctan(np.hypot(gx, gy))), 75)
    s75 = p75_slope(h)
    if s75 > 7.0:
        h *= math.tan(math.radians(7.0)) / math.tan(math.radians(s75))

    spawn = [(-0.35 * size_m, -0.35 * size_m), (-0.35 * size_m, -0.25 * size_m)]
    scout_spawn = (-0.28 * size_m, -0.32 * size_m)

    # ridges: impassable at 2.2 g (~26 deg flanks) but crossable at 1 g; tapered ends to walk around
    ridges = []
    n_ridges = 1 if size_m <= 50 else 2
    for k in range(n_ridges):
        if k == 0:  # across the diagonal between the spawn pad and the sector centre
            c = np.array([-0.08 * size_m, -0.08 * size_m]) + rng.uniform(-0.03, 0.03, 2) * size_m
            ang = math.radians(-45 + rng.uniform(-15, 15))
        else:
            c = rng.uniform(0.1, 0.3, 2) * size_m * np.array([1, rng.choice([-1, 1])])
            ang = rng.uniform(0, math.pi)
        length = rng.uniform(0.35, 0.45) * size_m
        rh = float(np.clip(0.025 * size_m, 1.6, 3.5))
        flank = math.radians(rng.uniform(25.0, 29.0))
        hw = 1.5 * rh / math.tan(flank)  # max slope of smoothstep tent = 1.5·H/w
        a = (float(c[0] - math.cos(ang) * length / 2), float(c[1] - math.sin(ang) * length / 2))
        b = (float(c[0] + math.cos(ang) * length / 2), float(c[1] + math.sin(ang) * length / 2))
        d, t = _segment_distance(X, Y, a, b)
        taper = _smoothstep(np.minimum(t, 1 - t) / 0.18)
        h += rh * _smoothstep(1 - d / hw) * taper
        ridges.append((a, b, rh, hw))

    # craters: bowl + raised rim, kept away from the spawn pad
    craters = []
    tries = 0
    while len(craters) < CRATER_COUNT[size_m] and tries < 200:
        tries += 1
        cr = float(rng.uniform(0.05, 0.08) * size_m)
        cx, cy = rng.uniform(-half + cr + 3, half - cr - 3, 2)
        if min(math.hypot(cx - sx, cy - sy) for sx, sy in spawn) < cr + 10:
            continue
        if any(math.hypot(cx - o.x, cy - o.y) < cr + o.radius + 4 for o in craters):
            continue
        if any(float(_segment_distance(np.array(cx), np.array(cy), ra, rb)[0]) < cr + rhw + 3
               for ra, rb, _, rhw in ridges):
            continue
        depth = float(0.35 * cr * rng.uniform(0.9, 1.0))   # wall slope ~ 1.5*d/R ~ 27 deg
        r = np.hypot(X - cx, Y - cy)
        bowl = depth * (1 - _smoothstep(r / cr))
        rim = 0.15 * depth * np.exp(-((r - cr) / (0.3 * cr)) ** 2)
        h += rim - bowl
        craters.append(Crater(float(cx), float(cy), cr, depth))

    # flatten all spawn pads onto one shared plateau level
    pads = spawn + [scout_spawn]
    level = float(np.mean([h[int(round((py + half) / res)), int(round((px + half) / res))] for px, py in pads]))
    w = np.zeros_like(h)
    for sx, sy in pads:
        w = np.maximum(w, 1 - _smoothstep((np.hypot(X - sx, Y - sy) - 4.5) / 8.0))
    h = h * (1 - w) + level * w

    h = _thermal_erosion(h, res)
    h = (h - h.min()).astype(np.float32)
    terr = Terrain(heights=h, size_m=size_m, seed=seed, ridges=ridges, craters=craters,
                   spawn_xy=spawn, scout_spawn_xy=scout_spawn)

    # boulders (separate objects) and cyan science crystals
    rocks: list[Rock] = []
    tries = 0
    while len(rocks) < ROCK_COUNT[size_m] and tries < 5000:
        tries += 1
        x, y = (float(v) for v in rng.uniform(-half + 2, half - 2, 2))
        rad = float(rng.uniform(0.25, 0.7))
        if min(math.hypot(x - sx, y - sy) for sx, sy in spawn) < 6.0 + rad:
            continue
        if any(math.hypot(x - o.x, y - o.y) < rad + o.radius + 0.8 for o in rocks):
            continue
        rocks.append(Rock(round(x, 3), round(y, 3), round(rad, 3),
                          round(float(np.clip(rad * rng.uniform(1.0, 1.6), 0.25, 1.1)), 3),
                          round(float(rng.uniform(0, math.pi)), 3)))
    science = []
    tries = 0
    while len(science) < SCIENCE_COUNT[size_m] and tries < 5000:
        tries += 1
        x, y = (float(v) for v in rng.uniform(-half + 4, half - 4, 2))
        if min(math.hypot(x - sx, y - sy) for sx, sy in spawn) < 10.0:
            continue
        if any(math.hypot(x - o.x, y - o.y) < o.radius + 1.5 for o in rocks):
            continue
        if any(math.hypot(x - a, y - b) < 8.0 for a, b in science):
            continue
        science.append((round(x, 3), round(y, 3)))
    terr.rocks = rocks
    terr.science = science
    return terr
