"""Procedural super-Earth terrain: fractal hills, craters, ridges, rocks, science targets, and (protocol
v4) lakes, a river, alien crystal spires and the home areas of grazer herds.

Everything is seeded and deterministic. Heights live on the protocol grid: heights[j, i] is the
sample at x = origin_x + i * res, y = origin_y + j * res. Feature counts scale with the map area
(the reference is 80 m x 80 m, so an 80 m map is laid out exactly as before the v4 features).

The v4 features use their own random stream, so adding them leaves the base terrain unchanged.
Lakes are carved basins with a flat liquid surface at `level`: inside the outline the ground lies
below the level, and a guard ring outside it is kept just above, so a flat surface geom at the
level shows exactly the lake. Every feature is checked against a flood fill from the lander: a lake
or river that would cut off ground the rovers could reach before is rejected (no traps).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import REACH_SLOPE, TARGET_PATH_SLOPE, Config  # noqa: F401 (the 2 g values, still importable here)
from . import v7_landforms

KINDS = ("mineral", "ice", "vent")            # kinds placed in clusters; "trace" is placed near herds
ALL_KINDS = ("mineral", "ice", "vent", "trace")
INSTRUMENT = {"mineral": "drill", "ice": "scoop", "vent": "sniffer", "trace": "swab"}
BREACH_STRETCH = 1.5   # crater bowl is (1 + this) times longer in the breach direction
BREACH_WIDTH = 0.7     # angular half width of the breach (rad)
KIND_LABEL = {"mineral": "mineral outcrop", "ice": "ice deposit", "vent": "thermal vent", "trace": "organic trace"}
VALUE_RANGE = {"mineral": (2, 7), "ice": (3, 7), "vent": (3, 7), "trace": (6, 10)}   # plus 2 jackpots (8, 9)
# on-board analysis results, by kind and value tier (low <= 4, mid 5 to 7, high >= 8)
ANALYSIS = {
    "mineral": (("Plain fine-grained basalt", "Weathered regolith with iron oxides", "Dusty basalt, no alteration"),
                ("Layered sediment with sulfate veins", "Basalt with olivine crystals", "Carbonate crust over basalt"),
                ("Olivine-rich basalt with water-bearing clays", "Hydrated silica with layered clays")),
    "ice": (("Dusty water ice, low purity", "Ice mixed with fine grit"),
            ("Clean water ice with trapped CO2 bubbles", "Layered ice with thin dust bands"),
            ("Layered ice with dissolved salts and organics", "Briny ice with nitrates")),
    "vent": (("Mostly CO2, traces of SO2", "Warm CO2, nothing unusual"),
             ("Methane and hydrogen sulfide plume", "Hydrogen and CO2 at 40 C"),
             ("Methane with hydrogen and ammonia, out of equilibrium", "Phosphine traces in a warm methane plume")),
    "trace": (("Simple carbon compounds",),
              ("Simple organics, amino-acid like compounds", "Lipid-like films on the soil"),
              ("Complex organics, possible biosignature", "Chiral organic molecules, possible biosignature")),
}


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
    value: int = 5            # true science value (revealed by the on-board analysis)
    analysis: str = ""        # what the analysis will say
    est_value: int = 5        # what the spectrometer suggests from afar (value +- 1)

    @property
    def instrument(self) -> str:
        return INSTRUMENT.get(self.kind, "drill")


@dataclass
class Spire:
    id: str
    x: float
    y: float
    z: float                  # ground height at the base
    height: float
    radius: float
    sides: list               # small side crystals: (azimuth, lean rad, length, radius)


@dataclass
class WaterBody:
    id: str
    kind: str                 # "lake" | "river"
    level: float
    outline: list             # closed polygon [[x, y], ...]
    cx: float
    cy: float
    radius: float = 0.0       # lake: radius of the surface disc geom
    boxes: list = None        # river: surface boxes (cx, cy, half length, half width, yaw)

    def to_msg(self) -> dict:
        return {"id": self.id, "kind": self.kind, "level": round(self.level, 2),
                "outline": [[round(float(x), 2), round(float(y), 2)] for x, y in self.outline]}


@dataclass
class HerdSpec:
    id: str
    cx: float
    cy: float
    radius: float
    members: list             # (x, y, yaw, size m, cruise speed m/s)


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
        frng = np.random.default_rng(cfg.seed * 7717 + 11)    # v4 features: own stream
        self.area = (cfg.map_size / 80.0) ** 2
        xs = self.origin[0] + np.arange(self.nx) * self.res
        ys = self.origin[1] + np.arange(self.ny) * self.res
        self.X, self.Y = np.meshgrid(xs, ys)
        self.features = []
        self._placed = []                  # (x, y, radius) of mounds and craters
        self._ridges = []                  # ridge spines (p0, p1)
        self.heights = self._make_heights(rng)
        self.landform_mask = None                         # v7 landforms: cells nothing else may use
        self.landforms = None
        if v7_landforms.any_on(cfg):
            self.heights = v7_landforms.apply(self, self.heights)
        self.wet = np.zeros(self.heights.shape, bool)     # under liquid
        self.wet_id = np.zeros(self.heights.shape, np.int8)   # 1 + index into self.water, 0 = dry
        self.water: list = []
        if cfg.water:
            self._make_water(frng)
        self.min_h = float(self.heights.min())
        self.max_h = float(self.heights.max())
        self.slope = slope_deg(self.heights, self.res)
        self.rocks = self._make_rocks(rng)
        self.bfield = v7_landforms.make_boulder_fields(self) if cfg.boulder_fields else []
        self.truth_nospire = self._with_rocks()           # terrain and rocks
        self.spires = self._make_spires(frng) if cfg.spires else []
        self.truth_heights = self.with_spires(self.truth_nospire, self.spires)
        self.reach = self.reachable(cfg.target_path_slope_deg)    # every target reachable without steeper ground
        self.herds = self._make_herds(frng) if cfg.lifeforms else []
        self.targets = self._make_targets(rng)
        self._add_traces(frng)
        self._assign_values(frng)

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
        area = (cfg.map_size / 80.0) ** 2
        n_mounds = int(round(int(rng.integers(2, 4)) * area))
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
        n_ridges = int(round(int(rng.integers(1, 3)) * area))
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
            self._ridges.append((p0, p1))
            self.features.append(("ridge", tuple(round(c, 1) for c in p0), tuple(round(c, 1) for c in p1)))

        # craters: bowl with a raised rim. On one side (the breach) the bowl is stretched into a long
        # gentle ramp and the rim is lowered, so a rover that drives in can always drive back out.
        # (Stretching the radius keeps the floor smooth; scaling the depth by angle would leave a
        # steep pinwheel crease at the centre.)
        n_craters = int(round(int(rng.integers(3, 5)) * area))
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

        self._placed = [p for p in placed if not any(abs(p[0] - (a[0] + b[0]) / 2) < 1e-9 and
                                                    abs(p[1] - (a[1] + b[1]) / 2) < 1e-9 for a, b in self._ridges)]
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
        """Samples connected to the lander through dry ground no steeper than max_slope (rocks and
        spires included)."""
        ok = (slope_deg(self.truth_heights, self.res) <= max_slope) & ~self.wet
        return self.flood(ok)

    def flood(self, ok: np.ndarray) -> np.ndarray:
        """Samples of `ok` connected (4-neighbour) to the lander."""
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
            if self.landform_mask is not None and not v7_landforms.spot_clear(self, x, y, 0.5):
                continue
            size = rng.uniform(0.3, 1.0)
            sx = size / 2 * rng.uniform(0.8, 1.2)
            sy = size / 2 * rng.uniform(0.7, 1.1)
            sz = size / 2 * rng.uniform(0.6, 0.9)
            if any(math.hypot(x - r.x, y - r.y) < 1.5 for r in rocks):
                continue
            if self.wet_near(x, y, size + 0.8):
                continue                    # no rocks in or right at the edge of the water
            z = self.height_at(x, y) + sz * 0.45
            rocks.append(Rock(float(x), float(y), float(z), float(sx), float(sy), float(sz), float(rng.uniform(0, math.pi))))
        return rocks

    # ------------------------------------------------------------------ targets
    def _make_targets(self, rng: np.random.Generator) -> list:
        cfg = self.cfg
        half = cfg.map_size / 2.0
        scale = cfg.map_size / 80.0
        grow = self.area ** 0.75           # more clusters and targets on bigger maps (same draws at 80 m)
        n_clusters = int(round(int(rng.integers(2, 4)) * grow))
        perm = list(rng.permutation(KINDS))
        kinds = [perm[k % len(perm)] for k in range(n_clusters)]
        total = int(round(int(rng.integers(6, 10)) * grow))
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
            if self.bfield and any(math.hypot(x - r.x, y - r.y) < r.sx + r.sy + 1.5 for r in self.bfield):
                return False
            if not v7_landforms.spot_clear(self, x, y, 1.5):
                return False
            if self.wet_near(x, y, 2.5) or any(math.hypot(x - sp.x, y - sp.y) < 3.0 for sp in self.spires):
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
        for r in (self.rocks + self.bfield if self.bfield else self.rocks):
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

    # ------------------------------------------------------------------ water (protocol v4)
    def wet_near(self, x: float, y: float, radius: float) -> bool:
        """Is there liquid within radius of (x, y)?"""
        if not self.water:
            return False
        i, j = self.ij(x, y)
        k = max(0, int(math.ceil(radius / self.res)))
        return bool(self.wet[max(0, j - k):j + k + 1, max(0, i - k):i + k + 1].any())

    def water_at(self, x: float, y: float):
        """The water body (WaterBody) whose liquid covers (x, y), or None."""
        if not self.water:
            return None
        i, j = self.ij(x, y)
        k = int(self.wet_id[j, i])
        return self.water[k - 1] if k > 0 else None

    def _clear_of_features(self, x: float, y: float, margin: float) -> bool:
        """Away from mounds and craters (their footprint) and ridge spines (about 7 m wide)."""
        if any(math.hypot(x - px, y - py) < 0.75 * pr + margin for px, py, pr in self._placed):
            return False
        if not v7_landforms.spot_clear(self, x, y, margin + 6.0):
            return False
        for a, b in self._ridges:
            d, _ = _seg_dist(np.array([x]), np.array([y]), a, b)
            if float(d[0]) < 7.0 + margin:
                return False
        return True

    @staticmethod
    def _bilinear(H, ox, oy, res, x, y) -> float:
        ny, nx = H.shape
        fx, fy = (x - ox) / res, (y - oy) / res
        i0 = int(np.clip(math.floor(fx), 0, nx - 2))
        j0 = int(np.clip(math.floor(fy), 0, ny - 2))
        tx, ty = min(max(fx - i0, 0.0), 1.0), min(max(fy - j0, 0.0), 1.0)
        return float(H[j0, i0] * (1 - tx) * (1 - ty) + H[j0, i0 + 1] * tx * (1 - ty)
                     + H[j0 + 1, i0] * (1 - tx) * ty + H[j0 + 1, i0 + 1] * tx * ty)

    def _no_trap(self, H: np.ndarray, wet: np.ndarray, reach0: np.ndarray) -> bool:
        """True if every sample reachable before (reach0) is still reachable with this water, apart
        from the water itself and a small margin around it."""
        ok = (slope_deg(H, self.res) <= self.cfg.reach_slope_deg) & ~wet
        reach = self.flood(ok)
        k = 4
        near = np.zeros_like(wet)
        p = np.pad(wet, k)
        for dj in range(-k, k + 1):
            for di in range(-k, k + 1):
                near |= p[k + dj:k + dj + wet.shape[0], k + di:k + di + wet.shape[1]]
        lost = reach0 & ~reach & ~near
        return float(lost.sum()) * self.res * self.res <= 25.0

    def _make_water(self, frng: np.random.Generator) -> None:
        """One or two lakes away from the lander (and on big maps maybe a short river flowing into
        one). Rejected if a basin would cut off ground the rovers could reach before."""
        cfg = self.cfg
        half = cfg.map_size / 2.0
        scale = cfg.map_size / 80.0
        n_lakes = 1 if self.area < 1.5 else int(frng.integers(1, 3))
        H = self.heights
        reach0 = self.flood(slope_deg(H, self.res) <= cfg.reach_slope_deg)
        sl0 = slope_deg(H, self.res)
        X, Y = self.X, self.Y
        tries = [(12.0, 20.0, 1.5)] * 120 + [(14.0, 24.0, 2.2)] * 120 + [(17.0, 28.0, 3.2)] * 200
        for k in range(n_lakes):
            for (p90_max, slope_max, range_max) in tries:
                if k > 0 and p90_max > 12.0:
                    break                   # a second lake only where it fits easily
                R = float(frng.uniform(4.0, 6.0))
                a2, a3 = float(frng.uniform(0.0, 0.14)), float(frng.uniform(0.0, 0.1))
                p2, p3 = (float(v) for v in frng.uniform(0.0, 2 * math.pi, 2))
                rmax = R * (1.0 + a2 + a3)
                guard = 2.2 * rmax
                dist = _u(frng, 16.0 + 4.0 * min(max(scale - 1.0, 0.0) * 2.0, 1.0), half - guard - 4.0)
                ang = float(frng.uniform(0.0, 2 * math.pi))
                cx, cy = dist * math.cos(ang), dist * math.sin(ang)
                if abs(cx) > half - guard - 4.0 or abs(cy) > half - guard - 4.0:
                    continue
                if not self._clear_of_features(cx, cy, 2.0):
                    continue
                if any(math.hypot(cx - w.cx, cy - w.cy) < guard + 2.2 * w.radius + 8.0 for w in self.water):
                    continue
                dx, dy = X - cx, Y - cy
                r = np.hypot(dx, dy)
                disk = r < guard
                if float(np.percentile(sl0[disk], 90)) > p90_max or float(sl0[disk].max()) > slope_max:
                    continue
                hs = H[disk]
                if float(hs.max() - hs.min()) > range_max:
                    continue
                th = np.arctan2(dy, dx)
                rt = R * (1.0 + a2 * np.sin(2 * th + p2) + a3 * np.sin(3 * th + p3))
                rho = r / rt
                ring = (rho >= 1.0) & (rho < 2.2)
                level = float(H[ring].min()) - 0.05
                depth = float(frng.uniform(0.8, 1.2))
                inside = rho < 1.0
                Hn = H.copy()
                # a shallow shelf at the shore (a rover that drives in can back out), deep in the middle
                Hn[inside] = np.minimum(H, level - 0.05 - depth * (1.0 - rho ** 2) ** 2)[inside]
                w = _smoothstep(1.0, 2.2, rho)
                blend = w * H + (1.0 - w) * (level + 0.05 + 0.3 * (rho - 1.0))
                Hn[ring] = np.maximum(blend, level + 0.03)[ring]
                wet_k = inside & (Hn < level)
                wet_all = self.wet | wet_k
                if not self._no_trap(Hn, wet_all, reach0):
                    continue
                idx = len(self.water) + 1
                thp = np.linspace(0.0, 2 * math.pi, 28, endpoint=False)
                rp = R * (1.0 + a2 * np.sin(2 * thp + p2) + a3 * np.sin(3 * thp + p3))
                outline = [[cx + float(a) * math.cos(t_), cy + float(a) * math.sin(t_)] for a, t_ in zip(rp, thp)]
                lake = WaterBody(f"w{idx}", "lake", level, outline, cx, cy, radius=rmax + 0.25)
                lake._rt = (R, a2, p2, a3, p3)
                H = Hn
                self.wet |= wet_k
                self.wet_id[wet_k] = idx
                self.water.append(lake)
                self.features.append(("lake", round(cx, 1), round(cy, 1), round(R, 1)))
                if k == 0 and self.area >= 1.5 and frng.uniform() < 0.7:
                    H = self._make_river(frng, lake, H, reach0)
                break
        self.heights = H

    def _make_river(self, frng: np.random.Generator, lake: WaterBody, H: np.ndarray, reach0) -> np.ndarray:
        """A short channel flowing into the lake at the same level. Its course follows ground just
        above the water level so the banks stay gentle. Returns the new heights (or H unchanged)."""
        cfg = self.cfg
        half = cfg.map_size / 2.0
        ox, oy, res = self.origin[0], self.origin[1], self.res
        R, a2, p2, a3, p3 = lake._rt
        level = lake.level
        for _ in range(12):
            th0 = float(frng.uniform(0.0, 2 * math.pi))
            rt0 = R * (1.0 + a2 * math.sin(2 * th0 + p2) + a3 * math.sin(3 * th0 + p3))
            p = (lake.cx + 0.6 * rt0 * math.cos(th0), lake.cy + 0.6 * rt0 * math.sin(th0))
            heading = th0
            pts = [p]
            length = float(frng.uniform(14.0, 22.0))
            walked = 0.0
            while walked < length:
                best = None
                for dh in (-0.35, -0.17, 0.0, 0.17, 0.35):
                    hd = heading + dh
                    q = (p[0] + math.cos(hd), p[1] + math.sin(hd))
                    if abs(q[0]) > half - 7.0 or abs(q[1]) > half - 7.0 or math.hypot(*q) < cfg.lander_flat_radius + 12.0:
                        continue
                    dq = math.hypot(q[0] - lake.cx, q[1] - lake.cy)
                    if dq < math.hypot(p[0] - lake.cx, p[1] - lake.cy) - 0.05 and walked < 8.0:
                        continue            # keep flowing away from the lake at first
                    if walked > 4.0 and dq < 1.3 * R * (1.0 + a2 + a3):
                        continue
                    if not self._clear_of_features(q[0], q[1], 2.0):
                        continue
                    hq = self._bilinear(H, ox, oy, res, q[0], q[1])
                    if walked > 3.0 and not (level - 0.1 <= hq <= level + 0.9):
                        continue
                    cost = abs(hq - (level + 0.35)) + 0.1 * abs(dh)
                    if best is None or cost < best[0]:
                        best = (cost, q, hd)
                if best is None:
                    break
                _, p, heading = best
                pts.append(p)
                walked += 1.0
            if walked < 10.0:
                continue
            hw = float(frng.uniform(1.2, 1.5))
            X, Y = self.X, self.Y
            d = np.full(H.shape, np.inf)
            for a, b in zip(pts, pts[1:]):
                dd, _ = _seg_dist(X, Y, a, b)
                d = np.minimum(d, dd)
            core = d < hw
            Hn = H.copy()
            Hn[core] = np.minimum(H, level - 0.05 - 0.45 * (1.0 - (d / hw) ** 2))[core]
            ring = (d >= hw) & (d < hw + 2.6) & ~self.wet
            wr = _smoothstep(hw, hw + 2.6, d)
            blend = wr * H + (1.0 - wr) * (level + 0.05 + 0.15 * (d - hw))
            Hn[ring] = np.maximum(blend, level + 0.03)[ring]
            wet_k = core & (Hn < level) & ~self.wet
            if not self._no_trap(Hn, self.wet | wet_k, reach0):
                continue
            idx = len(self.water) + 1
            # surface boxes along the simplified course, starting at the lake shore
            course = [q for q in pts if math.hypot(q[0] - lake.cx, q[1] - lake.cy) > 0.9 * R * (1.0 - a2 - a3)]
            out = _rdp(course, 0.3) if len(course) > 2 else course
            boxes = []
            for a, b in zip(out, out[1:]):
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                if L < 1e-3:
                    continue
                boxes.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, L / 2.0 + 0.25, hw + 0.4,
                              math.atan2(b[1] - a[1], b[0] - a[0])))
            if not boxes:
                continue
            left, right = [], []
            for n_, q in enumerate(course):
                a = course[max(n_ - 1, 0)]
                b = course[min(n_ + 1, len(course) - 1)]
                t_ = math.atan2(b[1] - a[1], b[0] - a[0])
                nx_, ny_ = -math.sin(t_), math.cos(t_)
                left.append([q[0] + hw * nx_, q[1] + hw * ny_])
                right.append([q[0] - hw * nx_, q[1] - hw * ny_])
            river = WaterBody(f"w{idx}", "river", level, left + right[::-1], out[len(out) // 2][0],
                              out[len(out) // 2][1], boxes=boxes)
            self.wet |= wet_k
            self.wet_id[wet_k] = idx
            self.water.append(river)
            self.features.append(("river", round(river.cx, 1), round(river.cy, 1), round(walked, 1)))
            return Hn
        return H

    # ------------------------------------------------------------------ spires (protocol v4)
    def _make_spires(self, frng: np.random.Generator) -> list:
        """Tall alien crystal spires on open, gentle ground (never in narrow passages, water or the
        landing site). LiDAR sees them like rocks."""
        cfg = self.cfg
        half = cfg.map_size / 2.0
        n = int(round(6 * self.area))
        out = []
        tries = 0
        while len(out) < n and tries < n * 200:
            tries += 1
            x, y = (float(v) for v in frng.uniform(-half + 5.0, half - 5.0, 2))
            if math.hypot(x, y) < cfg.lander_flat_radius + 9.0 or self.wet_near(x, y, 3.0):
                continue
            if self.max_slope_near(x, y, 1.0) > 14.0:
                continue
            if not v7_landforms.spot_clear(self, x, y, 3.0):
                continue
            i, j = self.ij(x, y)
            k = int(round(3.0 / self.res))
            win = self.slope[max(0, j - k):j + k + 1, max(0, i - k):i + k + 1]
            if float((win <= 18.0).mean()) < 0.92:
                continue                    # room to drive round it on every side
            if any(math.hypot(x - r.x, y - r.y) < max(r.sx, r.sy) + 2.5 for r in self.rocks):
                continue
            if any(math.hypot(x - sp.x, y - sp.y) < 7.0 for sp in out):
                continue
            hgt = float(frng.uniform(2.0, 4.0))
            rad = float(frng.uniform(0.28, 0.4))
            a0 = float(frng.uniform(0.0, 2 * math.pi))
            sides = [(a0, float(frng.uniform(0.35, 0.6)), hgt * float(frng.uniform(0.2, 0.3)), 0.55 * rad),
                     (a0 + float(frng.uniform(2.0, 2.8)), float(frng.uniform(0.35, 0.6)),
                      hgt * float(frng.uniform(0.15, 0.25)), 0.5 * rad)]
            out.append(Spire(f"s{len(out) + 1}", x, y, self.height_at(x, y), hgt, rad, sides))
        return out

    def with_spires(self, base: np.ndarray, spires: list) -> np.ndarray:
        """base heights plus standing spires (a column of the spire's radius)."""
        if not spires:
            return base
        H = base.copy()
        for sp in spires:
            ext = sp.radius + 0.1
            i0, j0 = self.ij(sp.x - ext, sp.y - ext)
            i1, j1 = self.ij(sp.x + ext, sp.y + ext)
            dx = self.X[j0:j1 + 1, i0:i1 + 1] - sp.x
            dy = self.Y[j0:j1 + 1, i0:i1 + 1] - sp.y
            top = np.where(dx * dx + dy * dy <= ext * ext, sp.z + sp.height, -np.inf)
            H[j0:j1 + 1, i0:i1 + 1] = np.maximum(H[j0:j1 + 1, i0:i1 + 1], top)
        return H

    # ------------------------------------------------------------------ lifeform herds (protocol v4)
    def _make_herds(self, frng: np.random.Generator) -> list:
        """Home areas of the grazer herds: open, gentle, dry ground the rovers can reach."""
        cfg = self.cfg
        half = cfg.map_size / 2.0
        scale = cfg.map_size / 80.0
        n = 2 if self.area < 1.5 else 3
        out = []
        for k in range(n):
            for _ in range(400):
                dist = _u(frng, 18.0 * min(scale, 1.0), half - 10.0)
                ang = float(frng.uniform(0.0, 2 * math.pi))
                cx, cy = dist * math.cos(ang), dist * math.sin(ang)
                i, j = self.ij(cx, cy)
                if not self.reach[j, i] or self.wet_near(cx, cy, 9.0):
                    continue
                kk = int(round(6.0 / self.res))
                win = self.slope[max(0, j - kk):j + kk + 1, max(0, i - kk):i + kk + 1]
                if float((win <= 12.0).mean()) < 0.85:
                    continue
                if any(math.hypot(cx - h.cx, cy - h.cy) < 25.0 * min(scale, 1.0) for h in out):
                    continue
                members = []
                m = int(frng.integers(3, 5))
                for _q in range(60):
                    if len(members) >= m:
                        break
                    rr, aa = float(frng.uniform(0.5, 3.0)), float(frng.uniform(0.0, 2 * math.pi))
                    x, y = cx + rr * math.cos(aa), cy + rr * math.sin(aa)
                    if self.max_slope_near(x, y, 0.6) > 14.0 or self.wet_near(x, y, 1.0):
                        continue
                    if any(math.hypot(x - mx, y - my) < 1.5 for mx, my, *_r in members):
                        continue
                    members.append((x, y, float(frng.uniform(0.0, 2 * math.pi)), float(frng.uniform(1.0, 1.5)),
                                    float(frng.uniform(0.2, 0.4))))
                if len(members) >= 2:
                    out.append(HerdSpec(f"g{k + 1}", cx, cy, float(frng.uniform(6.0, 8.0)), members))
                    break
        return out

    # ------------------------------------------------------------------ traces and science values (v4b)
    def _add_traces(self, frng: np.random.Generator) -> None:
        """Organic traces near each herd's home area (the creatures themselves are never sampled)."""
        tid = len(self.targets) + 1
        for h in self.herds:
            want = int(frng.integers(1, 3))
            got = 0
            for _ in range(200):
                if got >= want:
                    break
                rr, aa = float(frng.uniform(2.5, 6.5)), float(frng.uniform(0.0, 2 * math.pi))
                x, y = h.cx + rr * math.cos(aa), h.cy + rr * math.sin(aa)
                half = self.cfg.map_size / 2.0
                if abs(x) > half - 5 or abs(y) > half - 5 or self.max_slope_near(x, y, 1.5) > 12.0:
                    continue
                i, j = self.ij(x, y)
                if not self.reach[j, i] or self.wet_near(x, y, 2.5):
                    continue
                if any(math.hypot(x - r.x, y - r.y) < r.sx + r.sy + 1.5 for r in self.rocks):
                    continue
                if any(math.hypot(x - sp.x, y - sp.y) < 3.0 for sp in self.spires):
                    continue
                if any(math.hypot(x - t.x, y - t.y) < 2.5 for t in self.targets):
                    continue
                self.targets.append(TargetSpec(f"t{tid}", x, y, self.height_at(x, y), "trace"))
                tid += 1
                got += 1

    def _assign_values(self, frng: np.random.Generator) -> None:
        """Science value (1 to 10) and on-board analysis line of every target, deterministic per seed:
        traces near the herds tend to score high, and a couple of other finds per map are jackpots."""
        others = [k for k, t in enumerate(self.targets) if t.kind != "trace"]
        jackpots = set(int(k) for k in frng.permutation(others)[:2]) if others else set()
        for k, t in enumerate(self.targets):
            lo, hi = VALUE_RANGE[t.kind]
            v = lo + int((hi - lo + 1) * float(frng.uniform()) ** 1.5)    # most finds are ordinary
            if k in jackpots:
                v = max(v, int(frng.integers(8, 10)))
            tier = 0 if v <= 4 else (1 if v <= 7 else 2)
            lines = ANALYSIS[t.kind][tier]
            t.value = v
            t.analysis = lines[int(frng.integers(len(lines)))]
            t.est_value = int(min(10, max(1, v + int(frng.integers(-1, 2)))))


def _rdp(pts: list, tol: float) -> list:
    """Ramer-Douglas-Peucker on a list of (x, y)."""
    if len(pts) < 3:
        return list(pts)
    (ax, ay), (bx, by) = pts[0], pts[-1]
    L = math.hypot(bx - ax, by - ay) or 1e-9
    best, bi = -1.0, -1
    for k in range(1, len(pts) - 1):
        d = abs((pts[k][0] - ax) * (by - ay) - (pts[k][1] - ay) * (bx - ax)) / L
        if d > best:
            best, bi = d, k
    if best > tol:
        return _rdp(pts[:bi + 1], tol)[:-1] + _rdp(pts[bi:], tol)
    return [pts[0], pts[-1]]
