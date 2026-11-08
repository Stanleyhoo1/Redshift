"""Version 7 landforms: mesas and buttes, one winding canyon with two end ramps, pits (sinkholes and
lava-tube skylights) and boulder fields.

Everything here is ground truth used to build the world. Rover logic never reads it: rovers see the
cliffs, pits and boulders through their LiDAR like any other ground.

Each landform draws only from its own random stream, np.random.default_rng([seed, K]):
7101 mesas, 7102 canyon, 7104 boulder fields, 7105 pits.
"""
from __future__ import annotations

import math

import numpy as np

from . import v7_biomes

K_MESAS, K_CANYON, K_BFIELD, K_PITS = 7101, 7102, 7104, 7105
WALL_DEG = 65.0          # mesa and canyon wall steepness (well above any climb limit)
LANDER_CLEAR = 20.0      # landforms keep at least this far from the lander (scaled down on small maps)


def any_on(cfg) -> bool:
    return bool(cfg.mesas or cfg.canyon or cfg.pits)


def _flood(ter, ok: np.ndarray) -> np.ndarray:
    return ter.flood(ok)


def _reach(ter, H: np.ndarray) -> np.ndarray:
    from .terrain import slope_deg
    return ter.flood(slope_deg(H, ter.res) <= ter.cfg.reach_slope_deg)


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    out = m.copy()
    p = np.pad(m, k)
    ny, nx = m.shape
    for dj in range(-k, k + 1):
        for di in range(-k, k + 1):
            out |= p[k + dj:k + dj + ny, k + di:k + di + nx]
    return out


def _ok_trap(ter, H: np.ndarray, reach0: np.ndarray, foot: np.ndarray, max_lost: float = 40.0) -> bool:
    """Generalised no-trap check: ground reachable from the lander before the landform stays reachable,
    apart from the landform's own footprint and a small margin round it."""
    reach = _reach(ter, H)
    lost = reach0 & ~reach & ~_dilate(foot, 4)
    return float(lost.sum()) * ter.res * ter.res <= max_lost


def _clear_dist(ter) -> float:
    near = min(ter.cfg.map_size / 80.0, 1.0)
    return max(LANDER_CLEAR * near, ter.cfg.lander_flat_radius + 10.0)


# ---------------------------------------------------------------------- canyon
def _canyon(ter, h: np.ndarray, reach0: np.ndarray):
    cfg = ter.cfg
    rng = np.random.default_rng([cfg.seed, K_CANYON])
    half = cfg.map_size / 2.0
    X, Y = ter.X, ter.Y
    lim = float(cfg.tow_limit_deg)
    clear = _clear_dist(ter)
    for _ in range(40):
        width = float(rng.uniform(6.0, 9.0))
        depth = float(rng.uniform(3.0, 3.6))
        length = float(rng.uniform(0.75, 0.95)) * cfg.map_size
        # a smoothed random walk across the map, its middle well away from the lander
        ang = float(rng.uniform(0, 2 * math.pi))
        off = float(rng.uniform(clear + width + 4.0, max(clear + width + 5.0, half - 14.0)))
        cx, cy = off * math.cos(ang), off * math.sin(ang)
        head = ang + math.pi / 2 + float(rng.uniform(-0.4, 0.4))
        n = int(length / 0.5)
        turn = np.cumsum(rng.normal(0.0, 0.02, n))
        turn = np.clip(turn, -0.5, 0.5)
        hd = head + turn
        px = np.cumsum(0.5 * np.cos(hd))
        py = np.cumsum(0.5 * np.sin(hd))
        px += cx - px[n // 2]
        py += cy - py[n // 2]
        if np.any(np.abs(px) > half - 6.0) or np.any(np.abs(py) > half - 6.0):
            # clip the walk to the map instead of rejecting it
            inside = (np.abs(px) <= half - 6.0) & (np.abs(py) <= half - 6.0)
            idx = np.flatnonzero(inside)
            if len(idx) < 60:
                continue
            # longest contiguous run
            runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
            run = max(runs, key=len)
            if len(run) * 0.5 < 0.5 * cfg.map_size:
                continue
            px, py = px[run], py[run]
        if float(np.min(np.hypot(px, py))) < clear + width:
            continue
        s = np.arange(len(px)) * 0.5
        total = float(s[-1])
        # centreline ground, smoothed along the path (window about 16 m)
        hc = np.array([ter.height_at(float(x), float(y)) for x, y in zip(px, py)])
        win = 61
        pad = np.pad(hc, win // 2, mode="edge")
        g = np.convolve(pad, np.ones(win) / win, mode="valid")
        budget = math.tan(math.radians(lim - 1.5))
        ramp_len = None
        for L in np.arange(18.0, 0.45 * total, 1.0):
            fs = np.abs(np.gradient(g - depth * np.clip(np.minimum(s, total - s) / L, 0.0, 1.0), 0.5))
            if max(float(fs[s <= L + 1.0].max()), float(fs[s >= total - L - 1.0].max())) <= budget:
                ramp_len = float(L)
                break
        if ramp_len is None:
            continue
        dep = depth * np.clip(np.minimum(s, total - s) / ramp_len, 0.0, 1.0)
        floor = g - dep
        # nearest centreline sample for every grid cell in the bounding box
        pad_m = width / 2 + depth / math.tan(math.radians(WALL_DEG)) + 2.0
        i0, j0 = ter.ij(float(px.min()) - pad_m, float(py.min()) - pad_m)
        i1, j1 = ter.ij(float(px.max()) + pad_m, float(py.max()) + pad_m)
        sx, sy = X[j0:j1 + 1, i0:i1 + 1].ravel(), Y[j0:j1 + 1, i0:i1 + 1].ravel()
        best = np.full(sx.shape, np.inf)
        arg = np.zeros(sx.shape, int)
        for k in range(0, len(px), 2):
            d2 = (sx - px[k]) ** 2 + (sy - py[k]) ** 2
            better = d2 < best
            best[better] = d2[better]
            arg[better] = k
        d = np.sqrt(best)
        F = floor[arg] + np.maximum(0.0, d - width / 2) * math.tan(math.radians(WALL_DEG))
        sub = h[j0:j1 + 1, i0:i1 + 1]
        new = np.minimum(sub.ravel(), F).reshape(sub.shape)
        Hn = h.copy()
        Hn[j0:j1 + 1, i0:i1 + 1] = new
        foot = np.zeros(h.shape, bool)
        foot[j0:j1 + 1, i0:i1 + 1] = (new < sub - 0.05)
        if not _ok_trap(ter, Hn, reach0, foot, max_lost=80.0):
            continue
        # both ramps must be reachable from the lander
        rch = _reach(ter, Hn)
        ok = True
        for k in (int(len(px) * 0.5 * ramp_len / total), int(len(px) * (1 - 0.5 * ramp_len / total))):
            i, j = ter.ij(float(px[k]), float(py[k]))
            if not rch[j, i]:
                ok = False
        mid = len(px) // 2
        i, j = ter.ij(float(px[mid]), float(py[mid]))
        if not ok or not rch[j, i]:
            continue
        ramps = []
        for k, sgn in ((0, 1), (len(px) - 1, -1)):
            k2 = min(max(k + sgn * 4, 0), len(px) - 1)
            ramps.append({"x": round(float(px[k]), 2), "y": round(float(py[k]), 2),
                          "heading": round(math.degrees(math.atan2(py[k2] - py[k], px[k2] - px[k])), 1),
                          "length": round(ramp_len, 1)})
        pts = [[round(float(x), 2), round(float(y), 2)] for x, y in zip(px[::8], py[::8])]
        info = {"id": "canyon1", "path": pts, "width": round(width, 2), "depth": round(depth, 2),
                "ramps": ramps}
        ter._canyon_line = (px, py, s, ramp_len, width)
        return Hn, foot, info
    return h, np.zeros(h.shape, bool), None


# ---------------------------------------------------------------------- mesas and buttes
def _mesas(ter, h: np.ndarray, reach0: np.ndarray, taken: np.ndarray):
    cfg = ter.cfg
    rng = np.random.default_rng([cfg.seed, K_MESAS])
    half = cfg.map_size / 2.0
    X, Y = ter.X, ter.Y
    clear = _clear_dist(ter)
    area = ter.area
    n_mesa = max(1, int(round(int(rng.integers(1, 3)) * area)))
    n_butte = max(1, int(round(int(rng.integers(1, 4)) * area)))
    n_mesa = v7_biomes.count(ter, "mesas", n_mesa)            # v7 biomes: 1.0 (no change) when off
    n_butte = v7_biomes.count(ter, "mesas", n_butte)
    sizes = [("mesa", 6.0, 10.0)] * n_mesa + [("butte", 2.5, 4.5)] * n_butte
    out = []
    foot_all = np.zeros(h.shape, bool)
    kwall = math.tan(math.radians(WALL_DEG))
    for kind, rlo, rhi in sizes:
        for _ in range(60):
            R = float(rng.uniform(rlo, rhi))
            top_h = float(rng.uniform(3.0, 6.0))
            x, y = (float(v) for v in rng.uniform(-half + R + 8.0, half - R - 8.0, 2))
            if math.hypot(x, y) < clear + R + 4.0:
                continue
            if any(math.hypot(x - m["x"], y - m["y"]) < R + m["radius"] + 8.0 for m in out):
                continue
            phase = rng.uniform(0, 2 * math.pi, 3)
            amp = rng.uniform(0.04, 0.12, 3)
            ext = R * 1.4 + top_h / kwall + 3.0
            i0, j0 = ter.ij(x - ext, y - ext)
            i1, j1 = ter.ij(x + ext, y + ext)
            if taken[j0:j1 + 1, i0:i1 + 1].any():
                continue
            dx, dy = X[j0:j1 + 1, i0:i1 + 1] - x, Y[j0:j1 + 1, i0:i1 + 1] - y
            th = np.arctan2(dy, dx)
            rr = np.hypot(dx, dy)
            Rt = R * (1.0 + sum(a * np.sin((m + 2) * th + p) for m, (a, p) in enumerate(zip(amp, phase))))
            sub = h[j0:j1 + 1, i0:i1 + 1]
            base = float(np.mean(sub[rr < R])) if (rr < R).any() else ter.height_at(x, y)
            level = base + top_h
            wall = level - np.maximum(0.0, rr - Rt) * kwall
            talus = 0.5 * np.exp(-((rr - Rt - top_h / kwall - 1.0) / 1.5) ** 2)
            new = np.maximum(sub, np.maximum(wall, sub + talus * (rr > Rt)))
            Hn = h.copy()
            Hn[j0:j1 + 1, i0:i1 + 1] = new
            foot = np.zeros(h.shape, bool)
            foot[j0:j1 + 1, i0:i1 + 1] = rr < Rt + top_h / kwall + 0.5
            if not _ok_trap(ter, Hn, reach0, foot | foot_all):
                continue
            h = Hn
            foot_all |= foot
            out.append({"id": f"mesa{len(out) + 1}", "kind": kind, "x": round(x, 2), "y": round(y, 2),
                        "radius": round(R, 2), "height": round(top_h, 2), "level": round(level, 2)})
            break
    return h, foot_all, out


# ---------------------------------------------------------------------- pits
def _pits(ter, h: np.ndarray, reach0: np.ndarray, taken: np.ndarray):
    cfg = ter.cfg
    rng = np.random.default_rng([cfg.seed, K_PITS])
    half = cfg.map_size / 2.0
    X, Y = ter.X, ter.Y
    clear = _clear_dist(ter)
    n = max(2, int(round(int(rng.integers(2, 5)) * ter.area)))
    n = v7_biomes.count(ter, "pits", n)
    out = []
    foot_all = np.zeros(h.shape, bool)
    from .terrain import slope_deg
    sl = slope_deg(h, ter.res)
    for k in range(n):
        for _ in range(60):
            sky = bool(rng.uniform() < 0.4)
            R = float(rng.uniform(1.5, 2.0)) if sky else float(rng.uniform(2.0, 3.5))
            depth = float(rng.uniform(5.0, 6.0)) if sky else float(rng.uniform(4.0, 5.5))
            x, y = (float(v) for v in rng.uniform(-half + 8.0, half - 8.0, 2))
            if math.hypot(x, y) < clear + R:
                continue
            if any(math.hypot(x - p["x"], y - p["y"]) < R + p["radius"] + 8.0 for p in out):
                continue
            ext = R + 3.0
            i0, j0 = ter.ij(x - ext, y - ext)
            i1, j1 = ter.ij(x + ext, y + ext)
            if taken[j0:j1 + 1, i0:i1 + 1].any() or float(sl[j0:j1 + 1, i0:i1 + 1].max()) > 22.0:
                continue
            rr = np.hypot(X[j0:j1 + 1, i0:i1 + 1] - x, Y[j0:j1 + 1, i0:i1 + 1] - y)
            sub = h[j0:j1 + 1, i0:i1 + 1]
            new = sub - depth * np.clip((R - rr) / 0.5, 0.0, 1.0)
            Hn = h.copy()
            Hn[j0:j1 + 1, i0:i1 + 1] = new
            foot = np.zeros(h.shape, bool)
            foot[j0:j1 + 1, i0:i1 + 1] = rr < R + 0.6
            if not _ok_trap(ter, Hn, reach0, foot | foot_all):
                continue
            h = Hn
            foot_all |= foot
            out.append({"id": f"pit{len(out) + 1}", "kind": "skylight" if sky else "sinkhole",
                        "x": round(x, 2), "y": round(y, 2), "radius": round(R, 2), "depth": round(depth, 2)})
            break
    return h, foot_all, out


# ---------------------------------------------------------------------- the pass
def apply(ter, h: np.ndarray) -> np.ndarray:
    """The landform pass: runs after _make_heights and before water. Order: canyon, mesas, pits.
    Sets ter.landform_mask (cells on or near a landform where nothing else may be placed) and
    ter.landforms (the features message part)."""
    cfg = ter.cfg
    reach0 = _reach(ter, h)
    mask = np.zeros(h.shape, bool)
    info = {}
    ter._canyon_line = None
    if cfg.canyon and v7_biomes.has_canyon(ter):
        ter.heights = h
        h, foot, can = _canyon(ter, h, reach0)
        mask |= foot
        info["canyon"] = can
    elif cfg.canyon:                                       # v7 biomes: this world's ground has no canyon
        info["canyon"] = None
    if cfg.mesas:
        reach0 = reach0 & ~_dilate(mask, 4)
        h, foot, mesas = _mesas(ter, h, reach0, _dilate(mask, 6))
        mask |= foot
        info["mesas"] = mesas
    if cfg.pits:
        reach0 = reach0 & ~_dilate(mask, 4)
        h, foot, pits = _pits(ter, h, reach0, _dilate(mask, 6))
        mask |= foot
        info["pits"] = pits
    ter.landform_mask = _dilate(mask, 2)
    ter.landforms = info
    return h


def spot_clear(ter, x: float, y: float, radius: float) -> bool:
    """True if (x, y) is not inside a pit, on a mesa or cliff wall or in the canyon (within radius)."""
    m = getattr(ter, "landform_mask", None)
    if m is None:
        return True
    i, j = ter.ij(x, y)
    k = max(0, int(math.ceil(radius / ter.res)))
    return not bool(m[max(0, j - k):j + k + 1, max(0, i - k):i + k + 1].any())


# ---------------------------------------------------------------------- boulder fields
def make_boulder_fields(ter) -> list:
    """Dense patches of boulders on gentle open ground. Returns terrain.Rock objects (geoms bfield_<i>)."""
    from .terrain import Rock
    cfg = ter.cfg
    rng = np.random.default_rng([cfg.seed, K_BFIELD])
    half = cfg.map_size / 2.0
    clear = _clear_dist(ter)
    n_fields = max(1, int(round(1.4 * ter.area)))
    n_fields = v7_biomes.count(ter, "boulder_fields", n_fields)
    per = 32 if ter.area <= 2.5 else max(12, int(32 * 2.25 / ter.area))
    rocks = []
    fields = []
    for _ in range(n_fields):
        for _try in range(80):
            R = float(rng.uniform(5.0, 8.0))
            x, y = (float(v) for v in rng.uniform(-half + R + 4.0, half - R - 4.0, 2))
            if math.hypot(x, y) < clear + R:
                continue
            if not spot_clear(ter, x, y, R + 2.0) or ter.wet_near(x, y, R + 1.0):
                continue
            if ter.max_slope_near(x, y, R) > 16.0:
                continue
            if any(math.hypot(x - f["x"], y - f["y"]) < R + f["radius"] + 6.0 for f in fields):
                continue
            break
        else:
            continue
        got = 0
        first = len(rocks)
        for _k in range(per * 8):
            if got >= per:
                break
            rr = R * math.sqrt(float(rng.uniform()))
            aa = float(rng.uniform(0, 2 * math.pi))
            bx, by = x + rr * math.cos(aa), y + rr * math.sin(aa)
            size = float(rng.uniform(0.4, 1.2))
            sx = size / 2 * float(rng.uniform(0.8, 1.2))
            sy = size / 2 * float(rng.uniform(0.7, 1.1))
            sz = size / 2 * float(rng.uniform(0.6, 0.9))
            yaw = float(rng.uniform(0, math.pi))
            if any(math.hypot(bx - r.x, by - r.y) < 0.8 + max(sx, sy, r.sx, r.sy) for r in rocks):
                continue
            if any(math.hypot(bx - r.x, by - r.y) < 1.2 + max(r.sx, r.sy) for r in ter.rocks):
                continue
            z = ter.height_at(bx, by) + sz * 0.45
            rocks.append(Rock(bx, by, z, sx, sy, sz, yaw))
            got += 1
        # The field's boulders for the map view: [x, y, size], size the mean horizontal half-size (m).
        field_rocks = [[round(r.x, 2), round(r.y, 2), round((r.sx + r.sy) / 2, 2)] for r in rocks[first:]]
        fields.append({"id": f"bfield{len(fields) + 1}", "x": round(x, 2), "y": round(y, 2),
                       "radius": round(R, 2), "count": got, "rocks": field_rocks})
    ter.landforms = dict(getattr(ter, "landforms", None) or {})
    ter.landforms["boulder_fields"] = fields
    return rocks


def features(ter) -> dict:
    """The optional landform keys of the features message (only the flags that are on)."""
    info = getattr(ter, "landforms", None)
    return dict(info) if info else {}
