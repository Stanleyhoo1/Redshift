"""Belief grid built ONLY from sensor readings. Mirror of web/brain/grid.ts (numpy-vectorised)."""
import math
import numpy as np

from .config import G, RES, HALF, N, NCH, CFG


class Belief:
    def __init__(self):
        self.elev = np.full(N, np.nan)
        self.dry_elev = np.full(N, np.nan)
        self.struct = np.zeros(N, dtype=bool)
        self.obst = np.zeros(N)
        self.water = np.full(N, -1.0)
        self.water_t = np.full(N, -1.0)
        self.heat = np.full(N, np.nan)
        self.heat_t = np.full(N, -1.0)
        self.last_seen = np.full(N, -1.0)
        self.channels = [np.zeros(N) for _ in range(NCH)]
        self.risk = np.zeros(N)
        self.predicted = np.zeros(N)
        self.pred_water = np.zeros(N)
        self.haz_dist = np.full(N, 999.0)
        self.water_dist = np.full(N, 999.0)
        self.struct_dist = np.full(N, 999.0)
        self.conf = np.zeros(N)
        self.t = 0.0
        # cell centre coordinates
        ii = np.arange(N)
        self.xs = -HALF + ((ii % G) + 0.5) * RES
        self.ys = -HALF + ((ii // G) + 0.5) * RES

    @staticmethod
    def idx(x, y):
        cx = int(math.floor((x + HALF) / RES))
        cy = int(math.floor((y + HALF) / RES))
        cx = 0 if cx < 0 else G - 1 if cx >= G else cx
        cy = 0 if cy < 0 else G - 1 if cy >= G else cy
        return cy * G + cx

    @staticmethod
    def inside(x, y):
        return -HALF < x < HALF and -HALF < y < HALF

    @staticmethod
    def cx(i):
        return -HALF + ((i % G) + 0.5) * RES

    @staticmethod
    def cy(i):
        return -HALF + ((i // G) + 0.5) * RES

    def update_conf(self, t):
        self.t = t
        seen = self.last_seen >= 0
        self.conf = np.where(seen, np.exp(-(t - self.last_seen) / CFG['confTau']), 0.0)

    def known_frac(self):
        return float(np.count_nonzero(self.last_seen >= 0)) / N

    def blocked_arr(self):
        return self.obst > 0.55

    def blocked(self, i):
        return self.obst[i] > 0.55

    def plan_risk_arr(self):
        return np.maximum(self.risk, self.predicted * 0.85)

    def plan_risk(self, i):
        p = self.predicted[i] * 0.85
        return self.risk[i] if self.risk[i] > p else p


# ---------------------------------------------------------------- grid helpers

def shift2(a2, dx, dy, fill):
    """out[y, x] = a2[y - dy, x - dx] (values move by +dx, +dy)."""
    out = np.full_like(a2, fill)
    h, w = a2.shape
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    out[ys0:ys1, xs0:xs1] = a2[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def dilate_falloff(src, R, f):
    """max over a disk of radius R cells of src(shifted) * f(d) -- vectorised 'paint every cell'."""
    s2 = src.reshape(G, G)
    out = np.zeros_like(s2)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            d = math.sqrt(dx * dx + dy * dy)
            if d > R:
                continue
            k = f(d)
            if k <= 0:
                continue
            np.maximum(out, shift2(s2, dx, dy, 0.0) * k, out=out)
    return out.reshape(-1)


def paint(arr, i, R, f):
    """paint a disk of radius R cells around cell i with value f(d), keeping the max (in place)."""
    x0, y0 = i % G, i // G
    a2 = arr.reshape(G, G)
    ylo, yhi = max(0, y0 - R), min(G - 1, y0 + R)
    xlo, xhi = max(0, x0 - R), min(G - 1, x0 + R)
    yy, xx = np.mgrid[ylo:yhi + 1, xlo:xhi + 1]
    d = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    v = np.where(d <= R, np.vectorize(f, otypes=[float])(d) if callable(f) else f, 0.0)
    sub = a2[ylo:yhi + 1, xlo:xhi + 1]
    np.maximum(sub, v, out=sub)


def paint_lin(arr, i, R, peak, c, E):
    """fast paint for the common TS shape f(d) = peak * clamp01(c - d/E), disk radius R cells."""
    x0, y0 = i % G, i // G
    a2 = arr.reshape(G, G)
    ylo, yhi = max(0, y0 - R), min(G - 1, y0 + R)
    xlo, xhi = max(0, x0 - R), min(G - 1, x0 + R)
    yy, xx = np.mgrid[ylo:yhi + 1, xlo:xhi + 1]
    d = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    v = np.where(d <= R, peak * np.clip(c - d / E, 0.0, 1.0), 0.0)
    sub = a2[ylo:yhi + 1, xlo:xhi + 1]
    np.maximum(sub, v, out=sub)


def distance_field(src_mask, cap_cells=40):
    """8-connected chamfer-style distance (metres) from True cells, capped (beyond cap -> 999)."""
    big = 1e6
    d = np.where(src_mask, 0.0, big).reshape(G, G)
    a, b = RES, RES * 1.4142
    for _ in range(cap_cells):
        nd = d.copy()
        for dx, dy, c in ((1, 0, a), (-1, 0, a), (0, 1, a), (0, -1, a), (1, 1, b), (1, -1, b), (-1, 1, b), (-1, -1, b)):
            np.minimum(nd, shift2(d, dx, dy, big) + c, out=nd)
        if np.array_equal(nd, d):
            break
        d = nd
    d = d.reshape(-1)
    d[d >= big] = 999.0
    return d
