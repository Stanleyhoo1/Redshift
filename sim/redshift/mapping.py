"""Shared fleet map built from LiDAR hits.

Per grid sample we keep the LiDAR hit closest to the sample centre (closer hits replace farther
ones, so estimates converge as rovers pass by). Slopes come from a least squares plane through the
3x3 neighbourhood using the true hit positions; a step term (largest deviation of a neighbour from
that plane) makes rocks and ledges show up as steep. Small holes between scan lines are filled from
their neighbours. Changed samples are tracked so each tick only sends deltas.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Config

FILL_OFFSET = 0.40   # pseudo offset for hole-filled samples: any real hit in the footprint replaces them
EDGE_MARGIN_M = 1.5  # no paths or frontiers this close to the map edge
_NBR = [(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1)]


class FleetMap:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nx, self.ny, self.res = cfg.nx, cfg.ny, cfg.res
        self.ox, self.oy = cfg.origin
        shape = (self.ny, self.nx)
        self.known = np.zeros(shape, bool)
        self.h = np.zeros(shape, np.float32)          # height estimate at the sample centre
        self.slope = np.zeros(shape, np.float32)      # effective slope (deg), includes step term
        self.blocked = np.zeros(shape, bool)          # obstacles learned from stalls
        self.hz = np.zeros(shape, np.float32)         # stored hit height
        self.hdx = np.zeros(shape, np.float32)        # stored hit offset from sample centre
        self.hdy = np.zeros(shape, np.float32)
        self.hoff = np.full(shape, np.inf, np.float32)
        self.sent_h = np.full(shape, np.nan, np.float32)
        self.sent_s = np.full(shape, np.nan, np.float32)
        self.dirty = np.zeros(shape, bool)
        # keep a margin along the map edge out of bounds for planning (walls stand just outside it)
        m = max(1, int(math.ceil(EDGE_MARGIN_M / self.res)))
        self.blocked[:m, :] = True
        self.blocked[-m:, :] = True
        self.blocked[:, :m] = True
        self.blocked[:, -m:] = True
        self.known_count = 0
        self.version = 0
        self.xs = (self.ox + np.arange(self.nx) * self.res).astype(np.float32)
        self.ys = (self.oy + np.arange(self.ny) * self.res).astype(np.float32)

    # ------------------------------------------------------------------ coordinates
    def ij(self, x: float, y: float):
        i = int(round((x - self.ox) / self.res))
        j = int(round((y - self.oy) / self.res))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def xy(self, i: int, j: int):
        return self.ox + i * self.res, self.oy + j * self.res

    def explored_pct(self) -> float:
        return 100.0 * self.known_count / float(self.nx * self.ny)

    # ------------------------------------------------------------------ integration
    def integrate(self, pts: np.ndarray) -> None:
        """pts: (N, 3) world hit points on terrain or rocks."""
        if len(pts) == 0:
            return
        fi = (pts[:, 0] - self.ox) / self.res
        fj = (pts[:, 1] - self.oy) / self.res
        i = np.rint(fi).astype(np.int64)
        j = np.rint(fj).astype(np.int64)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        if not ok.any():
            return
        i, j, fi, fj, z = i[ok], j[ok], fi[ok], fj[ok], pts[ok, 2]
        dx = ((fi - i) * self.res).astype(np.float32)
        dy = ((fj - j) * self.res).astype(np.float32)
        off = np.hypot(dx, dy)
        flat = j * self.nx + i
        order = np.lexsort((off, flat))
        flat_s = flat[order]
        first = np.ones(len(flat_s), bool)
        first[1:] = flat_s[1:] != flat_s[:-1]
        sel = order[first]
        f = flat[sel]
        better = off[sel] < self.hoff.ravel()[f]
        if not better.any():
            return
        sel, f = sel[better], f[better]
        newly = ~self.known.ravel()[f]
        self.hz.ravel()[f] = z[sel]
        self.hdx.ravel()[f] = dx[sel]
        self.hdy.ravel()[f] = dy[sel]
        self.hoff.ravel()[f] = off[sel]
        self.known.ravel()[f] = True
        self.known_count += int(newly.sum())
        jj, ii = j[sel], i[sel]
        self._refit(int(jj.min()) - 2, int(jj.max()) + 2, int(ii.min()) - 2, int(ii.max()) + 2)

    def _refit(self, j0: int, j1: int, i0: int, i1: int) -> None:
        """Recompute plane fits, heights and slopes for samples in [j0..j1] x [i0..i1]."""
        j0, i0 = max(j0, 0), max(i0, 0)
        j1, i1 = min(j1, self.ny - 1), min(i1, self.nx - 1)
        # padded views (1 sample border) for neighbour access
        pj0, pj1 = max(j0 - 1, 0), min(j1 + 1, self.ny - 1)
        pi0, pi1 = max(i0 - 1, 0), min(i1 + 1, self.nx - 1)
        K = self.known[pj0:pj1 + 1, pi0:pi1 + 1]
        Z = self.hz[pj0:pj1 + 1, pi0:pi1 + 1]
        DX = self.hdx[pj0:pj1 + 1, pi0:pi1 + 1]
        DY = self.hdy[pj0:pj1 + 1, pi0:pi1 + 1]
        H, W = K.shape
        # pad to full 3x3 access
        Kp = np.zeros((H + 2, W + 2), bool)
        Kp[1:-1, 1:-1] = K
        Zp = np.zeros((H + 2, W + 2), np.float32)
        Zp[1:-1, 1:-1] = Z
        DXp = np.zeros_like(Zp)
        DXp[1:-1, 1:-1] = DX
        DYp = np.zeros_like(Zp)
        DYp[1:-1, 1:-1] = DY
        # centre values
        sj = slice(1, H + 1)
        si = slice(1, W + 1)
        res = self.res
        S1 = np.zeros((H, W), np.float32)
        Su = np.zeros_like(S1)
        Sv = np.zeros_like(S1)
        Suu = np.zeros_like(S1)
        Suv = np.zeros_like(S1)
        Svv = np.zeros_like(S1)
        Sz = np.zeros_like(S1)
        Suz = np.zeros_like(S1)
        Svz = np.zeros_like(S1)
        nbr = []
        for dj, di in _NBR:
            ks = Kp[1 + dj:H + 1 + dj, 1 + di:W + 1 + di]
            w = ks.astype(np.float32)
            u = (di * res + DXp[1 + dj:H + 1 + dj, 1 + di:W + 1 + di]) * w
            v = (dj * res + DYp[1 + dj:H + 1 + dj, 1 + di:W + 1 + di]) * w
            z = Zp[1 + dj:H + 1 + dj, 1 + di:W + 1 + di] * w
            S1 += w
            Su += u
            Sv += v
            Suu += u * u
            Suv += u * v
            Svv += v * v
            Sz += z
            Suz += u * z
            Svz += v * z
            nbr.append((ks, u, v, z))
        # regularise the gradient terms a little (keeps 2 or 3 point fits sane)
        lam = 0.02
        a11, a12, a13 = S1, Su, Sv
        a22, a23, a33 = Suu + lam, Suv, Svv + lam
        det = (a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13) + a13 * (a12 * a23 - a22 * a13))
        good = (S1 >= 3) & (np.abs(det) > 1e-9)
        det = np.where(good, det, 1.0)
        # Cramer's rule for [a, gx, gy]
        a = (Sz * (a22 * a33 - a23 * a23) - a12 * (Suz * a33 - a23 * Svz) + a13 * (Suz * a23 - a22 * Svz)) / det
        gx = (a11 * (Suz * a33 - a23 * Svz) - Sz * (a12 * a33 - a23 * a13) + a13 * (a12 * Svz - Suz * a13)) / det
        gy = (a11 * (a22 * Svz - Suz * a23) - a12 * (a12 * Svz - Suz * a13) + Sz * (a12 * a23 - a22 * a13)) / det
        gx = np.where(good, gx, 0.0)
        gy = np.where(good, gy, 0.0)
        Kc = Kp[sj, si]
        Zc = Zp[sj, si]
        DXc = DXp[sj, si]
        DYc = DYp[sj, si]
        # height at sample centre from its own hit, corrected along the local gradient
        corr = np.clip(gx * DXc + gy * DYc, -0.12, 0.12)
        hc = np.where(Kc, Zc - corr, a)
        # step term: largest deviation of a known neighbour from the plane through the centre point
        step = np.zeros((H, W), np.float32)
        for (ks, u, v, z), (dj, di) in zip(nbr, _NBR):
            if dj == 0 and di == 0:
                continue
            pred = Zc + gx * (u - DXc) + gy * (v - DYc)
            dev = np.where(ks & Kc, np.abs(z - pred), 0.0)
            np.maximum(step, dev, out=step)
        plane = np.degrees(np.arctan(np.hypot(gx, gy)))
        stepdeg = np.degrees(np.arctan(step / res))
        slope = np.maximum(plane, np.where(step > 0.06, stepdeg, 0.0))
        # hole filling: unknown samples well surrounded by known ones
        fill = (~Kc) & (S1 >= 5) & good
        # write back only the inner window
        oj, oi = j0 - pj0, i0 - pi0
        hh, ww = j1 - j0 + 1, i1 - i0 + 1
        win = (slice(j0, j1 + 1), slice(i0, i1 + 1))
        inner = (slice(oj, oj + hh), slice(oi, oi + ww))
        fill_in = fill[inner]
        if fill_in.any():
            self.known[win] |= fill_in
            self.hz[win] = np.where(fill_in, a[inner], self.hz[win])
            self.hdx[win] = np.where(fill_in, 0.0, self.hdx[win])
            self.hdy[win] = np.where(fill_in, 0.0, self.hdy[win])
            self.hoff[win] = np.where(fill_in, FILL_OFFSET, self.hoff[win])
            self.known_count += int(fill_in.sum())
            hc = np.where(fill, a, hc)
        kn = self.known[win]
        self.h[win] = np.where(kn, hc[inner], 0.0)
        self.slope[win] = np.where(kn, slope[inner], 0.0)
        self.dirty[win] |= kn
        self.version += 1

    # ------------------------------------------------------------------ obstacles
    def mark_blocked(self, x: float, y: float, radius: float) -> int:
        i, j = self.ij(x, y)
        k = int(math.ceil(radius / self.res))
        j0, j1 = max(0, j - k), min(self.ny - 1, j + k)
        i0, i1 = max(0, i - k), min(self.nx - 1, i + k)
        yy, xx = np.mgrid[j0:j1 + 1, i0:i1 + 1]
        m = (xx - i) ** 2 + (yy - j) ** 2 <= (radius / self.res) ** 2 + 1e-6
        before = int(self.blocked[j0:j1 + 1, i0:i1 + 1].sum())
        self.blocked[j0:j1 + 1, i0:i1 + 1] |= m
        self.version += 1
        return int(self.blocked[j0:j1 + 1, i0:i1 + 1].sum()) - before

    def traversable(self, limit_deg: float) -> np.ndarray:
        return self.known & (self.slope <= limit_deg) & ~self.blocked

    # ------------------------------------------------------------------ deltas for the protocol
    def take_changed_cells(self) -> list:
        idx = np.flatnonzero(self.dirty)
        self.dirty[:] = False
        if len(idx) == 0:
            return []
        h = self.h.ravel()[idx]
        s = self.slope.ravel()[idx]
        sh = self.sent_h.ravel()[idx]
        ss = self.sent_s.ravel()[idx]
        changed = np.isnan(sh) | (np.abs(h - sh) >= 0.02) | (np.abs(s - ss) >= 0.5)
        idx = idx[changed]
        if len(idx) == 0:
            return []
        h, s = h[changed], s[changed]
        self.sent_h.ravel()[idx] = h
        self.sent_s.ravel()[idx] = s
        return self._cells(idx, h, s)

    def all_cells(self) -> list:
        idx = np.flatnonzero(self.known)
        return self._cells(idx, self.h.ravel()[idx], self.slope.ravel()[idx])

    def _cells(self, idx, h, s) -> list:
        jj, ii = np.divmod(idx, self.nx)
        hq = np.round(h.astype(np.float64), 2).tolist()
        sq = np.round(s.astype(np.float64), 1).tolist()
        return [list(c) for c in zip(ii.tolist(), jj.tolist(), hq, sq)]

    # ------------------------------------------------------------------ frontiers
    def frontier_clusters(self, limit_deg: float, bucket_m: float = 4.0, min_size: int = 4) -> list:
        """Frontier samples: known, traversable, next to a sizeable unknown region.
        Returns [(x, y, size, flat_index)] with the centre snapped onto a frontier sample."""
        trav = self.traversable(limit_deg)
        unk = ~self.known
        # morphological opening of the unknown mask (3x3) drops speckle holes inside explored areas
        e = np.ones((self.ny + 2, self.nx + 2), bool)
        e[1:-1, 1:-1] = unk
        er = np.ones_like(unk)
        for dj, di in _NBR:
            er &= e[1 + dj:self.ny + 1 + dj, 1 + di:self.nx + 1 + di]
        d = np.zeros((self.ny + 2, self.nx + 2), bool)
        d[1:-1, 1:-1] = er
        op = np.zeros_like(unk)
        for dj, di in _NBR:
            op |= d[1 + dj:self.ny + 1 + dj, 1 + di:self.nx + 1 + di]
        op &= unk
        p = np.zeros((self.ny + 2, self.nx + 2), bool)
        p[1:-1, 1:-1] = op
        adj = (p[0:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, 0:-2] | p[1:-1, 2:])
        fr = trav & adj
        idx = np.flatnonzero(fr)
        if len(idx) == 0:
            return []
        jj, ii = np.divmod(idx, self.nx)
        B = max(1, int(round(bucket_m / self.res)))
        nbx = (self.nx + B - 1) // B
        bid = (jj // B) * nbx + (ii // B)
        nb = int(bid.max()) + 1
        cnt = np.bincount(bid, minlength=nb)
        sx = np.bincount(bid, weights=ii, minlength=nb)
        sy = np.bincount(bid, weights=jj, minlength=nb)
        mi = sx / np.maximum(cnt, 1)
        mj = sy / np.maximum(cnt, 1)
        dist = (ii - mi[bid]) ** 2 + (jj - mj[bid]) ** 2
        order = np.lexsort((dist, bid))
        b_sorted = bid[order]
        first = np.ones(len(order), bool)
        first[1:] = b_sorted[1:] != b_sorted[:-1]
        out = []
        for k in order[first]:
            b = bid[k]
            if cnt[b] < min_size:
                continue
            x, y = self.xy(int(ii[k]), int(jj[k]))
            out.append((x, y, int(cnt[b]), int(idx[k])))
        return out
