"""Weighted A* (8-connected) on the map grid.

Step cost = distance * multiplier. The multiplier is 1 + k * slope / limit for known samples,
infinite above the current learned slope limit, on learned obstacles, on soft ground, on identified
water and in areas the operator put off limits, high on unknown samples that kept giving no LiDAR
return (a hole: liquid or a drop-off), and
raised for unknown samples (optimism: paths may run into the unknown and get replanned as it is
revealed; a rover that cannot see far raises this a lot to stay on mapped ground).

The search is anytime: AStar.run(deadline) stops when the wall clock passes the deadline and can be
resumed in the next decision cycle, so a rover never blows its compute budget on one plan.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .mapping import FleetMap

INF = float("inf")
SQ2 = math.sqrt(2.0)
FOUND, FAIL = "found", "fail"


class _Pending:
    """Returned instead of a path when a plan ran out of time this cycle (resumed next cycle)."""

    def __repr__(self) -> str:
        return "PENDING"

    def __bool__(self) -> bool:
        return False


PENDING = _Pending()


def cost_grid(fmap: FleetMap, limit_deg: float, k_slope: float = 1.0, unknown_cost: float = 1.0) -> np.ndarray:
    """Per-sample cost multiplier (float64, shape ny x nx). inf = impassable."""
    known = fmap.known
    slope = fmap.slope
    bad = (known & ((slope > limit_deg) | fmap.blocked)) | fmap.soft | fmap.avoid | fmap.water
    mult = np.where(known, 1.0 + k_slope * slope / max(limit_deg, 1.0), 1.0 + unknown_cost)
    mult = np.where(fmap.holes(), 8.0 + unknown_cost, mult)      # seen holes: only if nothing else works
    # soft inflation: samples touching an impassable sample cost more (keeps wheels off rock edges)
    p = np.zeros((fmap.ny + 2, fmap.nx + 2), bool)
    p[1:-1, 1:-1] = bad
    near = np.zeros_like(bad)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj or di:
                near |= p[1 + dj:fmap.ny + 1 + dj, 1 + di:fmap.nx + 1 + di]
    mult = np.where(near, mult * 3.0, mult)
    mult[bad] = INF
    return mult


def add_avoid(mult: np.ndarray, fmap: FleetMap, spots: list) -> np.ndarray:
    """Copy of mult with extra cost around (x, y, radius) spots (other rovers)."""
    if not spots:
        return mult
    out = mult.copy()
    for (x, y, rad) in spots:
        i, j = fmap.ij(x, y)
        k = int(math.ceil(rad / fmap.res))
        j0, j1 = max(0, j - k), min(fmap.ny - 1, j + k)
        i0, i1 = max(0, i - k), min(fmap.nx - 1, i + k)
        out[j0:j1 + 1, i0:i1 + 1] *= 4.0
    return out


class AStar:
    """Resumable weighted A*. start/goal: (i, j). The cost grid is copied when the search starts,
    so a search resumed next cycle sees the map as it was (paths are re-checked against the live
    map while driving)."""

    NBRS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (-1, -1, SQ2))

    def __init__(self, mult: np.ndarray, start: tuple, goal: tuple, weight: float = 1.6,
                 max_expansions: int = 14000, escape_radius: int = 2):
        ny, nx = mult.shape
        self.nx, self.ny = nx, ny
        si, sj = start
        self.gi, self.gj = goal
        self.s = sj * nx + si
        self.gidx = self.gj * nx + self.gi
        cost = mult.ravel().tolist()
        # escape: the rover may sit on (or right next to) a sample that is now above the limit
        for dj in range(-escape_radius, escape_radius + 1):
            for di in range(-escape_radius, escape_radius + 1):
                i, j = si + di, sj + dj
                if 0 <= i < nx and 0 <= j < ny:
                    n = j * nx + i
                    if cost[n] == INF:
                        cost[n] = 6.0
        if cost[self.gidx] == INF:
            cost[self.gidx] = 6.0
        self.cost = cost
        N = nx * ny
        self.g = [INF] * N
        self.parent = [-1] * N
        self.closed = bytearray(N)
        self.g[self.s] = 0.0
        self.heap = [(0.0, self.s)]
        self.weight = weight
        self.max_expansions = max_expansions
        self.exp = 0
        self.status = None          # None while running, then FOUND or FAIL
        self.slices = 0

    def run(self, deadline: float | None = None, min_exp: int = 150, max_exp: int | None = None):
        """Expand until the goal is found, the search fails, the wall clock passes deadline
        (perf_counter seconds) or max_exp expansions were done in this call (at least min_exp per
        call either way). Returns FOUND, FAIL or PENDING."""
        if self.status is not None:
            return self.status
        stop_exp = None if max_exp is None else self.exp + max(min_exp, max_exp)
        self.slices += 1
        nx, ny = self.nx, self.ny
        gi, gj, gidx = self.gi, self.gj, self.gidx
        cost, g, parent, closed, heap = self.cost, self.g, self.parent, self.closed, self.heap
        push, pop = heapq.heappush, heapq.heappop
        nbrs = self.NBRS
        weight = self.weight
        c2 = SQ2 - 1.0
        exp = self.exp
        max_exp = self.max_expansions
        check_at = exp + min_exp
        clock = time.perf_counter
        while heap:
            if deadline is not None and exp >= check_at:
                check_at = exp + 128
                if clock() > deadline:
                    self.exp = exp
                    return PENDING
            if stop_exp is not None and exp >= stop_exp:
                self.exp = exp
                return PENDING
            _, cur = pop(heap)
            if closed[cur]:
                continue
            if cur == gidx:
                self.exp = exp
                self.status = FOUND
                return FOUND
            closed[cur] = 1
            exp += 1
            if exp > max_exp:
                break
            cj, ci = divmod(cur, nx)
            gc = g[cur]
            for di, dj, step in nbrs:
                i = ci + di
                j = cj + dj
                if i < 0 or j < 0 or i >= nx or j >= ny:
                    continue
                n = cur + dj * nx + di
                if closed[n]:
                    continue
                c = cost[n]
                if c == INF:
                    continue
                if di and dj and (cost[cur + di] == INF or cost[cur + dj * nx] == INF):
                    continue  # no corner cutting past impassable samples
                ng = gc + step * c
                if ng < g[n]:
                    g[n] = ng
                    parent[n] = cur
                    hx = i - gi if i > gi else gi - i
                    hy = j - gj if j > gj else gj - j
                    hv = hx + c2 * hy if hx > hy else hy + c2 * hx
                    push(heap, (ng + weight * hv, n))
        self.exp = exp
        self.status = FAIL
        return FAIL

    def cells(self) -> list:
        """The path as (i, j) from start to goal (only after FOUND)."""
        out = []
        n = self.gidx
        nx = self.nx
        while n != -1:
            j, i = divmod(n, nx)
            out.append((i, j))
            n = self.parent[n]
        out.reverse()
        return out


def astar(mult: np.ndarray, start: tuple, goal: tuple, weight: float = 1.6, max_expansions: int = 14000,
          escape_radius: int = 2):
    """One-shot search. start/goal: (i, j). Returns (list of (i, j) from start to goal or None, expansions)."""
    s = AStar(mult, start, goal, weight, max_expansions, escape_radius)
    st = s.run(None)
    return (s.cells() if st == FOUND else None), s.exp


def path_length(pts: list) -> float:
    return sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1))


def simplify(pts: list, tol: float = 0.12) -> list:
    """Ramer-Douglas-Peucker on a list of (x, y)."""
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        ax, ay = pts[a]
        bx, by = pts[b]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) or 1e-9
        best, bi = -1.0, -1
        for k in range(a + 1, b):
            px, py = pts[k]
            d = abs((px - ax) * dy - (py - ay) * dx) / L
            if d > best:
                best, bi = d, k
        if best > tol and bi > 0:
            keep[bi] = True
            stack.append((a, bi))
            stack.append((bi, b))
    return [p for p, k in zip(pts, keep) if k]


def downsample(pts: list, max_pts: int = 30) -> list:
    if len(pts) <= max_pts:
        return list(pts)
    idx = np.linspace(0, len(pts) - 1, max_pts).round().astype(int)
    return [pts[k] for k in idx]
