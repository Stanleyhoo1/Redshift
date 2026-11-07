"""Weighted A* (8-connected) on the map grid.

Step cost = distance * multiplier. The multiplier is 1 + k * slope / limit for known samples,
infinite above the current learned slope limit or on learned obstacles, and moderately raised for
unknown samples (optimism: paths may run into the unknown and get replanned as it is revealed).
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .mapping import FleetMap

INF = float("inf")
SQ2 = math.sqrt(2.0)


def cost_grid(fmap: FleetMap, limit_deg: float, k_slope: float = 1.0, unknown_cost: float = 1.0) -> np.ndarray:
    """Per-sample cost multiplier (float64, shape ny x nx). inf = impassable."""
    known = fmap.known
    slope = fmap.slope
    bad = known & ((slope > limit_deg) | fmap.blocked)
    mult = np.where(known, 1.0 + k_slope * slope / max(limit_deg, 1.0), 1.0 + unknown_cost)
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


def astar(mult: np.ndarray, start: tuple, goal: tuple, weight: float = 1.6, max_expansions: int = 14000,
          escape_radius: int = 2):
    """start/goal: (i, j). Returns (list of (i, j) from start to goal or None, expansions)."""
    ny, nx = mult.shape
    si, sj = start
    gi, gj = goal
    s = sj * nx + si
    gidx = gj * nx + gi
    cost = mult.ravel().tolist()
    # escape: the rover may sit on (or right next to) a sample that is now above the limit
    for dj in range(-escape_radius, escape_radius + 1):
        for di in range(-escape_radius, escape_radius + 1):
            i, j = si + di, sj + dj
            if 0 <= i < nx and 0 <= j < ny:
                n = j * nx + i
                if cost[n] == INF:
                    cost[n] = 6.0
    if cost[gidx] == INF:
        cost[gidx] = 6.0
    N = nx * ny
    g = [INF] * N
    parent = [-1] * N
    closed = bytearray(N)
    g[s] = 0.0
    heap = [(0.0, s)]
    push, pop = heapq.heappush, heapq.heappop
    nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (-1, -1, SQ2))
    c2 = SQ2 - 1.0
    exp = 0
    found = False
    while heap:
        _, cur = pop(heap)
        if closed[cur]:
            continue
        if cur == gidx:
            found = True
            break
        closed[cur] = 1
        exp += 1
        if exp > max_expansions:
            return None, exp
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
    if not found:
        return None, exp
    out = []
    n = gidx
    while n != -1:
        j, i = divmod(n, nx)
        out.append((i, j))
        n = parent[n]
    out.reverse()
    return out, exp


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
