"""Unexplored gaps inside the fleet's map (PROTOCOL.md version 8): "explore one of those holes", "prioritize the
gaps".

A gap is a connected pocket of unknown, still mappable ground (not identified water, not a hole LiDAR keeps
seeing into) of at least MIN_M2, with known ground around most of its edge. Unknown channels narrower than
about 3 m are sealed first (a morphological closing of the known ground), so a pocket joined to the open unknown
by a thin corridor still counts; a pocket that touches the map edge or is huge is the open unknown, not a gap.

    gaps, grid = find_gaps(fmap, frontiers)

gaps: [{"id", "x", "y", "m2", "r", "enclosed", "frontiers": [(x, y, size, flat_index)], "lab"}], biggest first.
grid: int32 label grid, each gap's label painted on its samples and NEAR samples around them, so the frontier
choice can tell in O(1) whether a frontier sits on a gap's edge (grid.ravel()[frontier_index]).

Pure numpy plus one Python loop over the runs of unknown samples (union find over row runs): a few ms on the
120 m map. The fleet caches the result and refreshes it every few seconds, only while something needs it
(a gaps priority, a command or a question), never in every decision cycle.
"""
from __future__ import annotations

import math

import numpy as np

from .mapping import NORET_HOLE

MIN_M2 = 4.0          # smallest gap (m2)
MIN_ENCLOSED = 0.6    # share of the gap's edge that must be known ground
SEAL = 3              # samples: unknown channels up to about 2 x SEAL x res (3 m) wide are sealed off
MAX_SHARE = 0.25      # a pocket bigger than this share of the map is the open unknown, not a gap
NEAR = 4              # samples: a frontier this close to a gap's samples (2 m) is on its edge
EDGE = 4              # samples: a pocket this close to the map edge is open to the outside


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    """Square dilation of a boolean grid by k samples (separable, edges padded with False)."""
    out = m.copy()
    for _ in range(k):
        o = out.copy()
        o[1:, :] |= out[:-1, :]
        o[:-1, :] |= out[1:, :]
        out = o
    for _ in range(k):
        o = out.copy()
        o[:, 1:] |= out[:, :-1]
        o[:, :-1] |= out[:, 1:]
        out = o
    return out


def _grow_labels(g: np.ndarray, k: int) -> np.ndarray:
    """Spread positive labels k samples into unlabelled neighbours (square, the bigger label wins a tie)."""
    out = g.copy()
    for _ in range(k):
        o = out.copy()
        np.maximum(o[1:, :], out[:-1, :], out=o[1:, :])
        np.maximum(o[:-1, :], out[1:, :], out=o[:-1, :])
        np.maximum(o[:, 1:], out[:, :-1], out=o[:, 1:])
        np.maximum(o[:, :-1], out[:, 1:], out=o[:, :-1])
        out = np.where(g > 0, g, o)
    return out


def label_runs(mask: np.ndarray) -> tuple:
    """4-connected components of a boolean grid. Returns (label grid int32 (0 = background), count)."""
    ny, nx = mask.shape
    p = np.zeros((ny, nx + 2), np.int8)
    p[:, 1:-1] = mask
    d = np.diff(p, axis=1)
    sj, si = np.nonzero(d == 1)              # run starts (row-major order)
    _ej, ei = np.nonzero(d == -1)            # run ends (exclusive), same order
    n = len(sj)
    if n == 0:
        return np.zeros((ny, nx), np.int32), 0
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rows = np.searchsorted(sj, np.arange(ny + 1))
    s_l, e_l = si.tolist(), ei.tolist()
    rows_l = rows.tolist()
    for j in range(1, ny):
        a, a1 = rows_l[j - 1], rows_l[j]
        b, b1 = rows_l[j], rows_l[j + 1]
        while a < a1 and b < b1:
            if s_l[a] < e_l[b] and s_l[b] < e_l[a]:      # overlapping columns: connected
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            if e_l[a] < e_l[b]:
                a += 1
            else:
                b += 1
    roots = np.array([find(k) for k in range(n)], np.int64)
    _u, comp = np.unique(roots, return_inverse=True)
    comp = comp.astype(np.int32) + 1
    lens = ei - si
    flat = np.repeat(sj * nx + si - np.concatenate(([0], np.cumsum(lens)[:-1])), lens) + np.arange(int(lens.sum()))
    grid = np.zeros(ny * nx, np.int32)
    grid[flat] = np.repeat(comp, lens)
    return grid.reshape(ny, nx), int(comp.max())


def find_gaps(fmap, frontiers: list | None = None) -> tuple:
    """The unexplored gaps of the fleet map (see the module doc). frontiers: the fleet's usable frontiers
    [(x, y, size, flat_index)], to tell which gaps a rover can reach the edge of."""
    res = fmap.res
    ny, nx = fmap.known.shape
    unk = ~fmap.known & ~fmap.water & (fmap.noret < NORET_HOLE)
    solid = ~unk
    # seal narrow unknown channels: closing of the known ground (outside the map counts as known here)
    pad = SEAL + 1
    big = np.ones((ny + 2 * pad, nx + 2 * pad), bool)
    big[pad:-pad, pad:-pad] = solid
    closed = ~_dilate(~_dilate(big, SEAL), SEAL)[pad:-pad, pad:-pad]
    mask = unk & ~closed
    lab, n = label_runs(mask)
    if n == 0:
        return [], np.zeros((ny, nx), np.int32)
    flat = lab.ravel()
    area = np.bincount(flat, minlength=n + 1)
    # open to the outside: touches the band along the map edge
    band = np.zeros((ny, nx), bool)
    band[:EDGE, :] = band[-EDGE:, :] = True
    band[:, :EDGE] = band[:, -EDGE:] = True
    edge = np.zeros(n + 1, bool)
    edge[np.unique(lab[band])] = True
    # how much of each pocket's edge is known ground (sealed channels count as open)
    tot = np.zeros(n + 1, np.int64)
    kn = np.zeros(n + 1, np.int64)
    known_edge = solid & ~unk
    for sl_a, sl_b in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                       ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                       ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        nb = lab[sl_b]                       # the neighbour's label
        here = ~mask[sl_a] & (nb > 0)        # an edge sample next to a pocket
        tot += np.bincount(nb[here], minlength=n + 1)
        kn += np.bincount(nb[here & known_edge[sl_a]], minlength=n + 1)
    min_cells = MIN_M2 / (res * res)
    max_cells = MAX_SHARE * nx * ny
    keep = [k for k in range(1, n + 1) if min_cells <= area[k] <= max_cells and not edge[k]
            and tot[k] > 0 and kn[k] >= MIN_ENCLOSED * tot[k]]
    grid = np.zeros((ny, nx), np.int32)
    gaps = []
    for new, k in enumerate(keep, start=1):
        jj, ii = np.nonzero(lab == k)
        grid[jj, ii] = new
        cx, cy = float(ii.mean()), float(jj.mean())
        c = int(np.argmin((ii - cx) ** 2 + (jj - cy) ** 2))      # a gap sample nearest the centroid
        x, y = fmap.xy(int(ii[c]), int(jj[c]))
        dd = np.hypot(ii - ii[c], jj - jj[c]) * res
        gaps.append({"lab": new, "x": round(float(x), 1), "y": round(float(y), 1),
                     "m2": round(float(len(ii)) * res * res, 1), "r": round(float(np.percentile(dd, 90)), 1),
                     "enclosed": round(float(kn[k]) / float(tot[k]), 2), "frontiers": []})
    if not gaps:
        return [], grid
    grid = _grow_labels(grid, NEAR)
    g = grid.ravel()
    by = {gp["lab"]: gp for gp in gaps}
    for fr in frontiers or ():
        k = int(g[fr[3]])
        if k:
            by[k]["frontiers"].append(fr)
    gaps.sort(key=lambda gp: -gp["m2"])
    for k, gp in enumerate(gaps, start=1):
        gp["id"] = f"g{k}"
    return gaps, grid


def gap_at(grid, fmap, x: float, y: float) -> int:
    """Label of the gap at (x, y) in a find_gaps grid (0 = none)."""
    if grid is None:
        return 0
    i, j = fmap.ij(x, y)
    return int(grid[j, i])


def score(gap: dict, x: float, y: float) -> float:
    """How good a gap is for a rover at (x, y): bigger is better, closer is better."""
    return 0.15 * min(gap["m2"], 300.0) - math.hypot(gap["x"] - x, gap["y"] - y)
