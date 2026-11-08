"""Gravity-aware grid planning: cost maps, 8-connected A*, frontier extraction and pure pursuit.

Cost model (per metre): time proxy `d · cell_cost` plus climb energy `w_energy · g · max(0, Δh)`.
`cell_cost = 1 + 2·(slope/θ_max)²` on known ground, ×3 in caution bands, `unknown_cost` on
unexplored cells (optimistic exploration), and lethal where slope > θ_max(g), step/rock obstacles or
permanent hazards — dilated by the robot radius.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

RING = 1.0e6          # inflated-but-not-core lethal (escapable when starting inside it)
RING_STEP = 30.0      # per-cell cost of crossing an inflation ring in escape mode
CORE_STEP = 100.0     # ... and of the lethal core the robot stands in: out by the shortest way
ESCAPE_CORE_M = 3.0   # how far from the start a core escape may run (a 2.5 m no-go zone plus margin)
SQ2 = math.sqrt(2.0)
_NBRS = [(-1, -1, SQ2), (-1, 0, 1.0), (-1, 1, SQ2), (0, -1, 1.0),
         (0, 1, 1.0), (1, -1, SQ2), (1, 0, 1.0), (1, 1, SQ2)]


@dataclass
class CostMap:
    cost: np.ndarray        # float32; RING for inflation ring, inf for core lethal
    elev: np.ndarray        # float32, NaN where unknown
    known: np.ndarray
    lethal: np.ndarray      # inflated lethal mask (core | ring)
    res: float
    origin: tuple
    g: float
    w_energy: float


def disk_kernel(radius_cells: int) -> np.ndarray:
    d = 2 * radius_cells + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def build_costmap(known, lethal, caution, slope_deg, elev, limits, res, origin, inflation_m=0.6,
                  caution_mode=False, unknown_cost=2.5, w_energy=4.0) -> CostMap:
    theta = limits.max_slope_deg
    s = np.nan_to_num(np.asarray(slope_deg, dtype=np.float32), nan=0.0)
    cost = (1.0 + 2.0 * (s / theta) ** 2).astype(np.float32)
    cost[caution] *= 3.0
    cost[~known] = unknown_cost * (2.0 if caution_mode else 1.0)
    core = (lethal | (known & (s > theta))).astype(np.uint8)
    infl = inflation_m + (0.4 if caution_mode else 0.0)
    r = int(math.ceil(infl / res)) if infl > 0 else 0
    inflated = cv2.dilate(core, disk_kernel(r)).astype(bool) if r > 0 else core.astype(bool)
    cost[inflated] = RING
    cost[core.astype(bool)] = np.inf
    return CostMap(cost=cost, elev=np.asarray(elev, dtype=np.float32), known=np.asarray(known, bool),
                   lethal=inflated, res=float(res), origin=tuple(origin), g=float(limits.g),
                   w_energy=float(w_energy))


def core_escape_mask(cm: CostMap, start, radius_m: float = ESCAPE_CORE_M):
    """Core lethal cells a robot standing ON a core cell may cross to get out: the 8-connected core blob under it,
    within radius_m of the start. None when the start is not a core cell. A robot that slid, drifted or was pushed
    into a no-go zone (or onto an obstacle cell) can then plan its way out, but never through another obstacle."""
    sr, sc = start
    if not cm.cost[sr, sc] == np.inf:
        return None
    _, labels = cv2.connectedComponents((cm.cost == np.inf).astype(np.uint8), connectivity=8)
    H, W = cm.cost.shape
    rr, cc = np.ogrid[0:H, 0:W]
    near = (rr - sr) ** 2 + (cc - sc) ** 2 <= (radius_m / cm.res) ** 2
    return (labels == labels[sr, sc]) & near


def astar(cm: CostMap, start, goal, max_expansions: int = 250_000):
    """8-connected A*. Returns (path [(r, c)...] or None, stats)."""
    t0 = time.perf_counter()
    H, W = cm.cost.shape
    stats = {'expansions': 0, 'ms': 0.0, 'cost': math.inf, 'capped': False, 'length_m': 0.0}
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return None, stats
    cost_l = cm.cost.ravel().tolist()
    goal_i = gr * W + gc
    if cost_l[goal_i] >= RING:
        stats['ms'] = (time.perf_counter() - t0) * 1000
        return None, stats
    escape = cost_l[sr * W + sc] >= RING
    core_ok = core_escape_mask(cm, start) if escape else None
    core_l = core_ok.ravel().tolist() if core_ok is not None else None
    elev_l = np.nan_to_num(cm.elev, nan=0.0).ravel().tolist()
    known_l = cm.known.ravel().tolist()
    res = cm.res
    climb_w = cm.w_energy * cm.g
    N = H * W
    gbest = [math.inf] * N
    came = [-1] * N
    closed = bytearray(N)
    start_i = sr * W + sc
    gbest[start_i] = 0.0

    def h(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return res * ((dr + dc) + (SQ2 - 2.0) * min(dr, dc))

    heap = [(h(sr, sc), 0.0, start_i)]
    found = False
    exp = 0
    while heap:
        _, gcur, i = heapq.heappop(heap)
        if closed[i]:
            continue
        closed[i] = 1
        exp += 1
        if i == goal_i:
            found = True
            break
        if exp > max_expansions:
            stats['capped'] = True
            break
        r, c = divmod(i, W)
        ci = cost_l[i]
        if ci >= RING:
            ci = RING_STEP if ci < math.inf else CORE_STEP
        for dr, dc, dd in _NBRS:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            j = rr * W + cc
            if closed[j]:
                continue
            cj = cost_l[j]
            if cj >= RING:
                if escape and cj < math.inf:
                    cj = RING_STEP
                elif core_l is not None and core_l[j]:
                    cj = CORE_STEP
                else:
                    continue
            if dr and dc:
                a, b = r * W + cc, rr * W + c
                if ((cost_l[a] == math.inf and not (core_l and core_l[a]))
                        or (cost_l[b] == math.inf and not (core_l and core_l[b]))):
                    continue  # no diagonal squeeze between core obstacles
            step = res * dd * 0.5 * (ci + cj)
            if known_l[i] and known_l[j]:
                dz = elev_l[j] - elev_l[i]
                if dz > 0:
                    step += climb_w * dz
            ng = gcur + step
            if ng < gbest[j]:
                gbest[j] = ng
                came[j] = i
                heapq.heappush(heap, (ng + h(rr, cc), ng, j))
    stats['expansions'] = exp
    stats['ms'] = (time.perf_counter() - t0) * 1000
    if not found:
        return None, stats
    path = []
    i = goal_i
    while i != -1:
        path.append(divmod(i, W))
        i = came[i]
    path.reverse()
    stats['cost'] = gbest[goal_i]
    stats['length_m'] = path_length_m(path, res)
    return path, stats


def path_length_m(path, res) -> float:
    return sum(res * math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))


def cell_to_world(rc, res, origin):
    r, c = rc
    return (origin[0] + (c + 0.5) * res, origin[1] + (r + 0.5) * res)


def world_to_cell(x, y, res, origin, shape):
    c = int(math.floor((x - origin[0]) / res))
    r = int(math.floor((y - origin[1]) / res))
    return min(max(r, 0), shape[0] - 1), min(max(c, 0), shape[1] - 1)


def cells_to_world(path, res, origin) -> np.ndarray:
    if not path:
        return np.zeros((0, 2))
    p = np.asarray(path, dtype=np.float64)
    return np.stack([origin[0] + (p[:, 1] + 0.5) * res, origin[1] + (p[:, 0] + 0.5) * res], axis=1)


def path_blocked(path, lethal, start_idx: int = 0) -> int:
    for k in range(max(0, start_idx), len(path)):
        r, c = path[k]
        if lethal[r, c]:
            return k
    return -1


def clearance_mask(lethal, clearance_m: float, res: float) -> np.ndarray:
    """Cells within `clearance_m` (at least one cell) of a lethal cell: the footprint a route check tests."""
    r = max(1, int(round(clearance_m / res)))
    return cv2.dilate(np.asarray(lethal, np.uint8), disk_kernel(r)).astype(bool)


def path_conflicts(path, blocked) -> frozenset:
    """Indices of the route's cells that were already inside `blocked` when it was planned: an escape out of an
    inflation ring or a tight-clearance route passes them on purpose, so they are no reason to halt later."""
    return frozenset(k for k, (r, c) in enumerate(path or ()) if blocked[r, c])


def first_block(path, path_xy, blocked, lethal, start_idx: int = 0, skip_m: float = 0.0,
                accepted: frozenset = frozenset(), accepted_lethal: frozenset = frozenset()):
    """First route cell from `start_idx` that blocks it: inside `blocked` and not passed on purpose (`accepted`,
    from `path_conflicts` at planning time), or on a lethal cell itself unless it was already lethal when the route
    was planned (`accepted_lethal` = path_conflicts(path, lethal): the core escape out of a zone the robot stood in).
    Returns (index, metres along the route) or None; the first `skip_m` metres (the segment under the robot) are
    ignored."""
    dist = 0.0
    for k in range(max(0, start_idx), len(path)):
        if k > start_idx:
            a, b = path_xy[k - 1], path_xy[k]
            dist += math.hypot(b[0] - a[0], b[1] - a[1])
        if dist < skip_m:
            continue
        r, c = path[k]
        if blocked[r, c] and (k not in accepted or (lethal[r, c] and k not in accepted_lethal)):
            return k, dist
    return None


def add_keepout(cm: CostMap, xy, radius_m: float) -> CostMap:
    """Mark a disc of cell centres within radius_m of xy as core lethal, without an inflation ring (in place). For a
    teammate the rescuer must come close to: its footprint plus the rescuer's own radius, nothing more."""
    H, W = cm.cost.shape
    r0, c0 = world_to_cell(xy[0], xy[1], cm.res, cm.origin, (H, W))
    span = int(math.ceil(radius_m / cm.res)) + 1
    rs, cs = slice(max(0, r0 - span), min(H, r0 + span + 1)), slice(max(0, c0 - span), min(W, c0 + span + 1))
    rr, cc = np.mgrid[rs, cs]
    cx = cm.origin[0] + (cc + 0.5) * cm.res
    cy = cm.origin[1] + (rr + 0.5) * cm.res
    disc = np.hypot(cx - xy[0], cy - xy[1]) <= radius_m
    cm.cost[rs, cs][disc] = np.inf
    cm.lethal[rs, cs][disc] = True
    return cm


def reachable_from(cm: CostMap, start) -> np.ndarray:
    """Cells `astar` can reach from `start` (8-connected): free and caution ground, plus the inflation rings when the
    start is inside one, and the core blob under it when it stands on one (astar's escape mode). Lets a caller skip
    goals in other regions without running A*."""
    sr, sc = start
    escape = cm.cost[sr, sc] >= RING
    passable = cm.cost < (np.inf if escape else RING)
    core_ok = core_escape_mask(cm, start) if escape else None
    if core_ok is not None:
        passable |= core_ok
    passable = passable.astype(np.uint8)
    passable[sr, sc] = 1
    _, labels = cv2.connectedComponents(passable, connectivity=8)
    return labels == labels[sr, sc]


def standoff_cells(cm: CostMap, target_xy, approach_rad: float, r_min: float = 2.0, r_max: float = 2.45,
                   r_pref: float = 2.2, step_deg: float = 15.0) -> list:
    """Every plannable cell (outside lethal cores and inflation rings) whose centre lies r_min..r_max from a teammate,
    best first: the side facing `approach_rad` (in step_deg bands), then the distance closest to r_pref. A goal
    there is reached within the 0.6 m pursuit tolerance at <= r_max + 0.6 m from the teammate's centre."""
    tx, ty = target_xy
    H, W = cm.cost.shape
    r0, c0 = world_to_cell(tx, ty, cm.res, cm.origin, (H, W))
    span = int(math.ceil(r_max / cm.res)) + 1
    step = math.radians(step_deg)
    out = []
    for r in range(max(0, r0 - span), min(H, r0 + span + 1)):
        for c in range(max(0, c0 - span), min(W, c0 + span + 1)):
            if cm.cost[r, c] >= RING:
                continue
            x, y = cell_to_world((r, c), cm.res, cm.origin)
            d = math.hypot(x - tx, y - ty)
            if r_min <= d <= r_max:
                dev = abs(wrap(math.atan2(y - ty, x - tx) - approach_rad))
                out.append((round(dev / step), abs(d - r_pref), (r, c)))
    out.sort()
    return [rc for _, _, rc in out]


@dataclass
class Frontier:
    target: tuple          # (r, c) frontier cell nearest the cluster centroid
    centroid: tuple        # (r, c) float
    size: int

    def target_xy(self, res, origin):
        return cell_to_world(self.target, res, origin)


def find_frontiers(known, free, min_size: int = 4, tile_m: float = 8.0, res: float = 0.5):
    """Known-free cells with an unknown 4-neighbour, clustered (8-conn) and split into ~tile_m tiles."""
    unknown = ~known
    unk_n = np.zeros_like(known)
    unk_n[1:, :] |= unknown[:-1, :]
    unk_n[:-1, :] |= unknown[1:, :]
    unk_n[:, 1:] |= unknown[:, :-1]
    unk_n[:, :-1] |= unknown[:, 1:]
    fmask = known & free & unk_n
    if not fmask.any():
        return []
    n, labels = cv2.connectedComponents(fmask.astype(np.uint8), connectivity=8)
    rows, cols = np.nonzero(fmask)
    T = max(1, int(round(tile_m / res)))
    tiles_per_row = labels.shape[1] // T + 1
    key = labels[rows, cols].astype(np.int64) * 1_000_000 + (rows // T) * tiles_per_row + (cols // T)
    order = np.argsort(key, kind='stable')
    key, rows, cols = key[order], rows[order], cols[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    out = []
    for rs, cs in zip(np.split(rows, bounds), np.split(cols, bounds)):
        if len(rs) < min_size:
            continue
        cr, cc = rs.mean(), cs.mean()
        k = int(np.argmin((rs - cr) ** 2 + (cs - cc) ** 2))
        out.append(Frontier(target=(int(rs[k]), int(cs[k])), centroid=(float(cr), float(cc)), size=int(len(rs))))
    return out


def score_frontiers(frontiers, robot_xy, res, origin, claims=(), claim_radius=8.0, heading=None,
                    w_size=0.35, w_dist=1.0, min_dist=1.5):
    scored = []
    for f in frontiers:
        x, y = f.target_xy(res, origin)
        d = math.hypot(x - robot_xy[0], y - robot_xy[1])
        if d < min_dist:
            continue
        s = w_size * f.size * res - w_dist * d
        if any(math.hypot(x - cx, y - cy) < claim_radius for cx, cy in claims):
            s -= 40.0
        if heading is not None:
            s += 3.0 * math.cos(math.atan2(y - robot_xy[1], x - robot_xy[0]) - heading)
        scored.append((s, f))
    scored.sort(key=lambda t: -t[0])
    return [f for _, f in scored]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def pure_pursuit(pose, path_xy, lookahead=1.5, v_max=0.8, w_max=1.0, goal_tol=0.6):
    """Returns (v, w, closest_idx, done)."""
    x, y, yaw = pose
    if path_xy is None or len(path_xy) == 0:
        return 0.0, 0.0, 0, True
    p = np.asarray(path_xy, dtype=np.float64)
    d = np.hypot(p[:, 0] - x, p[:, 1] - y)
    idx = int(np.argmin(d))
    d_goal = float(d[-1])
    if d_goal < goal_tol:
        return 0.0, 0.0, len(p) - 1, True
    ahead = np.flatnonzero(d[idx:] >= lookahead)
    tx, ty = p[idx + ahead[0]] if len(ahead) else p[-1]
    alpha = wrap(math.atan2(ty - y, tx - x) - yaw)
    if abs(alpha) > 1.0:
        return 0.0, math.copysign(0.8 * w_max, alpha), idx, False
    v = v_max * (1.0 - 0.5 * abs(alpha)) * min(1.0, d_goal / 1.5)
    ld = max(math.hypot(tx - x, ty - y), 1e-3)
    w = float(np.clip(v * 2.0 * math.sin(alpha) / ld, -w_max, w_max))
    return float(v), w, idx, False
