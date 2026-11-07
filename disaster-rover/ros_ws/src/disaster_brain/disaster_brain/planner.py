"""PLANNER: A* weighted by predicted risk + bounded Dijkstra. Mirror of web/brain/planner.ts.
The per-cell entry cost is computed for the whole grid at once (numpy), then the searches run on plain lists."""
import heapq
import math
import numpy as np

from .config import G, N, RES, WATER, SLOPE
from .grid import shift2

INF = float('inf')
NB = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))


def cost_grid(b, relaxed=False):
    """cost of entering each cell (inf = impassable); same formula as cellCost() in planner.ts"""
    blocked = b.blocked_arr()
    r = b.plan_risk_arr()
    sl = b.channels[SLOPE]
    w = b.channels[WATER]
    c = 1 + 9 * r * r + 6 * sl + np.where(b.last_seen < 0, np.where(b.water_dist < 6, 5.0, 0.6), 0.0)
    b2 = blocked.reshape(G, G)
    near = np.zeros((G, G), dtype=bool)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near |= shift2(b2, dx, dy, False)
    c = c + np.where(near.reshape(-1), 2.5, 0.0)
    imp = blocked | (sl >= (0.85 if relaxed else 0.6)) | (w > 0.7)
    if not relaxed:
        imp |= r > 0.85
    c = np.where(imp, INF, c)
    return c.tolist(), blocked.tolist()


def astar(b, sx, sy, gx, gy, relaxed=False, max_expand=12000, grid=None):
    s, g = b.idx(sx, sy), b.idx(gx, gy)
    cost, blocked = grid if grid is not None else cost_grid(b, relaxed)
    if cost[g] == INF and g != s:
        return None
    gs = [INF] * N
    frm = [-1] * N
    closed = [False] * N
    gxc, gyc = g % G, g // G

    def h(i):
        return math.hypot(i % G - gxc, i // G - gyc)

    gs[s] = 0.0
    op = [(h(s), s)]
    exp = 0
    while op:
        _, i = heapq.heappop(op)
        if closed[i]:
            continue
        closed[i] = True
        if i == g:
            break
        exp += 1
        if exp > max_expand:
            return None
        x, y = i % G, i // G
        gi = gs[i]
        for dx, dy, dl in NB:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= G or ny >= G:
                continue
            j = ny * G + nx
            if closed[j]:
                continue
            if dx and dy and (blocked[y * G + nx] or blocked[ny * G + x]):
                continue
            c = 1.0 if j == s else cost[j]
            if c == INF:
                continue
            ng = gi + dl * c
            if ng < gs[j]:
                gs[j] = ng
                frm[j] = i
                heapq.heappush(op, (ng + h(j), j))
    if not closed[g]:
        return None
    cells = []
    i = g
    while i != -1:
        cells.append(i)
        i = frm[i]
    cells.reverse()
    path = [(b.cx(i), b.cy(i)) for i in cells]
    if path:
        path[-1] = (gx, gy)
    return dict(path=path, cost=gs[g] * RES, cells=cells)


def reach_map(b, sx, sy, max_dist, relaxed=False, grid=None):
    """Dijkstra from the rover out to max_dist metres. Returns numpy array of travel cost (m-equivalent)."""
    s = b.idx(sx, sy)
    cost, blocked = grid if grid is not None else cost_grid(b, relaxed)
    gs = [INF] * N
    gs[s] = 0.0
    op = [(0.0, s)]
    lim = max_dist / RES * 12  # effectively geometric
    sxc, syc = s % G, s // G
    rc = max_dist / RES
    while op:
        gi, i = heapq.heappop(op)
        if gi > gs[i] or gi > lim:
            continue
        x, y = i % G, i // G
        for dx, dy, dl in NB:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= G or ny >= G:
                continue
            if (nx - sxc) ** 2 + (ny - syc) ** 2 > rc * rc:
                continue
            j = ny * G + nx
            if dx and dy and (blocked[y * G + nx] or blocked[ny * G + x]):
                continue
            c = 1.0 if j == s else cost[j]
            if c == INF:
                continue
            ng = gi + dl * c
            if ng < gs[j]:
                gs[j] = ng
                heapq.heappush(op, (ng, j))
    return np.array(gs) * RES


def path_risk(b, cells, metres):
    worst, at = 0.0, -1.0
    lim = int(math.ceil(metres / RES))
    for k in range(1, min(len(cells), lim)):
        r = b.plan_risk(cells[k])
        if r > worst:
            worst, at = r, k * RES
    return worst, at
