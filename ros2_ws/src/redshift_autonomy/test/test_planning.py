import math

import numpy as np
import pytest

from redshift_autonomy.physics_model import RoverPhysics
from redshift_autonomy.planning import (astar, build_costmap, find_frontiers, path_blocked,
                                        pure_pursuit, score_frontiers)

RES = 0.5
ORIGIN = (0.0, 0.0)


def flat(n=40):
    known = np.ones((n, n), bool)
    lethal = np.zeros((n, n), bool)
    caution = np.zeros((n, n), bool)
    elev = np.zeros((n, n), np.float32)
    return known, lethal, caution, elev


def slope_of(elev):
    gy, gx = np.gradient(elev.astype(float), RES)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def lim(g):
    return RoverPhysics().limits(g)


def test_astar_routes_around_wall():
    known, lethal, caution, elev = flat()
    lethal[0:34, 20] = True
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.0)
    path, stats = astar(cm, (5, 5), (5, 35))
    assert path is not None
    assert all(not lethal[r, c] for r, c in path)
    assert max(r for r, c in path) >= 34          # went through the gap
    assert stats['expansions'] > 0 and stats['ms'] >= 0


def test_inflation_blocks_narrow_gap():
    known, lethal, caution, elev = flat()
    lethal[:, 20] = True
    lethal[19:21, 20] = False                       # 1 m gap, robot radius 0.6 m -> blocked
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.6)
    path, _ = astar(cm, (5, 5), (5, 35))
    assert path is None


def _ridge_world():
    n = 40
    known, lethal, caution, elev = flat(n)
    # east-west ridge across rows 14..26 for cols 0..29 with ~20 deg flanks; flat corridor at cols >= 30
    rise = math.tan(math.radians(20.0)) * RES
    for r in range(14, 27):
        hgt = (r - 14) * rise if r <= 20 else (26 - r) * rise
        elev[r, :30] = hgt
    return known, lethal, caution, elev


def _crosses_ridge_west(path):
    return any(14 < r < 26 and c < 28 for r, c in path)


def test_gravity_changes_route_choice():
    known, lethal, caution, elev = _ridge_world()
    slope = slope_of(elev)
    start, goal = (5, 5), (35, 5)
    cm1 = build_costmap(known, lethal, caution, slope, elev, lim(1.0), RES, ORIGIN, inflation_m=0.0)
    p1, _ = astar(cm1, start, goal)
    cm22 = build_costmap(known, lethal, caution, slope, elev, lim(2.2), RES, ORIGIN, inflation_m=0.0)
    p22, _ = astar(cm22, start, goal)
    assert p1 is not None and p22 is not None
    assert _crosses_ridge_west(p1)        # 20 deg is fine at 1 g -> short way over the ridge
    assert not _crosses_ridge_west(p22)   # 20 deg > 16.8 deg limit at 2.2 g -> detour
    assert len(p22) > len(p1)


def test_energy_term_grows_with_gravity():
    known, lethal, caution, elev = flat()
    elev[:, 10:30] = np.linspace(0, 1.5, 20)[None, :]
    elev[:, 30:] = 1.5   # plateau (no cliff at the ramp top)
    slope = slope_of(elev)
    c1 = build_costmap(known, lethal, caution, slope, elev, lim(1.0), RES, ORIGIN, inflation_m=0.0)
    c3 = build_costmap(known, lethal, caution, slope, elev, lim(3.0), RES, ORIGIN, inflation_m=0.0)
    _, s1 = astar(c1, (20, 2), (20, 37))
    _, s3 = astar(c3, (20, 2), (20, 37))
    assert s3['cost'] > s1['cost']


def test_enclosed_goal_returns_none():
    known, lethal, caution, elev = flat()
    lethal[10:20, 10] = lethal[10:20, 20] = True
    lethal[10, 10:21] = lethal[20, 10:21] = True
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.0)
    path, stats = astar(cm, (2, 2), (15, 15))
    assert path is None


def test_expansion_cap():
    known, lethal, caution, elev = flat(80)
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.0)
    path, stats = astar(cm, (0, 0), (79, 79), max_expansions=50)
    assert path is None and stats['capped']


def test_unknown_cells_are_traversable_but_penalised():
    known, lethal, caution, elev = flat()
    known[:, 15:25] = False
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.0)
    path, _ = astar(cm, (20, 5), (20, 35))
    assert path is not None
    assert cm.cost[20, 20] > cm.cost[20, 5]


def test_frontiers_two_clusters():
    n = 40
    known = np.zeros((n, n), bool)
    known[2:10, 2:10] = True
    known[25:35, 25:35] = True
    free = known.copy()
    fr = find_frontiers(known, free, min_size=3, tile_m=1000.0)
    assert len(fr) == 2
    for f in fr:
        r, c = f.target
        assert known[r, c] and free[r, c]


def test_long_frontier_is_split_into_tiles():
    known = np.zeros((40, 40), bool)
    known[:, :20] = True                       # 20 m long straight frontier at col 19
    fr = find_frontiers(known, known.copy(), min_size=3, tile_m=4.0, res=RES)
    assert len(fr) >= 4


def test_score_frontiers_penalises_claims():
    n = 40
    known = np.zeros((n, n), bool)
    known[2:10, 2:10] = True
    known[2:10, 25:33] = True
    fr = find_frontiers(known, known.copy(), min_size=3, tile_m=1000.0)
    robot = (12 * RES, 6 * RES)
    ranked = score_frontiers(fr, robot, RES, ORIGIN, claims=[])
    claimed_first = ranked[0]
    ranked2 = score_frontiers(fr, robot, RES, ORIGIN, claims=[claimed_first.target_xy(RES, ORIGIN)])
    assert ranked2[0] is not claimed_first


def test_path_blocked_index():
    lethal = np.zeros((10, 10), bool)
    path = [(0, i) for i in range(10)]
    assert path_blocked(path, lethal) == -1
    lethal[0, 6] = True
    assert path_blocked(path, lethal) == 6
    assert path_blocked(path, lethal, start_idx=7) == -1


def test_pure_pursuit_turns_toward_path():
    path = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.5], [3.0, 3.0], [4.0, 5.0]])
    v, w, idx, done = pure_pursuit((0.0, 0.0, 0.0), path, lookahead=1.5, v_max=0.8)
    assert v > 0 and w > 0 and not done
    v, w, idx, done = pure_pursuit((4.0, 5.0, 1.0), path, lookahead=1.5, v_max=0.8)
    assert done and v == 0.0
