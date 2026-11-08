import math

import numpy as np
import pytest

from redshift_autonomy.physics_model import RoverPhysics
from redshift_autonomy.planning import (RING, add_keepout, astar, build_costmap, cell_to_world, cells_to_world,
                                        clearance_mask, core_escape_mask,
                                        find_frontiers, first_block, is_detour, path_blocked, path_conflicts,
                                        prefer_tight, pure_pursuit,
                                        reachable_from, score_frontiers, standoff_cells, world_to_cell)

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


def ring_corridor():
    """A 2-cell corridor between two lethal walls: every corridor cell lies in the inflation ring, so a rover
    standing in it can only leave by the ring-escape route, one cell from the walls."""
    known, lethal, caution, elev = flat()
    lethal[0:26, 0:10] = True
    lethal[0:26, 12:40] = True
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.6)
    path, _ = astar(cm, (2, 10), (35, 20))
    assert path is not None
    return lethal, path, cells_to_world(path, RES, ORIGIN)


def test_route_planned_through_its_own_ring_is_not_blocked_by_the_unchanged_map():
    lethal, path, xy = ring_corridor()
    blocked = clearance_mask(lethal, 0.5, RES)
    # the bare check flags the planner's own escape route: halt → re-plan the same route → halt (the livelock)
    assert first_block(path, xy, blocked, lethal, skip_m=1.0) is not None
    accepted = path_conflicts(path, blocked)
    assert accepted
    assert first_block(path, xy, blocked, lethal, skip_m=1.0, accepted=accepted) is None


def test_new_lethal_cells_still_block_an_accepted_route():
    lethal, path, xy = ring_corridor()
    accepted = path_conflicts(path, clearance_mask(lethal, 0.5, RES))
    k_rock = next(k for k, (r, c) in enumerate(path) if r == 15)        # a boulder lands in the corridor
    rocked = lethal.copy()
    rocked[path[k_rock]] = True
    hit = first_block(path, xy, clearance_mask(rocked, 0.5, RES), rocked, skip_m=1.0, accepted=accepted)
    assert hit is not None and hit[0] == k_rock
    k_open = next(k for k, (r, c) in enumerate(path) if r >= 32)        # and one beside the route in the open
    r, c = path[k_open]
    beside = lethal.copy()
    beside[r, c + 1] = True
    hit = first_block(path, xy, clearance_mask(beside, 0.5, RES), beside, skip_m=1.0, accepted=accepted)
    assert hit is not None and hit[0] <= k_open
    assert hit[1] == pytest.approx(float(np.sum(np.hypot(*np.diff(xy[:hit[0] + 1], axis=0).T))))


def test_first_block_ignores_the_segment_under_the_robot():
    known, lethal, _, _ = flat(10)
    path = [(0, i) for i in range(10)]
    xy = cells_to_world(path, RES, ORIGIN)
    lethal[1, 1] = True                                  # beside cell (0, 1), 0.5 m along the path
    blocked = clearance_mask(lethal, 0.5, RES)
    assert first_block(path, xy, blocked, lethal, skip_m=1.0) is None
    assert first_block(path, xy, blocked, lethal, skip_m=0.0) == (1, pytest.approx(0.5))


def target_costmap(tx, ty, n=60, extra=None):
    """Flat ground with a damaged teammate at (tx, ty), planned as the rescue plans it: an uninflated 1.5 m keep-out
    (its 0.9 m footprint + the rescuer's 0.6 m radius) on top of the inflated map hazards."""
    known, lethal, caution, elev = flat(n)
    if extra is not None:
        lethal |= extra
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.6)
    return add_keepout(cm, (tx, ty), 1.5)


def test_keepout_is_lethal_without_a_ring():
    cm = target_costmap(15.0, 15.0)
    for rc in [(30, 30), (30, 32), (27, 30)]:                     # centres 0.25 / 1.03 / 1.27 m from the teammate
        assert cm.cost[rc] == np.inf and cm.lethal[rc]
    assert cm.cost[30, 34] < RING                                  # 2.03 m out: free, no inflation around it


@pytest.mark.parametrize('tx,ty', [(15.0, 15.0), (15.3, 14.8), (14.6, 15.4), (15.24, 15.11)])
def test_standoff_cells_ring_the_target_outside_its_inflation(tx, ty):
    cm = target_costmap(tx, ty)
    for approach in np.linspace(-math.pi, math.pi, 13):
        cells = standoff_cells(cm, (tx, ty), approach)
        assert cells, 'open ground must always offer a stand-off'
        for rc in cells:
            assert cm.cost[rc] < RING
            x, y = cell_to_world(rc, RES, ORIGIN)
            assert 2.0 <= math.hypot(x - tx, y - ty) <= 2.45     # goal tolerance 0.6 keeps docking start ≤ 3.05 m
        x, y = cell_to_world(cells[0], RES, ORIGIN)
        dev = abs((math.atan2(y - ty, x - tx) - approach + math.pi) % (2 * math.pi) - math.pi)
        assert dev <= math.radians(22.5)                          # the first choice is in the 2nd 15° band or better


def test_standoff_cells_skip_the_side_a_hazard_covers():
    n = 60
    hazard = np.zeros((n, n), bool)
    hazard[:, 32:] = True                                         # everything east of x = 16 m is lethal
    cm = target_costmap(15.0, 15.0, n, extra=hazard)
    cells = standoff_cells(cm, (15.0, 15.0), 0.0)                 # approaching from the (blocked) east
    assert cells
    assert all(cell_to_world(rc, RES, ORIGIN)[0] < 15.0 for rc in cells)


def test_reachable_from_excludes_an_enclosed_pocket():
    known, lethal, caution, elev = flat()
    lethal[10:30, 10] = lethal[10:30, 29] = True
    lethal[10, 10:30] = lethal[29, 10:30] = True                  # a walled pocket
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.6)
    reach = reachable_from(cm, (2, 2))
    assert reach[2, 2] and reach[37, 37]
    assert not reach[20, 20]
    inside = reachable_from(cm, (20, 20))
    assert inside[20, 20] and not inside[2, 2]
    assert cm.cost[9, 9] == RING                                  # inside the wall's inflation ring: may escape it
    ring_start = reachable_from(cm, (9, 9))
    assert ring_start[9, 9] and ring_start[2, 2] and not ring_start[20, 20]


def zone_costmap(n=40, zones=((20, 20, 2),), walls=()):
    """Flat ground with lethal discs (row, col, radius in cells), e.g. a permanent no-go zone a rover slid into."""
    known, lethal, caution, elev = flat(n)
    rr, cc = np.mgrid[0:n, 0:n]
    for r0, c0, rad in zones:
        lethal |= (rr - r0) ** 2 + (cc - c0) ** 2 <= rad ** 2
    for sl in walls:
        lethal[sl] = True
    return lethal, build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN,
                                 inflation_m=0.6)


def core_run(cm, path):
    """Indices of the route's cells that are core lethal."""
    return [k for k, rc in enumerate(path) if cm.cost[rc] == np.inf]


def test_astar_leaves_the_lethal_core_it_stands_in():
    """A rover that slid (or was pushed) into the middle of a no-go zone must still be able to plan its way out:
    only through the blob it stands in, and never back into a core cell once out."""
    _, cm = zone_costmap()
    assert cm.cost[20, 20] == np.inf
    assert core_escape_mask(cm, (20, 20))[20, 20]
    path, _ = astar(cm, (20, 20), (36, 36))
    assert path is not None and path[0] == (20, 20) and path[-1] == (36, 36)
    k = core_run(cm, path)
    assert k == list(range(len(k)))                    # the core cells are the leading run of the route
    assert len(k) <= 3                                 # out by the shortest way: start + 2 cells of a 2-cell zone


def test_core_escape_never_crosses_another_obstacle():
    # a separate wall between the zone and the goal, with a gap at its south end: go round it, never through it
    _, cm = zone_costmap(zones=((20, 12, 2),), walls=[np.s_[6:40, 18]])
    path, _ = astar(cm, (20, 12), (20, 30))
    assert path is not None
    assert all(c != 18 or r < 6 for r, c in path)
    assert all(rc in [path[i] for i in range(4)] for rc in [path[i] for i in core_run(cm, path)])


def test_core_escape_is_bounded():
    # standing in the middle of a lethal area far wider than the 3 m escape radius: no route, as before
    known, lethal, caution, elev = flat()
    lethal[5:35, 5:35] = True
    cm = build_costmap(known, lethal, caution, slope_of(elev), elev, lim(2.2), RES, ORIGIN, inflation_m=0.6)
    assert astar(cm, (20, 20), (38, 38))[0] is None
    assert not reachable_from(cm, (20, 20))[38, 38]
    assert core_escape_mask(cm, (2, 2)) is None        # not in a core: no escape cells at all


def test_reachable_from_a_core_start_includes_the_way_out():
    _, cm = zone_costmap()
    reach = reachable_from(cm, (20, 20))
    assert reach[20, 20] and reach[36, 36] and reach[2, 2]
    assert not reachable_from(*zone_costmap(walls=[np.s_[:, 30]])[1:], (20, 20))[20, 36]   # walled off: still not


def test_escape_route_through_its_own_core_is_not_blocked_by_the_unchanged_map():
    lethal, cm = zone_costmap()
    path, _ = astar(cm, (20, 20), (36, 36))
    xy = cells_to_world(path, RES, ORIGIN)
    blocked = clearance_mask(lethal, 0.5, RES)
    accepted, already = path_conflicts(path, blocked), path_conflicts(path, lethal)
    assert already and first_block(path, xy, blocked, lethal, accepted=accepted) is not None
    assert first_block(path, xy, blocked, lethal, accepted=accepted, accepted_lethal=already) is None
    k_rock = len(path) // 2                             # a boulder later lands on the route: that still blocks it
    rocked = lethal.copy()
    rocked[path[k_rock]] = True
    hit = first_block(path, xy, clearance_mask(rocked, 0.5, RES), rocked, accepted=accepted, accepted_lethal=already)
    assert hit is not None and hit[0] <= k_rock


def test_pure_pursuit_turns_toward_path():
    path = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.5], [3.0, 3.0], [4.0, 5.0]])
    v, w, idx, done = pure_pursuit((0.0, 0.0, 0.0), path, lookahead=1.5, v_max=0.8)
    assert v > 0 and w > 0 and not done
    v, w, idx, done = pure_pursuit((4.0, 5.0, 1.0), path, lookahead=1.5, v_max=0.8)
    assert done and v == 0.0


def test_prefer_tight_only_for_a_long_detour_it_shortens_or_when_nothing_else_routes():
    assert prefer_tight(None, 20.0, 10.0)                  # no route at full clearance
    assert not prefer_tight(None, None, 10.0)
    assert not prefer_tight(30.0, 12.0, 10.0)              # 30 m for a 10 m gap is no detour (limit 2.5 × 10 + 10)
    assert prefer_tight(100.0, 50.0, 10.0)                 # a 100 m loop vs. 50 m at 0.3 m clearance
    assert not prefer_tight(100.0, 70.0, 10.0)             # the tight route must save ≥ 40 %
    assert not prefer_tight(100.0, None, 10.0)
    assert is_detour(35.5, 10.0) and not is_detour(35.0, 10.0)
