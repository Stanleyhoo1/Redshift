import math

import numpy as np
import pytest

from redshift_sim.scene_events import (DEFER_MAX_S, JOLT, PILLAR, PILLAR_REST_RAD, PillarFall, jolt_cut_angle_deg,
                                       jolt_wrench_world, pillar_drape, pillar_fall_angle, pillar_fall_time,
                                       pillar_footprint, pillar_layout, pillar_segments, rockslide_due,
                                       rockslide_layout)


def test_rockslide_lies_ahead_on_path_and_across_it():
    path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    rocks, impact = rockslide_layout(path, pose=(1.0, 0.0, 0.0), ahead_m=7.0, width_m=7.0, n=6, seed=1)
    assert impact[0] == pytest.approx(8.0, abs=0.01) and impact[1] == pytest.approx(0.0, abs=0.01)
    ys = [r[1] for r in rocks]
    assert min(ys) <= -3.0 and max(ys) >= 3.0           # spans the corridor across the path
    assert all(abs(r[0] - 8.0) < 1.0 for r in rocks)     # line perpendicular to tangent
    assert all(0.5 <= r[2] <= 1.1 for r in rocks)
    assert len(rocks) == 6


def test_rockslide_follows_path_bends():
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 10.0]])
    rocks, impact = rockslide_layout(path, pose=(0.0, 0.0, 0.0), ahead_m=7.0, seed=2)
    assert impact == pytest.approx((3.0, 4.0), abs=0.01)
    assert all(abs(r[1] - 4.0) < 1.0 for r in rocks)     # tangent is +y, rocks spread along x


def test_rockslide_falls_back_to_heading_when_no_path():
    rocks, impact = rockslide_layout(np.zeros((0, 2)), pose=(2.0, 3.0, math.pi / 2), ahead_m=7.0, seed=3)
    assert impact == pytest.approx((2.0, 10.0), abs=0.01)


def test_rockslide_short_path_extends_along_last_segment():
    path = np.array([[0.0, 0.0], [2.0, 0.0]])
    _, impact = rockslide_layout(path, pose=(0.0, 0.0, 0.0), ahead_m=7.0, seed=4)
    assert impact == pytest.approx((7.0, 0.0), abs=0.01)


def test_rockslide_stays_on_a_short_remaining_route():
    # 6 m left to the goal: the line moves back onto the route (60 % of it) instead of landing past the end
    path = np.array([[0.0, 0.0], [6.0, 0.0]])
    rocks, impact = rockslide_layout(path, pose=(0.0, 0.0, 0.0), ahead_m=7.0, seed=5)
    assert impact == pytest.approx((3.6, 0.0), abs=0.01)
    assert all(abs(r[0] - 3.6) < 1.0 for r in rocks)


def test_rockslide_keeps_room_before_the_goal():
    path = np.array([[0.0, 0.0], [10.0, 0.0]])          # 10 m left: 6 m ahead, goal stays 4 m clear
    _, impact = rockslide_layout(path, pose=(0.0, 0.0, 0.0), ahead_m=7.0, seed=6)
    assert impact == pytest.approx((6.0, 0.0), abs=0.01)


def test_rockslide_never_lands_on_the_robot():
    path = np.array([[0.0, 0.0], [4.0, 0.0]])           # 4 m left: clamp to the 3 m minimum stand-off
    _, impact = rockslide_layout(path, pose=(0.0, 0.0, 0.0), ahead_m=7.0, seed=7)
    assert impact == pytest.approx((3.0, 0.0), abs=0.01)


def test_rockslide_waits_for_the_next_route_near_the_goal():
    near_goal = np.array([[0.0, 0.0], [3.0, 0.0]])      # 3 m left: it would land past the goal, off every route
    assert not rockslide_due(near_goal, (0.0, 0.0, 0.0), waited_s=0.0)
    assert not rockslide_due(None, (0.0, 0.0, 0.0), waited_s=1.0)              # between routes
    new_route = np.array([[3.0, 0.0], [3.0, 12.0]])     # the rover picked its next goal: drop across that route
    assert rockslide_due(new_route, (3.0, 0.0, 0.0), waited_s=2.0)
    assert rockslide_due(np.array([[0.0, 0.0], [9.0, 0.0]]), (0.0, 0.0, 0.0), waited_s=0.0)
    assert rockslide_due(near_goal, (0.0, 0.0, 0.0), waited_s=DEFER_MAX_S)     # no new route: heading fallback


def test_jolt_profile():
    # tuned: cut 21.4 deg -> ~33.8 deg peak, 5 deg past the 2.2 g tip alarm (28.8 deg)
    assert jolt_cut_angle_deg(28.8) == pytest.approx(21.4, abs=0.1)
    f, t = jolt_wrench_world(0.0, 2.2)
    assert t[0] == pytest.approx(JOLT['torque_nm_at_2_2g'])
    _, t1 = jolt_wrench_world(math.pi / 2, 1.0)
    assert t1[1] == pytest.approx(JOLT['torque_nm_at_2_2g'] / 2.2) and abs(t1[0]) < 1e-9


# ---------------------------------------------------------------- pillar collapse (spec §1.1)
G22 = 9.81 * 2.2
STRAIGHT = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])


def _signed_offset(p, q):
    """Left (+) / right (−) distance of p from the route along +x through q."""
    return p[1] - q[1]


def test_pillar_stands_beside_the_route_and_its_trunk_crosses_it():
    for seed in range(6):
        lay = pillar_layout(STRAIGHT, (1.0, 0.0, 0.0), seed=seed)
        assert lay['cross'] == pytest.approx((9.0, 0.0), abs=0.01)                # 8 m ahead along the route
        assert lay['side'] in (1, -1)
        assert _signed_offset(lay['base'], lay['cross']) == pytest.approx(2.8 * lay['side'], abs=0.01)
        assert lay['fall_dir'] == pytest.approx((0.0, -float(lay['side'])), abs=1e-6)   # toward and across
        assert lay['height_m'] == PILLAR['height_m'] and lay['radius_m'] == PILLAR['radius_m']
        base, tip = pillar_footprint(lay)
        # the fallen trunk spans 2.8 m before the route and 3.2 m past it
        assert _signed_offset(base, lay['cross']) * lay['side'] == pytest.approx(2.8, abs=0.01)
        assert _signed_offset(tip, lay['cross']) * lay['side'] == pytest.approx(-3.2, abs=0.01)
        assert tip[0] == pytest.approx(9.0, abs=0.01)
        assert math.dist(base, tip) == pytest.approx(PILLAR['height_m'])


def test_pillar_layout_is_seeded_and_uses_both_sides():
    a = pillar_layout(STRAIGHT, (0.0, 0.0, 0.0), seed=11)
    assert a == pillar_layout(STRAIGHT, (0.0, 0.0, 0.0), seed=11)                  # deterministic
    assert {pillar_layout(STRAIGHT, (0.0, 0.0, 0.0), seed=s)['side'] for s in range(12)} == {1, -1}


def test_pillar_falls_where_the_rockslide_would_on_short_routes_and_bends():
    # same ROUTE_FRACTION / MIN_AHEAD_M rules as rockslide_layout
    assert pillar_layout(np.array([[0.0, 0.0], [10.0, 0.0]]), (0.0, 0.0, 0.0), seed=1)['cross'] == \
        pytest.approx((6.0, 0.0), abs=0.01)
    assert pillar_layout(np.array([[0.0, 0.0], [4.0, 0.0]]), (0.0, 0.0, 0.0), seed=1)['cross'] == \
        pytest.approx((3.0, 0.0), abs=0.01)
    bend = pillar_layout(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 20.0]]), (0.0, 0.0, 0.0), seed=2)
    assert bend['cross'] == pytest.approx((3.0, 5.0), abs=0.01)                     # tangent +y here
    assert abs(bend['base'][0] - 3.0) == pytest.approx(2.8, abs=0.01) and bend['base'][1] == pytest.approx(5.0)
    assert bend['fall_dir'][1] == pytest.approx(0.0, abs=1e-6)
    heading = pillar_layout(None, (2.0, 3.0, math.pi / 2), seed=3)                  # no route: along the heading
    assert heading['cross'] == pytest.approx((2.0, 11.0), abs=0.01)


def test_pillar_topples_in_about_one_and_a_half_seconds_at_2_2g():
    t_land = pillar_fall_time(6.0, G22, 4.0)
    assert 1.2 <= t_land <= 2.0
    assert pillar_fall_time(6.0, 9.81, 4.0) > t_land + 0.4                          # slower at 1 g
    assert pillar_fall_angle(0.0) == pytest.approx(math.radians(4.0))
    assert pillar_fall_angle(-1.0) == pytest.approx(math.radians(4.0))              # not released yet
    ts = np.linspace(0.0, 3.0, 601)
    th = np.array([pillar_fall_angle(t, 6.0, G22, 4.0) for t in ts])
    assert np.all(np.diff(th) >= 0.0)                                               # monotonic
    assert th[ts < t_land - 0.05].max() < PILLAR_REST_RAD
    assert np.all(th[ts >= t_land] == PILLAR_REST_RAD)                              # clamped where it lies
    assert pillar_fall_angle(0.5 * t_land) < math.radians(30.0)                     # slow start, fast finish
    assert pillar_fall_angle(1.234) == pillar_fall_angle(1.234)                     # deterministic


def test_pillar_fall_matches_the_rigid_rod_energy_balance():
    # θ'' = (3 g / 2L) sin θ from rest at θ0  ⇒  ω² = (3 g / L)(cos θ0 − cos θ); check the time to 45° by quadrature
    L, th0 = 6.0, math.radians(4.0)
    k = 3.0 * G22 / L
    target = math.radians(45.0)
    u = np.linspace(0.0, math.sqrt(target - th0), 20001)[1:]           # θ = θ0 + u²: a regular integrand
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz                    # numpy 2 renamed it
    t45 = float(trapezoid(2.0 * u / np.sqrt(k * (math.cos(th0) - np.cos(th0 + u * u))), u))
    t45 += u[0] * 2.0 / math.sqrt(k * math.sin(th0))                     # first interval: the integrand → 2/√(k sin θ0)
    assert pillar_fall_angle(t45, L, G22, 4.0) == pytest.approx(target, abs=math.radians(0.2))


def test_pillar_segments_are_overlapping_chunks_along_the_tilted_axis():
    lay = pillar_layout(STRAIGHT, (0.0, 0.0, 0.0), seed=4)
    r = PILLAR['radius_m'] * 1.15
    bx, by = lay['base']
    ux, uy = lay['fall_dir']
    for theta in (math.radians(4.0), 0.7, PILLAR_REST_RAD):
        segs = pillar_segments(lay, theta)
        assert len(segs) == 9 and all(s[3] == pytest.approx(r) for s in segs)
        c = np.array([s[:3] for s in segs])
        gaps = np.linalg.norm(np.diff(c, axis=0), axis=1)
        assert np.all(gaps < 2.0 * r) and np.allclose(gaps, gaps[0])                # even spacing, overlapping
        along = (c[:, 0] - bx) * ux + (c[:, 1] - by) * uy                           # horizontal reach
        side = -(c[:, 0] - bx) * uy + (c[:, 1] - by) * ux
        assert np.allclose(side, 0.0, atol=1e-9)                                    # tilted toward fall_dir only
        assert np.all(np.diff(along) >= 0.0) and np.all(c[:, 2] >= r - 1e-9)        # never below the ground
    stand = np.array(pillar_segments(lay, 0.0))
    assert stand[0, 2] == pytest.approx(r) and stand[-1, 2] == pytest.approx(PILLAR['height_m'] - r)
    lie = np.array(pillar_segments(lay, PILLAR_REST_RAD))
    assert np.allclose(lie[:, 2], r)                                                # lying flat on its side
    base, tip = pillar_footprint(lay)
    reach = np.hypot(lie[:, 0] - base[0], lie[:, 1] - base[1])
    assert reach.min() - r == pytest.approx(0.0, abs=1e-9)                          # chunks cover base → tip
    assert reach.max() + r == pytest.approx(PILLAR['height_m'], abs=1e-9)
    assert math.dist(lie[-1, :2], tip) == pytest.approx(r, abs=1e-9)


def test_pillar_drapes_over_uneven_ground_once_it_lies_down():
    lay = pillar_layout(STRAIGHT, (0.0, 0.0, 0.0), seed=4)
    r = PILLAR['radius_m'] * 1.15
    segs = pillar_segments(lay, PILLAR_REST_RAD)
    ground = np.linspace(0.0, 1.2, len(segs))                                       # ground rises under the trunk
    c = pillar_drape(segs, PILLAR_REST_RAD, base_z=0.0, ground_z=ground)
    assert c.shape == (9, 3) and np.allclose(c[:, 2], ground + r)
    upright = pillar_segments(lay, 0.0)
    cu = pillar_drape(upright, 0.0, base_z=2.0, ground_z=np.full(9, 2.0))
    assert np.allclose(cu[:, 2], 2.0 + np.array([s[2] for s in upright]))


def test_pillar_fall_moves_its_chunks_until_it_lands():
    lay = pillar_layout(STRAIGHT, (0.0, 0.0, 0.0), seed=5)
    fall = PillarFall(lay, t_release=10.0, ground_at=lambda x, y: 0.5 + 0.0 * np.asarray(x), g=G22)
    c0, radii = fall.centres(10.0)
    assert c0.shape == (9, 3) and radii == pytest.approx(PILLAR['radius_m'] * 1.15)
    assert c0[-1, 2] > 5.0 and not fall.landed                                       # still standing
    mid, _ = fall.centres(10.0 + 0.8 * fall.t_land)
    assert 0.5 < mid[-1, 2] < c0[-1, 2] and not fall.landed
    end, _ = fall.centres(10.0 + fall.t_land + 0.01)
    assert fall.landed and np.allclose(end[:, 2], 0.5 + PILLAR['radius_m'] * 1.15)
    base, tip = pillar_footprint(lay)
    assert math.dist(end[-1, :2], tip) == pytest.approx(PILLAR['radius_m'] * 1.15, abs=1e-6)
