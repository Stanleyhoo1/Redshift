import math

import numpy as np
import pytest

from redshift_sim.scene_events import JOLT, jolt_cut_angle_deg, jolt_wrench_world, rockslide_layout


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


def test_jolt_profile():
    # tuned: cut 21.4 deg -> ~33.8 deg peak, 5 deg past the 2.2 g tip alarm (28.8 deg)
    assert jolt_cut_angle_deg(28.8) == pytest.approx(21.4, abs=0.1)
    f, t = jolt_wrench_world(0.0, 2.2)
    assert t[0] == pytest.approx(JOLT['torque_nm_at_2_2g'])
    _, t1 = jolt_wrench_world(math.pi / 2, 1.0)
    assert t1[1] == pytest.approx(JOLT['torque_nm_at_2_2g'] / 2.2) and abs(t1[0]) < 1e-9
