import math

import numpy as np
import pytest

from exoswarm_autonomy.grid_map import (CAUTION, FREE, HAZARD, OBSTACLE, PERMANENT, SCIENCE, UNKNOWN,
                                        GlobalMap)
from exoswarm_autonomy.physics_model import RoverPhysics


def lim(g):
    return RoverPhysics().limits(g)


def patch(x0, x1, y0, y1, z_fn, step=0.1):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.stack([xs.ravel(), ys.ravel(), z_fn(xs.ravel(), ys.ravel())], axis=1)


def test_starts_unknown():
    m = GlobalMap(20.0, 0.5)
    assert m.n == 40
    assert m.percent_known() == 0.0
    assert (m.cls == UNKNOWN).all()


def test_flat_patch_marks_known_free():
    m = GlobalMap(20.0, 0.5)
    m.integrate(patch(-2, 2, -2, 2, lambda x, y: np.zeros_like(x)), t=1.0)
    m.recompute(lim(2.2))
    i, j = m.cell_of(0.0, 0.0)
    assert m.known[i, j] and m.cls[i, j] == FREE
    assert 0 < m.percent_known() < 10


def test_ramp_hazard_depends_on_gravity():
    ramp = patch(-4, 4, -4, 4, lambda x, y: np.tan(math.radians(25)) * x)
    m = GlobalMap(20.0, 0.5)
    m.integrate(ramp, t=0.0)
    m.recompute(lim(2.2))
    i, j = m.cell_of(0.0, 0.0)
    assert m.cls[i, j] == HAZARD and m.lethal[i, j]
    m.recompute(lim(1.0))
    assert m.cls[i, j] in (FREE, CAUTION) and not m.lethal[i, j]


def test_box_is_obstacle_via_step_height():
    m = GlobalMap(20.0, 0.5)
    ground = patch(-4, 4, -4, 4, lambda x, y: np.zeros_like(x))
    box = patch(1.0, 1.5, 1.0, 1.5, lambda x, y: np.full_like(x, 0.8))
    m.integrate(np.vstack([ground, box]), t=0.0)
    m.integrate(box, t=0.1)
    m.recompute(lim(2.2))
    i, j = m.cell_of(1.25, 1.25)
    assert m.cls[i, j] == OBSTACLE and m.lethal[i, j]


def test_new_obstacle_on_known_free_cell_is_detected():
    m = GlobalMap(20.0, 0.5)
    m.integrate(patch(-4, 4, -4, 4, lambda x, y: np.zeros_like(x)), t=0.0)
    m.recompute(lim(2.2))
    i, j = m.cell_of(2.2, 2.2)
    assert m.cls[i, j] == FREE
    rock = patch(2.0, 2.5, 2.0, 2.5, lambda x, y: np.full_like(x, 0.9))
    m.integrate(rock, t=1.0)
    m.integrate(rock, t=1.1)
    m.recompute(lim(2.2))
    assert m.cls[i, j] == OBSTACLE


def test_organized_scan_fill_between_rings():
    m = GlobalMap(40.0, 0.5)
    az = np.radians(np.arange(-30, 31, 2.0))
    ranges = np.array([4.0, 8.0, 12.0])
    pts = np.full((3, len(az), 3), np.nan)
    for v, r in enumerate(ranges):
        pts[v, :, 0] = r * np.cos(az)
        pts[v, :, 1] = r * np.sin(az)
        pts[v, :, 2] = 0.0
    m.integrate_organized(pts, sensor_xyz=(0.0, 0.0, 1.0), t=0.0)
    i, j = m.cell_of(6.0, 0.0)          # between ring 0 and ring 1 -> filled
    assert m.known[i, j]
    i, j = m.cell_of(16.0, 0.0)         # beyond the last ring -> unknown
    assert not m.known[i, j]


def test_min_range_filter_drops_dust_backscatter():
    m = GlobalMap(20.0, 0.5)
    near = np.array([[0.6, 0.0, 0.9], [1.0, 0.2, 0.7]])
    m.integrate(near, t=0.0, sensor_xyz=(0.0, 0.0, 1.0), min_range=2.0)
    assert m.percent_known() == 0.0


def test_permanent_hazard_persists():
    m = GlobalMap(20.0, 0.5)
    flat = patch(-4, 4, -4, 4, lambda x, y: np.zeros_like(x))
    m.integrate(flat, t=0.0)
    m.mark_permanent((0.0, 0.0), 1.5)
    m.integrate(flat, t=1.0)
    m.recompute(lim(2.2))
    i, j = m.cell_of(0.0, 0.0)
    assert m.cls[i, j] == PERMANENT and m.lethal[i, j]
    assert m.stats()['permanent_hazards'] == [[0.0, 0.0, 1.5]]


def test_science_marking_and_stats():
    m = GlobalMap(20.0, 0.5)
    m.integrate(patch(-4, 4, -4, 4, lambda x, y: np.zeros_like(x)), t=0.0)
    m.mark_science((1.0, -1.0))
    m.mark_science((1.1, -1.05))   # same target, deduplicated
    m.recompute(lim(2.2))
    assert m.cls[m.cell_of(1.0, -1.0)] == SCIENCE
    st = m.stats()
    assert len(st['science']) == 1
    assert set(st) >= {'percent', 'known_cells', 'total_cells', 'hazard_cells', 'science',
                       'permanent_hazards', 'elev_min', 'elev_max'}


def test_layers_rgb_and_occupancy_encoding():
    m = GlobalMap(20.0, 0.5)
    m.integrate(patch(-4, 4, -4, 4, lambda x, y: 0.1 * x), t=10.0)
    m.recompute(lim(2.2))
    img = m.layers_rgb(t=10.0)
    assert img.shape == (40, 40, 3) and img.dtype == np.uint8
    i, j = m.cell_of(0.0, 0.0)
    assert img[i, j, 1] == FREE * 40
    assert img[i, j, 2] == 255                      # just revealed
    assert m.layers_rgb(t=20.0)[i, j, 2] == 0       # glow decays after 6 s
    assert img[0, 0, 1] == 0 and img[0, 0, 0] == 0  # unknown corner
    occ = m.to_occupancy()
    assert occ[0, 0] == -1 and occ[i, j] == 0


def test_far_range_pose_error_between_scans_is_not_an_obstacle():
    # two scans of the same flat ground 18-22 m away whose heights disagree by 0.32 m (≈1° pitch error)
    m = GlobalMap(60.0, 0.5)
    far = patch(18, 22, -2, 2, lambda x, y: np.zeros_like(x))
    m.integrate(far, t=0.0, sensor_xyz=(0.0, 0.0, 0.9))
    m.integrate(far + np.array([0.0, 0.0, 0.32]), t=0.1, sensor_xyz=(0.0, 0.0, 0.9))
    m.integrate(far + np.array([0.0, 0.0, 0.32]), t=0.2, sensor_xyz=(0.0, 0.0, 0.9))
    m.recompute(lim(2.2))
    assert not (m.cls == OBSTACLE).any()
    # ...while the same step seen at close range is an obstacle
    near = patch(3, 4, -0.5, 0.5, lambda x, y: np.zeros_like(x))
    m.integrate(near, t=0.3, sensor_xyz=(0.0, 0.0, 0.9))
    m.integrate(near + np.array([0.0, 0.0, 0.5]), t=0.4, sensor_xyz=(0.0, 0.0, 0.9))
    m.integrate(near + np.array([0.0, 0.0, 0.5]), t=0.5, sensor_xyz=(0.0, 0.0, 0.9))
    m.recompute(lim(2.2))
    i, j = m.cell_of(3.5, 0.0)
    assert m.cls[i, j] == OBSTACLE
