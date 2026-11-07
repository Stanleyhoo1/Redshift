import numpy as np
import pytest

from redshift_sim.mock_render import Scene, camera_frame, camera_rotation, lidar_scan, robot_body, rpy_matrix
from redshift_sim.render_pool import RenderPool, scene_objects
from redshift_sim.robot_models import ROVER_SPEC


def small_scene():
    yy, xx = np.mgrid[0:65, 0:65]
    h = 0.6 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    sc = Scene(h, 32.0, rocks=[(3.0, 0.5, 0.6, 0.7, 0.5, 0.4, 0.3, 0.42)], crystals=[(5.0, -1.0)])
    parts = robot_body('rover_1')
    ids = sc.add_body(parts)
    sc.set_body_pose(ids, parts, (4.0, 1.5, 0.5), rpy_matrix(0.0, 0.0, 0.4), 0.4)
    sc.add_rocks([(6.0, -2.0, 0.6)], t=0.0, drop_m=4.0)
    sc.update(0.5)                                         # a rockslide boulder mid-fall
    return sc


def views():
    R = rpy_matrix(0.0, 0.0, 0.1)
    o = np.array([0.0, 0.0, 1.0])
    return o, R, camera_rotation(R, ROVER_SPEC['camera_pitch_rad'])


@pytest.mark.parametrize('procs', [0, 2])
def test_pool_renders_exactly_what_the_scene_renders(procs):
    sc = small_scene()
    o, R, Rc = views()
    want_rgb, want_depth = camera_frame(sc, o, Rc, 160, 120, 1.396, 40.0, scale=0.5)
    want_pts = lidar_scan(sc, o, R, ROVER_SPEC)
    pool = RenderPool(sc, procs)
    try:
        assert pool.procs == procs
        objs = scene_objects(sc)
        rgb, depth = pool.camera_frame(objs, o, Rc, 160, 120, 1.396, 40.0, scale=0.5)
        pts = pool.lidar_scan(objs, o, R, ROVER_SPEC)
    finally:
        pool.close()
    np.testing.assert_array_equal(rgb, want_rgb)
    np.testing.assert_array_equal(depth, want_depth)
    np.testing.assert_array_equal(pts, want_pts)


def test_pool_sees_objects_added_after_it_started():
    sc = small_scene()
    o, R, Rc = views()
    pool = RenderPool(sc, 1)
    try:
        before = pool.lidar_scan(scene_objects(sc), o, R, ROVER_SPEC)
        sc.add_rocks([(2.5, 0.0, 0.9)], t=0.0, drop_m=0.0)  # a boulder right in front of the sensor
        after = pool.lidar_scan(scene_objects(sc), o, R, ROVER_SPEC)
    finally:
        pool.close()
    np.testing.assert_array_equal(after, lidar_scan(sc, o, R, ROVER_SPEC))
    assert not np.array_equal(before, after)


def test_pool_excludes_a_sensors_own_body():
    sc = small_scene()
    parts = robot_body('rover_0')
    own = sc.add_body(parts)
    o, R, Rc = views()
    sc.set_body_pose(own, parts, o - np.array([0.0, 0.0, 0.5]), R, 0.1)
    pool = RenderPool(sc, 1)
    try:
        pts = pool.lidar_scan(scene_objects(sc), o, R, ROVER_SPEC, exclude=frozenset(own))
    finally:
        pool.close()
    np.testing.assert_array_equal(pts, lidar_scan(sc, o, R, ROVER_SPEC, exclude=frozenset(own)))
