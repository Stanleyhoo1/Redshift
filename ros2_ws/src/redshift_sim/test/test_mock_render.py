import math
import time

import cv2
import numpy as np
import pytest

from redshift_sim.mock_render import (LIDAR_SHAPE, Scene, camera_frame, camera_intrinsics, camera_rotation,
                                      lidar_scan, rpy_matrix)
from redshift_sim.robot_models import ROVER_SPEC, SCOUT_SPEC
from redshift_sim.terrain_gen import generate_terrain


def flat_scene(size=60.0, n=121, **kw):
    return Scene(np.zeros((n, n), np.float32), size, **kw)


def test_lidar_layout_matches_gazebo_cloud():
    """Gazebo gpu_lidar → PointCloud2: rows = rings from −15° ascending, column 0 at azimuth −180°, ccw."""
    sc = flat_scene()
    pts = lidar_scan(sc, np.array([0.0, 0.0, 0.9]), np.eye(3), ROVER_SPEC)
    assert pts.shape == LIDAR_SHAPE + (3,) and pts.dtype == np.float32
    with np.errstate(invalid='ignore'):
        elev = np.degrees(np.arctan2(pts[..., 2], np.hypot(pts[..., 0], pts[..., 1])))
    assert np.nanmedian(elev[0]) == pytest.approx(-15.0, abs=0.2)
    assert np.all(np.diff(np.nanmedian(elev[:6], axis=1)) > 0)
    az = np.degrees(np.arctan2(pts[0, :, 1], pts[0, :, 0]))
    assert abs(abs(az[0]) - 180.0) < 0.5 and az[90] == pytest.approx(-90.0, abs=0.5)
    assert az[270] == pytest.approx(90.0, abs=0.5)
    assert not np.isfinite(pts[-1]).any()                # rays above the horizon: no return (±inf like Gazebo)


def test_lidar_ring_hits_flat_ground_at_expected_range():
    sc = flat_scene()
    pts = lidar_scan(sc, np.array([0.0, 0.0, 0.9]), np.eye(3), ROVER_SPEC)
    rng = np.linalg.norm(pts[0], axis=1)
    assert np.median(rng) == pytest.approx(0.9 / math.sin(math.radians(15)), abs=0.1)


def test_horizontal_ray_hits_wall_at_expected_range():
    h = np.zeros((121, 121), np.float32)
    h[:, 80:] = 4.0                                    # wall at x = (80 - 60) * 0.5 = 10 m (one-cell ramp)
    sc = Scene(h, 60.0)
    t, kind, _, _ = sc.raycast(np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]), 0.1, 30.0)
    assert kind[0] == Scene.GROUND
    assert t[0] == pytest.approx(9.62, abs=0.15)       # bilinear ramp between x=9.5 (z=0) and 10 (z=4) → z=1 at 9.625


def test_depth_is_planar_and_sky_is_inf():
    sc = flat_scene()
    w, h, hfov = 320, 240, ROVER_SPEC['camera_hfov']
    R = camera_rotation(np.eye(3), 0.0)
    rgb, depth = camera_frame(sc, np.array([0.0, 0.0, 1.0]), R, w, h, hfov, 40.0, scale=1.0)
    assert rgb.shape == (h, w, 3) and rgb.dtype == np.uint8 and depth.dtype == np.float32
    fx, cx, cy = camera_intrinsics(w, h, hfov)
    v = 200
    expect = 1.0 * fx / (v - cy)                        # planar depth of a ground pixel below the horizon
    assert float(np.median(depth[v, 100:220])) == pytest.approx(expect, rel=0.05)
    assert np.isposinf(depth[10]).all()               # sky row: +inf like the Gazebo depth camera
    assert np.isfinite(depth[-1]).all()


def test_camera_intrinsics_match_gazebo_camera_info():
    assert camera_intrinsics(320, 240, ROVER_SPEC['camera_hfov']) == pytest.approx((190.7, 160.0, 120.0), abs=0.1)
    assert camera_intrinsics(160, 120, SCOUT_SPEC['camera_hfov']) == pytest.approx((80.0, 80.0, 60.0), abs=0.1)


def test_rock_pixels_gray_and_occluding_crystal_pixels_cyan():
    sc = flat_scene(rocks=[(6.0, 0.0, 0.42, 0.7, 0.56, 0.6, 0.0, 0.42)], crystals=[(4.0, 1.2)])
    R = camera_rotation(rpy_matrix(0.0, 0.0, 0.0), ROVER_SPEC['camera_pitch_rad'])
    rgb, depth = camera_frame(sc, np.array([0.45, 0.0, 0.78]), R, 320, 240, ROVER_SPEC['camera_hfov'], 40.0)
    fx, cx, cy = camera_intrinsics(320, 240, ROVER_SPEC['camera_hfov'])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # project the rock centre into the image (camera link: x forward, u → −y, v → −z)
    pc = R.T @ (np.array([6.0, 0.0, 0.42]) - np.array([0.45, 0.0, 0.78]))
    ru, rv = int(round(cx - fx * pc[1] / pc[0])), int(round(cy - fx * pc[2] / pc[0]))
    rock = hsv[rv - 3:rv + 4, ru - 3:ru + 4].reshape(-1, 3)
    assert np.median(rock[:, 1]) < 75                  # low saturation: grey, like vision.detect_objects expects
    assert np.median(depth[rv - 3:rv + 4, ru - 3:ru + 4]) < 6.2
    ground = hsv[230, 20:300]
    assert np.median(ground[:, 1]) > 90                # rust regolith is saturated
    sci = (hsv[..., 0] >= 78) & (hsv[..., 0] <= 102) & (hsv[..., 1] > 110) & (hsv[..., 2] > 90)
    ys, xs = np.nonzero(sci)
    assert len(xs) > 15 and np.median(xs) < cx         # crystal at +y (left of the image centre)


def test_scout_nadir_camera_sees_ground_at_altitude():
    sc = flat_scene()
    R = camera_rotation(np.eye(3), SCOUT_SPEC['camera_pitch_rad'])
    _, depth = camera_frame(sc, np.array([0.0, 0.0, 12.0]), R, 160, 120, SCOUT_SPEC['camera_hfov'], 40.0)
    assert float(np.median(depth[50:70, 70:90])) == pytest.approx(12.0, abs=0.1)
    # camera below the surface (landed scout, lens under the skid line): nothing within the clip range
    _, d2 = camera_frame(sc, np.array([0.0, 0.0, -0.03]), R, 160, 120, SCOUT_SPEC['camera_hfov'], 40.0)
    assert np.isposinf(d2).all()


def test_dynamic_rocks_show_up_and_fall():
    sc = flat_scene()
    ids = sc.add_rocks([(8.0, 0.0, 0.8)], t=0.0, drop_m=4.0)
    assert len(ids) == 1
    o, d = np.array([[0.0, 0.0, 0.5]]), np.array([[1.0, 0.0, 0.0]])
    sc.update(0.1)                                     # still falling: well above the ray
    t, kind, _, _ = sc.raycast(o, d, 0.1, 30.0)
    assert kind[0] == Scene.NONE
    sc.update(2.0)                                     # landed
    t, kind, _, _ = sc.raycast(o, d, 0.1, 30.0)
    assert kind[0] == Scene.ROCK and t[0] == pytest.approx(7.2, abs=0.25)
    assert sc.blocked(7.5, 0.0, 0.5) and not sc.blocked(0.0, 0.0, 0.5)


def test_full_rover_and_scout_frames_are_fast_enough():
    terr = generate_terrain(100.0, 42)
    sc = Scene.from_terrain(terr)
    x, y = terr.spawn_xy[0]
    z = float(terr.height_at(x, y)) + 0.28
    Rb = rpy_matrix(0.0, 0.0, 0.7)
    Rc = camera_rotation(Rb, ROVER_SPEC['camera_pitch_rad'])
    for _ in range(2):                                 # warm-up
        camera_frame(sc, np.array([x, y, z + 0.5]), Rc, 320, 240, ROVER_SPEC['camera_hfov'], 40.0)
    t0 = time.perf_counter()
    rgb, depth = camera_frame(sc, np.array([x, y, z + 0.5]), Rc, 320, 240, ROVER_SPEC['camera_hfov'], 40.0)
    pts = lidar_scan(sc, np.array([x, y, z + 0.62]), Rb, ROVER_SPEC)
    dt_rover = time.perf_counter() - t0
    Rs = camera_rotation(rpy_matrix(0.0, 0.0, 0.0), SCOUT_SPEC['camera_pitch_rad'])
    t0 = time.perf_counter()
    camera_frame(sc, np.array([x, y, z + 12.0]), Rs, 160, 120, SCOUT_SPEC['camera_hfov'], 40.0)
    dt_scout = time.perf_counter() - t0
    assert np.isfinite(depth).mean() > 0.4 and np.isfinite(pts).all(axis=2).mean() > 0.3
    assert dt_rover < 0.08, f'rover camera + lidar took {dt_rover * 1000:.0f} ms'
    assert dt_scout < 0.04, f'scout camera took {dt_scout * 1000:.0f} ms'
