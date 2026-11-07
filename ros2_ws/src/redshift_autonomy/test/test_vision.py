import math

import cv2
import numpy as np
import pytest

from redshift_autonomy.vision import (CameraModel, ImageQualityBaseline, LidarBaseline, annotate,
                                      camera_points_body, detect_objects, fuse_uncertainty, image_quality,
                                      lidar_corridor, lidar_quality)

W, H = 160, 120
CAM = CameraModel.from_hfov(W, H, math.radians(80))
PITCH = math.radians(12)
CAM_H = 1.0
RUST = np.array([150, 82, 52], np.float64)


def rays():
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xo = (u - CAM.cx) / CAM.fx
    yo = (v - CAM.cy) / CAM.fy
    # optical (x right, y down, z fwd) -> body-level (x fwd, y left, z up), pitched down
    xf, yl, zu = np.ones_like(xo), -xo, -yo
    c, s = math.cos(PITCH), math.sin(PITCH)
    return xf * c + zu * s, yl, -xf * s + zu * c          # direction per unit optical-z


def render(rock=None, crater=None, science=None, seed=0):
    """Returns (rgb uint8, planar depth float32)."""
    dx, dy, dz = rays()
    t = np.where(dz < -1e-6, CAM_H / -dz, np.inf)          # ground z = -CAM_H relative to camera
    gx, gy = dx * t, dy * t
    rng = np.random.default_rng(seed)
    tex = rng.normal(0, 6, (H, W, 1))
    dist = np.where(np.isfinite(t), np.hypot(gx, gy), 60.0)[..., None]
    fall = np.clip(1.15 - dist / 40.0, 0.45, 1.15)               # nearer ground brighter
    sky_v = (np.arange(H)[:, None, None] / H) * np.ones((H, W, 1))
    sky = np.array([90, 60, 50]) + sky_v * np.array([120, 80, 50])
    img = np.where(np.isfinite(t)[..., None], np.clip(RUST * fall + tex, 0, 255), sky)
    if crater is not None:                                 # bowl 0.8 m deep
        cx0, cx1, hw = crater
        t2 = np.where(dz < -1e-6, (CAM_H + 0.8) / -dz, np.inf)
        x2, y2 = dx * t2, dy * t2
        inside = (x2 > cx0) & (x2 < cx1) & (np.abs(y2) < hw)
        t = np.where(inside, t2, t)
        img[inside] = img[inside] * 0.55                   # shadowed bowl
    if rock is not None:                                   # vertical face at x = rx
        rx, half_w, hgt = rock
        tr = rx / np.maximum(dx, 1e-6)
        ry, rz = dy * tr, dz * tr + CAM_H
        hit = (tr < t) & (np.abs(ry) < half_w) & (rz > 0) & (rz < hgt)
        t = np.where(hit, tr, t)
        shade = np.clip(0.6 + 0.6 * rz[hit] / hgt, 0.6, 1.2)[:, None]   # lit top, darker base
        img[hit] = np.array([108, 106, 104]) * shade + rng.normal(0, 4, (int(hit.sum()), 3))
    if science is not None:
        sx, sy, sr = science
        hit = (np.hypot(gx - sx, gy - sy) < sr) & np.isfinite(t)
        img[hit] = [40, 205, 215]
    depth = np.where(np.isfinite(t), t, np.nan).astype(np.float32)
    depth[depth > 30] = np.nan
    return img.astype(np.uint8), depth


def classes(dets):
    return [d.cls for d in dets]


def test_camera_points_body_ground_is_flat():
    _, depth = render()
    pts = camera_points_body(depth, CAM, PITCH, CAM_H, stride=4)
    assert pts.shape[1] == 3
    assert np.nanmax(np.abs(pts[:, 2])) < 0.05             # z relative to ground ~ 0


def test_detects_rock():
    rgb, depth = render(rock=(5.0, 0.4, 0.8))
    dets = detect_objects(rgb, depth, CAM, PITCH, CAM_H)
    rocks = [d for d in dets if d.cls == 'ROCK']
    assert rocks and rocks[0].conf > 0.6
    assert rocks[0].range_m == pytest.approx(5.0, abs=0.6)
    assert abs(rocks[0].bearing_rad) < 0.15


def test_detects_science_target():
    rgb, depth = render(science=(4.0, -0.8, 0.35))
    dets = detect_objects(rgb, depth, CAM, PITCH, CAM_H)
    sci = [d for d in dets if d.cls == 'SCIENCE']
    assert sci and sci[0].bearing_rad < 0                  # target is to the right
    assert 'ROCK' not in classes(dets)


def test_detects_crater():
    rgb, depth = render(crater=(6.0, 9.0, 1.5))
    dets = detect_objects(rgb, depth, CAM, PITCH, CAM_H)
    assert 'CRATER' in classes(dets)


def test_clear_ground_has_no_detections():
    rgb, depth = render()
    assert detect_objects(rgb, depth, CAM, PITCH, CAM_H) == []


def test_world_xy_uses_pose():
    rgb, depth = render(rock=(5.0, 0.4, 0.8))
    d = [x for x in detect_objects(rgb, depth, CAM, PITCH, CAM_H, pose=(10.0, 2.0, math.pi / 2)) if x.cls == 'ROCK'][0]
    assert d.world_xy[0] == pytest.approx(10.0, abs=0.8)
    assert d.world_xy[1] == pytest.approx(7.0, abs=0.8)


def haze(rgb, level=0.85, seed=1):
    dust = np.array([170, 120, 80], np.float64)
    out = rgb.astype(np.float64) * (1 - 0.65 * level) + dust * 0.65 * level
    out = cv2.GaussianBlur(out, (0, 0), 1.0 + 3 * level)
    out += np.random.default_rng(seed).normal(0, 25 * level, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def test_dust_raises_uncertainty():
    rgb, _ = render(rock=(5.0, 0.4, 0.8), science=(4.0, -0.8, 0.35))
    base = ImageQualityBaseline()
    for _ in range(5):
        base.update(image_quality(rgb))
    clear_conf = base.confidence(image_quality(rgb))
    dusty_conf = base.confidence(image_quality(haze(rgb)))
    assert clear_conf > 0.85
    assert dusty_conf < 0.45
    u_clear = fuse_uncertainty(clear_conf, 0.95, 0.05, 'FUSED')
    u_dust = fuse_uncertainty(dusty_conf, 0.6, 0.3, 'FUSED')
    assert u_dust - u_clear >= 0.2
    assert 0.45 < u_dust < 0.8                             # caution band, not safe-hold
    assert fuse_uncertainty(dusty_conf, 0.6, 0.3, 'LIDAR_ONLY') < u_dust


def test_camera_baseline_follows_a_scene_change_only_with_clear_air():
    rich, _ = render(rock=(5.0, 0.4, 0.8), science=(4.0, -0.8, 0.35))
    rich = rich.copy()
    rich[100:160, 120:200] = 245                           # a teammate's white chassis in the first frames
    rich[120:130, 120:200] = (255, 120, 0)
    plain, _ = render()
    q_rich, q_plain = image_quality(rich), image_quality(plain)

    locked = ImageQualityBaseline()
    for _ in range(5):
        locked.update(q_rich)
    assert locked.confidence(q_plain) < 0.7                # a smoother scene looks like a degraded camera
    for _ in range(30):
        locked.update(q_plain)                             # no independent confirmation: never re-learned
    assert locked.confidence(q_plain) < 0.7

    base = ImageQualityBaseline()
    for _ in range(5):
        base.update(q_rich)
    for _ in range(20):
        base.update(q_plain, clear_air=True)               # LiDAR says the air is clear: the scene changed
    assert base.confidence(q_plain) > 0.85

    dusty = ImageQualityBaseline()
    for _ in range(5):
        dusty.update(q_plain)
    q_dust = image_quality(haze(plain))
    for _ in range(30):
        dusty.update(q_dust, clear_air=True)               # added pixel noise is the optics, never learned
    assert dusty.confidence(q_dust) < 0.45


def organized_scan(obstacle_x=None, noise=0.0, dropout=0.0, backscatter=0.0, seed=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        return _organized_scan(obstacle_x, noise, dropout, backscatter, seed)


def _organized_scan(obstacle_x, noise, dropout, backscatter, seed):
    rng = np.random.default_rng(seed)
    elev = np.radians(np.linspace(-15, 10, 16))
    az = np.radians(np.arange(-180, 180, 1.0))
    E, A = np.meshgrid(elev, az, indexing='ij')
    dx, dy, dz = np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)
    t = np.where(dz < -1e-6, 1.0 / -dz, np.inf)           # sensor 1 m above flat ground
    if obstacle_x is not None:
        to = obstacle_x / np.maximum(dx, 1e-6)
        oy, oz = dy * to, dz * to + 1.0
        hit = (to < t) & (np.abs(oy) < 1.5) & (oz < 1.2) & (dx > 0)
        t = np.where(hit, to, t)
    t = np.where(t > 25, np.inf, t)
    t = t + rng.normal(0, noise, t.shape)
    drop = rng.random(t.shape) < dropout
    t = np.where(drop, np.inf, t)
    bs = rng.random(t.shape) < backscatter
    t = np.where(bs, rng.uniform(0.4, 2.0, t.shape), t)
    pts = np.stack([dx * t, dy * t, dz * t], axis=-1)
    pts[~np.isfinite(t)] = np.nan
    return pts


def test_lidar_quality_and_baseline():
    clear = organized_scan()
    base = LidarBaseline()
    for _ in range(3):
        base.update(lidar_quality(clear))
    assert base.confidence(lidar_quality(clear)) > 0.9
    dusty = organized_scan(noise=0.08, dropout=0.25, backscatter=0.02)
    c = base.confidence(lidar_quality(dusty))
    assert 0.3 < c < 0.85


def test_solid_obstacle_close_by_is_not_mistaken_for_dust_backscatter():
    base = LidarBaseline()
    for _ in range(3):
        base.update(lidar_quality(organized_scan()))
    rock = base.confidence(lidar_quality(organized_scan(obstacle_x=1.2)))   # boulder face 1.2 m away, ~100° wide
    assert rock > 0.85
    dusty = base.confidence(lidar_quality(organized_scan(backscatter=0.06)))  # same sensor, scattered near returns
    assert dusty < rock - 0.1


def test_lidar_corridor_finds_obstacle_and_slope():
    pts = organized_scan(obstacle_x=6.0).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)] + np.array([0, 0, 1.0])   # base frame: ground z = 0
    cor = lidar_corridor(pts)
    assert cor['obstacle_dist_m'] == pytest.approx(6.0, abs=0.4)
    assert abs(cor['slope_ahead_deg']) < 3.0 and cor['conf'] > 0.8
    clear = organized_scan().reshape(-1, 3)
    clear = clear[np.isfinite(clear).all(axis=1)] + np.array([0, 0, 1.0])
    assert math.isinf(lidar_corridor(clear)['obstacle_dist_m'])


def test_lidar_corridor_ignores_sparse_dust_backscatter():
    rng = np.random.default_rng(3)
    scan = organized_scan(noise=0.07, dropout=0.2).reshape(-1, 3)
    scan = scan[np.isfinite(scan).all(axis=1)] + np.array([0, 0, 1.0])
    # ~2 % of a 16x360 scan as backscatter: random returns 0.3-3 m from the sensor at sensor height ±0.5 m
    def dust(n=115):
        r, a = rng.uniform(0.4, 2.0, n), rng.uniform(-math.pi, math.pi, n)
        return np.c_[r * np.cos(a), r * np.sin(a), 1.0 + rng.uniform(-0.5, 0.2, n)]

    prev = None
    for _ in range(10):      # consecutive scans, independently re-drawn dust each time
        cor = lidar_corridor(np.vstack([scan, dust()]), prev_cells=prev)
        if prev is not None:
            assert math.isinf(cor['obstacle_dist_m'])
        prev = cor['cells']
    # a real boulder persists across scans and is still found through the dust
    rock = organized_scan(obstacle_x=2.0).reshape(-1, 3)
    rock = rock[np.isfinite(rock).all(axis=1)] + np.array([0, 0, 1.0])
    c1 = lidar_corridor(np.vstack([rock, dust()]))
    c2 = lidar_corridor(np.vstack([rock, dust()]), prev_cells=c1['cells'])
    assert c2['obstacle_dist_m'] == pytest.approx(2.0, abs=0.4)


def test_annotate_shape_and_modes():
    rgb, depth = render(rock=(5.0, 0.4, 0.8))
    dets = detect_objects(rgb, depth, CAM, PITCH, CAM_H)
    info = {'robot': 'rover_0', 'mode': 'FUSED', 'uncertainty': 0.12, 'fps': 9.8, 'slope_ahead_deg': 4.0}
    out = annotate(rgb, dets, info)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    out2 = annotate(rgb, dets, dict(info, mode='LIDAR_ONLY'))
    assert not np.array_equal(out, out2)


def test_detect_objects_class_subset_and_highpass_nadir():
    # nadir view of gently undulating ground with one boulder: only ROCK/SCIENCE requested
    cam = CameraModel.from_hfov(80, 60, math.pi / 2)
    yy, xx = np.mgrid[0:60, 0:80]
    depth = (12.0 + 0.6 * np.sin(xx / 9.0) + 0.4 * np.cos(yy / 7.0)).astype(np.float32)
    depth[25:33, 35:44] -= 0.9                       # boulder top 0.9 m above the ground
    rgb = np.full((60, 80, 3), (150, 80, 50), np.uint8)
    dets = detect_objects(rgb, depth, cam, math.pi / 2, 12.0, classes=('ROCK', 'SCIENCE'), rock_h=0.35,
                          tophat_px=6, max_range=40.0)
    assert {d.cls for d in dets} == {'ROCK'}
    assert detect_objects(rgb, depth, cam, math.pi / 2, 12.0, classes=()) == []


# ---------------------------------------------------------------------- teammate perception
from redshift_autonomy.vision import CameraModel as _CM, depth_colormap, project_box, teammate_detection  # noqa: E402


def _cam():
    return _CM.from_hfov(320, 240, 1.396)


def test_project_box_geometry():
    cam = _cam()
    bb, xf = project_box(cam, 0.0, (5.0, 0.0, 0.0), 0.0)
    x, y, w, h = bb
    assert abs((x + w / 2) - cam.cx) < 2 and abs((y + h / 2) - cam.cy) < 2 and xf == pytest.approx(5.0)
    left, _ = project_box(cam, 0.0, (5.0, 1.5, 0.0), 0.0)
    assert left[0] + left[2] / 2 < cam.cx - 20                      # +y (left) → smaller u
    near, _ = project_box(cam, 0.0, (2.5, 0.0, 0.0), 0.0)
    assert near[2] > w * 1.6                                        # closer → bigger
    tilted, _ = project_box(cam, math.radians(12), (5.0, 0.0, 0.0), 0.0)
    assert tilted[1] + tilted[3] / 2 < cam.cy                        # camera pitched down: level target appears higher
    assert project_box(cam, 0.0, (-3.0, 0.0, 0.0), 0.0) is None     # behind the camera
    nadir, _ = project_box(cam, math.pi / 2, (2.0, 0.0, -12.0), 0.0)
    assert nadir[1] + nadir[3] / 2 < cam.cy                          # nadir: ahead appears toward the top


def _depth_with_target(cam, rel, far=30.0):
    depth = np.full((cam.height, cam.width), far, np.float32)
    (x, y, w, h), xf = project_box(cam, 0.0, rel, 0.0)
    depth[y:y + h, x:x + w] = xf - 0.3
    return depth


def test_teammate_detected_only_when_depth_confirms():
    cam = _cam()
    rel = (4.0, 0.3, -0.2)
    depth = _depth_with_target(cam, rel)
    d = teammate_detection(cam, depth, 0.0, rel, 0.0, rel_base_xy=(4.45, 0.3), robot='rover_0', damaged=True,
                           label='PERCEIVED: DAMAGED UNIT ROVER_1 | DOCKING FOR REPAIR')
    assert d is not None and d.cls == 'ROVER' and d.robot == 'rover_0' and d.damaged
    assert 'DEPTH' in d.sources and 'CAM' in d.sources and d.conf > 0.75
    assert d.range_m == pytest.approx(4.46, abs=0.6)
    dd = d.as_dict()
    assert dd['label'].startswith('PERCEIVED') and dd['robot'] == 'rover_0'
    # occluded (a boulder in front fills the box at 1.5 m): not perceived
    occl = depth.copy()
    (x, y, w, h), _ = project_box(cam, 0.0, rel, 0.0)
    occl[y:y + h, x:x + w] = 1.5
    assert teammate_detection(cam, occl, 0.0, rel, 0.0, rel_base_xy=(4.45, 0.3), robot='rover_0') is None


def test_teammate_lidar_confirmation_and_range():
    cam = _cam()
    rel = (3.0, 0.0, -0.2)
    blank = np.full((cam.height, cam.width), np.nan, np.float32)      # camera blinded (dust)
    rng = np.random.default_rng(0)
    pts = np.column_stack([3.15 + 0.1 * rng.standard_normal(40), 0.3 * rng.standard_normal(40),
                           0.3 + 0.1 * rng.random(40)])
    d = teammate_detection(cam, blank, 0.0, rel, 0.0, rel_base_xy=(3.45, 0.0), lidar_pts=pts, robot='rover_0')
    assert d is not None and d.sources == ['LIDAR']
    assert d.range_m == pytest.approx(3.45, abs=0.35)
    assert teammate_detection(cam, blank, 0.0, rel, 0.0, rel_base_xy=(3.45, 0.0), lidar_pts=pts[:3],
                              robot='rover_0') is None


def test_depth_colormap_near_is_warm_and_missing_is_black():
    depth = np.full((10, 20), np.nan, np.float32)
    depth[:, :10] = 1.0
    depth[:, 10:15] = 30.0
    img = depth_colormap(depth, far_m=30.0)
    assert img.shape == (10, 20, 3) and img.dtype == np.uint8
    near, far, none = img[5, 2], img[5, 12], img[5, 18]
    assert near[0] > near[2]                     # RGB: near → red
    assert far[2] >= far[0]                      # far → blue
    assert none.sum() == 0
