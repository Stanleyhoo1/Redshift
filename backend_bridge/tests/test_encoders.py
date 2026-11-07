import pytest
import base64
import math

import cv2
import numpy as np

from backend_bridge.encoders import (RobotCache, RtfMeter, Trail, decode_clock_cdr, downsample_path, lidar_points_body, png_b64,
                                     robot_telemetry)

ROBOT_KEYS = {'type', 'pose', 'speed', 'state', 'mode', 'goal', 'v_limit', 'slip', 'tip_alarm_deg', 'max_slope_deg',
              'altitude_agl', 'imu', 'uncertainty', 'edge', 'path', 'trail', 'last_safe'}


def test_png_roundtrip_keeps_rgb_channels_and_row_order():
    img = np.zeros((4, 6, 3), np.uint8)
    img[0, 0] = (200, 40, 255)          # row 0 = south-west corner
    dec = cv2.imdecode(np.frombuffer(base64.b64decode(png_b64(img)), np.uint8), cv2.IMREAD_COLOR)
    assert tuple(dec[0, 0][::-1]) == (200, 40, 255)


def test_lidar_points_body_frame_cap_and_nan_drop():
    grid = np.random.default_rng(0).uniform(-10, 10, (16, 360, 3)).astype(np.float32)
    grid[0, :10] = np.nan
    pts = lidar_points_body(grid, (0.25, 0.0, 0.62), max_points=1500)
    assert len(pts) <= 1500 and all(len(p) == 3 and all(math.isfinite(v) for v in p) for p in pts)
    one = lidar_points_body(np.array([[1.0, 2.0, 3.0]]), (0.25, 0.0, 0.62))
    assert one == [[1.25, 2.0, 3.62]]


def test_downsample_path_keeps_endpoints():
    p = [(float(i), 0.0) for i in range(1000)]
    d = downsample_path(p, 150)
    assert len(d) <= 150 and d[0] == [0.0, 0.0] and d[-1] == [999.0, 0.0]
    assert downsample_path([], 150) == []


def test_trail_spacing_and_cap():
    t = Trail(spacing_m=0.5, cap=10)
    for i in range(100):
        t.add(i * 0.1, 0.0)
    pts = t.points()
    assert len(pts) == 10 and all(abs((b[0] - a[0]) - 0.5) < 1e-6 for a, b in zip(pts, pts[1:]))


def test_robot_telemetry_complete_with_partial_cache():
    c = RobotCache('rover_0')
    t = robot_telemetry(c)
    assert set(t) >= ROBOT_KEYS and t['type'] == 'rover' and t['state'] == 'BOOT'
    c.pose = {'x': 1.0, 'y': 2.0, 'z': 0.3, 'roll': 0.1, 'pitch': -0.05, 'yaw': 1.0}
    c.agent_state = {'state': 'EXPLORING', 'mode': 'FUSED', 'goal': [5, 6], 'speed': 0.7, 'v_limit': 0.83,
                     'slip': 0.02, 'tip_alarm_deg': 28.9, 'max_slope_deg': 16.9, 'altitude_agl': None,
                     'last_safe': [0.5, 1.0]}
    c.edge = {'robot': 'rover_0', 'load_pct': 50.0}
    c.detections = {'uncertainty': {'score': 0.12, 'camera_conf': 0.95, 'lidar_conf': 0.97, 'imu_vib': 0.1}}
    c.path = [[1.0, 2.0], [5.0, 6.0]]
    t = robot_telemetry(c)
    assert t['state'] == 'EXPLORING' and t['goal'] == [5, 6] and t['uncertainty']['score'] == 0.12
    assert t['imu']['roll_deg'] == round(math.degrees(0.1), 2) and t['edge']['load_pct'] == 50.0
    s = robot_telemetry(RobotCache('scout_0'))
    assert s['type'] == 'scout' and s['slip'] == 0.0


def test_decode_clock_cdr():
    import struct
    raw = b'\x00\x01\x00\x00' + struct.pack('<iI', 12, 500_000_000)       # CDR LE header + builtin Time
    assert decode_clock_cdr(raw) == 12.5
    assert decode_clock_cdr(b'\x00') is None
    try:
        from rclpy.serialization import serialize_message
        from rosgraph_msgs.msg import Clock
        from builtin_interfaces.msg import Time
    except ImportError:
        return
    assert decode_clock_cdr(serialize_message(Clock(clock=Time(sec=7, nanosec=250_000_000)))) == 7.25


def test_rtf_meter_tracks_speed_resets_and_stalls():
    m = RtfMeter(window_s=2.0)
    assert m.rtf(0.0) is None
    for k in range(300):                       # 3 s wall at 200 Hz, sim at 2x
        m.add(k * 0.01, k * 0.02)
    assert m.rtf(2.99) == pytest.approx(2.0, abs=0.02) and m.sim_t == pytest.approx(5.98)
    m.add(3.0, 0.0)                            # new world: clock restarted
    assert m.sim_t == 0.0 and m.rtf(3.0) is None
    for k in range(1, 100):
        m.add(3.0 + k * 0.01, k * 0.005)       # 0.5x
    assert m.rtf(3.99) == pytest.approx(0.5, abs=0.02)
    assert m.rtf(6.0) == 0.0                   # no clock for > 1 s: simulation stalled / paused
