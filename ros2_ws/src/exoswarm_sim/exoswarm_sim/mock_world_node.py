"""Mock world: a drop-in replacement for Gazebo + ros_gz_bridge + world_interface_node.

Publishes exactly the raw topics the bridge produces (names, types, frame ids, encodings, rates, point-cloud
layout, no-return encoding; see scripts/check_topics.py), driven by mock_dynamics and rendered by mock_render
from the same seeded terrain.npz, and implements /world/command `spawn_rockslide` and `jolt` with the same
/world/events echoes. Physics and IMU run at 100 Hz in one callback group, sensor rendering in another, so
camera/LiDAR rendering never stalls the IMU stream.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from std_msgs.msg import String

from .mock_dynamics import RoverSim, ScoutSim
from .mock_render import Scene, camera_frame, camera_intrinsics, camera_rotation, lidar_scan, rpy_matrix
from .robot_models import ROVER_SPEC, SCOUT_SPEC
from .scene_events import rockslide_layout
from .terrain_gen import Terrain

PHYSICS_HZ = 100.0
ODOM_HZ = 30.0
WHEEL_ODOM_HZ = 20.0
LIDAR_HZ = 10.0
OUT_QOS = QoSProfile(depth=10)          # reliable, like the ros_gz bridge: compatible with every subscriber
CLOUD_DTYPE = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'],
                        'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'],
                        'offsets': [0, 4, 8, 16, 24], 'itemsize': 32})
CLOUD_FIELDS = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='ring', offset=24, datatype=PointField.UINT16, count=1)]


def quat(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class Robot:
    """Per-robot publishers, sim state and sensor schedule."""

    def __init__(self, node: 'MockWorld', name: str, spec: dict, spawn):
        self.name, self.spec = name, spec
        self.rover = name.startswith('rover')
        self.sim = (RoverSim(name, spec, spawn, node.g, np.random.default_rng(sum(name.encode())))
                    if self.rover else ScoutSim(name, spec, spawn, node.g))
        self.state = self.sim.snapshot()
        self.path = None
        self.cam_hz = float(spec.get('camera_hz', 10.0))
        self.fx, self.cx, self.cy = camera_intrinsics(spec['camera_w'], spec['camera_h'], spec['camera_hfov'])
        self.scale = node.render_scale if self.rover else 1.0
        n, r = node, name
        self.pub_odom = n.create_publisher(Odometry, f'/{r}/odom', OUT_QOS)
        self.pub_imu = n.create_publisher(Imu, f'/{r}/imu_raw', OUT_QOS)
        self.pub_img = n.create_publisher(Image, f'/{r}/camera/image_raw', OUT_QOS)
        self.pub_depth = n.create_publisher(Image, f'/{r}/camera/depth_raw', OUT_QOS)
        self.pub_info = n.create_publisher(CameraInfo, f'/{r}/camera/camera_info', OUT_QOS)
        if self.rover:
            self.pub_wheel = n.create_publisher(Odometry, f'/{r}/wheel_odom', OUT_QOS)
            self.pub_cloud = n.create_publisher(PointCloud2, f'/{r}/lidar/points_raw', OUT_QOS)
            n.create_subscription(PathMsg, f'/{r}/planned_path', self.on_path, 5, callback_group=n.cmd_group)
        n.create_subscription(Twist, f'/{r}/cmd_vel', self.on_cmd, qos_profile_sensor_data,
                              callback_group=n.physics_group)

    def on_cmd(self, m: Twist):
        if self.rover:
            self.sim.cmd = (m.linear.x, m.angular.z)
        else:
            self.sim.cmd = (m.linear.x, m.linear.y, m.linear.z, m.angular.z)

    def on_path(self, m: PathMsg):
        self.path = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses]).reshape(-1, 2)


class MockWorld(Node):
    def __init__(self):
        super().__init__('mock_world')
        for k, v in (('terrain_path', ''), ('robots', ['rover_0', 'scout_0']), ('g_mult', 2.2), ('spawn_json', '{}'),
                     ('specs_json', '{}'), ('render_scale', 0.5)):
            self.declare_parameter(k, v)
        gp = self.get_parameter
        self.g = float(gp('g_mult').value)
        self.render_scale = float(gp('render_scale').value)
        tp = Path(gp('terrain_path').value)
        terr = Terrain.load(tp)
        tex_path = tp.parent / 'regolith_diffuse.png'
        tex = None
        if tex_path.exists():
            tex = cv2.cvtColor(cv2.imread(str(tex_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.scene = Scene.from_terrain(terr, texture=tex, g=9.80665 * self.g)
        spawn = json.loads(gp('spawn_json').value or '{}')
        specs = json.loads(gp('specs_json').value or '{}')
        self.physics_group = MutuallyExclusiveCallbackGroup()
        self.cmd_group = self.physics_group               # commands mutate the scene: same thread as physics
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.lock = threading.Lock()                      # guards robot state snapshots
        self.robots: list[Robot] = []
        for r in gp('robots').value:
            spec = specs.get(r) or (ROVER_SPEC if r.startswith('rover') else SCOUT_SPEC)
            sp = spawn.get(r)
            if sp is None:
                x, y = terr.spawn_xy[len(self.robots) % len(terr.spawn_xy)] if r.startswith('rover') else terr.scout_spawn_xy
                sp = (x, y, float(terr.height_at(x, y)), 0.0)
            self.robots.append(Robot(self, r, spec, sp))
        self.events = self.create_publisher(String, '/world/events', 10)
        self.create_subscription(String, '/world/command', self.on_command, 10, callback_group=self.cmd_group)
        self.t0 = time.monotonic()
        self.sim_t = 0.0
        self._last = time.monotonic()
        self._due = {}                                    # (robot, sensor) → next sim time
        self._slide = 0
        self.create_timer(1.0 / PHYSICS_HZ, self.physics, callback_group=self.physics_group)
        self.create_timer(0.005, self.sensors, callback_group=self.sensor_group)
        self.get_logger().info(f'mock world ready: {len(terr.rocks)} boulders, {len(terr.science)} crystals, '
                               f'g={self.g}, robots={[r.name for r in self.robots]}, render_scale={self.render_scale}')

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def stamp(t: float) -> Time:
        sec = int(t)
        return Time(sec=sec, nanosec=int((t - sec) * 1e9))

    def emit(self, ev: dict):
        ev.setdefault('t', time.time())
        self.events.publish(String(data=json.dumps(ev)))

    def _due_now(self, key, period: float, t: float, phase: float = 0.0) -> bool:
        nxt = self._due.get(key)
        if nxt is None:
            self._due[key] = t + phase
            return False
        if t + 1e-6 >= nxt:
            self._due[key] = nxt + period if t - nxt < period else t + period   # no burst after a stall
            return True
        return False

    # ------------------------------------------------------------------ physics thread (100 Hz)
    def physics(self):
        now = time.monotonic()
        dt = min(0.05, max(1e-4, now - self._last))
        self._last = now
        self.sim_t = now - self.t0
        t = self.sim_t
        self.scene.update(t)
        stamp = self.stamp(t)
        for rb in self.robots:
            rb.sim.step(dt, self.scene, t)
            s = rb.sim.snapshot()
            with self.lock:
                rb.state = s
            for ev in (rb.sim.pop_events() if rb.rover else ()):
                self.get_logger().info(f'jolt {rb.name}: released at roll {ev["release_roll_deg"]:.1f} deg')
                self.emit(ev)
            self.publish_imu(rb, s, stamp)
            if self._due_now((rb.name, 'odom'), 1.0 / ODOM_HZ, t):
                self.publish_odom(rb, s, stamp)
            if rb.rover and self._due_now((rb.name, 'wheel'), 1.0 / WHEEL_ODOM_HZ, t):
                self.publish_wheel(rb, s, stamp)

    def publish_imu(self, rb: Robot, s: dict, stamp):
        m = Imu()
        m.header.stamp, m.header.frame_id = stamp, f'{rb.name}/base_link/imu'
        qx, qy, qz, qw = quat(s['roll'], s['pitch'], s['imu_yaw'])  # gz IMU: relative to the spawn heading
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = qx, qy, qz, qw
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(v) for v in s['gyro'])
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (float(v) for v in s['accel'])
        rb.pub_imu.publish(m)

    def publish_odom(self, rb: Robot, s: dict, stamp):
        m = Odometry()
        m.header.stamp, m.header.frame_id = stamp, 'world'
        m.child_frame_id = f'{rb.name}/base_link'
        p = m.pose.pose
        p.position.x, p.position.y, p.position.z = s['x'], s['y'], s['z']
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat(s['roll'], s['pitch'], s['yaw'])
        tw = m.twist.twist
        if rb.rover:
            tw.linear.x = s['v_true']
        else:
            tw.linear.x, tw.linear.y, tw.linear.z = (float(v) for v in s['v_body'])
        tw.angular.x, tw.angular.y, tw.angular.z = (float(v) for v in s['gyro'])
        rb.pub_odom.publish(m)

    def publish_wheel(self, rb: Robot, s: dict, stamp):
        m = Odometry()
        m.header.stamp, m.header.frame_id = stamp, f'{rb.name}/odom'
        m.child_frame_id = f'{rb.name}/base_link'
        wx, wy, wyaw = s['wheel_pose']
        m.pose.pose.position.x, m.pose.pose.position.y = wx, wy
        _, _, qz, qw = quat(0.0, 0.0, wyaw)
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = qz, qw
        m.twist.twist.linear.x = s['v_wheel']
        m.twist.twist.angular.z = s['w']
        rb.pub_wheel.publish(m)

    # ------------------------------------------------------------------ /world/command
    def on_command(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        op = cmd.get('op')
        rb = next((r for r in self.robots if r.name == cmd.get('robot', 'rover_0')), None)
        if op == 'spawn_rockslide':
            if rb is None or not rb.rover:
                self.emit({'op': 'spawn_rockslide', 'ok': False, 'robot': cmd.get('robot'), 'detail': 'unknown rover'})
                return
            s = rb.state
            rocks, impact = rockslide_layout(rb.path, (s['x'], s['y'], s['yaw']), seed=self._slide)
            self._slide += 1
            self.scene.add_rocks([tuple(r) for r in rocks], t=self.sim_t, drop_m=4.0)
            self.get_logger().info(f'rockslide #{self._slide} across {rb.name} path at {np.round(impact, 1)}')
            self.emit({'op': 'spawn_rockslide', 'ok': True, 'robot': rb.name, 'impact': [float(v) for v in impact],
                       'rocks': [[float(v) for v in r] for r in rocks]})
        elif op == 'jolt':
            if rb is None or not rb.rover:
                self.emit({'op': 'jolt', 'ok': False, 'robot': cmd.get('robot'), 'detail': 'unknown rover'})
                return
            rb.sim.start_jolt()

    # ------------------------------------------------------------------ sensor thread
    def sensors(self):
        t = time.monotonic() - self.t0
        for k, rb in enumerate(self.robots):
            if rb.rover and self._due_now((rb.name, 'lidar'), 1.0 / LIDAR_HZ, t, phase=0.05 * k):
                self.publish_lidar(rb)
            if self._due_now((rb.name, 'cam'), 1.0 / rb.cam_hz, t, phase=0.02 + 0.05 * k):
                self.publish_camera(rb)

    def _pose(self, rb: Robot):
        with self.lock:
            s = rb.state
        return s, np.array([s['x'], s['y'], s['z']]), rpy_matrix(s['roll'], s['pitch'], s['yaw'])

    def publish_lidar(self, rb: Robot):
        s, pos, R = self._pose(rb)
        stamp = self.stamp(time.monotonic() - self.t0)
        pts = lidar_scan(self.scene, pos + R @ np.asarray(rb.spec['lidar_xyz']), R, rb.spec)
        h, w = pts.shape[:2]
        arr = np.zeros(h * w, CLOUD_DTYPE)
        flat = pts.reshape(-1, 3)
        arr['x'], arr['y'], arr['z'] = flat[:, 0], flat[:, 1], flat[:, 2]
        arr['ring'] = np.repeat(np.arange(h, dtype=np.uint16), w)
        m = PointCloud2(height=h, width=w, fields=CLOUD_FIELDS, is_bigendian=False, point_step=32,
                        row_step=32 * w, is_dense=False)
        m.header.stamp, m.header.frame_id = stamp, f'{rb.name}/base_link/lidar'
        m.data = arr.tobytes()
        rb.pub_cloud.publish(m)

    def publish_camera(self, rb: Robot):
        s, pos, R = self._pose(rb)
        stamp = self.stamp(time.monotonic() - self.t0)
        sp = rb.spec
        Rc = camera_rotation(R, sp['camera_pitch_rad'])
        rgb, depth = camera_frame(self.scene, pos + R @ np.asarray(sp['camera_xyz']), Rc, sp['camera_w'],
                                  sp['camera_h'], sp['camera_hfov'], sp.get('camera_far', 40.0), scale=rb.scale)
        frame = f'{rb.name}/base_link/camera'
        h, w = depth.shape
        img = Image(height=h, width=w, encoding='rgb8', is_bigendian=0, step=3 * w, data=rgb.tobytes())
        img.header.stamp, img.header.frame_id = stamp, frame
        dep = Image(height=h, width=w, encoding='32FC1', is_bigendian=0, step=4 * w, data=depth.tobytes())
        dep.header.stamp, dep.header.frame_id = stamp, frame
        info = CameraInfo(width=w, height=h, distortion_model='plumb_bob', d=[0.0] * 5,
                          k=[rb.fx, 0.0, rb.cx, 0.0, rb.fx, rb.cy, 0.0, 0.0, 1.0],
                          r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                          p=[rb.fx, 0.0, rb.cx, 0.0, 0.0, rb.fx, rb.cy, 0.0, 0.0, 0.0, 1.0, 0.0])
        info.header.stamp, info.header.frame_id = stamp, frame
        rb.pub_img.publish(img)
        rb.pub_depth.publish(dep)
        rb.pub_info.publish(info)


def main():
    rclpy.init()
    node = MockWorld()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
