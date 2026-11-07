"""Mock world: a drop-in replacement for Gazebo + ros_gz_bridge + world_interface_node.

Publishes exactly the raw topics the bridge produces (names, types, frame ids, encodings, rates, point-cloud
layout, no-return encoding; see scripts/check_topics.py), driven by mock_dynamics and rendered by mock_render
from the same seeded terrain.npz, and implements /world/command `spawn_rockslide`, `jolt`, `set_speed`,
`critical_damage` and `repair` with the same /world/events echoes. Physics and IMU run at 100 Hz in one callback
group, sensor rendering in another, so camera/LiDAR rendering never stalls the IMU stream.

Simulated rover hardware (rover_hw, docs/protocol.md §8): the drivetrain fault is applied inside the dynamics and
`/<rover>/hw_status` is published at 5 Hz of simulated time. Every robot is also a body in the rendered scene, so the
robots see each other (each sensor excludes its own body); robots do not collide with each other in the mock.

Ray casting runs in a pool of worker processes (render_pool; parameter `render_procs`, -1 = auto, 0 = in-process):
the numpy renderer holds the GIL for much of a frame, so a two-rover fleet rendered in this process could not hold
1× real time. The physics thread snapshots the scene's objects when a frame comes due and a dispatch thread per
sensor hands the job to a worker.
"""
from __future__ import annotations

import json
import math
import os
import queue
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
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from std_msgs.msg import String

from .mock_dynamics import RoverSim, ScoutSim
from .mock_render import Scene, camera_intrinsics, camera_rotation, robot_body, rpy_matrix
from .render_pool import RenderPool, scene_objects
from .robot_models import ROVER_SPEC, SCOUT_SPEC
from .rover_hw import REPAIR_RANGE_M, REPAIR_S, hw_status_dict, planar_distance, solar_output
from .scene_events import rockslide_due, rockslide_layout
from .sim_pacing import SimPacer
from .terrain_gen import Terrain

PHYSICS_HZ = 100.0          # wall-clock rate of the physics thread; simulated time advances speed × faster
IMU_HZ = 100.0              # simulated-time rates, as in the Gazebo models
CLOCK_PERIOD_S = 0.01       # /clock granularity (simulated), same as the Gazebo republisher
ODOM_HZ = 30.0
WHEEL_ODOM_HZ = 20.0
LIDAR_HZ = 10.0
HW_STATUS_HZ = 5.0
AUTO_RENDER_PROCS = max(0, min(4, (os.cpu_count() or 2) - 4))   # leave cores for the autonomy stack
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
        self.hw = self.sim.drivetrain if self.rover else None
        self.hw_rng = np.random.default_rng(sum(name.encode()) + 1)
        self.repair: dict | None = None               # the repair event under way (echoed again when done)
        self.body = robot_body(name)
        self.body_ids = node.scene.add_body(self.body)
        self.body_off = np.array([part[0] for part in self.body], np.float64)
        self.exclude = frozenset(self.body_ids)       # a robot's sensors never see its own body
        self.path = None
        self.slide_armed: float | None = None   # sim time a rockslide was requested (rockslide_due)
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
            self.pub_hw = n.create_publisher(String, f'/{r}/hw_status', OUT_QOS)
            n.create_subscription(PathMsg, f'/{r}/planned_path', self.on_path, 5, callback_group=n.cmd_group)
        n.create_subscription(Twist, f'/{r}/cmd_vel', self.on_cmd, qos_profile_sensor_data,
                              callback_group=n.physics_group)

    def on_cmd(self, m: Twist):
        if self.rover:
            self.sim.cmd = (m.linear.x, m.angular.z)
        else:
            self.sim.cmd = (m.linear.x, m.linear.y, m.linear.z, m.angular.z)

    def place_body(self, scene: Scene, s: dict):
        scene.set_body_pose(self.body_ids, self.body_off, (s['x'], s['y'], s['z']),
                            rpy_matrix(s['roll'], s['pitch'], s['yaw']), s['yaw'])

    def on_path(self, m: PathMsg):
        self.path = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses]).reshape(-1, 2)


class MockWorld(Node):
    def __init__(self):
        super().__init__('mock_world')
        for k, v in (('terrain_path', ''), ('robots', ['rover_0', 'scout_0']), ('g_mult', 2.2), ('spawn_json', '{}'),
                     ('specs_json', '{}'), ('render_scale', 0.5), ('sim_speed', 1.0), ('render_procs', -1)):
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
        self.lock = threading.Lock()                      # guards robot state snapshots
        self.robots: list[Robot] = []
        for r in gp('robots').value:
            spec = specs.get(r) or (ROVER_SPEC if r.startswith('rover') else SCOUT_SPEC)
            sp = spawn.get(r)
            if sp is None:
                x, y = terr.spawn_xy[len(self.robots) % len(terr.spawn_xy)] if r.startswith('rover') else terr.scout_spawn_xy
                sp = (x, y, float(terr.height_at(x, y)), 0.0)
            self.robots.append(Robot(self, r, spec, sp))
        for rb in self.robots:
            rb.place_body(self.scene, rb.state)
        self.dust_level = 0.0
        self.create_subscription(String, '/world/env_state', self.on_env, 10, callback_group=self.cmd_group)
        self.events = self.create_publisher(String, '/world/events', 10)
        self.create_subscription(String, '/world/command', self.on_command, 10, callback_group=self.cmd_group)
        # this node is the clock source: simulated time is paced by SimPacer (lockstep with the renderers) and
        # published on /clock; every other node runs with use_sim_time
        self.pacer = SimPacer(float(gp('sim_speed').value))
        self.by_name = {rb.name: rb for rb in self.robots}
        for k, rb in enumerate(self.robots):
            if rb.rover:
                self.pacer.add_sensor((rb.name, 'lidar'), 1.0 / LIDAR_HZ, phase=0.05 * k)
            self.pacer.add_sensor((rb.name, 'cam'), 1.0 / rb.cam_hz, phase=0.02 + 0.05 * k)
        self.clock_pub = self.create_publisher(Clock, '/clock', QoSProfile(depth=10))
        self._last_clock = -1.0
        self._last = time.monotonic()
        self._due = {}                                    # (robot, stream) → next simulated time
        self._slide = 0
        procs = int(gp('render_procs').value)
        self.render = RenderPool(self.scene, AUTO_RENDER_PROCS if procs < 0 else procs)
        if self.render.error is not None:
            self.get_logger().warn(f'render pool unavailable ({self.render.error}); rendering in-process')
        self._jobs = {'lidar': queue.Queue(), 'cam': queue.Queue()}
        # with a pool, one dispatch thread per sensor (SimPacer keeps one frame per sensor in flight, so frames of
        # one sensor never overtake each other); in-process, one per kind: more threads would only thrash the GIL
        n_threads = {'lidar': sum(rb.rover for rb in self.robots), 'cam': len(self.robots)}
        for kind in self._jobs:
            for k in range(max(1, n_threads[kind]) if self.render.procs else 1):
                threading.Thread(target=self._render_loop, args=(kind,), name=f'render_{kind}_{k}',
                                 daemon=True).start()
        self.create_timer(1.0 / PHYSICS_HZ, self.physics, callback_group=self.physics_group)
        self.get_logger().info(f'mock world ready: {len(terr.rocks)} boulders, {len(terr.science)} crystals, '
                               f'g={self.g}, robots={[r.name for r in self.robots]}, render_scale={self.render_scale}, '
                               f'render_procs={self.render.procs}, '
                               f'speed={self.pacer.speed:.2f}x')

    @property
    def sim_t(self) -> float:
        return self.pacer.t

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def stamp(t: float) -> Time:
        sec = int(t)
        return Time(sec=sec, nanosec=int((t - sec) * 1e9))

    def on_env(self, msg: String):
        try:
            self.dust_level = float(json.loads(msg.data).get('dust_level') or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass

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

    # ------------------------------------------------------------------ physics thread (100 Hz wall)
    def physics(self):
        now = time.monotonic()
        wall_dt = now - self._last
        self._last = now
        for dt, due in self.pacer.steps(wall_dt):
            t = self.pacer.t
            if dt > 0.0:
                self._step(dt, t)
            objs = scene_objects(self.scene) if due else None   # boulders, falling rocks, robot bodies now
            for name, kind in due:                        # sample the frame at exactly its due time
                rb = self.by_name[name]
                with self.lock:
                    s = rb.state
                self._jobs[kind].put((rb, s, t, objs))
        if self.pacer.t > self._last_clock:
            self._publish_clock(self.pacer.t)

    def _step(self, dt: float, t: float):
        self.scene.update(t)
        stamp = self.stamp(t)
        for rb in self.robots:
            if rb.rover and rb.hw.update(t):
                self._repair_done(rb)
            rb.sim.step(dt, self.scene, t)
            s = rb.sim.snapshot()
            with self.lock:
                rb.state = s
            rb.place_body(self.scene, s)
            if rb.slide_armed is not None:
                self._try_rockslide(rb, t)
            for ev in (rb.sim.pop_events() if rb.rover else ()):
                if ev['op'] == 'jolt':
                    self.get_logger().info(f'jolt {rb.name}: released at roll {ev["release_roll_deg"]:.1f} deg')
                else:
                    self.get_logger().info(f'debris strike on {rb.name}: peak {ev["peak_g"]:.2f} g above gravity, '
                                           f'wheel health {ev["wheel_health"]}')
                self.emit(ev)
            if rb.rover and self._due_now((rb.name, 'hw'), 1.0 / HW_STATUS_HZ, t):
                self.publish_hw(rb, s)
            if self._due_now((rb.name, 'imu'), 1.0 / IMU_HZ, t):
                self.publish_imu(rb, s, stamp)
            if self._due_now((rb.name, 'odom'), 1.0 / ODOM_HZ, t):
                self.publish_odom(rb, s, stamp)
            if rb.rover and self._due_now((rb.name, 'wheel'), 1.0 / WHEEL_ODOM_HZ, t):
                self.publish_wheel(rb, s, stamp)
        if t - self._last_clock >= CLOCK_PERIOD_S - 1e-9:
            self._publish_clock(t)

    def _publish_clock(self, t: float):
        self._last_clock = t
        self.clock_pub.publish(Clock(clock=self.stamp(t)))

    def publish_hw(self, rb: Robot, s: dict):
        solar_w, irr = solar_output(self.dust_level, s['roll'], s['pitch'], s['yaw'])
        v_cmd, w_cmd = rb.sim.cmd
        cur = rb.hw.motor_current(v_cmd, w_cmd, rb.hw_rng)
        rb.pub_hw.publish(String(data=json.dumps(hw_status_dict(rb.name, time.time(), rb.hw, solar_w, irr, cur))))

    def _repair_done(self, rb: Robot):
        ev = dict(rb.repair or {'op': 'repair', 'ok': True, 'robot': rb.name})
        rb.repair = None
        ev.update(done=True, wheel_health=list(rb.hw.wheel_health), detail='drivetrain restored', t=time.time())
        self.get_logger().info(f'repair of {rb.name} complete')
        self.emit(ev)

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
            if rb.slide_armed is None:
                rb.slide_armed = self.sim_t
                self._try_rockslide(rb, self.sim_t)
        elif op == 'jolt':
            if rb is None or not rb.rover:
                self.emit({'op': 'jolt', 'ok': False, 'robot': cmd.get('robot'), 'detail': 'unknown rover'})
                return
            rb.sim.start_jolt()
        elif op == 'critical_damage':
            if rb is None or not rb.rover:
                self.emit({'op': 'critical_damage', 'ok': False, 'robot': cmd.get('robot'), 'detail': 'unknown rover'})
                return
            if not rb.sim.start_strike():
                self.emit({'op': 'critical_damage', 'ok': False, 'robot': rb.name, 'detail': 'strike already under way'})
                return
            rb.repair = None                           # a strike cancels any repair in progress
            self.get_logger().info(f'debris strike on {rb.name} at ({rb.state["x"]:.1f}, {rb.state["y"]:.1f})')
        elif op == 'repair':
            self._repair(rb, cmd)
        elif op == 'set_speed':
            try:
                speed = self.pacer.set_speed(float(cmd.get('speed', 1.0)))
            except (TypeError, ValueError):
                self.emit({'op': 'set_speed', 'ok': False, 'detail': f'bad speed {cmd.get("speed")!r}'})
                return
            self.get_logger().info(f'simulation speed set to {speed:.2f}x')
            self.emit({'op': 'set_speed', 'ok': True, 'speed': speed})

    def _try_rockslide(self, rb: Robot, t: float):
        """Drop the armed rockslide once rockslide_due: across the route ahead, or the next one near a goal."""
        s = rb.state
        waited = t - rb.slide_armed
        if not rockslide_due(rb.path, (s['x'], s['y'], s['yaw']), waited):
            return
        rb.slide_armed = None
        rocks, impact = rockslide_layout(rb.path, (s['x'], s['y'], s['yaw']), seed=self._slide)
        self._slide += 1
        self.scene.add_rocks([tuple(r) for r in rocks], t=t, drop_m=4.0)
        self.get_logger().info(f'rockslide #{self._slide} across {rb.name} path at {np.round(impact, 1)}'
                               + (f' (waited {waited:.1f} s for the next route)' if waited > 0 else ''))
        self.emit({'op': 'spawn_rockslide', 'ok': True, 'robot': rb.name, 'impact': [float(v) for v in impact],
                   'rocks': [[float(v) for v in r] for r in rocks], 'waited_s': round(waited, 2)})

    def _repair(self, rb: Robot | None, cmd: dict):
        by = self.by_name.get(cmd.get('by'))
        ev = {'op': 'repair', 'robot': rb.name if rb else cmd.get('robot'), 'by': cmd.get('by')}
        if rb is None or not rb.rover:
            self.emit(dict(ev, ok=False, detail='unknown rover'))
            return
        if by is None or not by.rover or by is rb:
            self.emit(dict(ev, ok=False, detail=f'unknown repairing rover {cmd.get("by")!r}'))
            return
        with self.lock:
            a, b = rb.state, by.state
        dist = round(planar_distance((a['x'], a['y']), (b['x'], b['y'])), 2)
        ev['distance_m'] = dist
        if dist > REPAIR_RANGE_M:
            self.emit(dict(ev, ok=False, detail=f'{by.name} is {dist:.2f} m from {rb.name} '
                                                f'(repair needs <= {REPAIR_RANGE_M:.1f} m)'))
            return
        if not rb.hw.start_repair(self.sim_t, REPAIR_S):
            self.emit(dict(ev, ok=True, done=False, detail='repair already in progress'))
            return
        rb.repair = dict(ev, ok=True)
        self.get_logger().info(f'repair of {rb.name} by {by.name} started at {dist:.2f} m')
        self.emit(dict(ev, ok=True, done=False, detail=f'repair started: drivetrain restoring over {REPAIR_S:.1f} s'))

    # ------------------------------------------------------------------ render threads
    def _render_loop(self, kind: str):
        jobs = self._jobs[kind]
        while rclpy.ok():
            try:
                rb, s, t, objs = jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                (self.publish_lidar if kind == 'lidar' else self.publish_camera)(rb, s, t, objs)
            except Exception as e:  # noqa: BLE001 - a bad frame must not stall the clock forever
                self.get_logger().warn(f'{kind} render failed for {rb.name}: {e}')
            finally:
                self.pacer.done((rb.name, kind))

    @staticmethod
    def _pose(s: dict):
        return np.array([s['x'], s['y'], s['z']]), rpy_matrix(s['roll'], s['pitch'], s['yaw'])

    def publish_lidar(self, rb: Robot, s: dict, t: float, objs: dict):
        pos, R = self._pose(s)
        stamp = self.stamp(t)
        pts = self.render.lidar_scan(objs, pos + R @ np.asarray(rb.spec['lidar_xyz']), R, rb.spec, exclude=rb.exclude)
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

    def publish_camera(self, rb: Robot, s: dict, t: float, objs: dict):
        pos, R = self._pose(s)
        stamp = self.stamp(t)
        sp = rb.spec
        Rc = camera_rotation(R, sp['camera_pitch_rad'])
        rgb, depth = self.render.camera_frame(objs, pos + R @ np.asarray(sp['camera_xyz']), Rc, sp['camera_w'],
                                              sp['camera_h'], sp['camera_hfov'], sp.get('camera_far', 40.0),
                                              scale=rb.scale, exclude=rb.exclude)
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
        node.render.close()
        node.destroy_node()
        rclpy.try_shutdown()
