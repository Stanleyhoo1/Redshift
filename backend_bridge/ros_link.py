"""rclpy side of the backend: one node in a background thread caching the fleet's topics for the WebSocket
broadcaster, forwarding traces/events into asyncio, and publishing /world/command."""
from __future__ import annotations

import base64
import itertools
import json
import math
import threading
import time
from typing import Callable

import numpy as np

from . import _paths  # noqa: F401
from .encoders import RobotCache, downsample_path
from .mission_manager import readiness_from_marks

try:
    import rclpy
    from nav_msgs.msg import Odometry, Path
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2
    from std_msgs.msg import String
    HAVE_ROS = True
except ImportError:          # lets the app start (and say so) without a sourced ROS environment
    HAVE_ROS = False

from exoswarm_autonomy.ros_utils import cloud_to_numpy, quat_to_rpy

MAP_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL) if HAVE_ROS else None


class RosLink:
    def __init__(self, on_trace: Callable[[dict], None], on_event: Callable[[dict], None]):
        if not HAVE_ROS:
            raise RuntimeError('rclpy not importable: source /opt/ros/jazzy/setup.bash and ros2_ws/install/setup.bash')
        self.on_trace, self.on_event = on_trace, on_event
        self.lock = threading.RLock()
        self._ids = itertools.count(1)
        # uvicorn owns SIGINT/SIGTERM: its lifespan aborts the mission, then shutdown() tears ROS down in order
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node('redshift_backend')
        n = self.node
        self.cmd_pub = n.create_publisher(String, '/world/command', 10)
        self.robots: dict[str, RobotCache] = {}
        self.specs: dict = {}
        self._robot_subs: list = []
        self.reset()
        n.create_subscription(String, '/agent_decision_trace', self._on_trace, 100)
        n.create_subscription(String, '/world/events', lambda m: self._on_event_json(m.data, None), 20)
        n.create_subscription(String, '/fleet/scan_request', lambda m: self._on_event_json(m.data, 'scan_request'), 10)
        n.create_subscription(String, '/fleet/scan_complete', lambda m: self._on_event_json(m.data, 'scan_complete'), 10)
        n.create_subscription(String, '/fleet/hazard_marks', lambda m: self._on_event_json(m.data, 'hazard_mark'), 10)
        n.create_subscription(String, '/world/env_state', self._on_env, 10)
        n.create_subscription(Image, '/planet_map/layers', self._on_layers, MAP_QOS)
        n.create_subscription(String, '/planet_map/stats', self._on_stats, MAP_QOS)
        self._stop = threading.Event()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(n)
        self._thread = threading.Thread(target=self._spin, name='ros_link', daemon=True)
        self._thread.start()

    def _spin(self):
        while rclpy.ok() and not self._stop.is_set():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:  # noqa: BLE001 - keep the link alive across a bad message
                self.node.get_logger().warn(f'ros_link callback error: {e}')

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.executor.remove_node(self.node)
            self.node.destroy_node()
        finally:
            rclpy.try_shutdown()

    # ------------------------------------------------------------------ fleet (re)configuration
    def reset(self):
        with self.lock:
            self.env = {'dust_level': 0.0, 'seismic': False, 'until_t': None}
            self.layers: np.ndarray | None = None
            self.layers_version = 0
            self.stats: dict = {}
            self.first_global: dict = {}

    def set_fleet(self, robots: list[str], specs: dict):
        with self.lock:
            for s in self._robot_subs:
                self.node.destroy_subscription(s)
            self._robot_subs = []
            self.robots = {r: RobotCache(r) for r in robots}
            self.specs = specs
        self.reset()
        n = self.node
        for r, c in self.robots.items():
            subs = [
                n.create_subscription(Odometry, f'/{r}/odom', lambda m, c=c: self._on_odom(c, m), SENSOR_QOS),
                n.create_subscription(Imu, f'/{r}/imu', lambda m, c=c: self._on_imu(c, m), SENSOR_QOS),
                n.create_subscription(String, f'/{r}/agent_state', lambda m, c=c: self._json(c, 'agent_state', m), 10),
                n.create_subscription(String, f'/{r}/edge_telemetry', lambda m, c=c: self._json(c, 'edge', m), 10),
                n.create_subscription(String, f'/{r}/perception/detections',
                                      lambda m, c=c: self._json(c, 'detections', m), 10),
                n.create_subscription(CompressedImage, f'/{r}/perception/annotated/compressed',
                                      lambda m, c=c: self._on_jpeg(c, m), SENSOR_QOS),
                n.create_subscription(Path, f'/{r}/planned_path', lambda m, c=c: self._on_path(c, m), 10),
                # proof that the simulator renders: a clean (post sensor-effects) camera frame. raw=True skips
                # deserialising the pixels; only the arrival is recorded.
                n.create_subscription(Image, f'/{r}/camera/image', lambda _m, c=c: self._on_frame(c), SENSOR_QOS,
                                      raw=True),
            ]
            if c.rtype == 'rover':
                subs.append(n.create_subscription(PointCloud2, f'/{r}/lidar/points',
                                                  lambda m, c=c: self._on_cloud(c, m), SENSOR_QOS))
            self._robot_subs += subs

    # ------------------------------------------------------------------ callbacks (ROS thread)
    def _mark(self, c: RobotCache, key: str):
        if key not in c.first:
            c.first[key] = time.time()

    def _on_odom(self, c: RobotCache, m):
        p = m.pose.pose
        r, pt, y = quat_to_rpy(p.orientation)
        with self.lock:
            c.pose = {'x': p.position.x, 'y': p.position.y, 'z': p.position.z, 'roll': r, 'pitch': pt, 'yaw': y}
            c.trail.add(p.position.x, p.position.y)
            self._mark(c, 'odom')

    def _on_frame(self, c: RobotCache):
        if 'frame' not in c.first:
            with self.lock:
                self._mark(c, 'frame')

    def _on_imu(self, c: RobotCache, m):
        r, p, _ = quat_to_rpy(m.orientation)
        w = m.angular_velocity
        c.imu = {'roll_deg': round(math.degrees(r), 2), 'pitch_deg': round(math.degrees(p), 2),
                 'gyro': [round(w.x, 3), round(w.y, 3), round(w.z, 3)]}

    def _json(self, c: RobotCache, attr: str, m):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            setattr(c, attr, d)
            self._mark(c, attr)

    def _on_jpeg(self, c: RobotCache, m):
        b = base64.b64encode(bytes(m.data)).decode('ascii')
        with self.lock:
            c.jpeg_b64, c.jpeg_t = b, time.time()
            self._mark(c, 'camera')

    def _on_path(self, c: RobotCache, m):
        xy = [(ps.pose.position.x, ps.pose.position.y) for ps in m.poses]
        with self.lock:
            c.path = downsample_path(xy, 150)

    def _on_cloud(self, c: RobotCache, m):
        with self.lock:
            c.cloud, c.cloud_t = m, time.time()

    def _on_trace(self, m):
        try:
            t = json.loads(m.data)
        except json.JSONDecodeError:
            return
        t['id'] = next(self._ids)      # traces come from several processes: give them one global sequence
        self.on_trace(t)

    def _on_event_json(self, data: str, op: str | None):
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            return
        if op:
            ev = dict(ev, op=op)
        ev.setdefault('t', time.time())
        self.on_event(ev)

    def _on_env(self, m):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.env = d
            self.first_global.setdefault('env', time.time())

    def _on_layers(self, m):
        arr = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        with self.lock:
            self.layers = arr
            self.layers_version += 1
            self.first_global.setdefault('map', time.time())

    def _on_stats(self, m):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.stats = d
            self.first_global.setdefault('stats', time.time())

    # ------------------------------------------------------------------ queries (asyncio thread)
    def readiness(self, backend: str) -> set:
        """Checklist keys whose data has arrived (see mission_manager.checklist_keys)."""
        with self.lock:
            marks = {r: dict(c.first) for r, c in self.robots.items()}
            glob = dict(self.first_global)
        return readiness_from_marks(backend, marks, glob)

    def lidar_cloud(self, robot: str):
        with self.lock:
            c = self.robots.get(robot)
            msg = c.cloud if c else None
        if msg is None:
            return None
        return cloud_to_numpy(msg)

    def publish_command(self, cmd: dict) -> int:
        self.cmd_pub.publish(String(data=json.dumps(cmd)))
        return self.cmd_pub.get_subscription_count()
