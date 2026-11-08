"""rclpy side of the backend: one node in a background thread caching the fleet's topics for the WebSocket
broadcaster, forwarding traces/events/brain decisions into asyncio, and publishing /world/command and
/operator/command."""
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
from .encoders import RobotCache, RtfMeter, decode_clock_cdr, downsample_path
from .mission_manager import readiness_from_marks

try:
    import rclpy
    from nav_msgs.msg import Odometry, Path
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
    from rclpy.signals import SignalHandlerOptions
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2
    from std_msgs.msg import String
    HAVE_ROS = True
except ImportError:          # lets the app start (and say so) without a sourced ROS environment
    HAVE_ROS = False

from redshift_autonomy.ros_utils import cloud_to_numpy, quat_to_rpy

MAP_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL) if HAVE_ROS else None
# /brain/status comes at 1 Hz: a status whose 't' is older than this (wall s) means the brain node stopped reporting
BRAIN_STATUS_STALE_S = 3.0


def parse_json_object(data) -> dict | None:
    """A JSON object from a std_msgs/String payload, or None for anything else (bad JSON, a list, a number): a
    malformed message from another node must never reach the browser or raise in a callback."""
    try:
        d = json.loads(data)
    except (TypeError, ValueError):
        return None
    return d if isinstance(d, dict) else None


class RosLink:
    def __init__(self, on_trace: Callable[[dict], None], on_event: Callable[[dict], None],
                 on_brain: Callable[[dict], None] | None = None):
        if not HAVE_ROS:
            raise RuntimeError('rclpy not importable: source /opt/ros/jazzy/setup.bash and ros2_ws/install/setup.bash')
        self.on_trace, self.on_event = on_trace, on_event
        self.on_brain = on_brain or (lambda _d: None)
        self.lock = threading.RLock()
        self._ids = itertools.count(1)
        # uvicorn owns SIGINT/SIGTERM: its lifespan aborts the mission, then shutdown() tears ROS down in order
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node('redshift_backend')
        n = self.node
        self.cmd_pub = n.create_publisher(String, '/world/command', 10)
        self.op_pub = n.create_publisher(String, '/operator/command', 10)
        self.robots: dict[str, RobotCache] = {}
        self.specs: dict = {}
        self._robot_subs: list = []
        self.reset()
        n.create_subscription(String, '/agent_decision_trace', self._on_trace, 100)
        n.create_subscription(String, '/world/events', lambda m: self._on_event_json(m.data, None), 20)
        n.create_subscription(String, '/fleet/scan_request', lambda m: self._on_event_json(m.data, 'scan_request'), 10)
        n.create_subscription(String, '/fleet/scan_complete', lambda m: self._on_event_json(m.data, 'scan_complete'), 10)
        n.create_subscription(String, '/fleet/hazard_marks', lambda m: self._on_event_json(m.data, 'hazard_mark'), 10)
        n.create_subscription(String, '/fleet/sos', lambda m: self._on_event_json(m.data, 'sos'), 10)
        n.create_subscription(String, '/fleet/rescue', lambda m: self._on_event_json(m.data, 'rescue'), 10)
        n.create_subscription(String, '/world/env_state', self._on_env, 10)
        # advisory brain (fleet_brain_node): every decision goes to the browser at once; the 1 Hz status is cached
        # for the telemetry mission block. Depth 50: decisions come in bursts (an event fans out to several robots).
        n.create_subscription(String, '/brain/decision', self._on_brain_decision, 50)
        n.create_subscription(String, '/brain/status', self._on_brain_status, 10)
        n.create_subscription(Image, '/planet_map/layers', self._on_layers, MAP_QOS)
        n.create_subscription(String, '/planet_map/stats', self._on_stats, MAP_QOS)
        # simulated time (Gazebo or the mock world) → mission clock + achieved real-time factor. ~100 Hz, so raw CDR.
        n.create_subscription(Clock, '/clock', self._on_clock, SENSOR_QOS, raw=True)
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
            self.env = {'dust_level': 0.0, 'seismic': False, 'remaining_s': None}
            self.layers: np.ndarray | None = None
            self.layers_version = 0
            self.stats: dict = {}
            self.first_global: dict = {}
            self.clock = RtfMeter()
            self.brain: dict | None = None        # latest /brain/status of this mission's fleet_brain_node
            self._brain_rx = 0.0                  # wall time it arrived (age fallback when its 't' is unusable)

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
                n.create_subscription(CompressedImage, f'/{r}/perception/depth/compressed',
                                      lambda m, c=c: self._on_depth(c, m), SENSOR_QOS),
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
        a = m.linear_acceleration
        c.imu = {'roll_deg': round(math.degrees(r), 2), 'pitch_deg': round(math.degrees(p), 2),
                 'gyro': [round(w.x, 3), round(w.y, 3), round(w.z, 3)],
                 'accel': [round(a.x, 3), round(a.y, 3), round(a.z, 3)]}

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

    def _on_depth(self, c: RobotCache, m):
        b = base64.b64encode(bytes(m.data)).decode('ascii')
        with self.lock:
            c.depth_b64, c.depth_t = b, time.time()

    def _on_path(self, c: RobotCache, m):
        xy = [(ps.pose.position.x, ps.pose.position.y) for ps in m.poses]
        with self.lock:
            c.path = downsample_path(xy, 150)

    def _on_cloud(self, c: RobotCache, m):
        with self.lock:
            c.cloud, c.cloud_t = m, time.time()

    def _on_trace(self, m):
        t = parse_json_object(m.data)
        if t is None:
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

    def _on_brain_decision(self, m):
        d = parse_json_object(m.data)
        if d is not None:
            self.on_brain(d)               # forwarded verbatim: the HUD shows exactly what the brain node decided

    def _on_brain_status(self, m):
        d = parse_json_object(m.data)
        if d is not None:
            with self.lock:
                self.brain, self._brain_rx = d, time.time()

    def _on_env(self, m):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.env = d
            self.first_global.setdefault('env', time.time())

    def _on_clock(self, raw: bytes):
        t = decode_clock_cdr(raw)
        if t is None:
            return
        with self.lock:
            self.clock.add(time.monotonic(), t)

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

    def sim_clock(self) -> tuple[float | None, float | None]:
        """(latest simulated time s, achieved real-time factor) — None until /clock has been seen."""
        with self.lock:
            return self.clock.sim_t, self.clock.rtf(time.monotonic())

    def lidar_cloud(self, robot: str):
        with self.lock:
            c = self.robots.get(robot)
            msg = c.cloud if c else None
        if msg is None:
            return None
        return cloud_to_numpy(msg)

    def brain_status(self, now: float | None = None) -> dict | None:
        """Latest /brain/status of the running mission: None until the brain node has reported, else its last fields
        plus "stale" — true once its 't' (wall epoch) is more than BRAIN_STATUS_STALE_S old, so the HUD never shows a
        dead brain node's last numbers as live. A status without a usable 't' ages from when it arrived."""
        with self.lock:
            if self.brain is None:
                return None
            d, rx = dict(self.brain), self._brain_rx
        t = d.get('t')
        try:                       # a huge JSON integer would overflow float(): age it from its arrival instead
            ref = float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) else rx
        except OverflowError:
            ref = rx
        d['stale'] = (time.time() if now is None else now) - ref > BRAIN_STATUS_STALE_S
        return d

    def publish_command(self, cmd: dict) -> int:
        self.cmd_pub.publish(String(data=json.dumps(cmd)))
        return self.cmd_pub.get_subscription_count()

    def publish_operator_command(self, cmd: dict) -> int:
        """/operator/command for fleet_brain_node; returns how many subscribers will receive it."""
        self.op_pub.publish(String(data=json.dumps(cmd)))
        return self.op_pub.get_subscription_count()
