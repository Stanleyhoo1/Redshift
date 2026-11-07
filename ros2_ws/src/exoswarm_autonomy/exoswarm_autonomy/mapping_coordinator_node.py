"""Global mapping coordinator: fuses every robot's LiDAR / depth into one gravity-aware /planet_map.

Inputs are only degraded sensor topics + odometry pose (no ground truth). Points are taken from the sensor frame
to the world with the robot extrinsics from `specs_json` (robot_models.ROVER_SPEC / SCOUT_SPEC via the launch file).
"""
from __future__ import annotations

import bisect
import collections
import json
import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String

from .grid_map import GlobalMap
from .physics_model import RoverPhysics
from .ros_utils import cloud_to_numpy, image_to_numpy, quat_to_matrix
from .vision import CameraModel

MAP_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
LIDAR_MIN_RANGE = 2.0          # dust backscatter lives in the first ~2 m
SELF_FILTER_M = 0.9            # drop returns from robot bodies (own and teammates)


def _ry(p: float) -> np.ndarray:
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def depth_to_cam_link(depth: np.ndarray, cam: CameraModel, stride: int, max_range: float) -> np.ndarray:
    """Planar depth image → points in the Gazebo camera link frame (x forward, y left, z up)."""
    d = depth[::stride, ::stride]
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    u = u + 0.5
    v = v + 0.5
    ok = np.isfinite(d) & (d > 0.05) & (d < max_range)
    d, u, v = d[ok], u[ok], v[ok]
    return np.stack([d, -(u - cam.cx) * d / cam.fx, -(v - cam.cy) * d / cam.fy], axis=1)


class RobotFeed:
    def __init__(self, name: str, spec: dict):
        self.name = name
        self.rover = name.startswith('rover')
        self.spec = spec
        self.pos: np.ndarray | None = None
        self.R = np.eye(3)
        self.cam: CameraModel | None = None
        self.ms = 0.0
        self.pending_cloud = None
        self.pending_depth = None
        self.last_depth = 0.0
        self.hist = collections.deque(maxlen=150)    # (stamp_s, pos, R) at odom rate (~3 s)

    def pose_at(self, stamp) -> tuple[np.ndarray, np.ndarray]:
        """Odometry pose closest to a sensor timestamp (both stamped in simulation time)."""
        t = stamp.sec + stamp.nanosec * 1e-9
        if not self.hist or t <= 0:
            return self.pos, self.R
        ts = [h[0] for h in self.hist]
        k = bisect.bisect_left(ts, t)
        cands = [c for c in (k - 1, k) if 0 <= c < len(ts)]
        best = min(cands, key=lambda c: abs(ts[c] - t))
        if abs(ts[best] - t) > 0.25:
            return self.pos, self.R
        return self.hist[best][1], self.hist[best][2]


class MappingCoordinator(Node):
    def __init__(self):
        super().__init__('mapping_coordinator')
        for k, v in (('g_mult', 2.2), ('edge_profile', 'orin_nano_15w'), ('size_m', 100.0), ('map_res', 0.5),
                     ('robots', ['rover_0']), ('specs_json', '{}'), ('publish_hz', 3.0)):
            self.declare_parameter(k, v)
        g = float(self.get_parameter('g_mult').value)
        self.limits = RoverPhysics().limits(g)
        size_m = float(self.get_parameter('size_m').value)
        res = float(self.get_parameter('map_res').value)
        specs = json.loads(self.get_parameter('specs_json').value or '{}')
        self.map = GlobalMap(size_m, res)
        self.t0 = time.monotonic()
        self.feeds = {r: RobotFeed(r, specs.get(r, {})) for r in self.get_parameter('robots').value}
        for r, f in self.feeds.items():
            self.create_subscription(Odometry, f'/{r}/odom', lambda m, f=f: self.on_odom(f, m), SENSOR_QOS)
            self.create_subscription(Image, f'/{r}/camera/depth', lambda m, f=f: setattr(f, 'pending_depth', m),
                                     SENSOR_QOS)
            self.create_subscription(CameraInfo, f'/{r}/camera/camera_info', lambda m, f=f: self.on_info(f, m),
                                     SENSOR_QOS)
            self.create_subscription(String, f'/{r}/perception/detections', self.on_detections, 10)
            if f.rover:
                self.create_subscription(PointCloud2, f'/{r}/lidar/points',
                                         lambda m, f=f: setattr(f, 'pending_cloud', m), SENSOR_QOS)
        self.create_subscription(String, '/fleet/hazard_marks', self.on_hazard, 10)
        self.pub_occ = self.create_publisher(OccupancyGrid, '/planet_map', MAP_QOS)
        self.pub_elev = self.create_publisher(Image, '/planet_map/elevation', MAP_QOS)
        self.pub_layers = self.create_publisher(Image, '/planet_map/layers', MAP_QOS)
        self.pub_stats = self.create_publisher(String, '/planet_map/stats', MAP_QOS)
        self._recompute_ms = 0.0
        self._marks: list = []
        self.create_timer(0.05, self.fuse)
        self.create_timer(1.0 / float(self.get_parameter('publish_hz').value), self.publish)
        self.get_logger().info(f'mapping {size_m:.0f} m @ {res} m, g={g} (θmax {self.limits.max_slope_deg:.1f}°), '
                               f'robots={list(self.feeds)}')

    def now(self) -> float:
        return time.monotonic() - self.t0

    # ------------------------------------------------------------------ inputs
    def on_odom(self, f: RobotFeed, m: Odometry):
        p = m.pose.pose.position
        f.pos = np.array([p.x, p.y, p.z])
        f.R = quat_to_matrix(m.pose.pose.orientation)
        f.hist.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, f.pos, f.R))

    def on_info(self, f: RobotFeed, m: CameraInfo):
        if f.cam is None and m.k[0] > 0:
            f.cam = CameraModel.from_K(m.width, m.height, m.k)

    def on_detections(self, m: String):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        for det in d.get('detections', []):
            if det.get('cls') == 'SCIENCE' and det.get('world_xy') and det.get('conf', 0) >= 0.55:
                self.map.mark_science(det['world_xy'])

    def on_hazard(self, m: String):
        try:
            d = json.loads(m.data)
            self.map.mark_permanent(d['xy'], float(d.get('radius_m', 2.5)))
            self.get_logger().info(f"permanent hazard at {d['xy']} r={d.get('radius_m')} ({d.get('reason')})")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.get_logger().warn(f'bad hazard mark: {e}')

    # ------------------------------------------------------------------ fusion
    def _self_filter(self, pts: np.ndarray) -> np.ndarray:
        keep = np.ones(len(pts), bool)
        for f in self.feeds.values():
            if f.pos is None:
                continue
            if f.rover:
                keep &= np.hypot(pts[:, 0] - f.pos[0], pts[:, 1] - f.pos[1]) > SELF_FILTER_M
            else:
                keep &= np.linalg.norm(pts - f.pos, axis=1) > 1.2
        return pts[keep]

    def fuse(self):
        t = self.now()
        for f in self.feeds.values():
            if f.pos is None:
                continue
            t0 = time.perf_counter()
            if f.pending_cloud is not None:
                msg, f.pending_cloud = f.pending_cloud, None
                grid = cloud_to_numpy(msg).astype(np.float64)
                pos, R = f.pose_at(msg.header.stamp)
                lx, ly, lz = f.spec.get('lidar_xyz', [0.25, 0.0, 0.62])
                sensor_w = R @ np.array([lx, ly, lz]) + pos
                world = grid.reshape(-1, 3) @ R.T + sensor_w
                near = np.linalg.norm(world - sensor_w, axis=1) < LIDAR_MIN_RANGE
                bodies = np.zeros(len(world), bool)
                for o in self.feeds.values():
                    if o.pos is not None:
                        lim = SELF_FILTER_M if o.rover else 1.2
                        bodies |= np.hypot(world[:, 0] - o.pos[0], world[:, 1] - o.pos[1]) < lim
                world[near | bodies] = np.nan
                self.map.integrate_organized(world.reshape(grid.shape), sensor_w, t, min_range=LIDAR_MIN_RANGE)
            # depth: rovers 5 Hz near-field, scouts every frame (their only mapping sensor)
            if f.pending_depth is not None and (not f.rover or t - f.last_depth >= 0.2):
                msg, f.pending_depth = f.pending_depth, None
                f.last_depth = t
                depth = image_to_numpy(msg)
                cam = f.cam or CameraModel.from_hfov(depth.shape[1], depth.shape[0], f.spec.get('camera_hfov', 1.396))
                stride = 6 if f.rover else 2
                pc = depth_to_cam_link(depth, cam, stride, 12.0 if f.rover else 40.0)
                if len(pc):
                    cx, cy, cz = f.spec.get('camera_xyz', [0.45, 0.0, 0.5])
                    pb = pc @ _ry(f.spec.get('camera_pitch_rad', 0.2094)).T + np.array([cx, cy, cz])
                    pos, R = f.pose_at(msg.header.stamp)
                    cam_w = R @ np.array([cx, cy, cz]) + pos
                    pw = self._self_filter(pb @ R.T + pos)
                    self.map.integrate(pw, t, sensor_xyz=cam_w, min_range=0.3)
            f.ms = 0.8 * f.ms + 0.2 * (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ outputs
    def publish(self):
        t0 = time.perf_counter()
        t = self.now()
        m = self.map
        m.recompute(self.limits)
        self._recompute_ms = (time.perf_counter() - t0) * 1000
        stamp = self.get_clock().now().to_msg()

        occ = OccupancyGrid()
        occ.header.stamp = stamp
        occ.header.frame_id = 'world'
        occ.info.resolution = m.res
        occ.info.width = occ.info.height = m.n
        occ.info.origin.position.x, occ.info.origin.position.y = m.origin
        occ.info.origin.orientation.w = 1.0
        occ.data = m.to_occupancy().ravel().tolist()
        self.pub_occ.publish(occ)

        for pub, arr, enc, step in ((self.pub_elev, m.ground(), '32FC1', 4), (self.pub_layers, m.layers_rgb(t), 'rgb8', 3)):
            img = Image()
            img.header.stamp = stamp
            img.header.frame_id = 'world'
            img.height = img.width = m.n
            img.encoding = enc
            img.step = m.n * step
            img.data = np.ascontiguousarray(arr).tobytes()
            pub.publish(img)

        st = m.stats()
        st.update({'t': time.time(), 'size_m': m.size_m, 'res': m.res, 'cells': m.n, 'origin': list(m.origin),
                   'g': self.limits.g, 'max_slope_deg': round(self.limits.max_slope_deg, 2),
                   'mapping_ms': {r: round(f.ms + self._recompute_ms / max(1, len(self.feeds)), 2)
                                  for r, f in self.feeds.items()},
                   'recompute_ms': round(self._recompute_ms, 2)})
        self.pub_stats.publish(String(data=json.dumps(st)))


def main():
    rclpy.init()
    node = MappingCoordinator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
