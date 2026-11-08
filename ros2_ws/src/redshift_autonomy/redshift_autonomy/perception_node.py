"""Per-robot local perception: RGB-D + LiDAR + IMU → detections, measured uncertainty, annotated feed."""
from __future__ import annotations

import collections
import dataclasses
import json
import math
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, LaserScan, PointCloud2
from std_msgs.msg import Float32, String

from .edge_budget import PROFILES
from .physics_model import RoverPhysics
from .ros_utils import cloud_to_numpy, image_to_numpy, quat_to_rpy, rot_rp
from .agent_fsm import unit_label
from .vision import (CLEAR_AIR_LIDAR, CameraModel, ImageQualityBaseline, LidarBaseline, annotate, depth_colormap,
                     detect_objects, fit_ground_plane, fuse_uncertainty, image_quality, lidar_corridor, lidar_quality,
                     teammate_detection)

TEAMMATE_MAX_M = 25.0
TEAMMATE_FRESH_S = 1.0
OBSTACLE_CLASSES = ('ROCK', 'PILLAR')      # obstacle bumps a teammate's chassis could be mistaken for


class Perception(Node):
    def __init__(self):
        super().__init__('perception')
        for k, v in (('robot', 'rover_0'), ('robot_type', 'rover'), ('spec_json', '{}'), ('edge_profile', 'orin_nano_15w'),
                     ('g_mult', 2.2), ('size_m', 100.0), ('map_res', 0.5), ('robots', ['rover_0'])):
            self.declare_parameter(k, v)
        self.robot = self.get_parameter('robot').value
        self.rtype = self.get_parameter('robot_type').value
        self.spec = json.loads(self.get_parameter('spec_json').value or '{}')
        self.limits = RoverPhysics().limits(float(self.get_parameter('g_mult').value))
        self.profile = PROFILES[self.get_parameter('edge_profile').value]
        self.hz = self.profile.perception_hz_max
        self.mode = 'FUSED'
        r = self.robot
        self.rgb = self.depth = self.cloud = None
        self.cam: CameraModel | None = None
        self.pose = (0.0, 0.0, 0.0)
        self.z = 0.0
        self.rp = (0.0, 0.0)
        self.acc = collections.deque(maxlen=100)
        self.img_base = ImageQualityBaseline()
        self.lidar_base = LidarBaseline()
        self.lidar_conf = 1.0
        self.lidar_q = None
        self.corridor = {'obstacle_dist_m': math.inf, 'slope_ahead_deg': 0.0, 'conf': 0.0, 'points': 0}
        self.scan_pts = np.zeros((0, 3))
        self._last_frame = -1e9
        self._fps = collections.deque(maxlen=10)
        self.create_subscription(Image, f'/{r}/camera/image', lambda m: setattr(self, 'rgb', m), SENSOR_QOS)
        self.create_subscription(Image, f'/{r}/camera/depth', lambda m: setattr(self, 'depth', m), SENSOR_QOS)
        self.create_subscription(CameraInfo, f'/{r}/camera/camera_info', self.on_info, SENSOR_QOS)
        self.create_subscription(Imu, f'/{r}/imu', self.on_imu, SENSOR_QOS)
        self.create_subscription(Odometry, f'/{r}/odom', self.on_odom, SENSOR_QOS)
        self.create_subscription(String, f'/{r}/perception/mode_request', self.on_mode, 10)
        self.create_subscription(String, f'/{r}/edge_telemetry', self.on_edge, 10)
        if self.rtype == 'rover':
            self.create_subscription(PointCloud2, f'/{r}/lidar/points', self.on_cloud, SENSOR_QOS)
        self.pub_jpg = self.create_publisher(CompressedImage, f'/{r}/perception/annotated/compressed', SENSOR_QOS)
        self.pub_det = self.create_publisher(String, f'/{r}/perception/detections', 10)
        self.pub_unc = self.create_publisher(Float32, f'/{r}/perception/uncertainty', 10)
        self.pub_scan = self.create_publisher(LaserScan, f'/{r}/perception/scan2d', SENSOR_QOS)
        self.pub_depth = self.create_publisher(CompressedImage, f'/{r}/perception/depth/compressed', SENSOR_QOS)
        # teammates: fleet-reported poses say where to look; this robot's own sensors confirm them
        self.mates: dict = {}               # robot → (x, y, z, yaw, t_sim)
        self.damaged: dict = {}             # robot → SOS id while its SOS is active
        self.docking: set = set()           # teammates this robot is docking with / repairing
        for mate in (x for x in self.get_parameter('robots').value if x.startswith('rover') and x != r):
            self.create_subscription(Odometry, f'/{mate}/odom', lambda m, mate=mate: self.on_mate(mate, m), SENSOR_QOS)
        self.create_subscription(String, '/fleet/sos', self.on_sos, 10)
        self.create_subscription(String, '/fleet/rescue', self.on_rescue, 10)
        self.create_timer(0.02, self.tick)
        self.get_logger().info(f'perception up for {r} ({self.rtype}), {self.profile.label}')

    # ------------------------------------------------------------------ callbacks
    def on_info(self, m: CameraInfo):
        if self.cam is None and m.k[0] > 0:
            self.cam = CameraModel.from_K(m.width, m.height, m.k)

    def on_imu(self, m: Imu):
        r, p, _ = quat_to_rpy(m.orientation)
        self.rp = (r, p)
        a = m.linear_acceleration
        self.acc.append(math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z))

    def on_odom(self, m: Odometry):
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, quat_to_rpy(p.orientation)[2])
        self.z = p.position.z

    def on_mate(self, robot: str, m: Odometry):
        p = m.pose.pose
        self.mates[robot] = (p.position.x, p.position.y, p.position.z, quat_to_rpy(p.orientation)[2],
                             self.get_clock().now().nanoseconds * 1e-9)

    def on_sos(self, m: String):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        if d.get('status') == 'active':
            self.damaged[d.get('robot')] = d.get('id')
        else:
            self.damaged.pop(d.get('robot'), None)

    def on_rescue(self, m: String):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        if d.get('rescuer') != self.robot:
            return
        if d.get('phase') in ('DOCKING', 'REPAIRING'):
            self.docking.add(d.get('target'))
        else:
            self.docking.discard(d.get('target'))

    def teammates(self, cam: CameraModel, depth, cam_pitch: float, cam_xyz, now: float):
        """ROVER detections for teammates this robot can actually see (camera+depth and/or LiDAR)."""
        out = []
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        for mate, (mx, my, mz, myaw, t) in self.mates.items():
            if now - t > TEAMMATE_FRESH_S:
                continue
            dx, dy = mx - x, my - y
            if math.hypot(dx, dy) > TEAMMATE_MAX_M:
                continue
            bx, by = c * dx + s * dy, -s * dx + c * dy               # levelled body frame
            rel = (bx - cam_xyz[0], by - cam_xyz[1], (mz - self.z) - cam_xyz[2])
            pts = self.scan_pts if self.rtype == 'rover' else None
            damaged = mate in self.damaged
            label = None
            if damaged:
                label = f'[PERCEIVED: DAMAGED UNIT {unit_label(mate)}' + (
                    ' | DOCKING FOR REPAIR]' if mate in self.docking else ']')
            else:
                label = f'TEAMMATE {unit_label(mate)}'
            d = teammate_detection(cam, depth, cam_pitch, rel, myaw - yaw, (bx, by), lidar_pts=pts, robot=mate,
                                   damaged=damaged, label=label, world_xy=(mx, my))
            if d is not None:
                out.append(d)
        return out

    def on_mode(self, m: String):
        if m.data in ('FUSED', 'LIDAR_ONLY'):
            self.mode = m.data

    def on_edge(self, m: String):
        try:
            self.hz = float(json.loads(m.data).get('perception_hz', self.hz)) or self.hz
        except (ValueError, json.JSONDecodeError):
            pass

    def on_cloud(self, m: PointCloud2):
        t0 = time.perf_counter()
        grid = cloud_to_numpy(m)
        self.lidar_q = lidar_quality(grid)
        if self.lidar_q['n'] > 50:
            self.lidar_base.update(self.lidar_q)
        self.lidar_conf = self.lidar_base.confidence(self.lidar_q)
        pts = grid.reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)].astype(np.float64)
        lx, ly, lz = self.spec.get('lidar_xyz', [0.25, 0.0, 0.62])
        pts = pts + np.array([lx, ly, lz + self.spec.get('base_height_m', 0.28)])
        pts = pts @ rot_rp(*self.rp).T
        self.corridor = lidar_corridor(pts, half_width=self.spec.get('radius_m', 0.6) + 0.3,
                                       prev_cells=self.corridor.get('cells'))
        self.scan_pts = pts
        self._lidar_ms = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ main loop
    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9     # simulated: the edge-profile rate is per sim second
        if self.rgb is None or self.depth is None or now - self._last_frame < 1.0 / max(self.hz, 0.5):
            return
        rgb_msg, depth_msg = self.rgb, self.depth
        self.rgb = None
        self._last_frame = now
        self._fps.append(now)
        st = {}
        t0 = time.perf_counter()
        rgb = image_to_numpy(rgb_msg)
        depth = image_to_numpy(depth_msg)
        cam = self.cam or CameraModel.from_hfov(rgb.shape[1], rgb.shape[0], self.spec.get('camera_hfov', 1.396))
        q = image_quality(rgb)
        cam_conf = self.img_base.confidence(q)
        scout_h = None
        if self.rtype != 'rover':
            c = depth[depth.shape[0] // 3:2 * depth.shape[0] // 3, depth.shape[1] // 3:2 * depth.shape[1] // 3]
            c = c[np.isfinite(c) & (c > 0.05)]
            scout_h = float(np.median(c)) if len(c) > 20 else 0.0
        # clear-condition baseline. Rover: in any mode, re-learning a changed scene when its LiDAR confirms clear
        # air (else a texture-poor stretch after the warm-up would hold it in CAUTION). Scout (no LiDAR): only in
        # FUSED mode at survey altitude.
        if self.rtype == 'rover':
            self.img_base.update(q, clear_air=self.lidar_conf >= CLEAR_AIR_LIDAR)
        elif self.mode == 'FUSED' and scout_h > 8.0:
            self.img_base.update(q)
        imu_vib = float(np.std(self.acc)) if len(self.acc) > 10 else 0.0
        lidar_conf = self.lidar_conf if self.rtype == 'rover' else cam_conf
        score = fuse_uncertainty(cam_conf, lidar_conf, imu_vib, self.mode)
        st['quality'] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        # rover: half-res depth is plenty at 2-15 m; scout: boulders are only a few pixels from 12 m AGL
        d_small = (cv2.resize(depth, (depth.shape[1] // 2, depth.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
                   if self.rtype == 'rover' else depth)
        if self.rtype == 'rover':
            pitch = self.spec.get('camera_pitch_rad', 0.2094) + self.rp[1]
            cam_h = self.spec.get('base_height_m', 0.28) + self.spec.get('camera_xyz', [0, 0, 0.5])[2]
        else:
            pitch = math.pi / 2
            cam_h = scout_h if scout_h and scout_h > 0.5 else 12.0
        if self.rtype == 'rover':
            kw = {'max_range': 15.0}          # default classes, PILLAR included
        else:   # nadir view: rocks + science only, terrain undulation removed (~3 m high-pass). No PILLAR: its
            # extent / height test needs a forward view (from overhead a fallen trunk is reported as ROCK)
            px_m = cam_h / cam.fx * (cam.width / d_small.shape[1])     # ground metres per depth pixel
            kw = {'max_range': 40.0, 'classes': ('ROCK', 'SCIENCE') if cam_h > 3.0 else (), 'rock_h': 0.3,
                  'tophat_px': int(np.clip(1.6 / max(px_m, 1e-3), 2, 15)),      # ~1.6 m radius kernel
                  'min_rock_px': 5, 'rock_gray_min': 0.3}
        dets = detect_objects(rgb, d_small, cam, pitch, cam_h, pose=self.pose, cam_conf=cam_conf,
                              slope_limit_deg=self.limits.max_slope_deg, use_camera=self.mode == 'FUSED', **kw)
        if self.mates:
            cam_xyz = (self.spec.get('camera_xyz', [0.45, 0.0, 0.5]) if self.rtype == 'rover'
                       else self.spec.get('camera_xyz', [0.0, 0.0, -0.08]))
            mates = self.teammates(cam, depth, pitch, cam_xyz, now)
            if mates:      # a teammate's chassis is not an obstacle: drop rock / pillar boxes centred inside its box
                def inside(d, m):
                    x, y, w, h = d.bbox
                    mx, my, mw, mh = m.bbox
                    return mx <= x + w / 2 <= mx + mw and my <= y + h / 2 <= my + mh
                dets = mates + [d for d in dets
                                if not (d.cls in OBSTACLE_CLASSES and any(inside(d, m) for m in mates))]
        st['detect'] = (time.perf_counter() - t1) * 1000
        st['lidar'] = getattr(self, '_lidar_ms', 0.0)

        t2 = time.perf_counter()
        fps = (len(self._fps) - 1) / (self._fps[-1] - self._fps[0]) if len(self._fps) > 1 else 0.0
        info = {'robot': self.robot, 'mode': self.mode, 'uncertainty': score, 'fps': fps,
                'slope_ahead_deg': self.corridor['slope_ahead_deg'] if self.rtype == 'rover' else None}
        if rgb.shape[1] < 320:   # small nadir frames: annotate at 2x so overlays stay legible in the UI
            k = 320 / rgb.shape[1]
            big = cv2.resize(rgb, (320, int(rgb.shape[0] * k)), interpolation=cv2.INTER_LINEAR)
            ann = annotate(big, [dataclasses.replace(d, bbox=tuple(v * k for v in d.bbox)) for d in dets], info)
        else:
            ann = annotate(rgb, dets, info)
        ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(ann, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            msg = CompressedImage(format='jpeg', data=jpg.tobytes())
            msg.header = rgb_msg.header
            self.pub_jpg.publish(msg)
        dvis = depth_colormap(depth, far_m=25.0 if self.rtype == 'rover' else 30.0)
        for d in dets:
            bx_, by_, bw_, bh_ = (int(v) for v in d.bbox)
            if bw_ > 0:
                cv2.rectangle(dvis, (bx_, by_), (bx_ + bw_, by_ + bh_), (255, 255, 255), 1)
        ok, djpg = cv2.imencode('.jpg', cv2.cvtColor(dvis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            dm = CompressedImage(format='jpeg', data=djpg.tobytes())
            dm.header = rgb_msg.header
            self.pub_depth.publish(dm)
        st['annotate'] = (time.perf_counter() - t2) * 1000

        out = {'robot': self.robot, 't': time.time(), 'mode': self.mode,
               'detections': [d.as_dict() for d in dets[:12]],
               'corridor': {k: (None if isinstance(v, float) and not math.isfinite(v) else
                                (round(v, 3) if isinstance(v, float) else v))
                            for k, v in self.corridor.items() if k != 'cells'},
               'uncertainty': {'score': round(score, 3), 'camera_conf': round(cam_conf, 3),
                               'lidar_conf': round(lidar_conf, 3), 'imu_vib': round(imu_vib, 3)},
               'stage_ms': {k: round(v, 2) for k, v in st.items()}, 'perception_hz': round(fps, 2),
               'lidar': self.lidar_summary()}
        self.pub_det.publish(String(data=json.dumps(out)))
        self.pub_unc.publish(Float32(data=float(score)))
        if self.rtype == 'rover':
            self.publish_scan(rgb_msg.header)

    def lidar_summary(self):
        if self.rtype != 'rover' or self.lidar_q is None:
            return None
        pts = self.scan_pts
        near = None
        if len(pts):
            a, b, c = fit_ground_plane(pts, x_max=12.0, inlier_m=0.2)
            h = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
            obs = pts[(h > 0.3) & (h < 2.5)]
            if len(obs):
                near = round(float(np.hypot(obs[:, 0], obs[:, 1]).min()), 2)
        return {'returns_pct': round(100.0 * float(self.lidar_q.get('return_ratio', 0.0)), 1), 'nearest_m': near}

    def publish_scan(self, header):
        pts = self.scan_pts
        scan = LaserScan()
        scan.header = header
        scan.header.frame_id = f'{self.robot}/base_link'
        scan.angle_min, scan.angle_max = -math.pi, math.pi - math.radians(2)
        scan.angle_increment = math.radians(2)
        scan.range_min, scan.range_max = 0.3, 25.0
        ranges = np.full(180, np.inf, np.float32)
        if len(pts):
            a, b, c = fit_ground_plane(pts, x_max=12.0, inlier_m=0.2)
            h = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
            obs = pts[(h > 0.3) & (h < 2.5)]
            if len(obs):
                ang = np.arctan2(obs[:, 1], obs[:, 0])
                rng = np.hypot(obs[:, 0], obs[:, 1])
                idx = np.clip(((ang + math.pi) / scan.angle_increment).astype(int), 0, 179)
                np.minimum.at(ranges, idx, rng.astype(np.float32))
        scan.ranges = ranges.tolist()
        self.pub_scan.publish(scan)


def main():
    rclpy.init()
    node = Perception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
