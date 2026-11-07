"""Atmospheric / seismic environment model applied to raw sensor streams (both world backends).

Clear conditions: zero-copy passthrough  *_raw → clean topic names.
Dust anomaly (level L ramps in/out over ~1.5 s):
  RGB   haze toward dust colour (0.65·L), blur σ = 1 + 3L, sensor noise σ = 25L, drifting particles
  depth visibility clip at 4 + 36(1 − L) m, 40·L % dropout, range noise
  LiDAR 25·L % dropout, range noise σ = 0.08·L m, 2·L % near-field dust backscatter returns
Seismic: accelerometer/gyro vibration on the IMU.
The autonomy stack never reads the dust flag — it must infer degradation from the data itself.
"""
from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import String

DUST_RGB = np.array([170, 120, 80], np.float32)   # RGB order used on rgb8 images below
RAMP_S = 1.5


class SensorEffects(Node):
    def __init__(self):
        super().__init__('sensor_effects')
        self.declare_parameter('robot', 'rover_0')
        self.declare_parameter('robot_type', 'rover')
        self.declare_parameter('publish_env', False)
        self.robot = self.get_parameter('robot').value
        rtype = self.get_parameter('robot_type').value
        self.publish_env = bool(self.get_parameter('publish_env').value)
        self.rng = np.random.default_rng(abs(hash(self.robot)) % 2 ** 32)
        r = self.robot
        self.target = 0.0
        self.level = 0.0
        self.until = 0.0
        self.seismic_until = 0.0
        self._last_t = time.monotonic()
        self.pub_img = self.create_publisher(Image, f'/{r}/camera/image', qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, f'/{r}/camera/depth', qos_profile_sensor_data)
        self.pub_imu = self.create_publisher(Imu, f'/{r}/imu', qos_profile_sensor_data)
        self.create_subscription(Image, f'/{r}/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(Image, f'/{r}/camera/depth_raw', self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Imu, f'/{r}/imu_raw', self.on_imu, qos_profile_sensor_data)
        if rtype == 'rover':
            self.pub_pts = self.create_publisher(PointCloud2, f'/{r}/lidar/points', qos_profile_sensor_data)
            self.create_subscription(PointCloud2, f'/{r}/lidar/points_raw', self.on_points, qos_profile_sensor_data)
        self.create_subscription(String, '/world/command', self.on_cmd, 10)
        self.env_pub = self.create_publisher(String, '/world/env_state', 10)
        self.events = self.create_publisher(String, '/world/events', 10)
        self.create_timer(0.1, self.tick)
        self._particles = self.rng.random((60, 4)).astype(np.float32)

    # ------------------------------------------------------------------ control
    def on_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        now = time.time()
        if cmd.get('op') == 'anomaly':
            self.target = float(np.clip(cmd.get('dust_level', 0.85), 0.0, 1.0))
            self.until = now + float(cmd.get('duration_s', 25.0))
            self.seismic_until = now + float(cmd.get('seismic_s', 4.0))
            if self.publish_env:
                self.events.publish(String(data=json.dumps({'op': 'anomaly', 'ok': True, 'dust_level': self.target,
                                                            'until_t': self.until, 't': now})))
        elif cmd.get('op') == 'clear_anomaly':
            self.until = now
            self.seismic_until = now

    def tick(self):
        now = time.time()
        t = time.monotonic()
        dt = t - self._last_t
        self._last_t = t
        goal = self.target if now < self.until else 0.0
        step = dt / RAMP_S
        self.level = min(goal, self.level + step) if goal > self.level else max(goal, self.level - step)
        if self.publish_env:
            self.env_pub.publish(String(data=json.dumps({'dust_level': round(self.level, 3),
                                                         'seismic': now < self.seismic_until,
                                                         'until_t': self.until if now < self.until else None})))

    # ------------------------------------------------------------------ streams
    def on_image(self, msg: Image):
        L = self.level
        if L < 0.02 or msg.encoding not in ('rgb8', 'bgr8'):
            self.pub_img.publish(msg)
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).astype(np.float32)
        dust = DUST_RGB if msg.encoding == "rgb8" else DUST_RGB[::-1]
        img = img * (1 - 0.65 * L) + dust * (0.65 * L)
        img = cv2.GaussianBlur(img, (0, 0), 1.0 + 3.0 * L)
        img += self.rng.normal(0, 25.0 * L, img.shape).astype(np.float32)
        h, w = img.shape[:2]
        p = self._particles
        p[:, 0] = (p[:, 0] + 0.02 + 0.03 * p[:, 2]) % 1.0
        p[:, 1] = (p[:, 1] + 0.004 * (p[:, 3] - 0.5)) % 1.0
        for k in range(int(60 * L)):
            x0, y0 = int(p[k, 0] * w), int(p[k, 1] * h)
            cv2.line(img, (x0, y0), (x0 + int(4 + 8 * p[k, 2]), y0 + 1), (235, 205, 165), 1, cv2.LINE_AA)
        out = Image(header=msg.header, height=msg.height, width=msg.width, encoding=msg.encoding,
                    is_bigendian=msg.is_bigendian, step=msg.step)
        out.data = np.clip(img, 0, 255).astype(np.uint8).tobytes()
        self.pub_img.publish(out)

    def on_depth(self, msg: Image):
        L = self.level
        if L < 0.02 or msg.encoding != '32FC1':
            self.pub_depth.publish(msg)
            return
        d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
        vis = 4.0 + 36.0 * (1 - L)
        d[d > vis] = np.nan
        d[self.rng.random(d.shape) < 0.4 * L] = np.nan
        d += (self.rng.normal(0, 0.02 * L, d.shape) * np.nan_to_num(d, nan=0.0)).astype(np.float32)
        out = Image(header=msg.header, height=msg.height, width=msg.width, encoding=msg.encoding,
                    is_bigendian=msg.is_bigendian, step=msg.step)
        out.data = d.astype(np.float32).tobytes()
        self.pub_depth.publish(out)

    def on_points(self, msg: PointCloud2):
        L = self.level
        if L < 0.02:
            self.pub_pts.publish(msg)
            return
        offs = {f.name: f.offset for f in msg.fields}
        stride = msg.point_step // 4
        arr = np.frombuffer(msg.data, np.float32).reshape(-1, stride).copy()
        xi, yi, zi = offs['x'] // 4, offs['y'] // 4, offs['z'] // 4
        xyz = arr[:, [xi, yi, zi]]
        r = np.linalg.norm(xyz, axis=1)
        fin = np.isfinite(r) & (r > 0)
        scale = np.ones_like(r)
        scale[fin] = (r[fin] + self.rng.normal(0, 0.08 * L, int(fin.sum()))) / r[fin]
        n = len(r)
        back = self.rng.random(n) < 0.02 * L
        dirs = np.where(fin[:, None], xyz / np.where(fin, r, 1.0)[:, None], 0.0)
        xyz = xyz * scale[:, None]
        if back.any():
            rr = self.rng.uniform(0.4, 2.0, int(back.sum()))
            # rays without a return still have a direction implied by their organized index; use a random one
            d_back = dirs[back]
            empty = ~np.any(d_back, axis=1)
            if empty.any():
                az = self.rng.uniform(-math.pi, math.pi, int(empty.sum()))
                d_back[empty] = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)
            xyz[back] = d_back * rr[:, None]
        drop = self.rng.random(n) < 0.25 * L
        xyz[drop] = np.inf
        arr[:, [xi, yi, zi]] = xyz
        out = PointCloud2(header=msg.header, height=msg.height, width=msg.width, fields=msg.fields,
                          is_bigendian=msg.is_bigendian, point_step=msg.point_step, row_step=msg.row_step,
                          is_dense=False)
        out.data = arr.tobytes()
        self.pub_pts.publish(out)

    def on_imu(self, msg: Imu):
        if time.time() >= self.seismic_until:
            self.pub_imu.publish(msg)
            return
        t = time.time()
        a = msg.linear_acceleration
        w = msg.angular_velocity
        n = self.rng.normal(0, 1.0, 6)
        s = math.sin(2 * math.pi * 6.0 * t)
        a.x += 3.0 * n[0] + 4.0 * s
        a.y += 3.0 * n[1] + 3.0 * math.cos(2 * math.pi * 7.3 * t)
        a.z += 3.0 * n[2]
        w.x += 0.3 * n[3]
        w.y += 0.3 * n[4]
        w.z += 0.15 * n[5]
        self.pub_imu.publish(msg)


def main():
    rclpy.init()
    node = SensorEffects()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
