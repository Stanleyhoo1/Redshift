"""Gazebo implementation of /world/command (rockslide spawning, tip/slip jolts, simulation speed) via gz-transport.

Dust/seismic anomalies are handled by sensor_effects_node in both backends; this node only touches
the physical world. Every executed command is echoed on /world/events.

It also republishes Gazebo's clock on ROS /clock, throttled to CLOCK_PERIOD_S of simulated time: gz-sim publishes
it every 4 ms step (≈500 Hz at 2×), and every use_sim_time node would otherwise wake for each one. The per-step
clock reaches this node through ros_gz_bridge (/gz/clock), NOT a Python gz-transport subscription: the Python
binding holds the GIL inside a blocking gz request, so any Python gz subscription in this process starves every
service reply (/create, /set_physics) until its timeout. Keep this node gz-publish/request only.
"""
from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from redshift_autonomy.physics_model import RoverPhysics

from .scene_events import JOLT, jolt_cut_angle_deg, jolt_wrench_world, rockslide_layout
from .terrain_gen import Terrain
from .world_builder import GZ_CLOCK_TOPIC


CLOCK_PERIOD_S = 0.01


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def rock_model_sdf(name: str, x: float, y: float, z: float, r: float, yaw: float) -> str:
    h = r * 1.3
    return f'''<sdf version="1.10"><model name="{name}">
  <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>
  <link name="link">
    <inertial><mass>400</mass><inertia><ixx>60</ixx><iyy>60</iyy><izz>60</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
    <collision name="c"><geometry><box><size>{1.5 * r:.3f} {1.3 * r:.3f} {h:.3f}</size></box></geometry>
      <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>
    <visual name="v"><geometry><ellipsoid><radii>{r:.3f} {0.85 * r:.3f} {h / 2:.3f}</radii></ellipsoid></geometry>
      <material><ambient>0.25 0.24 0.24 1</ambient><diffuse>0.40 0.39 0.38 1</diffuse><specular>0.1 0.1 0.1 1</specular></material>
    </visual>
  </link>
</model></sdf>'''


class WorldInterface(Node):
    def __init__(self):
        super().__init__('world_interface')
        self.declare_parameter('world_name', 'redshift')
        self.declare_parameter('terrain_path', '')
        self.declare_parameter('g_mult', 2.2)
        self.declare_parameter('robots', ['rover_0'])
        self.declare_parameter('max_step_s', 0.004)
        self.step_s = float(self.get_parameter('max_step_s').value)
        self.world = self.get_parameter('world_name').value
        self.g = float(self.get_parameter('g_mult').value)
        tp = self.get_parameter('terrain_path').value
        self.terrain = Terrain.load(tp) if tp else None
        from gz.transport13 import Node as GzNode
        from gz.msgs10.entity_wrench_pb2 import EntityWrench
        from gz.msgs10.entity_pb2 import Entity
        self._EntityWrench, self._Entity = EntityWrench, Entity
        self.gz = GzNode()
        self.wrench_pub = self.gz.advertise(f'/world/{self.world}/wrench/persistent', EntityWrench)
        self.clear_pub = self.gz.advertise(f'/world/{self.world}/wrench/clear', Entity)
        self.sim_t = 0.0                   # Gazebo simulated time, s (jolt timing)
        self._last_clock = -1.0
        self.clock_pub = self.create_publisher(Clock, '/clock', QoSProfile(depth=10))
        self.create_subscription(Clock, GZ_CLOCK_TOPIC, self._on_gz_clock, QoSProfile(depth=50))
        self.paths: dict[str, np.ndarray] = {}
        self.poses: dict[str, tuple] = {}
        self.roll: dict[str, float] = {}
        self.cut_deg = jolt_cut_angle_deg(RoverPhysics().limits(self.g).tip_alarm_deg)
        self._jolting: set[str] = set()
        for r in self.get_parameter('robots').value:
            if r.startswith('rover'):
                self.create_subscription(Path, f'/{r}/planned_path', lambda m, r=r: self._on_path(r, m), 5)
                self.create_subscription(Odometry, f'/{r}/odom', lambda m, r=r: self._on_odom(r, m), 10)
                self.create_subscription(Imu, f'/{r}/imu_raw', lambda m, r=r: self._on_imu(r, m), 50)
        self.events = self.create_publisher(String, '/world/events', 10)
        self.create_subscription(String, '/world/command', self._on_cmd, 10)
        self._slide = 0
        self.get_logger().info(f'world interface ready for world "{self.world}" (g={self.g})')

    def _on_gz_clock(self, msg: Clock):
        t = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.sim_t = t
        if t - self._last_clock >= CLOCK_PERIOD_S - 1e-9 or t < self._last_clock:
            self._last_clock = t
            self.clock_pub.publish(msg)

    def _on_path(self, robot, msg: Path):
        self.paths[robot] = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses]).reshape(-1, 2)

    def _on_odom(self, robot, msg: Odometry):
        p = msg.pose.pose
        self.poses[robot] = (p.position.x, p.position.y, yaw_of(p.orientation))

    def _on_imu(self, robot, msg: Imu):
        q = msg.orientation
        self.roll[robot] = math.degrees(math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)))

    def _emit(self, ev: dict):
        ev.setdefault('t', time.time())
        self.events.publish(String(data=json.dumps(ev)))

    def _on_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        op = cmd.get('op')
        if op == 'spawn_rockslide':
            threading.Thread(target=self._rockslide, args=(cmd.get('robot', 'rover_0'),), daemon=True).start()
        elif op == 'jolt':
            self._jolt(cmd.get('robot', 'rover_0'))
        elif op == 'set_speed':
            threading.Thread(target=self._set_speed, args=(cmd.get('speed', 1.0),), daemon=True).start()

    def _set_speed(self, speed):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.physics_pb2 import Physics
        try:
            speed = min(4.0, max(0.1, float(speed)))
        except (TypeError, ValueError):
            self._emit({'op': 'set_speed', 'ok': False, 'detail': f'bad speed {speed!r}'})
            return
        req = Physics(max_step_size=self.step_s, real_time_factor=speed)
        ok, rep = self.gz.request(f'/world/{self.world}/set_physics', req, Physics, Boolean, 3000)
        ok = bool(ok and rep.data)
        self.get_logger().info(f'simulation speed → {speed:.2f}x: {"ok" if ok else "set_physics failed"}')
        self._emit({'op': 'set_speed', 'ok': ok, 'speed': speed})

    def _rockslide(self, robot: str):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.entity_factory_pb2 import EntityFactory
        pose = self.poses.get(robot)
        if pose is None:
            self._emit({'op': 'spawn_rockslide', 'ok': False, 'robot': robot, 'detail': 'no odometry yet'})
            return
        rocks, impact = rockslide_layout(self.paths.get(robot), pose, seed=self._slide)
        self._slide += 1
        ok_all = True
        for k, (x, y, r) in enumerate(rocks):
            ground = self.terrain.height_at(x, y) if self.terrain else 0.0
            req = EntityFactory(sdf=rock_model_sdf(f'rockslide_{self._slide}_{k}', x, y, ground + 4.0 + 0.4 * k, r,
                                                   k * 0.9), allow_renaming=True)
            ok, rep = self.gz.request(f'/world/{self.world}/create', req, EntityFactory, Boolean, 2000)
            ok_all &= bool(ok and rep.data)
        self.get_logger().info(f'rockslide #{self._slide} across {robot} path at {impact}: ok={ok_all}')
        self._emit({'op': 'spawn_rockslide', 'ok': ok_all, 'robot': robot, 'impact': list(impact),
                    'rocks': [list(r) for r in rocks]})

    def _jolt(self, robot: str):
        if robot in self._jolting:
            return
        self._jolting.add(robot)
        threading.Thread(target=self._jolt_loop, args=(robot,), daemon=True).start()

    def _jolt_loop(self, robot: str):
        """Subsidence: roll torque until the *true* roll reaches the cut-off, then release."""
        pose = self.poses.get(robot, (0.0, 0.0, 0.0))
        force, torque = jolt_wrench_world(pose[2], self.g)
        ent = self._Entity()
        ent.name = f'{robot}::base_link'
        ent.type = self._Entity.LINK
        msg = self._EntityWrench()
        msg.entity.CopyFrom(ent)
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = force
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = torque
        roll0 = self.roll.get(robot, 0.0)
        t0, w0 = self.sim_t, time.monotonic()
        self.wrench_pub.publish(msg)
        peak = 0.0
        # the cut-off is in simulated time; the wall-clock guard only matters if Gazebo stops publishing its clock
        while self.sim_t - t0 < JOLT['max_duration_s'] and time.monotonic() - w0 < 10.0:
            r = abs(self.roll.get(robot, 0.0))
            peak = max(peak, r)
            if r >= self.cut_deg:
                break
            time.sleep(0.002)
        for _ in range(3):
            self.clear_pub.publish(ent)
        self.get_logger().info(f'jolt {robot}: torque {np.round(torque, 0)} released at roll {peak:.1f} '
                               f'(cut {self.cut_deg:.1f}, start {roll0:.1f}) after {self.sim_t - t0:.2f} s sim')
        self._emit({'op': 'jolt', 'ok': True, 'robot': robot, 'release_roll_deg': round(peak, 1)})
        self._jolting.discard(robot)


def main():
    rclpy.init()
    node = WorldInterface()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
