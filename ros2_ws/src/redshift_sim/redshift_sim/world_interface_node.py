"""Gazebo implementation of /world/command (rockslide spawning, pillar collapse, tip/slip jolts, simulation speed,
debris strike / critical damage, repair) via gz-transport, plus the simulated rover hardware (rover_hw, docs/protocol.md
§8) and the dust storm's wind on the drone (§10.1).

Dust/seismic anomalies are handled by sensor_effects_node in both backends; this node only touches
the physical world. Every executed command is echoed on /world/events.

It also republishes Gazebo's clock on ROS /clock, throttled to CLOCK_PERIOD_S of simulated time: gz-sim publishes
it every 4 ms step (≈500 Hz at 2×), and every use_sim_time node would otherwise wake for each one. The per-step
clock reaches this node through ros_gz_bridge (/gz/clock), NOT a Python gz-transport subscription: the Python
binding holds the GIL inside a blocking gz request, so any Python gz subscription in this process starves every
service reply (/create, /set_physics) until its timeout. Keep this node gz-publish/request only.

Rover drivetrain: the bridge feeds gz `/<rover>/cmd_vel` from ROS `/<rover>/cmd_vel_hw`, which this node publishes
for every `/<rover>/cmd_vel` message, scaled by the drivetrain health (rover_hw.Drivetrain.scale), so a damaged rover
crawls and veers exactly as in the mock. `/<rover>/hw_status` is published at 5 Hz of simulated time from the /gz/clock
callback. The debris strike is a short two-phase wrench on <rover>::base_link that never lifts a wheel (persistent
wrench, delta wrench at half time, clear; timed in simulated time like the jolt; see rover_hw); its event reports the
|a| peak measured on imu_raw.

Pillar collapse: armed and deferred like the rockslide (rockslide_due), then one dynamic basalt cylinder
(world_builder.pillar_model_sdf) is spawned beside the route at scene_events.pillar_layout's base, leaning across it;
gravity topples it. Its event carries the planned footprint (base, tip, cross).

Scout: the bridge feeds gz `/<scout>/cmd_vel` (VelocityControl, body frame) from ROS `/<scout>/cmd_vel_hw`, which this
node relays from `/<scout>/cmd_vel` plus the storm's wind drift: rover_hw.StormWind (fixed bearing, seeded gusts) at
the scout's AGL (odom z minus the terrain under it), with env_state `wind_mps` as the reference wind, rotated into the
body frame (world_builder.wind_drift_body). The drone only learns the wind from its own odometry.
"""
from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from redshift_autonomy.physics_model import RoverPhysics

from .robot_models import ROVER_SPEC, rover_com_z
from .rover_hw import (REPAIR_RANGE_M, REPAIR_S, STRIKE, Drivetrain, StormWind, hw_status_dict, peak_g_above_gravity,
                       planar_distance, solar_output, strike_wrenches_world)
from .scene_events import (JOLT, PILLAR, jolt_cut_angle_deg, jolt_wrench_world, pillar_footprint, pillar_layout,
                           rockslide_due, rockslide_layout)
from .terrain_gen import Terrain
from .world_builder import (GZ_CLOCK_TOPIC, WIND_MIN_AGL_M, cmd_topic_ros, disc_ground_z, pillar_model_sdf,
                            wind_drift_body)


CLOCK_PERIOD_S = 0.01
HW_STATUS_PERIOD_S = 0.2
PILLAR_CLEARANCE_M = 0.02     # the pillar's low base rim spawns this far above the highest ground under its base
WIND_TAU_S = 0.5              # the drone picks up a change of drift through drag over ≈ this long (as the mock does)


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
        self._sliding: set[str] = set()     # rovers with a rockslide armed (waiting for rockslide_due)
        self._pillaring: set[str] = set()   # rovers with a pillar collapse armed (same deferral)
        self._pillar = 0
        self.poses: dict[str, tuple] = {}
        self.roll: dict[str, float] = {}
        self.cut_deg = jolt_cut_angle_deg(RoverPhysics().limits(self.g).tip_alarm_deg)
        self._jolting: set[str] = set()
        self._striking: set[str] = set()
        robots = list(self.get_parameter('robots').value)
        self.rovers = [r for r in robots if r.startswith('rover')]
        self.scouts = [r for r in robots if not r.startswith('rover')]
        self.tilt: dict[str, tuple] = {}            # (roll, pitch) rad from imu_raw
        self.accel: dict[str, float] = {}           # |a| m/s² from imu_raw
        self._peak: dict[str, float] = {}           # running |a| max while a strike is being measured
        self.hw = {r: Drivetrain(ROVER_SPEC['track_m']) for r in self.rovers}
        self.hw_rng = {r: np.random.default_rng(sum(r.encode()) + 1) for r in self.rovers}
        self.cmd_in: dict[str, Twist] = {}
        self.repairs: dict[str, dict] = {}          # repair event under way, echoed again when done
        self.dust_level = 0.0
        self.env_kind: str | None = None             # 'dust_storm' | 'dust' | None (env_state)
        self.wind_mps = 0.0                           # storm wind at WIND_REF_AGL_M (env_state, ramped)
        self._storm = 0                               # storms so far: seeds each one's gusts (as the mock world)
        self.wind_field = StormWind()
        self.scout_odom: dict[str, tuple] = {}        # (x, y, z, yaw)
        self.scout_ground0: dict[str, float] = {}     # odom z at rest on the pad: the AGL datum without a terrain
        self.scout_cmd: dict[str, Twist] = {}
        self.scout_drift: dict[str, tuple] = {}       # (t, wx, wy): lagged world-frame drift last relayed
        self._next_hw: float | None = None
        self.hw_pub, self.cmd_hw_pub = {}, {}
        for r in self.rovers:
            self.create_subscription(Path, f'/{r}/planned_path', lambda m, r=r: self._on_path(r, m), 5)
            self.create_subscription(Odometry, f'/{r}/odom', lambda m, r=r: self._on_odom(r, m), 10)
            self.create_subscription(Imu, f'/{r}/imu_raw', lambda m, r=r: self._on_imu(r, m), 50)
            self.hw_pub[r] = self.create_publisher(String, f'/{r}/hw_status', QoSProfile(depth=10))
            self.cmd_hw_pub[r] = self.create_publisher(Twist, cmd_topic_ros(r), QoSProfile(depth=10))
            self.create_subscription(Twist, f'/{r}/cmd_vel', lambda m, r=r: self._on_cmd_vel(r, m),
                                     qos_profile_sensor_data)
        for s in self.scouts:
            self.create_subscription(Odometry, f'/{s}/odom', lambda m, s=s: self._on_scout_odom(s, m), 10)
            self.cmd_hw_pub[s] = self.create_publisher(Twist, cmd_topic_ros(s), QoSProfile(depth=10))
            self.create_subscription(Twist, f'/{s}/cmd_vel', lambda m, s=s: self._on_scout_cmd(s, m),
                                     qos_profile_sensor_data)
        self.create_subscription(String, '/world/env_state', self._on_env, 10)
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
        for r in self.rovers:
            if self.hw[r].update(t):
                self._repair_done(r)
        nxt = self._next_hw                           # 5 Hz of sim time, scheduled like mock_world_node._due_now
        if nxt is None or t < nxt - 2 * HW_STATUS_PERIOD_S:                    # first clock, or Gazebo reset
            self._next_hw = t + HW_STATUS_PERIOD_S
        elif t + 1e-6 >= nxt:
            self._next_hw = nxt + HW_STATUS_PERIOD_S if t - nxt < HW_STATUS_PERIOD_S else t + HW_STATUS_PERIOD_S
            for r in self.rovers:
                self._publish_hw(r)
                if self.hw[r].repairing:
                    self._relay(r)                    # health is ramping: apply it even between commands
            for s in self.scouts:
                if self.wind_mps > 0.0 or s in self.scout_drift:
                    self._relay_scout(s)              # gusts and the wind ramp act between commands too

    # ------------------------------------------------------------------ simulated rover hardware
    def _on_env(self, msg: String):
        try:
            env = json.loads(msg.data)
            dust = float(env.get('dust_level') or 0.0)
            wind = max(0.0, float(env.get('wind_mps') or 0.0))
            kind = env.get('kind')
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return
        self.dust_level = dust
        if wind > 0.0 and self.wind_mps <= 0.0:       # a new storm: its own gusts, the fixed bearing
            self._storm += 1
            self.wind_field = StormWind(seed=self._storm)
            self.get_logger().info(f'storm #{self._storm} ({kind or "wind"}): wind rising toward the scouts '
                                   f'(bearing {math.degrees(self.wind_field.toward):.0f}° ENU)')
        self.wind_mps, self.env_kind = wind, kind

    def _on_cmd_vel(self, robot: str, msg: Twist):
        self.cmd_in[robot] = msg
        self._relay(robot)

    def _relay(self, robot: str):
        """cmd_vel → cmd_vel_hw through the drivetrain (zero-latency: called for every incoming command)."""
        m = self.cmd_in.get(robot)
        if m is None:
            return
        hw = self.hw[robot]
        if not hw.damaged:
            self.cmd_hw_pub[robot].publish(m)
            return
        v, w = hw.scale(m.linear.x, m.angular.z)
        out = Twist()
        out.linear.x, out.linear.y, out.linear.z = float(v), m.linear.y, m.linear.z
        out.angular.x, out.angular.y, out.angular.z = m.angular.x, m.angular.y, float(w)
        self.cmd_hw_pub[robot].publish(out)

    # ------------------------------------------------------------------ scout: storm wind drift (protocol §10.1)
    def _on_scout_odom(self, scout: str, msg: Odometry):
        p = msg.pose.pose
        self.scout_odom[scout] = (p.position.x, p.position.y, p.position.z, yaw_of(p.orientation))
        self.scout_ground0.setdefault(scout, p.position.z)

    def _on_scout_cmd(self, scout: str, msg: Twist):
        self.scout_cmd[scout] = msg
        self._relay_scout(scout)

    def _scout_agl(self, scout: str) -> float | None:
        o = self.scout_odom.get(scout)
        if o is None:
            return None
        ground = self.terrain.height_at(o[0], o[1]) if self.terrain else self.scout_ground0[scout]
        return o[2] - ground

    def _relay_scout(self, scout: str):
        """cmd_vel → cmd_vel_hw plus the body-frame wind drift (zero-latency for every command, and 5 Hz in a storm)."""
        m = self.scout_cmd.get(scout)
        if m is None:
            return
        o, agl = self.scout_odom.get(scout), self._scout_agl(scout)
        want = (0.0, 0.0)
        if self.wind_mps > 0.0 and agl is not None and agl >= WIND_MIN_AGL_M:     # airborne: on its skids it grips
            want = self.wind_field.vector(self.sim_t, agl, self.wind_mps)
        t0, wx, wy = self.scout_drift.get(scout, (self.sim_t, 0.0, 0.0))
        k = min(1.0, max(0.0, self.sim_t - t0) / WIND_TAU_S)
        wx, wy = wx + (want[0] - wx) * k, wy + (want[1] - wy) * k
        if want == (0.0, 0.0) and math.hypot(wx, wy) < 0.05:
            self.scout_drift.pop(scout, None)         # calm again: relay unchanged
            wx = wy = 0.0
        else:
            self.scout_drift[scout] = (self.sim_t, wx, wy)
        bx, by = wind_drift_body((wx, wy), o[3], agl) if o is not None and agl is not None else (0.0, 0.0)
        if bx == 0.0 and by == 0.0:
            self.cmd_hw_pub[scout].publish(m)
            return
        out = Twist()
        out.linear.x, out.linear.y, out.linear.z = m.linear.x + bx, m.linear.y + by, m.linear.z
        out.angular.x, out.angular.y, out.angular.z = m.angular.x, m.angular.y, m.angular.z
        self.cmd_hw_pub[scout].publish(out)

    def _publish_hw(self, robot: str):
        roll, pitch = self.tilt.get(robot, (0.0, 0.0))
        yaw = self.poses.get(robot, (0.0, 0.0, 0.0))[2]
        solar_w, irr = solar_output(self.dust_level, roll, pitch, yaw)
        m = self.cmd_in.get(robot)
        v, w = (m.linear.x, m.angular.z) if m is not None else (0.0, 0.0)
        cur = self.hw[robot].motor_current(v, w, self.hw_rng[robot])
        self.hw_pub[robot].publish(String(data=json.dumps(
            hw_status_dict(robot, time.time(), self.hw[robot], solar_w, irr, cur))))

    def _on_path(self, robot, msg: Path):
        self.paths[robot] = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses]).reshape(-1, 2)

    def _on_odom(self, robot, msg: Odometry):
        p = msg.pose.pose
        self.poses[robot] = (p.position.x, p.position.y, yaw_of(p.orientation))

    def _on_imu(self, robot, msg: Imu):
        q = msg.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.roll[robot] = math.degrees(roll)
        self.tilt[robot] = (roll, pitch)
        a = msg.linear_acceleration
        an = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
        self.accel[robot] = an
        pk = self._peak.get(robot)                    # the strike thread pops it concurrently: never index it here
        if pk is not None and an > pk:
            self._peak[robot] = an

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
            robot = cmd.get('robot', 'rover_0')
            if robot not in self._sliding:                 # one armed slide per rover
                self._sliding.add(robot)
                threading.Thread(target=self._rockslide, args=(robot,), daemon=True).start()
        elif op == 'pillar_collapse':
            robot = cmd.get('robot', 'rover_0')
            if robot not in self.hw:
                self._emit({'op': 'pillar_collapse', 'ok': False, 'robot': robot, 'detail': 'unknown rover'})
            elif robot not in self._pillaring:             # one armed pillar per rover
                self._pillaring.add(robot)
                threading.Thread(target=self._pillar_collapse, args=(robot,), daemon=True).start()
        elif op == 'jolt':
            self._jolt(cmd.get('robot', 'rover_0'))
        elif op == 'critical_damage':
            self._critical_damage(cmd.get('robot', 'rover_0'))
        elif op == 'repair':
            self._repair(cmd.get('robot', 'rover_0'), cmd.get('by'))
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
        try:
            pose = self.poses.get(robot)
            if pose is None:
                self._emit({'op': 'spawn_rockslide', 'ok': False, 'robot': robot, 'detail': 'no odometry yet'})
                return
            # near its goal: wait (simulated time) for the next route; the wall guard only covers a stopped clock
            t0, w0 = self.sim_t, time.monotonic()
            while not rockslide_due(self.paths.get(robot), self.poses[robot], self.sim_t - t0) \
                    and time.monotonic() - w0 < 30.0:
                time.sleep(0.05)
            self._drop_rockslide(robot, self.poses[robot], self.sim_t - t0)
        finally:
            self._sliding.discard(robot)

    def _drop_rockslide(self, robot: str, pose, waited: float):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.entity_factory_pb2 import EntityFactory
        rocks, impact = rockslide_layout(self.paths.get(robot), pose, seed=self._slide)
        self._slide += 1
        ok_all = True
        for k, (x, y, r) in enumerate(rocks):
            ground = self.terrain.height_at(x, y) if self.terrain else 0.0
            req = EntityFactory(sdf=rock_model_sdf(f'rockslide_{self._slide}_{k}', x, y, ground + 4.0 + 0.4 * k, r,
                                                   k * 0.9), allow_renaming=True)
            ok, rep = self.gz.request(f'/world/{self.world}/create', req, EntityFactory, Boolean, 2000)
            ok_all &= bool(ok and rep.data)
        self.get_logger().info(f'rockslide #{self._slide} across {robot} path at {impact}: ok={ok_all}'
                               + (f' (waited {waited:.1f} s for the next route)' if waited > 0 else ''))
        self._emit({'op': 'spawn_rockslide', 'ok': ok_all, 'robot': robot, 'impact': list(impact),
                    'rocks': [list(r) for r in rocks], 'waited_s': round(waited, 2)})

    def _pillar_collapse(self, robot: str):
        try:
            if self.poses.get(robot) is None:
                self._emit({'op': 'pillar_collapse', 'ok': False, 'robot': robot, 'detail': 'no odometry yet'})
                return
            # like the rockslide: near its goal, wait (simulated time) for the next route; the wall guard only covers
            # a stopped clock
            t0, w0 = self.sim_t, time.monotonic()
            while not rockslide_due(self.paths.get(robot), self.poses[robot], self.sim_t - t0) \
                    and time.monotonic() - w0 < 30.0:
                time.sleep(0.05)
            self._drop_pillar(robot, self.poses[robot], self.sim_t - t0)
        except Exception as e:  # noqa: BLE001 - the inject must always answer
            self.get_logger().error(f'pillar collapse for {robot} failed: {e}')
            self._emit({'op': 'pillar_collapse', 'ok': False, 'robot': robot, 'detail': f'pillar failed: {e}'})
        finally:
            self._pillaring.discard(robot)

    def _drop_pillar(self, robot: str, pose, waited: float):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.entity_factory_pb2 import EntityFactory
        lay = pillar_layout(self.paths.get(robot), pose, seed=self._pillar)
        self._pillar += 1
        (bx, by), tip = pillar_footprint(lay)
        r, h = lay['radius_m'], lay['height_m']
        ground = disc_ground_z(self.terrain.height_at, bx, by, r) if self.terrain else 0.0
        sdf = pillar_model_sdf(f'pillar_{self._pillar}', (bx, by, ground + PILLAR_CLEARANCE_M), lay['fall_dir'],
                               PILLAR['tilt0_deg'], h, r, PILLAR['mass_kg'], PILLAR['albedo_rgb'])
        ok, rep = self.gz.request(f'/world/{self.world}/create', EntityFactory(sdf=sdf, allow_renaming=True),
                                  EntityFactory, Boolean, 2000)
        ok = bool(ok and rep.data)
        self.get_logger().info(f'pillar #{self._pillar} beside {robot} route at ({bx:.1f}, {by:.1f}) toppling across '
                               f'{lay["cross"]}: ok={ok}' + (f' (waited {waited:.1f} s for the next route)'
                                                             if waited > 0 else ''))
        ev = {'op': 'pillar_collapse', 'ok': ok, 'robot': robot, 'base': [bx, by], 'tip': list(tip),
              'cross': list(lay['cross']), 'height_m': h, 'radius_m': r, 'waited_s': round(waited, 2)}
        if not ok:
            ev['detail'] = 'gz create request failed'
        self._emit(ev)

    def _jolt(self, robot: str):
        if robot in self._jolting or robot in self._striking:
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


    # ------------------------------------------------------------------ debris strike / repair (protocol §8.5)
    def _wrench_msg(self, robot: str, force, torque, offset=None):
        """EntityWrench on <robot>::base_link; `offset` = point of application in the link frame (gz applies the
        force at the link origin otherwise)."""
        ent = self._Entity()
        ent.name = f'{robot}::base_link'
        ent.type = self._Entity.LINK
        msg = self._EntityWrench()
        msg.entity.CopyFrom(ent)
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = (float(v) for v in force)
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = (float(v) for v in torque)
        if offset is not None:
            msg.wrench.force_offset.x, msg.wrench.force_offset.y, msg.wrench.force_offset.z = (float(v) for v in offset)
        return ent, msg

    def _critical_damage(self, robot: str):
        if robot not in self.hw:
            self._emit({'op': 'critical_damage', 'ok': False, 'robot': robot, 'detail': 'unknown rover'})
            return
        pose = self.poses.get(robot)
        if pose is None:
            self._emit({'op': 'critical_damage', 'ok': False, 'robot': robot, 'detail': 'no odometry yet'})
            return
        if robot in self._striking or robot in self._jolting:
            self._emit({'op': 'critical_damage', 'ok': False, 'robot': robot,
                        'detail': 'strike already under way' if robot in self._striking else 'jolt under way'})
            return
        self._striking.add(robot)
        self.hw[robot].apply_critical_damage()
        self.repairs.pop(robot, None)                 # a strike cancels any repair in progress
        self._relay(robot)
        threading.Thread(target=self._strike_loop, args=(robot, pose), daemon=True).start()

    def _strike_loop(self, robot: str, pose):
        """Two-phase strike wrench over STRIKE['gz_duration_s'] of sim time, then the IMU peak over gz_measure_s."""
        ent = None
        try:
            roll, pitch = self.tilt.get(robot, (0.0, 0.0))
            (f1, t1), (f2, t2) = strike_wrenches_world(roll, pitch, pose[2], self.g, ROVER_SPEC['mass_kg'])
            com = (0.0, 0.0, rover_com_z(ROVER_SPEC))         # whole rover (chassis + wheels), base_link frame
            ent, first = self._wrench_msg(robot, f1, t1, com)  # at the centre of mass: the push does not roll it
            _, switch = self._wrench_msg(robot, f2, t2, com)
            self._peak[robot] = self.accel.get(robot, 0.0)
            dur = STRIKE['gz_duration_s']
            t0, w0 = self.sim_t, time.monotonic()
            self.wrench_pub.publish(first)
            # timed in simulated time; the wall-clock guards only matter if Gazebo stops publishing its clock
            while self.sim_t - t0 < dur / 2 and time.monotonic() - w0 < 5.0:
                time.sleep(0.0005)
            self.wrench_pub.publish(switch)           # persistent wrenches accumulate: this flips the lateral push
            t_switch = self.sim_t - t0
            # second half as long as the first really was (a late switch only lengthens the pulse): net Δv ≈ 0
            while self.sim_t - t0 < max(dur, 2.0 * t_switch) and time.monotonic() - w0 < 5.0:
                time.sleep(0.0005)
            for _ in range(3):
                self.clear_pub.publish(ent)
            pulse = self.sim_t - t0
            while self.sim_t - t0 < STRIKE['gz_measure_s'] and time.monotonic() - w0 < 10.0:
                time.sleep(0.005)
            peak = self._peak.pop(robot, 0.0)
            peak_g = peak_g_above_gravity(peak, self.g)
            health = [round(h, 3) for h in self.hw[robot].wheel_health]
            self.get_logger().info(f'debris strike on {robot}: force {np.round(f1, 0)} N, then {np.round(f2, 0)} N '
                                   f'more at {t_switch:.3f} s, cleared at {pulse:.3f} s sim; IMU peak '
                                   f'{peak / 9.80665:.2f} g ({peak_g:.2f} g above gravity), wheel health {health}')
            self._emit({'op': 'critical_damage', 'ok': True, 'robot': robot,
                        'impact': [round(float(pose[0]), 3), round(float(pose[1]), 3)], 'peak_g': peak_g,
                        'wheel_health': health})
        except Exception as e:  # noqa: BLE001 - never leave a persistent wrench on the rover
            if ent is not None:
                for _ in range(3):
                    self.clear_pub.publish(ent)
            self.get_logger().error(f'debris strike on {robot} failed: {e}')
            self._emit({'op': 'critical_damage', 'ok': False, 'robot': robot, 'detail': f'strike failed: {e}'})
        finally:
            self._peak.pop(robot, None)
            self._striking.discard(robot)

    def _repair(self, robot: str, by):
        ev = {'op': 'repair', 'robot': robot, 'by': by}
        if robot not in self.hw:
            self._emit(dict(ev, ok=False, detail='unknown rover'))
            return
        if by not in self.hw or by == robot:
            self._emit(dict(ev, ok=False, detail=f'unknown repairing rover {by!r}'))
            return
        a, b = self.poses.get(robot), self.poses.get(by)
        if a is None or b is None:
            self._emit(dict(ev, ok=False, detail='no odometry yet'))
            return
        dist = round(planar_distance(a, b), 2)
        ev['distance_m'] = dist
        if dist > REPAIR_RANGE_M:
            self._emit(dict(ev, ok=False, detail=f'{by} is {dist:.2f} m from {robot} '
                                                 f'(repair needs <= {REPAIR_RANGE_M:.1f} m)'))
            return
        if not self.hw[robot].start_repair(self.sim_t, REPAIR_S):
            self._emit(dict(ev, ok=True, done=False, detail='repair already in progress'))
            return
        self.repairs[robot] = dict(ev, ok=True)
        self.get_logger().info(f'repair of {robot} by {by} started at {dist:.2f} m')
        self._emit(dict(ev, ok=True, done=False, detail=f'repair started: drivetrain restoring over {REPAIR_S:.1f} s'))

    def _repair_done(self, robot: str):
        ev = dict(self.repairs.pop(robot, None) or {'op': 'repair', 'ok': True, 'robot': robot})
        ev.update(done=True, wheel_health=list(self.hw[robot].wheel_health), detail='drivetrain restored',
                  t=time.time())
        self._relay(robot)
        self.get_logger().info(f'repair of {robot} complete')
        self._emit(ev)


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
