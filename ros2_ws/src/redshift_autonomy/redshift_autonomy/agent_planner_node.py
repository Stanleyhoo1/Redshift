"""Per-robot agent: rover (FSM + frontier selection + gravity-aware A* + pure pursuit + safe-hold recovery)
or scout (aerial survey + altitude hold + hazard scans). Publishes cmd_vel, planned_path, agent_state,
edge_telemetry, fleet coordination messages and the traceable decision log."""
from __future__ import annotations

import collections
import json
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String

from . import grid_map as gm
from .agent_fsm import (DOCK_M, ImuWindow, ReplanGate, RoverFSM, RoverInputs, ScoutFSM, ScoutInputs, display_name,
                        role_of, unit_label)
from .brain_bridge import (CAPPED_DRIVES, BrainAltitude, BrainCap, DropGuard, altitude_trace, cap_trace,
                           parse_decision)
from .edge_budget import PROFILES, EdgeBudget
from .grid_map import slope_from_elevation
from .physics_model import RoverPhysics
from .rover_health import DroneHealth, RoverHealth
from .planning import (RING, add_keepout, astar, build_costmap, cell_to_world, cells_to_world, clearance_mask,
                       find_frontiers, first_block, path_conflicts, path_length_m, pure_pursuit, reachable_from,
                       score_frontiers, standoff_cells, world_to_cell, wrap)
from .ros_utils import image_to_numpy, quat_to_rpy

MAP_QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
LOOP_HZ = 10.0
BLOCK_LOOKAHEAD_M = 12.0     # lethal cells on the path closer than this = blocked (visible re-plan)
BLOCK_SKIP_M = 1.0           # ignore the path segment under the robot
REPLAN_PERIOD_S = 4.0        # quiet refresh of the route as the map grows
LETHAL_CLASSES = (gm.HAZARD, gm.PERMANENT, gm.OBSTACLE)
BORDER_M = 3.0               # sector edge keep-out (the heightmap ends there)
SCOUT_LEASH_M = 40.0         # scout surveys within reach of the rovers it supports
STANDOFF_M = 2.2             # rescue A* goal: preferred distance from the damaged rover's centre (2.0–2.45 m band)
RESCUE_TRIES = 4             # A* attempts per rescue plan, best stand-off cells first
PEER_DISC_M = 0.9            # teammates are lethal discs in the planning cost map (the map filters their returns)
PEER_FRESH_S = 2.0
SOS_FRESH_S = 4.0
DOCK_CLEAR_M = 0.9           # a LiDAR corridor return this much short of the teammate's centre is something else in
                             # the way: the docking creep stops (the teammate's own near face is ≈0.45–0.6 m short).
                             # The corridor ignores returns within 1.0 m, so the last ≈0.45 m of the creep is unguarded.
START_BATTERY = {'rover_0': 82.0, 'rover_1': 91.0}
# advisory brain (/brain/decision, fleet_brain_node): a cap only ever lowers the local speed (brain_bridge.BrainCap)
BRAIN_REPLAN_STATES = ('EXPLORING', 'CAUTION', 'RESCUE_TRANSIT')   # SLOW_AND_REPLAN → quiet re-plan in these
BRAIN_AGL_STATES = ('SURVEY', 'FLY_TO_SCAN')   # scout states that fly an advisory cruise AGL (never take-off / scan)
AGL_HOLD_STATES = BRAIN_AGL_STATES + ('SCANNING',)   # ... plus a scan, which holds the AGL it started at


class MapView:
    """Planner-side copy of /planet_map/layers + /planet_map/elevation."""

    def __init__(self, size_m: float, res: float):
        self.size_m, self.res = size_m, res
        self.n = int(round(size_m / res))
        self.origin = (-size_m / 2.0, -size_m / 2.0)
        self.cls = np.zeros((self.n, self.n), np.uint8)
        self.elev = np.full((self.n, self.n), np.nan, np.float32)
        self.version = 0
        self._slope = None
        self._slope_v = -1

    @property
    def ready(self) -> bool:
        return self.version > 0

    def set_layers(self, img: np.ndarray):
        self.cls = np.rint(img[..., 1].astype(np.float32) / 40.0).astype(np.uint8)
        self.version += 1

    @property
    def known(self):
        return self.cls != gm.UNKNOWN

    @property
    def lethal(self):
        return np.isin(self.cls, LETHAL_CLASSES)

    @property
    def caution(self):
        return self.cls == gm.CAUTION

    @property
    def slope(self):
        if self._slope_v != self.version:
            self._slope = slope_from_elevation(self.elev, self.res)
            self._slope_v = self.version
        return self._slope

    def cell(self, x, y):
        return world_to_cell(x, y, self.res, self.origin, self.cls.shape)

    def xy(self, rc):
        return cell_to_world(rc, self.res, self.origin)


class AgentBase(Node):
    def now_s(self) -> float:
        """Simulated seconds (use_sim_time). Every duration, timeout and rate in the agents uses this, so behaviour
        is identical at any simulation speed; message 't' fields stay wall-clock epoch for the UI."""
        return self.get_clock().now().nanoseconds * 1e-9

    def __init__(self, name: str):
        super().__init__(name)
        for k, v in (('robot', 'rover_0'), ('robot_type', 'rover'), ('spec_json', '{}'), ('robots', ['rover_0']),
                     ('g_mult', 2.2), ('edge_profile', 'orin_nano_15w'), ('size_m', 100.0), ('map_res', 0.5)):
            self.declare_parameter(k, v)
        self.robot = self.get_parameter('robot').value
        self.role, self.name = role_of(self.robot), display_name(self.robot)
        self.edge_power_w = 0.0
        self.spec = json.loads(self.get_parameter('spec_json').value or '{}')
        self.robots = list(self.get_parameter('robots').value)
        self.g = float(self.get_parameter('g_mult').value)
        self.physics = RoverPhysics()
        self.limits = self.physics.limits(self.g)
        self.profile = PROFILES[self.get_parameter('edge_profile').value]
        self.edge = EdgeBudget(self.profile.id)
        self.map = MapView(float(self.get_parameter('size_m').value), float(self.get_parameter('map_res').value))
        r = self.robot
        self.pose = None                        # (x, y, yaw)
        self.z = 0.0
        self.det = None
        self.det_t = 0.0
        self.map_stats: dict = {}
        self.planner_ms = 0.0
        self.claims: dict = {}
        self._seen_science: list = []
        self.create_subscription(Odometry, f'/{r}/odom', self.on_odom, SENSOR_QOS)
        self.create_subscription(String, f'/{r}/perception/detections', self.on_det, 10)
        self.create_subscription(Image, '/planet_map/layers', self.on_layers, MAP_QOS)
        self.create_subscription(Image, '/planet_map/elevation', self.on_elev, MAP_QOS)
        self.create_subscription(String, '/planet_map/stats', self.on_stats, MAP_QOS)
        self.create_subscription(String, '/fleet/claims', self.on_claim, 10)
        self.pub_cmd = self.create_publisher(Twist, f'/{r}/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, f'/{r}/planned_path', 10)
        self.pub_state = self.create_publisher(String, f'/{r}/agent_state', 10)
        self.pub_edge = self.create_publisher(String, f'/{r}/edge_telemetry', 10)
        self.pub_trace = self.create_publisher(String, '/agent_decision_trace', 50)
        self.create_timer(0.5, self.publish_edge)

    # ------------------------------------------------------------------ common inputs
    def on_odom(self, m: Odometry):
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, quat_to_rpy(p.orientation)[2])
        self.z = p.position.z
        self.odom_twist = m.twist.twist

    def on_det(self, m: String):
        try:
            self.det = json.loads(m.data)
            self.det_t = self.now_s()
        except json.JSONDecodeError:
            return
        self._science_traces()

    def on_layers(self, m: Image):
        self.map.set_layers(image_to_numpy(m))

    def on_elev(self, m: Image):
        self.map.elev = image_to_numpy(m).copy()

    def on_stats(self, m: String):
        try:
            self.map_stats = json.loads(m.data)
        except json.JSONDecodeError:
            pass

    def on_claim(self, m: String):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        if d.get('robot') != self.robot and d.get('goal'):
            self.claims[d['robot']] = (tuple(d['goal']), self.now_s())

    def other_claims(self):
        now = self.now_s()
        return [g for r, (g, t) in self.claims.items() if now - t < 15.0]

    # ------------------------------------------------------------------ outputs
    def emit(self, traces):
        for t in traces:
            if t:
                self.pub_trace.publish(String(data=json.dumps(t)))
                self.get_logger().info(f"[{t['state']}] {t['kind']}: {t['input']} -> {t['action']}")

    def tb_state(self) -> str:
        return self.fsm.state

    def trace(self, kind, sensors, inp, reasoning, conf, action, severity='info', latency_ms=None, force=False):
        t = self.fsm.tb.make(kind, self.tb_state(), sensors, inp, reasoning, conf, action, severity,
                             latency_ms=latency_ms, now=self.now_s(), force=force)
        if t:
            self.emit([t])

    def send_cmd(self, vx=0.0, wz=0.0, vy=0.0, vz=0.0):
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z = float(vx), float(vy), float(vz)
        t.angular.z = float(wz)
        self.pub_cmd.publish(t)

    def publish_path(self, xy):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        for x, y in (xy if xy is not None else []):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def perception_ms(self) -> tuple[float, float]:
        if not self.det or self.now_s() - self.det_t > 3.0:
            return 0.0, 0.0
        return float(sum(self.det.get('stage_ms', {}).values())), float(self.det.get('perception_hz', 0.0))

    def publish_edge(self):
        p_ms, p_hz = self.perception_ms()
        m_ms = float(self.map_stats.get('mapping_ms', {}).get(self.robot, 0.0))
        host = {'perception': p_ms, 'planner': self.planner_ms, 'mapping': m_ms}
        hz = {'perception': p_hz, 'planner': LOOP_HZ, 'mapping': self.mapping_hz}
        st = self.edge.update(host, hz, now=self.now_s())
        self.edge_power_w = float(st.power_w)
        unc = (self.det or {}).get('uncertainty', {}).get('score')
        out = {'robot': self.robot, 'profile': st.profile, 'device': self.profile.device,
               'host_ms': {k: round(v, 2) for k, v in host.items()}, 'device_ms': round(st.device_ms, 1),
               'loop_hz': round(p_hz, 2), 'load_pct': round(st.load_pct, 1), 'power_w': round(st.power_w, 2),
               'watts_max': self.profile.watts_max, 'perception_hz': round(st.perception_hz, 2),
               'throttle_level': st.throttle_level, 'uncertainty': unc, 't': time.time()}
        self.pub_edge.publish(String(data=json.dumps(out)))
        if st.changed:
            self.trace('edge_throttle', ['EDGE'], f'Device time {st.device_ms:.0f} ms/frame on {self.profile.label} '
                       f'({st.load_pct:.0f}% load, {st.power_w:.1f} W)', st.change_reason, 0.95,
                       f'Perception rate set to {st.perception_hz:.1f} Hz', 'warn' if st.throttle_level else 'info',
                       force=True)

    def _science_traces(self):
        for d in self.det.get('detections', []):
            if d.get('cls') != 'SCIENCE' or d.get('conf', 0) < 0.6 or not d.get('world_xy'):
                continue
            x, y = d['world_xy']
            if any(math.hypot(x - a, y - b) < 2.5 for a, b in self._seen_science):
                continue
            self._seen_science.append((x, y))
            rng = d.get('range_m')
            self.trace('science_target', d.get('sources', ['CAM']),
                       'Cyan crystalline outcrop' + (f' at {rng:.1f} m' if rng else '') +
                       f', bearing {math.degrees(d.get("bearing_rad") or 0):+.0f}°',
                       f'Hue/saturation signature match {d["conf"] * 100:.0f}% (local OpenCV HSV + depth)',
                       d['conf'], f'Logged science target at ({x:.1f}, {y:.1f}) on /planet_map', force=True)


# ====================================================================== rover
class RoverAgent(AgentBase):
    mapping_hz = 10.0

    def __init__(self):
        super().__init__('rover_agent')
        r = self.robot
        self.fsm = RoverFSM(r, self.limits, g_label=f'{self.g:.1f} g', role=self.role)
        self.has_scout = any(x.startswith('scout') for x in self.robots)
        self.peers_list = [x for x in self.robots if x.startswith('rover') and x != r]
        self.rescuer_available = any(role_of(x) == 'recovery' for x in self.peers_list)
        self.health = RoverHealth(r, self.g, battery_pct=START_BATTERY.get(r, 85.0),
                                  tip_alarm_deg=self.limits.tip_alarm_deg, physics=self.physics)
        self.accel = (0.0, 0.0, 0.0)
        self.peers: dict = {}               # teammate → (x, y, yaw, t_sim) from their odometry (fleet link)
        self.sos_seen: dict = {}            # SOS id → (msg, t_sim)
        self.rescue_seen: dict = {}         # SOS id → latest /fleet/rescue msg
        self._sos_msg = None                # our own SOS while active / just resolved
        self._sos_until = -1.0
        self._sos_last = -1e9
        self._rescue_last = -1e9
        self._repairing_by = None
        self._t_health = None
        self.radius = float(self.spec.get('radius_m', 0.6))
        self.w_max = float(self.spec.get('w_max', 1.0))
        self.roll = self.pitch = 0.0
        self.gyro = 0.0
        self.imu_win = ImuWindow()          # 100 Hz attitude/rate between the 10 Hz ticks (the FSM sees the window)
        self.v_wheel = 0.0
        self.slip = 0.0
        self._vhist = collections.deque(maxlen=8)
        self.path = None            # [(r, c)]
        self._path_ok = frozenset()  # route indices planned past lethal cells on purpose (see check_blocked)
        self._path_core = frozenset()  # ... and on cells already lethal at planning time (a core escape)
        self.path_xy = None
        self.path_idx = 0
        self.goal = None            # (x, y)
        self.goal_rc = None
        self.goal_kind = 'frontier'
        self._blacklist: dict = {}
        self._visited_science: list = []
        self.frontiers_left = True
        self.goal_reached = False
        self.blocked = False
        self.blocked_info = ('', ('LIDAR', 'MAP'), 0.9)
        self._last_map_v = -1
        self._last_replan = -1e9
        self._replan_gate = ReplanGate(1.0)
        self._own_marks: list = []
        self._marks_mask = None
        self._marks_n = 0
        b = int(round(BORDER_M / self.map.res))
        self._border = np.ones((self.map.n, self.map.n), bool)
        self._border[b:-b, b:-b] = False
        self._last_select = -1e9
        self.scan_complete = None
        self.out = None
        self.cmd = (0.0, 0.0)
        self.rev_idx = 0
        self._drive_since = self.now_s()
        self._last_drive = 'stop'
        self._mode_sent = None
        self._started = False
        self._t_boot = self.now_s()
        self.brain_cap = BrainCap(r)
        self._brain_replan = False
        self._cap_applied = False           # the last drive ran under brain_cap.apply ('follow' with a path / 'dock')
        self._brain_guard = DropGuard(self.get_logger().warning)
        self.create_subscription(String, '/brain/decision', self.on_brain, 10)
        self.create_subscription(Imu, f'/{r}/imu', self.on_imu, SENSOR_QOS)
        self.create_subscription(Odometry, f'/{r}/wheel_odom', self.on_wheel, SENSOR_QOS)
        self.create_subscription(String, '/fleet/scan_complete', self.on_scan_complete, 10)
        self.create_subscription(String, f'/{r}/hw_status', self.on_hw, 10)
        self.create_subscription(String, '/fleet/sos', self.on_sos, 10)
        self.create_subscription(String, '/fleet/rescue', self.on_rescue, 10)
        for pr in self.peers_list:
            self.create_subscription(Odometry, f'/{pr}/odom', lambda m, pr=pr: self.on_peer(pr, m), SENSOR_QOS)
        self.pub_sos = self.create_publisher(String, '/fleet/sos', 10)
        self.pub_rescue = self.create_publisher(String, '/fleet/rescue', 10)
        self.pub_world = self.create_publisher(String, '/world/command', 10)    # repair tool actuation
        self.pub_mode = self.create_publisher(String, f'/{r}/perception/mode_request', 10)
        self.pub_claim = self.create_publisher(String, '/fleet/claims', 10)
        self.pub_hazard = self.create_publisher(String, '/fleet/hazard_marks', 10)
        self.pub_scan_req = self.create_publisher(String, '/fleet/scan_request', 10)
        self.create_timer(1.0 / LOOP_HZ, self.tick)
        self.create_timer(1.0, self.publish_claim)
        self.create_timer(0.2, self.publish_state)
        self.get_logger().info(f'{r}: θmax {self.limits.max_slope_deg:.1f}° ({self.limits.factor}), '
                               f'tip alarm {self.limits.tip_alarm_deg:.1f}°, v_max {self.limits.v_max:.2f} m/s, '
                               f'scout={self.has_scout}, role={self.role}')

    # ------------------------------------------------------------------ inputs
    def on_imu(self, m: Imu):
        r, p, _ = quat_to_rpy(m.orientation)
        self.roll, self.pitch = math.degrees(r), math.degrees(p)
        w = m.angular_velocity
        self.gyro = math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)
        self.imu_win.add(self.roll, self.pitch, self.gyro)
        a = m.linear_acceleration
        self.accel = (a.x, a.y, a.z)
        self.health.on_imu(self.now_s(), math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z),
                           max(abs(self.roll), abs(self.pitch)))

    def on_wheel(self, m: Odometry):
        self.v_wheel = m.twist.twist.linear.x
        self.health.on_drive(self.now_s(), self.cmd[0], self.v_wheel)

    def on_hw(self, m: String):
        try:
            d = json.loads(m.data)
            self.health.on_hw_status(self.now_s(), d.get('wheel_health') or [1.0] * 4, d.get('faults') or [],
                                     float(d.get('solar_w') or 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    def on_peer(self, robot: str, m: Odometry):
        p = m.pose.pose
        self.peers[robot] = (p.position.x, p.position.y, quat_to_rpy(p.orientation)[2], self.now_s())

    def on_sos(self, m: String):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        if d.get('robot') != self.robot and d.get('id'):
            self.sos_seen[d['id']] = (d, self.now_s())

    def on_rescue(self, m: String):
        try:
            d = json.loads(m.data)
        except json.JSONDecodeError:
            return
        if not d.get('id'):
            return
        self.rescue_seen[d['id']] = d
        if d.get('target') == self.robot and d.get('phase') == 'REPAIRING':
            self._repairing_by = d.get('rescuer')

    def on_brain(self, m: String):
        """Advisory decision for this rover: a speed cap (min with the local speed, sim-time expiry) and, for
        SLOW_AND_REPLAN, a quiet re-plan once no hold binds. Malformed or foreign decisions are ignored; one that
        cannot be handled is dropped with a throttled warning."""
        self._brain_guard.call('/brain/decision', self._on_brain, m.data)

    def _on_brain(self, data: str):
        now = self.now_s()
        cap = self.brain_cap.offer(parse_decision(data), now)
        if cap is None:
            return
        if cap.action == 'SLOW_AND_REPLAN' and not cap.released:
            self._brain_replan = True
        if (self._last_drive in CAPPED_DRIVES and self.brain_cap.holding(now, self.fsm.state)
                and self.cmd != (0.0, 0.0)):
            self.cmd = (0.0, 0.0)          # a hold takes effect now, not after the next tick's A*
            self.send_cmd(0.0, 0.0)
        tx = cap_trace(cap)
        self.trace('brain_cap', ['FLEET'], tx['input'], tx['reasoning'], tx['confidence'], tx['action'],
                   tx['severity'], force=True)   # one per decision (rate-limited upstream): a lifted stop must show

    # ------------------------------------------------------------------ fleet: SOS / rescue
    def peer_xy(self, robot: str, fallback=None):
        p = self.peers.get(robot)
        if p is not None and self.now_s() - p[3] < PEER_FRESH_S:
            return p[0], p[1]
        return fallback

    def active_sos(self):
        """Freshest active SOS from a teammate (recovery role only), or None."""
        if self.role != 'recovery':
            return None
        now = self.now_s()
        live = [d for d, t in self.sos_seen.values() if now - t < SOS_FRESH_S and d.get('status') == 'active']
        if self.fsm.rescue:
            mine = [d for d in live if d.get('id') == self.fsm.rescue['id']]
            if mine:
                return mine[0]
        return live[0] if live else None

    def rescue_target_xy(self):
        r = self.fsm.rescue
        sos = self.active_sos()
        if r is not None:
            msg = self.sos_seen.get(r['id'], (None, 0))[0] or {}
            return self.peer_xy(r['peer'], tuple(msg.get('xy') or ()) or None)
        if sos is not None:
            return self.peer_xy(sos['robot'], tuple(sos.get('xy') or ()) or None)
        return None

    def perceived_teammate(self, target: str):
        if not self.det or self.now_s() - self.det_t > 1.0:
            return None
        for d in self.det.get('detections', []):
            if d.get('cls') == 'ROVER' and d.get('robot') == target and d.get('range_m'):
                return {'robot': target, 'range_m': float(d['range_m']), 'conf': float(d.get('conf', 0.8)),
                        'sources': list(d.get('sources', ['CAM']))}
        return None

    def publish_sos(self, status: str | None = None):
        if self._sos_msg is None or self.pose is None:
            return
        hs = self.health.state()
        self._sos_msg.update({'xy': [round(self.pose[0], 2), round(self.pose[1], 2)], 'yaw': round(self.pose[2], 3),
                              'health_pct': round(hs.pct, 1), 'faults': hs.faults, 't': time.time()})
        if status:
            self._sos_msg['status'] = status
        self.pub_sos.publish(String(data=json.dumps(self._sos_msg)))
        self._sos_last = self.now_s()

    def publish_rescue(self, phase: str | None = None, act: dict | None = None):
        """/fleet/rescue update. `act` is the FSM's rescue action: on DONE / ABORTED the FSM has already dropped its
        rescue record, so the id and target come from the action."""
        r = self.fsm.rescue
        if r is None and phase is None:
            return
        info = dict(r or {})
        act = act or {}
        msg = {'id': info.get('id') or act.get('id'), 'rescuer': self.robot, 'target': info.get('peer') or act.get('target'),
               'phase': phase or info.get('phase'), 'range_m': info.get('range_m'), 'locked': bool(info.get('locked')),
               't': time.time()}
        self.pub_rescue.publish(String(data=json.dumps(msg)))
        self._rescue_last = self.now_s()

    def plan_rescue(self, visible: bool = True) -> bool:
        """A* to a free stand-off cell 2.0–2.45 m from the damaged teammate, on the side facing our approach.

        The teammate is planned as an uninflated keep-out (its footprint + our radius): the stand-off is meant to be
        close to it, and its usual peer disc + inflation ring (≈2.0–2.5 m) would reject a quarter of the cells at
        2.2 m on open ground, and all of them beside another hazard. Cells outside our reachable region are skipped
        without running A*."""
        txy = self.rescue_target_xy()
        if txy is None or self.pose is None:
            self._set_path(None)
            return False
        tx, ty = txy
        peer = self.fsm.rescue['peer'] if self.fsm.rescue else None
        target = (peer, txy)
        cm = self._costmap(target=target)
        base = math.atan2(self.pose[1] - ty, self.pose[0] - tx)
        t0 = time.perf_counter()
        start = self.map.cell(self.pose[0], self.pose[1])
        cells = standoff_cells(cm, txy, base, r_pref=STANDOFF_M)
        reach = reachable_from(cm, start)
        usable = [rc for rc in cells if reach[rc]]
        tight = False
        if not usable:
            # boxed in by inflation rings (the case _plan's tight retry covers): try 0.3 m clearance, cores still lethal
            cm_t = self._costmap(tight=True, target=target)
            cells_t = standoff_cells(cm_t, txy, base, r_pref=STANDOFF_M)
            reach_t = reachable_from(cm_t, start)
            usable_t = [rc for rc in cells_t if reach_t[rc]]
            if usable_t:
                cm, cells, usable, tight = cm_t, cells_t, usable_t, True
        for rc in usable[:RESCUE_TRIES]:
            path, st, _ = self._plan(rc, cm, target=target)
            if path is None:
                continue
            gx, gy = self.map.xy(rc)
            self.goal, self.goal_rc, self.goal_kind = (gx, gy), rc, 'rescue'
            self._set_path(path)
            self._last_replan = self.now_s()
            if visible:
                side = math.degrees(math.atan2(gy - ty, gx - tx)) % 360
                self.trace('rescue_route', ['MAP', 'FLEET'],
                           f'{unit_label(peer or "?")} at ({tx:.1f}, {ty:.1f}); stand-off '
                           f'{math.hypot(gx - tx, gy - ty):.1f} m on the {side:.0f}° side',
                           f'Gravity-aware A* over the shared map at {self.g:.1f} g', 0.9,
                           f'Rescue route {st["length_m"]:.1f} m ({st["expansions"]} nodes, {st["ms"]:.0f} ms)'
                           + (' at 0.3 m clearance' if tight or st.get('tight') else ''),
                           latency_ms=(time.perf_counter() - t0) * 1000, force=True)
            self.publish_claim()
            return True
        if visible:
            self.trace('rescue_noroute', ['MAP', 'FLEET'],
                       f'No route to a stand-off beside {unit_label(peer or "?")}: {len(cells)} free cells '
                       f'2.0–2.45 m from it, {len(usable)} in this rover\'s reachable region',
                       'Hazards or their inflation enclose the teammate or this rover on the current map', 0.7,
                       'Holding; re-planning as the map grows', 'warn')
        self._set_path(None)
        return False

    def on_odom(self, m: Odometry):
        super().on_odom(m)
        # slip compares wheel speed with ground speed differentiated over these samples: time them by the odometry's
        # own (simulated) stamp, not by arrival: under CPU load deliveries bunch up, and arrival times then stretch
        # the window so ground speed reads near zero while the wheels report full speed (a false 70–97 % slip)
        st = m.header.stamp
        t = st.sec + st.nanosec * 1e-9
        self._vhist.append((t if t > 0.0 else self.now_s(), self.pose[0], self.pose[1]))

    def on_scan_complete(self, m: String):
        try:
            self.scan_complete = json.loads(m.data)
        except json.JSONDecodeError:
            pass

    def _update_slip(self):
        now = self.now_s()
        steady = now - self._drive_since > 1.2 and self._last_drive == 'follow'
        if len(self._vhist) >= 4 and steady and abs(self.v_wheel) > 0.15:
            t0, x0, y0 = self._vhist[0]
            t1, x1, y1 = self._vhist[-1]
            yaw = self.pose[2]
            v_true = ((x1 - x0) * math.cos(yaw) + (y1 - y0) * math.sin(yaw)) / max(t1 - t0, 1e-3)
            s = float(np.clip(1.0 - v_true / self.v_wheel, 0.0, 1.0))
            self.slip = 0.7 * self.slip + 0.3 * s
        else:
            self.slip *= 0.7

    # ------------------------------------------------------------------ planning
    def _lethal_core(self):
        """Map lethal classes + this rover's own permanent marks (applied before the map round-trips)."""
        if len(self._own_marks) != self._marks_n:
            m = self.map
            ii, jj = np.mgrid[0:m.n, 0:m.n]
            cx = m.origin[0] + (jj + 0.5) * m.res
            cy = m.origin[1] + (ii + 0.5) * m.res
            mask = np.zeros((m.n, m.n), bool)
            for (x, y, rad) in self._own_marks:
                mask |= np.hypot(cx - x, cy - y) <= rad
            self._marks_mask, self._marks_n = mask, len(self._own_marks)
        lethal = self.map.lethal
        return lethal | self._marks_mask if self._marks_mask is not None else lethal

    def _costmap(self, tight: bool = False, target=None):
        """Planning cost map. `target` = (robot, xy) of a teammate we are docking with: planned as an uninflated
        keep-out (its footprint + our radius) instead of an inflated peer disc."""
        lethal = self._lethal_core()
        skip = target[0] if target is not None else None
        peers = [xy for xy in (self.peer_xy(pr) for pr in self.peers_list if pr != skip) if xy is not None
                 and math.hypot(xy[0] - self.pose[0], xy[1] - self.pose[1]) > 2.5]
        if peers:
            lethal = lethal.copy()
            r_px = max(1, int(round(PEER_DISC_M / self.map.res)))
            mask = np.zeros(lethal.shape, np.uint8)
            for x, y in peers:
                rr, cc = self.map.cell(x, y)
                cv2.circle(mask, (int(cc), int(rr)), r_px, 1, -1)
            lethal |= mask.astype(bool)
        cm = build_costmap(self.map.known, lethal, self.map.caution, self.map.slope, self.map.elev, self.limits,
                           self.map.res, self.map.origin, inflation_m=0.3 if tight else self.radius,
                           caution_mode=self.fsm.caution and not tight)
        # sector edge keep-out behaves like an inflation ring: never entered, but escapable if drifted into
        cm.cost[self._border] = np.maximum(cm.cost[self._border], RING)
        cm.lethal |= self._border
        if target is not None:
            add_keepout(cm, target[1], PEER_DISC_M + self.radius)
        return cm

    def _plan(self, goal_rc, cm=None, target=None):
        cm = cm if cm is not None else self._costmap(target=target)
        start = self.map.cell(self.pose[0], self.pose[1])
        path, st = astar(cm, start, goal_rc, max_expansions=self.profile.max_expansions)
        if path is None and not st['capped'] and st['expansions'] < 1500:
            # boxed in by inflation rings: retry with tight clearance (core hazards still excluded)
            cm_t = self._costmap(tight=True, target=target)
            path, st2 = astar(cm_t, start, goal_rc, max_expansions=self.profile.max_expansions)
            st = dict(st2, tight=True)
            if path is not None:
                self.trace('tight_clearance', ['MAP'], f'Only {st["expansions"]} cells reachable at '
                           f'{self.radius:.1f} m clearance', 'Inflation rings close the pocket; core hazards still '
                           'excluded', 0.7, 'Planning with 0.3 m clearance at reduced speed', 'warn')
        if path is not None and cm.cost[start] == np.inf:
            n_core = sum(1 for rc in path if cm.cost[rc] == np.inf)
            self.trace('core_escape', ['ODOM', 'MAP'],
                       f'Standing inside a no-go cell at ({self.pose[0]:.1f}, {self.pose[1]:.1f}): slid, drifted or '
                       'was pushed in', 'Leaving the lethal blob under the robot by the shortest way; no other '
                       'obstacle may be crossed', 0.8,
                       f'Escape route: {n_core * self.map.res:.1f} m across the zone, then {st["length_m"]:.1f} m '
                       'in all', 'warn')
        return path, st, cm

    def _set_path(self, path):
        self.path = path
        self.path_xy = cells_to_world(path, self.map.res, self.map.origin) if path else None
        if self.path_xy is not None and len(self.path_xy):
            self.path_xy[0] = self.pose[:2]
        self.path_idx = 0
        self.goal_reached = False
        # cells this route passes on purpose (ring escape / tight clearance): the unchanged map must not halt it
        self._path_ok = path_conflicts(path, self._path_footprint()) if path else frozenset()
        self._path_core = path_conflicts(path, self._lethal_core()) if path else frozenset()
        self.publish_path(self.path_xy)

    def _remaining_m(self):
        if self.path_xy is None:
            return 0.0
        p = self.path_xy[self.path_idx:]
        return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0

    def select_goal(self) -> bool:
        """Frontier exploration with fleet claims; falls back to unvisited science targets."""
        m = self.map
        now = self.now_s()
        known = m.known
        free = known & ~self._lethal_core() & ~self._border
        fr_all = find_frontiers(known, free, min_size=4, tile_m=8.0, res=m.res)
        fr = [f for f in fr_all if now - self._blacklist.get(f.target, -1e9) > 30.0]
        claims = self.other_claims()
        ranked = score_frontiers(fr, self.pose[:2], m.res, m.origin, claims=claims, heading=self.pose[2])
        cm = self._costmap()
        t0 = time.perf_counter()
        for f in ranked[:4]:
            path, st, _ = self._plan(f.target, cm)
            if path is None:
                self._blacklist[f.target] = now
                continue
            gx, gy = f.target_xy(m.res, m.origin)
            self.goal, self.goal_rc, self.goal_kind = (gx, gy), f.target, 'frontier'
            self._set_path(path)
            self.frontiers_left = True
            self._last_replan = now
            d = math.hypot(gx - self.pose[0], gy - self.pose[1])
            self.trace('frontier_selected', ['MAP', 'LIDAR'],
                       f'{len(fr)} frontier clusters; best {f.size} cells at ({gx:.1f}, {gy:.1f}), {d:.1f} m away',
                       f'Information gain vs. climb energy at {self.g:.1f} g'
                       + (f'; {len(claims)} teammate goal(s) de-prioritised' if claims else ''),
                       0.85, f'A* route {st["length_m"]:.1f} m ({st["expansions"]} nodes, {st["ms"]:.0f} ms)',
                       latency_ms=(time.perf_counter() - t0) * 1000)
            self.publish_claim()
            return True
        # no reachable frontier: visit science targets for close-range imaging
        for sx, sy in self.map_stats.get('science', []):
            if any(math.hypot(sx - a, sy - b) < 2.0 for a, b in self._visited_science):
                continue
            rc = self._nearest_free(sx, sy, cm)
            if rc is None:
                self._visited_science.append((sx, sy))
                continue
            path, st, _ = self._plan(rc, cm)
            if path is None:
                self._visited_science.append((sx, sy))
                continue
            self.goal, self.goal_rc, self.goal_kind = m.xy(rc), rc, 'science'
            self._set_path(path)
            self.frontiers_left = True
            self.trace('science_approach', ['MAP', 'CAM'], f'Frontiers exhausted; science target at ({sx:.1f}, {sy:.1f})',
                       'Close-range imaging raises sample confidence', 0.8,
                       f'A* route {st["length_m"]:.1f} m to approach point', force=True)
            return True
        self.frontiers_left = bool(fr_all)
        self._set_path(None)
        return False

    def _nearest_free(self, x, y, cm, max_r=6):
        r0, c0 = self.map.cell(x, y)
        best, bd = None, math.inf
        for dr in range(-max_r, max_r + 1):
            for dc in range(-max_r, max_r + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < self.map.n and 0 <= c < self.map.n and cm.cost[r, c] < 1e5 and cm.known[r, c]:
                    d = dr * dr + dc * dc
                    if 4 <= d < bd:
                        best, bd = (r, c), d
        return best

    def _path_footprint(self, lethal=None):
        """Cells too close to a lethal cell for the route check (robot radius less a 0.1 m margin)."""
        return clearance_mask(self._lethal_core() if lethal is None else lethal, self.radius - 0.1, self.map.res)

    def check_blocked(self):
        """New lethal cells (fresh map) on the next BLOCK_LOOKAHEAD_M of the path → visible re-plan; further out →
        quiet. Cells the route was planned past on purpose (escaping an inflation ring, a tight-clearance route) only
        count once they turn lethal themselves: otherwise the same route is halted and re-planned forever."""
        if self.path is None or self.path_xy is None:
            return None
        lethal = self._lethal_core()
        return first_block(self.path, self.path_xy, self._path_footprint(lethal), lethal, self.path_idx,
                           BLOCK_SKIP_M, self._path_ok, self._path_core)

    def _blocked_reason(self, k, dist):
        """Describe what blocks the path, from the map classes + camera detections around the blocking cell."""
        r, c = self.path[k]
        n = 4
        win = self.map.cls[max(0, r - n):r + n + 1, max(0, c - n):c + n + 1]
        n_obs = int((win == gm.OBSTACLE).sum())
        n_haz = int((win == gm.HAZARD).sum())
        n_perm = int((win == gm.PERMANENT).sum())
        rocks = [d for d in (self.det or {}).get('detections', []) if d.get('cls') == 'ROCK'
                 and d.get('range_m') and abs(d['range_m'] - dist) < 5.0]
        sens = ['LIDAR', 'MAP'] + (['CAM'] if rocks else [])
        if n_perm >= max(n_obs, n_haz):
            return f'Permanent hazard zone {dist:.1f} m ahead on route', ('MAP', 'FLEET'), 0.95
        if n_obs >= n_haz:
            txt = f'{n_obs} obstacle cells {dist:.1f} m ahead across path (LiDAR step > 0.30 m)'
            if rocks:
                txt += f'; camera confirms {len(rocks)} boulder{"s" if len(rocks) > 1 else ""}'
            conf = min(0.98, 0.75 + 0.03 * n_obs + 0.05 * len(rocks))
            return txt, tuple(sens), conf
        sl = float(np.nanmax(self.map.slope[max(0, r - n):r + n + 1, max(0, c - n):c + n + 1]))
        return (f'Slope {sl:.1f}° mapped {dist:.1f} m ahead > {self.limits.max_slope_deg:.1f}° limit at {self.g:.1f} g',
                ('LIDAR', 'MAP'), 0.88)

    def replan(self, visible: bool) -> bool:
        now = self.now_s()
        if self.fsm.state == 'RESCUE_TRANSIT':
            self._last_replan = now
            return self.plan_rescue(visible)
        if self.fsm.state in ('RESCUE_DOCKING', 'REPAIRING', 'AWAIT_RESCUE', 'SAFE_HOLD'):
            return False
        old_len = self._remaining_m()
        if self.goal_rc is not None:
            path, st, _ = self._plan(self.goal_rc)
            if path is not None:
                self._set_path(path)
                self._last_replan = now
                if visible:
                    new_len = path_length_m(path, self.map.res)
                    self.trace('replanned', ['MAP', 'LIDAR'],
                               f'Obstacle/hazard cells excluded from cost map (inflation {self.radius:.1f} m'
                               + (', +0.4 m caution' if self.fsm.caution else '') + ')',
                               f'Gravity-aware A* at {self.g:.1f} g: slope ≤ {self.limits.max_slope_deg:.1f}°, '
                               f'climb energy weighted ×{self.g:.1f}', 0.9,
                               f'New route {new_len:.1f} m ({new_len - old_len:+.1f} m), {st["expansions"]} nodes, '
                               f'{st["ms"]:.0f} ms', 'info', latency_ms=st['ms'], force=True)
                return True
        ok = self.select_goal()
        self._last_replan = now
        return ok

    # ------------------------------------------------------------------ control loop
    def tick(self):
        t_start = time.perf_counter()
        now = self.now_s()               # simulated seconds: every duration below follows the sim speed
        mono = now
        imu_win = self.imu_win.take()    # every tick starts a new window, even one that returns early
        if self.pose is None or not self.map.ready:
            return
        if not self._started:
            r0, c0 = self.map.cell(*self.pose[:2])
            near = int(self.map.known[max(0, r0 - 20):r0 + 21, max(0, c0 - 20):c0 + 21].sum())
            if (near < 30 or self.det is None) and mono - self._t_boot < 30.0:
                return
            self._started = True
        self._update_slip()

        # map-driven path checks (only on fresh maps)
        if self.map.version != self._last_map_v and self.fsm.state in ('EXPLORING', 'CAUTION', 'RESCUE_TRANSIT'):
            self._last_map_v = self.map.version
            hit = self.check_blocked()
            if hit is not None:
                k, dist = hit
                if dist < BLOCK_LOOKAHEAD_M:
                    self.blocked = True
                    self.blocked_info = self._blocked_reason(k, dist)
                else:
                    self.replan(visible=False)
            elif self.path is not None and mono - self._last_replan > REPLAN_PERIOD_S:
                self.replan(visible=False)

        # health: battery integration + repair verification
        dt = 0.0 if self._t_health is None else min(1.0, max(0.0, now - self._t_health))
        self._t_health = now
        tw = getattr(self, 'odom_twist', None)
        self.health.step(now, dt, v_ground=tw.linear.x if tw else 0.0, pitch_rad=math.radians(self.pitch),
                         compute_w=self.edge_power_w)
        if self._repairing_by and self.health.drive_pct >= 95.0:
            self.health.repaired(now, self._repairing_by)
            self._repairing_by = None
        hs = self.health.state().as_dict()
        sos = self.active_sos()
        txy = self.rescue_target_xy()
        rng = math.hypot(txy[0] - self.pose[0], txy[1] - self.pose[1]) if txy else None
        target = (self.fsm.rescue or {}).get('peer') or (sos or {}).get('robot')
        lock = self.perceived_teammate(target) if target and self.fsm.state == 'RESCUE_DOCKING' else None
        resolved = False
        if self.fsm.rescue:
            msg = self.sos_seen.get(self.fsm.rescue['id'], (None, 0))[0]
            resolved = bool(msg and msg.get('status') == 'resolved')

        d = self.det or {}
        unc = d.get('uncertainty', {})
        cor = d.get('corridor', {})
        cell = self.map.cell(*self.pose[:2])
        roll_w, pitch_w, gyro_w = imu_win or (self.roll, self.pitch, self.gyro)
        inp = RoverInputs(
            now=now, pose=self.pose, roll_deg=roll_w, pitch_deg=pitch_w, gyro_rate=gyro_w, slip=self.slip,
            uncertainty=float(unc.get('score', 0.0)), camera_conf=float(unc.get('camera_conf', 1.0)),
            lidar_conf=float(unc.get('lidar_conf', 1.0)),
            corridor_obstacle_m=float(cor.get('obstacle_dist_m') or math.inf), corridor_conf=float(cor.get('conf', 0.0)),
            slope_ahead_deg=float(cor.get('slope_ahead_deg') or 0.0), path_blocked=self.blocked,
            blocked_reason=self.blocked_info[0], blocked_sensors=self.blocked_info[1], blocked_conf=self.blocked_info[2],
            has_path=self.path is not None, goal_reached=self.goal_reached, frontiers_left=self.frontiers_left,
            percent_known=float(self.map_stats.get('percent', 0.0)), scan_complete=self.scan_complete,
            scout_available=self.has_scout, on_hazard_cell=bool(self.map.lethal[cell]), health=hs,
            rescuer_available=self.rescuer_available, sos=sos, rescue_range_m=rng, rescue_lock=lock,
            sos_resolved=resolved)
        self.scan_complete = None
        out = self.fsm.step(inp)
        self.out = out
        if self.fsm.rescue is None and self.goal_kind == 'rescue':
            # the rescue ended in this step (done or aborted): the re-plan it requests must pick a fresh frontier,
            # not drive back to the stand-off beside the teammate
            self.goal = self.goal_rc = None
            self.goal_kind = 'frontier'
        self.emit(out.traces)
        self._slope_trace(inp)
        brain_hold = self.brain_cap.holding(now, self.fsm.state)
        if (out.drive == 'stop' or (brain_hold and out.drive in CAPPED_DRIVES)) and self.cmd != (0.0, 0.0):
            self.cmd = (0.0, 0.0)          # halt before the (possibly long) A* below, not after it
            self.send_cmd(0.0, 0.0)

        for a in out.actions:
            if a['type'] == 'hazard_mark':
                x, y = a['xy']
                self._own_marks.append((x, y, a['radius_m']))
                self.pub_hazard.publish(String(data=json.dumps({'robot': self.robot, 'xy': [x, y],
                                                                'radius_m': a['radius_m'], 'reason': a['reason'],
                                                                't': time.time()})))
            elif a['type'] == 'scan_request':
                self.pub_scan_req.publish(String(data=json.dumps({'id': a['id'], 'robot': self.robot,
                                                                  'xy': list(a['xy']), 'reason': a['reason'],
                                                                  't': time.time()})))
            elif a['type'] == 'new_goal' and mono - self._last_select > 1.0:
                self._last_select = mono
                self.select_goal()
            elif a['type'] == 'sos':
                if a['status'] == 'active':
                    self._sos_msg = {'id': a['id'], 'robot': self.robot, 'status': 'active'}
                    self._set_path(None)
                self.publish_sos(a['status'])
                if a['status'] == 'resolved':
                    self._sos_until = now + 3.0          # repeat the resolution a few times (lossy fleet link)
            elif a['type'] == 'rescue':
                self.publish_rescue(a['phase'], a)
                if a['phase'] in ('DOCKING', 'REPAIRING'):
                    self._set_path(None)
            elif a['type'] == 'repair':
                self.pub_world.publish(String(data=json.dumps({'op': 'repair', 'robot': a['target'],
                                                               'by': self.robot})))
        # heartbeats
        if self.fsm.state == 'AWAIT_RESCUE' and now - self._sos_last >= 1.0:
            self.publish_sos()
        elif self._sos_msg is not None and self._sos_msg.get('status') == 'resolved':
            if now < self._sos_until and now - self._sos_last >= 1.0:
                self.publish_sos()
            elif now >= self._sos_until:
                self._sos_msg = None
        if self.fsm.rescue and now - self._rescue_last >= 1.0:
            self.publish_rescue()

        if self._replan_gate.due(out.request_replan, out.state, mono):
            self.blocked = False
            self.replan(visible=True)
            self.publish_claim()
        if self._brain_replan and not brain_hold:
            # advisory SLOW_AND_REPLAN: refresh the route quietly unless the FSM has just re-planned it (deferred
            # while a brain hold binds: the rover is stopped, the re-plan runs when it may move again)
            self._brain_replan = False
            if self.fsm.state in BRAIN_REPLAN_STATES and mono - self._last_replan > 1.0:
                self.replan(visible=False)

        if out.mode != self._mode_sent:
            self.pub_mode.publish(String(data=out.mode))
            self._mode_sent = out.mode

        self.drive(out)
        self.planner_ms = 0.8 * self.planner_ms + 0.2 * (time.perf_counter() - t_start) * 1000

    def _slope_trace(self, i: RoverInputs):
        s = i.slope_ahead_deg
        if s > self.limits.max_slope_deg and self.fsm.state in ('EXPLORING', 'CAUTION'):
            lim1 = self.physics.limits(1.0).max_slope_deg
            self.trace('slope_hazard', ['LIDAR', 'IMU'], f'LiDAR ground slope {s:.1f}° in the corridor ahead',
                       f'Exceeds {self.limits.max_slope_deg:.1f}° {self.limits.factor.replace("_", " ")} limit at '
                       f'{self.g:.1f} g (would be passable up to {lim1:.0f}° at 1 g)', 0.85,
                       'Cells marked lethal; route keeps to gentler ground', 'warn')

    def drive(self, out):
        if out.drive != self._last_drive:
            self._drive_since = self.now_s()
            self._last_drive = out.drive
            self.rev_idx = 0
        x, y, yaw = self.pose
        v = w = 0.0
        self._cap_applied = False
        if out.drive == 'follow' and self.path_xy is not None:
            vmax = self.limits.v_max * out.speed_scale
            v, w, idx, done = pure_pursuit(self.pose, self.path_xy, lookahead=1.5, v_max=vmax, w_max=self.w_max)
            v, w = self.brain_cap.apply(v, w, self.now_s(), self.fsm.state)   # advisory: never faster than local
            self._cap_applied = True
            self.path_idx = max(self.path_idx, idx)
            if done:
                self.goal_reached = True
                if self.goal_kind == 'science':
                    self._visited_science.append(self.goal)
        elif out.drive == 'dock':
            txy = self.rescue_target_xy()
            cor = (self.det or {}).get('corridor', {})
            near = float(cor.get('obstacle_dist_m') or math.inf)
            if txy is not None:
                alpha = wrap(math.atan2(txy[1] - y, txy[0] - x) - yaw)
                rng = math.hypot(txy[0] - x, txy[1] - y)
                if abs(alpha) > 0.35:
                    v, w = 0.0, float(np.clip(1.5 * alpha, -0.5, 0.5))
                elif rng > DOCK_M - 0.05 and near >= rng - DOCK_CLEAR_M:
                    v = self.limits.v_max * out.speed_scale
                    w = float(np.clip(1.5 * alpha, -0.4, 0.4))
            v, w = self.brain_cap.apply(v, w, self.now_s(), self.fsm.state)   # operator caps only while docking
            self._cap_applied = True
        elif out.drive == 'reverse' and out.reverse_path:
            rp = out.reverse_path
            while self.rev_idx < len(rp) - 1 and math.hypot(rp[self.rev_idx][0] - x, rp[self.rev_idx][1] - y) < 0.6:
                self.rev_idx += 1
            tx, ty = rp[self.rev_idx]
            alpha = wrap(math.atan2(ty - y, tx - x) - (yaw + math.pi))
            if abs(alpha) > 0.9:
                v, w = 0.0, math.copysign(0.5, alpha)
            else:
                v, w = -out.speed_scale, float(np.clip(1.5 * alpha, -0.6, 0.6))
        self.cmd = (v, w)
        self.send_cmd(v, w)

    # ------------------------------------------------------------------ status
    def publish_claim(self):
        if self.goal is not None:
            self.pub_claim.publish(String(data=json.dumps({'robot': self.robot, 'goal': list(self.goal),
                                                           't': time.time()})))

    def publish_state(self):
        if self.pose is None:
            return
        out = self.out
        tw = getattr(self, 'odom_twist', None)
        speed = math.hypot(tw.linear.x, tw.linear.y) if tw else 0.0
        st = {'robot': self.robot, 'type': 'rover', 'state': self.fsm.state,
              'mode': out.mode if out else 'FUSED', 'goal': list(self.goal) if self.goal else None,
              'goal_kind': self.goal_kind, 'speed': round(speed, 3),
              'v_limit': round(self.limits.v_max * (out.speed_scale if out else 1.0), 3),
              'cmd': [round(self.cmd[0], 3), round(self.cmd[1], 3)],
              'tilt': {'roll_deg': round(self.roll, 2), 'pitch_deg': round(self.pitch, 2)},
              'slip': round(self.slip, 3), 'breadcrumbs': self.fsm.breadcrumb_count(),
              'last_safe': list(self.fsm.last_safe) if self.fsm.last_safe else None,
              'tip_alarm_deg': round(self.limits.tip_alarm_deg, 2), 'max_slope_deg': round(self.limits.max_slope_deg, 2),
              'altitude_agl': None, 'pose': [round(v, 3) for v in self.pose] + [round(self.z, 3)],
              'path_remaining_m': round(self._remaining_m(), 1), 'caution': self.fsm.caution,
              'role': self.role, 'name': self.name, 'health': self.health.state().as_dict(),
              'rescue': self._rescue_info(), 'v_cmd': round(self.cmd[0], 3), 'v_wheel': round(self.v_wheel, 3),
              'v_ground': round(tw.linear.x, 3) if tw else 0.0,
              'brain_cap': self.brain_cap.info(self.now_s(), self.fsm.state, self._cap_applied), 't': time.time()}
        self.pub_state.publish(String(data=json.dumps(st)))

    def _rescue_info(self):
        r = self.fsm.rescue
        if r is not None:
            return {k: r.get(k) for k in ('id', 'role', 'peer', 'phase', 'range_m', 'locked')}
        if self._sos_msg is not None and self._sos_msg.get('status') == 'active':
            rs = self.rescue_seen.get(self._sos_msg['id']) or {}
            if rs.get('phase') in ('ABORTED', 'DONE'):     # that rescuer gave up / left: waiting for help again
                rs = {}
            return {'id': self._sos_msg['id'], 'role': 'target', 'peer': rs.get('rescuer'),
                    'phase': rs.get('phase') or 'AWAITING', 'range_m': rs.get('range_m'),
                    'locked': bool(rs.get('locked'))}
        return None


# ====================================================================== scout
class ScoutAgent(AgentBase):
    mapping_hz = 5.0
    CRUISE_V = 2.0
    DASH_V = 4.0
    ORBIT_R = 3.0

    def __init__(self):
        super().__init__('scout_agent')
        r = self.robot
        self.cruise = float(self.spec.get('cruise_agl_m', 12.0))
        self.vz_max = float(self.spec.get('vz_max', 2.0))
        self.fsm = ScoutFSM(r, cruise_agl=self.cruise, max_slope_deg=self.limits.max_slope_deg)
        self.health = DroneHealth(r, self.g, battery_pct=96.0, mass_kg=float(self.spec.get('mass_kg', 2.0)))
        self._t_health = None
        self.accel = (0.0, 0.0, 0.0)
        self.create_subscription(Imu, f'/{r}/imu', self.on_imu, SENSOR_QOS)
        self.agl = 0.0
        self.ground_z = None
        self.vz = 0.0
        self._z_prev = None
        self._integ = 0.0
        self.target = None
        self.goal_reached = False
        self.frontiers_left = True
        self.pending_req = None
        self.out = None
        self._orbit_a = 0.0
        self._blacklist: dict = {}
        self._lawn = self._lawnmower()
        self.brain_alt = BrainAltitude(r)
        self._agl_cmd = self.cruise            # AGL the altitude hold is flying to (cruise or the advisory one)
        self._brain_guard = DropGuard(self.get_logger().warning)
        self.create_subscription(String, '/brain/decision', self.on_brain, 10)
        self.rover_xy: dict = {}
        for rv in (x for x in self.robots if x.startswith('rover')):
            self.create_subscription(Odometry, f'/{rv}/odom', lambda m, rv=rv: self.rover_xy.__setitem__(
                rv, (m.pose.pose.position.x, m.pose.pose.position.y)), SENSOR_QOS)
        self.create_subscription(Image, f'/{r}/camera/depth', self.on_depth, SENSOR_QOS)
        self.create_subscription(String, '/fleet/scan_request', self.on_req, 10)
        self.pub_scan_done = self.create_publisher(String, '/fleet/scan_complete', 10)
        self.create_timer(1.0 / LOOP_HZ, self.tick)
        self.create_timer(0.2, self.publish_state)
        self.get_logger().info(f'{r}: cruise {self.cruise:.0f} m AGL')

    def _lawnmower(self):
        h = self.map.size_m / 2 - 8.0
        ys = np.arange(-h, h + 0.1, 20.0)
        pts = []
        for k, y in enumerate(ys):
            xs = (-h, h) if k % 2 == 0 else (h, -h)
            pts += [(xs[0], y), (xs[1], y)]
        return pts

    def on_depth(self, m: Image):
        d = image_to_numpy(m)
        h, w = d.shape
        c = d[h // 2 - h // 6:h // 2 + h // 6, w // 2 - w // 6:w // 2 + w // 6]
        c = c[np.isfinite(c) & (c > 0.05)]
        cam_dz = -float(self.spec.get('camera_xyz', [0, 0, -0.08])[2])
        if len(c) > 20:
            self.agl = float(np.median(c)) + cam_dz
            self.ground_z = self.z - self.agl
        elif self.ground_z is not None:
            self.agl = self.z - self.ground_z

    def on_odom(self, m: Odometry):
        super().on_odom(m)
        now = self.now_s()
        if self._z_prev is not None and now > self._z_prev[0]:
            self.vz = 0.7 * self.vz + 0.3 * (self.z - self._z_prev[1]) / (now - self._z_prev[0])
        self._z_prev = (now, self.z)
        if self.ground_z is None:
            self.ground_z = self.z - 0.1

    def on_imu(self, m: Imu):
        a = m.linear_acceleration
        self.accel = (a.x, a.y, a.z)

    def on_req(self, m: String):
        try:
            self.pending_req = json.loads(m.data)
        except json.JSONDecodeError:
            pass

    def on_brain(self, m: String):
        """Advisory ADJUST_DRONE_ALTITUDE for this drone: cruise AGL clamped to 6-20 m for 60 s (sim). A decision
        that cannot be handled is dropped with a throttled warning."""
        self._brain_guard.call('/brain/decision', self._on_brain, m.data)

    def _on_brain(self, data: str):
        d = parse_decision(data)
        agl = self.brain_alt.offer(d, self.now_s())
        if agl is None:
            return
        tx = altitude_trace(agl, d, self.fsm.state in BRAIN_AGL_STATES)
        self.trace('brain_cap', ['FLEET'], tx['input'], tx['reasoning'], tx['confidence'], tx['action'],
                   tx['severity'], force=True)   # one per decision (rate-limited upstream): a lifted stop must show

    def agl_target(self) -> float:
        """Altitude-hold set-point. The advisory AGL applies only while surveying / flying to a scan: take-off keeps
        the cruise AGL (its completion test is 90 % of it) and a scan holds the altitude it started at."""
        if self.fsm.state == 'SCANNING':
            return self._agl_cmd
        agl = self.brain_alt.target(self.now_s()) if self.fsm.state in BRAIN_AGL_STATES else None
        self._agl_cmd = self.cruise if agl is None else agl
        return self._agl_cmd

    def scan_stats(self, xy, radius=4.0):
        m = self.map
        if not m.ready:
            return None
        r0, c0 = m.cell(*xy)
        k = int(math.ceil(radius / m.res))
        sl = m.slope[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1]
        kn = m.known[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1]
        v = sl[kn]
        if len(v) < 10:
            return None
        return {'max_slope_deg': float(np.percentile(v, 98)), 'mean_slope_deg': float(v.mean())}

    def pick_target(self):
        m = self.map
        now = self.now_s()
        if m.ready and self.pose is not None:
            known = m.known
            fr = find_frontiers(known, known, min_size=6, tile_m=16.0, res=m.res)
            lim = m.size_m / 2 - 6.0
            fr = [f for f in fr if now - self._blacklist.get(f.target, -1e9) > 60.0
                  and all(abs(v) < lim for v in f.target_xy(m.res, m.origin))]
            if self.rover_xy and fr:
                # stay within reach of the rovers so a scan request is answered within seconds
                def leash(f):
                    x, y = f.target_xy(m.res, m.origin)
                    return min(math.hypot(x - a, y - b) for a, b in self.rover_xy.values())
                near = [f for f in fr if leash(f) < SCOUT_LEASH_M]
                fr = near or sorted(fr, key=leash)[:3]
            ranked = score_frontiers(fr, self.pose[:2], m.res, m.origin, claims=self.other_claims(),
                                     claim_radius=12.0, heading=self.pose[2], w_size=0.25, min_dist=8.0)
            if ranked:
                f = ranked[0]
                self._blacklist[f.target] = now
                self.target = f.target_xy(m.res, m.origin)
                self.trace('survey_target', ['MAP', 'DEPTH'],
                           f'{len(fr)} unmapped frontier regions; next {f.size} cells at '
                           f'({self.target[0]:.1f}, {self.target[1]:.1f})',
                           f'Nadir footprint ~{2 * self._agl_cmd:.0f} m wide maximises new coverage per Wh', 0.85,
                           f'Flying survey leg at {self.CRUISE_V:.0f} m/s, {self._agl_cmd:.0f} m AGL')
                return
            if float(self.map_stats.get('percent', 0.0)) > 95.0:
                self.frontiers_left = False
                self.target = None
                return
        if self._lawn:
            self.target = self._lawn.pop(0)

    def tick(self):
        t_start = time.perf_counter()
        if self.pose is None:
            return
        now = self.now_s()
        dt = 0.0 if self._t_health is None else min(1.0, max(0.0, now - self._t_health))
        self._t_health = now
        self.health.step(now, dt, airborne=self.agl > 0.5, compute_w=self.edge_power_w)
        x, y, yaw = self.pose
        inp = ScoutInputs(now=now, pose=(x, y, self.z), altitude_agl=self.agl, goal_reached=self.goal_reached,
                          scan_request=self.pending_req, frontiers_left=self.frontiers_left)
        self.pending_req = None
        if self.fsm.state == 'SCANNING' and self.fsm._job:
            inp.scan_stats = self.scan_stats(self.fsm._job['xy'])
        out = self.fsm.step(inp)
        self.out = out
        self.emit(out.traces)
        for a in out.actions:
            if a['type'] == 'scan_complete':
                msg = dict(a, scout=self.robot, t=time.time())
                msg.pop('type', None)
                self.pub_scan_done.publish(String(data=json.dumps(msg)))
            elif a['type'] == 'new_goal':
                self.pick_target()
        if out.state == 'SURVEY' and self.target is None:
            self.pick_target()

        # ---- altitude hold (PID on AGL; VelocityControl sinks slightly at zero command)
        agl_cmd = self.agl_target()
        err = agl_cmd - self.agl
        self._integ = float(np.clip(self._integ + 0.25 * err / LOOP_HZ, -0.6, 0.6))
        vz = float(np.clip(1.0 * err - 0.4 * self.vz + self._integ, -self.vz_max, self.vz_max))

        # ---- horizontal guidance in the world frame
        vx = vy = 0.0
        if out.orbit and out.target_xy:
            tx, ty = out.target_xy
            self._orbit_a += 0.5 / LOOP_HZ
            px, py = tx + self.ORBIT_R * math.cos(self._orbit_a), ty + self.ORBIT_R * math.sin(self._orbit_a)
            vx, vy = self._toward(px, py, 1.5)
        elif out.target_xy:
            vx, vy = self._toward(*out.target_xy, self.DASH_V)
        elif out.state == 'SURVEY' and self.target is not None and self.agl > 0.7 * agl_cmd:
            vx, vy = self._toward(*self.target, self.CRUISE_V)
            self.goal_reached = math.hypot(self.target[0] - x, self.target[1] - y) < 2.5
            if self.goal_reached:
                self.target = None
        if out.state in ('TAKEOFF',) or out.hover:
            vx = vy = 0.0
        # body frame (yaw held near zero)
        c, s = math.cos(yaw), math.sin(yaw)
        self.send_cmd(c * vx + s * vy, -1.0 * wrap(yaw), -s * vx + c * vy, vz)
        self.planner_ms = 0.8 * self.planner_ms + 0.2 * (time.perf_counter() - t_start) * 1000

    def _toward(self, tx, ty, vmax):
        dx, dy = tx - self.pose[0], ty - self.pose[1]
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return 0.0, 0.0
        v = min(vmax, 0.8 * d)
        return v * dx / d, v * dy / d

    def publish_state(self):
        if self.pose is None:
            return
        tw = getattr(self, 'odom_twist', None)
        speed = math.hypot(tw.linear.x, tw.linear.y) if tw else 0.0
        out = self.out
        tgt = (out.target_xy if out and out.target_xy else self.target)
        st = {'robot': self.robot, 'type': 'scout', 'state': self.fsm.state, 'mode': 'FUSED',
              'goal': list(tgt) if tgt else None, 'speed': round(speed, 3), 'v_limit': self.DASH_V,
              'tilt': None, 'slip': None, 'breadcrumbs': None, 'last_safe': None,
              'altitude_agl': round(self.agl, 2), 'pose': [round(v, 3) for v in self.pose] + [round(self.z, 3)],
              'scan_job': (self.fsm._job or {}).get('id'), 'orbit': bool(out and out.orbit),
              'role': self.role, 'name': self.name, 'health': self.health.state().as_dict(), 'rescue': None,
              'v_cmd': None, 'v_wheel': None, 'v_ground': round(speed, 3),
              'brain_agl': self.brain_alt.info(self.now_s(),
                                               self._agl_cmd if self.fsm.state in AGL_HOLD_STATES else None),
              't': time.time()}
        self.pub_state.publish(String(data=json.dumps(st)))


def main():
    rclpy.init()
    probe = rclpy.create_node('agent_probe')
    probe.declare_parameter('robot_type', 'rover')
    rtype = probe.get_parameter('robot_type').value
    probe.destroy_node()
    node = RoverAgent() if rtype == 'rover' else ScoutAgent()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.send_cmd()
        except Exception:  # noqa: BLE001 - context may already be gone
            pass
        node.destroy_node()
        rclpy.try_shutdown()
