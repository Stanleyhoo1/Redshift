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
from .agent_fsm import RoverFSM, RoverInputs, ScoutFSM, ScoutInputs
from .edge_budget import PROFILES, EdgeBudget
from .grid_map import slope_from_elevation
from .physics_model import RoverPhysics
from .planning import (RING, astar, build_costmap, cell_to_world, cells_to_world, find_frontiers, path_length_m,
                       pure_pursuit, score_frontiers, world_to_cell, wrap)
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


class MapView:
    """Planner-side copy of /planet_map/layers + /planet_map/elevation."""

    def __init__(self, size_m: float, res: float):
        self.size_m, self.res = size_m, res
        self.n = int(round(size_m / res))
        self.origin = (-size_m / 2.0, -size_m / 2.0)
        self.cls = np.zeros((self.n, self.n), np.uint8)
        self.elev = np.full((self.n, self.n), np.nan, np.float32)
        self.stamp = 0.0
        self.version = 0
        self._slope = None
        self._slope_v = -1

    @property
    def ready(self) -> bool:
        return self.version > 0

    def set_layers(self, img: np.ndarray):
        self.cls = np.rint(img[..., 1].astype(np.float32) / 40.0).astype(np.uint8)
        self.version += 1
        self.stamp = time.monotonic()

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
    def __init__(self, name: str):
        super().__init__(name)
        for k, v in (('robot', 'rover_0'), ('robot_type', 'rover'), ('spec_json', '{}'), ('robots', ['rover_0']),
                     ('g_mult', 2.2), ('edge_profile', 'orin_nano_15w'), ('size_m', 100.0), ('map_res', 0.5)):
            self.declare_parameter(k, v)
        self.robot = self.get_parameter('robot').value
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
            self.det_t = time.monotonic()
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
            self.claims[d['robot']] = (tuple(d['goal']), time.monotonic())

    def other_claims(self):
        now = time.monotonic()
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
                             latency_ms=latency_ms, now=time.time(), force=force)
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
        if not self.det or time.monotonic() - self.det_t > 3.0:
            return 0.0, 0.0
        return float(sum(self.det.get('stage_ms', {}).values())), float(self.det.get('perception_hz', 0.0))

    def publish_edge(self):
        p_ms, p_hz = self.perception_ms()
        m_ms = float(self.map_stats.get('mapping_ms', {}).get(self.robot, 0.0))
        host = {'perception': p_ms, 'planner': self.planner_ms, 'mapping': m_ms}
        hz = {'perception': p_hz, 'planner': LOOP_HZ, 'mapping': self.mapping_hz}
        st = self.edge.update(host, hz)
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
        self.fsm = RoverFSM(r, self.limits, g_label=f'{self.g:.1f} g')
        self.has_scout = any(x.startswith('scout') for x in self.robots)
        self.radius = float(self.spec.get('radius_m', 0.6))
        self.w_max = float(self.spec.get('w_max', 1.0))
        self.roll = self.pitch = 0.0
        self.gyro = 0.0
        self.v_wheel = 0.0
        self.slip = 0.0
        self._vhist = collections.deque(maxlen=8)
        self.path = None            # [(r, c)]
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
        self._last_replan = 0.0
        self._last_fail_replan = 0.0
        self._own_marks: list = []
        self._marks_mask = None
        self._marks_n = 0
        b = int(round(BORDER_M / self.map.res))
        self._border = np.ones((self.map.n, self.map.n), bool)
        self._border[b:-b, b:-b] = False
        self._last_select = 0.0
        self.scan_complete = None
        self.out = None
        self.cmd = (0.0, 0.0)
        self.rev_idx = 0
        self._drive_since = time.monotonic()
        self._last_drive = 'stop'
        self._mode_sent = None
        self._started = False
        self._t_boot = time.monotonic()
        self.create_subscription(Imu, f'/{r}/imu', self.on_imu, SENSOR_QOS)
        self.create_subscription(Odometry, f'/{r}/wheel_odom', self.on_wheel, SENSOR_QOS)
        self.create_subscription(String, '/fleet/scan_complete', self.on_scan_complete, 10)
        self.pub_mode = self.create_publisher(String, f'/{r}/perception/mode_request', 10)
        self.pub_claim = self.create_publisher(String, '/fleet/claims', 10)
        self.pub_hazard = self.create_publisher(String, '/fleet/hazard_marks', 10)
        self.pub_scan_req = self.create_publisher(String, '/fleet/scan_request', 10)
        self.create_timer(1.0 / LOOP_HZ, self.tick)
        self.create_timer(1.0, self.publish_claim)
        self.create_timer(0.2, self.publish_state)
        self.get_logger().info(f'{r}: θmax {self.limits.max_slope_deg:.1f}° ({self.limits.factor}), '
                               f'tip alarm {self.limits.tip_alarm_deg:.1f}°, v_max {self.limits.v_max:.2f} m/s, '
                               f'scout={self.has_scout}')

    # ------------------------------------------------------------------ inputs
    def on_imu(self, m: Imu):
        r, p, _ = quat_to_rpy(m.orientation)
        self.roll, self.pitch = math.degrees(r), math.degrees(p)
        w = m.angular_velocity
        self.gyro = math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)

    def on_wheel(self, m: Odometry):
        self.v_wheel = m.twist.twist.linear.x

    def on_odom(self, m: Odometry):
        super().on_odom(m)
        self._vhist.append((time.monotonic(), self.pose[0], self.pose[1]))

    def on_scan_complete(self, m: String):
        try:
            self.scan_complete = json.loads(m.data)
        except json.JSONDecodeError:
            pass

    def _update_slip(self):
        now = time.monotonic()
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

    def _costmap(self, tight: bool = False):
        lethal = self._lethal_core()
        cm = build_costmap(self.map.known, lethal, self.map.caution, self.map.slope, self.map.elev, self.limits,
                           self.map.res, self.map.origin, inflation_m=0.3 if tight else self.radius,
                           caution_mode=self.fsm.caution and not tight)
        # sector edge keep-out behaves like an inflation ring: never entered, but escapable if drifted into
        cm.cost[self._border] = np.maximum(cm.cost[self._border], RING)
        cm.lethal |= self._border
        return cm

    def _plan(self, goal_rc, cm=None):
        cm = cm if cm is not None else self._costmap()
        start = self.map.cell(self.pose[0], self.pose[1])
        path, st = astar(cm, start, goal_rc, max_expansions=self.profile.max_expansions)
        if path is None and not st['capped'] and st['expansions'] < 1500:
            # boxed in by inflation rings: retry with tight clearance (core hazards still excluded)
            cm_t = self._costmap(tight=True)
            path, st2 = astar(cm_t, start, goal_rc, max_expansions=self.profile.max_expansions)
            st = dict(st2, tight=True)
            if path is not None:
                self.trace('tight_clearance', ['MAP'], f'Only {st["expansions"]} cells reachable at '
                           f'{self.radius:.1f} m clearance', 'Inflation rings close the pocket; core hazards still '
                           'excluded', 0.7, 'Planning with 0.3 m clearance at reduced speed', 'warn')
        return path, st, cm

    def _set_path(self, path):
        self.path = path
        self.path_xy = cells_to_world(path, self.map.res, self.map.origin) if path else None
        if self.path_xy is not None and len(self.path_xy):
            self.path_xy[0] = self.pose[:2]
        self.path_idx = 0
        self.goal_reached = False
        self.publish_path(self.path_xy)

    def _remaining_m(self):
        if self.path_xy is None:
            return 0.0
        p = self.path_xy[self.path_idx:]
        return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0

    def select_goal(self) -> bool:
        """Frontier exploration with fleet claims; falls back to unvisited science targets."""
        m = self.map
        now = time.monotonic()
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

    def check_blocked(self):
        """Lethal cells (fresh map) on the next BLOCK_LOOKAHEAD_M of the path → visible re-plan; further out → quiet."""
        if self.path is None or self.path_xy is None:
            return None
        lethal = self._lethal_core()
        r_cells = max(1, int(round((self.radius - 0.1) / self.map.res)))
        dil = cv2.dilate(lethal.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_cells + 1,) * 2))
        dist = 0.0
        for k in range(self.path_idx, len(self.path)):
            if k > self.path_idx:
                a, b = self.path_xy[k - 1], self.path_xy[k]
                dist += math.hypot(b[0] - a[0], b[1] - a[1])
            if dist < BLOCK_SKIP_M:
                continue
            r, c = self.path[k]
            if dil[r, c]:
                return k, dist
        return None

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
        now = time.monotonic()
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
        now = time.time()
        mono = time.monotonic()
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
        if self.map.version != self._last_map_v and self.fsm.state in ('EXPLORING', 'CAUTION'):
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

        d = self.det or {}
        unc = d.get('uncertainty', {})
        cor = d.get('corridor', {})
        cell = self.map.cell(*self.pose[:2])
        inp = RoverInputs(
            now=now, pose=self.pose, roll_deg=self.roll, pitch_deg=self.pitch, gyro_rate=self.gyro, slip=self.slip,
            uncertainty=float(unc.get('score', 0.0)), camera_conf=float(unc.get('camera_conf', 1.0)),
            lidar_conf=float(unc.get('lidar_conf', 1.0)),
            corridor_obstacle_m=float(cor.get('obstacle_dist_m') or math.inf), corridor_conf=float(cor.get('conf', 0.0)),
            slope_ahead_deg=float(cor.get('slope_ahead_deg') or 0.0), path_blocked=self.blocked,
            blocked_reason=self.blocked_info[0], blocked_sensors=self.blocked_info[1], blocked_conf=self.blocked_info[2],
            has_path=self.path is not None, goal_reached=self.goal_reached, frontiers_left=self.frontiers_left,
            percent_known=float(self.map_stats.get('percent', 0.0)), scan_complete=self.scan_complete,
            scout_available=self.has_scout, on_hazard_cell=bool(self.map.lethal[cell]))
        self.scan_complete = None
        out = self.fsm.step(inp)
        self.out = out
        self.emit(out.traces)
        self._slope_trace(inp)
        if out.drive == 'stop' and self.cmd != (0.0, 0.0):
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

        if out.request_replan and mono - self._last_fail_replan > 1.0:
            self._last_fail_replan = mono
            self.blocked = False
            self.replan(visible=True)
            self.publish_claim()

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
            self._drive_since = time.monotonic()
            self._last_drive = out.drive
            self.rev_idx = 0
        x, y, yaw = self.pose
        v = w = 0.0
        if out.drive == 'follow' and self.path_xy is not None:
            vmax = self.limits.v_max * out.speed_scale
            v, w, idx, done = pure_pursuit(self.pose, self.path_xy, lookahead=1.5, v_max=vmax, w_max=self.w_max)
            self.path_idx = max(self.path_idx, idx)
            if done:
                self.goal_reached = True
                if self.goal_kind == 'science':
                    self._visited_science.append(self.goal)
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
              'path_remaining_m': round(self._remaining_m(), 1), 'caution': self.fsm.caution, 't': time.time()}
        self.pub_state.publish(String(data=json.dumps(st)))


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
        now = time.monotonic()
        if self._z_prev is not None and now > self._z_prev[0]:
            self.vz = 0.7 * self.vz + 0.3 * (self.z - self._z_prev[1]) / (now - self._z_prev[0])
        self._z_prev = (now, self.z)
        if self.ground_z is None:
            self.ground_z = self.z - 0.1

    def on_req(self, m: String):
        try:
            self.pending_req = json.loads(m.data)
        except json.JSONDecodeError:
            pass

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
        now = time.monotonic()
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
                           f'Nadir footprint ~{2 * self.cruise:.0f} m wide maximises new coverage per Wh', 0.85,
                           f'Flying survey leg at {self.CRUISE_V:.0f} m/s, {self.cruise:.0f} m AGL')
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
        now = time.time()
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
        err = self.cruise - self.agl
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
        elif out.state == 'SURVEY' and self.target is not None and self.agl > 0.7 * self.cruise:
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
              'scan_job': (self.fsm._job or {}).get('id'), 'orbit': bool(out and out.orbit), 't': time.time()}
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
