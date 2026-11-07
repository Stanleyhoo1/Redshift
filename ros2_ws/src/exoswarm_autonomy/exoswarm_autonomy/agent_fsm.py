"""Pure state machines for the rover and the aerial scout.

Every transition emits a decision trace linking sensor input → reasoning → action. The ROS planner
node feeds `RoverInputs` at 10 Hz and executes the returned `RoverOutput` (drive mode, re-plan
requests and fleet actions such as hazard marks and scout scan requests).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .trace import TraceBuilder

_req_ids = itertools.count(1)

# thresholds (docs/superpowers/plans: "Physics & autonomy models")
CAUTION_ENTER_U = 0.45
CAUTION_ENTER_CAM = 0.40
CAUTION_ENTER_S = 0.5
CAUTION_EXIT_U = 0.35
CAUTION_EXIT_CAM = 0.70
CAUTION_EXIT_S = 2.0
SAFEHOLD_U = 0.80
SAFEHOLD_U_S = 1.5
TILT_S = 0.10
GYRO_SPIKE = 2.0
SLIP_LIMIT = 0.6
SLIP_S = 0.8
HOLD_S = 1.5
REVERSE_MIN_M = 3.0
REVERSE_SPEED = 0.3
REVERSE_TIMEOUT_S = 15.0
ARRIVE_M = 0.5
HAZARD_RADIUS_M = 2.5
AWAIT_TIMEOUT_S = 45.0
CAUTION_SPEED = 0.35
STOP_DIST_M = 2.5
CORRIDOR_COOLDOWN_S = 3.0
CRUMB_SPACING_M = 0.5


@dataclass
class RoverInputs:
    now: float
    pose: tuple                      # (x, y, yaw)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    gyro_rate: float = 0.0           # |ω| rad/s
    slip: float = 0.0
    uncertainty: float = 0.0
    camera_conf: float = 1.0
    lidar_conf: float = 1.0
    corridor_obstacle_m: float = math.inf
    corridor_conf: float = 0.0
    slope_ahead_deg: float = 0.0
    path_blocked: bool = False
    blocked_reason: str = ''
    blocked_sensors: tuple = ('LIDAR', 'MAP')
    blocked_conf: float = 0.9
    has_path: bool = True
    goal_reached: bool = False
    frontiers_left: bool = True
    percent_known: float = 0.0
    scan_complete: dict | None = None
    scout_available: bool = True
    on_hazard_cell: bool = False


@dataclass
class RoverOutput:
    state: str
    mode: str = 'FUSED'
    speed_scale: float = 1.0
    drive: str = 'follow'            # follow | stop | reverse
    request_replan: bool = False
    reverse_path: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    traces: list = field(default_factory=list)


class RoverFSM:
    def __init__(self, robot: str, limits, g_label: str | None = None):
        self.robot = robot
        self.lim = limits
        self.g_label = g_label or f'{limits.g:.1f}g'
        self.tb = TraceBuilder(robot)
        self.state = 'BOOT'
        self.caution = False
        self._crumbs: list[tuple] = []
        self._since: dict[str, float | None] = {}
        self._entered = 0.0
        self._fail_xy = None
        self._fail_reason = ''
        self._reverse_path: list = []
        self._req = None
        self._corridor_quiet_until = -1.0

    # ---------------------------------------------------------------- helpers
    def breadcrumb_count(self) -> int:
        return len(self._crumbs)

    @property
    def last_safe(self):
        return self._crumbs[-1][:2] if self._crumbs else None

    def _held(self, key: str, cond: bool, now: float, dur: float) -> bool:
        if not cond:
            self._since[key] = None
            return False
        if self._since.get(key) is None:
            self._since[key] = now
        return now - self._since[key] >= dur - 1e-9

    def _go(self, state: str, now: float):
        self.state = state
        self._entered = now

    def _tr(self, out, kind, sensors, inp, reason, conf, action, sev='info', now=0.0, force=True):
        t = self.tb.make(kind, self.state, sensors, inp, reason, conf, action, sev, now=now, force=force)
        if t:
            out.traces.append(t)

    def _out(self, **kw) -> RoverOutput:
        o = RoverOutput(state=self.state, mode='LIDAR_ONLY' if self.caution else 'FUSED',
                        speed_scale=CAUTION_SPEED if self.caution else 1.0)
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def _record_crumb(self, i: RoverInputs):
        tilt = max(abs(i.roll_deg), abs(i.pitch_deg))
        if tilt >= 0.5 * self.lim.tip_alarm_deg or i.uncertainty >= 0.6 or i.on_hazard_cell:
            return
        x, y = i.pose[0], i.pose[1]
        if not self._crumbs or math.hypot(x - self._crumbs[-1][0], y - self._crumbs[-1][1]) >= CRUMB_SPACING_M:
            self._crumbs.append((x, y, i.now))
            if len(self._crumbs) > 600:
                self._crumbs = self._crumbs[-600:]

    def _plan_retreat(self, pose):
        """Breadcrumb trail back to the most recent crumb at least REVERSE_MIN_M from the failure."""
        fx, fy = self._fail_xy
        for k in range(len(self._crumbs) - 1, -1, -1):
            cx, cy, _ = self._crumbs[k]
            if math.hypot(cx - fx, cy - fy) >= REVERSE_MIN_M:
                trail = [(c[0], c[1]) for c in reversed(self._crumbs[k:])]
                return trail, k
        # no usable crumbs: straight back along heading
        x, y, yaw = pose
        return [(x - REVERSE_MIN_M * math.cos(yaw), y - REVERSE_MIN_M * math.sin(yaw))], -1

    # ---------------------------------------------------------------- failure detection
    def _check_failure(self, i: RoverInputs, out: RoverOutput) -> bool:
        tilt = max(abs(i.roll_deg), abs(i.pitch_deg))
        axis = 'Roll' if abs(i.roll_deg) >= abs(i.pitch_deg) else 'Pitch'
        alarm = self.lim.tip_alarm_deg
        spike = i.gyro_rate > GYRO_SPIKE and tilt > self.lim.max_slope_deg
        if self._held('tilt', tilt > alarm, i.now, TILT_S) or spike:
            self._enter_hold(i, out, 'tip',
                             ['IMU'], f'{axis} {tilt:.1f}° (rate {i.gyro_rate:.1f} rad/s) on {self.g_label} terrain',
                             f'Exceeds {self.g_label} tip alarm {alarm:.1f}° (static tip {self.lim.tip_deg:.1f}°); '
                             f'continued motion risks rollover',
                             min(0.99, 0.75 + (tilt - alarm) / 40.0))
            return True
        if self._held('slip', i.slip > SLIP_LIMIT, i.now, SLIP_S):
            self._enter_hold(i, out, 'slip', ['ODOM', 'IMU'],
                             f'Wheel slip {i.slip * 100:.0f}% sustained {SLIP_S:.1f} s (encoders vs. odometry)',
                             f'Traction lost on {self.g_label} regolith (> {SLIP_LIMIT * 100:.0f}% slip limit)',
                             min(0.97, 0.7 + (i.slip - SLIP_LIMIT)))
            return True
        if self._held('blind', i.uncertainty > SAFEHOLD_U, i.now, SAFEHOLD_U_S):
            self._enter_hold(i, out, 'uncertainty', ['CAM', 'LIDAR'],
                             f'Uncertainty {i.uncertainty:.2f} (cam {i.camera_conf:.2f}, LiDAR {i.lidar_conf:.2f}) '
                             f'for {SAFEHOLD_U_S:.1f} s',
                             f'Above safe-hold threshold {SAFEHOLD_U:.2f}: cannot verify terrain ahead',
                             i.uncertainty)
            return True
        return False

    def _enter_hold(self, i, out, reason, sensors, inp_txt, reasoning, conf):
        self._fail_xy = (i.pose[0], i.pose[1])
        self._fail_reason = reason
        self._go('SAFE_HOLD', i.now)
        out.state = self.state
        out.drive = 'stop'
        out.speed_scale = 0.0
        self._tr(out, 'safe_hold', sensors, inp_txt, reasoning, conf,
                 'SAFE-HOLD: motors halted, brakes engaged; planning retreat to last verified-safe cell',
                 'critical', i.now)

    # ---------------------------------------------------------------- main step
    def step(self, i: RoverInputs) -> RoverOutput:
        out = self._out()
        now = i.now
        if self.state == 'BOOT':
            self._go('EXPLORING', now)
            out = self._out()
            self._tr(out, 'mission_start', ['IMU', 'ODOM'],
                     f'Systems nominal at ({i.pose[0]:.1f}, {i.pose[1]:.1f}); gravity {self.g_label}',
                     f'Max traversable slope {self.lim.max_slope_deg:.1f}° ({self.lim.factor.replace("_", " ")}-bound), '
                     f'tip alarm {self.lim.tip_alarm_deg:.1f}°, v_max {self.lim.v_max:.2f} m/s',
                     0.99, 'Begin frontier exploration', 'info', now)
            return out

        if self.state == 'MISSION_COMPLETE':
            return self._out(drive='stop', speed_scale=0.0)

        active = self.state in ('EXPLORING', 'CAUTION', 'REPLANNING', 'RESUMING')
        if active and self._check_failure(i, out):
            return out

        if self.state == 'SAFE_HOLD':
            tilt = max(abs(i.roll_deg), abs(i.pitch_deg))
            if now - self._entered >= HOLD_S and tilt < 0.6 * self.lim.tip_alarm_deg:
                self._reverse_path, k = self._plan_retreat(i.pose)
                tx, ty = self._reverse_path[-1]
                dist = math.hypot(tx - self._fail_xy[0], ty - self._fail_xy[1])
                self._go('RECOVERY_REVERSE', now)
                out = self._out(drive='reverse', reverse_path=list(self._reverse_path),
                                speed_scale=REVERSE_SPEED)
                self._tr(out, 'recovery_reverse', ['IMU', 'ODOM', 'MAP'],
                         f'Tilt settled at {tilt:.1f}°; last verified-safe cell {dist:.1f} m behind'
                         + (f' (breadcrumb #{k})' if k >= 0 else ''),
                         'Retracing own verified track avoids unmapped ground while the zone is unresolved',
                         0.93, f'Reversing along breadcrumbs at {REVERSE_SPEED:.1f} m/s to ({tx:.1f}, {ty:.1f})',
                         'warn', now)
                return out
            return self._out(drive='stop', speed_scale=0.0)

        if self.state == 'RECOVERY_REVERSE':
            tx, ty = self._reverse_path[-1]
            arrived = math.hypot(i.pose[0] - tx, i.pose[1] - ty) < ARRIVE_M
            if arrived or now - self._entered > REVERSE_TIMEOUT_S:
                out = self._out(drive='stop', speed_scale=0.0)
                fx, fy = self._fail_xy
                out.actions.append({'type': 'hazard_mark', 'xy': (fx, fy), 'radius_m': HAZARD_RADIUS_M,
                                    'reason': self._fail_reason})
                self._tr(out, 'hazard_marked', ['IMU', 'MAP'],
                         f'Reached safe cell ({i.pose[0]:.1f}, {i.pose[1]:.1f})',
                         f'{self._fail_reason.capitalize()} event at ({fx:.1f}, {fy:.1f}) must never be re-entered',
                         0.95, f'Marked {HAZARD_RADIUS_M:.1f} m zone as PERMANENT hazard on /planet_map', 'warn', now)
                if i.scout_available:
                    self._req = f'{self.robot}-{next(_req_ids)}'
                    out.actions.append({'type': 'scan_request', 'id': self._req, 'xy': (fx, fy),
                                        'reason': self._fail_reason})
                    self._go('AWAIT_SCOUT', now)
                    out.state = self.state
                    self._tr(out, 'scout_tasked', ['FLEET'],
                             'Ground sensors cannot see the failure zone from a safe stand-off',
                             'Aerial nadir depth scan resolves slope/subsidence without risking the rover',
                             0.9, f'Tasked scout to scan ({fx:.1f}, {fy:.1f}); holding position', 'info', now)
                else:
                    self._go('RESUMING', now)
                    out.state = self.state
                    out.request_replan = True
                return out
            return self._out(drive='reverse', reverse_path=list(self._reverse_path), speed_scale=REVERSE_SPEED)

        if self.state == 'AWAIT_SCOUT':
            sc = i.scan_complete
            if (sc and sc.get('id') == self._req) or now - self._entered > AWAIT_TIMEOUT_S:
                self._go('RESUMING', now)
                out = self._out(request_replan=True)
                if sc and sc.get('id') == self._req:
                    ms = sc.get('max_slope_deg')
                    self._tr(out, 'scan_complete', ['FLEET', 'MAP'],
                             f'Scout nadir scan: max slope {ms:.1f}°' if ms is not None else 'Scout scan complete',
                             str(sc.get('verdict', 'zone characterised')), 0.9,
                             'Re-planning route around the permanent hazard', 'info', now)
                else:
                    self._tr(out, 'scan_timeout', ['FLEET'], f'No scout report after {AWAIT_TIMEOUT_S:.0f} s',
                             'Hazard zone stays permanent; proceeding conservatively', 0.7,
                             'Re-planning route around the permanent hazard', 'warn', now)
                return out
            return self._out(drive='stop', speed_scale=0.0)

        if self.state == 'RESUMING':
            self._go('CAUTION' if self.caution else 'EXPLORING', now)
            out = self._out()
            self._tr(out, 'resume', ['MAP'], 'Recovery complete; hazard zone excluded from cost map',
                     'Safe path available on the updated global map', 0.9, 'Resuming exploration', 'info', now)
            return out

        # ---------------- nominal driving states: EXPLORING / CAUTION / REPLANNING
        self._record_crumb(i)
        self._update_caution(i, out)

        if self.state == 'REPLANNING':
            if i.has_path and not i.path_blocked:
                self._go('CAUTION' if self.caution else 'EXPLORING', now)
                self._corridor_quiet_until = now + CORRIDOR_COOLDOWN_S
                o = self._out()
                o.traces = out.traces
                return o
            o = self._out(drive='stop', request_replan=True)
            o.traces = out.traces
            return o

        corridor_hit = (i.corridor_obstacle_m < STOP_DIST_M and i.corridor_conf > 0.5
                        and now >= self._corridor_quiet_until)
        if i.path_blocked or corridor_hit:
            self._go('REPLANNING', now)
            o = self._out(drive='stop', request_replan=True)
            o.traces = out.traces
            if i.path_blocked:
                self._tr(o, 'path_blocked', list(i.blocked_sensors), i.blocked_reason or 'Planned path crosses new lethal cells',
                         'Global map update invalidates the current path', i.blocked_conf,
                         'Halting and re-planning A* path around obstacle', 'warn', now)
            else:
                self._tr(o, 'path_blocked', ['LIDAR'],
                         f'LiDAR corridor return {i.corridor_obstacle_m:.1f} m ahead',
                         f'Inside {STOP_DIST_M:.1f} m stopping envelope at {self.g_label}', i.corridor_conf,
                         'Emergency stop; re-planning around obstacle', 'warn', now)
            return o

        if i.goal_reached or not i.has_path:
            if not i.frontiers_left:
                self._go('MISSION_COMPLETE', now)
                o = self._out(drive='stop', speed_scale=0.0)
                self._tr(o, 'mission_complete', ['MAP'], f'No reachable frontiers; {i.percent_known:.1f}% mapped',
                         'Sector exploration complete', 0.95, 'Holding position', 'info', now)
                return o
            out.actions.append({'type': 'new_goal'})
        out.state = self.state
        out.mode = 'LIDAR_ONLY' if self.caution else 'FUSED'
        out.speed_scale = CAUTION_SPEED if self.caution else 1.0
        return out

    def _update_caution(self, i: RoverInputs, out: RoverOutput):
        now = i.now
        if not self.caution:
            bad = i.uncertainty > CAUTION_ENTER_U or i.camera_conf < CAUTION_ENTER_CAM
            if self._held('caution_in', bad, now, CAUTION_ENTER_S):
                self.caution = True
                self._since['caution_out'] = None
                if self.state == 'EXPLORING':
                    self.state = 'CAUTION'
                self._tr(out, 'caution_enter', ['CAM', 'LIDAR'],
                         f'Camera confidence {i.camera_conf:.2f}, LiDAR {i.lidar_conf:.2f} → uncertainty {i.uncertainty:.2f}',
                         f'Above caution threshold {CAUTION_ENTER_U:.2f}: visual detections unreliable',
                         max(0.6, i.uncertainty), f'CAUTION: LiDAR-only navigation at {CAUTION_SPEED * 100:.0f}% speed, '
                         f'+0.4 m hazard inflation', 'warn', now)
        else:
            good = i.uncertainty < CAUTION_EXIT_U and i.camera_conf > CAUTION_EXIT_CAM
            if self._held('caution_out', good, now, CAUTION_EXIT_S):
                self.caution = False
                self._since['caution_in'] = None
                if self.state == 'CAUTION':
                    self.state = 'EXPLORING'
                self._tr(out, 'caution_exit', ['CAM', 'LIDAR'],
                         f'Camera confidence recovered to {i.camera_conf:.2f} for {CAUTION_EXIT_S:.0f} s',
                         f'Uncertainty {i.uncertainty:.2f} below {CAUTION_EXIT_U:.2f}', 0.9,
                         'Restoring fused camera+LiDAR navigation at full speed', 'info', now)


# ====================================================================== scout
@dataclass
class ScoutInputs:
    now: float
    pose: tuple                 # (x, y, z)
    altitude_agl: float
    goal_reached: bool = False
    scan_request: dict | None = None
    scan_stats: dict | None = None
    frontiers_left: bool = True


@dataclass
class ScoutOutput:
    state: str
    target_xy: tuple | None = None
    orbit: bool = False
    hover: bool = False
    actions: list = field(default_factory=list)
    traces: list = field(default_factory=list)


class ScoutFSM:
    SCAN_S = 4.0
    ARRIVE_M = 1.0

    def __init__(self, robot: str, cruise_agl: float = 12.0, max_slope_deg: float = 16.8):
        self.robot = robot
        self.cruise_agl = cruise_agl
        self.max_slope = max_slope_deg
        self.tb = TraceBuilder(robot)
        self.state = 'BOOT'
        self._entered = 0.0
        self._queue: list[dict] = []
        self._job: dict | None = None

    def _tr(self, out, kind, sensors, inp, reason, conf, action, sev='info', now=0.0):
        t = self.tb.make(kind, self.state, sensors, inp, reason, conf, action, sev, now=now, force=True)
        if t:
            out.traces.append(t)

    def step(self, i: ScoutInputs) -> ScoutOutput:
        now = i.now
        if i.scan_request and all(j['id'] != i.scan_request['id'] for j in self._queue) and \
                (self._job is None or self._job['id'] != i.scan_request['id']):
            self._queue.append(dict(i.scan_request))
        if self.state == 'BOOT':
            self.state, self._entered = 'TAKEOFF', now
        if self.state == 'TAKEOFF':
            if i.altitude_agl >= 0.9 * self.cruise_agl:
                self.state, self._entered = 'SURVEY', now
                out = ScoutOutput(self.state)
                self._tr(out, 'airborne', ['IMU', 'ODOM'], f'Cruise altitude {i.altitude_agl:.1f} m AGL reached',
                         'Nadir depth footprint ~2x altitude; aerial survey maximises map coverage per watt', 0.98,
                         'Begin aerial frontier survey', 'info', now)
                return out
            return ScoutOutput(self.state, hover=True)
        if self.state == 'SURVEY' and self._queue:
            self._job = self._queue.pop(0)
            self.state, self._entered = 'FLY_TO_SCAN', now
            out = ScoutOutput(self.state, target_xy=tuple(self._job['xy']))
            self._tr(out, 'scan_tasked', ['FLEET'],
                     f"Scan request {self._job['id']} from {self._job.get('robot', '?')} ({self._job.get('reason', '')})",
                     'Priority task: characterise ground-failure zone the rover cannot safely observe', 0.95,
                     f"Pre-empting survey; flying to ({self._job['xy'][0]:.1f}, {self._job['xy'][1]:.1f})", 'warn', now)
            return out
        if self.state == 'FLY_TO_SCAN':
            tx, ty = self._job['xy']
            if math.hypot(i.pose[0] - tx, i.pose[1] - ty) < self.ARRIVE_M:
                self.state, self._entered = 'SCANNING', now
                out = ScoutOutput(self.state, target_xy=(tx, ty), orbit=True)
                self._tr(out, 'scanning', ['DEPTH', 'CAM'], f'Over target at {i.altitude_agl:.1f} m AGL',
                         'Orbiting to build dense nadir depth coverage of the zone', 0.9,
                         f'Scanning for {self.SCAN_S:.0f} s', 'info', now)
                return out
            return ScoutOutput(self.state, target_xy=(tx, ty))
        if self.state == 'SCANNING':
            tx, ty = self._job['xy']
            elapsed = now - self._entered
            if elapsed >= self.SCAN_S and (i.scan_stats is not None or elapsed >= self.SCAN_S + 2.0):
                st = i.scan_stats or {}
                ms = st.get('max_slope_deg')
                if ms is not None and ms > self.max_slope:
                    verdict = f'Confirmed: slopes up to {ms:.1f}° exceed the {self.max_slope:.1f}° limit'
                elif ms is not None:
                    verdict = (f'Surface slope {ms:.1f}° within limit: subsurface instability suspected, '
                               f'zone kept as permanent hazard')
                else:
                    verdict = 'Insufficient depth returns: zone kept as permanent hazard'
                act = {'type': 'scan_complete', 'id': self._job['id'], 'xy': [tx, ty],
                       'max_slope_deg': ms, 'mean_slope_deg': st.get('mean_slope_deg'), 'verdict': verdict}
                self.state, self._entered = 'SURVEY', now
                self._job = None
                out = ScoutOutput(self.state, actions=[act])
                self._tr(out, 'scan_complete', ['DEPTH', 'MAP'],
                         f'Nadir scan of ({tx:.1f}, {ty:.1f}) complete' + (f': max slope {ms:.1f}°' if ms is not None else ''),
                         verdict, 0.9, 'Reported to fleet; resuming survey', 'info', now)
                return out
            return ScoutOutput(self.state, target_xy=(tx, ty), orbit=True)
        if self.state == 'SURVEY' and not i.frontiers_left:
            self.state = 'MISSION_COMPLETE'
        if self.state == 'MISSION_COMPLETE':
            return ScoutOutput(self.state, hover=True)
        return ScoutOutput(self.state, actions=[{'type': 'new_goal'}] if i.goal_reached else [])
