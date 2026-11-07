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
HAZARD_MIN_M = 1.0             # smallest permanent zone when the reverse could not back out of the full one
HAZARD_CLEAR_M = 0.6           # the zone always stays this far (≈ robot radius) from where the rover stands
AWAIT_TIMEOUT_S = 45.0
CAUTION_SPEED = 0.35
STOP_DIST_M = 2.5
CORRIDOR_COOLDOWN_S = 3.0
CRUMB_SPACING_M = 0.5
# critical damage → SOS → rescue (docs/protocol.md §8.6)
CRIT_HOLD_S = 1.0
DOCK_START_M = 3.0             # stand-off reached (A* goal) within this centre distance → perception-guided docking
DOCK_NEAR_M = 2.4              # ... or this close regardless of the path (the stand-off is 2.2 m out)
DOCK_M = 1.45                  # docked: centre distance (the world accepts a repair within 1.6 m)
DOCK_SCALE = 0.18              # ≈0.15 m/s at 2.2 g
DOCK_NOLOCK_S = 5.0            # docked by odometry alone if perception cannot confirm for this long
DOCK_TIMEOUT_S = 45.0
TRANSIT_TIMEOUT_S = 150.0
NO_PATH_ABORT_S = 6.0
REPAIR_RETRY_S = 8.0
REPAIR_TIMEOUT_S = 40.0
BACKOFF_M = 2.0
BACKOFF_TIMEOUT_S = 10.0
RESCUE_RETRY_S = 20.0          # don't re-accept an SOS we just abandoned
RESCUE_STATES = ('RESCUE_TRANSIT', 'RESCUE_DOCKING', 'REPAIRING')


DISPLAY_NAMES = {'scout': 'Scout Rover', 'recovery': 'Heavy Recovery Rover', 'drone': 'Survey Drone'}


class ReplanGate:
    """At most one FSM-requested re-plan per `min_gap_s`, without losing any. A request made inside the gap is held
    and served when the gap ends, provided the FSM is still in the state that asked: a later state that needs a route
    asks for its own. (Dropping it lost the rescue route when rescue_start followed RESUMING's frontier re-plan by
    0.1 s: the rover drove the frontier route away from its teammate until a quiet periodic refresh.)"""

    def __init__(self, min_gap_s: float = 1.0):
        self.min_gap_s = min_gap_s
        self._last = -math.inf
        self._pending: str | None = None

    def due(self, requested: bool, state: str, now: float) -> bool:
        if requested:
            self._pending = state
        if self._pending is None or now - self._last <= self.min_gap_s:
            return False
        ok = self._pending == state
        self._pending = None
        if ok:
            self._last = now
        return ok


class ImuWindow:
    """What the 100 Hz IMU saw between two 10 Hz decision ticks: the attitude at the largest tilt and the largest
    angular rate held for two consecutive samples (one-sample rigid-contact chatter is not a rotation). Feeding the
    FSM these instead of the instantaneous values at tick time keeps a fast jolt that has already decayed by the
    next tick (a Gazebo roll overshoot lasts well under 100 ms)."""

    def __init__(self):
        self._att = None                    # (roll, pitch, tilt) at the window's largest tilt
        self._rate = 0.0
        self._prev_rate = 0.0

    def add(self, roll_deg: float, pitch_deg: float, rate: float) -> None:
        if not (math.isfinite(roll_deg) and math.isfinite(pitch_deg) and math.isfinite(rate)):
            return                          # an initialising IMU (zero quaternion) is not an attitude
        tilt = max(abs(roll_deg), abs(pitch_deg))
        if self._att is None or tilt > self._att[2]:
            self._att = (roll_deg, pitch_deg, tilt)
        self._rate = max(self._rate, min(rate, self._prev_rate))
        self._prev_rate = rate

    def take(self):
        """(roll, pitch, rate) of the window since the last take, or None if no sample arrived; starts a new one."""
        if self._att is None:
            return None
        out = (self._att[0], self._att[1], self._rate)
        self._att, self._rate = None, 0.0
        return out


def role_of(robot: str) -> str:
    """rover_0 is the scout rover, every other rover is a recovery rover, scouts are drones (§8.1)."""
    if robot.startswith('rover'):
        return 'scout' if robot == 'rover_0' else 'recovery'
    return 'drone'


def display_name(robot: str) -> str:
    kind, _, k = robot.partition('_')
    n = int(k) + 1 if k.isdigit() else 1
    role = role_of(robot)
    return DISPLAY_NAMES[role] if role == 'drone' and n == 1 else f'{DISPLAY_NAMES[role]} {n}'


def unit_label(robot: str) -> str:
    """rover_0 → ROVER_1, rover_1 → ROVER_2, scout_0 → DRONE (the names operators see)."""
    kind, _, k = robot.partition('_')
    n = int(k) + 1 if k.isdigit() else 1
    return f'ROVER_{n}' if kind == 'rover' else ('DRONE' if n == 1 else f'DRONE_{n}')


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
    health: dict | None = None       # RoverHealth.state().as_dict()
    rescuer_available: bool = False  # a recovery rover other than this one is in the fleet
    sos: dict | None = None          # active SOS from a teammate (/fleet/sos), recovery role only
    rescue_range_m: float | None = None   # centre distance to the SOS position (own odometry vs. reported pose)
    rescue_lock: dict | None = None  # perception confirmation of the teammate {range_m, conf, sources}
    sos_resolved: bool = False       # the rescue target reported itself repaired


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
    def __init__(self, robot: str, limits, g_label: str | None = None, role: str = 'scout'):
        self.robot = robot
        self.role = role
        self.rescue: dict | None = None          # {id, role, peer, phase, range_m, locked}
        self._abandoned: dict[str, float] = {}
        self._sos_seq = 0
        self._sos_id: str | None = None
        self._repair_sent = 0.0
        self._backoff: tuple | None = None
        self._resume_txt = ('Recovery complete; hazard zone excluded from cost map',
                            'Safe path available on the updated global map')
        self.lim = limits
        self.g_label = g_label or f'{limits.g:.1f}g'
        self.tb = TraceBuilder(robot)
        self.state = 'BOOT'
        self.caution = False
        self._crumbs: list[tuple] = []
        self._since: dict[str, float | None] = {}
        self._entered = 0.0
        self._fail_xy = None
        self._fail_yaw = 0.0
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
        if self.rescue:
            self._abort_rescue(out, i.now, f'{reason} failure on the rescuer itself', resume=False)
        self._fail_xy = (i.pose[0], i.pose[1])
        self._fail_yaw = i.pose[2]
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

        h = i.health or {}
        if h.get('critical') and not (self._fail_reason == 'critical' and self.state in ('SAFE_HOLD', 'AWAIT_RESCUE')):
            self._enter_critical(i, out, h)
            return out

        if self.state == 'MISSION_COMPLETE':
            return self._out(drive='stop', speed_scale=0.0)

        active = self.state in ('EXPLORING', 'CAUTION', 'REPLANNING', 'RESUMING') + RESCUE_STATES
        if active and self._check_failure(i, out):
            return out

        if self.state == 'SAFE_HOLD' and self._fail_reason == 'critical':
            if now - self._entered >= CRIT_HOLD_S:
                self._go('AWAIT_RESCUE', now)
                out = self._out(drive='stop', speed_scale=0.0)
                self._sos_seq += 1
                self._sos_id = f'sos-{self.robot}-{self._sos_seq}'
                out.actions.append({'type': 'sos', 'status': 'active', 'id': self._sos_id,
                                    'health_pct': float(h.get('pct', 0.0)), 'faults': list(h.get('faults', []))})
                if i.rescuer_available:
                    act = 'SOS broadcast on /fleet/sos; requesting the recovery rover for a field repair'
                else:
                    act = 'SOS broadcast; no recovery unit in this fleet: holding safe for the operator'
                self._tr(out, 'sos', ['DIAG', 'FLEET'],
                         f'Immobilised at ({i.pose[0]:.1f}, {i.pose[1]:.1f}), health {h.get("pct", 0):.0f}%',
                         'Self-recovery impossible with a failed drivetrain; a teammate can dock and repair', 0.95,
                         act, 'critical', now)
                return out
            return self._out(drive='stop', speed_scale=0.0)

        if self.state == 'AWAIT_RESCUE':
            if not h.get('critical'):
                self._fail_reason = ''
                self._go('RESUMING', now)
                out = self._out(request_replan=True)
                out.actions.append({'type': 'sos', 'status': 'resolved', 'id': self._sos_id,
                                    'health_pct': float(h.get('pct', 100.0)), 'faults': list(h.get('faults', []))})
                self._resume_txt = ('Field repair complete; drivetrain verified by motion test pending',
                                    'Health back above the recovery threshold')
                self._tr(out, 'repaired', ['DIAG', 'IMU'],
                         f'Motor-controller diagnostics nominal: health {h.get("pct", 100):.0f}% '
                         f'(drivetrain {h.get("drive_pct", 100):.0f}%, structure {h.get("structure_pct", 100):.0f}%)',
                         'Above the recovery threshold; SOS resolved', 0.93,
                         'Releasing brakes; re-planning and resuming exploration', 'info', now)
                return out
            return self._out(drive='stop', speed_scale=0.0)

        if self.state in RESCUE_STATES:
            return self._step_rescue(i, out)

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
                (hx, hy), hr, d = self._hazard_zone(i.pose)
                out.actions.append({'type': 'hazard_mark', 'xy': (hx, hy), 'radius_m': hr,
                                    'reason': self._fail_reason})
                where = f'({i.pose[0]:.1f}, {i.pose[1]:.1f})'
                txt = (f'Reached safe cell {where}' if arrived else
                       f'Reverse timed out after {REVERSE_TIMEOUT_S:.0f} s at {where}, {d:.1f} m from the failure point')
                why = f'{self._fail_reason.capitalize()} event at ({fx:.1f}, {fy:.1f}) must never be re-entered'
                if hr < HAZARD_RADIUS_M:
                    why += '; zone sized to keep this rover outside it so it can still plan a way out'
                self._tr(out, 'hazard_marked', ['IMU', 'MAP'], txt, why, 0.95 if arrived else 0.8,
                         f'Marked {hr:.1f} m zone as PERMANENT hazard on /planet_map', 'warn', now)
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
            txt, why = self._resume_txt
            self._resume_txt = ('Recovery complete; hazard zone excluded from cost map',
                                'Safe path available on the updated global map')
            self._tr(out, 'resume', ['MAP'], txt, why, 0.9, 'Resuming exploration', 'info', now)
            return out

        if self.role == 'recovery' and self.state in ('EXPLORING', 'CAUTION', 'REPLANNING'):
            sos = i.sos
            if (sos and sos.get('status', 'active') == 'active' and sos.get('robot') != self.robot
                    and now - self._abandoned.get(sos.get('id'), -1e9) >= RESCUE_RETRY_S):
                return self._start_rescue(i, sos)

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

    # ---------------------------------------------------------------- critical damage / rescue
    def _hazard_zone(self, pose):
        """Permanent zone for the failure: HAZARD_RADIUS_M around the failure point, never containing the rover.

        A rover inside its own lethal core cannot plan out (A* only escapes inflation rings) and no rescuer can reach
        it. If the reverse stopped short, the zone shrinks to keep HAZARD_CLEAR_M from the rover; if not even
        HAZARD_MIN_M fits, a HAZARD_MIN_M disc goes just beyond the rover, towards the failure point (or ahead along
        the heading it had when it failed). Returns ((x, y), radius, distance rover → failure point)."""
        fx, fy = self._fail_xy
        d = math.hypot(fx - pose[0], fy - pose[1])
        if d >= HAZARD_RADIUS_M:
            return (fx, fy), HAZARD_RADIUS_M, d
        if d - HAZARD_CLEAR_M >= HAZARD_MIN_M:
            return (fx, fy), round(d - HAZARD_CLEAR_M, 2), d
        if d > 0.05:
            ux, uy = (fx - pose[0]) / d, (fy - pose[1]) / d
        else:
            ux, uy = math.cos(self._fail_yaw), math.sin(self._fail_yaw)
        c = HAZARD_CLEAR_M + HAZARD_MIN_M
        return (pose[0] + c * ux, pose[1] + c * uy), HAZARD_MIN_M, d

    def _enter_critical(self, i: RoverInputs, out: RoverOutput, h: dict):
        if self.rescue:
            self._abort_rescue(out, i.now, 'rescuer critically damaged', resume=False)
        self._fail_xy = (i.pose[0], i.pose[1])
        self._fail_yaw = i.pose[2]
        self._fail_reason = 'critical'
        self._go('SAFE_HOLD', i.now)
        out.state, out.drive, out.speed_scale = self.state, 'stop', 0.0
        out.request_replan, out.reverse_path = False, []
        faults = ', '.join(h.get('faults', [])) or 'no fault codes'
        self._tr(out, 'critical_damage', ['IMU', 'DIAG', 'ODOM'],
                 f'Health {h.get("pct", 0):.1f}% (structure {h.get("structure_pct", 0):.0f}%, drivetrain '
                 f'{h.get("drive_pct", 0):.0f}%): {faults}; last IMU shock {h.get("last_shock_g", h.get("peak_g", 0)):.1f} g',
                 'Below the 15% critical threshold: the drivetrain cannot deliver commanded torque and further '
                 'motion risks losing the unit', 0.96,
                 'SAFE-HOLD: motors disabled, brakes engaged; preparing SOS', 'critical', i.now)

    def _start_rescue(self, i: RoverInputs, sos: dict) -> RoverOutput:
        now = i.now
        target = sos['robot']
        self._backoff = None
        self.rescue = {'id': sos.get('id'), 'role': 'rescuer', 'peer': target, 'phase': 'TRANSIT',
                       'range_m': i.rescue_range_m, 'locked': False}
        self._go('RESCUE_TRANSIT', now)
        out = self._out(request_replan=True, drive='follow')
        out.actions.append({'type': 'rescue', 'phase': 'TRANSIT', 'target': target, 'id': sos.get('id')})
        rng = f', {i.rescue_range_m:.1f} m away' if i.rescue_range_m is not None else ''
        xy = sos.get('xy') or (0.0, 0.0)
        self._tr(out, 'rescue_start', ['FLEET', 'DIAG'],
                 f'SOS from {target} ({unit_label(target)}): health {float(sos.get("health_pct", 0)):.0f}% at '
                 f'({xy[0]:.1f}, {xy[1]:.1f}){rng}',
                 'This unit is the recovery rover with a nominal drivetrain; saving a fleet asset outweighs '
                 'exploration', 0.94,
                 f'AUTONOMOUS RESCUE MODE: A* to a stand-off ≈2.2 m beside {unit_label(target)}', 'warn', now)
        return out

    def _set_phase(self, out: RoverOutput, phase: str):
        self.rescue['phase'] = phase
        out.actions.append({'type': 'rescue', 'phase': phase, 'target': self.rescue['peer'], 'id': self.rescue['id']})

    def _abort_rescue(self, out: RoverOutput, now: float, why: str, resume: bool = True):
        r = self.rescue
        self.rescue = None
        self._backoff = None                 # a back-off cut short must not leak into the next rescue
        if r is None:
            return
        self._abandoned[r['id']] = now
        out.actions.append({'type': 'rescue', 'phase': 'ABORTED', 'target': r['peer'], 'id': r['id']})
        self._tr(out, 'rescue_abort', ['FLEET'], f'Rescue of {unit_label(r["peer"])} interrupted: {why}',
                 'Cannot complete the rescue safely', 0.8,
                 'Rescue aborted; resuming exploration' if resume else 'Rescue aborted', 'warn', now)
        if resume:
            self._go('RESUMING', now)
            out.state, out.drive, out.request_replan = self.state, 'stop', True
            self._resume_txt = (f'Rescue of {unit_label(r["peer"])} ended', 'Exploration goal re-selected')

    def _step_rescue(self, i: RoverInputs, out: RoverOutput) -> RoverOutput:
        now = i.now
        r = self.rescue
        sos = i.sos
        if r is None:
            self._go('RESUMING', now)
            return self._out(request_replan=True)
        if i.rescue_range_m is not None:
            r['range_m'] = round(float(i.rescue_range_m), 2)
        lock = i.rescue_lock
        r['locked'] = lock is not None
        label = unit_label(r['peer'])
        withdrawn = (sos is None or sos.get('id') != r['id'] or sos.get('status') == 'resolved')

        if self.state == 'RESCUE_TRANSIT':
            self._update_caution(i, out)
            if withdrawn:
                self._abort_rescue(out, now, 'SOS withdrawn')
                return out
            if now - self._entered > TRANSIT_TIMEOUT_S:
                self._abort_rescue(out, now, f'no stand-off reached in {TRANSIT_TIMEOUT_S:.0f} s')
                return out
            rng = i.rescue_range_m
            if rng is not None and (rng <= DOCK_NEAR_M or (i.goal_reached and rng <= DOCK_START_M)):
                self._go('RESCUE_DOCKING', now)
                o = self._out(drive='dock', speed_scale=DOCK_SCALE)
                o.traces = out.traces
                self._set_phase(o, 'DOCKING')
                self._tr(o, 'rescue_dock', ['ODOM', 'FLEET'], f'Stand-off reached: {label} {rng:.1f} m ahead',
                         'Final approach needs centimetre-level range, so control switches from the map to '
                         'direct camera / LiDAR perception of the teammate', 0.9,
                         f'Perception-guided docking at {DOCK_SCALE * self.lim.v_max:.2f} m/s', 'info', now)
                return o
            if self._held('rescue_nopath', not i.has_path, now, NO_PATH_ABORT_S):
                self._abort_rescue(out, now, f'no traversable route to {label} for {NO_PATH_ABORT_S:.0f} s')
                return out
            if not i.has_path:
                o = self._out(drive='stop', request_replan=True)
                o.traces = out.traces
                return o
            corridor_hit = (i.corridor_obstacle_m < STOP_DIST_M and i.corridor_conf > 0.5
                            and now >= self._corridor_quiet_until
                            and (rng is None or rng > STOP_DIST_M + 1.5))    # near the target the return IS the target
            if i.path_blocked or corridor_hit or i.goal_reached:
                self._corridor_quiet_until = now + CORRIDOR_COOLDOWN_S
                o = self._out(drive='stop', request_replan=True)
                o.traces = out.traces
                if i.path_blocked or corridor_hit:
                    self._tr(o, 'path_blocked', list(i.blocked_sensors) if i.path_blocked else ['LIDAR'],
                             i.blocked_reason or f'LiDAR corridor return {i.corridor_obstacle_m:.1f} m ahead',
                             f'Rescue route to {label} obstructed', i.blocked_conf if i.path_blocked else i.corridor_conf,
                             'Re-planning the rescue route', 'warn', now, force=False)
                return o
            o = self._out(drive='follow')
            o.traces = out.traces
            return o

        if self.state == 'RESCUE_DOCKING':
            if withdrawn:
                self._abort_rescue(out, now, 'SOS withdrawn')
                return out
            if now - self._entered > DOCK_TIMEOUT_S:
                self._abort_rescue(out, now, f'docking not achieved in {DOCK_TIMEOUT_S:.0f} s')
                return out
            if lock is not None and not r.get('lock_traced'):
                r['lock_traced'] = True
                self._tr(out, 'rescue_lock', list(lock.get('sources', ['CAM'])),
                         f'[PERCEIVED: DAMAGED UNIT {label} | DOCKING FOR REPAIR] at {float(lock["range_m"]):.1f} m',
                         'Camera box, depth and LiDAR returns coincide with the SOS position: target confirmed',
                         float(lock.get('conf', 0.9)), 'Visual lock-on; perception-locked final approach', 'info', now)
            rng = i.rescue_range_m
            close = rng is not None and rng <= DOCK_M
            blind = self._held('dock_nolock', close and lock is None, now, DOCK_NOLOCK_S)
            if close and (lock is not None or blind):
                self._go('REPAIRING', now)
                o = self._out(drive='stop', speed_scale=0.0)
                o.traces = out.traces
                self._set_phase(o, 'REPAIRING')
                o.actions.append({'type': 'repair', 'target': r['peer']})
                self._repair_sent = now
                how = (f'perceived at {float(lock["range_m"]):.2f} m ({"+".join(lock.get("sources", []))})'
                       if lock is not None else 'odometry only: perception could not confirm')
                self._tr(o, 'docked', (list(lock.get('sources', [])) if lock else []) + ['ODOM'],
                         f'Docked {rng:.2f} m from {label}, {how}', 'Within the 1.5 m repair envelope',
                         float(lock.get('conf', 0.9)) if lock else 0.6, 'Holding position for repair', 'info', now)
                self._tr(o, 'repair_start', ['FLEET'], f'Repair link to {label} established',
                         'Field repair restores the drivetrain; the damaged unit verifies it by diagnostics', 0.9,
                         f'Executing field repair on {label}', 'info', now)
                return o
            o = self._out(drive='dock', speed_scale=DOCK_SCALE)
            o.traces = out.traces
            return o

        # REPAIRING (+ backing off once the target reports itself repaired)
        if self._backoff is not None:
            bx, by = self._backoff
            if math.hypot(i.pose[0] - bx, i.pose[1] - by) < ARRIVE_M or now - self._entered > BACKOFF_TIMEOUT_S:
                self._backoff = None
                self._set_phase(out, 'DONE')
                self.rescue = None
                self._go('RESUMING', now)
                out.state, out.drive, out.request_replan = self.state, 'stop', True
                self._resume_txt = (f'Rescue of {label} complete; backed off {BACKOFF_M:.0f} m',
                                    'Both units operational; exploration goals re-claimed')
                return out
            return self._out(drive='reverse', reverse_path=[self._backoff], speed_scale=REVERSE_SPEED)
        if i.sos_resolved:
            x, y, yaw = i.pose
            self._backoff = (x - BACKOFF_M * math.cos(yaw), y - BACKOFF_M * math.sin(yaw))
            self._entered = now
            out = self._out(drive='reverse', reverse_path=[self._backoff], speed_scale=REVERSE_SPEED)
            self._tr(out, 'repair_complete', ['FLEET', 'DIAG'], f'{label} reports drivetrain nominal; SOS resolved',
                     'Repair verified by the damaged unit\'s own diagnostics', 0.95,
                     f'Backing off {BACKOFF_M:.0f} m to clear its path', 'info', now)
            return out
        if now - self._entered > REPAIR_TIMEOUT_S:
            self._abort_rescue(out, now, f'repair not confirmed in {REPAIR_TIMEOUT_S:.0f} s')
            return out
        o = self._out(drive='stop', speed_scale=0.0)
        if now - self._repair_sent >= REPAIR_RETRY_S:
            self._repair_sent = now
            o.actions.append({'type': 'repair', 'target': r['peer']})
        return o

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
