"""Per-rover embodied agent: LiDAR + spectrometer, skid-steer controller, battery, state machine.

States (protocol enum): exploring, investigating, returning, charging, recovering, idle.
"""
from __future__ import annotations

import math
from collections import deque

import mujoco
import numpy as np

from .config import Config
from .planner import downsample, path_length
from .terrain import KIND_LABEL
from .world import GROUP_ROCK

LIDAR_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)   # terrain, rocks, targets; never rovers (group 3)
BUMPER_GROUPS = np.array([0, 1, 0, 0, 0, 0], np.uint8)  # rocks only


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Rover:
    def __init__(self, sim, idx: int, rid: str, name: str, color: str, dock: tuple):
        self.sim = sim
        self.cfg: Config = sim.cfg
        self.fleet = sim.fleet
        m = sim.model
        self.idx, self.id, self.name, self.color = idx, rid, name, color
        self.dock = dock
        self.body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rid)
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{rid}_root")
        self.qadr = int(m.jnt_qposadr[jid])
        self.vadr = int(m.jnt_dofadr[jid])
        jnts = [j for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == self.body]
        self.nq = sum(7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1 for j in jnts)
        self.nv = sum(6 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1 for j in jnts)
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{rid}_lidar")
        # own spec sheet: wheel torque needed per unit sin(slope) to hold the whole rover against gravity
        self.mass = float(m.body_subtreemass[self.body])
        self.grav_torque = self.mass * self.cfg.gravity * self.cfg.wheel_radius
        self.acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{rid}_m{k}") for k in range(4)]
        self.wdofs = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{rid}_w{k}")])
                      for k in range(4)]
        self.rng = np.random.default_rng(self.cfg.seed * 101 + idx)

        # lidar buffers
        self.el = np.array(self.cfg.lidar_el)
        self.n_az = self.cfg.lidar_az
        self.nray = self.n_az * len(self.el)
        self.ray_gid = np.zeros(self.nray, np.int32)
        self.ray_dist = np.zeros(self.nray, np.float64)
        self.bump_gid = np.zeros(3, np.int32)
        self.bump_dist = np.zeros(3, np.float64)

        # state
        self.state = "exploring"
        self.prev_state = "exploring"
        self.battery = 1.0
        self.goal = None
        self.goal_size = 0
        self.goal_t = -1e9
        self.path: list = []
        self.wp = 0
        self.plan_t = -1e9
        self.plan_limit_version = -1
        self.next_goal_try = 0.0
        self.next_goal_eval = 0.0
        self.last_goal_event_t = -1e9
        self.target_idx = None
        self.dwell_until = None
        self.inv_fails = 0
        self.next_target_check = 0.0
        self.rec_phase = None
        self.rec_until = 0.0
        self.rec_dir = 1.0
        self.stall_cooldown = 3.0
        self.hold_until = 0.0
        self.yield_since = None
        self.flip_since = None
        self.recent_stalls: deque = deque(maxlen=6)
        self.hist: deque = deque(maxlen=int(round(self.cfg.stall_window_s * self.cfg.agent_hz)) + 1)
        self.stuck_ref = None
        self.climb_hist: deque = deque(maxlen=int(round(2.0 * self.cfg.agent_hz)) + 1)
        self.odometer = 0.0
        self.last_xy = None
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.e_per_m = self.cfg.battery_motor_k * 45.0      # battery fraction per metre (motor), learned
        self.e_acc_drain = 0.0
        self.e_acc_dist = 0.0
        self.home_dist = None
        self.home_dist_t = -1e9
        self.home_dist_xy = (0.0, 0.0)
        self.need_home = 0.0
        self.next_home_check = 0.0
        self.no_goal_since = None
        self.return_reason = None
        self.dock_near_since = None
        self.goal_fail_reason = None
        self.blacklist: dict = {}          # frontier bucket -> time until which this rover skips it
        self.x = self.y = self.z = self.yaw = 0.0
        self.tilt = self.pitch = 0.0
        self.speed = 0.0
        self.speed_fwd = 0.0
        self.read_pose()
        self.last_xy = (self.x, self.y)

    # ================================================================== sensing
    def read_pose(self) -> None:
        d = self.sim.data
        q = d.qpos
        a = self.qadr
        self.x, self.y, self.z = float(q[a]), float(q[a + 1]), float(q[a + 2])
        R = d.xmat[self.body]
        self.yaw = math.atan2(R[3], R[0])
        self.tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[8]))))
        self.pitch = math.degrees(math.asin(max(-1.0, min(1.0, R[6]))))
        v = d.qvel
        b = self.vadr
        vx, vy, vz = float(v[b]), float(v[b + 1]), float(v[b + 2])
        self.speed_fwd = vx * R[0] + vy * R[3] + vz * R[6]
        self.speed = math.hypot(vx, vy)

    def scan(self, t: float) -> None:
        """One LiDAR sweep (one mj_multiRay call). The head is gimballed: it yaws with the rover but
        stays level. Hits on target markers within spectrometer range count as discoveries."""
        cfg = self.cfg
        m, d = self.sim.model, self.sim.data
        pnt = d.site_xpos[self.site].copy()
        az = self.yaw + (np.arange(self.n_az) + self.rng.uniform()) * (2 * math.pi / self.n_az)
        el = self.el + self.rng.uniform(-0.6, 0.6, len(self.el)) * (math.pi / 180.0)
        ce, se = np.cos(el)[:, None], np.sin(el)[:, None]
        vec = np.empty((len(self.el), self.n_az, 3))
        vec[:, :, 0] = ce * np.cos(az)[None, :]
        vec[:, :, 1] = ce * np.sin(az)[None, :]
        vec[:, :, 2] = se
        vec = vec.reshape(-1, 3)
        mujoco.mj_multiRay(m, d, pnt, vec.ravel(), LIDAR_GROUPS, 1, -1, self.ray_gid, self.ray_dist, None,
                           self.nray, cfg.lidar_range)
        gid = self.ray_gid
        dist = self.ray_dist
        hit = (gid >= 0) & (dist >= 0) & (dist <= cfg.lidar_range)
        tmap = self.sim.world.target_geom_ids
        if tmap:
            is_t = np.isin(gid, self.sim.target_gid_array)
            th = np.flatnonzero(hit & is_t & (dist <= cfg.spectro_range))
            for k in th:
                self.fleet.on_target_seen(t, self, tmap[int(gid[k])], float(dist[k]))
            hit &= ~is_t
        idx = np.flatnonzero(hit)
        if len(idx):
            pts = pnt[None, :] + vec[idx] * dist[idx, None]
            self.fleet.map.integrate(pts)

    def bumper(self):
        """Three short rays from the chassis nose at wheel height: rocks right ahead?"""
        m, d = self.sim.model, self.sim.data
        R = d.xmat[self.body].reshape(3, 3)
        origin = d.xpos[self.body] + R @ np.array([0.25, 0.0, -0.02])
        best = None
        for k, a in enumerate((-0.4, 0.0, 0.4)):
            v = R @ np.array([math.cos(a), math.sin(a), 0.0])
            gid = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(m, d, origin, v, BUMPER_GROUPS, 1, -1, gid)
            if 0 <= dist < 0.8 and m.geom_group[gid[0]] == GROUP_ROCK:
                p = origin + v * dist
                if best is None or dist < best[0]:
                    best = (dist, float(p[0]), float(p[1]))
        return best

    # ================================================================== energy
    def energy_for_distance(self, dist: float) -> float:
        """Battery fraction to drive dist metres: learned motor cost per metre plus 25%, and the
        electronics drain at a cautious average speed (turns and slopes slow us down)."""
        cfg = self.cfg
        return dist * (1.25 * self.e_per_m + cfg.battery_base_drain / (0.55 * cfg.v_max))

    def _battery(self, t: float, dt: float, energy_j: float) -> None:
        cfg = self.cfg
        parked = self.state == "idle" and math.hypot(self.x - self.dock[0], self.y - self.dock[1]) < 1.6
        if self.state == "charging" or parked:
            self.battery = min(1.0, self.battery + dt / cfg.charge_time_s)
            return
        motor = cfg.battery_motor_k * energy_j
        self.battery = max(0.0, self.battery - cfg.battery_base_drain * dt - motor)
        moved = math.hypot(self.x - self.last_xy[0], self.y - self.last_xy[1])
        self.e_acc_drain += motor
        self.e_acc_dist += moved
        if self.e_acc_dist > 6.0:
            rate = self.e_acc_drain / self.e_acc_dist
            self.e_per_m = max(cfg.battery_motor_k * 12.0, 0.8 * self.e_per_m + 0.2 * rate)
            self.e_acc_drain = self.e_acc_dist = 0.0

    def home_distance(self, t: float) -> float:
        moved = math.hypot(self.x - self.home_dist_xy[0], self.y - self.home_dist_xy[1])
        if self.home_dist is None or t - self.home_dist_t > 12.0 or moved > 8.0:
            path = self.fleet.plan(t, self, self.dock, max_exp=9000)
            if path is None:
                path = self.fleet.plan(t, self, self.dock, relax=4.0, max_exp=9000)
            d = path_length([(self.x, self.y)] + path) if path else 1.5 * math.hypot(self.x - self.dock[0], self.y - self.dock[1])
            self.home_dist = d
            self.home_dist_t = t
            self.home_dist_xy = (self.x, self.y)
            return d
        return self.home_dist + moved

    # ================================================================== main tick
    def tick(self, t: float, dt: float, energy_j: float) -> None:
        self.read_pose()
        self._battery(t, dt, energy_j)
        self.odometer += math.hypot(self.x - self.last_xy[0], self.y - self.last_xy[1])
        self.last_xy = (self.x, self.y)
        self.scan(t)
        if self.state == "idle":
            self._set_cmd(0.0, 0.0)
            if self.battery > 0.3 and not self.fleet.complete and t >= self.next_goal_try:
                self.state = "exploring"
            self._apply()
            return
        if self._check_flip(t):
            self._apply()
            return
        if self.battery <= 0.0:
            self.battery = 0.0
            self.state = "idle"
            self._drop_target(t)
            self.goal, self.path = None, []
            self.fleet.log(t, "warn", f"{self.id} battery depleted. Stopped in the field.", self.id)
            self._set_cmd(0.0, 0.0)
            self._apply()
            return
        if self.state == "charging":
            self._charging(t)
            self._apply()
            return
        if self.state == "recovering":
            self._recovering(t)
            self._apply()
            return

        if self.state in ("exploring", "investigating"):
            if self.fleet.complete:
                self._go_home(t, "complete")
            elif self._need_return(t):
                self._go_home(t, "battery")

        self._track(t)
        if self._detect_stall(t):
            self._apply()
            return

        if self.state == "exploring":
            self._explore(t)
        elif self.state == "investigating":
            if self._investigate(t):
                self._apply()
                return
        elif self.state == "returning":
            if self._return(t):
                self._apply()
                return
        self._drive(t)
        self._apply()

    # ================================================================== state logic
    def _need_return(self, t: float) -> bool:
        if t >= self.next_home_check:
            self.next_home_check = t + 3.0
            self.need_home = self.energy_for_distance(self.home_distance(t))
        return self.battery < self.need_home + self.cfg.battery_margin

    def _go_home(self, t: float, reason: str) -> None:
        self._drop_target(t, "battery low" if reason in ("battery", "range") else None)
        self.state = "returning"
        self.return_reason = reason
        self.dock_near_since = None
        self.goal = self.dock
        self.path = []
        self.plan_t = -1e9
        self.dwell_until = None
        self.next_goal_try = t + 4.0
        if reason == "battery":
            self.fleet.log(t, "info", f"{self.id} battery at {self.battery * 100:.0f}%, needs about "
                                      f"{self.need_home * 100:.0f}% to get home plus a "
                                      f"{self.cfg.battery_margin * 100:.0f}% margin. Returning to the lander.",
                           self.id)
        elif reason == "range":
            self.fleet.log(t, "info", f"{self.id} battery at {self.battery * 100:.0f}% is not enough to reach the "
                                      f"remaining frontiers and get back. Returning to charge.", self.id)
        elif reason == "regroup":
            self.fleet.log(t, "info", f"{self.id} can't reach any frontier from here. "
                                      f"Heading toward the lander to find a way round.", self.id)

    def _drop_target(self, t: float, reason: str | None = None) -> None:
        if self.target_idx is not None:
            self.fleet.release_target(t, self, self.target_idx, reason)
            self.target_idx = None
            self.dwell_until = None

    def _charging(self, t: float) -> None:
        self._set_cmd(0.0, 0.0)
        if self.battery >= 0.995:
            self.battery = 1.0
            if self.fleet.complete:
                self.state = "idle"
                self.fleet.log(t, "info", f"{self.id} fully charged and parked at the lander.", self.id)
            else:
                self.state = "exploring"
                self.goal, self.path = None, []
                self.next_goal_try = t
                self.fleet.log(t, "info", f"{self.id} fully charged. Heading back out.", self.id)

    def _return(self, t: float) -> bool:
        dd = math.hypot(self.x - self.dock[0], self.y - self.dock[1])
        if dd < 2.5:
            self.dock_near_since = t if self.dock_near_since is None else self.dock_near_since
        else:
            self.dock_near_since = None
        # park on our own spot (so we never block another rover's), unless we have been
        # trying to get onto it for a while
        if dd < 0.7 or (dd < 1.6 and self.dock_near_since is not None and t - self.dock_near_since > 25.0):
            self._set_cmd(0.0, 0.0)
            self.path = []
            self.goal = None
            if self.fleet.complete and self.battery > 0.9:
                self.state = "idle"
                self.fleet.log(t, "info", f"{self.id} parked at the lander.", self.id)
            else:
                self.state = "charging"
                self.fleet.log(t, "info", f"{self.id} docked at {self.battery * 100:.0f}%. Charging.", self.id)
            return True
        if self.return_reason == "regroup" and t >= self.next_goal_try:
            # on the way back, keep checking whether a frontier has become reachable
            self.next_goal_try = t + 4.0
            res = self.fleet.choose_goal(t, self)
            if res is not None:
                goal, path, size = res
                self.state = "exploring"
                self.return_reason = None
                self.no_goal_since = None
                self.goal, self.goal_size, self.goal_t = goal, size, t
                self.next_goal_eval = t + 15.0
                self._set_path(t, path)
                self.fleet.log(t, "info", f"{self.id} found a way through. Back to exploring.", self.id)
                return False
        self.goal = self.dock
        if self._need_replan(t):
            self._set_path(t, self._plan_home(t))
        return False

    def _plan_home(self, t: float, max_exp: int = 14000) -> list:
        """Path to the dock. If the learned limit boxes us in, allow slightly steeper ground."""
        for relax in (0.0, 2.0, 4.0):
            path = self.fleet.plan(t, self, self.dock, relax=relax, max_exp=max_exp)
            if path is not None:
                return path
        return [(self.x, self.y), self.dock]

    def _explore(self, t: float) -> None:
        f = self.fleet
        # an already found target that nobody is sampling?
        if t >= self.next_target_check:
            self.next_target_check = t + 3.0
            k, d = f.unclaimed_target(t, self)
            if k is not None and d < 30.0 and self.can_investigate(t, f.targets[k]):
                ts = f.targets[k]
                ts.status = "investigating"
                ts.investigator = self.id
                self.start_investigation(t, k)
                f.log(t, "info", f"{self.id} heading to sample {ts.spec.id} ({KIND_LABEL[ts.spec.kind]}, "
                                 f"found by {ts.found_by}), {d:.0f} m away.", self.id)
                return
        need = self.goal is None
        if not need:
            dgoal = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
            if dgoal < 1.5 or not self._goal_still_frontier() or t >= self.next_goal_eval:
                need = True
        if need and t >= self.next_goal_try:
            res = f.choose_goal(t, self)
            if res is None:
                self.goal, self.path = None, []
                self.next_goal_try = t + 2.0
                if self.goal_fail_reason == "battery" and self.battery < 0.9:
                    # every frontier is too far for the charge left: top up first
                    self._go_home(t, "range")
                    return
                if self.no_goal_since is None:
                    self.no_goal_since = t
                elif t - self.no_goal_since > 6.0 and not f.complete and f.usable_frontiers(t) \
                        and not f.in_reach(self.x, self.y):
                    # the fleet can still reach frontiers, but not from here: boxed in by steep ground
                    self.no_goal_since = None
                    self._go_home(t, "regroup")
                return
            self.no_goal_since = None
            goal, path, size = res
            old = self.goal
            self.goal, self.goal_size, self.goal_t = goal, size, t
            self.next_goal_eval = t + 15.0
            self._set_path(t, path)
            big_change = old is None or math.hypot(goal[0] - old[0], goal[1] - old[1]) > 15.0
            if t - self.last_goal_event_t > 75.0 and big_change:
                self.last_goal_event_t = t
                dx, dy = goal[0] - self.x, goal[1] - self.y
                from .fleet import compass
                extra = " near a sampled target" if f.bonus_area(goal[0], goal[1]) else ""
                f.log(t, "info", f"{self.id} heading to a frontier {math.hypot(dx, dy):.0f} m {compass(dx, dy)}"
                                 f"{extra} ({size} cells).", self.id)
            return
        if self.goal is not None and self._need_replan(t):
            path = f.plan(t, self, self.goal)
            if path is None:
                self.blacklist[f._bucket(*self.goal)] = t + 40.0
                self.goal, self.path = None, []
                self.next_goal_try = t
            else:
                self._set_path(t, path)

    def _goal_still_frontier(self) -> bool:
        gx, gy = self.goal
        return any(abs(fx - gx) < 3.0 and abs(fy - gy) < 3.0 for (fx, fy, _s, _i) in self.fleet.frontiers)

    def can_investigate(self, t: float, ts) -> bool:
        if self.state != "exploring" or self.target_idx is not None:
            return False
        d = math.hypot(ts.spec.x - self.x, ts.spec.y - self.y)
        need = self.energy_for_distance(1.3 * d + 1.3 * math.hypot(ts.spec.x, ts.spec.y)) + self.cfg.battery_margin + 0.05
        return self.battery > need

    def start_investigation(self, t: float, tidx: int) -> None:
        ts = self.fleet.targets[tidx]
        self.state = "investigating"
        self.target_idx = tidx
        self.goal = (ts.spec.x, ts.spec.y)
        self.path = []
        self.plan_t = -1e9
        self.dwell_until = None
        self.inv_fails = 0

    def _investigate(self, t: float) -> bool:
        f = self.fleet
        if self.target_idx is None:
            self.state = "exploring"
            return False
        ts = f.targets[self.target_idx]
        if self.dwell_until is not None:
            self._set_cmd(0.0, 0.0)
            if t >= self.dwell_until:
                f.target_sampled(t, self, self.target_idx)
                self.target_idx = None
                self.dwell_until = None
                self.state = "exploring"
                self.goal, self.path = None, []
                self.next_goal_try = t
            return True
        d = math.hypot(ts.spec.x - self.x, ts.spec.y - self.y)
        if d <= self.cfg.investigate_radius:
            self.dwell_until = t + self.cfg.investigate_dwell_s
            self._set_cmd(0.0, 0.0)
            self.path = []
            return True
        self.goal = (ts.spec.x, ts.spec.y)
        if self._need_replan(t):
            path = f.plan(t, self, self.goal)
            if path is None:
                self.inv_fails += 1
                if self.inv_fails >= 3:
                    self._drop_target(t, "no safe path found")
                    self.state = "exploring"
                    self.goal, self.path = None, []
                    return False
                self.plan_t = t
            else:
                self._set_path(t, path)
        return False

    # ================================================================== recovery, stalls, flips
    def _track(self, t: float) -> None:
        intended_fwd = self.cmd_v > 0.2 and self.path
        self.hist.append((t, self.x, self.y, self.yaw, self.pitch, bool(intended_fwd), self.cmd_w, self.tilt))
        self.climb_hist.append((self.x, self.y, self.pitch, self.speed_fwd))
        if len(self.climb_hist) == self.climb_hist.maxlen:
            x0, y0 = self.climb_hist[0][0], self.climb_hist[0][1]
            prog = math.hypot(self.x - x0, self.y - y0)
            minp = min(c[2] for c in self.climb_hist)
            if prog > 0.9 and minp > 6.0 and all(c[3] > 0.15 for c in self.climb_hist):
                self.fleet.record_climb(t, self, minp)

    def _detect_stall(self, t: float) -> bool:
        if t < self.stall_cooldown:
            return False
        h = self.hist
        if len(h) == h.maxlen and all(e[5] for e in h):
            t0, x0, y0, yaw0 = h[0][0], h[0][1], h[0][2], h[0][3]
            prog = (self.x - x0) * math.cos(yaw0) + (self.y - y0) * math.sin(yaw0)
            if prog < self.cfg.stall_progress_m:
                self._on_stall(t)
                return True
        # generic stuck check: no real movement (or turning) for 9 s while trying to drive somewhere
        if self.path and self.state in ("exploring", "investigating", "returning"):
            if self.stuck_ref is None or math.hypot(self.x - self.stuck_ref[1], self.y - self.stuck_ref[2]) > 0.4 \
                    or abs(wrap(self.yaw - self.stuck_ref[3])) > 0.6:
                self.stuck_ref = (t, self.x, self.y, self.yaw)
            elif t - self.stuck_ref[0] > 9.0:
                self.stuck_ref = None
                self._start_recovery(t, None)
                return True
        else:
            self.stuck_ref = None
        return False

    def _on_stall(self, t: float) -> None:
        """No forward progress for a few seconds while driving forward. Work out why."""
        f = self.fleet
        h = list(self.hist)
        # traffic: another rover right in front
        for o in f.rovers:
            if o is not self and math.hypot(o.x - self.x, o.y - self.y) < 1.9:
                if abs(wrap(math.atan2(o.y - self.y, o.x - self.x) - self.yaw)) < 1.0:
                    self._start_recovery(t, None)
                    return
        bump = self.bumper()
        self.recent_stalls.append((t, self.x, self.y))
        if bump is not None:
            _, bx, by = bump
            f.map.mark_blocked(bx, by, 0.6)
            f.log(t, "info", f"{self.id} blocked by a rock. Marking it as an obstacle and rerouting.", self.id)
            self._start_recovery(t, (bx, by))
            return
        pitches = sorted(e[4] for e in h)
        p_med, p_max = pitches[len(pitches) // 2], pitches[-1]
        turning = sum(abs(e[6]) for e in h) / max(len(h), 1) > 0.45
        ax = self.x + 0.7 * math.cos(self.yaw)
        ay = self.y + 0.7 * math.sin(self.yaw)
        # only blame the slope if gravity alone would eat most of the motors' torque at this pitch
        gravity_share = self.grav_torque * math.sin(math.radians(max(p_med, 0.0))) / (4.0 * self.cfg.motor_torque)
        if p_med >= 12.0 and p_med >= 0.8 * p_max and not turning and gravity_share >= 0.8:
            # nose up on a steady slope with full power and no progress: the slope beat the motors.
            # The tilt (pitch and roll together) is the slope the planner's map measures.
            tilts = sorted(e[7] for e in h)
            if not f.report_slope_stall(t, self, max(p_med, tilts[len(tilts) // 2])):
                f.map.mark_blocked(ax, ay, 0.5)    # steeper than the map said: avoid this exact spot
            self._start_recovery(t, None)
            return
        if p_med >= 12.0 or (turning and p_med >= 8.0):
            # wheels lost grip while turning or on uneven ground: not evidence about the slope limit
            f.log(t, "info", f"{self.id} lost grip on a {p_med:.0f} deg slope. Backing off to try another line.",
                  self.id)
            self._start_recovery(t, None)
            return
        # unexplained stall on flat ground: treat the spot ahead as an obstacle
        f.map.mark_blocked(ax, ay, 0.5)
        f.log(t, "info", f"{self.id} stuck on rough ground. Marking the spot as untraversable.", self.id)
        self._start_recovery(t, (ax, ay))

    def _start_recovery(self, t: float, obstacle) -> None:
        if self.state != "recovering":
            self.prev_state = self.state
        self.state = "recovering"
        self.rec_phase = "reverse"
        self.rec_until = t + 1.5
        if obstacle is not None:
            side = wrap(math.atan2(obstacle[1] - self.y, obstacle[0] - self.x) - self.yaw)
            self.rec_dir = -1.0 if side > 0 else 1.0
        else:
            self.rec_dir = 1.0 if self.rng.uniform() < 0.5 else -1.0
        self.path = []
        self.hist.clear()
        # repeated stalls around the same spot: give up on this goal for a while
        near = [s for s in self.recent_stalls if t - s[0] < 45.0 and math.hypot(s[1] - self.x, s[2] - self.y) < 3.0]
        if len(near) >= 3 and self.goal is not None and self.prev_state == "exploring":
            self.blacklist[self.fleet._bucket(*self.goal)] = t + 60.0
            self.goal = None
        if self.prev_state == "exploring":
            self.goal = None

    def _recovering(self, t: float) -> None:
        if self.rec_phase == "reverse":
            self._set_cmd(-0.4, 0.0)
            if t >= self.rec_until:
                self.rec_phase = "turn"
                self.rec_until = t + 1.4
        elif self.rec_phase == "turn":
            self._set_cmd(0.0, self.rec_dir * self.cfg.w_max)
            if t >= self.rec_until:
                self.rec_phase = None
                self.state = self.prev_state if self.prev_state != "recovering" else "exploring"
                self.stall_cooldown = t + 2.0
                self.plan_t = -1e9
                self.next_goal_try = t
                self.hist.clear()
        else:
            self.state = "exploring"

    def _check_flip(self, t: float) -> bool:
        if self.tilt > self.cfg.flip_tilt_deg:
            if self.flip_since is None:
                self.flip_since = t
            elif t - self.flip_since > self.cfg.flip_time_s:
                self.flip_since = None
                self.sim.reset_rover_upright(self)
                self.read_pose()
                self.fleet.log(t, "warn", f"{self.id} tipped over. Self-righting and replanning.", self.id)
                self.path = []
                self.plan_t = -1e9
                self.hist.clear()
                if self.state == "recovering":
                    self.state = self.prev_state
                self.stall_cooldown = t + 2.0
                return True
        else:
            self.flip_since = None
        return False

    # ================================================================== driving
    def _need_replan(self, t: float) -> bool:
        if not self.path or self.wp >= len(self.path):
            return True
        if self.plan_limit_version != self.fleet.limit_version:
            return True
        if t - self.plan_t > 2.5:
            return True
        if self.fleet.path_blocked(self.path, max(self.wp - 1, 0), horizon=8):
            return True
        return False

    def _set_path(self, t: float, path: list) -> None:
        self.path = list(path)
        self.wp = 1 if len(self.path) > 1 else 0
        self.plan_t = t
        self.plan_limit_version = self.fleet.limit_version

    def _carrot(self):
        pts = self.path
        n = len(pts)
        if n == 0:
            return None
        L = self.cfg.lookahead
        while self.wp < n - 1 and math.hypot(pts[self.wp][0] - self.x, pts[self.wp][1] - self.y) < L:
            self.wp += 1
        if self.wp == 0:
            return pts[0]
        ax, ay = pts[self.wp - 1]
        bx, by = pts[self.wp]
        dx, dy = bx - ax, by - ay
        fx, fy = ax - self.x, ay - self.y
        A = dx * dx + dy * dy
        B = 2 * (fx * dx + fy * dy)
        C = fx * fx + fy * fy - L * L
        disc = B * B - 4 * A * C
        if A > 1e-9 and disc >= 0:
            t2 = (-B + math.sqrt(disc)) / (2 * A)
            if 0.0 <= t2 <= 1.0:
                return (ax + t2 * dx, ay + t2 * dy)
        return (bx, by)

    def _drive(self, t: float) -> None:
        cfg = self.cfg
        c = self._carrot()
        if c is None or self.goal is None:
            self._set_cmd(0.0, 0.0)
            return
        e = wrap(math.atan2(c[1] - self.y, c[0] - self.x) - self.yaw)
        # Skid steer on a slope: a tight arc leaves the inner wheels with almost no speed target, so
        # only the outer pair pushes and the rover bogs down. On sloped ground make big corrections in
        # place and keep arcs wide enough that all four wheels drive forward.
        sloped = self.tilt > 7.0
        if abs(e) > (0.6 if sloped else 0.85):
            v = 0.0
            w = math.copysign(cfg.w_max, e)
        else:
            v = cfg.v_max * max(0.0, math.cos(e)) ** 2
            w = max(-cfg.w_max, min(cfg.w_max, 2.2 * e))
            if sloped:
                w = max(-0.9 * v, min(0.9 * v, w))
        dgoal = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        if self.wp >= len(self.path) - 1:
            v = min(v, 0.2 + 0.5 * dgoal)
        if abs(self.pitch) > 10.0:
            v *= 0.8
        intended_v = v
        # give way: lower id has priority, parked rovers are obstacles
        blocked_by = None
        for o in self.fleet.rovers:
            if o is self:
                continue
            dd = math.hypot(o.x - self.x, o.y - self.y)
            if dd < 2.2 and abs(wrap(math.atan2(o.y - self.y, o.x - self.x) - self.yaw)) < 0.8:
                moving = o.speed >= 0.05 and o.state not in ("idle", "charging")
                # moving rovers: lower id has right of way. Parked ones: the planner routes round
                # them, so only stop if we are about to touch.
                if dd < 1.3 or (moving and o.idx < self.idx) or (not moving and dd < 1.5):
                    blocked_by = o
                    break
        if blocked_by is not None and v > 0:
            v = 0.0
            if self.yield_since is None:
                self.yield_since = t
            elif t - self.yield_since > 3.0:
                self.yield_since = None
                self.plan_t = -1e9  # replan around (others are soft obstacles in the planner)
        else:
            self.yield_since = None
        # hill hold: if we are rolling backwards while trying to climb, brake instead of spinning
        if v > 0.05 and self.speed_fwd < -0.06:
            self.hold_until = t + 0.6
        if t < self.hold_until:
            v, w = 0.0, 0.0
        self._set_cmd(v, w)
        self.cmd_v = intended_v if blocked_by is None else 0.0

    def _set_cmd(self, v: float, w: float) -> None:
        self.cmd_v = v
        self.cmd_w = w

    def _apply(self) -> None:
        cfg = self.cfg
        m, d = self.sim.model, self.sim.data
        r = cfg.wheel_radius
        wl = (self.cmd_v - self.cmd_w * cfg.track_eff) / r
        wr = (self.cmd_v + self.cmd_w * cfg.track_eff) / r
        T = cfg.motor_torque
        Tb = 3.0 * T
        for k, a in enumerate(self.acts):
            wt = wl if k < 2 else wr
            d.ctrl[a] = wt
            # motors drive with up to T, brakes and back driving resist with up to 3T
            if wt > 0.05:
                m.actuator_forcerange[a] = (-Tb, T)
            elif wt < -0.05:
                m.actuator_forcerange[a] = (-T, Tb)
            else:
                m.actuator_forcerange[a] = (-Tb, Tb)

    # ================================================================== protocol
    def to_msg(self) -> dict:
        pts = []
        if self.path and self.state not in ("charging", "idle"):
            rest = self.path[max(self.wp, 0):]
            pts = downsample([(self.x, self.y)] + rest, 30)
        return {
            "id": self.id,
            "x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2),
            "yaw": round(self.yaw, 3),
            "speed": round(self.speed, 2),
            "battery": round(self.battery, 3),
            "state": self.state,
            "goal": None if self.goal is None else [round(self.goal[0], 2), round(self.goal[1], 2)],
            "path": [[round(p[0], 2), round(p[1], 2)] for p in pts],
            "target_id": None if self.target_idx is None else self.fleet.targets[self.target_idx].spec.id,
        }
