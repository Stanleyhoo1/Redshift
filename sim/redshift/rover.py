"""Per-rover embodied agent: sensors, skid-steer controller, battery, state machine.

Sensors (all the rover knows about the world):
  LiDAR          spinning, on a mast; hits go into the shared map (mj_multiRay)
  spectrometer   LiDAR hits on science targets within range
  IMU            MuJoCo framequat + gyro on the chassis: pitch, roll, tilt, heading, yaw rate
  wheel encoders wheel joint speeds -> wheel surface speed
  motor feedback actuator torques -> motor load; controller status (a fault shows up here)
  wheel load     normal force on each wheel (strain gauges; read only while slip is suspected)
  bumper         three short rays at the nose (mj_ray)
  position       ground truth, treated as GPS-like localisation (no SLAM)

  water sensor   a wetness probe on the chassis belly (only matters if the rover ever drives into liquid)
  camera         mast camera (camera.py), used by the rules for a quick colour check of a LiDAR hole

States (protocol enum): exploring, investigating, returning, charging, recovering, idle, fault, observing.

LiDAR returns are split three ways: liquid gives no return at all (water shows up as a hole in the
map); hits on creatures go to the fleet's motion tracker (tracker.py; a stated simplification, a real
rover would separate them by scan-to-map differencing); everything else builds the map. A hole where
the ground should have answered (several steep rays straight ahead with nothing back) is an anomaly
("no_returns"): the rover slows down and takes a camera picture. Blue liquid in the lower half of
the picture means a lake: the fleet marks it impassable and replans. Otherwise it is a drop-off and
the rover keeps to mapped ground.

Each decision cycle is timed (the "brain": map integration, frontier logic, planning, state machine;
not the simulated sensor hardware or physics) and planning stops when the cycle's share of the
compute budget runs out, then resumes next cycle.

Anything perception did not expect (new obstacles on the path, wheel slip, LiDAR returns collapsing)
goes through Rover.on_anomaly(): it records the anomaly, slows the rover down at once (before any
re-planning) and logs what was seen. Other layers can subscribe with Rover.anomaly_hooks.

Tier 2 (advisor.py: Gemma advice and operator commands) can only nudge the rover through a few
fields: held (operator hold: parked in state idle), op_goal (operator go_to), dir_bias (operator
"explore north"), sticky_goal (a frontier Gemma picked), pause_until and crawl_until / crawl_v
(wait or creep while a question is open). The safety rules below still run first every cycle.
"""
from __future__ import annotations

import math
import time
from collections import deque

import mujoco
import numpy as np

from .config import Config
from .planner import PENDING, downsample, path_length
from .terrain import KIND_LABEL
from .world import GROUP_ROCK

LIDAR_GROUPS = np.array([1, 1, 1, 0, 0, 1], np.uint8)   # terrain, rocks, targets and creatures, liquid; never
                                                         # rovers (3) or the walls (4)
BUMPER_GROUPS = np.array([0, 1, 0, 0, 0, 0], np.uint8)  # rocks only
DRIVING = ("exploring", "investigating", "returning", "observing")
PARKED = ("idle", "charging", "fault")
ANOMALY_KINDS = ("new_obstacle", "slip", "low_visibility", "no_returns")
ANOMALY_TYPE = {"new_obstacle": "new_obstacle", "slip": "wheel_slip", "low_visibility": "low_visibility",
                "no_returns": "no_returns"}
CAUTION_S = {"new_obstacle": 3.0, "slip": 1.0, "low_visibility": 3.0, "no_returns": 4.0}   # cautious slowdown
HOLE_MIN_RAYS = 7          # missing steep returns straight ahead before a hole counts
HOLE_FRONT = math.radians(35.0)   # only straight ahead (inside the camera's view)
LIQUID_MIN_PCT = 3.0       # blue liquid in at least this share of the lower camera image
CREATURE_STOP_M = 3.5      # any rover stops if a tracked creature is this close ahead
VERB = {"drill": "drilled a core from", "scoop": "scooped ice from", "sniffer": "sniffed the gas at",
        "swab": "swabbed the trace at"}
BRAKE_KV = 400.0       # wheel brake stiffness while the motor controller is down (normal servo: motor_kv)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _camera_frame(sim, rid: str):
    """A small mast camera picture (camera hardware: rendered in this thread with its own GL context)."""
    from . import camera
    return camera._local_renderer().frame(None, sim, rid, "fpv", 160, 120, depth=False, hide_groups=(3, 4),
                                          smooth=False)


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
        self.imu_q = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"{rid}_imu_quat")])
        self.imu_g = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"{rid}_imu_gyro")])
        # own spec sheet: wheel torque needed per unit sin(slope) to hold the whole rover against gravity
        self.mass = float(m.body_subtreemass[self.body])
        self.grav_torque = self.mass * self.cfg.gravity * self.cfg.wheel_radius
        self.acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{rid}_m{k}") for k in range(4)]
        self.wdofs = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{rid}_w{k}")])
                      for k in range(4)]
        self.wheel_gidx = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{rid}_wheel{k}"): k for k in range(4)}
        self._f6 = np.zeros(6)
        self.rng = np.random.default_rng(self.cfg.seed * 101 + idx)
        self.imu_rng = np.random.default_rng(self.cfg.seed * 131 + idx)

        # lidar buffers
        self.el = np.array(self.cfg.lidar_el)
        self.n_az = self.cfg.lidar_az
        self.nray = self.n_az * len(self.el)
        self.ray_gid = np.zeros(self.nray, np.int32)
        self.ray_dist = np.zeros(self.nray, np.float64)

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
        self.plan_relax = 0.0
        self._last_relax = 0.0
        self.next_goal_try = 0.0
        self.next_goal_eval = 0.0
        self.last_goal_event_t = -1e9
        self.target_idx = None
        self.dwell_until = None
        self.inv_fails = 0
        self.next_inv_plan = 0.0
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
        self.e_per_m = self.cfg.battery_motor_k * 45.0 / self.cfg.battery_capacity   # battery per metre (motor), learned
        self.e_acc_drain = 0.0
        self.e_acc_dist = 0.0
        self.home_dist = None
        self.home_dist_t = -1e9
        self.home_dist_xy = (0.0, 0.0)
        self.home_pending = False
        self.need_home = 0.0
        self.next_home_check = 0.0
        self.no_goal_since = None
        self.return_reason = None
        self.dock_near_since = None
        self.goal_fail_reason = None
        self.blacklist: dict = {}          # frontier bucket -> time until which this rover skips it

        # planning (anytime) and re-routing
        self.searches: dict = {}           # unfinished A* searches, resumed next cycle
        self.plan_fails: dict = {}         # (goal cell, relax) -> (t, x, y) of a recent failed search
        self.pending_goal_key = None
        self.block_info = None             # (x, y, metres ahead, reason) if the path is blocked
        self.hold_for_plan = False
        self.reroute = None                # pending "new obstacle" announcement
        self.last_rocks_t = -1e9
        self.rocks_seen: list = []         # (x, y, t) obstacles already announced
        self.soft_seen: set = set()

        # sensors
        self.x = self.y = self.z = self.yaw = 0.0
        self.tilt = self.pitch = self.roll = 0.0
        self.gyro_z = 0.0
        self.speed = 0.0
        self.speed_fwd = 0.0
        self.v_wheel = 0.0
        self.slip = 0.0
        self.spin_load = 0.0
        self.motor_load = 0.0
        self.lidar_hits = 0
        self.lidar_eff = self.cfg.lidar_range
        self.far_hist: deque = deque(maxlen=3)
        self.degraded = False
        self.spectro = None                # (target index, distance) nearest target seen this sweep
        self.scan_pts = np.zeros((0, 3))
        self.scan_changes = 0

        # hazards: hardware state set by the world, and the rover's reactions
        self.motor_fault = False           # motor controller offline (seen through motor feedback)
        self._brakes_locked = False
        self.fault_prev = "exploring"
        self.trail: deque = deque(maxlen=80)
        self.slip_since = None
        self.slip_xy = None
        self.slip_peak = 0.0
        self.slip_n = 0
        self.slip_spin = np.zeros(4)
        self.slip_load = np.zeros(4)
        self.sand_zone = None
        self.sand_t0 = 0.0
        self.sand_freed = False
        self.escape: list = []

        # uncertainty: the last thing perception did not expect, and the cautious slowdown it caused
        self.anomaly = None                # {"t", "kind", "at", ...details}
        self.anomaly_log: deque = deque(maxlen=20)
        self.anomaly_hooks: list = []      # callables (rover, anomaly dict), e.g. a reasoning layer
        self.caution_until = -1e9
        self.caution_v = self.cfg.caution_v

        # tier 2 nudges (advisor.py); the rules keep the final say
        self.advisor = None                # set by the sim; rover_tick() runs after each decision cycle
        self.held = False                  # operator "hold": parked in state idle until "resume"
        self.op_goal = None                # operator go_to point (x, y)
        self.op_fails = 0
        self.dir_bias = None               # operator "explore <dir>": (unit x, unit y, until t)
        self.sticky_goal = None            # frontier picked by Gemma: kept until reached or gone
        self.goal_options: list = []       # ranked frontier options from the last choose_goal
        self.pause_until = -1e9            # stand still (wait) until then
        self.pause_dust = False            # the wait ends early when visibility comes back
        self.crawl_until = -1e9            # creep at crawl_v until then (a question is open)
        self.crawl_v = 0.25

        # water, lifeforms and samples (protocol v4 / v4b)
        self.hole_prev = None              # (x, y) of a hole candidate in the previous sweep
        self.hole_last_t = -1e9
        self.ext_last_t = -1e9
        self.water_entry = None            # (x, y) where the water sensor got wet
        self.obs = None                    # observation of a lifeform track (state "observing")
        self.last_backoff_t = -1e9
        self.samples: list = []            # target indices in the sample tubes
        self.capacity = int(self.cfg.sample_capacity)
        self.sampling = None               # {"tidx", "instrument", "t0", "dur"} while an instrument runs
        self.cache_note_t = -1e9

        # charging telemetry (the lander's panels can be dusty)
        self.battery_margin = self.cfg.battery_margin
        self.charge_hist: deque = deque(maxlen=12)
        self.slow_charge = False

        # compute budget
        self.budget_s = self.cfg.brain_budget_ms / 1000.0
        self._c0 = None
        self._hw = 0.0
        self.brain_last = 0.0
        self.brain_hist: deque = deque(maxlen=int(10 * self.cfg.agent_hz) + 5)
        self.brain_all: list = []
        self.share_until = -1e9
        self.last_budget_warn = -1e9
        self.t_now = 0.0
        self.exp_used = 0

        self.read_pose()
        self.last_xy = (self.x, self.y)

    # ================================================================== sensing
    def read_pose(self) -> None:
        """Position from GPS-like localisation (ground truth); attitude and heading from the IMU."""
        d = self.sim.data
        q = d.qpos
        a = self.qadr
        self.x, self.y, self.z = float(q[a]), float(q[a + 1]), float(q[a + 2])
        sd = d.sensordata
        w, qx, qy, qz = (float(v) for v in sd[self.imu_q:self.imu_q + 4])
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r10 = 2.0 * (qx * qy + w * qz)
        r20 = 2.0 * (qx * qz - w * qy)
        r21 = 2.0 * (qy * qz + w * qx)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        self.yaw = math.atan2(r10, r00)
        noise = self.imu_rng.normal(0.0, 0.1, 2)
        self.pitch = math.degrees(math.asin(max(-1.0, min(1.0, r20)))) + float(noise[0])
        self.roll = math.degrees(math.atan2(r21, r22)) + float(noise[1])
        cr = math.cos(math.radians(self.pitch)) * math.cos(math.radians(self.roll))
        if r22 < 0.0:
            cr = -abs(cr)                  # upside down: keep the IMU's sign
        self.tilt = math.degrees(math.acos(max(-1.0, min(1.0, cr))))
        self.gyro_z = float(sd[self.imu_g + 2])
        v = d.qvel
        b = self.vadr
        vx, vy, vz = float(v[b]), float(v[b + 1]), float(v[b + 2])
        self.speed_fwd = vx * r00 + vy * r10 + vz * r20
        self.speed = math.hypot(vx, vy)

    def _read_drive_sensors(self, t: float) -> None:
        """Wheel encoders, motor feedback and the slip they imply."""
        d = self.sim.data
        cfg = self.cfg
        w = d.qvel[self.wdofs]
        tau = np.abs(d.actuator_force[self.acts])
        self.v_wheel = float(w.mean()) * cfg.wheel_radius
        self.motor_load = min(1.0, float(tau.mean()) / cfg.motor_torque)
        # wheels turning faster than the rover moves, and how hard their motors work (smoothed): digging
        # into sand takes torque, a wheel spinning in the air takes almost none
        spin = w * cfg.wheel_radius > self.speed_fwd + 0.2
        n_spin = int(spin.sum())
        self.spin_load = 0.5 * self.spin_load + 0.5 * (float(tau[spin].mean()) / cfg.motor_torque if n_spin else 0.0)
        if abs(self.v_wheel) > cfg.slip_min_wheel_speed:
            raw = max(0.0, min(1.0, 1.0 - self.speed_fwd / self.v_wheel))
        else:
            raw = 0.0
        self.slip = 0.5 * self.slip + 0.5 * raw          # smoothed over about a second
        # soft ground: wheels turning well, body barely moving, on gentle ground, not turning.
        # Near soft ground the fleet already knows about, weaker evidence is enough.
        driving = self.state in DRIVING and self.cmd_v > 0.2 and not self.motor_fault
        turning = abs(self.gyro_z) > 0.8               # body really rotating (IMU gyro)
        thr = cfg.slip_threshold
        if self.fleet.soft_zones and self.fleet.soft_zone_at(self.x, self.y, margin=1.5) is not None:
            thr = 0.75 * thr
        suspect = (driving and not turning and self.tilt < 14.0 and self.v_wheel > cfg.slip_min_wheel_speed
                   and self.slip > thr and n_spin >= 2 and self.spin_load > 0.2)
        if suspect:
            if self.slip_since is None:
                self.slip_since = t
                self.slip_xy = (self.x, self.y)
                self.slip_peak = 0.0
                self.slip_n = 0
                self.slip_spin = np.zeros(4)
                self.slip_load = np.zeros(4)
            self.slip_peak = max(self.slip_peak, self.slip)
            # wheel loads while it lasts (contact forces flicker, so they are averaged over the window)
            self.slip_n += 1
            self.slip_spin += spin
            self.slip_load += np.where(spin, self._hw_call(self.wheel_loads), 0.0)
        else:
            self.slip_since = None

    def _slip_on_ground(self) -> bool:
        """Were the wheels that kept spinning actually on the ground? Sunk in sand they carry their share
        of the weight; a rover hung up on a rock spins wheels that hang in the air."""
        share = 0.25 * self.mass * self.cfg.gravity / 4.0
        busy = self.slip_spin >= 0.5 * self.slip_n
        if not busy.any():
            return False
        mean = self.slip_load[busy] / np.maximum(self.slip_spin[busy], 1)
        return bool((mean > share).all())

    def wheel_loads(self) -> list:
        """Normal force on each wheel (N), from the wheel contacts: simulated strain gauges."""
        m, d = self.sim.model, self.sim.data
        loads = [0.0, 0.0, 0.0, 0.0]
        gidx = self.wheel_gidx
        for k in range(d.ncon):
            c = d.contact[k]
            w = gidx.get(c.geom1, gidx.get(c.geom2))
            if w is not None:
                mujoco.mj_contactForce(m, d, k, self._f6)
                loads[w] += abs(float(self._f6[0]))
        return loads

    def _hw_call(self, fn, *args, **kw):
        """Run simulated sensor hardware (ray casts, camera renders) outside the brain's compute budget."""
        c = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            self._hw += time.perf_counter() - c

    def scan(self, t: float) -> None:
        """One LiDAR sweep (one mj_multiRay call). The head is gimballed: it yaws with the rover but
        stays level. Hits on target markers within spectrometer range count as discoveries."""
        cfg = self.cfg
        m, d = self.sim.model, self.sim.data
        hz = self.sim.hazards
        rng_max = hz.lidar_range() if hz is not None else cfg.lidar_range
        pnt = d.site_xpos[self.site].copy()
        az = self.yaw + (np.arange(self.n_az) + self.rng.uniform()) * (2 * math.pi / self.n_az)
        el = self.el + self.rng.uniform(-0.6, 0.6, len(self.el)) * (math.pi / 180.0)
        ce, se = np.cos(el)[:, None], np.sin(el)[:, None]
        vec = np.empty((len(self.el), self.n_az, 3))
        vec[:, :, 0] = ce * np.cos(az)[None, :]
        vec[:, :, 1] = ce * np.sin(az)[None, :]
        vec[:, :, 2] = se
        vec = vec.reshape(-1, 3)
        self._hw_call(mujoco.mj_multiRay, m, d, pnt, vec.ravel(), LIDAR_GROUPS, 1, -1, self.ray_gid,
                      self.ray_dist, None, self.nray, rng_max)
        gid = self.ray_gid
        dist = self.ray_dist
        hit = (gid >= 0) & (dist >= 0) & (dist <= rng_max)
        wg = self.sim.world.water_gids
        if wg is not None and len(wg):
            hit &= ~np.isin(gid, wg)       # liquid absorbs the beam or glances it away: no return
        drop, sigma = hz.lidar_noise() if hz is not None else (0.0, 0.0)
        if drop > 0.0:                     # dust: lost returns and noisy ranges
            hit &= self.rng.uniform(size=self.nray) >= drop
            dist = dist + self.rng.normal(0.0, sigma, self.nray)
            hit &= (dist > 0.05) & (dist <= rng_max)
        self.lidar_hits = int(hit.sum())
        returned = hit.copy()              # anything that echoed back (ground, rocks, markers, creatures)
        tmap = self.sim.world.target_geom_ids
        self.spectro = None
        if tmap:
            is_t = np.isin(gid, self.sim.target_gid_array)
            th = np.flatnonzero(hit & is_t & (dist <= cfg.spectro_range))
            best = None
            for k in th:
                tidx = tmap[int(gid[k])]
                if best is None or dist[k] < best[1]:
                    best = (tidx, float(dist[k]))
                self.fleet.on_target_seen(t, self, tidx, float(dist[k]))
            self.spectro = best
            hit &= ~is_t
        lg = self.sim.world.life_gids
        if lg is not None and len(lg):
            is_l = hit & np.isin(gid, lg)
            if is_l.any():                 # creature returns go to the motion tracker, not the map
                k = np.flatnonzero(is_l)
                self.fleet.tracker.add_scan(self.id, pnt[None, :] + vec[k] * dist[k, None], t)
                hit &= ~is_l
        idx = np.flatnonzero(hit)
        # how far the ground and rocks still echo back (visibility); target markers are ignored, the
        # head can pass right through one when the rover drives over a target
        if len(idx) >= 30:
            self.far_hist.append(float(dist[idx].max()))
        if len(idx):
            pts = pnt[None, :] + vec[idx] * dist[idx, None]
            self.scan_pts = pts
            self.scan_changes = self.fleet.map.integrate(pts, t)
        else:
            self.scan_pts = np.zeros((0, 3))
            self.scan_changes = 0
        if not self.degraded and self.state not in PARKED:
            self._check_holes(t, pnt, vec, el, az, returned, dist)

    # ================================================================== holes in the LiDAR returns (liquid?)
    def _ground_z(self) -> float:
        m = self.fleet.map
        i, j = m.ij(self.x, self.y)
        return float(m.h[j, i]) if m.known[j, i] else self.z - 0.2

    def _check_holes(self, t, pnt, vec, el, az, returned, dist) -> None:
        """Steep rays straight ahead that should have hit the ground inside the LiDAR range but came
        back empty: a hole (liquid, or a drop-off). In each column the ground line through its two
        farthest steep returns is extended; a missing ray only counts if it would meet that line
        within range (so a downslope that falls away from the beam is not a hole). The expected ground
        points of missing rays become no-return evidence in the map; a hole in two sweeps in a row
        is an anomaly."""
        cfg = self.cfg
        nom = cfg.lidar_range
        if not self.far_hist or self.far_hist[-1] < 0.75 * nom:
            self.hole_prev = None          # no far returns anywhere: dust, not a hole
            return
        n_el, n_az = len(el), self.n_az
        steep = np.flatnonzero(el < math.radians(-9.0))
        if len(steep) < 3:
            return
        rel = np.angle(np.exp(1j * (az - self.yaw)))
        cols = np.flatnonzero(np.abs(rel) < HOLE_FRONT)
        R = returned.reshape(n_el, n_az)[np.ix_(steep, cols)]
        miss = ~R
        if int(miss.sum()) < HOLE_MIN_RAYS:
            self.hole_prev = None
            return
        V = vec.reshape(n_el, n_az, 3)[np.ix_(steep, cols)]
        D = dist.reshape(n_el, n_az)[np.ix_(steep, cols)]
        hor = np.hypot(V[..., 0], V[..., 1]) * D            # horizontal distance of each return
        Z = pnt[2] + V[..., 2] * D
        # two farthest returns per column -> ground line z = z1 + (h - h1) * g
        Hm = np.where(R, hor, -1.0)
        order = np.argsort(-Hm, axis=0)
        ci = np.arange(len(cols))
        i1, i2 = order[0], order[1]
        ok2 = (Hm[i1, ci] > 0.5) & (Hm[i2, ci] > 0.3) & (Hm[i1, ci] - Hm[i2, ci] > 0.3)
        h1, z1 = Hm[i1, ci], Z[i1, ci]
        g = np.where(ok2, (z1 - Z[i2, ci]) / np.maximum(Hm[i1, ci] - Hm[i2, ci], 1e-3), 0.0)
        g = np.clip(g, -0.6, 0.6)
        ground = self._ground_z()
        flat = ~ok2
        h1 = np.where(flat, 0.0, h1)
        z1 = np.where(flat, ground, z1)
        tan_el = np.tan(el[steep])[:, None]
        denom = tan_el - g[None, :]
        hx = (z1[None, :] - h1[None, :] * g[None, :] - pnt[2]) / np.where(np.abs(denom) > 1e-3, denom, -1e-3)
        reach = nom * np.cos(el[steep])[:, None] - 0.6
        expect = (denom < -0.03) & (hx > np.maximum(h1[None, :], 1.0) + 0.2) & (hx < reach)
        cosel = np.cos(el[steep])[:, None, None]
        P = pnt[None, None, :] + V * (hx[..., None] / np.maximum(cosel, 1e-3))
        half = (cfg.nx - 1) * cfg.res / 2.0 - 1.0
        inside = (np.abs(P[..., 0]) < half) & (np.abs(P[..., 1]) < half)
        mask = miss & expect & inside
        n = int(mask.sum())
        if n == 0:
            self.hole_prev = None
            return
        pts = P[mask][:, :2]
        self.fleet.map.add_noret(pts)
        steep12 = (el[steep] < math.radians(-11.0))[:, None]
        if n < HOLE_MIN_RAYS or len(set(np.nonzero(mask)[1].tolist())) < 3 or not (mask & steep12).any():
            self.hole_prev = None
            return
        f = self.fleet
        new = np.array([p for p in pts if f.water_zone_at(p[0], p[1], margin=0.4) is None])
        if len(new) < 4:
            self.hole_prev = None          # all of it is water we already know
            return
        cx, cy = float(new[:, 0].mean()), float(new[:, 1].mean())
        if f.water_zones and len(new) >= 5 and min(f.water_zone_gap(p[0], p[1]) for p in new) < 2.5:
            if t - self.ext_last_t > 1.0:  # more of a lake we already know: extend it quietly
                self.ext_last_t = t
                f.add_water_zone(t, self, *self._fit_zone(new), quiet=True)
            return
        if f.dropoff_near(cx, cy, t) or t - self.hole_last_t < 6.0:
            return
        prev, self.hole_prev = self.hole_prev, (cx, cy)
        if prev is None or math.hypot(prev[0] - cx, prev[1] - cy) > 3.0:
            return                         # needs two sweeps in a row
        self.hole_prev = None
        self.hole_last_t = t
        hc = sorted(set(np.nonzero(mask)[1].tolist()))
        edge = z1[hc][~flat[hc]]
        spread = float(edge.std()) if len(edge) >= 3 else None
        self._on_hole(t, (cx, cy), n, new, spread)

    @staticmethod
    def _fit_zone(pts: np.ndarray) -> tuple:
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        r = float(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).max()) + 0.8
        return cx, cy, min(6.0, max(1.5, r))

    def camera_liquid_pct(self):
        """Share (%) of the lower part of a small mast camera picture that shows blue liquid, or None if
        the camera cannot render here. The render is camera hardware (outside the brain budget); the
        colour check is perception (inside it)."""
        if getattr(self.sim, "camera_ok", True) is False:
            return None
        try:
            fr = self._hw_call(_camera_frame, self.sim, self.id)
        except Exception:                  # noqa: BLE001 (no OpenGL here: LiDAR only)
            self.sim.camera_ok = False
            return None
        rgb = fr.rgb[int(0.45 * fr.rgb.shape[0]):].astype(np.float32)
        r_, g_, b_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        blue = (b_ > 45.0) & (b_ > 1.8 * r_) & (b_ > 1.25 * g_)
        return round(100.0 * float(blue.mean()), 1)

    def _on_hole(self, t: float, at: tuple, rays: int, pts: np.ndarray, spread) -> None:
        """A hole ahead in two sweeps: slow down, look with the camera, and decide liquid or drop-off."""
        d = math.hypot(at[0] - self.x, at[1] - self.y)
        pct = self.camera_liquid_pct()
        details = {"at": at, "dist": round(d, 1), "rays": int(rays)}
        if pct is not None:
            details["camera_liquid_pct"] = pct
        if spread is not None:
            details["edge_spread"] = round(spread, 2)
        liquid = pct is not None and pct >= LIQUID_MIN_PCT
        flat_guess = pct is None and spread is not None and spread < 0.08
        f = self.fleet
        if liquid or flat_guess:
            self.on_anomaly(t, "no_returns", details, f"Slowing to {self.cfg.caution_v:.1f} m/s to look.")
            zx, zy, zr = self._fit_zone(pts)
            f.add_water_zone(t, self, zx, zy, zr)
            if liquid:
                msg = (f"{self.id} LiDAR gets no returns {d:.0f} m ahead and the camera shows a liquid surface "
                       f"({pct:.0f}% of the lower view is blue). Marking a lake and routing around it.")
            else:
                msg = (f"{self.id} LiDAR gets no returns {d:.0f} m ahead and the hole has a flat, level edge "
                       f"(spread {spread:.2f} m; no camera here). Probably liquid: marking it and routing around it.")
            f.log(t, "discovery", msg, self.id, at=(zx, zy))
            self._replan_now(t)
            self.plan_t = -1e9
        else:                              # a drop-off: just slow down and keep to mapped ground
            self.caution_until = t + CAUTION_S["no_returns"]
            self.caution_v = self.cfg.caution_v
            f.dropoffs.append((at[0], at[1], t))
            seen = "no liquid" if pct is not None else "nothing (no camera here)"
            f.log(t, "info", f"{self.id} LiDAR gets no returns {d:.0f} m ahead but the camera shows {seen}. "
                             f"Treating it as a drop-off and keeping to mapped ground.", self.id, at=at)

    def mark_water_here(self, t: float, at, source: str = "rules") -> bool:
        """Mark a hole around `at` as water (tier 2 or the operator may ask for it). True if new."""
        f = self.fleet
        if f.water_zone_at(at[0], at[1], margin=0.5) is not None:
            return False
        f.add_water_zone(t, self, at[0], at[1], 3.0, source=source)
        self._replan_now(t)
        self.plan_t = -1e9
        return True

    def _check_immersion(self, t: float) -> bool:
        """Failsafe: the belly probe is wet (the chassis is below a liquid surface). Stop, mark the water
        for the fleet and back out the way we came."""
        w = self.sim.terrain.water_at(self.x, self.y)
        if w is None:
            return False
        depth = w.level - (self.z - self.cfg.chassis_half[2])
        if depth <= 0.0 or (self.state == "recovering" and self.rec_phase == "water"):
            return False
        f = self.fleet
        ax, ay = self.x + 1.5 * math.cos(self.yaw), self.y + 1.5 * math.sin(self.yaw)
        f.add_water_zone(t, self, ax, ay, 2.5)
        f.log(t, "warn", f"{self.id} water sensor wet: chassis {depth:.2f} m under a liquid surface. Stopping, "
                         f"marking it for the fleet and backing out.", self.id, at=(self.x, self.y))
        if self.state != "recovering":
            self.prev_state = self.state
        if self.state == "observing":
            self.end_observing(t, "water", quiet=True)
            self.prev_state = "exploring"
        self.state = "recovering"
        self.rec_phase = "water"
        self.water_entry = (self.x, self.y)
        self.sand_t0 = t
        self.escape = list(reversed(self.trail))
        self.path = []
        self.searches.clear()
        self.hist.clear()
        self._set_cmd(0.0, 0.0)
        return True

    def _water_escape(self, t: float) -> None:
        dry = self.sim.terrain.water_at(self.x, self.y) is None
        ex, ey = self.water_entry if self.water_entry is not None else (self.x, self.y)
        if (dry and math.hypot(self.x - ex, self.y - ey) > 1.2) or t - self.sand_t0 > 40.0:
            self._set_cmd(0.0, 0.0)
            self.rec_phase = None
            self.state = self.prev_state if self.prev_state in ("exploring", "investigating", "returning") else "exploring"
            self.stall_cooldown = t + 2.0
            self.plan_t = -1e9
            self.next_goal_try = t
            self.goal = self.goal if self.state != "exploring" else None
            self.hist.clear()
            self.fleet.log(t, "info", f"{self.id} out of the water and dry. Planning around it.", self.id,
                           at=(self.x, self.y))
            return
        while self.escape and math.hypot(self.escape[0][0] - self.x, self.escape[0][1] - self.y) < 0.7:
            self.escape.pop(0)
        if self.escape:
            tx, ty = self.escape[0]
        else:
            tx, ty = self.x - 1.5 * math.cos(self.yaw), self.y - 1.5 * math.sin(self.yaw)
        e = wrap(math.atan2(ty - self.y, tx - self.x) - (self.yaw + math.pi))
        self._set_cmd(-0.3 * max(0.4, math.cos(e)), max(-0.6, min(0.6, 1.5 * e)))

    def bumper(self):
        """Three short rays from the chassis nose at wheel height: rocks right ahead?"""
        m, d = self.sim.model, self.sim.data
        R = d.xmat[self.body].reshape(3, 3)
        origin = d.xpos[self.body] + R @ np.array([0.25, 0.0, -0.02])
        best = None
        for k, a in enumerate((-0.4, 0.0, 0.4)):
            v = R @ np.array([math.cos(a), math.sin(a), 0.0])
            gid = np.zeros(1, np.int32)
            dist = self._hw_call(mujoco.mj_ray, m, d, origin, v, BUMPER_GROUPS, 1, -1, gid)
            if 0 <= dist < 0.8 and m.geom_group[gid[0]] == GROUP_ROCK:
                p = origin + v * dist
                if best is None or dist < best[0]:
                    best = (dist, float(p[0]), float(p[1]))
        return best

    def _check_visibility(self, t: float) -> None:
        """Infer LiDAR visibility from the rover's own returns: in dust, nothing comes back from
        beyond a few metres. Degraded = unsure: slow down, stay on mapped ground, stay close."""
        cfg = self.cfg
        if len(self.far_hist) < self.far_hist.maxlen:
            return
        far_max, far_min = max(self.far_hist), min(self.far_hist)
        nominal = cfg.lidar_range
        if not self.degraded and far_max < 0.55 * nominal:
            self.degraded = True
            self.lidar_eff = round(far_max, 1)
            if self.state in PARKED:
                tail = "Will drive slowly on mapped ground."
            else:
                tail = f"Slowing down, then {cfg.degraded_v_max:.1f} m/s on mapped ground only."
            self.on_anomaly(t, "low_visibility", {"lidar_range": round(far_max, 1)}, tail)
            self._replan_now(t)
        elif self.degraded and far_min > 0.75 * nominal:
            self.degraded = False
            self.lidar_eff = nominal
            self.fleet.log(t, "info", f"{self.id} visibility back to {nominal:.0f} m. Resuming normal speed.",
                           self.id, at=(self.x, self.y))
            self._replan_now(t)
        elif self.degraded:
            self.lidar_eff = round(far_max, 1)

    def on_anomaly(self, t: float, kind: str, details: dict, then: str = "") -> dict:
        """Perception flagged something it did not expect. Record it, slow down at once (before any
        re-planning: about caution_v m/s, or a brief stop for slip), and log what was seen plus
        `then`, what the rover does about it. kind: "new_obstacle", "slip" or "low_visibility".
        Every function in anomaly_hooks is then called with (rover, anomaly dict)."""
        assert kind in ANOMALY_KINDS, kind
        at = details.get("at", (self.x, self.y))
        a = {"t": round(float(t), 1), "kind": kind, "type": ANOMALY_TYPE[kind],
             "at": [round(float(at[0]), 1), round(float(at[1]), 1)]}
        a.update({k: v for k, v in details.items() if k != "at"})
        self.anomaly = a
        self.anomaly_log.append(a)
        self.caution_until = t + CAUTION_S[kind]
        self.caution_v = 0.0 if kind == "slip" else self.cfg.caution_v
        if kind == "new_obstacle" and details.get("shape") == "long":
            seen = (f"{self.id} sees a new long, low obstacle across its path {details.get('dist', 0.0):.0f} m ahead "
                    f"(LiDAR: about {details.get('length', 0.0):.1f} m long, {details.get('height', 0.0):.1f} m tall, "
                    f"{details.get('cells', 0)} map cells changed).")
        elif kind == "new_obstacle":
            seen = (f"{self.id} sees new rocks blocking its path {details.get('dist', 0.0):.0f} m ahead "
                    f"(LiDAR, {details.get('cells', 0)} map cells changed).")
        elif kind == "no_returns":
            seen = (f"{self.id} LiDAR gets no returns {details.get('dist', 0.0):.0f} m ahead where ground was "
                    f"expected ({details.get('rays', 0)} rays).")
        elif kind == "slip":
            seen = f"{self.id} wheels spinning but not moving (slip {details.get('slip', 0.0) * 100:.0f}%)."
        else:
            seen = f"{self.id} LiDAR returns down to {details.get('lidar_range', 0.0):.0f} m."
        self.fleet.log(t, "warn", f"{seen} {then}".strip(), self.id, at=at)
        for hook in list(self.anomaly_hooks):
            try:
                hook(self, a)
            except Exception:              # a broken add-on must never stop the rover
                pass
        return a

    def _check_new_obstacles(self, t: float) -> None:
        """Our own sweep just changed mapped ground (something new stands there). If it is on or right
        next to our route, say what we saw and replan, even if the route is not blocked yet."""
        if self.state not in DRIVING or t - self.last_rocks_t <= 8.0 or self.reroute is not None:
            return
        m = self.fleet.map
        i, j = m.ij(self.x, self.y)
        r = 18
        j0, i0 = max(j - r, 0), max(i - r, 0)
        win = m.changed_t[j0:j + r + 1, i0:i + r + 1] >= t - 3.0
        if int(win.sum()) < 2:
            return
        jj, ii = np.nonzero(win)
        cx = float(m.ox + (ii + i0).mean() * m.res)
        cy = float(m.oy + (jj + j0).mean() * m.res)
        if any(math.hypot(cx - ax, cy - ay) < 4.0 and t - at < 120.0 for ax, ay, at in self.rocks_seen):
            return
        dist = math.hypot(cx - self.x, cy - self.y)
        ahead = dist < 9.0 and abs(wrap(math.atan2(cy - self.y, cx - self.x) - self.yaw)) < 0.8
        pts = [(self.x, self.y)] + list(self.path[max(self.wp, 0):])
        near = False
        along = 0.0
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            L = math.hypot(bx - ax, by - ay)
            if L > 1e-6:
                q = max(0.0, min(1.0, ((cx - ax) * (bx - ax) + (cy - ay) * (by - ay)) / (L * L)))
                if math.hypot(cx - (ax + q * (bx - ax)), cy - (ay + q * (by - ay))) < 3.0:
                    near = True
                    break
            along += L
            if along > 15.0:
                break
        if not (near or ahead):
            return
        old = path_length(pts) if len(pts) > 1 else 0.0
        det = {"at": (cx, cy), "dist": round(dist, 1), "cells": int(win.sum())}
        shape = m.change_shape(cx, cy, t - 20.0)
        if shape is not None:
            det.update(shape)
        self.rocks_seen = [rs for rs in self.rocks_seen if t - rs[2] < 120.0] + [(cx, cy, t)]
        self.reroute = ("long" if det.get("shape") == "long" else "rocks", cx, cy, dist, old, None)
        self.on_anomaly(t, "new_obstacle", det, f"Slowing to {self.cfg.caution_v:.1f} m/s and replanning.")
        self.plan_t = -1e9

    def _replan_now(self, t: float) -> None:
        self.searches.clear()
        self.pending_goal_key = None
        self.plan_t = -1e9
        self.next_goal_eval = t

    # ================================================================== energy
    def energy_for_distance(self, dist: float) -> float:
        """Battery fraction to drive dist metres: learned motor cost per metre plus 25%, and the
        electronics drain at a cautious average speed (turns and slopes slow us down)."""
        cfg = self.cfg
        return dist * (1.25 * self.e_per_m + cfg.battery_base_drain / cfg.battery_capacity / (0.55 * cfg.v_max))

    def _battery(self, t: float, dt: float, energy_j: float) -> None:
        cfg = self.cfg
        parked = self.state == "idle" and math.hypot(self.x - self.dock[0], self.y - self.dock[1]) < 1.6
        if self.state == "charging" or parked:
            power = self.sim.hazards.charge_factor() if self.sim.hazards is not None else 1.0
            self.battery = min(1.0, self.battery + power * dt / cfg.charge_time_s)
            return
        cap = cfg.battery_capacity
        motor = cfg.battery_motor_k * energy_j / cap
        self.battery = max(0.0, self.battery - cfg.battery_base_drain / cap * dt - motor)
        moved = math.hypot(self.x - self.last_xy[0], self.y - self.last_xy[1])
        self.e_acc_drain += motor
        self.e_acc_dist += moved
        if self.e_acc_dist > 6.0:
            rate = self.e_acc_drain / self.e_acc_dist
            self.e_per_m = max(cfg.battery_motor_k * 12.0 / cfg.battery_capacity, 0.8 * self.e_per_m + 0.2 * rate)
            self.e_acc_drain = self.e_acc_dist = 0.0

    def home_distance(self, t: float) -> float:
        moved = math.hypot(self.x - self.home_dist_xy[0], self.y - self.home_dist_xy[1])
        if self.home_dist is None or t - self.home_dist_t > 12.0 or moved > 8.0:
            est = (self.home_dist + moved if self.home_dist is not None
                   else 1.5 * math.hypot(self.x - self.dock[0], self.y - self.dock[1]))
            path = self.fleet.plan(t, self, self.dock, max_exp=9000)
            if path is None:
                path = self.fleet.plan(t, self, self.dock, relax=4.0, max_exp=9000)
            if path is PENDING:
                self.home_pending = True        # out of compute this cycle: estimate, finish next cycle
                return est
            self.home_pending = False
            d = path_length([(self.x, self.y)] + path) if path else 1.5 * math.hypot(self.x - self.dock[0], self.y - self.dock[1])
            self.home_dist = d
            self.home_dist_t = t
            self.home_dist_xy = (self.x, self.y)
            return d
        return self.home_dist + moved

    # ================================================================== compute budget
    def plan_limits(self):
        """(wall clock deadline, expansion cap) for planning in this decision cycle. With the wall
        clock the planner stops when this cycle's share of the budget is used up; with the model
        clock it may do as many A* expansions as fit in that share at exp_cost_us each."""
        if self._c0 is None:
            return None, None
        cfg = self.cfg
        share = cfg.plan_share * (0.5 if self.t_now < self.share_until else 1.0)
        if cfg.plan_clock == "model":
            cap = int(self.budget_s * share / (cfg.exp_cost_us * 1e-6)) - self.exp_used
            return None, max(0, cap)
        used = time.perf_counter() - self._c0 - self._hw
        return time.perf_counter() + max(0.0005, self.budget_s * share - used), None

    def _record_brain(self, t: float, ms: float) -> None:
        self.brain_last = ms
        h = self.brain_hist
        h.append((t, ms))
        while h and h[0][0] < t - 10.0:
            h.popleft()
        self.brain_all.append(ms)
        budget = self.cfg.brain_budget_ms
        if ms > budget:
            over = sum(1 for _, v in h if v > budget)
            if over >= 3 and t - self.last_budget_warn > 60.0:
                self.last_budget_warn = t
                if self.cfg.plan_clock == "wall":
                    self.share_until = t + 30.0
                peak = max(v for _, v in h)
                self.fleet.log(t, "warn", f"{self.id} brain went over its {budget:.0f} ms budget {over} times in "
                                          f"10 s (peak {peak:.0f} ms). Planning in smaller slices.", self.id)

    def brain_msg(self) -> dict:
        h = self.brain_hist
        vals = [v for _, v in h] or [0.0]
        budget = self.cfg.brain_budget_ms
        return {"last_ms": round(self.brain_last, 1), "avg_ms": round(sum(vals) / len(vals), 1),
                "peak_ms": round(max(vals), 1), "over": sum(1 for v in vals if v > budget)}

    # ================================================================== main tick
    def tick(self, t: float, dt: float, energy_j: float, duty: bool = False) -> None:
        """One decision cycle. duty: this rover also runs the shared fleet bookkeeping this cycle
        (frontiers, reachability), and that time counts toward its brain."""
        self._hw = 0.0
        self.exp_used = 0
        self.t_now = t
        self._c0 = time.perf_counter()
        try:
            self._tick(t, dt, energy_j, duty)
            if self.advisor is not None:
                self.advisor.rover_tick(self, t)    # tier 2 bookkeeping (no model calls in here)
        finally:
            ms = (time.perf_counter() - self._c0 - self._hw) * 1000.0
            self._c0 = None
            self._record_brain(t, ms)

    def _tick(self, t: float, dt: float, energy_j: float, duty: bool) -> None:
        self.read_pose()
        self._battery(t, dt, energy_j)
        self.odometer += math.hypot(self.x - self.last_xy[0], self.y - self.last_xy[1])
        self.last_xy = (self.x, self.y)
        if self.rec_phase != "sand" and (not self.trail or
                                          math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 0.3):
            self.trail.append((self.x, self.y))
        self.scan(t)
        if duty:
            self.fleet.update(t)
        for k in [k for k, s in self.searches.items() if t - s.t0 > 3.5]:
            del self.searches[k]
        if len(self.plan_fails) > 20:
            self.plan_fails = {k: v for k, v in self.plan_fails.items() if t - v[0] < 3.0}
        self._read_drive_sensors(t)
        self._check_visibility(t)
        if self.scan_changes > 0:
            self._check_new_obstacles(t)
        self.hold_for_plan = False
        self.block_info = None

        # motor controller status (motor feedback)
        if self.motor_fault:
            if self.state != "fault":
                self._enter_fault(t)
            self._set_cmd(0.0, 0.0)
            self._apply()
            return
        if self.state == "fault":
            self._reboot(t)
        if self.obs is not None and self.state not in ("observing", "recovering"):
            self.end_observing(t, "interrupted", quiet=True)     # held, faulted or sent elsewhere
        elif self.state == "observing" and self.obs is None:
            self.state = "exploring"
        if self.state in DRIVING and self._check_immersion(t):
            self._apply()
            return

        if self.state == "idle":
            self._set_cmd(0.0, 0.0)
            if self.held:
                away = math.hypot(self.x - self.dock[0], self.y - self.dock[1]) > 1.6
                if away and self.battery > 0.0 and self._need_return(t):
                    # safety first: a long hold must not strand the rover
                    self.held = False
                    self._say(t, f"{self.id}: battery {self.battery * 100:.0f}%, leaving the hold to return "
                                 f"to the lander.", "info")
                    self.state = "exploring"
                    self._go_home(t, "battery")
            elif self.battery > 0.3 and not self.fleet.complete and t >= self.next_goal_try:
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
            self.fleet.log(t, "warn", f"{self.id} battery depleted. Stopped in the field.", self.id, at=(self.x, self.y))
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
        if self.slip_since is not None and t - self.slip_since >= self.cfg.slip_time_s:
            if self._slip_on_ground() and self.bumper() is None:
                self._on_soft_ground(t)
                self._apply()
                return
            self.slip_since = None             # hung up on a rock, not sand: leave it to the stall logic
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
        elif self.state == "observing":
            if self._observe(t):
                self._apply()
                return
        self._drive(t)
        self._apply()

    # ================================================================== state logic
    def _need_return(self, t: float) -> bool:
        if t >= self.next_home_check:
            self.need_home = self.energy_for_distance(self.home_distance(t))
            self.next_home_check = t + (0.2 if self.home_pending else 3.0)
        return self.battery < self.need_home + self.battery_margin

    def _say(self, t: float, text: str, kind: str = "info") -> None:
        """A chat line from this rover to the operator (protocol v3 chat), if tier 2 is attached."""
        if self.advisor is not None:
            self.advisor.say(self.id, t, text, kind)

    def _go_home(self, t: float, reason: str) -> None:
        if self.op_goal is not None and reason in ("battery", "range", "complete"):
            gx, gy = self.op_goal
            why = "exploration is complete" if reason == "complete" else f"battery {self.battery * 100:.0f}%"
            self._say(t, f"{self.id}: {why}, giving up on ({gx:g}, {gy:g}) and returning to the lander.")
        self.op_goal = None
        self.sticky_goal = None
        if reason in ("battery", "range"):
            self.pause_until = min(self.pause_until, t)     # no waiting around when the battery says go home
        self._drop_target(t, "battery low" if reason in ("battery", "range") else None)
        self.state = "returning"
        self.return_reason = reason
        self.dock_near_since = None
        self.goal = self.dock
        self.path = []
        self.plan_t = -1e9
        self.dwell_until = None
        self.next_goal_try = t + 4.0
        here = (self.x, self.y)
        if reason == "battery":
            self.fleet.log(t, "info", f"{self.id} battery at {self.battery * 100:.0f}%, needs about "
                                      f"{self.need_home * 100:.0f}% to get home plus a "
                                      f"{self.battery_margin * 100:.0f}% margin. Returning to the lander.",
                           self.id, at=here)
        elif reason == "range":
            self.fleet.log(t, "info", f"{self.id} battery at {self.battery * 100:.0f}% is not enough to reach the "
                                      f"remaining frontiers and get back. Returning to charge.", self.id, at=here)
        elif reason == "regroup":
            self.fleet.log(t, "info", f"{self.id} can't reach any frontier from here. "
                                      f"Heading toward the lander to find a way round.", self.id, at=here)
        elif reason == "cache":
            self.next_goal_try = t + 1.0

    def _drop_target(self, t: float, reason: str | None = None, retry_s: float = 20.0) -> None:
        if self.target_idx is not None:
            self.fleet.release_target(t, self, self.target_idx, reason, retry_s)
            self.target_idx = None
            self.dwell_until = None
            self.sampling = None

    def _check_charge_rate(self, t: float) -> None:
        """Watch the battery's own charge rate while docked. Slow charging (dust on the lander's
        panels) means a trip home leaves less time to recover: keep a bigger reserve until it clears."""
        cfg = self.cfg
        if self.battery >= 0.99:
            self.charge_hist.clear()
            return
        self.charge_hist.append((t, self.battery))
        if len(self.charge_hist) < self.charge_hist.maxlen:
            return
        (t0, b0), (t1, b1) = self.charge_hist[0], self.charge_hist[-1]
        share = (b1 - b0) / max(t1 - t0, 1e-6) * cfg.charge_time_s      # 1.0 = normal charging power
        if not self.slow_charge and share < 0.6:
            self.slow_charge = True
            self.battery_margin = max(cfg.battery_margin, cfg.slow_charge_margin)
            self.fleet.log(t, "warn", f"{self.id} charging at {share * 100:.0f}% of normal (dust on the lander's "
                                      f"panels). Raising return margin to {self.battery_margin * 100:.0f}%.",
                           self.id, at=self.dock)
        elif self.slow_charge and share > 0.9:
            self.slow_charge = False
            self.battery_margin = cfg.battery_margin
            self.fleet.log(t, "info", f"{self.id} charging at full rate again. Return margin back to "
                                      f"{self.battery_margin * 100:.0f}%.", self.id, at=self.dock)

    def _charging(self, t: float) -> None:
        self._set_cmd(0.0, 0.0)
        self._check_charge_rate(t)
        if self.battery >= 0.995:
            self.battery = 1.0
            if self.held:
                self.state = "idle"
                self.fleet.log(t, "info", f"{self.id} fully charged. Holding at the lander for the operator.",
                               self.id, at=self.dock)
            elif self.fleet.complete:
                self.state = "idle"
                self.fleet.log(t, "info", f"{self.id} fully charged and parked at the lander.", self.id, at=self.dock)
            else:
                self.state = "exploring"
                self.goal, self.path = None, []
                self.next_goal_try = t
                self.fleet.log(t, "info", f"{self.id} fully charged. Heading back out.", self.id, at=self.dock)

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
            self.fleet.cache_samples(t, self)
            if self.return_reason == "cache" and self.battery >= 0.6 and not self.fleet.complete:
                self.state = "exploring"
                self.next_goal_try = t
                self.fleet.log(t, "info", f"{self.id} heading back out with {self.capacity} free tubes "
                                          f"(battery {self.battery * 100:.0f}%).", self.id, at=self.dock)
                return True
            if self.fleet.complete and self.battery > 0.9:
                self.state = "idle"
                self.fleet.log(t, "info", f"{self.id} parked at the lander.", self.id, at=self.dock)
            else:
                self.state = "charging"
                self.fleet.log(t, "info", f"{self.id} docked at {self.battery * 100:.0f}%. Charging.", self.id,
                               at=self.dock)
            return True
        if self.return_reason == "regroup" and t >= self.next_goal_try:
            # on the way back, keep checking whether a frontier has become reachable
            self.next_goal_try = t + 4.0
            res = self.fleet.choose_goal(t, self)
            if res is PENDING:
                self.next_goal_try = t          # finish that plan next cycle
            elif res is not None:
                goal, path, size = res
                self.state = "exploring"
                self.return_reason = None
                self.no_goal_since = None
                self.goal, self.goal_size, self.goal_t = goal, size, t
                self.next_goal_eval = t + 15.0
                self._set_path(t, path)
                self.fleet.log(t, "info", f"{self.id} found a way through. Back to exploring.", self.id,
                               at=(self.x, self.y))
                return False
        self.goal = self.dock
        if self._need_replan(t):
            path = self._plan_home(t)
            self._after_replan(t, path)
            if path is PENDING:
                self._hold_if_blocked()
            elif path is None:
                if self.block_info is not None and self.block_info[3] != "terrain":
                    self.path = []             # do not drive into what blocked us; try again shortly
                    self.plan_t = t
                else:
                    self._last_relax = 4.0     # nothing found: head straight for the lander, carefully
                    self._set_path(t, [(self.x, self.y), self.dock])
            else:
                self._set_path(t, path)
        return False

    def _plan_home(self, t: float, max_exp: int = 14000):
        """Path to the dock. If the learned limit boxes us in, allow slightly steeper ground."""
        for relax in (0.0, 2.0, 4.0):
            path = self.fleet.plan(t, self, self.dock, relax=relax, max_exp=max_exp)
            if path is PENDING:
                return PENDING
            if path is not None:
                self._last_relax = relax
                return path
        return None

    def _explore(self, t: float) -> None:
        f = self.fleet
        if self.op_goal is not None:
            self._op_drive(t)
            return
        # an already found target that nobody is sampling?
        if t >= self.next_target_check:
            self.next_target_check = t + 3.0
            if self.samples and len(self.samples) >= self.capacity and t - self.cache_note_t > 60.0:
                k, d = f.valuable_target(t, self)
                if k is not None:
                    ts_ = f.targets[k]
                    need = self.energy_for_distance(1.3 * (self.home_dist or math.hypot(self.x, self.y)))
                    if self.battery > need + self.battery_margin + 0.1:
                        self.cache_note_t = t
                        f.log(t, "info", f"{self.id} tubes full ({len(self.samples)} of {self.capacity}) and "
                                         f"{ts_.spec.id} looks valuable (spectrometer estimate "
                                         f"{ts_.spec.est_value}/10, {d:.0f} m away). Returning to cache the samples "
                                         f"first.", self.id, at=(ts_.spec.x, ts_.spec.y))
                        self._go_home(t, "cache")
                        return
            k, d = f.unclaimed_target(t, self)
            reach = 8.0 if self.degraded else 30.0     # no far investigations while unsure
            b = self.dir_bias
            if k is not None and b is not None and t < b[2]:    # operator: "explore north"
                ts_ = f.targets[k].spec
                if ((ts_.x - self.x) * b[0] + (ts_.y - self.y) * b[1]) < 0.3 * max(d, 1e-6):
                    k = None
            if k is not None and d < reach and self.can_investigate(t, f.targets[k]):
                ts = f.targets[k]
                ts.status = "investigating"
                ts.investigator = self.id
                self.start_investigation(t, k)
                f.log(t, "info", f"{self.id} heading to sample {ts.spec.id} ({KIND_LABEL[ts.spec.kind]}, "
                                 f"found by {ts.found_by}), {d:.0f} m away.", self.id, at=(ts.spec.x, ts.spec.y))
                return
        need = self.goal is None
        reached = False
        if not need:
            dgoal = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
            reached = dgoal < 1.5
            # a frontier Gemma picked (sticky) is kept until reached or no longer a frontier
            periodic = t >= self.next_goal_eval and self.goal != self.sticky_goal
            if reached or not self._goal_still_frontier() or periodic:
                need = True
        replan = self.goal is not None and self._need_replan(t)
        if need and t >= self.next_goal_try:
            res = f.choose_goal(t, self)
            if res is PENDING:
                self._hold_if_blocked()       # out of compute: keep the current plan, finish next cycle
                return
            if res is None:
                self._after_replan(t, None)
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
            self._after_replan(t, path)
            old = self.goal
            self.goal, self.goal_size, self.goal_t = goal, size, t
            self.sticky_goal = None
            self.next_goal_eval = t + 15.0
            self._set_path(t, path)
            f.takeover_note(t, self, goal)
            if self.advisor is not None:          # tier 2 may suggest another option; we drive this one now
                self.advisor.on_goal_chosen(self, t, goal, path, size, reached)
            big_change = old is None or math.hypot(goal[0] - old[0], goal[1] - old[1]) > 15.0
            if t - self.last_goal_event_t > 75.0 and big_change:
                self.last_goal_event_t = t
                dx, dy = goal[0] - self.x, goal[1] - self.y
                from .fleet import compass
                extra = " near a sampled target" if f.bonus_area(goal[0], goal[1]) else ""
                f.log(t, "info", f"{self.id} heading to a frontier {math.hypot(dx, dy):.0f} m {compass(dx, dy)}"
                                 f"{extra} ({size} cells).", self.id, at=goal)
            return
        if replan:
            path = f.plan(t, self, self.goal)
            self._after_replan(t, path)
            if path is PENDING:
                self._hold_if_blocked()
            elif path is None:
                self.blacklist[f._bucket(*self.goal)] = t + 40.0
                self.goal, self.path = None, []
                self.next_goal_try = t
            else:
                self._set_path(t, path)

    def _op_drive(self, t: float) -> None:
        """Drive to the operator's go_to point (state stays "exploring"). The usual safety checks
        (battery, blocked path, stalls) still apply; autonomy resumes on arrival."""
        gx, gy = self.op_goal
        if math.hypot(gx - self.x, gy - self.y) < 1.5:
            self.op_goal = None
            self.goal, self.path = None, []
            self.next_goal_try = t
            self._say(t, f"{self.id}: arrived at ({gx:g}, {gy:g}). Back to exploring.")
            self.fleet.log(t, "info", f"{self.id} reached the operator's point ({gx:.0f}, {gy:.0f}). "
                                      f"Back to exploring.", self.id, at=(gx, gy), source="operator")
            return
        self.goal = self.op_goal
        if not self._need_replan(t):
            return
        path = self.fleet.plan(t, self, self.goal, max_exp=30000)
        self._after_replan(t, path)
        if path is PENDING:
            self._hold_if_blocked()
        elif path is None:
            self.op_fails += 1
            if self.block_info is not None:
                self.path = []                 # never keep driving into what blocked us
            self.plan_t = t
            if self.op_fails >= 3:
                self.op_goal = None
                self.goal, self.path = None, []
                self.next_goal_try = t
                self._say(t, f"{self.id}: no safe path to ({gx:g}, {gy:g}) any more "
                             f"(slope limit {self.fleet.slope_limit:.0f} deg). Back to exploring.")
        else:
            self.op_fails = 0
            self._set_path(t, path)

    def _goal_still_frontier(self) -> bool:
        gx, gy = self.goal
        return any(abs(fx - gx) < 3.0 and abs(fy - gy) < 3.0 for (fx, fy, _s, _i) in self.fleet.frontiers)

    def free_tubes(self) -> int:
        return self.capacity - len(self.samples)

    def tube_verdict(self, ts) -> str | None:
        """Why this rover should not sample this target with its tubes, or None. With one tube left,
        ordinary finds (spectrometer estimate under 6/10) are skipped to keep it for a better one."""
        free = self.free_tubes()
        if free <= 0:
            return f"all {self.capacity} tubes full"
        if free == 1 and self.capacity > 1 and ts.spec.est_value < 6:
            return (f"estimated value {ts.spec.est_value}/10 and only 1 of {self.capacity} tubes left, keeping it "
                    f"for a better find")
        return None

    def can_investigate(self, t: float, ts) -> bool:
        if self.state != "exploring" or self.target_idx is not None:
            return False
        if self.tube_verdict(ts) is not None:
            return False
        d = math.hypot(ts.spec.x - self.x, ts.spec.y - self.y)
        if self.degraded and d > 8.0:
            return False
        need = self.energy_for_distance(1.3 * d + 1.3 * math.hypot(ts.spec.x, ts.spec.y)) + self.battery_margin + 0.05
        return self.battery > need

    def instrument_time(self, inst: str) -> float:
        cfg = self.cfg
        return {"drill": cfg.drill_s, "scoop": cfg.scoop_s, "sniffer": cfg.sniffer_s,
                "swab": cfg.swab_s}.get(inst, cfg.investigate_dwell_s)

    def start_investigation(self, t: float, tidx: int) -> None:
        ts = self.fleet.targets[tidx]
        self.state = "investigating"
        self.target_idx = tidx
        self.goal = (ts.spec.x, ts.spec.y)
        self.path = []
        self.plan_t = -1e9
        self.dwell_until = None
        self.inv_fails = 0
        self.next_inv_plan = t

    def _investigate(self, t: float) -> bool:
        f = self.fleet
        if self.target_idx is None:
            self.state = "exploring"
            return False
        ts = f.targets[self.target_idx]
        if self.dwell_until is not None:
            self._set_cmd(0.0, 0.0)
            if t >= self.dwell_until:
                self.sampling = None
                f.target_sampled(t, self, self.target_idx)
                self.target_idx = None
                self.dwell_until = None
                self.state = "exploring"
                self.goal, self.path = None, []
                self.next_goal_try = t
                self.next_target_check = t
            return True
        d = math.hypot(ts.spec.x - self.x, ts.spec.y - self.y)
        if d <= self.cfg.investigate_radius:
            if self.free_tubes() <= 0:             # filled up on the way (operator, timing): leave it for later
                self._drop_target(t, f"all {self.capacity} tubes full")
                self.state = "exploring"
                self.goal, self.path = None, []
                return False
            inst = ts.spec.instrument
            dur = self.instrument_time(inst)
            self.dwell_until = t + dur
            self.sampling = {"tidx": self.target_idx, "instrument": inst, "t0": t, "dur": dur}
            self._set_cmd(0.0, 0.0)
            self.path = []
            return True
        self.goal = (ts.spec.x, ts.spec.y)
        if not self.path and t < self.next_inv_plan:
            self._set_cmd(0.0, 0.0)
            return True
        if self._need_replan(t):
            path = f.plan(t, self, self.goal)
            self._after_replan(t, path)
            if path is PENDING:
                self._hold_if_blocked()
            elif path is None:
                self.inv_fails += 1
                if self.block_info is not None:
                    self.path = []                 # never keep driving into what blocked us
                    self.next_inv_plan = t + 2.0
                if self.inv_fails >= 3:
                    self._drop_target(t, "no safe path found")
                    self.state = "exploring"
                    self.goal, self.path = None, []
                    return False
                self.plan_t = t
            else:
                self._set_path(t, path)
        return False

    # ================================================================== soft ground (wheel slip)
    def _on_soft_ground(self, t: float) -> None:
        cfg, f = self.cfg, self.fleet
        R = cfg.soft_zone_radius
        slip = max(self.slip_peak, self.slip)
        self.on_anomaly(t, "slip", {"slip": round(slip, 2), "wheel_speed": round(self.v_wheel, 2),
                                    "ground_speed": round(self.speed_fwd, 2)},
                        "Soft ground. Backing out and marking it for the fleet.")
        # the soft ground starts about where the front wheels began to slip and lies ahead; its size is
        # unknown, so assume a generous patch around that
        ex, ey = self.slip_xy if self.slip_xy is not None else (self.x, self.y)
        cx = ex + math.cos(self.yaw) * (cfg.wheel_x + 0.65 * R)
        cy = ey + math.sin(self.yaw) * (cfg.wheel_x + 0.65 * R)
        zone = f.soft_zone_at(cx, cy, margin=-1.0)
        if zone is None:
            f.add_soft_zone(t, self, cx, cy, R)
            zone = f.soft_zones[-1]
        self.soft_seen.add(zone[:2])
        if self.state != "recovering":
            self.prev_state = self.state
        if self.prev_state == "exploring":
            self.goal = None
        self.state = "recovering"
        self.rec_phase = "sand"
        self.sand_zone = zone
        self.sand_t0 = t
        self.sand_freed = False
        self.escape = list(reversed(self.trail))      # newest first: drive back the way we came
        self.path = []
        self.searches.clear()
        self.slip_since = None
        self.hist.clear()
        self._set_cmd(0.0, 0.0)

    def _sand_escape(self, t: float) -> None:
        cfg = self.cfg
        zx, zy, zr = self.sand_zone[0], self.sand_zone[1], self.sand_zone[2]
        if math.hypot(self.x - zx, self.y - zy) >= zr + 0.9:
            self._set_cmd(0.0, 0.0)
            self.rec_phase = None
            self.state = self.prev_state if self.prev_state not in ("recovering", "fault") else "exploring"
            self.stall_cooldown = t + 2.0
            self.plan_t = -1e9
            self.next_goal_try = t
            self.next_inv_plan = t
            self.hist.clear()
            self.fleet.log(t, "info", f"{self.id} backed out of the soft ground. Planning around it.", self.id,
                           at=(self.x, self.y))
            return
        if t - self.sand_t0 > cfg.sand_free_s and not self.sand_freed:
            self.sand_freed = True
            if self.sim.hazards is not None:
                self.sim.hazards.free_rover(self.id, t)
            self.fleet.log(t, "warn", f"{self.id} dug itself out.", self.id, at=(self.x, self.y))
        if t - self.sand_t0 > cfg.sand_free_s + 30.0:
            self.sand_zone = (self.x, self.y, 0.0)     # give up on backing out; carry on from here
            return
        while self.escape and math.hypot(self.escape[0][0] - self.x, self.escape[0][1] - self.y) < 0.7:
            self.escape.pop(0)
        if self.escape:
            tx, ty = self.escape[0]
        else:
            tx, ty = self.x - 1.5 * math.cos(self.yaw), self.y - 1.5 * math.sin(self.yaw)
        e = wrap(math.atan2(ty - self.y, tx - self.x) - (self.yaw + math.pi))
        self._set_cmd(-0.25 * max(0.4, math.cos(e)), max(-0.6, min(0.6, 1.5 * e)))

    # ================================================================== motor fault
    def _enter_fault(self, t: float) -> None:
        f = self.fleet
        st = self.state if self.state != "recovering" else self.prev_state
        had_frontier = self.goal is not None and st == "exploring"
        self.fault_prev = st if st not in ("fault", "recovering") else "exploring"
        tid = None if self.target_idx is None else f.targets[self.target_idx].spec.id
        f.rover_faulted(t, self)
        self._drop_target(t, None, retry_s=0.0)
        self.goal, self.path = None, []
        self.searches.clear()
        self.pending_goal_key = None
        self.rec_phase = None
        self.dwell_until = None
        self.state = "fault"
        if tid:
            what = f"Brakes on, releasing {tid} to the fleet."
        elif had_frontier:
            what = "Brakes on, releasing its frontier to the fleet."
        else:
            what = "Brakes on, telling the fleet it is parked here."
        f.log(t, "warn", f"{self.id} motor controller not responding. {what}", self.id, at=(self.x, self.y))

    def _reboot(self, t: float) -> None:
        at_dock = math.hypot(self.x - self.dock[0], self.y - self.dock[1]) < 1.6
        if self.fault_prev == "charging" and at_dock and self.battery < 0.995:
            self.state = "charging"
        elif self.fault_prev == "idle" and self.fleet.complete:
            self.state = "idle"
        else:
            self.state = "exploring"
        self.next_goal_try = t
        self.next_goal_eval = t
        self.plan_t = -1e9
        self.stall_cooldown = t + 2.0
        self.hist.clear()
        self.fleet.log(t, "info", f"{self.id} rebooted. Back online.", self.id, at=(self.x, self.y))

    # ================================================================== lifeforms: observe from a safe distance
    def can_observe(self, t: float, tr) -> bool:
        """Free, healthy, enough battery, close enough: may switch to observing this track."""
        if self.state != "exploring" or self.held or self.op_goal is not None or self.degraded:
            return False
        if self.dir_bias is not None and t < self.dir_bias[2]:
            return False
        px, py = tr.predict(t)
        d = math.hypot(px - self.x, py - self.y)
        if d > 20.0:
            return False
        home = self.energy_for_distance(1.3 * math.hypot(px, py) + 1.3 * d)
        return self.battery >= max(0.4, home + self.battery_margin + 0.15)

    def start_observing(self, t: float, tr) -> None:
        from .fleet import compass
        px, py = tr.predict(t)
        d = math.hypot(px - self.x, py - self.y)
        self.prev_state = self.state
        self.state = "observing"
        self.sticky_goal = None
        self.goal, self.path = None, []
        self.searches.clear()
        self.obs = {"track": tr.id, "t0": t, "x0": self.x, "y0": self.y, "mode": "look", "look_until": t + 1.5,
                    "seen": {tr.id}, "speeds": [], "next_plan": t, "goal": None, "life": False, "decided": False,
                    "most": 1, "start": (px, py), "far": 0.0}
        tr.observer = self.id
        self.fleet.log(t, "discovery", f"{self.id} sees something moving {d:.0f} m {compass(px - self.x, py - self.y)} "
                                       f"(LiDAR track {tr.id}, {tr.speed:.1f} m/s, about {tr.size:.1f} m). Stopping to "
                                       f"look.", self.id, at=(px, py))
        if self.advisor is not None and self.advisor.on_sighting(self, tr, t):
            self.obs["look_until"] = t + 12.0           # tier 2 decides (Gemma's look, maybe the operator)

    def set_obs_mode(self, t: float, mode: str) -> None:
        """mode: "follow" (approach to the standoff distance and follow), "here" (watch from where it
        is, never closer) or "stop" (resume exploring)."""
        if self.obs is None:
            return
        self.obs["decided"] = True
        if mode == "stop":
            self.end_observing(t, "operator")
            return
        self.obs["mode"] = mode
        self.obs["next_plan"] = t

    def _face(self, px: float, py: float) -> None:
        e = wrap(math.atan2(py - self.y, px - self.x) - self.yaw)
        self._set_cmd(0.0, 0.0 if abs(e) < 0.12 else max(-0.6, min(0.6, 1.5 * e)))

    def _safe_cell(self, x: float, y: float) -> bool:
        f = self.fleet
        c = f._cost if f._cost is not None else f.cost(self.t_now)
        i, j = f.map.ij(x, y)
        return bool(f.map.known[j, i] and np.isfinite(c[j, i]))

    def _standoff_goal(self, px: float, py: float, d: float):
        """A safe spot about the standoff distance from the creature, on our side of it."""
        sd = self.cfg.observe_standoff
        base = math.atan2(self.y - py, self.x - px)
        for off in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
            gx, gy = px + sd * math.cos(base + off), py + sd * math.sin(base + off)
            if math.hypot(gx - self.obs["x0"], gy - self.obs["y0"]) > self.cfg.observe_leash - 1.0:
                continue
            if self._safe_cell(gx, gy) and self.fleet.water_zone_at(gx, gy, margin=1.0) is None:
                return gx, gy
        return None

    def _observe(self, t: float) -> bool:
        """One cycle of observing. Returns True if it set the motor command itself."""
        cfg, f, o = self.cfg, self.fleet, self.obs
        tr = f.tracker.get(o["track"])
        if tr is None or (tr.status == "lost" and t - tr.last_seen > 8.0):
            self.end_observing(t, "lost")
            return False
        if tr.id != o["track"]:            # the tracker merged our track into an older one
            o["track"] = tr.id
            tr.observer = self.id
        px, py = tr.predict(t)
        d = math.hypot(px - self.x, py - self.y)
        fresh = []
        for o2 in f.tracker.near(px, py, 12.0, t):
            o["seen"].add(o2.id)
            if t - o2.last_seen < 1.0:
                fresh.append(o2)
                if o2.speed > 0.05:
                    o["speeds"].append(o2.speed)
        # individuals: the most separate tracks seen at the same time
        o["most"] = max(o["most"], f.tracker.distinct(fresh, t))
        o["far"] = max(o["far"], math.hypot(px - o["start"][0], py - o["start"][1]))
        if tr.status == "lost":            # the cluster may have come back as a new track nearby
            alt = [o2 for o2 in f.tracker.near(px, py, 3.0, t) if o2 is not tr and o2.observer is None]
            if alt:
                tr.observer = None
                tr = alt[0]
                tr.observer = self.id
                o["track"] = tr.id
                px, py = tr.predict(t)
                d = math.hypot(px - self.x, py - self.y)
        if t - o["t0"] >= cfg.observe_max_s:
            self.end_observing(t, "time")
            return False
        if self._need_return(t):
            self.end_observing(t, "battery")
            self._go_home(t, "battery")
            return False
        if math.hypot(self.x - o["x0"], self.y - o["y0"]) > cfg.observe_leash:
            self.end_observing(t, "leash")
            return False
        if o["mode"] == "look" and t >= o["look_until"]:
            o["mode"] = "follow"           # rules alone (no tier 2 answer): follow carefully
            if not o["decided"]:
                f.log(t, "info", f"{self.id} keeps watching track {tr.id} with the rules alone. Approaching to "
                                 f"{cfg.observe_standoff:.0f} m to observe.", self.id, at=(px, py))
        # the closest tracked creature (any of the herd) decides when to back off
        dn, near = d, (px, py)
        for o2 in f.tracker.near(self.x, self.y, cfg.observe_min_dist + 1.0, t):
            qx, qy = o2.predict(t)
            dq = math.hypot(qx - self.x, qy - self.y)
            if dq < dn:
                dn, near = dq, (qx, qy)
        if dn < cfg.observe_min_dist + 0.5:
            bx, by = self.x - 1.2 * math.cos(self.yaw), self.y - 1.2 * math.sin(self.yaw)
            self._face(*near)
            if self._safe_cell(bx, by) and f.water_zone_at(bx, by, margin=0.5) is None:
                self.cmd_v = -0.25
            if t - self.last_backoff_t > 20.0:
                self.last_backoff_t = t
                f.log(t, "info", f"{self.id} a creature is {dn:.1f} m away. Backing off to keep at least "
                                 f"{cfg.observe_min_dist:.0f} m (planetary protection).", self.id, at=near)
            self.path = []
            return True
        if o["mode"] != "follow" or d <= cfg.observe_standoff + 1.0:
            self.path = []
            self.goal = None
            self._face(px, py)
            return True
        # follow: approach to the standoff distance, replanning as it moves
        moved = o["goal"] is None or math.hypot(px - o["goal"][2], py - o["goal"][3]) > 1.5
        if t >= o["next_plan"] or moved or not self.path:
            g = self._standoff_goal(px, py, d)
            if g is None:
                o["mode"] = "here"
                f.log(t, "info", f"{self.id} can't get closer to track {tr.id}: no safe ground near it (steep, "
                                 f"soft or water). Observing from here, {d:.0f} m away.", self.id, at=(px, py))
                return False
            path = f.plan(t, self, g)
            if path is PENDING:
                self._face(px, py)
                return True
            o["next_plan"] = t + 2.0
            if path is None:
                o["mode"] = "here"
                f.log(t, "info", f"{self.id} no safe path closer to track {tr.id}. Observing from here, "
                                 f"{d:.0f} m away.", self.id, at=(px, py))
                return False
            o["goal"] = (g[0], g[1], px, py)
            self.goal = g
            self._set_path(t, path)
        return False

    def end_observing(self, t: float, why: str, quiet: bool = False) -> None:
        o, f = self.obs, self.fleet
        if o is None:
            return
        self.obs = None
        tr = f.tracker.get(o["track"])
        px, py = tr.predict(t) if tr is not None else (self.x, self.y)
        dur = t - o["t0"]
        seen = [f.tracker.get(k) for k in o["seen"]]
        seen = [s_ for s_ in seen if s_ is not None and s_.confirmed]
        n = max(1, o["most"])
        sp = o["speeds"]
        avg = sum(sp) / len(sp) if sp else (tr.speed if tr is not None else 0.0)
        # the rules confirm a lifeform on their own if it kept moving by itself while we watched
        life = o.get("life") or (o["far"] > 1.5 and dur > 20.0 and avg > 0.1)
        for s_ in seen:
            s_.observer = None
            s_.done_until = t + 300.0
            if life:
                s_.lifeform = True
        if life:
            f.confirm_herd(px, py, n, t)
        f.observed_areas.append((px, py, t))
        if self.state == "observing":
            self.state = "exploring"
        self.goal, self.path = None, []
        self.next_goal_try = t
        self.next_goal_eval = t
        if quiet:
            return
        who = "the herd" if n > 1 else "the creature"
        tail = {"time": "Resuming exploration.", "lost": "Lost sight of it. Resuming exploration.",
                "leash": f"Not going more than {self.cfg.observe_leash:.0f} m from where it started. Resuming exploration.",
                "battery": f"Battery {self.battery * 100:.0f}%, returning to the lander.",
                "operator": "Resuming exploration (operator).",
                "not_life": "Not a lifeform. Resuming exploration."}.get(why, "Resuming exploration.")
        f.log(t, "discovery", f"{self.id} observed {who} for {dur:.0f} s: {n} individual{'s' if n > 1 else ''}, "
                              f"average speed {avg:.1f} m/s. {tail}", self.id, at=(px, py))

    # ================================================================== recovery, stalls, flips
    def _track(self, t: float) -> None:
        intended_fwd = self.cmd_v > 0.2 and self.path
        self.hist.append((t, self.x, self.y, self.yaw, self.pitch, bool(intended_fwd), self.cmd_w, self.tilt,
                          self.motor_load))
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
        if t < self.pause_until:               # standing still on purpose (waiting)
            self.stuck_ref = None
            self.hist.clear()
            return False
        h = self.hist
        if len(h) == h.maxlen and all(e[5] for e in h):
            t0, x0, y0, yaw0 = h[0][0], h[0][1], h[0][2], h[0][3]
            prog = (self.x - x0) * math.cos(yaw0) + (self.y - y0) * math.sin(yaw0)
            if prog < self.cfg.stall_progress_m:
                self._on_stall(t)
                return True
        # generic stuck check: no real movement (or turning) for 9 s while trying to drive somewhere
        if self.path and self.state in DRIVING:
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
        here = (self.x, self.y)
        # traffic: another rover right in front
        for o in f.rovers:
            if o is not self and math.hypot(o.x - self.x, o.y - self.y) < 1.9:
                if abs(wrap(math.atan2(o.y - self.y, o.x - self.x) - self.yaw)) < 1.0:
                    self._start_recovery(t, None)
                    return
        bump = self.bumper()
        self.recent_stalls.append((t, self.x, self.y))
        if bump is not None:
            bd, bx, by = bump
            f.map.mark_blocked(bx, by, 0.6)
            f.log(t, "info", f"{self.id} blocked by a rock (bumper ray hit at {bd:.1f} m). Marking it as an obstacle "
                             f"and rerouting.", self.id, at=(bx, by))
            self._start_recovery(t, (bx, by))
            return
        pitches = sorted(e[4] for e in h)
        p_med, p_max = pitches[len(pitches) // 2], pitches[-1]
        loads = sorted(e[8] for e in h)
        load = loads[int(0.8 * (len(loads) - 1))]
        turning = sum(abs(e[6]) for e in h) / max(len(h), 1) > 0.45
        ax = self.x + 0.7 * math.cos(self.yaw)
        ay = self.y + 0.7 * math.sin(self.yaw)
        # only blame the slope if gravity alone would eat most of the motors' torque at this pitch
        gravity_share = self.grav_torque * math.sin(math.radians(max(p_med, 0.0))) / (4.0 * self.cfg.motor_torque)
        if p_med >= 12.0 and p_med >= 0.8 * p_max and not turning and gravity_share >= 0.8:
            # nose up on a steady slope with full power and no progress: the slope beat the motors.
            # The tilt (pitch and roll together) is the slope the planner's map measures.
            tilts = sorted(e[7] for e in h)
            if not f.report_slope_stall(t, self, max(p_med, tilts[len(tilts) // 2]), load):
                f.map.mark_blocked(ax, ay, 0.5)    # steeper than the map said: avoid this exact spot
            self._start_recovery(t, None)
            return
        if p_med >= 12.0 or (turning and p_med >= 8.0):
            # wheels lost grip while turning or on uneven ground: not evidence about the slope limit
            f.log(t, "info", f"{self.id} lost grip on a {p_med:.0f} deg slope (IMU pitch). Backing off to try "
                             f"another line.", self.id, at=here)
            self._start_recovery(t, None)
            return
        # unexplained stall on flat ground: treat the spot ahead as an obstacle
        f.map.mark_blocked(ax, ay, 0.5)
        f.log(t, "info", f"{self.id} stuck on rough ground (no progress in {self.cfg.stall_window_s:.0f} s, motors at "
                         f"{load * 100:.0f}%). Marking the spot as untraversable.", self.id, at=(ax, ay))
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
        if self.rec_phase == "sand":
            self._sand_escape(t)
        elif self.rec_phase == "water":
            self._water_escape(t)
        elif self.rec_phase == "reverse":
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
                tilt = self.tilt
                self._hw_call(self.sim.reset_rover_upright, self)
                self.read_pose()
                self.fleet.log(t, "warn", f"{self.id} tipped over (IMU tilt {tilt:.0f} deg). Self-righting and "
                                          f"replanning.", self.id, at=(self.x, self.y))
                self.path = []
                self.plan_t = -1e9
                self.hist.clear()
                if self.state == "recovering":
                    self.state = self.prev_state
                    self.rec_phase = None
                self.stall_cooldown = t + 2.0
                return True
        else:
            self.flip_since = None
        return False

    # ================================================================== driving
    def _need_replan(self, t: float) -> bool:
        self.block_info = None
        if not self.path or self.wp >= len(self.path):
            return True
        f = self.fleet
        if self.plan_limit_version != f.limit_version:
            return True
        if f._cost is None or t - f._cost_t > 0.6:
            f.cost(t)                       # keep the shared cost grid fresh for everyone's checks
        b = f.blocked_ahead(t, self, f.slope_limit + self.plan_relax)
        if b is not None:
            self.block_info = b
            x, y, dist, why = b
            if self.reroute is None:
                old = path_length([(self.x, self.y)] + self.path[max(self.wp, 0):])
                if why == "rocks" and t - self.last_rocks_t > 8.0 and not any(
                        math.hypot(x - ax, y - ay) < 4.0 and t - at < 120.0 for ax, ay, at in self.rocks_seen):
                    self.rocks_seen = [r for r in self.rocks_seen if t - r[2] < 120.0] + [(x, y, t)]
                    self.reroute = ("rocks", x, y, dist, old, None)
                    i, j = f.map.ij(x, y)
                    cells = int((f.map.changed_t[max(j - 3, 0):j + 4, max(i - 3, 0):i + 4] >= t - 20.0).sum())
                    det = {"at": (x, y), "dist": round(dist, 1), "cells": cells}
                    shape = f.map.change_shape(x, y, t - 20.0)
                    if shape is not None:
                        det.update(shape)
                        if shape.get("shape") == "long":
                            self.reroute = ("long",) + self.reroute[1:]
                    self.on_anomaly(t, "new_obstacle", det, f"Slowing to {self.cfg.caution_v:.1f} m/s and replanning.")
                elif why == "soft":
                    z = f.soft_zone_at(x, y)
                    if z is not None and z[:2] not in self.soft_seen:
                        self.soft_seen.add(z[:2])
                        self.reroute = ("soft", x, y, dist, old, z[3])
            return True
        if t - self.plan_t > 2.5:
            return True
        return False

    def _after_replan(self, t: float, path) -> None:
        """Announce a re-route caused by something newly perceived on the path (once the new
        plan is ready), with the sensor reason and the extra distance."""
        ri = self.reroute
        if ri is None or path is PENDING:
            return
        self.reroute = None
        why, x, y, dist, old_len, owner = ri
        what = {"rocks": "the new rocks", "long": "the fallen obstacle"}.get(why, f"the soft ground {owner} reported")
        if path is None:
            msg = f"{self.id} finds no way past {what}. Picking another goal."
        else:
            extra = path_length([(self.x, self.y)] + list(path)) - old_len
            tail = f" (+{extra:.0f} m)" if extra >= 1.0 else ""
            if why in ("rocks", "long"):
                msg = f"{self.id} rerouting around {what}{tail}."
            else:
                msg = f"{self.id} rerouting around soft ground reported by {owner}{tail}."
        if why in ("rocks", "long"):
            self.last_rocks_t = t
        self.fleet.log(t, "info", msg, self.id, at=(x, y))

    def _hold_if_blocked(self) -> None:
        """The new plan is not ready yet: if the old path runs into something close, wait."""
        if self.block_info is not None and self.block_info[2] < 3.5:
            self.hold_for_plan = True

    def _set_path(self, t: float, path: list) -> None:
        self.path = list(path)
        self.wp = 1 if len(self.path) > 1 else 0
        self.plan_t = t
        self.plan_limit_version = self.fleet.limit_version
        self.plan_relax = self._last_relax
        self._last_relax = 0.0

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
        v_max = cfg.degraded_v_max if self.degraded else cfg.v_max
        w_max = 0.5 * cfg.w_max if self.degraded else cfg.w_max      # turn gently too when it cannot see far
        # Skid steer on a slope: a tight arc leaves the inner wheels with almost no speed target, so
        # only the outer pair pushes and the rover bogs down. On sloped ground make big corrections in
        # place and keep arcs wide enough that all four wheels drive forward.
        sloped = self.tilt > 7.0
        if abs(e) > (0.6 if sloped else 0.85):
            v = 0.0
            w = math.copysign(w_max, e)
        else:
            v = v_max * max(0.0, math.cos(e)) ** 2
            w = max(-w_max, min(w_max, 2.2 * e))
            if sloped:
                w = max(-0.9 * v, min(0.9 * v, w))
        dgoal = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        if self.wp >= len(self.path) - 1:
            v = min(v, 0.2 + 0.5 * dgoal)
        if abs(self.pitch) > 10.0:
            v *= 0.8
        if t < self.caution_until:
            v = min(v, self.caution_v)       # just saw something unexpected: go slow while re-planning
        if t < self.crawl_until and not (self.state == "returning" and self.return_reason in ("battery", "range")):
            v = min(v, self.crawl_v)         # a question to the operator is open: creep along the safe route
        if t < self.pause_until:
            v, w = 0.0, 0.0                  # waiting (Gemma or the operator said wait)
        if self.hold_for_plan:
            v, w = 0.0, 0.0                  # something blocks the path just ahead; wait for the new plan
        if self.state == "observing":
            v = min(v, cfg.observe_v)
        if v > 0.0 and self.fleet.tracker.tracks:
            # planetary protection: never drive at a creature the LiDAR tracks
            for tr in self.fleet.tracker.near(self.x, self.y, CREATURE_STOP_M, t):
                px, py = tr.predict(t)
                if abs(wrap(math.atan2(py - self.y, px - self.x) - self.yaw)) < 0.9:
                    v = 0.0
                    if t - self.last_backoff_t > 30.0:
                        self.last_backoff_t = t
                        self.fleet.log(t, "info", f"{self.id} stops: a moving creature {math.hypot(px - self.x, py - self.y):.1f} m "
                                                  f"ahead (LiDAR track {tr.id}). Waiting for it to move off.", self.id,
                                       at=(px, py))
                    if self.yield_since is None:
                        self.yield_since = t
                    elif t - self.yield_since > 4.0:
                        self.yield_since = None
                        self.plan_t = -1e9   # replan round it (tracks cost extra in the planner)
                    break
        intended_v = v
        # give way: lower id has priority, parked rovers are obstacles
        blocked_by = None
        for o in self.fleet.rovers:
            if o is self:
                continue
            dd = math.hypot(o.x - self.x, o.y - self.y)
            if dd < 2.2 and abs(wrap(math.atan2(o.y - self.y, o.x - self.x) - self.yaw)) < 0.8:
                moving = o.speed >= 0.05 and o.state not in PARKED
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
        T = cfg.motor_torque
        Tb = 3.0 * T
        if self.motor_fault:                 # controller offline: the parking brakes lock the wheels
            for a in self.acts:
                d.ctrl[a] = 0.0
                m.actuator_forcerange[a] = (-Tb, Tb)
                m.actuator_gainprm[a][0] = BRAKE_KV
                m.actuator_biasprm[a][2] = -BRAKE_KV
            self._brakes_locked = True
            return
        if self._brakes_locked:
            for a in self.acts:
                m.actuator_gainprm[a][0] = cfg.motor_kv
                m.actuator_biasprm[a][2] = -cfg.motor_kv
            self._brakes_locked = False
        wl = (self.cmd_v - self.cmd_w * cfg.track_eff) / r
        wr = (self.cmd_v + self.cmd_w * cfg.track_eff) / r
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
    def sensors_msg(self) -> dict:
        sp = None
        if self.spectro is not None:
            sp = {"id": self.fleet.targets[self.spectro[0]].spec.id, "dist": round(self.spectro[1], 1)}
        return {
            "lidar_range": round(float(self.lidar_eff), 1), "lidar_hits": int(self.lidar_hits),
            "pitch": round(self.pitch, 1), "roll": round(self.roll, 1),
            "wheel_speed": round(abs(self.v_wheel), 2), "ground_speed": round(self.speed, 2),
            "slip": round(self.slip, 2), "motor_load": round(self.motor_load, 2),
            "spectro": sp,
        }

    def scan_msg(self) -> list:
        pts = self.scan_pts
        n = len(pts)
        if n == 0:
            return []
        k = self.cfg.scan_points
        if n > k:
            pts = pts[np.linspace(0, n - 1, k).astype(np.int64)]
        return np.round(pts, 1).ravel().tolist()

    def to_msg(self) -> dict:
        pts = []
        if self.path and self.state not in ("charging", "idle", "fault"):
            rest = self.path[max(self.wp, 0):]
            pts = downsample([(self.x, self.y)] + rest, 30)
        msg = {
            "id": self.id,
            "x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2),
            "yaw": round(self.yaw, 3),
            "speed": round(self.speed, 2),
            "battery": round(self.battery, 3),
            "state": self.state,
            "goal": None if self.goal is None else [round(self.goal[0], 2), round(self.goal[1], 2)],
            "path": [[round(p[0], 2), round(p[1], 2)] for p in pts],
            "target_id": None if self.target_idx is None else self.fleet.targets[self.target_idx].spec.id,
            "sensors": self.sensors_msg(),
            "scan": self.scan_msg(),
            "brain": self.brain_msg(),
            "track_id": None if self.obs is None else self.obs["track"],
            "samples": [self.fleet.targets[k].spec.id for k in self.samples],
            "capacity": self.capacity,
        }
        if self.sampling is not None:
            sm = self.sampling
            msg["sampling"] = {"target": self.fleet.targets[sm["tidx"]].spec.id, "instrument": sm["instrument"],
                               "progress": round(min(1.0, max(0.0, (self.t_now - sm["t0"]) / sm["dur"])), 2)}
        return msg
