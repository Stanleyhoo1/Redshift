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
  camera         mast camera (camera.py), used by the rules for a quick colour check of a LiDAR hole, and
                 its onboard object detector (vision.py, protocol v8): self.vision holds the latest
                 detections; the decision cycle only reads them (water, creatures, new obstacles, targets)

States (protocol enum): exploring, investigating, returning, charging, recovering, idle, fault, observing, and
protocol v9: disabled (immobile, waiting for help), rescuing (a helper on its way or working), towing, towed,
repairing (at the lander) and wrecked (left behind). Health (health.py) changes how the rover drives through the
physics (a weak drive side has less torque and top speed), its sensors (LiDAR range and rays, camera noise), its
battery drain and its radio range; the rover adapts (lower top speed, its own lower slope limit, home for a repair
when its health is low) and backs away from a creature that charges at it (a fast LiDAR track coming straight at
it, confirmed by the camera when it sees one).

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
"explore north"), sticky_goal (a frontier Gemma picked), focus (protocol v7: an area to explore),
pause_until and crawl_until / crawl_v (wait or creep while a question is open). The safety rules below
still run first every cycle.

Protocol v7: a focus {"x", "y", "r", "reason", ...} makes the frontier choice (fleet.choose_goal) take the
unknown ground inside that circle first, then the ground nearest its edge; the rover stays until the time is
up (counted from when it reaches the circle) or nothing unknown is left nearby, then the advisor reports
what it mapped and found there. reason "operator" comes from a command ("explore around the lake"),
"interest" from the rover's own curiosity: when it finds something new (a science target, water, a
cluster of spires, a herd it finished watching, ground that suddenly looks different) it sticks around for
about a minute, with a cooldown, never while returning, on a low battery, during another focus or where
another rover's focus already covers the spot. While a lifeform question is open the rover follows the
herd at the observation standoff instead of holding still; hazards, water and a low battery end that.
"""
from __future__ import annotations

import math
import time
from collections import deque

import mujoco
import numpy as np

from .config import Config
from .health import Health
from .memory import RoverMemory
from .planner import PENDING, downsample, path_length
from .terrain import KIND_LABEL
from .vision import VisionBrain
from .world import GROUP_ROCK

LIDAR_GROUPS = np.array([1, 1, 1, 0, 0, 1], np.uint8)   # terrain, rocks, targets and creatures, liquid; never
                                                         # rovers (3) or the walls (4)
BUMPER_GROUPS = np.array([0, 1, 0, 0, 0, 0], np.uint8)  # rocks only
DRIVING = ("exploring", "investigating", "returning", "observing", "rescuing", "towing")
PARKED = ("idle", "charging", "fault", "disabled", "towed", "wrecked", "repairing")
HURT = ("disabled", "towed", "wrecked")    # protocol v9: out of action (rescue.py decides what happens to them)
CHARGE_SPEED = 0.65    # protocol v9: a LiDAR track closing this fast (m/s; grazers walk at 0.5 at most) ...
CHARGE_RANGE = 9.0     # ... from this close, heading straight at the rover, is a creature charging
EVADE_S = 2.5          # back away for this long
REPAIR_NOTE_S = 120.0  # a rover that needs a repair says so at most this often
ANOMALY_KINDS = ("new_obstacle", "slip", "low_visibility", "no_returns")
ANOMALY_TYPE = {"new_obstacle": "new_obstacle", "slip": "wheel_slip", "low_visibility": "low_visibility",
                "no_returns": "no_returns"}
CAUTION_S = {"new_obstacle": 3.0, "slip": 1.0, "low_visibility": 3.0, "no_returns": 4.0}   # cautious slowdown
FOLLOW_HAZARD = {"new_obstacle": "new rocks on my path", "slip": "soft ground (my wheels slipped)",
                 "low_visibility": "dust (my LiDAR range dropped)", "no_returns": "liquid ahead (no LiDAR returns)"}
HOLE_MIN_RAYS = 7          # missing steep returns straight ahead before a hole counts
HOLE_FRONT = math.radians(35.0)   # only straight ahead (inside the camera's view)
LIQUID_MIN_PCT = 3.0       # blue liquid in at least this share of the lower camera image
CREATURE_STOP_M = 3.5      # any rover stops if a tracked creature is this close ahead
VERB = {"drill": "drilled a core from", "scoop": "scooped ice from", "sniffer": "sniffed the gas at",
        "swab": "swabbed the trace at"}
BRAKE_KV = 400.0       # wheel brake stiffness while the motor controller is down (normal servo: motor_kv)
SURVEY_S = 30.0        # protocol v6: survey on arrival with "investigate" (sim s) ...
SURVEY_W = 0.55        # ... turn command in place (rad/s), corrected so a full circle takes about 26 s
FOCUS_CHECK_S = 2.0    # protocol v7: how often a focused rover counts the unknown ground left in its circle ...
FOCUS_MIN_UNKNOWN = 12  # ... and fewer unexplored samples than this (3 m2) means the area is done
INTEREST_DONE_SHARE = 0.08  # curiosity: done when less than this share of its circle is still unknown ...
INTEREST_START_SHARE = 0.2  # ... and only starts when at least this share is unknown
FOCUS_STALL_S = {"interest": 15.0, "operator": 30.0}   # nothing new mapped for this long: nothing more to find
INTEREST_SPOT_M = 15.0  # curiosity: not again within this distance of an earlier interest spot ...
INTEREST_SPOT_S = 600.0  # ... for this long (sim s)
INTEREST_STALE_S = 120.0  # a find not acted on within this long (busy sampling, say) is let go
SPIRE_CHECK_S = 2.0    # how often a rover looks for spires its LiDAR newly mapped
SPIRE_CLUSTER_M = 10.0  # spires this close together are a cluster (2 or more)
GROUND_MIN_CELLS = 40  # a sweep must map this many new samples to compare the ground ...
GROUND_WARMUP = 25     # ... after this many such sweeps
GROUND_ALPHA = 0.08    # running mean of the median slope and height of newly mapped ground
GROUND_SLOPE_DEG = 8.0  # "different ground": median slope this much above the running mean (and 1.8 times it) ...
GROUND_HEIGHT_M = 2.5  # ... or median height this far from it
RESURVEY_SHARE = 0.15  # protocol v8: an operator focus on an area with less unknown than this share is a re-survey
RESURVEY_EARLY_S = 10.0  # ... and so is one whose area turns out to be explored this soon after arriving
RESURVEY_RING = (0.6, 0.8, 0.4)  # loop waypoints at these fractions of the circle's radius (first safe one wins)
RESURVEY_POINTS = 8    # waypoints around the loop


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _camera_frame(sim, rid: str):
    """A small mast camera picture (camera hardware: rendered in this thread with its own GL context)."""
    from . import camera
    return camera._local_renderer().frame(None, sim, rid, "fpv", 160, 120, depth=False, hide_groups=(3, 4),
                                          smooth=False, look=False)


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
        self.op_task = None                # protocol v6: the place behind op_goal {"label", "investigate", "cid", ...}
        self.survey = None                 # protocol v6: surveying on arrival (stop, turn 360 deg, LiDAR tracker on)
        self.op_fails = 0
        self.dir_bias = None               # operator "explore <dir>": (unit x, unit y, until t)
        self.sticky_goal = None            # frontier picked by Gemma: kept until reached or gone
        self.goal_options: list = []       # ranked frontier options from the last choose_goal
        self.pause_until = -1e9            # stand still (wait) until then
        self.pause_dust = False            # the wait ends early when visibility comes back
        self.crawl_until = -1e9            # creep at crawl_v until then (a question is open)
        self.crawl_v = 0.25
        self.question_hold = False         # a question to the operator is open: stay put (safety moves allowed)
        self.focus = None                  # protocol v7: the area this rover concentrates on (see start_focus)
        self.pending_interest = None       # curiosity: something new found, the focus starts at the next goal choice
        self.interest_next = 0.0           # no interest focus before then (cooldown)
        self._ground = None                # [median slope, median height, sweeps] of newly mapped ground
        self._next_spire_check = 0.0

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

        # mission memory: what it mapped, found, sampled and met (answers questions, protocol v5 stats)
        self.mem = RoverMemory(self)
        self.health = Health(self)         # protocol v9: parts, status, self diagnosis
        self._health_v = 0                 # health version last acted on (replan on a change)
        self._drive_sig = (1.0, 1.0, None)  # drive factors and own slope limit last planned with
        self.carried_spares: list = []     # salvaged parts on board: (spare kind, condition, from rover)
        self.evade_until = -1e9            # backing away from a charging creature until then
        self.evade_from = None
        self.charge_note_t = -1e9
        self.repair_go_t = -1e9
        self.repair_t0 = 0.0
        self.repair_note = None
        self._drive_mode = None            # last actuator setup written by _apply (only rewritten on a change)
        self._count_replan = False
        self.vis_lost_t = None
        self.vision = VisionBrain(self)    # protocol v8: onboard object detection (results from the camera thread)

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
        if self.state in DRIVING:                       # protocol v9: motor strain (heat) per side
            T = cfg.motor_torque
            self.health.strain(t, 1.0 / cfg.agent_hz, (float(tau[:2].mean()) / T, float(tau[2:].mean()) / T),
                               self.cmd_v > 0.1 or abs(self.cmd_w) > 0.3)

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
        he = self.health.effects()
        if he["lidar_range"] < 1.0:                     # protocol v9: a damaged LiDAR sees less far
            rng_max = min(rng_max, cfg.lidar_range * he["lidar_range"])
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
        if he["lidar_drop"] > 0.0:         # protocol v9: a damaged LiDAR loses rays
            hit &= self.rng.uniform(size=self.nray) >= he["lidar_drop"]
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
            fm = self.fleet.map
            k0 = fm.known_count
            self.scan_changes = fm.integrate(pts, t, origin=(float(pnt[0]), float(pnt[1])))
            if fm.known_count > k0:
                self.mem.on_mapped(fm, fm.last_new, fm.known_count - k0)
                if self.cfg.curiosity:
                    self._ground_novelty(t, fm, fm.last_new)
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
        cam = self.vision.water_near(at, t)          # protocol v8: the onboard detector saw water there
        pct = None if cam is not None else self.camera_liquid_pct()
        details = {"at": at, "dist": round(d, 1), "rays": int(rays)}
        if cam is not None:
            details["camera_water"] = cam["conf"]
        if pct is not None:
            details["camera_liquid_pct"] = pct
        if spread is not None:
            details["edge_spread"] = round(spread, 2)
        liquid = cam is not None or (pct is not None and pct >= LIQUID_MIN_PCT)
        flat_guess = pct is None and spread is not None and spread < 0.08
        f = self.fleet
        if liquid or flat_guess:
            self.on_anomaly(t, "no_returns", details, f"Slowing to {self.cfg.caution_v:.1f} m/s to look.")
            zx, zy, zr = self._fit_zone(pts)
            new_body = f.water_zone_gap(zx, zy) > zr + 3.0      # not just more of a lake the fleet knows
            f.add_water_zone(t, self, zx, zy, zr)
            if new_body:
                self.note_interest(t, zx, zy, "water", "water", "the water")
            if cam is not None:
                msg = (f"{self.id} Camera: water ahead ({cam['conf']:.2f}) where LiDAR gets no returns {d:.0f} m "
                       f"ahead. Confirmed liquid: marking a lake and routing around it.")
            elif liquid:
                msg = (f"{self.id} LiDAR gets no returns {d:.0f} m ahead and the camera shows a liquid surface "
                       f"({pct:.0f}% of the lower view is blue). Marking a lake and routing around it.")
            else:
                msg = (f"{self.id} LiDAR gets no returns {d:.0f} m ahead and the hole has a flat, level edge "
                       f"(spread {spread:.2f} m; no camera here). Probably liquid: marking it and routing around it.")
            f.log(t, "discovery", msg, self.id, at=(zx, zy))
            self.mem.on_water(t, zx, zy, zr, "camera" if liquid else "flat edge")
            self._replan_now(t)
            self.plan_t = -1e9
        else:                              # a drop-off: just slow down and keep to mapped ground
            self.caution_until = t + CAUTION_S["no_returns"]
            self.caution_v = self.cfg.caution_v
            f.dropoffs.append((at[0], at[1], t))
            self.mem.on_hazard("dropoff", t, at, f"a drop-off {d:.0f} m ahead where my LiDAR got no returns")
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
        self.mem.on_hazard("wet", t, (self.x, self.y))
        # protocol v9: water in the chassis shorts the power system and gets into the wheel motors (once a minute:
        # backing out it may dip in again)
        if depth >= 0.05 and t - getattr(self, "_wet_hit_t", -1e9) > 60.0:   # a dunk, not a wet belly at the shore
            self._wet_hit_t = t
            sev = min(20.0, 6.0 + 30.0 * depth)
            self.health.hit(t, [("power", 1.0), ("drive_l", 0.35), ("drive_r", 0.35)], sev, "water",
                            kinds={"drive_l": "broken", "drive_r": "broken"})
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
        if len(self.far_hist) < self.far_hist.maxlen or self.state in HURT:
            return
        far_max, far_min = max(self.far_hist), min(self.far_hist)
        nominal = cfg.lidar_range * self.health.effects()["lidar_range"]   # its own range (damage, protocol v9)
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
            if self.vis_lost_t is not None:
                self.mem.note_outcome("dust", f"visibility came back after {t - self.vis_lost_t:.0f} s", t, 600.0)
                self.vis_lost_t = None
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
        self._remember_anomaly(t, kind, details, at)
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
        if self.obs is not None and self.obs.get("asking") is not None:
            # following a herd while the operator decides (protocol v7): safety wins, the follow ends here
            self.obs["hazard"] = FOLLOW_HAZARD[kind]
            self.end_observing(t, "hazard")
        return a

    def _remember_anomaly(self, t: float, kind: str, d: dict, at) -> None:
        """Mission memory: what the anomaly was, in the rover's own words, with its numbers."""
        if kind == "new_obstacle" and d.get("shape") == "long":
            self.mem.on_hazard("spire", t, at, f"a long, low obstacle across my path {d.get('dist', 0.0):.0f} m ahead "
                                               f"(about {d.get('length', 0.0):.1f} m long, a fallen spire)")
        elif kind == "new_obstacle":
            self.mem.on_hazard("rocks", t, at, f"new rocks across my path {d.get('dist', 0.0):.0f} m ahead "
                                               f"({d.get('cells', 0)} map cells changed)")
        elif kind == "slip":
            self.mem.on_hazard("sand", t, at, f"soft ground: wheels spinning at {d.get('slip', 0.0) * 100:.0f}% slip")
        elif kind == "low_visibility":
            self.vis_lost_t = t
            self.mem.on_hazard("dust", t, at, f"dust: LiDAR range down to {d.get('lidar_range', 0.0):.0f} m")
        elif kind == "no_returns":
            pct = d.get("camera_liquid_pct")
            seen = (f"the camera saw liquid ({pct:.0f}% of the lower view blue)" if pct is not None
                    else "the hole had a flat, level edge")
            if d.get("camera_water") is not None:
                seen = f"the camera's detector saw water ({d['camera_water']:.2f})"
            self.mem.on_hazard("water", t, at, f"no LiDAR returns {d.get('dist', 0.0):.0f} m ahead and {seen}")

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
        e = self.health.effects()
        v = 0.55 * cfg.v_max * min(1.0, e["v_cap"] + 0.1)
        return dist * (1.25 * self.e_per_m + e["drain"] * cfg.battery_base_drain / cfg.battery_capacity / v)

    def _battery(self, t: float, dt: float, energy_j: float) -> None:
        cfg = self.cfg
        parked = self.state == "idle" and math.hypot(self.x - self.dock[0], self.y - self.dock[1]) < 1.6
        if self.state == "repairing":
            return                                      # rescue.repair_step charges it
        if self.state == "charging" or parked:
            power = self.sim.hazards.charge_factor() if self.sim.hazards is not None else 1.0
            self.battery = min(1.0, self.battery + power * dt / cfg.charge_time_s)
            return
        cap = cfg.battery_capacity
        drain = self.health.effects()["drain"]          # protocol v9: a damaged power system drains faster
        motor = cfg.battery_motor_k * energy_j / cap * (1.0 + 0.4 * (drain - 1.0))
        base = cfg.battery_base_drain / cap * dt * drain
        if self.state in HURT:
            base *= 0.3                                 # out of action: hibernating, only the heaters and radio
        self.battery = max(0.0, self.battery - base - motor)
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
        self.vision.tick(t)                # decisions from the latest camera detections (protocol v8)
        self.hold_for_plan = False
        self.block_info = None
        if self._health_tick(t, dt):       # protocol v9: out of action, or being repaired at the lander
            self._apply()
            return

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
        if self.survey is not None and (self.state != "exploring" or self.held or self.op_goal is not None):
            self.end_survey(t, self.state if self.state != "exploring" else "new command")
        if self.op_task is not None and self.op_goal is None:
            self.op_task = None                # the operator trip ended some other way (home, hold, resume)
        if self.focus is not None:
            self._focus_step(t)
        if self.cfg.curiosity and t >= self._next_spire_check:
            self._next_spire_check = t + SPIRE_CHECK_S
            self._check_spires(t)
        if self.state in DRIVING and self._check_immersion(t):
            self._apply()
            return

        if self.state == "idle":
            self._set_cmd(0.0, 0.0)
            if math.hypot(self.x - self.dock[0], self.y - self.dock[1]) > 1.6 and self._evade(t):
                self._apply()                  # protocol v9: even a parked rover gets out of a charging creature's way
                return
            if self.held:
                away = math.hypot(self.x - self.dock[0], self.y - self.dock[1]) > 1.6
                if away and self.battery > 0.0 and self._need_return(t):
                    # safety first: a long hold must not strand the rover
                    self.held = False
                    self._say(t, f"{self.id}: battery {self.battery * 100:.0f}%, leaving the hold to return "
                                 f"to the lander.", "info")
                    self.state = "exploring"
                    self._go_home(t, "battery")
                elif not away and self.battery < 0.9:
                    self.state = "charging"    # held on the pad: keep charging (still held once full)
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

        if self.state == "rescuing" and self._need_return(t):
            self._say(t, f"{self.id}: battery {self.battery * 100:.0f}%, I have to leave the rescue and charge.")
            self._go_home(t, "battery")
        if self._evade(t):                 # protocol v9: a creature charging at us
            self._apply()
            return
        self._track(t)
        if self.state == "towing":         # the rescue watches a tow's progress (no reversing with a load)
            self.sim.rescue.drive(self, t) or self._drive(t)
            self._apply()
            return
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
            if self.survey is not None:
                if self._survey(t):
                    self._apply()
                    return
            else:
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
        elif self.state in ("rescuing", "towing"):
            if self.sim.rescue.drive(self, t):       # protocol v9: the rescue sets the motors itself
                self._apply()
                return
        self._drive(t)
        self._apply()

    # ================================================================== protocol v9: health
    def _health_tick(self, t: float, dt: float) -> bool:
        """Health bookkeeping once per cycle (cheap). Returns True when the rover is out of action or being
        repaired, i.e. the rest of the decision cycle is skipped."""
        h = self.health
        hz = self.sim.hazards
        h.dust(t, dt, hz is not None and hz.storm_active() and self.state not in ("charging", "repairing"))
        if h.version != self._health_v:
            self._health_v = h.version
            e = h.effects()
            sig = (e["drive_l"], e["drive_r"], e["slope_cap"])
            if sig != self._drive_sig:
                self._drive_sig = sig
                self._replan_now(t)        # a weaker drive means another slope limit and speed
                self.plan_t = -1e9
            if not self.degraded:
                self.lidar_eff = round(self.cfg.lidar_range * h.effects()["lidar_range"], 1)
            if not h.mobile() and self.state not in HURT + ("repairing",):
                self.sim.rescue.on_immobile(self, t)
        st = self.state
        if st in HURT:
            self._set_cmd(0.0, 0.0)
            self.goal, self.path = None, []
            return True
        if st == "repairing":
            self.sim.rescue.repair_step(self, t, dt)
            return self.state == "repairing"
        if (st in ("exploring", "investigating", "observing") and h.needs_repair() and t - self.repair_go_t > 30.0
                and not self.held):
            self.repair_go_t = t
            if self.obs is not None:
                self.end_observing(t, "interrupted", quiet=True)
            self.state = "exploring" if st == "observing" else st
            self._go_home(t, "repair")
        return False

    def release_all(self, t: float, why: str, quiet: bool = False) -> None:
        """Protocol v9: drop every task and claim (frontier, target, focus, survey, observation, operator trip, open
        questions) so the fleet can take them over. why: "immobile" (out of action) or "rescue" (off to help)."""
        if self.survey is not None:
            self.end_survey(t, why)
        if self.focus is not None:
            self.end_focus(t, "battery" if why == "immobile" else "new command", quiet=True)
        if self.obs is not None:
            self.end_observing(t, "interrupted", quiet=True)
        if why == "immobile":
            self.fleet.rover_faulted(t, self)  # its frontier goes to the others (takeover note)
        self._drop_target(t, None, retry_s=0.0)
        self.op_goal = None
        self.op_task = None
        self.sticky_goal = None
        self.dir_bias = None
        self.pending_interest = None
        self.goal, self.path = None, []
        self.searches.clear()
        self.pending_goal_key = None
        self.rec_phase = None
        self.dwell_until = None
        self.pause_until = min(self.pause_until, t)
        self.crawl_until = min(self.crawl_until, t)
        self.vision.cam_goal = None
        if why == "immobile":
            self.held = False
            if self.advisor is not None:
                self.advisor.on_out_of_action(self, t)

    def limit_now(self) -> float:
        """The slope limit this rover plans with: the fleet's, its own (a damaged drive climbs less) and a
        stricter one while towing (protocol v9)."""
        lim = self.fleet.slope_limit
        cap = self.health.effects()["slope_cap"]
        if cap is not None:
            lim = min(lim, cap)
        if self.state == "towing":
            from .rescue import TOW_SLOPE_MARGIN
            lim = min(lim, self.fleet.slope_limit - TOW_SLOPE_MARGIN)
        return lim

    def _evade(self, t: float) -> bool:
        """A creature charging: a LiDAR track closing fast and heading straight at us (the camera confirms it when
        it sees one). Back away from it for a moment if the ground behind is safe. True while doing that."""
        if self.state == "recovering" or (self.state not in DRIVING and self.state != "idle"):
            return False
        tr_ = self.fleet.tracker
        if tr_.tracks and t >= self.evade_until - EVADE_S + 0.6:
            for tr in tr_.near(self.x, self.y, CHARGE_RANGE, t):
                sp = tr.speed
                if sp < CHARGE_SPEED:
                    continue
                px, py = tr.predict(t)
                dx, dy = self.x - px, self.y - py
                d = math.hypot(dx, dy)
                if d < 0.3:
                    continue
                closing = (tr.vx * dx + tr.vy * dy) / d
                if closing < CHARGE_SPEED or closing < 0.85 * sp or d / closing > 4.0 or d > 6.5:
                    continue                # not at us, or not soon (time to contact over 4 s)
                self.evade_until = t + EVADE_S
                self.evade_from = (px, py)
                if t - self.charge_note_t > 8.0:
                    self.charge_note_t = t
                    cam = ""
                    res = self.vision.fresh(t)
                    if res is not None:
                        seen = [o for o in res.objects if o["label"] == "creature"
                                and math.hypot(o["x"] - px, o["y"] - py) < 2.5 + sp * 1.0]
                        if seen:
                            cam = f" Camera confirms a creature ({max(o['conf'] for o in seen):.2f})."
                    self.fleet.log(t, "warn", f"{self.id} LiDAR track {tr.id} is coming straight at me ({closing:.1f} m/s, "
                                              f"{d:.1f} m away).{cam} Backing away.", self.id, at=(px, py))
                    self.mem.on_hazard("charge", t, (px, py), f"a creature charging at me ({closing:.1f} m/s)",
                                       handled="backed away")
                break
        if t >= self.evade_until or self.evade_from is None:
            return False
        px, py = self.evade_from
        bearing = wrap(math.atan2(py - self.y, px - self.x) - self.yaw)
        v = -0.45 if abs(bearing) < math.pi / 2 else 0.5
        sx = self.x + (1.2 if v > 0 else -1.2) * math.cos(self.yaw)
        sy = self.y + (1.2 if v > 0 else -1.2) * math.sin(self.yaw)
        if not self._safe_cell(sx, sy) or self.fleet.water_zone_at(sx, sy, margin=0.5) is not None:
            v = 0.0                         # nowhere safe to go: brace
        self._set_cmd(v, 0.0)
        return True

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
        if reason in ("battery", "range", "repair"):
            self.pause_until = min(self.pause_until, t)     # no waiting around when the battery says go home
            self.crawl_until = min(self.crawl_until, t)
            self.pending_interest = None
        self._drop_target(t, "battery low" if reason in ("battery", "range") else None)
        self.state = "returning"
        self.return_reason = reason
        if reason in ("battery", "range", "repair"):
            # protocol v7: waiting for the operator never outlasts the battery; a survey, a focus and any open
            # question end here, with a chat line saying why (protocol v9: nor a repair)
            if self.survey is not None:
                self.end_survey(t, "returning")
            if self.focus is not None:
                self.end_focus(t, "battery")
            if self.advisor is not None:
                self.advisor.on_battery_return(self, t)
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
        elif reason == "repair":
            self.fleet.log(t, "warn", f"{self.id} health {self.health.hp()} ({self.health.short()}). Heading home for "
                                      f"a repair at the lander.", self.id, at=here)
            self._say(t, f"{self.id}: health {self.health.hp()} ({self.health.short()}), heading home for a repair.")

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
            self.sim.rescue.deliver(self, t)             # protocol v9: salvaged parts become fleet spares
            if self.return_reason == "repair" or self.health.status() != "ok":
                self.state = "repairing"                 # protocol v9: repaired at the lander (charges too)
                self.repair_t0 = t
                self.repair_note = None
                self.fleet.log(t, "info", f"{self.id} docked for a repair (health {self.health.hp()}: "
                                          f"{self.health.short()}).", self.id, at=self.dock)
                return True
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
        if self.vision.drive_check(t):     # protocol v8: checking something the camera saw beyond the spectrometer
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
            if not self.degraded and f.priority is not None and f.priority["kind"] == "sampling":
                reach = 90.0                           # protocol v8: "prioritize sampling"
            fc = self.focus
            if k is not None and fc is not None and fc.get("mode") == "resurvey" and math.hypot(
                    f.targets[k].spec.x - fc["x"], f.targets[k].spec.y - fc["y"]) > fc["r"] + self.cfg.focus_margin:
                k = None                               # a re-survey stops only for what is inside its circle
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
        if self.focus is not None and self.focus.get("mode") == "resurvey":
            self._resurvey_drive(t)           # protocol v8: a loop through the circle, rescanning
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
            if self.pending_interest is not None and self.focus is None:
                self._start_interest(t)       # curiosity: something new was found, look around it first
            res = f.choose_goal(t, self)
            if res is PENDING:
                self._hold_if_blocked()       # out of compute: keep the current plan, finish next cycle
                return
            if res is None and self.focus is not None:
                if self.goal_fail_reason == "battery":
                    why = "battery"
                else:                         # nothing unknown left nearby, or what is left cannot be reached
                    fc = self.focus
                    _kn, unk = self.fleet.map.area_cells(fc["x"], fc["y"], fc["r"] + self.cfg.focus_margin)
                    why = "explored" if unk < FOCUS_MIN_UNKNOWN else "blocked"
                self.end_focus(t, why)
                if why in ("explored", "blocked"):
                    self.goal, self.path = None, []
                    self.next_goal_try = t
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
            if self.focus is not None:
                self.mem.on_goal(t, self.focus["reason"], f"unknown ground around {self.focus['label']} ({size} cells "
                                                          f"at its edge)")
            else:
                self.mem.on_goal(t, "rules", f"best frontier for size and distance ({size} cells of unknown ground "
                                             f"at its edge)")
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
            task, self.op_task = self.op_task, None
            if task is not None:               # a place the operator named (protocol v6)
                if task.get("investigate"):
                    self.start_survey(t, task)     # protocol v7: the area focus follows the survey
                elif task.get("focus") is not None:
                    self._task_focus(t, task)      # protocol v7: "explore around the lake"
                if self.advisor is not None:
                    self.advisor.on_place_arrived(self, task, t)
                return
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
                self.op_task = None
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

    def on_rescue(self) -> bool:
        """True while this rover is the helper of a rescue: the stranded rover comes first, no side trips."""
        rm = getattr(self.fleet, "rescue", None)
        return rm is not None and rm.by_helper(self.id) is not None

    def can_investigate(self, t: float, ts) -> bool:
        if self.state != "exploring" or self.target_idx is not None or self.on_rescue():
            return False
        if self.tube_verdict(ts) is not None:
            return False
        d = math.hypot(ts.spec.x - self.x, ts.spec.y - self.y)
        if self.degraded and d > 8.0:
            return False
        fc = self.focus
        if fc is not None and math.hypot(ts.spec.x - fc["x"], ts.spec.y - fc["y"]) > fc["r"] + self.cfg.spectro_range:
            return False                   # exploring an area: finds well outside it wait (logged for later)
        need = self.energy_for_distance(1.3 * d + 1.3 * math.hypot(ts.spec.x, ts.spec.y)) + self.battery_margin + 0.05
        return self.battery > need

    def instrument_time(self, inst: str) -> float:
        cfg = self.cfg
        return {"drill": cfg.drill_s, "scoop": cfg.scoop_s, "sniffer": cfg.sniffer_s,
                "swab": cfg.swab_s}.get(inst, cfg.investigate_dwell_s)

    def start_investigation(self, t: float, tidx: int) -> None:
        if self.on_rescue():
            return                         # on a rescue: no side trips (the target stays for later)
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
            self.mem.note_outcome("sand", f"backed out along my tracks in {t - self.sand_t0:.0f} s", t)
            return
        if t - self.sand_t0 > cfg.sand_free_s and not self.sand_freed:
            self.sand_freed = True
            if self.sim.hazards is not None:
                self.sim.hazards.free_rover(self.id, t)
            self.fleet.log(t, "warn", f"{self.id} dug itself out.", self.id, at=(self.x, self.y))
            self.mem.note_outcome("sand", f"dug myself out after {t - self.sand_t0:.0f} s", t)
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
        self.fault_t0 = t
        self.mem.on_hazard("fault", t, (self.x, self.y), handled="locked the brakes and " + (
            f"released {tid} to the fleet" if tid else "handed my frontier to the fleet" if had_frontier
            else "told the fleet I was parked"))

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
        self.mem.note_outcome("fault", f"rebooted after {t - getattr(self, 'fault_t0', t):.0f} s", t, 600.0)

    # ================================================================== surveying a place (protocol v6)
    def start_survey(self, t: float, task: dict) -> None:
        """Arrived at a place the operator asked to investigate: stop and turn slowly through 360 deg (so the
        camera sweeps the area) for about SURVEY_S while the LiDAR tracker watches for movement. State stays
        "exploring"; goal and path are empty. Ends in end_survey (the advisor writes the report)."""
        self.survey = {"t0": t, "last_yaw": self.yaw, "turned": 0.0, "label": task.get("label", "the place"),
                       "cid": task.get("cid"), "qid": task.get("qid"), "at": task.get("at"), "tracks": {},
                       "targets": {}, "mapped0": self.mem.mapped_cells, "observing": None,
                       "task": task if task.get("focus") is not None else None}
        self.goal, self.path = None, []
        self.sticky_goal = None
        self.fleet.log(t, "info", f"{self.id} surveying {self.survey['label']} for the operator: turning slowly "
                                  f"through 360 deg for about {SURVEY_S:.0f} s.", self.id, at=(self.x, self.y),
                       source="operator")

    def _survey(self, t: float) -> bool:
        """One survey cycle (a few microseconds: the tracks near the rover and the spectrometer reading).
        Returns True if it set the motor command."""
        from .fleet import compass
        sv = self.survey
        for tr in self.fleet.tracker.near(self.x, self.y, self.lidar_eff + 4.0, t):
            if t - tr.last_seen > 1.0:
                continue
            px, py = tr.predict(t)
            d = math.hypot(px - self.x, py - self.y)
            e = sv["tracks"].get(tr.id)
            if e is None:
                e = sv["tracks"][tr.id] = {"id": tr.id, "dist": d, "dir": compass(px - self.x, py - self.y),
                                           "speed": tr.speed, "size": tr.size, "label": tr.label}
            else:
                if d < e["dist"]:
                    e["dist"], e["dir"] = d, compass(px - self.x, py - self.y)
                e["speed"], e["size"], e["label"] = max(e["speed"], tr.speed), tr.size, tr.label
            # something moves on its own: the normal observation flow (Gemma's look, maybe the operator)
            if tr.observer is None and not tr.ignored and tr.speed >= 0.08 and self.can_observe(t, tr):
                sv["observing"] = tr.id
                self.end_survey(t, "observing")
                self.start_observing(t, tr)
                return True
        if self.spectro is not None:
            k, dk = self.spectro
            sv["targets"][k] = min(dk, sv["targets"].get(k, math.inf))
        sv["turned"] += abs(wrap(self.yaw - sv["last_yaw"]))
        sv["last_yaw"] = self.yaw
        el = t - sv["t0"]
        if (el >= SURVEY_S and sv["turned"] >= 2.0 * math.pi - 0.15) or el >= SURVEY_S + 15.0:
            self.end_survey(t, "done")
            self._set_cmd(0.0, 0.0)
            return True
        hold = (self.question_hold and not self._may_move_while_asking(t)) or t < self.pause_until
        self.goal, self.path = None, []
        # closed loop on the IMU heading: skid steer turns in place slower than commanded at low wheel speed
        plan = min(2.0 * math.pi, 2.0 * math.pi * el / (SURVEY_S - 4.0))
        w = 0.0 if sv["turned"] >= 2.0 * math.pi else max(0.3, min(1.0, SURVEY_W + 1.5 * (plan - sv["turned"])))
        self._set_cmd(0.0, 0.0 if hold else w)
        return True

    def end_survey(self, t: float, why: str) -> None:
        """why: "done", "observing", "new command", or the state that interrupted it (returning, idle, ...)."""
        sv, self.survey = self.survey, None
        if sv is None:
            return
        self.next_goal_try = self.next_goal_eval = t
        if why == "observing" and self.obs is not None:
            sv["observing"] = sv["observing"] or self.obs["track"]
            tr = self.fleet.tracker.get(sv["observing"])
            if tr is not None and tr.id not in sv["tracks"]:
                from .fleet import compass
                px, py = tr.predict(t)
                sv["tracks"][tr.id] = {"id": tr.id, "dist": math.hypot(px - self.x, py - self.y),
                                       "dir": compass(px - self.x, py - self.y), "speed": tr.speed, "size": tr.size,
                                       "label": tr.label}
        then = sv.get("task") is not None and why in ("done", "observing")
        sv["then_focus"] = then
        if self.advisor is not None:
            self.advisor.survey_report(self, sv, t, why)
        if then:
            self._task_focus(t, sv["task"])        # protocol v7: "investigate X": survey first, then the focus

    # ================================================================== exploring an area (protocol v7 focus)
    def start_focus(self, t: float, x: float, y: float, r: float, reason: str, label: str, dur: float,
                    cid=None, qid=None, what: str | None = None, kind: str | None = None,
                    mode: str | None = None) -> dict:
        """Concentrate on the unknown ground within r of (x, y) for about dur sim seconds once there (reason
        "operator" or "interest"). The frontier choice does the rest (fleet.choose_goal); _focus_step ends it.
        mode (protocol v8): "explore", or "resurvey" (a loop through the circle for the whole time, rescanning);
        None: an operator focus on an area that is already mostly mapped is a re-survey."""
        if self.focus is not None:
            self.end_focus(t, "replaced", quiet=self.focus["reason"] == "interest")
        if mode is None:
            mode = "resurvey" if reason == "operator" and self.mostly_mapped(x, y, r) else "explore"
        fc = {"x": float(x), "y": float(y), "r": float(r), "reason": reason, "label": label, "dur": float(dur),
              "t0": t, "arrived": None, "until": None, "cid": cid, "qid": qid, "what": what, "kind": kind,
              "mapped0": self.mem.mapped_cells, "next_check": t + FOCUS_CHECK_S,
              "gain_t": t, "gain_cells": self.mem.mapped_cells, "mode": mode, "loop": None, "wp": 0, "laps": 0,
              "odo0": None, "found0": None}
        self.focus = fc
        self._focus_arrival(t)
        self.sticky_goal = None
        if self.state == "exploring" and self.op_goal is None and self.survey is None:
            self.goal, self.path = None, []        # pick a frontier in the circle now
            self.next_goal_try = self.next_goal_eval = t
        return fc

    def mostly_mapped(self, x: float, y: float, r: float) -> bool:
        """Is the circle already mapped but for a little unknown ground (protocol v8 re-survey)?"""
        kn, unk = self.fleet.map.area_cells(x, y, r)
        return unk < max(4 * FOCUS_MIN_UNKNOWN, RESURVEY_SHARE * (kn + unk))

    def _task_focus(self, t: float, task: dict) -> None:
        fs = task["focus"]
        self.start_focus(t, fs["x"], fs["y"], fs["r"], "operator", fs["label"], self.cfg.focus_s,
                         cid=task.get("cid"), qid=task.get("qid"))

    def _focus_arrival(self, t: float) -> None:
        fc = self.focus
        if fc["arrived"] is None and math.hypot(self.x - fc["x"], self.y - fc["y"]) <= fc["r"] + self.cfg.focus_margin:
            fc["arrived"] = t                    # the clock starts when the rover is there
            fc["until"] = t + fc["dur"]
            fc["mapped0"] = self.mem.mapped_cells
            fc["odo0"] = self.odometer
            fc["found0"] = self._found_in(fc)
            near = fc["r"] + self.cfg.focus_margin
            if self.goal is not None and math.hypot(self.goal[0] - fc["x"], self.goal[1] - fc["y"]) > near:
                self.next_goal_eval = t          # a goal picked on the way in: choose one inside now

    def _focus_step(self, t: float) -> None:
        """Every cycle while focused (microseconds; the unknown ground is counted every FOCUS_CHECK_S)."""
        fc = self.focus
        self._focus_arrival(t)
        if self.held or self.state in ("returning", "charging", "idle"):
            self.end_focus(t, "held" if self.held else self.state)
        elif fc["until"] is not None and t >= fc["until"]:
            self.end_focus(t, "time")
        elif fc["arrived"] is None and t - fc["t0"] > self.cfg.focus_travel_s:
            self.end_focus(t, "unreached")
        elif fc["reason"] == "interest" and self.op_goal is not None:
            self.end_focus(t, "new command", quiet=True)
        elif fc["arrived"] is not None and t >= fc["next_check"] and fc.get("mode") != "resurvey":
            fc["next_check"] = t + FOCUS_CHECK_S
            if self.mem.mapped_cells > fc["gain_cells"]:
                fc["gain_t"], fc["gain_cells"] = t, self.mem.mapped_cells
            kn, unk = self.fleet.map.area_cells(fc["x"], fc["y"], fc["r"] + self.cfg.focus_margin)
            done = FOCUS_MIN_UNKNOWN if fc["reason"] == "operator" else max(FOCUS_MIN_UNKNOWN,
                                                                             INTEREST_DONE_SHARE * (kn + unk))
            stalled = self.state == "exploring" and t - fc["gain_t"] > FOCUS_STALL_S.get(fc["reason"], 30.0)
            if unk < done and fc["reason"] == "operator" and t - fc["arrived"] < RESURVEY_EARLY_S:
                self._to_resurvey(t)             # protocol v8: nothing left to explore there: re-survey it
            elif unk < done or stalled:
                self.end_focus(t, "explored" if unk < done else "blocked")

    def end_focus(self, t: float, why: str, quiet: bool = False) -> None:
        """why: "time", "explored" (nothing unknown left nearby), "blocked" (unknown ground left that the rover
        cannot reach or stopped gaining), "battery", "unreached", "replaced", "new
        command", "held" or the state that ended it. The advisor reports what was mapped and found there."""
        fc, self.focus = self.focus, None
        if fc is None:
            return
        if fc["reason"] == "interest":
            self.interest_next = t + self.cfg.interest_cooldown_s
        summary = self.focus_summary(fc, t)
        summary["why"] = why
        self.mem.on_focus(summary)
        self.next_goal_eval = t
        if self.advisor is not None:
            self.advisor.focus_report(self, fc, summary, t, why, quiet)
        elif not quiet:
            self.fleet.log(t, "info", f"{self.id} done exploring around {fc['label']} ({summary['dur']:.0f} s).",
                           self.id, at=(fc["x"], fc["y"]))

    def focus_summary(self, fc: dict, t: float) -> dict:
        """What this rover mapped and found while focused, from its own memory (grounded numbers only)."""
        t0 = fc["t0"]
        there = fc["arrived"] is not None
        kn, unk = self.fleet.map.area_cells(fc["x"], fc["y"], fc["r"])
        spires = [sid for sid, (ts, rid, _x, _y) in self.fleet.spires_seen.items() if ts >= t0 and rid == self.id]
        return {"label": fc["label"], "reason": fc["reason"], "x": round(fc["x"], 1), "y": round(fc["y"], 1),
                "r": round(fc["r"]), "t": round(t, 1), "there": there,
                "dur": round(max(0.0, t - fc["arrived"]), 1) if there else 0.0,
                "mapped": round((self.mem.mapped_cells - fc["mapped0"]) * self.cfg.res ** 2) if there else 0,
                "pct": round(100.0 * kn / max(kn + unk, 1)),
                "targets": [dict(d) for d in self.mem.targets if d["t"] >= t0],
                "samples": [dict(d) for d in self.mem.samples if d["t"] >= t0],
                "water": [dict(w) for w in self.mem.water if w["t"] >= t0],
                "life": [dict(e) for e in self.mem.life if e["t"] >= t0],
                "spires": spires, "what": fc.get("what"), "mode": fc.get("mode", "explore"),
                "laps": int(fc.get("laps", 0)),
                "driven": round(self.odometer - fc["odo0"]) if there and fc.get("odo0") is not None else 0,
                "changes": self.fleet.map.changes_in(fc["x"], fc["y"], fc["r"], fc["arrived"]) if there else 0,
                "new_found": sorted(self._found_in(fc) - (fc.get("found0") or set())) if there else [],
                "hazards": [dict(h) for h in self.mem.hazards if h["t"] >= t0]}

    def focus_msg(self) -> dict | None:
        fc = self.focus
        if fc is None:
            return None
        left = fc["dur"] if fc["until"] is None else max(0.0, fc["until"] - self.t_now)
        return {"x": round(fc["x"], 1), "y": round(fc["y"], 1), "r": round(fc["r"], 1), "reason": fc["reason"],
                "label": fc["label"], "ends_in": round(left, 1), "mode": fc.get("mode", "explore")}

    # ================================================================== re-survey (protocol v8)
    def _found_in(self, fc: dict) -> set:
        """Ids of the targets the fleet knows inside the focus circle."""
        r = fc["r"] + self.cfg.focus_margin
        return {ts.spec.id for ts in self.fleet.targets
                if ts.status is not None and math.hypot(ts.spec.x - fc["x"], ts.spec.y - fc["y"]) <= r}

    def _to_resurvey(self, t: float) -> None:
        fc = self.focus
        fc["mode"], fc["loop"], fc["wp"] = "resurvey", None, 0
        self.goal, self.path = None, []
        self.fleet.log(t, "info", f"{self.id} {fc['label']} is already mapped: re-surveying it instead, a loop through "
                                  f"it for the rest of the {fc['dur']:.0f} s.", self.id, at=(fc["x"], fc["y"]))

    def _resurvey_loop(self, fc: dict) -> list:
        """Up to RESURVEY_POINTS safe waypoints on a ring through the circle (mapped, passable, off the water),
        starting with the one nearest the rover and going round."""
        f = self.fleet
        pts = []
        for k in range(RESURVEY_POINTS):
            a = 2.0 * math.pi * k / RESURVEY_POINTS
            for frac in RESURVEY_RING:
                rr = max(3.0, frac * fc["r"])
                x, y = fc["x"] + rr * math.cos(a), fc["y"] + rr * math.sin(a)
                if self._safe_cell(x, y) and f.water_zone_at(x, y, margin=1.5) is None and f.in_reach(x, y):
                    pts.append((round(x, 1), round(y, 1)))
                    break
        if not pts:
            return []
        k0 = min(range(len(pts)), key=lambda k: math.hypot(pts[k][0] - self.x, pts[k][1] - self.y))
        return pts[k0:] + pts[:k0]

    def _resurvey_drive(self, t: float) -> None:
        """Drive the loop through the circle again and again while the focus lasts; LiDAR, the spectrometer and
        the tracker keep working as usual (a target found inside the circle is sampled, a creature is watched,
        new rocks are rerouted around). A waypoint with no path is skipped."""
        fc = self.focus
        if fc["loop"] is None:
            fc["loop"], fc["wp"] = self._resurvey_loop(fc), 0
            if not fc["loop"]:
                self.end_focus(t, "noloop")
                self.goal, self.path = None, []
                self.next_goal_try = t
                return
        loop = fc["loop"]
        gx, gy = loop[fc["wp"] % len(loop)]
        if math.hypot(gx - self.x, gy - self.y) < 1.5:
            fc["wp"] += 1
            if fc["wp"] % len(loop) == 0:
                fc["laps"] += 1
            self.goal, self.path = None, []
            return
        if self.goal != (gx, gy):
            self.goal, self.path, self.goal_t = (gx, gy), [], t
        if not self._need_replan(t):
            return
        path = self.fleet.plan(t, self, self.goal)
        self._after_replan(t, path)
        if path is PENDING:
            self._hold_if_blocked()
        elif path is None:
            fc["wp"] += 1                      # no way to that waypoint now: the next one
            self.goal, self.path = None, []
        else:
            self._set_path(t, path)

    # ================================================================== curiosity (interest focus, protocol v7)
    def note_interest(self, t: float, x: float, y: float, kind: str, what: str, label: str | None = None) -> bool:
        """Something new was found at (x, y): stick around it at the next goal choice (if allowed then)."""
        if not self.cfg.curiosity or self.pending_interest is not None or not self._interest_ok(t, x, y):
            return False
        self.pending_interest = {"t": t, "x": float(x), "y": float(y), "kind": kind, "what": what,
                                 "label": label or f"the {kind}"}
        return True

    def _interest_ok(self, t: float, x: float, y: float) -> bool:
        if t < self.interest_next or self.focus is not None or self.held or self.op_goal is not None:
            return False
        if self.state in ("returning", "charging", "fault", "idle") or self.degraded or self.survey is not None:
            return False
        if self.on_rescue() or self.state in ("rescuing", "towing", "disabled", "towed", "wrecked", "repairing"):
            return False                   # a rescue comes first: no curious detours
        if self.dir_bias is not None and t < self.dir_bias[2]:
            return False                   # the operator asked for a direction: that comes first
        if self.fleet.priority is not None:
            return False                   # protocol v8: and so does a fleet priority
        if self.battery < max(0.4, self.need_home + self.battery_margin + 0.15):
            return False                   # low battery: no detours
        f = self.fleet
        for o in f.rovers:
            fc = getattr(o, "focus", None)
            if o is not self and fc is not None and math.hypot(x - fc["x"], y - fc["y"]) <= fc["r"]:
                return False               # another rover already concentrates on that spot
        return not any(math.hypot(x - sx, y - sy) < INTEREST_SPOT_M and t - st < INTEREST_SPOT_S
                       for sx, sy, st in f.interest_spots)

    def _start_interest(self, t: float) -> bool:
        if self.on_rescue():
            self.pending_interest = None
            return False
        pi, self.pending_interest = self.pending_interest, None
        if pi is None or t - pi["t"] > INTEREST_STALE_S or not self._interest_ok(t, pi["x"], pi["y"]):
            return False
        r = self.cfg.interest_r
        kn, unk = self.fleet.map.area_cells(pi["x"], pi["y"], r + self.cfg.focus_margin)
        if unk < max(4 * FOCUS_MIN_UNKNOWN, INTEREST_START_SHARE * (kn + unk)):
            return False                   # little left to look at there: no need to stay
        f = self.fleet
        f.interest_spots = [s for s in f.interest_spots if t - s[2] < INTEREST_SPOT_S] + [(pi["x"], pi["y"], t)]
        self.start_focus(t, pi["x"], pi["y"], r, "interest", pi["label"], self.cfg.interest_s, what=pi["what"],
                         kind=pi["kind"])
        f.log(t, "info", f"{self.id} found {pi['what']} here. Sticking around for about {self.cfg.interest_s:.0f} s "
                         f"to look for more.", self.id, at=(pi["x"], pi["y"]))
        return True

    def _ground_novelty(self, t: float, fm, new) -> None:
        """Ground that differs from what this rover has been driving through: the median slope (roughness) or
        height of the samples a sweep mapped first jumps away from their running mean. A few microseconds."""
        if new is None or len(new) < GROUND_MIN_CELLS or self.state != "exploring":
            return
        ms = float(np.median(fm.slope.ravel()[new]))
        mh = float(np.median(fm.h.ravel()[new]))
        g = self._ground
        if g is None:
            self._ground = [ms, mh, 1]
            return
        rough = ms > g[0] + GROUND_SLOPE_DEG and ms > 1.8 * g[0]
        step = abs(mh - g[1]) > GROUND_HEIGHT_M
        if g[2] >= GROUND_WARMUP and (rough or step) and self.pending_interest is None and self.focus is None \
                and t >= self.interest_next:
            jj, ii = np.divmod(new, fm.nx)
            cx, cy = fm.ox + float(ii.mean()) * fm.res, fm.oy + float(jj.mean()) * fm.res
            if rough:
                what = f"rougher ground (median slope {ms:.0f} deg after {g[0]:.0f} deg)"
            else:
                what = f"a sharp change in height ({mh - g[1]:+.1f} m)"
            self.note_interest(t, cx, cy, "ground", what, "the rough ground" if rough else "the height change")
        g[0] += GROUND_ALPHA * (ms - g[0])
        g[1] += GROUND_ALPHA * (mh - g[1])
        g[2] += 1

    def _check_spires(self, t: float) -> None:
        """Spires the LiDAR newly mapped near this rover (something tall at a spire site, as places.py sees
        them); two or more within SPIRE_CLUSTER_M of each other are a cluster worth a closer look."""
        hz = self.sim.hazards
        if hz is None:
            return
        f, m = self.fleet, self.fleet.map
        reach = self.lidar_eff + 2.0
        new = []
        for sp in hz.all_spires():
            if sp.id in f.spires_seen or abs(sp.x - self.x) > reach or abs(sp.y - self.y) > reach:
                continue
            if not hz.spire_standing.get(sp.id, True):
                continue
            i, j = m.ij(sp.x, sp.y)
            if np.isfinite(m.tall_t[max(j - 2, 0):j + 3, max(i - 2, 0):i + 3]).any():
                f.spires_seen[sp.id] = (t, self.id, float(sp.x), float(sp.y))
                new.append(sp)
        for sp in new:
            near = [(v[2], v[3]) for k, v in f.spires_seen.items()
                    if hz.spire_standing.get(k, True) and math.hypot(v[2] - sp.x, v[3] - sp.y) < SPIRE_CLUSTER_M]
            if len(near) >= 2:
                cx, cy = sum(p[0] for p in near) / len(near), sum(p[1] for p in near) / len(near)
                self.note_interest(t, cx, cy, "spires", f"a cluster of {len(near)} spires", "the spires")
                break

    # ================================================================== lifeforms: observe from a safe distance
    def can_observe(self, t: float, tr, busy_ok: bool = False) -> bool:
        """Free, healthy, enough battery, close enough: may switch to observing this track. busy_ok: an
        injected demo herd may also interrupt a trip to a target (investigating)."""
        if self.state not in (("exploring", "investigating") if busy_ok else ("exploring",)) or self.held or self.op_goal is not None or self.degraded:
            return False
        if self.on_rescue():
            return False                   # on a rescue: the stranded rover comes first
        if self.dir_bias is not None and t < self.dir_bias[2]:
            return False
        px, py = tr.predict(t)
        d = math.hypot(px - self.x, py - self.y)
        if d > 20.0:
            return False
        home = self.energy_for_distance(1.3 * math.hypot(px, py) + 1.3 * d)
        return self.battery >= max(0.4, home + self.battery_margin + 0.15)

    def start_observing(self, t: float, tr) -> None:
        if self.on_rescue():
            return                         # on a rescue: keep going to the stranded rover
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
                    "most": 1, "start": (px, py), "far": 0.0,
                    "t_lim": t,          # the time limit counts from here (reset when the operator answers)
                    "asking": None,      # protocol v7: id of the open lifeform request (following meanwhile)
                    "retry_at": -1e9, "blocked_note": False}
        tr.observer = self.id
        self.mem.on_sighting(t)
        self.fleet.log(t, "discovery", f"{self.id} sees something moving {d:.0f} m {compass(px - self.x, py - self.y)} "
                                       f"(LiDAR track {tr.id}, {tr.speed:.1f} m/s, about {tr.size:.1f} m). Stopping to "
                                       f"look.", self.id, at=(px, py))
        if self.advisor is not None and self.advisor.on_sighting(self, tr, t):
            if self.obs is not None and self.obs["mode"] == "look":
                self.obs["look_until"] = t + 12.0       # tier 2 decides (Gemma's look, maybe the operator)

    def follow_while_asking(self, t: float, qid: str) -> None:
        """Protocol v7: a new herd and a question to the operator about it: follow at the observation standoff
        now (the creatures may wander off) until the answer comes; the time limit waits for the answer."""
        if self.obs is None:
            return
        self.obs["asking"] = qid
        self.obs["mode"] = "follow"
        self.obs["decided"] = True
        self.obs["next_plan"] = t

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
        self.obs["retry_at"] = -1e9

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
        asking = o.get("asking") is not None       # protocol v7: following while the operator decides
        if not asking and t - o["t_lim"] >= cfg.observe_max_s:
            self.end_observing(t, "time")
            return False
        if self._need_return(t):
            self.end_observing(t, "battery")
            self._go_home(t, "battery")
            return False
        if math.hypot(self.x - o["x0"], self.y - o["y0"]) > cfg.observe_leash + (3.0 if asking else 0.0):
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
        if t < o["retry_at"]:              # following while asking, no safe spot closer just now: watch, retry
            self.path, self.goal = [], None
            self._face(px, py)
            return True
        moved = o["goal"] is None or math.hypot(px - o["goal"][2], py - o["goal"][3]) > 1.5
        if t >= o["next_plan"] or moved or not self.path:
            g = self._standoff_goal(px, py, d)
            if g is None:
                if asking:
                    return self._follow_blocked(t, tr, px, py, d)
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
                if asking:
                    return self._follow_blocked(t, tr, px, py, d)
                o["mode"] = "here"
                f.log(t, "info", f"{self.id} no safe path closer to track {tr.id}. Observing from here, "
                                 f"{d:.0f} m away.", self.id, at=(px, py))
                return False
            o["goal"] = (g[0], g[1], px, py)
            self.goal = g
            self._set_path(t, path)
        return False

    def _follow_blocked(self, t: float, tr, px: float, py: float, d: float) -> bool:
        """Following while asking, but no safe spot closer within the leash (steep, soft, water, too far from
        where it started): watch from here and try again in a few seconds (the herd may come back)."""
        o = self.obs
        o["retry_at"] = t + 3.0
        if not o["blocked_note"]:
            o["blocked_note"] = True
            self.fleet.log(t, "info", f"{self.id} can't follow track {tr.id} closer right now (no safe ground within "
                                      f"{self.cfg.observe_leash:.0f} m of where it first saw it). Watching from here, "
                                      f"{d:.0f} m away, until the operator answers.", self.id, at=(px, py))
        self.path, self.goal = [], None
        self._face(px, py)
        return True

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
        label = tr.label if (tr is not None and tr.label and tr.label != "moving object") else None
        self.mem.on_observed(t, n, dur, avg, bool(life), label, (px, py))
        if self.state == "observing":
            self.state = "exploring"
        self.goal, self.path = None, []
        self.next_goal_try = t
        self.next_goal_eval = t
        if o.get("asking") is not None and self.advisor is not None:
            self.advisor.on_follow_ended(self, o["asking"], why, t, o)     # closes the open question, says why
        if life and dur >= 15.0 and why in ("time", "leash", "lost"):
            self.note_interest(t, px, py, "herd", f"a herd of {n} lifeform{'s' if n > 1 else ''}" if n > 1 else
                               "a lifeform", "the herd's grazing ground")    # curiosity: look around where they graze
        if quiet:
            return
        who = "the herd" if n > 1 else "the creature"
        tail = {"time": "Resuming exploration.", "lost": "Lost sight of it. Resuming exploration.",
                "leash": f"Not going more than {self.cfg.observe_leash:.0f} m from where it started. Resuming exploration.",
                "battery": f"Battery {self.battery * 100:.0f}%, returning to the lander.",
                "operator": "Resuming exploration (operator).",
                "hazard": f"Stopped following: {o.get('hazard', 'a hazard')}.",
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
            self.mem.on_hazard("rock", t, (bx, by), f"a rock under my bumper ({bd:.1f} m ahead)")
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
        he = self.health.effects()
        avail = 2.0 * self.cfg.motor_torque * (he["drive_l"] + he["drive_r"])     # protocol v9: a weak side
        gravity_share = self.grav_torque * math.sin(math.radians(max(p_med, 0.0))) / avail
        if he["slope_cap"] is not None and p_med >= 0.8 * p_max and not turning and gravity_share >= 0.8:
            # a damaged drive: its own limit, not evidence about the fleet's (it does not lower the fleet limit)
            f.map.mark_blocked(ax, ay, 0.5)
            f.log(t, "info", f"{self.id} stalled on a {p_med:.0f} deg slope with its damaged drive (own slope limit "
                             f"{he['slope_cap']:.0f} deg). Backing off and replanning.", self.id, at=here)
            self._start_recovery(t, None)
            return
        if p_med >= 12.0 and p_med >= 0.8 * p_max and not turning and gravity_share >= 0.8:
            # nose up on a steady slope with full power and no progress: the slope beat the motors.
            # The tilt (pitch and roll together) is the slope the planner's map measures.
            tilts = sorted(e[7] for e in h)
            sl = max(p_med, tilts[len(tilts) // 2])
            lowered = f.report_slope_stall(t, self, sl, load)
            self.mem.on_hazard("slope", t, here, f"a {sl:.0f} deg slope I could not climb (motors at {load * 100:.0f}%)",
                               f"backed off and the fleet lowered its slope limit to {f.slope_limit:.0f} deg" if lowered
                               else "backed off and replanned")
            if not lowered:
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
                # protocol v9: tipping over knocks the mast (LiDAR, camera) and the antenna and deck
                self.health.hit(t, [("lidar", 1.0), ("camera", 0.8), ("comms", 0.6), ("power", 0.4)], 30.0,
                                "rollover")
                self._hw_call(self.sim.reset_rover_upright, self)
                self.read_pose()
                self.mem.on_hazard("tipped", t, (self.x, self.y), f"tipping over (IMU tilt {tilt:.0f} deg)")
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
        b = f.blocked_ahead(t, self, self.limit_now() + self.plan_relax)
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
            self.mem.note_outcome(("rocks", "spire"), "found no way past, picked another goal" if path is None else (
                f"rerouted (+{extra:.0f} m)" if extra >= 1.0 else "rerouted"), t)
        if path is not None:
            self._count_replan = True
        self.fleet.log(t, "info", msg, self.id, at=(x, y))

    def _may_move_while_asking(self, t: float) -> bool:
        """A rover waiting for the operator stays put, unless staying would be the danger: backing out of
        soft ground or liquid, heading home on a low battery, or making room for a creature that came
        closer than the planetary protection distance."""
        if self.state == "recovering":
            return True
        if self.state == "returning" and self.return_reason in ("battery", "range"):
            return True
        if self.fleet.tracker.tracks and self.fleet.tracker.near(self.x, self.y, self.cfg.observe_min_dist, t):
            return True
        return False

    def _hold_if_blocked(self) -> None:
        """The new plan is not ready yet: if the old path runs into something close, wait."""
        if self.block_info is not None and self.block_info[2] < 3.5:
            self.hold_for_plan = True

    def _set_path(self, t: float, path: list) -> None:
        if self._count_replan or self.block_info is not None or (
                self.path and self.plan_limit_version != self.fleet.limit_version):
            self.mem.replans += 1          # replanned because of something newly perceived or learned
        self._count_replan = False
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
        v_max = min(v_max, cfg.v_max * self.health.effects()["v_cap"])  # protocol v9: a damaged rover drives slower
        if self.state == "towing":
            from .rescue import TOW_V, TOW_W
            v_max, w_max = min(v_max, TOW_V), min(w_max, TOW_W)
        # Skid steer on a slope: a tight arc leaves the inner wheels with almost no speed target, so
        # only the outer pair pushes and the rover bogs down. On sloped ground make big corrections in
        # place and keep arcs wide enough that all four wheels drive forward.
        sloped = self.tilt > 7.0
        if abs(e) > (0.6 if sloped else 0.85):
            v = 0.12 if self.state == "towing" else 0.0      # protocol v9: towing turns in wide arcs, never pivots
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
        if self.question_hold and not self._may_move_while_asking(t):
            v, w = 0.0, 0.0                  # asked the operator what to do: stay put until the answer
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
        if self.motor_fault or self.state in ("disabled", "wrecked"):
            # controller offline, or out of action (protocol v9): the parking brakes lock the wheels
            for a in self.acts:
                d.ctrl[a] = 0.0
                m.actuator_forcerange[a] = (-Tb, Tb)
                m.actuator_gainprm[a][0] = BRAKE_KV
                m.actuator_biasprm[a][2] = -BRAKE_KV
            self._brakes_locked = True
            return
        if self.state == "towed":            # protocol v9: drive clutches open, the wheels roll freely
            for a in self.acts:
                d.ctrl[a] = 0.0
                m.actuator_forcerange[a] = (0.0, 0.0)
                m.actuator_gainprm[a][0] = 0.0
                m.actuator_biasprm[a][2] = 0.0
            self._brakes_locked = True       # the gains are written again when it drives
            return
        e = self.health.effects()
        fl, fr = e["drive_l"], e["drive_r"]
        if self._brakes_locked or self._drive_mode != (fl, fr):
            # protocol v9: a damaged drive side has a weaker servo (it lags, so the rover pulls that way)
            for k, a in enumerate(self.acts):
                f = fl if k < 2 else fr
                kv = cfg.motor_kv if f >= 1.0 else cfg.motor_kv * (0.4 + 0.6 * f)
                m.actuator_gainprm[a][0] = kv
                m.actuator_biasprm[a][2] = -kv
            self._brakes_locked = False
            self._drive_mode = (fl, fr)
        wl = (self.cmd_v - self.cmd_w * cfg.track_eff) / r
        wr = (self.cmd_v + self.cmd_w * cfg.track_eff) / r
        for k, a in enumerate(self.acts):
            wt = wl if k < 2 else wr
            f = fl if k < 2 else fr
            Tk, Tbk = T, Tb
            if f < 1.0:                      # ... less torque and a lower top wheel speed
                wm = cfg.max_wheel_speed * f
                wt = max(-wm, min(wm, wt))
                Tk, Tbk = T * f, Tb * max(f, 0.25)
            d.ctrl[a] = wt
            # motors drive with up to T, brakes and back driving resist with up to 3T
            if wt > 0.05:
                m.actuator_forcerange[a] = (-Tbk, Tk)
            elif wt < -0.05:
                m.actuator_forcerange[a] = (-Tk, Tbk)
            else:
                m.actuator_forcerange[a] = (-Tbk, Tbk)

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
        if self.path and self.state not in PARKED:
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
            "stats": self.mem.stats(self),
            "health": self.health.msg(),   # protocol v9
        }
        if self.focus is not None:
            msg["focus"] = self.focus_msg()
        vis = self.vision.msg()
        if vis is not None:
            msg["vision"] = vis            # protocol v8: latest detections of the onboard camera
        if self.sampling is not None:
            sm = self.sampling
            msg["sampling"] = {"target": self.fleet.targets[sm["tidx"]].spec.id, "instrument": sm["instrument"],
                               "progress": round(min(1.0, max(0.0, (self.t_now - sm["t0"]) / sm["dur"])), 2)}
        return msg
