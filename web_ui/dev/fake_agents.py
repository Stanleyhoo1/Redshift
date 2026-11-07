"""Scripted-but-reactive rover and scout agents for the fake backend."""
from __future__ import annotations

import math

import numpy as np

from fake_render import CLS_OBSTACLE, CLS_PERMANENT
from fake_world import World

G0 = 9.80665
# docs/protocol.md §8.1
NAMES = {"rover_0": ("scout", "Scout Rover 1", "ROVER_1"), "rover_1": ("recovery", "Heavy Recovery Rover 2", "ROVER_2"),
         "scout_0": ("drone", "Survey Drone", "DRONE")}
BATTERY_START = {"rover_0": 0.82, "rover_1": 0.91, "scout_0": 0.96}
DAMAGED_WHEELS = [0.08, 0.35, 0.05, 0.30]
DAMAGE_FAULTS = ["DRIVE_FL_OVERCURRENT", "DRIVE_RL_ENCODER_LOST"]
LOCK_LABEL = "[PERCEIVED: DAMAGED UNIT {short} | DOCKING FOR REPAIR]"


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def short(rid: str) -> str:
    return NAMES.get(rid, ("", rid, rid.upper()))[2]


def display(rid: str) -> str:
    return NAMES.get(rid, ("", rid, rid))[1]


class Agent:
    kind = "rover"

    def __init__(self, rid: str, x: float, y: float, yaw: float):
        self.id, self.x, self.y, self.yaw = rid, x, y, yaw
        self.z = 0.0
        self.speed = 0.0
        self.state = "BOOT"
        self.mode = "FUSED"
        self.path: list = []
        self.trail: list = [[round(x, 2), round(y, 2)]]
        self.goal: list | None = None
        self.state_t = 0.0
        self.roll = self.pitch = 0.0
        self.gyro = [0.0, 0.0, 0.0]
        self.unc = 0.14
        self.throttle = 0
        self.extra_ms = 0.0
        self.role, self.name, _ = NAMES.get(rid, ("recovery" if rid.startswith("rover") else "drone", rid, rid.upper()))
        self.capacity_wh = 600.0 if self.kind == "rover" else 300.0
        self.battery_wh = self.capacity_wh * BATTERY_START.get(rid, 0.9)
        self.v_cmd = 0.0
        self.solar_w = 0.0
        self.load_w = 0.0
        self.accel = [0.0, 0.0, G0]
        self.structure = 100.0
        self.shocks = 0
        self.peak_g = 0.0
        self.rescue: dict | None = None  # §8.7 telemetry block
        self._edge: dict = {}

    def power(self, w: World, dt: float):
        """Battery integration: solar − (avionics + edge compute + drive / hover)."""
        dust = w.env["dust_level"]
        edge_w = float(self._edge.get("power_w", 10.0)) if self._edge else 10.0
        if self.kind == "scout":
            self.solar_w = 0.0
            self.load_w = (60.0 * w.g ** 1.5 if self.alt > 0.5 else 8.0) + edge_w + float(w.rng.normal(0, 1.5))
        else:
            tilt = math.radians(math.hypot(self.roll, self.pitch))
            irr = 0.93 * (1 - 0.85 * dust) * (1 + float(w.rng.normal(0, 0.006)))
            self.solar_w = max(0.0, 220.0 * irr * max(0.0, math.cos(tilt)))
            self.load_w = 25.0 + edge_w + w.limits["energy_j_per_m"] * abs(self.speed) + float(w.rng.normal(0, 0.8))
        self.battery_wh = min(self.capacity_wh, max(0.0, self.battery_wh + (self.solar_w - self.load_w) * dt / 3600.0))

    def imu_accel(self, w: World, now: float, extra: tuple = (0.0, 0.0, 0.0)):
        """Specific force in the IMU frame (m/s²): gravity tilted by roll/pitch + vibration + impulses."""
        g = w.g * G0
        r, p = math.radians(self.roll), math.radians(self.pitch)
        sd = 0.10 + 0.35 * abs(self.speed) + (1.8 if w.env["seismic"] else 0.0)
        a = [-g * math.sin(p) + extra[0], g * math.sin(r) * math.cos(p) + extra[1], g * math.cos(r) * math.cos(p) + extra[2]]
        self.accel = [round(v + float(w.rng.normal(0, sd)), 3) for v in a]
        dev = abs(math.sqrt(sum(v * v for v in self.accel)) / G0 - w.g)
        if dev > 1.2:  # a shock (§8.7 health model)
            if not getattr(self, "_in_shock", False):
                self.shocks += 1
                if self.kind == "rover":
                    self.structure = max(0.0, self.structure - min(4.0, 0.8 * dev))
            self._in_shock = True
            self.peak_g = max(self.peak_g, round(dev, 1))
        else:
            self._in_shock = False

    def health_json(self) -> dict:
        drive = self.drive_pct()
        pct = self.structure * drive / 100.0
        net = self.solar_w - self.load_w
        return {"pct": round(pct, 1), "structure_pct": round(self.structure, 1), "drive_pct": round(drive, 1),
                "battery_pct": round(100 * self.battery_wh / self.capacity_wh, 1), "battery_wh": round(self.battery_wh, 1),
                "capacity_wh": self.capacity_wh, "solar_w": round(self.solar_w, 1), "load_w": round(self.load_w, 1),
                "net_w": round(net, 1), "faults": list(getattr(self, "faults", [])), "critical": pct < 15,
                "shocks": self.shocks, "peak_g": self.peak_g}

    def drive_pct(self) -> float:
        return 100.0

    def set_state(self, s: str, now: float):
        self.state, self.state_t = s, now

    def clock(self, now: float):
        if self.state_t == 0.0:
            self.state_t = now

    def drive_factor(self) -> float:
        return 1.0

    def follow(self, dt: float, v: float, reverse: bool = False) -> bool:
        """Pure-pursuit along self.path; returns True at the end."""
        while len(self.path) > 1 and math.hypot(self.path[0][0] - self.x, self.path[0][1] - self.y) < 1.2:
            self.path.pop(0)
        if not self.path or (len(self.path) == 1 and math.hypot(self.path[0][0] - self.x, self.path[0][1] - self.y) < 0.6):
            self.speed = 0.0
            self.v_cmd = 0.0
            return True
        tx, ty = self.path[min(1, len(self.path) - 1)] if len(self.path) > 1 else self.path[0]
        want = math.atan2(ty - self.y, tx - self.x) + (math.pi if reverse else 0.0)
        err = wrap(want - self.yaw)
        rate = 3.0 if self.kind == "scout" else 1.4
        self.yaw = wrap(self.yaw + max(-rate * dt, min(rate * dt, err)))
        if self.kind == "scout":
            # a multirotor yaws on the spot instead of looping round a waypoint, and brakes into the last one
            end = self.path[-1]
            v = min(v, 1.0 + 1.5 * math.hypot(end[0] - self.x, end[1] - self.y))
            self.speed = v * max(0.0, math.cos(err)) ** 2
            self.v_cmd = self.speed
        else:
            self.v_cmd = v * max(0.25, math.cos(err))
            self.speed = self.v_cmd * self.drive_factor() * (1 - min(0.9, getattr(self, "slip", 0.0)))
        sgn = -1.0 if reverse else 1.0
        self.x += sgn * self.speed * math.cos(self.yaw) * dt
        self.y += sgn * self.speed * math.sin(self.yaw) * dt
        if math.hypot(self.trail[-1][0] - self.x, self.trail[-1][1] - self.y) > 0.5:
            self.trail.append([round(self.x, 2), round(self.y, 2)])
            del self.trail[:-300]
        return False


class Rover(Agent):
    kind = "rover"

    def __init__(self, rid: str, x: float, y: float, yaw: float):
        super().__init__(rid, x, y, yaw)
        self.slip = 0.03
        self.last_safe = [round(x, 2), round(y, 2)]
        self.breadcrumbs: list = [[x, y]]
        self.jolt_t: float | None = None
        self.jolt_xy: list | None = None
        self.seen_rocks = 0
        self.replan_t = 0.0
        self.scan_id = 0
        self.prev_roll = 0.0
        self.caution_since: float | None = None
        self.goal_t = 0.0
        # §8.3 motor controllers (slight per-wheel wear) and §8.7 health
        self.wheel_health = [0.985, 0.978, 0.991, 0.982]
        self.faults: list[str] = []
        self.structure = 97.5 + (0.8 if rid.endswith("1") else 0.0)
        self.damage_t: float | None = None
        self.damage_handled = False
        self.repair_t0: float | None = None
        self.repair_from: list | None = None
        self.repair_done = False
        self.sos_id: str | None = None
        self.sos_n = 0
        self.rescuer_id: str | None = None
        self.lone_noted = False
        self.peers: list = []
        # recovery side
        self.rescue_target = None
        self.locked = False
        self.docked = False
        self.repair_requested = False

    # ---- drivetrain / health ---------------------------------------------------------------
    def drive_factor(self) -> float:
        return sum(self.wheel_health) / 4

    def drive_pct(self) -> float:
        # mean wheel health, cross-checked against commanded vs measured wheel speed (faults weigh in hard)
        return 100.0 * self.drive_factor() ** 1.4

    def health_pct(self) -> float:
        return self.structure * self.drive_pct() / 100.0

    def strike(self, w: World, now: float) -> dict:
        """§8.5 critical_damage: debris impulse + drivetrain fault."""
        self.damage_t = now
        self.damage_handled = False
        self.wheel_health = list(DAMAGED_WHEELS)
        self.faults = list(DAMAGE_FAULTS)
        self.structure = min(self.structure, 74.0 + float(w.rng.uniform(-1.0, 1.0)))
        self.repair_t0, self.repair_done, self.rescuer_id, self.lone_noted = None, False, None, False
        return {"op": "critical_damage", "ok": True, "robot": self.id, "impact": [round(self.x, 2), round(self.y, 2)],
                "peak_g": 5.1, "wheel_health": list(DAMAGED_WHEELS)}

    def begin_repair(self, now: float):
        self.repair_t0 = now
        self.repair_from = list(self.wheel_health)

    def sensors_json(self, w: World, others: list) -> dict:
        nearest = 25.0
        for x, y, r in w.rocks:
            d = math.hypot(x - self.x, y - self.y) - r
            if d < nearest:
                nearest = d
        for o in others:
            if o is not self and o.kind == "rover":
                nearest = min(nearest, math.hypot(o.x - self.x, o.y - self.y) - 0.55)
        dust = w.env["dust_level"]
        if dust > 0 and w.rng.random() < 0.8 * dust:
            nearest = min(nearest, float(w.rng.uniform(0.4, 3.5)))  # backscatter off the dust
        v_wheel = self.v_cmd * self.drive_factor()
        return {"accel": self.accel, "accel_g": round(math.sqrt(sum(v * v for v in self.accel)) / G0, 3),
                "gyro": self.gyro, "roll_deg": round(self.roll, 2), "pitch_deg": round(self.pitch, 2),
                "lidar_nearest_m": round(max(0.2, nearest + float(w.rng.normal(0, 0.03))), 2),
                "lidar_returns_pct": round(max(5.0, 93.0 - 34.0 * dust + float(w.rng.normal(0, 1.2))), 1),
                "camera_conf": round(max(0.05, 1 - self.unc * 1.2), 2), "v_cmd": round(self.v_cmd, 3),
                "v_wheel": round(v_wheel + float(w.rng.normal(0, 0.006)) * (v_wheel > 0), 3), "v_ground": round(self.speed, 3),
                "slip": round(self.slip, 3), "altitude_agl": None}

    # ---- rescue (recovery side) --------------------------------------------------------------
    def start_rescue(self, target: "Rover", w: World, now: float):
        self.rescue_target = target
        self.locked = self.docked = self.repair_requested = False
        d = math.hypot(target.x - self.x, target.y - self.y) or 1.0
        sx = target.x + (self.x - target.x) / d * 2.2
        sy = target.y + (self.y - target.y) / d * 2.2
        path = w.plan((self.x, self.y), (sx, sy)) or World._resample([(self.x, self.y), (sx, sy)], 0.8)
        self.path = path
        self.goal = [round(sx, 1), round(sy, 1)]
        self.mode = "FUSED"
        self.set_state("RESCUE_TRANSIT", now)
        h = target.health_pct()
        w.trace(self.id, "RESCUE_TRANSIT", "rescue_start", ["FLEET", "MAP"],
                f"SOS {target.sos_id} from {short(target.id)} at ({target.x:.1f}, {target.y:.1f}): health {h:.0f} %, {', '.join(target.faults) or 'no faults'}",
                f"Only able rover in the fleet; A* route {len(path) * 0.8:.1f} m to a 2.2 m stand-off beside the damaged unit",
                0.94, f"Abandoning frontier goal: AUTONOMOUS RESCUE MODE, transit at {0.85 * w.limits['v_max']:.2f} m/s", "warn", 31.0)

    def rescue_block(self, role: str) -> dict | None:
        t = self.rescue_target
        if t is None:
            return None
        phase = {"RESCUE_TRANSIT": "TRANSIT", "RESCUE_DOCKING": "DOCKING", "REPAIRING": "REPAIRING"}.get(self.state, "DONE")
        rng_m = round(math.hypot(t.x - self.x, t.y - self.y), 2)
        return {"id": t.sos_id, "role": role, "peer": t.id if role == "rescuer" else self.id, "phase": phase, "range_m": rng_m,
                "locked": self.locked and phase in ("DOCKING", "REPAIRING")}

    def _rescue_step(self, w: World, dt: float, now: float) -> None:
        t = self.rescue_target
        st = self.state
        rng_m = math.hypot(t.x - self.x, t.y - self.y)
        if st == "RESCUE_TRANSIT":
            if self.follow(dt, 0.85 * w.limits["v_max"]) or rng_m <= 2.5:
                self.speed = self.v_cmd = 0.0
                self.set_state("RESCUE_DOCKING", now)
                w.trace(self.id, "RESCUE_DOCKING", "rescue_dock", ["ODOM", "MAP", "LIDAR"],
                        f"Stand-off reached: {rng_m:.1f} m from {short(t.id)} by odometry",
                        "Map navigation is good to ~0.5 m; the last metre needs a perception lock on the damaged rover",
                        0.9, "Switching to perception-guided docking at 0.20 m/s", "info", 22.0)
        elif st == "RESCUE_DOCKING":
            want = math.atan2(t.y - self.y, t.x - self.x)
            err = wrap(want - self.yaw)
            self.yaw = wrap(self.yaw + max(-1.2 * dt, min(1.2 * dt, err)))
            if not self.locked and abs(err) < 0.12 and now - self.state_t > 1.0:
                self.locked = True
                w.trace(self.id, "RESCUE_DOCKING", "rescue_lock", ["CAM", "DEPTH", "LIDAR"],
                        LOCK_LABEL.format(short=short(t.id)),
                        f"ROVER detection conf 0.93 at {rng_m:.2f} m, bearing {err:+.2f} rad; depth and LiDAR ranges agree within 0.04 m",
                        0.93, "Lock acquired: creeping in on the fused range at 0.20 m/s", "info", 44.0)
            if self.locked and not self.docked:
                if rng_m > 1.38:
                    self.v_cmd = 0.2
                    self.speed = 0.2 * self.drive_factor()
                    self.x += self.speed * math.cos(self.yaw) * dt
                    self.y += self.speed * math.sin(self.yaw) * dt
                else:
                    self.speed = self.v_cmd = 0.0
                    self.docked = True
                    w.trace(self.id, "RESCUE_DOCKING", "docked", ["LIDAR", "DEPTH", "ODOM"],
                            f"Centre distance {rng_m:.2f} m; LiDAR nearest {rng_m - 0.55:.2f} m to {short(t.id)}'s hull",
                            "Inside the 1.5 m docking envelope with the lock held",
                            0.95, "Docked: requesting the field repair", "info", 18.0)
            else:
                self.speed = self.v_cmd = 0.0
        elif st == "REPAIRING":
            self.speed = self.v_cmd = 0.0

    def jolt(self, now: float):
        self.jolt_t = now
        self.jolt_xy = [round(self.x, 2), round(self.y, 2)]

    def step(self, w: World, dt: float, now: float, scouts: list):
        self.clock(now)
        lim = w.limits
        v_lim = lim["v_max"]
        T = w.terrain
        self.z = T.height_at(self.x, self.y)
        gx, gy = T.gradient_at(self.x, self.y)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pitch = -math.degrees(math.atan(gx * c + gy * s))
        roll = math.degrees(math.atan(-gx * s + gy * c))
        vib = 0.0
        if self.jolt_t is not None:
            tj = now - self.jolt_t
            amp = 37.5 * (math.sin(min(1.0, tj / 0.22) * math.pi / 2) if tj < 0.22 else math.exp(-(tj - 0.22) / 0.55) * math.cos((tj - 0.22) * 5))
            roll += amp
            pitch += amp * 0.18
            if tj > 4:
                self.jolt_t = None
        extra = (0.0, 0.0, 0.0)
        if self.damage_t is not None and now - self.damage_t < 1.5:
            td = now - self.damage_t
            k = math.exp(-td / 0.25)
            roll += 7.0 * math.exp(-td / 0.3) * math.cos(td * 14)
            extra = (1.8 * G0 * k * math.cos(td * 30), 1.2 * G0 * k, 5.4 * G0 * k)
        if self.jolt_t is not None and now - self.jolt_t < 0.5:
            tj = now - self.jolt_t
            extra = (0.4 * G0 * math.exp(-tj / 0.15), 1.5 * G0 * math.exp(-tj / 0.15), 0.6 * G0 * math.exp(-tj / 0.15))
        if w.env["seismic"]:
            vib = 2.2
        self.roll = roll + float(w.rng.normal(0, 0.15 + vib))
        self.pitch = pitch + float(w.rng.normal(0, 0.15 + vib * 0.6))
        self.gyro = [round((self.roll - self.prev_roll) / max(dt, 1e-3) * math.pi / 180, 3), round(float(w.rng.normal(0, 0.02 + vib * 0.05)), 3), 0.0]
        self.prev_roll = self.roll
        dust = w.env["dust_level"]
        self.unc += ((0.14 + 0.42 * dust + (0.08 if vib else 0.0)) - self.unc) * min(1.0, dt * 1.5)
        self.slip = 0.7 if (self.jolt_t is not None and now - self.jolt_t < 0.8) else 0.03 + 0.002 * abs(pitch) + float(w.rng.uniform(0, 0.01))
        self.imu_accel(w, now, extra)
        self.power(w, dt)
        if not self.faults and self.repair_t0 is None:  # slow wear on healthy wheels
            self.wheel_health = [min(1.0, max(0.95, h + float(w.rng.normal(0, 0.0004)))) for h in self.wheel_health]
        if self.repair_t0 is not None and not self.repair_done:  # §8.5 repair: ramp to 1.0 over 4 s
            f = min(1.0, (now - self.repair_t0) / 4.0)
            self.wheel_health = [a + (1.0 - a) * f for a in self.repair_from]
            if f >= 1.0:
                self.repair_done = True
                self.faults = []
        self._step_fsm(w, dt, now, scouts)
        if math.hypot(self.trail[-1][0] - self.x, self.trail[-1][1] - self.y) > 0.5:  # docking creeps outside follow()
            self.trail.append([round(self.x, 2), round(self.y, 2)])
            del self.trail[:-300]

    def _step_fsm(self, w: World, dt: float, now: float, scouts: list):
        lim = w.limits
        v_lim = lim["v_max"]
        st = self.state
        alarm = lim["tip_alarm_deg"]
        # a struck rover keeps commanding for half a second before the diagnosis lands: wheels lag the command
        if self.damage_t is not None and not self.damage_handled and now - self.damage_t > 0.5:
            self.damage_handled = True
            self.set_state("SAFE_HOLD", now)
            d, h = self.drive_pct(), self.health_pct()
            w.trace(self.id, "SAFE_HOLD", "critical_damage", ["IMU", "DIAG", "ODOM"],
                    f"Shock {self.peak_g:.1f} g; motor controllers: {', '.join(self.faults)}; wheels {self.v_cmd * self.drive_factor():.2f} m/s vs {self.v_cmd:.2f} commanded",
                    f"Health {h:.0f} % < 15 % critical (structure {self.structure:.0f} %, drive {d:.0f} %): the drivetrain cannot move the rover safely",
                    0.97, "SAFE_HOLD: drive stopped, brakes on; preparing SOS", "critical", 7.4)
            self.speed = self.v_cmd = 0.0
            return
        if st in ("RESCUE_TRANSIT", "RESCUE_DOCKING", "REPAIRING"):
            self._rescue_step(w, dt, now)
            return
        if st == "AWAIT_RESCUE":
            self.speed = self.v_cmd = 0.0
            if self.repair_done and self.health_pct() >= 60:
                self.set_state("RESUMING", now)
                w.trace(self.id, "RESUMING", "repaired", ["DIAG", "ODOM", "IMU"],
                        f"Drive health {self.drive_pct():.0f} % after the field repair; motor controllers report no faults",
                        f"Health {self.health_pct():.0f} % ≥ 60 % resume threshold; wheel speeds track commands again",
                        0.95, "SOS resolved: resuming exploration", "info", 12.0)
            return
        if st == "SAFE_HOLD" and self.faults:
            self.speed = self.v_cmd = 0.0
            if now - self.state_t > 1.0:
                self.sos_n += 1
                self.sos_id = f"sos-{self.id}-{self.sos_n}"
                self.set_state("AWAIT_RESCUE", now)
                lone = not [p for p in self.peers if p.kind == "rover"]
                w.trace(self.id, "AWAIT_RESCUE", "sos", ["FLEET", "DIAG"],
                        f"Health {self.health_pct():.0f} %; {', '.join(self.faults)}; position ({self.x:.1f}, {self.y:.1f})",
                        f"Cannot self-recover: drive health {self.drive_pct():.0f} % is far below the 60 % needed to move" +
                        ("; no other rover in the fleet can reach me" if lone else "; a teammate can repair the drivetrain in the field"),
                        0.96, f"Broadcasting SOS {self.sos_id} at 1 Hz; holding position" + (" (no rescuer available)" if lone else ""), "critical", 9.0)
            return
        if st not in ("SAFE_HOLD", "RECOVERY_REVERSE") and abs(self.roll) > alarm:
            self.set_state("SAFE_HOLD", now)
            self.speed = 0.0
            w.trace(self.id, "SAFE_HOLD", "safe_hold", ["IMU", "ODOM"],
                    f"Roll {self.roll:.1f}° at {abs(self.gyro[0]) * 57.3:.0f}°/s; wheel slip {self.slip:.2f}",
                    f"|roll| {abs(self.roll):.1f}° > tip alarm {alarm:.1f}° for 100 ms at {w.g:.1f} g; chassis unstable",
                    0.97, "SAFE-HOLD: drive stopped, brakes on for 1.5 s", "critical", 6.2)
            return
        if st == "BOOT":
            if now - self.state_t > 1.5:
                self.set_state("EXPLORING", now)
                w.trace(self.id, "EXPLORING", "mission_start", ["ODOM", "MAP", "EDGE"],
                        f"Spawn pad at ({self.x:.1f}, {self.y:.1f}); map 0 % known; {w.profile_id} budget",
                        f"Gravity {w.g:.1f} g limits slope to {lim['max_slope_deg']:.1f}° ({lim['factor'].replace('_', ' ')}) and speed to {v_lim:.2f} m/s",
                        0.99, "Begin frontier exploration")
                self._new_goal(w, now, scouts)
        elif st in ("EXPLORING", "CAUTION"):
            self._caution_logic(w, now)
            fresh_rocks = w.rock_version != self.seen_rocks
            blocker = w.path_blocked(self.path, 14) if (fresh_rocks or now - self.replan_t > 3.0) else 0
            if blocker == CLS_OBSTACLE and fresh_rocks:
                self.seen_rocks = w.rock_version
                self._replan(w, now)
                return
            if blocker:  # newly mapped slope / no-go cells: reroute on the move
                self._reroute(w, now, blocker)
            v = v_lim * (0.35 if self.state == "CAUTION" else 0.78)
            if self.follow(dt, v) and now - self.goal_t > 1.0:
                self.goal_t = now
                self._new_goal(w, now, scouts)
            if self.unc < 0.4 and abs(self.roll) < 0.5 * alarm and math.hypot(self.x - self.breadcrumbs[-1][0], self.y - self.breadcrumbs[-1][1]) > 0.5:
                self.breadcrumbs.append([self.x, self.y])
                self.last_safe = [round(self.x, 2), round(self.y, 2)]
        elif st == "REPLANNING":
            self.speed = 0.0
            if now - self.state_t > 1.2 and not getattr(self, "_replanned", False):
                self._replanned = True
                self._finish_replan(w, now)
            if now - self.state_t > 2.2:
                self._replanned = False
                self.set_state("CAUTION" if w.env["dust_level"] > 0 else "EXPLORING", now)
        elif st == "SAFE_HOLD":
            self.speed = 0.0
            if now - self.state_t > 1.5:
                back = [b for b in self.breadcrumbs if math.hypot(b[0] - self.x, b[1] - self.y) >= 3.0]
                tgt = back[-1] if back else self.breadcrumbs[0]
                self.path = [[self.x, self.y], [round(tgt[0], 2), round(tgt[1], 2)]]
                self.last_safe = [round(tgt[0], 2), round(tgt[1], 2)]
                self.set_state("RECOVERY_REVERSE", now)
                d = math.hypot(tgt[0] - self.x, tgt[1] - self.y)
                w.trace(self.id, "RECOVERY_REVERSE", "recovery_reverse", ["ODOM", "IMU", "MAP"],
                        f"Roll settled to {self.roll:.1f}°; {len(self.breadcrumbs)} breadcrumbs logged",
                        f"Last safe cell ({tgt[0]:.1f}, {tgt[1]:.1f}) is {d:.1f} m back along the driven track",
                        0.93, f"Reversing {d:.1f} m along breadcrumbs at 0.50 m/s", "warn")
        elif st == "RECOVERY_REVERSE":
            if self.follow(dt, 0.5, reverse=True):
                xy = self.jolt_xy or [self.x, self.y]
                w.perm.append([xy[0], xy[1], 2.5])
                w.refresh_truth()
                w.event({"op": "hazard_mark", "robot": self.id, "xy": xy, "radius_m": 2.5, "reason": "tip_event"})
                w.trace(self.id, "AWAIT_SCOUT", "hazard_marked", ["IMU", "MAP", "FLEET"],
                        f"Tip event at ({xy[0]:.1f}, {xy[1]:.1f}); local slope estimate unreliable",
                        "Terrain caused a tip-over; unseen relief may exceed the 16.8° limit", 0.9,
                        "Marked permanent no-go zone r 2.5 m; shared with fleet", "warn")
                self.scan_id += 1
                if scouts:
                    w.event({"op": "scan_request", "id": self.scan_id, "robot": self.id, "xy": xy, "reason": "tip_event"})
                    scouts[0].task(xy, self.scan_id, self.id, w, now)
                    w.trace(self.id, "AWAIT_SCOUT", "scout_tasked", ["FLEET"],
                            f"Scan request #{self.scan_id} for ({xy[0]:.1f}, {xy[1]:.1f})",
                            f"{scouts[0].id} is {math.hypot(scouts[0].x - xy[0], scouts[0].y - xy[1]):.0f} m away with a downward camera",
                            0.88, f"Holding at last safe cell until {scouts[0].id} reports slope stats")
                self.set_state("AWAIT_SCOUT", now)
        elif st == "AWAIT_SCOUT":
            self.speed = 0.0
            if not scouts and now - self.state_t > 4:
                self.resume(w, now, None)
        elif st == "RESUMING":
            if now - self.state_t > 1.8:
                self.set_state("CAUTION" if w.env["dust_level"] > 0 else "EXPLORING", now)
                self._new_goal(w, now, scouts)

    def resume(self, w: World, now: float, verdict: dict | None):
        self.set_state("RESUMING", now)
        w.trace(self.id, "RESUMING", "resume", ["FLEET", "MAP"],
                f"Scout verdict: {verdict['verdict']} (max slope {verdict['max_slope_deg']:.1f}°)" if verdict else "Scout timeout after 25 s",
                "No-go zone stays permanent; remaining frontiers reachable around it", 0.9,
                "Re-planning around the no-go zone and resuming exploration")

    def _caution_logic(self, w: World, now: float):
        if self.state == "EXPLORING" and self.unc > 0.45:
            self.caution_since = self.caution_since or now
            if now - self.caution_since > 0.5:
                self.set_state("CAUTION", now)
                self.mode = "LIDAR_ONLY"
                self.extra_ms = 5.0
                w.trace(self.id, "CAUTION", "caution_enter", ["CAM", "LIDAR", "IMU"],
                        f"Camera sharpness −64 %, contrast 0.19; LiDAR returns {int(100 - 30 * w.env['dust_level'])} % with near-field backscatter",
                        f"Fused uncertainty {self.unc:.2f} > 0.45 for 0.5 s; camera confidence {max(0.05, 0.9 - self.unc):.2f} < 0.40",
                        0.86, f"CAUTION: LiDAR-only perception, speed ×0.35 ({0.35 * w.limits['v_max']:.2f} m/s), +0.4 m inflation", "warn")
        elif self.state == "CAUTION" and self.unc < 0.3:
            self.set_state("EXPLORING", now)
            self.mode = "FUSED"
            self.extra_ms = 0.0
            self.caution_since = None
            w.trace(self.id, "EXPLORING", "caution_exit", ["CAM", "LIDAR"],
                    f"Camera confidence {1 - self.unc:.2f} > 0.70 for 2 s; LiDAR returns 97 %",
                    "Dust cleared; fused perception trustworthy again", 0.9, "Back to FUSED perception at full speed")

    def _new_goal(self, w: World, now: float, scouts: list):
        others = [s.goal for s in scouts if s.goal] + [p.goal for p in self.peers if p.goal] + [[p.x, p.y] for p in self.peers]
        g = w.frontier_goal(self.x, self.y, self.yaw, others)
        if g is None:
            self.path = []
            if w.stats()["percent"] >= 95:
                self.set_state("MISSION_COMPLETE", now)
            return
        self.path = w.plan((self.x, self.y), g)
        if not self.path:  # walled in: remember it so the next pick (and the trace) is a real route
            w.dead_goals.append(g)
            self.goal = None
            return
        self.goal = [round(g[0], 1), round(g[1], 1)]
        dist = math.hypot(g[0] - self.x, g[1] - self.y)
        w.trace(self.id, self.state, "frontier_selected", ["MAP", "FLEET"],
                f"{int(w.rng.integers(3, 9))} frontier clusters; best at ({g[0]:.1f}, {g[1]:.1f}), {dist:.0f} m",
                f"Score = size − λ·cost; {len(self.path)}-point A* path, {sum(1 for _ in self.path) * 0.8:.0f} m at {w.g:.1f} g energy cost",
                float(w.rng.uniform(0.78, 0.92)), f"Driving to frontier at {0.78 * w.limits['v_max']:.2f} m/s")

    def _replan(self, w: World, now: float):
        """Blocked: stop and announce. The new route is computed in REPLANNING after 1.2 s."""
        self.set_state("REPLANNING", now)
        self.speed = 0.0
        w.trace(self.id, "REPLANNING", "path_blocked", ["LIDAR", "CAM"],
                f"Boulder field {float(w.rng.uniform(5.5, 7.5)):.1f} m ahead across path ({int(w.rng.integers(4, 7))} objects)",
                f"Corridor occupancy {float(w.rng.uniform(0.7, 0.9)):.2f}; step height {float(w.rng.uniform(0.7, 1.0)):.1f} m > 0.30 m clearance",
                0.91, "Stop and re-plan A* around the obstacle", "warn", 41.7)

    def _reroute(self, w: World, now: float, blocker: int):
        old = len(self.path) * 0.8
        goal = tuple(self.goal) if self.goal else (self.x, self.y)
        path = w.plan((self.x, self.y), goal)
        self.replan_t = now
        if not path:
            self._new_goal(w, now, [])
            return
        self.path = path
        what = {CLS_PERMANENT: "No-go zone", CLS_OBSTACLE: "Boulder"}.get(blocker, f"Slope above {w.limits['max_slope_deg']:.1f}°")
        w.trace(self.id, self.state, "replanned", ["LIDAR", "MAP"],
                f"{what} newly mapped across the planned route",
                f"Lethal at {w.g:.1f} g; cheaper to detour than to stop", 0.87,
                f"Re-planned on the move ({len(path) * 0.8 - old:+.1f} m)", "info", 24.0)

    def _finish_replan(self, w: World, now: float):
        old_len = len(self.path) * 0.8
        goal = tuple(self.goal) if self.goal else (self.x, self.y)
        path = w.plan((self.x, self.y), goal)
        if not path:
            g = w.frontier_goal(self.x, self.y, self.yaw + math.pi / 2, [])
            if g:
                self.goal = [round(g[0], 1), round(g[1], 1)]
                path = w.plan((self.x, self.y), g)
        self.path = path
        self.replan_t = now
        self.seen_rocks = w.rock_version
        self.extra_ms = 12.0
        d = len(path) * 0.8 - old_len
        w.trace(self.id, "REPLANNING", "replanned", ["MAP", "EDGE"],
                f"{len(path)} waypoints; lethal cells dilated 0.6 m",
                f"Detour {d:+.1f} m vs blocked route; within the {w.profile_id} expansion budget",
                0.89, f"Re-planned A* around obstacle ({d:+.1f} m, {int(w.rng.integers(900, 1800))} nodes, 38 ms)", "info", 38.0)


class Scout(Agent):
    kind = "scout"

    def __init__(self, rid: str, x: float, y: float, L: float):
        super().__init__(rid, x, y, 0.0)
        self.alt = 0.0
        self.lanes = self._lawnmower(L)
        self.lane_i = 0
        self.req: dict | None = None
        self.orbit_c = (0.0, 0.0)

    @staticmethod
    def _lawnmower(L: float) -> list:
        pts, half, step = [], 0.42 * L, max(10.0, 0.18 * L)
        ys = np.arange(-half, half + 0.1, step)
        for k, y in enumerate(ys):
            xs = (-half, half) if k % 2 == 0 else (half, -half)
            pts += [[xs[0], float(y)], [xs[1], float(y)]]
        return pts

    def task(self, xy: list, sid: int, robot: str, w: World, now: float):
        self.req = {"xy": xy, "id": sid, "robot": robot}
        self.path = World._resample([(self.x, self.y), tuple(xy)], 1.5)
        self.set_state("FLY_TO_SCAN", now)
        w.trace(self.id, "FLY_TO_SCAN", "scout_tasked", ["FLEET", "ODOM"],
                f"Scan request #{sid} from {robot} at ({xy[0]:.1f}, {xy[1]:.1f})",
                f"Pausing survey lane {self.lane_i // 2 + 1}; {math.hypot(xy[0] - self.x, xy[1] - self.y):.0f} m transit at 12 m/s",
                0.95, "Flying to the tip site for a nadir slope scan")

    def step(self, w: World, dt: float, now: float, rovers: list):
        self.clock(now)
        T = w.terrain
        ground = T.height_at(self.x, self.y)
        self.roll = float(w.rng.normal(0, 0.8)) + (2.0 if self.speed > 1 else 0.0)
        self.pitch = float(w.rng.normal(0, 0.8)) - 3.0 * min(1.0, self.speed / 3.0)
        self.gyro = [round(float(w.rng.normal(0, 0.03)), 3) for _ in range(3)]
        self.unc += ((0.12 + 0.3 * w.env["dust_level"]) - self.unc) * min(1.0, dt)
        st = self.state
        if st == "BOOT" and now - self.state_t > 1.0:
            self.set_state("TAKEOFF", now)
        elif st == "TAKEOFF":
            self.alt = min(12.0, self.alt + 4.0 * dt)
            if self.alt >= 12.0:
                self.set_state("SURVEY", now)
                self._next_lane(w)
        elif st == "SURVEY":
            if self.follow(dt, 3.0):
                self._next_lane(w)
        elif st == "FLY_TO_SCAN":
            if self.follow(dt, 12.0):
                self.set_state("SCANNING", now)
                self.orbit_c = tuple(self.req["xy"]) if self.req else (self.x, self.y)
        elif st == "SCANNING":
            a = (now - self.state_t) * 1.6
            cx, cy = self.orbit_c
            self.x, self.y = cx + 3 * math.cos(a), cy + 3 * math.sin(a)
            self.yaw = wrap(a + math.pi / 2)
            self.speed = 4.8
            if now - self.state_t > 4.0 and self.req:
                ci, cj = T.cell(cx, cy)
                sl = T.slope_deg[max(0, ci - 5): ci + 6, max(0, cj - 5): cj + 6]
                mx = max(float(sl.max()) + 9.0, w.limits["max_slope_deg"] + 6)
                verdict = {"verdict": "impassable", "max_slope_deg": round(mx, 1), "mean_slope_deg": round(float(sl.mean()) + 6, 1)}
                w.event({"op": "scan_complete", "id": self.req["id"], "scout": self.id, "xy": list(self.orbit_c), **verdict})
                w.trace(self.id, "SCANNING", "scan_complete", ["CAM", "DEPTH"],
                        f"Nadir depth over r 3 m: max slope {mx:.1f}°, mean {verdict['mean_slope_deg']:.1f}°",
                        f"{mx:.1f}° > {w.limits['max_slope_deg']:.1f}° limit at {w.g:.1f} g", 0.92,
                        "Verdict impassable: confirm permanent no-go zone", "info")
                for r in rovers:
                    if r.id == self.req["robot"] and r.state == "AWAIT_SCOUT":
                        r.resume(w, now, verdict)
                self.req = None
                self.set_state("SURVEY", now)
                self._next_lane(w)
        self.z = ground + self.alt
        if st == "SCANNING":
            self.v_cmd = self.speed
        self.imu_accel(w, now)
        self.power(w, dt)
        if math.hypot(self.trail[-1][0] - self.x, self.trail[-1][1] - self.y) > 1.0:
            self.trail.append([round(self.x, 2), round(self.y, 2)])
            del self.trail[:-300]

    def sensors_json(self, w: World, others: list) -> dict:
        return {"accel": self.accel, "accel_g": round(math.sqrt(sum(v * v for v in self.accel)) / G0, 3), "gyro": self.gyro,
                "roll_deg": round(self.roll, 2), "pitch_deg": round(self.pitch, 2), "lidar_nearest_m": None, "lidar_returns_pct": None,
                "camera_conf": round(max(0.05, 1 - self.unc * 1.2), 2), "v_cmd": round(self.v_cmd, 2), "v_wheel": None,
                "v_ground": round(self.speed, 2), "slip": None, "altitude_agl": round(self.alt, 1)}

    def _next_lane(self, w: World):
        if self.lane_i >= len(self.lanes):
            self.lane_i = 0
        tgt = self.lanes[self.lane_i]
        self.lane_i += 1
        self.goal = [round(tgt[0], 1), round(tgt[1], 1)]
        self.path = World._resample([(self.x, self.y), tuple(tgt)], 1.5)
        if self.lane_i % 2 == 0:
            w.trace(self.id, "SURVEY", "frontier_selected", ["MAP", "FLEET"],
                    f"Survey lane {self.lane_i // 2} of {len(self.lanes) // 2}; strip {w.stats()['percent']:.0f} % mapped overall",
                    "Largest unmapped strip not claimed by a rover within 8 m", 0.84,
                    "Flying lawnmower lane at 12 m AGL, 3.0 m/s")
