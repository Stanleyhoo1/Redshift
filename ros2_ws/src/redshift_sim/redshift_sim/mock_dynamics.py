"""Mock-world robot dynamics, matching what the Gazebo models do closely enough for the autonomy stack:

* Rover: DiffDrive command limits and acceleration limits, terrain following (z, roll, pitch from the four
  wheel contacts), the same torque-limited stall `physics_model` predicts (it is what the Gazebo wheel effort
  limit produces), loose-regolith slip on steep grades, boulders block the chassis while the wheels keep
  turning, and the subsidence jolt as a second-order roll mode driven closed-loop to a peak just past the tip
  alarm (like world_interface_node does in Gazebo).
* Scout: VelocityControl-like body-frame velocity commands with a first-order lag, resting on its skids.
"""
from __future__ import annotations

import math

import numpy as np

from redshift_autonomy.physics_model import RoverPhysics

from .mock_render import rpy_matrix
from .scene_events import JOLT

G0 = 9.80665
# DiffDrive limits written into the rover SDF (robot_models.rover_sdf)
V_LIM, W_LIM = 1.2, 1.5
ACC_UP, ACC_DOWN, W_ACC = 1.5, 2.0, 4.0
ATTITUDE_TAU_S = 0.08
FOOTPRINT_R = 0.5
# roll mode of the chassis on its suspension/tyres, and the jolt's drive
ROLL_W = 5.0                  # rad/s
ROLL_ZETA = 0.4
JOLT_ACCEL = 40.0             # rad/s² of roll acceleration while the subsidence torque is applied
SKID_Z = 0.05                 # scout base height above ground when landed
SCOUT_TAU_S, SCOUT_YAW_TAU_S = 0.25, 0.2


class RoverSim:
    def __init__(self, name: str, spec: dict, spawn, g_mult: float, rng: np.random.Generator | None = None):
        self.name, self.spec, self.g = name, spec, float(g_mult)
        self.lim = RoverPhysics.from_spec(spec).limits(self.g)
        x, y, _, yaw = spawn
        self.x, self.y, self.yaw, self.yaw0 = float(x), float(y), float(yaw), float(yaw)
        self.z = 0.0
        self.cmd = (0.0, 0.0)
        self.v_drive = self.w_drive = 0.0
        self.v_wheel = self.v_true = self._v_prev = 0.0
        self.roll_t = self.pitch_t = 0.0
        self.roll = self.pitch = 0.0
        self.roll_rate = self.pitch_rate = 0.0
        self.phi = self.phid = 0.0
        self.jolting = False
        self._jolt_t = 0.0
        self._events: list[dict] = []
        self.wx = self.wy = self.wyaw = 0.0
        self.accel = np.array([0.0, 0.0, self.g * G0])
        self.rng = rng or np.random.default_rng(0)
        self._init = False

    # ------------------------------------------------------------------ terrain contact
    def _contacts(self, scene, x, y, yaw):
        hb, ht = self.spec['wheelbase_m'] / 2.0, self.spec['track_m'] / 2.0
        c, s = math.cos(yaw), math.sin(yaw)
        bx = np.array([hb, hb, -hb, -hb])
        by = np.array([ht, -ht, ht, -ht])
        h = scene.height_at(x + c * bx - s * by, y + s * bx + c * by)     # FL, FR, RL, RR
        front, rear = (h[0] + h[1]) / 2, (h[2] + h[3]) / 2
        left, right = (h[0] + h[2]) / 2, (h[1] + h[3]) / 2
        pitch = -math.atan2(front - rear, self.spec['wheelbase_m'])
        roll = math.atan2(left - right, self.spec['track_m'])
        return float(h.mean()), roll, pitch

    def start_jolt(self) -> bool:
        if self.jolting:
            return False
        self.jolting, self._jolt_t = True, 0.0
        return True

    def pop_events(self) -> list[dict]:
        ev, self._events = self._events, []
        return ev

    def _predicted_peak(self) -> float:
        phi, phid, dt = self.phi, self.phid, 0.004
        for _ in range(400):
            if phid <= 0:
                break
            phid += (-ROLL_W ** 2 * phi - 2 * ROLL_ZETA * ROLL_W * phid) * dt
            phi += phid * dt
        return phi

    # ------------------------------------------------------------------ integration
    def step(self, dt: float, scene, t: float) -> None:
        if not self._init:
            _, self.roll_t, self.pitch_t = self._contacts(scene, self.x, self.y, self.yaw)
            self._init = True
        v_cmd = float(np.clip(self.cmd[0], -V_LIM, V_LIM))
        w_cmd = float(np.clip(self.cmd[1], -W_LIM, W_LIM))
        self.v_drive += float(np.clip(v_cmd - self.v_drive, -ACC_DOWN * dt, ACC_UP * dt))
        self.w_drive += float(np.clip(w_cmd - self.w_drive, -W_ACC * dt, W_ACC * dt))

        # grade in the direction of travel → torque stall and slip
        grade = math.degrees(-self.pitch_t) * (1.0 if self.v_drive >= 0 else -1.0)
        torque_ok = float(np.clip((self.lim.drive_deg - grade) / 3.0, 0.0, 1.0))
        slip = 0.8 * float(np.clip((grade - 0.6 * self.lim.slip_deg) / (0.4 * self.lim.slip_deg), 0.0, 1.0))
        self.v_wheel = self.v_drive * torque_ok
        v_true = self.v_wheel * (1.0 - slip)

        yaw_new = self.yaw + self.w_drive * dt
        xn = self.x + v_true * math.cos(yaw_new) * dt
        yn = self.y + v_true * math.sin(yaw_new) * dt
        edge = scene.half - 0.6
        if (abs(xn) > edge or abs(yn) > edge or
                (scene.blocked(xn, yn, FOOTPRINT_R) and not scene.blocked(self.x, self.y, FOOTPRINT_R))):
            xn, yn, v_true = self.x, self.y, 0.0
        self.x, self.y, self.yaw = xn, yn, (yaw_new + math.pi) % (2 * math.pi) - math.pi
        self.v_true = v_true

        hmean, roll_c, pitch_c = self._contacts(scene, self.x, self.y, self.yaw)
        a = min(1.0, dt / ATTITUDE_TAU_S)
        self.roll_t += (roll_c - self.roll_t) * a
        self.pitch_t += (pitch_c - self.pitch_t) * a
        self.z = hmean + self.spec['base_height_m']

        # subsidence jolt: roll mode driven until the predicted peak reaches the target, then released
        u = 0.0
        if self.jolting:
            self._jolt_t += dt
            u = JOLT_ACCEL
        self.phid += (u - ROLL_W ** 2 * self.phi - 2 * ROLL_ZETA * ROLL_W * self.phid) * dt
        self.phi += self.phid * dt
        if self.jolting:
            target = math.radians(self.lim.tip_alarm_deg + JOLT['peak_over_alarm_deg'])
            if abs(self.roll_t + self._predicted_peak()) >= target or self._jolt_t >= JOLT['max_duration_s']:
                self.jolting = False
                self._events.append({'op': 'jolt', 'ok': True, 'robot': self.name,
                                     'release_roll_deg': round(math.degrees(abs(self.roll_t + self.phi)), 1)})

        roll_new, pitch_new = self.roll_t + self.phi, self.pitch_t
        self.roll_rate = (roll_new - self.roll) / dt
        self.pitch_rate = (pitch_new - self.pitch) / dt
        self.roll, self.pitch = roll_new, pitch_new

        self.wyaw += self.w_drive * dt
        self.wx += self.v_wheel * math.cos(self.wyaw) * dt
        self.wy += self.v_wheel * math.sin(self.wyaw) * dt

        R = rpy_matrix(self.roll, self.pitch, self.yaw)
        a_fwd = (self.v_true - self._v_prev) / dt
        self._v_prev = self.v_true
        vib = 0.03 * min(1.0, abs(self.v_true)) * self.rng.standard_normal(3)
        self.accel = R.T @ np.array([0.0, 0.0, self.g * G0]) + np.array([a_fwd, self.v_true * self.w_drive, 0.0]) + vib

    def snapshot(self) -> dict:
        return {'x': self.x, 'y': self.y, 'z': self.z, 'yaw': self.yaw, 'roll': self.roll, 'pitch': self.pitch,
                'v_wheel': self.v_wheel, 'v_true': self.v_true, 'w': self.w_drive,
                'gyro': (self.roll_rate, self.pitch_rate, self.w_drive), 'accel': tuple(self.accel),
                'wheel_pose': (self.wx, self.wy, self.wyaw), 'imu_yaw': self.yaw - self.yaw0}


class ScoutSim:
    def __init__(self, name: str, spec: dict, spawn, g_mult: float):
        self.name, self.spec, self.g = name, spec, float(g_mult)
        x, y, z, yaw = spawn
        self.x, self.y, self.z, self.yaw, self.yaw0 = float(x), float(y), float(z), float(yaw), float(yaw)
        self.cmd = (0.0, 0.0, 0.0, 0.0)          # body vx, vy, vz, yaw rate
        self.vel = np.zeros(3)
        self.w = 0.0
        self.accel = np.array([0.0, 0.0, self.g * G0])

    def vel_world(self) -> np.ndarray:
        return self.vel.copy()

    def step(self, dt: float, scene, t: float) -> None:
        vmax, vzmax = self.spec.get('v_max', 4.0), self.spec.get('vz_max', 2.0)
        vx, vy, vz, wz = self.cmd
        sp = math.hypot(vx, vy)
        if sp > vmax:
            vx, vy = vx * vmax / sp, vy * vmax / sp
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        target = np.array([c * vx - s * vy, s * vx + c * vy, float(np.clip(vz, -vzmax, vzmax))])
        prev = self.vel.copy()
        self.vel += (target - self.vel) * min(1.0, dt / SCOUT_TAU_S)
        self.w += (wz - self.w) * min(1.0, dt / SCOUT_YAW_TAU_S)
        ground = float(scene.height_at(self.x, self.y)) + SKID_Z
        if self.z - ground < 0.1 and target[2] <= 0.0:          # on (or settling onto) its skids
            self.vel[:] = 0.0 if abs(target[0]) + abs(target[1]) < 1e-3 else self.vel
            self.vel[2] = 0.0
            self.z = ground
        self.x += self.vel[0] * dt
        self.y += self.vel[1] * dt
        self.z = max(ground, self.z + self.vel[2] * dt)
        if self.z <= ground:
            self.vel[2] = max(0.0, self.vel[2])
        edge = scene.half - 0.5
        self.x, self.y = float(np.clip(self.x, -edge, edge)), float(np.clip(self.y, -edge, edge))
        self.yaw = (self.yaw + self.w * dt + math.pi) % (2 * math.pi) - math.pi
        a_w = (self.vel - prev) / dt + np.array([0.0, 0.0, self.g * G0])
        self.accel = np.array([c * a_w[0] + s * a_w[1], -s * a_w[0] + c * a_w[1], a_w[2]])

    def snapshot(self) -> dict:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        v = self.vel
        return {'x': self.x, 'y': self.y, 'z': self.z, 'yaw': self.yaw, 'roll': 0.0, 'pitch': 0.0,
                'v_body': (c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]), 'gyro': (0.0, 0.0, self.w),
                'accel': tuple(self.accel), 'imu_yaw': self.yaw - self.yaw0}
