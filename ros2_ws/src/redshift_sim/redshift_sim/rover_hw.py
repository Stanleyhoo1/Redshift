"""Simulated rover hardware shared by both world backends (pure; no ROS): drivetrain health, motor-controller
currents, the solar panel, the debris-strike profile (docs/protocol.md §8.2, §8.3, §8.5) and the dust-storm wind
on the drone.

The drivetrain fault is applied by the *world*, before the wheels: the mock scales the command inside its dynamics,
Gazebo scales it in world_interface_node's `cmd_vel` → `cmd_vel_hw` relay. The autonomy only learns about it from
its sensors (`hw_status` motor-controller diagnostics, commanded vs. measured wheel speed, the IMU shock).

Model, kept deliberately simple:

* Drivetrain: per-wheel health h ∈ [0, 1] (FL, FR, RL, RR) = fraction of the commanded wheel speed the actuator
  delivers. A skid-steer command (v, w) is split into left/right wheel surface speeds v ∓ w·track/2, each side is
  scaled by the mean health of its two wheels, and the result recombined, so a damaged rover both crawls and veers
  toward its weaker side.
* Motor current (24 V bus): idle draw + a term proportional to the commanded wheel speed and the yaw rate; a damaged
  (binding) wheel draws more current per unit of output, an OVERCURRENT wheel sits near the stall current whenever
  it is commanded (high current, almost no output), an ENCODER_LOST wheel drops to the driver's idle draw.
* Solar panel on the chassis deck: P = 220 W × irradiance × max(0, n̂·ŝ), with irradiance = dust transmittance
  1 − 0.85·dust_level, n̂ the chassis up axis and ŝ the sun direction (elevation 55° from the contract, azimuth of
  the rendered sun, world.sdf / mock_render SUN_DIR). A level rover therefore gets sin 55° = 0.82 of peak.
* Debris strike, mock: a half-sine shock of STRIKE['duration_s'] — specific force `vertical_g` up and `lateral_g`
  sideways (body frame, g0 units, on top of gravity: |a| peaks ≈ 5.1 g above the mission gravity at 2.2 g) and a
  small roll kick (≈ 5° peak, well below the 28.8° tip alarm and the 16.9° gyro-spike gate: tip recovery never fires).
* Debris strike, Gazebo (strike_wrenches_world): an accelerometer only reads what the body really accelerates, and a
  rigid-contact touchdown is resolved in one 4 ms step, so *any* lift-off lands as a random 5–30 g single-step spike
  (and the airborne |a| ≈ 0 samples chain it into the same shock episode for rover_health). The Gazebo strike
  therefore never lifts a wheel: for `gz_duration_s` a force along the chassis normal carries (1 − gz_reserve) of the
  weight, so the wheels barely grip, while a lateral push along the chassis y axis — +L for the first half, −L for
  the second (net Δv ≈ 0, ≈ 3 cm sideways and back) — gives the same |a| peak as the mock: √((g·cos tilt)² + L²).
  No roll torque, and the push is applied at the centre of mass (force_offset): lifting one side would land the
  same way. Persistent wrenches accumulate in gz-sim (ApplyLinkWrench), so the mid-pulse switch is published as a
  delta wrench and the end as /wrench/clear.
* Storm wind (the drone only): a power-law shear profile, wind_at(h) = wind_ref·clip(h / 12 m, 0.05, 2)^0.6, so lower
  is calmer, along a fixed world bearing per storm with ±25 % gusts at 0.3 Hz (StormWind, seeded phase). The world
  adds it to the drone's commanded velocity; the drone only learns it from its own odometry (cmd vs. measured).
"""
from __future__ import annotations

import math

import numpy as np

G0 = 9.80665
WHEELS = ('FL', 'FR', 'RL', 'RR')
LEFT, RIGHT = (0, 2), (1, 3)
TRACK_M = 0.85                       # robot_models.ROVER_SPEC['track_m']

# --- drivetrain ------------------------------------------------------------------------------------------------
DAMAGED_HEALTH = (0.08, 0.35, 0.05, 0.30)
DAMAGE_FAULTS = ('DRIVE_FL_OVERCURRENT', 'DRIVE_FR_TORQUE_DERATED', 'DRIVE_RL_ENCODER_LOST',
                 'DRIVE_RR_TORQUE_DERATED')
REPAIR_S = 4.0
REPAIR_RANGE_M = 1.6                 # true centre distance (xy) required for a repair

# --- motor controllers (24 V bus) ------------------------------------------------------------------------------
I_IDLE_A = 0.4
I_PER_MPS = 3.4                      # per wheel, per m/s of commanded wheel surface speed (≈3.1 A at 0.8 m/s)
I_PER_RADPS = 1.2                    # skid-steer scrub, per rad/s of commanded yaw rate
I_STALL_A = 11.5                     # binding/shorted motor commanded to move
I_LIMIT_A = 12.0                     # controller current limit
I_NOISE_A = 0.05

# --- solar panel -------------------------------------------------------------------------------------------------
SOLAR_PEAK_W = 220.0
SUN_ELEVATION_DEG = 55.0
SUN_AZIMUTH_DEG = math.degrees(math.atan2(-0.35, 0.55))   # toward the sun: −SUN_DIR of mock_render / world.sdf
DUST_EXTINCTION = 0.85

# --- storm wind (drone) -----------------------------------------------------------------------------------------
WIND_SHEAR_EXP = 0.6                 # power-law wind-shear exponent (rough terrain, stormy boundary layer)
WIND_REF_AGL_M = 12.0                # wind_mps of /world/env_state is the wind at this height (the drone's cruise AGL)
WIND_AGL_CLIP = (0.05, 2.0)          # h / WIND_REF_AGL_M is clipped to this range (no calm at 0 m, no runaway aloft)
STORM_WIND_TOWARD_DEG = 30.0         # world direction (ENU, ccw from +x east) the storm blows toward: compass 060°,
                                     # i.e. a wind "from 240°" (west-south-west), as the drone's trace reports it
GUST_FRAC = 0.25                     # gust amplitude, fraction of the mean wind
GUST_HZ = 0.3                        # gust frequency

# --- debris strike (critical_damage) -------------------------------------------------------------------------
STRIKE = {
    'duration_s': 0.12,              # mock half-sine shock
    'vertical_g': 5.0,               # body +z specific force peak on top of gravity, g0 units
    'lateral_g': 1.2,                # body +y specific force peak, g0 units  → |a| ≈ 5.1 g above gravity at 2.2 g
    'roll_accel': 9.5,               # rad/s² peak roll acceleration into the chassis roll mode → ≈ 5° roll kick
    'settle_s': 0.05,                # keep measuring the peak this long after the shock
    # Gazebo: two-phase wrench on <rover>::base_link (persistent wrench, delta at half time, clear; sim-timed)
    'gz_duration_s': 0.04,           # 10 physics steps of 4 ms: ≥ 3 IMU samples (100 Hz) inside the pulse
    'gz_reserve': 0.08,              # fraction of the normal load left on the wheels during the pulse (no lift-off)
    'gz_measure_s': 0.5,             # sim seconds after the start over which the IMU peak is taken
}


def strike_shape(t_rel: float) -> float:
    """Half-sine 0 → 1 → 0 over STRIKE['duration_s'] (0 outside)."""
    T = STRIKE['duration_s']
    return math.sin(math.pi * t_rel / T) if 0.0 <= t_rel <= T else 0.0


def strike_specific_force(t_rel: float) -> np.ndarray:
    """Extra body-frame specific force (m/s²) the IMU measures during the mock strike."""
    s = strike_shape(t_rel)
    return np.array([0.0, STRIKE['lateral_g'], STRIKE['vertical_g']]) * G0 * s


def strike_peak_accel(g_mult: float) -> float:
    """|a| (m/s²) at the peak of the mock strike on level ground — the target for both backends."""
    return G0 * math.hypot(STRIKE['lateral_g'], g_mult + STRIKE['vertical_g'])


def strike_wrenches_world(roll: float, pitch: float, yaw: float, g_mult: float = 2.2, mass_kg: float = 60.0):
    """Gazebo strike (world-frame forces at the base link, see the module docstring): returns (first, switch), each
    (force[3], torque[3]); `first` is published at the start, `switch` (a delta: persistent wrenches accumulate)
    at half time, then the wrenches are cleared."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    y_b = np.array([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr])      # chassis axes in the world
    z_b = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])
    cos_tilt = max(0.0, float(z_b[2]))
    weight_n = mass_kg * G0 * g_mult
    lift = (1.0 - STRIKE['gz_reserve']) * weight_n * cos_tilt * z_b            # carries the normal load
    lat_ms2 = math.sqrt(max(0.0, strike_peak_accel(g_mult) ** 2 - (G0 * g_mult * cos_tilt) ** 2))
    push = mass_kg * lat_ms2 * y_b
    zero = [0.0, 0.0, 0.0]
    return ([float(v) for v in lift + push], zero), ([float(v) for v in -2.0 * push], zero)


def peak_g_above_gravity(accel_peak_ms2: float, g_mult: float) -> float:
    """`peak_g` of the critical_damage event: how far the measured |a| peak rose above the mission gravity, in g0."""
    return round(max(0.0, accel_peak_ms2 / G0 - g_mult), 2)


# --- solar -------------------------------------------------------------------------------------------------------
def _sun_vector() -> np.ndarray:
    el, az = math.radians(SUN_ELEVATION_DEG), math.radians(SUN_AZIMUTH_DEG)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


SUN_VEC = _sun_vector()


def solar_output(dust_level: float, roll_rad: float, pitch_rad: float, yaw_rad: float = 0.0) -> tuple[float, float]:
    """(solar_w, irradiance): panel output and the dust transmittance 1 − 0.85·dust_level.

    The panel normal is the chassis up axis Rz(yaw)·Ry(pitch)·Rx(roll)·ẑ; its incidence factor is max(0, n̂·ŝ).
    """
    irr = float(np.clip(1.0 - DUST_EXTINCTION * float(np.clip(dust_level, 0.0, 1.0)), 0.0, 1.0))
    cr, sr, cp, sp = math.cos(roll_rad), math.sin(roll_rad), math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    n = (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr)
    inc = max(0.0, n[0] * SUN_VEC[0] + n[1] * SUN_VEC[1] + n[2] * SUN_VEC[2])
    return SOLAR_PEAK_W * irr * inc, irr


# --- storm wind ----------------------------------------------------------------------------------------------------
def wind_at(h_agl: float, wind_ref_mps: float) -> float:
    """Mean wind speed (m/s) at h_agl metres above ground: wind_ref·clip(h / WIND_REF_AGL_M, 0.05, 2)^WIND_SHEAR_EXP."""
    lo, hi = WIND_AGL_CLIP
    return float(wind_ref_mps) * min(hi, max(lo, float(h_agl) / WIND_REF_AGL_M)) ** WIND_SHEAR_EXP


class StormWind:
    """World-frame wind of one storm: wind_at shear along a fixed bearing, times a gust factor 1 ± GUST_FRAC at
    GUST_HZ whose phase is seeded per storm (deterministic for a given seed)."""

    def __init__(self, seed: int = 0, toward_deg: float = STORM_WIND_TOWARD_DEG, gust_frac: float = GUST_FRAC):
        self.toward = math.radians(toward_deg)
        self.gust_frac = float(gust_frac)
        self.phase = float(np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi))

    def gust(self, t: float) -> float:
        """Gust factor at simulated time t."""
        return 1.0 + self.gust_frac * math.sin(2.0 * math.pi * GUST_HZ * float(t) + self.phase)

    def vector(self, t: float, h_agl: float, wind_ref_mps: float) -> tuple[float, float]:
        """(wx, wy) m/s, world frame, at h_agl above ground and time t."""
        if wind_ref_mps <= 0.0:
            return 0.0, 0.0
        v = wind_at(h_agl, wind_ref_mps) * self.gust(t)
        return v * math.cos(self.toward), v * math.sin(self.toward)


# --- drivetrain --------------------------------------------------------------------------------------------------
class Drivetrain:
    """Per-rover drivetrain health: critical damage, timed repair, command scaling and motor currents."""

    def __init__(self, track_m: float = TRACK_M):
        self.track_m = float(track_m)
        self.wheel_health = [1.0, 1.0, 1.0, 1.0]
        self.faults: list[str] = []
        self._repair: tuple[float, float, list[float]] | None = None    # (t0, duration, health at start)

    # state
    @property
    def damaged(self) -> bool:
        return bool(self.faults) or min(self.wheel_health) < 1.0

    @property
    def repairing(self) -> bool:
        return self._repair is not None

    @property
    def health_pct(self) -> float:
        return 100.0 * sum(self.wheel_health) / 4.0

    def side_health(self) -> tuple[float, float]:
        h = self.wheel_health
        return (h[LEFT[0]] + h[LEFT[1]]) / 2.0, (h[RIGHT[0]] + h[RIGHT[1]]) / 2.0

    # events
    def apply_critical_damage(self) -> None:
        """Debris strike: drive actuators lose most of their output; cancels any repair in progress."""
        self.wheel_health = list(DAMAGED_HEALTH)
        self.faults = list(DAMAGE_FAULTS)
        self._repair = None

    def start_repair(self, t: float, duration_s: float = REPAIR_S) -> bool:
        """Ramp every wheel linearly back to 1.0 over `duration_s` (sim) from time t. False if already repairing."""
        if self._repair is not None:
            return False
        self._repair = (float(t), max(1e-3, float(duration_s)), list(self.wheel_health))
        return True

    def update(self, t: float) -> bool:
        """Advance a repair to time t; True exactly once, when it completes (health 1.0, faults cleared)."""
        if self._repair is None:
            return False
        t0, dur, h0 = self._repair
        frac = min(1.0, max(0.0, (float(t) - t0) / dur))
        if frac >= 1.0:
            self.wheel_health = [1.0, 1.0, 1.0, 1.0]
            self.faults = []
            self._repair = None
            return True
        self.wheel_health = [h + (1.0 - h) * frac for h in h0]
        return False

    # physics
    def scale(self, v: float, w: float, track_m: float | None = None) -> tuple[float, float]:
        """What the wheels deliver for a commanded (v, w): each side's wheel speed × its mean health."""
        hl, hr = self.side_health()
        if hl >= 1.0 and hr >= 1.0:
            return float(v), float(w)
        tr = self.track_m if track_m is None else float(track_m)
        left = (v - w * tr / 2.0) * hl
        right = (v + w * tr / 2.0) * hr
        return (left + right) / 2.0, (right - left) / tr

    def motor_current(self, v_cmd: float, w_cmd: float, rng: np.random.Generator | None = None) -> list[float]:
        """Per-wheel motor current (A) for the commanded (v, w), FL, FR, RL, RR."""
        tr = self.track_m
        side = (v_cmd - w_cmd * tr / 2.0, v_cmd + w_cmd * tr / 2.0)
        out = []
        for k, name in enumerate(WHEELS):
            u = abs(side[0 if k in LEFT else 1])
            h = self.wheel_health[k]
            if f'DRIVE_{name}_OVERCURRENT' in self.faults:
                i = I_IDLE_A + I_STALL_A * min(1.0, u / 0.1)
            elif f'DRIVE_{name}_ENCODER_LOST' in self.faults:
                i = I_IDLE_A
            else:                                   # a damaged gearbox draws more current per unit of output
                i = I_IDLE_A + (I_PER_MPS * u + I_PER_RADPS * abs(w_cmd)) * (1.0 + 1.5 * (1.0 - h))
            if rng is not None:
                i += I_NOISE_A * float(rng.standard_normal())
            out.append(float(np.clip(i, 0.0, I_LIMIT_A)))
        return out


def hw_status_dict(robot: str, t: float, drivetrain: Drivetrain, solar_w: float, irradiance: float,
                   currents) -> dict:
    """The /<rover>/hw_status payload (docs/protocol.md §8.3). `t` is wall-clock epoch seconds (§2.1)."""
    return {'robot': robot, 't': round(float(t), 3),
            'wheel_health': [round(float(h), 3) for h in drivetrain.wheel_health],
            'faults': list(drivetrain.faults),
            'motor_current_a': [round(float(i), 2) for i in currents],
            'solar_w': round(float(solar_w), 1), 'irradiance': round(float(irradiance), 3)}


def planar_distance(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
