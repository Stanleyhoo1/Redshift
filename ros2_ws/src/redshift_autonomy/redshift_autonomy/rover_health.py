"""On-board health estimation from the robot's own sensors (docs/protocol.md §8.7).

A rover never reads the world's damage state. It estimates its own condition from:

* **IMU** — shocks: |a| more than `SHOCK_G` *above* the mission gravity for ≥ 2 consecutive samples is an
  impact; structural damage grows with how far the measured peak exceeds the threshold. A reading below gravity
  means the chassis is partly airborne (a tip, a bounce): that is the tip detector's business, not an impact, and
  a single-sample spike as the wheels land again is rigid-contact chatter (rejected by the 2-sample rule). Shock detection arms only
  after the accelerometer has read steady gravity for `SHOCK_ARM_SAMPLES` samples: a starting IMU is not an impact
  (Gazebo's reads 0 until it initialises, then spikes ~40 g on its first physics step). Sustained tilt beyond the
  tip alarm adds a little fatigue damage.
* **Motor-controller diagnostics** (`/<r>/hw_status`) — per-wheel delivered-torque fraction and fault codes,
  cross-checked against commanded vs. measured wheel speed (wheel odometry) to flag a stall even without them.
* **Power** — battery state of charge by coulomb counting: solar panel output (hw_status) minus avionics,
  edge compute (the edge budget's power estimate) and drive power (the physics model's energy per metre).

`pct = structure_pct × drive_pct / 100`; below `CRITICAL_PCT` the rover is critical and stays so (latched)
until the estimate is back above `RECOVERED_PCT`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .physics_model import RoverPhysics

G0 = 9.80665
CRITICAL_PCT = 15.0
RECOVERED_PCT = 60.0
SHOCK_G = 1.2                    # |a| above mission gravity (g) that counts as an impact
SHOCK_MIN_SAMPLES = 2            # consecutive IMU samples above threshold (rejects single-sample contact glitches)
SHOCK_END_SAMPLES = 3
SHOCK_DAMAGE_PER_G = 12.0        # structure % lost per g of peak above the threshold
SHOCK_ARM_SAMPLES = 50           # consecutive at-gravity samples (0.5 s at 100 Hz) before shocks count (IMU boot)
TILT_DAMAGE_PER_DEG_S = 0.25     # structure % per (degree beyond the tip alarm × second)
STALL_RATIO = 0.5                # measured / commanded wheel speed below this while commanding = stall
STALL_MIN_CMD = 0.2
STALL_S = 1.5
STALL_CLEAR_S = 1.0
FIELD_REPAIR_STRUCTURE_PCT = 92.0
ROVER_CAPACITY_WH = 600.0
ROVER_AVIONICS_W = 25.0
DRONE_CAPACITY_WH = 250.0
DRONE_AVIONICS_W = 15.0


@dataclass
class HealthState:
    pct: float
    structure_pct: float
    drive_pct: float
    battery_pct: float
    battery_wh: float
    capacity_wh: float
    solar_w: float
    load_w: float
    net_w: float
    faults: list = field(default_factory=list)
    critical: bool = False
    shocks: int = 0
    peak_g: float = 0.0              # largest shock this mission (g above gravity)
    last_shock_g: float = 0.0        # the most recent shock's peak

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d


class _Battery:
    def __init__(self, capacity_wh: float, pct: float):
        self.capacity = float(capacity_wh)
        self.wh = self.capacity * min(100.0, max(0.0, pct)) / 100.0
        self.solar_w = 0.0
        self.load_w = 0.0

    def integrate(self, dt: float, load_w: float):
        self.load_w = load_w
        self.wh = min(self.capacity, max(0.0, self.wh + (self.solar_w - load_w) * dt / 3600.0))

    @property
    def pct(self) -> float:
        return 100.0 * self.wh / self.capacity


class RoverHealth:
    def __init__(self, robot: str, g_mult: float, battery_pct: float = 80.0, capacity_wh: float = ROVER_CAPACITY_WH,
                 tip_alarm_deg: float | None = None, physics: RoverPhysics | None = None):
        self.robot, self.g = robot, float(g_mult)
        self.phys = physics or RoverPhysics()
        self.tip_alarm = tip_alarm_deg if tip_alarm_deg is not None else self.phys.limits(self.g).tip_alarm_deg
        self.bat = _Battery(capacity_wh, battery_pct)
        self.structure = 100.0
        self.wheel_health: list | None = None
        self.hw_faults: list = []
        self.measured_drive = 1.0          # EMA of wheel/commanded speed (used only without diagnostics)
        self.stall = False
        self._stall_since = self._ok_since = None
        self.shocks = 0
        self.peak_g = 0.0
        self.last_shock_g = 0.0
        self._run = self._below = 0            # consecutive samples above / below the shock threshold
        self._run_peak = 0.0                   # peak of the current above-threshold run
        self._qualified = False                # this episode held >= SHOCK_MIN_SAMPLES consecutive samples
        self._episode_peak = 0.0               # peak over the qualifying runs only (a lone spike never sets it)
        self._settled = 0                  # consecutive at-gravity samples while the shock detector is unarmed
        self._last_imu_t = None
        self.critical = False

    # ---------------------------------------------------------------- sensor inputs
    @property
    def shock_armed(self) -> bool:
        return self._settled >= SHOCK_ARM_SAMPLES

    def on_imu(self, t: float, accel_norm: float, tilt_deg: float):
        dev = accel_norm / G0 - self.g              # specific force above gravity, g0
        if not self.shock_armed:
            self._settled = self._settled + 1 if abs(dev) <= SHOCK_G else 0
        elif dev > SHOCK_G:
            self._run += 1
            self._below = 0
            self._run_peak = max(self._run_peak, dev)
            if self._run >= SHOCK_MIN_SAMPLES:              # sustained: this run is an impact
                self._qualified = True
                self._episode_peak = max(self._episode_peak, self._run_peak)
        else:
            self._run, self._run_peak = 0, 0.0
            self._below += 1
            if self._below == SHOCK_END_SAMPLES:          # episode over
                if self._qualified:
                    self.shocks += 1
                    self.peak_g = max(self.peak_g, self._episode_peak)
                    self.last_shock_g = self._episode_peak
                    self._damage(SHOCK_DAMAGE_PER_G * (self._episode_peak - SHOCK_G))
                self._qualified, self._episode_peak = False, 0.0
        dt = 0.0 if self._last_imu_t is None else min(0.05, max(0.0, t - self._last_imu_t))
        self._last_imu_t = t
        if tilt_deg > self.tip_alarm:
            self._damage(TILT_DAMAGE_PER_DEG_S * (tilt_deg - self.tip_alarm) * dt)

    def on_hw_status(self, t: float, wheel_health, faults, solar_w: float):
        self.wheel_health = [min(1.0, max(0.0, float(v))) for v in wheel_health]
        self.hw_faults = [str(f) for f in faults]
        self.bat.solar_w = max(0.0, float(solar_w))

    def on_drive(self, t: float, v_cmd: float, v_wheel: float):
        if abs(v_cmd) < STALL_MIN_CMD:
            return
        ratio = min(1.0, max(0.0, v_wheel / v_cmd))
        self.measured_drive += 0.2 * (ratio - self.measured_drive)
        if ratio < STALL_RATIO:
            self._ok_since = None
            self._stall_since = t if self._stall_since is None else self._stall_since
            if t - self._stall_since >= STALL_S - 1e-9:
                self.stall = True
        else:
            self._stall_since = None
            self._ok_since = t if self._ok_since is None else self._ok_since
            if t - self._ok_since >= STALL_CLEAR_S - 1e-9:
                self.stall = False

    def step(self, t: float, dt: float, v_ground: float, pitch_rad: float, compute_w: float):
        """Integrate the battery over `dt` seconds of driving at `v_ground` (m/s) on the current grade."""
        grade = math.tan(-pitch_rad) * (1.0 if v_ground >= 0 else -1.0)
        drive_w = self.phys.energy_per_m(self.g, grade, 1.0) * abs(v_ground)
        self.bat.integrate(dt, ROVER_AVIONICS_W + max(0.0, compute_w) + drive_w)

    def repaired(self, t: float, by: str):
        """A teammate completed a field repair (structure patched; the drivetrain is verified via diagnostics)."""
        self.structure = max(self.structure, FIELD_REPAIR_STRUCTURE_PCT)
        self.measured_drive = 1.0
        self.stall = False

    # ---------------------------------------------------------------- estimate
    def _damage(self, pct: float):
        self.structure = max(0.0, self.structure - max(0.0, pct))

    @property
    def drive_pct(self) -> float:
        if self.wheel_health:
            return 100.0 * sum(self.wheel_health) / len(self.wheel_health)
        return 100.0 * self.measured_drive

    def state(self) -> HealthState:
        pct = self.structure * self.drive_pct / 100.0
        if pct < CRITICAL_PCT:
            self.critical = True
        elif pct >= RECOVERED_PCT:
            self.critical = False
        faults = list(self.hw_faults) + (['DRIVE_STALL_DETECTED'] if self.stall else [])
        b = self.bat
        return HealthState(pct=pct, structure_pct=self.structure, drive_pct=self.drive_pct, battery_pct=b.pct,
                           battery_wh=b.wh, capacity_wh=b.capacity, solar_w=b.solar_w, load_w=b.load_w,
                           net_w=b.solar_w - b.load_w, faults=faults, critical=self.critical, shocks=self.shocks,
                           peak_g=self.peak_g, last_shock_g=self.last_shock_g)


def drone_hover_power_w(mass_kg: float, g_mult: float, rho: float = 1.2, rotor_area_m2: float = 0.18,
                        figure_of_merit: float = 0.6) -> float:
    """Momentum-theory hover power: T^1.5 / sqrt(2 ρ A) / FoM, with T = m·g."""
    thrust = mass_kg * G0 * g_mult
    return thrust ** 1.5 / math.sqrt(2.0 * rho * rotor_area_m2) / figure_of_merit


class DroneHealth:
    def __init__(self, robot: str, g_mult: float, battery_pct: float = 95.0, capacity_wh: float = DRONE_CAPACITY_WH,
                 mass_kg: float = 2.0):
        self.robot, self.g, self.mass = robot, float(g_mult), float(mass_kg)
        self.bat = _Battery(capacity_wh, battery_pct)

    def step(self, t: float, dt: float, airborne: bool, compute_w: float, rho: float = 1.2):
        hover = drone_hover_power_w(self.mass, self.g, rho) if airborne else 0.0
        self.bat.integrate(dt, DRONE_AVIONICS_W + max(0.0, compute_w) + hover)

    def state(self) -> HealthState:
        b = self.bat
        return HealthState(pct=100.0, structure_pct=100.0, drive_pct=100.0, battery_pct=b.pct, battery_wh=b.wh,
                           capacity_wh=b.capacity, solar_w=0.0, load_w=b.load_w, net_w=-b.load_w)
