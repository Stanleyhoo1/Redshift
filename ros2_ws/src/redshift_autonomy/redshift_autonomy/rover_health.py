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
  `PowerSaver` switches the rover into battery conservation when its own pyranometer / panel readings collapse
  (a dust storm): it never reads the world's dust level.

`pct = structure_pct × drive_pct / 100`; below `CRITICAL_PCT` the rover is critical and stays so (latched)
until the estimate is back above `RECOVERED_PCT`.
"""
from __future__ import annotations

import collections
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
DRONE_WIND_W_PER_MPS2 = 0.25     # extra hover power fighting the wind: 0.25 W per (m/s)² of estimated wind
# battery conservation (PowerSaver), from the rover's own hw_status solar_w + irradiance, simulated seconds
PS_ENTER_IRR = 0.35              # pyranometer irradiance below this ...
PS_SOLAR_FRAC = 0.25             # ... or solar output below this fraction of the clear-sky reference ...
PS_ENTER_S = 2.0                 # ... for this long → conserve
PS_EXIT_IRR = 0.6                # irradiance above this (and solar back above PS_SOLAR_FRAC of the reference) ...
PS_EXIT_S = 5.0                  # ... for this long → normal operation
PS_REF_IRR = 0.8                 # clear-sky reference: rolling max of solar_w over samples brighter than this ...
PS_REF_S = 60.0                  # ... within this window
PS_SPEED_SCALE = 0.6             # drive speed factor while conserving (multiplies the FSM's own speed scale)


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
    def battery_pct(self) -> float:
        return self.bat.pct

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

    def step(self, t: float, dt: float, airborne: bool, compute_w: float, rho: float = 1.2, wind_mps: float = 0.0):
        """Integrate hover + avionics + compute power; `wind_mps` is the drone's own wind estimate (wind.py)."""
        hover = 0.0
        if airborne:
            w = wind_mps if wind_mps is not None and math.isfinite(wind_mps) else 0.0
            hover = drone_hover_power_w(self.mass, self.g, rho) + DRONE_WIND_W_PER_MPS2 * w * w
        self.bat.integrate(dt, DRONE_AVIONICS_W + max(0.0, compute_w) + hover)

    def state(self) -> HealthState:
        b = self.bat
        return HealthState(pct=100.0, structure_pct=100.0, drive_pct=100.0, battery_pct=b.pct, battery_wh=b.wh,
                           capacity_wh=b.capacity, solar_w=0.0, load_w=b.load_w, net_w=-b.load_w)


class PowerSaver:
    """Battery-conservation mode from the rover's own power readings (hw_status solar_w + irradiance).

    Enter: irradiance < ENTER_IRR (0.35) for ENTER_S (2.0 s, sim), or solar_w < 25 % of the clear-sky reference
    (rolling max of solar_w over the last 60 s while irradiance > 0.8) for ENTER_S. Exit: irradiance > EXIT_IRR
    (0.6) for EXIT_S (5.0 s), with solar_w back above 25 % of the reference (so a panel that stays starved under a
    bright sky does not flap in and out); the reference does not age while active. update(t, solar_w, irradiance, battery_pct) -> 'enter' | 'exit' | None;
    .active; .speed_scale (0.6 when active, else 1.0); .reason (trace input with the measured numbers);
    .trigger ('irradiance' | 'solar': what entered it). A sample without an irradiance reading changes nothing."""
    ENTER_IRR, ENTER_S, EXIT_IRR, EXIT_S = PS_ENTER_IRR, PS_ENTER_S, PS_EXIT_IRR, PS_EXIT_S
    SOLAR_FRAC, SPEED_SCALE = PS_SOLAR_FRAC, PS_SPEED_SCALE

    def __init__(self):
        self.active = False
        self.reason = ''
        self.trigger: str | None = None
        self.solar_w: float | None = None
        self.irradiance: float | None = None
        self.battery_pct: float | None = None
        self._ref: collections.deque = collections.deque()   # (t, solar_w), decreasing solar_w: front = rolling max
        self._since: float | None = None                    # start of the current run that would flip the mode

    @property
    def speed_scale(self) -> float:
        return self.SPEED_SCALE if self.active else 1.0

    @property
    def ref_w(self) -> float | None:
        """Clear-sky solar reference (W), None without a bright sample in the last PS_REF_S seconds."""
        return self._ref[0][1] if self._ref else None

    def update(self, t: float, solar_w, irradiance, battery_pct=None) -> str | None:
        try:
            sw, irr = max(0.0, float(solar_w)), float(irradiance)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(sw) and math.isfinite(irr)):
            return None
        bp = None
        if battery_pct is not None:
            try:
                bp = float(battery_pct)
            except (TypeError, ValueError):
                bp = None
        self.solar_w, self.irradiance, self.battery_pct = sw, irr, bp
        # the clear-sky reference does not age while conserving: else the starved samples would become the new
        # "clear sky" after PS_REF_S and the mode would end with the panel still starved
        while not self.active and self._ref and t - self._ref[0][0] > PS_REF_S:
            self._ref.popleft()
        if irr > PS_REF_IRR:
            while self._ref and self._ref[-1][1] <= sw:
                self._ref.pop()
            self._ref.append((t, sw))
        ref = self.ref_w
        dark = irr < self.ENTER_IRR
        starved = ref is not None and sw < self.SOLAR_FRAC * ref
        flip = (irr > self.EXIT_IRR and not starved) if self.active else (dark or starved)
        if not flip:
            self._since = None
            return None
        if self._since is None:
            self._since = t
        if t - self._since < (self.EXIT_S if self.active else self.ENTER_S) - 1e-9:
            return None
        self._since = None
        self.active = not self.active
        if self.active:
            self.trigger = 'irradiance' if dark else 'solar'
        self.reason = self._text(ref)
        return 'enter' if self.active else 'exit'

    def _text(self, ref: float | None) -> str:
        sw = self.solar_w or 0.0
        txt = f'Solar {sw:.0f} W'
        if ref is not None and ref > 0.0:
            if self.active:
                txt += f' (was {ref:.0f} W, {100.0 * (sw / ref - 1.0):+.0f} %)'.replace('-', '−')
            else:
                txt += f' ({100.0 * sw / ref:.0f} % of the clear-sky {ref:.0f} W)'
        txt += f', irradiance {self.irradiance:.2f}'
        if self.battery_pct is not None:
            txt += f', battery {self.battery_pct:.0f} %'
        return txt

    def texts(self, event: str) -> dict:
        """Trace texts for 'enter' (kind power_save) / 'exit' (kind power_restored)."""
        if event == 'enter':
            if self.trigger == 'solar':
                why = (f'Solar output below {self.SOLAR_FRAC:.0%} of the clear-sky reference for '
                       f'{self.ENTER_S:.0f} s: the panel is starved')
            else:
                why = (f'Pyranometer irradiance below {self.ENTER_IRR:.2f} for {self.ENTER_S:.0f} s: dust is '
                       'blocking the solar panel')
            return {'input': self.reason, 'reasoning': why + '; driving slower draws less power',
                    'confidence': 0.9, 'action': f'Battery conservation: speed ×{self.SPEED_SCALE:.1f}',
                    'severity': 'warn'}
        return {'input': self.reason,
                'reasoning': f'Irradiance above {self.EXIT_IRR:.2f} for {self.EXIT_S:.0f} s: the panel is charging again',
                'confidence': 0.9, 'action': 'Battery conservation off: full drive speed restored', 'severity': 'info'}
