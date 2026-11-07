"""Gravity-aware rover traversability model (pure Python, shared by planner, mock world and UI).

The maximum traversable slope is the most restrictive of three physical limits:

* **drive torque** – wheels must supply m·g·sin(θ):            θ_d = asin(F_max / (m·g))
* **soil traction** – Mohr-Coulomb shear (cohesion + friction): θ_s = atan(tan φ + c / (m·g))
* **static tip-over** – geometric, independent of g:             θ_t = atan((track/2) / h_cg)

Heavier effective weight (Super-Earth gravity) shrinks θ_d fastest, so the binding constraint moves
from *slip* at 1 g to *drive torque* at ~2 g – which is exactly what the planner reports in its traces.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

G_EARTH = 9.81


@dataclass(frozen=True)
class RoverSpec:
    mass_kg: float = 60.0
    wheel_radius_m: float = 0.20
    track_m: float = 0.85
    cg_height_m: float = 0.35
    wheels: int = 4
    tau_max_nm: float = 20.8          # per wheel; also written into the Gazebo joint effort limit
    soil_friction_deg: float = 32.0   # regolith internal friction angle φ
    cohesion_force_n: float = 150.0   # contact-patch area × cohesion
    safety_factor: float = 0.9
    v_nominal: float = 1.0            # m/s at 1 g
    rolling_resistance: float = 0.08
    drivetrain_eff: float = 0.75


@dataclass(frozen=True)
class Limits:
    g: float
    max_slope_deg: float
    factor: str            # 'slip' | 'drive_torque' | 'tip'
    drive_deg: float
    slip_deg: float
    tip_deg: float
    tip_alarm_deg: float
    v_max: float
    energy_j_per_m: float  # flat ground

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


class RoverPhysics:
    def __init__(self, spec: RoverSpec | None = None):
        self.spec = spec or RoverSpec()

    @classmethod
    def from_spec(cls, spec: dict) -> 'RoverPhysics':
        names = {f.name for f in fields(RoverSpec)}
        return cls(RoverSpec(**{k: v for k, v in spec.items() if k in names}))

    def weight_n(self, g_mult: float) -> float:
        return self.spec.mass_kg * G_EARTH * g_mult

    def drive_limit_deg(self, g_mult: float) -> float:
        s = self.spec
        f_max = s.wheels * s.tau_max_nm / s.wheel_radius_m
        return math.degrees(math.asin(min(1.0, f_max / self.weight_n(g_mult))))

    def slip_limit_deg(self, g_mult: float) -> float:
        s = self.spec
        mu_eff = math.tan(math.radians(s.soil_friction_deg)) + s.cohesion_force_n / self.weight_n(g_mult)
        return math.degrees(math.atan(mu_eff))

    def tip_limit_deg(self) -> float:
        s = self.spec
        return math.degrees(math.atan((s.track_m / 2.0) / s.cg_height_m))

    def limits(self, g_mult: float) -> Limits:
        drive = self.drive_limit_deg(g_mult)
        slip = self.slip_limit_deg(g_mult)
        tip = self.tip_limit_deg()
        factor, binding = min((('drive_torque', drive), ('slip', slip), ('tip', tip)), key=lambda kv: kv[1])
        max_slope = self.spec.safety_factor * binding
        tip_alarm = min(0.7 * tip, max_slope + 12.0)
        v_max = self.spec.v_nominal * min(1.0, math.sqrt(1.5 / g_mult))
        return Limits(g=g_mult, max_slope_deg=max_slope, factor=factor, drive_deg=drive, slip_deg=slip,
                      tip_deg=tip, tip_alarm_deg=tip_alarm, v_max=v_max,
                      energy_j_per_m=self.energy_per_m(g_mult, 0.0, 1.0))

    def energy_per_m(self, g_mult: float, dh: float, d: float) -> float:
        """Electrical energy (J) per metre travelled for a segment of length d rising dh."""
        s = self.spec
        grade = max(0.0, dh / d) if d > 0 else 0.0
        return self.weight_n(g_mult) * (s.rolling_resistance + grade) / s.drivetrain_eff
