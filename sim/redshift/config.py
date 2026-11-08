"""Configuration: environment variables from PROTOCOL.md plus tunables."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


ROVER_COLORS = ["#ff7a45", "#36cfc9", "#9254de", "#fadb14", "#73d13d", "#f759ab", "#40a9ff", "#ffc069"]


@dataclass(frozen=True)
class Config:
    # ---- protocol env vars ----
    seed: int = 42
    num_rovers: int = 3
    sim_speed: float = 3.0
    map_size: float = 120.0
    gravity_g: float = 2.0
    viewer: bool = False

    # ---- world ----
    world_name: str = "Redshift-b"
    res: float = 0.5               # map grid spacing (m), same grid for hfield and protocol
    hfield_tile: int = 10          # heightfield tile size in cells (tiles make ray casts fast)
    lander_radius: float = 2.5
    lander_flat_radius: float = 4.0
    num_rocks: int = 55            # per 80 m x 80 m; terrain feature counts scale with the map area
    water: bool = True             # lakes (and maybe a river), see terrain.py
    spires: bool = True            # alien crystal spires
    lifeforms: bool = True         # herds of grazers (lifeforms.py)
    reserve_spires: int = 4        # hidden spare spire bodies for "drop a new spire here"

    # ---- physics ----
    timestep: float = 0.005
    cone: str = "pyramidal"
    impratio: float = 1.0
    agent_hz: float = 5.0          # decision rate (sim time)
    lidar_hz: float = 5.0          # scan rate per rover (sim time)

    # ---- rover body (tuned so the climb limit at 2 g is about 20 deg) ----
    # scripts/climb_test.py: climbs 20 deg at full speed, stalls at 21, slides back at 22. The limit is
    # set by motor torque (so it depends on gravity), not by grip: at 1 g the same rover climbs ~26 deg.
    chassis_half: tuple = (0.4, 0.25, 0.1)
    chassis_mass: float = 18.0
    wheel_radius: float = 0.18
    wheel_mass: float = 1.0
    wheel_x: float = 0.22
    wheel_y: float = 0.40          # wide track, short wheelbase: easy skid steer turns
    wheel_z: float = -0.02
    wheel_friction: float = 0.8
    terrain_friction: float = 0.5
    motor_torque: float = 7.2      # torque cap per wheel (N m)
    motor_kv: float = 25.0         # velocity servo gain (N m s / rad)
    max_wheel_speed: float = 8.0   # rad/s
    lidar_height: float = 0.75     # sensor site above chassis centre (on a mast)

    # ---- driving ----
    v_max: float = 0.8             # m/s
    w_max: float = 1.2             # rad/s
    track_eff: float = 0.55        # effective half track for skid steer kinematics (m)
    lookahead: float = 1.3

    # ---- perception ----
    lidar_range: float = 8.0
    lidar_az: int = 72
    lidar_el_deg: tuple = (-40.0, -31.0, -24.0, -19.0, -15.0, -12.0, -9.5, -7.5, -6.0, -4.7, -3.6, -2.5, -1.0, 2.0)
    spectro_range: float = 8.0

    # ---- decisions ----
    slope_limit_initial_deg: float = 30.0
    slope_limit_floor_deg: float = 12.0
    stall_window_s: float = 3.0
    stall_progress_m: float = 0.15
    flip_tilt_deg: float = 70.0
    flip_time_s: float = 2.0
    battery_base_drain: float = 1.0 / 1500.0   # per second, electronics and heaters
    battery_motor_k: float = 1.0 / 70000.0     # per joule of motor work (incl. copper losses)
    battery_heat_k: float = 0.02               # copper loss coefficient (W per (N m)^2)
    battery_margin: float = 0.15
    charge_time_s: float = 90.0     # empty to full
    investigate_dwell_s: float = 3.0   # fallback instrument time (s) for an unknown kind
    investigate_radius: float = 2.0
    target_bonus_radius: float = 15.0
    sample_capacity: int = 3           # sample tubes per rover (protocol v4b)
    drill_s: float = 10.0              # instrument run times (sim s): mineral drill core
    scoop_s: float = 8.0               # ice scoop
    sniffer_s: float = 6.0             # vent gas sniffer
    swab_s: float = 6.0                # organic trace swab
    high_value: int = 8                # an estimated value this high is worth a trip to free a tube

    # ---- lifeforms: observing from a safe distance (planetary protection) ----
    observe_standoff: float = 6.0      # follow at about this distance (m)
    observe_min_dist: float = 4.0      # never closer than this to a tracked creature (m)
    observe_max_s: float = 75.0        # an observation lasts at most this long (sim s)
    observe_v: float = 0.35            # top speed while observing (m/s)
    observe_leash: float = 25.0        # never further than this from where the observation started (m)

    # ---- protocol v7: exploring an area (focus) and curiosity ----
    focus_s: float = 120.0             # operator "explore around X": stay this long once there (sim s)
    focus_r: float = 15.0              # ... exploring the unknown ground within this radius (m)
    focus_margin: float = 4.0          # unknown ground this far outside the circle still counts as nearby (m)
    focus_offset: float = 20.0         # "explore north of the lander": the circle's centre this far that way (m)
    focus_travel_s: float = 180.0      # an area focus not reached in this long is given up (sim s)
    curiosity: bool = True             # rovers stick around something new they found (interest focus)
    interest_s: float = 60.0           # ... for about this long (sim s)
    interest_r: float = 10.0           # ... within this radius (m)
    interest_cooldown_s: float = 150.0  # per rover, after an interest focus ends (sim s)

    # ---- edge compute budget (per rover per decision cycle, wall clock) ----
    brain_budget_ms: float = 40.0
    plan_share: float = 0.75        # share of the budget the planner may use in one cycle
    plan_clock: str = "wall"        # "wall": stop planning on the wall clock (real budget, not repeatable);
                                    # "model": count A* expansions at exp_cost_us each (repeatable runs)
    exp_cost_us: float = 3.0        # measured cost of one A* expansion on the dev machine

    # ---- hazards (see hazards.py) ----
    num_boulders: int = 24          # hidden boulder bodies allocated in the model for rockfalls (3 to 5 per rockfall)
    sand_radius: float = 2.5
    sand_friction: float = 0.3      # wheel friction inside a sand patch (normal: wheel_friction)
    sand_wheel_damping: float = 0.3  # extra wheel joint damping inside a patch (rolling resistance)
    sand_drag_fwd: float = 5.0      # sinkage drag on the rover body, x weight, per m/s, when pushing in
    sand_drag_rev: float = 0.5      # same when backing out along its own compacted ruts
    dust_storm_s: float = 90.0
    dust_lidar_range: float = 3.0
    dust_charge_factor: float = 0.3  # lander solar charging power left during a dust storm
    fault_s: float = 90.0

    # ---- reactions ----
    slip_threshold: float = 0.6
    slip_time_s: float = 1.5
    slip_min_wheel_speed: float = 0.2
    soft_zone_radius: float = 3.2   # how big the fleet assumes a soft patch is (true ones: sand_radius)
    sand_free_s: float = 30.0       # failsafe: a rover still stuck after this long digs itself out
    degraded_v_max: float = 0.4     # top speed while the LiDAR is degraded
    degraded_unknown_cost: float = 8.0
    slow_charge_margin: float = 0.25  # return margin while the lander charges slowly
    caution_v: float = 0.3           # speed right after perception flags something unexpected

    # ---- telemetry ----
    scan_points: int = 200          # max LiDAR points per rover in a tick message (protocol: at most 300)

    # ---- tier 2: Gemma advice, operator chat and questions (advisor.py) ----
    gemma_ask_every_s: float = 20.0      # a rover asks Gemma about its goal at most this often (sim s), or on arrival
    operator_timeout_s: float = 0.0      # WALL seconds a question waits before the safe default; 0 = never expires
    operator_ask_below: float = 0.75     # Gemma confidence below this: ask the operator
    operator_ask_first: bool = True      # always check the first anomaly after start or reset with the operator
    question_cooldown_s: float = 60.0    # per rover, sim seconds between two questions
    max_open_questions: int = 2          # fleet wide

    @property
    def nx(self) -> int:
        return int(round(self.map_size / self.res)) + 1

    @property
    def ny(self) -> int:
        return self.nx

    @property
    def origin(self) -> tuple:
        half = (self.nx - 1) * self.res / 2.0
        return (-half, -half)

    @property
    def battery_capacity(self) -> float:
        """Relative battery size: bigger maps get a bigger battery (1.0 at 80 m, 1.5 at 120 m), so a trip
        to the far edge and back takes about the same share of a charge as on the 80 m map."""
        return max(1.0, self.map_size / 80.0)

    @property
    def gravity(self) -> float:
        return 9.81 * self.gravity_g

    @property
    def lidar_el(self) -> tuple:
        return tuple(math.radians(e) for e in self.lidar_el_deg)

    def with_seed(self, seed: int) -> "Config":
        return replace(self, seed=int(seed))

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        base = cls(
            seed=_env_int("SEED", 42),
            num_rovers=max(1, min(8, _env_int("NUM_ROVERS", 3))),
            sim_speed=max(0.1, _env_float("SIM_SPEED", 3.0)),
            map_size=max(30.0, _env_float("MAP_SIZE", 120.0)),
            gravity_g=max(0.1, _env_float("GRAVITY_G", 2.0)),
            viewer=_env_bool("VIEWER", False),
            brain_budget_ms=max(1.0, _env_float("BRAIN_BUDGET_MS", 40.0)),
            gemma_ask_every_s=max(2.0, _env_float("GEMMA_ASK_EVERY_S", 20.0)),
            operator_timeout_s=max(0.0, _env_float("OPERATOR_TIMEOUT_S", 0.0)),
            operator_ask_below=min(1.01, max(0.0, _env_float("OPERATOR_ASK_BELOW", 0.75))),
            operator_ask_first=_env_bool("OPERATOR_ASK_FIRST", True),
        )
        return replace(base, **overrides) if overrides else base
