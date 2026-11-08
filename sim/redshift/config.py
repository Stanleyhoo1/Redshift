"""Configuration: environment variables from PROTOCOL.md plus tunables."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields, replace


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError, OverflowError):
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


# ====================================================================== gravity (v7)
# The slope thresholds below were tuned for 2 g. With a launch gravity they follow the rover's predicted
# climb limit instead, chosen so that each one is exactly its old value at 2 g.
ROVER_AXLE_MASS = 0.5        # world.py: the front axle bar (kg)
CLIMB_CAP_DEG = 30.0         # above this, grip and tipping limit the climb (climb_test at 1 g: 28 deg ok, tips at 32)
TOW_FACTOR = 0.8             # a tow train keeps this share of the two rovers' motor limit
TARGET_PATH_SLOPE = 15.0     # terrain.py: every target reachable from the lander without steeper ground
REACH_SLOPE = 18.0           # terrain.py: flood fill slope of the "no traps" checks for lakes and rivers
PASSABLE_SLOPE = 20.0        # hazards.py: steepest ground a placed hazard counts as drivable
SAND_SPIN_LOAD = 0.2         # rover.py: wheel load (share of the torque cap) that tells sand from free spin


def motor_climb_deg(gravity_g: float, mass: float, motor_torque: float, wheel_radius: float) -> float:
    """Steepest slope (deg) four wheel motors can hold `mass` on: asin(4 T / r / (m g)); 90 when they
    could lift it straight up."""
    ratio = 4.0 * motor_torque / wheel_radius / (mass * 9.81 * gravity_g)
    return math.degrees(math.asin(min(1.0, ratio)))


ROVER_COLORS = ["#ff7a45", "#36cfc9", "#9254de", "#fadb14", "#73d13d", "#f759ab", "#40a9ff", "#ffc069"]


# ====================================================================== v7 launch settings
# SETTINGS is the single source of the settings a mission can be launched with: keys, ranges, labels and
# help. GET /api/config sends it to the setup screen, Config.from_settings checks requests against it and
# Config.from_env takes its clamps from it. Later steps never add a setting anywhere else.
def _num_text(v) -> str:
    """3.0 -> "3", 0.5 -> "0.5", 2147483647 -> "2147483647" (never "2.14748e+09")."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _with_unit(v, unit: str) -> str:
    t = _num_text(v)
    return t + unit if unit == "x" else (f"{t} {unit}" if unit else t)


@dataclass(frozen=True)
class Setting:
    key: str                   # a Config field (wear stands for the wear and wear_rate pair)
    type: str                  # "int" | "float" | "bool" | "choice"
    label: str
    group: str                 # "planet" | "fleet" | "sector" | "features" | "wear" | "simulation"
    help: str                  # one line for the setup screen: what it is, what it does
    min: float | None = None   # int and float only
    max: float | None = None
    step: float | None = None
    unit: str = ""
    choices: tuple = ()        # choice only: ((value, label, help), ...)

    def to_json(self) -> dict:
        out = {"key": self.key, "type": self.type, "label": self.label, "group": self.group, "help": self.help}
        if self.type in ("int", "float"):
            out.update(min=self.min, max=self.max, step=self.step, unit=self.unit)
        if self.choices:
            out["choices"] = [{"value": v, "label": lab, "help": h} for v, lab, h in self.choices]
        return out

    def parse(self, value) -> tuple:
        """Check one value from a request. Returns (value, None) or (None, error text)."""
        k = self.key
        if self.type == "bool":
            return (value, None) if isinstance(value, bool) else (None, f"{k} must be true or false")
        if self.type == "choice":
            names = [c[0] for c in self.choices]
            if isinstance(value, str) and value in names:
                return value, None
            return None, f"{k} must be one of {', '.join(names)}"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"{k} must be a number"
        if isinstance(value, float) and not math.isfinite(value):
            return None, f"{k} must be a finite number"
        if self.type == "int" and isinstance(value, float) and not value.is_integer():
            return None, f"{k} must be a whole number"
        if not self.min <= value <= self.max:        # before any conversion: a huge int cannot overflow
            return None, f"{k} must be between {_num_text(self.min)} and {_with_unit(self.max, self.unit)}"
        if self.type == "int":
            value = int(value)
            if self.step > 1 and (value - int(self.min)) % int(self.step):
                return None, f"{k} must be in steps of {_with_unit(self.step, self.unit)}"
            return value, None
        # a float on the step grid, give or take float noise (1.7000000000000002 is 1.7); 2.04 g is refused,
        # so every launched gravity is one the setup screen and the physics table show
        n = (float(value) - self.min) / self.step
        if abs(n - round(n)) > 1e-6:
            return None, f"{k} must be in steps of {_with_unit(self.step, self.unit)}"
        return round(self.min + round(n) * self.step, 6), None

    def clamp(self, value):
        """value limited to [min, max] (from_env: an env var out of range is clamped, not refused)."""
        return min(self.max, max(self.min, value))


SETTINGS: tuple = (
    Setting("gravity_g", "float", "Gravity", "planet",
            "Heavier gravity makes every slope harder: the rovers climb and tow less.", 0.5, 3.0, 0.1, "g"),
    Setting("seed", "int", "Seed", "planet", "Same seed, same terrain, targets and life.", 0, 2147483647, 1),
    Setting("num_rovers", "int", "Rovers", "fleet", "How many rovers land. More rovers map faster.", 1, 8, 1),
    Setting("map_size", "int", "Sector size", "sector", "Width of the square sector to explore.", 60, 200, 10, "m"),
    Setting("water", "bool", "Water", "features",
            "Lakes and a river. Liquid gives no LiDAR echo, so rovers must work it out."),
    Setting("spires", "bool", "Crystal spires", "features",
            "Tall alien crystals that block routes and can topple across one."),
    Setting("lifeforms", "bool", "Lifeforms", "features",
            "Herds of grazers. Rovers watch from 4 m or more and never touch them."),
    Setting("mesas", "bool", "Mesas", "features", "Flat-topped hills with cliff walls no rover can climb."),
    Setting("canyon", "bool", "Canyon", "features", "A winding trench with two ramps, the only ways across."),
    Setting("dunes", "bool", "Dunes", "features", "Rippled loose sand where the wheels slip."),
    Setting("boulder_fields", "bool", "Boulder fields", "features", "Dense patches of boulders to steer around."),
    Setting("pits", "bool", "Pits", "features", "Sinkholes and lava-tube skylights: sheer drops in the ground."),
    Setting("geysers", "bool", "Geysers", "features", "Vents that erupt on a schedule. Steam scatters the LiDAR."),
    Setting("ice_sheets", "bool", "Ice sheets", "features", "Frozen ground with almost no grip."),
    Setting("flora", "bool", "Alien flora", "features", "Fungal trees and glowing bushes. Rovers keep 4 m away."),
    Setting("ground_detail", "bool", "Ground detail", "features",
            "Pebbles, hoodoos, rock bands and wind streaks. Looks only."),
    Setting("wear", "choice", "Wear", "wear",
            "How fast wheels and motors wear out. A worn-out rover breaks down and needs a tow.",
            choices=(("off", "Off", "Rovers never wear out."),
                     ("normal", "Normal", "A drivetrain lasts about 2 km of driving."),
                     ("harsh", "Harsh", "Parts wear out about 4 times sooner."))),
    Setting("sim_speed", "float", "Speed", "simulation", "Sim seconds per real second.", 0.5, 10.0, 0.5, "x"),
)
SETTINGS_BY_KEY = {s.key: s for s in SETTINGS}
WEAR_RATES = {"normal": 1.0, "harsh": 4.0}          # Config.wear_rate per wear setting ("off": wear=False)
PRESETS = {"gravity_g": [{"label": "Earth", "value": 1.0}, {"label": "Redshift-b", "value": 2.0},
                         {"label": "Heavy", "value": 3.0}]}


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

    # ---- v7 world features and wear. All off here, so Config(seed=42) still builds today's world;
    # from_env turns them on. Each is built in its own v7 step; until then the flag is only stored. ----
    mesas: bool = False
    canyon: bool = False
    dunes: bool = False
    boulder_fields: bool = False
    pits: bool = False
    geysers: bool = False
    ice_sheets: bool = False
    ground_detail: bool = False
    flora: bool = False
    wear: bool = False
    wear_rate: float = 1.0          # how fast parts wear: 1.0 normal, 4.0 harsh (WEAR_RATES)

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

    # ---- v7: thresholds derived from gravity (each exactly its old value at 2 g) ----
    @property
    def rover_mass(self) -> float:
        """One rover (kg): chassis, four wheels and the axle bar, as world.py builds it."""
        return self.chassis_mass + 4.0 * self.wheel_mass + ROVER_AXLE_MASS

    @property
    def climb_limit_deg(self) -> float:
        """Predicted climb limit: the motor torque limit, capped where grip takes over."""
        return min(CLIMB_CAP_DEG, motor_climb_deg(self.gravity_g, self.rover_mass, self.motor_torque,
                                                  self.wheel_radius))

    @property
    def tow_limit_deg(self) -> float:
        """Predicted steepest slope one rover can tow another up (two rovers' mass on four motors):
        0.8 * asin(4 T / r / ((m1 + m2) g)), the asin capped at the single-rover grip limit (a tow train never
        climbs steeper than one rover). The one tow limit: the setup screen and GET /api/config show it, and
        the canyon ramps and tow paths of later v7 steps use it instead of their own copy of the formula."""
        two = motor_climb_deg(self.gravity_g, 2.0 * self.rover_mass, self.motor_torque, self.wheel_radius)
        return TOW_FACTOR * min(CLIMB_CAP_DEG, two)

    @property
    def target_path_slope_deg(self) -> float:
        return min(TARGET_PATH_SLOPE, self.climb_limit_deg - 2.0)

    @property
    def reach_slope_deg(self) -> float:
        return min(REACH_SLOPE, self.climb_limit_deg - 1.0)

    @property
    def passable_slope_deg(self) -> float:
        return min(PASSABLE_SLOPE, self.climb_limit_deg - 1.0)

    @property
    def slope_floor_deg(self) -> float:
        """The learned fleet slope limit never drops below this."""
        return min(self.slope_limit_floor_deg, self.climb_limit_deg - 2.0)

    @property
    def sand_spin_load(self) -> float:
        """Wheel load that marks sand: sinkage drag grows with weight, so lighter gravity needs less."""
        return SAND_SPIN_LOAD * min(1.0, self.gravity_g / 2.0)

    def with_seed(self, seed: int) -> "Config":
        return replace(self, seed=int(seed))

    @classmethod
    def from_settings(cls, base: "Config", data) -> tuple:
        """Apply launch settings (a JSON object keyed like SETTINGS; keys left out keep base's values).
        Returns (new Config, []) or (None, [every problem found, readable]). A number equal to base's own
        value is no change and always passes, even off the table's range or step (an env var or a reset may
        give a world one, e.g. MAP_SIZE=125), so GET /api/config `current` can be sent back as it is."""
        if not isinstance(data, dict):
            return None, ['Body must be a JSON object like {"gravity_g": 1.5, "num_rovers": 4}']
        changes, errors = {}, []
        own = base.settings()
        for key, value in data.items():
            s = SETTINGS_BY_KEY.get(key)
            if s is None:
                errors.append(f"unknown setting {key!r} (use one of {', '.join(SETTINGS_BY_KEY)})")
                continue
            val, err = s.parse(value)
            if err and s.type in ("int", "float") and _same_number(value, own.get(key)):
                val, err = own[key], None
            if err:
                errors.append(err)
            elif key == "wear":
                changes["wear"] = val != "off"
                changes["wear_rate"] = WEAR_RATES.get(val, 1.0)
            elif _FIELD_TYPES[key] == "float":
                changes[key] = float(val)      # map_size is a float field set from a whole number
            else:
                changes[key] = val
        if errors:
            return None, errors
        return (replace(base, **changes) if changes else base), []

    def settings(self) -> dict:
        """The effective launch settings, keyed and ordered like SETTINGS (GET /api/config `current`,
        init `config`)."""
        out = {}
        for s in SETTINGS:
            if s.key == "wear":
                if not self.wear:
                    out["wear"] = "off"
                else:
                    out["wear"] = min(WEAR_RATES, key=lambda n: abs(WEAR_RATES[n] - self.wear_rate))
                continue
            v = getattr(self, s.key)
            if s.type == "bool":
                v = bool(v)
            elif s.type == "int":
                v = int(v) if float(v).is_integer() else float(v)
            elif s.type == "float":
                v = float(v)
            out[s.key] = v
        return out

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        S = SETTINGS_BY_KEY
        wear = (os.environ.get("WEAR") or "normal").strip().lower()
        if wear not in ("off", *WEAR_RATES):
            wear = "normal"
        base = cls(
            seed=max(0, _env_int("SEED", 42)),      # not clamped above: as before v7, SEED=3000000000 builds that world
            num_rovers=int(S["num_rovers"].clamp(_env_int("NUM_ROVERS", 3))),
            sim_speed=float(S["sim_speed"].clamp(_env_float("SIM_SPEED", 3.0))),
            map_size=float(S["map_size"].clamp(_env_float("MAP_SIZE", 120.0))),
            gravity_g=float(S["gravity_g"].clamp(_env_float("GRAVITY_G", 2.0))),
            viewer=_env_bool("VIEWER", False),
            brain_budget_ms=max(1.0, _env_float("BRAIN_BUDGET_MS", 40.0)),
            gemma_ask_every_s=max(2.0, _env_float("GEMMA_ASK_EVERY_S", 20.0)),
            operator_timeout_s=max(0.0, _env_float("OPERATOR_TIMEOUT_S", 0.0)),
            operator_ask_below=min(1.01, max(0.0, _env_float("OPERATOR_ASK_BELOW", 0.75))),
            operator_ask_first=_env_bool("OPERATOR_ASK_FIRST", True),
            # v7 world features: on unless their env var says otherwise (MESAS=0 leaves mesas out)
            mesas=_env_bool("MESAS", True),
            canyon=_env_bool("CANYON", True),
            dunes=_env_bool("DUNES", True),
            boulder_fields=_env_bool("BOULDER_FIELDS", True),
            pits=_env_bool("PITS", True),
            geysers=_env_bool("GEYSERS", True),
            ice_sheets=_env_bool("ICE_SHEETS", True),
            ground_detail=_env_bool("GROUND_DETAIL", True),
            flora=_env_bool("FLORA", True),
            wear=wear != "off",
            wear_rate=WEAR_RATES.get(wear, 1.0),
        )
        return replace(base, **overrides) if overrides else base


# declared type of each Config field ("float", "int", ...): from_settings stores floats as floats
_FIELD_TYPES = {f.name: f.type for f in fields(Config)}


def _same_number(value, own) -> bool:
    """value (from a request) is a real, finite number equal to own (base's setting)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or isinstance(own, bool):
        return False
    if not isinstance(own, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
        return False
    return value == own


def physics_table(cfg: Config) -> list:
    """Predicted climb and tow limits for this config's rover at every launch gravity step (GET /api/config
    `physics`): [{gravity_g, climb_deg, tow_deg, grip_limited}, ...]."""
    s = SETTINGS_BY_KEY["gravity_g"]
    rows = []
    for k in range(int(round((s.max - s.min) / s.step)) + 1):
        g = round(s.min + k * s.step, 1)
        c = replace(cfg, gravity_g=g)
        motor = motor_climb_deg(g, c.rover_mass, c.motor_torque, c.wheel_radius)
        rows.append({"gravity_g": g, "climb_deg": round(c.climb_limit_deg, 1), "tow_deg": round(c.tow_limit_deg, 1),
                     "grip_limited": motor > CLIMB_CAP_DEG})
    return rows
