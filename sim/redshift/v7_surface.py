"""Version 7 surface features: dune fields and ice sheets.

Both are ground truth. The world builder shapes the height field with them and the physics composer
(compose_wheel, called from HazardManager.update_physics) sets wheel friction by wheel position. Rover
logic never reads them: a rover finds an ice sheet only through its wheel encoders, motor load and IMU
(check_low_traction below), and the fleet then keeps an ice zone it has learned.

Own random streams: dunes default_rng([seed, 7103]), ice sheets default_rng([seed, 7107]).
"""
from __future__ import annotations

import math

import numpy as np

DUNE_K = 7103
ICE_K = 7107

ICE_MU = 0.05              # wheel friction on an ice sheet (sand detector trips above about 0.073)
DUNE_MU = 0.5              # loose dune sand: less grip than firm ground (0.8), more than a sand trap (0.3)
DUNE_WHEEL_DAMPING = 0.08  # mild rolling resistance in loose sand (a sand trap adds 0.3)

NONE, DUNE, ICE = 0, 1, 2

# low traction rule: wheels spin, the body does not follow, the motors barely work, ground is level
LT_SLIP = 0.5
LT_SPIN_LOAD = 0.3         # sand needs more than cfg.sand_spin_load; ice about 2.76 * mu = 0.14
LT_TILT = 4.0
LT_TIME_S = 0.6
ICE_ZONE_R = 5.0           # how big the fleet assumes an ice sheet is (true ones: 3 to 7 m)


def _smooth(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _count(rng: np.random.Generator, area: float) -> int:
    return int(rng.integers(1, 3)) + (1 if area >= 2.0 else 0)


def _clear(x: float, y: float, r: float, placed: list, cfg, half: float, terrain=None) -> bool:
    if abs(x) > half - r - 4.0 or abs(y) > half - r - 4.0:
        return False
    if math.hypot(x, y) < cfg.lander_flat_radius + r + 8.0:     # lander and rover starts stay clear
        return False
    if terrain is not None and getattr(terrain, "landform_mask", None) is not None:
        from . import v7_landforms                               # keep off mesas, pits and the canyon
        if not v7_landforms.spot_clear(terrain, x, y, r + 2.0):
            return False
    return all(math.hypot(x - px, y - py) > r + pr + 3.0 for px, py, pr in placed)


def make_dunes(terrain, cfg) -> list:
    """Shape 1 to 3 dune fields into terrain.heights: long parallel ripples plus a few larger dunes,
    all aligned with one wind direction. Gentle enough for the rovers to cross."""
    rng = np.random.default_rng([cfg.seed, DUNE_K])
    half = cfg.map_size / 2.0
    X, Y, H = terrain.X, terrain.Y, terrain.heights
    out, placed = [], []
    for k in range(_count(rng, terrain.area)):
        for _ in range(60):
            r = float(rng.uniform(7.5, 12.5))
            x, y = (float(v) for v in rng.uniform(-half, half, 2))
            if _clear(x, y, r, placed, cfg, half, terrain):
                break
        else:
            continue
        wind = float(rng.uniform(0.0, 2.0 * math.pi))
        lam_r, amp_r = float(rng.uniform(3.5, 5.0)), float(rng.uniform(0.03, 0.045))
        lam_d, amp_d = float(rng.uniform(16.0, 24.0)), float(rng.uniform(0.3, 0.42))
        lam_c = float(rng.uniform(10.0, 16.0))
        ph = rng.uniform(0.0, 2.0 * math.pi, 3)
        d = np.hypot(X - x, Y - y)
        w = 1.0 - _smooth(0.6 * r, r, d)
        s = (X - x) * math.cos(wind) + (Y - y) * math.sin(wind)      # along the wind
        c = -(X - x) * math.sin(wind) + (Y - y) * math.cos(wind)     # across it
        ripple = amp_r * np.sin(2.0 * math.pi * s / lam_r + ph[0])
        dune = amp_d * np.sin(2.0 * math.pi * s / lam_d + ph[1]) * (0.7 + 0.3 * np.cos(2.0 * math.pi * c / lam_c + ph[2]))
        H += w * (ripple + dune)
        placed.append((x, y, r))
        out.append({"id": f"dune{k + 1}", "x": round(x, 2), "y": round(y, 2), "radius": round(r, 2),
                    "wind_deg": round(math.degrees(wind) % 360.0, 1)})
    return out


def make_ice(terrain, cfg, avoid: list) -> list:
    """Flatten 1 to 3 level frozen patches (radius 3 to 7 m) on ground that is already near flat."""
    rng = np.random.default_rng([cfg.seed, ICE_K])
    half = cfg.map_size / 2.0
    X, Y, H = terrain.X, terrain.Y, terrain.heights
    out = []
    placed = [(a["x"], a["y"], a["radius"]) for a in avoid]
    for k in range(_count(rng, terrain.area)):
        for _ in range(200):
            r = float(rng.uniform(3.0, 7.0))
            x, y = (float(v) for v in rng.uniform(-half, half, 2))
            if not _clear(x, y, r, placed, cfg, half, terrain):
                continue
            d = np.hypot(X - x, Y - y)
            disc = d <= r + 3.0
            h0 = float(H[d <= r].mean())
            if float(np.abs(H[disc] - h0).max()) < 0.25:             # gentle: under about 3 deg
                break
        else:
            continue
        w = 1.0 - _smooth(r, r + 3.0, d)
        H[:] = H * (1.0 - w) + h0 * w                                # level ice, blended into the ground
        placed.append((x, y, r))
        out.append({"id": f"ice{k + 1}", "x": round(x, 2), "y": round(y, 2), "radius": round(r, 2)})
    return out


def build(terrain, cfg) -> None:
    """Terrain hook, right after the height field is made and before water."""
    terrain.dunes = make_dunes(terrain, cfg) if cfg.dunes else []
    terrain.ice_sheets = make_ice(terrain, cfg, terrain.dunes) if cfg.ice_sheets else []


def after_water(terrain) -> None:
    """A lake carved over an ice sheet wins: drop that sheet."""
    ice = getattr(terrain, "ice_sheets", None)
    if ice and terrain.water:
        terrain.ice_sheets = [s for s in ice if not terrain.wet_near(s["x"], s["y"], s["radius"] + 1.0)]


def features(terrain) -> dict:
    """Optional features message keys (only when a feature is on)."""
    out = {}
    if terrain.cfg.dunes:
        out["dunes"] = list(getattr(terrain, "dunes", []))
    if terrain.cfg.ice_sheets:
        out["ice"] = list(getattr(terrain, "ice_sheets", []))
    return out


# ---------------------------------------------------------------------- physics composer
class Surface:
    """Region causes for the wheel contact: which wheel is on dune sand or ice."""

    def __init__(self, terrain):
        self.ice = [(s["x"], s["y"], s["radius"] ** 2) for s in getattr(terrain, "ice_sheets", [])]
        self.dunes = [(s["x"], s["y"], (0.85 * s["radius"]) ** 2) for s in getattr(terrain, "dunes", [])]
        self.state: dict = {}              # rover id -> region per wheel

    def region_at(self, x: float, y: float) -> int:
        for cx, cy, r2 in self.ice:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                return ICE
        for cx, cy, r2 in self.dunes:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                return DUNE
        return NONE


def surface_of(sim):
    """The sim's Surface, or None when no region feature is on (cached on the sim)."""
    s = getattr(sim, "_v7_surface", False)
    if s is False:
        t = sim.terrain
        s = Surface(t) if (getattr(t, "ice_sheets", None) or getattr(t, "dunes", None)) else None
        sim._v7_surface = s
    return s


def compose_wheel(base_fric: float, base_damp: float, in_sand: bool, region: int, cfg) -> tuple:
    """Final wheel friction and wheel joint damping from every active cause. With no region cause this
    returns exactly what the sand model wrote before v7."""
    if in_sand:
        return cfg.sand_friction, base_damp + cfg.sand_wheel_damping
    if region == ICE:
        return ICE_MU, base_damp
    if region == DUNE:
        return min(base_fric, DUNE_MU), base_damp + DUNE_WHEEL_DAMPING
    return base_fric, base_damp + 0.0


# ---------------------------------------------------------------------- rover side (sensed only)
def check_low_traction(rover, t: float) -> bool:
    """Ice: wheels turn well, the body does not follow, the motors barely work, ground is level.
    Uses only wheel encoders, motor torque and the IMU. Returns True when the rover reacted."""
    cfg = rover.cfg
    driving = rover.state in ("exploring", "investigating", "returning") and rover.cmd_v > 0.2 \
        and not rover.motor_fault
    f = rover.fleet
    known = ice_zone_at(f, rover.x, rover.y, 1.0) is not None
    hit = (driving and not known and abs(rover.gyro_z) < 0.8 and rover.tilt < LT_TILT
           and rover.v_wheel > cfg.slip_min_wheel_speed and rover.slip > LT_SLIP
           and rover.spin_load < LT_SPIN_LOAD)
    since = getattr(rover, "lt_since", None)
    if not hit:
        if since is not None and t - since > 0.4:
            rover.lt_since = None
        return False
    if since is None:
        rover.lt_since = t
        rover.lt_xy = (rover.x, rover.y)
        rover.lt_peak = rover.slip
        return False
    rover.lt_peak = max(rover.lt_peak, rover.slip)
    if t - since < LT_TIME_S:
        return False
    _on_low_traction(rover, t)
    return True


def ice_zone_at(fleet, x: float, y: float, margin: float = 0.5):
    for z in getattr(fleet, "ice_zones", ()):
        if math.hypot(x - z[0], y - z[1]) <= z[2] + margin:
            return z
    return None


def add_ice_zone(fleet, t: float, rover, x: float, y: float, radius: float) -> None:
    """Fleet belief: frozen ground here. Impassable for planning (the avoid layer), so routes go around."""
    if not hasattr(fleet, "ice_zones"):
        fleet.ice_zones = []
    fleet.ice_zones.append((float(x), float(y), float(radius), rover.id, float(t)))
    fleet.map.mark_avoid(x, y, radius)
    fleet.invalidate_costs()
    fleet.frontiers_t = -1e9
    fleet.refresh_reach()


def _on_low_traction(rover, t: float) -> None:
    f = rover.fleet
    slip = max(rover.lt_peak, rover.slip)
    load = rover.spin_load
    ex, ey = rover.lt_xy
    cx = ex + math.cos(rover.yaw) * (rover.cfg.wheel_x + 0.5 * ICE_ZONE_R)
    cy = ey + math.sin(rover.yaw) * (rover.cfg.wheel_x + 0.5 * ICE_ZONE_R)
    a = {"t": round(float(t), 1), "kind": "low_traction", "type": "low_traction",
         "at": [round(float(cx), 1), round(float(cy), 1)], "slip": round(slip, 2), "load": round(load, 2)}
    rover.anomaly = a
    rover.anomaly_log.append(a)
    rover.caution_until = t + 3.0
    rover.caution_v = rover.cfg.caution_v
    zone = ice_zone_at(f, cx, cy, -1.0)
    if zone is None:
        add_ice_zone(f, t, rover, cx, cy, ICE_ZONE_R)
        zone = f.ice_zones[-1]
    f.log(t, "warn", f"{rover.id} low traction: wheels turning but the rover barely follows (slip {slip * 100:.0f}% "
                     f"at load {load:.2f}, level ground). Frozen ground. Slowing, backing out and marking an ice "
                     f"zone for the fleet.", rover.id, at=(cx, cy))
    rover.mem.on_hazard("ice", t, (cx, cy), f"frozen ground: slip {slip * 100:.0f}% at motor load {load:.2f}")
    if rover.state != "recovering":
        rover.prev_state = rover.state
    if rover.prev_state == "exploring":
        rover.goal = None
    rover.state = "recovering"
    rover.rec_phase = "ice"
    rover.ice_zone = zone
    rover.ice_t0 = t
    rover.escape = list(reversed(rover.trail))
    rover.path = []
    rover.searches.clear()
    rover.lt_since = None
    rover.slip_since = None
    rover.hist.clear()
    rover._set_cmd(0.0, 0.0)


def ice_escape(rover, t: float) -> None:
    """Back out slowly along our own tracks until clear of the ice zone."""
    zx, zy, zr = rover.ice_zone[0], rover.ice_zone[1], rover.ice_zone[2]
    if math.hypot(rover.x - zx, rover.y - zy) >= zr + 0.9 or t - rover.ice_t0 > 40.0:
        rover._set_cmd(0.0, 0.0)
        rover.rec_phase = None
        rover.state = rover.prev_state if rover.prev_state not in ("recovering", "fault") else "exploring"
        rover.stall_cooldown = t + 2.0
        rover.plan_t = -1e9
        rover.next_goal_try = t
        rover.next_inv_plan = t
        rover.hist.clear()
        rover.fleet.log(t, "info", f"{rover.id} off the frozen ground. Planning around the ice zone.", rover.id,
                        at=(rover.x, rover.y))
        return
    while rover.escape and math.hypot(rover.escape[0][0] - rover.x, rover.escape[0][1] - rover.y) < 0.7:
        rover.escape.pop(0)
    if rover.escape:
        tx, ty = rover.escape[0]
    else:
        tx, ty = rover.x - 1.5 * math.cos(rover.yaw), rover.y - 1.5 * math.sin(rover.yaw)
    e = (math.atan2(ty - rover.y, tx - rover.x) - (rover.yaw + math.pi) + math.pi) % (2.0 * math.pi) - math.pi
    rover._set_cmd(-0.15 * max(0.4, math.cos(e)), max(-0.4, min(0.4, 1.2 * e)))
