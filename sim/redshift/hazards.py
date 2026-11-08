"""World changes for the demo: rockfall, soft sand, dust storm and rover fault.

The hazard list is ground truth for the UI only. Rovers never read it; they find out through their
own sensors:

  rockfall     boulders land on the route; LiDAR sees them, the shared map changes, the path is
               blocked and the rover re-plans.
  sand         hidden from LiDAR. Inside a patch the wheels lose grip and the rover sinks in, so
               the wheel encoders say "moving" while the body barely moves (wheel slip).
  dust_storm   the LiDAR's returns stop at about 3 m; the rovers notice their scans got short. Dust
               also covers the lander's solar panels, so docked rovers charge slower (they notice
               their own charge rate).
  rover_fault  one rover's motor controller drops out for a while (motor feedback); it tells the
               fleet, which takes over its work.
  spire_fall   an alien crystal spire topples across a rover's route (animated over about 1.5 s by
               rotating its mocap body about its base); LiDAR change detection sees a new long, low
               obstacle and the rover replans.
  lifeform     a demo convenience: moves a herd of grazers into view of a rover (or to a point). The
               rovers still have to notice them (moving LiDAR clusters, tracker.py).

Point and click: rockfall, sand, spire_fall and lifeform accept a point (x, y). It is refused (400) if
it is off the map, on the lander, within 2 m of a rover, or unsuitable (water, steep ground for
sand, nowhere safe to put it). dust_storm and rover_fault ignore the point.

Sand physics, per rover: each wheel inside a patch gets low friction and extra joint damping
(rolling resistance), and the body gets a horizontal drag that grows with the square of the share of
wheels in the sand (sinkage and bulldozing). Pushing forward means shoving fresh sand, so the drag is high;
backing out runs in the rover's own compacted ruts, so it is much lower. A rover therefore bogs
down driving in but can slowly reverse out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import v7_landforms

TYPES = ("rockfall", "sand", "dust_storm", "rover_fault", "spire_fall", "lifeform")
POINT_TYPES = ("rockfall", "sand", "spire_fall", "lifeform")
DROP_HEIGHT = 5.0      # boulders start this far above the ground
DROP_TIME = 0.9        # and land after this long (s)
FALL_TIME = 1.5        # a toppling spire lies on the ground after this long (s)
ROVER_HALF_DIAG = 0.75
NUDGES = [(0.0, 0.0)] + [(r * math.cos(a), r * math.sin(a)) for r in (0.7, 1.4, 2.0)
                         for a in np.linspace(0.0, 2 * math.pi, 8, endpoint=False)]


class HazardError(Exception):
    """A hazard request that cannot be honoured (becomes HTTP 400)."""


@dataclass
class Hazard:
    id: str
    type: str
    t: float
    rover: str | None = None
    x: float | None = None
    y: float | None = None
    radius: float | None = None
    rocks: list = field(default_factory=list)       # [[x, y, z_centre, radius]]
    known: bool = False
    until: float | None = None
    active: bool = True
    spire: str | None = None
    seg: tuple | None = None                        # spire_fall: (x0, y0, x1, y1) base and tip

    def to_msg(self, t: float) -> dict:
        m = {"id": self.id, "type": self.type, "t": round(self.t, 1), "rover": self.rover}
        if self.type == "spire_fall":
            m["spire"] = self.spire
            m["x0"], m["y0"], m["x1"], m["y1"] = (round(v, 2) for v in self.seg)
        elif self.x is not None:
            m["x"], m["y"] = round(self.x, 2), round(self.y, 2)
        if self.radius is not None:
            m["radius"] = round(self.radius, 2)
        if self.type == "rockfall":
            m["rocks"] = [[round(v, 2) for v in r] for r in self.rocks]
        if self.type == "sand":
            m["known"] = bool(self.known)
        if self.until is not None:
            m["remaining"] = round(max(0.0, self.until - t), 1) if self.active else 0.0
        m["active"] = bool(self.active)
        return m


def _remaining_path(rover) -> list:
    if not rover.path:
        return []
    return [(rover.x, rover.y)] + list(rover.path[max(rover.wp, 0):])


def _point_at(pts: list, s: float):
    """Point and unit tangent at arc length s along a polyline (None if it is shorter)."""
    acc = 0.0
    for k in range(len(pts) - 1):
        ax, ay = pts[k]
        bx, by = pts[k + 1]
        L = math.hypot(bx - ax, by - ay)
        if L < 1e-6:
            continue
        if acc + L >= s:
            f = (s - acc) / L
            return (ax + f * (bx - ax), ay + f * (by - ay)), ((bx - ax) / L, (by - ay) / L)
        acc += L
    return None


def _length(pts: list) -> float:
    return sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1))


class HazardManager:
    def __init__(self, sim):
        self.sim = sim
        self.cfg = sim.cfg
        self.items: list = []
        self.seq = 0
        self.rng = np.random.default_rng(self.cfg.seed * 7919 + 3)
        self.free_boulders = list(range(len(sim.world.boulders)))
        self.drops: list = []          # [mocap id, x, y, z_rest, t0]
        self.placed: list = []         # (x, y, z_rest, sx, sy, sz, yaw) boulders now on the map
        self.storm: Hazard | None = None
        self.freed_until: dict = {}    # rover id -> sim time until which sand ignores it (failsafe)
        self.version = 0               # bumps whenever the true heights change
        m = sim.model
        self._wheels = {}              # rover id -> (wheel geom ids, wheel dof ids)
        for r in sim.rovers:
            g = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{r.id}_wheel{k}") for k in range(4)]
            self._wheels[r.id] = (g, list(r.wdofs))
        self._base_fric = {gid: float(m.geom_friction[gid][0]) for gs, _ in self._wheels.values() for gid in gs}
        self._base_wdamp = {dof: float(m.dof_damping[dof]) for _, ds in self._wheels.values() for dof in ds}
        self._in_sand = {r.id: [False] * 4 for r in sim.rovers}
        self._body_damped = {r.id: False for r in sim.rovers}
        # spires: terrain ones first, then the spares parked below the map (same order as world.spires)
        ter = sim.terrain
        self.spires = list(getattr(ter, "spires", []))
        self.spire_mocap = {}
        wsp = list(sim.world.spires or [])
        for k, sp in enumerate(self.spires):
            if k < len(wsp):
                self.spire_mocap[sp.id] = wsp[k][0]
        self.spare_mocap = [m_ for m_, _g in wsp[len(self.spires):]]
        self.spire_standing = {sp.id: True for sp in self.spires}
        self.falls: list = []          # toppling spires: [mocap id, x0, y0, z0, ax, ay, angle, lift, t0]
        self.fallen: list = []         # (spire, x0, y0, z0, x1, y1, z1, from terrain)
        self.features_version = 0      # bumps when a spire falls (tick carries an updated `features`)
        self._conn_cache: dict = {}    # (passable grid, rover) -> connected before a placement (one request)

    # ================================================================== queries
    @property
    def patches(self) -> list:
        return [h for h in self.items if h.type == "sand"]

    def storm_active(self) -> bool:
        return self.storm is not None and self.storm.active

    def lidar_range(self) -> float:
        """Physical LiDAR range right now (what the hardware can see, not what the rover believes)."""
        return self.cfg.dust_lidar_range if self.storm_active() else self.cfg.lidar_range

    def charge_factor(self) -> float:
        """Share of the normal charging power the lander's solar panels deliver right now."""
        return self.cfg.dust_charge_factor if self.storm_active() else 1.0

    def lidar_noise(self):
        """(dropout probability, range noise sigma in m) for the LiDAR right now."""
        return (0.12, 0.04) if self.storm_active() else (0.0, 0.0)

    def to_msgs(self, t: float) -> list:
        return [h.to_msg(t) for h in self.items]

    def all_spires(self) -> list:
        return list(self.spires)

    def truth_heights(self, base: np.ndarray) -> np.ndarray:
        """True terrain heights including boulders that have been dropped and spires that fell."""
        if not self.placed and not self.fallen:
            return base
        ter = self.sim.terrain
        H = base.copy()
        for (sp, x0, y0, z0, x1, y1, z1, native) in self.fallen:
            ext = sp.radius + 0.1
            if native:                  # the column where it stood is open ground now
                i0, j0 = ter.ij(x0 - ext, y0 - ext)
                i1, j1 = ter.ij(x0 + ext, y0 + ext)
                H[j0:j1 + 1, i0:i1 + 1] = ter.truth_nospire[j0:j1 + 1, i0:i1 + 1]
            i0, j0 = ter.ij(min(x0, x1) - ext, min(y0, y1) - ext)
            i1, j1 = ter.ij(max(x0, x1) + ext, max(y0, y1) + ext)
            X = ter.X[j0:j1 + 1, i0:i1 + 1]
            Y = ter.Y[j0:j1 + 1, i0:i1 + 1]
            d, f = _seg_dist_np(X, Y, (x0, y0), (x1, y1))
            top = np.where(d <= sp.radius, z0 + f * (z1 - z0) + 2.0 * sp.radius, -np.inf)
            H[j0:j1 + 1, i0:i1 + 1] = np.maximum(H[j0:j1 + 1, i0:i1 + 1], top)
        for (x, y, z, sx, sy, sz, yaw) in self.placed:
            ext = max(sx, sy)
            i0, j0 = ter.ij(x - ext, y - ext)
            i1, j1 = ter.ij(x + ext, y + ext)
            X = ter.X[j0:j1 + 1, i0:i1 + 1] - x
            Y = ter.Y[j0:j1 + 1, i0:i1 + 1] - y
            c, s = math.cos(-yaw), math.sin(-yaw)
            lx = c * X - s * Y
            ly = s * X + c * Y
            q = (lx / sx) ** 2 + (ly / sy) ** 2
            top = np.where(q < 1, z + sz * np.sqrt(np.clip(1 - q, 0, 1)), -np.inf)
            H[j0:j1 + 1, i0:i1 + 1] = np.maximum(H[j0:j1 + 1, i0:i1 + 1], top)
        return H

    # ================================================================== triggering
    def trigger(self, kind, rover_id=None, x=None, y=None) -> dict:
        """Start a hazard now, optionally at a point (x, y) for rockfall, sand, spire_fall and
        lifeform. Returns its message dict, or raises HazardError."""
        sim = self.sim
        t = sim.t
        if kind not in TYPES:
            raise HazardError(f"unknown hazard type {kind!r} (use {', '.join(TYPES)})")
        rovers = {r.id: r for r in sim.rovers}
        if rover_id is not None and rover_id not in rovers:
            raise HazardError(f"unknown rover {rover_id!r} (rovers: {', '.join(rovers)})")
        self._conn_cache = {}
        if (x is None) != (y is None):
            raise HazardError("give both x and y, or neither")
        point = None
        if x is not None and kind in POINT_TYPES:
            point = (float(x), float(y))
            if not (math.isfinite(point[0]) and math.isfinite(point[1])):
                raise HazardError("x and y must be finite numbers")
            self._check_point(point)
        working = [r for r in sim.rovers if not r.motor_fault]
        if not working:
            raise HazardError("every rover is already faulted")
        if kind == "dust_storm":
            h = self._dust_storm(t)
        elif kind == "rover_fault":
            h = self._fault(t, rovers.get(rover_id), working)
        elif kind == "lifeform":
            h = self._lifeform(t, rovers.get(rover_id), working, point)
        elif point is not None:
            h = {"rockfall": self._rockfall_at, "sand": self._sand_at, "spire_fall": self._spire_fall_at}[kind](t, point)
        else:
            if rover_id is not None:
                r = rovers[rover_id]
                if r.motor_fault:
                    raise HazardError(f"{rover_id} is faulted and parked; pick a moving rover")
                cands = [r]
            else:
                cands = self._route_candidates(working)
                if not cands:
                    raise HazardError("no working rover is driving a planned route right now")
            h = None
            fn = {"rockfall": self._rockfall, "sand": self._sand, "spire_fall": self._spire_fall}[kind]
            for r in cands:
                h = fn(t, r)
                if h is not None:
                    break
            if h is None and kind == "spire_fall":
                for r in cands:         # no standing spire near any route: a spare one beside it
                    h = self._spire_fall(t, r, spare=True)
                    if h is not None:
                        break
            if h is None:
                who = cands[0].id if len(cands) == 1 else "any rover"
                what = {"rockfall": "rockfall", "sand": "sand patch (needs gentle open ground)",
                        "spire_fall": "falling spire"}[kind]
                raise HazardError(f"no valid placement for a {what} on the route of {who} right now")
        return h.to_msg(t)

    # ------------------------------------------------------------------ point checks
    def _check_point(self, p) -> None:
        cfg, sim = self.cfg, self.sim
        half = (cfg.nx - 1) * cfg.res / 2.0
        x, y = p
        if abs(x) > half - 1.0 or abs(y) > half - 1.0:
            raise HazardError(f"point ({x:g}, {y:g}) is off the map (x and y from {-half + 1:g} to {half - 1:g} m)")
        if math.hypot(x, y) < cfg.lander_radius + 1.0:
            raise HazardError(f"point ({x:g}, {y:g}) is on the lander pad")
        for r in sim.rovers:
            d = math.hypot(x - r.x, y - r.y)
            if d < 2.0:
                raise HazardError(f"point ({x:g}, {y:g}) is within 2 m of {r.id} ({d:.1f} m away)")
        if sim.terrain.wet_near(x, y, 0.3):
            raise HazardError(f"point ({x:g}, {y:g}) is in the water")

    def _rover_crossing(self, blocks: list):
        """The driving rover whose remaining route runs through any of these circles (x, y, r)."""
        best = None
        for r in self.sim.rovers:
            pts = _remaining_path(r)
            if len(pts) < 2 or r.motor_fault:
                continue
            acc = 0.0
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                L = math.hypot(bx - ax, by - ay)
                n = max(1, int(L / 0.3))
                for q in range(n + 1):
                    px, py = ax + (bx - ax) * q / n, ay + (by - ay) * q / n
                    if any(math.hypot(px - cx, py - cy) < cr + 0.5 for cx, cy, cr in blocks):
                        s_ = acc + L * q / n
                        if best is None or s_ < best[1]:
                            best = (r, s_)
                        break
                else:
                    acc += L
                    continue
                break
        return None if best is None else best[0]

    def _new(self, kind: str, t: float, **kw) -> Hazard:
        self.seq += 1
        h = Hazard(id=f"h{self.seq}", type=kind, t=float(t), **kw)
        self.items.append(h)
        return h

    def _route_candidates(self, working: list) -> list:
        driving = [r for r in working if r.path and r.goal is not None and
                   r.state in ("exploring", "investigating", "returning")]
        driving.sort(key=lambda r: (r.state != "exploring", -_length(_remaining_path(r))))
        return driving

    # ------------------------------------------------------------------ rockfall
    def _spot_ok(self, x: float, y: float, rad: float, rover_clear: float) -> bool:
        sim, cfg = self.sim, self.cfg
        half = (cfg.nx - 1) * cfg.res / 2.0
        if abs(x) > half - 2.5 or abs(y) > half - 2.5:
            return False
        if math.hypot(x, y) < cfg.lander_radius + rad + 1.5:
            return False
        if sim.terrain.wet_near(x, y, rad + 0.3):
            return False
        if not v7_landforms.spot_clear(sim.terrain, x, y, rad + 0.5):
            return False                       # v7: not in a pit, on a mesa wall or in the canyon
        for r in sim.rovers:
            if math.hypot(x - r.x, y - r.y) < rad + rover_clear:
                return False
        for tg in sim.terrain.targets:
            if math.hypot(x - tg.x, y - tg.y) < rad + 1.2:
                return False
        for b in self.placed:
            if math.hypot(x - b[0], y - b[1]) < rad + max(b[3], b[4]) + 0.2:
                return False
        for sp in self.spires:
            if self.spire_standing.get(sp.id, True) and math.hypot(x - sp.x, y - sp.y) < rad + sp.radius + 0.6:
                return False
        return True

    def _passable_base(self):
        """Ground a rover can really drive on (true terrain, spires and boulders already down, no water)."""
        from .terrain import slope_deg
        H = self.truth_heights(self.sim.terrain.truth_heights)
        lim = min(self.sim.fleet.slope_limit, self.cfg.passable_slope_deg)
        return (slope_deg(H, self.cfg.res) <= lim) & ~self.sim.terrain.wet

    def _still_connected(self, ok: np.ndarray, rover, spots, sizes) -> bool:
        """Would the rover still have a way home and to its goal with these boulders down? A rockfall
        should force a detour, not lock a rover in."""
        return self._connected_with(ok, rover, [(x, y, sx + 0.6) for (x, y), (sx, _sy, _sz) in zip(spots, sizes)])

    def _nobody_trapped(self, ok: np.ndarray, blocks: list) -> bool:
        """No working rover that can get home (and to its goal) now would be cut off by these
        obstacles. A rover whose goal is unreachable anyway does not veto a placement."""
        for r in self.sim.rovers:
            if r.motor_fault:
                continue
            key = (id(ok), r.id)
            before = self._conn_cache.get(key)
            if before is None:
                before = self._connected_with(ok, r, [])
                self._conn_cache[key] = before
            if before and not self._connected_with(ok, r, blocks):
                return False
        return True

    def _connected_with(self, ok: np.ndarray, rover, blocks: list) -> bool:
        """Same check for any obstacle given as circles (x, y, radius)."""
        ter = self.sim.terrain
        ok = ok.copy()
        for (x, y, rr) in blocks:
            i0, j0 = ter.ij(x - rr, y - rr)
            i1, j1 = ter.ij(x + rr, y + rr)
            X = ter.X[j0:j1 + 1, i0:i1 + 1] - x
            Y = ter.Y[j0:j1 + 1, i0:i1 + 1] - y
            ok[j0:j1 + 1, i0:i1 + 1] &= (X * X + Y * Y) > rr * rr
        reach = np.zeros_like(ok)
        i, j = ter.ij(rover.x, rover.y)
        reach[max(j - 2, 0):j + 3, max(i - 2, 0):i + 3] = True
        reach &= ok
        goals = [ter.ij(0.0, 0.0)]
        if rover.goal is not None:
            goals.append(ter.ij(rover.goal[0], rover.goal[1]))
        k = 0
        while True:
            g = reach.copy()
            g[1:, :] |= reach[:-1, :]
            g[:-1, :] |= reach[1:, :]
            g[:, 1:] |= reach[:, :-1]
            g[:, :-1] |= reach[:, 1:]
            g &= ok
            k += 1
            if all(g[max(gj - 2, 0):gj + 3, max(gi - 2, 0):gi + 3].any() for gi, gj in goals):
                return True
            if k % 8 == 0 and (g == reach).all():
                return False
            reach = g

    def _rockfall(self, t: float, rover, only_s: float | None = None):
        if len(self.free_boulders) < 3:
            raise HazardError("no boulders left for another rockfall (reset to clear the map)")
        pts = _remaining_path(rover)
        total = _length(pts)
        if total < 4.5:
            return None
        n = min(int(self.rng.integers(3, 6)), len(self.free_boulders))
        chosen = [self.free_boulders[k] for k in self.rng.permutation(len(self.free_boulders))[:n]]
        sizes = [self.sim.world.boulders[k][2] for k in chosen]
        gaps = [float(self.rng.uniform(0.45, 0.7)) for _ in range(n - 1)]   # narrower than a rover (1.16 m)
        fmap, ter = self.sim.map, self.sim.terrain
        base_ok = None
        for need_known in (True, False):
            for s in ((only_s,) if only_s is not None else (5.5, 5.0, 6.0, 4.5, 6.5, 4.0, 7.0)):
                if s > total - 0.3:
                    continue
                at = _point_at(pts, s)
                if at is None:
                    continue
                (cx, cy), (ux, uy) = at
                nx_, ny_ = -uy, ux
                # lay the boulders out side by side across the path, centred on it
                offs = [0.0]
                for k in range(1, n):
                    offs.append(offs[-1] + sizes[k - 1][0] + gaps[k - 1] + sizes[k][0])
                mid = (offs[0] + offs[-1]) / 2.0
                spots, ok = [], True
                for k in range(n):
                    o = offs[k] - mid
                    jit = float(self.rng.uniform(-0.25, 0.25))
                    x, y = cx + nx_ * o + ux * jit, cy + ny_ * o + uy * jit
                    sx = sizes[k][0]
                    i, j = fmap.ij(x, y)
                    if not self._spot_ok(x, y, sx, ROVER_HALF_DIAG + 1.0) or ter.max_slope_near(x, y, sx) > 24.0:
                        ok = False
                        break
                    if need_known and not (fmap.known[j, i] and fmap.slope[j, i] <= 15.0):
                        ok = False              # first choice: smooth ground the rovers already mapped
                        break
                    spots.append((x, y))
                if not ok:
                    continue
                if base_ok is None:
                    base_ok = self._passable_base()
                if not self._still_connected(base_ok, rover, spots, sizes):
                    continue
                return self._drop(t, rover, (cx, cy), chosen, sizes, spots, (offs[-1] - offs[0]) / 2.0)
        return None

    def _drop(self, t, rover, centre, chosen, sizes, spots, half_len) -> Hazard:
        d, ter = self.sim.data, self.sim.terrain
        rocks = []
        for k, (x, y), (sx, sy, sz) in zip(chosen, spots, sizes):
            mid = self.sim.world.boulders[k][0]
            yaw = float(self.rng.uniform(0, math.pi))
            ground = min(ter.height_at(x + dx, y + dy) for dx, dy in ((0, 0), (sx * 0.6, 0), (-sx * 0.6, 0),
                                                                       (0, sy * 0.6), (0, -sy * 0.6)))
            z = ground + 0.45 * sz
            d.mocap_quat[mid] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            d.mocap_pos[mid] = (x, y, z + DROP_HEIGHT)
            self.drops.append([mid, x, y, z, t])
            self.placed.append((x, y, z, sx, sy, sz, yaw))
            self.free_boulders.remove(k)
            rocks.append([x, y, z, sx])
        self.version += 1
        h = self._new("rockfall", t, rover=rover.id if rover is not None else None, x=centre[0], y=centre[1],
                      radius=half_len + max(s[0] for s in sizes), rocks=rocks)
        if rover is not None:
            self.sim.fleet.log(t, "hazard", f"Rockfall on {rover.name}'s route.", None, at=centre)
        else:
            self.sim.fleet.log(t, "hazard", f"Rockfall at ({centre[0]:.0f}, {centre[1]:.0f}).", None, at=centre)
        return h

    def animate(self, t: float) -> None:
        """Called every physics substep while boulders drop or spires topple."""
        d = self.sim.data
        keep = []
        for rec in self.drops:
            mid, x, y, z, t0 = rec
            f = min(1.0, max(0.0, (t - t0) / DROP_TIME))
            d.mocap_pos[mid] = (x, y, z + DROP_HEIGHT * (1.0 - f * f))
            if f < 1.0:
                keep.append(rec)
        self.drops = keep
        keep = []
        for rec in self.falls:
            mid, x0, y0, z0, ax, ay, angle, lift, t0 = rec
            f = min(1.0, max(0.0, (t - t0) / FALL_TIME))
            a = angle * f * f                          # slow at first, then faster, like a falling pole
            sa = math.sin(a / 2)
            d.mocap_quat[mid] = (math.cos(a / 2), ax * sa, ay * sa, 0.0)
            d.mocap_pos[mid] = (x0, y0, z0 + lift * f)
            if f < 1.0:
                keep.append(rec)
        self.falls = keep

    # ------------------------------------------------------------------ sand
    def _gentle(self, cx: float, cy: float, R: float) -> bool:
        """Is the ground under a patch gentle (so slip, not slope, is what the rover will feel)?"""
        ter = self.sim.terrain
        i0, j0 = ter.ij(cx - R, cy - R)
        i1, j1 = ter.ij(cx + R, cy + R)
        X = ter.X[j0:j1 + 1, i0:i1 + 1] - cx
        Y = ter.Y[j0:j1 + 1, i0:i1 + 1] - cy
        sl = ter.slope[j0:j1 + 1, i0:i1 + 1][(X * X + Y * Y) <= R * R]
        return len(sl) > 0 and float(sl.mean()) <= 10.5 and float(np.percentile(sl, 90)) <= 14.0

    def _sand(self, t: float, rover):
        cfg, sim = self.cfg, self.sim
        R = cfg.sand_radius
        pts = _remaining_path(rover)
        total = _length(pts)
        half = (cfg.nx - 1) * cfg.res / 2.0
        for s in (4.0, 4.5, 3.5, 5.0, 3.2, 5.5):
            if s > total + 0.5:
                continue
            at = _point_at(pts, min(s, total - 1e-6))
            if at is None:
                continue
            (cx, cy), (ux, uy) = at
            if s > total:                       # just past the goal, straight on
                cx, cy = cx + ux * (s - total), cy + uy * (s - total)
            if math.hypot(cx - rover.x, cy - rover.y) < R + 0.7:
                continue
            if self._sand_spot_ok(cx, cy, rover) is not None:
                continue
            h = self._new("sand", t, rover=rover.id, x=cx, y=cy, radius=R, known=False)
            sim.fleet.log(t, "hazard", f"Soft sand on {rover.name}'s route (invisible to LiDAR).", None, at=(cx, cy))
            return h
        return None

    def _sand_spot_ok(self, cx: float, cy: float, rover=None):
        """None if a sand patch can be centred here, else the reason."""
        cfg, sim = self.cfg, self.sim
        R = cfg.sand_radius
        half = (cfg.nx - 1) * cfg.res / 2.0
        if abs(cx) > half - R - 2.0 or abs(cy) > half - R - 2.0:
            return "too close to the map edge"
        if math.hypot(cx, cy) < cfg.lander_flat_radius + R:
            return "too close to the lander"
        if sim.terrain.wet_near(cx, cy, R + 0.5):
            return "too close to the water"
        if not self._gentle(cx, cy, R + 0.5):
            return "ground too steep or rough for sand"
        if any(math.hypot(cx - rk.x, cy - rk.y) < R + 0.5 + max(rk.sx, rk.sy) for rk in sim.terrain.rocks):
            return "rocks in the way"
        if any(math.hypot(cx - b[0], cy - b[1]) < R + 0.8 + max(b[3], b[4]) for b in self.placed):
            return "boulders in the way"
        if any(math.hypot(cx - sp.x, cy - sp.y) < R + 1.0 for sp in self.spires if self.spire_standing.get(sp.id, True)):
            return "a spire in the way"
        if any(math.hypot(cx - tg.x, cy - tg.y) < R + 1.5 for tg in sim.terrain.targets):
            return "a science target in the way"
        if any(o is not rover and math.hypot(cx - o.x, cy - o.y) < R + 1.5 for o in sim.rovers):
            return "a rover in the way"
        if any(math.hypot(cx - p.x, cy - p.y) < R + p.radius + 1.0 for p in self.patches):
            return "another sand patch there"
        return None

    def _sand_at(self, t: float, p) -> Hazard:
        R = self.cfg.sand_radius
        first = None
        for dx, dy in NUDGES:
            cx, cy = p[0] + dx, p[1] + dy
            why = self._sand_spot_ok(cx, cy, None)
            if why is None and any(math.hypot(cx - o.x, cy - o.y) < R + 0.7 for o in self.sim.rovers):
                why = "a rover in the way"
            if why is not None:
                first = first or why
                continue
            r = self._rover_crossing([(cx, cy, R)])
            h = self._new("sand", t, rover=r.id if r is not None else None, x=cx, y=cy, radius=R, known=False)
            where = f"{r.name}'s route" if r is not None else f"({cx:.0f}, {cy:.0f})"
            self.sim.fleet.log(t, "hazard", f"Soft sand on {where} (invisible to LiDAR).", None, at=(cx, cy))
            return h
        raise HazardError(f"no room for a {R:g} m sand patch at ({p[0]:g}, {p[1]:g}): {first}")

    # ------------------------------------------------------------------ rockfall at a point
    def _rockfall_at(self, t: float, p) -> Hazard:
        """Across the route of a rover heading that way if there is one, otherwise a cluster here."""
        if len(self.free_boulders) < 3:
            raise HazardError("no boulders left for another rockfall (reset to clear the map)")
        rover = None
        best = None
        for r in self.sim.rovers:
            pts = _remaining_path(r)
            if len(pts) < 2 or r.motor_fault:
                continue
            total = _length(pts)
            s_ = 3.0
            while s_ < total - 0.3:
                at = _point_at(pts, s_)
                if at is not None and math.hypot(at[0][0] - p[0], at[0][1] - p[1]) < 2.5:
                    if best is None or s_ < best[1]:
                        best = (r, s_)
                    break
                s_ += 0.3
        if best is not None:
            rover = best[0]
            h = self._rockfall(t, rover, only_s=best[1])
            if h is not None:
                return h
        n = min(int(self.rng.integers(3, 5)), len(self.free_boulders))
        chosen = [self.free_boulders[k] for k in self.rng.permutation(len(self.free_boulders))[:n]]
        sizes = [self.sim.world.boulders[k][2] for k in chosen]
        ter = self.sim.terrain
        base_ok = None
        for dx, dy in NUDGES:
            cx, cy = p[0] + dx, p[1] + dy
            spots = []
            for k in range(n):
                sx = sizes[k][0]
                for _ in range(30):
                    a, rr = float(self.rng.uniform(0, 2 * math.pi)), (0.0 if k == 0 else float(self.rng.uniform(0.9, 1.8)))
                    x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
                    if any(math.hypot(x - qx, y - qy) < sx + sizes[q][0] + 0.1 for q, (qx, qy) in enumerate(spots)):
                        continue
                    if self._spot_ok(x, y, sx, ROVER_HALF_DIAG + 1.0) and ter.max_slope_near(x, y, sx) <= 24.0:
                        spots.append((x, y))
                        break
                else:
                    break
            if len(spots) < n:
                continue
            if base_ok is None:
                base_ok = self._passable_base()
            if self._nobody_trapped(base_ok, [(x, y, sizes[k][0] + 0.6) for k, (x, y) in enumerate(spots)]):
                r = self._rover_crossing([(x, y, sizes[k][0]) for k, (x, y) in enumerate(spots)])
                return self._drop(t, r, (cx, cy), chosen, sizes, spots, 1.8)
        raise HazardError(f"no safe spot for a rockfall at ({p[0]:g}, {p[1]:g}) (steep ground, objects in the way, "
                          f"or it would trap a rover)")

    # ------------------------------------------------------------------ spire fall
    def _fall_blocks(self, x0, y0, x1, y1, r) -> list:
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(L / 0.4) + 1)
        return [(x0 + (x1 - x0) * k / (n - 1), y0 + (y1 - y0) * k / (n - 1), r + 0.6) for k in range(n)]

    def _fall_ok(self, x0, y0, ux, uy, length, r, base_ok, near_rover=None) -> tuple | None:
        """Where a spire at (x0, y0) falling along (ux, uy) would lie: (x1, y1) if that is safe (on the
        map, dry, clear of rovers, the lander and targets, nobody trapped), else None."""
        cfg, sim = self.cfg, self.sim
        ter = sim.terrain
        half = (cfg.nx - 1) * cfg.res / 2.0
        reach = length - r
        x1, y1 = x0 + ux * reach, y0 + uy * reach
        if abs(x1) > half - 2.0 or abs(y1) > half - 2.0:
            return None
        for k in range(1, 11):
            px, py = x0 + ux * reach * k / 10, y0 + uy * reach * k / 10
            if ter.wet_near(px, py, r + 0.2) or math.hypot(px, py) < cfg.lander_radius + 1.5:
                return None
            if any(math.hypot(px - tg.x, py - tg.y) < 1.0 for tg in ter.targets):
                return None
            if any(math.hypot(px - b[0], py - b[1]) < max(b[3], b[4]) + r for b in self.placed):
                return None
        for o in sim.rovers:
            d, _ = _seg_dist_np(np.array([o.x]), np.array([o.y]), (x0, y0), (x1, y1))
            if float(d[0]) < 1.7:
                return None
        blocks = self._fall_blocks(x0, y0, x1, y1, r)
        if not self._nobody_trapped(base_ok, blocks):
            return None
        return x1, y1

    def _spare_spire(self, x: float, y: float):
        """Bring a spare spire up at (x, y), standing. Returns its Spire spec (or None if none left)."""
        if not self.spare_mocap:
            return None
        from .terrain import Spire
        mid = self.spare_mocap.pop(0)
        n = len(self.spires) + 1
        z = self.sim.terrain.height_at(x, y)
        sp = Spire(f"s{n}", x, y, z, 3.0, 0.32, [])
        self.spires.append(sp)
        self.spire_mocap[sp.id] = mid
        self.spire_standing[sp.id] = True
        d = self.sim.data
        d.mocap_pos[mid] = (x, y, z)
        d.mocap_quat[mid] = (1.0, 0.0, 0.0, 0.0)
        return sp

    def _spire_fall(self, t: float, rover, spare: bool = False):
        """Topple a standing spire across the rover's route 4.5 to 8.5 m ahead (or, with spare=True, a
        spare spire that comes up beside the route). None if nothing fits."""
        pts = _remaining_path(rover)
        total = _length(pts)
        if total < 4.0:
            return None
        ter = self.sim.terrain
        base_ok = None
        for s_ in (5.5, 6.5, 4.5, 7.5, 8.5):
            if s_ > total - 0.5:
                continue
            at = _point_at(pts, s_)
            if at is None:
                continue
            (px, py), (ux, uy) = at
            if not spare:
                cands = sorted((sp for sp in self.spires if self.spire_standing.get(sp.id)),
                               key=lambda sp: math.hypot(sp.x - px, sp.y - py))
                for sp in cands:
                    dd = math.hypot(px - sp.x, py - sp.y)
                    if dd > sp.height - 0.7:
                        break
                    if dd < 0.8:
                        continue
                    if base_ok is None:
                        base_ok = self._passable_base()
                    end = self._fall_ok(sp.x, sp.y, (px - sp.x) / dd, (py - sp.y) / dd, sp.height, sp.radius, base_ok)
                    if end is not None:
                        return self._topple(t, rover, sp, end, (px, py))
                continue
            nx_, ny_ = -uy, ux
            for side in (1.0, -1.0):
                for off in (1.6, 2.2):
                    bx, by = px + nx_ * side * off, py + ny_ * side * off
                    if not self._spot_ok(bx, by, 0.4, 2.5) or ter.max_slope_near(bx, by, 0.6) > 15.0:
                        continue
                    if base_ok is None:
                        base_ok = self._passable_base()
                    end = self._fall_ok(bx, by, -nx_ * side, -ny_ * side, 3.0, 0.32, base_ok)
                    if end is None:
                        continue
                    sp = self._spare_spire(bx, by)
                    if sp is None:
                        return None
                    return self._topple(t, rover, sp, end, (px, py))
        return None

    def _spire_fall_at(self, t: float, p) -> Hazard:
        """Topple the nearest standing spire within about 6 m toward the point, or drop a new one there."""
        base_ok = self._passable_base()
        cands = sorted((sp for sp in self.spires if self.spire_standing.get(sp.id)),
                       key=lambda sp: math.hypot(sp.x - p[0], sp.y - p[1]))
        if cands and math.hypot(cands[0].x - p[0], cands[0].y - p[1]) <= 6.0:
            sp = cands[0]
            dd = math.hypot(p[0] - sp.x, p[1] - sp.y)
            if dd > 0.5:
                dirs = [((p[0] - sp.x) / dd, (p[1] - sp.y) / dd)]
            else:
                dirs = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
            for ux, uy in dirs:
                end = self._fall_ok(sp.x, sp.y, ux, uy, sp.height, sp.radius, base_ok)
                if end is not None:
                    return self._topple(t, None, sp, end, p)
            raise HazardError(f"spire {sp.id} near ({p[0]:g}, {p[1]:g}) cannot fall that way safely (it would hit "
                              f"a rover, the water or the lander, or trap a rover)")
        if not self.spare_mocap:
            raise HazardError("no spare spires left to drop (reset to restore them)")
        ter = self.sim.terrain
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            ux, uy = math.cos(a), math.sin(a)
            bx, by = p[0] - 1.5 * ux, p[1] - 1.5 * uy
            if not self._spot_ok(bx, by, 0.4, 2.5) or ter.max_slope_near(bx, by, 0.6) > 15.0:
                continue
            end = self._fall_ok(bx, by, ux, uy, 3.0, 0.32, base_ok)
            if end is None:
                continue
            sp = self._spare_spire(bx, by)
            return self._topple(t, None, sp, end, p)
        raise HazardError(f"no safe spot to drop a spire at ({p[0]:g}, {p[1]:g}) (steep ground, objects in the way, "
                          f"or it would trap a rover)")

    def _topple(self, t: float, rover, sp, end, cross) -> Hazard:
        """Start the fall: rotate the spire's mocap body about its base over FALL_TIME seconds so it
        ends up lying from the base to `end`."""
        ter = self.sim.terrain
        x1, y1 = end
        z0, z1 = sp.z, ter.height_at(x1, y1)
        dx, dy, dz = x1 - sp.x, y1 - sp.y, z1 - z0
        n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        wx, wy, wz = dx / n, dy / n, dz / n
        h = math.hypot(wx, wy) or 1.0
        ax, ay = -wy / h, wx / h                      # rotation axis: z cross w (horizontal)
        angle = math.acos(max(-1.0, min(1.0, wz)))
        lift = sp.radius * (1.0 - wz)
        self.falls.append([self.spire_mocap[sp.id], sp.x, sp.y, z0, ax, ay, angle, lift, t])
        native = sp in getattr(ter, "spires", [])
        self.fallen.append((sp, sp.x, sp.y, z0, x1, y1, z1, native))
        self.spire_standing[sp.id] = False
        self.features_version += 1
        self.version += 1
        if rover is None:
            rover = self._rover_crossing(self._fall_blocks(sp.x, sp.y, x1, y1, sp.radius))
        hz = self._new("spire_fall", t, rover=rover.id if rover is not None else None, spire=sp.id,
                       seg=(sp.x, sp.y, x1, y1))
        if rover is not None:
            self.sim.fleet.log(t, "hazard", f"Spire fell across {rover.name}'s route.", None, at=cross)
        else:
            self.sim.fleet.log(t, "hazard", f"A spire fell near ({cross[0]:.0f}, {cross[1]:.0f}).", None, at=cross)
        return hz

    # ------------------------------------------------------------------ lifeform (demo)
    def _lifeform(self, t: float, rover, working: list, p) -> Hazard:
        sim = self.sim
        lf = sim.lifeforms
        if not lf.herds:
            raise HazardError("no lifeforms on this planet (lifeforms are switched off)")
        spot = None
        if p is not None:
            for dx, dy in NUDGES:
                x, y = p[0] + dx, p[1] + dy
                if lf.spot_ok(x, y) and all(math.hypot(x - r.x, y - r.y) >= 3.0 for r in sim.rovers):
                    spot = (x, y)
                    break
            if spot is None:
                raise HazardError(f"no room for a herd at ({p[0]:g}, {p[1]:g}) (steep ground, rocks or water)")
            if rover is None:
                rover = min(sim.rovers, key=lambda r: math.hypot(r.x - spot[0], r.y - spot[1]))
            face = math.atan2(spot[1] - rover.y, spot[0] - rover.x) + math.pi / 2
        else:
            pool = [rover] if rover is not None else sorted(
                working, key=lambda r: (r.state != "exploring", r.state not in ("exploring", "investigating"),
                                        r.state == "charging"))
            for r in pool:
                for dist in (6.5, 7.0, 6.0, 7.5):
                    for a in (0.0, 0.3, -0.3, 0.6, -0.6):
                        x, y = r.x + dist * math.cos(r.yaw + a), r.y + dist * math.sin(r.yaw + a)
                        if lf.spot_ok(x, y) and all(math.hypot(x - o.x, y - o.y) >= 4.5 for o in sim.rovers):
                            spot, rover = (x, y), r
                            break
                    if spot is not None:
                        break
                if spot is not None:
                    break
            if spot is None:
                raise HazardError("no open ground in view of a rover for a herd right now")
            face = rover.yaw + math.pi / 2           # they walk across the rover's view
        # bring a herd nobody is watching (the nearest such), preferably one the fleet has not counted yet,
        # and let the fleet notice it afresh
        f = sim.fleet
        watched = [(r.x, r.y) for r in sim.rovers if r.state == "observing"]
        free = [hd for hd in lf.herds if not any(math.hypot(hd.centre()[0] - wx, hd.centre()[1] - wy) < 15.0
                                                 for wx, wy in watched)] or lf.herds

        def counted(hd):
            return next((k for k, (hx, hy, _c) in enumerate(f.lifeform_herds)
                         if math.hypot(hd.centre()[0] - hx, hd.centre()[1] - hy) < 20.0), None)

        fresh = [hd for hd in free if counted(hd) is None] or free
        herd = min(fresh, key=lambda hd: math.hypot(hd.centre()[0] - spot[0], hd.centre()[1] - spot[1]))
        k = counted(herd)
        lf.move_herd(herd, spot[0], spot[1], face)
        if k is not None:
            # demo bookkeeping: the teleported herd is the same individuals, so its count moves with it
            # (otherwise lifeforms_seen would count them twice; the rovers never learn about the move)
            hx, hy, c = f.lifeform_herds[k]
            f.lifeform_herds[k] = (spot[0], spot[1], c)
        f.observed_areas = [a for a in f.observed_areas if math.hypot(a[0] - spot[0], a[1] - spot[1]) > 20.0]
        f.lifeforms_seen = sum(c for _, _, c in f.lifeform_herds)
        for tr in f.tracker.tracks:
            if math.hypot(tr.x - spot[0], tr.y - spot[1]) < 20.0 and tr.observer is None:
                tr.done_until = -1e9
                tr.ignored = False
        h = self._new("lifeform", t, rover=rover.id, x=spot[0], y=spot[1])
        sim.fleet.log(t, "hazard", f"A herd of creatures wanders into view of {rover.name}.", None, at=spot)
        return h

    def on_soft_zone(self, x: float, y: float, radius: float) -> None:
        """A rover reported soft ground: mark the matching hidden patch as known (UI bookkeeping)."""
        for p in self.patches:
            if math.hypot(p.x - x, p.y - y) < p.radius + radius:
                p.known = True

    def free_rover(self, rid: str, t: float, seconds: float = 20.0) -> None:
        """Failsafe: the sand lets go of this rover for a while (it dug itself out)."""
        self.freed_until[rid] = t + seconds

    def update_physics(self, t: float) -> None:
        """Once per decision cycle, before the physics substeps: sand contact properties."""
        from . import v7_surface
        patches = self.patches
        surf = v7_surface.surface_of(self.sim)     # v7 region causes (dunes, ice), None when off
        if not patches and not any(self._body_damped.values()) and surf is None:
            return
        m, d, cfg = self.sim.model, self.sim.data, self.cfg
        for r in self.sim.rovers:
            gids, dofs = self._wheels[r.id]
            freed = self.freed_until.get(r.id, -1.0) > t
            n_in = 0
            regs = surf.state.setdefault(r.id, [0] * len(gids)) if surf is not None else None
            for k, gid in enumerate(gids):
                inside = False
                if patches and not freed:
                    wx, wy = d.geom_xpos[gid][0], d.geom_xpos[gid][1]
                    inside = any((wx - p.x) ** 2 + (wy - p.y) ** 2 <= p.radius ** 2 for p in patches)
                reg_changed = False
                if regs is not None:
                    reg = surf.region_at(d.geom_xpos[gid][0], d.geom_xpos[gid][1])
                    reg_changed, regs[k] = reg != regs[k], reg
                n_in += inside
                if inside != self._in_sand[r.id][k] or reg_changed:
                    self._in_sand[r.id][k] = inside
                    fric, damp = v7_surface.compose_wheel(self._base_fric[gid], self._base_wdamp[dofs[k]], inside,
                                                          regs[k] if regs is not None else 0, cfg)
                    m.geom_friction[gid][0] = fric
                    m.dof_damping[dofs[k]] = damp
            b = r.vadr
            if n_in:
                weight = r.mass * cfg.gravity
                frac = n_in / 4.0
                k_lin = cfg.sand_drag_rev if r.cmd_v < -0.05 else cfg.sand_drag_fwd
                # one wheel on the sand barely matters; all four sunk in and the rover is stuck
                m.dof_damping[b] = m.dof_damping[b + 1] = frac * frac * k_lin * weight
                # turning in place means scraping all four wheels sideways through the sand
                m.dof_damping[b + 5] = frac * frac * cfg.sand_friction * weight * cfg.wheel_y / 0.15
                self._body_damped[r.id] = True
            elif self._body_damped[r.id]:
                m.dof_damping[b] = m.dof_damping[b + 1] = m.dof_damping[b + 5] = 0.0
                self._body_damped[r.id] = False

    # ------------------------------------------------------------------ dust storm
    def _dust_storm(self, t: float) -> Hazard:
        if self.storm_active():
            raise HazardError("a dust storm is already blowing")
        self.storm = self._new("dust_storm", t, until=t + self.cfg.dust_storm_s)
        self.sim.fleet.log(t, "hazard", "Dust storm rolling in.", None)
        return self.storm

    # ------------------------------------------------------------------ rover fault
    def _fault(self, t: float, rover, working: list) -> Hazard:
        if rover is not None and rover.motor_fault:
            raise HazardError(f"{rover.id} is already faulted")
        if len(working) <= 1:
            raise HazardError("that would leave no working rover")
        if rover is None:
            busy = [r for r in working if r.state not in ("charging", "idle")]
            pool = busy or working
            rover = pool[int(self.rng.integers(len(pool)))]
        rover.motor_fault = True
        h = self._new("rover_fault", t, rover=rover.id, until=t + self.cfg.fault_s)
        self.sim.fleet.log(t, "hazard", f"{rover.name} motor controller fault.", None, at=(rover.x, rover.y))
        return h

    # ================================================================== timers
    def step(self, t: float) -> None:
        """Once per decision cycle, after the physics: hazards that run out."""
        for h in self.items:
            if not h.active or h.until is None or t < h.until:
                continue
            h.active = False
            if h.type == "dust_storm":
                self.sim.fleet.log(t, "hazard", "Dust storm clearing.", None)
            elif h.type == "rover_fault":
                for r in self.sim.rovers:
                    if r.id == h.rover:
                        r.motor_fault = False


def _seg_dist_np(X, Y, p0, p1):
    """Distance from points to the segment p0-p1, and the position along it (0..1)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L2 = dx * dx + dy * dy or 1e-9
    f = np.clip(((X - p0[0]) * dx + (Y - p0[1]) * dy) / L2, 0.0, 1.0)
    return np.hypot(X - (p0[0] + f * dx), Y - (p0[1] + f * dy)), f
