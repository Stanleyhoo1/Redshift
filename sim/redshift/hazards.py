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

TYPES = ("rockfall", "sand", "dust_storm", "rover_fault")
DROP_HEIGHT = 5.0      # boulders start this far above the ground
DROP_TIME = 0.9        # and land after this long (s)
ROVER_HALF_DIAG = 0.75


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

    def to_msg(self, t: float) -> dict:
        m = {"id": self.id, "type": self.type, "t": round(self.t, 1), "rover": self.rover}
        if self.x is not None:
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

    def truth_heights(self, base: np.ndarray) -> np.ndarray:
        """True terrain heights including boulders that have been dropped."""
        if not self.placed:
            return base
        ter = self.sim.terrain
        H = base.copy()
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
    def trigger(self, kind, rover_id=None) -> dict:
        """Start a hazard now. Returns its message dict, or raises HazardError."""
        sim = self.sim
        t = sim.t
        if kind not in TYPES:
            raise HazardError(f"unknown hazard type {kind!r} (use rockfall, sand, dust_storm or rover_fault)")
        rovers = {r.id: r for r in sim.rovers}
        if rover_id is not None and rover_id not in rovers:
            raise HazardError(f"unknown rover {rover_id!r} (rovers: {', '.join(rovers)})")
        working = [r for r in sim.rovers if not r.motor_fault]
        if not working:
            raise HazardError("every rover is already faulted")
        if kind == "dust_storm":
            h = self._dust_storm(t)
        elif kind == "rover_fault":
            h = self._fault(t, rovers.get(rover_id), working)
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
            for r in cands:
                h = self._rockfall(t, r) if kind == "rockfall" else self._sand(t, r)
                if h is not None:
                    break
            if h is None:
                who = cands[0].id if len(cands) == 1 else "any rover"
                what = "rockfall" if kind == "rockfall" else "sand patch (needs gentle open ground)"
                raise HazardError(f"no valid placement for a {what} on the route of {who} right now")
        return h.to_msg(t)

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
        for r in sim.rovers:
            if math.hypot(x - r.x, y - r.y) < rad + rover_clear:
                return False
        for tg in sim.terrain.targets:
            if math.hypot(x - tg.x, y - tg.y) < rad + 1.2:
                return False
        for b in self.placed:
            if math.hypot(x - b[0], y - b[1]) < rad + max(b[3], b[4]) + 0.2:
                return False
        return True

    def _passable_base(self):
        """Ground a rover can really drive on (true terrain and boulders already down)."""
        from .terrain import slope_deg
        H = self.truth_heights(self.sim.terrain.truth_heights)
        lim = min(self.sim.fleet.slope_limit, 20.0)
        return slope_deg(H, self.cfg.res) <= lim

    def _still_connected(self, ok: np.ndarray, rover, spots, sizes) -> bool:
        """Would the rover still have a way home and to its goal with these boulders down? A rockfall
        should force a detour, not lock a rover in."""
        ter = self.sim.terrain
        ok = ok.copy()
        for (x, y), (sx, _sy, _sz) in zip(spots, sizes):
            rr = sx + 0.6
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

    def _rockfall(self, t: float, rover):
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
            for s in (5.5, 5.0, 6.0, 4.5, 6.5, 4.0, 7.0):
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
        h = self._new("rockfall", t, rover=rover.id, x=centre[0], y=centre[1],
                      radius=half_len + max(s[0] for s in sizes), rocks=rocks)
        self.sim.fleet.log(t, "hazard", f"Rockfall on {rover.name}'s route.", None, at=centre)
        return h

    def animate(self, t: float) -> None:
        """Called every physics substep while boulders are falling."""
        d = self.sim.data
        keep = []
        for rec in self.drops:
            mid, x, y, z, t0 = rec
            f = min(1.0, max(0.0, (t - t0) / DROP_TIME))
            d.mocap_pos[mid] = (x, y, z + DROP_HEIGHT * (1.0 - f * f))
            if f < 1.0:
                keep.append(rec)
        self.drops = keep

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
            if abs(cx) > half - R - 2.0 or abs(cy) > half - R - 2.0:
                continue
            if math.hypot(cx, cy) < cfg.lander_flat_radius + R:
                continue
            if not self._gentle(cx, cy, R + 0.5):
                continue
            if any(math.hypot(cx - rk.x, cy - rk.y) < R + 0.5 + max(rk.sx, rk.sy) for rk in sim.terrain.rocks):
                continue
            if any(math.hypot(cx - b[0], cy - b[1]) < R + 0.8 + max(b[3], b[4]) for b in self.placed):
                continue
            if any(math.hypot(cx - tg.x, cy - tg.y) < R + 1.5 for tg in sim.terrain.targets):
                continue
            if any(o is not rover and math.hypot(cx - o.x, cy - o.y) < R + 1.5 for o in sim.rovers):
                continue
            if any(math.hypot(cx - p.x, cy - p.y) < R + p.radius + 1.0 for p in self.patches):
                continue
            h = self._new("sand", t, rover=rover.id, x=cx, y=cy, radius=R, known=False)
            sim.fleet.log(t, "hazard", f"Soft sand on {rover.name}'s route (invisible to LiDAR).", None, at=(cx, cy))
            return h
        return None

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
        patches = self.patches
        if not patches and not any(self._body_damped.values()):
            return
        m, d, cfg = self.sim.model, self.sim.data, self.cfg
        for r in self.sim.rovers:
            gids, dofs = self._wheels[r.id]
            freed = self.freed_until.get(r.id, -1.0) > t
            n_in = 0
            for k, gid in enumerate(gids):
                inside = False
                if patches and not freed:
                    wx, wy = d.geom_xpos[gid][0], d.geom_xpos[gid][1]
                    inside = any((wx - p.x) ** 2 + (wy - p.y) ** 2 <= p.radius ** 2 for p in patches)
                n_in += inside
                if inside != self._in_sand[r.id][k]:
                    self._in_sand[r.id][k] = inside
                    m.geom_friction[gid][0] = cfg.sand_friction if inside else self._base_fric[gid]
                    m.dof_damping[dofs[k]] = self._base_wdamp[dofs[k]] + (cfg.sand_wheel_damping if inside else 0.0)
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
