"""v7 geysers: vents that erupt on a fixed schedule, and steam plumes that scatter the LiDAR.

Placement and the schedule come from the feature's own random stream, default_rng([seed, 7106]), so
they are fixed per seed and never touch the other streams. Each vent is a low solid cone (a squat
ellipsoid geom "geyser_<i>", LiDAR group 1), so the map sees it like a rock.

While a vent erupts, a steam column (radius PLUME_R, a few metres tall) stands over it. A LiDAR ray
that passes through the column is lost, and some of those rays come back as short noisy returns
inside the steam. The column has no MuJoCo geom: the scatter is worked out per ray from the ray
geometry, and the scattered returns are kept apart from the real ones. That routing is a stated
simplification, like the motion tracker's routing of creature hits by geom id: a real rover would
tell steam from rock by the diffuse, short, flickering returns. Plume returns go into the rover's raw
scan only. They never reach the shared height map, the motion tracker or change detection.

What the rovers learn comes only from those plume returns: the rover that sees a plume estimates its
centre, the fleet marks a keep-out zone (fleet.geyser_zones) so every rover stays clear for good, and a
second eruption seen at the same vent gives the fleet an estimate of the period.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K_GEYSERS = 7106
PLUME_R = 1.5            # steam column radius (m)
VENT_R = 0.9             # vent cone footprint radius (m)
VENT_H = 0.45            # vent cone height above the ground (m)
KEEP_OUT_M = 6.0         # keep-out radius the fleet marks around a vent it saw erupting
PLUME_RETURN = 0.35      # share of the rays through the steam that come back as short returns
MIN_PLUME_PTS = 6        # scattered returns before a rover calls it a plume
SAME_VENT_M = 4.0        # two plume sightings this close are the same vent
NEW_ERUPTION_S = 20.0    # a sighting this long after the last one is a new eruption


@dataclass
class Vent:
    id: str
    x: float
    y: float
    z: float                # ground height at the vent
    period_s: float
    dur_s: float
    phase_s: float
    plume_h: float

    def active(self, t: float) -> bool:
        return ((t + self.phase_s) % self.period_s) < self.dur_s

    def next_change(self, t: float) -> float:
        """Sim time of the next eruption start or end after t."""
        u = (t + self.phase_s) % self.period_s
        return t + (self.dur_s - u if u < self.dur_s else self.period_s - u)


def plan_vents(cfg, ter) -> list:
    """1 to 3 vents (more on big maps) on gentle, reachable ground away from the lander, targets,
    rocks, spires and water. Every number comes from the geyser stream."""
    rng = np.random.default_rng([int(cfg.seed), K_GEYSERS])
    half = cfg.map_size / 2.0 - 6.0
    n = int(rng.integers(1, 4))
    if ter.area > 1.0:
        n = min(6, int(round(n * ter.area)))
    lander_gap = min(20.0, 0.4 * cfg.map_size)
    vents: list = []
    for _ in range(600):
        if len(vents) >= n:
            break
        x, y = float(rng.uniform(-half, half)), float(rng.uniform(-half, half))
        period = float(rng.uniform(60.0, 120.0))
        dur = float(rng.uniform(8.0, 15.0))
        phase = float(rng.uniform(0.0, period))
        plume_h = float(rng.uniform(6.0, 10.0))
        if math.hypot(x, y) < lander_gap:
            continue
        i, j = ter.ij(x, y)
        if not ter.reach[j, i] or ter.max_slope_near(x, y, 2.0) > 8.0 or ter.wet_near(x, y, 4.0):
            continue
        if any(math.hypot(x - t.x, y - t.y) < 9.0 for t in ter.targets):
            continue
        if any(math.hypot(x - r.x, y - r.y) < 2.5 for r in ter.rocks):
            continue
        if any(math.hypot(x - s.x, y - s.y) < 5.0 for s in ter.spires):
            continue
        if any(math.hypot(x - v.x, y - v.y) < 15.0 for v in vents):
            continue
        vents.append(Vent(f"g{len(vents) + 1}", round(x, 2), round(y, 2), round(ter.height_at(x, y), 3),
                          round(period, 1), round(dur, 1), round(phase, 1), round(plume_h, 1)))
    return vents


def vent_geoms_xml(vents: list) -> list:
    """The vent cones: squat ellipsoids half sunk in the ground, solid and seen by the LiDAR."""
    return [f'<geom name="geyser_{k}" type="ellipsoid" size="{VENT_R:.3f} {VENT_R:.3f} {VENT_H:.3f}" '
            f'pos="{v.x:.3f} {v.y:.3f} {v.z:.3f}" material="rock" group="1" friction="0.8 0.005 0.0001"/>'
            for k, v in enumerate(vents)]


class GeyserManager:
    """Runs the eruption schedule, scatters LiDAR rays in active plumes and keeps what the fleet
    learned (keep-out zones and period estimates) from the plume returns the rovers saw."""

    def __init__(self, sim):
        self.sim = sim
        self.vents: list = list(getattr(sim.terrain, "geysers", []))
        self.rng = np.random.default_rng([int(sim.cfg.seed), K_GEYSERS, 1])    # scatter noise
        self._state = [v.active(0.0) for v in self.vents]
        self.eruptions = 0                # eruption starts seen by the schedule (truth side)
        sim.fleet.geyser_zones = []       # [(x, y, radius, rover id, t)] learned keep-out zones
        self._seen: list = []             # per zone: [last sighting t, [eruption start times]]

    # ------------------------------------------------------------------ schedule
    def active_vents(self, t: float) -> list:
        return [v for v in self.vents if v.active(t)]

    def update(self, t: float) -> bool:
        """Advance the schedule. Returns True (and bumps the features version, so the next tick
        resends `features`) when an eruption starts or ends."""
        changed = False
        for k, v in enumerate(self.vents):
            a = v.active(t)
            if a != self._state[k]:
                self._state[k] = a
                changed = True
                if a:
                    self.eruptions += 1
        if changed and self.sim.hazards is not None:
            self.sim.hazards.features_version += 1
        return changed

    def to_msg(self, t: float) -> list:
        return [{"id": v.id, "x": v.x, "y": v.y, "radius": PLUME_R, "active": bool(v.active(t)),
                 "period_s": v.period_s, "next_s": round(v.next_change(t), 1), "height": v.plume_h}
                for v in self.vents]

    # ------------------------------------------------------------------ LiDAR scatter
    def scatter(self, rover, t: float, pnt, vec, dist, hit, rng_max: float):
        """Rays through an active plume are lost; some come back short and noisy. Returns the new
        hit mask and (plume points (k, 3), mask of the rays the steam took). Sets rover.plume_pts."""
        rover.plume_pts = None
        if not self.vents:
            return hit, None
        through = np.zeros(len(dist), bool)
        pts_all = []
        for v in self.vents:
            if not v.active(t):
                continue
            dx, dy = pnt[0] - v.x, pnt[1] - v.y
            if math.hypot(dx, dy) > rng_max + PLUME_R:
                continue
            vx, vy = vec[:, 0], vec[:, 1]
            a = vx * vx + vy * vy
            b = 2.0 * (dx * vx + dy * vy)
            c = dx * dx + dy * dy - PLUME_R * PLUME_R
            disc = b * b - 4.0 * a * c
            ok = (disc > 0.0) & (a > 1e-9)
            sq = np.sqrt(np.where(ok, disc, 0.0))
            a2 = np.where(ok, 2.0 * a, 1.0)
            s_in = np.maximum((-b - sq) / a2, 0.0)
            s_out = (-b + sq) / a2
            end = np.where(hit, dist, rng_max)
            ok &= (s_out > 0.0) & (s_in < end)
            zm = pnt[2] + vec[:, 2] * 0.5 * (s_in + np.minimum(s_out, end))
            ok &= (zm >= v.z - 0.2) & (zm <= v.z + v.plume_h)
            ok &= ~through
            if not ok.any():
                continue
            through |= ok
            k = np.flatnonzero(ok)
            back = k[self.rng.uniform(size=len(k)) < PLUME_RETURN]
            if len(back):
                span = np.minimum(s_out[back], end[back]) - s_in[back]
                d = s_in[back] + self.rng.uniform(0.0, 1.0, len(back)) * np.maximum(span, 0.05) * 0.6
                d = d + self.rng.normal(0.0, 0.15, len(back))
                d = np.clip(d, 0.1, rng_max)
                pts_all.append(pnt[None, :] + vec[back] * d[:, None])
        if not through.any():
            return hit, None
        hit = hit & ~through
        pts = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
        rover.plume_pts = pts
        if len(pts) >= MIN_PLUME_PTS:
            self._on_plume(rover, t, pts)
        return hit, (pts, through)

    # ------------------------------------------------------------------ fleet knowledge
    def _on_plume(self, rover, t: float, pts: np.ndarray) -> None:
        """A rover saw a plume in its own returns: estimate where the vent is and keep clear of it."""
        f = self.sim.fleet
        mx, my = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        ux, uy = mx - rover.x, my - rover.y
        dn = math.hypot(ux, uy)
        if dn > 1e-6:                      # the returns sit on the near side of the column
            mx, my = mx + 0.8 * PLUME_R * ux / dn, my + 0.8 * PLUME_R * uy / dn
        dist = math.hypot(mx - rover.x, my - rover.y)
        zones = f.geyser_zones
        k = next((n for n, z in enumerate(zones) if math.hypot(z[0] - mx, z[1] - my) < SAME_VENT_M), None)
        rmax = min(math.hypot(r.x - mx, r.y - my) for r in f.rovers) - 1.0
        r = max(2.5, min(KEEP_OUT_M, rmax))
        if k is None:
            zones.append((mx, my, r, rover.id, float(t)))
            self._seen.append([float(t), [float(t)]])
            self._mark(mx, my, r)
            f.log(t, "warn", f"{rover.id} sees a steam plume erupting at ({mx:.0f}, {my:.0f}), {dist:.1f} m away: "
                  f"{len(pts)} scattered LiDAR returns. Marking a {r:.1f} m keep-out zone.", rover.id, at=(mx, my))
            return
        z = zones[k]
        seen = self._seen[k]
        if r > z[2] + 0.3:                 # nobody inside any more: widen the zone to the full radius
            zones[k] = (z[0], z[1], r, z[3], z[4])
            self._mark(z[0], z[1], r)
        if t - seen[0] > NEW_ERUPTION_S:
            seen[1].append(float(t))
            if len(seen[1]) >= 2:
                per = seen[1][-1] - seen[1][-2]
                f.log(t, "info", f"{rover.id} sees the vent at ({z[0]:.0f}, {z[1]:.0f}) erupt again. The fleet "
                      f"estimates it erupts every {per:.0f} s.", rover.id, at=(z[0], z[1]))
        seen[0] = float(t)

    def _mark(self, x: float, y: float, r: float) -> None:
        f = self.sim.fleet
        f.map.mark_avoid(x, y, r)
        f.invalidate_costs()
        f.frontiers_t = -1e9
        f.refresh_reach()
        for rv in f.rovers:
            rv.plan_t = -1e9
