"""Fleet level reasoning: shared map, frontier assignment, learned slope limit, targets, event log."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .mapping import FleetMap
from .planner import astar, add_avoid, cost_grid, path_length, simplify
from .terrain import KIND_LABEL, TargetSpec

KIND_REASON = {
    "ice": "Ice often clusters, prioritising nearby frontiers.",
    "mineral": "Mineral veins tend to run in groups, prioritising nearby frontiers.",
    "vent": "Vents usually come in fields, prioritising nearby frontiers.",
}

# goal utility weights
W_SIZE = 0.5
W_DIST = 1.0
W_TURN = 3.0
OVERLAP_RADIUS = 8.0
OVERLAP_PENALTY = 22.0
SAMPLE_BONUS = 14.0
NEAR_ROVER_PENALTY = 8.0


def compass(dx: float, dy: float) -> str:
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    a = math.degrees(math.atan2(dy, dx)) % 360.0
    return names[int((a + 22.5) // 45) % 8]


@dataclass
class TargetState:
    spec: TargetSpec
    status: str | None = None      # None = undiscovered, "found", "investigating", "sampled"
    found_by: str | None = None
    found_t: float | None = None
    investigator: str | None = None
    fails: int = 0
    retry_after: float = 0.0

    def to_msg(self) -> dict:
        return {
            "id": self.spec.id, "x": round(self.spec.x, 2), "y": round(self.spec.y, 2), "kind": self.spec.kind,
            "status": self.status, "found_by": self.found_by,
            "found_t": None if self.found_t is None else round(self.found_t, 1),
        }


class Fleet:
    def __init__(self, cfg: Config, fmap: FleetMap, targets: list):
        self.cfg = cfg
        self.map = fmap
        self.slope_limit = float(cfg.slope_limit_initial_deg)
        self.limit_version = 0
        self.best_climb = 0.0
        self.best_climb_by = None
        self.events: list = []
        self.event_seq = 0
        self.targets = [TargetState(t) for t in targets]
        self.rovers: list = []
        self.frontiers: list = []
        self.frontiers_t = -1e9
        self.blacklist: dict = {}          # bucket key -> expiry time
        self.sampled_points: list = []
        self.complete = False
        self.exhausted: list = []          # frontiers left over when exploration was declared done
        self.no_frontier_since = None
        self._cost = None
        self._cost_key = None
        self._cost_t = -1e9
        self.reach = None                  # samples connected to the lander through passable ground
        self.reach_t = -1e9
        self.plan_calls = 0
        self.plan_expansions = 0

    # ------------------------------------------------------------------ events
    def log(self, t: float, level: str, msg: str, rover: str | None = None) -> None:
        self.event_seq += 1
        self.events.append({"t": round(float(t), 1), "rover": rover, "level": level, "msg": msg, "_seq": self.event_seq})
        if len(self.events) > 400:
            del self.events[:100]

    @staticmethod
    def public_event(e: dict) -> dict:
        return {"t": e["t"], "rover": e["rover"], "level": e["level"], "msg": e["msg"]}

    # ------------------------------------------------------------------ periodic
    def update(self, t: float) -> None:
        if t - self.reach_t >= 2.0 or self.reach is None:
            self._update_reach(t)
        if t - self.frontiers_t >= 1.0:
            if self.reach_t < self.frontiers_t:      # limit changed since the last flood fill
                self._update_reach(t)
            self.frontiers = self.map.frontier_clusters(self.slope_limit)
            self.frontiers_t = t
            for k in [k for k, exp in self.blacklist.items() if exp < t]:
                del self.blacklist[k]
            self._check_complete(t)

    def _update_reach(self, t: float) -> None:
        """Flood fill from the lander over every sample the planner may cross (known and gentle
        enough, or unknown). Frontiers outside this region cannot be reached by the fleet."""
        ok = np.isfinite(self.cost(t))
        reach = np.zeros_like(ok)
        for (x, y) in [(0.0, 0.0)] + [r.dock for r in self.rovers]:
            i, j = self.map.ij(x, y)
            reach[max(j - 2, 0):j + 3, max(i - 2, 0):i + 3] = True
        reach &= ok
        k = 0
        while True:
            grown = reach.copy()
            grown[1:, :] |= reach[:-1, :]
            grown[:-1, :] |= reach[1:, :]
            grown[:, 1:] |= reach[:, :-1]
            grown[:, :-1] |= reach[:, 1:]
            grown &= ok
            k += 1
            if k % 6 == 0 and (grown == reach).all():
                break
            reach = grown
        self.reach = reach
        self.reach_t = t

    def in_reach(self, x: float, y: float, radius: int = 2) -> bool:
        """Is (x, y), or a sample within radius cells of it, connected to the lander?"""
        if self.reach is None:
            return True
        i, j = self.map.ij(x, y)
        return bool(self.reach[max(j - radius, 0):j + radius + 1, max(i - radius, 0):i + radius + 1].any())

    def usable_frontiers(self, t: float) -> list:
        """Frontiers the fleet can still reach from the lander."""
        out = []
        reach = None if self.reach is None else self.reach.ravel()
        for f in self.frontiers:
            if reach is not None and not reach[f[3]]:
                continue
            if self._bucket(f[0], f[1]) in self.blacklist:
                continue
            out.append(f)
        return out

    def _check_complete(self, t: float) -> None:
        if self.complete:
            fresh = [f for f in self.usable_frontiers(t) if f[2] >= 6 and
                     all(math.hypot(f[0] - ex, f[1] - ey) > 6.0 for ex, ey in self.exhausted)]
            if fresh:
                self.complete = False
                self.no_frontier_since = None
                self.log(t, "info", "New frontier spotted. Resuming exploration.")
            return
        usable = self.usable_frontiers(t)
        busy = any(r.state in ("exploring", "investigating") and r.goal is not None for r in self.rovers)
        if usable or busy or t < 30.0:
            self.no_frontier_since = None
            return
        if self.no_frontier_since is None:
            self.no_frontier_since = t
        elif t - self.no_frontier_since > 15.0:
            self.complete = True
            self.exhausted = [(f[0], f[1]) for f in self.frontiers]
            pct = self.map.explored_pct()
            self.log(t, "info", f"Exploration complete: no reachable frontiers left ({pct:.0f}% of the map explored). "
                                f"All rovers returning to the lander.")

    # ------------------------------------------------------------------ planning
    def _bucket(self, x: float, y: float) -> tuple:
        return (int(math.floor(x / 4.0)), int(math.floor(y / 4.0)))

    def cost(self, t: float, limit: float | None = None) -> np.ndarray:
        lim = self.slope_limit if limit is None else limit
        key = (lim, self.limit_version)
        if self._cost is None or key != self._cost_key or t - self._cost_t > 0.6:
            self._cost = cost_grid(self.map, lim)
            self._cost_key = key
            self._cost_t = t
        return self._cost

    def plan(self, t: float, rover, goal: tuple, relax: float = 0.0, max_exp: int = 14000):
        """Plan from the rover to goal (x, y). Returns list of (x, y) waypoints (dense) or None."""
        lim = self.slope_limit + relax
        base = self.cost(t, lim) if relax == 0.0 else cost_grid(self.map, lim, unknown_cost=0.6)
        others = [(o.x, o.y, 1.3 if (o.speed < 0.05 or o.state in ("idle", "charging")) else 0.9)
                  for o in self.rovers if o is not rover]
        mult = add_avoid(base, self.map, others)
        s = self.map.ij(rover.x, rover.y)
        g = self.map.ij(goal[0], goal[1])
        self.plan_calls += 1
        cells, exp = astar(mult, s, g, max_expansions=max_exp)
        self.plan_expansions += exp
        if cells is None:
            return None
        pts = [self.map.xy(i, j) for (i, j) in cells]
        pts[-1] = (float(goal[0]), float(goal[1]))
        return simplify(pts) if len(pts) > 2 else pts

    def path_blocked(self, pts: list, start_idx: int, horizon: int = 30) -> bool:
        if self._cost is None:
            return False
        c = self._cost
        for (x, y) in pts[start_idx + 1:start_idx + 1 + horizon]:
            i, j = self.map.ij(x, y)
            if c[j, i] == math.inf:
                return True
        return False

    def segment_blocked(self, a: tuple, b: tuple) -> bool:
        """True if the straight segment a->b crosses an impassable sample (current cost grid)."""
        if self._cost is None:
            return False
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(L / (self.map.res * 0.7)))
        c = self._cost
        for k in range(1, n + 1):
            f = k / n
            i, j = self.map.ij(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
            if c[j, i] == math.inf:
                return True
        return False

    # ------------------------------------------------------------------ goals
    def choose_goal(self, t: float, rover, max_tries: int = 6):
        """Pick the best frontier cluster for this rover and plan to it. Returns (goal, path, size) or None."""
        rover.goal_fail_reason = None
        if self.complete or not self.frontiers:
            return None
        cands = []
        reach = self.reach.ravel() if (self.reach is not None and self.in_reach(rover.x, rover.y)) else None
        skipped_battery = 0
        for (x, y, size, idx) in self.frontiers:
            key = self._bucket(x, y)
            if key in self.blacklist or rover.blacklist.get(key, -1.0) > t:
                continue
            if reach is not None and not reach[idx]:
                continue
            d = math.hypot(x - rover.x, y - rover.y)
            if d < 2.5:
                continue
            # battery feasibility: get there and then home with margin
            need = rover.energy_for_distance(d * 1.3 + math.hypot(x, y) * 1.3)
            if need + self.cfg.battery_margin > rover.battery:
                skipped_battery += 1
                continue
            u = W_SIZE * min(size, 40) - W_DIST * d
            ang = abs(math.atan2(math.sin(math.atan2(y - rover.y, x - rover.x) - rover.yaw),
                                 math.cos(math.atan2(y - rover.y, x - rover.x) - rover.yaw)))
            u -= W_TURN * ang
            for o in self.rovers:
                if o is rover:
                    continue
                if o.goal is not None and o.state in ("exploring", "investigating") and \
                        math.hypot(x - o.goal[0], y - o.goal[1]) < OVERLAP_RADIUS:
                    u -= OVERLAP_PENALTY
                if math.hypot(x - o.x, y - o.y) < 6.0:
                    u -= NEAR_ROVER_PENALTY
            if any(math.hypot(x - sx, y - sy) < self.cfg.target_bonus_radius for sx, sy in self.sampled_points):
                u += SAMPLE_BONUS
            cands.append((u, x, y, size, key))
        if not cands and skipped_battery:
            rover.goal_fail_reason = "battery"
        cands.sort(key=lambda c: -c[0])
        for (u, x, y, size, key) in cands[:max_tries]:
            path = self.plan(t, rover, (x, y))
            if path is not None:
                return (x, y), path, size
            rover.blacklist[key] = t + 40.0     # unreachable from here; other rovers may still get there
        return None

    def bonus_area(self, x: float, y: float) -> bool:
        return any(math.hypot(x - sx, y - sy) < self.cfg.target_bonus_radius for sx, sy in self.sampled_points)

    # ------------------------------------------------------------------ targets
    def on_target_seen(self, t: float, rover, tidx: int, dist: float) -> None:
        ts = self.targets[tidx]
        if ts.status is not None:
            return
        ts.status = "found"
        ts.found_by = rover.id
        ts.found_t = t
        label = KIND_LABEL[ts.spec.kind]
        dirn = compass(ts.spec.x - rover.x, ts.spec.y - rover.y)
        if rover.can_investigate(t, ts):
            ts.status = "investigating"
            ts.investigator = rover.id
            rover.start_investigation(t, tidx)
            self.log(t, "discovery", f"{rover.id} spotted a possible {label} ({ts.spec.id}) {dist:.0f} m {dirn}. "
                                     f"Moving in to sample it.", rover.id)
        else:
            why = "battery too low" if rover.state == "returning" or rover.battery < 0.35 else "busy"
            self.log(t, "discovery", f"{rover.id} spotted a possible {label} ({ts.spec.id}) {dist:.0f} m {dirn}. "
                                     f"Logged for later ({why}).", rover.id)

    def unclaimed_target(self, t: float, rover):
        best, bd = None, 1e9
        for k, ts in enumerate(self.targets):
            if ts.status == "found" and ts.investigator is None and t >= ts.retry_after:
                d = math.hypot(ts.spec.x - rover.x, ts.spec.y - rover.y)
                if d < bd:
                    best, bd = k, d
        return best, bd

    def target_sampled(self, t: float, rover, tidx: int) -> None:
        ts = self.targets[tidx]
        ts.status = "sampled"
        ts.investigator = None
        self.sampled_points.append((ts.spec.x, ts.spec.y))
        label = KIND_LABEL[ts.spec.kind]
        self.log(t, "discovery", f"{rover.id} sampled {label} {ts.spec.id}. {KIND_REASON[ts.spec.kind]}", rover.id)

    def release_target(self, t: float, rover, tidx: int, reason: str | None = None) -> None:
        ts = self.targets[tidx]
        if ts.status == "investigating" and ts.investigator == rover.id:
            ts.status = "found"
            ts.investigator = None
            ts.retry_after = t + 20.0
            if reason:
                self.log(t, "info", f"{rover.id} left {ts.spec.id} for later: {reason}.", rover.id)

    # ------------------------------------------------------------------ learning
    def report_slope_stall(self, t: float, rover, slope_deg: float) -> bool:
        """A rover could not climb a slope. Returns True if the fleet limit was lowered."""
        measured = int(round(slope_deg))
        floor = max(self.cfg.slope_limit_floor_deg, math.floor(self.best_climb))
        new = max(floor, measured - 2)
        if new < self.slope_limit - 0.01:
            self.slope_limit = float(new)
            self.limit_version += 1
            self._cost = None
            msg = f"{rover.id} stalled on a {measured} deg slope. Lowering fleet slope limit to {new:.0f} deg."
            if self.best_climb >= 1.0:
                msg += f" Steepest successful climb so far: {math.floor(self.best_climb)} deg."
            self.log(t, "warn", msg, rover.id)
            self.frontiers_t = -1e9   # recompute frontiers with the new limit
            return True
        self.log(t, "warn", f"{rover.id} stalled on a {measured} deg slope. Backing off and replanning.", rover.id)
        return False

    def record_climb(self, t: float, rover, pitch_deg: float) -> None:
        if pitch_deg > self.best_climb + 0.5:
            self.best_climb = pitch_deg
            self.best_climb_by = rover.id

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        found = sum(1 for ts in self.targets if ts.status is not None)
        return {
            "explored_pct": round(self.map.explored_pct(), 1),
            "slope_limit_deg": round(self.slope_limit, 1),
            "slope_limit_initial_deg": round(float(self.cfg.slope_limit_initial_deg), 1),
            "targets_found": found,
            "targets_total": len(self.targets),
            "sim_speed": float(self.cfg.sim_speed),
        }

    def discovered_targets(self) -> list:
        return [ts.to_msg() for ts in self.targets if ts.status is not None]
