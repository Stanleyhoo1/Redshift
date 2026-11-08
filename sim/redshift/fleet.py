"""Fleet level reasoning: shared map, frontier assignment, learned slope limit, targets and samples,
soft ground and water reports, the motion tracker and who observes a lifeform, rover status (faults)
and the event log."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .mapping import FleetMap
from .planner import FAIL, PENDING, AStar, add_avoid, cost_grid, path_length, simplify
from .terrain import KIND_LABEL, TargetSpec
from .tracker import Tracker

KIND_REASON = {
    "ice": "Ice often clusters, prioritising nearby frontiers.",
    "mineral": "Mineral veins tend to run in groups, prioritising nearby frontiers.",
    "vent": "Vents usually come in fields, prioritising nearby frontiers.",
    "trace": "Traces mark where the creatures graze, prioritising nearby frontiers.",
}
W_VALUE = 4.0               # target choice: utility per point of (estimated) science value ...
W_TDIST = 0.5               # ... minus this per metre away

# goal utility weights
W_SIZE = 0.5
W_DIST = 1.0
W_TURN = 3.0
OVERLAP_RADIUS = 8.0
OVERLAP_PENALTY = 22.0
SAMPLE_BONUS = 14.0
NEAR_ROVER_PENALTY = 8.0
DEGRADED_DIST_K = 2.5       # a rover that cannot see far weighs distance this much more
DEGRADED_MAX_GOAL = 14.0    # and avoids frontiers farther than this (unless there is nothing closer)
SEARCH_TTL = 3.0            # an unfinished plan older than this (sim s) is dropped and restarted
DIR_BIAS = 30.0             # operator "explore <direction>": utility bonus for frontiers straight that way


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
    value: int | None = None           # revealed by the on-board analysis
    analysis: str | None = None
    carried_by: str | None = None
    cached: bool = False

    def to_msg(self) -> dict:
        return {
            "id": self.spec.id, "x": round(self.spec.x, 2), "y": round(self.spec.y, 2), "kind": self.spec.kind,
            "status": self.status, "found_by": self.found_by,
            "found_t": None if self.found_t is None else round(self.found_t, 1),
            "instrument": self.spec.instrument, "value": self.value, "analysis": self.analysis,
            "carried_by": self.carried_by, "cached": bool(self.cached),
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
        self._reach_job = None             # a flood fill in progress (anytime, a slice per duty cycle)
        self.plan_calls = 0
        self.plan_expansions = 0
        self.plan_slices = 0
        self._costs: dict = {}             # (limit, limit_version, unknown_cost) -> (grid, t)
        self.soft_zones: list = []         # [(x, y, radius, rover id, t)] soft ground found by wheel slip
        self.ice_zones: list = []          # v7: [(x, y, radius, rover id, t)] frozen ground found by low traction
        self.orphan = None                 # (x, y, rover id, t): area a faulted rover had to drop
        self.hazards = None                # set by the sim; only told about soft zones (for the UI)
        self.water_zones: list = []        # [(x, y, radius, rover id, t)] liquid the fleet identified
        self.dropoffs: list = []           # [(x, y, t)] LiDAR holes the camera said were not liquid
        self.tracker = Tracker()           # moving LiDAR clusters (lifeforms)
        self.observed_areas: list = []     # [(x, y, t)] where an observation ended (no repeat for a while)
        self.lifeform_herds: list = []     # [(x, y, most individuals seen at once)] confirmed lifeforms
        self.lifeforms_seen = 0
        self.announced: list = []          # [(x, y, t)] sightings already logged without an observer
        self.samples_cached = 0
        self.science_score = 0
        self.skip_logged: set = set()      # (rover id, target index) trade-offs already logged

    # ------------------------------------------------------------------ events
    def log(self, t: float, level: str, msg: str, rover: str | None = None, at=None, source: str | None = None) -> None:
        """source: "gemma" or "operator" when the decision came from tier 2 or a typed command
        (protocol v3); None means the rules (the field is then left out)."""
        self.event_seq += 1
        e = {"t": round(float(t), 1), "rover": rover, "level": level, "msg": msg, "_seq": self.event_seq}
        if at is not None:
            e["at"] = [round(float(at[0]), 1), round(float(at[1]), 1)]
        if source is not None and source != "rules":
            e["source"] = source
        self.events.append(e)
        if len(self.events) > 400:
            del self.events[:100]
        if rover is not None:
            for r in self.rovers:              # the rover's own mission memory (answers.py)
                if r.id == rover:
                    r.mem.on_event(e)
                    break

    @staticmethod
    def public_event(e: dict) -> dict:
        out = {"t": e["t"], "rover": e["rover"], "level": e["level"], "msg": e["msg"]}
        if "at" in e:
            out["at"] = e["at"]
        if "source" in e:
            out["source"] = e["source"]
        return out

    # ------------------------------------------------------------------ periodic
    def update(self, t: float) -> None:
        self.tracker.update(t)
        if self.tracker.tracks:
            self._assign_observers(t)
        if self.reach is None:
            self._update_reach(t, max_iters=None)     # first time: all at once
        elif self._reach_job is not None or t - self.reach_t >= 2.0:
            self._update_reach(t)                     # anytime: a slice per duty cycle
        if t - self.frontiers_t >= 1.0:
            self.frontiers = self.map.frontier_clusters(self.slope_limit)
            self.frontiers_t = t
            for k in [k for k, exp in self.blacklist.items() if exp < t]:
                del self.blacklist[k]
            self._check_complete(t)

    def _update_reach(self, t: float, max_iters: int | None = 80) -> None:
        """Flood fill from the lander over every sample the planner may cross (known and gentle
        enough, or unknown). Frontiers outside this region cannot be reached by the fleet. Anytime:
        at most max_iters growth steps per call (a few ms on the 120 m map); the previous result is
        used until the new one has converged."""
        job = self._reach_job
        if job is None:
            ok = np.isfinite(self.cost(t))
            reach = np.zeros_like(ok)
            for (x, y) in [(0.0, 0.0)] + [r.dock for r in self.rovers]:
                i, j = self.map.ij(x, y)
                reach[max(j - 2, 0):j + 3, max(i - 2, 0):i + 3] = True
            reach &= ok
            job = self._reach_job = {"ok": ok, "reach": reach, "k": 0, "t": t}
        ok, reach = job["ok"], job["reach"]
        grown = np.empty_like(reach)
        n = 0
        while True:
            np.copyto(grown, reach)
            grown[1:, :] |= reach[:-1, :]
            grown[:-1, :] |= reach[1:, :]
            grown[:, 1:] |= reach[:, :-1]
            grown[:, :-1] |= reach[:, 1:]
            grown &= ok
            job["k"] += 1
            n += 1
            if job["k"] % 6 == 0 and np.array_equal(grown, reach):
                break
            reach, grown = grown, reach
            if max_iters is not None and n >= max_iters:
                job["reach"] = reach
                return                      # carry on next duty cycle
        self.reach = reach
        self.reach_t = job["t"]
        self._reach_job = None

    def refresh_reach(self) -> None:
        """The passable map changed (water, soft ground, an avoid area, a new slope limit): start the
        flood fill again on the next duty cycle."""
        self.reach_t = -1e9
        self._reach_job = None

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
        busy = any((r.state in ("exploring", "investigating") and r.goal is not None) or r.state in ("fault", "observing")
                   for r in self.rovers)
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

    def cost(self, t: float, limit: float | None = None, unknown_cost: float = 1.0) -> np.ndarray:
        """Cost grid for planning, cached for 0.6 s. The default grid (current limit, normal
        unknown cost) is also kept as self._cost for quick path checks."""
        lim = self.slope_limit if limit is None else limit
        key = (lim, self.limit_version, unknown_cost)
        hit = self._costs.get(key)
        if hit is None or t - hit[1] > 0.6 or t < hit[1]:
            grid = cost_grid(self.map, lim, unknown_cost=unknown_cost)
            if len(self._costs) > 6:
                self._costs.clear()
            self._costs[key] = (grid, t)
            if key == (self.slope_limit, self.limit_version, 1.0):
                self._cost, self._cost_key, self._cost_t = grid, key, t
            return grid
        return hit[0]

    def invalidate_costs(self) -> None:
        self._cost = None
        self._costs.clear()

    def plan(self, t: float, rover, goal: tuple, relax: float = 0.0, max_exp: int = 14000):
        """Plan from the rover to goal (x, y). Returns a list of (x, y) waypoints, None if there is
        no path, or PENDING if the rover's compute budget for this cycle ran out first (the search
        resumes on the next call with the same goal)."""
        g = self.map.ij(goal[0], goal[1])
        key = (g, round(relax, 1))
        fail = rover.plan_fails.get(key)
        if fail is not None and t - fail[0] < 2.0 and math.hypot(rover.x - fail[1], rover.y - fail[2]) < 1.0:
            return None                     # same question, same answer a moment ago
        srch = rover.searches.get(key)
        if srch is not None and (srch.limit_version != self.limit_version or t - srch.t0 > SEARCH_TTL
                                 or math.hypot(rover.x - srch.sx, rover.y - srch.sy) > 1.0):
            srch = None
        if srch is None:
            lim = self.slope_limit + relax
            unk = self.cfg.degraded_unknown_cost if rover.degraded else (0.6 if relax > 0 else 1.0)
            base = self.cost(t, lim, unk)
            others = [(o.x, o.y, 1.3 if (o.speed < 0.05 or o.state in ("idle", "charging", "fault")) else 0.9)
                      for o in self.rovers if o is not rover]
            others += [(*tr.predict(t), 3.0) for tr in self.tracker.near(rover.x, rover.y, 25.0, t)]
            mult = add_avoid(base, self.map, others)
            srch = AStar(mult, self.map.ij(rover.x, rover.y), g, max_expansions=max_exp)
            srch.t0, srch.sx, srch.sy, srch.limit_version = t, rover.x, rover.y, self.limit_version
            self.plan_calls += 1
        deadline, cap = rover.plan_limits()
        exp0 = srch.exp
        status = srch.run(deadline, max_exp=cap)
        rover.exp_used += srch.exp - exp0
        self.plan_slices += 1
        if status is PENDING:
            rover.searches[key] = srch
            return PENDING
        rover.searches.pop(key, None)
        self.plan_expansions += srch.exp
        if status == FAIL:
            rover.plan_fails[key] = (t, rover.x, rover.y)
            return None
        pts = [self.map.xy(i, j) for (i, j) in srch.cells()]
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

    def blocked_ahead(self, t: float, rover, limit: float, near_m: float = 12.0, horizon: int = 8):
        """Walk the rover's remaining path. Within near_m metres every segment is sampled; further on
        only the next waypoints are checked. Returns None or (x, y, metres ahead, reason) for the
        first sample that is now impassable. reason: "rocks" (ground that changed under a recent
        rockfall, seen by LiDAR), "soft" (soft ground a rover reported), "avoid" (an area the
        operator put off limits), "water" (liquid the fleet identified) or "terrain" (steep ground
        that was unknown when the path was planned)."""
        pts = rover.path
        if not pts:
            return None
        m = self.map
        known, slope, blocked, soft, avoid, water = m.known, m.slope, m.blocked, m.soft, m.avoid, m.water
        start = pts[0]
        si, sj = m.ij(start[0], start[1])
        end = pts[-1]

        def bad(x, y):
            i, j = m.ij(x, y)
            if abs(i - si) <= 2 and abs(j - sj) <= 2:
                return None          # the planner's escape zone around the path start
            if soft[j, i]:
                return "soft"
            if avoid[j, i]:
                return "avoid"
            if water[j, i]:
                return "water"
            if blocked[j, i] or (known[j, i] and slope[j, i] > limit):
                # new rocks: several samples near here changed lately (a boulder changes many)
                return "rocks" if m.recently_changed(i, j, t - 20.0, r=3, min_count=2) else "terrain"
            return None

        px, py = rover.x, rover.y
        along = 0.0
        step = 0.3
        for k in range(max(rover.wp, 1), len(pts)):
            bx, by = pts[k]
            L = math.hypot(bx - px, by - py)
            n = max(1, int(L / step))
            for q in range(1, n + 1):
                f = q / n
                x, y = px + f * (bx - px), py + f * (by - py)
                d = along + f * L
                if d < 1.0 or math.hypot(x - end[0], y - end[1]) < 0.6:
                    continue
                why = bad(x, y)
                if why is not None:
                    return (x, y, d, why)
            along += L
            px, py = bx, by
            if along > near_m:
                for (x, y) in pts[k + 1:k + 1 + horizon]:
                    if math.hypot(x - end[0], y - end[1]) < 0.6:
                        continue
                    why = bad(x, y)
                    if why is not None:
                        return (x, y, along, why)
                break
        return None

    # ------------------------------------------------------------------ soft ground (from wheel slip)
    def add_soft_zone(self, t: float, rover, x: float, y: float, radius: float) -> None:
        self.soft_zones.append((float(x), float(y), float(radius), rover.id, float(t)))
        self.map.mark_soft(x, y, radius)
        self.invalidate_costs()
        self.frontiers_t = -1e9
        self.refresh_reach()
        if self.hazards is not None:
            self.hazards.on_soft_zone(x, y, radius)

    def soft_zone_at(self, x: float, y: float, margin: float = 0.5):
        for z in self.soft_zones:
            if math.hypot(x - z[0], y - z[1]) <= z[2] + margin:
                return z
        return None

    # ------------------------------------------------------------------ water (LiDAR holes + camera)
    def add_water_zone(self, t: float, rover, x: float, y: float, radius: float, quiet: bool = True,
                       source: str | None = None) -> None:
        """Liquid identified around (x, y): impassable for every rover from now on."""
        self.water_zones.append((float(x), float(y), float(radius), rover.id if rover is not None else None, float(t)))
        self.map.mark_water(x, y, radius)
        self.invalidate_costs()
        self.frontiers_t = -1e9
        self.refresh_reach()
        for r in self.rovers:
            r.plan_t = -1e9                 # everyone re-checks their route against the new water
        if not quiet:
            self.log(t, "info", f"{rover.id} marks water at ({x:.0f}, {y:.0f}), radius {radius:.1f} m.", rover.id,
                     at=(x, y), source=source)

    def water_zone_at(self, x: float, y: float, margin: float = 0.0):
        for z in self.water_zones:
            if math.hypot(x - z[0], y - z[1]) <= z[2] + margin:
                return z
        return None

    def water_zone_gap(self, x: float, y: float) -> float:
        """Distance from (x, y) to the edge of the nearest known water zone (inf if none)."""
        return min((math.hypot(x - z[0], y - z[1]) - z[2] for z in self.water_zones), default=math.inf)

    def dropoff_near(self, x: float, y: float, t: float, radius: float = 8.0, ttl: float = 120.0) -> bool:
        self.dropoffs = [d for d in self.dropoffs if t - d[2] < ttl]
        return any(math.hypot(x - d[0], y - d[1]) < radius for d in self.dropoffs)

    # ------------------------------------------------------------------ lifeforms (motion tracks)
    def _assign_observers(self, t: float) -> None:
        """A new moving track: the nearest free rover with enough battery observes it (one at a time,
        one per herd); the others keep exploring."""
        tracker = self.tracker
        fresh = tracker.new_confirmed
        tracker.new_confirmed = []
        self.observed_areas = [a for a in self.observed_areas if t - a[2] < 300.0]
        observing = [r for r in self.rovers if r.state == "observing" and r.obs is not None]
        max_obs = 2 if len(self.rovers) >= 3 else 1     # the others keep exploring
        for tr in tracker.confirmed():
            if tr.observer is not None or tr.ignored or tr.status == "lost" or t < tr.done_until:
                continue
            px, py = tr.predict(t)
            if any(o2.observer is not None and math.hypot(o2.x - px, o2.y - py) < 12.0 for o2 in tracker.confirmed()):
                continue                    # its herd is already being watched
            if any(math.hypot(px - ax, py - ay) < 15.0 for ax, ay, _ in self.observed_areas):
                continue                    # that herd was watched a moment ago
            cands = [r for r in self.rovers if r.can_observe(t, tr)] if len(observing) < max_obs else []
            if cands:
                r = min(cands, key=lambda r_: math.hypot(r_.x - px, r_.y - py))
                r.start_observing(t, tr)
                observing.append(r)
                continue
            if tr in fresh and not any(math.hypot(px - ax, py - ay) < 15.0 and t - at < 120.0
                                       for ax, ay, at in self.announced):
                self.announced = [a for a in self.announced if t - a[2] < 120.0] + [(px, py, t)]
                rid = sorted(tr.seen_by)[0]
                rv = next((r for r in self.rovers if r.id == rid), None)
                if rv is not None:
                    d = math.hypot(px - rv.x, py - rv.y)
                    self.log(t, "discovery", f"{rid} sees something moving {d:.0f} m {compass(px - rv.x, py - rv.y)} "
                                             f"(LiDAR track {tr.id}, {tr.speed:.1f} m/s). No rover free to observe it now; "
                                             f"logged.", rid, at=(px, py))

    def confirm_lifeform(self, tr, t: float) -> None:
        """Gemma (or the rules) confirmed this track is a lifeform: count it as one individual of the
        herd around it (see confirm_herd)."""
        tr.lifeform = True
        px, py = tr.predict(t)
        self.confirm_herd(px, py, 1, t)

    def confirm_herd(self, x: float, y: float, n: int, t: float) -> None:
        """At least n lifeforms seen at once around (x, y). Track ids churn when a creature is lost and
        picked up again, so the fleet counts herds by place: a herd within 20 m of one already counted
        is the same herd, and its count is the most individuals ever seen at the same time."""
        for k, (hx, hy, cnt) in enumerate(self.lifeform_herds):
            if math.hypot(x - hx, y - hy) < 20.0:
                self.lifeform_herds[k] = (0.5 * (hx + x), 0.5 * (hy + y), max(cnt, n))
                break
        else:
            self.lifeform_herds.append((x, y, n))
        self.lifeforms_seen = sum(c for _, _, c in self.lifeform_herds)

    # ------------------------------------------------------------------ rover status
    def rover_faulted(self, t: float, rover) -> None:
        """A rover reported that its motors are offline: free everything it had claimed so the
        others can re-plan and take over."""
        area = rover.goal if (rover.goal is not None and rover.state in ("exploring", "investigating")) \
            else (rover.x, rover.y)
        self.orphan = (float(area[0]), float(area[1]), rover.id, t)
        for o in self.rovers:
            if o is not rover:
                o.next_goal_eval = t        # everyone reconsiders their goal now
                o.next_target_check = t
                o.plan_t = -1e9

    def takeover_note(self, t: float, rover, goal: tuple) -> None:
        if self.orphan is None:
            return
        ox, oy, rid, t0 = self.orphan
        if t - t0 > 40.0:
            self.orphan = None
            return
        if rid != rover.id and math.hypot(goal[0] - ox, goal[1] - oy) < 12.0:
            self.orphan = None
            self.log(t, "info", f"{rover.id} takes over {rid}'s area {math.hypot(goal[0] - rover.x, goal[1] - rover.y):.0f} m "
                                f"{compass(goal[0] - rover.x, goal[1] - rover.y)}.", rover.id, at=goal)

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
        w_dist = W_DIST * (DEGRADED_DIST_K if rover.degraded else 1.0)
        bias = getattr(rover, "dir_bias", None)      # operator: "explore north" (unit vector, until t)
        if bias is not None and t >= bias[2]:
            bias = None
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
            if need + rover.battery_margin > rover.battery:
                skipped_battery += 1
                continue
            u = W_SIZE * min(size, 40) - w_dist * d
            if rover.degraded and d > DEGRADED_MAX_GOAL:
                u -= 200.0                  # only if nothing nearer can be reached
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
            if bias is not None:
                u += DIR_BIAS * ((x - rover.x) * bias[0] + (y - rover.y) * bias[1]) / max(d, 1e-6)
            cands.append((u, x, y, size, key))
        if not cands and skipped_battery:
            rover.goal_fail_reason = "battery"
        cands.sort(key=lambda c: -c[0])
        cands = cands[:max_tries]
        rover.goal_options = list(cands)        # ranked options: tier 2 (Gemma) may pick another one
        # a plan that ran out of time last cycle is finished first (anytime planning)
        pk = rover.pending_goal_key
        if pk is not None:
            first = [c for c in cands if c[4] == pk]
            cands = first + [c for c in cands if c[4] != pk]
        rover.pending_goal_key = None
        for (u, x, y, size, key) in cands:
            path = self.plan(t, rover, (x, y))
            if path is PENDING:
                rover.pending_goal_key = key
                return PENDING
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
        rover.mem.on_target(ts, t)
        label = KIND_LABEL[ts.spec.kind]
        dirn = compass(ts.spec.x - rover.x, ts.spec.y - rover.y)
        at = (ts.spec.x, ts.spec.y)
        est = f"estimated value {ts.spec.est_value}/10"
        if rover.can_investigate(t, ts):
            ts.status = "investigating"
            ts.investigator = rover.id
            rover.start_investigation(t, tidx)
            self.log(t, "discovery", f"{rover.id} spectrometer flags a possible {label} ({ts.spec.id}) {dist:.0f} m "
                                     f"{dirn}, {est}. Moving in to sample it.", rover.id, at=at)
        else:
            tube = rover.tube_verdict(ts)
            if tube is not None:
                why = tube
                self.skip_logged.add((rover.id, tidx))
            elif rover.state == "returning" or rover.battery < 0.35:
                why = "battery too low"
            elif rover.degraded:
                why = "LiDAR degraded, staying on mapped ground"
            elif rover.state == "fault":
                why = "motors offline"
            else:
                why = "busy"
            self.log(t, "discovery", f"{rover.id} spectrometer flags a possible {label} ({ts.spec.id}) {dist:.0f} m "
                                     f"{dirn}, {est}. Logged for later ({why}).", rover.id, at=at)

    def unclaimed_target(self, t: float, rover):
        """The found, unclaimed target with the best value for distance that this rover's tubes allow.
        Returns (index, distance) or (None, inf). A low value find skipped because the tubes are nearly
        full is logged once."""
        best, bu, bd = None, -1e9, math.inf
        for k, ts in enumerate(self.targets):
            if ts.status == "found" and ts.investigator is None and t >= ts.retry_after:
                d = math.hypot(ts.spec.x - rover.x, ts.spec.y - rover.y)
                why = rover.tube_verdict(ts)
                if why is not None:
                    if rover.free_tubes() > 0 and (rover.id, k) not in self.skip_logged:
                        self.skip_logged.add((rover.id, k))
                        self.log(t, "info", f"{rover.id} skips {ts.spec.id} ({why}).", rover.id, at=(ts.spec.x, ts.spec.y))
                    continue
                u = W_VALUE * ts.spec.est_value - W_TDIST * d
                if u > bu:
                    best, bu, bd = k, u, d
        return best, bd

    def valuable_target(self, t: float, rover):
        """A found, unclaimed target whose spectrometer estimate is high enough to fetch a free tube for."""
        best, bd = None, math.inf
        for k, ts in enumerate(self.targets):
            if ts.status == "found" and ts.investigator is None and ts.spec.est_value >= self.cfg.high_value:
                d = math.hypot(ts.spec.x - rover.x, ts.spec.y - rover.y)
                if d < bd:
                    best, bd = k, d
        return best, bd

    def target_sampled(self, t: float, rover, tidx: int) -> None:
        """The instrument finished: the sample goes into a tube and the on-board analysis reports."""
        from .rover import VERB
        ts = self.targets[tidx]
        ts.status = "sampled"
        ts.investigator = None
        ts.value = int(ts.spec.value)
        ts.analysis = ts.spec.analysis
        ts.carried_by = rover.id
        if tidx not in rover.samples:
            rover.samples.append(tidx)
        rover.mem.on_sample(ts, t)
        self.sampled_points.append((ts.spec.x, ts.spec.y))
        used = len(rover.samples)
        verb = VERB.get(ts.spec.instrument, "sampled")
        high = " High-value find!" if ts.value >= self.cfg.high_value else ""
        self.log(t, "discovery", f"{rover.id} {verb} {ts.spec.id}: {ts.analysis[0].lower() + ts.analysis[1:]} "
                                 f"(value {ts.value}/10). {used} of {rover.capacity} tubes used.{high} "
                                 f"{KIND_REASON[ts.spec.kind]}", rover.id, at=(ts.spec.x, ts.spec.y))
        if used >= rover.capacity:
            self.log(t, "info", f"{rover.id} all {rover.capacity} tubes full. Caching them at the lander on the next "
                                f"return.", rover.id, at=(rover.x, rover.y))

    def cache_samples(self, t: float, rover) -> int:
        """Docked at the lander: hand the tubes over to the sample cache. Returns how many."""
        if not rover.samples:
            return 0
        n = len(rover.samples)
        for k in rover.samples:
            ts = self.targets[k]
            ts.cached = True
            ts.carried_by = None
            self.samples_cached += 1
            self.science_score += int(ts.value or 0)
        rover.samples = []
        rover.mem.on_cache(n, t)
        self.log(t, "discovery", f"{rover.id} cached {n} sample{'s' if n != 1 else ''} at the lander (fleet total "
                                 f"{self.samples_cached}, science score {self.science_score}).", rover.id, at=rover.dock)
        return n

    def release_target(self, t: float, rover, tidx: int, reason: str | None = None, retry_s: float = 20.0) -> None:
        ts = self.targets[tidx]
        if ts.status == "investigating" and ts.investigator == rover.id:
            ts.status = "found"
            ts.investigator = None
            ts.retry_after = t + retry_s
            if reason:
                self.log(t, "info", f"{rover.id} left {ts.spec.id} for later: {reason}.", rover.id,
                         at=(ts.spec.x, ts.spec.y))

    # ------------------------------------------------------------------ learning
    def report_slope_stall(self, t: float, rover, slope_deg: float, load: float | None = None) -> bool:
        """A rover could not climb a slope. Returns True if the fleet limit was lowered."""
        measured = int(round(slope_deg))
        floor = max(self.cfg.slope_floor_deg, math.floor(self.best_climb))
        new = max(floor, measured - 2)
        why = f"slope {measured} deg" + (f" with motors at {load * 100:.0f}%" if load is not None else "")
        if new < self.slope_limit - 0.01:
            self.slope_limit = float(new)
            self.limit_version += 1
            self.invalidate_costs()
            msg = f"{rover.id} stalled ({why}). Lowering fleet slope limit to {new:.0f} deg."
            if self.best_climb >= 1.0:
                msg += f" Steepest successful climb so far: {math.floor(self.best_climb)} deg."
            self.log(t, "warn", msg, rover.id, at=(rover.x, rover.y))
            self.frontiers_t = -1e9   # recompute frontiers with the new limit
            self.refresh_reach()
            return True
        self.log(t, "warn", f"{rover.id} stalled ({why}). Backing off and replanning.", rover.id,
                 at=(rover.x, rover.y))
        return False

    def record_climb(self, t: float, rover, pitch_deg: float) -> None:
        if pitch_deg > self.best_climb + 0.5:
            self.best_climb = pitch_deg
            self.best_climb_by = rover.id

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        found = sum(1 for ts in self.targets if ts.status is not None)
        out = {
            "explored_pct": round(self.map.explored_pct(), 1),
            "slope_limit_deg": round(self.slope_limit, 1),
            "slope_limit_initial_deg": round(float(self.cfg.slope_limit_initial_deg), 1),
            "targets_found": found,
            "targets_total": len(self.targets),
            "sim_speed": float(self.cfg.sim_speed),
            "budget_ms": round(float(self.cfg.brain_budget_ms), 1),
            "cycle_ms": round(1000.0 / self.cfg.agent_hz, 1),
            "soft_zones": [[round(z[0], 2), round(z[1], 2), round(z[2], 2)] for z in self.soft_zones],
            "water_zones": [[round(z[0], 2), round(z[1], 2), round(z[2], 2)] for z in self.water_zones],
            "lifeforms_seen": int(self.lifeforms_seen),
            "samples_cached": int(self.samples_cached),
            "samples_carried": int(sum(len(r.samples) for r in self.rovers)),
            "science_score": int(self.science_score),
            **({"ice_zones": [[round(z[0], 2), round(z[1], 2), round(z[2], 2)] for z in self.ice_zones]}
               if self.cfg.ice_sheets and self.ice_zones else {}),
        }
        gz = getattr(self, "geyser_zones", None)
        if gz:                              # v7: only when geysers are on and the fleet has seen one
            out["geyser_zones"] = [[round(z[0], 2), round(z[1], 2), round(z[2], 2)] for z in gz]
        if getattr(self, "flora", None) is not None:
            out.update(self.flora.stats())     # v7: only when flora is on and some is known
        return out

    def discovered_targets(self) -> list:
        return [ts.to_msg() for ts in self.targets if ts.status is not None]
