"""Scripted-but-reactive rover and scout agents for the fake backend."""
from __future__ import annotations

import math

import numpy as np

from fake_render import CLS_OBSTACLE, CLS_PERMANENT
from fake_world import World


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Agent:
    kind = "rover"

    def __init__(self, rid: str, x: float, y: float, yaw: float):
        self.id, self.x, self.y, self.yaw = rid, x, y, yaw
        self.z = 0.0
        self.speed = 0.0
        self.state = "BOOT"
        self.mode = "FUSED"
        self.path: list = []
        self.trail: list = [[round(x, 2), round(y, 2)]]
        self.goal: list | None = None
        self.state_t = 0.0
        self.roll = self.pitch = 0.0
        self.gyro = [0.0, 0.0, 0.0]
        self.unc = 0.14
        self.throttle = 0
        self.extra_ms = 0.0

    def set_state(self, s: str, now: float):
        self.state, self.state_t = s, now

    def clock(self, now: float):
        if self.state_t == 0.0:
            self.state_t = now

    def follow(self, dt: float, v: float, reverse: bool = False) -> bool:
        """Pure-pursuit along self.path; returns True at the end."""
        while len(self.path) > 1 and math.hypot(self.path[0][0] - self.x, self.path[0][1] - self.y) < 1.2:
            self.path.pop(0)
        if not self.path or (len(self.path) == 1 and math.hypot(self.path[0][0] - self.x, self.path[0][1] - self.y) < 0.6):
            self.speed = 0.0
            return True
        tx, ty = self.path[min(1, len(self.path) - 1)] if len(self.path) > 1 else self.path[0]
        want = math.atan2(ty - self.y, tx - self.x) + (math.pi if reverse else 0.0)
        err = wrap(want - self.yaw)
        rate = 3.0 if self.kind == "scout" else 1.4
        self.yaw = wrap(self.yaw + max(-rate * dt, min(rate * dt, err)))
        if self.kind == "scout":
            # a multirotor yaws on the spot instead of looping round a waypoint, and brakes into the last one
            end = self.path[-1]
            v = min(v, 1.0 + 1.5 * math.hypot(end[0] - self.x, end[1] - self.y))
            self.speed = v * max(0.0, math.cos(err)) ** 2
        else:
            self.speed = v * max(0.25, math.cos(err))
        sgn = -1.0 if reverse else 1.0
        self.x += sgn * self.speed * math.cos(self.yaw) * dt
        self.y += sgn * self.speed * math.sin(self.yaw) * dt
        if math.hypot(self.trail[-1][0] - self.x, self.trail[-1][1] - self.y) > 0.5:
            self.trail.append([round(self.x, 2), round(self.y, 2)])
            del self.trail[:-300]
        return False


class Rover(Agent):
    kind = "rover"

    def __init__(self, rid: str, x: float, y: float, yaw: float):
        super().__init__(rid, x, y, yaw)
        self.slip = 0.03
        self.last_safe = [round(x, 2), round(y, 2)]
        self.breadcrumbs: list = [[x, y]]
        self.jolt_t: float | None = None
        self.jolt_xy: list | None = None
        self.seen_rocks = 0
        self.replan_t = 0.0
        self.scan_id = 0
        self.prev_roll = 0.0
        self.caution_since: float | None = None
        self.goal_t = 0.0

    def jolt(self, now: float):
        self.jolt_t = now
        self.jolt_xy = [round(self.x, 2), round(self.y, 2)]

    def step(self, w: World, dt: float, now: float, scouts: list):
        self.clock(now)
        lim = w.limits
        v_lim = lim["v_max"]
        T = w.terrain
        self.z = T.height_at(self.x, self.y)
        gx, gy = T.gradient_at(self.x, self.y)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pitch = -math.degrees(math.atan(gx * c + gy * s))
        roll = math.degrees(math.atan(-gx * s + gy * c))
        vib = 0.0
        if self.jolt_t is not None:
            tj = now - self.jolt_t
            amp = 37.5 * (math.sin(min(1.0, tj / 0.22) * math.pi / 2) if tj < 0.22 else math.exp(-(tj - 0.22) / 0.55) * math.cos((tj - 0.22) * 5))
            roll += amp
            pitch += amp * 0.18
            if tj > 4:
                self.jolt_t = None
        if w.env["seismic"]:
            vib = 2.2
        self.roll = roll + float(w.rng.normal(0, 0.15 + vib))
        self.pitch = pitch + float(w.rng.normal(0, 0.15 + vib * 0.6))
        self.gyro = [round((self.roll - self.prev_roll) / max(dt, 1e-3) * math.pi / 180, 3), round(float(w.rng.normal(0, 0.02 + vib * 0.05)), 3), 0.0]
        self.prev_roll = self.roll
        dust = w.env["dust_level"]
        self.unc += ((0.14 + 0.42 * dust + (0.08 if vib else 0.0)) - self.unc) * min(1.0, dt * 1.5)
        self.slip = 0.7 if (self.jolt_t is not None and now - self.jolt_t < 0.8) else 0.03 + 0.002 * abs(pitch) + float(w.rng.uniform(0, 0.01))

        st = self.state
        alarm = lim["tip_alarm_deg"]
        if st not in ("SAFE_HOLD", "RECOVERY_REVERSE") and abs(self.roll) > alarm:
            self.set_state("SAFE_HOLD", now)
            self.speed = 0.0
            w.trace(self.id, "SAFE_HOLD", "safe_hold", ["IMU", "ODOM"],
                    f"Roll {self.roll:.1f}° at {abs(self.gyro[0]) * 57.3:.0f}°/s; wheel slip {self.slip:.2f}",
                    f"|roll| {abs(self.roll):.1f}° > tip alarm {alarm:.1f}° for 100 ms at {w.g:.1f} g; chassis unstable",
                    0.97, "SAFE-HOLD: drive stopped, brakes on for 1.5 s", "critical", 6.2)
            return
        if st == "BOOT":
            if now - self.state_t > 1.5:
                self.set_state("EXPLORING", now)
                w.trace(self.id, "EXPLORING", "mission_start", ["ODOM", "MAP", "EDGE"],
                        f"Spawn pad at ({self.x:.1f}, {self.y:.1f}); map 0 % known; {w.profile_id} budget",
                        f"Gravity {w.g:.1f} g limits slope to {lim['max_slope_deg']:.1f}° ({lim['factor'].replace('_', ' ')}) and speed to {v_lim:.2f} m/s",
                        0.99, "Begin frontier exploration")
                self._new_goal(w, now, scouts)
        elif st in ("EXPLORING", "CAUTION"):
            self._caution_logic(w, now)
            fresh_rocks = w.rock_version != self.seen_rocks
            blocker = w.path_blocked(self.path, 14) if (fresh_rocks or now - self.replan_t > 3.0) else 0
            if blocker == CLS_OBSTACLE and fresh_rocks:
                self.seen_rocks = w.rock_version
                self._replan(w, now)
                return
            if blocker:  # newly mapped slope / no-go cells: reroute on the move
                self._reroute(w, now, blocker)
            v = v_lim * (0.35 if self.state == "CAUTION" else 0.78)
            if self.follow(dt, v) and now - self.goal_t > 1.0:
                self.goal_t = now
                self._new_goal(w, now, scouts)
            if self.unc < 0.4 and abs(self.roll) < 0.5 * alarm and math.hypot(self.x - self.breadcrumbs[-1][0], self.y - self.breadcrumbs[-1][1]) > 0.5:
                self.breadcrumbs.append([self.x, self.y])
                self.last_safe = [round(self.x, 2), round(self.y, 2)]
        elif st == "REPLANNING":
            self.speed = 0.0
            if now - self.state_t > 1.2 and not getattr(self, "_replanned", False):
                self._replanned = True
                self._finish_replan(w, now)
            if now - self.state_t > 2.2:
                self._replanned = False
                self.set_state("CAUTION" if w.env["dust_level"] > 0 else "EXPLORING", now)
        elif st == "SAFE_HOLD":
            self.speed = 0.0
            if now - self.state_t > 1.5:
                back = [b for b in self.breadcrumbs if math.hypot(b[0] - self.x, b[1] - self.y) >= 3.0]
                tgt = back[-1] if back else self.breadcrumbs[0]
                self.path = [[self.x, self.y], [round(tgt[0], 2), round(tgt[1], 2)]]
                self.last_safe = [round(tgt[0], 2), round(tgt[1], 2)]
                self.set_state("RECOVERY_REVERSE", now)
                d = math.hypot(tgt[0] - self.x, tgt[1] - self.y)
                w.trace(self.id, "RECOVERY_REVERSE", "recovery_reverse", ["ODOM", "IMU", "MAP"],
                        f"Roll settled to {self.roll:.1f}°; {len(self.breadcrumbs)} breadcrumbs logged",
                        f"Last safe cell ({tgt[0]:.1f}, {tgt[1]:.1f}) is {d:.1f} m back along the driven track",
                        0.93, f"Reversing {d:.1f} m along breadcrumbs at 0.50 m/s", "warn")
        elif st == "RECOVERY_REVERSE":
            if self.follow(dt, 0.5, reverse=True):
                xy = self.jolt_xy or [self.x, self.y]
                w.perm.append([xy[0], xy[1], 2.5])
                w.refresh_truth()
                w.event({"op": "hazard_mark", "robot": self.id, "xy": xy, "radius_m": 2.5, "reason": "tip_event"})
                w.trace(self.id, "AWAIT_SCOUT", "hazard_marked", ["IMU", "MAP", "FLEET"],
                        f"Tip event at ({xy[0]:.1f}, {xy[1]:.1f}); local slope estimate unreliable",
                        "Terrain caused a tip-over; unseen relief may exceed the 16.8° limit", 0.9,
                        "Marked permanent no-go zone r 2.5 m; shared with fleet", "warn")
                self.scan_id += 1
                if scouts:
                    w.event({"op": "scan_request", "id": self.scan_id, "robot": self.id, "xy": xy, "reason": "tip_event"})
                    scouts[0].task(xy, self.scan_id, self.id, w, now)
                    w.trace(self.id, "AWAIT_SCOUT", "scout_tasked", ["FLEET"],
                            f"Scan request #{self.scan_id} for ({xy[0]:.1f}, {xy[1]:.1f})",
                            f"{scouts[0].id} is {math.hypot(scouts[0].x - xy[0], scouts[0].y - xy[1]):.0f} m away with a downward camera",
                            0.88, f"Holding at last safe cell until {scouts[0].id} reports slope stats")
                self.set_state("AWAIT_SCOUT", now)
        elif st == "AWAIT_SCOUT":
            self.speed = 0.0
            if not scouts and now - self.state_t > 4:
                self.resume(w, now, None)
        elif st == "RESUMING":
            if now - self.state_t > 1.8:
                self.set_state("CAUTION" if w.env["dust_level"] > 0 else "EXPLORING", now)
                self._new_goal(w, now, scouts)

    def resume(self, w: World, now: float, verdict: dict | None):
        self.set_state("RESUMING", now)
        w.trace(self.id, "RESUMING", "resume", ["FLEET", "MAP"],
                f"Scout verdict: {verdict['verdict']} (max slope {verdict['max_slope_deg']:.1f}°)" if verdict else "Scout timeout after 25 s",
                "No-go zone stays permanent; remaining frontiers reachable around it", 0.9,
                "Re-planning around the no-go zone and resuming exploration")

    def _caution_logic(self, w: World, now: float):
        if self.state == "EXPLORING" and self.unc > 0.45:
            self.caution_since = self.caution_since or now
            if now - self.caution_since > 0.5:
                self.set_state("CAUTION", now)
                self.mode = "LIDAR_ONLY"
                self.extra_ms = 5.0
                w.trace(self.id, "CAUTION", "caution_enter", ["CAM", "LIDAR", "IMU"],
                        f"Camera sharpness −64 %, contrast 0.19; LiDAR returns {int(100 - 30 * w.env['dust_level'])} % with near-field backscatter",
                        f"Fused uncertainty {self.unc:.2f} > 0.45 for 0.5 s; camera confidence {max(0.05, 0.9 - self.unc):.2f} < 0.40",
                        0.86, f"CAUTION: LiDAR-only perception, speed ×0.35 ({0.35 * w.limits['v_max']:.2f} m/s), +0.4 m inflation", "warn")
        elif self.state == "CAUTION" and self.unc < 0.3:
            self.set_state("EXPLORING", now)
            self.mode = "FUSED"
            self.extra_ms = 0.0
            self.caution_since = None
            w.trace(self.id, "EXPLORING", "caution_exit", ["CAM", "LIDAR"],
                    f"Camera confidence {1 - self.unc:.2f} > 0.70 for 2 s; LiDAR returns 97 %",
                    "Dust cleared; fused perception trustworthy again", 0.9, "Back to FUSED perception at full speed")

    def _new_goal(self, w: World, now: float, scouts: list):
        others = [s.goal for s in scouts if s.goal]
        g = w.frontier_goal(self.x, self.y, self.yaw, others)
        if g is None:
            self.path = []
            if w.stats()["percent"] >= 95:
                self.set_state("MISSION_COMPLETE", now)
            return
        self.goal = [round(g[0], 1), round(g[1], 1)]
        self.path = w.plan((self.x, self.y), g)
        dist = math.hypot(g[0] - self.x, g[1] - self.y)
        w.trace(self.id, self.state, "frontier_selected", ["MAP", "FLEET"],
                f"{int(w.rng.integers(3, 9))} frontier clusters; best at ({g[0]:.1f}, {g[1]:.1f}), {dist:.0f} m",
                f"Score = size − λ·cost; {len(self.path)}-point A* path, {sum(1 for _ in self.path) * 0.8:.0f} m at {w.g:.1f} g energy cost",
                float(w.rng.uniform(0.78, 0.92)), f"Driving to frontier at {0.78 * w.limits['v_max']:.2f} m/s")

    def _replan(self, w: World, now: float):
        """Blocked: stop and announce. The new route is computed in REPLANNING after 1.2 s."""
        self.set_state("REPLANNING", now)
        self.speed = 0.0
        w.trace(self.id, "REPLANNING", "path_blocked", ["LIDAR", "CAM"],
                f"Boulder field {float(w.rng.uniform(5.5, 7.5)):.1f} m ahead across path ({int(w.rng.integers(4, 7))} objects)",
                f"Corridor occupancy {float(w.rng.uniform(0.7, 0.9)):.2f}; step height {float(w.rng.uniform(0.7, 1.0)):.1f} m > 0.30 m clearance",
                0.91, "Stop and re-plan A* around the obstacle", "warn", 41.7)

    def _reroute(self, w: World, now: float, blocker: int):
        old = len(self.path) * 0.8
        goal = tuple(self.goal) if self.goal else (self.x, self.y)
        path = w.plan((self.x, self.y), goal)
        self.replan_t = now
        if not path:
            self._new_goal(w, now, [])
            return
        self.path = path
        what = {CLS_PERMANENT: "No-go zone", CLS_OBSTACLE: "Boulder"}.get(blocker, f"Slope above {w.limits['max_slope_deg']:.1f}°")
        w.trace(self.id, self.state, "replanned", ["LIDAR", "MAP"],
                f"{what} newly mapped across the planned route",
                f"Lethal at {w.g:.1f} g; cheaper to detour than to stop", 0.87,
                f"Re-planned on the move ({len(path) * 0.8 - old:+.1f} m)", "info", 24.0)

    def _finish_replan(self, w: World, now: float):
        old_len = len(self.path) * 0.8
        goal = tuple(self.goal) if self.goal else (self.x, self.y)
        path = w.plan((self.x, self.y), goal)
        if not path:
            g = w.frontier_goal(self.x, self.y, self.yaw + math.pi / 2, [])
            if g:
                self.goal = [round(g[0], 1), round(g[1], 1)]
                path = w.plan((self.x, self.y), g)
        self.path = path
        self.replan_t = now
        self.seen_rocks = w.rock_version
        self.extra_ms = 12.0
        d = len(path) * 0.8 - old_len
        w.trace(self.id, "REPLANNING", "replanned", ["MAP", "EDGE"],
                f"{len(path)} waypoints; lethal cells dilated 0.6 m",
                f"Detour {d:+.1f} m vs blocked route; within the {w.profile_id} expansion budget",
                0.89, f"Re-planned A* around obstacle ({d:+.1f} m, {int(w.rng.integers(900, 1800))} nodes, 38 ms)", "info", 38.0)


class Scout(Agent):
    kind = "scout"

    def __init__(self, rid: str, x: float, y: float, L: float):
        super().__init__(rid, x, y, 0.0)
        self.alt = 0.0
        self.lanes = self._lawnmower(L)
        self.lane_i = 0
        self.req: dict | None = None
        self.orbit_c = (0.0, 0.0)

    @staticmethod
    def _lawnmower(L: float) -> list:
        pts, half, step = [], 0.42 * L, max(10.0, 0.18 * L)
        ys = np.arange(-half, half + 0.1, step)
        for k, y in enumerate(ys):
            xs = (-half, half) if k % 2 == 0 else (half, -half)
            pts += [[xs[0], float(y)], [xs[1], float(y)]]
        return pts

    def task(self, xy: list, sid: int, robot: str, w: World, now: float):
        self.req = {"xy": xy, "id": sid, "robot": robot}
        self.path = World._resample([(self.x, self.y), tuple(xy)], 1.5)
        self.set_state("FLY_TO_SCAN", now)
        w.trace(self.id, "FLY_TO_SCAN", "scout_tasked", ["FLEET", "ODOM"],
                f"Scan request #{sid} from {robot} at ({xy[0]:.1f}, {xy[1]:.1f})",
                f"Pausing survey lane {self.lane_i // 2 + 1}; {math.hypot(xy[0] - self.x, xy[1] - self.y):.0f} m transit at 12 m/s",
                0.95, "Flying to the tip site for a nadir slope scan")

    def step(self, w: World, dt: float, now: float, rovers: list):
        self.clock(now)
        T = w.terrain
        ground = T.height_at(self.x, self.y)
        self.roll = float(w.rng.normal(0, 0.8)) + (2.0 if self.speed > 1 else 0.0)
        self.pitch = float(w.rng.normal(0, 0.8)) - 3.0 * min(1.0, self.speed / 3.0)
        self.gyro = [round(float(w.rng.normal(0, 0.03)), 3) for _ in range(3)]
        self.unc += ((0.12 + 0.3 * w.env["dust_level"]) - self.unc) * min(1.0, dt)
        st = self.state
        if st == "BOOT" and now - self.state_t > 1.0:
            self.set_state("TAKEOFF", now)
        elif st == "TAKEOFF":
            self.alt = min(12.0, self.alt + 4.0 * dt)
            if self.alt >= 12.0:
                self.set_state("SURVEY", now)
                self._next_lane(w)
        elif st == "SURVEY":
            if self.follow(dt, 3.0):
                self._next_lane(w)
        elif st == "FLY_TO_SCAN":
            if self.follow(dt, 12.0):
                self.set_state("SCANNING", now)
                self.orbit_c = tuple(self.req["xy"]) if self.req else (self.x, self.y)
        elif st == "SCANNING":
            a = (now - self.state_t) * 1.6
            cx, cy = self.orbit_c
            self.x, self.y = cx + 3 * math.cos(a), cy + 3 * math.sin(a)
            self.yaw = wrap(a + math.pi / 2)
            self.speed = 4.8
            if now - self.state_t > 4.0 and self.req:
                ci, cj = T.cell(cx, cy)
                sl = T.slope_deg[max(0, ci - 5): ci + 6, max(0, cj - 5): cj + 6]
                mx = max(float(sl.max()) + 9.0, w.limits["max_slope_deg"] + 6)
                verdict = {"verdict": "impassable", "max_slope_deg": round(mx, 1), "mean_slope_deg": round(float(sl.mean()) + 6, 1)}
                w.event({"op": "scan_complete", "id": self.req["id"], "scout": self.id, "xy": list(self.orbit_c), **verdict})
                w.trace(self.id, "SCANNING", "scan_complete", ["CAM", "DEPTH"],
                        f"Nadir depth over r 3 m: max slope {mx:.1f}°, mean {verdict['mean_slope_deg']:.1f}°",
                        f"{mx:.1f}° > {w.limits['max_slope_deg']:.1f}° limit at {w.g:.1f} g", 0.92,
                        "Verdict impassable: confirm permanent no-go zone", "info")
                for r in rovers:
                    if r.id == self.req["robot"] and r.state == "AWAIT_SCOUT":
                        r.resume(w, now, verdict)
                self.req = None
                self.set_state("SURVEY", now)
                self._next_lane(w)
        self.z = ground + self.alt
        if math.hypot(self.trail[-1][0] - self.x, self.trail[-1][1] - self.y) > 1.0:
            self.trail.append([round(self.x, 2), round(self.y, 2)])
            del self.trail[:-300]

    def _next_lane(self, w: World):
        if self.lane_i >= len(self.lanes):
            self.lane_i = 0
        tgt = self.lanes[self.lane_i]
        self.lane_i += 1
        self.goal = [round(tgt[0], 1), round(tgt[1], 1)]
        self.path = World._resample([(self.x, self.y), tuple(tgt)], 1.5)
        if self.lane_i % 2 == 0:
            w.trace(self.id, "SURVEY", "frontier_selected", ["MAP", "FLEET"],
                    f"Survey lane {self.lane_i // 2} of {len(self.lanes) // 2}; strip {w.stats()['percent']:.0f} % mapped overall",
                    "Largest unmapped strip not claimed by a rover within 8 m", 0.84,
                    "Flying lawnmower lane at 12 m AGL, 3.0 m/s")
