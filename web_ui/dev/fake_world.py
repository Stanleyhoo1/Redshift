"""Mission state for the fake backend: fog-of-war map, coarse A*, traces, events, injections."""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from fake_render import (CLS_CAUTION, CLS_FREE, CLS_HAZARD, CLS_OBSTACLE, CLS_PERMANENT, CLS_SCIENCE,
                         albedo_texture, generate_terrain, physics_preview)

SECTORS = {"small": (50.0, 0.5), "medium": (100.0, 0.5), "large": (200.0, 1.0)}
PROFILES = {
    "orin_nano_15w": dict(perf=2.6, idle=5, max=15, hz=10),
    "orin_nx_25w": dict(perf=1.6, idle=7, max=25, hz=15),
    "agx_orin_60w": dict(perf=0.9, idle=15, max=60, hz=20),
    "rpi5_8w": dict(perf=5.0, idle=3, max=8, hz=5),
}
LETHAL = (CLS_HAZARD, CLS_PERMANENT, CLS_OBSTACLE)

# GET /api/options (docs/protocol.md)
OPTIONS = {
    "sectors": [
        {"id": "small", "label": "Small", "size_m": 50, "res_m": 0.5, "cells": 100},
        {"id": "medium", "label": "Medium", "size_m": 100, "res_m": 0.5, "cells": 200},
        {"id": "large", "label": "Large", "size_m": 200, "res_m": 1.0, "cells": 200},
    ],
    "gravity": {"min": 1.0, "max": 3.0, "step": 0.1, "default": 2.2},
    "fleet": {"rovers": {"min": 1, "max": 2, "default": 2}, "drones": {"min": 0, "max": 1, "default": 1}},
    "edge_profiles": [
        {"id": "orin_nano_15w", "label": "Orin Nano 15 W", "device": "NVIDIA Jetson Orin Nano", "watts_idle": 5, "watts_max": 15, "perf_scale": 2.6, "perception_hz_max": 10, "max_expansions": 60000},
        {"id": "orin_nx_25w", "label": "Orin NX 25 W", "device": "NVIDIA Jetson Orin NX", "watts_idle": 7, "watts_max": 25, "perf_scale": 1.6, "perception_hz_max": 15, "max_expansions": 120000},
        {"id": "agx_orin_60w", "label": "AGX Orin 60 W", "device": "NVIDIA Jetson AGX Orin", "watts_idle": 15, "watts_max": 60, "perf_scale": 0.9, "perception_hz_max": 20, "max_expansions": 250000},
        {"id": "rpi5_8w", "label": "Raspberry Pi 5", "device": "Raspberry Pi 5 (8 GB)", "watts_idle": 3, "watts_max": 8, "perf_scale": 5.0, "perception_hz_max": 5, "max_expansions": 30000},
    ],
    "backends": [{"id": "gazebo", "label": "Gazebo Harmonic", "available": True}, {"id": "mock", "label": "Mock physics", "available": True}],
    "defaults": {"sector": "medium", "gravity_g": 2.2, "fleet": {"rovers": 2, "drones": 1}, "edge_profile": "orin_nano_15w", "sim_backend": "gazebo", "seed": 42, "gazebo_gui": False, "sim_speed": 1.0},
    "sim_speed": {"choices": [0.5, 1.0, 1.5, 2.0], "min": 0.25, "max": 3.0, "default": 1.0},
}


class World:
    def __init__(self, config: dict):
        self.config = config
        self.L, self.res = SECTORS.get(config.get("sector", "medium"), SECTORS["medium"])
        self.g = float(config.get("gravity_g", 2.2))
        self.limits = physics_preview(self.g)
        self.profile_id = config.get("edge_profile", "orin_nano_15w")
        self.profile = PROFILES.get(self.profile_id, PROFILES["orin_nano_15w"])
        self.terrain = generate_terrain(self.L, self.res, int(config.get("seed", 42)))
        n = self.terrain.n
        self.n = n
        self.px_m = 0.1 if self.L <= 100 else 0.2
        self.tex = albedo_texture(self.terrain, self.px_m)
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
        self.X = (xs + 0.5) * self.res - self.L / 2
        self.Y = (ys + 0.5) * self.res - self.L / 2
        self.cls = np.zeros((n, n), np.uint8)  # what the fleet knows
        self.seen_t = np.full((n, n), np.inf, np.float64)
        self.rocks = list(self.terrain.rocks)
        self.rock_version = 0
        self.perm: list[list[float]] = []
        self.found_science: list[list[float]] = []
        # kind / wind_mps (docs/protocol.md §10.1): "dust" for a plain anomaly, "dust_storm" with its 12 m-AGL wind
        self.env = {"dust_level": 0.0, "seismic": False, "until_t": 0.0, "seismic_until": 0.0, "kind": None, "wind_mps": 0.0}
        self.pillars: list[list] = []  # landed pillars, each a list of [x, y, r] chunks (also in self.rocks)
        self.truth = self._truth_classes()
        self.new_hazard = (0, 0.0)
        self.last_trace_t = 0.0
        self.trace_id = 0
        self.dead_goals: list[tuple] = []  # frontiers A* could not reach; skipped from then on
        self.out_traces: list[dict] = []
        self.out_events: list[dict] = []
        self.rng = np.random.default_rng(int(config.get("seed", 42)) + 1)
        self.cs = 2.0 if self.L <= 100 else 4.0  # coarse planning cell (m)
        self.cn = int(self.L / self.cs)

    # ---- map -----------------------------------------------------------------------------
    def _truth_classes(self) -> np.ndarray:
        mx = self.limits["max_slope_deg"]
        s = self.terrain.slope_deg
        c = np.full(s.shape, CLS_FREE, np.uint8)
        c[s > 0.72 * mx] = CLS_CAUTION
        c[s > mx] = CLS_HAZARD
        for x, y, r in self.rocks:
            c[(self.X - x) ** 2 + (self.Y - y) ** 2 <= (r + 0.15) ** 2] = CLS_OBSTACLE
        for x, y in self.terrain.science:
            c[(self.X - x) ** 2 + (self.Y - y) ** 2 <= 0.8 ** 2] = CLS_SCIENCE
        for x, y, r in self.perm:
            c[(self.X - x) ** 2 + (self.Y - y) ** 2 <= r ** 2] = CLS_PERMANENT
        return c

    def refresh_truth(self):
        self.truth = self._truth_classes()
        known = self.cls > 0
        self.cls[known] = self.truth[known]

    def reveal(self, x: float, y: float, radius: float, now: float) -> tuple[int, list]:
        """Mark a disc as observed. Returns (#new cells, newly found science targets); new hazard
        cells accumulate in self.new_hazard (count, max slope) for slope_hazard traces."""
        i0, j0 = self.terrain.cell(x - radius, y - radius)
        i1, j1 = self.terrain.cell(x + radius, y + radius)
        sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        disc = (self.X[sl] - x) ** 2 + (self.Y[sl] - y) ** 2 <= radius ** 2
        # ragged LiDAR-like edge
        disc &= self.rng.random(disc.shape) > 0.08
        fresh = disc & ~np.isfinite(self.seen_t[sl])
        self.seen_t[sl][fresh] = now
        hz = fresh & (self.truth[sl] == CLS_HAZARD)
        if hz.any():
            c, m = self.new_hazard
            self.new_hazard = (c + int(hz.sum()), max(m, float(self.terrain.slope_deg[sl][hz].max())))
        self.cls[sl][disc] = self.truth[sl][disc]
        found = []
        for sx, sy in self.terrain.science:
            if [sx, sy] in self.found_science:
                continue
            ci, cj = self.terrain.cell(sx, sy)
            if self.cls[ci, cj] == CLS_SCIENCE:
                self.found_science.append([sx, sy])
                found.append([sx, sy])
        return int(fresh.sum()), found

    def stats(self) -> dict:
        known = int((self.cls > 0).sum())
        total = self.n * self.n
        hz = int(np.isin(self.cls, (CLS_HAZARD, CLS_PERMANENT)).sum())
        h = self.terrain.heights
        return {"percent": round(100.0 * known / total, 2), "known_cells": known, "total_cells": total,
                "hazard_cells": hz, "science": self.found_science, "permanent_hazards": self.perm,
                "elev_min": float(h.min()), "elev_max": float(h.max())}

    def meta(self) -> dict:
        h = self.terrain.heights
        return {"size_m": self.L, "res": self.res, "cells": self.n, "origin": [-self.L / 2, -self.L / 2],
                "elev_min": float(h.min()), "elev_max": float(h.max())}

    # ---- planning ----------------------------------------------------------------------------
    def _coarse(self) -> tuple[np.ndarray, np.ndarray]:
        k = int(self.cs / self.res)
        c = self.cls[: self.cn * k, : self.cn * k].reshape(self.cn, k, self.cn, k)
        lethal = np.isin(c, LETHAL).any(axis=(1, 3))
        known = (c > 0).mean(axis=(1, 3))
        return lethal, known

    def plan(self, start: tuple, goal: tuple) -> list:
        lethal, known = self._coarse()
        cn, cs, L = self.cn, self.cs, self.L
        to_c = lambda p: (min(cn - 1, max(0, int((p[1] + L / 2) / cs))), min(cn - 1, max(0, int((p[0] + L / 2) / cs))))
        s, gl = to_c(start), to_c(goal)
        lethal[s] = False
        lethal[gl] = False
        openq = [(0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == gl:
                break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not di and not dj:
                        continue
                    nb = (cur[0] + di, cur[1] + dj)
                    if not (0 <= nb[0] < cn and 0 <= nb[1] < cn) or lethal[nb]:
                        continue
                    step = math.hypot(di, dj) * (1.0 + 0.6 * (1 - known[nb]))
                    nc = cost[cur] + step
                    if nc < cost.get(nb, 1e18):
                        cost[nb] = nc
                        came[nb] = cur
                        h = math.hypot(nb[0] - gl[0], nb[1] - gl[1])
                        heapq.heappush(openq, (nc + h, nb))
        if gl not in came:
            return []
        cells = []
        c = gl
        while c is not None:
            cells.append(c)
            c = came[c]
        cells.reverse()
        pts = [start] + [((j + 0.5) * cs - L / 2, (i + 0.5) * cs - L / 2) for i, j in cells[1:-1]] + [goal]
        smooth = self._resample(self._smooth(pts), 0.8)
        if not self.path_blocked(smooth, len(smooth), skip_m=1.0):
            return smooth
        return self._resample(pts, 0.8)  # corner-cutting hit something: keep the cell-centre route

    @staticmethod
    def _smooth(pts: list) -> list:
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for k in range(1, len(pts) - 1):
            a, b, c = out[-1], pts[k], pts[k + 1]
            out.append(((a[0] + 2 * b[0] + c[0]) / 4, (a[1] + 2 * b[1] + c[1]) / 4))
        out.append(pts[-1])
        return out

    @staticmethod
    def _resample(pts: list, step: float) -> list:
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            for k in range(1, max(1, int(d / step)) + 1):
                f = k / max(1, int(d / step))
                out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
        return [[round(x, 2), round(y, 2)] for x, y in out][:150]

    def path_blocked(self, path: list, ahead: int = 30, skip_m: float = 2.0) -> int:
        """Class code of the first lethal cell on the path ahead (0 = clear)."""
        if not path:
            return 0
        x0, y0 = path[0]
        for x, y in path[1:ahead]:
            if math.hypot(x - x0, y - y0) < skip_m:
                continue
            i, j = self.terrain.cell(x, y)
            if self.cls[i, j] in LETHAL:
                return int(self.cls[i, j])
        return 0

    def frontier_goal(self, x: float, y: float, yaw: float, avoid: list) -> tuple | None:
        lethal, known = self._coarse()
        best, best_s = None, -1e9
        cn, cs, L = self.cn, self.cs, self.L
        for i in range(cn):
            for j in range(cn):
                if lethal[i, j] or known[i, j] > 0.45:
                    continue
                nb = known[max(0, i - 1): i + 2, max(0, j - 1): j + 2]
                if nb.max() < 0.5:
                    continue
                gx, gy = (j + 0.5) * cs - L / 2, (i + 0.5) * cs - L / 2
                d = math.hypot(gx - x, gy - y)
                if d < 6 or abs(gx) > 0.47 * L or abs(gy) > 0.47 * L:
                    continue
                if any(math.hypot(gx - ax, gy - ay) < 3 for ax, ay in self.dead_goals):
                    continue
                turn = abs(math.atan2(math.sin(math.atan2(gy - y, gx - x) - yaw), math.cos(math.atan2(gy - y, gx - x) - yaw)))
                s = -0.5 * d - 4.0 * turn - sum(20.0 for ax, ay in avoid if math.hypot(gx - ax, gy - ay) < 8) + self.rng.uniform(0, 3)
                if s > best_s:
                    best, best_s = (gx, gy), s
        return best

    def pillar_ahead(self, x: float, y: float, yaw: float, max_m: float = 16.0, half_fov: float = 0.7) -> float | None:
        """Range to the nearest landed pillar chunk in view ahead of (x, y, yaw), or None."""
        best = None
        for chunks in self.pillars:
            for cx, cy, _ in chunks:
                fwd = (cx - x) * math.cos(yaw) + (cy - y) * math.sin(yaw)
                lat = -(cx - x) * math.sin(yaw) + (cy - y) * math.cos(yaw)
                if 1.0 <= fwd <= max_m and abs(math.atan2(lat, fwd)) <= half_fov:
                    d = math.hypot(fwd, lat)
                    best = d if best is None else min(best, d)
        return best

    # ---- traces & events -----------------------------------------------------------------
    def trace(self, robot: str, state: str, kind: str, sensors: list, inp: str, reasoning: str, conf: float,
              action: str, severity: str = "info", latency_ms: float | None = None):
        self.trace_id += 1
        e = {"id": self.trace_id, "t": round(time.time(), 2), "robot": robot, "state": state, "kind": kind,
             "sensors": sensors, "input": inp, "reasoning": reasoning, "confidence": round(conf, 2),
             "action": action, "severity": severity,
             "latency_ms": round(latency_ms if latency_ms is not None else float(self.rng.uniform(18, 60)), 1)}
        self.out_traces.append(e)
        self.last_trace_t = time.time()

    def event(self, ev: dict):
        ev.setdefault("t", round(time.time(), 2))
        self.out_events.append(ev)

    def edge(self, robot: str, extra_ms: float, throttle: int, unc: float) -> dict:
        p = self.profile
        perc = float(self.rng.normal(10.5, 0.8)) + extra_ms
        plan = float(self.rng.normal(3.4, 0.5)) + (25.0 if extra_ms > 8 else 0.0)
        mapping = float(self.rng.normal(6.0, 0.6))
        hz = min(10.0, p["hz"]) * (1 - 0.25 * throttle)
        load = (perc * p["perf"] * hz + plan * p["perf"] * 10 + mapping * p["perf"] * 3) / 1000.0
        return {"robot": robot, "profile": self.profile_id,
                "host_ms": {"perception": round(perc, 1), "planner": round(plan, 1), "mapping": round(mapping, 1)},
                "device_ms": round((perc + plan + mapping) * p["perf"] * float(self.rng.uniform(0.95, 1.05)), 1),
                "loop_hz": round(min(10.0, 1000.0 / max(1.0, (perc + plan) * p["perf"])), 1),
                "load_pct": round(100 * load, 1), "power_w": round(p["idle"] + (p["max"] - p["idle"]) * min(1.0, load), 1),
                "perception_hz": round(hz, 1), "throttle_level": throttle, "uncertainty": round(unc, 2)}
