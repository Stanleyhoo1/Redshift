#!/usr/bin/env python3
"""UI-only demo harness: serves the full docs/protocol.md contract with animated fake data.

    python3 web_ui/dev/fake_backend.py [--port 8000] [--fast-boot]

REST /api/*, WebSocket /ws, and (if built) the static UI from web_ui/dist. No ROS required.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from fake_agents import LOCK_LABEL, Rover, Scout, display, short
from fake_render import b64_jpeg, lidar_points, map_png_b64, physics_preview, rover_camera, scout_camera
from fake_world import OPTIONS, World

DIST = Path(__file__).resolve().parent.parent / "dist"
# docs/protocol.md §10.1 / §10.6
PILLAR_H, PILLAR_R, PILLAR_SIDE, PILLAR_AHEAD, PILLAR_LAND_S = 6.0, 0.35, 2.8, 8.0, 1.65
STORM_S, STORM_WIND = 45.0, 14.0


class Hub:
    """Mission lifecycle + broadcast to every connected client."""

    def __init__(self, fast_boot: bool):
        self.fast_boot = fast_boot
        self.status = "idle"
        self.config: dict | None = None
        self.backend: str | None = None
        self.checklist: dict = {}
        self.detail: str | None = None
        self.started_at: float | None = None
        self.world: World | None = None
        self.rovers: list[Rover] = []
        self.scouts: list[Scout] = []
        self.clients: dict[WebSocket, dict] = {}
        self.task: asyncio.Task | None = None
        self.t_run = 0.0
        self.speed = 1.0     # UI dev only: scales motion; event timers stay on wall time
        self.pending_rocks: list = []
        self.rng = np.random.default_rng(3)
        self._fleet_t = 0.0                 # last 1 Hz /fleet/sos + /fleet/rescue repeat
        self._phase: dict[str, str] = {}    # rescuer id -> last announced phase
        self._done_until: dict[str, float] = {}  # keep a DONE rescue block in telemetry briefly

    # ---- outbound ------------------------------------------------------------------------
    def status_msg(self) -> dict:
        return {"type": "status", "status": self.status, "backend": self.backend, "detail": self.detail, "checklist": self.checklist, "config": self.config}

    async def send(self, ws: WebSocket, msg: dict | str):
        try:
            await ws.send_text(msg if isinstance(msg, str) else json.dumps(msg))
        except Exception:
            self.clients.pop(ws, None)

    async def broadcast(self, msg: dict):
        text = json.dumps(msg)
        await asyncio.gather(*(self.send(ws, text) for ws in list(self.clients)))

    async def set_status(self, status: str, detail: str | None = None):
        self.status, self.detail = status, detail
        await self.broadcast(self.status_msg())

    # ---- lifecycle -----------------------------------------------------------------------
    async def deploy(self, cfg: dict):
        await self.abort(silent=True)
        self.config = {**OPTIONS["defaults"], **cfg}
        self.backend = self.config["sim_backend"]
        self.speed = float(self.config.get("sim_speed") or 1.0)
        f = self.config["fleet"]
        robots = [f"rover_{i}" for i in range(int(f.get("rovers", 1)))] + [f"scout_{i}" for i in range(int(f.get("drones", 1)))]
        world_key = "gazebo" if self.backend == "gazebo" else "mock_world"
        keys = [world_key] + (["bridge"] if self.backend == "gazebo" else []) + ["sensor_effects", "perception", "mapping"] + robots
        self.checklist = {k: False for k in keys}
        self.status, self.detail, self.started_at = "launching", None, time.time()
        self.task = asyncio.create_task(self._run(robots, keys))

    async def abort(self, silent: bool = False):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        self.task, self.world = None, None
        self.pending_rocks = []
        if not silent and self.status != "idle":
            await self.set_status("stopped", "Mission aborted by operator")

    async def _boot(self, robots: list, keys: list):
        cfg = self.config
        log = lambda line: self.broadcast({"type": "boot_log", "line": line, "t": round(time.time(), 3)})
        g = float(cfg["gravity_g"])
        steps = [
            (f"[mission] writing runtime/mission_config.yaml (sector={cfg['sector']}, g={g:.1f}, seed={cfg['seed']})", None),
            (f"[world_builder] terrain {cfg['sector']}: heightmap 257×257, {len(self.world.terrain.rocks)} boulders, {len(self.world.terrain.science)} science crystals", None),
            (f"[world_builder] gravity <0 0 {-9.81 * g:.2f}>, wheel effort limit 20.8 N·m", None),
        ]
        if self.backend == "gazebo":
            steps += [("[gz] gz sim -s -r --headless-rendering runtime/world.sdf", None), ("[gz] Gazebo Harmonic 8.15 server started, RTF 0.97", "gazebo"),
                      ("[bridge] parameter_bridge: 6 topics per robot bridged", "bridge")]
        else:
            steps += [("[mock_world] kinematic world up, numpy raycast lidar 360×16 @10 Hz", "mock_world")]
        steps += [("[sensor_effects] dust/seismic layer passthrough (clear)", "sensor_effects"),
                  (f"[perception] {len(robots)} nodes ready, edge profile {cfg['edge_profile']}", "perception"),
                  ("[mapping_coordinator] global grid ready, publishing layers @3 Hz", "mapping")]
        for r in robots:
            steps.append((f"[{r}] odom + first camera frame received, autonomy online", r))
        delay = 0.15 if self.fast_boot else 0.75
        for line, key in steps:
            await log(line)
            if key:
                self.checklist[key] = True
                await self.broadcast(self.status_msg())
            await asyncio.sleep(delay * float(self.rng.uniform(0.6, 1.4)))
        await log("[mission] all subsystems ready, handing control to the agents")

    async def _run(self, robots: list, keys: list):
        try:
            self.world = World(self.config)
            w = self.world
            await self.broadcast(self.status_msg())
            sx, sy = w.terrain.spawn
            # rovers fan out from the pad on diverging headings, so a rescue later has real ground to cover
            self.rovers = [Rover(r, sx + 3.0 * k, sy - 1.5 * k, math.radians(45 - 55 * k)) for k, r in enumerate(robots) if r.startswith("rover")]
            self.scouts = [Scout(r, sx - 2.0, sy - 2.0, w.L) for r in robots if r.startswith("scout")]
            for r in self.rovers:
                r.peers = [o for o in self.rovers if o is not r]
            self._phase, self._done_until, self._fleet_t = {}, {}, 0.0
            await self._boot(robots, keys)
            await self.set_status("running")
            self.t_run = time.time()
            tick, dt = 0, 1 / 15
            nxt = time.perf_counter()
            while True:
                now = time.time()
                self._step(dt * self.speed, now)
                await self._emit(tick, now)
                tick += 1
                nxt += dt
                await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surface crashes to the UI as a failed launch
            import traceback
            traceback.print_exc()
            await self.set_status("failed", f"fake backend crashed: {e}")

    def _step(self, dt: float, now: float):
        w = self.world
        env = w.env
        if env["dust_level"] > 0 and now > env["until_t"]:
            env.update({"dust_level": 0.0, "kind": None, "wind_mps": 0.0})
        env["seismic"] = now < env.get("seismic_until", 0)
        for item in [p for p in self.pending_rocks if p[0] <= now]:
            self.pending_rocks.remove(item)
            w.rocks.extend(item[1])
            if len(item) > 3 and item[3] == "pillar":
                w.pillars.append(item[1])
            w.rock_version += 1
            w.refresh_truth()
            w.reveal(item[2][0], item[2][1], 4.5, now)
        for r in self.rovers:
            r.step(w, dt, now, self.scouts)
            _, found = w.reveal(r.x, r.y, 11.0 if env["dust_level"] == 0 else 7.0, now)
            for sx, sy in found:
                w.trace(r.id, r.state, "science_target", ["CAM", "DEPTH"], f"Cyan crystalline outcrop 0.6 m wide at ({sx:.1f}, {sy:.1f})",
                        "Hue 178°, saturation 0.71 and a 0.3 m depth bump match the SCIENCE signature", 0.87, f"Logged science target #{len(w.found_science)}")
        for s in self.scouts:
            s.step(w, dt, now, self.rovers)
            if s.alt > 6:
                _, found = w.reveal(s.x, s.y, 12.0, now)
                for sx, sy in found:
                    w.trace(s.id, s.state, "science_target", ["CAM"], f"Nadir frame: cyan cluster at ({sx:.1f}, {sy:.1f})",
                            "Colour and texture match crystal signature from 12 m AGL", 0.81, f"Logged science target #{len(w.found_science)}; tagged on map")
        self._rescue(now)
        self._narrate(now)
        for a in self.rovers + self.scouts:  # edge budget throttle with hysteresis
            e = w.edge(a.id, a.extra_ms, a.throttle, a.unc)
            a._edge = e
            a.extra_ms = max(0.0, a.extra_ms - dt * 2) if a.mode == "FUSED" else a.extra_ms
            over = e["load_pct"] > 90
            under = e["load_pct"] < 60
            a._over = (getattr(a, "_over", 0.0) + dt) if over else 0.0
            a._under = (getattr(a, "_under", 0.0) + dt) if under else 0.0
            cool = now - getattr(a, "_thr_t", 0.0) > 4.0
            if a._over > 1.0 and cool and a.throttle < 2:
                a.throttle += 1
                a._thr_t = now
                w.trace(a.id, a.state, "edge_throttle", ["EDGE"], f"Compute load {e['load_pct']:.0f} % of {w.profile_id} budget, {e['power_w']:.1f} W",
                        "Load > 90 %: perception rate must drop to keep the control loop at 10 Hz", 0.95, f"Perception throttled −25 % to {e['perception_hz'] * 0.75:.1f} Hz", "warn")
            elif a._under > 1.5 and cool and a.throttle > 0:
                a.throttle -= 1
                a._thr_t = now
                w.trace(a.id, a.state, "edge_throttle", ["EDGE"], f"Compute load {e['load_pct']:.0f} %", "Load < 60 %: headroom restored", 0.93, "Perception rate restored")

    # ---- §8 SOS / rescue orchestration (the fleet + world side of the scripted sequence) ----------
    def _rescue(self, now: float):
        w = self.world
        by_id = {r.id: r for r in self.rovers}
        for r in self.rovers:
            if r.state == "AWAIT_RESCUE" and r.sos_id and not r.rescuer_id:
                able = [o for o in self.rovers if o is not r and not o.faults and o.rescue_target is None
                        and o.state not in ("AWAIT_RESCUE", "SAFE_HOLD", "RECOVERY_REVERSE")]
                if able:
                    o = min(able, key=lambda q: math.hypot(q.x - r.x, q.y - r.y))
                    r.rescuer_id = o.id
                    o.start_rescue(r, w, now)
        for o in self.rovers:
            t = o.rescue_target
            if t is None:
                continue
            # world side of §8.5 repair: checks the true centre distance, then ramps the target's wheels
            if o.state == "RESCUE_DOCKING" and o.docked and not o.repair_requested:
                o.repair_requested = True
                dist = math.hypot(t.x - o.x, t.y - o.y)
                ok = dist <= 1.6
                w.event({"op": "repair", "ok": ok, "robot": t.id, "by": o.id, "distance_m": round(dist, 2),
                         "detail": f"Repairing {t.id} drivetrain" if ok else f"Too far for a repair: {dist:.2f} m > 1.6 m"})
                if ok:
                    t.begin_repair(now)
                    o.set_state("REPAIRING", now)
                    w.trace(o.id, "REPAIRING", "repair_start", ["DIAG", "FLEET"],
                            f"Docked {dist:.2f} m from {short(t.id)}; its motor controllers report {', '.join(t.faults)}",
                            "Within the 1.6 m repair reach; the world confirms contact", 0.94,
                            "REPAIRING: swapping the failed drive modules (≈ 4 s)", "info", 15.0)
            elif o.state == "REPAIRING" and t.repair_done:
                w.event({"op": "repair", "ok": True, "done": True, "robot": t.id, "by": o.id,
                         "distance_m": round(math.hypot(t.x - o.x, t.y - o.y), 2), "detail": f"{t.id} drivetrain restored"})
                w.trace(o.id, "REPAIRING", "repair_complete", ["DIAG", "FLEET"],
                        f"{short(t.id)} wheel health {min(t.wheel_health):.2f}–{max(t.wheel_health):.2f}; faults cleared",
                        f"{short(t.id)} health {t.health_pct():.0f} % ≥ 60 %: it can drive itself again", 0.96,
                        "Rescue complete: resuming my own exploration", "info", 11.0)
                self._rescue_event(o, "DONE")
                self._done_until[o.id] = now + 3.0
                o.set_state("RESUMING", now)
                o.locked = False
        # resolved SOS: the damaged rover left AWAIT_RESCUE on its own sensors
        for r in self.rovers:
            if r.sos_id and r.state in ("RESUMING", "EXPLORING", "CAUTION") and getattr(r, "_sos_open", False):
                r._sos_open = False
                w.event(self._sos_event(r, "resolved"))
            if r.state == "AWAIT_RESCUE":
                r._sos_open = True
        # phase changes go out immediately, everything repeats at 1 Hz
        for o in self.rovers:
            if o.rescue_target is not None and o.state in ("RESCUE_TRANSIT", "RESCUE_DOCKING", "REPAIRING"):
                ph = o.rescue_block("rescuer")["phase"]
                if self._phase.get(o.id) != ph:
                    self._phase[o.id] = ph
                    self._rescue_event(o, ph)
        if now - self._fleet_t >= 1.0:
            self._fleet_t = now
            for r in self.rovers:
                if r.state == "AWAIT_RESCUE" and r.sos_id:
                    w.event(self._sos_event(r, "active"))
            for o in self.rovers:
                if o.rescue_target is not None and o.state in ("RESCUE_TRANSIT", "RESCUE_DOCKING", "REPAIRING"):
                    self._rescue_event(o, o.rescue_block("rescuer")["phase"])
        # drop finished rescue links after the DONE block has been shown for a moment
        for oid, until in list(self._done_until.items()):
            if now > until:
                del self._done_until[oid]
                o = by_id.get(oid)
                if o is not None:
                    o.rescue_target = None
                    self._phase.pop(oid, None)

    def _sos_event(self, r, status: str) -> dict:
        h = r.health_json()
        return {"op": "sos", "id": r.sos_id, "robot": r.id, "xy": [round(r.x, 2), round(r.y, 2)], "yaw": round(r.yaw, 3),
                "health_pct": h["pct"], "faults": h["faults"], "status": status}

    def _rescue_event(self, o, phase: str):
        b = o.rescue_block("rescuer")
        self.world.event({"op": "rescue", "id": b["id"], "rescuer": o.id, "target": o.rescue_target.id, "phase": phase,
                          "range_m": b["range_m"], "locked": b["locked"] and phase in ("DOCKING", "REPAIRING")})

    def _rescue_json(self, a) -> dict | None:
        if a.kind != "rover":
            return None
        if a.rescue_target is not None:
            return a.rescue_block("rescuer")
        for o in self.rovers:
            if o.rescue_target is a:
                return o.rescue_block("target")
        return None

    def _narrate(self, now: float):
        """Slope-hazard discoveries plus a routine status trace whenever the feed goes quiet."""
        w = self.world
        cnt, mx = w.new_hazard
        lim = w.limits
        busy = ("SAFE_HOLD", "AWAIT_RESCUE", "RESCUE_TRANSIT", "RESCUE_DOCKING", "REPAIRING")
        r = next((q for q in self.rovers if q.state not in busy), None)
        if cnt >= 30 and r and now - getattr(self, "_slope_t", 0) > 7:
            self._slope_t = now
            w.new_hazard = (0, 0.0)
            w.trace(r.id, r.state, "slope_hazard", ["LIDAR", "MAP"], f"{cnt} newly mapped cells steeper than the limit, peak {mx:.1f}°",
                    f"Slope {mx:.1f}° > {lim['max_slope_deg']:.1f}° traversable at {w.g:.1f} g ({lim['factor'].replace('_', ' ')} bound)", 0.9,
                    "Marked lethal in the costmap; inflated 0.6 m for robot radius", "warn")
            return
        # routine chatter pauses while a rescue is under way, so the rescue reads as one story
        rescue_on = any(q.state in busy[1:] for q in self.rovers)
        if now - w.last_trace_t < (9.0 if rescue_on else 2.3):
            return
        agents = self.rovers + self.scouts
        if not agents:
            return
        self._rr = (getattr(self, "_rr", -1) + 1) % len(agents)
        a = agents[self._rr]
        st = w.stats()
        if a.kind == "scout":
            if a.state not in ("SURVEY", "TAKEOFF"):
                return
            w.trace(a.id, a.state, "path_planned", ["CAM", "ODOM", "MAP"], f"Nadir coverage at {a.alt:.0f} m AGL; sector {st['percent']:.1f} % mapped",
                    "Lane on schedule; no scan requests pending", 0.86, f"Continue survey at {a.speed:.1f} m/s")
        elif a.state == "CAUTION":
            w.trace(a.id, a.state, "path_planned", ["LIDAR", "IMU"], f"LiDAR-only corridor clear {float(w.rng.uniform(7, 11)):.1f} m; uncertainty {a.unc:.2f}",
                    "Camera still degraded; LiDAR alone keeps the corridor check within budget", 0.74, f"Creeping at {a.speed:.2f} m/s with extra inflation", "warn")
        elif a.state == "EXPLORING":
            slope = float(w.terrain.slope_deg[w.terrain.cell(a.x, a.y)])
            w.trace(a.id, a.state, "path_planned", ["LIDAR", "CAM", "IMU"], f"Corridor clear {float(w.rng.uniform(11, 19)):.1f} m; slope underfoot {slope:.1f}°; slip {a.slip:.2f}",
                    f"Slope {slope:.1f}° < {lim['max_slope_deg']:.1f}° limit; tilt {abs(a.roll):.1f}° well under the {lim['tip_alarm_deg']:.1f}° alarm", 0.9,
                    f"Hold course at {a.speed:.2f} m/s (limit {lim['v_max']:.2f} m/s at {w.g:.1f} g)")

    def robot_json(self, a) -> dict:
        lim = self.world.limits
        e = a._edge
        return {
            "type": a.kind, "pose": {"x": round(a.x, 3), "y": round(a.y, 3), "z": round(a.z, 3), "roll": round(math.radians(a.roll), 4), "pitch": round(math.radians(a.pitch), 4), "yaw": round(a.yaw, 4)},
            "speed": round(a.speed, 2), "state": a.state, "mode": a.mode, "goal": a.goal,
            "v_limit": round(lim["v_max"] * (0.35 if a.state == "CAUTION" else 1.0), 2) if a.kind == "rover" else 4.0,
            "slip": round(getattr(a, "slip", 0.0), 3), "tip_alarm_deg": round(lim["tip_alarm_deg"], 1), "max_slope_deg": round(lim["max_slope_deg"], 1),
            "altitude_agl": round(a.alt, 1) if a.kind == "scout" else None,
            "imu": {"roll_deg": round(a.roll, 2), "pitch_deg": round(a.pitch, 2), "gyro": a.gyro},
            "uncertainty": {"score": round(a.unc, 3), "camera_conf": round(max(0.05, 1 - a.unc * 1.2), 2), "lidar_conf": round(max(0.2, 1 - a.unc * 0.6), 2), "imu_vib": 0.4 if self.world.env["seismic"] else 0.05},
            "edge": e, "path": [list(p) for p in a.path[:150]], "trail": a.trail[-300:], "last_safe": getattr(a, "last_safe", None),
            # §8.7
            "role": a.role, "name": a.name, "health": a.health_json(), "rescue": self._rescue_json(a),
            "sensors": a.sensors_json(self.world, self.rovers + self.scouts),
            # §10.3 / §10.6
            "power_save": a.kind == "rover" and a.power_save,
            "wind_mps": round(a.wind_est, 2) if a.kind == "scout" and a.alt > 1.0 else None,
            "wind_descent": a.kind == "scout" and a.wind_descent,
            "detections": self._detections(a),
        }

    def _detections(self, a) -> list:
        """Compact §10.6 detections: a landed pillar in a rover's camera view."""
        rng_m = self.world.pillar_ahead(a.x, a.y, a.yaw) if a.kind == "rover" else None
        if rng_m is None:
            return []
        return [{"cls": "PILLAR", "conf": round(0.86 * (1 - 0.4 * self.world.env["dust_level"]), 2), "range_m": round(rng_m, 1),
                 "world_xy": [round(a.x + rng_m * math.cos(a.yaw), 2), round(a.y + rng_m * math.sin(a.yaw), 2)]}]

    async def _emit(self, tick: int, now: float):
        w = self.world
        st = w.stats()
        mission = {"status": self.status, "backend": self.backend, "elapsed_s": round(now - self.t_run, 1), "percent_mapped": st["percent"],
                   "sim_speed": self.speed, "rtf": self.speed, "sim_time_s": round(now - self.t_run, 2),
                   "science_found": len(w.found_science), "hazard_cells": st["hazard_cells"],
                   "env": {"dust_level": w.env["dust_level"], "seismic": w.env["seismic"],
                           "kind": w.env["kind"] if w.env["dust_level"] > 0 else None,
                           "wind_mps": w.env["wind_mps"] if w.env["dust_level"] > 0 else 0.0},
                   "config": self.config}
        await self.broadcast({"type": "telemetry", "t": round(now, 3), "mission": mission, "robots": {a.id: self.robot_json(a) for a in self.rovers + self.scouts}})
        for tr in w.out_traces:
            await self.broadcast({"type": "trace", "entry": tr})
        for ev in w.out_events:
            await self.broadcast({"type": "event", "event": ev})
        w.out_traces.clear()
        w.out_events.clear()
        if tick % 5 == 0:
            await self.broadcast({"type": "map", "png_b64": map_png_b64(w.terrain, w.cls, w.seen_t, now), "meta": w.meta(), "stats": st})
        wanted = {c["camera"] for c in self.clients.values()}
        agents = {a.id: a for a in self.rovers + self.scouts}
        full, low = tick % 2 == 0, tick % 4 == 0
        rgb, dep, lids = {}, {}, {}
        storm = w.env["kind"] == "dust_storm" and w.env["dust_level"] > 0
        pillar_chunks = {(c[0], c[1]) for chunks in w.pillars for c in chunks}
        for rid, a in agents.items():
            sel = rid in wanted
            if not (low or (full and sel)):
                continue
            pose = {"x": a.x, "y": a.y, "z": a.z, "yaw": a.yaw}
            mates = [{"x": o.x, "y": o.y, "short": short(o.id), "damaged": bool(o.faults),
                      "lock": (LOCK_LABEL.format(short=short(o.id)) if a.kind == "rover" and a.rescue_target is o and a.locked else None)}
                     for o in self.rovers if o is not a]
            if a.kind == "rover":
                out = rover_camera(w.terrain, w.tex, w.px_m, pose, w.rocks, w.env["dust_level"], w.env["seismic"], a.id, a.mode, a.unc, w.rng,
                                   others=mates, want_depth=sel and full, storm=storm, pillar_chunks=pillar_chunks)
            else:
                out = scout_camera(w.tex, w.px_m, w.L, pose, w.terrain.science, w.perm, a.state == "SCANNING", w.env["dust_level"], a.id,
                                   rovers=mates, terrain=w.terrain, alt=a.alt, want_depth=sel and full, storm=storm)
            img, depth = out if isinstance(out, tuple) else (out, None)
            rgb[rid] = json.dumps({"type": "camera", "robot": rid, "kind": "rgb", "jpeg_b64": b64_jpeg(img), "t": round(now, 3)})
            if depth is not None:
                dep[rid] = json.dumps({"type": "camera", "robot": rid, "kind": "depth", "jpeg_b64": b64_jpeg(depth, 80), "t": round(now, 3)})
        for rid in wanted:
            a = agents.get(rid)
            if a and tick % 3 == 0 and a.kind == "rover":
                pose = {"x": a.x, "y": a.y, "z": a.z, "yaw": a.yaw}
                lids[rid] = json.dumps({"type": "lidar", "robot": rid, "points": lidar_points(w.terrain, pose, w.rocks, w.env["dust_level"], w.rng), "max_range": 25, "t": round(now, 3)})
        for ws, c in list(self.clients.items()):
            mine = c["camera"]
            for rid, msg in rgb.items():
                if rid == mine or low:
                    await self.send(ws, msg)
            if mine in dep:
                await self.send(ws, dep[mine])
            if mine in lids:
                await self.send(ws, lids[mine])

    # ---- injections ----------------------------------------------------------------------
    def _rover(self, rid: str | None) -> Rover | None:
        for r in self.rovers:
            if rid is None or r.id == rid:
                return r
        return None

    def inject_rockslide(self, rid: str | None) -> dict:
        r = self._rover(rid)
        if not r or not self.world or self.status != "running":
            return {"ok": False, "detail": "No running mission with that rover"}
        w = self.world
        ahead = [p for p in r.path if 5.5 <= math.hypot(p[0] - r.x, p[1] - r.y) <= 8.0]
        cx, cy = ahead[0] if ahead else (r.x + 7 * math.cos(r.yaw), r.y + 7 * math.sin(r.yaw))
        k = r.path.index(ahead[0]) if ahead else -1
        tx, ty = ((r.path[k + 1][0] - r.path[k][0]), (r.path[k + 1][1] - r.path[k][1])) if 0 <= k < len(r.path) - 1 else (math.cos(r.yaw), math.sin(r.yaw))
        n = math.hypot(tx, ty) or 1.0
        px, py = -ty / n, tx / n
        rocks = [[round(cx + px * o + float(w.rng.normal(0, 0.15)), 2), round(cy + py * o + float(w.rng.normal(0, 0.15)), 2), round(float(w.rng.uniform(0.65, 1.0)), 2)]
                 for o in np.linspace(-3.5, 3.5, 7)]
        w.event({"op": "spawn_rockslide", "ok": True, "robot": r.id, "impact": [round(cx, 2), round(cy, 2)], "rocks": rocks})
        self.pending_rocks.append((time.time() + 1.1, rocks, (cx, cy)))
        return {"ok": True, "detail": f"Rockslide of {len(rocks)} boulders 7 m ahead of {r.id}"}

    def inject_anomaly(self, dust: float | None, duration: float | None) -> dict:
        if not self.world or self.status != "running":
            return {"ok": False, "detail": "No running mission"}
        now = time.time()
        lvl, dur = float(dust if dust is not None else 0.85), float(duration if duration is not None else 25)
        self.world.env.update({"dust_level": lvl, "until_t": now + dur, "seismic_until": now + 4, "kind": "dust", "wind_mps": 0.0})
        self.world.event({"op": "anomaly", "ok": True, "dust_level": lvl, "until_t": round(now + dur, 2)})
        return {"ok": True, "detail": f"Dust {lvl:.2f} for {dur:.0f} s with a 4 s tremor"}

    def inject_pillar(self, rid: str | None) -> dict:
        """§10.1: a 6 m basalt pillar 2.8 m off the route topples across it; its chunks land as obstacles."""
        r = self._rover(rid)
        if not r or not self.world or self.status != "running":
            return {"ok": False, "detail": "No running mission with that rover"}
        w = self.world
        ahead = [p for p in r.path if 6.0 <= math.hypot(p[0] - r.x, p[1] - r.y) <= PILLAR_AHEAD]
        cx, cy = ahead[-1] if ahead else (r.x + PILLAR_AHEAD * math.cos(r.yaw), r.y + PILLAR_AHEAD * math.sin(r.yaw))
        k = r.path.index(ahead[-1]) if ahead else -1
        tx, ty = ((r.path[k + 1][0] - r.path[k][0]), (r.path[k + 1][1] - r.path[k][1])) if 0 <= k < len(r.path) - 1 else (math.cos(r.yaw), math.sin(r.yaw))
        n = math.hypot(tx, ty) or 1.0
        side = 1.0 if w.rng.random() < 0.5 else -1.0
        ux, uy = side * ty / n, -side * tx / n          # fall direction: across the route, from the base's side
        bx, by = cx - PILLAR_SIDE * ux, cy - PILLAR_SIDE * uy
        tip = (bx + PILLAR_H * ux, by + PILLAR_H * uy)
        chunks = [[round(bx + ux * d, 2), round(by + uy * d, 2), round(PILLAR_R * 1.15, 2)] for d in np.linspace(0.3, PILLAR_H - 0.3, 9)]
        w.event({"op": "pillar_collapse", "ok": True, "robot": r.id, "base": [round(bx, 2), round(by, 2)], "tip": [round(tip[0], 2), round(tip[1], 2)],
                 "cross": [round(cx, 2), round(cy, 2)], "height_m": PILLAR_H, "radius_m": PILLAR_R, "waited_s": 0.0})
        self.pending_rocks.append((time.time() + PILLAR_LAND_S, chunks, (cx, cy), "pillar"))
        return {"ok": True, "detail": f"Basalt pillar toppling across {r.id} planned path"}

    def inject_storm(self, duration: float | None, wind: float | None) -> dict:
        if not self.world or self.status != "running":
            return {"ok": False, "detail": "No running mission"}
        now = time.time()
        dur, mps = float(duration if duration is not None else STORM_S), float(wind if wind is not None else STORM_WIND)
        self.world.env.update({"dust_level": 1.0, "until_t": now + dur, "kind": "dust_storm", "wind_mps": mps})
        self.world.event({"op": "anomaly", "ok": True, "kind": "dust_storm", "dust_level": 1.0, "duration_s": dur, "wind_mps": mps})
        return {"ok": True, "detail": f"Dust storm for {dur:.0f} s: solar blackout, {mps:.0f} m/s wind"}

    def inject_damage(self, rid: str | None) -> dict:
        r = self._rover(rid)
        if not r or not self.world or self.status != "running":
            return {"ok": False, "detail": "No running mission with that rover"}
        if r.faults or r.state in ("SAFE_HOLD", "AWAIT_RESCUE") or r.rescue_target is not None:
            return {"ok": False, "detail": f"{display(r.id)} is already damaged or busy with a rescue"}
        if any(o.rescue_target is r for o in self.rovers):
            return {"ok": False, "detail": f"{display(r.id)} is being rescued"}
        self.world.event(r.strike(self.world, time.time()))
        return {"ok": True, "detail": f"Debris strike: critical damage on {display(r.id)}"}

    def inject_tip(self, rid: str | None) -> dict:
        r = self._rover(rid)
        if not r or not self.world or self.status != "running":
            return {"ok": False, "detail": "No running mission with that rover"}
        r.jolt(time.time())
        self.world.event({"op": "jolt", "ok": True, "robot": r.id})
        return {"ok": True, "detail": f"Jolt applied to {r.id} chassis"}


def build_app(hub: Hub) -> FastAPI:
    app = FastAPI(title="Redshift fake backend")

    @app.get("/api/options")
    async def options():
        return OPTIONS

    @app.get("/api/physics_preview")
    async def preview(g: float = 2.2):
        return physics_preview(min(3.0, max(1.0, g)))

    @app.post("/api/mission/deploy")
    async def deploy(cfg: dict):
        await hub.deploy(cfg)
        return {"ok": True, "mission_id": f"m{int(time.time())}", "status": hub.status}

    @app.post("/api/mission/abort")
    async def abort():
        await hub.abort()
        return {"ok": True}

    @app.get("/api/mission/status")
    async def status():
        return {"status": hub.status, "backend": hub.backend, "config": hub.config, "started_at": hub.started_at, "checklist": hub.checklist, "detail": hub.detail}

    @app.post("/api/inject/rockslide")
    async def rockslide(body: dict | None = None):
        return JSONResponse(hub.inject_rockslide((body or {}).get("robot")))

    @app.post("/api/inject/anomaly")
    async def anomaly(body: dict | None = None):
        b = body or {}
        return JSONResponse(hub.inject_anomaly(b.get("dust_level"), b.get("duration_s")))

    @app.post("/api/inject/tip")
    async def tip(body: dict | None = None):
        return JSONResponse(hub.inject_tip((body or {}).get("robot")))

    @app.post("/api/inject/pillar")
    async def pillar(body: dict | None = None):
        return JSONResponse(hub.inject_pillar((body or {}).get("robot")))

    @app.post("/api/inject/storm")
    async def storm(body: dict | None = None):
        b = body or {}
        return JSONResponse(hub.inject_storm(b.get("duration_s"), b.get("wind_mps")))

    @app.post("/api/inject/damage")
    async def damage(body: dict | None = None):
        return JSONResponse(hub.inject_damage((body or {}).get("robot")))

    @app.post("/api/sim/speed")
    async def sim_speed(body: dict | None = None):
        x = (body or {}).get("speed")
        if not isinstance(x, (int, float)) or not 0.25 <= x <= 3.0:
            return JSONResponse({"ok": False, "detail": "speed must be 0.25–3.0"}, status_code=422)
        if hub.status not in ("running", "degraded"):
            return JSONResponse({"ok": False, "detail": "No running mission"})
        hub.speed = float(x)
        if hub.config:
            hub.config["sim_speed"] = hub.speed
        return JSONResponse({"ok": True, "speed": hub.speed, "detail": f"Simulation speed set to {hub.speed:g}×"})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients[ws] = {"camera": "rover_0"}
        await hub.send(ws, hub.status_msg())
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "select_camera" and isinstance(msg.get("robot"), str):
                    hub.clients[ws]["camera"] = msg["robot"]
        except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            hub.clients.pop(ws, None)

    if DIST.is_dir():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fast-boot", action="store_true", help="shorten the fake launch sequence")
    a = ap.parse_args()
    print(f"Redshift fake backend on http://{a.host}:{a.port}  (UI from {DIST if DIST.is_dir() else 'vite dev server'})")
    uvicorn.run(build_app(Hub(a.fast_boot)), host=a.host, port=a.port, log_level="warning")
