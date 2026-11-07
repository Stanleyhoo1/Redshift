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

from fake_agents import Rover, Scout
from fake_render import b64_jpeg, lidar_points, map_png_b64, physics_preview, rover_camera, scout_camera
from fake_world import World

DIST = Path(__file__).resolve().parent.parent / "dist"

OPTIONS = {
    "sectors": [
        {"id": "small", "label": "Small", "size_m": 50, "res_m": 0.5, "cells": 100},
        {"id": "medium", "label": "Medium", "size_m": 100, "res_m": 0.5, "cells": 200},
        {"id": "large", "label": "Large", "size_m": 200, "res_m": 1.0, "cells": 200},
    ],
    "gravity": {"min": 1.0, "max": 3.0, "step": 0.1, "default": 2.2},
    "fleet": {"rovers": {"min": 1, "max": 2, "default": 1}, "drones": {"min": 0, "max": 1, "default": 1}},
    "edge_profiles": [
        {"id": "orin_nano_15w", "label": "Orin Nano 15 W", "device": "NVIDIA Jetson Orin Nano", "watts_idle": 5, "watts_max": 15, "perf_scale": 2.6, "perception_hz_max": 10, "max_expansions": 60000},
        {"id": "orin_nx_25w", "label": "Orin NX 25 W", "device": "NVIDIA Jetson Orin NX", "watts_idle": 7, "watts_max": 25, "perf_scale": 1.6, "perception_hz_max": 15, "max_expansions": 120000},
        {"id": "agx_orin_60w", "label": "AGX Orin 60 W", "device": "NVIDIA Jetson AGX Orin", "watts_idle": 15, "watts_max": 60, "perf_scale": 0.9, "perception_hz_max": 20, "max_expansions": 250000},
        {"id": "rpi5_8w", "label": "Raspberry Pi 5", "device": "Raspberry Pi 5 (8 GB)", "watts_idle": 3, "watts_max": 8, "perf_scale": 5.0, "perception_hz_max": 5, "max_expansions": 30000},
    ],
    "backends": [{"id": "gazebo", "label": "Gazebo Harmonic", "available": True}, {"id": "mock", "label": "Mock physics", "available": True}],
    "defaults": {"sector": "medium", "gravity_g": 2.2, "fleet": {"rovers": 1, "drones": 1}, "edge_profile": "orin_nano_15w", "sim_backend": "gazebo", "seed": 42, "gazebo_gui": False},
}


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
        self.rng = np.random.default_rng(3)

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
            self.rovers = [Rover(r, sx + 3.0 * k, sy, math.radians(45)) for k, r in enumerate(robots) if r.startswith("rover")]
            self.scouts = [Scout(r, sx - 2.0, sy - 2.0, w.L) for r in robots if r.startswith("scout")]
            await self._boot(robots, keys)
            await self.set_status("running")
            self.t_run = time.time()
            tick, dt = 0, 1 / 15
            nxt = time.perf_counter()
            while True:
                now = time.time()
                self._step(dt, now)
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
            env["dust_level"] = 0.0
        env["seismic"] = now < env.get("seismic_until", 0)
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
        self._narrate(now)
        for a in self.rovers + self.scouts:  # edge budget throttle with hysteresis
            e = w.edge(a.id, a.extra_ms, a.throttle, a.unc)
            a._edge = e
            a.extra_ms = max(0.0, a.extra_ms - dt * 2) if a.mode == "FUSED" else a.extra_ms
            if e["load_pct"] > 90 and a.throttle < 2:
                a.throttle += 1
                w.trace(a.id, a.state, "edge_throttle", ["EDGE"], f"Compute load {e['load_pct']:.0f} % of {w.profile_id} budget, {e['power_w']:.1f} W",
                        "Load > 90 %: perception rate must drop to keep the control loop at 10 Hz", 0.95, f"Perception throttled −25 % to {e['perception_hz'] * 0.75:.1f} Hz", "warn")
            elif e["load_pct"] < 60 and a.throttle > 0:
                a.throttle -= 1
                w.trace(a.id, a.state, "edge_throttle", ["EDGE"], f"Compute load {e['load_pct']:.0f} %", "Load < 60 %: headroom restored", 0.93, "Perception rate restored")

    def _narrate(self, now: float):
        """Slope-hazard discoveries plus a routine status trace whenever the feed goes quiet."""
        w = self.world
        cnt, mx = w.new_hazard
        lim = w.limits
        r = self.rovers[0] if self.rovers else None
        if cnt >= 30 and r and now - getattr(self, "_slope_t", 0) > 7:
            self._slope_t = now
            w.new_hazard = (0, 0.0)
            w.trace(r.id, r.state, "slope_hazard", ["LIDAR", "MAP"], f"{cnt} newly mapped cells steeper than the limit, peak {mx:.1f}°",
                    f"Slope {mx:.1f}° > {lim['max_slope_deg']:.1f}° traversable at {w.g:.1f} g ({lim['factor'].replace('_', ' ')} bound)", 0.9,
                    "Marked lethal in the costmap; inflated 0.6 m for robot radius", "warn")
            return
        if now - w.last_trace_t < 2.3:
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
        }

    async def _emit(self, tick: int, now: float):
        w = self.world
        st = w.stats()
        mission = {"status": self.status, "backend": self.backend, "elapsed_s": round(now - self.t_run, 1), "percent_mapped": st["percent"],
                   "science_found": len(w.found_science), "hazard_cells": st["hazard_cells"],
                   "env": {"dust_level": w.env["dust_level"], "seismic": w.env["seismic"]}, "config": self.config}
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
        cams, lids = {}, {}
        for rid in wanted:
            a = agents.get(rid)
            if not a:
                continue
            if tick % 2 == 0:
                pose = {"x": a.x, "y": a.y, "z": a.z, "yaw": a.yaw}
                if a.kind == "rover":
                    img = rover_camera(w.terrain, w.tex, w.px_m, pose, w.rocks, w.env["dust_level"], w.env["seismic"], a.id, a.mode, a.unc, w.rng)
                else:
                    img = scout_camera(w.tex, w.px_m, w.L, pose, w.terrain.science, w.perm, a.state == "SCANNING", w.env["dust_level"], a.id)
                cams[rid] = json.dumps({"type": "camera", "robot": rid, "jpeg_b64": b64_jpeg(img), "t": round(now, 3)})
            if tick % 3 == 0 and a.kind == "rover":
                pose = {"x": a.x, "y": a.y, "z": a.z, "yaw": a.yaw}
                lids[rid] = json.dumps({"type": "lidar", "robot": rid, "points": lidar_points(w.terrain, pose, w.rocks, w.env["dust_level"], w.rng), "max_range": 25, "t": round(now, 3)})
        for ws, c in list(self.clients.items()):
            if c["camera"] in cams:
                await self.send(ws, cams[c["camera"]])
            if c["camera"] in lids:
                await self.send(ws, lids[c["camera"]])

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
        rocks = [[round(cx + px * o + float(w.rng.normal(0, 0.4)), 2), round(cy + py * o + float(w.rng.normal(0, 0.4)), 2), round(float(w.rng.uniform(0.5, 1.0)), 2)]
                 for o in np.linspace(-3.5, 3.5, 6)]
        w.rocks.extend(rocks)
        w.refresh_truth()
        w.reveal(cx, cy, 4.5, time.time())
        w.event({"op": "spawn_rockslide", "ok": True, "robot": r.id, "impact": [round(cx, 2), round(cy, 2)], "rocks": rocks})
        r.pending_block = time.time() + 1.2
        return {"ok": True, "detail": f"Rockslide of {len(rocks)} boulders 7 m ahead of {r.id}"}

    def inject_anomaly(self, dust: float | None, duration: float | None) -> dict:
        if not self.world or self.status != "running":
            return {"ok": False, "detail": "No running mission"}
        now = time.time()
        lvl, dur = float(dust if dust is not None else 0.85), float(duration if duration is not None else 25)
        self.world.env.update({"dust_level": lvl, "until_t": now + dur, "seismic_until": now + 4})
        self.world.event({"op": "anomaly", "ok": True, "dust_level": lvl, "until_t": round(now + dur, 2)})
        return {"ok": True, "detail": f"Dust {lvl:.2f} for {dur:.0f} s with a 4 s tremor"}

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
