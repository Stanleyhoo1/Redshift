"""Simulation: owns the terrain, the MuJoCo world, the shared map, the fleet and the rovers.

One call to step() advances physics by one agent period (1 / agent_hz sim seconds), then runs every
rover's decision tick. One rover per cycle (rotating) is on "fleet duty" and also refreshes the
shared frontiers and reachability, so that shared work counts toward a rover's compute budget.
The server calls step() from a background thread; the CLI below runs it flat out and can script
hazards and operator messages:

    python -m redshift.sim --seconds 300 [--seed N] [--hazard rockfall@60] [--hazard sand@120:r1]
    GEMMA_FAKE=1 python -m redshift.sim --gemma --seconds 200 --say "40:rover 2 explore north"

Tier 2 (advisor.py: Gemma advice, operator chat, clarification questions) is attached as
self.advisor; without a planner it is off and the rovers run on the rules alone.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import mujoco
import numpy as np

from .advisor import Advisor
from .config import ROVER_COLORS, Config
from .fleet import Fleet
from .hazards import HazardError, HazardManager
from .lifeforms import LifeformManager
from .mapping import FleetMap
from .rover import Rover
from .terrain import Terrain
from .world import build_world, rover_spawn_height


def dock_spots(cfg: Config, n: int) -> list:
    """Parking spots on the lander pad, spread on a circle so charging rovers never touch."""
    if n <= 0:
        return []
    radius = 1.8 if n <= 4 else min(cfg.lander_radius - 0.2, max(1.8, 0.95 / math.sin(math.pi / n)))
    return [(round(radius * math.cos(2 * math.pi * k / n), 3), round(radius * math.sin(2 * math.pi * k / n), 3))
            for k in range(n)]


class Simulation:
    def __init__(self, cfg: Config | None = None, planner=None, frame_source=None, wait_for_gemma: bool = False):
        """planner: a gemma.GemmaPlanner for tier 2 (None: rules only, chat through the keyword
        parser). frame_source(rover_id) -> JPEG bytes, for Gemma's anomaly checks (worker thread)."""
        self.cfg = cfg if cfg is not None else Config.from_env()
        cfg = self.cfg
        self.terrain = Terrain(cfg)

        n = cfg.num_rovers
        docks = dock_spots(cfg, n)
        spawns = []
        for k in range(n):
            dx, dy = docks[k]
            spawns.append((f"r{k + 1}", dx, dy, math.atan2(dy, dx), ROVER_COLORS[k % len(ROVER_COLORS)]))
        self.world = build_world(cfg, self.terrain, spawns)
        self.model = self.world.model
        self.data = self.world.data
        self.target_gid_array = np.array(sorted(self.world.target_geom_ids), dtype=np.int32)

        self.map = FleetMap(cfg)
        self.fleet = Fleet(cfg, self.map, self.terrain.targets)
        self.hazards = None
        self.rovers = [Rover(self, k, rid, f"Rover {k + 1}", color, docks[k])
                       for k, (rid, _x, _y, _yaw, color) in enumerate(spawns)]
        self.fleet.rovers = self.rovers
        self.hazards = HazardManager(self)
        self.fleet.hazards = self.hazards
        self.lifeforms = LifeformManager(self)
        self.camera_ok = None           # rules' camera checks: None = not tried yet, False = no OpenGL here
        self._features_sent = -1
        self.advisor = Advisor(self, planner, frame_source, wait_for_gemma=wait_for_gemma)

        # per rover motor energy bookkeeping (actuator ids and wheel dof ids, shape n x 4)
        self._act = np.array([r.acts for r in self.rovers], dtype=np.int64).reshape(n, 4)
        self._dof = np.array([r.wdofs for r in self.rovers], dtype=np.int64).reshape(n, 4)
        self.energy = np.zeros(n)
        self.steps_per_tick = max(1, int(round(1.0 / (cfg.agent_hz * self.model.opt.timestep))))
        self.agent_dt = self.steps_per_tick * self.model.opt.timestep

        self.ticks = 0
        self.prof = {"physics": 0.0, "agents": 0.0}
        self.glitches = 0               # physics blow ups caught and undone (should stay 0)
        self._last_glitch_t = -1e9
        self._snap_qpos = self.data.qpos.copy()
        self._half = (cfg.nx - 1) * cfg.res / 2.0
        self._event_cursor = 0          # _seq of the last event handed out in a tick message
        self._truth_json = None
        self._truth_version = -1
        self.flips = 0

        # rovers start on the pad: one tick so the map has the lander area in it from the start
        for r in self.rovers:
            r.read_pose()
            r.scan(0.0)
        self.fleet.update(0.0)

    # ------------------------------------------------------------------ stepping
    @property
    def t(self) -> float:
        return float(self.data.time)

    def step(self) -> None:
        """Advance one agent period: physics substeps, then fleet update and rover decisions."""
        m, d = self.model, self.data
        dt = m.opt.timestep
        k_heat = self.cfg.battery_heat_k
        act, dof, energy = self._act, self._dof, self.energy
        t_start = float(d.time)
        self._snap_qpos[:] = d.qpos
        hz = self.hazards
        c0 = time.perf_counter()
        hz.update_physics(t_start)
        self.lifeforms.step(t_start, self.agent_dt)
        for _ in range(self.steps_per_tick):
            if hz.drops or hz.falls:
                hz.animate(float(d.time))
            mujoco.mj_step(m, d)
            f = d.actuator_force[act]
            energy += (np.abs(f * d.qvel[dof]) + k_heat * f * f).sum(axis=1) * dt
        # MuJoCo resets the whole state if it ever goes unstable: undo that and carry on
        if d.time < t_start + 0.5 * self.agent_dt or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            self._recover_physics(t_start)
        self._check_rovers_in_world()
        t = float(d.time)
        hz.step(t)
        c1 = time.perf_counter()
        n = len(self.rovers)
        s0 = self.ticks % n
        for k in range(n):
            r = self.rovers[(s0 + k) % n]
            r.tick(t, self.agent_dt, float(energy[r.idx]), duty=(k == 0))
            energy[r.idx] = 0.0
        self.advisor.update(t)
        self.ticks += 1
        c2 = time.perf_counter()
        self.prof["physics"] += c1 - c0
        self.prof["agents"] += c2 - c1

    def _recover_physics(self, t_start: float) -> None:
        m, d = self.model, self.data
        repeat = t_start - self._last_glitch_t < 3.0
        d.qpos[:] = self._snap_qpos
        d.qvel[:] = 0.0
        d.time = t_start + self.agent_dt
        mujoco.mj_forward(m, d)
        self.energy[:] = 0.0
        self.glitches += 1
        self._last_glitch_t = t_start
        print(f"[sim] physics went unstable near t={t_start:.1f} s; restored the last good state", file=sys.stderr)
        if repeat:   # it blew up again straight away: put every rover back on its wheels
            for r in self.rovers:
                r.read_pose()
                self.reset_rover_upright(r)

    def _check_rovers_in_world(self) -> None:
        """Safety net: a rover that left the map or sank into the ground is set back on its wheels."""
        q = self.data.qpos
        lim = self._half + 0.3
        for r in self.rovers:
            x, y, z = float(q[r.qadr]), float(q[r.qadr + 1]), float(q[r.qadr + 2])
            bad = not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z))
            if not bad:
                bad = abs(x) > lim or abs(y) > lim or z < self.terrain.height_at(x, y) - 1.0
            if bad:
                r.read_pose()
                self.reset_rover_upright(r)
                r.read_pose()
                self.fleet.log(self.t, "warn", f"{r.id} got stuck in the terrain. Self-righting.", r.id)

    def run_for(self, seconds: float) -> None:
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.step()

    def reset_rover_upright(self, rover) -> None:
        """Self-righting: put the rover back on its wheels at the flattest spot within a metre."""
        m, d, ter = self.model, self.data, self.terrain
        half = self._half - 2.0
        x0 = rover.x if math.isfinite(rover.x) else 0.0
        y0 = rover.y if math.isfinite(rover.y) else 0.0
        x0, y0 = min(max(x0, -half), half), min(max(y0, -half), half)
        yaw = rover.yaw if math.isfinite(rover.yaw) else 0.0
        best = (x0, y0)
        best_s = 1e9
        for dx in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for dy in (-1.0, -0.5, 0.0, 0.5, 1.0):
                x, y = x0 + dx, y0 + dy
                s = ter.max_slope_near(x, y, 0.6) + 2.0 * math.hypot(dx, dy)
                if s < best_s:
                    best_s, best = s, (x, y)
        x = min(max(best[0], -half), half)
        y = min(max(best[1], -half), half)
        z = ter.height_at(x, y) + rover_spawn_height(self.cfg) + 0.12
        a, b = rover.qadr, rover.vadr
        d.qpos[a:a + 3] = (x, y, z)
        d.qpos[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qpos[a + 7:a + rover.nq] = 0.0        # axle and wheel angles
        d.qvel[b:b + rover.nv] = 0.0
        mujoco.mj_forward(m, d)
        self.flips += 1

    # ------------------------------------------------------------------ protocol messages
    def _new_events(self) -> list:
        out = []
        cur = self._event_cursor
        for e in self.fleet.events:
            if e["_seq"] > cur:
                out.append(Fleet.public_event(e))
        if self.fleet.events:
            self._event_cursor = max(cur, self.fleet.events[-1]["_seq"])
        return out

    def init_message(self) -> dict:
        """Full state for a newly connected client. Events stop at the tick cursor so that the
        next tick message carries exactly the events this message does not."""
        cfg = self.cfg
        truth = self.terrain.truth_nospire          # colour scale: terrain and rocks, not the tall spires
        evs = [Fleet.public_event(e) for e in self.fleet.events if e["_seq"] <= self._event_cursor][-50:]
        self._features_sent = self.hazards.features_version
        return {
            "type": "init",
            "world": {"name": cfg.world_name, "gravity": round(cfg.gravity, 2),
                      "gravity_g": float(cfg.gravity_g), "seed": int(cfg.seed)},
            "map": {"nx": cfg.nx, "ny": cfg.ny, "res": cfg.res,
                    "origin": [float(cfg.origin[0]), float(cfg.origin[1])],
                    "min_h": round(float(truth.min()), 2), "max_h": round(float(truth.max()), 2)},
            "features": self.features_msg(),
            "lifeforms": self.lifeforms.to_msgs(),
            "tracks": self.fleet.tracker.to_msgs(),
            "lander": {"x": 0.0, "y": 0.0, "radius": float(cfg.lander_radius)},
            "rovers": [{"id": r.id, "name": r.name, "color": r.color} for r in self.rovers],
            "known": {"cells": self.map.all_cells()},
            "targets": self.fleet.discovered_targets(),
            "events": evs,
            "fleet": self.fleet.stats(),
            "hazards": self.hazards.to_msgs(self.t),
            "planner": self.advisor.planner_msg(),
            "chat": self.advisor.init_chat(),
            "requests": self.advisor.requests_msg(),
        }

    def tick_message(self) -> dict:
        """Deltas since the previous tick message (cells and events), plus full rover, frontier,
        target and fleet state. Each call consumes the deltas, so build it once per broadcast. An updated
        `features` object is included only in the tick after it changed (a spire fell)."""
        msg = {
            "type": "tick",
            "t": round(self.t, 1),
            "rovers": [r.to_msg() for r in self.rovers],
            "cells": self.map.take_changed_cells(),
            "frontiers": [[round(float(x), 2), round(float(y), 2), int(size)]
                          for (x, y, size, _idx) in self.fleet.frontiers],
            "targets": self.fleet.discovered_targets(),
            "events": self._new_events(),
            "fleet": self.fleet.stats(),
            "hazards": self.hazards.to_msgs(self.t),
            "planner": self.advisor.planner_msg(),
            "chat": self.advisor.new_chat(),
            "requests": self.advisor.requests_msg(),
            "lifeforms": self.lifeforms.to_msgs(),
            "tracks": self.fleet.tracker.to_msgs(),
        }
        if self._features_sent != self.hazards.features_version:
            self._features_sent = self.hazards.features_version
            msg["features"] = self.features_msg()
        return msg

    def features_msg(self) -> dict:
        """Ground truth of the static features (protocol v4): water outlines and spires."""
        standing = self.hazards.spire_standing
        return {"water": [w.to_msg() for w in self.terrain.water],
                "spires": [{"id": sp.id, "x": round(sp.x, 2), "y": round(sp.y, 2), "height": round(sp.height, 2),
                            "radius": round(sp.radius, 2), "standing": bool(standing.get(sp.id, True))}
                           for sp in self.hazards.all_spires()]}

    def close(self) -> None:
        """Stop tier 2's worker thread (the shared Gemma planner itself keeps running)."""
        self.advisor.close()

    def truth_heights(self) -> np.ndarray:
        """True heights: terrain, rocks and any rockfall boulders."""
        return self.hazards.truth_heights(self.terrain.truth_heights)

    def truth_payload(self) -> dict:
        cfg = self.cfg
        return {
            "nx": cfg.nx, "ny": cfg.ny, "res": cfg.res,
            "origin": [float(cfg.origin[0]), float(cfg.origin[1])],
            "heights": np.round(self.truth_heights().astype(np.float64), 2).ravel().tolist(),
            "targets": [{"id": t.id, "x": round(t.x, 2), "y": round(t.y, 2), "kind": t.kind}
                        for t in self.terrain.targets],
        }

    def truth_json(self) -> str:
        if self._truth_json is None or self._truth_version != self.hazards.version:
            self._truth_version = self.hazards.version
            self._truth_json = json.dumps(self.truth_payload(), separators=(",", ":"))
        return self._truth_json

    # ------------------------------------------------------------------ summaries
    def summary(self) -> dict:
        f = self.fleet
        return {
            "t": round(self.t, 1),
            "explored_pct": round(self.map.explored_pct(), 1),
            "slope_limit_deg": round(f.slope_limit, 1),
            "best_climb_deg": round(f.best_climb, 1),
            "targets_total": len(f.targets),
            "targets_found": sum(1 for ts in f.targets if ts.status is not None),
            "targets_sampled": sum(1 for ts in f.targets if ts.status == "sampled"),
            "samples_cached": f.samples_cached,
            "science_score": f.science_score,
            "water_zones": len(f.water_zones),
            "lifeforms_seen": f.lifeforms_seen,
            "events": f.event_seq,
            "flips": self.flips,
            "complete": f.complete,
            "rovers": [{"id": r.id, "state": r.state, "battery": round(r.battery, 2),
                        "odometer_m": round(r.odometer, 1)} for r in self.rovers],
        }

    def brain_stats(self) -> dict:
        """Per rover brain time over the whole run (ms): avg, p95, max, cycles over budget."""
        out = {}
        for r in self.rovers:
            a = np.array(r.brain_all) if r.brain_all else np.zeros(1)
            out[r.id] = {"avg": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max()),
                         "over": int((a > self.cfg.brain_budget_ms).sum()), "cycles": len(r.brain_all)}
        return out


def parse_hazard(spec: str):
    """'rockfall@60', 'sand@120:r1', 'spire_fall@150:12,-30' -> (type, sim time, rover id or None,
    point (x, y) or None)."""
    kind, _, rest = spec.partition("@")
    when, _, who = rest.partition(":")
    who = who.strip()
    point = None
    if "," in who:
        x, y = who.split(",", 1)
        point, who = (float(x), float(y)), ""
    return kind.strip(), float(when or 0.0), (who or None), point


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Redshift simulation headless, as fast as possible.")
    ap.add_argument("--seconds", type=float, default=300.0, help="sim seconds to run (default 300)")
    ap.add_argument("--seed", type=int, default=None, help="terrain seed (default: SEED env var or 42)")
    ap.add_argument("--rovers", type=int, default=None, help="number of rovers (default: NUM_ROVERS or 3)")
    ap.add_argument("--report-every", type=float, default=60.0, help="progress line every N sim seconds (0 = off)")
    ap.add_argument("--quiet", action="store_true", help="do not print the event log")
    ap.add_argument("--hazard", action="append", default=[], metavar="TYPE@T[:ROVER|:X,Y]",
                    help="trigger a hazard at sim time T, e.g. rockfall@60, sand@120:r1, dust_storm@200, "
                         "rover_fault@260:r3, lifeform@60, spire_fall@150, rockfall@90:12,-30 (a point) "
                         "(repeatable; retried for up to 60 s if it cannot be placed)")
    ap.add_argument("--ticks", action="store_true", help="build a tick message every 0.6 sim s and report its size")
    ap.add_argument("--model-clock", action="store_true",
                    help="cap planning by counted A* expansions instead of the wall clock (repeatable runs)")
    ap.add_argument("--budget-ms", type=float, default=None, help="brain budget per rover per cycle (default: "
                                                                  "BRAIN_BUDGET_MS or 40)")
    ap.add_argument("--gemma", action="store_true",
                    help="attach the Gemma planner from the GEMMA_* env vars (GEMMA_FAKE=1 for the fake model)")
    ap.add_argument("--say", action="append", default=[], metavar="T:TEXT",
                    help="operator chat message at sim time T, e.g. '40:rover 2 explore north' (repeatable)")
    args = ap.parse_args(argv)
    try:
        plan = sorted((parse_hazard(h) for h in args.hazard), key=lambda h: h[1])
    except ValueError:
        ap.error("--hazard must look like TYPE@SECONDS or TYPE@SECONDS:ROVER")

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.rovers is not None:
        overrides["num_rovers"] = max(1, min(8, args.rovers))
    if args.model_clock:
        overrides["plan_clock"] = "model"
    if args.budget_ms is not None:
        overrides["brain_budget_ms"] = max(1.0, args.budget_ms)
    cfg = Config.from_env(**overrides)
    try:
        says = sorted(((float(s.split(":", 1)[0]), s.split(":", 1)[1]) for s in args.say), key=lambda x: x[0])
    except (ValueError, IndexError):
        ap.error("--say must look like SECONDS:TEXT")
    planner = None
    if args.gemma:
        from .gemma import get_planner
        planner = get_planner()
        planner.wait_ready(5.0)
        print(f"Gemma planner: {planner.stats()['state']} ({planner.stats()['model']}, "
              f"{planner.stats()['backend']})")

    w0 = time.perf_counter()
    sim = Simulation(cfg, planner=planner)
    build_s = time.perf_counter() - w0
    print(f"Redshift sim: seed {cfg.seed}, {cfg.num_rovers} rovers, {cfg.gravity_g:g} g, map {cfg.map_size:g} m, "
          f"{len(sim.terrain.targets)} targets (built in {build_s:.2f} s)")
    w1 = time.perf_counter()
    next_report = args.report_every if args.report_every > 0 else float("inf")
    pending = [[k, when, rover, when, pt] for (k, when, rover, pt) in plan]    # [3]: next try
    sizes = []
    next_msg = 0.6
    try:
        while sim.t < args.seconds - 1e-9:
            sim.step()
            while says and sim.t >= says[0][0]:
                sim.advisor.operator_message(says.pop(0)[1])
            for hp in list(pending):
                if sim.t >= hp[3]:
                    try:
                        pt = hp[4]
                        h = sim.hazards.trigger(hp[0], hp[2], *(pt if pt is not None else (None, None)))
                        print(f"  t={sim.t:6.1f}s hazard {h['id']} {h['type']}"
                              f"{' on ' + h['rover'] if h.get('rover') else ''}", flush=True)
                        pending.remove(hp)
                    except HazardError as e:
                        if sim.t - hp[1] > 60.0:
                            print(f"  t={sim.t:6.1f}s hazard {hp[0]} gave up: {e}", flush=True)
                            pending.remove(hp)
                        else:
                            hp[3] = sim.t + 2.0
            if args.ticks and sim.t >= next_msg:
                next_msg += 0.6
                sizes.append(len(json.dumps(sim.tick_message(), separators=(",", ":"))))
            if sim.t >= next_report - 1e-9:
                s = sim.summary()
                wall = time.perf_counter() - w1
                states = " ".join(f"{r['id']}:{r['state']}/{r['battery'] * 100:.0f}%" for r in s["rovers"])
                print(f"  t={s['t']:6.0f}s explored {s['explored_pct']:5.1f}%  limit {s['slope_limit_deg']:4.1f} deg  "
                      f"targets {s['targets_found']}/{s['targets_total']} (sampled {s['targets_sampled']})  "
                      f"{states}  [{sim.t / max(wall, 1e-9):.1f}x]", flush=True)
                next_report += args.report_every
    except KeyboardInterrupt:
        print("interrupted")
    wall = time.perf_counter() - w1
    s = sim.summary()
    print("")
    print("Summary")
    print(f"  sim time            {s['t']:.0f} s")
    print(f"  explored            {s['explored_pct']:.1f}% of the map")
    print(f"  slope limit         {s['slope_limit_deg']:.1f} deg (started at {cfg.slope_limit_initial_deg:.0f}, "
          f"steepest climb seen {s['best_climb_deg']:.1f})")
    print(f"  targets             {s['targets_found']} found, {s['targets_sampled']} sampled, of {s['targets_total']}")
    print(f"  samples cached      {s['samples_cached']}, science score {s['science_score']}")
    print(f"  water zones         {s['water_zones']} (lakes and rivers in the world: {len(sim.terrain.water)})")
    print(f"  lifeforms seen      {s['lifeforms_seen']} (creatures in the world: {len(sim.lifeforms.creatures)})")
    print(f"  events              {s['events']}")
    print(f"  self-rights         {s['flips']}")
    print(f"  physics glitches    {sim.glitches}")
    print(f"  exploration done    {'yes' if s['complete'] else 'no'}")
    for r in s["rovers"]:
        print(f"  {r['id']}                  {r['state']}, battery {r['battery'] * 100:.0f}%, "
              f"travelled {r['odometer_m']:.0f} m")
    p = sim.prof
    print(f"  speed               {sim.t / max(wall, 1e-9):.1f} sim s per wall s "
          f"(physics {p['physics']:.1f} s, agents {p['agents']:.1f} s, wall {wall:.1f} s)")
    print(f"  planner             {sim.fleet.plan_calls} A* searches, {sim.fleet.plan_slices} slices, "
          f"{sim.fleet.plan_expansions} expansions")
    print(f"  map changes         {sim.map.changes} samples overwritten by change detection")
    print(f"  brain budget        {cfg.brain_budget_ms:.0f} ms per rover per {1000.0 / cfg.agent_hz:.0f} ms cycle")
    for rid, b in sim.brain_stats().items():
        print(f"  brain {rid}            avg {b['avg']:.2f} ms, p95 {b['p95']:.2f} ms, max {b['max']:.1f} ms, "
              f"over budget {b['over']} of {b['cycles']} cycles")
    if sizes:
        a = np.array(sizes)
        print(f"  tick size           median {np.median(a) / 1024:.1f} KB, p95 {np.percentile(a, 95) / 1024:.1f} KB, "
              f"max {a.max() / 1024:.1f} KB ({len(a)} ticks)")
    pm = sim.advisor.planner_msg()
    print(f"  gemma               {pm['state']}, {pm['calls']} calls, {pm['ok']} ok, {pm['fallbacks']} fallbacks, "
          f"avg {pm['avg_ms']:.0f} ms, p95 {pm['p95_ms']:.0f} ms (budget {pm['budget_ms']:.0f} ms)")
    if not args.quiet:
        print("")
        print("Event log")
        for e in sim.fleet.events:
            who = e["rover"] or "fleet"
            at = f"  @({e['at'][0]:.0f}, {e['at'][1]:.0f})" if "at" in e else ""
            src = f" [{e['source']}]" if "source" in e else ""
            print(f"  [{e['t']:7.1f}] {e['level']:9s} {who:5s} {e['msg']}{at}{src}")
        if sim.advisor.chat:
            print("")
            print("Chat")
            for m in sim.advisor.chat:
                src = f" [{m['source']}]" if m["source"] else ""
                print(f"  [{m['t']:7.1f}] {m['from']:8s} {m['kind']:8s} {m['text']}{src}")
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
