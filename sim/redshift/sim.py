"""Simulation: owns the terrain, the MuJoCo world, the shared map, the fleet and the rovers.

One call to step() advances physics by one agent period (1 / agent_hz sim seconds), then runs the
fleet planner and every rover's decision tick. The server calls it from a background thread; the
CLI below runs it flat out:

    python -m redshift.sim --seconds 300 [--seed N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import mujoco
import numpy as np

from .config import ROVER_COLORS, Config
from .fleet import Fleet
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
    def __init__(self, cfg: Config | None = None):
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
        self.rovers = [Rover(self, k, rid, f"Rover {k + 1}", color, docks[k])
                       for k, (rid, _x, _y, _yaw, color) in enumerate(spawns)]
        self.fleet.rovers = self.rovers

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
        c0 = time.perf_counter()
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(m, d)
            f = d.actuator_force[act]
            energy += (np.abs(f * d.qvel[dof]) + k_heat * f * f).sum(axis=1) * dt
        # MuJoCo resets the whole state if it ever goes unstable: undo that and carry on
        if d.time < t_start + 0.5 * self.agent_dt or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            self._recover_physics(t_start)
        self._check_rovers_in_world()
        c1 = time.perf_counter()
        t = float(d.time)
        self.fleet.update(t)
        for k, r in enumerate(self.rovers):
            r.tick(t, self.agent_dt, float(energy[k]))
            energy[k] = 0.0
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
        truth = self.terrain.truth_heights
        evs = [Fleet.public_event(e) for e in self.fleet.events if e["_seq"] <= self._event_cursor][-50:]
        return {
            "type": "init",
            "world": {"name": cfg.world_name, "gravity": round(cfg.gravity, 2),
                      "gravity_g": float(cfg.gravity_g), "seed": int(cfg.seed)},
            "map": {"nx": cfg.nx, "ny": cfg.ny, "res": cfg.res,
                    "origin": [float(cfg.origin[0]), float(cfg.origin[1])],
                    "min_h": round(float(truth.min()), 2), "max_h": round(float(truth.max()), 2)},
            "lander": {"x": 0.0, "y": 0.0, "radius": float(cfg.lander_radius)},
            "rovers": [{"id": r.id, "name": r.name, "color": r.color} for r in self.rovers],
            "known": {"cells": self.map.all_cells()},
            "targets": self.fleet.discovered_targets(),
            "events": evs,
            "fleet": self.fleet.stats(),
        }

    def tick_message(self) -> dict:
        """Deltas since the previous tick message (cells and events), plus full rover, frontier,
        target and fleet state. Each call consumes the deltas, so build it once per broadcast."""
        return {
            "type": "tick",
            "t": round(self.t, 1),
            "rovers": [r.to_msg() for r in self.rovers],
            "cells": self.map.take_changed_cells(),
            "frontiers": [[round(float(x), 2), round(float(y), 2), int(size)]
                          for (x, y, size, _idx) in self.fleet.frontiers],
            "targets": self.fleet.discovered_targets(),
            "events": self._new_events(),
            "fleet": self.fleet.stats(),
        }

    def truth_payload(self) -> dict:
        cfg = self.cfg
        return {
            "nx": cfg.nx, "ny": cfg.ny, "res": cfg.res,
            "origin": [float(cfg.origin[0]), float(cfg.origin[1])],
            "heights": np.round(self.terrain.truth_heights.astype(np.float64), 2).ravel().tolist(),
            "targets": [{"id": t.id, "x": round(t.x, 2), "y": round(t.y, 2), "kind": t.kind}
                        for t in self.terrain.targets],
        }

    def truth_json(self) -> str:
        if self._truth_json is None:
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
            "events": f.event_seq,
            "flips": self.flips,
            "complete": f.complete,
            "rovers": [{"id": r.id, "state": r.state, "battery": round(r.battery, 2),
                        "odometer_m": round(r.odometer, 1)} for r in self.rovers],
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Redshift simulation headless, as fast as possible.")
    ap.add_argument("--seconds", type=float, default=300.0, help="sim seconds to run (default 300)")
    ap.add_argument("--seed", type=int, default=None, help="terrain seed (default: SEED env var or 42)")
    ap.add_argument("--rovers", type=int, default=None, help="number of rovers (default: NUM_ROVERS or 3)")
    ap.add_argument("--report-every", type=float, default=60.0, help="progress line every N sim seconds (0 = off)")
    ap.add_argument("--quiet", action="store_true", help="do not print the event log")
    args = ap.parse_args(argv)

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.rovers is not None:
        overrides["num_rovers"] = max(1, min(8, args.rovers))
    cfg = Config.from_env(**overrides)

    w0 = time.perf_counter()
    sim = Simulation(cfg)
    build_s = time.perf_counter() - w0
    print(f"Redshift sim: seed {cfg.seed}, {cfg.num_rovers} rovers, {cfg.gravity_g:g} g, map {cfg.map_size:g} m, "
          f"{len(sim.terrain.targets)} targets (built in {build_s:.2f} s)")
    w1 = time.perf_counter()
    next_report = args.report_every if args.report_every > 0 else float("inf")
    try:
        while sim.t < args.seconds - 1e-9:
            sim.step()
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
    print(f"  planner             {sim.fleet.plan_calls} A* calls, {sim.fleet.plan_expansions} expansions")
    if not args.quiet:
        print("")
        print("Event log")
        for e in sim.fleet.events:
            who = e["rover"] or "fleet"
            print(f"  [{e['t']:7.1f}] {e['level']:9s} {who:5s} {e['msg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
