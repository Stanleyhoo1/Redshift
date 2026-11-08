"""Headless smoke test: about 400 sim seconds with the default world (seed 42, 120 m map).

Run with `python -m pytest -q tests/` (add -s to see the event log) or `python tests/test_smoke.py`.
The simulation runs once and every test checks a different part of the result.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift.config import Config  # noqa: E402
from redshift.fleet import Fleet  # noqa: E402
from redshift.sim import Simulation  # noqa: E402

SIM_SECONDS = 400.0
MSG_EVERY = 0.6        # sim seconds between tick messages (5 Hz at 3x real time)

# protocol v1 keys plus the "Version 2 additions" (hazards, sensors, edge budget), the "Version 3
# additions" (planner, chat, requests; checked in detail in test_brain.py) and the "Version 4 / 4b
# additions" (features, lifeforms, tracks, samples; checked in detail in test_v4.py)
INIT_KEYS = {"type", "world", "map", "lander", "rovers", "known", "targets", "events", "fleet", "hazards",
             "planner", "chat", "requests", "features", "lifeforms", "tracks"}
TICK_KEYS = {"type", "t", "rovers", "cells", "frontiers", "targets", "events", "fleet", "hazards",
             "planner", "chat", "requests", "lifeforms", "tracks"}          # plus "features" when it changed
ROVER_KEYS = {"id", "x", "y", "z", "yaw", "speed", "battery", "state", "goal", "path", "target_id",
              "sensors", "scan", "brain", "track_id", "samples", "capacity"}   # plus "sampling" while sampling
SENSOR_KEYS = {"lidar_range", "lidar_hits", "pitch", "roll", "wheel_speed", "ground_speed", "slip", "motor_load",
               "spectro"}
BRAIN_KEYS = {"last_ms", "avg_ms", "peak_ms", "over"}
TARGET_KEYS = {"id", "x", "y", "kind", "status", "found_by", "found_t", "instrument", "value", "analysis",
               "carried_by", "cached"}
EVENT_KEYS = {"t", "rover", "level", "msg"}           # plus optional "at": [x, y] and "source" (v3)
FLEET_KEYS = {"explored_pct", "slope_limit_deg", "slope_limit_initial_deg", "targets_found", "targets_total",
              "sim_speed", "budget_ms", "cycle_ms", "soft_zones", "water_zones", "lifeforms_seen", "samples_cached",
              "samples_carried", "science_score", "mode",           # mode: protocol v8 ("priority" only while set)
              "spares", "rescues", "wrecks"}                        # protocol v9
STATES = {"exploring", "investigating", "returning", "charging", "recovering", "idle", "fault", "observing",
          "disabled", "rescuing", "towing", "towed", "repairing", "wrecked"}             # protocol v9
LEVELS = {"info", "warn", "discovery", "hazard"}
EM_DASH = chr(0x2014)   # banned everywhere in this project

_RESULT = None


class Result:
    pass


def run() -> Result:
    """Run the simulation once (cached) and record everything the tests look at."""
    global _RESULT
    if _RESULT is not None:
        return _RESULT
    R = Result()
    sim = Simulation(Config(seed=42))
    R.sim = sim
    R.errors = []
    R.inits = [sim.init_message()]
    R.ticks = []
    R.explored = []
    R.max_tilt_run = {r.id: 0.0 for r in sim.rovers}
    tilt_since = {r.id: None for r in sim.rovers}
    R.nonfinite = []
    next_msg = MSG_EVERY
    next_init = 200.0
    while sim.t < SIM_SECONDS:
        sim.step()
        t = sim.t
        for r in sim.rovers:
            if not all(math.isfinite(v) for v in (r.x, r.y, r.z, r.yaw, r.battery)):
                R.nonfinite.append((t, r.id))
            if r.tilt > 70.0:
                tilt_since[r.id] = t if tilt_since[r.id] is None else tilt_since[r.id]
                R.max_tilt_run[r.id] = max(R.max_tilt_run[r.id], t - tilt_since[r.id])
            else:
                tilt_since[r.id] = None
        if t >= next_msg:
            next_msg += MSG_EVERY
            R.ticks.append(sim.tick_message())
        if t >= next_init:
            next_init = 1e9
            R.inits.append(sim.init_message())
        if int(t * 10) % 100 == 0:
            R.explored.append((t, sim.map.explored_pct()))
    R.ticks.append(sim.tick_message())
    R.inits.append(sim.init_message())
    R.explored.append((sim.t, sim.map.explored_pct()))
    _RESULT = R
    return R


# ---------------------------------------------------------------------- checks
def _check_cells(cells, nx, ny):
    for c in cells:
        assert isinstance(c, list) and len(c) == 4, c
        i, j, h, s = c
        assert isinstance(i, int) and isinstance(j, int), c
        assert 0 <= i < nx and 0 <= j < ny, c
        assert math.isfinite(h) and math.isfinite(s) and 0.0 <= s <= 90.0, c


def _check_events(events):
    for e in events:
        assert EVENT_KEYS <= set(e) <= EVENT_KEYS | {"at", "source", "decision", "damage"}, e
        assert e.get("source", "rules") in ("rules", "gemma", "operator"), e
        assert e["level"] in LEVELS, e
        assert isinstance(e["msg"], str) and e["msg"], e
        assert EM_DASH not in e["msg"], e
        if "at" in e:
            assert len(e["at"]) == 2 and all(math.isfinite(v) for v in e["at"]), e
        if e["level"] == "hazard":
            assert e["rover"] is None, e
        elif e["rover"] is not None:
            assert e["msg"].startswith(e["rover"] + " "), e


def _check_rover_v2(r):
    s = r["sensors"]
    assert set(s) == SENSOR_KEYS, set(s) ^ SENSOR_KEYS
    for k in SENSOR_KEYS - {"spectro"}:
        assert isinstance(s[k], (int, float)) and math.isfinite(s[k]), (k, s[k])
    assert 0.0 <= s["slip"] <= 1.0 and 0.0 <= s["motor_load"] <= 1.0 and s["lidar_range"] > 0
    assert s["spectro"] is None or (set(s["spectro"]) == {"id", "dist"} and isinstance(s["spectro"]["id"], str))
    scan = r["scan"]
    assert len(scan) % 3 == 0 and len(scan) <= 900, len(scan)
    assert all(isinstance(v, (int, float)) and round(v, 1) == v for v in scan[:60])
    assert set(r["brain"]) == BRAIN_KEYS and r["brain"]["over"] >= 0


def test_no_nan_poses():
    R = run()
    assert not R.nonfinite, R.nonfinite[:5]
    for r in R.sim.rovers:
        assert all(math.isfinite(v) for v in (r.x, r.y, r.z, r.yaw))
    assert R.sim.glitches == 0, "physics went unstable and had to be restored"


def test_rovers_travelled():
    R = run()
    for r in R.sim.rovers:
        assert r.odometer > 15.0, f"{r.id} only travelled {r.odometer:.1f} m"


def test_explored_and_growing():
    R = run()
    final = R.sim.map.explored_pct()
    assert final > 25.0, f"explored only {final:.1f}%"
    pcts = [p for _, p in R.explored]
    assert all(b >= a for a, b in zip(pcts, pcts[1:])), "explored percentage went down"
    mid = [p for t, p in R.explored if t <= SIM_SECONDS / 3]
    assert mid and final > mid[-1] + 5.0, "exploration stopped growing"


def test_no_permanent_flips():
    R = run()
    for rid, longest in R.max_tilt_run.items():
        assert longest < 10.0, f"{rid} stayed tipped over for {longest:.1f} s"
    for r in R.sim.rovers:
        assert r.tilt < 60.0, f"{r.id} ends the run tilted {r.tilt:.0f} deg"


def test_target_discovered():
    R = run()
    found = [ts for ts in R.sim.fleet.targets if ts.status is not None]
    assert found, "no science target discovered"


def test_messages_match_protocol():
    R = run()
    cfg = R.sim.cfg
    nx, ny = cfg.nx, cfg.ny
    for m in R.inits:
        json.dumps(m, allow_nan=False)
        assert set(m) == INIT_KEYS, set(m) ^ INIT_KEYS
        assert m["type"] == "init"
        assert set(m["world"]) == {"name", "gravity", "gravity_g", "seed"}
        assert set(m["map"]) == {"nx", "ny", "res", "origin", "min_h", "max_h"}
        assert m["map"]["nx"] == nx and m["map"]["ny"] == ny and len(m["map"]["origin"]) == 2
        assert set(m["lander"]) == {"x", "y", "radius"}
        assert [set(r) for r in m["rovers"]] == [{"id", "name", "color"}] * cfg.num_rovers
        assert set(m["known"]) == {"cells"}
        _check_cells(m["known"]["cells"], nx, ny)
        assert len(m["events"]) <= 50
        _check_events(m["events"])
        for t in m["targets"]:
            assert set(t) == TARGET_KEYS, t
        assert set(m["fleet"]) == FLEET_KEYS
        assert isinstance(m["hazards"], list)
    assert len(R.ticks) > 100
    for m in R.ticks:
        json.dumps(m, allow_nan=False)
        assert TICK_KEYS <= set(m) <= TICK_KEYS | {"features"}, set(m) ^ TICK_KEYS
        assert m["type"] == "tick" and math.isfinite(m["t"])
        assert len(m["rovers"]) == cfg.num_rovers
        for r in m["rovers"]:
            # stats: v5; focus: v7 (only while a rover concentrates on an area)
            assert ROVER_KEYS <= set(r) <= ROVER_KEYS | {"sampling", "stats", "focus", "health", "vision"}, \
                set(r) ^ ROVER_KEYS
            if "health" in r:                                                     # protocol v9
                hl = r["health"]
                assert set(hl) == {"hp", "status", "parts", "diagnosis", "last_hit"}, hl
                assert hl["status"] in ("ok", "damaged", "immobile", "wrecked") and 0 <= hl["hp"] <= 100
                assert set(hl["parts"]) == {"drive_l", "drive_r", "lidar", "camera", "power", "comms"}
            if "focus" in r:
                assert set(r["focus"]) == {"x", "y", "r", "reason", "label", "ends_in", "mode"}, r["focus"]   # mode: v8
                assert r["focus"]["reason"] in ("operator", "interest") and r["focus"]["ends_in"] >= 0.0
            assert r["state"] in STATES, r["state"]
            assert 0.0 <= r["battery"] <= 1.0
            assert r["goal"] is None or (len(r["goal"]) == 2 and all(math.isfinite(v) for v in r["goal"]))
            assert len(r["path"]) <= 30 and all(len(p) == 2 for p in r["path"])
            assert r["target_id"] is None or isinstance(r["target_id"], str)
            _check_rover_v2(r)
        _check_cells(m["cells"], nx, ny)
        for f in m["frontiers"]:
            assert len(f) == 3 and isinstance(f[2], int) and f[2] > 0, f
        for t in m["targets"]:
            assert set(t) == TARGET_KEYS, t
            assert t["status"] in ("found", "investigating", "sampled") and t["kind"] in ("mineral", "ice", "vent", "trace")
        _check_events(m["events"])
        assert set(m["fleet"]) == FLEET_KEYS
        f = m["fleet"]
        assert f["budget_ms"] == cfg.brain_budget_ms and f["cycle_ms"] == 200.0 and isinstance(f["soft_zones"], list)
        assert m["hazards"] == []          # nothing triggered in this run
        assert m["planner"]["state"] == "disabled" and m["chat"] == [] and m["requests"] == []   # rules only


def test_no_false_hazard_reactions():
    """Without hazards: no soft ground, no degraded LiDAR, no faults, no map change detections."""
    R = run()
    msgs = [e["msg"] for e in R.sim.fleet.events]
    assert not any("Soft ground" in m or "LiDAR returns down" in m or "new rocks" in m for m in msgs), msgs
    assert R.sim.fleet.soft_zones == []
    assert R.sim.map.changes <= 3, R.sim.map.changes


def test_brain_within_budget():
    R = run()
    for r in R.sim.rovers:
        a = sorted(r.brain_all)
        p95 = a[int(0.95 * (len(a) - 1))]
        assert p95 < R.sim.cfg.brain_budget_ms, (r.id, p95)


def test_deltas_rebuild_the_map():
    """A client that applies init + every tick ends up with what a late joiner gets in its init."""
    R = run()
    cells = {(c[0], c[1]): (c[2], c[3]) for c in R.inits[0]["known"]["cells"]}
    for m in R.ticks:
        for c in m["cells"]:
            cells[(c[0], c[1])] = (c[2], c[3])
    late = {(c[0], c[1]): (c[2], c[3]) for c in R.inits[-1]["known"]["cells"]}
    assert set(cells) == set(late), f"{len(set(cells) ^ set(late))} cells differ"
    worst_h = max(abs(cells[k][0] - late[k][0]) for k in late)
    worst_s = max(abs(cells[k][1] - late[k][1]) for k in late)
    assert worst_h <= 0.03 and worst_s <= 0.6, (worst_h, worst_s)


def test_events_delivered_once():
    R = run()
    streamed = list(R.inits[0]["events"])
    for m in R.ticks:
        streamed.extend(m["events"])
    everything = [Fleet.public_event(e) for e in R.sim.fleet.events]
    assert streamed[-len(everything):] == everything
    assert len(streamed) == R.sim.fleet.event_seq


def test_truth_payload():
    R = run()
    p = R.sim.truth_payload()
    cfg = R.sim.cfg
    assert set(p) == {"nx", "ny", "res", "origin", "heights", "targets"}
    assert len(p["heights"]) == cfg.nx * cfg.ny and all(math.isfinite(h) for h in p["heights"])
    lo, hi = (6, 9) if cfg.map_size <= 80 else (12, 24)     # more targets (and traces) on the 120 m map
    assert len(p["targets"]) == len(R.sim.terrain.targets) and lo <= len(p["targets"]) <= hi
    for t in p["targets"]:
        assert set(t) == {"id", "x", "y", "kind"}
    json.loads(R.sim.truth_json())


def test_no_em_dash_in_sources():
    for path in list((ROOT / "redshift").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        assert EM_DASH not in path.read_text(encoding="utf-8"), path


def test_zz_print_event_log():
    R = run()
    s = R.sim.summary()
    print("")
    print(f"t={s['t']:.0f} s  explored {s['explored_pct']:.1f}%  slope limit {s['slope_limit_deg']:.0f} deg  "
          f"targets found {s['targets_found']}/{s['targets_total']} sampled {s['targets_sampled']}  "
          f"events {s['events']}")
    for r in s["rovers"]:
        print(f"  {r['id']}: {r['state']}, battery {r['battery'] * 100:.0f}%, travelled {r['odometer_m']:.0f} m")
    print("Event log:")
    for e in R.sim.fleet.events:
        print(f"  [{e['t']:6.1f}] {e['level']:9s} {e['rover'] or 'fleet':5s} {e['msg']}")


if __name__ == "__main__":
    failed = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
