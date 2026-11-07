"""Hazards, sensors, edge budget and the v2 protocol additions.

One scripted run (cached, model clock so it repeats exactly) triggers every hazard in turn and
records what the rovers did; each test checks one part. A few small extra sims cover charging in a
dust storm, the anomaly hook, anytime planning and the /api/hazard endpoint.

    python -m pytest -q tests/test_hazards.py        (about 30 s)
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift.config import Config  # noqa: E402
from redshift.hazards import HazardError  # noqa: E402
from redshift.sim import Simulation  # noqa: E402

STORM_S = 40.0
FAULT_S = 40.0
SCRIPT = [("rockfall", 40.0, None), ("sand", 85.0, None), ("dust_storm", 150.0, None), ("rover_fault", 200.0, "r3")]
END_T = 262.0
EM_DASH = chr(0x2014)

_R = None


class Run:
    pass


def run() -> Run:
    global _R
    if _R is not None:
        return _R
    R = Run()
    cfg = Config(seed=42, plan_clock="model", dust_storm_s=STORM_S, fault_s=FAULT_S)
    sim = Simulation(cfg)
    R.sim = sim
    R.hz = {}            # type -> hazard message at trigger
    R.trig_t = {}
    R.ticks = [sim.init_message()]
    R.contacts = 0       # rover touching a rockfall boulder (should never happen)
    R.storm = []         # (t, rover id, lidar_range, cmd_v, forward ground speed, degraded, state)
    R.fault = []         # (t, state, goal, target, odometers)
    R.odo = []
    boulder_gids = {g for (_, g, _) in sim.world.boulders}
    pending = list(SCRIPT)
    next_msg = 0.6
    while sim.t < END_T:
        sim.step()
        t = sim.t
        for item in list(pending):
            kind, when, rid = item
            if t >= when:
                try:
                    R.hz[kind] = sim.hazards.trigger(kind, rid)
                    R.trig_t[kind] = t
                    pending.remove(item)
                except HazardError:
                    pass         # no route to place it on right now: try again next cycle
        d = sim.data
        for c in d.contact[:d.ncon]:
            if c.geom1 in boulder_gids or c.geom2 in boulder_gids:
                R.contacts += 1
        if "dust_storm" in R.trig_t:
            for r in sim.rovers:
                R.storm.append((t, r.id, r.lidar_eff, r.cmd_v, abs(r.speed_fwd), r.degraded, r.state))
        if "rover_fault" in R.trig_t:
            r3 = sim.rovers[2]
            R.fault.append((t, r3.state, r3.goal, r3.target_idx, [r.odometer for r in sim.rovers]))
        R.odo.append((t, [r.odometer for r in sim.rovers]))
        if t >= next_msg:
            next_msg += 0.6
            R.ticks.append(sim.tick_message())
    R.ticks.append(sim.init_message())
    R.events = list(sim.fleet.events)
    _R = R
    return R


def _events(R, pred):
    return [e for e in R.events if pred(e)]


def _odo_at(R, t):
    for tt, o in R.odo:
        if tt >= t:
            return o
    return R.odo[-1][1]


# ---------------------------------------------------------------------- the script ran
def test_all_hazards_triggered():
    R = run()
    assert set(R.hz) == {"rockfall", "sand", "dust_storm", "rover_fault"}, R.hz.keys()
    for kind, when, _ in SCRIPT:
        assert R.trig_t[kind] - when < 40.0, (kind, R.trig_t[kind])


# ---------------------------------------------------------------------- rockfall
def test_rockfall_replan_and_no_collision():
    R = run()
    h = R.hz["rockfall"]
    t0 = R.trig_t["rockfall"]
    assert 3 <= len(h["rocks"]) <= 5 and h["rover"] in {"r1", "r2", "r3"}
    assert _events(R, lambda e: e["level"] == "hazard" and e["msg"].startswith("Rockfall") and "at" in e)
    seen = _events(R, lambda e: e["rover"] == h["rover"] and "sees new rocks" in e["msg"] and t0 <= e["t"] <= t0 + 15)
    assert seen, "target rover never noticed the boulders"
    rx, ry = seen[0]["at"]
    assert min(math.hypot(rx - b[0], ry - b[1]) for b in h["rocks"]) < 3.0, seen[0]
    assert _events(R, lambda e: e["rover"] == h["rover"] and ("rerouting around the new rocks" in e["msg"] or
                                                              "no way past the new rocks" in e["msg"])
                   and e["t"] >= seen[0]["t"]), "no re-plan after seeing the rocks"
    assert R.contacts == 0, f"{R.contacts} rover-boulder contacts"


def test_rockfall_in_map_and_truth():
    R = run()
    sim, h = R.sim, R.hz["rockfall"]
    fm, ter = sim.map, sim.terrain
    truth = sim.truth_heights()
    shown = 0
    for (x, y, z, rad) in h["rocks"]:
        i, j = fm.ij(x, y)
        k = max(1, int(rad / fm.res))
        win = (slice(max(j - k, 0), j + k + 1), slice(max(i - k, 0), i + k + 1))
        ground = float(ter.heights[j, i])
        assert truth[j, i] > ground + 0.3, "boulder missing from /api/truth heights"
        if fm.known[win].any() and float(fm.h[win][fm.known[win]].max()) > ground + 0.3 \
                and float(fm.slope[win].max()) > sim.fleet.slope_limit:
            shown += 1
    assert shown >= 2, f"only {shown} boulders visible in the shared map"
    assert fm.changes > 0
    p = sim.truth_payload()
    assert len(p["heights"]) == sim.cfg.nx * sim.cfg.ny


# ---------------------------------------------------------------------- sand
def test_sand_detected_and_escaped():
    R = run()
    h = R.hz["sand"]
    t0 = R.trig_t["sand"]
    rid = h["rover"]
    det = _events(R, lambda e: e["rover"] == rid and "wheels spinning but not moving" in e["msg"] and e["t"] >= t0)
    assert det and det[0]["t"] - t0 < 30.0, "slip never revealed the sand"
    assert "slip" in det[0]["msg"] and "Soft ground" in det[0]["msg"]
    zones = R.sim.fleet.soft_zones
    assert zones and min(math.hypot(z[0] - h["x"], z[1] - h["y"]) for z in zones) < 2.0, zones
    assert [p for p in R.sim.hazards.patches if p.id == h["id"]][0].known
    out = _events(R, lambda e: e["rover"] == rid and ("backed out of the soft ground" in e["msg"] or
                                                      "dug itself out" in e["msg"]) and e["t"] >= det[0]["t"])
    assert out and out[0]["t"] - det[0]["t"] < 45.0, "rover never got out of the sand"
    k = int(rid[1:]) - 1
    t_out = out[0]["t"]
    assert _odo_at(R, t_out + 30.0)[k] - _odo_at(R, t_out)[k] > 3.0, "rover did not drive on after the sand"
    # the soft layer is impassable for everyone
    fm = R.sim.map
    zi, zj = fm.ij(zones[0][0], zones[0][1])
    assert fm.soft[zj, zi] and not R.sim.map.traversable(90.0)[zj, zi]
    assert math.isinf(R.sim.fleet.cost(R.sim.t)[zj, zi])


# ---------------------------------------------------------------------- dust storm
def test_dust_storm_slows_and_recovers():
    R = run()
    t0 = R.trig_t["dust_storm"]
    cfg = R.sim.cfg
    down = _events(R, lambda e: "LiDAR returns down to 3 m" in e["msg"] and t0 <= e["t"] <= t0 + 3.0)
    assert {e["rover"] for e in down} == {"r1", "r2", "r3"}, down
    mid = [s for s in R.storm if t0 + 4.0 <= s[0] <= t0 + STORM_S - 1.0 and s[6] != "fault"]
    assert mid
    assert all(s[5] for s in mid), "a rover forgot its LiDAR was degraded"
    assert max(s[2] for s in mid) <= 3.5
    assert max(s[3] for s in mid) <= cfg.degraded_v_max + 1e-6, "commanded faster than the storm limit"
    assert max(s[4] for s in mid) <= 0.55, "drove too fast in the storm"
    back = _events(R, lambda e: "visibility back to 8 m" in e["msg"] and e["t"] >= t0 + STORM_S)
    assert {e["rover"] for e in back} == {"r1", "r2", "r3"}
    after = [s for s in R.storm if s[0] >= t0 + STORM_S + 3.0]
    assert after and all((not s[5]) and s[2] >= 7.5 for s in after)
    assert _events(R, lambda e: e["level"] == "hazard" and e["msg"] == "Dust storm rolling in.")
    # one event per change, not spam
    assert len(_events(R, lambda e: "LiDAR returns down" in e["msg"])) == 3


# ---------------------------------------------------------------------- rover fault
def test_fault_releases_claims_and_reboots():
    R = run()
    t0 = R.trig_t["rover_fault"]
    f = [x for x in R.fault if t0 + 0.5 <= x[0] <= t0 + FAULT_S - 0.5]
    assert f and all(x[1] == "fault" and x[2] is None and x[3] is None for x in f), f[:3]
    msg = _events(R, lambda e: e["rover"] == "r3" and "motor controller not responding" in e["msg"])
    assert msg and msg[0]["t"] - t0 < 1.0
    o0, o1 = f[0][4], f[-1][4]
    assert o1[2] - o0[2] < 0.3, "faulted rover moved"
    assert o1[0] - o0[0] > 5.0 or o1[1] - o0[1] > 5.0, "the others stopped exploring"
    up = _events(R, lambda e: e["rover"] == "r3" and e["msg"] == "r3 rebooted. Back online.")
    assert up and abs(up[0]["t"] - (t0 + FAULT_S)) < 1.0
    assert R.sim.rovers[2].state != "fault" and not R.sim.rovers[2].motor_fault


# ---------------------------------------------------------------------- protocol v2
def test_v2_protocol_fields():
    R = run()
    kinds = set()
    for m in R.ticks:
        text = json.dumps(m, allow_nan=False)
        assert EM_DASH not in text
        assert "hazards" in m and isinstance(m["hazards"], list)
        f = m["fleet"]
        assert {"budget_ms", "cycle_ms", "soft_zones"} <= set(f)
        assert all(len(z) == 3 for z in f["soft_zones"])
        for h in m["hazards"]:
            kinds.add(h["type"])
            assert {"id", "type", "t", "rover", "active"} <= set(h), h
            if h["type"] == "rockfall":
                assert {"x", "y", "radius", "rocks"} <= set(h) and all(len(r) == 4 for r in h["rocks"])
            elif h["type"] == "sand":
                assert {"x", "y", "radius", "known"} <= set(h)
            else:
                assert "remaining" in h and h["remaining"] >= 0.0
                if not h["active"]:
                    assert h["remaining"] == 0.0
        if m["type"] != "tick":
            continue
        for r in m["rovers"]:
            assert {"sensors", "scan", "brain"} <= set(r)
            assert r["state"] in {"exploring", "investigating", "returning", "charging", "recovering", "idle",
                                  "fault"}
            assert len(r["scan"]) <= 900 and len(r["scan"]) % 3 == 0
        for e in m["events"]:
            if e["level"] == "hazard":
                assert e["rover"] is None
    assert kinds == {"rockfall", "sand", "dust_storm", "rover_fault"}
    states = {r["state"] for m in R.ticks if m["type"] == "tick" for r in m["rovers"]}
    assert "fault" in states and "recovering" in states
    # at least one tick shows the sand as known and a soft zone in the fleet
    assert any(h["type"] == "sand" and h["known"] for m in R.ticks for h in m["hazards"])
    assert any(m["fleet"]["soft_zones"] for m in R.ticks)


def test_events_say_where_and_why():
    R = run()
    for e in R.events:
        assert EM_DASH not in e["msg"]
        if e["rover"] is not None:
            assert e["msg"].startswith(e["rover"] + " "), e
    for key in ("sees new rocks", "wheels spinning", "motor controller not responding", "Rockfall", "Soft sand"):
        hits = [e for e in R.events if key in e["msg"]]
        assert hits and all("at" in e for e in hits), key
    assert any("slip" in e["msg"] and "%" in e["msg"] for e in R.events)


def test_anomalies_recorded():
    R = run()
    kinds = {a["kind"] for r in R.sim.rovers for a in r.anomaly_log}
    assert {"new_obstacle", "slip", "low_visibility"} <= kinds, kinds


def test_brain_within_budget_with_hazards():
    R = run()
    for r in R.sim.rovers:
        a = sorted(r.brain_all)
        assert a[int(0.95 * (len(a) - 1))] < R.sim.cfg.brain_budget_ms, r.id


# ---------------------------------------------------------------------- small extra sims
def test_dusty_panels_slow_charging():
    sim = Simulation(Config(seed=7, plan_clock="model", dust_storm_s=12.0))
    r = sim.rovers[0]
    r.battery = 0.3
    r.state = "charging"
    sim.hazards.trigger("dust_storm")
    sim.run_for(5.0)
    assert r.slow_charge and abs(r.battery_margin - sim.cfg.slow_charge_margin) < 1e-9
    ev = [e for e in sim.fleet.events if e["rover"] == "r1" and "of normal (dust on the lander's panels)" in e["msg"]]
    assert ev and "Raising return margin to 25%" in ev[0]["msg"], ev
    assert "charging at 30% of normal" in ev[0]["msg"]
    sim.run_for(12.0)
    assert not r.slow_charge and r.battery_margin == sim.cfg.battery_margin


def test_anomaly_hook_and_caution():
    sim = Simulation(Config(seed=42, plan_clock="model"))
    sim.run_for(10.0)
    r = sim.rovers[1]
    got = []
    r.anomaly_hooks.append(lambda rover, a: got.append((rover.id, a)))
    r.anomaly_hooks.append(lambda rover, a: 1 / 0)      # a broken add-on must not stop the rover
    a = r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
    assert got == [("r2", a)] and r.anomaly is a and a["kind"] == "new_obstacle" and a["dist"] == 4.0
    assert sim.fleet.events[-1]["msg"].startswith("r2 sees new rocks blocking its path 4 m ahead")
    sim.run_for(1.0)
    if r.state in ("exploring", "investigating", "returning"):
        assert r.cmd_v <= sim.cfg.caution_v + 1e-6


def test_anytime_planning_splits_but_keeps_exploring():
    sim = Simulation(Config(seed=42, plan_clock="model", brain_budget_ms=0.5))
    sim.run_for(60.0)
    f = sim.fleet
    assert f.plan_slices > f.plan_calls, (f.plan_slices, f.plan_calls)
    assert all(r.odometer > 10.0 for r in sim.rovers)
    assert sim.map.explored_pct() > 10.0


def test_hazard_endpoint():
    from redshift import server

    class Req:
        def __init__(self, body: bytes):
            self._b = body

        async def body(self):
            return self._b

    def call(body: bytes):
        out = asyncio.run(server.hazard(Req(body)))
        if hasattr(out, "status_code"):
            return out.status_code, json.loads(out.body)
        return 200, json.loads(json.dumps(out))

    old = server.state.runner
    server.state.runner = server.SimRunner(Config(seed=42, plan_clock="model"))   # not started: no thread
    try:
        code, body = call(b'{"type": "dust_storm"}')
        assert code == 200 and body["ok"] and body["hazard"]["type"] == "dust_storm" and body["hazard"]["active"]
        assert call(b'{"type": "dust_storm"}')[0] == 400                 # one storm at a time
        code, body = call(b'{"type": "meteor"}')
        assert code == 400 and body == {"ok": False, "error": body["error"]} and "unknown hazard type" in body["error"]
        assert call(b"not json")[0] == 400
        assert call(b'{"type": "sand", "rover": "r9"}')[0] == 400
        assert call(b'{"type": "rockfall"}')[0] == 400                   # nobody is driving yet at t=0
        assert call(b'{"type": "rover_fault", "rover": "r1"}')[0] == 200
        code, body = call(b'{"type": "rover_fault", "rover": "r1"}')
        assert code == 400 and "already faulted" in body["error"]
        assert call(b'{"type": "rover_fault"}')[0] == 200                 # picks a working rover
        code, body = call(b'{"type": "rover_fault"}')
        assert code == 400 and "no working rover" in body["error"]
        with server.state.runner.locked() as sim:
            msg = sim.tick_message()
        assert [h["type"] for h in msg["hazards"]] == ["dust_storm", "rover_fault", "rover_fault"]
    finally:
        server.state.runner = old


if __name__ == "__main__":
    failed = 0
    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_") and callable(fn)]
    for n, fn in tests:
        try:
            fn()
            print(f"PASS {n}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {n}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
