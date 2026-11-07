"""Protocol v4 / v4b: water, alien spires, lifeforms, point-and-click hazards and sample collection.

Each test builds a small scenario on the default 120 m map (seed 42 has a lake, 14 spires and three
grazer herds). Rovers are sometimes moved into place by hand (a test harness move, like carrying the
rover there); after that they only learn through their sensors.

    python -m pytest -q tests/test_v4.py        (a few minutes)
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift.config import Config  # noqa: E402
from redshift.gemma import FakeBackend, GemmaConfig, GemmaPlanner  # noqa: E402
from redshift.hazards import HazardError  # noqa: E402
from redshift.sim import Simulation  # noqa: E402
from redshift.terrain import Terrain  # noqa: E402
from redshift.world import rover_spawn_height  # noqa: E402

EM_DASH = chr(0x2014)


def make(seed=42, planner=None, **cfg) -> Simulation:
    return Simulation(Config(seed=seed, plan_clock="model", **cfg), planner=planner, wait_for_gemma=planner is not None)


def teleport(sim, r, x, y, yaw) -> None:
    """Test harness: carry a rover to (x, y) facing yaw, standing on its wheels."""
    d = sim.data
    a = r.qadr
    z = sim.terrain.height_at(x, y) + rover_spawn_height(sim.cfg) + 0.04
    d.qpos[a:a + 3] = (x, y, z)
    d.qpos[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    d.qpos[a + 7:a + r.nq] = 0.0
    d.qvel[r.vadr:r.vadr + r.nv] = 0.0
    mujoco.mj_forward(sim.model, d)
    r.read_pose()
    r.trail.clear()
    r.last_xy = (r.x, r.y)
    r.goal, r.path = None, []
    r.home_dist = None


def fake_planner() -> GemmaPlanner:
    p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0), backend=FakeBackend(0.0))
    assert p.wait_ready(2.0)
    return p


def lake_approach(sim, gap=8.0):
    """A dry, gentle spot about `gap` m from the lake's edge facing it, and a goal on the far side."""
    ter = sim.terrain
    lake = next(w for w in ter.water if w.kind == "lake")
    for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        sx, sy = lake.cx + (lake.radius + gap) * math.cos(a), lake.cy + (lake.radius + gap) * math.sin(a)
        gx, gy = lake.cx - (lake.radius + 6.0) * math.cos(a), lake.cy - (lake.radius + 6.0) * math.sin(a)
        ok = all(not ter.wet_near(x, y, min(2.0, gap - 1.0)) and ter.max_slope_near(x, y, 1.5) < 10.0
                 and abs(x) < 55 and abs(y) < 55 for x, y in ((sx, sy), (gx, gy)))
        # the straight line crosses the lake, and the ground before the shore is gentle
        line = [(sx + (gx - sx) * k / 20, sy + (gy - sy) * k / 20) for k in range(21)]
        if ok and all(ter.max_slope_near(x, y, 0.5) < 14.0 for x, y in line[:5]):
            return lake, (sx, sy, a + math.pi), (gx, gy)
    pytest.skip("no clean approach to the lake on this seed")


# ====================================================================== water
def test_lakes_generated_with_outline_and_no_traps():
    for seed in (42, 7, 123):
        ter = Terrain(Config(seed=seed))
        assert ter.water, seed
        for w in ter.water:
            assert w.kind in ("lake", "river") and len(w.outline) >= (8 if w.kind == "lake" else 4)
            assert math.hypot(w.cx, w.cy) > 12.0                 # away from the lander
        reach = ter.reachable(18.0)
        assert reach.sum() > 0.3 * reach.size                    # most of the map still reachable
        # every target is reachable around the water
        for tg in ter.targets:
            i, j = ter.ij(tg.x, tg.y)
            assert ter.reach[j, i] and not ter.wet_near(tg.x, tg.y, 1.0)


def test_water_is_a_hole_in_the_map():
    sim = make()
    lake, (sx, sy, yaw), _goal = lake_approach(sim, gap=3.0)
    r = sim.rovers[0]
    teleport(sim, r, sx, sy, yaw)
    for _ in range(6):
        r.scan(sim.t)
    wet = sim.terrain.wet
    m = sim.map
    # LiDAR rays that hit the liquid get no return: the lake stays unknown (apart from hole filling at
    # the very edge), and the rays that should have hit ground there leave no-return evidence
    real = m.known & (m.hoff < 0.39)
    interior = wet.copy()                     # lake samples whose neighbours are all liquid too (not the shoreline)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            interior &= np.roll(np.roll(wet, dj, 0), di, 1)
    assert int((interior & real).sum()) == 0
    assert int((wet & (m.noret > 0)).sum()) > 0


def test_water_identified_and_avoided():
    sim = make()
    lake, (sx, sy, yaw), (gx, gy) = lake_approach(sim)
    r = sim.rovers[0]
    teleport(sim, r, sx, sy, yaw)
    sim.advisor._cmd_go_to(r.id, round(gx, 1), round(gy, 1), sim.t, "rules")
    wet_steps = 0
    for _ in range(int(90 / 0.2)):
        sim.step()
        for o in sim.rovers:
            w = sim.terrain.water_at(o.x, o.y)
            if w is not None and o.z - sim.cfg.chassis_half[2] < w.level:
                wet_steps += 1
        if sim.fleet.water_zones and r.path:
            break
    found = [e for e in sim.fleet.events if e["rover"] == r.id and e["level"] == "discovery"
             and ("liquid surface" in e["msg"] or "flat, level edge" in e["msg"])]
    assert found, [e["msg"] for e in sim.fleet.events if e["rover"] == r.id]
    assert "Marking a lake and routing around it" in found[0]["msg"] or "routing around it" in found[0]["msg"]
    assert sim.fleet.water_zones and sim.fleet.stats()["water_zones"]
    assert any(a["kind"] == "no_returns" for a in r.anomaly_log)
    # the planner keeps every path off identified water
    for _ in range(int(40 / 0.2)):
        sim.step()
        for (px, py) in r.path[r.wp:]:
            i, j = sim.map.ij(px, py)
            assert not sim.map.water[j, i], (px, py)
        for o in sim.rovers:
            w = sim.terrain.water_at(o.x, o.y)
            if w is not None and o.z - sim.cfg.chassis_half[2] < w.level:
                wet_steps += 1
    assert wet_steps == 0
    # frontiers do not sit on the lake: unknown water is no longer "unexplored"
    for (fx, fy, _s, _i) in sim.fleet.frontiers:
        assert sim.fleet.water_zone_at(fx, fy, margin=-0.5) is None


def test_failsafe_backs_out_of_water():
    sim = make()
    lake = next(w for w in sim.terrain.water if w.kind == "lake")
    ter = sim.terrain
    r = sim.rovers[1]
    # a spot just inside the shore, deep enough to wet the chassis, facing into the lake
    best = None
    for a in np.linspace(0, 2 * math.pi, 36, endpoint=False):
        for k in np.linspace(0.5, 1.0, 11):
            x, y = lake.cx + k * lake.radius * math.cos(a), lake.cy + k * lake.radius * math.sin(a)
            if ter.water_at(x, y) is not None and lake.level - ter.height_at(x, y) > 0.2:
                best = (x, y, a + math.pi)
        if best:
            break
    assert best
    x, y, _ = best
    teleport(sim, r, x, y, math.atan2(lake.cy - y, lake.cx - x))
    r.trail.append((x - 2.0 * math.cos(r.yaw), y - 2.0 * math.sin(r.yaw)))
    sim.step()
    assert r.state == "recovering" and r.rec_phase == "water"
    assert any("water sensor wet" in e["msg"] for e in sim.fleet.events if e["rover"] == r.id)
    assert sim.fleet.water_zones
    for _ in range(int(40 / 0.2)):
        sim.step()
        if r.rec_phase != "water":
            break
    assert sim.terrain.water_at(r.x, r.y) is None
    assert any("out of the water" in e["msg"] for e in sim.fleet.events if e["rover"] == r.id)


# ====================================================================== spires
def test_spire_fall_makes_the_rover_replan():
    sim = make()
    sim.run_for(40.0)
    h = None
    for _ in range(300):
        try:
            h = sim.hazards.trigger("spire_fall")
            break
        except HazardError:
            sim.step()
    assert h is not None and h["type"] == "spire_fall"
    assert {"id", "type", "t", "rover", "spire", "x0", "y0", "x1", "y1", "active"} <= set(h)
    assert any(e["msg"].startswith("Spire fell across") and e["level"] == "hazard" for e in sim.fleet.events)
    tick = sim.tick_message()
    assert "features" in tick                                   # sent once, right after it changed
    sp = next(s for s in tick["features"]["spires"] if s["id"] == h["spire"])
    assert sp["standing"] is False
    assert "features" not in sim.tick_message()
    rid = h["rover"]
    r = next(o for o in sim.rovers if o.id == rid)
    for _ in range(int(20 / 0.2)):
        sim.step()
    seen = [e for e in sim.fleet.events if e["rover"] == rid and e["t"] >= h["t"]
            and ("sees a new long, low obstacle" in e["msg"] or "sees new rocks" in e["msg"])]
    assert seen, [e["msg"] for e in sim.fleet.events if e["t"] >= h["t"]]
    assert any("rerouting around" in e["msg"] or "finds no way past" in e["msg"]
               for e in sim.fleet.events if e["rover"] == rid and e["t"] >= h["t"])
    # it lies there: truth heights along it rose, and the mocap body is horizontal
    H = sim.truth_heights()
    mx, my = (h["x0"] + h["x1"]) / 2, (h["y0"] + h["y1"]) / 2
    i, j = sim.terrain.ij(mx, my)
    assert H[j, i] > sim.terrain.heights[j, i] + 0.4
    mid = sim.hazards.spire_mocap[h["spire"]]
    w, qx, qy, qz = sim.data.mocap_quat[mid]
    assert abs(1.0 - 2.0 * (qx * qx + qy * qy)) < 0.35          # body z axis close to horizontal
    # the rover's route does not run through the fallen spire
    for (px, py) in r.path[r.wp:]:
        d = abs((py - h["y0"]) * (h["x1"] - h["x0"]) - (px - h["x0"]) * (h["y1"] - h["y0"])) / math.hypot(
            h["x1"] - h["x0"], h["y1"] - h["y0"])
        f = ((px - h["x0"]) * (h["x1"] - h["x0"]) + (py - h["y0"]) * (h["y1"] - h["y0"])) / (
            (h["x1"] - h["x0"]) ** 2 + (h["y1"] - h["y0"]) ** 2)
        assert not (0.05 < f < 0.95 and d < 0.4), (px, py)


def test_no_false_change_detections_near_spires():
    sim = make()
    sim.run_for(150.0)
    assert sim.map.changes <= 3, sim.map.changes
    assert not any("new rocks" in e["msg"] or "long, low obstacle" in e["msg"] for e in sim.fleet.events)


# ====================================================================== lifeforms
def _bring_herd(sim, rid="r1", t=5.0):
    sim.run_for(t)
    for _ in range(100):
        try:
            return sim.hazards.trigger("lifeform", rid)
        except HazardError:
            sim.step()
    raise AssertionError("could not bring a herd")


def test_tracks_form_from_moving_creatures():
    sim = make()
    h = _bring_herd(sim)
    assert h["type"] == "lifeform" and {"x", "y", "rover", "active"} <= set(h)
    for _ in range(int(10 / 0.2)):
        sim.step()
    conf = sim.fleet.tracker.confirmed()
    assert conf, "no moving-object track"
    live = [tr for tr in conf if sim.t - tr.last_seen < 1.0]
    assert live
    for tr in live:
        d = min(math.hypot(tr.x - c.x, tr.y - c.y) for c in sim.lifeforms.creatures)
        assert d < 1.0, (tr.id, d)
        assert tr.speed < 1.0
    msgs = sim.fleet.tracker.to_msgs()
    assert all(set(m) == {"id", "x", "y", "vx", "vy", "size", "last_seen", "observer", "label", "status"} for m in msgs)
    assert all(m["status"] in ("tracking", "lost") for m in msgs)


def test_observer_keeps_its_distance_and_gives_up_in_time():
    sim = make(observe_max_s=45.0)
    _bring_herd(sim)
    r = next(o for o in sim.rovers if o.id == "r1")
    for _ in range(int(10 / 0.2)):
        sim.step()
        if r.state == "observing":
            break
    assert r.state == "observing" and r.obs is not None
    t0, x0, y0 = r.obs["t0"], r.obs["x0"], r.obs["y0"]
    dists, follow = [], []
    while r.obs is not None and sim.t < t0 + 60.0:
        sim.step()
        d = min(math.hypot(c.x - r.x, c.y - r.y) for c in sim.lifeforms.creatures)
        dists.append(d)
        if r.obs is not None and r.obs["mode"] == "follow" and sim.t > t0 + 8.0:
            follow.append(d)
        # never leaves the safe area: within the leash, never on water, soft ground or unknown steep ground
        assert math.hypot(r.x - x0, r.y - y0) <= sim.cfg.observe_leash + 1.0
        assert sim.terrain.water_at(r.x, r.y) is None
        assert sim.fleet.soft_zone_at(r.x, r.y, margin=0.0) is None
    assert r.obs is None and r.state != "observing"
    ended = sim.t - t0
    assert ended <= 45.0 + 1.0, ended
    assert min(dists) >= 3.3, min(dists)
    if len(follow) > 20:
        med = float(np.median(follow))
        assert 3.8 <= med <= 7.6, med
    ev = [e for e in sim.fleet.events if e["rover"] == "r1" and "observed the" in e["msg"]]
    assert ev and "individual" in ev[0]["msg"] and "average speed" in ev[0]["msg"]
    assert sim.tick_message()["rovers"][0]["track_id"] is None


def test_first_sighting_asks_the_operator():
    p = fake_planner()
    sim = make(planner=p, operator_timeout_s=60.0)
    try:
        _bring_herd(sim)
        r = next(o for o in sim.rovers if o.id == "r1")
        q = None
        for _ in range(int(15 / 0.2)):
            sim.step()
            reqs = sim.advisor.requests_msg()
            if reqs:
                q = reqs[0]
                break
        assert q is not None, [e["msg"] for e in sim.fleet.events][-10:]
        assert q["rover"] == "r1" and q["question"].startswith("Possible lifeform")
        assert q["question"].endswith("Follow it or keep exploring?")
        assert [o["id"] for o in q["options"]] == ["follow", "observe_here", "keep_exploring"]
        assert q["default"] == "observe_here"
        assert r.state == "observing" and r.obs["mode"] == "here"
        tick = sim.tick_message()
        assert tick["rovers"][0]["track_id"] == r.obs["track"]
        tr = sim.fleet.tracker.get(r.obs["track"])
        assert tr.label.startswith("possible lifeform")
        assert any(e.get("source") == "gemma" and "possible lifeform" in e["msg"] for e in sim.fleet.events)
        assert sim.advisor.answer(q["id"], "follow") == "ok"
        assert r.obs["mode"] == "follow"
        assert sim.fleet.lifeforms_seen >= 1
        # later sightings only ask when Gemma is unsure: the fake is sure (0.8), so no new question
        n_req = sim.advisor.req_seq
        for _ in range(int(10 / 0.2)):
            sim.step()
        try:
            sim.hazards.trigger("lifeform", "r2")
        except HazardError:
            pass
        for _ in range(int(10 / 0.2)):
            sim.step()
        assert sim.advisor.req_seq == n_req
    finally:
        p.close()
        sim.close()


def test_lifeform_question_expires_to_observe_from_here():
    p = fake_planner()
    sim = make(planner=p, operator_timeout_s=1.0)
    try:
        _bring_herd(sim)
        r = next(o for o in sim.rovers if o.id == "r1")
        for _ in range(int(15 / 0.2)):
            sim.step()
            if sim.advisor.requests:
                break
        assert sim.advisor.requests
        import time as _t
        _t.sleep(1.2)
        sim.step()
        assert not sim.advisor.requests
        assert r.obs is not None and r.obs["mode"] == "here"
        assert any("no reply in 1 s, taking the safe option: observe from here" in m["text"].lower()
                   for m in sim.advisor.chat)
    finally:
        p.close()
        sim.close()


# ====================================================================== point and click
def _call_hazard(body: dict, runner):
    from redshift import server

    class Req:
        def __init__(self, b: bytes):
            self._b = b

        async def body(self):
            return self._b

    old = server.state.runner
    server.state.runner = runner
    try:
        out = asyncio.run(server.hazard(Req(json.dumps(body).encode())))
    finally:
        server.state.runner = old
    if hasattr(out, "status_code"):
        return out.status_code, json.loads(out.body)
    return 200, json.loads(json.dumps(out))


def test_point_placement_success_and_errors():
    from redshift import server
    runner = server.SimRunner(Config(seed=42, plan_clock="model"))     # not started: no thread
    sim = runner.sim
    ter = sim.terrain

    used = []
    rng = np.random.default_rng(3)

    def gentle_point(min_r=15.0):
        for _ in range(4000):
            x, y = (float(v) for v in rng.uniform(-50, 50, 2))
            if math.hypot(x, y) < min_r or ter.wet_near(x, y, 5.0) or ter.max_slope_near(x, y, 3.5) > 9.0:
                continue
            if any(math.hypot(x - ux, y - uy) < 12.0 for ux, uy in used):
                continue
            if any(math.hypot(x - tg.x, y - tg.y) < 5.0 for tg in ter.targets):
                continue
            if any(math.hypot(x - r.x, y - r.y) < max(r.sx, r.sy) + 3.5 for r in ter.rocks):
                continue
            if any(math.hypot(x - s.x, y - s.y) < 4.5 for s in ter.spires):
                continue
            used.append((x, y))
            return round(x, 1), round(y, 1)
        raise AssertionError("no gentle point")

    x, y = gentle_point()
    code, body = _call_hazard({"type": "sand", "x": x, "y": y}, runner)
    assert code == 200 and body["hazard"]["type"] == "sand", body
    assert math.hypot(body["hazard"]["x"] - x, body["hazard"]["y"] - y) <= 2.1

    x, y = gentle_point(25.0)
    code, body = _call_hazard({"type": "rockfall", "x": x, "y": y}, runner)
    assert code == 200 and len(body["hazard"]["rocks"]) >= 3, body

    sp = ter.spires[0]
    code, body = _call_hazard({"type": "spire_fall", "x": round(sp.x + 2.0, 1), "y": round(sp.y, 1)}, runner)
    assert code == 200 and body["hazard"]["spire"] == sp.id, body        # the nearest standing spire toppled
    assert sim.features_msg()["spires"][0]["standing"] is False

    x, y = gentle_point(20.0)
    if min(math.hypot(x - s.x, y - s.y) for s in ter.spires) > 6.5:
        code, body = _call_hazard({"type": "spire_fall", "x": x, "y": y}, runner)
        assert code == 200 and body["hazard"]["spire"] not in [s.id for s in ter.spires[:14]], body   # a new one

    x, y = gentle_point(20.0)
    code, body = _call_hazard({"type": "lifeform", "x": x, "y": y}, runner)
    assert code == 200 and body["hazard"]["type"] == "lifeform", body
    assert any(math.hypot(c.x - body["hazard"]["x"], c.y - body["hazard"]["y"]) < 3.0 for c in sim.lifeforms.creatures)

    # 400s: off the map, on the lander, within 2 m of a rover, unsuitable ground, bad x / y
    r = sim.rovers[0]
    teleport(sim, r, 14.0, 3.0, 0.0)
    lake = ter.water[0]
    steep = None
    jj, ii = np.nonzero(ter.slope > 24.0)
    for j, i in zip(jj, ii):
        px, py = ter.origin[0] + i * ter.res, ter.origin[1] + j * ter.res
        if math.hypot(px, py) > 10.0 and not ter.wet_near(px, py, 3.0) and abs(px) < 50 and abs(py) < 50:
            steep = (round(px, 1), round(py, 1))
            break
    cases = [({"type": "rockfall", "x": 500.0, "y": 0.0}, "off the map"),
             ({"type": "sand", "x": 0.5, "y": 0.5}, "on the lander"),
             ({"type": "lifeform", "x": round(r.x + 1.0, 2), "y": round(r.y + 1.0, 2)}, "within 2 m of"),
             ({"type": "spire_fall", "x": round(lake.cx, 1), "y": round(lake.cy, 1)}, "in the water"),
             ({"type": "sand", "x": steep[0], "y": steep[1]}, "steep"),
             ({"type": "rockfall", "x": 3.0}, "both x and y"),
             ({"type": "rockfall", "x": "a", "y": 2}, "must be numbers")]
    for req, part in cases:
        code, body = _call_hazard(req, runner)
        assert code == 400 and body["ok"] is False and part in body["error"], (req, body)
    # dust_storm ignores the point
    code, body = _call_hazard({"type": "dust_storm", "x": 500.0, "y": 0.0}, runner)
    assert code == 200


# ====================================================================== samples (v4b)
def _start_next_to(sim, r, tidx):
    ts = sim.fleet.targets[tidx]
    teleport(sim, r, ts.spec.x + 1.2, ts.spec.y, math.pi)
    ts.status, ts.found_by, ts.found_t = "investigating", r.id, sim.t
    ts.investigator = r.id
    r.state = "exploring"
    r.start_investigation(sim.t, tidx)


def test_sampling_runs_the_instrument_then_analyses():
    sim = make()
    r = sim.rovers[0]
    k = next(k for k, ts in enumerate(sim.fleet.targets) if ts.spec.kind == "mineral")
    _start_next_to(sim, r, k)
    progress = []
    for _ in range(int(20 / 0.2)):
        sim.step()
        m = r.to_msg()
        if "sampling" in m:
            assert m["sampling"]["instrument"] == "drill" and m["sampling"]["target"] == sim.fleet.targets[k].spec.id
            progress.append(m["sampling"]["progress"])
        if sim.fleet.targets[k].status == "sampled":
            break
    ts = sim.fleet.targets[k]
    assert ts.status == "sampled" and ts.value == ts.spec.value and ts.analysis == ts.spec.analysis
    assert ts.carried_by == r.id and r.samples == [k] and not ts.cached
    assert progress and progress == sorted(progress) and progress[-1] > 0.85
    assert len(progress) >= int(sim.cfg.drill_s / 0.2) - 3                 # the drill takes about 10 s
    msg = ts.to_msg()
    assert msg["instrument"] == "drill" and 1 <= msg["value"] <= 10 and msg["analysis"]
    ev = [e for e in sim.fleet.events if e["rover"] == r.id and "drilled a core from" in e["msg"]]
    assert ev and f"(value {ts.value}/10). 1 of 3 tubes used." in ev[0]["msg"]
    assert sim.fleet.stats()["samples_carried"] == 1


def test_capacity_trade_offs_and_caching():
    sim = make()
    f = sim.fleet
    r = sim.rovers[0]
    kinds = [k for k, ts in enumerate(f.targets)]
    for k in kinds[:2]:
        f.target_sampled(sim.t, r, k)
    low = next(k for k in kinds[2:] if f.targets[k].spec.est_value < 6)
    assert "only 1 of 3 tubes left" in r.tube_verdict(f.targets[low])
    f.targets[low].status = "found"
    k, _d = f.unclaimed_target(sim.t, r)
    assert k != low
    assert sum("skips" in e["msg"] for e in f.events) == 1
    f.unclaimed_target(sim.t, r)
    assert sum("skips" in e["msg"] for e in f.events) == 1                  # logged once
    f.target_sampled(sim.t, r, kinds[2])
    assert r.free_tubes() == 0 and not r.can_investigate(sim.t, f.targets[low])
    assert any("all 3 tubes full" in e["msg"] for e in f.events)
    # tubes full and a valuable find known: a dedicated trip to the lander cache
    hi = next((k for k in kinds[3:] if f.targets[k].spec.est_value >= sim.cfg.high_value), None)
    if hi is not None:
        f.targets[hi].status = "found"
        teleport(sim, r, 12.0, 3.0, 0.0)
        r.state, r.next_target_check = "exploring", 0.0
        sim.step()
        assert r.state == "returning" and r.return_reason == "cache"
        assert any("Returning to cache the samples first" in e["msg"] for e in f.events)
    # docking caches everything
    values = [f.targets[k].value for k in r.samples]
    teleport(sim, r, r.dock[0], r.dock[1], 0.0)
    r.state, r.return_reason = "returning", "cache"
    sim.step()
    assert r.samples == [] and f.samples_cached == 3 and f.science_score == sum(values)
    for k in kinds[:3]:
        assert f.targets[k].cached and f.targets[k].carried_by is None
    ev = [e for e in f.events if "cached 3 samples at the lander" in e["msg"]]
    assert ev and f"(fleet total 3, science score {sum(values)})" in ev[0]["msg"]
    st = f.stats()
    assert st["samples_cached"] == 3 and st["samples_carried"] == 0 and st["science_score"] == sum(values)


def test_sample_command_and_gemma_summaries_know_about_tubes():
    p = fake_planner()
    sim = make(planner=p)
    try:
        f = sim.fleet
        r = sim.rovers[0]
        for k in range(3):
            f.target_sampled(sim.t, r, k)
        f.targets[5].status = "found"
        f.targets[5].retry_after = 1e9          # keep the rovers' own choice off it: only the operator sends one
        sim.advisor.operator_message(f"r1 sample {f.targets[5].spec.id}")
        for _ in range(5):
            sim.step()
        rep = [m for m in sim.advisor.chat if m["from"] == "r1" and m["kind"] == "refuse"]
        assert rep and "tubes are full" in rep[-1]["text"], [m["text"] for m in sim.advisor.chat]
        # a rover with free tubes takes the job and says how many tubes are free
        sim.advisor.operator_message(f"r2 sample {f.targets[5].spec.id}")
        for _ in range(5):
            sim.step()
        ack = [m for m in sim.advisor.chat if m["from"] == "r2" and m["kind"] == "ack"]
        assert ack and "3 of 3 tubes free" in ack[-1]["text"]
        # candidate summaries for Gemma quote value and tubes
        from redshift.advisor import _Choice
        r2 = sim.rovers[1]
        job = _Choice(sim.t, (r2.x + 5, r2.y))
        job.cands = [{"id": "f1", "kind": "frontier", "goal": (r2.x + 5, r2.y), "path": [(r2.x + 5, r2.y)], "size": 9,
                      "plen": 5.0, "smax": 4.0},
                     {"id": f.targets[6].spec.id, "kind": "target", "goal": (f.targets[6].spec.x, f.targets[6].spec.y),
                      "path": [(0, 0)], "tidx": 6, "plen": 12.0, "smax": 5.0}]
        summ = sim.advisor._summaries(r2, job)
        t_s = next(c["summary"] for c in summ if c["kind"] == "target")
        assert f"value {f.targets[6].spec.est_value}/10" in t_s and "tubes free" in t_s
    finally:
        p.close()
        sim.close()


def test_values_deterministic_and_varied():
    a, b = Terrain(Config(seed=42)), Terrain(Config(seed=42))
    assert [(t.id, t.value, t.analysis, t.kind) for t in a.targets] == [(t.id, t.value, t.analysis, t.kind) for t in b.targets]
    vals = [t.value for t in a.targets]
    assert len(set(vals)) >= 4 and max(vals) >= 8 and min(vals) <= 4
    traces = [t for t in a.targets if t.kind == "trace"]
    assert traces and all(t.instrument == "swab" for t in traces)
    assert all(min(math.hypot(t.x - h.cx, t.y - h.cy) for h in a.herds) < 8.0 for t in traces)


# ====================================================================== v4 protocol fields
def test_v4_protocol_fields_and_json():
    sim = make()
    _bring_herd(sim)
    sim.run_for(15.0)
    init = sim.init_message()
    tick = sim.tick_message()
    for m in (init, tick):
        text = json.dumps(m, allow_nan=False)
        assert EM_DASH not in text
        assert isinstance(m["lifeforms"], list) and isinstance(m["tracks"], list)
        for lf in m["lifeforms"]:
            assert set(lf) == {"id", "kind", "x", "y", "z", "yaw", "speed", "size"} and lf["kind"] == "grazer"
        f = m["fleet"]
        assert {"water_zones", "lifeforms_seen", "samples_cached", "samples_carried", "science_score"} <= set(f)
        for r in m["rovers"] if m["type"] == "tick" else []:
            assert {"track_id", "samples", "capacity"} <= set(r) and r["capacity"] == 3
            assert r["state"] in {"exploring", "investigating", "returning", "charging", "recovering", "idle",
                                  "fault", "observing"}
        for tg in m["targets"]:
            assert {"instrument", "value", "analysis", "carried_by", "cached"} <= set(tg)
    feats = init["features"]
    assert feats["water"] and all({"id", "kind", "level", "outline"} == set(w) for w in feats["water"])
    assert all(len(w["outline"]) >= 8 and all(len(p) == 2 for p in w["outline"]) for w in feats["water"])
    assert len(feats["spires"]) == 14
    assert all({"id", "x", "y", "height", "radius", "standing"} == set(s) for s in feats["spires"])
    assert all(2.0 <= s["height"] <= 4.0 for s in feats["spires"])
    assert len(init["lifeforms"]) == len(sim.lifeforms.creatures) >= 6
    assert init["map"]["nx"] == 241 and init["map"]["origin"] == [-60.0, -60.0]
    assert [h["type"] for h in tick["hazards"]] == ["lifeform"]
