"""Protocol v9: rover health and damage, distress, rescue, towing, salvage, repair.

Scenarios on the default 120 m map (seed 42). Rovers are sometimes carried into place by hand (a test harness move);
after that every reaction comes from their sensors and the physics.

    python -m pytest -q tests/test_v9.py        (about 2 minutes)
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import gemma as g  # noqa: E402
from redshift.camera import _sensor_noise  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.gemma import FakeBackend, GemmaConfig, GemmaPlanner  # noqa: E402
from redshift.health import PARTS  # noqa: E402
from redshift.rescue import REPAIR_RADIUS  # noqa: E402
from redshift.sim import Simulation  # noqa: E402
from redshift.world import rover_spawn_height  # noqa: E402

EM_DASH = chr(0x2014)


def make(planner=None, **cfg) -> Simulation:
    return Simulation(Config(seed=42, plan_clock="model", **cfg), planner=planner, wait_for_gemma=planner is not None)


def fake_planner() -> GemmaPlanner:
    p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0), backend=FakeBackend(0.0))
    assert p.wait_ready(2.0)
    return p


def teleport(sim, r, x, y, yaw) -> None:
    d = sim.data
    a = r.qadr
    d.qpos[a:a + 3] = (x, y, sim.terrain.height_at(x, y) + rover_spawn_height(sim.cfg) + 0.04)
    d.qpos[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    d.qpos[a + 7:a + r.nq] = 0.0
    d.qvel[r.vadr:r.vadr + r.nv] = 0.0
    mujoco.mj_forward(sim.model, d)
    r.read_pose()
    r.trail.clear()
    r.last_xy = (r.x, r.y)
    r.goal, r.path = None, []
    r.home_dist = None


def flat_spot(sim, x, y, avoid=(), min_lander=6.0):
    """Gentle, dry, rock-free ground near (x, y)."""
    ter = sim.terrain
    rocky = (ter.truth_heights - ter.heights) > 0.08
    for rr in np.arange(0.0, 14.0, 0.5):
        for a in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
            px, py = x + rr * math.cos(a), y + rr * math.sin(a)
            if math.hypot(px, py) < min_lander or any(math.hypot(px - ax, py - ay) < 4.0 for ax, ay in avoid):
                continue
            if ter.max_slope_near(px, py, 2.0) > 6.0 or ter.wet_near(px, py, 3.0):
                continue
            i, j = ter.ij(px, py)
            if rocky[max(j - 6, 0):j + 7, max(i - 6, 0):i + 7].any():
                continue
            return px, py
    raise AssertionError(f"no flat spot near ({x}, {y})")


def run_until(sim, cond, max_s: float) -> bool:
    end = sim.t + max_s
    while sim.t < end:
        sim.step()
        if cond():
            return True
    return False


def events_since(sim, n0: int) -> list:
    return sim.fleet.events[n0:]


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


# ====================================================================== damage from the physics
def test_boulder_and_spire_hits_damage_while_normal_driving_does_not():
    sim = make()
    sim.run_for(45.0)
    assert not [e for e in sim.fleet.events if e.get("damage")], "normal driving caused damage"
    assert all(r.health.hp() == 100 for r in sim.rovers)
    r2, r3 = sim.rovers[1], sim.rovers[2]
    n0 = len(sim.fleet.events)
    h = sim.hazards.trigger("rockfall", None, r2.x + 0.3, r2.y + 0.2)      # clicked on the rover: aimed at it
    assert h["type"] == "rockfall" and h["rover"] == "r2"
    assert run_until(sim, lambda: any(e.get("damage", {}).get("rover") == "r2" for e in events_since(sim, n0)), 5.0)
    hit = next(e for e in events_since(sim, n0) if e.get("damage", {}).get("rover") == "r2")
    dmg = hit["damage"]
    assert dmg["cause"] == "boulder" and dmg["after"] < dmg["before"] and dmg["part"] in PARTS
    assert set(dmg) == {"rover", "part", "before", "after", "cause", "shock_g"}
    assert r2.health.hp() < 100 and r2.health.last_hit["cause"] == "boulder"
    msg = r2.to_msg()["health"]
    assert set(msg) == {"hp", "status", "parts", "diagnosis", "last_hit"} and msg["hp"] == r2.health.hp()
    # a spire toppling onto another rover
    n1 = len(sim.fleet.events)
    h = sim.hazards.trigger("spire_fall", None, r3.x - 0.2, r3.y + 0.3)
    assert h["type"] == "spire_fall" and h["rover"] == "r3"
    assert run_until(sim, lambda: any(e.get("damage", {}).get("rover") == "r3" for e in events_since(sim, n1)), 6.0)
    d3 = next(e["damage"] for e in events_since(sim, n1) if e.get("damage", {}).get("rover") == "r3")
    assert d3["cause"] == "spire"
    assert sim.glitches == 0
    # still refused: the lander, and lifeforms within 2 m of a rover
    import pytest
    from redshift.hazards import HazardError
    with pytest.raises(HazardError):
        sim.hazards.trigger("rockfall", None, 0.5, 0.5)
    with pytest.raises(HazardError):
        sim.hazards.trigger("lifeform", None, r3.x + 1.0, r3.y)


def test_part_effects_drive_pull_lidar_range_power_and_camera():
    sim = make()
    sim.run_for(1.0)
    r = sim.rovers[0]
    others = [o for o in sim.rovers if o is not r]
    x, y = flat_spot(sim, 12.0, 0.0)

    def open_loop(seconds: float) -> float:
        teleport(sim, r, x, y, 0.0)
        y0 = r.yaw
        for _ in range(int(seconds / sim.model.opt.timestep)):
            r.cmd_v, r.cmd_w = sim.cfg.v_max, 0.0
            r._apply()
            for o in others:
                o._set_cmd(0.0, 0.0)
                o._apply()
            mujoco.mj_step(sim.model, sim.data)
        r.read_pose()
        return (r.yaw - y0 + math.pi) % (2 * math.pi) - math.pi

    straight = open_loop(4.0)
    r.health.set_parts({"drive_r": 35.0})
    pulled = open_loop(4.0)
    assert abs(straight) < 0.1, straight
    assert pulled < straight - 0.12, (straight, pulled)         # a weak right side: it turns right (clockwise)
    e = r.health.effects()
    assert abs(e["v_cap"] * sim.cfg.v_max - 0.4) < 0.01        # top speed it allows itself: 0.4 m/s
    assert e["slope_cap"] is not None and e["slope_cap"] < 16.0
    assert r.limit_now() <= e["slope_cap"]
    assert "pulls right" in r.health.diagnosis() and r.health.status() == "damaged"
    # LiDAR: shorter range and fewer returns
    r.health.set_parts({"drive_r": 100.0})
    teleport(sim, r, x, y, 0.0)
    r.scan(sim.t)
    healthy_hits, healthy_far = r.lidar_hits, float(np.max(np.hypot(r.scan_pts[:, 0] - r.x, r.scan_pts[:, 1] - r.y)))
    r.health.set_parts({"lidar": 30.0})
    r.scan(sim.t)
    far = float(np.max(np.hypot(r.scan_pts[:, 0] - r.x, r.scan_pts[:, 1] - r.y)))
    assert far <= sim.cfg.lidar_range * (0.35 + 0.65 * 0.3) + 0.3 < healthy_far, (far, healthy_far)
    assert r.lidar_hits < 0.75 * healthy_hits
    # power: drains faster; camera: noisy frames
    r.health.set_parts({"lidar": 100.0})
    r.state, b0 = "exploring", 0.8
    r.battery = b0
    r._battery(sim.t, 1.0, 0.0)
    healthy = b0 - r.battery
    r.health.set_parts({"power": 40.0})
    r.battery = b0
    r._battery(sim.t, 1.0, 0.0)
    assert (b0 - r.battery) > 2.0 * healthy
    img = np.full((24, 32, 3), 120, np.uint8)
    noisy = _sensor_noise(img, r.health.effects()["camera_noise"] + 30.0, 7)
    assert noisy.std() > 10.0 and np.array_equal(noisy, _sensor_noise(img, r.health.effects()["camera_noise"] + 30.0, 7))


# ====================================================================== the demo hazard
def test_damage_hazard_endpoint_every_severity_and_errors():
    from redshift import server
    runner = server.SimRunner(Config(seed=42, plan_clock="model"))     # not started: no thread
    sim = runner.sim
    sim.run_for(20.0)
    code, body = _call_hazard({"type": "damage", "rover": "r1", "severity": "light", "part": "camera"}, runner)
    assert code == 200 and body["hazard"]["type"] == "damage" and body["hazard"]["severity"] == "light"
    assert body["health"]["parts"]["camera"] < 100 and body["health"]["status"] in ("ok", "damaged")
    code, body = _call_hazard({"type": "damage", "rover": "r1", "severity": "heavy", "part": "drive_l"}, runner)
    assert code == 200 and body["health"]["status"] == "damaged" and 25 <= body["health"]["parts"]["drive_l"] <= 45
    code, body = _call_hazard({"type": "damage", "rover": "r2", "severity": "disable"}, runner)
    assert code == 200 and body["health"]["status"] == "immobile"
    assert sim.rovers[1].state == "disabled"                                         # immediately
    assert any(e["msg"].startswith("Distress signal from r2") for e in sim.fleet.events)
    assert body["health"]["parts"]["drive_r"] == 0
    code, body = _call_hazard({"type": "damage", "rover": "r2", "severity": "disable"}, runner)
    assert code == 400 and "already immobile" in body["error"]
    code, body = _call_hazard({"type": "damage", "rover": "r3", "severity": "wreck"}, runner)
    assert code == 200 and body["health"]["status"] == "wrecked" and body["health"]["hp"] < 40
    code, body = _call_hazard({"type": "damage", "rover": "r3", "severity": "light"}, runner)
    assert code == 400 and "already wrecked" in body["error"]
    for bad, part in (({"type": "damage", "rover": "r1", "severity": "medium"}, "unknown severity"),
                      ({"type": "damage", "rover": "r1", "severity": "light", "part": "wings"}, "unknown part"),
                      ({"type": "damage", "rover": "r9", "severity": "light"}, "unknown rover"),
                      ({"type": "damage", "rover": "r1", "severity": 3}, "severity must be")):
        code, body = _call_hazard(bad, runner)
        assert code == 400 and part in body["error"], (bad, body)
    # disable naming a sensor part: that part is hit and the rover is still immobile (right drive burnt out)
    runner2 = server.SimRunner(Config(seed=42, plan_clock="model"))
    code, body = _call_hazard({"type": "damage", "rover": "r1", "severity": "disable", "part": "lidar"}, runner2)
    assert code == 200 and body["health"]["status"] == "immobile"
    assert body["health"]["parts"]["lidar"] <= 10 and body["health"]["parts"]["drive_r"] == 0
    code, body = _call_hazard({"type": "damage", "rover": "r2", "severity": "disable", "part": "power"}, runner2)
    assert code == 200 and body["health"]["status"] == "immobile" and body["health"]["parts"]["power"] <= 10
    # jam (a sim extra): immobile, but everything is fixable in the field
    code, body = _call_hazard({"type": "damage", "rover": "r3", "severity": "jam"}, runner2)
    r3 = runner2.sim.rovers[2]
    assert code == 200 and body["health"]["status"] == "immobile" and r3.health.mobile(field_fixed=True)
    assert sim.hazards.to_msgs(sim.t)[0]["type"] == "damage" and "health" not in sim.hazards.to_msgs(sim.t)[0]


# ====================================================================== distress, helper, field repair
def test_distress_helper_choice_and_field_repair():
    sim = make()
    sim.run_for(15.0)
    r1, r2, r3 = sim.rovers
    x2, y2 = flat_spot(sim, 10.0, 8.0)
    teleport(sim, r2, x2, y2, math.atan2(-y2, -x2))
    x3, y3 = flat_spot(sim, x2 + 3.0, y2, avoid=[(x2, y2)])
    teleport(sim, r3, x3, y3, 0.0)                    # r3 is closest, but its battery is too low to help
    r3.battery = 0.2
    sim.run_for(0.4)
    n0 = len(sim.fleet.events)
    sim.hazards.trigger("damage", "r2", severity="jam")
    assert r2.state == "disabled" and r2.goal is None and r2.target_idx is None
    distress = [e for e in events_since(sim, n0) if e["msg"].startswith("Distress signal from r2")]
    assert distress and "Requesting help at" in distress[0]["msg"]
    assert any(m["from"] == "r2" and m["kind"] == "info" and "Requesting help at" in m["text"] for m in sim.advisor.chat)
    rec = sim.rescue.active("r2")
    assert rec is not None and rec["helper"] == "r1" and r1.state == "rescuing"
    assert "battery" in sim.rescue.can_help(r3, rec)
    assert sim.fleet.stats()["rescues"] == [{"rover": "r2", "helper": "r1", "phase": "en_route", "plan": None}]
    assert run_until(sim, lambda: rec["phase"] == "repairing", 120.0), rec["phase"]
    diag = next(e["msg"] for e in events_since(sim, n0) if "diagnosed r2" in e["msg"])
    assert "jammed wheel" in diag and "tripped motor controller" in diag and "Fixable here" in diag
    assert run_until(sim, lambda: rec["phase"] == "done", 30.0)
    assert r2.health.mobile() and r2.state not in ("disabled", "towed", "wrecked")
    assert rec["plan"] is None and r1.state not in ("rescuing", "towing")
    assert any("can drive again" in e["msg"] for e in events_since(sim, n0))


def test_silent_rover_found_by_its_heartbeat():
    sim = make()
    sim.run_for(10.0)
    r2 = sim.rovers[1]
    x2, y2 = flat_spot(sim, 14.0, -6.0)
    teleport(sim, r2, x2, y2, 0.0)
    sim.run_for(0.4)
    n0 = len(sim.fleet.events)
    r2.health.set_parts({"comms": 0.0, "drive_l": 0.0})
    sim.rescue.check_now(r2)
    assert r2.state == "disabled" and sim.rescue.active("r2") is None          # no distress: its radio is dead
    assert any("radio too weak" in e["msg"] for e in events_since(sim, n0))
    assert run_until(sim, lambda: sim.rescue.active("r2") is not None, 20.0)
    rec = sim.rescue.active("r2")
    assert rec["silent"] and any("no heartbeat from r2" in e["msg"] for e in events_since(sim, n0))
    assert run_until(sim, lambda: rec["phase"] == "diagnosing", 120.0), rec["phase"]
    assert any("could not call for help" in e["msg"] for e in events_since(sim, n0))


# ====================================================================== tow, repair at the lander
def test_tow_home_and_repair_with_a_spare():
    sim = make()
    sim.run_for(15.0)
    r1, r2, r3 = sim.rovers
    x2, y2 = flat_spot(sim, 9.0, -9.0, min_lander=9.0)
    teleport(sim, r2, x2, y2, math.atan2(-y2, -x2))     # facing the lander: room to back up to its hitch
    sim.run_for(0.4)
    spares0 = dict(sim.fleet.spares)
    n0 = len(sim.fleet.events)
    sim.hazards.trigger("damage", "r2", severity="disable", part="drive_r")
    rec = sim.rescue.active("r2")
    assert rec is not None and rec["helper"] is not None
    assert run_until(sim, lambda: r2.state == "towed", 200.0), (rec["phase"], rec.get("hitch"))
    dec = [e for e in events_since(sim, n0) if e.get("decision", {}).get("choice") == "tow"]
    assert dec and dec[0]["decision"]["by"] == "rules" and dec[0]["decision"]["options"][0] == "tow"
    helper = sim.rescue._rover(rec["helper"])
    assert helper.state == "towing" and helper.limit_now() <= sim.fleet.slope_limit - 4.0 + 1e-6
    assert run_until(sim, lambda: r2.state == "repairing", 260.0), (rec["phase"], r2.state)
    assert math.hypot(r2.x, r2.y) <= REPAIR_RADIUS
    how = "rigid bar" if any("rigid bar" in e["msg"] for e in events_since(sim, n0)) else "tow bar"
    print(f"towed on the {how}")
    sim.run_for(0.4)
    assert sim.fleet.spares["drive"] == spares0["drive"] - 1                      # the burnt-out motor replaced
    assert run_until(sim, lambda: r2.state != "repairing", 60.0)
    assert r2.health.hp() >= 99 and r2.health.status() == "ok"
    assert rec["phase"] == "done" and rec["plan"] == "tow"
    b = sim.brain_stats()
    assert all(v["p95"] < sim.cfg.brain_budget_ms for v in b.values()), b
    assert sim.glitches == 0


def test_lander_repair_waits_for_a_missing_spare_and_salvage_brings_one():
    sim = make()
    sim.run_for(5.0)
    r2 = sim.rovers[1]
    sim.fleet.spares["lidar"] = 0
    r2.health.set_parts({"lidar": 10.0, "camera": 60.0})
    r2.state = "repairing"
    sim.run_for(2.0)
    assert r2.state == "repairing" and r2.health.value("lidar") < 30.0                # no spare: still waiting
    assert any("needs a spare lidar" in e["msg"] for e in sim.fleet.events)
    r2.state = "repairing"
    r1 = sim.rovers[0]
    r1.carried_spares = [("lidar", 90.0, "r3")]
    sim.rescue.deliver(r1, sim.t)
    assert sim.fleet.spares["lidar"] == 1
    assert run_until(sim, lambda: r2.state != "repairing", 60.0)
    assert r2.health.value("lidar") >= 99 and r2.health.value("camera") >= 99


# ====================================================================== salvage and leave, Gemma and the operator
def test_autonomous_gemma_salvages_a_wreck_and_spares_count_on_delivery():
    p = fake_planner()
    sim = make(planner=p)
    try:
        sim.run_for(10.0)
        r2 = sim.rovers[1]
        x2, y2 = flat_spot(sim, -10.0, 9.0)
        teleport(sim, r2, x2, y2, 0.0)
        sim.run_for(0.4)
        sim.fleet.spares = {"drive": 0, "lidar": 1, "camera": 1, "power": 1}    # the repair could not be done
        n0 = len(sim.fleet.events)
        sim.hazards.trigger("damage", "r2", severity="wreck")
        assert r2.health.status() == "wrecked"
        rec = sim.rescue.active("r2")
        assert run_until(sim, lambda: rec["phase"] in ("salvaging", "done"), 150.0), rec["phase"]
        dec = next(e["decision"] for e in events_since(sim, n0) if e.get("decision", {}).get("choice"))
        assert dec["choice"] == "salvage" and dec["by"] == "gemma" and dec["confidence"] is not None
        assert set(dec) == {"question", "options", "choice", "by", "confidence", "why"}
        assert run_until(sim, lambda: rec["phase"] == "done", 40.0)
        helper = sim.rescue._rover(rec["helper"])
        assert r2.state == "wrecked" and sim.fleet.stats()["wrecks"][0]["rover"] == "r2"
        taken = [k for k, _c, _w in helper.carried_spares]
        assert taken and all(k in ("lidar", "camera") for k in taken)
        before = dict(sim.fleet.spares)
        sim.rescue.deliver(helper, sim.t)
        for k in taken:
            assert sim.fleet.spares[k] == before[k] + taken.count(k)
        assert helper.carried_spares == []
        assert r2.health.value("lidar") == 0.0                                   # the part left the wreck
    finally:
        sim.close()
        p.close()


def test_supervised_rescue_request_marks_safe_options_and_the_answer_is_applied():
    sim = make(autonomy_mode="supervised")
    sim.run_for(10.0)
    r2 = sim.rovers[1]
    x2, y2 = flat_spot(sim, 10.0, 10.0)
    teleport(sim, r2, x2, y2, 0.0)
    sim.run_for(0.4)
    sim.hazards.trigger("damage", "r2", severity="disable", part="drive_l")
    rec = sim.rescue.active("r2")
    assert run_until(sim, lambda: rec.get("qid") is not None, 150.0), rec["phase"]
    q = next(q for q in sim.advisor.requests_msg() if q["id"] == rec["qid"])
    assert q["kind"] == "rescue" and q["about"] == "r2" and q["rover"] == rec["helper"]
    ids = [o["id"] for o in q["options"]]
    assert sorted(ids) == ["leave", "salvage", "tow"]
    assert all(isinstance(o["safe"], bool) and o["note"] for o in q["options"])
    assert next(o for o in q["options"] if o["id"] == q["default"])["safe"]
    assert next(o for o in q["options"] if o["id"] == "leave")["safe"]
    # an option the rules marked unsafe cannot be picked; the request stays open
    sim.advisor.requests[rec["qid"]]["meta"]["salvage"]["safe"] = False
    assert sim.advisor.answer(rec["qid"], "salvage") == "unsafe" and rec["qid"] in sim.advisor.requests
    assert sim.advisor.answer(rec["qid"], "leave") == "ok"
    assert r2.state == "wrecked" and rec["phase"] == "done" and rec["plan"] == "leave"
    assert sim.fleet.wrecks == [{"rover": "r2", "x": round(r2.x, 2), "y": round(r2.y, 2)}]


# ====================================================================== chat: health questions and rescue orders
FLEET = {"rovers": [{"id": "r1", "name": "Rover 1", "color": "#ff7a45", "x": 1.0, "y": 2.0, "state": "exploring"},
                    {"id": "r2", "name": "Rover 2", "color": "#36cfc9", "x": 3.0, "y": 4.0, "state": "disabled"},
                    {"id": "r3", "name": "Rover 3", "color": "#9254de", "x": 5.0, "y": 6.0, "state": "exploring"}],
         "targets": [], "lander": [0.0, 0.0], "bounds": [-60, -60, 60, 60]}
ORDERS = [
    ("rover 1, go help rover 2", {"type": "rescue", "rover": "r1", "target": "r2", "plan": "help"}),
    ("send rover 3 to rescue rover 1", {"type": "rescue", "rover": "r3", "target": "r1", "plan": "help"}),
    ("tow rover 2 home", {"type": "rescue", "rover": None, "target": "r2", "plan": "tow"}),
    ("r3, tow it home", {"type": "rescue", "rover": "r3", "target": None, "plan": "tow"}),
    ("salvage rover 3", {"type": "rescue", "rover": None, "target": "r3", "plan": "salvage"}),
    ("salvage it", {"type": "rescue", "rover": None, "target": None, "plan": "salvage"}),
    ("leave it", {"type": "rescue", "rover": None, "target": None, "plan": "leave"}),
    ("rover 2, are you damaged?", {"type": "question", "rover": "r2", "topic": "health"}),
    ("how is rover 3?", {"type": "question", "rover": "r3", "topic": "health"}),
    ("bring the orange one home", {"type": "return_home", "rover": "r1"}),
    ("leave the herd alone", None),
]


def test_rescue_orders_and_health_questions_parse():
    p = fake_planner()
    try:
        for text, want in ORDERS:
            got = g.parse_command(text, FLEET)["command"]
            if want is None:
                assert got["type"] != "rescue", (text, got)
                continue
            assert got == want, (text, got)
            tk = p.submit_command(1.0, text, FLEET)
            res = None
            for _ in range(300):
                res = p.poll(tk)
                if res is not None:
                    break
                time.sleep(0.01)
            assert res is not None and res.ok and res.command == want, (text, res)
        # a small model's rescue for a message without a rescue word is the keyword parser's reading
        req = g.make_command_request(1, 0.0, "go to 10, 5", FLEET)
        r = g.validate(req, json.dumps({"command": "rescue", "rover": "r1", "plan": "tow", "target_rover": "r2",
                                        "confidence": 0.9}))
        assert r.ok and r.command["type"] == "go_to"
        bad = g.validate(req, json.dumps({"command": "rescue", "rover": "r1", "plan": "fly", "confidence": 0.9}))
        assert not bad.ok and "plan" in bad.error
        system, user = g.build_prompt(g.make_command_request(1, 0.0, "rover 2 go to 10, -5", FLEET))
        assert "rescue" in system and g.estimate_tokens(system + user) < 400
        props = g.build_schema(req)["properties"]
        assert props["plan"]["enum"] == ["help", "tow", "salvage", "leave", "none"] and "health" in props["topic"]["enum"]
    finally:
        p.close()


def _say(sim, text, steps=15):
    n = len(sim.advisor.chat)
    sim.advisor.operator_message(text)
    for _ in range(steps):
        sim.step()
    return [m for m in sim.advisor.chat[n:] if m["from"] != "operator"]


def test_chat_health_answers_and_rescue_orders_in_the_sim():
    sim = make()
    sim.run_for(10.0)
    r1, r2, r3 = sim.rovers
    sim.hazards.trigger("damage", "r3", severity="light", part="camera")
    out = _say(sim, "rover 3, are you damaged?")
    rep = next(m for m in out if m["from"] == "r3" and m["kind"] == "reply")
    assert f"health is {r3.health.hp()}" in rep["text"] and "camera" in rep["text"]
    out = _say(sim, "how is rover 1?")
    assert any(m["from"] == "r1" and m["kind"] == "reply" and "health is 100" in m["text"] for m in out)
    x2, y2 = flat_spot(sim, 8.0, -8.0)
    teleport(sim, r2, x2, y2, 0.0)
    sim.run_for(0.4)
    sim.hazards.trigger("damage", "r2", severity="disable")
    rec = sim.rescue.active("r2")
    first = rec["helper"]
    other = "r3" if first == "r1" else "r1"
    out = _say(sim, f"rover {other[1]}, go help rover 2")
    assert rec["helper"] == other, (rec["helper"], out)
    assert any(m["from"] == other and m["kind"] == "ack" and "help r2" in m["text"] for m in out)
    out = _say(sim, "rover 2, how are you?")
    assert any(m["from"] == "r2" and "can't drive" in m["text"] for m in out), out
    out = _say(sim, "tow rover 2 home")
    assert rec["preset"] == "tow" or rec["plan"] == "tow", out
    out = _say(sim, "leave it")
    assert r2.state == "wrecked" and rec["phase"] == "done" and rec["plan"] == "leave"
    assert sim.fleet.stats()["wrecks"][0]["rover"] == "r2"
    out = _say(sim, "rover 2 go to 5, 5")
    assert any(m["from"] == "r2" and m["kind"] == "refuse" and "wreck" in m["text"] for m in out), out
    out = _say(sim, "salvage rover 2")
    rec2 = sim.rescue.active("r2")
    assert rec2 is not None and rec2["preset"] == "salvage" and rec2["helper"] is not None


# ====================================================================== lifeforms: a charge
def test_charge_is_seen_and_a_disabled_rover_gets_rammed():
    sim = make()
    sim.run_for(10.0)
    r2 = sim.rovers[1]
    x2, y2 = flat_spot(sim, 14.0, 14.0)
    teleport(sim, r2, x2, y2, 0.0)
    sim.run_for(0.4)
    r2.health.set_parts({"drive_r": 0.0})
    sim.rescue.check_now(r2)
    assert r2.state == "disabled"
    lf = sim.lifeforms
    herd = lf.herds[0]
    lf.move_herd(herd, x2 + 5.0, y2, face=math.pi)
    lf.provoke(herd, r2, sim.t + 1.0)
    hp0 = r2.health.hp()
    n0 = len(sim.fleet.events)
    assert run_until(sim, lambda: any(e.get("damage", {}).get("cause") == "lifeform" for e in events_since(sim, n0)),
                     15.0)
    assert any("rammed Rover 2" in e["msg"] for e in events_since(sim, n0))
    assert r2.health.hp() < hp0
    # a working rover sees a charge coming (a fast LiDAR track straight at it) and backs away
    sim = make()
    sim.run_for(10.0)
    r1 = sim.rovers[0]
    x1, y1 = flat_spot(sim, -12.0, 12.0)
    teleport(sim, r1, x1, y1, 0.0)
    r1.held = True                               # parked by the operator: only safety moves
    sim.run_for(0.4)
    lf = sim.lifeforms
    lf.move_herd(lf.herds[1], x1 + 5.0, y1, face=math.pi)
    lf.provoke(lf.herds[1], r1, sim.t + 2.0)
    n1 = len(sim.fleet.events)
    seen = run_until(sim, lambda: any("coming straight at me" in e["msg"] and e["rover"] == "r1"
                                      for e in events_since(sim, n1)), 12.0)
    sim.run_for(3.0)
    out = [e["msg"] for e in events_since(sim, n1)]
    assert seen or any("stopped short" in m for m in out), out


def test_no_em_dash_in_v9_files():
    for rel in ("redshift/health.py", "redshift/rescue.py", "redshift/hazards.py", "redshift/lifeforms.py",
                "redshift/rover.py", "redshift/advisor.py", "redshift/gemma.py", "redshift/answers.py",
                "tests/test_v9.py", "README.md", "GEMMA.md", "../PROTOCOL.md"):
        assert EM_DASH not in (ROOT / rel).read_text(encoding="utf-8"), rel
