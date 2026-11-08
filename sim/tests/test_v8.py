"""Protocol v8: autonomy mode (Gemma decides and logs, or the operator is asked everything), unexplored gaps as a
place kind and a fleet priority, the other fleet priorities, the re-survey focus, area questions, and the
operator command table (keyword parser and fake Gemma).

    python -m pytest -q tests/test_v8.py        (about 90 s)
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import gaps as G  # noqa: E402
from redshift import gemma as g  # noqa: E402
from redshift import places as P  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.fleet import PRIORITY_IN  # noqa: E402
from redshift.gemma import FakeBackend, GemmaConfig, GemmaPlanner  # noqa: E402
from redshift.hazards import HazardError  # noqa: E402
from redshift.mapping import FleetMap  # noqa: E402
from redshift.sim import Simulation  # noqa: E402

EM_DASH = chr(0x2014)
FLEET = {"rovers": [{"id": "r1", "name": "Rover 1", "color": "#ff7a45", "x": 3.2, "y": -1.0},
                    {"id": "r2", "name": "Rover 2", "color": "#36cfc9", "x": -8.5, "y": 12.1},
                    {"id": "r3", "name": "Rover 3", "color": "#9254de", "x": 15.0, "y": 4.4}],
         "targets": [{"id": "t4", "kind": "ice", "x": 12.0, "y": -4.5}, {"id": "t7", "kind": "vent", "x": -3.0, "y": 9.0}],
         "lander": [0.0, 0.0], "map_size": 120}
LEGACY = dict(map_size=80.0, water=False, spires=False, lifeforms=False, plan_clock="model")
DECISION_KEYS = {"question", "options", "choice", "by", "confidence", "why"}


# ====================================================================== helpers
def fake_planner(backend=None) -> GemmaPlanner:
    p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0),
                     backend=backend if backend is not None else FakeBackend(0.0))
    assert p.wait_ready(2.0)
    return p


def offline_planner() -> GemmaPlanner:
    p = GemmaPlanner(GemmaConfig(url="http://127.0.0.1:9", fallback_urls=(), check_every_s=3600.0,
                                 check_timeout_s=0.3), start=False)
    p.check_now()
    return p


def step_until(sim, cond, seconds):
    for _ in range(int(seconds / sim.agent_dt)):
        sim.step()
        if cond():
            return True
    return False


def chat_after(sim, seq):
    return [m for m in sim.advisor.chat if m["_seq"] > seq]


def replies_to(sim, cid):
    return [m for m in sim.advisor.chat if m["reply_to"] == cid]


def say(sim, text, seconds=4.0, rover=None):
    """Operator message, then run until its first reply. Returns that reply."""
    cid = sim.advisor.operator_message(text, rover)
    assert step_until(sim, lambda: replies_to(sim, cid), seconds), f"no reply to {text!r}"
    return replies_to(sim, cid)[0]


def decisions(sim, rid=None):
    return [e for e in sim.fleet.events if e.get("decision") and (rid is None or e["rover"] == rid)]


class Unsure(FakeBackend):
    """Gemma answers anomalies with a set confidence (and reroute)."""

    def __init__(self, conf):
        super().__init__(0.0)
        self.conf = conf

    def _interpret(self, req):
        out = super()._interpret(req)
        out["confidence"] = self.conf
        out["action"] = "proceed_slowly"
        return out


def call(fn, body: bytes):
    class Req:
        async def body(self):
            return body

    out = asyncio.run(fn(Req()))
    if hasattr(out, "status_code"):
        return out.status_code, json.loads(out.body)
    return 200, json.loads(json.dumps(out))


def spot(sim, x0, y0, rmin, rmax, avoid=()):
    fm, cost = sim.map, sim.fleet.cost(sim.t)
    best = None
    for a in range(0, 360, 10):
        for rr in np.arange(rmin, rmax, 1.0):
            x, y = x0 + rr * math.cos(math.radians(a)), y0 + rr * math.sin(math.radians(a))
            if math.hypot(x, y) < 8.0 or any(math.hypot(x - r.x, y - r.y) < 6.0 for r in sim.rovers):
                continue
            if any(math.hypot(x - ax, y - ay) < ar for ax, ay, ar in avoid):
                continue
            ok = 0
            for b in range(0, 360, 30):
                for d in (4.5, 5.5):
                    i, j = fm.ij(x + d * math.cos(math.radians(b)), y + d * math.sin(math.radians(b)))
                    ok += bool(fm.known[j, i] and np.isfinite(cost[j, i]))
            if best is None or ok > best[0]:
                best = (ok, round(float(x), 1), round(float(y), 1))
    assert best is not None and best[0] >= 12, best
    return best[1], best[2]


def punch_gap(sim, radius=3.5, rmin=10.0, rmax=22.0):
    """Make an unexplored pocket inside mapped, passable ground (as rovers leave between their tracks): forget a
    disc of the shared map. Returns its centre."""
    fm, cost = sim.map, sim.fleet.cost(sim.t)
    best = None
    for a in range(0, 360, 15):
        for rr in np.arange(rmin, rmax, 1.0):
            x, y = rr * math.cos(math.radians(a)), rr * math.sin(math.radians(a))
            if any(math.hypot(x - r.x, y - r.y) < radius + 4.0 for r in sim.rovers):
                continue
            kn, unk = fm.area_cells(x, y, radius + 3.0)
            ring = 0
            for b in range(0, 360, 30):
                i, j = fm.ij(x + (radius + 1.5) * math.cos(math.radians(b)), y + (radius + 1.5) * math.sin(math.radians(b)))
                ring += bool(fm.known[j, i] and np.isfinite(cost[j, i]))
            if unk == 0 and (best is None or ring > best[0]):
                best = (ring, x, y)
    assert best is not None and best[0] >= 10, best
    _r, x, y = best
    i, j = fm.ij(x, y)
    k = int(radius / fm.res)
    jj, ii = np.mgrid[j - k:j + k + 1, i - k:i + k + 1]
    disc = (ii - i) ** 2 + (jj - j) ** 2 <= k * k
    fm.known[jj[disc], ii[disc]] = False
    fm.known_count = int(fm.known.sum())
    sim.fleet.frontiers_t = -1e9
    sim.fleet.gaps_t = -1e9
    sim.fleet.invalidate_costs()
    sim.step()
    return x, y


# ====================================================================== 1. autonomy mode: settings, API, chat
def test_mode_defaults_env_api_and_chat(monkeypatch):
    assert Config().autonomy_mode == "autonomous"
    monkeypatch.delenv("AUTONOMY_MODE", raising=False)
    assert Config.from_env().autonomy_mode == "autonomous"
    monkeypatch.setenv("AUTONOMY_MODE", "supervised")
    assert Config.from_env().autonomy_mode == "supervised"
    monkeypatch.setenv("AUTONOMY_MODE", "nonsense")
    assert Config.from_env().autonomy_mode == "autonomous"
    from redshift import server
    old = server.state.runner
    server.state.runner = server.SimRunner(Config(seed=42, **LEGACY))      # not started
    try:
        sim = server.state.runner.sim
        assert sim.tick_message()["fleet"]["mode"] == "autonomous" and "priority" not in sim.fleet.stats()
        assert call(server.mode, b'{"mode": "supervised"}') == (200, {"ok": True, "mode": "supervised"})
        assert sim.fleet.mode == "supervised" and sim.init_message()["fleet"]["mode"] == "supervised"
        assert sim.advisor.chat[-1]["from"] == "fleet" and sim.advisor.chat[-1]["text"].startswith("Supervised mode")
        for bad in (b'{"mode": "manual"}', b"not json", b"{}"):
            assert call(server.mode, bad)[0] == 400
        assert call(server.mode, b'{"mode": "autonomous"}')[1]["mode"] == "autonomous"
        # the chat switches it too, with one reply
        m = say(sim, "switch to supervised mode")
        assert m["from"] == "fleet" and m["kind"] == "ack" and sim.fleet.mode == "supervised", m
        m = say(sim, "you decide from now on")
        assert sim.fleet.mode == "autonomous" and m["text"].startswith("Autonomous mode")
        m = say(sim, "let the robots decide")
        assert m["text"].startswith("Already in autonomous mode")
        m = say(sim, "ask me before you decide anything")
        assert sim.fleet.mode == "supervised"
        m = say(sim, "are you in autonomous mode?")
        assert m["kind"] == "reply" and "supervised mode" in m["text"].lower(), m
    finally:
        server.state.runner.sim.close()
        server.state.runner = old


# ====================================================================== 2. autonomous decisions
def test_autonomous_anomaly_is_decided_by_gemma_and_logged():
    p = fake_planner()
    sim = Simulation(Config(seed=42, **LEGACY), planner=p, wait_for_gemma=True)
    try:
        sim.run_for(12.0)
        r = next(o for o in sim.rovers if o.state == "exploring")
        seq = sim.advisor.chat_seq
        r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
        sim.step()
        assert not sim.advisor.requests, "the default (autonomous) never asks the operator"
        (e,) = decisions(sim, r.id)
        assert re.match(rf"^{r.id} decided \(Gemma 0\.80\): go around\. New rocks 4 m ahead, \d+\.\d m tall\.$", e["msg"]), e
        d = e["decision"]
        assert set(d) == DECISION_KEYS and d["choice"] == "reroute" and d["by"] == "gemma" and d["confidence"] == 0.8
        assert d["options"] == ["reroute", "proceed_slowly", "wait"] and re.search(r"\d", d["why"])
        assert e.get("source") == "gemma" and sim.fleet.public_event(e)["decision"] == d
        line = [m for m in chat_after(sim, seq) if m["from"] == r.id and m["kind"] == "info"]
        assert line and line[0]["text"].startswith("Decided (Gemma 0.80): go around. New rocks 4 m ahead")
        tick = sim.tick_message()
        assert any(ev.get("decision") for ev in tick["events"]) and tick["requests"] == []
        # the same decision again right away is logged but not repeated in the chat
        n_chat = len([m for m in sim.advisor.chat if m["kind"] == "info" and m["from"] == r.id])
        r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Again.")
        sim.step()
        assert len(decisions(sim, r.id)) == 2
        assert len([m for m in sim.advisor.chat if m["kind"] == "info" and m["from"] == r.id]) == n_chat
    finally:
        p.close()
        sim.close()


@pytest.mark.parametrize("conf,choice,by", [(0.4, "wait", "rules"), (0.52, "proceed_slowly", "gemma")])
def test_confidence_floor_takes_the_safest_option(conf, choice, by):
    p = fake_planner(Unsure(conf))
    sim = Simulation(Config(seed=42, **LEGACY), planner=p, wait_for_gemma=True)
    try:
        sim.run_for(12.0)
        r = next(o for o in sim.rovers if o.state == "exploring")
        r.on_anomaly(sim.t, "low_visibility", {"lidar_range": 2.9}, "Testing.")
        sim.step()
        (e,) = decisions(sim, r.id)
        assert e["decision"]["choice"] == choice and e["decision"]["by"] == by, e
        if by == "rules":
            assert "unsure" in e["decision"]["why"] and r.pause_until > sim.t        # waiting for the dust to clear
    finally:
        p.close()
        sim.close()


def test_rules_decide_when_gemma_is_offline():
    p = offline_planner()
    sim = Simulation(Config(seed=42, **LEGACY), planner=p)
    try:
        sim.run_for(12.0)
        r = next(o for o in sim.rovers if o.state == "exploring")
        r.on_anomaly(sim.t, "slip", {"slip": 0.8, "wheel_speed": 0.4, "ground_speed": 0.05}, "Testing.")
        sim.step()
        assert not sim.advisor.requests
        (e,) = decisions(sim, r.id)
        assert e["decision"]["by"] == "rules" and e["decision"]["confidence"] is None
        assert e["msg"].startswith(f"{r.id} decided (rules): back out and go around. Wheel slip 80%")
        assert "source" not in e
    finally:
        sim.close()


def test_autonomous_lifeform_sighting_follows_without_asking():
    p = fake_planner()
    sim = Simulation(Config(seed=42, plan_clock="model"), planner=p, wait_for_gemma=True)
    try:
        sim.run_for(5.0)
        for _ in range(100):
            try:
                sim.hazards.trigger("lifeform", "r1")
                break
            except HazardError:
                sim.step()
        assert step_until(sim, lambda: decisions(sim, "r1"), 20.0), [e["msg"] for e in sim.fleet.events][-6:]
        e = decisions(sim, "r1")[0]
        assert e["decision"]["choice"] == "follow" and e["decision"]["options"] == ["follow", "observe_here",
                                                                                   "keep_exploring"]
        assert e["msg"].startswith("r1 decided (Gemma 0.80): keep following. Possible lifeform")
        assert not sim.advisor.requests and sim.rovers[0].obs is not None and sim.rovers[0].obs["mode"] == "follow"
    finally:
        p.close()
        sim.close()


# ====================================================================== 3. supervised: every decision asks
def test_supervised_asks_every_anomaly_with_gemmas_suggestion_then_autonomous_decides_them():
    p = fake_planner()
    sim = Simulation(Config(seed=42, autonomy_mode="supervised", operator_ask_first=False, **LEGACY), planner=p,
                     wait_for_gemma=True)
    try:
        sim.run_for(12.0)
        rs = [o for o in sim.rovers if o.state == "exploring"][:2]
        assert len(rs) == 2
        for r in rs:                          # not only the first or uncertain ones: every anomaly
            r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
            sim.step()
        reqs = sim.advisor.requests_msg()
        assert len(reqs) == 2 and {q["rover"] for q in reqs} == {r.id for r in rs}
        for q in reqs:
            sug = [o for o in q["options"] if o.get("suggested")]
            assert [o["id"] for o in sug] == ["reroute"], q["options"]
            assert "Gemma suggests: go around (0.80)." in q["question"] and q["question"].endswith("What should I do?")
        # a second anomaly for a rover that is already waiting: noted, not decided behind the operator's back
        n_dec = len(decisions(sim, rs[0].id))
        rs[0].on_anomaly(sim.t, "slip", {"slip": 0.8, "wheel_speed": 0.4, "ground_speed": 0.05}, "Testing.")
        sim.step()
        assert len(sim.advisor.requests) == 2 and len(decisions(sim, rs[0].id)) == n_dec
        assert any("Still waiting for your answer" in e["msg"] for e in sim.fleet.events if e.get("rover") == rs[0].id)
        # "you decide" (no "from now on"): the open questions are decided now, the mode stays supervised
        m = say(sim, "you decide")
        assert "deciding the 2 open questions myself" in m["text"] and sim.fleet.mode == "supervised"
        assert not sim.advisor.requests and all(sim.advisor.closed[q["id"]] == "decided" for q in reqs)
        per_q = [m_ for m_ in sim.advisor.chat if m_["request_id"] in {q["id"] for q in reqs} and m_["kind"] == "info"]
        assert len(per_q) == 2 and all(m_["text"].startswith("Decided (Gemma 0.80): go around") for m_ in per_q)
        # switching to autonomous with a question open decides it at once
        r = rs[1]
        sim.run_for(1.0)
        r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
        sim.step()
        (q,) = sim.advisor.requests_msg()
        assert sim.advisor.set_mode("autonomous") == "ok" and sim.fleet.mode == "autonomous"
        assert not sim.advisor.requests and sim.advisor.answer(q["id"], "reroute") == "closed"
        assert decisions(sim, r.id)[-1]["decision"]["choice"] == "reroute" and not r.question_hold
    finally:
        p.close()
        sim.close()


# ====================================================================== 4. places: Gemma picks, or suggests
def _water_world(planner=None, mode="autonomous"):
    sim = Simulation(Config(seed=42, plan_clock="model", lifeforms=False, focus_s=40.0, curiosity=False,
                            autonomy_mode=mode), planner=planner, wait_for_gemma=planner is not None)
    sim.run_for(40.0)
    f = sim.fleet
    px, py = spot(sim, 0.0, 0.0, 9.0, 16.0)
    f.add_water_zone(sim.t, sim.rovers[0], px, py, 2.0)
    lx, ly = spot(sim, 0.0, 0.0, 9.0, 18.0, avoid=[(px, py, 16.0)])
    f.add_water_zone(sim.t + 1.0, sim.rovers[1], lx, ly, 3.0)
    sim.step()
    return sim


@pytest.mark.parametrize("gemma", [False, True])
def test_autonomous_place_choice_is_decided_and_said_in_the_ack(gemma):
    p = fake_planner() if gemma else None
    sim = _water_world(p)
    try:
        seq = sim.advisor.chat_seq
        m = say(sim, "go to the water source", 6.0)
        assert not sim.advisor.requests and m["kind"] == "ack", [x["text"] for x in chat_after(sim, seq)]
        who = "Gemma picked it, 0.80" if gemma else "my rules picked it"
        assert re.match(rf"I'm the closest free rover \(\d+ m\)\. I know 2 bodies of water; going for the (pond|lake) "
                        rf"about \d+ m across, .*\({who}\)\. Heading to the (pond|lake)'s shore", m["text"]), m["text"]
        e = decisions(sim, m["from"])[-1]
        assert e["decision"]["options"] == ["p1", "p2"] and e["decision"]["by"] == ("gemma" if gemma else "rules")
    finally:
        sim.close()
        if p is not None:
            p.close()


def test_supervised_place_request_shows_gemmas_suggestion():
    p = fake_planner()
    sim = _water_world(p, mode="supervised")
    try:
        sim.advisor.operator_message("go to the water source")
        assert step_until(sim, lambda: sim.advisor.requests, 3.0)
        (q,) = sim.advisor.requests_msg()
        assert q["kind"] == "choose_place" and [o["id"] for o in q["options"] if o.get("suggested")] == ["p1"]
        assert re.search(r"Which one\? Gemma suggests: (pond|lake) about \d+ m across, .+ \(0\.80\)\.$", q["question"])
    finally:
        p.close()
        sim.close()


# ====================================================================== 5. unexplored gaps
def _flood_labels(mask):
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    for j, i in zip(*np.nonzero(mask)):
        if lab[j, i]:
            continue
        n += 1
        stack = [(j, i)]
        lab[j, i] = n
        while stack:
            a, b = stack.pop()
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c, d = a + da, b + db
                if 0 <= c < mask.shape[0] and 0 <= d < mask.shape[1] and mask[c, d] and not lab[c, d]:
                    lab[c, d] = n
                    stack.append((c, d))
    return lab, n


def test_gap_detection_unit():
    rng = np.random.default_rng(3)
    mask = rng.random((40, 50)) < 0.45
    lab, n = G.label_runs(mask)
    ref, n_ref = _flood_labels(mask)
    assert n == n_ref and ((lab > 0) == mask).all()
    for k in range(1, n_ref + 1):           # the same partition, whatever the numbering
        assert len(np.unique(lab[ref == k])) == 1
    cfg = Config(seed=1)
    fm = FleetMap(cfg)
    fm.known[:] = True
    fm.known[:30, :] = False                 # the open unknown along the south edge: not a gap
    fm.known[100:116, 100:116] = False       # an 8 m x 8 m pocket: a gap
    fm.known[150:153, 60:63] = False         # 2.25 m2: too small
    fm.known[60:72, 180:192] = False         # a pocket joined to the open unknown by a 1 m corridor: still a gap
    fm.known[30:60, 185:187] = False
    gaps, grid = G.find_gaps(fm, [])
    assert len(gaps) == 2, gaps
    big = next(g_ for g_ in gaps if abs(g_["x"] - fm.xy(108, 108)[0]) < 2.0)
    assert 60.0 <= big["m2"] <= 66.0 and big["enclosed"] == 1.0 and 2.5 <= big["r"] <= 5.5
    assert G.gap_at(grid, fm, *fm.xy(108, 108)) == big["lab"] and G.gap_at(grid, fm, *fm.xy(5, 5)) == 0
    t0 = time.perf_counter()
    for _ in range(5):
        G.find_gaps(fm, [])
    assert (time.perf_counter() - t0) / 5 < 0.05


@pytest.fixture(scope="module")
def gap_world():
    """The 80 m map after 60 s, with an unexplored pocket punched into mapped ground; the keyword parser."""
    sim = Simulation(Config(seed=42, curiosity=False, focus_s=40.0, **LEGACY))
    sim.run_for(60.0)
    gx, gy = punch_gap(sim)
    yield sim, (gx, gy)
    sim.close()


def test_gaps_place_and_fleet_priority(gap_world):
    sim, (gx, gy) = gap_world
    f, adv = sim.fleet, sim.advisor
    cands = P.candidates(sim, "gaps")
    assert cands and any(math.hypot(c.x - gx, c.y - gy) < 3.0 for c in cands), [(c.x, c.y) for c in cands]
    c = min(cands, key=lambda c_: math.hypot(c_.x - gx, c_.y - gy))
    assert c.kind == "gaps" and c.noun().startswith("Unexplored gap about") and c.dest().startswith("the unexplored gap at")
    # "can you explore one of those holes on the map": the closest free rover explores the best gap
    m = say(sim, "can you explore one of those holes on the map")
    assert m["kind"] == "ack" and "Exploring around the unexplored gap at" in m["text"] and "m2 of unknown ground" in m["text"]
    r = adv._rover(m["from"])
    assert r.focus is not None and r.focus["mode"] == "explore" and r.focus["label"].startswith("the unexplored gap")
    say(sim, "resume")
    # "no like there are multiple holes ... I want those prioritized": a fleet priority with a plan per rover
    m = say(sim, "no like there are multiple holes in the map that are unexplored, I want those prioritized")
    assert m["from"] == "fleet" and m["kind"] == "ack", m
    assert re.match(r"Prioritizing \d+ unexplored gaps? inside the map.*: r\d( and r\d)* takes? the one at \(-?\d+, -?\d+\) "
                    r"\(\d+ m2, [\d and]+ m away\)", m["text"]), m["text"]
    assert f.priority is not None and f.priority["kind"] == "gaps"
    pm = sim.tick_message()["fleet"]["priority"]
    assert {"kind", "label", "count", "since"} <= set(pm) <= {"kind", "label", "count", "since", "places"}
    assert pm["kind"] == "gaps" and pm["count"] >= 1
    assert all({"x", "y", "r"} == set(q) for q in pm.get("places", []))
    # every free rover's frontier choice now weighs the gaps strongly
    rv = next(o for o in sim.rovers if o.state == "exploring" and o.focus is None and o.op_goal is None)
    res = f.choose_goal(sim.t, rv)
    assert res is not None
    top = rv.goal_options[0]
    idx = next(fr[3] for fr in f.frontiers if (fr[0], fr[1]) == (top[1], top[2]))
    assert f.gap_grid.ravel()[idx] > 0 and f.priority_bonus(rv, top[1], top[2], idx) >= PRIORITY_IN
    # the gaps get mapped: the priority ends, announced in the chat
    seq = adv.chat_seq
    f.map.known[:] = f.map.known | (f.gap_grid > 0)
    f.gaps_t = -1e9
    assert step_until(sim, lambda: f.priority is None, 8.0)
    done = [x for x in chat_after(sim, seq) if x["text"].startswith("Done: no unexplored gaps left")]
    assert done and done[0]["from"] == "fleet" and done[0]["reply_to"] == m["reply_to"]
    assert "priority" not in sim.tick_message()["fleet"]


def test_other_priorities_and_back_to_normal(gap_world):
    sim, _ = gap_world
    f = sim.fleet
    m = say(sim, "prioritize the north")
    if m["kind"] == "ack":
        assert m["text"].startswith("Prioritizing the north: ") and f.priority["direction"] == "north"
        fr = next(fr for fr in f.usable_frontiers(sim.t) if f.in_direction(fr[0], fr[1], "north"))
        assert f.priority_bonus(sim.rovers[0], fr[0], fr[1], fr[3]) == PRIORITY_IN
        assert f.priority_bonus(sim.rovers[0], 0.0, -30.0, fr[3]) == 0.0
    m = say(sim, "prioritize water")                   # no water in this world: low ground first
    assert m["text"].startswith(("Replacing the priority", "No water found yet")) and "low ground" in m["text"], m
    assert f.priority["kind"] == "water" and f.priority["low_h"] is not None
    m = say(sim, "back to normal")
    assert m["text"].startswith("Back to normal exploration: dropped the priority") and f.priority is None
    m = say(sim, "back to normal")                     # nothing to drop: normal exploration, short and clear
    assert m["from"] == "fleet" and "resum" in m["text"].lower(), m
    ts = min(f.targets, key=lambda ts_: math.hypot(ts_.spec.x, ts_.spec.y))
    if ts.status is None:
        ts.status, ts.found_by, ts.found_t = "found", "r1", sim.t
    m = say(sim, "prioritize sampling")
    assert m["text"].startswith("Prioritizing sampling: ") and "90 m" in m["text"] and f.priority["kind"] == "sampling"
    assert re.search(r"r\d (takes|is already on) t\d+", m["text"]) or "caches" in m["text"], m["text"]
    m = say(sim, "cancel the priority")
    assert f.priority is None and m["text"].startswith("Back to normal exploration")


# ====================================================================== 6. re-survey
def test_operator_focus_on_mapped_ground_is_a_resurvey_then_area_question():
    sim = Simulation(Config(seed=42, curiosity=False, focus_s=40.0, **LEGACY))
    try:
        sim.run_for(40.0)
        r = sim.rovers[0]
        pts = [(x, y) for x in range(-12, 13, 3) for y in range(-12, 13, 3) if math.hypot(x, y) > 3.0]
        x, y = min(pts, key=lambda p_: sim.map.area_cells(p_[0], p_[1], 8.0)[1] - 0.01 * math.hypot(*p_))
        assert r.mostly_mapped(x, y, 8.0)
        adv = sim.advisor
        # through the command path the ack says re-survey (radius from the config: use a small circle here)
        seq = adv.chat_seq
        adv._reply_to = None
        adv._cmd_area("r1", float(x), float(y), f"({x}, {y})", sim.t, "rules", radius=8.0)
        ack = chat_after(sim, seq)[-1]
        assert re.match(rf"\({x}, {y}\) is already \d+% mapped, so I'll re-survey it for about 40 s: a loop through it "
                        rf"within 8 m", ack["text"]), ack["text"]
        assert r.focus["mode"] == "resurvey" and sim.tick_message()["rovers"][0]["focus"]["mode"] == "resurvey"
        goals = set()

        def done():
            if r.focus is not None and r.goal is not None:
                goals.add(r.goal)
            return r.focus is None
        assert step_until(sim, done, 120.0)
        s = r.mem.focus[-1]
        assert s["mode"] == "resurvey" and s["why"] == "time" and s["dur"] >= 39.0, s     # the whole time
        assert s["driven"] > 5 and len(goals) >= 2                                         # a loop, not standing still
        assert all(math.hypot(gx - x, gy - y) <= 8.0 for gx, gy in goals)
        rep = [m for m in adv.chat if m["from"] == "r1" and m["text"].startswith("Re-surveyed")][-1]
        assert re.match(rf"Re-surveyed \({x}, {y}\) for \d+ s: drove \d+ m through it", rep["text"]), rep["text"]
        assert "No changes since it was mapped" in rep["text"] or "Changes:" in rep["text"]
        # "what did we find there?": the fleet answers about that place only
        m = say(sim, "what did we find there")
        assert m["from"] == "fleet" and m["kind"] == "reply", m
        assert f"({x}, {y})" in m["text"] and "re-surveyed" in m["text"] and "of 20 targets" not in m["text"], m["text"]
        assert "targets: " not in m["text"].lower() or "within 15 m" in m["text"]
        # an explore focus that finds nothing to explore right after arriving turns into a re-survey
        r2 = sim.rovers[1]
        r2.start_focus(sim.t, float(x), float(y), 3.0, "operator", "the test spot", 30.0, mode="explore")
        assert r2.focus["mode"] == "explore"
        r2.focus["arrived"], r2.focus["until"], r2.focus["next_check"] = sim.t, sim.t + 30.0, sim.t
        r2._focus_step(sim.t)
        assert r2.focus is not None and r2.focus["mode"] == "resurvey"
        assert any("re-surveying it instead" in e["msg"] for e in sim.fleet.events if e["rover"] == r2.id)
    finally:
        sim.close()


def test_area_questions_are_scoped_to_the_place():
    p = fake_planner()
    sim = _water_world(p)
    seen = []
    orig = p.submit_answer

    def rec(speaker, t, question, topic, facts, default, timeout_s=None, context=None):
        seen.append((speaker, question, list(facts), default))
        return orig(speaker, t, question, topic, facts, default, timeout_s=timeout_s, context=context)

    p.submit_answer = rec
    try:
        f = sim.fleet
        lake = max(P.candidates(sim, "water"), key=lambda c: c.size)
        near = [ts for ts in f.targets if math.hypot(ts.spec.x - lake.x, ts.spec.y - lake.y) < 14.0]
        far = [ts for ts in f.targets if math.hypot(ts.spec.x - lake.x, ts.spec.y - lake.y) > 40.0]
        for ts in near[:1] + far[:2]:
            ts.status, ts.found_by, ts.found_t = "found", "r2", sim.t
        m = say(sim, "anything near the lake?")
        speaker, _q, facts, rules = seen[-1]
        assert any("the lake there is about" in ln or "pond there" in ln for ln in facts), facts
        assert all(ts.spec.id not in " ".join(facts) for ts in far[:2]), facts
        if near:
            assert near[0].spec.id in " ".join(facts)
        assert len(facts) <= 7 and m["kind"] == "reply"
        # "we" questions are the fleet's
        m = say(sim, "what did we find there?")
        assert m["from"] == "fleet" and seen[-1][0] == "fleet"
    finally:
        p.close()
        sim.close()


# ====================================================================== 7. the operator command table
TABLE = [
    ("explore water terrain", "go_to_place", {"place": "water", "explore": True}),
    ("can you explore one of those holes on the map", "go_to_place", {"place": "gaps", "explore": True}),
    ("no like there are multiple holes in the map that are unexplored, I want those prioritized", "prioritize",
     {"priority": "gaps"}),
    ("what did we find there", "question", {"topic": "finds"}),
    ("explore the dark patch near rover 2", "go_to_place", {"place": "gaps", "near_rover": "r2"}),
    ("go look at the crystals", "go_to_place", {"place": "spire"}),
    ("check the area east of the lake", "go_to_place", {"place": "water", "direction": "east", "explore": True}),
    ("send someone to the vent", "go_to_place", {"place": "vent"}),
    ("prioritize sampling", "prioritize", {"priority": "sampling"}),
    ("back to normal", "prioritize", {"priority": "none"}),
    ("switch to supervised", "set_mode", {"mode": "supervised"}),
    ("you decide", "set_mode", {"mode": "autonomous"}),
    ("what's rover 3 doing", "question", {"rover": "r3", "topic": "status"}),
    ("why did you go there", "question", {"topic": "why"}),
    ("go back to where you saw the creatures", "go_to_place", {"place": "lifeform"}),
    ("survey the lander area", "go_to_place", {"place": "lander", "investigate": True}),
    ("focus on the unexplored holes first", "prioritize", {"priority": "gaps"}),
    ("prioritize the north", "prioritize", {"priority": "direction", "direction": "north"}),
    ("prioritize water", "prioritize", {"priority": "water"}),
    ("cancel the priority", "prioritize", {"priority": "none"}),
    ("you decide from now on", "set_mode", {"mode": "autonomous"}),
    ("ask me before you decide anything", "set_mode", {"mode": "supervised"}),
    ("what's at the vent?", "question", {}),
    ("anything near the lake?", "question", {"topic": "water"}),
    ("fill in the gaps", "prioritize", {"priority": "gaps"}),
    ("map the blank spots", "prioritize", {"priority": "gaps"}),
    ("go to the biggest gap", "go_to_place", {"place": "gaps", "qualifier": "biggest"}),
    ("explore the unexplored area east of the lake", "go_to_place", {"place": "water", "direction": "east"}),
    ("focus on the lake", "go_to_place", {"place": "water", "explore": True}),
    ("check on rover 1", "question", {"rover": "r1", "topic": "status"}),
    ("rover 2, explore the gaps", "go_to_place", {"place": "gaps", "rover": "r2"}),
    ("explore the shiny stuff", "unclear", {}),
    ("keep exploring", "resume", {}),
    ("what's left to explore?", "question", {"topic": "exploration"}),
    ("rover 2, go check out the area north of the lander", "explore_direction", {"direction": "north"}),
    ("investigate the lake for signs of life", "go_to_place", {"place": "water", "investigate": True}),
]


def _check(cmd, typ, want, text):
    assert cmd["type"] == typ, (text, cmd)
    for k, v in want.items():
        assert cmd.get(k) == v, (text, k, cmd)


def test_command_table_keyword_parser():
    for text, typ, want in TABLE:
        _check(g.parse_command(text, FLEET)["command"], typ, want, text)


def test_command_table_fake_gemma():
    p = fake_planner()
    try:
        for text, typ, want in TABLE:
            tk = p.submit_command(1.0, text, FLEET)
            res = None
            for _ in range(3000):
                res = p.poll(tk)
                if res is not None:
                    break
                threading.Event().wait(0.001)
            assert res is not None and res.ok and res.source == "gemma", (text, res and res.error)
            _check(res.command, typ, want, text)
    finally:
        p.close()


def _gemma(text, **answer):
    req = g.make_command_request(1, 10.0, text, FLEET)
    base = {"command": "unclear", "rover": "none", "place": "none", "qualifier": "none", "ordinal": 0,
            "near_rover": "none", "investigate": False, "explore": False, "direction": "none", "priority": "none",
            "mode": "none", "x": 0, "y": 0, "radius": 0, "target_id": "none", "topic": "none", "question": "", "ack": "",
            "confidence": 0.9}
    return g.validate(req, json.dumps(dict(base, **answer)))


def test_gemma_checks_for_priorities_modes_and_resumes():
    # what a 2B model gets wrong: the operator's words win
    r = _gemma("can you explore one of those holes on the map", command="resume")
    assert r.ok and r.command["type"] == "go_to_place" and r.command["place"] == "gaps"
    r = _gemma("no like there are multiple holes in the map that are unexplored, I want those prioritized",
               command="resume", rover="r1")
    assert r.ok and r.command == {"type": "prioritize", "rover": None, "priority": "gaps", "direction": None}
    r = _gemma("explore the shiny stuff", command="resume")
    assert r.ok and r.command["type"] == "unclear" and r.command["question"].startswith("Explore what?")
    r = _gemma("prioritize the north", command="prioritize", priority="direction", direction="north")
    assert r.ok and r.command["direction"] == "north" and r.ack == "Prioritizing the north."
    # a priority or a mode the words do not ask for is not accepted
    r = _gemma("go to the lake", command="prioritize", priority="water")
    assert not r.ok and "does not ask for a priority" in r.error and r.command["type"] == "go_to_place"
    r = _gemma("what is the battery level of rover 2", command="set_mode", mode="supervised")
    assert r.command["type"] == "question"
    r = _gemma("hello there", command="set_mode", mode="supervised")
    assert not r.ok and "set_mode" in r.error
    r = _gemma("prioritize the volcano", command="prioritize", priority="volcano")
    assert not r.ok and "priority" in r.error
    # schema and prompt stay small for the 2B model
    req = g.make_command_request(1, 0.0, "can we investigate the water source again for any signs of life", FLEET)
    props = g.build_schema(req)["properties"]
    assert props["priority"]["enum"] == list(g.PRIORITIES) and props["mode"]["enum"] == ["autonomous", "supervised", "none"]
    assert "gaps" in props["place"]["enum"] and {"prioritize", "set_mode"} <= set(props["command"]["enum"])
    system, user = g.build_prompt(req)
    assert "prioritize" in system and "set_mode" in system and g.estimate_tokens(system + user) < 400


def test_no_em_dash_in_v8_files():
    for rel in ("redshift/gaps.py", "redshift/advisor.py", "redshift/gemma.py", "redshift/fleet.py", "redshift/rover.py",
                "redshift/answers.py", "redshift/places.py", "redshift/server.py", "tests/test_v8.py", "README.md",
                "GEMMA.md", "../PROTOCOL.md"):
        assert EM_DASH not in (ROOT / rel).read_text(encoding="utf-8"), rel
