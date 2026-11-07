"""Tier 2 of the two-tier brain (redshift/advisor.py): Gemma advice, anomaly checks, operator
questions and chat commands, with the fake model (no Ollama needed) and the rule-based parser.

    python -m pytest -q tests/test_brain.py        (about 60 s)

Most runs use wait_for_gemma=True (tests only): each Gemma answer is awaited before the sim goes
on, so the runs repeat exactly. Two runs (offline planner, performance) use the real, non-blocking
behaviour.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import gemma as g  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.gemma import FakeBackend, GemmaConfig, GemmaPlanner  # noqa: E402
from redshift.hazards import HazardError  # noqa: E402
from redshift.sim import Simulation  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + bytes(300) + b"\xff\xd9"      # stands in for a camera frame
EM_DASH = chr(0x2014)
PLANNER_KEYS = {"model", "state", "online", "calls", "ok", "fallbacks", "avg_ms", "p95_ms", "last_ms", "budget_ms",
                "over_budget"}
CHAT_KEYS = {"id", "t", "from", "to", "text", "kind", "source", "request_id"}
REQUEST_KEYS = {"id", "rover", "t", "question", "options", "default", "expires_in", "image"}
KINDS = {"command", "ack", "refuse", "question", "answer", "info"}


# ====================================================================== helpers
class RecordingFake(FakeBackend):
    """The fake model, plus a record of what it was asked (and whether a picture came with it)."""

    def __init__(self, latency_s=0.0):
        super().__init__(latency_s)
        self.asked = []

    def chat(self, req, system, user, schema, timeout):
        self.asked.append((req.kind, req.rover_id, req.image_b64 is not None, req.payload))
        return super().chat(req, system, user, schema, timeout)


class PickAnother(RecordingFake):
    """Always prefers a frontier other than the rules' pick (f1), quoting its numbers."""

    def _choice(self, req):
        cands = req.payload["candidates"]
        pick = next((c for c in cands if c["id"] != "f1" and c["kind"] == "frontier"), cands[-1])
        nums = [p.strip() for p in pick["summary"].split(",") if re.search(r"slope|battery", p)]
        return {"reason": f"{pick['id']}: " + ", ".join(nums), "choice": pick["id"], "confidence": 0.8}


class AlwaysAgree(RecordingFake):
    def _choice(self, req):
        c = req.payload["candidates"][0]
        nums = [p.strip() for p in c["summary"].split(",") if "battery" in p]
        return {"reason": f"f1 is fine, {nums[0]}", "choice": "f1", "confidence": 0.9}


class LikeGemmaOnSlip(RecordingFake):
    """What gemma4:e2b really answered for wheel slip before the prompt fix: right class, unsafe action."""

    def _interpret(self, req):
        return {"classification": "soft_ground", "reason": "Slip 82% while ground speed 0.11 m/s",
                "action": "proceed_slowly", "confidence": 0.85}


def fake_planner(backend=None) -> GemmaPlanner:
    p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0),
                     backend=backend if backend is not None else RecordingFake())
    assert p.wait_ready(2.0), p.stats()
    return p


def make_sim(planner=None, seed=42, wait=True, frames=True, **cfg) -> Simulation:
    return Simulation(Config(seed=seed, plan_clock="model", **cfg), planner=planner,
                      frame_source=(lambda rid: JPEG) if frames else None, wait_for_gemma=wait)


def gemma_events(sim, part=""):
    return [e for e in sim.fleet.events if e.get("source") == "gemma" and part in e["msg"]]


def replies(sim, since: int = 0):
    return [m for m in sim.advisor.chat if m["_seq"] > since and m["from"] != "operator"]


def say(sim, text, steps=10, rover=None):
    """Operator message, then run until its one reply arrives. Returns that reply."""
    seq = sim.advisor.chat_seq
    sim.advisor.operator_message(text, rover)
    for _ in range(steps):
        sim.step()
        got = [m for m in replies(sim, seq) if not m["text"].endswith("thanks.")]
        if got:
            return got[0]
    raise AssertionError(f"no reply to {text!r}")


def trigger(sim, kind, rover=None, tries=200):
    for _ in range(tries):
        try:
            return sim.hazards.trigger(kind, rover)
        except HazardError:
            sim.run_for(0.4)
    raise AssertionError(f"could not place {kind}")


# ====================================================================== 1. goal choice
def test_provisional_goal_then_gemma_switch():
    p = fake_planner(PickAnother())
    sim = make_sim(p)
    seen = []
    orig = p.submit_choice

    def rec(rid, t, perception, cands, default=None):
        r, job = sim.advisor._rover(rid), sim.advisor.choices[rid]
        seen.append((rid, r.goal == job.rule_goal and bool(r.path), default, [c["id"] for c in cands], cands))
        return orig(rid, t, perception, cands, default)

    p.submit_choice = rec
    try:
        sim.run_for(80.0)
    finally:
        p.close()
        sim.close()
    assert seen, "nobody asked Gemma"
    assert all(ok and default == "f1" for _, ok, default, _, _ in seen), "a rover waited for Gemma"
    for _, _, _, ids, cands in seen:
        assert ids[0] == "f1" and 2 <= len(ids) <= 5
        for c in cands:
            if c["kind"] == "frontier":     # A* length, max slope, battery after return
                assert re.search(r"path \d+ m, slope max \d+ deg, battery after return -?\d+%", c["summary"]), c
    sw = gemma_events(sim, "switching to")
    assert sw, "Gemma never changed a goal"
    for e in sw:
        assert re.match(rf"{e['rover']} switching to (frontier|sample) .+ \(Gemma, \d+\.\d s\): '.*\d+.*'$", e["msg"]), e
        assert "at" in e
    # one event per switch, never more than the questions asked
    assert len(sw) <= len(seen)
    # the asks are rate limited: about every 20 s per rover, or when a goal is reached
    for rid in ("r1", "r2", "r3"):
        assert sum(1 for s in seen if s[0] == rid) <= 80.0 / 5.0


def test_agreement_logged_at_most_once_a_minute():
    p = fake_planner(AlwaysAgree())
    sim = make_sim(p)
    try:
        sim.run_for(150.0)
    finally:
        p.close()
        sim.close()
    asked = [a for a in p.backend.asked if a[0] == "choice"]
    agree = gemma_events(sim, "Gemma agrees with")
    assert agree and len(asked) > len(agree), (len(asked), len(agree))
    for rid in ("r1", "r2", "r3"):
        ts = [e["t"] for e in agree if e["rover"] == rid]
        assert all(b - a >= 60.0 - 1e-6 for a, b in zip(ts, ts[1:])), (rid, ts)
    assert not gemma_events(sim, "switching to")


# ====================================================================== 2. anomalies
def test_anomaly_goes_to_gemma_with_the_camera_frame():
    backend = RecordingFake()
    p = fake_planner(backend)
    sim = make_sim(p, operator_ask_first=False)
    try:
        sim.run_for(12.0)
        r = sim.rovers[0]
        assert r.state in ("exploring", "investigating")
        r.on_anomaly(sim.t, "slip", {"slip": 0.82, "wheel_speed": 0.61, "ground_speed": 0.11}, "Testing.")
        sim.step()
    finally:
        p.close()
        sim.close()
    kind, rid, image, payload = [a for a in backend.asked if a[0] == "interpretation"][-1]
    assert rid == "r1" and image, "the camera frame did not reach Gemma"
    assert payload["anomaly"]["type"] == "wheel_slip" and payload["anomaly"]["slip"] == 0.82
    ev = gemma_events(sim, "soft ground")
    assert ev and ev[-1]["rover"] == "r1" and ev[-1]["msg"].startswith("r1 Gemma (camera and sensors, ")
    assert "82%" in ev[-1]["msg"] and "0.11 m/s" in ev[-1]["msg"]
    assert "Agrees with the rules: backing out" in ev[-1]["msg"] and sim.advisor.frames == 1
    assert not sim.advisor.requests


def test_unsafe_gemma_action_never_beats_a_safety_reaction():
    """Real sand: the rules back out at once; Gemma's (old, wrong) 'proceed_slowly' is refused."""
    p = fake_planner(LikeGemmaOnSlip())
    sim = make_sim(p, operator_ask_first=False)
    try:
        sim.run_for(30.0)
        h = trigger(sim, "sand")
        rid = h["rover"]
        r = sim.advisor._rover(rid)
        for _ in range(int(40 / 0.2)):
            sim.step()
            if gemma_events(sim, "soft ground"):
                break
        ev = gemma_events(sim, "soft ground")
        assert ev, "slip never reached Gemma"
        assert "Safety rule wins: no driving on into soft ground" in ev[0]["msg"], ev[0]
        assert r.state == "recovering" and r.rec_phase == "sand"
        sim.run_for(30.0)
    finally:
        p.close()
        sim.close()
    out = [e for e in sim.fleet.events if e["rover"] == rid and ("backed out of the soft ground" in e["msg"]
                                                                  or "dug itself out" in e["msg"])]
    assert out, "the rover did not get out of the sand"


# ====================================================================== 3. questions
def test_first_anomaly_asks_the_operator_and_the_answer_is_applied():
    p = fake_planner()
    sim = make_sim(p)
    try:
        sim.run_for(40.0)
        h = trigger(sim, "rockfall")
        rid = h["rover"]
        for _ in range(int(30 / 0.2)):
            sim.step()
            if sim.advisor.requests:
                break
        msg = sim.tick_message()
        assert len(msg["requests"]) == 1, "no question opened"
        q = msg["requests"][0]
        assert set(q) == REQUEST_KEYS and q["rover"] == rid
        ids = [o["id"] for o in q["options"]]
        assert 2 <= len(ids) <= 4 and q["default"] == "reroute" == ids[0] and set(ids) <= set(g.ACTIONS)
        assert q["image"] == f"/api/camera/{rid}.jpg?view=fpv" and 0 < q["expires_in"] <= 20.0
        assert "block my path" in q["question"] and "Gemma thinks: rock" in q["question"]
        ask = [m for m in msg["chat"] if m["request_id"] == q["id"]]
        assert ask and ask[0]["kind"] == "question" and not ask[0]["text"].startswith(f"{rid}")
        assert ask[0]["source"] == "gemma"
        r = sim.advisor._rover(rid)
        speeds = []
        for _ in range(15):
            sim.step()
            if r.state in ("exploring", "investigating", "returning"):
                speeds.append(r.cmd_v)
        assert speeds and max(speeds) <= 0.25 + 1e-9, speeds        # creeping while it waits
        assert sim.advisor.answer(q["id"], "teleport") == "bad_option"
        assert sim.advisor.answer(q["id"], "wait") == "ok"
        assert sim.advisor.answer(q["id"], "wait") == "closed" and sim.advisor.answer("q99", "wait") == "unknown"
        chat = sim.tick_message()["chat"]
        assert [m["text"] for m in chat if m["kind"] == "answer"] == ["Wait"]
        assert "Waiting 6 s, then going around, thanks." in [m["text"] for m in chat]
        assert r.pause_until > sim.t and r.crawl_until <= sim.t and not sim.advisor.requests
        assert [e for e in sim.fleet.events if e.get("source") == "operator" and "operator chose 'Wait'" in e["msg"]]
    finally:
        p.close()
        sim.close()


def test_question_expires_to_the_safe_default():
    p = fake_planner()
    sim = make_sim(p)
    now = [1000.0]
    sim.advisor.clock = lambda: now[0]
    try:
        sim.run_for(12.0)
        r = next(r for r in sim.rovers if r.state == "exploring")
        r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
        sim.step()
        assert len(sim.advisor.requests) == 1
        qid = next(iter(sim.advisor.requests))
        now[0] += 19.0
        sim.step()
        assert sim.advisor.requests_msg()[0]["expires_in"] == 1.0
        now[0] += 1.5
        sim.step()
        assert not sim.advisor.requests and sim.advisor.answer(qid, "reroute") == "closed"
    finally:
        p.close()
        sim.close()
    last = sim.advisor.chat[-1]
    assert last["text"] == "No reply in 20 s, taking the safe option: go around."
    assert last["request_id"] == qid and last["kind"] == "info"


def test_rules_ask_on_their_own_when_gemma_is_offline():
    p = GemmaPlanner(GemmaConfig(url="http://127.0.0.1:9", fallback_urls=(), check_every_s=3600.0,
                                 check_timeout_s=0.3), start=False)
    p.check_now()
    sim = make_sim(p)
    try:
        sim.run_for(12.0)
        r = next(r for r in sim.rovers if r.state == "exploring")
        r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
        sim.step()
        q = sim.advisor.requests_msg()
        assert q and "Gemma" not in q[0]["question"] and q[0]["default"] == "reroute"
        assert sim.advisor.chat[-1]["source"] == "rules"
        r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Again.")
        sim.step()
        assert len(sim.advisor.requests) == 1        # only the first anomaly asks while Gemma is offline
    finally:
        sim.close()


# ====================================================================== 4. chat commands
def _found_target(sim):
    ts = min(sim.fleet.targets, key=lambda ts: math.hypot(ts.spec.x, ts.spec.y))
    if ts.status is None:
        ts.status, ts.found_by, ts.found_t = "found", "r1", sim.t
    return ts


def _chat_commands(planner, src):
    sim = make_sim(planner, frames=False)
    try:
        sim.run_for(20.0)
        r2 = sim.rovers[1]
        gx, gy = round(r2.x + 3.0), round(r2.y)
        m = say(sim, f"rover 2 go to {gx}, {gy}")
        assert m["from"] == "r2" and m["kind"] == "ack" and m["source"] == src, m
        assert m["text"].startswith(f"Heading to ({gx}, {gy}), ") and "Max slope on the way" in m["text"]
        assert r2.op_goal == (gx, gy) and r2.goal == (gx, gy)
        assert any(e.get("source") == "operator" and e["msg"].startswith("r2 heading to") for e in sim.fleet.events)

        m = say(sim, "r1 explore north")
        assert m["kind"] == "ack" and m["text"].startswith("Exploring north") and m["source"] == src, m
        assert sim.rovers[0].dir_bias[:2] == (0.0, 1.0)

        ts = _found_target(sim)
        m = say(sim, f"the purple rover should sample {ts.spec.id}")
        assert m["from"] == "r3" and m["kind"] == "ack" and f"Going to sample {ts.spec.id}" in m["text"], m
        assert sim.rovers[2].state == "investigating" and ts.investigator == "r3"

        m = say(sim, "hold")
        assert m["from"] == "fleet" and m["kind"] == "ack" and m["source"] == src
        assert all(r.held and r.state == "idle" and r.goal is None for r in sim.rovers), [r.state for r in sim.rovers]
        assert r2.op_goal is None and "dropped" not in m["text"]
        sim.run_for(3.0)
        assert all(r.state == "idle" and r.speed < 0.2 for r in sim.rovers)

        m = say(sim, "resume")
        assert m["kind"] == "ack" and m["text"].count("resuming exploration") == 3
        sim.step()
        assert not any(r.held for r in sim.rovers) and all(r.state != "idle" for r in sim.rovers)

        m = say(sim, "all return")
        assert m["from"] == "fleet" and m["text"].count("returning to the lander") == 3, m
        assert all(r.state == "returning" for r in sim.rovers)

        m = say(sim, "avoid 20, 20 radius 3")
        assert m["from"] == "fleet" and m["text"].startswith("Marked (20, 20) radius 3 m as off limits"), m
        i, j = sim.map.ij(20.0, 20.0)
        assert sim.map.avoid[j, i] and math.isinf(sim.fleet.cost(sim.t)[j, i])

        m = say(sim, "do the thing")
        assert m["kind"] == "question" and "did not understand" in m["text"] and m["source"] == src
        m = say(sim, "resume", rover="r1")
        assert m["from"] == "r1" and m["text"] == "Resuming exploration."
        return sim
    finally:
        sim.close()


def test_chat_commands_through_gemma_fake():
    p = fake_planner()
    try:
        sim = _chat_commands(p, "gemma")
    finally:
        p.close()
    assert sum(1 for a in p.backend.asked if a[0] == "command") == 9


def test_chat_commands_through_the_keyword_parser():
    sim = _chat_commands(None, "rules")
    assert sim.advisor.planner_msg()["state"] == "disabled"


def test_unsafe_commands_are_refused_with_numbers():
    sim = make_sim(None)
    try:
        sim.run_for(15.0)
        r1 = sim.rovers[0]
        r1.battery = 0.35                  # enough to get home, not for a trip to the far corner
        m = say(sim, "r1 go to 30, 30")
        assert m["kind"] == "refuse" and m["source"] == "rules", m
        assert m["text"].startswith("Can't, battery 35% would not get me there and back (needs about ")
        assert r1.state != "returning"
        assert r1.op_goal is None
        m = say(sim, "r2 go to 60, 0")
        assert m["kind"] == "refuse" and "outside the map" in m["text"]
        sim.fleet.add_soft_zone(sim.t, sim.rovers[2], 12.0, -12.0, 3.0)
        m = say(sim, "r3 go to 12, -12")
        assert m["kind"] == "refuse" and "soft ground r3 found" in m["text"]
        m = say(sim, "avoid 1, 1 radius 3")
        assert m["kind"] == "refuse" and "lander pad" in m["text"]
        r1.battery = 0.12
        r1._go_home(sim.t, "battery")
        m = say(sim, "r1 hold")
        assert m["kind"] == "refuse" and "battery 12%" in m["text"] and not r1.held
    finally:
        sim.close()


def test_a_long_hold_never_strands_a_rover():
    sim = make_sim(None)
    try:
        sim.run_for(25.0)
        say(sim, "hold")
        r = max(sim.rovers, key=lambda r: math.hypot(r.x, r.y))
        r.battery = 0.16
        sim.run_for(4.0)
    finally:
        sim.close()
    assert not r.held and r.state == "returning", r.state
    assert any(m["text"].startswith("Battery 16%, leaving the hold") for m in sim.advisor.chat)


# ====================================================================== offline, protocol, budgets
def test_gemma_offline_never_blocks_and_rules_carry_on():
    p = GemmaPlanner(GemmaConfig(url="http://127.0.0.1:9", fallback_urls=(), check_every_s=3600.0,
                                 check_timeout_s=0.3))
    deadline = time.perf_counter() + 5.0
    while p.state not in ("offline",) and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert p.state == "offline"
    sim = make_sim(p, wait=False, operator_ask_first=False)
    try:
        worst = 0.0
        w0 = time.perf_counter()
        while sim.t < 60.0:
            c = time.perf_counter()
            sim.step()
            worst = max(worst, time.perf_counter() - c)
        speed = sim.t / (time.perf_counter() - w0)
        m = say(sim, "r1 explore north")
        assert m["source"] == "rules" and m["kind"] == "ack"
        pm = sim.tick_message()["planner"]
    finally:
        p.close()
        sim.close()
    assert pm["state"] == "offline" and not pm["online"] and pm["calls"] == 0
    assert speed >= 3.0 and worst < 0.5, (speed, worst)
    assert not [e for e in sim.fleet.events if e.get("source") == "gemma"]
    assert all(r.odometer > 10.0 for r in sim.rovers)


def test_protocol_v3_fields():
    p = fake_planner()
    sim = make_sim(p)
    msgs = [sim.init_message()]
    try:
        for k in range(int(70 / 0.6)):
            sim.run_for(0.6)
            if k == 20:
                sim.advisor.operator_message("rover 2 explore east")
            if k == 40:
                sim.advisor.operator_message("send the purple one home")
            if k == 60:
                r = next(r for r in sim.rovers if r.state in ("exploring", "investigating"))
                r.on_anomaly(sim.t, "low_visibility", {"lidar_range": 2.9}, "Testing.")
            msgs.append(sim.tick_message())
        msgs.append(sim.init_message())
    finally:
        p.close()
        sim.close()
    ids = {r.id for r in sim.rovers}
    streamed = list(msgs[0]["chat"])
    for m in msgs:
        text = json.dumps(m, allow_nan=False)
        assert EM_DASH not in text
        assert set(m["planner"]) == PLANNER_KEYS and m["planner"]["state"] in ("ready", "warming", "offline",
                                                                               "disabled")
        assert m["planner"]["budget_ms"] == 3000.0
        for c in m["chat"]:
            assert set(c) == CHAT_KEYS and c["kind"] in KINDS, c
            assert c["from"] in ids | {"operator", "fleet"} and c["to"] in ids | {"all", "operator", None}
            assert c["source"] in ("gemma", "rules", None) and (c["source"] is None) == (c["from"] == "operator")
            if c["from"] in ids:
                assert not c["text"].startswith(c["from"]), c
        for q in m["requests"]:
            assert set(q) == REQUEST_KEYS
        for e in m["events"]:
            assert e.get("source", "rules") in ("rules", "gemma", "operator")
        if m["type"] == "tick" and m is not msgs[-1]:
            streamed.extend(m["chat"])
    assert [c["id"] for c in streamed] == [f"c{n}" for n in range(1, len(streamed) + 1)]
    assert len(streamed) >= 5 and msgs[-1]["chat"] == streamed[-50:]
    assert any(m["requests"] for m in msgs), "the first anomaly should have opened a question"
    assert msgs[-1]["planner"]["calls"] > 0 and msgs[-1]["planner"]["ok"] > 0


def test_speed_and_brain_budget_with_gemma():
    """Real non-blocking mode: the fake model answers after 50 ms on its own thread."""
    p = fake_planner(RecordingFake(0.05))
    sim = make_sim(p, wait=False)
    try:
        w0 = time.perf_counter()
        sim.run_for(40.0)
        trigger(sim, "rockfall")
        sim.advisor.operator_message("r3 explore west")
        sim.run_for(40.0)
        trigger(sim, "sand")
        sim.advisor.operator_message("hold")
        sim.run_for(5.0)
        sim.advisor.operator_message("resume")
        sim.run_for(55.0)
        speed = sim.t / (time.perf_counter() - w0)
        pm = sim.advisor.planner_msg()
    finally:
        p.close()
        sim.close()
    assert speed >= 3.0, speed
    for r in sim.rovers:
        a = sorted(r.brain_all)
        assert a[int(0.95 * (len(a) - 1))] < sim.cfg.brain_budget_ms, r.id
    assert pm["calls"] >= 5 and pm["over_budget"] == 0 and pm["avg_ms"] < pm["budget_ms"]
    assert gemma_events(sim)


def test_chat_and_answer_endpoints():
    from redshift import server

    class Req:
        def __init__(self, body: bytes):
            self._b = body

        async def body(self):
            return self._b

    def call(fn, body: bytes):
        out = asyncio.run(fn(Req(body)))
        if hasattr(out, "status_code"):
            return out.status_code, json.loads(out.body)
        return 200, json.loads(json.dumps(out))

    p = fake_planner()
    old = server.state.runner
    server.state.runner = server.SimRunner(Config(seed=42, plan_clock="model"), planner=p)   # not started
    try:
        sim = server.state.runner.sim
        sim.advisor.frame_source = None
        sim.advisor.wait_for_gemma = True
        assert call(server.chat, b'{"text": "  "}')[0] == 400
        assert call(server.chat, b"not json")[0] == 400
        assert call(server.chat, b'{"text": "hi", "rover": 2}')[0] == 400
        code, body = call(server.chat, b'{"text": "explore north", "rover": "r2"}')
        assert code == 200 and body == {"ok": True, "id": "c1"}
        sim.run_for(12.0)
        assert [m["text"] for m in sim.advisor.chat if m["from"] == "r2"][0].startswith("Exploring north")
        assert call(server.answer, b'{"id": "q1", "option": "reroute"}')[0] == 404
        r = next(r for r in sim.rovers if r.state == "exploring")
        r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
        sim.step()
        assert call(server.answer, b'{"id": "q1", "option": "nope"}')[0] == 400
        assert call(server.answer, b'{"id": "q1", "option": "reroute"}') == (200, {"ok": True})
        assert call(server.answer, b'{"id": "q1", "option": "reroute"}')[0] == 409
        assert call(server.answer, b'{"id": 1}')[0] == 400
    finally:
        server.state.runner.sim.close()
        server.state.runner = old
        p.close()


def test_no_em_dash_in_tier2_files():
    for rel in ("redshift/advisor.py", "tests/test_brain.py", "README.md", "GEMMA.md"):
        assert EM_DASH not in (ROOT / rel).read_text(encoding="utf-8"), rel


if __name__ == "__main__":
    failed = 0
    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_") and callable(fn)]
    for n, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {n} ({time.perf_counter() - t0:.1f} s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {n}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
