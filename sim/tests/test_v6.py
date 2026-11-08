"""Protocol v6: commanding by place ("go to the water source", "can we investigate the lake for signs of
life"), choosing between candidates (choose_place requests answered by POST /api/answer or by chat), the
survey on arrival and its report, and requests that wait for the operator (OPERATOR_TIMEOUT_S 0).

    python -m pytest -q tests/test_v6.py        (about 40 s)

The sims run on the 120 m default map with the model clock (runs repeat exactly). Water is put into the
fleet's knowledge directly (fleet.add_water_zone, as a rover does when its camera sees liquid) on mapped
ground, so the tests do not wait for the rovers to find a lake.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import gemma as g  # noqa: E402
from redshift import places as P  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.gemma import FakeBackend, GemmaConfig, GemmaPlanner  # noqa: E402
from redshift.hazards import HazardError  # noqa: E402
from redshift.sim import Simulation  # noqa: E402

EM_DASH = chr(0x2014)
FLEET = {"rovers": [{"id": "r1", "name": "Rover 1", "color": "#ff7a45", "x": 3.2, "y": -1.0},
                    {"id": "r2", "name": "Rover 2", "color": "#36cfc9", "x": -8.5, "y": 12.1},
                    {"id": "r3", "name": "Rover 3", "color": "#9254de", "x": 15.0, "y": 4.4}],
         "targets": [{"id": "t4", "kind": "ice", "x": 12.0, "y": -4.5}, {"id": "t7", "kind": "vent", "x": -3.0, "y": 9.0}],
         "lander": [0.0, 0.0], "map_size": 120}


# ====================================================================== helpers
def spot(sim, x0, y0, rmin, rmax, avoid=()):
    """A spot rmin..rmax m from (x0, y0) with mapped, passable ground 4.5 to 5.5 m around it (for a water
    zone whose shore the rovers can reach), away from the lander, the rovers and the `avoid` circles."""
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


def step_until(sim, cond, seconds):
    for _ in range(int(seconds / sim.agent_dt)):
        sim.step()
        if cond():
            return True
    return False


def chat_after(sim, seq):
    return [m for m in sim.advisor.chat if m["_seq"] > seq]


class FakeRunner:
    def __init__(self, sim):
        self.sim, self.cfg = sim, sim.cfg
        self._lock = threading.Lock()

    @contextmanager
    def locked(self):
        with self._lock:
            yield self.sim


def post_answer(sim, body: dict):
    from redshift import server

    class Req:
        def __init__(self, b: bytes):
            self._b = b

        async def body(self):
            return self._b

    old = server.state.runner
    server.state.runner = FakeRunner(sim)
    try:
        out = asyncio.run(server.answer(Req(json.dumps(body).encode())))
    finally:
        server.state.runner = old
    if hasattr(out, "status_code"):
        return out.status_code, json.loads(out.body)
    return 200, out


@pytest.fixture(scope="module")
def world():
    """No lifeforms (a survey with nothing moving), the keyword parser (no Gemma), two bodies of water: a
    pond (one 2 m zone) and a lake (two overlapping 3 m zones, about 9 m across) on opposite sides."""
    sim = Simulation(Config(seed=42, plan_clock="model", lifeforms=False))
    sim.run_for(40.0)
    f = sim.fleet
    px, py = spot(sim, 0.0, 0.0, 9.0, 16.0)
    f.add_water_zone(sim.t, sim.rovers[0], px, py, 2.0)
    lx, ly = spot(sim, 0.0, 0.0, 9.0, 18.0, avoid=[(px, py, 16.0)])
    f.add_water_zone(sim.t + 1.0, sim.rovers[1], lx, ly, 3.0)
    f.add_water_zone(sim.t + 1.0, sim.rovers[1], lx + 3.0 * (lx - px) / math.hypot(lx - px, ly - py),
                     ly + 3.0 * (ly - py) / math.hypot(lx - px, ly - py), 3.0)
    sim.step()
    yield sim
    sim.close()


# ====================================================================== intent and parsing
POLITE = ["can we investigate the water source again for any signs of life around there", "go to the water source",
          "could you check out the vent", "let's survey the spires", "I want you to look at the herd",
          "please head to the lake near rover 2", "can rover 2 investigate the lake", "go back to the herd",
          "return to the lake", "would you take a look at t4"]
QUESTIONS = [("can you see any water?", "water"), ("is there water near the lander?", "water"),
             ("where is the lake?", "water"), ("did you investigate the lake?", "water"),
             ("found any water?", "water"), ("seen any life?", "lifeforms")]


def test_polite_commands_are_commands_and_real_questions_stay_questions():
    for text in POLITE:
        assert g.question_intent(text, FLEET["rovers"])[0] == "command", text
        assert g.parse_command(text, FLEET)["command"]["type"] == "go_to_place", text
    for text, topic in QUESTIONS:
        assert g.question_intent(text, FLEET["rovers"])[0] == "strong", text
        assert g.parse_command(text, FLEET)["command"] == {"type": "question", "rover": None, "topic": topic}, text
    # older behaviour stays
    assert g.parse_command("rover 2, go check out the area north of the lander", FLEET)["command"]["type"] == \
        "explore_direction"
    assert g.parse_command("return to the lander", FLEET)["command"]["type"] == "return_home"
    assert g.parse_command("go to the lake at 10, -5", FLEET)["command"] == {"type": "go_to", "rover": None,
                                                                           "x": 10.0, "y": -5.0}
    assert g.parse_command("investigate t4", FLEET)["command"]["type"] == "sample_target"


def test_place_parsing():
    def cmd(text):
        return g.parse_command(text, FLEET)["command"]

    c = cmd("can we investigate the water source again for any signs of life around there")
    assert c == {"type": "go_to_place", "rover": None, "place": "water", "target_id": None, "qualifier": "none",
                 "ordinal": None, "near_rover": None, "investigate": True}
    c = cmd("go to the water source")
    assert (c["place"], c["investigate"], c["rover"]) == ("water", False, None)
    assert cmd("rover 3, survey the spires")["rover"] == "r3" and cmd("rover 3, survey the spires")["investigate"]
    c = cmd("please head to the lake near rover 2")
    assert (c["rover"], c["qualifier"], c["near_rover"]) == (None, "near_rover", "r2")     # r2 is a reference
    assert cmd("go to the lake closest to the orange rover")["near_rover"] == "r1"
    assert cmd("go to the second lake")["qualifier"] == "ordinal" and cmd("go to the second lake")["ordinal"] == 2
    assert cmd("go to the biggest lake")["qualifier"] == "biggest"
    assert cmd("head to the northern pond")["qualifier"] == "north"
    assert cmd("go to the newest vent")["qualifier"] == "newest"
    assert cmd("rover 1, look at the closest spire")["qualifier"] == "nearest"
    c = cmd("look for life")
    assert (c["place"], c["investigate"]) == ("lifeform", True)
    c = cmd("take a look at t7")
    assert (c["place"], c["target_id"]) == ("target", "t7")
    assert cmd("check out the rockfall")["place"] == "hazard" and cmd("go to the fallen spire")["place"] == "hazard"
    assert cmd("go to the lander")["place"] == "lander"
    assert cmd("take a look at t9")["type"] == "unclear"                      # unknown target id
    assert g.parse_command("go to the water source", FLEET)["ack"] == "Going to the water."


def _gemma(text, **answer):
    req = g.make_command_request(1, 10.0, text, FLEET)
    base = {"command": "unclear", "rover": "none", "place": "none", "qualifier": "none", "ordinal": 0,
            "near_rover": "none", "investigate": False, "direction": "none", "x": 0, "y": 0, "radius": 0,
            "target_id": "none", "topic": "none", "question": "", "ack": "", "confidence": 0.9}
    return g.validate(req, json.dumps(dict(base, **answer)))


def test_gemma_place_answers_are_validated_and_cross_checked():
    r = _gemma("check out the glowing thing", command="go_to_place", place="spire", qualifier="nearest",
               investigate="true")
    assert r.ok and r.command == {"type": "go_to_place", "rover": None, "place": "spire", "target_id": None,
                                  "qualifier": "nearest", "ordinal": None, "near_rover": None, "investigate": True}
    # what went wrong before v6: the polite order read as a question, the go_to at the rover's own position
    r = _gemma("can we investigate the water source again for any signs of life around there", command="question",
               topic="water")
    assert r.ok and r.command["type"] == "go_to_place" and r.command["investigate"]
    r = _gemma("go to the water source", command="go_to", rover="r1", x=3.2, y=-1.0)
    assert r.ok and r.command["type"] == "go_to_place" and r.command["place"] == "water"
    # the operator's words win for the place; a qualifier the model missed is added
    r = _gemma("go to the biggest lake", command="go_to_place", place="lifeform")
    assert r.ok and (r.command["place"], r.command["qualifier"]) == ("water", "biggest")
    # enums and ids are checked: anything invalid falls back to the keyword parser
    for bad, part in ((dict(place="volcano"), "place"), (dict(place="target", target_id="t99"), "target"),
                      (dict(place="water", qualifier="weird"), "qualifier"),
                      (dict(place="water", qualifier="near_rover", near_rover="r9"), "near_rover")):
        r = _gemma("go to the lake near rover 2", command="go_to_place", **bad)
        assert not r.ok and part in r.error and r.command["type"] == "go_to_place", (bad, r.error)
        assert r.command["near_rover"] == "r2"
    # a question stays a question; explicit coordinates stay a go_to
    assert _gemma("can you see any water?", command="go_to_place", place="water").command["type"] == "question"
    assert _gemma("go to the lake at 10, -5", command="go_to_place", place="water").command == \
        {"type": "go_to", "rover": None, "x": 10.0, "y": -5.0}
    req = g.make_command_request(1, 0.0, "can we investigate the water source again for any signs of life", FLEET)
    props = g.build_schema(req)["properties"]
    assert props["place"]["enum"] == list(g.PLACES) + ["none"] and "near_rover" in props["qualifier"]["enum"]
    assert props["near_rover"]["enum"] == ["r1", "r2", "r3", "lander", "none"] and props["investigate"]["type"] == "boolean"
    system, user = g.build_prompt(req)
    assert "go_to_place" in system and g.estimate_tokens(system + user) < 400


def test_fake_backend_understands_place_commands():
    p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0), backend=FakeBackend(0.0))
    try:
        assert p.wait_ready(2.0)
        for text, place, inv in (("go to the water source", "water", False),
                                 ("can we investigate the lake for signs of life", "water", True),
                                 ("rover 2, check out the vent near the lander", "vent", False)):
            tk = p.submit_command(1.0, text, FLEET)
            res = None
            for _ in range(2000):
                res = p.poll(tk)
                if res is not None:
                    break
                threading.Event().wait(0.001)
            assert res is not None and res.ok and res.source == "gemma", (text, res and res.error)
            assert (res.command["type"], res.command["place"], res.command["investigate"]) == ("go_to_place", place, inv)
    finally:
        p.close()


# ====================================================================== resolver
def test_resolver_bodies_qualifiers_goals_and_nothing_known(world):
    sim = world
    f = sim.fleet
    cands = P.candidates(sim, "water")
    assert [c.sub for c in cands] == ["pond", "lake"], [(c.sub, c.size) for c in cands]
    pond, lake = cands
    assert 3.5 <= pond.size <= 4.5 and 8.0 <= lake.size <= 10.0 and len(lake.zones) == 2
    r = sim.rovers[0]
    for c in cands:
        lab = P.option_label(c, (r.x, r.y))
        assert re.fullmatch(r"(Pond|Lake) about \d+ m across, (\d+ m [a-z-]+ of me|right here)( near the lander)?",
                            lab), lab
    assert P.apply_qualifier(cands, "biggest") == ([lake], True)
    assert P.apply_qualifier(cands, "smallest") == ([pond], True)
    assert P.apply_qualifier(cands, "newest") == ([lake], True)
    near = min(cands, key=lambda c: math.hypot(c.x - r.x, c.y - r.y))
    assert P.apply_qualifier(cands, "nearest", ref=(r.x, r.y)) == ([near], True)
    assert P.apply_qualifier(cands, "ordinal", 1, ref=(r.x, r.y)) == ([near], True)
    assert P.apply_qualifier(cands, "ordinal", 5, ref=(r.x, r.y))[1] is False
    assert P.apply_qualifier(cands, "north")[0] == [max(cands, key=lambda c: c.y)]
    assert P.apply_qualifier(cands, "none") == (cands, False)
    for c in cands:
        goals = P.approach_goals(sim, c, (r.x, r.y), sim.t)
        assert goals, c
        cost = f.cost(sim.t)
        for gx, gy in goals:
            i, j = sim.map.ij(gx, gy)
            assert sim.map.known[j, i] and np.isfinite(cost[j, i]) and not sim.map.water[j, i]
            assert 1.8 <= f.water_zone_gap(gx, gy) <= 4.6 and f.water_zone_at(gx, gy, margin=1.5) is None
    # nothing known: the fleet says so and lists what it knows (no lifeforms in this world)
    seq = sim.advisor.chat_seq
    cid = sim.advisor.operator_message("go to the herd")
    step_until(sim, lambda: any(m["reply_to"] == cid for m in chat_after(sim, seq)), 3.0)
    rep = [m for m in chat_after(sim, seq) if m["reply_to"] == cid]
    assert rep and rep[0]["from"] == "fleet" and rep[0]["kind"] == "info", rep
    assert rep[0]["text"].startswith("We don't know any herds or creatures yet. Known: 2 bodies of water")
    assert not sim.advisor.go_jobs and not sim.advisor.requests


def test_closest_free_rover_goes(world):
    sim, adv = world, world.advisor
    lake = max(P.candidates(sim, "water"), key=lambda c: c.size)
    order = sorted(sim.rovers, key=lambda r: math.hypot(r.x - lake.x, r.y - lake.y))
    busy, want = order[0], order[1]
    old = (busy.state, busy.return_reason)
    busy.state, busy.return_reason = "returning", "battery"         # not free: going home on a low battery
    try:
        seq = adv.chat_seq
        cid = adv.operator_message("can you go to the biggest lake")
        sim.step()
        busy.state, busy.return_reason = old
        assert step_until(sim, lambda: any(m["reply_to"] == cid for m in chat_after(sim, seq)), 10.0)
    finally:
        busy.state, busy.return_reason = old
    rep = [m for m in chat_after(sim, seq) if m["reply_to"] == cid]
    assert len(rep) == 1 and rep[0]["from"] == want.id and rep[0]["kind"] == "ack", rep
    assert re.match(r"I'm the closest free rover \(\d+ m\)\. Heading to the lake's shore at \(-?[\d.]+, -?[\d.]+\), "
                    r"\d+ m\. Max slope on the way \d+ deg, battery after return about \d+%\.$", rep[0]["text"]), rep
    assert want.op_goal is not None and want.op_task["label"] == "the lake's shore"
    assert 1.8 <= sim.fleet.water_zone_gap(*want.op_goal) <= 4.6
    adv.operator_message("resume")
    sim.run_for(1.0)
    assert want.op_goal is None and want.op_task is None


def test_choose_place_request_answered_by_api_and_by_chat(world):
    sim, adv = world, world.advisor
    # 1. no rover named, nothing to choose by: a choose_place request for the closest free rover
    seq = adv.chat_seq
    cid = adv.operator_message("go to the water source")
    assert step_until(sim, lambda: adv.requests, 3.0)
    msg = sim.tick_message()
    json.dumps(msg, allow_nan=False)                  # no Infinity anywhere
    (q,) = msg["requests"]
    assert set(q) == {"id", "rover", "kind", "t", "question", "options", "default", "expires_in", "image"}
    assert q["kind"] == "choose_place" and q["default"] is None and q["expires_in"] is None and q["image"] is None
    assert re.match(r"I'm the closest free rover \(\d+ m\)\. I know 2 bodies of water\. Which one\?$", q["question"])
    assert [o["id"] for o in q["options"]] == ["p1", "p2", "cancel"] and q["options"][-1]["label"] == "Never mind"
    for o in q["options"][:2]:
        assert isinstance(o["x"], float) and isinstance(o["y"], float) and " m across, " in o["label"]
    ask = [m for m in chat_after(sim, seq) if m["request_id"] == q["id"]]
    assert ask[0]["kind"] == "question" and ask[0]["reply_to"] == cid and ask[0]["from"] == q["rover"]
    r = adv._rover(q["rover"])
    assert r.question_hold
    sim.run_for(3.0)
    assert r.speed < 0.15 and adv.requests                       # holds, never expires
    # answered through POST /api/answer
    assert post_answer(sim, {"id": q["id"], "option": "p9"})[0] == 400
    assert post_answer(sim, {"id": q["id"], "option": "p2"}) == (200, {"ok": True})
    assert post_answer(sim, {"id": q["id"], "option": "p2"})[0] == 409
    assert not r.question_hold
    assert step_until(sim, lambda: r.op_goal is not None, 5.0)
    acks = [m for m in chat_after(sim, seq) if m["kind"] == "ack" and m["from"] == r.id]
    assert acks and acks[0]["request_id"] == q["id"] and acks[0]["reply_to"] == cid
    p2 = q["options"][1]
    assert math.hypot(r.op_goal[0] - p2["x"], r.op_goal[1] - p2["y"]) < 9.0
    # 2. a named rover asks too; "the second one" typed in the chat answers it
    seq = adv.chat_seq
    cid = adv.operator_message("rover 3, go to the water source")
    assert step_until(sim, lambda: adv.requests, 3.0)
    (q,) = adv.requests_msg()
    assert q["rover"] == "r3" and not q["question"].startswith("I'm the closest")
    pick = adv.operator_message("the second one")
    op = next(m for m in adv.chat if m["id"] == pick)
    assert op["kind"] == "answer" and op["request_id"] == q["id"] and not adv.requests
    r3 = adv._rover("r3")
    assert step_until(sim, lambda: r3.op_goal is not None, 5.0)
    ack = [m for m in chat_after(sim, seq) if m["from"] == "r3" and m["kind"] == "ack"]
    assert ack and ack[0]["request_id"] == q["id"] and ack[0]["reply_to"] == cid
    assert math.hypot(r3.op_goal[0] - q["options"][1]["x"], r3.op_goal[1] - q["options"][1]["y"]) < 9.0
    # 3. "the one near the lander" and "never mind"
    for text, want in (("the one near the lander", None), ("never mind", "cancel")):
        cid = adv.operator_message("r1, go to the water source")
        assert step_until(sim, lambda: adv.requests, 3.0)
        (q,) = adv.requests_msg()
        if want is None:
            want = min(q["options"][:2], key=lambda o: math.hypot(o["x"], o["y"]))["id"]
        cands = adv.requests[q["id"]]["job"]["cands"]
        assert P.pick_option(text, cands, (0.0, 0.0), {}) == want
        adv.operator_message(text)
        assert not adv.requests and adv.closed[q["id"]] == "answered"
    assert adv.chat[-1]["text"] == "OK, never mind. Carrying on with my own plan." and adv.chat[-1]["from"] == "r1"
    # a new order to that rover is not taken as a pick
    adv.operator_message("r1, go to the water source")
    assert step_until(sim, lambda: adv.requests, 3.0)
    adv.operator_message("r1 explore north")
    assert step_until(sim, lambda: not adv.requests, 3.0)        # the order settles the open question
    adv.operator_message("resume")
    sim.run_for(1.0)


def test_investigate_surveys_on_arrival_and_reports(world):
    sim, adv = world, world.advisor
    pond = min(P.candidates(sim, "water"), key=lambda c: c.size)
    r = min(sim.rovers, key=lambda o: math.hypot(o.x - pond.x, o.y - pond.y))
    n = r.id[1:]
    seq = adv.chat_seq
    cid = adv.operator_message(f"rover {n}, could you investigate the smallest pond for any signs of life")
    yaws = []

    def report():
        if r.survey is not None:
            yaws.append(r.yaw)
        return any(m["reply_to"] == cid and m["text"].startswith("Survey") for m in chat_after(sim, seq))

    assert step_until(sim, report, 150.0), [m["text"] for m in chat_after(sim, seq)]
    msgs = [m for m in chat_after(sim, seq) if m["reply_to"] == cid]
    assert [m["kind"] for m in msgs] == ["ack", "info", "info"], [m["text"] for m in msgs]
    assert msgs[0]["text"].endswith("I'll survey there for about 30 s.")
    assert msgs[1]["text"].startswith("Arrived at the pond's shore. Surveying for about 30 s")
    rep = msgs[2]
    assert rep["from"] == r.id and rep["source"] == "rules" and rep["request_id"] is None
    assert re.match(r"Survey of the pond's shore done: \d+ s, turned a full circle\. Nothing moved within my \d+ m "
                    r"LiDAR range\.", rep["text"]), rep["text"]
    assert "The water's edge is about" in rep["text"] and EM_DASH not in rep["text"]
    assert len(yaws) > 100 and r.survey is None and r.state != "observing"
    assert any(e.get("source") == "operator" and e["msg"].startswith(f"{r.id} survey of the pond's shore")
               for e in sim.fleet.events)


def test_survey_sees_movement_and_starts_observing():
    sim = Simulation(Config(seed=42, plan_clock="model"))
    try:
        sim.run_for(40.0)
        r = sim.rovers[0]
        wx, wy = spot(sim, r.x, r.y, 9.0, 15.0)
        sim.fleet.add_water_zone(sim.t, r, wx, wy, 2.0)
        adv = sim.advisor
        seq = adv.chat_seq
        cid = adv.operator_message("rover 1, investigate the water for signs of life")
        assert step_until(sim, lambda: r.survey is not None or r.state == "observing", 60.0)
        for _ in range(20):
            if r.survey is None:
                break
            try:
                sim.hazards.trigger("lifeform", "r1")
                break
            except HazardError:
                sim.step()
        assert step_until(sim, lambda: any(m["reply_to"] == cid and m["text"].startswith("Survey")
                                           for m in chat_after(sim, seq)), 40.0)
        rep = [m for m in chat_after(sim, seq) if m["reply_to"] == cid][-1]
        assert rep["kind"] == "info" and "Movement: k" in rep["text"] and "Stopping to watch k" in rep["text"], rep
        assert any(e["rover"] == "r1" and "sees something moving" in e["msg"] for e in sim.fleet.events)
        nums = [v for _, v in g.numbers_in_text(rep["text"])]
        assert all(math.isfinite(v) for v in nums)
    finally:
        sim.close()


# ====================================================================== requests wait for the operator
def test_requests_never_expire_by_default(monkeypatch):
    assert Config().operator_timeout_s == 0.0
    monkeypatch.delenv("OPERATOR_TIMEOUT_S", raising=False)
    assert Config.from_env().operator_timeout_s == 0.0
    monkeypatch.setenv("OPERATOR_TIMEOUT_S", "20")
    assert Config.from_env().operator_timeout_s == 20.0
    legacy = dict(map_size=80.0, water=False, spires=False, lifeforms=False, plan_clock="model")
    for timeout in (0.0, 5.0):
        p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0), backend=FakeBackend(0.0))
        assert p.wait_ready(2.0)
        sim = Simulation(Config(seed=42, operator_timeout_s=timeout, **legacy), planner=p, wait_for_gemma=True)
        now = [1000.0]
        sim.advisor.clock = lambda: now[0]
        try:
            sim.run_for(12.0)
            r = next(o for o in sim.rovers if o.state == "exploring")
            r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
            sim.step()
            (q,) = sim.advisor.requests_msg()
            assert q["kind"] == "anomaly" and q["default"] == "reroute"
            json.dumps(sim.tick_message(), allow_nan=False)
            if timeout == 0.0:
                assert q["expires_in"] is None
                now[0] += 1e7                               # a very long wait: still open, still holding
                sim.run_for(3.0)
                assert sim.advisor.requests and r.question_hold
                assert r.speed < 0.15 or r._may_move_while_asking(sim.t)
                assert not any("No reply" in m["text"] for m in sim.advisor.chat)
            else:
                assert q["expires_in"] == 5.0
                now[0] += 6.0
                sim.step()
                assert not sim.advisor.requests and not r.question_hold
                assert sim.advisor.chat[-1]["text"] == "No reply in 5 s, taking the safe option: go around."
        finally:
            sim.close()
            p.close()


def test_no_em_dash_in_v6_files():
    for rel in ("redshift/places.py", "redshift/advisor.py", "redshift/rover.py", "redshift/gemma.py",
                "tests/test_v6.py", "README.md", "GEMMA.md", "../PROTOCOL.md"):
        assert EM_DASH not in (ROOT / rel).read_text(encoding="utf-8"), rel
