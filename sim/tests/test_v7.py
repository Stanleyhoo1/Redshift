"""Protocol v7: exploring an area (rover `focus`), curiosity (interest focus), chat memory and references,
following a newly sighted herd while asking, and leaving for the lander when waiting would drain the battery.

    python -m pytest -q tests/test_v7.py        (about a minute)

The sims run on the 120 m default map with the model clock (runs repeat exactly). Water is put into the
fleet's knowledge directly (fleet.add_water_zone, as a rover does when its camera sees liquid), so the tests
do not wait for the rovers to find a lake.
"""
from __future__ import annotations

import json
import math
import re
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import gemma as g  # noqa: E402
from redshift import places as P  # noqa: E402
from redshift.chatmem import ChatMemory, Ref  # noqa: E402
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
LEGACY = dict(map_size=80.0, water=False, spires=False, lifeforms=False, plan_clock="model")


# ====================================================================== helpers
def fake_planner() -> GemmaPlanner:
    p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0), backend=FakeBackend(0.0))
    assert p.wait_ready(2.0)
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


def spot(sim, x0, y0, rmin, rmax, avoid=()):
    """A spot rmin..rmax m from (x0, y0) with mapped, passable ground around it (for a water zone whose shore
    the rovers can reach), away from the lander, the rovers and the `avoid` circles."""
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


def _water_world(planner=None):
    sim = Simulation(Config(seed=42, plan_clock="model", lifeforms=False, focus_s=40.0, curiosity=False,
                            autonomy_mode="supervised"), planner=planner, wait_for_gemma=planner is not None)
    sim.run_for(40.0)
    f = sim.fleet
    px, py = spot(sim, 0.0, 0.0, 9.0, 16.0)
    f.add_water_zone(sim.t, sim.rovers[0], px, py, 2.0)
    lx, ly = spot(sim, 0.0, 0.0, 9.0, 18.0, avoid=[(px, py, 16.0)])
    f.add_water_zone(sim.t + 1.0, sim.rovers[1], lx, ly, 3.0)
    sim.step()
    return sim


@pytest.fixture(scope="module", params=["keywords", "fake_gemma"])
def world(request):
    """Two bodies of water, no lifeforms, curiosity off (only the operator's focus); the commands go through the
    keyword parser, or through the fake Gemma and its checks."""
    p = fake_planner() if request.param == "fake_gemma" else None
    sim = _water_world(p)
    yield sim
    sim.close()
    if p is not None:
        p.close()


# ====================================================================== parsing (keyword parser)
def test_parser_area_commands_and_references():
    def cmd(text):
        return g.parse_command(text, FLEET)["command"]

    c = cmd("explore around the lake")
    assert (c["type"], c["place"], c.get("explore"), c["investigate"]) == ("go_to_place", "water", True, False)
    c = cmd("go to the vent and look around")
    assert (c["place"], c.get("explore"), c["investigate"]) == ("vent", True, False)
    c = cmd("explore north of the lander")
    assert (c["place"], c.get("explore"), c.get("direction")) == ("lander", True, "north")
    assert cmd("explore the area south-west of the lake")["direction"] == "south-west"
    assert cmd("explore around 10, -5") == {"type": "go_to", "rover": None, "x": 10.0, "y": -5.0, "explore": True}
    c = cmd("explore there")
    assert (c["place"], c.get("explore")) == ("there", True)
    c = cmd("investigate the pond")
    assert c["investigate"] and "explore" not in c                  # survey first, then the focus (advisor)
    # references
    assert cmd("go back there")["place"] == "there" and cmd("go back there")["type"] == "go_to_place"
    c = cmd("the second one")
    assert (c["place"], c["qualifier"], c["ordinal"]) == ("there", "ordinal", 2)
    assert cmd("go to the last one")["ordinal"] == -1
    assert cmd("go to that lake")["qualifier"] == "mentioned" and cmd("that lake")["place"] == "water"
    c = cmd("check out the vent you mentioned")
    assert (c["place"], c["qualifier"]) == ("vent", "mentioned")
    assert cmd("investigate it")["place"] == "there" and cmd("investigate it")["investigate"]
    assert cmd("do that again") == {"type": "repeat", "rover": None}
    assert cmd("same for rover 3") == {"type": "repeat", "rover": "r3"}
    assert cmd("can you do that again?")["type"] == "repeat" and cmd("r2 too")["rover"] == "r2"
    # older behaviour stays
    assert cmd("rover 2, go check out the area north of the lander")["type"] == "explore_direction"
    assert cmd("explore north") == {"type": "explore_direction", "rover": None, "direction": "north"}
    assert cmd("go back to the lander")["type"] == "return_home" and cmd("stay there")["type"] == "hold"
    assert cmd("can we investigate the water source again for any signs of life around there")["place"] == "water"
    assert cmd("what did you find there?") == {"type": "question", "rover": None, "topic": "finds"}
    assert g.parse_command("explore around the lake", FLEET)["ack"] == "Exploring around the water."


# ====================================================================== Gemma: schema, prompt, checks, fake
def _gemma(text, fleet=FLEET, **answer):
    req = g.make_command_request(1, 10.0, text, fleet)
    base = {"command": "unclear", "rover": "none", "place": "none", "qualifier": "none", "ordinal": 0,
            "near_rover": "none", "investigate": False, "explore": False, "direction": "none", "x": 0, "y": 0,
            "radius": 0, "target_id": "none", "topic": "none", "question": "", "ack": "", "confidence": 0.9}
    return g.validate(req, json.dumps(dict(base, **answer)))


def test_gemma_schema_prompt_context_and_checks():
    chat = {"lines": ["operator: rover 2, go to the pond", "r2: Heading to the pond's shore at (15.7, 1.7), 22 m."],
            "refs": "there = the pond's shore at (15.7, 1.7); last rover r2; last command: 'rover 2, go to the pond' (r2)"}
    fleet = dict(FLEET, chat=chat)
    req = g.make_command_request(1, 0.0, "go back there", fleet)
    props = g.build_schema(req)["properties"]
    assert "repeat" in props["command"]["enum"] and "there" in props["place"]["enum"]
    assert "mentioned" in props["qualifier"]["enum"] and props["explore"]["type"] == "boolean"
    system, user = g.build_prompt(req)
    assert "Recent chat: operator: rover 2, go to the pond | r2: Heading" in user
    assert "Context: there = the pond's shore at (15.7, 1.7)" in user and "repeat" in system
    assert g.estimate_tokens(system + user) < 520
    plain = g.make_command_request(1, 0.0, "can we investigate the water source again for any signs of life", FLEET)
    assert g.estimate_tokens("".join(g.build_prompt(plain))) < 400        # no chat yet: as small as before
    # the model answered go_to at the remembered coordinates: the operator's words ("there") win
    r = _gemma("go back there", fleet, command="go_to", rover="r2", x=15.7, y=1.7)
    assert r.ok and r.command["type"] == "go_to_place" and r.command["place"] == "there"
    # "do that again" is a repeat whatever the model says; a model's repeat for a concrete order is that order
    assert _gemma("do that again", command="go_to_place", place="water").command == {"type": "repeat", "rover": None}
    assert _gemma("rover 3, same for you", command="hold").command["type"] == "repeat"
    assert _gemma("go to the lake", command="repeat").command["type"] == "go_to_place"
    r = _gemma("same for rover 3", command="repeat", rover="r3")
    assert r.ok and r.command == {"type": "repeat", "rover": "r3"}
    # explore and the side come from the operator's words
    r = _gemma("explore around the lake", command="go_to_place", place="water", explore=False)
    assert r.ok and r.command.get("explore") is True
    r = _gemma("go to the lake", command="go_to_place", place="water", explore=True)
    assert r.ok and "explore" not in r.command                       # a plain go to still just goes
    r = _gemma("explore north of the lander", command="explore_direction", direction="north")
    assert r.ok and (r.command["type"], r.command["place"], r.command["direction"]) == ("go_to_place", "lander", "north")
    r = _gemma("the second one", command="go_to_place", place="there", qualifier="ordinal", ordinal=2)
    assert r.ok and (r.command["place"], r.command["ordinal"]) == ("there", 2)
    r = _gemma("go to that lake", command="go_to_place", place="water")
    assert r.ok and r.command["qualifier"] == "mentioned"
    # answers get a context line, and still only numbers from the facts
    areq = g.make_answer_request(1, "r2", 1.0, "what did you find there?", "finds",
                                 ["about the pond's shore at (16, 2): I explored around it for 40 s"], "rules",
                                 context=["'there' or 'it' means the pond's shore at (16, 2), from the chat"])
    assert "Chat context: 'there' or 'it' means the pond's shore" in g.build_prompt(areq)[1]


def test_fake_backend_handles_the_new_fields():
    p = fake_planner()
    try:
        for text, want in (("explore around the lake", ("go_to_place", "water", True)),
                           ("go back there", ("go_to_place", "there", False)),
                           ("do that again", ("repeat", None, False)),
                           ("explore around 10, -5", ("go_to", None, True))):
            tk = p.submit_command(1.0, text, FLEET)
            res = None
            for _ in range(2000):
                res = p.poll(tk)
                if res is not None:
                    break
                threading.Event().wait(0.001)
            assert res is not None and res.ok and res.source == "gemma", (text, res and res.error)
            c = res.command
            assert (c["type"], c.get("place"), bool(c.get("explore"))) == want, (text, c)
    finally:
        p.close()


def test_chat_memory_unit():
    m = ChatMemory()
    assert m.context() == {} and m.there is None and m.pick(1) is None
    a = Ref("water", 10.0, 5.0, "the pond's shore", id="w1", rover="r2")
    b = Ref("vent", -4.0, 9.0, "t7 (thermal vent)", id="t7", rover="r1")
    c = Ref("ice", 12.0, -4.5, "t4 (ice deposit)", id="t4", rover="r1")
    m.mention(a)
    m.mention_list([b, c])
    assert m.there is a and m.pick(2) is c and m.pick(-1) is c and m.pick(3) is None
    assert m.last_of("vent") is b and m.last_of("water") is a and m.last_of("target") is c
    m.add_report(Ref("area", 10.0, 5.0, "the pond's shore", rover="r2", r=15.0), "Explored around...", "r2", 50.0)
    assert m.report_near(11.0, 5.0)["rover"] == "r2" and m.there.kind == "area"
    for k in range(12):
        m.add_msg("operator", f"message {k}")
    ctx = m.context()
    assert len(m.msgs) == 8 and len(ctx["lines"]) == 3 and ctx["refs"].startswith("there = the pond's shore")


# ====================================================================== area focus (operator)
def test_focus_choice_prefers_the_circle_and_keeps_claims():
    sim = Simulation(Config(seed=42, plan_clock="model", lifeforms=False, curiosity=False))
    try:
        sim.run_for(40.0)
        f = sim.fleet
        r, o = sim.rovers[0], sim.rovers[1]
        r.focus, r.goal = None, None
        base = f.choose_goal(sim.t, r)
        assert base is not None
        far = max(f.usable_frontiers(sim.t), key=lambda fr: math.hypot(fr[0] - base[0][0], fr[1] - base[0][1]))
        r.start_focus(sim.t, far[0], far[1], 8.0, "operator", "the test area", 60.0)
        r.focus["arrived"] = sim.t                     # as if there: only unknown ground nearby counts
        res = f.choose_goal(sim.t, r)
        assert res is not None
        (gx, gy), _path, _size = res
        assert math.hypot(gx - far[0], gy - far[1]) <= 8.0 + sim.cfg.focus_margin, (res[0], far)
        # another rover heading there claims it: the focused rover never duplicates that work
        o.goal, o_state = (gx, gy), o.state
        o.state = "exploring"
        res2 = f.choose_goal(sim.t, r)
        assert res2 is None or math.hypot(res2[0][0] - gx, res2[0][1] - gy) >= 8.0
        # other rovers leave the ground inside someone's focus circle alone (a utility penalty)
        r.focus["r"] = 30.0
        o.goal = None
        f.choose_goal(sim.t, o)
        with_focus = {c[4]: c for c in o.goal_options}
        saved, r.focus = r.focus, None
        f.choose_goal(sim.t, o)
        without = {c[4]: c for c in o.goal_options}
        r.focus = saved
        for key in set(with_focus) & set(without):
            c = without[key]
            inside = math.hypot(c[1] - far[0], c[2] - far[1]) <= 30.0
            assert with_focus[key][0] == pytest.approx(c[0] - (15.0 if inside else 0.0)), (key, inside)
        o.state = o_state
    finally:
        sim.close()


def test_operator_focus_stays_then_reports(world):
    sim, adv = world, world.advisor
    r = adv._rover("r1")
    fr = [f for f in sim.fleet.usable_frontiers(sim.t) if 10.0 < math.hypot(f[0] - r.x, f[1] - r.y) < 30.0]
    assert fr
    fx, fy = round(fr[0][0]), round(fr[0][1])
    seq = adv.chat_seq
    cid = adv.operator_message(f"rover 1, explore around {fx}, {fy}")
    assert step_until(sim, lambda: replies_to(sim, cid), 3.0)
    ack = replies_to(sim, cid)[0]
    assert ack["kind"] == "ack" and ack["from"] == "r1"
    assert re.match(rf"Exploring around \({fx}, {fy}\) for about 40 s, within 15 m; ", ack["text"]), ack["text"]
    assert r.focus is not None and r.focus["reason"] == "operator"
    fm = sim.tick_message()["rovers"][0]["focus"]
    assert set(fm) == {"x", "y", "r", "reason", "label", "ends_in", "mode"} and fm["reason"] == "operator"   # v8 mode
    assert fm["label"] == f"({fx}, {fy})" and 0.0 < fm["ends_in"] <= 40.0
    goals = []

    def done():
        fc = r.focus
        if fc is not None and fc["arrived"] is not None and r.goal is not None and r.state == "exploring" \
                and r.goal_t > fc["arrived"] + 0.5 and r.op_goal is None:
            goals.append(math.hypot(r.goal[0] - fx, r.goal[1] - fy))     # goals picked while there
        return any(m["text"].startswith(("Explored around", "Around", "Didn't")) for m in replies_to(sim, cid))

    assert step_until(sim, done, 230.0), [m["text"] for m in chat_after(sim, seq)]
    rep = [m for m in replies_to(sim, cid) if m["kind"] == "info"][-1]
    s = r.mem.focus[-1]
    assert rep["from"] == "r1" and rep["source"] == "rules" and EM_DASH not in rep["text"]
    if s["there"] and s["dur"] >= 3.0:
        assert rep["text"].startswith(f"Explored around ({fx}, {fy}) for {s['dur']:.0f} s: mapped {s['mapped']} m2 of "
                                      f"new ground; the area within 15 m is now {s['pct']}% mapped."), rep["text"]
    assert all(d <= 15.0 + sim.cfg.focus_margin + 0.5 for d in goals), goals
    assert r.focus is None and "focus" not in sim.tick_message()["rovers"][0]
    # "what did you find there?": the rover that explored it answers from its memory
    cid2 = adv.operator_message("what did you find there?")
    assert step_until(sim, lambda: replies_to(sim, cid2), 4.0)
    ans = replies_to(sim, cid2)[0]
    assert ans["from"] == "r1" and ans["kind"] == "reply" and f"({fx}, {fy})" in ans["text"], ans
    assert any(m["kind"] == "reply" for m in replies_to(sim, cid2))
    adv.operator_message("resume")
    sim.run_for(0.6)


def test_explore_side_of_the_lander_and_survey_then_focus(world):
    sim, adv = world, world.advisor
    r3 = adv._rover("r3")
    cid = adv.operator_message("rover 3, explore north of the lander")
    assert step_until(sim, lambda: replies_to(sim, cid), 3.0)
    assert r3.focus is not None and (r3.focus["x"], r3.focus["y"]) == (0.0, sim.cfg.focus_offset)
    assert r3.focus["label"] == "the area north of the lander"
    adv.operator_message("rover 3, resume")
    sim.run_for(0.6)
    assert r3.focus is None
    # "investigate X": survey first, then the focus around the place
    r2 = next(o for o in sim.rovers if o.state == "exploring" and o.op_goal is None and o.obs is None)
    task = {"label": "the test spot", "investigate": True, "cid": None, "qid": None, "at": (r2.x, r2.y),
            "focus": {"x": r2.x, "y": r2.y, "r": 12.0, "label": "the test spot"}}
    r2.start_survey(sim.t, task)
    assert step_until(sim, lambda: r2.survey is None, 50.0)
    assert r2.focus is not None and r2.focus["label"] == "the test spot" and r2.focus["reason"] == "operator"
    rep = [m for m in sim.advisor.chat if m["from"] == r2.id and m["text"].startswith("Survey of the test spot")][-1]
    assert rep["text"].endswith("Now exploring around it for about 40 s.") or len(rep["text"]) > 330, rep["text"]
    adv.operator_message(f"rover {r2.id[1:]}, resume")
    sim.run_for(0.6)


# ====================================================================== chat references end to end
def test_references_resolve_against_the_chat(world):
    sim, adv = world, world.advisor
    adv.operator_message("resume")
    sim.run_for(0.6)
    # a choose_place list, cancelled, then "go to the second one" picks from it
    cid = adv.operator_message("rover 2, go to the water source")
    assert step_until(sim, lambda: adv.requests, 3.0)
    (q,) = adv.requests_msg()
    assert q["kind"] == "choose_place" and q["rover"] == "r2"
    adv.operator_message("never mind")
    assert not adv.requests
    p2 = q["options"][1]
    seq = adv.chat_seq
    cid = adv.operator_message("go to the second one")
    r2, r3, r1 = adv._rover("r2"), adv._rover("r3"), adv._rover("r1")
    assert step_until(sim, lambda: r2.op_goal is not None, 5.0), [m["text"] for m in chat_after(sim, seq)]
    ack = replies_to(sim, cid)[0]
    assert ack["from"] == "r2" and ack["kind"] == "ack"            # the rover the chat was about goes
    assert math.hypot(r2.op_goal[0] - p2["x"], r2.op_goal[1] - p2["y"]) < 9.0
    # "same for rover 3": the same place
    cid = adv.operator_message("same for rover 3")
    assert step_until(sim, lambda: r3.op_goal is not None, 5.0), [m["text"] for m in chat_after(sim, seq)]
    assert replies_to(sim, cid)[0]["from"] == "r3"
    assert math.hypot(r3.op_goal[0] - p2["x"], r3.op_goal[1] - p2["y"]) < 9.0
    # "that lake": the body of water the chat mentioned last
    cid = adv.operator_message("rover 1, go to that lake")
    assert step_until(sim, lambda: r1.op_goal is not None, 5.0), [m["text"] for m in chat_after(sim, seq)]
    assert math.hypot(r1.op_goal[0] - p2["x"], r1.op_goal[1] - p2["y"]) < 9.0 and not adv.requests
    # "do that again": the same command once more (r1, the same lake)
    goal = r1.op_goal
    cid = adv.operator_message("do that again")
    assert step_until(sim, lambda: replies_to(sim, cid), 5.0), [m["text"] for m in chat_after(sim, seq)]
    rep = replies_to(sim, cid)[0]
    assert rep["from"] == "r1" and rep["kind"] == "ack", rep
    assert r1.op_goal is not None and math.hypot(r1.op_goal[0] - goal[0], r1.op_goal[1] - goal[1]) < 9.0
    # an unclear reference gets a short question
    adv.operator_message("resume")
    sim.run_for(0.6)
    saved, adv.mem = adv.mem, ChatMemory()
    try:
        cid = adv.operator_message("investigate it")
        assert step_until(sim, lambda: replies_to(sim, cid), 3.0)
        rep = replies_to(sim, cid)[0]
        assert rep["kind"] == "question" and rep["text"].startswith("Where do you mean?")
        cid = adv.operator_message("do that again")
        assert step_until(sim, lambda: replies_to(sim, cid), 3.0)
        assert replies_to(sim, cid)[0]["text"].startswith("Which command should I repeat?")
    finally:
        adv.mem = saved


# ====================================================================== curiosity (interest focus)
def test_interest_trigger_cooldown_and_limits():
    sim = Simulation(Config(seed=42, plan_clock="model", lifeforms=False))
    try:
        sim.run_for(30.0)
        f = sim.fleet
        assert step_until(sim, lambda: any(o.state == "exploring" and o.op_goal is None for o in sim.rovers), 60.0)
        r = next(o for o in sim.rovers if o.state == "exploring" and o.op_goal is None)
        r.focus, r.pending_interest, r.interest_next = None, None, 0.0
        fr = min(f.usable_frontiers(sim.t), key=lambda fr_: math.hypot(fr_[0] - r.x, fr_[1] - r.y))
        x, y = fr[0], fr[1]
        f.interest_spots = []
        assert r.note_interest(sim.t, x, y, "ground", "rougher ground (test)", "the rough ground")
        assert not r.note_interest(sim.t, x + 30.0, y, "water", "water")       # one pending at a time
        assert step_until(sim, lambda: r.focus is not None or r.pending_interest is None, 10.0)
        assert r.focus is not None and r.focus["reason"] == "interest", r.pending_interest
        ev = [e for e in f.events if e["rover"] == r.id and "Sticking around" in e["msg"]]
        assert ev and ev[-1]["msg"] == (f"{r.id} found rougher ground (test) here. Sticking around for about 60 s to "
                                        f"look for more.")
        assert sim.tick_message()["rovers"][r.idx]["focus"]["reason"] == "interest"
        # not during a focus, and no other rover where this one already concentrates
        assert not r.note_interest(sim.t, x + 20.0, y, "water", "water")
        other = next(o for o in sim.rovers if o is not r)
        other.focus, other.pending_interest, other.interest_next = None, None, 0.0
        assert not other._interest_ok(sim.t, x + 1.0, y)
        # the end: cooldown, and never the same spot twice
        r.end_focus(sim.t, "time")
        assert r.focus is None and r.interest_next == pytest.approx(sim.t + sim.cfg.interest_cooldown_s)
        assert not r.note_interest(sim.t, x + 40.0, y, "water", "water")
        r.interest_next = 0.0
        assert not r._interest_ok(sim.t, x + 2.0, y)                             # within 15 m of the last spot
        assert r._interest_ok(sim.t, x + 40.0, y) or r.state != "exploring"
        # never while returning or on a low battery
        st, bat = r.state, r.battery
        r.state = "returning"
        assert not r._interest_ok(sim.t, x + 40.0, y)
        r.state, r.battery = st, 0.3
        assert not r._interest_ok(sim.t, x + 40.0, y)
        r.battery = bat
        # what triggers it: a new target, ground that differs, a cluster of spires
        r.pending_interest = None
        k = next((k for k, ts in enumerate(f.targets) if ts.status is None), None)
        if k is not None and r._interest_ok(sim.t, f.targets[k].spec.x, f.targets[k].spec.y):
            f.on_target_seen(sim.t, r, k, 6.0)
            assert r.pending_interest is not None and r.pending_interest["kind"] == "target"
            assert r.pending_interest["what"].endswith(f"({f.targets[k].spec.id})")
        r.pending_interest, r.state = None, "exploring"
        f.interest_spots = []
        for o in sim.rovers:
            o.focus = None
        fm = sim.map
        idx = np.flatnonzero(fm.known.ravel())[:80]
        old = fm.slope.ravel()[idx].copy()
        r._ground = [3.0, float(np.median(fm.h.ravel()[idx])), 30]
        try:
            fm.slope.ravel()[idx] = 20.0
            r._ground_novelty(sim.t, fm, idx)
        finally:
            fm.slope.ravel()[idx] = old
        assert r.pending_interest is not None and r.pending_interest["kind"] == "ground", r.pending_interest
        assert r.pending_interest["what"] == "rougher ground (median slope 20 deg after 3 deg)"
        hz = sim.hazards
        sp = hz.all_spires()
        pair = next(((a, b) for a in sp for b in sp if a is not b and math.hypot(a.x - b.x, a.y - b.y) < 8.0), None)
        if pair is not None:
            r.pending_interest = None
            f.spires_seen.clear()
            f.interest_spots = []
            for s_ in pair:
                i, j = fm.ij(s_.x, s_.y)
                fm.tall_t[j, i] = sim.t
            r.x, r.y = (pair[0].x + pair[1].x) / 2 + 3.0, (pair[0].y + pair[1].y) / 2
            r._check_spires(sim.t)
            assert r.pending_interest is not None and r.pending_interest["kind"] == "spires"
            assert r.pending_interest["what"].startswith("a cluster of ")
    finally:
        sim.close()


# ====================================================================== following a new herd while asking
def _bring_herd(sim, rid="r1"):
    for _ in range(100):
        try:
            return sim.hazards.trigger("lifeform", rid)
        except HazardError:
            sim.step()
    raise AssertionError("could not bring a herd")


def test_follow_a_new_herd_while_asking():
    p = fake_planner()
    sim = Simulation(Config(seed=42, plan_clock="model", autonomy_mode="supervised"), planner=p, wait_for_gemma=True)
    try:
        sim.run_for(5.0)
        _bring_herd(sim)
        r, adv = sim.rovers[0], sim.advisor
        assert step_until(sim, lambda: adv.requests, 15.0), [e["msg"] for e in sim.fleet.events][-8:]
        (q,) = adv.requests_msg()
        assert q["kind"] == "lifeform" and q["rover"] == "r1" and q["expires_in"] is None
        assert "I'm following at about 6 m" in q["question"]
        assert [o["label"] for o in q["options"]] == ["Keep following", "Observe from here", "Keep exploring"]
        assert not r.question_hold and r.obs["mode"] == "follow" and r.obs["asking"] == q["id"]
        # the herd walks off: the rover goes with it, never closer than the planetary protection distance
        lf = sim.lifeforms
        herd = min(lf.herds, key=lambda h: math.hypot(h.centre()[0] - r.x, h.centre()[1] - r.y))
        cx, cy = herd.centre()
        base = math.atan2(cy - r.y, cx - r.x)
        home = None
        for off in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
            for d in (12.0, 10.0, 8.0):
                hx, hy = cx + d * math.cos(base + off), cy + d * math.sin(base + off)
                if lf.spot_ok(hx, hy) and math.hypot(hx - r.x, hy - r.y) < sim.cfg.observe_leash - 4.0:
                    home = (hx, hy)
                    break
            if home is not None:
                break
        assert home is not None
        herd.cx, herd.cy = home                         # they wander off to graze over there
        for c in herd.members:
            c.goal, c.next_goal, c.pause_until = home, sim.t + 120.0, 0.0
        x0, y0, odo0 = r.x, r.y, r.odometer
        dmin = 1e9
        for _ in range(int(40.0 / sim.agent_dt)):
            sim.step()
            dmin = min(dmin, min(math.hypot(c.x - r.x, c.y - r.y) for c in lf.creatures))
        assert math.hypot(r.x - x0, r.y - y0) > 1.5, (r.odometer - odo0, r.x - x0, r.y - y0)
        assert dmin >= sim.cfg.observe_min_dist - 0.7, dmin
        assert adv.requests and adv.requests_msg()[0]["id"] == q["id"]          # still waiting, still following
        assert r.state == "observing" and r.obs["asking"] == q["id"]
        assert sim.t - r.obs["t0"] > 40.0                                     # no time limit while asking
        # safety wins: a hazard ends the follow and closes the question, with a chat line
        seq = adv.chat_seq
        r.on_anomaly(sim.t, "slip", {"slip": 0.8, "wheel_speed": 0.3, "ground_speed": 0.02}, "Testing.")
        assert r.obs is None and not adv.requests and adv.closed[q["id"]] == "superseded"
        msg = [m for m in chat_after(sim, seq) if m["request_id"] == q["id"]]
        assert msg and msg[0]["kind"] == "info" and msg[0]["text"].startswith(
            "Stopped following the herd: soft ground (my wheels slipped).")
        # that herd is not asked about again (the slip itself may raise an anomaly question)
        n_life = len(adv.asked_herds)
        for _ in range(int(20.0 / sim.agent_dt)):
            sim.step()
        assert len(adv.asked_herds) == n_life and not any(q_["kind"] == "lifeform" for q_ in adv.requests.values())
    finally:
        p.close()
        sim.close()


# ====================================================================== waiting never outlasts the battery
def test_waiting_rover_returns_before_the_battery_runs_out():
    p = fake_planner()
    sim = Simulation(Config(seed=42, operator_timeout_s=0.0, autonomy_mode="supervised", **LEGACY), planner=p,
                     wait_for_gemma=True)
    try:
        sim.run_for(12.0)
        adv = sim.advisor
        r = next(o for o in sim.rovers if o.state == "exploring")
        r.on_anomaly(sim.t, "new_obstacle", {"at": (r.x + 4.0, r.y), "dist": 4.0, "cells": 6}, "Testing.")
        sim.step()
        (q,) = adv.requests_msg()
        assert q["rover"] == r.id and r.question_hold
        r.start_focus(sim.t, r.x, r.y, 10.0, "operator", "the test area", 120.0)
        sim.run_for(2.0)
        assert r.speed < 0.15 and adv.requests                         # holds while it waits
        r.home_distance(sim.t)
        r.battery = r.energy_for_distance(r.home_distance(sim.t)) + r.battery_margin + 0.004
        r.next_home_check = 0.0
        seq = adv.chat_seq
        assert step_until(sim, lambda: r.state == "returning", 30.0), (r.state, r.battery, r.need_home)
        assert r.return_reason == "battery" and not r.question_hold and r.focus is None
        assert not adv.requests and adv.closed[q["id"]] == "superseded"
        msg = [m for m in chat_after(sim, seq) if m["request_id"] == q["id"]]
        assert msg and msg[0]["kind"] == "info" and re.match(
            r"Battery \d+%, I can't wait any longer: heading back to the lander\. I'll ask again if it still matters\.$",
            msg[0]["text"]), msg
        x0, y0 = r.x, r.y
        sim.run_for(4.0)
        assert math.hypot(r.x - x0, r.y - y0) > 0.5                     # on its way home
        # a rover held on the pad keeps charging instead of sitting until the battery is empty
        h = next(o for o in sim.rovers if o is not r)
        h.held, h.state = True, "idle"
        a = h.qadr
        h.read_pose()
        sim.data.qpos[a:a + 2] = h.dock
        h.x, h.y = h.dock
        h.battery = 0.5
        sim.step()
        assert h.state == "charging" and h.held
    finally:
        p.close()
        sim.close()


def test_no_em_dash_in_v7_files():
    for rel in ("redshift/chatmem.py", "redshift/advisor.py", "redshift/rover.py", "redshift/gemma.py",
                "redshift/fleet.py", "redshift/memory.py", "tests/test_v7.py", "README.md", "GEMMA.md", "../PROTOCOL.md"):
        assert EM_DASH not in (ROOT / rel).read_text(encoding="utf-8"), rel
