"""Protocol v5: asking the rovers questions (intent, memory, fact sheets, grounded answers from Gemma or
the rules, fleet vs per-rover routing, reply_to), rover stats, and the larger camera frames.

    python -m pytest -q tests/test_v5.py        (about 60 s)

One sim (seed 42, the 120 m default map, model clock) runs about 230 sim seconds with a rockfall, a
herd walking into view and a dust storm, with the fake Gemma model; most tests then ask it questions.
"""
from __future__ import annotations

import io
import math
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import gemma as g  # noqa: E402
from redshift.answers import TOPICS, fact_lines, fleet_sheet, rover_sheet, rule_answer  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.gemma import FakeBackend, GemmaConfig, GemmaPlanner  # noqa: E402
from redshift.hazards import HazardError  # noqa: E402
from redshift.sim import Simulation  # noqa: E402

EM_DASH = chr(0x2014)
STATS_KEYS = {"distance_m", "mapped_m2", "targets_found", "samples_taken", "hazards_met", "lifeforms_seen", "replans"}
FLEET = {"rovers": [{"id": "r1", "name": "Rover 1", "color": "#ff7a45"}, {"id": "r2", "name": "Rover 2", "color": "#36cfc9"},
                    {"id": "r3", "name": "Rover 3", "color": "#9254de"}],
         "targets": [{"id": "t4", "kind": "ice"}]}


class BadNumberFake(FakeBackend):
    """Answers with a number that is in no fact sheet."""

    def _answer(self, req):
        return {"text": "I have driven 98765 m and found everything."}


def planner(backend=None, **cfg) -> GemmaPlanner:
    p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0, **cfg),
                     backend=backend if backend is not None else FakeBackend(0.01))
    assert p.wait_ready(2.0), p.stats()
    return p


def trigger(sim, kind, rover=None, tries=150):
    for _ in range(tries):
        try:
            return sim.hazards.trigger(kind, rover)
        except HazardError:
            sim.run_for(1.0)
    raise AssertionError(f"could not place {kind}")


@pytest.fixture(scope="module")
def world():
    p = planner()
    sim = Simulation(Config(seed=42, plan_clock="model"), planner=p, wait_for_gemma=True)
    ticks = []
    sim.run_for(90.0)
    ticks.append(sim.tick_message())
    trigger(sim, "rockfall")
    sim.run_for(30.0)
    trigger(sim, "lifeform", "r3")
    sim.run_for(60.0)
    ticks.append(sim.tick_message())
    trigger(sim, "dust_storm")
    sim.run_for(50.0)
    ticks.append(sim.tick_message())
    yield sim, p, ticks
    sim.close()
    p.close()


def ask(sim, text, rover=None, steps=12):
    """Operator message, then step until the replies to it are in. Returns (operator id, replies)."""
    cid = sim.advisor.operator_message(text, rover)
    for _ in range(steps):
        sim.step()
        if not sim.advisor.answers and not sim.advisor.commands and not sim.advisor.go_jobs:
            break
    return cid, [m for m in sim.advisor.chat if m.get("reply_to") == cid]


def numbers(text: str) -> list:
    return g.numbers_in_text(g._strip_thousands(text))


def fact_numbers(sheet) -> list:
    out = []
    for topic in TOPICS:
        for ln in fact_lines(sheet, topic):
            out += [v for _, v in numbers(ln)]
    return out


# ====================================================================== intent
QUESTIONS = [
    ("rover 2, what have you explored?", "r2", "exploration"), ("what did you find?", None, "finds"),
    ("how's your battery?", None, "battery"), ("why are you going there?", None, "why"),
    ("seen any life?", None, "lifeforms"), ("where are you?", None, "location"), ("any problems?", None, "hazards"),
    ("what has the fleet found so far?", "all", "finds"), ("everyone, report battery", "all", "battery"),
    ("status", None, "status"), ("r3 status report", "r3", "status"), ("tell me about your samples", None, "samples"),
    ("did you sample t4?", None, "samples"), ("found any water?", None, "water"), ("battery?", None, "battery"),
    ("the purple rover, where are you headed?", "r3", "why"), ("rover 1, what's the temperature?", "r1", "unknown"),
]
COMMANDS = [("r1 go to 10, -5", "go_to"), ("can you go home?", "return_home"), ("go north?", "explore_direction"),
            ("report back to the lander", "return_home"), ("rover 2 hold", "hold"), ("sample t4", "sample_target"),
            ("could you explore east", "explore_direction")]


def test_parser_tells_questions_from_commands():
    for text, rover, topic in QUESTIONS:
        p = g.parse_command(text, FLEET)
        assert p["command"] == {"type": "question", "rover": rover, "topic": topic}, (text, p["command"])
        assert p["confidence"] > 0.5
    for text, typ in COMMANDS:
        assert g.parse_command(text, FLEET)["command"]["type"] == typ, text
    assert g.question_intent("can you go home?", FLEET["rovers"])[0] == "command"
    assert g.question_intent("battery?", FLEET["rovers"])[0] == "weak"
    assert g.wants_each("everyone, report battery") and not g.wants_each("what has the fleet found?")


def _command_result(text, answer: dict):
    req = g.make_command_request(1, 10.0, text, FLEET)
    base = {"command": "unclear", "rover": "none", "direction": "none", "x": 0, "y": 0, "radius": 0,
            "target_id": "none", "topic": "none", "question": "", "ack": "", "confidence": 0.9}
    return g.validate(req, __import__("json").dumps(dict(base, **answer)))


def test_gemma_question_intent_is_validated_and_cross_checked():
    r = _command_result("rover 2, what have you found?", {"command": "question", "rover": "r2", "topic": "finds"})
    assert r.ok and r.command == {"type": "question", "rover": "r2", "topic": "finds"}
    # a small model turning a question into an order is caught by the keyword rules
    r = _command_result("did you sample t4?", {"command": "sample_target", "target_id": "t4"})
    assert r.ok and r.command["type"] == "question" and r.command["topic"] == "samples"
    # ... and an order it called a question
    r = _command_result("r1 go to 10, -5", {"command": "question", "rover": "r1", "topic": "status"})
    assert r.command == {"type": "go_to", "rover": "r1", "x": 10.0, "y": -5.0}
    # an unknown topic takes the keyword topic; low confidence never blocks a question
    r = _command_result("how is your battery", {"command": "question", "topic": "banana", "confidence": 0.2})
    assert r.ok and r.command["topic"] == "battery"
    schema = g.build_schema(g.make_command_request(1, 0.0, "status", FLEET))
    assert "question" in schema["properties"]["command"]["enum"] and "topic" in schema["required"]


# ====================================================================== answer validation (Gemma's reply)
FACTS = ["I've mapped about 1,904 m2 (13.2% of the 14,400 m2 map), mostly north-east of the lander",
         "I've driven 410 m", "I found 3 targets: t4 ice, t7 mineral and t9 ice",
         "my battery is at 64%; I head home below about 31% (about 16% to get back plus a 15% margin)"]


def _answer(text, facts=FACTS, speaker="r2"):
    req = g.make_answer_request(1, speaker, 50.0, "what have you explored?", "exploration", facts, "RULES")
    return g.validate(req, __import__("json").dumps({"text": text}))


def test_answer_validation_numbers_ids_and_length():
    ok = _answer("I've mapped about 1,900 m2, mostly north-east of the lander, and driven 410 m.")
    assert ok.ok and ok.source == "gemma" and ok.answer.startswith("I've mapped")
    assert _answer("About 2,000 m2 so far, 13% of the map. Battery 0.64.").ok         # rounding and 64% as 0.64
    assert _answer("Found t4 and t9, both ice.").ok
    bad = _answer("I've mapped 2,500 m2 and driven 410 m.")
    assert not bad.ok and bad.answer == "RULES" and "number 2500" in bad.error
    bad = _answer("I found t12, a vent.")
    assert not bad.ok and "id t12" in bad.error
    assert not _answer("One. Two. Three. Four sentences.").ok
    assert not _answer("word " * 70).ok
    assert not _answer("").ok
    req = g.make_answer_request(1, "r2", 50.0, "q?", "exploration", FACTS, "RULES")
    assert not g.validate(req, '{"txt": "no"}').ok
    em = _answer(f"I've driven 410 m {EM_DASH} mostly north-east.")
    assert em.ok and EM_DASH not in em.answer
    prompt = g.build_prompt(req)[1]
    assert "You are rover r2." in prompt and "- I've driven 410 m" in prompt
    assert g.build_schema(req) == {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def test_fake_backend_answers_are_grounded():
    fb = FakeBackend(0.0)
    for topic in TOPICS:
        req = g.make_answer_request(1, "r2", 5.0, "q?", topic, FACTS, "RULES")
        content, _ = fb.chat(req, *g.build_prompt(req), g.build_schema(req), 1.0)
        res = g.validate(req, content)
        assert res.ok, (topic, res.error)
    req = g.make_answer_request(1, "r2", 5.0, "what's the temperature?", "unknown", FACTS, "RULES")
    assert "don't know" in g.validate(req, fb.chat(req, *g.build_prompt(req), g.build_schema(req), 1.0)[0]).answer


# ====================================================================== memory, fact sheets, rule answers
def test_rule_answers_cover_every_topic_and_stay_grounded(world):
    sim, _p, _ticks = world
    sheets = [rover_sheet(r) for r in sim.rovers] + [fleet_sheet(sim)]
    for sheet in sheets:
        nums = fact_numbers(sheet)
        for topic in TOPICS:
            a = rule_answer(sheet, topic)
            assert 10 < len(a) <= 360 and EM_DASH not in a and a[-1] in ".!?", (sheet["speaker"], topic, a)
            for written, v in numbers(a):
                assert any(g._matches_rounded(written, v, w) for w in nums), (sheet["speaker"], topic, written, a)
            for m in re.finditer(r"\b[tk]\d+\b", a):
                assert m.group(0) in " ".join(fact_lines(sheet, "unknown") + fact_lines(sheet, topic)), (topic, a)
    r = sim.rovers[1]
    s = rover_sheet(r)
    assert f"{s['battery']}%" in rule_answer(s, "battery")
    assert f"{s['mapped']:,} m2" in rule_answer(s, "exploration") and f"{s['driven']:,} m" in rule_answer(s, "exploration")
    assert f"({round(r.x)}, {round(r.y)})" in rule_answer(s, "location")
    assert "Why:" in rule_answer(s, "why")
    assert "can't answer" in rule_answer(s, "unknown")
    for rv in sim.rovers:
        sv = rover_sheet(rv)
        found = [ts.spec.id for ts in sim.fleet.targets if ts.found_by == rv.id]
        assert sv["n_targets"] == len(found)
        if found:
            assert found[0] in rule_answer(sv, "finds")
        assert ("haven't seen anything move" in rule_answer(sv, "lifeforms")) == (not rv.mem.life and not rv.mem.sightings)
    fs = fleet_sheet(sim)
    assert f"{fs['n_found']} of {fs['n_total']} targets" in rule_answer(fs, "finds")
    assert rule_answer(fs, "battery").count("%") == len(sim.rovers)


def test_memory_records_hazards_life_and_mapping(world):
    sim, _p, _ticks = world
    mems = [r.mem for r in sim.rovers]
    assert sum(m.n_hazards for m in mems) >= len(sim.rovers)            # everyone met the dust storm
    assert all(m.hazard_kinds.get("dust") for m in mems)
    assert any(h["kind"] == "dust" and "LiDAR range down" in h["what"] for m in mems for h in m.hazards)
    assert any(m.sightings for m in mems)
    total_cells = sum(m.mapped_cells for m in mems)
    assert abs(total_cells - sim.map.known_count) <= 0.02 * sim.map.known_count + 5
    for m in mems:
        assert len(m.hazards) <= 12 and len(m.events) <= 8 and len(m.life) <= 6 and len(m.targets) <= 24
        if m.mapped_cells:
            assert m.sectors() and abs(sum(sh for _, sh in m.sectors()) - 1.0) < 1e-6


# ====================================================================== chat: routing, sources, reply_to
def test_questions_get_grounded_replies_from_the_right_speaker(world):
    sim, _p, _ticks = world
    cid, rep = ask(sim, "rover 2, what have you explored?")
    assert len(rep) == 1 and rep[0]["from"] == "r2" and rep[0]["kind"] == "reply" and rep[0]["source"] == "gemma"
    assert rep[0]["to"] == "operator" and rep[0]["reply_to"] == cid and "m2" in rep[0]["text"]
    cid, rep = ask(sim, "what has the fleet found so far?", rover="r1")
    assert [m["from"] for m in rep] == ["fleet"] and "targets" in rep[0]["text"]
    cid, rep = ask(sim, "everyone, report battery")
    assert sorted(m["from"] for m in rep) == ["r1", "r2", "r3"] and all("%" in m["text"] for m in rep)
    cid, rep = ask(sim, "how's your battery?", rover="r3")
    assert [m["from"] for m in rep] == ["r3"]
    cid, rep = ask(sim, "where are you?")
    assert [m["from"] for m in rep] == ["fleet"]
    cid, rep = ask(sim, "rover 1, what's the temperature?")
    assert rep[0]["from"] == "r1" and "know" in rep[0]["text"]
    assert sim.advisor.n_answers["gemma"] >= 6


def test_bad_or_slow_or_missing_gemma_falls_back_to_rules(world):
    sim, good, _ticks = world
    adv = sim.advisor
    for p, why in ((planner(BadNumberFake(0.01)), "bad number"),
                   (planner(FakeBackend(0.6), budget_ms=150.0), "slow"),
                   (GemmaPlanner(GemmaConfig(enabled=False), start=False), "disabled")):
        adv.planner = p
        try:
            cid, rep = ask(sim, "rover 3, what did you find?")
            assert len(rep) == 1 and rep[0]["source"] == "rules" and rep[0]["kind"] == "reply", why
            assert "98765" not in rep[0]["text"]
            assert rep[0]["text"] == rule_answer(rover_sheet(sim.rovers[2]), "finds"), why
        finally:
            adv.planner = good
            p.close()


def test_command_acks_and_refusals_carry_reply_to(world):
    sim, _p, _ticks = world
    cid, rep = ask(sim, "rover 2 go to 500, 500")
    assert rep and rep[0]["kind"] == "refuse" and rep[0]["reply_to"] == cid
    r1 = sim.rovers[0]
    x, y = round(r1.x + 3.0), round(r1.y)
    cid, rep = ask(sim, f"r1 go to {x}, {y}", steps=40)
    assert rep and rep[-1]["kind"] in ("ack", "refuse") and all(m["reply_to"] == cid for m in rep), rep
    cid, rep = ask(sim, "resume")
    assert rep and rep[0]["kind"] == "ack" and rep[0]["reply_to"] == cid
    op = [m for m in sim.advisor.chat if m["id"] == cid][0]
    assert op["from"] == "operator" and op["reply_to"] is None


# ====================================================================== stats
def test_rover_stats_in_every_tick(world):
    sim, _p, ticks = world
    for t in ticks + [sim.tick_message()]:
        for r in t["rovers"]:
            st = r["stats"]
            assert set(st) == STATS_KEYS, st
            assert all(isinstance(v, (int, float)) and v >= 0 for v in st.values())
    last = {r["id"]: r["stats"] for r in sim.tick_message()["rovers"]}
    first = {r["id"]: r["stats"] for r in ticks[0]["rovers"]}
    for rv in sim.rovers:
        assert last[rv.id]["distance_m"] == round(rv.odometer, 1)
        assert all(last[rv.id][k] >= first[rv.id][k] for k in STATS_KEYS)
    f = sim.fleet
    assert sum(s["targets_found"] for s in last.values()) == sum(1 for ts in f.targets if ts.status is not None)
    assert sum(s["samples_taken"] for s in last.values()) == sum(1 for ts in f.targets if ts.status == "sampled")
    assert sum(s["mapped_m2"] for s in last.values()) == pytest.approx(f.map.known_count * 0.25, rel=0.03)
    assert all(s["hazards_met"] >= 1 for s in last.values())


# ====================================================================== camera frames up to 1280 x 720
class FakeRunner:
    def __init__(self, sim):
        self.sim, self.cfg = sim, sim.cfg
        self._lock = threading.Lock()

    @contextmanager
    def locked(self):
        with self._lock:
            yield self.sim


def test_camera_sizes_up_to_1280x720(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from PIL import Image

    from redshift import camera, server

    st = camera.renderer_status()
    if not st["ok"]:
        pytest.skip(f"no OpenGL here: {st['error']}")
    assert (camera.MAX_W, camera.MAX_H) == (1280, 720)
    sim = Simulation(Config(seed=42))
    assert int(sim.model.vis.global_.offwidth) >= 1280 and int(sim.model.vis.global_.offheight) >= 720
    monkeypatch.setattr(server.state, "runner", FakeRunner(sim))
    app = FastAPI()
    app.include_router(camera.router)
    with TestClient(app) as client:
        for (w, h), want in (((1280, 720), (1280, 720)), ((1920, 1080), (1280, 720)), ((2000, 300), (1280, 300)),
                             ((640, 480), (640, 480))):
            r = client.get("/api/camera/r1.jpg", params={"w": w, "h": h, "view": "chase"})
            assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == want, (w, h)
    rgb = camera.render_rgb(sim, "r2", "fpv", 1280, 720)
    assert rgb.shape == (720, 1280, 3) and rgb.std() > 8.0


def test_no_em_dash_in_v5_code():
    for name in ("answers.py", "memory.py", "visuals.py", "gemma.py", "advisor.py", "camera.py", "world.py"):
        assert EM_DASH not in (ROOT / "redshift" / name).read_text(encoding="utf-8"), name
    assert EM_DASH not in Path(__file__).read_text(encoding="utf-8")
