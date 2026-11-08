"""Protocol v7, launch settings: the settings table and its parser, the thresholds derived from gravity,
Gemma's gravity text, the setup phase, GET /api/config, POST /api/launch and POST /api/reset.

    python -m pytest -q tests/test_v7_settings.py        (about a minute and a half)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import config as C  # noqa: E402
from redshift.config import PRESETS, SETTINGS, SETTINGS_BY_KEY, Config  # noqa: E402

EM_DASH = chr(0x2014)
V7_FLAGS = ("mesas", "canyon", "dunes", "boulder_fields", "pits", "geysers", "ice_sheets", "ground_detail", "flora")
V7_ENV = {f: f.upper() for f in V7_FLAGS}              # MESAS, CANYON, ..., FLORA
SETTING_KEYS = ("gravity_g", "seed", "num_rovers", "map_size", "water", "spires", "lifeforms", "mesas", "canyon",
                "dunes", "boulder_fields", "pits", "geysers", "ice_sheets", "flora", "ground_detail", "wear",
                "sim_speed")
SMALL = dict(map_size=60.0, water=False, spires=False, lifeforms=False, plan_clock="model")   # quick worlds
JPEG = b"\xff\xd8\xff\xe0" + bytes(300) + b"\xff\xd9"      # stands in for a camera frame


# ====================================================================== the settings table
def test_new_world_features_default_off():
    cfg = Config()
    for f in V7_FLAGS + ("wear",):
        assert getattr(cfg, f) is False, f
    assert cfg.wear_rate == 1.0


def test_settings_table_matches_the_spec():
    assert tuple(s.key for s in SETTINGS) == SETTING_KEYS
    assert set(SETTINGS_BY_KEY) == set(SETTING_KEYS)
    fields = {f.name for f in dataclasses.fields(Config)}
    for s in SETTINGS:
        assert s.key in fields, s.key
        assert s.type in ("int", "float", "bool", "choice"), s
        assert s.group in ("planet", "fleet", "sector", "features", "wear", "simulation"), s
        assert s.label and s.help and len(s.help) <= 90, s
        assert EM_DASH not in s.help + s.label, s
        if s.type in ("int", "float"):
            assert s.min < s.max and s.step > 0, s
    ranges = {s.key: (s.min, s.max, s.step, s.unit) for s in SETTINGS if s.type in ("int", "float")}
    assert ranges == {"gravity_g": (0.5, 3.0, 0.1, "g"), "seed": (0, 2147483647, 1, ""), "num_rovers": (1, 8, 1, ""),
                      "map_size": (60, 200, 10, "m"), "sim_speed": (0.5, 10.0, 0.5, "x")}
    assert {s.key for s in SETTINGS if s.type == "bool"} == {"water", "spires", "lifeforms", *V7_FLAGS}
    assert all(s.group == "features" for s in SETTINGS if s.type == "bool")
    wear = SETTINGS_BY_KEY["wear"]
    assert wear.type == "choice" and [c[0] for c in wear.choices] == ["off", "normal", "harsh"]
    assert C.WEAR_RATES == {"normal": 1.0, "harsh": 4.0}
    assert PRESETS == {"gravity_g": [{"label": "Earth", "value": 1.0}, {"label": "Redshift-b", "value": 2.0},
                                     {"label": "Heavy", "value": 3.0}]}
    js = [s.to_json() for s in SETTINGS]
    json.dumps(js)                                       # plain JSON
    assert js[0] == {"key": "gravity_g", "type": "float", "label": "Gravity", "group": "planet",
                     "help": js[0]["help"], "min": 0.5, "max": 3.0, "step": 0.1, "unit": "g"}
    w = next(j for j in js if j["key"] == "wear")
    assert [c["value"] for c in w["choices"]] == ["off", "normal", "harsh"] and all(c["help"] for c in w["choices"])
    assert "min" not in next(j for j in js if j["key"] == "water")


# ====================================================================== the parser
PARSE_CASES = [                                          # (request body, part of the one error, or None)
    ({}, None),
    ({"gravity_g": 1.5, "num_rovers": 4, "map_size": 100, "water": False, "mesas": True, "wear": "harsh",
      "sim_speed": 5, "seed": 7}, None),
    ({"gravity_g": 3}, None),                            # an int is a fine float
    ({"seed": 7.0}, None),                               # a whole float is a fine int
    ({"warp": 9}, "unknown setting 'warp' (use one of gravity_g, seed,"),
    ({"gravity_g": 3.5}, "gravity_g must be between 0.5 and 3 g"),
    ({"gravity_g": 0.4}, "gravity_g must be between 0.5 and 3 g"),
    ({"num_rovers": 0}, "num_rovers must be between 1 and 8"),
    ({"num_rovers": 9}, "num_rovers must be between 1 and 8"),
    ({"map_size": 300}, "map_size must be between 60 and 200 m"),
    ({"map_size": 65}, "map_size must be in steps of 10 m"),
    ({"sim_speed": 20}, "sim_speed must be between 0.5 and 10x"),
    ({"seed": -1}, "seed must be between 0 and 2147483647"),
    ({"seed": 2147483648}, "seed must be between 0 and 2147483647"),
    ({"seed": 10 ** 400}, "seed must be between 0 and 2147483647"),     # huge ints do not overflow
    ({"seed": 7.5}, "seed must be a whole number"),
    ({"num_rovers": True}, "num_rovers must be a number"),              # a bool where a number is expected
    ({"gravity_g": "2"}, "gravity_g must be a number"),
    ({"gravity_g": None}, "gravity_g must be a number"),
    ({"gravity_g": float("nan")}, "gravity_g must be a finite number"),
    ({"map_size": float("inf")}, "map_size must be a finite number"),
    ({"water": 1}, "water must be true or false"),
    ({"flora": "yes"}, "flora must be true or false"),
    ({"wear": "extreme"}, "wear must be one of off, normal, harsh"),
    ({"wear": True}, "wear must be one of off, normal, harsh"),
    ([1, 2], 'Body must be a JSON object like {"gravity_g": 1.5, "num_rovers": 4}'),
    ("gravity", "Body must be a JSON object"),
]


@pytest.mark.parametrize("body, part", PARSE_CASES)
def test_from_settings_table(body, part):
    cfg, errors = Config.from_settings(Config(seed=42), body)
    if part is None:
        assert errors == [] and isinstance(cfg, Config), errors
    else:
        assert cfg is None and len(errors) == 1 and part in errors[0], errors


def test_from_settings_applies_the_changes_and_keeps_the_rest():
    base = Config(seed=42, brain_budget_ms=25.0)
    cfg, errors = Config.from_settings(base, {"gravity_g": 1.5, "num_rovers": 4, "map_size": 100, "water": False,
                                              "mesas": True, "wear": "harsh", "seed": 7})
    assert errors == []
    assert (cfg.gravity_g, cfg.num_rovers, cfg.map_size, cfg.water, cfg.mesas, cfg.seed) == (1.5, 4, 100.0, False, True, 7)
    assert cfg.wear is True and cfg.wear_rate == 4.0
    assert isinstance(cfg.map_size, float) and isinstance(cfg.gravity_g, float) and isinstance(cfg.num_rovers, int)
    assert cfg.brain_budget_ms == 25.0 and cfg.spires is True and cfg.sim_speed == 3.0      # untouched
    off, _ = Config.from_settings(cfg, {"wear": "off"})
    assert off.wear is False and off.wear_rate == 1.0
    assert Config.from_settings(base, {})[0] is base                                      # nothing to change


def test_from_settings_reports_every_problem():
    cfg, errors = Config.from_settings(Config(), {"gravity_g": 9, "water": "no", "warp": 1, "num_rovers": 2})
    assert cfg is None and len(errors) == 3, errors


def test_settings_round_trip():
    s = Config().settings()
    assert tuple(s) == SETTING_KEYS
    assert s["map_size"] == 120 and isinstance(s["map_size"], int) and isinstance(s["seed"], int)
    assert s["gravity_g"] == 2.0 and isinstance(s["gravity_g"], float) and s["wear"] == "off"
    assert s["water"] is True and s["mesas"] is False
    for cfg in (Config(), Config(wear=True, water=False),
                Config(seed=7, gravity_g=1.0, num_rovers=5, map_size=150.0, mesas=True, flora=True, wear=True,
                       wear_rate=4.0, sim_speed=6.5)):
        sent = json.loads(json.dumps(cfg.settings()))          # through JSON, as the UI sends it back
        back, errors = Config.from_settings(Config(), sent)
        assert errors == [] and back == cfg, errors
    assert Config(wear=True, wear_rate=3.0).settings()["wear"] == "harsh"        # the nearest named rate
    assert Config(map_size=95.5).settings()["map_size"] == 95.5                # an odd env value, shown as is


# ====================================================================== from_env
def _clear_env(monkeypatch):
    for name in ("SEED", "NUM_ROVERS", "SIM_SPEED", "MAP_SIZE", "GRAVITY_G", "WEAR", *V7_ENV.values()):
        monkeypatch.delenv(name, raising=False)


def test_from_env_turns_the_new_features_on(monkeypatch):
    _clear_env(monkeypatch)
    cfg = Config.from_env()
    assert all(getattr(cfg, f) is True for f in V7_FLAGS)
    assert cfg.wear is True and cfg.wear_rate == 1.0
    assert (cfg.seed, cfg.num_rovers, cfg.sim_speed, cfg.map_size, cfg.gravity_g) == (42, 3, 3.0, 120.0, 2.0)


def test_from_env_switches(monkeypatch):
    _clear_env(monkeypatch)
    for name in V7_ENV.values():
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("WEAR", "off")
    cfg = Config.from_env()
    assert not any(getattr(cfg, f) for f in V7_FLAGS) and cfg.wear is False
    for value, want in (("harsh", (True, 4.0)), ("Normal ", (True, 1.0)), ("sometimes", (True, 1.0))):
        monkeypatch.setenv("WEAR", value)                       # unknown words give the default, normal
        cfg = Config.from_env()
        assert (cfg.wear, cfg.wear_rate) == want, value
    monkeypatch.setenv("FLORA", "yes")
    assert Config.from_env().flora is True and Config.from_env().mesas is False


def test_from_env_clamps_come_from_the_settings_table(monkeypatch):
    _clear_env(monkeypatch)
    for name, value, field, want in (("MAP_SIZE", "30", "map_size", 60.0), ("MAP_SIZE", "500", "map_size", 200.0),
                                     ("GRAVITY_G", "5", "gravity_g", 3.0), ("GRAVITY_G", "0.1", "gravity_g", 0.5),
                                     ("NUM_ROVERS", "12", "num_rovers", 8), ("NUM_ROVERS", "0", "num_rovers", 1),
                                     ("SIM_SPEED", "50", "sim_speed", 10.0), ("SIM_SPEED", "0.01", "sim_speed", 0.5),
                                     ("SEED", "-5", "seed", 0)):
        monkeypatch.setenv(name, value)
        got = getattr(Config.from_env(), field)
        assert got == want and type(got) is type(want), (name, value, got)
        monkeypatch.delenv(name)
    monkeypatch.setenv("MAP_SIZE", "125")                    # inside the range: kept as given (no step here)
    assert Config.from_env().map_size == 125.0
    assert Config.from_env(seed=9).seed == 9                  # overrides still win


# ====================================================================== gravity
def test_derived_thresholds_are_identical_at_2g():
    cfg = Config()
    assert cfg.rover_mass == 22.5
    assert cfg.climb_limit_deg == pytest.approx(21.25, abs=0.01)
    assert cfg.target_path_slope_deg == C.TARGET_PATH_SLOPE == 15.0
    assert cfg.reach_slope_deg == C.REACH_SLOPE == 18.0
    assert cfg.passable_slope_deg == C.PASSABLE_SLOPE == 20.0
    assert cfg.slope_floor_deg == cfg.slope_limit_floor_deg == 12.0
    assert cfg.sand_spin_load == C.SAND_SPIN_LOAD == 0.2
    assert cfg.tow_limit_deg == pytest.approx(8.35, abs=0.01)


def test_derived_thresholds_are_lower_at_3g():
    hi = Config(gravity_g=3.0)
    assert hi.climb_limit_deg == pytest.approx(13.98, abs=0.01)
    assert hi.target_path_slope_deg == pytest.approx(hi.climb_limit_deg - 2.0) and hi.target_path_slope_deg < 15.0
    assert hi.reach_slope_deg == pytest.approx(hi.climb_limit_deg - 1.0) and hi.reach_slope_deg < 18.0
    assert hi.passable_slope_deg == pytest.approx(hi.climb_limit_deg - 1.0) and hi.passable_slope_deg < 20.0
    assert hi.slope_floor_deg == pytest.approx(11.98, abs=0.01) and hi.slope_floor_deg < 12.0
    assert hi.sand_spin_load == 0.2                    # heavier than 2 g: sand loads the wheels at least as much
    assert hi.tow_limit_deg < Config().tow_limit_deg
    lo = Config(gravity_g=1.0)
    assert lo.sand_spin_load == pytest.approx(0.1)     # half the weight, half the wheel load in sand
    assert (lo.target_path_slope_deg, lo.reach_slope_deg, lo.passable_slope_deg, lo.slope_floor_deg) == \
        (15.0, 18.0, 20.0, 12.0)
    assert Config(gravity_g=0.5).climb_limit_deg == C.CLIMB_CAP_DEG == 30.0     # grip, not torque, limits it


def test_physics_table_for_the_setup_screen():
    rows = C.physics_table(Config())
    assert [r["gravity_g"] for r in rows] == [round(0.5 + 0.1 * k, 1) for k in range(26)]
    by_g = {r["gravity_g"]: r for r in rows}
    assert by_g[2.0] == {"gravity_g": 2.0, "climb_deg": 21.3, "tow_deg": 8.4, "grip_limited": False}
    assert by_g[0.5]["climb_deg"] == 30.0 and by_g[0.5]["grip_limited"] is True
    assert by_g[3.0]["climb_deg"] == 14.0
    climbs = [r["climb_deg"] for r in rows]
    assert climbs == sorted(climbs, reverse=True)
    assert all(r["tow_deg"] < r["climb_deg"] for r in rows)
    for r in rows:      # the wear and towing design's 0.8 * asin(4T/r / ((m1 + m2) g)), as is from 0.73 g up
        if r["gravity_g"] >= 0.8:
            plain = 0.8 * math.degrees(math.asin(4 * 7.2 / 0.18 / (2 * 22.5 * 9.81 * r["gravity_g"])))
            assert r["tow_deg"] == round(plain, 1), r
    assert by_g[0.5]["tow_deg"] == by_g[0.7]["tow_deg"] == 24.0     # below it the 30 deg grip cap holds
    json.dumps(rows)


def test_terrain_targets_and_traps_use_the_derived_slopes():
    from redshift import terrain
    from redshift.terrain import Terrain
    assert terrain.TARGET_PATH_SLOPE == 15.0 and terrain.REACH_SLOPE == 18.0       # still importable from terrain
    c2, c3 = Config(seed=42, water=False), Config(seed=42, water=False, gravity_g=3.0)
    t2, t3 = Terrain(c2), Terrain(c3)
    assert np.array_equal(t2.truth_heights, t3.truth_heights)        # same ground, only the limits change
    assert np.array_equal(t2.reach, t2.reachable(15.0))
    assert np.array_equal(t3.reach, t3.reachable(c3.target_path_slope_deg))
    assert t3.reach.sum() < t2.reach.sum() and not (t3.reach & ~t2.reach).any()
    for cfg in (c3, Config(seed=42, gravity_g=3.0), Config(seed=7, gravity_g=3.0)):      # with water too
        ter = t3 if cfg is c3 else Terrain(cfg)
        for tg in ter.targets:
            i, j = ter.ij(tg.x, tg.y)
            assert ter.reach[j, i], (cfg.seed, tg.id)                 # every target reachable at 3 g


def test_passable_ground_and_slope_floor_follow_gravity():
    from redshift.sim import Simulation
    from redshift.terrain import slope_deg
    for g, floor in ((2.0, 12.0), (3.0, Config(gravity_g=3.0).slope_floor_deg)):
        cfg = Config(seed=42, gravity_g=g, **SMALL)
        sim = Simulation(cfg)
        try:
            assert all(r.mass == pytest.approx(cfg.rover_mass) for r in sim.rovers)    # the model matches the formula
            H = sim.hazards.truth_heights(sim.terrain.truth_heights)
            want = (slope_deg(H, cfg.res) <= min(sim.fleet.slope_limit, cfg.passable_slope_deg)) & ~sim.terrain.wet
            assert np.array_equal(sim.hazards._passable_base(), want), g
            if g == 3.0:
                assert want.sum() < ((slope_deg(H, cfg.res) <= 20.0) & ~sim.terrain.wet).sum()
            assert sim.fleet.report_slope_stall(sim.t, sim.rovers[0], 13.0)
            assert sim.fleet.slope_limit == pytest.approx(floor), g
        finally:
            sim.close()


def _sand_suspect(sim, load: float) -> bool:
    """Feed rover 1's own wheel channels by hand: wheels at 0.54 m/s, body still, motors at `load` of the
    torque cap. True if the soft ground (sand) detector starts its slip timer."""
    r = sim.rovers[0]
    r.state, r.cmd_v, r.gyro_z, r.tilt, r.speed_fwd = "exploring", 0.5, 0.0, 0.0, 0.0
    r.spin_load, r.slip, r.slip_since = 0.0, 0.0, None
    for _ in range(8):
        sim.data.qvel[r.wdofs] = 3.0
        sim.data.actuator_force[r.acts] = load * sim.cfg.motor_torque
        r._read_drive_sensors(sim.t)
    return r.slip_since is not None


def test_sand_detector_scales_with_gravity():
    from redshift.sim import Simulation
    for g, light, heavy in ((2.0, False, True), (1.0, True, True)):
        sim = Simulation(Config(seed=42, gravity_g=g, **SMALL))
        try:
            assert _sand_suspect(sim, 0.15) is light, g        # 0.15 of the cap: sand at 1 g, not at 2 g
            assert _sand_suspect(sim, 0.30) is heavy, g
        finally:
            sim.close()


# ====================================================================== Gemma states the gravity
from redshift import gemma as g  # noqa: E402

PERCEPTION = {"battery": 0.64, "state": "exploring", "x": 3.21, "y": -1.04, "lidar_range": 8.0, "lidar_hits": 812,
              "pitch": 3.1, "roll": -1.2, "wheel_speed": 0.62, "ground_speed": 0.58, "slip": 0.06,
              "motor_load": 0.41, "spectro": None, "slope_limit_deg": 20.0, "home_distance": 31.2,
              "scan": [0.1] * 900}                       # the largest prompt inputs of test_gemma.py
CANDIDATES = [
    {"id": "f1", "kind": "frontier", "summary": "frontier 9 m north, 34 cells, slope max 12 deg, battery after return 58%"},
    {"id": "f2", "kind": "frontier", "summary": "frontier 21 m east, 80 cells, slope max 18 deg, battery after return 41%"},
    {"id": "f3", "kind": "frontier", "summary": "frontier 6 m west, 8 cells, slope max 6 deg, battery after return 66%"},
    {"id": "home", "kind": "return", "summary": "lander 31 m south-west, battery on arrival 60%"},
]
SLIP = {"type": "wheel_slip", "slip": 0.46, "duration_s": 2.4, "detail": "wheels spinning, not moving"}
OPTIONS = ["reroute", "back_out", "proceed_slowly", "wait"]
TRACK = {"distance": 6.2, "speed": 0.3, "size": 1.2}
FLEET = {"rovers": [{"id": "r1", "name": "Rover 1", "color": "#ff7a45", "x": 3.2, "y": -1.0, "state": "exploring",
                     "battery": 0.64},
                    {"id": "r2", "name": "Rover 2", "color": "#36cfc9", "x": -8.5, "y": 12.1, "state": "returning",
                     "battery": 0.31}],
         "targets": [{"id": "t4", "kind": "ice", "x": 12.0, "y": -4.5, "status": "found"}],
         "lander": [0.0, 0.0], "map_size": 80}
ANSWER_FACTS = ["I have mapped 1,904 m2, mostly north-east of the lander.", "I found 1 target: t4 (ice), 13 m east."]


def _prompts(gravity=None) -> dict:
    kw = {} if gravity is None else {"gravity_g": gravity}
    reqs = [g.make_choice_request(1, "r1", 0.0, PERCEPTION, CANDIDATES, **kw),
            g.make_interpretation_request(2, "r1", 0.0, PERCEPTION, SLIP, b"x", OPTIONS, **kw),
            g.make_description_request(3, "r1", 0.0, PERCEPTION, TRACK, b"x", **kw),
            g.make_command_request(4, 0.0, "rover 2 go to 10, -5", FLEET, **kw),
            g.make_answer_request(5, "r2", 0.0, "what have you found?", "finds", ANSWER_FACTS, "RULES", **kw)]
    return {r.kind: g.build_prompt(r) for r in reqs}


def test_gemma_prompts_at_2g_are_unchanged():
    assert g.SYSTEM_BASE == ("You advise an autonomous rover exploring an unknown super-Earth (gravity 2 g, no link "
                             "to Earth). Decide only from the rover's own sensor readings listed below. ")
    plain = _prompts()
    assert plain == _prompts(2.0)
    assert plain["choice"][0] == g.SYSTEM_CHOICE and plain["interpretation"][0] == g.SYSTEM_INTERPRET
    assert plain["describe"][0] == g.SYSTEM_DESCRIBE
    assert plain["command"][0] == g.SYSTEM_COMMAND and plain["answer"][0] == g.SYSTEM_ANSWER
    n = g.estimate_tokens("".join(plain["interpretation"]))
    assert n <= 397, n                    # the longest prompt is no longer than before v7 (budget: under 400)


@pytest.mark.parametrize("gravity, text", [(3.0, "an unknown super-Earth (gravity 3 g, no link to Earth)"),
                                           (2.5, "an unknown super-Earth (gravity 2.5 g, no link to Earth)"),
                                           (1.0, "an unknown planet (gravity 1 g, no link to Earth)"),
                                           (0.5, "an unknown planet (gravity 0.5 g, no link to Earth)")])
def test_gemma_prompts_state_the_configured_gravity(gravity, text):
    prompts = _prompts(gravity)
    for kind in ("choice", "interpretation", "describe"):
        system, user = prompts[kind]
        assert text in system and "gravity 2 g" not in system, kind
        assert g.estimate_tokens(system + user) < 400, kind
    # commands and answers name the same world, never a gravity number (they may only quote their own inputs)
    for kind, plain in (("command", g.SYSTEM_COMMAND), ("answer", g.SYSTEM_ANSWER)):
        system, user = prompts[kind]
        want = plain if gravity > 1.0 else plain.replace("a super-Earth", "a planet")
        assert system == want and ("super-Earth" in system) == (gravity > 1.0), kind
        assert "gravity" not in system and g.estimate_tokens(system + user) < 400, kind


class SystemRecorder(g.FakeBackend):
    """The fake model, keeping every system prompt it was sent."""

    def __init__(self):
        super().__init__(latency_s=0.0)
        self.systems = []

    def chat(self, req, system, user, schema, timeout):
        self.systems.append((req.kind, system))
        return super().chat(req, system, user, schema, timeout)


def _planner(backend):
    p = g.GemmaPlanner(g.GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0), backend=backend)
    assert p.wait_ready(2.0), p.stats()
    return p


def _result(p, ticket, timeout=5.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        r = p.poll(ticket)
        if r is not None:
            return r
        time.sleep(0.002)
    raise AssertionError(f"no result for ticket {ticket}")


def test_planner_submits_carry_the_gravity():
    rec = SystemRecorder()
    p = _planner(rec)
    try:
        assert _result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES, default="f1", gravity_g=3.0)).ok
        assert _result(p, p.submit_interpretation("r2", 1.0, PERCEPTION, SLIP, None, OPTIONS, gravity_g=0.5)).ok
        assert _result(p, p.submit_description("r3", 1.0, PERCEPTION, TRACK, None, gravity_g=1.0)) is not None
        assert _result(p, p.submit_choice("r1", 2.0, PERCEPTION, CANDIDATES, default="f1")).ok
        assert _result(p, p.submit_command(3.0, "rover 2 go to 10, -5", FLEET, gravity_g=0.5)) is not None
        assert _result(p, p.submit_answer("r2", 3.0, "what have you found?", "finds", ANSWER_FACTS, "RULES",
                                          gravity_g=1.0)) is not None
        assert _result(p, p.submit_command(4.0, "rover 2 go to 10, -5", FLEET)) is not None
    finally:
        p.close()
    (k1, s1), (k2, s2), (k3, s3), (k4, s4), (k5, s5), (k6, s6), (k7, s7) = rec.systems
    assert (k1, k2, k3, k4, k5, k6, k7) == ("choice", "interpretation", "describe", "choice", "command", "answer",
                                            "command")
    assert "super-Earth (gravity 3 g," in s1 and "planet (gravity 0.5 g," in s2 and "planet (gravity 1 g," in s3
    assert s4 == g.SYSTEM_CHOICE and s7 == g.SYSTEM_COMMAND          # the default is 2 g, as before
    assert s5 == g.SYSTEM_COMMAND.replace("a super-Earth", "a planet")
    assert s6 == g.SYSTEM_ANSWER.replace("a super-Earth", "a planet")


def test_advisor_sends_the_configured_gravity():
    from redshift.sim import Simulation
    rec = SystemRecorder()
    p = _planner(rec)
    legacy = dict(map_size=80.0, water=False, spires=False, lifeforms=False, drill_s=3.0, scoop_s=3.0, sniffer_s=3.0,
                  swab_s=3.0, sample_capacity=99, num_boulders=12)            # test_brain.py's scripted world
    sim = Simulation(Config(seed=42, gravity_g=1.0, plan_clock="model", operator_ask_first=False, **legacy),
                     planner=p, frame_source=lambda rid: JPEG, wait_for_gemma=True)
    try:
        sim.run_for(12.0)
        r = sim.rovers[0]
        r.on_anomaly(sim.t, "slip", {"slip": 0.82, "wheel_speed": 0.61, "ground_speed": 0.11}, "Testing.")
        sim.step()
        sim.advisor.operator_message("rover 1 hold")                    # Gemma parses a command
        sim.advisor.operator_message("what have you found?", "r1")      # then a question, answered by Gemma
        for _ in range(12):
            sim.step()
    finally:
        p.close()
        sim.close()
    kinds = [k for k, _ in rec.systems]
    assert {"interpretation", "command", "answer"} <= set(kinds), kinds
    for kind, system in rec.systems:
        if kind in ("choice", "interpretation", "describe"):
            assert "planet (gravity 1 g," in system and "gravity 2 g" not in system, kind
        else:
            assert "a planet" in system and "super-Earth" not in system, kind
