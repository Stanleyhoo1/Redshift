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
