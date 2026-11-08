"""Two hackathon fixes: injected lifeform herds always get a reaction, and the onboard detector's class comes
before (and, when it is confident and contradicts, wins over) Gemma's obstacle label."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import vision  # noqa: E402
from redshift.advisor import detector_overrides  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.gemma import FakeBackend, GemmaConfig, GemmaPlanner  # noqa: E402
from redshift.hazards import HazardError  # noqa: E402
from redshift.sim import Simulation  # noqa: E402

EM_DASH = chr(0x2014)


def _inject(sim, rid):
    for _ in range(100):
        try:
            return sim.hazards.trigger("lifeform", rid)
        except HazardError:
            sim.step()
    raise AssertionError("could not bring a herd")


def _sighted(sim, rid, h, secs=10.0):
    """True once the rover observes a track near the injected herd (or logs a sighting of it)."""
    r = next(o for o in sim.rovers if o.id == rid)
    n0, t0 = len(sim.fleet.events), sim.t
    while sim.t < t0 + secs:
        sim.step()
        if r.state == "observing" and r.obs is not None:
            tr = sim.fleet.tracker.get(r.obs["track"])
            if tr is not None and math.hypot(tr.x - h["x"], tr.y - h["y"]) < 10.0:
                return True
    return False


def test_injected_herd_gets_a_reaction_even_when_other_rovers_observe():
    sim = Simulation(Config(seed=42, plan_clock="model"))
    try:
        sim.run_for(5.0)
        for rid in ("r1", "r2", "r3"):           # the third used to hit the observer cap: "no rover free"
            h = _inject(sim, rid)
            r = next(o for o in sim.rovers if o.id == rid)
            d = math.hypot(h["x"] - r.x, h["y"] - r.y)
            assert 4.0 <= d <= 9.0, d
            assert _sighted(sim, rid, h), (rid, [e["msg"] for e in sim.fleet.events][-6:])
    finally:
        sim.close()


def test_detector_override_rule():
    assert detector_overrides({"label": "spire", "conf": 0.99}, "rock")
    assert not detector_overrides({"label": "spire", "conf": 0.8}, "rock")
    assert not detector_overrides({"label": "boulder", "conf": 0.95}, "rock")
    assert detector_overrides({"label": "rock", "conf": 0.95}, "clear")


def test_confident_detector_wins_over_gemma_label():
    p = GemmaPlanner(GemmaConfig(fake=True, check_every_s=3600.0, timeout_s=2.0), backend=FakeBackend(0.0))
    assert p.wait_ready(2.0)
    sim = Simulation(Config(seed=42, plan_clock="model"), planner=p, wait_for_gemma=True)
    try:
        sim.run_for(3.0)
        r = sim.rovers[0]
        at = (r.x + 4.0 * math.cos(r.yaw), r.y + 4.0 * math.sin(r.yaw))
        obj = {"label": "spire", "conf": 0.99, "box": [0.4, 0.4, 0.5, 0.5], "dist": 4.0, "bearing": 0.0,
               "x": at[0], "y": at[1], "height": 0.5, "width": 0.8}
        frame = (r.vision.latest.frame + 1) if r.vision.latest is not None else 1
        r.vision.publish(vision.VisionResult(frame=frame, t=sim.t, view="fpv", ms=5.0, objects=[obj]))
        facts = {"kind": "new_obstacle", "t": sim.t, "at": at, "dist": 4.0, "cells": 6, "height": 0.6}
        job = {"rover": r.id, "kind": "new_obstacle", "t": sim.t, "at": at, "first": False,
               "anomaly": {"type": "new_obstacle"}, "facts": facts, "image": True}
        sim.advisor._camera_evidence(job)
        assert job["anomaly"]["camera_detector"] == "spire 0.99"
        res = SimpleNamespace(ok=True, classification="rock", confidence=0.95, action="reroute", model_action=None,
                              reason="looks like a rock", latency_ms=800.0, error=None)
        n0 = len(sim.fleet.events)
        sim.advisor._on_interpretation(job, res, sim.t)
        msgs = [e["msg"] for e in sim.fleet.events[n0:]]
        line = next(m for m in msgs if "Gemma" in m)
        assert "Camera detector: spire 0.99, so a fallen spire" in line, msgs
        assert EM_DASH not in line
    finally:
        p.close()
        sim.close()
