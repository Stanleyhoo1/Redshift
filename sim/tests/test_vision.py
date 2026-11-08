"""Protocol v8: onboard object detection (redshift/vision.py) and the faster camera (redshift/camera.py).

Detector tests render real frames (MuJoCo segmentation is used here only to score them, as offline
evaluation does; the detector never sees it). Decision tests feed the rover hand-made detections, so they
run without OpenGL. Rendering tests skip with the reason if no GL backend works.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m pytest -q tests/test_vision.py
"""
from __future__ import annotations

import io
import json
import math
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from redshift import camera, server, vision  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.sim import Simulation  # noqa: E402
from redshift.world import rover_spawn_height  # noqa: E402

EM_DASH = chr(0x2014)


class FakeRunner:
    """The parts of server.SimRunner the camera uses, around a sim nobody steps."""

    def __init__(self, sim):
        self.sim = sim
        self.cfg = sim.cfg
        self._lock = threading.Lock()

    @contextmanager
    def locked(self):
        with self._lock:
            yield self.sim


def place(sim, rover, x: float, y: float, yaw: float) -> None:
    q, a = sim.data.qpos, rover.qadr
    q[a:a + 3] = (x, y, sim.terrain.height_at(x, y) + rover_spawn_height(sim.cfg) + 0.02)
    q[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    sim.data.qvel[rover.vadr:rover.vadr + rover.nv] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def face(sim, rover, tx: float, ty: float, dist: float, ang: float) -> None:
    """Put the rover dist metres from (tx, ty), looking at it."""
    place(sim, rover, tx - dist * math.cos(ang), ty - dist * math.sin(ang), ang)


def park_others(sim, keep=("r1",)) -> None:
    for r in sim.rovers:
        if r.id not in keep:
            place(sim, r, 50.0 - 3 * r.idx, -55.0, 0.0)


@pytest.fixture(scope="module")
def gl():
    st = camera.renderer_status()
    if not st["ok"]:
        pytest.skip(f"no OpenGL backend for rendering: {st['error']}")
    return st


@pytest.fixture(scope="module")
def sim():
    s = Simulation(Config(seed=42, plan_clock="model"))
    park_others(s)
    return s


def detect_view(sim, w=320, h=240, shadows=None):
    """(detections, ground truth instances that count) of r1's fpv camera."""
    fr, seg = camera.segment_ids(sim, "r1", "fpv", w, h, shadows=shadows)
    res = vision.detect_frame(fr)
    return res, vision.gt_instances(seg, sim.model, fr.depth), fr


def best_of(res, label):
    c = [o for o in res.objects if o["label"] == label]
    return max(c, key=lambda o: o["conf"]) if c else None


def box_hits(o, g, w, h) -> bool:
    b = (o["box"][0] * w, o["box"][1] * h, o["box"][2] * w, o["box"][3] * h)
    cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
    gb = g["box"]
    return gb[0] - 2 <= cx <= gb[2] + 3 and gb[1] - 2 <= cy <= gb[3] + 3


# ====================================================================== detector on rendered frames
@pytest.mark.parametrize("kind,dist", [("spire", 7.0), ("creature", 5.0), ("lander", 7.0), ("mineral", 4.0),
                                       ("ice", 3.5), ("vent", 4.5), ("trace", 3.5), ("rock", 4.0), ("water", 4.0)])
def test_detector_finds_each_kind(gl, sim, kind, dist):
    r1 = sim.rovers[0]
    t = sim.terrain
    if kind in ("mineral", "ice", "vent", "trace"):
        spots = [(o.x, o.y) for o in t.targets if o.kind == kind]
    elif kind == "spire":
        spots = [(o.x, o.y) for o in t.spires]
    elif kind == "creature":
        spots = [(sim.data.mocap_pos[m][0], sim.data.mocap_pos[m][1]) for m in sim.world.life_mocap]
    elif kind == "rock":
        spots = [(o.x, o.y) for o in t.rocks if o.sx > 0.3]
    elif kind == "water":
        spots = [(w.cx, w.cy, w.radius) for w in t.water if w.kind == "lake"]
    else:
        spots = [(0.0, 0.0)]
    tried = 0
    for sp in spots[:6]:
        for ang in (0.3, 1.9, 3.5, 5.0):
            d = dist + (sp[2] if len(sp) > 2 else 0.0)
            x, y = sp[0] - d * math.cos(ang), sp[1] - d * math.sin(ang)
            if t.water_at(x, y) is not None or t.max_slope_near(x, y, 0.6) > 20.0:
                continue
            face(sim, r1, sp[0], sp[1], d, ang)
            res, insts, fr = detect_view(sim)
            gts = [g for g in insts if g["label"] == kind and g["counts"] and g["npx"] > 60]
            if not gts:
                continue                  # hidden behind something from here
            tried += 1
            g = max(gts, key=lambda g_: g_["npx"])
            hits = [o for o in res.objects if o["label"] == kind and o["conf"] >= 0.5 and box_hits(o, g, 320, 240)]
            if hits:
                o = hits[0]
                assert 0.0 <= o["box"][0] < o["box"][2] <= 1.0 and 0.0 <= o["box"][1] < o["box"][3] <= 1.0
                assert abs(o["dist"] - math.hypot(sp[0] - fr.rover_pos[0], sp[1] - fr.rover_pos[1])) < 0.6 * d + 1.0
                return
    assert tried, f"no clear view of a {kind}"
    pytest.fail(f"{kind} not detected in {tried} clear views")


def test_detector_finds_rovers_and_boulders(gl):
    s = Simulation(Config(seed=7, plan_clock="model"))
    r1, r2, r3 = s.rovers[:3]
    # open ground: the flattest spot on a ring around the lander
    best = min(((14.0 * math.cos(a), 14.0 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)),
               key=lambda p: s.terrain.max_slope_near(p[0], p[1], 4.0) + 50.0 * bool(s.terrain.wet_near(p[0], p[1], 6.0)))
    x, y = best
    yaw = math.atan2(-y, -x) + 1.2
    place(s, r1, x, y, yaw)
    place(s, r2, x + 5.0 * math.cos(yaw + 0.25), y + 5.0 * math.sin(yaw + 0.25), yaw + 1.0)
    mid, _gid, (_sx, _sy, sz) = s.world.boulders[1]
    bx, by = x + 6.0 * math.cos(yaw - 0.3), y + 6.0 * math.sin(yaw - 0.3)
    s.data.mocap_pos[mid] = (bx, by, s.terrain.height_at(bx, by) + 0.45 * sz)
    place(s, r3, 50.0, -55.0, 0.0)
    res, insts, _fr = detect_view(s)
    labels = {o["label"] for o in res.objects if o["conf"] >= 0.5}
    assert "rover" in labels, res.objects
    assert labels & {"boulder", "rock"}, res.objects
    rover = best_of(res, "rover")
    assert 3.0 < rover["dist"] < 7.0 and rover["bearing"] > 0      # to the left


def test_dust_lowers_confidence(gl, sim):
    r1 = sim.rovers[0]
    sp = sim.terrain.spires[0]
    for ang in (0.3, 1.9, 3.5, 5.0):
        face(sim, r1, sp.x, sp.y, 14.0, ang)
        clear, _i, _f = detect_view(sim)
        if best_of(clear, "spire") is not None:
            break
    else:
        pytest.skip("no clear view of a spire")
    sim.hazards.trigger("dust_storm")
    try:
        sim.data.time += camera.DUST_RAMP_S + 1.0
        dusty, _i, fr = detect_view(sim)
        assert fr.dust > 0.9
    finally:
        sim.hazards.storm.active = False
        sim.hazards.storm = None
    c0 = best_of(clear, "spire")["conf"]
    d1 = best_of(dusty, "spire")
    assert d1 is None or d1["conf"] < c0, (c0, d1)
    assert len([o for o in dusty.objects if o["conf"] >= 0.5]) <= len([o for o in clear.objects if o["conf"] >= 0.5])


def test_detector_time_and_shapes(gl, sim):
    r1 = sim.rovers[0]
    face(sim, r1, 0.0, 0.0, 9.0, 2.2)
    fr = camera.render_frame(sim, "r1", "fpv", 640, 480, depth=True)
    vision.detect_frame(fr)                          # warm up the colour table
    res = vision.detect_frame(fr)
    assert res.ms < 400.0
    m = res.msg()
    assert set(m) == {"frame", "t", "view", "ms", "objects"} and m["view"] == "fpv" and m["frame"] == fr.frame
    for o in m["objects"]:
        assert set(o) == {"label", "conf", "box", "dist", "bearing"}
        assert o["label"] in vision.LABELS and 0.0 < o["conf"] <= 1.0
        assert len(o["box"]) == 4 and all(0.0 <= v <= 1.0 for v in o["box"])
    assert json.loads(res.header()) == m["objects"]


# ====================================================================== camera: culling, cache, headers
def test_culling_changes_no_pixel(gl, sim, monkeypatch):
    r1 = sim.rovers[0]
    face(sim, r1, sim.terrain.spires[1].x, sim.terrain.spires[1].y, 8.0, 0.7)
    for view, w, h in (("fpv", 320, 240), ("chase", 640, 480)):
        culled = camera.render_rgb(sim, "r1", view, w, h)
        monkeypatch.setattr(camera, "_cull_mask", lambda c, rb, *a, **k: np.ones(len(c), bool))
        full = camera.render_rgb(sim, "r1", view, w, h)
        monkeypatch.undo()
        assert np.array_equal(culled, full), f"{view}: culling changed {np.mean(np.any(culled != full, axis=2)):.4%}"


def camera_app() -> FastAPI:
    app = FastAPI()
    app.include_router(camera.router)
    return app


def test_endpoint_headers_and_frame_detections(gl, monkeypatch):
    s = Simulation(Config(seed=42, plan_clock="model"))
    park_others(s)
    sp = s.terrain.spires[0]
    face(s, s.rovers[0], sp.x, sp.y, 8.0, 0.4)
    monkeypatch.setattr(server.state, "runner", FakeRunner(s))
    with TestClient(camera_app()) as client:
        r = client.get("/api/camera/r1.jpg", params={"w": 320, "h": 240})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        num, age = int(r.headers["X-Frame"]), int(r.headers["X-Frame-Age-Ms"])
        dets = json.loads(r.headers["X-Detections"])
        assert num > 0 and age >= 0 and "X-Detections" in r.headers["Access-Control-Expose-Headers"]
        assert Image.open(io.BytesIO(r.content)).size == (320, 240)
        for o in dets:
            assert set(o) == {"label", "conf", "box", "dist", "bearing"}
        assert any(o["label"] == "spire" for o in dets), dets
        # the rover's own `vision` came from exactly that frame (or a newer detector frame)
        vis = s.rovers[0].to_msg()["vision"]
        assert vis["frame"] >= num
        if vis["frame"] == num:
            assert vis["objects"] == dets
        # chase is not an onboard camera: no detections
        r = client.get("/api/camera/r1.jpg", params={"view": "chase"})
        assert r.status_code == 200 and json.loads(r.headers["X-Detections"]) == []


def test_stream_answers_at_once_and_stops_when_idle(gl, monkeypatch):
    s = Simulation(Config(seed=42, plan_clock="model"))
    rt = camera.RenderThread()
    monkeypatch.setattr(camera, "_RT", rt)
    monkeypatch.setattr(camera, "STREAM_IDLE_S", 1.0)
    monkeypatch.setattr(server.state, "runner", FakeRunner(s))
    try:
        with TestClient(camera_app()) as client:
            first = client.get("/api/camera/r2.jpg", params={"w": 200, "h": 150})
            assert first.status_code == 200
            n0 = int(first.headers["X-Frame"])
            t0 = time.perf_counter()
            again = client.get("/api/camera/r2.jpg", params={"w": 200, "h": 150})
            fast = time.perf_counter() - t0
            assert again.status_code == 200 and int(again.headers["X-Frame"]) >= n0
            assert fast < 0.5, f"the newest frame should come back at once ({fast:.2f} s)"
            nums = {n0}
            for _ in range(12):           # polling keeps the view rendering in the background
                time.sleep(0.15)
                nums.add(int(client.get("/api/camera/r2.jpg", params={"w": 200, "h": 150}).headers["X-Frame"]))
            assert len(nums) >= 3, f"background rendering stalled: frames {sorted(nums)}"
            assert len(rt.streams) == 1
            time.sleep(2.2)               # nobody asks: the view stops rendering
            before = rt.stream_frames
            time.sleep(0.6)
            assert rt.stream_frames == before and not rt.streams
    finally:
        rt.shutdown()


# ====================================================================== decisions from detections
def make_result(rover, objects, t=None, frame=None) -> vision.VisionResult:
    """A detection result at the rover's current pose: objects are (label, conf, dist, bearing_deg)."""
    out = []
    for (label, conf, dist, bearing) in objects:
        a = rover.yaw + math.radians(bearing)
        out.append({"label": label, "conf": conf, "box": [0.4, 0.4, 0.5, 0.5], "dist": dist, "bearing": bearing,
                    "x": rover.x + dist * math.cos(a), "y": rover.y + dist * math.sin(a), "height": 0.5, "width": 0.8})
    t = rover.sim.t if t is None else t
    frame = (rover.vision.latest.frame + 1) if (frame is None and rover.vision.latest is not None) else (frame or 1)
    return vision.VisionResult(frame=frame, t=t, view="fpv", ms=5.0, objects=out)


@pytest.fixture()
def fresh():
    s = Simulation(Config(seed=42, plan_clock="model"))
    s.run_for(4.0)
    return s


def events_with(sim, text):
    return [e["msg"] for e in sim.fleet.events if text in e["msg"]]


def test_water_confirmed_by_camera(fresh):
    s, r = fresh, fresh.rovers[0]
    r.read_pose()
    at = (r.x + 5.0 * math.cos(r.yaw), r.y + 5.0 * math.sin(r.yaw))
    r.vision.publish(make_result(r, [("water", 0.86, 5.2, 2.0)]))
    zones = len(s.fleet.water_zones)
    pts = np.array([[at[0] + dx, at[1] + dy, 0.0] for dx in (-0.8, 0.0, 0.8) for dy in (-0.8, 0.0, 0.8)])
    r._on_hole(s.t, at, 9, pts, 0.3)
    assert len(s.fleet.water_zones) == zones + 1
    msg = events_with(s, "Camera: water")
    assert msg and "(0.86)" in msg[-1] and "Confirmed liquid" in msg[-1]
    assert r.anomaly["kind"] == "no_returns" and r.anomaly["camera_water"] == 0.86


def test_creature_confirms_a_lidar_track(fresh):
    s, r = fresh, fresh.rovers[0]
    r.read_pose()
    tk = s.fleet.tracker
    x, y = r.x + 7.0 * math.cos(r.yaw), r.y + 7.0 * math.sin(r.yaw)
    for k in range(6):                       # a cluster that moves: a confirmed moving-object track
        tk.add_scan("r2", np.array([[x + 0.1 * k, y, 0.4], [x + 0.1 * k + 0.3, y, 0.5], [x + 0.1 * k, y + 0.3, 0.4]]),
                    s.t + 0.2 * k)
        tk.update(s.t + 0.2 * k)
    tracks = [tr for tr in tk.tracks if tr.confirmed]
    assert tracks, "the test track did not confirm"
    tr = tracks[0]
    seen0 = s.fleet.lifeforms_seen
    px, py = tr.predict(s.t)
    d, b = math.hypot(px - r.x, py - r.y), math.degrees(math.atan2(py - r.y, px - r.x) - r.yaw)
    r.vision.publish(make_result(r, [("creature", 0.83, d, b)], t=s.t + 1.0))
    r.vision.tick(s.t + 1.0)
    assert tr.lifeform and tr.label == "creature" and s.fleet.lifeforms_seen > seen0
    assert events_with(s, f"where LiDAR track {tr.id} moves")


def test_new_obstacle_gets_its_class(fresh):
    s, r = fresh, fresh.rovers[0]
    r.read_pose()
    r.state = "exploring"
    at = (r.x + 5.0 * math.cos(r.yaw), r.y + 5.0 * math.sin(r.yaw))
    r.on_anomaly(s.t, "new_obstacle", {"at": at, "dist": 5.0, "cells": 6})
    blocked0 = int(s.fleet.map.blocked.sum()) if hasattr(s.fleet.map, "blocked") else None
    r.vision.publish(make_result(r, [("boulder", 0.78, 5.1, 1.0), ("rock", 0.6, 9.0, -30.0)], t=s.t + 0.5))
    r.vision.tick(s.t + 0.5)
    msg = events_with(s, "Camera: boulder")
    assert msg and "where the map changed" in msg[-1] and "extra room" in msg[-1]
    assert r.anomaly["camera"] == {"label": "boulder", "conf": 0.78}
    if blocked0 is not None:
        assert int(s.fleet.map.blocked.sum()) > blocked0
    r.vision.tick(s.t + 1.0)                  # once per obstacle
    assert len(events_with(s, "Camera: boulder")) == 1


def test_camera_check_of_a_distant_target(fresh):
    s, r = fresh, fresh.rovers[0]
    r.read_pose()
    r.state, r.held, r.op_goal, r.focus, r.obs = "exploring", False, None, None, None
    r.battery = 0.95
    r.vision.next_check = 0.0
    r.vision.publish(make_result(r, [("vent", 0.71, 12.0, 3.0)]))
    r.vision.tick(s.t)
    cg = r.vision.cam_goal
    assert cg is not None and cg["label"] == "vent"
    assert events_with(s, "Camera: possible vent 12 m ahead (0.71). Going to check it.")
    assert math.hypot(r.goal[0] - r.x, r.goal[1] - r.y) == pytest.approx(12.0 - vision.CHECK_SHORT_M, abs=0.3)
    # it drives there (the planner runs inside the normal decision cycle)
    for _ in range(10):
        s.step()
    assert r.vision.cam_goal is None or r.path, "no path to the camera check"
    # a second sighting during the cooldown starts nothing new
    r.vision.cam_goal = None
    r.vision.publish(make_result(r, [("mineral", 0.9, 14.0, -10.0)]))
    r.vision.tick(s.t)
    assert r.vision.cam_goal is None


def test_camera_check_ends_when_the_spectrometer_confirms(fresh):
    s, r = fresh, fresh.rovers[0]
    r.read_pose()
    r.state, r.battery = "exploring", 0.95
    ts = s.fleet.targets[0]
    d = math.hypot(ts.spec.x - r.x, ts.spec.y - r.y)
    b = math.degrees(math.atan2(ts.spec.y - r.y, ts.spec.x - r.x) - r.yaw)
    if not (vision.CHECK_RANGE_M[0] <= d <= vision.CHECK_RANGE_M[1]):
        d = 12.0           # out of range for a real check: put the sighting at 12 m and move the spot instead
    r.vision.publish(make_result(r, [(ts.spec.kind, 0.8, d, b)]))
    ts.status = None
    r.vision.tick(s.t)
    cg = r.vision.cam_goal
    assert cg is not None
    ts.spec.x, ts.spec.y = cg["x"], cg["y"]     # the spectrometer finds the target right there
    ts.status, ts.found_by = "found", r.id
    assert r.vision.drive_check(s.t + 1.0) is False and r.vision.cam_goal is None
    assert events_with(s, "the spectrometer confirms")


def test_vision_tick_is_cheap(fresh):
    s, r = fresh, fresh.rovers[0]
    r.read_pose()
    objs = [(lab, 0.7, 6.0 + k, -40.0 + 6 * k) for k, lab in enumerate(vision.LABELS)]
    r.vision.next_check = 1e9
    t0 = time.perf_counter()
    for k in range(50):
        r.vision.publish(make_result(r, objs, frame=1000 + k))
        r.vision.tick(s.t)
    ms = (time.perf_counter() - t0) * 1000.0 / 50
    assert ms < 5.0, f"vision.tick took {ms:.2f} ms"


def test_tick_message_vision_field(fresh):
    s, r = fresh, fresh.rovers[0]
    assert "vision" not in r.to_msg()
    r.read_pose()
    r.vision.publish(make_result(r, [("rock", 0.66, 4.0, 10.0)], frame=77))
    msg = s.tick_message()["rovers"][0]
    assert msg["vision"]["frame"] == 77 and msg["vision"]["view"] == "fpv"
    assert msg["vision"]["objects"] == [{"label": "rock", "conf": 0.66, "box": [0.4, 0.4, 0.5, 0.5], "dist": 4.0,
                                         "bearing": 10.0}]
    json.dumps(msg)


def test_older_results_never_replace_newer(fresh):
    r = fresh.rovers[0]
    r.vision.publish(make_result(r, [], frame=10))
    r.vision.publish(make_result(r, [("rock", 0.9, 3.0, 0.0)], frame=9))
    assert r.vision.latest.frame == 10


def test_gemma_gets_the_detections(fresh):
    s, r = fresh, fresh.rovers[0]
    r.read_pose()
    at = (r.x + 4.0 * math.cos(r.yaw), r.y + 4.0 * math.sin(r.yaw))
    r.vision.publish(make_result(r, [("boulder", 0.78, 4.1, 0.0)]))
    job = {"rover": r.id, "kind": "new_obstacle", "at": at, "t": s.t, "anomaly": {"type": "new_obstacle"},
           "facts": {}}
    s.advisor._camera_evidence(job)
    assert job["anomaly"]["camera_sees"] == "boulder 0.78 at 4.1 m"
    assert "Camera: boulder 0.78 at 4.1 m." in s.advisor._question(r, "new_obstacle", job, None, False)


def test_stats_against_segmentation(gl, sim):
    face(sim, sim.rovers[0], sim.terrain.spires[0].x, sim.terrain.spires[0].y, 8.0, 0.4)
    fr, seg = camera.segment_ids(sim, "r1", "fpv", 256, 192, shadows=False)
    st = vision.VisionStats()
    st.add(vision.detect_frame(fr), seg, sim.model, fr.depth)
    out = st.summary()
    assert out["frames"] == 1 and set(out["classes"]) == set(vision.LABELS)
    assert out["classes"]["spire"]["tp"] >= 1
    json.dumps(out)


def test_no_em_dash():
    for p in (ROOT / "redshift" / "vision.py", ROOT / "redshift" / "camera.py", ROOT / "scripts" / "vision_eval.py",
              Path(__file__)):
        assert EM_DASH not in p.read_text(encoding="utf-8"), p.name
