"""Onboard camera tests (redshift/camera.py): headless renders of both views, the HTTP endpoint,
and camera based target spotting.

Rendering needs an OpenGL backend: OSMesa (what Docker uses), EGL, or a display. Rendering tests
skip with the reason if none works; the endpoint error paths (404, 503) run anyway.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m pytest -q tests/test_camera.py
"""
from __future__ import annotations

import io
import math
import sys
import threading
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

from redshift import camera, server  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.sim import Simulation  # noqa: E402
from redshift.world import rover_spawn_height  # noqa: E402

EM_DASH = chr(0x2014)
SCENE_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)   # terrain, rocks, targets (what the camera sees, minus rovers)


# ====================================================================== helpers and fixtures
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


def camera_app() -> FastAPI:
    app = FastAPI()
    app.include_router(camera.router)
    return app


def decode(jpg: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(jpg))
    im.load()
    return im


def place(sim, rover, x: float, y: float, yaw: float) -> None:
    """Teleport a rover (upright, on the ground) and refresh the derived state. No physics step."""
    q, a = sim.data.qpos, rover.qadr
    q[a:a + 3] = (x, y, sim.terrain.height_at(x, y) + rover_spawn_height(sim.cfg) + 0.02)
    q[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    sim.data.qvel[rover.vadr:rover.vadr + rover.nv] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def ray(sim, pnt, vec):
    """Distance and geom id of the first scene geom (no rovers) hit along vec from pnt."""
    gid = np.array([-1], np.int32)
    d = mujoco.mj_ray(sim.model, sim.data, np.asarray(pnt, np.float64), np.asarray(vec, np.float64),
                      SCENE_GROUPS, 1, -1, gid)
    return d, int(gid[0])


@pytest.fixture(scope="module")
def gl():
    st = camera.renderer_status()
    if not st["ok"]:
        pytest.skip(f"no OpenGL backend for rendering: {st['error']}")
    return st


SPREAD = {"r1": (9.0, 4.0, 0.5), "r2": (-12.0, 8.0, 2.6), "r3": (4.0, -14.0, -1.2)}   # x, y, yaw


@pytest.fixture(scope="module")
def sim():
    """Rovers out on the terrain (placed, not driven, so these tests do not depend on the autonomy)."""
    s = Simulation(Config(seed=42))
    for r in s.rovers:
        if r.id in SPREAD:
            place(s, r, *SPREAD[r.id])
    return s


# ====================================================================== rendering
@pytest.mark.parametrize("view", camera.VIEWS)
def test_views_render_and_encode(gl, sim, view):
    for rid in [r.id for r in sim.rovers]:
        for w, h in ((320, 240), (160, 120), (640, 480)):
            rgb = camera.render_rgb(sim, rid, view, w, h)
            assert rgb.shape == (h, w, 3) and rgb.dtype == np.uint8
            assert rgb.std() > 8.0, f"{rid} {view} {w}x{h} looks blank"
            colours = len(np.unique(rgb.reshape(-1, 3), axis=0))
            assert colours > 50, f"{rid} {view} {w}x{h}: only {colours} colours"
            im = decode(camera.encode_jpeg(rgb))
            assert im.format == "JPEG" and im.mode == "RGB" and im.size == (w, h)


def test_upright_with_sky_above_and_ground_below(gl):
    """Not upside down, horizon sensible: sky (far away) at the top, ground (close) at the bottom."""
    s = Simulation(Config(seed=42))          # rovers parked on the flat lander pad, facing outwards
    for view in camera.VIEWS:
        fr = camera.render_frame(s, "r1", view, 160, 120, depth=True)
        top, bottom = fr.depth[:4], fr.depth[-4:]
        assert np.median(top) > 50.0, f"{view}: no sky at the top (depth {np.median(top):.1f} m)"
        assert np.median(bottom) < 6.0, f"{view}: no ground at the bottom (depth {np.median(bottom):.1f} m)"
        lum = fr.rgb.astype(float).mean(axis=2)
        assert lum[-30:].mean() > lum[:10].mean(), f"{view}: ground should be brighter than the dark sky"


def test_chase_view_shows_the_rover(gl, sim):
    with_rovers = camera.render_rgb(sim, "r2", "chase", 320, 240).astype(int)
    without = camera.render_rgb(sim, "r2", "chase", 320, 240, hide_groups=(3,)).astype(int)
    diff = np.abs(with_rovers - without).sum(axis=2) > 40
    frac = diff.mean()
    assert frac > 0.03, f"rover covers only {100 * frac:.1f}% of the chase view"
    ys, xs = np.nonzero(diff)
    assert 0.3 * 320 < np.median(xs) < 0.7 * 320, "rover should be near the middle"
    assert np.median(ys) > 0.4 * 240, "rover should be in the lower part, below the horizon"


def test_fpv_looks_forward_along_the_heading(gl, sim):
    for r in sim.rovers:
        fr = camera.render_frame(sim, r.id, "fpv", 64, 48)
        f = fr.spec.forward
        assert abs(math.remainder(math.atan2(f[1], f[0]) - fr.rover_yaw, 2 * math.pi)) < 1e-6
        pitch = math.degrees(math.asin(f[2]))
        assert -16.0 < pitch < -8.0
        # eye on top of the mast: about lidar_height above the chassis centre
        assert 0.5 < fr.spec.eye[2] - fr.rover_pos[2] < 1.1


def test_depth_matches_ray_casts(gl, sim):
    """Field of view, aspect ratio, clip planes and depth decoding agree with MuJoCo's own ray casts.

    Each pixel's depth must lie within the depths MuJoCo's ray casts give across that pixel's own
    footprint (its four corners). A multisampled depth buffer is read at one sample, not at the pixel
    centre (Mesa: about 0.35 px off), which matters at grazing angles, so the centre ray is not exact."""
    w, h = 64, 48
    fr = camera.render_frame(sim, "r1", "fpv", w, h, depth=True, hide_groups=(3, 4))
    f, right, up = fr.spec.basis()
    tan_v = math.tan(math.radians(fr.spec.fovy) / 2)
    tan_h = tan_v * w / h

    def view_depth(px, py):          # ray cast through image point (px, py), as depth along the view axis
        d = f + (px / w * 2 - 1) * tan_h * right + (1 - py / h * 2) * tan_v * up
        dist, _gid = ray(sim, fr.spec.eye, d / np.linalg.norm(d))
        return dist / np.linalg.norm(d) if dist >= 0 else np.inf

    good, total = 0, 0
    for row in range(2, h, 5):
        for col in range(2, w, 5):
            zs = [view_depth(col + a, row + b) for a in (0.0, 1.0) for b in (0.0, 1.0)]
            if not all(np.isfinite(zs)) or max(zs) > 40:
                continue
            total += 1
            z, tol = float(fr.depth[row, col]), 0.01 + 0.005 * max(zs)
            good += min(zs) - tol <= z <= max(zs) + tol
    assert total > 40
    assert good / total > 0.95, f"only {good}/{total} pixels have a depth inside their footprint"


def test_new_model_after_reset_is_picked_up(gl):
    a, b = Simulation(Config(seed=42)), Simulation(Config(seed=7))
    fa = camera.render_rgb(a, "r1", "chase", 160, 120)
    fb = camera.render_rgb(b, "r1", "chase", 160, 120)
    fa2 = camera.render_rgb(a, "r1", "chase", 160, 120)
    assert np.abs(fa.astype(int) - fb.astype(int)).mean() > 2.0, "different planets should look different"
    assert np.abs(fa.astype(int) - fa2.astype(int)).mean() < 1.0, "same state should render the same"


# ====================================================================== HTTP endpoint
def test_endpoint_with_a_running_sim(gl, monkeypatch):
    runner = server.SimRunner(Config(seed=42))
    runner.start()
    monkeypatch.setattr(server.state, "runner", runner)
    try:
        with TestClient(camera_app()) as client:
            t0 = runner.sim.t
            r = client.get("/api/camera/r1.jpg")
            assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
            assert decode(r.content).size == (camera.DEFAULT_W, camera.DEFAULT_H)
            r = client.get("/api/camera/r2.jpg", params={"view": "chase", "w": 200, "h": 150})
            assert r.status_code == 200 and decode(r.content).size == (200, 150)
            r = client.get("/api/camera/r3.jpg", params={"w": 5000, "h": 5000})
            assert r.status_code == 200 and decode(r.content).size == (camera.MAX_W, camera.MAX_H)
            r = client.get("/api/camera/r9.jpg")
            assert r.status_code == 404 and r.json()["ok"] is False
            r = client.get("/api/camera/r1.jpg", params={"view": "sideways"})
            assert r.status_code == 400 and r.json()["ok"] is False
            for _ in range(5):
                assert client.get("/api/camera/r1.jpg", params={"view": "chase"}).status_code == 200
            # the sim thread kept getting the lock between frames (steps ran, whatever they did)
            assert runner.sim.t > t0 or runner.errors > 0
    finally:
        runner.stop()


def test_endpoint_503_when_sim_not_running(monkeypatch):
    monkeypatch.setattr(server.state, "runner", None)
    with TestClient(camera_app()) as client:
        r = client.get("/api/camera/r1.jpg")
        assert r.status_code == 503 and r.json()["ok"] is False


def test_endpoint_503_when_rendering_is_unavailable(monkeypatch):
    """No GL backend: clear 503 JSON, unknown rovers still 404, nothing else breaks."""
    def no_gl(w, h):
        raise camera.RenderUnavailable("no OpenGL here (test)")

    rt = camera.RenderThread()
    monkeypatch.setattr(camera, "_make_gl_context", no_gl)
    monkeypatch.setattr(camera, "_RT", rt)
    monkeypatch.setattr(server.state, "runner", FakeRunner(Simulation(Config(seed=42))))
    try:
        with TestClient(camera_app()) as client:
            for view in camera.VIEWS:
                r = client.get("/api/camera/r1.jpg", params={"view": view})
                assert r.status_code == 503
                body = r.json()
                assert body["ok"] is False and "no OpenGL here" in body["error"]
            assert client.get("/api/camera/r9.jpg").status_code == 404
    finally:
        rt.shutdown()


# ====================================================================== spot_targets (stretch)
def test_spot_targets_finds_markers_where_they_are(gl):
    s = Simulation(Config(seed=42))
    rover = s.rovers[0]
    kinds = {t.kind for t in s.terrain.targets}
    checked, all_dets = 0, []
    for t in s.terrain.targets:
        centre = np.array([t.x, t.y, t.z + 0.6])
        gid_t = [g for g, idx in s.world.target_geom_ids.items() if s.terrain.targets[idx] is t][0]
        for dist in (4.0, 8.0):
            for ang, off in ((0.3, 12.0), (2.4, -15.0)):
                x, y = t.x - dist * math.cos(ang), t.y - dist * math.sin(ang)
                yaw = ang + math.radians(off)
                place(s, rover, x, y, yaw)
                eye = camera.rig("fpv", s, rover).eye
                _d, hit = ray(s, eye, (centre - eye) / np.linalg.norm(centre - eye))
                dets = camera.spot_targets(s, rover)
                all_dets += dets
                if hit != gid_t:
                    continue         # hidden behind terrain, a rock or another marker
                checked += 1
                same = [d for d in dets if d["kind"] == t.kind]
                assert same, f"{t.id} ({t.kind}) at {dist} m not detected: {dets}"
                best = min(same, key=lambda d: math.hypot(d["x"] - t.x, d["y"] - t.y))
                err = math.hypot(best["x"] - t.x, best["y"] - t.y)
                assert err < 0.5, f"{t.id}: position off by {err:.2f} m"
                true_bearing = math.degrees(math.remainder(math.atan2(t.y - y, t.x - x) - yaw, 2 * math.pi))
                assert abs(best["bearing_deg"] - true_bearing) < 3.0
                assert abs(best["distance"] - math.hypot(t.x - x, t.y - y)) < 0.5
    assert checked >= 12, f"only {checked} unobstructed views"
    for d in all_dets:            # no false positives anywhere
        assert d["kind"] in kinds
        near = min(math.hypot(d["x"] - t.x, d["y"] - t.y) for t in s.terrain.targets if t.kind == d["kind"])
        assert near < 1.0, f"false detection {d}"


def test_spot_targets_empty_view(gl):
    s = Simulation(Config(seed=42))
    rover = s.rovers[0]
    tx = np.array([t.x for t in s.terrain.targets])
    ty = np.array([t.y for t in s.terrain.targets])
    # face the direction with no marker within 25 m in a 90 degree cone
    best = None
    for yaw in np.linspace(-math.pi, math.pi, 36, endpoint=False):
        rel = np.remainder(np.arctan2(ty - rover.y, tx - rover.x) - yaw + math.pi, 2 * math.pi) - math.pi
        close = (np.abs(rel) < math.radians(45)) & (np.hypot(tx - rover.x, ty - rover.y) < 25)
        if not close.any():
            best = yaw
            break
    if best is None:
        pytest.skip("no empty direction on this layout")
    place(s, rover, rover.x, rover.y, best)
    assert camera.spot_targets(s, rover) == []


def test_no_em_dash():
    for p in (ROOT / "redshift" / "camera.py", Path(__file__)):
        assert EM_DASH not in p.read_text(encoding="utf-8"), p.name
