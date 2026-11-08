"""v7 alien flora: placement, geoms, identification by the rovers' own sensors, keep-out, contacts."""
import math
import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from redshift import v7_flora  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.planner import PENDING  # noqa: E402
from redshift.sim import Simulation  # noqa: E402
from redshift.terrain import Terrain  # noqa: E402
from test_v4 import teleport  # noqa: E402

EM_DASH = chr(0x2014)


def make(**cfg) -> Simulation:
    return Simulation(Config(seed=42, plan_clock="model", **cfg))


@pytest.fixture(scope="module")
def on():
    return make(flora=True)


def _gid_name(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""


def test_flag_off_world_unchanged(on):
    off = make()
    a, b = off.terrain, on.terrain
    assert off.flora is None and a.flora == []
    assert np.array_equal(a.heights, b.heights) and np.array_equal(a.truth_heights, b.truth_heights)
    assert [(t.x, t.y, t.kind) for t in a.targets] == [(t.x, t.y, t.kind) for t in b.targets]
    assert [(r.x, r.y) for r in a.rocks] == [(r.x, r.y) for r in b.rocks]
    assert [(s.x, s.y) for s in a.spires] == [(s.x, s.y) for s in b.spires]
    assert "flora" not in off.features_msg()
    assert not any(_gid_name(off.model, g).startswith("flora_") for g in range(off.model.ngeom))
    st = off.fleet.stats()
    assert "flora_zones" not in st and "flora_seen" not in st
    assert "flora" in on.features_msg() and len(on.features_msg()["flora"]) == len(b.flora)


def test_placement_deterministic_and_clear(on):
    cfg = Config(seed=42, flora=True)
    again = Terrain(cfg)
    pl = on.terrain.flora
    assert [p.to_msg() for p in pl] == [p.to_msg() for p in again.flora]
    trees = [p for p in pl if p.kind == "tree"]
    bushes = [p for p in pl if p.kind == "bush"]
    assert 15 <= len(trees) <= 30 and 20 <= len(bushes) <= 45
    t = on.terrain
    half = cfg.map_size / 2.0
    for p in pl:
        assert abs(p.x) < half - 3 and abs(p.y) < half - 3
        assert math.hypot(p.x, p.y) >= cfg.lander_flat_radius + 12.0
        assert not t.wet_near(p.x, p.y, p.radius + 2.0)
        i, j = t.ij(p.x, p.y)
        assert float(t.slope[j, i]) <= 12.0
        for tg in t.targets:
            assert math.hypot(p.x - tg.x, p.y - tg.y) >= v7_flora.NOMINAL_RADIUS["tree"] + cfg.observe_min_dist
        for sp in t.spires:
            assert math.hypot(p.x - sp.x, p.y - sp.y) > sp.radius + p.radius
        if p.kind == "tree":
            assert 2.5 <= p.height <= 4.5 and 1.2 <= p.radius <= 2.2
        else:
            assert 0.4 <= p.height <= 0.7 and 5 <= p.bulbs <= 12
    for a in range(len(pl)):
        for b in range(a + 1, len(pl)):
            assert math.hypot(pl[a].x - pl[b].x, pl[a].y - pl[b].y) > pl[a].radius + pl[b].radius
    other = Terrain(Config(seed=7, flora=True)).flora
    assert [p.to_msg() for p in other] != [p.to_msg() for p in pl]


def test_geom_groups_and_budget(on):
    m = on.model
    names = [(g, _gid_name(m, g)) for g in range(m.ngeom)]
    flora = [(g, n) for g, n in names if n.startswith("flora_")]
    assert 0 < len(flora) <= 160
    for g, n in flora:
        assert n.startswith("flora_tree_") or n.startswith("flora_bush_")
        if n.endswith("_stalk"):
            assert m.geom_group[g] == 2
            assert np.ptp(m.geom_rgba[g][:3]) < 0.1          # neutral grey brown, never a marker colour
            if n.startswith("flora_tree_"):
                assert (m.geom_contype[g], m.geom_conaffinity[g]) == (1, 0)
            else:
                assert (m.geom_contype[g], m.geom_conaffinity[g]) == (0, 0)
        else:
            assert m.geom_group[g] == 4
            assert (m.geom_contype[g], m.geom_conaffinity[g]) == (0, 0)
    stalks = [g for g, n in flora if n.endswith("_stalk")]
    assert set(stalks) == set(on.flora.gid_plant)
    life = set(on.world.life_gids.tolist()) if on.world.life_gids is not None else set()
    assert not life & set(stalks)


def _view_tree(sim, dist=7.0, which=0):
    trees = [p for p in sim.terrain.flora if p.kind == "tree"]
    p = trees[which]
    r = sim.rovers[0]
    ang = math.atan2(p.y, p.x)
    teleport(sim, r, p.x - dist * math.cos(ang), p.y - dist * math.sin(ang), ang)
    return p, r, ang


def test_spot_targets_ignores_proxies():
    from redshift import camera
    if not camera.renderer_status().get("ok", True):
        pytest.skip("no OpenGL here")
    sim = make(flora=True)
    for which in range(3):
        p, r, _ = _view_tree(sim, 6.0, which)
        for det in camera.spot_targets(sim, r):
            assert math.hypot(det["x"] - p.x, det["y"] - p.y) > 2.0


def test_rover_identifies_tree_keeps_out_and_paths_avoid():
    sim = make(flora=True)
    p, r, ang = _view_tree(sim, 7.0)
    for k in range(3):
        r.scan(sim.t + 2.5 * k + 0.1)
    if sim.camera_ok is False:
        pytest.skip("no OpenGL here")
    st = sim.fleet.stats()
    assert st["flora_seen"] == 1
    zx, zy, zr = st["flora_zones"][0]
    assert math.hypot(zx - p.x, zy - p.y) < 0.6
    assert zr == pytest.approx(v7_flora.NOMINAL_RADIUS["tree"] + sim.cfg.observe_min_dist)
    ev = [e for e in sim.fleet.events if "plant life" in e["msg"]]
    assert len(ev) == 1 and ev[0]["msg"].startswith("r1 sees possible plant life: fungal tree (camera ")
    assert "Keeping 4 m away." in ev[0]["msg"] and "at" in ev[0] and EM_DASH not in ev[0]["msg"]
    # the same plant is never logged twice
    r.scan(sim.t + 9.0)
    assert len([e for e in sim.fleet.events if "plant life" in e["msg"]]) == 1
    # static flora is a map obstacle, never a motion track
    assert not sim.fleet.tracker.to_msgs()
    # a path to the far side goes around the keep-out zone
    goal = (p.x + 8.0 * math.cos(ang), p.y + 8.0 * math.sin(ang))
    path, t = PENDING, sim.t + 10.0
    for k in range(200):
        path = sim.fleet.plan(t + 0.05 * k, r, goal, max_exp=40000)
        if path is not PENDING:
            break
    assert path, "no path around the plant"
    pts = np.array(path)
    assert float(np.hypot(pts[:, 0] - zx, pts[:, 1] - zy).min()) >= zr - 0.75


def test_headless_run_no_contacts():
    sim = make(flora=True)
    while sim.t < 150.0:
        sim.step()
    assert sim.flora.contacts == 0
    assert np.isfinite(sim.flora.closest) or sim.flora.closest == float("inf")
    for tr in sim.fleet.tracker.to_msgs():       # no track sits on a plant
        for pl in sim.terrain.flora:
            assert math.hypot(tr["x"] - pl.x, tr["y"] - pl.y) > 1.0
