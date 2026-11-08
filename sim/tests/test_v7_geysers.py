"""v7 geysers: placement and schedule per seed, plume scatter kept out of the map and the tracker,
keep-out zones learned from the rover's own returns, eruption state in `features`."""
import json
import math

import mujoco
import numpy as np

from redshift.config import Config
from redshift.planner import INF, cost_grid
from redshift.sim import Simulation
from redshift.terrain import Terrain
from redshift.v7_geysers import KEEP_OUT_M
from redshift.world import rover_spawn_height


def _place(sim, r, x, y, yaw):
    q, a = sim.data.qpos, r.qadr
    q[a:a + 3] = (x, y, sim.terrain.height_at(x, y) + rover_spawn_height(sim.cfg) + 0.05)
    q[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    sim.data.qvel[r.vadr:r.vadr + r.nv] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    r.read_pose()


def _facing(sim, v, d=8.0):
    ang = math.atan2(v.y, v.x)
    return v.x - d * math.cos(ang), v.y - d * math.sin(ang), ang


def _first_active(v, t0=0.0):
    t = t0
    while not v.active(t):
        t += 0.25
    return t + 0.5


def test_flags_off_nothing_changes():
    sim = Simulation(Config(seed=42))
    assert sim.geysers is None and sim.terrain.geysers == []
    assert set(sim.features_msg()) == {"water", "spires"}
    assert "geyser_zones" not in sim.fleet.stats()
    names = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(sim.model.ngeom)]
    assert not any(n.startswith("geyser") for n in names)
    init = json.dumps(sim.init_message())
    assert "geyser" not in init
    sim.run_for(2.0)
    assert "geyser" not in json.dumps(sim.tick_message())


def test_schedule_deterministic_and_in_range():
    a = Terrain(Config(seed=42, geysers=True)).geysers
    b = Terrain(Config(seed=42, geysers=True)).geysers
    c = Terrain(Config(seed=7, geysers=True)).geysers
    assert a == b and a != c
    assert 1 <= len(a) <= 6
    ter = Terrain(Config(seed=42))
    base = Terrain(Config(seed=42, geysers=True))
    assert np.array_equal(ter.heights, base.heights)            # vents leave the ground and targets alone
    assert [t.id for t in ter.targets] == [t.id for t in base.targets]
    for v in a:
        assert 60.0 <= v.period_s <= 120.0 and 8.0 <= v.dur_s <= 15.0 and 6.0 <= v.plume_h <= 10.0
        assert math.hypot(v.x, v.y) >= 20.0
        assert all(math.hypot(v.x - t.x, v.y - t.y) >= 9.0 for t in base.targets)
        on = sum(v.active(t) for t in np.arange(0.0, v.period_s, 0.1))
        assert abs(on * 0.1 - v.dur_s) < 0.3
        t0 = 3.0
        assert v.active(v.next_change(t0) + 0.05) != v.active(t0)


def test_vent_geoms_exist():
    sim = Simulation(Config(seed=42, geysers=True))
    for k, v in enumerate(sim.terrain.geysers):
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, f"geyser_{k}")
        assert gid >= 0 and int(sim.model.geom_group[gid]) == 1
        assert np.allclose(sim.model.geom_pos[gid][:2], (v.x, v.y), atol=0.01)


def test_plume_returns_only_in_raw_scan():
    sim = Simulation(Config(seed=42, geysers=True))
    v = sim.terrain.geysers[0]
    r = sim.rovers[0]
    x, y, yaw = _facing(sim, v)
    _place(sim, r, x, y, yaw)
    t = _first_active(v)
    h_before = sim.map.h.copy()
    r.scan(t)
    pp = r.plume_pts
    assert pp is not None and len(pp) >= 6
    # in the raw scan the rover sends
    raw = np.array(r.scan_msg()).reshape(-1, 3)
    assert len(raw) >= len(pp)
    # never in the map: the mapped heights near the vent stay at ground level
    m = sim.map
    for p in pp:
        i, j = m.ij(float(p[0]), float(p[1]))
        if m.known[j, i]:
            assert m.h[j, i] < p[2] - 0.3 or abs(m.h[j, i] - h_before[j, i]) < 1e-6 or p[2] < v.z + 0.6
    high = m.known & (m.h > v.z + 1.5)
    yy, xx = np.nonzero(high)
    if len(xx):
        d = np.hypot(m.ox + xx * m.res - v.x, m.oy + yy * m.res - v.y)
        assert d.min() > 1.6                                  # no steam column in the height map
    assert sim.fleet.tracker.to_msgs() == []                    # nor in the motion tracker
    # learned from the rover's own returns: a keep-out zone near the vent and an event with `at`
    zones = sim.fleet.geyser_zones
    assert len(zones) == 1 and math.hypot(zones[0][0] - v.x, zones[0][1] - v.y) < 2.0
    assert sim.fleet.stats()["geyser_zones"][0][2] > 0
    ev = sim.fleet.events[-1]
    assert ev["msg"].startswith("r1 sees a steam plume") and "at" in ev
    assert not any("new obstacle" in e["msg"].lower() for e in sim.fleet.events)


def test_no_plume_when_quiet():
    sim = Simulation(Config(seed=42, geysers=True))
    v = sim.terrain.geysers[0]
    r = sim.rovers[0]
    x, y, yaw = _facing(sim, v)
    _place(sim, r, x, y, yaw)
    t = 0.0
    while v.active(t):
        t += 0.25
    r.scan(t)
    assert r.plume_pts is None
    assert sim.fleet.geyser_zones == []


def test_rovers_keep_clear_of_learned_vent():
    sim = Simulation(Config(seed=42, geysers=True))
    v = sim.terrain.geysers[0]
    r = sim.rovers[0]
    x, y, yaw = _facing(sim, v, 9.0)
    _place(sim, r, x, y, yaw)
    r.scan(_first_active(v))
    zx, zy, zr = sim.fleet.geyser_zones[0][:3]
    assert zr >= 5.0
    m = sim.map
    i, j = m.ij(v.x, v.y)
    assert m.avoid[j, i]
    # every cell inside the zone is impassable for every rover's planner
    m.known[:] = True
    grid = cost_grid(m, sim.fleet.slope_limit)
    yy, xx = np.mgrid[0:m.ny, 0:m.nx]
    inside = np.hypot(m.ox + xx * m.res - zx, m.oy + yy * m.res - zy) < zr - 0.3
    assert inside.any() and (grid[inside] >= INF).all()


def test_features_resent_on_eruption_change():
    sim = Simulation(Config(seed=42, geysers=True))
    sim.init_message()
    v = sim.terrain.geysers[0]
    msg = sim.features_msg()["geysers"]
    assert {"id", "x", "y", "radius", "active", "period_s", "next_s"} <= set(msg[0])
    t_change = min(g.next_change(sim.t) for g in sim.terrain.geysers)
    seen = []
    while sim.t < t_change + 1.0:
        sim.step()
        tm = sim.tick_message()
        if "features" in tm:
            seen.append(tm["features"]["geysers"])
    assert seen, "features resent when an eruption starts or ends"
    act = {g["id"]: g["active"] for g in seen[-1]}
    assert act == {g.id: g.active(sim.t) for g in sim.terrain.geysers}
    assert v.id in act


def test_headless_run_with_geysers():
    sim = Simulation(Config(seed=42, geysers=True))
    sim.run_for(90.0)
    s = sim.summary()
    assert s["t"] >= 89.9
    assert sim.geysers.eruptions >= 1
    for r in sim.rovers:
        assert math.isfinite(r.x) and math.isfinite(r.y)
