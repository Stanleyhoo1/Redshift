"""v7 surface features: dune fields and ice sheets (flags dunes, ice_sheets)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import v7_surface  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.terrain import Terrain  # noqa: E402


def test_flags_off_keep_the_world_and_physics_values():
    cfg = Config(seed=42)
    t = Terrain(cfg)
    assert t.dunes == [] and t.ice_sheets == []
    # no region cause: exactly what the sand model wrote before
    assert v7_surface.compose_wheel(0.8, 0.02, False, v7_surface.NONE, cfg) == (0.8, 0.02)
    assert v7_surface.compose_wheel(0.8, 0.02, True, v7_surface.NONE, cfg) == (cfg.sand_friction,
                                                                                0.02 + cfg.sand_wheel_damping)


def test_dunes_and_ice_are_deterministic_and_local():
    base = Terrain(Config(seed=42, water=False, spires=False, lifeforms=False)).heights
    cfg = Config(seed=42, water=False, spires=False, lifeforms=False, dunes=True, ice_sheets=True)
    a, b = Terrain(cfg), Terrain(cfg)
    assert np.array_equal(a.heights, b.heights)
    assert a.dunes and a.ice_sheets
    assert a.dunes == b.dunes and a.ice_sheets == b.ice_sheets
    # the lander basin is untouched
    R = np.hypot(a.X, a.Y)
    near = R < cfg.lander_flat_radius + 4.0
    assert np.array_equal(a.heights[near], base[near])
    # ice is level, dunes stay gentle
    for s in a.ice_sheets:
        d = np.hypot(a.X - s["x"], a.Y - s["y"])
        h = a.heights[d <= s["radius"]]
        assert float(h.max() - h.min()) < 1e-6
    from redshift.terrain import slope_deg
    added = slope_deg(a.heights - base, a.res)        # what the dunes themselves add
    for s in a.dunes:
        d = np.hypot(a.X - s["x"], a.Y - s["y"])
        assert float(added[d <= s["radius"]].max()) < 16.0


def test_ice_mu_only_on_ice():
    cfg = Config(seed=42, water=False, spires=False, lifeforms=False, dunes=True, ice_sheets=True)
    t = Terrain(cfg)
    s = v7_surface.Surface(t)
    ice = t.ice_sheets[0]
    assert s.region_at(ice["x"], ice["y"]) == v7_surface.ICE
    assert s.region_at(0.0, 0.0) == v7_surface.NONE
    assert v7_surface.compose_wheel(0.8, 0.02, False, v7_surface.ICE, cfg)[0] == v7_surface.ICE_MU
    assert v7_surface.compose_wheel(0.8, 0.02, False, v7_surface.NONE, cfg)[0] == 0.8
    dune = t.dunes[0]
    assert s.region_at(dune["x"], dune["y"]) == v7_surface.DUNE
    assert 0.3 < v7_surface.compose_wheel(0.8, 0.02, False, v7_surface.DUNE, cfg)[0] < 0.8


def test_ice_zones_in_fleet_stats_only_when_on_and_learned():
    from redshift.sim import Simulation
    sim = Simulation(Config(seed=42, ice_sheets=True))
    assert "ice_zones" not in sim.fleet.stats()
    v7_surface.add_ice_zone(sim.fleet, 1.0, sim.rovers[0], 10.0, 10.0, 5.0)
    assert sim.fleet.stats()["ice_zones"] == [[10.0, 10.0, 5.0]]
    assert "ice" in sim.features_msg()
    off = Simulation(Config(seed=42))
    assert "ice_zones" not in off.fleet.stats()
    assert "ice" not in off.features_msg() and "dunes" not in off.features_msg()
