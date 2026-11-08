"""v7 biomes (sim/redshift/v7_biomes.py): the seeded pick, the region field, the feature weights, the
protocol fields, flags off giving byte-identical worlds, and camera safety on every biome's ground.

    python -m pytest -q tests/test_v7_biomes.py        (about two minutes; the camera part needs OpenGL)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import camera, v7_biomes as B, v7_flora  # noqa: E402
from redshift.config import SETTINGS_BY_KEY, Config  # noqa: E402
from redshift.rover import LIQUID_MIN_PCT, _camera_frame  # noqa: E402
from redshift.sim import Simulation  # noqa: E402
from redshift.terrain import Terrain  # noqa: E402
from redshift.world import rover_spawn_height  # noqa: E402

EM_DASH = chr(0x2014)
V7_ON = dict(mesas=True, canyon=True, dunes=True, boulder_fields=True, pits=True, geysers=True, ice_sheets=True,
             flora=True, ground_detail=True)


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _seed_for(primary: str, secondary: str | None = None) -> int:
    for s in range(2000):
        got = B.pick(Config(seed=s, biomes=True))
        if got[0] == primary and (secondary is None or got[1] == secondary):
            return s
    raise AssertionError(f"no seed for {primary}/{secondary}")


# ====================================================================== settings and the pick
def test_config_flags_default_off_and_from_env_on(monkeypatch):
    cfg = Config()
    assert cfg.biomes is False and cfg.life_variety is False
    for name in ("BIOMES", "LIFE_VARIETY"):
        monkeypatch.delenv(name, raising=False)
    env = Config.from_env()
    assert env.biomes is True and env.life_variety is True
    monkeypatch.setenv("BIOMES", "0")
    monkeypatch.setenv("LIFE_VARIETY", "off")
    env = Config.from_env()
    assert env.biomes is False and env.life_variety is False
    for key, label in (("biomes", "Biomes"), ("life_variety", "More lifeforms")):
        s = SETTINGS_BY_KEY[key]
        assert s.type == "bool" and s.group == "features" and s.label == label and len(s.help) <= 90
    back, errors = Config.from_settings(Config(), {"biomes": True, "life_variety": True})
    assert errors == [] and back.biomes and back.life_variety


def test_pick_is_deterministic_and_never_rust_primary():
    prim, sec = Counter(), Counter()
    for s in range(401):
        cfg = Config(seed=s, biomes=True)
        a, b = B.pick(cfg), B.pick(cfg)
        assert a == b and a[0] != a[1]
        assert a[0] != "rust" and a[0] in B.PRIMARY_IDS and a[1] in B.IDS
        prim[a[0]] += 1
        sec[a[1]] += 1
    assert set(prim) == set(B.PRIMARY_IDS) == {"basalt", "verdigris", "frost", "ash", "salt", "obsidian"}
    assert set(sec) == set(B.IDS) and sec["rust"] > 0
    assert B.pick(Config(seed=5)) is None


def test_weights_follow_the_table():
    class T:
        pass
    t = T()
    assert B.weight(t, "mesas") == 1.0                      # no biome: exactly 1
    t.biome = None
    assert B.weight(t, "flora") == 1.0 and B.count(t, "flora", 7) == 7
    t.biome = {"id": "frost", "name": "Frost plains", "secondary": "verdigris"}
    assert B.weight(t, "flora") == 0.0 and B.count(t, "flora", 9) == 0     # a primary 0 is absolute
    assert B.weight(t, "ice_sheets") == pytest.approx(0.75 * 2.5 + 0.25 * 0.0)
    t.biome = {"id": "salt", "name": "Salt flats", "secondary": "ash"}
    assert B.weight(t, "geysers") == 0.0
    assert B.weight(t, "dunes") == pytest.approx(0.75 * 2.0 + 0.25 * 1.2)
    t.biome = {"id": "verdigris", "name": "Verdigris lowlands", "secondary": "rust"}
    assert B.weight(t, "canyon") == pytest.approx(0.7) and B.has_canyon(t)
    for bid, b in B.BIOMES.items():
        assert set(b["weights"]) == set(B.FEATURES), bid
        assert b["life"] and all(k in ("grazer", "skitterer", "drifter", "crawler", "burrower") for k in b["life"])


# ====================================================================== the terrain
def test_region_field_and_terrain_api():
    covers = []
    for s in (0, 3, 11, 42, 77, 150):
        cfg = Config(seed=s, biomes=True)
        t = Terrain(cfg)
        p, sec = B.pick(cfg)
        assert t.biome == {"id": p, "name": B.BIOMES[p]["name"], "secondary": sec}
        assert t.biome_id_at(0.0, 0.0) == p                 # the lander stands on the primary ground
        bm = B.biome_map(t)
        w = np.array(bm["w"]).reshape(bm["ny"], bm["nx"])
        assert bm["origin"] == [cfg.origin[0], cfg.origin[1]] and 3.0 <= bm["res"] <= 4.5
        assert w.min() >= 0.0 and w.max() <= 1.0 and all(round(v, 2) == v for v in bm["w"])
        covers.append(float((w >= 0.5).mean()))
        ids = {t.biome_id_at(x, y) for x in np.linspace(-55, 55, 23) for y in np.linspace(-55, 55, 23)}
        assert ids == {p, sec}
        again = B.biome_map(Terrain(cfg))
        assert again == bm                                  # deterministic
    assert all(0.15 <= c <= 0.5 for c in covers), covers
    off = Terrain(Config(seed=42))
    assert off.biome is None and off.biome_id_at(1.0, 2.0) is None and B.biome_map(off) is None


def test_weights_change_the_features():
    frost = Terrain(Config(seed=_seed_for("frost"), biomes=True, flora=True))
    assert frost.flora == []
    salt = Terrain(Config(seed=_seed_for("salt"), biomes=True, geysers=True))
    assert salt.geysers == []
    ash = Terrain(Config(seed=_seed_for("ash"), biomes=True, ice_sheets=True))
    assert ash.ice_sheets == []
    s = _seed_for("verdigris")
    lush = Terrain(Config(seed=s, biomes=True, flora=True))
    plain = Terrain(Config(seed=s, flora=True))
    assert len(lush.flora) > len(plain.flora)


def test_flags_off_is_byte_identical():
    """The hashes the code before v7 biomes gave for this world (all v7 features on, 80 m, seed 42)."""
    sim = Simulation(Config(seed=42, map_size=80.0, **V7_ON))
    assert _h(np.ascontiguousarray(sim.terrain.truth_heights).tobytes()) == "17162fe5e4fe1c10"
    assert _h(json.dumps(sim.features_msg(), sort_keys=True).encode()) == "4a340ae6c6cad3e8"
    assert _h(np.asarray(sim.model.geom_matid).tobytes()) == "c3edb6dda4422383" and sim.model.ngeom == 797
    assert _h(np.asarray(sim.model.tex_data).tobytes()) == "0555dca1e6c493e8"
    assert _h(np.asarray(sim.model.light_diffuse).tobytes()) == "e6515d40f0ee09d8"
    assert _h(np.asarray(sim.model.vis.rgba.haze).tobytes()) == "772a0444302285c2"
    assert "biome" not in sim.init_message()["world"]
    assert "biome_map" not in sim.features_msg()


def test_protocol_fields_when_on():
    sim = Simulation(Config(seed=7, map_size=60.0, biomes=True, water=False, spires=False, lifeforms=False))
    f = sim.features_msg()
    assert set(f["biome_map"]) == {"nx", "ny", "origin", "res", "w"}
    assert len(f["biome_map"]["w"]) == f["biome_map"]["nx"] * f["biome_map"]["ny"]
    json.dumps(f)
    init = sim.init_message()
    assert init["world"]["biome"] == sim.terrain.biome and "biome_map" in init["features"]
    # the MuJoCo ground takes the biome's colours: primary tiles and secondary tiles
    names = {mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_MATERIAL, int(m)) for m in sim.model.geom_matid
             if m >= 0}
    assert {"ground4", "ground5"} & names and {"ground0", "ground1", "ground2", "ground3"} & names


def test_no_em_dash():
    for p in (ROOT / "redshift" / "v7_biomes.py", Path(__file__)):
        assert EM_DASH not in p.read_text(encoding="utf-8"), p.name


# ====================================================================== camera safety
@pytest.fixture(scope="module")
def gl():
    st = camera.renderer_status()
    if not st["ok"]:
        pytest.skip(f"no OpenGL backend for rendering: {st['error']}")
    return st


def _place(sim, rover, x: float, y: float, yaw: float) -> None:
    q, a = sim.data.qpos, rover.qadr
    q[a:a + 3] = (x, y, sim.terrain.height_at(x, y) + rover_spawn_height(sim.cfg) + 0.02)
    q[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    sim.data.qvel[rover.vadr:rover.vadr + rover.nv] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def test_ground_colours_keep_clear_of_the_markers():
    """Every soil, pebble and rock colour of every biome, in neutral light and in the camera's vision light,
    stays well outside the target spotting's colour tolerance (camera.DETECT_TOL) of every marker colour, and
    never passes the blue liquid rule. A patch of one biome's ground seen against another's can then never
    look like a marker, whatever the background."""
    refs = {k: v / v.max() for k, v in camera.target_palette().items()}
    vision = np.array([0.32, 0.31, 0.30])
    for bid, b in B.BIOMES.items():
        cols = [b["soil"][k] for k in ("dark", "light", "grey", "tint")] + list(b["soil"]["pebbles"])
        cols += [b["rock"][0], tuple(np.add(b["rock"][0], b["rock"][1]))]
        for c in cols:
            for lt in (np.ones(3), vision / vision.mean()):
                c2 = np.array(c) * lt
                ch = c2 / c2.max()
                for kind, ref in refs.items():
                    assert np.abs(ch - ref).max() >= camera.DETECT_TOL + 0.01, (bid, c, kind)
                assert not (c2[2] > 1.8 * c2[0] and c2[2] > 1.25 * c2[1]), (bid, c)


@pytest.mark.parametrize("bid", B.IDS)
def test_camera_checks_stay_quiet_on_every_ground(gl, monkeypatch, bid):
    """Every biome forced as the primary (its secondary another one): from open ground the camera's target
    spotting finds only real markers, the liquid check sees no liquid (there is none) and the plant check
    labels no ground as a plant."""
    other = "basalt" if bid != "basalt" else "salt"
    monkeypatch.setattr(B, "pick", lambda cfg: (bid, other) if getattr(cfg, "biomes", False) else None)
    for seed in (3, 42):
        sim = Simulation(Config(seed=seed, map_size=60.0, num_rovers=1, biomes=True, water=False, spires=False,
                                lifeforms=False))
        assert sim.terrain.biome["id"] == bid
        rover = sim.rovers[0]
        tiles = [g for g in range(sim.model.ngeom) if (mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                                                       or "").startswith("g_tile_")]
        for k, (x, y) in enumerate(((6.0, 3.0), (-9.0, 7.0), (12.0, -10.0), (-14.0, -13.0))):
            yaw = 0.9 + 1.7 * k
            _place(sim, rover, x, y, yaw)
            for d in camera.spot_targets(sim, rover):
                near = min((math.hypot(d["x"] - t.x, d["y"] - t.y) for t in sim.terrain.targets
                            if t.kind == d["kind"]), default=99.0)
                assert near < 1.0, f"{bid} seed {seed}: false {d['kind']} at ({d['x']}, {d['y']})"
            fr = _camera_frame(sim, rover.id)
            rgb = fr.rgb[int(0.45 * fr.rgb.shape[0]):].astype(np.float32)
            r_, g_, b_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            blue = 100.0 * float(((b_ > 45.0) & (b_ > 1.8 * r_) & (b_ > 1.25 * g_)).mean())
            assert blue < min(0.5, LIQUID_MIN_PCT), f"{bid} seed {seed}: {blue:.1f}% looks like liquid"
            seg = v7_flora._seg_frame(sim, rover.id)
            assert v7_flora.classify(sim.model, seg, tiles) == {}
