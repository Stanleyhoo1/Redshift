"""v7 "More lifeforms" (Config.life_variety): skitterers, drifters, crawlers and burrowers join the grazers,
chosen by the biome weights. With the flag off the world, the herds and every lifeform message are
exactly as before.

    python -m pytest -q tests/test_v7_life.py        (about two minutes)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import gemma as g  # noqa: E402
from redshift import v7_life  # noqa: E402
from redshift.config import SETTINGS, Config  # noqa: E402
from redshift.hazards import HazardError  # noqa: E402
from redshift.rover import LARGE_KEEP_M  # noqa: E402
from redshift.sim import Simulation  # noqa: E402
from redshift.terrain import Terrain  # noqa: E402
from redshift.tracker import Track  # noqa: E402

EM_DASH = chr(0x2014)
LIDAR_GROUPS = np.array([1, 1, 1, 0, 0, 1], np.uint8)
# sha256 of seed 42's herds, init lifeforms and four tick lifeform lists, computed with the code from
# before this feature (commit f3f2694): the flags-off world must not change at all
FLAGS_OFF_DIGEST = "c6ec9b2f6db6cf0c354e12abed0fd01d0ac66033d90804e02aa5d4ea07204872"


def make(seed=1, **cfg) -> Simulation:
    return Simulation(Config(seed=seed, plan_clock="model", **cfg))


def bring(sim, kind, rid="r1", t=5.0):
    """The lifeform hazard, forced to pick this kind (test harness)."""
    orig = v7_life.pick_kind
    v7_life.pick_kind = lambda rng, w, used=None: kind
    try:
        sim.run_for(t)
        for _ in range(100):
            try:
                return sim.hazards.trigger("lifeform", rid)
            except HazardError:
                sim.step()
    finally:
        v7_life.pick_kind = orig
    raise AssertionError("could not bring a group")


# ====================================================================== flags off: nothing changes
def test_config_flags_and_setting():
    cfg = Config()
    assert cfg.life_variety is False and cfg.biomes is False
    keys = [s.key for s in SETTINGS]
    assert keys[keys.index("lifeforms") + 1] == "life_variety"
    s = next(s for s in SETTINGS if s.key == "life_variety")
    assert (s.type, s.label, s.group) == ("bool", "More lifeforms", "features")
    assert s.help == "Skitterers, drifters, crawlers and burrowers join the grazers."
    assert EM_DASH not in s.help


def test_from_env_turns_variety_on(monkeypatch):
    monkeypatch.delenv("LIFE_VARIETY", raising=False)
    monkeypatch.delenv("BIOMES", raising=False)
    cfg = Config.from_env()
    assert cfg.life_variety is True and cfg.biomes is True
    monkeypatch.setenv("LIFE_VARIETY", "0")
    assert Config.from_env().life_variety is False


def test_flags_off_world_and_messages_identical():
    sim = make(seed=42)
    herds = [[h.id, round(h.cx, 4), round(h.cy, 4), round(h.radius, 4), [[round(v, 4) for v in m] for m in h.members]]
             for h in sim.terrain.herds]
    init = sim.init_message()
    lf0 = init.get("lifeforms") or init.get("world", {}).get("lifeforms")
    ticks = []
    for k in range(40):
        sim.step()
        if k % 10 == 9:
            ticks.append(sim.tick_message()["lifeforms"])
    blob = json.dumps({"herds": herds, "init_lifeforms": lf0, "ticks": ticks}, sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == FLAGS_OFF_DIGEST
    for msgs in ticks:
        for m in msgs:
            assert m["kind"] == "grazer"
            assert set(m) == {"id", "kind", "x", "y", "z", "yaw", "speed", "size"}
    assert all(getattr(h, "kind", "grazer") == "grazer" for h in sim.terrain.herds)


# ====================================================================== kinds by biome
class _Ter:
    def __init__(self, primary, at=None):
        self.biome = {"id": primary, "name": primary, "secondary": "rust"}
        if at is not None:
            self.biome_id_at = at


def test_weights_follow_the_biome():
    assert v7_life.weights_at(object(), 0, 0) == {k: 1.0 for k in v7_life.KIND_IDS}     # biomes off
    w = v7_life.weights_at(_Ter("frost"), 0, 0)
    assert w == {"grazer": 0.0, "skitterer": 2.0, "drifter": 0.0, "crawler": 0.0, "burrower": 3.0}
    # the region decides when the terrain can tell (the secondary biome there)
    ter = _Ter("frost", at=lambda x, y: "basalt" if x > 0 else "frost")
    assert v7_life.weights_at(ter, 5, 0)["crawler"] == 3.0 and v7_life.weights_at(ter, -5, 0)["crawler"] == 0.0
    assert v7_life.weights_at(_Ter("rust"), 0, 0) == {"grazer": 2.0, "skitterer": 1.0, "drifter": 0.0,
                                                       "crawler": 0.0, "burrower": 0.0}
    rng = np.random.default_rng(3)
    n = Counter(v7_life.pick_kind(rng, v7_life.weights_at(_Ter("ash"), 0, 0)) for _ in range(3000))
    assert set(n) == {"drifter", "skitterer", "burrower"}
    assert n["drifter"] > n["skitterer"] > n["burrower"]
    assert abs(n["drifter"] / 3000 - 0.5) < 0.05


def test_vary_uses_the_biome_at_each_home():
    cfg = Config(seed=42, plan_clock="model")
    ter = Terrain(cfg)
    assert all(not hasattr(h, "kind") for h in ter.herds)
    ter.biome = {"id": "frost", "name": "Frost plains", "secondary": "rust"}
    ter.biome_id_at = lambda x, y: "frost"
    v7_life.vary(ter, Config(seed=42, plan_clock="model", life_variety=True))
    kinds = [h.kind for h in ter.herds]
    assert len(kinds) >= 4 and set(kinds) <= {"burrower", "skitterer"}, kinds
    for h in ter.herds:
        spec = v7_life.KINDS[h.kind]
        assert spec.group[0] <= len(h.members) <= spec.group[1] or len(h.members) >= 2
        for (x, y, _yaw, size, cruise) in h.members:
            assert spec.size[0] <= size <= spec.size[1] and spec.cruise[0] <= cruise <= spec.cruise[1]


def test_variety_world_has_several_kinds_deterministic():
    a, b = Terrain(Config(seed=1, life_variety=True)), Terrain(Config(seed=1, life_variety=True))
    assert [(h.kind, h.members) for h in a.herds] == [(h.kind, h.members) for h in b.herds]
    assert len({h.kind for h in a.herds}) >= 3


# ====================================================================== behaviour per kind
def test_each_kind_moves_its_own_way():
    sim = make(seed=1, life_variety=True)
    lf = sim.lifeforms
    by = {}
    for c in lf.creatures:
        by.setdefault(c.kind, []).append(c)
    assert set(by) == {"grazer", "skitterer", "drifter", "crawler", "burrower"}
    rec = {k: {"speed": [], "hover": [], "up": []} for k in by}
    start = {c.id: (c.x, c.y) for c in lf.creatures}
    for _ in range(int(40 / 0.2)):
        sim.step()
        for k, cs in by.items():
            for c in cs:
                rec[k]["speed"].append(c.speed)
                rec[k]["hover"].append(c.z - sim.terrain.height_at(c.x, c.y))
                rec[k]["up"].append(getattr(c, "up", 1.0))
    msgs = {m["id"]: m for m in sim.tick_message()["lifeforms"]}
    assert {m["kind"] for m in msgs.values()} == set(by)
    assert max(rec["skitterer"]["speed"]) > 0.5                       # quick
    assert max(rec["crawler"]["speed"]) <= 0.16                       # slow
    assert max(rec["grazer"]["speed"]) <= 0.51
    hov = rec["drifter"]["hover"]
    assert 1.3 <= min(hov) and max(hov) <= 3.7                        # floats 1.5 to 3.5 m up
    ups = rec["burrower"]["up"]
    assert min(ups) == 0.0 and max(ups) == 1.0                         # pops up and sinks
    assert min(rec["burrower"]["hover"]) < -0.3                       # under the ground when down
    for c in by["drifter"]:                                            # drifts with its wind
        assert math.hypot(c.x - start[c.id][0], c.y - start[c.id][1]) > 2.0
    assert all("up" in msgs[c.id] for c in by["burrower"])
    assert all("up" not in msgs[c.id] for c in by["grazer"] + by["drifter"])


def test_skitterers_scatter_from_a_rover():
    sim = make(seed=1, life_variety=True)
    lf = sim.lifeforms
    herd = next(h for h in lf.herds if h.kind == "skitterer")
    r = sim.rovers[0]
    lf.move_herd(herd, r.x + 6.0, r.y, None)
    d0 = np.mean([math.hypot(c.x - r.x, c.y - r.y) for c in herd.members])
    for _ in range(int(4 / 0.2)):
        sim.step()
    d1 = np.mean([math.hypot(c.x - r.x, c.y - r.y) for c in herd.members])
    assert d1 > d0 + 2.0, (d0, d1)
    assert max(c.speed for c in herd.members) > 1.0


# ====================================================================== LiDAR
def _rays_hit_life(sim, frm, to_pts) -> bool:
    m, d = sim.model, sim.data
    gid = np.zeros(1, np.int32)
    lg = set(int(k) for k in sim.world.life_gids)
    for p in to_pts:
        v = np.array(p, float) - np.array(frm, float)
        dist = np.linalg.norm(v)
        x = mujoco.mj_ray(m, d, np.array(frm, float), v / dist, LIDAR_GROUPS, 1, -1, gid)
        if x >= 0 and int(gid[0]) in lg:
            return True
    return False


def test_burrowers_seen_by_lidar_only_while_up():
    sim = make(seed=1, life_variety=True)
    lf = sim.lifeforms
    c = next(c for c in lf.creatures if c.kind == "burrower")
    gz = sim.terrain.height_at(c.x, c.y)
    frm = (c.x + 3.0, c.y, gz + 1.1)
    aims = [(c.x, c.y, gz + h) for h in np.linspace(-0.2, 1.0, 13)]
    c.burrow, c.up = "down", 0.0
    lf._place(c, sim.t)
    mujoco.mj_forward(sim.model, sim.data)
    assert not _rays_hit_life(sim, frm, aims)
    c.burrow, c.up = "up", 1.0
    lf._place(c, sim.t)
    mujoco.mj_forward(sim.model, sim.data)
    assert _rays_hit_life(sim, frm, aims)


def test_drifters_seen_by_lidar_at_their_height():
    sim = make(seed=1, life_variety=True)
    lf = sim.lifeforms
    c = next(c for c in lf.creatures if c.kind == "drifter")
    lf._place(c, sim.t)
    mujoco.mj_forward(sim.model, sim.data)
    gz = sim.terrain.height_at(c.x, c.y)
    assert c.z - gz >= 1.4
    # a rover's LiDAR sits about 1.1 m up and looks at most 2 deg above level: it sees the tendrils
    frm = (c.x + 4.0, c.y, gz + 1.1)
    aims = [(c.x, c.y, gz + 1.1 + 4.0 * math.tan(math.radians(e))) for e in (2.0, -1.0, -2.5, -4.7, -7.5)]
    assert _rays_hit_life(sim, frm, aims)


# ====================================================================== rovers keep their distance
@pytest.mark.parametrize("kind,keep", [("crawler", 5.3), ("burrower", 3.3), ("drifter", 3.3)])
def test_rovers_keep_their_distance(kind, keep):
    sim = make(seed=1, life_variety=True)
    h = bring(sim, kind)
    assert h["type"] == "lifeform"
    ev = [e for e in sim.fleet.events if "wanders into view" in e["msg"]]
    assert ev and v7_life.KINDS[kind].many in ev[-1]["msg"] or "crawler" in ev[-1]["msg"]
    mine = [c for c in sim.lifeforms.creatures if c.kind == kind]
    mind = 99.0
    for _ in range(int(40 / 0.2)):
        sim.step()
        for r in sim.rovers:
            for c in mine:
                if kind == "burrower" and c.up < 0.5:
                    continue                    # underground: nobody can see it
                mind = min(mind, math.hypot(c.x - r.x, c.y - r.y))
    assert mind >= keep, (kind, mind)


def test_keep_distance_from_the_rovers_own_readings():
    sim = make(seed=1, life_variety=True)
    r = sim.rovers[0]
    tr = Track("k1", 0.0, 0.0, 1.0, 0.0, "r1")
    assert r.keep_m(tr) == sim.cfg.observe_min_dist
    tr.size = 1.7
    assert r.keep_m(tr) == LARGE_KEEP_M == 6.0
    tr.size = 1.0
    tr.label = "possible crawler, slow and large, about 1.4 m"
    assert r.keep_m(tr) == 6.0 and r.standoff_m(tr) == 8.0
    off = make(seed=1)
    tr.size = 2.0
    assert off.rovers[0].keep_m(tr) == off.cfg.observe_min_dist


def test_hazard_kind_follows_the_biome_at_the_spot():
    got = []
    for rid in ("r1", "r2", "r3"):
        sim = make(seed=1, life_variety=True)
        sim.terrain.biome = {"id": "frost", "name": "Frost plains", "secondary": "rust"}
        sim.terrain.biome_id_at = lambda x, y: "frost"
        sim.run_for(5.0)
        for _ in range(100):
            try:
                h = sim.hazards.trigger("lifeform", rid)
                break
            except HazardError:
                sim.step()
        moved = [hd for hd in sim.lifeforms.herds if math.hypot(hd.cx - h["x"], hd.cy - h["y"]) < 0.01]
        assert moved
        got.append(moved[0].kind)
        msg = [e["msg"] for e in sim.fleet.events if "wanders into view" in e["msg"]][-1]
        assert v7_life.KINDS[moved[0].kind].many in msg or v7_life.KINDS[moved[0].kind].one in msg, msg
    assert set(got) <= {"burrower", "skitterer"}, got     # frost: burrowers 3, skitterers 2, nothing else


# ====================================================================== Gemma names a kind from its camera
def test_describe_prompt_lists_kinds_only_when_on():
    obs = {"distance": 6.0, "speed": 0.1, "size": 1.7}
    plain = g.make_description_request(1, "r1", 0.0, {"battery": 0.8}, obs, None)
    s0, u0 = g.build_prompt(plain)
    assert "Kinds" not in u0 and "kinds" not in plain.payload
    req = g.make_description_request(1, "r1", 0.0, {"battery": 0.8}, obs, None, kinds=tuple(g.KIND_LOOKS))
    s1, u1 = g.build_prompt(req)
    assert "Kinds (name one if seen): grazer" in u1 and "crawler big domed shell" in u1
    assert g.estimate_tokens(s1 + u1) < 400
    fake = g.FakeBackend(0.0)
    assert "crawler" in fake._describe(req)["label"]
    assert "crawler" not in fake._describe(plain)["label"]
    small = g.make_description_request(1, "r1", 0.0, {}, {"speed": 0.9, "size": 0.6}, None, kinds=("skitterer",))
    assert "skitterer" in fake._describe(small)["label"]
