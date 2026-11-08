"""v7 landforms: mesas, canyon, pits and boulder fields."""
import hashlib
import math

import numpy as np

from redshift.config import Config
from redshift.sim import dock_spots
from redshift.terrain import Terrain, slope_deg


def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_flags_off_unchanged():
    t = Terrain(Config(seed=42))
    assert t.landform_mask is None and t.bfield == []
    assert t.landforms is None


def test_landforms_appear_and_starts_reach_lander():
    cfg = Config(seed=42, mesas=True, canyon=True, pits=True, boulder_fields=True)
    t = Terrain(cfg)
    lf = t.landforms
    assert lf["mesas"] and lf["pits"] and lf["boulder_fields"]
    assert 0 < len(t.bfield) < 300
    for x, y in dock_spots(cfg, cfg.num_rovers):
        i, j = t.ij(x, y)
        assert t.reach[j, i]
    for tg in t.targets:
        i, j = t.ij(tg.x, tg.y)
        assert not t.landform_mask[j, i]
    for sp in t.spires:
        i, j = t.ij(sp.x, sp.y)
        assert not t.landform_mask[j, i]
    assert float(t.reach.mean()) > 0.4
    plain = Terrain(Config(seed=42))
    assert _hash(plain.heights) != _hash(t.heights)


def test_canyon_ramps_within_tow_limit():
    cfg = Config(seed=42, canyon=True)
    t = Terrain(cfg)
    can = t.landforms["canyon"]
    assert can is not None and len(can["ramps"]) == 2 and can["width"] >= 4.0
    px, py, s, ramp_len, width = t._canyon_line
    hc = np.array([t.height_at(float(x), float(y)) for x, y in zip(px, py)])
    grade = np.degrees(np.arctan(np.abs(np.diff(hc)) / np.diff(s)))
    ramp = (s[1:] <= ramp_len) | (s[1:] >= s[-1] - ramp_len)
    # 2 m running mean of the centreline grade on both ramps stays under the tow limit
    run = np.convolve(grade, np.ones(4) / 4, mode="same")
    assert float(run[ramp].max()) <= cfg.tow_limit_deg
    for r in can["ramps"]:
        i, j = t.ij(r["x"], r["y"])
        assert t.reach[j, i] or t.reach[max(0, j - 2):j + 3, max(0, i - 2):i + 3].any()


def test_flags_off_hash_and_second_seed():
    a = Terrain(Config(seed=42))
    b = Terrain(Config(seed=42, boulder_fields=False, mesas=False))
    assert _hash(a.heights) == _hash(b.heights) and _hash(a.truth_heights) == _hash(b.truth_heights)
    # the same hashes the code before v7 landforms gave for seed 42
    assert _hash(a.heights).startswith("c49cde1d196f6f95")
    assert _hash(a.truth_heights).startswith("68d069f15687a2f4")
    t = Terrain(Config(seed=7, mesas=True, pits=True))
    assert t.landforms["mesas"] and t.landforms["pits"]
    m = t.landform_mask
    for sp in t.spires:
        i, j = t.ij(sp.x, sp.y)
        assert not m[j, i]
