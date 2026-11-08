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
    assert lf["mesas"] and lf["boulder_fields"]
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
