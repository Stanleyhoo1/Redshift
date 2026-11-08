"""v7 boulder fields: the per-field `rocks` list in `features` matches the bfield geoms, and a
features message with boulder fields off is byte-for-byte what it was before the list existed."""
import hashlib
import json
import math

import mujoco
import pytest

from redshift.config import Config
from redshift.sim import Simulation


def _digest(msg):
    return hashlib.sha256(json.dumps(msg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture(scope="module")
def on():
    return Simulation(Config(seed=42, boulder_fields=True))


def test_rocks_match_bfield_geoms(on):
    fields = on.features_msg()["boulder_fields"]
    rocks = [r for f in fields for r in f["rocks"]]
    m = on.model
    geoms = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("bfield_")]
    assert len(rocks) == len(geoms) == len(on.terrain.bfield) == sum(f["count"] for f in fields)
    assert 30 <= len(rocks) <= 120
    for k, (x, y, size) in enumerate(rocks):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"bfield_{k}")
        gx, gy = float(m.geom_pos[g][0]), float(m.geom_pos[g][1])
        assert abs(x - gx) <= 0.006 and abs(y - gy) <= 0.006
        sx, sy = float(m.geom_size[g][0]), float(m.geom_size[g][1])
        assert abs(size - (sx + sy) / 2) <= 0.006
        assert all(v == round(v, 2) for v in (x, y, size))
    for f in fields:
        for x, y, _ in f["rocks"]:
            assert math.hypot(x - f["x"], y - f["y"]) <= f["radius"] + 0.01


def test_flags_off_features_unchanged():
    # Digests of the features message before the rocks list was added (seed 42).
    off = Simulation(Config(seed=42)).features_msg()
    assert "boulder_fields" not in off
    assert _digest(off) == "29b6c77edb88abc3821d4e5531521d400d94b3fdd4ea83bc4718a9516094159f"
    other = Simulation(Config(seed=42, mesas=True, pits=True)).features_msg()
    assert "boulder_fields" not in other
    assert _digest(other) == "dc6f070ce1318d6f51212221c4cb9dbdfcb656917d759c5e4eb073232fd05553"
