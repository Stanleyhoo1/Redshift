import math

import numpy as np
import pytest

from exoswarm_sim.terrain_gen import SECTOR_GRID_N, generate_terrain


@pytest.fixture(scope='module')
def terr():
    return generate_terrain(100.0, seed=42)


def test_deterministic_by_seed():
    a = generate_terrain(50.0, seed=7)
    b = generate_terrain(50.0, seed=7)
    c = generate_terrain(50.0, seed=8)
    assert np.array_equal(a.heights, b.heights)
    assert a.rocks == b.rocks and a.science == b.science
    assert not np.array_equal(a.heights, c.heights)


def test_grid_size_and_resolution(terr):
    assert terr.n == SECTOR_GRID_N[100.0]
    assert terr.heights.shape == (terr.n, terr.n)
    assert terr.res == pytest.approx(100.0 / (terr.n - 1))
    assert terr.heights.dtype == np.float32
    assert float(terr.heights.min()) == pytest.approx(0.0, abs=1e-4)


def test_spawn_pad_is_flat(terr):
    slope = terr.slope_deg()
    for (x, y) in terr.spawn_xy:
        i, j = terr.index_of(x, y)
        r = int(math.ceil(2.5 / terr.res))
        assert slope[i - r:i + r + 1, j - r:j + r + 1].max() < 3.0


def test_has_steep_ridge_but_mostly_gentle(terr):
    slope = terr.slope_deg()
    assert slope.max() >= 22.0
    assert slope.max() < 42.0   # everything stays crossable at 1 g's ~37 deg limit plus margin
    assert np.percentile(slope, 75) < 12.0
    assert len(terr.ridges) >= 1


def test_rocks_and_science_avoid_spawn(terr):
    assert len(terr.rocks) >= 25
    assert 4 <= len(terr.science) <= 8
    for (sx, sy) in terr.spawn_xy:
        for r in terr.rocks:
            assert math.hypot(r.x - sx, r.y - sy) > 5.0
        for (x, y) in terr.science:
            assert math.hypot(x - sx, y - sy) > 6.0


def test_features_inside_sector(terr):
    half = terr.size_m / 2
    for r in terr.rocks:
        assert -half < r.x < half and -half < r.y < half
        assert 0.2 <= r.height <= 1.2


def test_height_at_matches_vertices_and_interpolates(terr):
    i, j = 10, 20
    x = -terr.size_m / 2 + j * terr.res
    y = -terr.size_m / 2 + i * terr.res
    assert terr.height_at(x, y) == pytest.approx(float(terr.heights[i, j]), abs=1e-4)
    mid = terr.height_at(x + terr.res / 2, y)
    lo, hi = sorted([terr.heights[i, j], terr.heights[i, j + 1]])
    assert lo - 1e-4 <= mid <= hi + 1e-4
    xs = np.array([x, x + 1.0])
    assert terr.height_at(xs, np.array([y, y])).shape == (2,)


def test_npz_roundtrip(tmp_path, terr):
    p = tmp_path / 't.npz'
    terr.save(p)
    from exoswarm_sim.terrain_gen import Terrain
    t2 = Terrain.load(p)
    assert np.array_equal(t2.heights, terr.heights)
    assert t2.rocks == terr.rocks and t2.spawn_xy == terr.spawn_xy
