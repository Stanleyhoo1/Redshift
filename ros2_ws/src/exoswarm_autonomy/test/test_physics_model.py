import math

import pytest

from exoswarm_autonomy.physics_model import RoverPhysics, RoverSpec


@pytest.fixture
def phys():
    return RoverPhysics(RoverSpec())


def test_earth_gravity_is_slip_bound(phys):
    lim = phys.limits(1.0)
    assert 35.0 <= lim.max_slope_deg <= 39.0
    assert lim.factor == 'slip'


def test_super_earth_is_drive_torque_bound(phys):
    lim = phys.limits(2.2)
    assert 15.5 <= lim.max_slope_deg <= 18.0
    assert lim.factor == 'drive_torque'


def test_max_slope_monotonically_decreases_with_gravity(phys):
    slopes = [phys.limits(g / 10).max_slope_deg for g in range(10, 31)]
    assert all(a > b for a, b in zip(slopes, slopes[1:]))


def test_tip_alarm_at_2_2g(phys):
    assert phys.limits(2.2).tip_alarm_deg == pytest.approx(28.8, abs=1.0)
    # alarm never exceeds 70% of the static tip angle
    assert phys.limits(1.0).tip_alarm_deg <= 0.7 * phys.limits(1.0).tip_deg + 1e-9


def test_speed_limit_drops_with_gravity(phys):
    assert phys.limits(2.2).v_max < phys.limits(1.0).v_max
    assert phys.limits(1.0).v_max == pytest.approx(1.0)


def test_energy_uphill_exceeds_flat_and_scales_with_g(phys):
    flat = phys.energy_per_m(2.2, dh=0.0, d=1.0)
    uphill = phys.energy_per_m(2.2, dh=0.3, d=1.0)
    downhill = phys.energy_per_m(2.2, dh=-0.3, d=1.0)
    assert uphill > flat
    assert downhill == pytest.approx(flat)
    assert phys.energy_per_m(3.0, 0.0, 1.0) > phys.energy_per_m(1.0, 0.0, 1.0)


def test_from_spec_ignores_unknown_keys():
    p = RoverPhysics.from_spec({'mass_kg': 60.0, 'tau_max_nm': 20.8, 'camera_xyz': [0, 0, 1]})
    assert math.isclose(p.spec.mass_kg, 60.0)


def test_limits_as_dict_is_json_ready(phys):
    d = phys.limits(2.2).as_dict()
    assert set(d) >= {'g', 'max_slope_deg', 'factor', 'drive_deg', 'slip_deg', 'tip_deg',
                      'tip_alarm_deg', 'v_max', 'energy_j_per_m'}
