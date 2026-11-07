import json
import math

import numpy as np
import pytest

from redshift_autonomy.physics_model import RoverPhysics
from redshift_sim.rover_hw import (DAMAGE_FAULTS, DAMAGED_HEALTH, G0, STRIKE, SOLAR_PEAK_W, SUN_ELEVATION_DEG,
                                   SUN_VEC, Drivetrain, hw_status_dict, peak_g_above_gravity, planar_distance,
                                   solar_output, strike_peak_accel, strike_shape, strike_specific_force,
                                   strike_wrenches_world)

SUN_AZ = math.atan2(SUN_VEC[1], SUN_VEC[0])


# ---------------------------------------------------------------- solar panel
def test_level_clean_panel_gets_sun_elevation_share_of_peak():
    w, irr = solar_output(0.0, 0.0, 0.0, 0.0)
    assert irr == pytest.approx(1.0)
    assert w == pytest.approx(SOLAR_PEAK_W * math.sin(math.radians(SUN_ELEVATION_DEG)), abs=0.1)   # ≈ 180 W
    assert math.degrees(math.asin(SUN_VEC[2])) == pytest.approx(55.0)


def test_dust_transmittance_scales_output():
    w0, _ = solar_output(0.0, 0.0, 0.0)
    for dust in (0.25, 0.85, 1.0):
        w, irr = solar_output(dust, 0.0, 0.0)
        assert irr == pytest.approx(1.0 - 0.85 * dust)
        assert w == pytest.approx(w0 * irr)
    assert solar_output(3.0, 0.0, 0.0)[1] == pytest.approx(0.15)          # clamped
    assert solar_output(-1.0, 0.0, 0.0)[1] == pytest.approx(1.0)


def test_tilt_toward_the_sun_raises_output_and_away_lowers_it_never_negative():
    flat, _ = solar_output(0.0, 0.0, 0.0, SUN_AZ)
    # facing the sun (yaw = sun azimuth): nose down (positive pitch) tips the deck normal toward the sun
    toward, _ = solar_output(0.0, 0.0, math.radians(20.0), SUN_AZ)
    away, _ = solar_output(0.0, 0.0, math.radians(-20.0), SUN_AZ)
    assert toward > flat > away > 0.0
    assert toward == pytest.approx(SOLAR_PEAK_W * math.cos(math.radians(35.0 - 20.0)), abs=0.5)
    # deck turned 90° + beyond the sun's elevation away from it: no direct sun at all
    assert solar_output(0.0, 0.0, math.radians(-150.0), SUN_AZ)[0] == 0.0


# ---------------------------------------------------------------- drivetrain
def test_healthy_drivetrain_passes_commands_through():
    d = Drivetrain()
    assert not d.damaged and not d.repairing and d.health_pct == 100.0
    assert d.scale(0.8, 0.3, 0.85) == (0.8, 0.3)


def test_critical_damage_makes_the_rover_crawl_and_veer():
    d = Drivetrain()
    d.apply_critical_damage()
    assert d.wheel_health == list(DAMAGED_HEALTH) == [0.08, 0.35, 0.05, 0.30]
    assert 'DRIVE_FL_OVERCURRENT' in d.faults and 'DRIVE_RL_ENCODER_LOST' in d.faults
    assert d.faults == list(DAMAGE_FAULTS) and d.damaged
    v, w = d.scale(0.8, 0.0, 0.85)
    left, right = (0.08 + 0.05) / 2 * 0.8, (0.35 + 0.30) / 2 * 0.8
    assert v == pytest.approx((left + right) / 2) and v < 0.25 * 0.8      # crawls
    assert w == pytest.approx((right - left) / 0.85) and w > 0.2          # veers toward the weaker (left) side
    # a turn command is distorted too
    v2, w2 = d.scale(0.0, 1.0, 0.85)
    assert w2 == pytest.approx(1.0 * (0.065 + 0.325) / 2, abs=1e-9) and v2 > 0.0


def test_repair_ramps_health_back_over_four_seconds_then_clears_faults_once():
    d = Drivetrain()
    d.apply_critical_damage()
    assert d.start_repair(10.0, duration_s=4.0)
    assert not d.start_repair(11.0)                                       # already under way
    assert d.update(12.0) is False
    assert d.wheel_health[0] == pytest.approx(0.08 + 0.92 * 0.5)
    assert d.faults and d.repairing
    assert d.update(14.0) is True
    assert d.wheel_health == [1.0] * 4 and d.faults == [] and not d.repairing and not d.damaged
    assert d.update(15.0) is False
    # a new strike mid-repair cancels it
    d.apply_critical_damage()
    d.start_repair(20.0)
    d.apply_critical_damage()
    assert not d.repairing and d.update(30.0) is False and d.wheel_health == list(DAMAGED_HEALTH)


def test_motor_currents_are_plausible_and_flag_the_faulty_wheels():
    d = Drivetrain()
    idle = d.motor_current(0.0, 0.0)
    assert idle == pytest.approx([0.4] * 4)
    cruise = d.motor_current(0.8, 0.0)
    assert cruise == pytest.approx([3.12] * 4, abs=0.05)                  # protocol example ≈ 3.1 A
    turning = d.motor_current(0.0, 1.0)
    assert all(i > 0.4 for i in turning)
    d.apply_critical_damage()
    fl, fr, rl, rr = d.motor_current(0.8, 0.0)
    assert fl > 10.0                                                      # overcurrent: high current, no output
    assert rl == pytest.approx(0.4)                                       # encoder lost: driver idles the motor
    assert fr > cruise[1] and rr > cruise[3]                              # binding gearboxes draw more per output
    assert all(0.0 <= i <= 12.0 for i in (fl, fr, rl, rr))
    noisy = Drivetrain().motor_current(0.8, 0.0, np.random.default_rng(1))
    assert noisy == pytest.approx(cruise, abs=0.25) and noisy != cruise


def test_hw_status_payload_matches_protocol():
    d = Drivetrain()
    d.apply_critical_damage()
    msg = hw_status_dict('rover_0', 1762550000.123456, d, 182.44, 0.931, d.motor_current(0.5, 0.1))
    assert set(msg) == {'robot', 't', 'wheel_health', 'faults', 'motor_current_a', 'solar_w', 'irradiance'}
    assert msg['robot'] == 'rover_0' and msg['solar_w'] == 182.4 and msg['irradiance'] == 0.931
    assert len(msg['wheel_health']) == 4 and len(msg['motor_current_a']) == 4
    assert msg['faults'] == list(DAMAGE_FAULTS)
    json.dumps(msg)                                                       # strict JSON
    msg['faults'].append('X')
    assert 'X' not in d.faults                                            # a copy, not the live list


# ---------------------------------------------------------------- debris strike profile
def test_strike_peak_is_about_five_g_above_gravity_at_2_2g():
    g = 2.2
    T = STRIKE['duration_s']
    assert strike_shape(0.0) == 0.0 and strike_shape(T / 2) == pytest.approx(1.0) and strike_shape(T + 0.01) == 0.0
    peak = np.linalg.norm(np.array([0.0, 0.0, g * G0]) + strike_specific_force(T / 2))
    assert 4.0 <= peak_g_above_gravity(peak, g) <= 6.0
    assert peak_g_above_gravity(peak, g) == pytest.approx(5.1, abs=0.05)
    assert peak_g_above_gravity(g * G0 * 0.9, g) == 0.0


def test_gazebo_strike_never_lifts_a_wheel_and_reads_the_same_peak():
    """Ground-contact model: with the wheels kept loaded, the IMU reads the normal load (g·cos tilt along the chassis
    normal) plus the lateral push; the push reverses at half time so the chassis ends where it started."""
    from redshift_sim.mock_render import rpy_matrix
    m, g = 60.0, 2.2
    for roll, pitch, yaw in ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.15, -0.2, -2.5)):
        (f1, t1), (df, dt) = strike_wrenches_world(roll, pitch, yaw, g, m)
        R = rpy_matrix(roll, pitch, yaw)
        f1, df = np.array(f1), np.array(df)
        assert t1 == [0.0] * 3 and dt == [0.0] * 3                         # no roll torque: no wheel lift-off
        cos_t = R[2, 2]
        weight_n = m * G0 * g
        normal_ext = f1 @ R[:, 2]                                          # along the chassis normal
        assert normal_ext == pytest.approx((1 - STRIKE['gz_reserve']) * weight_n * cos_t)
        assert normal_ext < weight_n * cos_t                               # the ground keeps a load on the wheels
        assert abs(f1 @ R[:, 0]) < 1e-6 and abs(df @ R[:, 0]) < 1e-6 and abs(df @ R[:, 2]) < 1e-6
        push = f1 @ R[:, 1]
        assert df @ R[:, 1] == pytest.approx(-2 * push)                    # first half +L, second half −L
        a_read = math.hypot(g * G0 * cos_t, push / m)                      # specific force while the wheels slide
        assert a_read == pytest.approx(strike_peak_accel(g))
        assert peak_g_above_gravity(a_read, g) == pytest.approx(5.1, abs=0.15)
    assert RoverPhysics().limits(2.2).tip_alarm_deg == pytest.approx(28.86, abs=0.05)


def test_planar_distance():
    assert planar_distance((0.0, 0.0, 5.0), (3.0, 4.0, -1.0)) == pytest.approx(5.0)
