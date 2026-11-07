import math

import pytest

from redshift_autonomy.rover_health import (CRITICAL_PCT, G0, RECOVERED_PCT, DroneHealth, RoverHealth,
                                            drone_hover_power_w)

G = 2.2


def at_rest(h: RoverHealth, t0: float, seconds: float, dt: float = 0.01, extra_g: float = 0.0):
    t = t0
    while t < t0 + seconds - 1e-9:
        h.on_imu(t, (G + extra_g) * G0, 0.0)
        t += dt
    return t


def strike(h: RoverHealth, t0: float, peak_dev_g: float, dur: float = 0.12, dt: float = 0.01):
    """Half-sine shock of `peak_dev_g` above gravity, then quiet (after a settle, so the shock detector is armed)."""
    t = at_rest(h, t0, 0.6)
    n = int(round(dur / dt))
    for k in range(n + 1):
        h.on_imu(t, (G + peak_dev_g * math.sin(math.pi * k / n)) * G0, 0.0)
        t += dt
    return at_rest(h, t, 0.2)


def test_nominal_rover_is_healthy_and_not_critical():
    h = RoverHealth('rover_0', G, battery_pct=82.0)
    at_rest(h, 0.0, 1.0)
    s = h.state()
    assert s.pct == pytest.approx(100.0) and not s.critical
    assert s.battery_pct == pytest.approx(82.0, abs=0.05)
    d = s.as_dict()
    for k in ('pct', 'structure_pct', 'drive_pct', 'battery_pct', 'battery_wh', 'capacity_wh', 'solar_w', 'load_w',
              'net_w', 'faults', 'critical', 'shocks', 'peak_g', 'last_shock_g'):
        assert k in d


def test_road_vibration_and_single_sample_spikes_do_no_damage():
    h = RoverHealth('rover_0', G)
    t = 0.0
    for k in range(500):                   # ±0.3 g vibration
        h.on_imu(t, (G + 0.3 * math.sin(k)) * G0, 2.0)
        t += 0.01
    h.on_imu(t, (G + 4.0) * G0, 0.0)       # one-sample contact glitch
    at_rest(h, t + 0.01, 0.3)
    s = h.state()
    assert s.structure_pct == pytest.approx(100.0) and s.shocks == 0


def test_imu_boot_transient_is_not_an_impact():
    # Gazebo's IMU reads 0 until it initialises, then spikes ~40 g on its first physics step. A real accelerometer
    # at rest always reads gravity, so shock detection arms only once it has read steady gravity for a while.
    h = RoverHealth('rover_0', G)
    t = 0.0
    for _ in range(8):
        h.on_imu(t, 0.0, 0.0)
        t += 0.01
    h.on_imu(t, 39.6 * G0, 0.0)
    h.on_imu(t + 0.01, 0.0, 0.0)
    t = at_rest(h, t + 0.02, 1.0)
    s = h.state()
    assert s.structure_pct == pytest.approx(100.0) and s.shocks == 0 and s.peak_g == 0.0
    strike(h, t, 5.0)                                 # armed now: a real strike still counts
    assert h.state().shocks == 1 and h.state().structure_pct < 70.0


def test_gazebo_jolt_landing_is_not_structural_damage():
    # measured on Gazebo /rover_0/imu_raw during a tip/slip jolt: one side lifts, the chassis is partly airborne
    # (|a| 0.7-1.0 g at 2.2 g gravity for 0.25 s), then one 65 g rigid-contact sample as the wheels land.
    h = RoverHealth('rover_0', G)
    t = at_rest(h, 0.0, 0.6)
    for a in [5.85, 2.06, 2.61] + [2.2] * 20 + [0.77, 0.92, 0.95, 1.23, 1.07, 1.03, 1.05, 1.06, 1.06, 1.05, 1.04,
                                                 1.27, 0.80, 0.97, 0.94, 0.73, 0.87, 0.04, 0.86, 0.76, 0.74, 0.08,
                                                 0.71, 0.69, 0.69, 0.70, 65.27, 1.17, 1.35, 2.25]:
        h.on_imu(t, a * G0, 20.0)
        t += 0.01
    at_rest(h, t, 0.5)
    s = h.state()
    assert s.shocks == 0 and s.structure_pct == pytest.approx(100.0)


def test_separate_single_sample_spikes_are_not_one_sustained_shock():
    # front axle lands, then the rear axle 20 ms later: two one-sample contact spikes with a normal sample between
    h = RoverHealth('rover_0', G)
    t = at_rest(h, 0.0, 0.6)
    for a in [0.8, 0.7, 0.75, 30.0, 2.2, 25.0, 2.2, 2.2, 2.2, 65.27, 1.17, 30.0, 1.35, 2.25]:
        h.on_imu(t, a * G0, 20.0)
        t += 0.01
    at_rest(h, t, 0.5)
    s = h.state()
    assert s.shocks == 0 and s.structure_pct == pytest.approx(100.0) and not s.critical


def test_an_isolated_spike_does_not_set_the_peak_of_a_real_shock():
    h = RoverHealth('rover_0', G)
    t = at_rest(h, 0.0, 0.6)
    for a in [7.2, 7.2, 2.2, 60.0, 2.2, 2.2, 2.2]:          # a 5 g sustained hit, then a lone 60 g contact sample
        h.on_imu(t, a * G0, 0.0)
        t += 0.01
    at_rest(h, t, 0.3)
    s = h.state()
    assert s.shocks == 1 and s.last_shock_g == pytest.approx(5.0, abs=0.01)


def test_gazebo_debris_strike_profile_is_damage():
    # measured on Gazebo /rover_1/imu_raw during the debris strike: a 40 ms plateau ~5.1 g above gravity
    h = RoverHealth('rover_1', G)
    t = at_rest(h, 0.0, 0.6)
    for a in [7.27, 7.26, 7.42, 7.28, 4.10, 2.31, 2.29]:
        h.on_imu(t, a * G0, 0.0)
        t += 0.01
    at_rest(h, t, 0.3)
    s = h.state()
    assert s.shocks == 1 and s.last_shock_g == pytest.approx(5.22, abs=0.01)
    assert s.structure_pct == pytest.approx(100.0 - 12.0 * (5.22 - 1.2), abs=0.5)


def test_debris_strike_damages_structure_in_proportion_to_measured_peak():
    small, big = RoverHealth('a', G), RoverHealth('b', G)
    strike(small, 0.0, 2.0)
    strike(big, 0.0, 5.0)
    assert small.state().shocks == 1 and big.state().shocks == 1
    assert big.state().peak_g == pytest.approx(5.0, abs=0.2)
    assert 85.0 < small.state().structure_pct < 100.0
    assert 40.0 < big.state().structure_pct < 70.0
    assert not big.state().critical               # a strike alone is survivable
    two = RoverHealth('c', G)
    strike(two, strike(two, 0.0, 5.0), 2.0)        # the trace reports the last shock, not the mission's largest
    assert two.state().last_shock_g == pytest.approx(2.0, abs=0.2)
    assert two.state().peak_g == pytest.approx(5.0, abs=0.2)


def test_drivetrain_fault_from_motor_diagnostics_plus_strike_is_critical():
    h = RoverHealth('rover_0', G)
    t = strike(h, 0.0, 5.0)
    h.on_hw_status(t, [0.08, 0.35, 0.05, 0.30], ['DRIVE_FL_OVERCURRENT', 'DRIVE_RL_ENCODER_LOST'], 180.0)
    s = h.state()
    assert s.drive_pct == pytest.approx(19.5, abs=0.1)
    assert s.pct < CRITICAL_PCT and s.critical
    assert 'DRIVE_FL_OVERCURRENT' in s.faults


def test_commanded_vs_measured_wheel_speed_flags_a_stall():
    h = RoverHealth('rover_0', G)
    t = 0.0
    for _ in range(25):                    # 2.5 s commanding 0.8 m/s, wheels turn at 0.1 m/s
        h.on_drive(t, 0.8, 0.1)
        t += 0.1
    assert 'DRIVE_STALL_DETECTED' in h.state().faults
    for _ in range(25):
        h.on_drive(t, 0.8, 0.78)
        t += 0.1
    assert 'DRIVE_STALL_DETECTED' not in h.state().faults


def test_critical_latches_until_repaired_and_repair_restores_structure():
    h = RoverHealth('rover_0', G)
    t = strike(h, 0.0, 5.0)
    h.on_hw_status(t, [0.08, 0.35, 0.05, 0.30], ['DRIVE_FL_OVERCURRENT'], 180.0)
    assert h.state().critical
    h.on_hw_status(t + 1, [0.5, 0.6, 0.5, 0.6], [], 180.0)        # partial: still not recovered
    assert h.state().critical
    h.on_hw_status(t + 2, [1.0, 1.0, 1.0, 1.0], [], 180.0)
    h.repaired(t + 2, 'rover_1')
    s = h.state()
    assert s.pct >= RECOVERED_PCT and not s.critical
    assert s.structure_pct >= 90.0 and s.faults == []


def test_battery_integrates_solar_minus_load():
    h = RoverHealth('rover_0', G, battery_pct=50.0, capacity_wh=600.0)
    h.on_hw_status(0.0, [1.0] * 4, [], 0.0)             # night: no solar
    for k in range(600):                                  # 60 s at 0.8 m/s on the flat, 10 W compute
        h.step(k * 0.1, 0.1, v_ground=0.8, pitch_rad=0.0, compute_w=10.0)
    s = h.state()
    assert s.load_w > 25.0 + 10.0
    assert s.net_w == pytest.approx(s.solar_w - s.load_w)
    used_wh = 300.0 - s.battery_wh
    assert used_wh == pytest.approx(s.load_w * 60.0 / 3600.0, rel=0.05)
    sunny = RoverHealth('rover_1', G, battery_pct=50.0)
    sunny.on_hw_status(0.0, [1.0] * 4, [], 400.0)         # parked in sun: charges
    for k in range(100):
        sunny.step(k * 0.1, 0.1, v_ground=0.0, pitch_rad=0.0, compute_w=5.0)
    assert sunny.state().battery_pct > 50.0


def test_uphill_drive_draws_more_power_than_flat():
    flat, up = RoverHealth('a', G), RoverHealth('b', G)
    for k in range(10):
        flat.step(k * 0.1, 0.1, v_ground=0.8, pitch_rad=0.0, compute_w=0.0)
        up.step(k * 0.1, 0.1, v_ground=0.8, pitch_rad=-math.radians(12.0), compute_w=0.0)   # nose up = -pitch
    assert up.state().load_w > flat.state().load_w * 1.5


def test_over_tilt_adds_a_little_damage():
    h = RoverHealth('rover_0', G, tip_alarm_deg=28.8)
    t = 0.0
    for _ in range(50):                       # 0.5 s at 33 deg
        h.on_imu(t, G * G0, 33.0)
        t += 0.01
    s = h.state()
    assert 95.0 < s.structure_pct < 100.0


def test_drone_battery_drains_with_hover_power_at_high_gravity():
    assert drone_hover_power_w(2.0, 2.2) > drone_hover_power_w(2.0, 1.0) * 2.5
    d = DroneHealth('scout_0', 2.2, battery_pct=90.0)
    for k in range(100):
        d.step(k * 0.1, 0.1, airborne=True, compute_w=8.0)
    s = d.state()
    assert s.battery_pct < 90.0 and s.solar_w == 0.0 and s.structure_pct == 100.0
    assert s.load_w > 100.0
    ground = DroneHealth('scout_0', 2.2, battery_pct=90.0)
    for k in range(100):
        ground.step(k * 0.1, 0.1, airborne=False, compute_w=8.0)
    assert ground.state().load_w < 40.0
