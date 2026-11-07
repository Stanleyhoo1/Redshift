import math

import numpy as np
import pytest

from redshift_sim.mock_dynamics import RoverSim, ScoutSim
from redshift_sim.mock_render import Scene
from redshift_sim.robot_models import ROVER_SPEC, SCOUT_SPEC
from redshift_autonomy.physics_model import RoverPhysics

DT = 0.01


def ramp_scene(deg: float, size=80.0, n=161):
    """Flat for x < 0, then a uniform uphill grade toward +x."""
    xs = np.linspace(-size / 2, size / 2, n)
    h = np.where(xs > 0, xs * math.tan(math.radians(deg)), 0.0)
    return Scene(np.tile(h, (n, 1)).astype(np.float32), size)


def run(sim, scene, seconds, v=0.0, w=0.0, vz=0.0):
    out = []
    for k in range(int(seconds / DT)):
        if isinstance(sim, RoverSim):
            sim.cmd = (v, w)
        else:
            sim.cmd = (v, 0.0, vz, w)
        sim.step(DT, scene, k * DT)
        out.append(sim.snapshot())
    return out


def test_rover_follows_cmd_with_accel_limit_and_sits_at_base_height():
    sc = ramp_scene(0.0)
    r = RoverSim('rover_0', ROVER_SPEC, (-30.0, 0.0, 0.3, 0.0), g_mult=2.2)
    s = run(r, sc, 0.2, v=0.8)
    assert s[-1]['v_wheel'] == pytest.approx(0.3, abs=0.02)          # 1.5 m/s² acceleration limit
    s = run(r, sc, 3.0, v=0.8)
    assert s[-1]['v_true'] == pytest.approx(0.8, abs=0.01)
    assert s[-1]['z'] == pytest.approx(ROVER_SPEC['base_height_m'], abs=0.01)
    assert s[-1]['x'] > -28.0 and abs(s[-1]['y']) < 1e-6


def test_rover_turns_in_place():
    r = RoverSim('rover_0', ROVER_SPEC, (0.0, 0.0, 0.3, 0.0), g_mult=2.2)
    run(r, ramp_scene(0.0), 2.0, w=0.5)
    assert r.yaw == pytest.approx(1.0, abs=0.05)


def test_torque_limit_stalls_on_a_grade_above_the_drive_limit_only_at_high_g():
    lim22 = RoverPhysics().limits(2.2)
    steep = lim22.drive_deg + 3.0                                      # ≈ 21.7°: too steep at 2.2 g, fine at 1 g
    for g, should_climb in ((2.2, False), (1.0, True)):
        r = RoverSim('rover_0', ROVER_SPEC, (2.0, 0.0, 0.3, 0.0), g_mult=g)
        sc = ramp_scene(steep)
        run(r, sc, 6.0, v=0.6)
        climbed = r.x - 2.0
        assert (climbed > 2.0) == should_climb, f'g={g} climbed {climbed:.2f} m'
    assert r.pitch == pytest.approx(-math.radians(steep), abs=0.03)    # nose up on the ramp


def test_rock_blocks_motion_while_wheels_keep_turning():
    sc = ramp_scene(0.0)
    sc.add_rocks([(3.0, 0.0, 0.8)], t=-10.0)
    r = RoverSim('rover_0', ROVER_SPEC, (0.0, 0.0, 0.3, 0.0), g_mult=2.2)
    s = run(r, sc, 6.0, v=0.6)
    assert r.x < 3.0 - 0.8 - 0.3
    assert s[-1]['v_wheel'] > 0.5 and abs(s[-1]['v_true']) < 0.05


def test_jolt_peaks_past_tip_alarm_for_over_100ms_then_self_rights():
    lim = RoverPhysics().limits(2.2)
    r = RoverSim('rover_0', ROVER_SPEC, (0.0, 0.0, 0.3, 0.0), g_mult=2.2)
    sc = ramp_scene(0.0)
    run(r, sc, 0.5)
    r.start_jolt()
    s = run(r, sc, 4.0)
    roll = np.degrees([abs(q['roll']) for q in s])
    above = (roll > lim.tip_alarm_deg).sum() * DT
    assert roll.max() == pytest.approx(lim.tip_alarm_deg + 5.0, abs=2.5)
    assert above > 0.12
    assert roll[-50:].max() < 3.0                                      # back on its wheels
    rel = r.pop_events()
    assert len(rel) == 1 and rel[0]['op'] == 'jolt' and rel[0]['release_roll_deg'] > 15.0
    gyro = max(abs(q['gyro'][0]) for q in s)
    assert gyro > 1.0                                                  # the IMU really measures the jolt


def test_imu_specific_force_scales_with_gravity():
    r = RoverSim('rover_0', ROVER_SPEC, (0.0, 0.0, 0.3, 0.0), g_mult=2.2)
    s = run(r, ramp_scene(0.0), 0.2)
    assert np.linalg.norm(s[-1]['accel']) == pytest.approx(2.2 * 9.80665, rel=0.02)


def test_scout_takes_off_holds_altitude_and_never_sinks_into_the_ground():
    sc = ramp_scene(0.0)
    s = ScoutSim('scout_0', SCOUT_SPEC, (0.0, 0.0, 0.12, 0.0), g_mult=2.2)
    run(s, sc, 1.0)
    assert s.z == pytest.approx(0.05, abs=0.02)                        # settles on its skids
    run(s, sc, 1.0, vz=-1.0)
    assert s.z == pytest.approx(0.05, abs=0.02)
    run(s, sc, 5.0, vz=2.0)
    z_up = s.z
    assert z_up > 8.0
    run(s, sc, 3.0)
    assert s.z == pytest.approx(z_up, abs=0.6)                         # first-order lag, then hold
    run(s, sc, 2.0, v=3.0)
    assert s.x > 4.0 and s.vel_world()[0] == pytest.approx(3.0, abs=0.2)
