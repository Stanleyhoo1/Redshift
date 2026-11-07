import pytest

from redshift_autonomy.agent_fsm import RoverFSM, RoverInputs, ScoutFSM, ScoutInputs
from redshift_autonomy.physics_model import RoverPhysics
from redshift_autonomy.trace import TraceBuilder

LIM = RoverPhysics().limits(2.2)
TRACE_KEYS = {'id', 't', 'robot', 'state', 'kind', 'sensors', 'input', 'reasoning', 'confidence',
              'action', 'severity'}


def inp(now, **kw):
    base = dict(now=now, pose=(0.0, 0.0, 0.0), roll_deg=0.0, pitch_deg=0.0, gyro_rate=0.0, slip=0.0,
                uncertainty=0.1, camera_conf=0.95, lidar_conf=0.95, corridor_obstacle_m=float('inf'),
                corridor_conf=0.9, slope_ahead_deg=0.0, path_blocked=False, blocked_reason='',
                has_path=True, goal_reached=False, frontiers_left=True, percent_known=10.0,
                scan_complete=None, scout_available=True, on_hazard_cell=False)
    base.update(kw)
    return RoverInputs(**base)


def drive_forward(fsm, t0, t1, dt=0.1, y=0.0):
    """Explore along +x leaving breadcrumbs; returns last output."""
    t, out = t0, None
    while t < t1:
        out = fsm.step(inp(t, pose=(t * 0.5, y, 0.0)))
        t += dt
    return out, t


def collect(fsm, t, n, dt=0.1, **kw):
    outs = []
    for _ in range(n):
        outs.append(fsm.step(inp(t, **kw)))
        t += dt
    return outs, t


def all_traces(outs):
    return [tr for o in outs for tr in o.traces]


def test_boot_to_exploring_with_trace():
    fsm = RoverFSM('rover_0', LIM)
    out = fsm.step(inp(0.0))
    assert out.state == 'EXPLORING'
    assert out.traces and TRACE_KEYS <= set(out.traces[0])


def test_full_tip_recovery_chain():
    fsm = RoverFSM('rover_0', LIM)
    out, t = drive_forward(fsm, 0.0, 16.0)          # ~8 m of breadcrumbs along +x
    assert out.state == 'EXPLORING'
    assert fsm.breadcrumb_count() >= 10
    fail_xy = (8.0, 0.0)
    # 31 deg roll for 120 ms (tip alarm @2.2 g ~28.8 deg)
    outs, t = collect(fsm, t, 3, dt=0.06, pose=(*fail_xy, 0.0), roll_deg=31.0, gyro_rate=1.2)
    assert outs[-1].state == 'SAFE_HOLD'
    assert outs[-1].drive == 'stop'
    crit = [tr for tr in all_traces(outs) if tr['kind'] == 'safe_hold']
    assert crit and crit[0]['severity'] == 'critical' and 'IMU' in crit[0]['sensors']
    # settles, then reverses after the 1.5 s hold
    outs, t = collect(fsm, t, 10, pose=(*fail_xy, 0.0), roll_deg=3.0)
    assert outs[-1].state == 'SAFE_HOLD'
    outs, t = collect(fsm, t, 8, pose=(*fail_xy, 0.0), roll_deg=2.0)
    assert outs[-1].state == 'RECOVERY_REVERSE'
    assert outs[-1].drive == 'reverse'
    target = outs[-1].reverse_path[-1]
    assert 3.0 <= fail_xy[0] - target[0] <= 5.0
    # arrive at the safe cell -> hazard mark + scout tasking -> AWAIT_SCOUT
    out = fsm.step(inp(t, pose=(target[0], target[1], 0.0)))
    t += 0.1
    types = [a['type'] for a in out.actions]
    assert 'hazard_mark' in types and 'scan_request' in types
    mark = next(a for a in out.actions if a['type'] == 'hazard_mark')
    assert mark['xy'] == pytest.approx(fail_xy) and mark['radius_m'] == pytest.approx(2.5)
    req = next(a for a in out.actions if a['type'] == 'scan_request')
    assert out.state == 'AWAIT_SCOUT' and out.drive == 'stop'
    # scout reports back -> RESUMING (replan) -> EXPLORING
    out = fsm.step(inp(t, pose=(target[0], target[1], 0.0),
                       scan_complete={'id': req['id'], 'max_slope_deg': 24.0, 'verdict': 'confirmed'}))
    assert out.state == 'RESUMING' and out.request_replan
    out = fsm.step(inp(t + 0.1, pose=(target[0], target[1], 0.0)))
    assert out.state == 'EXPLORING'


def test_no_scout_skips_waiting():
    fsm = RoverFSM('rover_0', LIM)
    out, t = drive_forward(fsm, 0.0, 12.0)
    collect(fsm, t, 3, dt=0.06, pose=(6.0, 0.0, 0.0), roll_deg=35.0)
    outs, t = collect(fsm, t + 0.2, 20, pose=(6.0, 0.0, 0.0), roll_deg=1.0, scout_available=False)
    target = outs[-1].reverse_path[-1]
    out = fsm.step(inp(t, pose=(target[0], target[1], 0.0), scout_available=False))
    assert 'scan_request' not in [a['type'] for a in out.actions]
    assert out.state == 'RESUMING'


def test_short_tilt_spike_is_ignored():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0))
    out = fsm.step(inp(0.1, roll_deg=31.0))
    out = fsm.step(inp(0.15, roll_deg=5.0))
    assert out.state == 'EXPLORING'


def test_sustained_slip_triggers_safe_hold():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0))
    outs, _ = collect(fsm, 0.1, 10, slip=0.75)
    assert outs[-1].state == 'SAFE_HOLD'
    assert 'ODOM' in all_traces(outs)[-1]['sensors']


def test_caution_hysteresis():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0))
    outs, t = collect(fsm, 0.1, 3, uncertainty=0.55, camera_conf=0.3)
    assert outs[-1].state == 'EXPLORING'                    # needs 0.5 s
    outs, t = collect(fsm, t, 5, uncertainty=0.55, camera_conf=0.3)
    assert outs[-1].state == 'CAUTION'
    assert outs[-1].mode == 'LIDAR_ONLY' and outs[-1].speed_scale == pytest.approx(0.35)
    outs, t = collect(fsm, t, 10, uncertainty=0.30, camera_conf=0.8)   # 1 s of good data: not enough
    assert outs[-1].state == 'CAUTION'
    outs, t = collect(fsm, t, 15, uncertainty=0.20, camera_conf=0.9)
    assert outs[-1].state == 'EXPLORING' and outs[-1].mode == 'FUSED'
    kinds = [tr['kind'] for tr in all_traces(outs)]
    assert 'caution_exit' in kinds


def test_extreme_uncertainty_triggers_safe_hold():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0))
    outs, _ = collect(fsm, 0.1, 20, uncertainty=0.9, camera_conf=0.05, lidar_conf=0.1)
    assert outs[-1].state == 'SAFE_HOLD'


def test_blocked_path_triggers_replanning_and_returns():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0))
    out = fsm.step(inp(0.1, path_blocked=True, blocked_reason='Boulder field 6.8 m ahead'))
    assert out.state == 'REPLANNING' and out.request_replan and out.drive == 'stop'
    tr = [x for x in out.traces if x['kind'] == 'path_blocked'][0]
    assert 'Boulder' in tr['input']
    out = fsm.step(inp(0.2, has_path=True, path_blocked=False))
    assert out.state == 'EXPLORING'


def test_corridor_obstacle_triggers_replanning():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0))
    out = fsm.step(inp(0.1, corridor_obstacle_m=1.8, corridor_conf=0.9))
    assert out.state == 'REPLANNING'


def test_goal_reached_requests_new_goal_and_completion():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0))
    out = fsm.step(inp(0.1, goal_reached=True))
    assert 'new_goal' in [a['type'] for a in out.actions]
    out = fsm.step(inp(0.2, goal_reached=True, frontiers_left=False, has_path=False))
    assert out.state == 'MISSION_COMPLETE'


def test_trace_dedup():
    tb = TraceBuilder('rover_0', dedup_s=2.0)
    a = tb.make('slope_hazard', 'EXPLORING', ['LIDAR'], 'x', 'y', 0.9, 'z', now=0.0)
    b = tb.make('slope_hazard', 'EXPLORING', ['LIDAR'], 'x2', 'y', 0.9, 'z', now=1.0)
    c = tb.make('slope_hazard', 'EXPLORING', ['LIDAR'], 'x3', 'y', 0.9, 'z', now=2.5)
    d = tb.make('slope_hazard', 'EXPLORING', ['LIDAR'], 'x4', 'y', 0.9, 'z', now=2.6, force=True)
    assert a and b is None and c and d
    assert c['id'] > a['id'] and TRACE_KEYS <= set(a)


def scout_inp(now, **kw):
    base = dict(now=now, pose=(0.0, 0.0, 12.0), altitude_agl=12.0, goal_reached=False, scan_request=None,
                scan_stats=None, frontiers_left=True)
    base.update(kw)
    return ScoutInputs(**base)


def test_scout_takeoff_survey_scan_cycle():
    s = ScoutFSM('scout_0', cruise_agl=12.0)
    out = s.step(scout_inp(0.0, altitude_agl=0.3))
    assert out.state == 'TAKEOFF'
    out = s.step(scout_inp(1.0, altitude_agl=11.8))
    assert out.state == 'SURVEY'
    req = {'id': 'r1', 'robot': 'rover_0', 'xy': [8.0, 0.0], 'reason': 'tip'}
    out = s.step(scout_inp(2.0, scan_request=req))
    assert out.state == 'FLY_TO_SCAN' and out.target_xy == pytest.approx((8.0, 0.0))
    out = s.step(scout_inp(3.0, pose=(8.0, 0.1, 12.0)))
    assert out.state == 'SCANNING'
    out = s.step(scout_inp(5.0, pose=(8.0, 0.1, 12.0)))
    assert out.state == 'SCANNING' and out.orbit
    out = s.step(scout_inp(7.5, pose=(8.0, 0.1, 12.0), scan_stats={'max_slope_deg': 22.0, 'mean_slope_deg': 9.0}))
    done = [a for a in out.actions if a['type'] == 'scan_complete']
    assert done and done[0]['id'] == 'r1' and done[0]['max_slope_deg'] == 22.0
    assert out.state == 'SURVEY'
