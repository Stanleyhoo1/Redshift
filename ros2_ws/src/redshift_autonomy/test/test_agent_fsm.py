import pytest

from redshift_autonomy.agent_fsm import GYRO_SPIKE, ImuWindow, ReplanGate, RoverFSM, RoverInputs, ScoutFSM, ScoutInputs
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


def reverse_until_timeout(stop_x):
    """Tip at (8, 0) after driving +x, then the reverse makes it only back to (stop_x, 0) before the 15 s timeout."""
    fsm = RoverFSM('rover_0', LIM)
    _, t = drive_forward(fsm, 0.0, 16.0)
    _, t = collect(fsm, t, 3, dt=0.06, pose=(8.0, 0.0, 0.0), roll_deg=31.0, gyro_rate=1.2)
    outs, t = collect(fsm, t, 18, pose=(8.0, 0.0, 0.0), roll_deg=2.0)
    assert outs[-1].state == 'RECOVERY_REVERSE'
    outs, t = collect(fsm, t, 160, pose=(stop_x, 0.0, 0.0))
    out = next(o for o in outs if any(a['type'] == 'hazard_mark' for a in o.actions))
    mark = next(a for a in out.actions if a['type'] == 'hazard_mark')
    tr = next(x for x in out.traces if x['kind'] == 'hazard_marked')
    return mark, tr


@pytest.mark.parametrize('stop_x', [7.6, 6.9, 6.0])
def test_stuck_reverse_never_marks_the_rover_inside_its_own_hazard(stop_x):
    mark, tr = reverse_until_timeout(stop_x)
    (cx, cy), r = mark['xy'], mark['radius_m']
    assert ((stop_x - cx) ** 2 + cy ** 2) ** 0.5 >= r + 0.6 - 1e-6     # it can still plan its way out
    assert r >= 1.0
    assert 'timed out' in tr['input'] and 'Reached safe cell' not in tr['input']
    assert f'{r:.1f} m zone' in tr['action']


def test_partial_reverse_keeps_the_hazard_on_the_failure_point():
    mark, _ = reverse_until_timeout(6.0)                               # 2.0 m back: a 1.4 m zone fits
    assert mark['xy'] == pytest.approx((8.0, 0.0)) and mark['radius_m'] == pytest.approx(1.4)


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


def test_trace_timestamp_is_wall_clock_while_dedup_follows_sim_time():
    import time
    tb = TraceBuilder('rover_0', dedup_s=2.0)
    a = tb.make('k', 'EXPLORING', ['MAP'], 'x', 'y', 0.9, 'z', now=5.0)     # now = simulated seconds
    assert abs(a['t'] - time.time()) < 5.0                                     # t = epoch, for the UI clock
    assert tb.make('k', 'EXPLORING', ['MAP'], 'x', 'y', 0.9, 'z', now=6.9) is None
    assert tb.make('k', 'EXPLORING', ['MAP'], 'x', 'y', 0.9, 'z', now=7.1) is not None


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


# ====================================================================== critical damage, SOS and rescue
OK_HEALTH = {'pct': 96.0, 'structure_pct': 98.0, 'drive_pct': 98.0, 'critical': False, 'faults': [], 'peak_g': 0.4}
BAD_HEALTH = {'pct': 7.5, 'structure_pct': 54.0, 'drive_pct': 19.5, 'critical': True,
              'faults': ['DRIVE_FL_OVERCURRENT'], 'peak_g': 5.1}
SOS = {'id': 'sos-rover_0-1', 'robot': 'rover_0', 'xy': [10.0, 0.0], 'health_pct': 7.5, 'status': 'active'}
LOCK = {'robot': 'rover_0', 'range_m': 1.4, 'conf': 0.93, 'sources': ['CAM', 'DEPTH', 'LIDAR']}
SOS2 = dict(SOS, id='sos-rover_0-2', xy=[21.4, 0.0])


def kinds(outs):
    return [tr['kind'] for tr in all_traces(outs)]


def test_critical_damage_holds_and_broadcasts_sos_then_resumes_when_repaired():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0, health=OK_HEALTH))
    outs, t = collect(fsm, 0.1, 2, health=BAD_HEALTH, rescuer_available=True)
    assert outs[0].state == 'SAFE_HOLD' and outs[0].drive == 'stop'
    tr = [x for x in all_traces(outs) if x['kind'] == 'critical_damage'][0]
    assert tr['severity'] == 'critical' and 'DIAG' in tr['sensors'] and 'IMU' in tr['sensors']
    assert '7' in tr['input'] and 'DRIVE_FL_OVERCURRENT' in tr['input']
    outs, t = collect(fsm, t, 12, health=BAD_HEALTH, rescuer_available=True)
    assert outs[-1].state == 'AWAIT_RESCUE' and outs[-1].drive == 'stop'
    sos = [a for o in outs for a in o.actions if a['type'] == 'sos']
    assert sos and sos[0]['status'] == 'active'
    assert 'sos' in kinds(outs)
    assert 'RECOVERY_REVERSE' not in [o.state for o in outs]          # immobilised: no retreat
    # still critical: keeps waiting, even with good tilt
    outs, t = collect(fsm, t, 30, health=BAD_HEALTH, rescuer_available=True)
    assert outs[-1].state == 'AWAIT_RESCUE'
    # repaired (diagnostics nominal) -> SOS resolved -> RESUMING -> EXPLORING
    outs, t = collect(fsm, t, 3, health=OK_HEALTH, rescuer_available=True)
    states = [o.state for o in outs]
    assert 'RESUMING' in states and states[-1] == 'EXPLORING'
    res = [a for o in outs for a in o.actions if a['type'] == 'sos']
    assert res and res[0]['status'] == 'resolved'
    assert 'repaired' in kinds(outs)


def test_critical_damage_without_a_rescuer_says_so():
    fsm = RoverFSM('rover_0', LIM)
    fsm.step(inp(0.0, health=OK_HEALTH))
    outs, _ = collect(fsm, 0.1, 15, health=BAD_HEALTH, rescuer_available=False)
    sos_tr = [x for x in all_traces(outs) if x['kind'] == 'sos'][0]
    assert 'no recovery' in sos_tr['action'].lower()
    assert outs[-1].state == 'AWAIT_RESCUE'


def test_critical_damage_preempts_tip_recovery():
    fsm = RoverFSM('rover_0', LIM)
    out, t = drive_forward(fsm, 0.0, 10.0)
    collect(fsm, t, 3, dt=0.06, pose=(5.0, 0.0, 0.0), roll_deg=31.0)
    outs, t = collect(fsm, t + 0.2, 20, pose=(5.0, 0.0, 0.0), roll_deg=1.0)
    assert outs[-1].state == 'RECOVERY_REVERSE'
    outs, t = collect(fsm, t, 15, pose=(5.0, 0.0, 0.0), health=BAD_HEALTH)
    assert outs[-1].state == 'AWAIT_RESCUE'


def rescuer(**kw):
    fsm = RoverFSM('rover_1', LIM, role='recovery')
    fsm.step(inp(0.0, pose=(0.0, 0.0, 0.0), health=OK_HEALTH))
    return fsm


def test_scout_role_ignores_sos_but_recovery_rover_starts_rescue():
    scout = RoverFSM('rover_0', LIM)
    scout.step(inp(0.0))
    out = scout.step(inp(0.1, sos=SOS, rescue_range_m=10.0))
    assert out.state == 'EXPLORING'
    fsm = rescuer()
    out = fsm.step(inp(0.1, health=OK_HEALTH, sos=SOS, rescue_range_m=10.0))
    assert out.state == 'RESCUE_TRANSIT' and out.request_replan and out.drive == 'follow'
    tr = [x for x in out.traces if x['kind'] == 'rescue_start'][0]
    assert 'rover_0' in tr['input'] and 'AUTONOMOUS RESCUE MODE' in tr['action'] and 'FLEET' in tr['sensors']
    act = [a for a in out.actions if a['type'] == 'rescue'][0]
    assert act['phase'] == 'TRANSIT' and act['target'] == 'rover_0' and act['id'] == SOS['id']
    assert fsm.rescue['phase'] == 'TRANSIT' and fsm.rescue['peer'] == 'rover_0'


def test_rescue_transit_ignores_target_in_lidar_corridor_and_docks_on_perceived_lock():
    fsm = rescuer()
    fsm.step(inp(0.1, sos=SOS, rescue_range_m=10.0))
    out = fsm.step(inp(0.2, sos=SOS, rescue_range_m=6.0, corridor_obstacle_m=5.0, corridor_conf=0.9))
    assert out.state == 'RESCUE_TRANSIT'
    out = fsm.step(inp(0.25, sos=SOS, rescue_range_m=3.8, corridor_obstacle_m=2.0, corridor_conf=0.9))
    assert out.state == 'RESCUE_TRANSIT' and out.drive == 'follow'       # the corridor return is the target itself
    out = fsm.step(inp(0.3, sos=SOS, rescue_range_m=2.9, corridor_obstacle_m=2.0, corridor_conf=0.9))
    assert out.state == 'RESCUE_TRANSIT'                                  # not at the stand-off yet
    out = fsm.step(inp(0.35, sos=SOS, rescue_range_m=2.9, goal_reached=True))
    assert out.state == 'RESCUE_DOCKING' and out.drive == 'dock'
    assert out.speed_scale <= 0.25
    # lock-on
    out = fsm.step(inp(0.4, sos=SOS, rescue_range_m=2.4, rescue_lock=dict(LOCK, range_m=2.4)))
    assert 'rescue_lock' in [x['kind'] for x in out.traces]
    assert 'PERCEIVED: DAMAGED UNIT ROVER_1' in [x for x in out.traces if x['kind'] == 'rescue_lock'][0]['input']
    assert fsm.rescue['locked']
    # inside 1.5 m with a lock -> REPAIRING + repair action
    out = fsm.step(inp(0.5, sos=SOS, rescue_range_m=1.42, rescue_lock=LOCK))
    assert out.state == 'REPAIRING' and out.drive == 'stop'
    rep = [a for a in out.actions if a['type'] == 'repair']
    assert rep and rep[0]['target'] == 'rover_0'
    assert {'docked', 'repair_start'} <= set(x['kind'] for x in out.traces)


def test_repair_completes_backs_off_and_resumes():
    fsm = rescuer()
    fsm.step(inp(0.1, sos=SOS, rescue_range_m=10.0))
    fsm.step(inp(0.2, sos=SOS, rescue_range_m=2.3))
    fsm.step(inp(0.3, sos=SOS, rescue_range_m=1.4, rescue_lock=LOCK, pose=(8.6, 0.0, 0.0)))
    assert fsm.state == 'REPAIRING'
    # waiting: retries the repair command periodically while unresolved
    outs, t = collect(fsm, 0.4, 100, sos=SOS, rescue_range_m=1.4, rescue_lock=LOCK, pose=(8.6, 0.0, 0.0))
    assert outs[-1].state == 'REPAIRING'
    assert sum(1 for o in outs for a in o.actions if a['type'] == 'repair') >= 1
    # target resolved its SOS
    out = fsm.step(inp(t, sos=None, sos_resolved=True, pose=(8.6, 0.0, 0.0)))
    assert 'repair_complete' in [x['kind'] for x in out.traces]
    assert out.drive == 'reverse' and out.reverse_path
    bx, by = out.reverse_path[-1]
    assert 1.5 <= 8.6 - bx <= 2.5
    out = fsm.step(inp(t + 0.1, pose=(bx, by, 0.0)))
    out2 = fsm.step(inp(t + 0.2, pose=(bx, by, 0.0)))
    assert 'RESUMING' in (out.state, out2.state) and fsm.state in ('RESUMING', 'EXPLORING')
    assert fsm.rescue is None or fsm.rescue['phase'] == 'DONE'


def test_failure_during_back_off_leaves_no_back_off_for_the_next_rescue():
    fsm = rescuer()
    fsm.step(inp(0.1, sos=SOS, rescue_range_m=10.0))
    fsm.step(inp(0.2, sos=SOS, rescue_range_m=2.3))
    fsm.step(inp(0.3, sos=SOS, rescue_range_m=1.4, rescue_lock=LOCK, pose=(8.6, 0.0, 0.0)))
    out = fsm.step(inp(0.4, sos=None, sos_resolved=True, pose=(8.6, 0.0, 0.0)))
    assert out.drive == 'reverse'                                       # backing off after the repair
    outs, t = collect(fsm, 0.5, 3, dt=0.06, pose=(8.4, 0.0, 0.0), roll_deg=31.0, gyro_rate=1.2)
    assert outs[-1].state == 'SAFE_HOLD' and 'rescue_abort' in kinds(outs)
    assert fsm._backoff is None
    # a later rescue that reaches REPAIRING holds for the repair instead of reversing to the stale point
    fsm.state, fsm.rescue = 'EXPLORING', None
    fsm.step(inp(t, sos=SOS2, rescue_range_m=10.0))
    fsm.step(inp(t + 0.1, sos=SOS2, rescue_range_m=2.3))
    out = fsm.step(inp(t + 0.2, sos=SOS2, rescue_range_m=1.4, rescue_lock=LOCK, pose=(20.0, 0.0, 0.0)))
    assert out.state == 'REPAIRING'
    out = fsm.step(inp(t + 0.3, sos=SOS2, rescue_range_m=1.4, rescue_lock=LOCK, pose=(20.0, 0.0, 0.0)))
    assert out.state == 'REPAIRING' and out.drive == 'stop'


def test_rescue_aborts_when_sos_disappears_or_transit_fails():
    fsm = rescuer()
    fsm.step(inp(0.1, sos=SOS, rescue_range_m=10.0))
    out = fsm.step(inp(0.2, sos=None, rescue_range_m=None))
    assert out.state == 'RESUMING' and 'rescue_abort' in [x['kind'] for x in out.traces]
    fsm = rescuer()
    fsm.step(inp(0.1, sos=SOS, rescue_range_m=10.0))
    outs, _ = collect(fsm, 0.2, 80, sos=SOS, rescue_range_m=10.0, has_path=False)
    assert outs[-1].state in ('RESUMING', 'EXPLORING')
    assert 'rescue_abort' in kinds(outs)


def test_critical_rescuer_abandons_rescue():
    fsm = rescuer()
    fsm.step(inp(0.1, sos=SOS, rescue_range_m=10.0))
    outs, _ = collect(fsm, 0.2, 3, sos=SOS, rescue_range_m=9.0, health=BAD_HEALTH)
    assert outs[-1].state == 'SAFE_HOLD'
    assert 'rescue_abort' in kinds(outs)


def test_fleet_roles_and_names():
    from redshift_autonomy.agent_fsm import display_name, role_of, unit_label
    assert [role_of(r) for r in ('rover_0', 'rover_1', 'scout_0')] == ['scout', 'recovery', 'drone']
    assert display_name('rover_0') == 'Scout Rover 1' and display_name('rover_1') == 'Heavy Recovery Rover 2'
    assert display_name('scout_0') == 'Survey Drone'
    assert unit_label('rover_0') == 'ROVER_1' and unit_label('scout_0') == 'DRONE'


def test_imu_window_keeps_a_jolt_that_falls_between_decision_ticks():
    # 100 Hz IMU, 10 Hz decisions: a fast roll overshoot that has already decayed by the next tick
    w = ImuWindow()
    roll = [3, 6, 11, 17, 23, 28, 32, 30, 24, 15]                # deg, one sample per 10 ms
    gyro = [0.3, 2.8, 4.6, 5.4, 5.1, 4.2, 2.6, 1.4, 1.0, 0.8]   # rad/s
    for r, g in zip(roll, gyro):
        w.add(r, 1.0, g)
    r, p, g = w.take()
    assert (r, p) == (32, 1.0)                              # attitude at the window's largest tilt
    assert g == pytest.approx(5.1)                          # the largest rate held for two samples
    assert g > GYRO_SPIKE and max(abs(r), abs(p)) > LIM.max_slope_deg   # the spike rule now sees it
    assert w.take() is None                                 # the next window starts empty


def test_imu_window_ignores_single_sample_gyro_chatter():
    w = ImuWindow()
    for g in (0.2, 6.0, 0.3, 0.2, 7.5, 0.1):                # rigid-contact spikes, one sample each
        w.add(18.0, 0.0, g)
    assert w.take()[2] < GYRO_SPIKE


# --- re-plan request gate ---------------------------------------------------------------------------------------------

def test_replan_gate_serves_the_first_request_at_once():
    g = ReplanGate(1.0)
    assert g.due(True, 'EXPLORING', 10.0)
    assert not g.due(False, 'EXPLORING', 10.1)


def test_replan_request_inside_the_gap_is_held_not_dropped():
    # RESUMING asks for a frontier route, then rescue_start asks again 0.1 s later: the second request must still be
    # served once the gap ends, or the rover keeps driving the frontier route away from its teammate
    g = ReplanGate(1.0)
    assert g.due(True, 'RESUMING', 94.5)
    assert not g.due(True, 'RESCUE_TRANSIT', 94.7)
    assert not g.due(False, 'RESCUE_TRANSIT', 95.2)
    assert g.due(False, 'RESCUE_TRANSIT', 95.6)
    assert not g.due(False, 'RESCUE_TRANSIT', 96.0)      # served once


def test_held_request_lapses_when_the_state_that_asked_has_ended():
    g = ReplanGate(1.0)
    assert g.due(True, 'EXPLORING', 0.0)
    assert not g.due(True, 'REPLANNING', 0.3)
    assert not g.due(False, 'SAFE_HOLD', 1.5)             # SAFE_HOLD did not ask; the REPLANNING request is stale
    assert not g.due(False, 'SAFE_HOLD', 3.0)
