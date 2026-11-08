"""Unit tests for brain_bridge: trace → event mapping, context building, message formatting and the planners'
advisory cap / altitude bookkeeping (no ROS, no network)."""
import json
import math
import time

import pytest

from redshift_autonomy import brain_bridge as bb
from redshift_autonomy.brain_bridge import (BrainAltitude, BrainCap, EventGate, EventSpec, KnownHazards, LatestJson,
                                            advice_trace, build_context, cap_trace, decision_message, effective_limit,
                                            event_for_trace, parse_decision, parse_operator_command)
from redshift_autonomy.gemma_agent_brain import (ALTITUDE, CLARIFY, DEPLOY, REJECT, SLOW, AgentBrain, GemmaClient,
                                                 assess_operator_command, deterministic_fallback, validate)
from redshift_autonomy.physics_model import RoverPhysics
from redshift_autonomy.trace import TraceBuilder

ROBOTS = ['rover_0', 'rover_1', 'scout_0']
TRACE_KEYS = {'id', 't', 'robot', 'state', 'kind', 'sensors', 'input', 'reasoning', 'confidence', 'action',
              'severity', 'latency_ms'}


def trace(kind, robot='rover_0', **kw):
    t = TraceBuilder(robot).make(kind, 'EXPLORING', ['LIDAR'], kw.pop('input', f'{kind} input'), 'why', 0.9,
                                 'act', force=True)
    t.update(kw)
    return t


def state(robot='rover_0', st='EXPLORING', v_limit=0.8, **kw):
    d = {'robot': robot, 'type': 'rover' if robot.startswith('rover') else 'scout', 'state': st, 'speed': 0.5,
         'v_limit': v_limit, 'altitude_agl': None,
         'health': {'pct': 88.0, 'battery_pct': 76.0, 'critical': False, 'faults': []}}
    d.update(kw)
    return d


def decision(**kw):
    d = {'action': SLOW, 'target_speed_mps': 0.3, 'target_altitude_m': None, 'reasoning_trace': 'r',
         'source': 'gemma', 'model': 'gemma-4-26b-a4b-it', 'event': 'novel_hazard', 'robot': 'rover_0',
         'target_robot': None, 'stale': False, 'command_id': None, 'command_seq': None, 'immediate': False}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------------------------------- event mapping
@pytest.mark.parametrize('kind', ['path_blocked', 'slope_hazard', 'hazard_marked'])
def test_hazard_traces_are_novel_hazards(kind):
    spec = event_for_trace(trace(kind, input='Boulder field 6.8 m ahead'), ROBOTS, 'rover_1')
    assert spec == EventSpec('novel_hazard', 'rover_0', kind, 'Boulder field 6.8 m ahead',
                             hazard='Boulder field 6.8 m ahead')


def test_caution_enter_is_low_confidence_and_critical_damage_targets_the_damaged_rover():
    assert event_for_trace(trace('caution_enter'), ROBOTS).event == 'low_confidence'
    spec = event_for_trace(trace('critical_damage', robot='rover_1', input='Health 7%'), ROBOTS, 'rover_1')
    assert (spec.event, spec.robot, spec.damage) == ('critical_damage', 'rover_1', 'Health 7%')


def test_sos_asks_the_recovery_rover_about_its_peer():
    spec = event_for_trace(trace('sos', input='Immobilised at (1.0, 2.0)'), ROBOTS, 'rover_1')
    assert (spec.event, spec.robot, spec.peer, spec.trigger) == ('peer_sos', 'rover_1', 'rover_0', 'sos')
    # no recovery rover, the recovery rover itself in trouble, or a resolved SOS → nothing to ask
    assert event_for_trace(trace('sos'), ROBOTS, None) is None
    assert event_for_trace(trace('sos', robot='rover_1'), ROBOTS, 'rover_1') is None
    assert event_for_trace(trace('sos', status='resolved'), ROBOTS, 'rover_1') is None


def test_brain_traces_unknown_robots_and_other_kinds_are_ignored():
    assert event_for_trace(trace('path_blocked', robot='rover_7'), ROBOTS) is None
    assert event_for_trace(trace('path_blocked', brain={'source': 'gemma'}), ROBOTS) is None
    assert event_for_trace(trace('brain_advice'), ROBOTS) is None
    assert event_for_trace(trace('brain_cap'), ROBOTS) is None
    assert event_for_trace(trace('frontier_selected'), ROBOTS) is None
    for bad in (None, [], 'x', {'kind': 'path_blocked'}, {'robot': 'rover_0', 'kind': 3}):
        assert event_for_trace(bad, ROBOTS) is None


def test_operator_command_parsing():
    msg = json.dumps({'id': 'cmd-1', 'text': '  slow down  ', 'robot': None, 't': 1.0})
    assert parse_operator_command(msg, ROBOTS) == {'id': 'cmd-1', 'text': 'slow down', 'robot': 'rover_0', 't': 1.0}
    scout = json.dumps({'id': 'c', 'text': 'x', 'robot': 'scout_0'})
    assert parse_operator_command(scout, ROBOTS)['robot'] == 'scout_0'
    assert parse_operator_command(json.dumps({'id': 'c', 'text': 'x'}), ['rover_3'])['robot'] == 'rover_3'
    for bad in ('{', '[]', json.dumps({'id': 'c', 'text': ''}), json.dumps({'id': 'c', 'text': 'x' * 201}),
                json.dumps({'id': '', 'text': 'x'}), json.dumps({'id': 'c', 'text': 'x', 'robot': 'rover_9'}),
                json.dumps({'id': 'c', 'text': 7}), json.dumps({'id': 5, 'text': 'x'})):
        assert parse_operator_command(bad, ROBOTS) is None


def test_known_hazards_from_world_events_and_marks():
    kh = KnownHazards(limit=2)
    kh.add(bb.hazard_name({'op': 'spawn_rockslide', 'ok': True}))
    kh.add(bb.hazard_name({'op': 'spawn_rockslide', 'ok': False}))
    kh.add(bb.hazard_name({'reason': 'tip'}, 'hazard_mark'))
    kh.add(bb.hazard_name({'op': 'spawn_rockslide', 'ok': True}))
    assert kh.names() == ['tip hazard zone', 'rockslide']
    kh.add(bb.hazard_name({'reason': 'slip'}, 'hazard_mark'))
    assert kh.names() == ['rockslide', 'slip hazard zone']


# ---------------------------------------------------------------------------------------------------- context
@pytest.mark.parametrize('st', ['SAFE_HOLD', 'AWAIT_RESCUE', 'REPAIRING', 'AWAIT_SCOUT', 'RECOVERY_REVERSE',
                                'MISSION_COMPLETE', 'BOOT'])
def test_effective_limit_is_zero_while_holding(st):
    assert effective_limit(state(st=st)) == 0.0


def test_effective_limit_follows_v_limit_when_driving():
    # REPLANNING is a momentary stop to re-route; its v_limit is the drive limit the rover resumes at
    for st in ('EXPLORING', 'CAUTION', 'REPLANNING', 'RESCUE_TRANSIT', 'RESCUE_DOCKING'):
        assert effective_limit(state(st=st, v_limit=0.42)) == pytest.approx(0.42)
    assert effective_limit(None) == 0.0
    assert effective_limit(state(v_limit=None)) == 0.0
    assert effective_limit(state(health={'critical': True})) == 0.0


def test_build_context_fills_the_documented_keys():
    states = {'rover_0': state(), 'rover_1': state('rover_1'),
              'scout_0': state('scout_0', 'SURVEY', 4.0, altitude_agl=11.6)}
    perception = {'uncertainty': {'score': 0.5, 'camera_conf': 0.3, 'lidar_conf': 0.9},
                  'corridor': {'obstacle_dist_m': 3.2}}
    ctx = build_context('rover_0', states=states, edge={'uncertainty': 0.1}, perception=perception,
                        env={'dust_level': 0.8}, recovery_rover='rover_1', drone='scout_0', v_max_mps=0.94,
                        hazard='Boulder', known_hazards=['rockslide'], operator_text='go', edge_profile='orin_nx_25w')
    assert ctx['speed_limit_mps'] == 0.8 and ctx['v_max_mps'] == 0.94 and ctx['speed_mps'] == 0.5
    assert (ctx['camera_conf'], ctx['lidar_conf'], ctx['uncertainty']) == (0.3, 0.9, 0.5)
    assert ctx['nearest_obstacle_m'] == 3.2 and ctx['dust'] is True and ctx['hazard'] == 'Boulder'
    assert ctx['health'] == pytest.approx(0.88) and ctx['battery'] == pytest.approx(0.76)
    assert ctx['known_hazards'] == ['rockslide'] and ctx['operator_text'] == 'go'
    assert ctx['edge_profile'] == 'orin_nx_25w' and ctx['robot_type'] == 'rover'
    assert ctx['fleet'] == {'recovery_rover': 'rover_1', 'recovery_available': True, 'drone': 'scout_0',
                            'drone_agl_m': 11.6}
    # the module reads its own keys out of it
    d = deterministic_fallback('novel_hazard', ctx)
    assert d.action == SLOW and d.target_speed_mps == pytest.approx(0.28)


def test_build_context_falls_back_and_marks_recovery_unavailable():
    states = {'rover_0': state(st='SAFE_HOLD'), 'rover_1': state('rover_1', 'AWAIT_RESCUE'), 'scout_0': None}
    ctx = build_context('rover_0', states=states, edge={'uncertainty': 0.6}, perception=None,
                        recovery_rover='rover_1', drone='scout_0')
    assert ctx['speed_limit_mps'] == 0.0 and ctx['uncertainty'] == 0.6 and 'camera_conf' not in ctx
    assert ctx['fleet']['recovery_available'] is False and 'drone_agl_m' not in ctx['fleet']
    assert ctx['dust'] is False
    # no state at all → a stopped robot of the right type; a critical recovery rover is unavailable too
    ctx = build_context('scout_0', states={}, recovery_rover='rover_1')
    assert ctx['speed_limit_mps'] == 0.0 and ctx['robot_type'] == 'scout' and ctx['fleet']['recovery_available']
    crit = {'rover_1': state('rover_1', health={'critical': True})}
    assert build_context('rover_0', states=crit, recovery_rover='rover_1')['fleet']['recovery_available'] is False
    assert build_context('rover_0', states={})['fleet']['recovery_available'] is False


def test_damage_defaults_to_the_robots_fault_codes():
    st = state(health={'pct': 7.0, 'faults': ['DRIVE_FL_OVERCURRENT', 'DRIVE_RR_STALL'], 'critical': True})
    ctx = build_context('rover_0', states={'rover_0': st})
    assert ctx['damage'] == 'DRIVE_FL_OVERCURRENT, DRIVE_RR_STALL'
    assert build_context('rover_0', states={'rover_0': st}, damage='strike')['damage'] == 'strike'


def test_latest_json_parses_lazily_and_expires():
    lj = LatestJson()
    lj.put('a', '{"x": 1}', 10.0)
    assert lj.get('a', 12.0, 5.0) == {'x': 1}
    assert lj.get('a', 16.0, 5.0) is None and lj.get('b', 0.0, 5.0) is None
    lj.put('a', 'not json', 20.0)
    assert lj.get('a', 20.0, 5.0) is None


def test_event_gate_reoffers_the_newest_deferred_event_after_its_interval():
    g = EventGate({'novel_hazard': 10.0})
    a = EventSpec('novel_hazard', 'rover_0', 'slope_hazard', 'slope')
    b = EventSpec('novel_hazard', 'rover_0', 'path_blocked', 'rockslide')
    g.submitted(a, 100.0)
    g.defer(a, 103.0)
    g.defer(b, 105.0)              # the rockslide outranks the slope warning
    assert g.due(109.0) == [] and g.pending() == 1
    assert g.due(110.1) == [b] and g.pending() == 0
    g.defer(EventSpec('peer_sos', 'rover_1', 'sos'), 0.0)     # never submitted: due after its own interval
    assert g.due(1.0) == [] and len(g.due(3.1)) == 1


# ---------------------------------------------------------------------------------------------------- outputs
def test_decision_message_carries_the_contract_fields_and_is_strict_json():
    ctx = build_context('rover_0', states={'rover_0': state()}, v_max_mps=0.94, operator_text='floor it')
    local = assess_operator_command('floor it', ctx)
    assert local['requested_speed_mps'] == pytest.approx(0.94)
    local = assess_operator_command('go faster', ctx)
    assert math.isinf(local['requested_speed_mps'])
    d = deterministic_fallback('operator_command', dict(ctx, operator_text='go faster'))
    d.robot, d.request_id = 'rover_0', 'rover_0:operator_command:1'
    msg = decision_message(d.as_dict(), trigger='operator', trigger_input='go faster', command_id='cmd-3',
                           command_text='go faster', local=local, command_seq=3, t=1762550000.1234)
    for k in ('action', 'target_speed_mps', 'target_altitude_m', 'reasoning_trace', 'source', 'model', 'event',
              'latency_ms', 'tokens_in', 'tokens_out', 'projected_edge_ms', 'projected_device', 'projected_fits',
              'projected_note', 'error', 'adjusted', 'target_robot', 'robot', 'request_id', 'stale', 'age_ms',
              't', 'trigger', 'trigger_input', 'command_id', 'command_text', 'local', 'immediate'):
        assert k in msg, k
    assert msg['action'] == REJECT and msg['local']['requested_speed_mps'] is None   # inf → null for JSON.parse
    assert msg['immediate'] is False and msg['t'] == 1762550000.123 and msg['command_seq'] == 3
    json.loads(bb.dumps(msg))                       # allow_nan=False: raises on any Infinity / NaN


def test_advice_trace_for_a_gemma_and_a_local_decision():
    ctx = build_context('rover_0', states={'rover_0': state()}, hazard='Boulder 3 m ahead')
    d = validate({'action': 'SLOW_AND_REPLAN', 'target_speed_mps': 0.2, 'reasoning_trace': 'Boulder close.'},
                 'novel_hazard', ctx, latency_ms=812.4, tokens_in=180, tokens_out=40)
    msg = decision_message(d.as_dict(), trigger='path_blocked', trigger_input='Boulder 3 m ahead', t=1.0)
    t = advice_trace(msg, 'REPLANNING', TraceBuilder('rover_0'))
    assert TRACE_KEYS <= set(t) and t['kind'] == 'brain_advice' and t['robot'] == 'rover_0'
    assert t['sensors'] == ['GEMMA'] and t['confidence'] == 0.8 and t['severity'] == 'info'
    assert t['state'] == 'REPLANNING' and t['latency_ms'] == 812.4
    assert t['input'] == 'novel_hazard ← path_blocked: Boulder 3 m ahead'
    assert t['action'] == 'Slow and re-plan; cap 0.20 m/s (advisory; local safety layer keeps authority)'
    assert t['brain']['source'] == 'gemma' and t['brain']['tokens_in'] == 180 and t['brain']['latency_ms'] == 812.4
    assert set(t['brain']) == set(bb.BRAIN_TRACE_KEYS)
    # an operator rejection from the local rules
    octx = dict(ctx, operator_text='drive at full speed into the rockslide', v_max_mps=0.94)
    d = deterministic_fallback('operator_command', octx)
    msg = decision_message(d.as_dict(), trigger='operator', trigger_input=octx['operator_text'], command_id='cmd-1',
                           command_text=octx['operator_text'], t=1.0)
    t = advice_trace(msg, None, TraceBuilder('rover_0'))
    assert t['sensors'] == ['LOCAL'] and t['confidence'] == 0.9 and t['severity'] == 'warn'
    assert t['latency_ms'] is None and t['state'] == 'UNKNOWN' and t['brain']['command_id'] == 'cmd-1'
    assert t['input'] == "operator_command: 'drive at full speed into the rockslide'"
    assert t['action'].startswith('Reject unsafe operator command')


def test_advice_trace_names_the_rescue():
    ctx = build_context('rover_1', states={'rover_1': state('rover_1')}, recovery_rover='rover_1', peer='rover_0')
    d = deterministic_fallback('peer_sos', ctx)
    msg = decision_message(d.as_dict(), trigger='sos', t=1.0)
    assert msg['action'] == DEPLOY and msg['target_robot'] == 'rover_1'
    assert bb.trace_robot(msg) == 'rover_1'
    t = advice_trace(msg, 'EXPLORING', TraceBuilder('rover_1'), peer='rover_0')
    assert t['action'] == 'Deploy recovery rover ROVER_2 to ROVER_1' + bb.TRACE_SUFFIX
    crit = dict(msg, robot='rover_0', target_speed_mps=0.0)          # critical_damage: the damaged rover holds anyway
    assert bb.action_text(crit) == 'Deploy recovery rover ROVER_2 to ROVER_1' + bb.TRACE_SUFFIX
    rejected = bb.action_text(dict(msg, action=REJECT, target_speed_mps=0.5))
    assert rejected == 'Reject unsafe operator command; keep the local plan' + bb.TRACE_SUFFIX
    assert '; hold at 0 m/s' in bb.action_text(dict(msg, action=REJECT, target_speed_mps=0.0))
    assert bb.trace_robot({'robot': None, 'target_robot': 'scout_0'}) == 'scout_0'
    assert bb.trace_robot({}) == 'fleet'


def test_parse_decision_rejects_garbage():
    assert parse_decision(json.dumps(decision()))['action'] == SLOW
    for bad in ('', '{', '[1]', json.dumps(decision(action='FLOOR_IT')), json.dumps(decision(robot=3)),
                json.dumps(decision(event=['x']))):
        assert parse_decision(bad) is None


# ---------------------------------------------------------------------------------------------------- BrainCap
def test_cap_applies_only_to_its_robot_and_never_raises_speed():
    c = BrainCap('rover_0')
    assert c.offer(decision(robot='rover_1'), 0.0) is None
    cap = c.offer(decision(target_speed_mps=0.3), 0.0)
    assert cap.speed_mps == 0.3 and cap.duration_s == bb.EVENT_CAP_S and not cap.operator
    assert c.apply(0.8, 0.4, 1.0) == pytest.approx((0.3, 0.15))     # curvature kept
    assert c.apply(0.1, 0.4, 1.0) == (0.1, 0.4)                    # already slower: untouched
    assert c.apply(0.0, 0.6, 1.0) == (0.0, 0.6)                    # turning on the spot is not speed
    assert c.info(5.0) == {'speed_mps': 0.3, 'action': SLOW, 'source': 'gemma', 'operator': False,
                           'remaining_s': 10.0, 'applied': False}
    assert c.info(5.0, 'EXPLORING', applied=True)['applied'] is True
    assert c.limit(15.0) is None and c.info(15.0) is None and c.apply(0.8, 0.4, 15.0) == (0.8, 0.4)


def test_cap_expiry_by_kind():
    c = BrainCap('rover_0')
    hold = c.offer(decision(action=CLARIFY, target_speed_mps=0.0, event='low_confidence'), 0.0)
    assert hold.duration_s == bb.EVENT_HOLD_S and c.apply(0.5, 0.3, 3.9) == (0.0, 0.0)
    assert c.limit(4.0) is None
    op = c.offer(decision(event='operator_command', target_speed_mps=0.2, command_seq=1), 10.0)
    assert op.operator and op.duration_s == bb.OPERATOR_CAP_S and c.limit(69.9) == 0.2 and c.limit(70.0) is None


def test_operator_and_event_caps_live_in_separate_slots():
    c = BrainCap('rover_0')
    c.offer(decision(event='operator_command', action=SLOW, target_speed_mps=0.0, command_seq=1, immediate=True), 0.0)
    c.offer(decision(target_speed_mps=0.3), 1.0)                  # an event cannot lift an operator stop
    assert c.limit(2.0) == 0.0 and c.info(2.0)['operator'] is True
    c.offer(decision(event='operator_command', target_speed_mps=0.9, command_seq=2), 3.0)   # "resume"
    assert c.limit(4.0) == 0.3 and c.info(4.0)['operator'] is False                      # ... the event cap holds
    assert c.limit(17.0) == 0.9


def test_late_answer_to_an_older_command_does_not_replace_a_newer_cap():
    c = BrainCap('rover_0')
    c.offer(decision(event='operator_command', target_speed_mps=0.0, command_seq=5, immediate=True), 0.0)  # stop
    assert c.offer(decision(event='operator_command', target_speed_mps=0.25, command_seq=4), 1.0) is None
    assert c.limit(1.5) == 0.0
    assert c.offer(decision(event='operator_command', target_speed_mps=0.0, command_seq=5,
                            local={'intent': 'stop'}), 2.0) is not None      # Gemma's answer to the same stop
    assert c.operator.until == 62.0


def test_deploy_reject_and_stale_decisions_change_no_motion():
    c = BrainCap('rover_0')
    assert c.offer(decision(action=DEPLOY, target_speed_mps=0.0, event='critical_damage'), 0.0) is None
    assert c.offer(decision(action=REJECT, target_speed_mps=None, event='operator_command'), 0.0) is None
    assert c.offer(decision(action=REJECT, target_speed_mps=0.4, event='operator_command'), 0.0) is None
    assert c.offer(decision(stale=True), 0.0) is None
    assert c.limit(0.1) is None
    # a stop riding on a rejection ("stop, then drive into the crater") is still honoured: the operator asked to stop
    stop = {'intent': 'stop', 'hazard': 'crater'}
    assert c.offer(decision(action=REJECT, target_speed_mps=0.0, event='operator_command', local=stop),
                   0.0).speed_mps == 0.0
    # a stale operator answer is applied only if it is a stop
    assert c.offer(decision(event='operator_command', target_speed_mps=0.1, stale=True), 1.0) is None
    assert c.offer(decision(event='operator_command', target_speed_mps=0.0, stale=True, local=stop), 1.0).operator


@pytest.mark.parametrize('bad', [None, 'x', -0.1, float('nan'), float('inf'), True, '0.2'])
def test_malformed_caps_are_ignored(bad):
    c = BrainCap('rover_0')
    assert c.offer(decision(target_speed_mps=bad), 0.0) is None
    assert c.offer(None, 0.0) is None and c.offer('x', 0.0) is None and c.offer(decision(action='GO'), 0.0) is None


def test_cap_trace_texts():
    c = BrainCap('rover_0')
    cap = c.offer(decision(event='operator_command', target_speed_mps=0.2, command_text='slow down to 0.2 m/s'), 0.0)
    tx = cap_trace(cap)
    assert tx['action'] == 'Speed capped to 0.20 m/s by an operator command (Gemma, validated) for 60 s'
    assert tx['input'] == "/brain/decision SLOW_AND_REPLAN for operator_command: 'slow down to 0.2 m/s'"
    assert tx['severity'] == 'info' and tx['confidence'] == 0.8
    cap = c.offer(decision(action=CLARIFY, target_speed_mps=0.0, event='low_confidence'), 0.0)
    assert cap_trace(cap)['action'] == 'Holding 4 s: Gemma requested clarification'
    assert cap_trace(cap)['severity'] == 'warn'
    cap = c.offer(decision(event='operator_command', target_speed_mps=0.0, source='local-fallback', immediate=True,
                           command_seq=9), 0.0)
    assert cap_trace(cap)['action'] == 'Holding 60 s: operator stop (local rules), immediate'
    cap = c.offer(decision(target_speed_mps=0.31, source='local-fallback'), 0.0)
    assert cap_trace(cap)['action'] == 'Speed capped to 0.31 m/s by a novel hazard event (local rules) for 15 s'


# ---------------------------------------------------------------------------------------------------- altitude
def test_brain_altitude_for_the_drone_is_clamped_and_expires():
    a = BrainAltitude('scout_0')
    assert a.offer(decision(action=ALTITUDE, target_altitude_m=8.0, target_robot='scout_1'), 0.0) is None
    assert a.offer(decision(action=SLOW, target_altitude_m=8.0, target_robot='scout_0'), 0.0) is None
    assert a.offer(decision(action=ALTITUDE, target_altitude_m=None, target_robot='scout_0'), 0.0) is None
    assert a.offer(decision(action=ALTITUDE, target_altitude_m=2.0, target_robot='scout_0'), 0.0) == 6.0
    assert a.offer(decision(action=ALTITUDE, target_altitude_m=40.0, target_robot='scout_0'), 0.0) == 20.0
    assert a.offer(decision(action=ALTITUDE, target_altitude_m=8.0, robot='scout_0'), 10.0) == 8.0
    assert a.info(40.0) == {'agl_m': 8.0, 'remaining_s': 30.0, 'applied': False}
    assert a.target(70.0) is None and a.info(70.0) is None
    assert a.offer(decision(action=ALTITUDE, target_altitude_m=8.0, target_robot='scout_0', stale=True), 0.0) is None
    tx = bb.altitude_trace(8.0, decision(action=ALTITUDE, event='operator_command', command_text='drone down'), False)
    assert tx['action'].startswith('Cruise altitude 8 m AGL for 60 s by an operator command (Gemma, validated)')
    assert 'applies once surveying' in tx['action']


def test_the_operator_scenario_end_to_end_with_the_local_rules():
    """The smoke test's operator sequence through context → local rules → BrainCap."""
    states = {'rover_0': state(v_limit=0.94)}
    ctx = lambda text: build_context('rover_0', states=states, v_max_mps=0.94, operator_text=text,  # noqa: E731
                                     known_hazards=['rockslide'])
    c = BrainCap('rover_0')
    for seq, (text, action, cap) in enumerate([('drive at full speed into the rockslide', REJECT, None),
                                               ('slow down to 0.2 m/s', SLOW, 0.2),
                                               ('stop', SLOW, 0.0),
                                               ('resume at full speed', SLOW, 'released')], 1):
        d = deterministic_fallback('operator_command', ctx(text))
        d.robot = 'rover_0'
        msg = json.loads(bb.dumps(decision_message(d.as_dict(), trigger='operator', command_id=f'cmd-{seq}',
                                                   command_text=text, local=assess_operator_command(text, ctx(text)),
                                                   command_seq=seq, t=1.0)))
        assert msg['action'] == action, text
        applied = c.offer(parse_decision(json.dumps(msg)), float(seq))
        if cap == 'released':                       # "full speed" lifts the operator's own stop: no cap is left
            assert applied is not None and applied.released, text
        else:
            assert (applied.speed_mps if applied else None) == (pytest.approx(cap) if cap is not None else None), text
    assert assess_operator_command('stop', ctx('stop'))['intent'] == 'stop'
    assert c.limit(5.0) is None and c.operator is None


# ---------------------------------------------------------------------------------------------------- BrainHub
class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def gemma_transport(answer, calls=None, tin=164, tout=38):
    """Fake generateContent transport (no network): always answers `answer`."""
    def transport(url, headers, body, timeout_s):
        if calls is not None:
            calls.append(json.loads(body))
        payload = {'candidates': [{'content': {'parts': [{'text': json.dumps(answer)}]}, 'finishReason': 'STOP'}],
                   'usageMetadata': {'promptTokenCount': tin, 'candidatesTokenCount': tout},
                   'modelVersion': 'gemma-4-26b-a4b-it'}
        return 200, {}, json.dumps(payload).encode()
    return transport


def make_hub(client=None, clock=None):
    clock = clock or Clock()
    out = {'decisions': [], 'traces': []}
    brain = AgentBrain(client, min_interval_s=bb.MIN_INTERVAL_S, clock=clock, device_id='orin_nano_15w')
    hub = bb.BrainHub(brain, robots=ROBOTS, recovery_rover='rover_1', drone='scout_0', rover_v_max=0.94,
                      publish_decision=lambda d: out['decisions'].append(json.loads(bb.dumps(d))),
                      publish_trace=lambda t: out['traces'].append(json.loads(bb.dumps(t))), clock=clock,
                      wall=lambda: 1762550000.0)
    for r in ROBOTS:
        hub.on_state(r, json.dumps(state(r, 'SURVEY' if r.startswith('scout') else 'EXPLORING',
                                         4.0 if r.startswith('scout') else 0.94, altitude_agl=12.0)))
    return hub, out, clock


def wait_poll(hub, n=1, timeout=5.0):
    t_end, got = time.monotonic() + timeout, 0
    while got < n and time.monotonic() < t_end:
        got += hub.poll()
        if got < n:
            time.sleep(0.01)
    return got


def cmd(cid, text, robot='rover_0'):
    return json.dumps({'id': cid, 'text': text, 'robot': robot, 't': 1762550000.0})


def test_hub_hazard_trace_to_decision_and_advice_trace():
    hub, out, clock = make_hub()
    hub.on_perception('rover_0', json.dumps({'uncertainty': {'score': 0.2, 'camera_conf': 0.9, 'lidar_conf': 0.9},
                                             'corridor': {'obstacle_dist_m': 6.0}}))
    hub.on_hazard(json.dumps({'op': 'spawn_rockslide', 'ok': True}))
    assert hub.on_trace(json.dumps(trace('path_blocked', input='Rockslide 6 m ahead'))).event == 'novel_hazard'
    assert hub.poll() == 1
    d, tr = out['decisions'][0], out['traces'][0]
    assert (d['event'], d['robot'], d['trigger'], d['source'], d['action']) == (
        'novel_hazard', 'rover_0', 'path_blocked', 'local-fallback', SLOW)
    assert d['target_speed_mps'] == pytest.approx(0.35 * 0.94, abs=1e-3) and d['immediate'] is False
    assert d['error'] == 'no Gemma client configured' and d['model'] == 'local-rules' and d['latency_ms'] is None
    assert d['t'] == 1762550000.0 and d['command_id'] is None and d['local'] is None
    assert tr['kind'] == 'brain_advice' and tr['robot'] == 'rover_0' and tr['state'] == 'EXPLORING'
    assert tr['sensors'] == ['LOCAL'] and tr['brain']['event'] == 'novel_hazard'
    assert hub.on_trace(json.dumps(tr)) is None              # its own trace never loops back as an event
    assert hub.status(False, True)['stats']['local'] == 1 and hub.last_error == 'no Gemma client configured'


def test_hub_defers_a_deduplicated_hazard_until_its_slot():
    hub, out, clock = make_hub()
    hub.on_trace(json.dumps(trace('slope_hazard', input='Slope 21 deg')))
    hub.poll()
    clock.t += 4.0
    hub.on_trace(json.dumps(trace('path_blocked', input='Rockslide 6 m ahead')))     # inside the 10 s window
    assert hub.poll() == 0 and hub.gate.pending() == 1
    clock.t += 6.1
    hub.poll()                                         # due: re-offered with the newest trigger
    assert hub.poll() == 1
    assert [d['trigger'] for d in out['decisions']] == ['slope_hazard', 'path_blocked']
    assert out['decisions'][1]['trigger_input'] == 'Rockslide 6 m ahead'


def test_hub_operator_stop_is_answered_immediately_then_replaced():
    hub, out, clock = make_hub()
    assert hub.on_command(cmd('cmd-1', 'stop'))['id'] == 'cmd-1'
    assert len(out['decisions']) == 1                  # before any poll
    imm = out['decisions'][0]
    assert imm['immediate'] is True and imm['target_speed_mps'] == 0.0 and imm['command_id'] == 'cmd-1'
    assert imm['local']['intent'] == 'stop' and imm['command_seq'] == 1 and imm['request_id'] == 'immediate:cmd-1'
    assert imm['trigger'] == 'operator' and imm['command_text'] == 'stop'
    assert hub.poll() == 1
    later = out['decisions'][1]
    assert later['immediate'] is False and later['command_id'] == 'cmd-1' and later['command_seq'] == 1
    assert out['traces'][0]['input'] == "operator_command: 'stop' (immediate local stop)"
    # the planner side: the immediate stop holds at once, the later answer replaces it in the same slot
    c = BrainCap('rover_0')
    assert c.offer(imm, 0.0).speed_mps == 0.0 and c.offer(later, 1.0).speed_mps == 0.0
    assert hub.last_error == 'no Gemma client configured'      # the immediate answer does not set last_error


def test_hub_duplicate_and_malformed_commands():
    hub, out, clock = make_hub()
    hub.on_command(cmd('cmd-1', 'slow down'))
    hub.on_command(cmd('cmd-2', 'slow down'))           # same text < 0.5 s later
    hub.poll()
    by_id = {d['command_id']: d for d in out['decisions']}
    assert set(by_id) == {'cmd-1', 'cmd-2'} and 'duplicate' in by_id['cmd-2']['error']
    assert by_id['cmd-2']['command_seq'] == 2
    n = len(out['decisions'])
    assert hub.on_command('{"id": "x"}') is None and hub.on_command(cmd('cmd-3', 'go', 'rover_9')) is None
    assert len(out['decisions']) == n
    hub.on_command(cmd('cmd-4', 'go faster'))           # unbounded request → strict JSON (no Infinity)
    hub.poll()
    assert out['decisions'][-1]['action'] == REJECT and out['decisions'][-1]['local']['requested_speed_mps'] is None


def test_hub_peer_sos_asks_the_recovery_rover():
    hub, out, clock = make_hub()
    hub.on_trace(json.dumps(trace('sos', input='Immobilised at (3.0, 4.0), health 7%')))
    hub.poll()
    d, tr = out['decisions'][0], out['traces'][0]
    assert (d['event'], d['robot'], d['action'], d['target_robot']) == ('peer_sos', 'rover_1', DEPLOY, 'rover_1')
    assert d['target_speed_mps'] is None
    assert tr['robot'] == 'rover_1' and tr['action'].startswith('Deploy recovery rover ROVER_2 to ROVER_1')
    assert BrainCap('rover_1').offer(d, 0.0) is None    # the local rescue already runs: no motion change


def test_hub_with_gemma_measures_the_round_trip_and_counts_tokens():
    calls = []
    answer = {'action': 'SLOW_AND_REPLAN', 'target_speed_mps': 0.2, 'reasoning_trace': 'Operator asked 0.2 m/s.'}
    client = GemmaClient(api_key='AIzaTEST-not-a-real-key', transport=gemma_transport(answer, calls))
    hub, out, clock = make_hub(client)
    hub.on_command(cmd('cmd-7', 'slow down to 0.2 m/s'))
    assert wait_poll(hub) == 1
    d, tr = out['decisions'][0], out['traces'][0]
    assert (d['source'], d['model'], d['action'], d['target_speed_mps']) == ('gemma', 'gemma-4-26b-a4b-it', SLOW, 0.2)
    assert d['latency_ms'] is not None and d['tokens_in'] == 164 and d['tokens_out'] == 38
    assert d['projected_fits'] is False and 'local rules are the on-device brain' in d['projected_note']
    assert d['command_id'] == 'cmd-7' and d['local']['intent'] == 'set_speed'
    assert tr['sensors'] == ['GEMMA'] and tr['brain']['tokens_in'] == 164 and tr['latency_ms'] == d['latency_ms']
    summary = calls[0]['contents'][0]['parts'][0]['text']
    assert '"operator":"slow down to 0.2 m/s"' in summary and '"v_limit_mps":0.94' in summary
    st = hub.status(True, True)
    assert st['stats']['cloud_calls'] == 1 and st['stats']['gemma'] == 1 and st['last_error'] is None
    assert set(st) == {'enabled', 'key_present', 'model', 'device', 'stats', 'backoff_s', 'last_error', 't'}
    hub.brain.close()


# ---------------------------------------------------------------------------------------------------- review fixes
BIG = 10 ** 400                # a valid JSON integer that float() cannot convert (OverflowError)


def op_decision(intent, hazard=None, **kw):
    """An operator-command decision for rover_0 whose local reading is `intent` (/ `hazard`)."""
    return decision(event='operator_command', local={'intent': intent, 'hazard': hazard}, **kw)


def test_only_a_real_operator_stop_holds_in_the_operator_slot():                                          # R1
    c = BrainCap('rover_0')
    # CLARIFY / REJECT answers to a non-stop command never change motion, whatever speed they carry
    assert c.offer(op_decision('unclear', action=CLARIFY, target_speed_mps=0.0, command_seq=1), 0.0) is None
    assert c.offer(op_decision('enter_hazard', 'crater', action=REJECT, target_speed_mps=0.0, command_seq=2),
                   0.0) is None
    assert c.limit(0.1) is None and c.operator is None and c.event is None
    # a SLOW_AND_REPLAN hold answering "slow down" changes no motion either: not a 60 s operator stop, no event hold
    assert c.offer(op_decision('slow', target_speed_mps=0.0, command_seq=3), 1.0) is None
    assert c.limit(1.1) is None and c.operator is None and c.event is None
    # a real stop (local intent "stop", or the immediate stop decision) holds 60 s in the operator slot
    stop = c.offer(op_decision('stop', target_speed_mps=0.0, command_seq=4), 10.0)
    assert stop.operator and stop.duration_s == bb.OPERATOR_CAP_S and c.limit(69.9) == 0.0
    assert cap_trace(stop)['action'] == 'Holding 60 s: operator stop (Gemma, validated)'
    imm = BrainCap('rover_0').offer(decision(event='operator_command', target_speed_mps=0.0, immediate=True,
                                             source='local-fallback'), 0.0)
    assert imm.operator and cap_trace(imm)['action'] == 'Holding 60 s: operator stop (local rules), immediate'
    rej = BrainCap('rover_0').offer(op_decision('stop', 'crater', action=REJECT, target_speed_mps=0.0), 0.0)
    assert rej.operator and cap_trace(rej)['action'] == (
        'Holding 60 s: operator stop (Gemma, validated); the rest of the command was rejected')


def test_gemma_holds_answering_non_stop_commands_do_not_pin_the_rover():                                 # R1
    """The reproduced defect: Gemma's 0 m/s survives validation on CLARIFY / REJECT / SLOW answers to commands that
    are not stops; none of them may become a 60 s operator stop."""
    c = BrainCap('rover_0')
    for seq, (text, action) in enumerate([('do a barrel roll', CLARIFY), ('go faster', REJECT),
                                          ('slow down', SLOW)], 1):
        ctx = build_context('rover_0', states={'rover_0': state(v_limit=0.94)}, v_max_mps=0.94, operator_text=text)
        d = validate({'action': action, 'target_speed_mps': 0, 'reasoning_trace': 'r'}, 'operator_command', ctx)
        d.robot = 'rover_0'
        assert (d.action, d.target_speed_mps) == (action, 0.0), text
        msg = decision_message(d.as_dict(), trigger='operator', command_text=text, command_seq=seq,
                               local=assess_operator_command(text, ctx), t=1.0)
        cap = c.offer(parse_decision(bb.dumps(msg)), float(seq))
        assert cap is None, text
    assert c.operator is None and c.event is None and c.limit(4.0) is None


def test_an_operator_answer_never_shortens_a_live_hazard_cap():
    # a 15 s hazard cap, then Gemma answers "go 0.5 m/s" with a 0 m/s hold: the hold is ignored and the hazard cap
    # keeps binding for its full 15 s (an event-slot write would have replaced it with a 4 s hold)
    c = BrainCap('rover_0')
    assert c.offer(decision(target_speed_mps=0.2), 0.0).speed_mps == 0.2
    assert c.offer(op_decision('set_speed', target_speed_mps=0.0, command_seq=1), 1.0) is None
    assert c.limit(5.1) == 0.2 and c.limit(14.9) == 0.2 and c.limit(15.1) is None


def test_near_zero_caps_are_holds():                                                                   # R3
    c = BrainCap('rover_0')
    cap = c.offer(decision(target_speed_mps=0.03), 0.0)                 # event: rounded to 0, a 4 s hold
    assert cap.speed_mps == 0.0 and cap.duration_s == bb.EVENT_HOLD_S and c.info(1.0)['speed_mps'] == 0.0
    assert cap_trace(cap)['action'] == 'Holding 4 s: Gemma requested a stop and re-plan'
    assert cap_trace(cap)['severity'] == 'warn' and c.apply(0.5, 0.3, 1.0) == (0.0, 0.0)
    floor = c.offer(decision(target_speed_mps=bb.HOLD_FLOOR_MPS), 10.0)  # the floor itself is a crawl cap
    assert floor.speed_mps == 0.05 and floor.duration_s == bb.EVENT_CAP_S
    c = BrainCap('rover_0')                                             # an operator crawl this slow is a stop
    crawl = c.offer(decision(event='operator_command', target_speed_mps=0.049, command_seq=1,
                             local={'intent': 'set_speed', 'hazard': None, 'requested_speed_mps': 0.049}), 0.0)
    assert crawl.operator and crawl.speed_mps == 0.0 and crawl.duration_s == bb.OPERATOR_CAP_S
    assert c.limit(30.0) == 0.0
    stop = c.offer(op_decision('stop', target_speed_mps=0.01, command_seq=2), 0.0)
    assert stop.operator and stop.speed_mps == 0.0 and stop.duration_s == bb.OPERATOR_CAP_S


def test_a_faster_or_set_speed_command_naming_no_hazard_lifts_the_operator_stop():                        # R7
    c = BrainCap('rover_0')
    c.offer(decision(event='operator_command', target_speed_mps=0.0, immediate=True, command_seq=1,
                     local={'intent': 'stop', 'hazard': None}), 0.0)
    # "drive at full speed into the rockslide": rejected and it names a hazard → the stop stays
    assert c.offer(op_decision('faster', 'rockslide', action=REJECT, target_speed_mps=None, command_seq=2),
                   1.0) is None
    assert c.limit(2.0) == 0.0
    # "go faster": rejected (unbounded), but it lifts the operator's own stop; the local limits apply again
    rel = c.offer(op_decision('faster', action=REJECT, target_speed_mps=None, command_seq=3,
                              command_text='go faster'), 3.0)
    assert rel.released and c.operator is None and c.limit(3.0) is None
    assert cap_trace(rel)['action'] == ('Operator stop lifted by a newer operator command (Gemma, validated); '
                                        'the local limits apply')
    assert cap_trace(rel)['severity'] == 'info'
    # a CLARIFY answer to "drive at 0.4 m/s" lifts an operator slow-down too, without applying its own speed
    c.offer(op_decision('slow', target_speed_mps=0.2, command_seq=4), 4.0)
    rel = c.offer(op_decision('set_speed', action=CLARIFY, target_speed_mps=0.4, command_seq=5), 5.0)
    assert rel.released and cap_trace(rel)['action'].startswith('Operator speed cap lifted') and c.limit(5.0) is None
    # never an event cap; with nothing live to lift there is nothing to trace, but the command still orders
    c.offer(decision(target_speed_mps=0.3), 6.0)
    assert c.offer(op_decision('faster', action=REJECT, target_speed_mps=None, command_seq=6), 7.0) is None
    assert c.limit(7.0) == 0.3 and c.seq == 6


def test_the_local_rules_stop_then_go_faster_lifts_the_stop():                                          # R7
    states = {'rover_0': state(v_limit=0.94)}
    c = BrainCap('rover_0')
    applied = []
    for seq, text in enumerate(['stop', 'go faster'], 1):
        ctx = build_context('rover_0', states=states, v_max_mps=0.94, operator_text=text)
        d = deterministic_fallback('operator_command', ctx)
        d.robot = 'rover_0'
        msg = decision_message(d.as_dict(), trigger='operator', command_text=text, command_seq=seq,
                               local=assess_operator_command(text, ctx), t=1.0)
        applied.append(c.offer(parse_decision(bb.dumps(msg)), float(seq)))
    assert applied[0].operator and applied[0].speed_mps == 0.0
    assert applied[1].action == REJECT and applied[1].released and c.limit(3.0) is None


def test_rescue_is_never_slowed_by_advice_about_the_sos():                                              # R8
    c = BrainCap('rover_1')
    for ev in bb.SOS_EVENTS:
        for a, sp in ((SLOW, 0.2), (CLARIFY, 0.0)):
            assert c.offer(decision(robot='rover_1', event=ev, action=a, target_speed_mps=sp), 0.0) is None
    c.offer(decision(robot='rover_1', target_speed_mps=0.3), 0.0)           # a hazard cap on the way there
    assert c.limit(1.0, 'RESCUE_TRANSIT') == 0.3
    for st in bb.RESCUE_STATES:                                             # ... ignored while docking / repairing
        assert c.limit(1.0, st) is None and c.apply(0.4, 0.2, 1.0, st) == (0.4, 0.2) and c.info(1.0, st) is None
    c.offer(decision(robot='rover_1', event='operator_command', target_speed_mps=0.1, command_seq=1,
                     local={'intent': 'set_speed', 'hazard': None}), 2.0)   # operator caps still apply
    assert c.limit(3.0, 'RESCUE_DOCKING') == 0.1 and c.info(3.0, 'RESCUE_DOCKING')['operator'] is True
    c.offer(decision(robot='rover_1', action=CLARIFY, target_speed_mps=0.0, event='low_confidence'), 4.0)
    assert c.holding(5.0, 'RESCUE_TRANSIT') and not c.holding(5.0, 'REPAIRING')               # (R5 uses holding)


def test_command_order_survives_cap_expiry_and_stale_answers_only_stop():                                # R9
    c = BrainCap('rover_0')
    c.offer(op_decision('set_speed', target_speed_mps=0.3, command_seq=5), 0.0)
    assert c.limit(61.0) is None                                          # that cap has expired ...
    assert c.offer(op_decision('stop', target_speed_mps=0.0, command_seq=4), 100.0) is None   # ... 4 is still older
    assert c.limit(100.0) is None
    c.offer(op_decision('faster', action=REJECT, target_speed_mps=None, command_seq=7), 101.0)  # a release orders too
    assert c.offer(op_decision('stop', target_speed_mps=0.0, command_seq=6), 102.0) is None
    assert c.offer(op_decision('slow', target_speed_mps=0.2, command_seq=8, stale=True), 103.0) is None
    assert c.seq == 7
    assert c.offer(op_decision('stop', target_speed_mps=0.0, command_seq=8, stale=True), 103.0).operator
    assert c.seq == 8


def test_brain_agl_reports_whether_it_is_flown():                                                       # R10
    a = BrainAltitude('scout_0')
    a.offer(decision(action=ALTITUDE, target_altitude_m=8.0, target_robot='scout_0'), 0.0)
    assert a.info(1.0)['applied'] is False                    # not in a state that flies it (take-off)
    assert a.info(1.0, 12.0)['applied'] is False              # set-point still the cruise AGL
    assert a.info(1.0, 8.0) == {'agl_m': 8.0, 'remaining_s': 59.0, 'applied': True}


@pytest.mark.parametrize('g', [1.6, 2.0, 3.0])
def test_full_speed_at_the_full_local_limit_is_not_above_it(g):                                        # R6
    v_max = RoverPhysics().limits(g).v_max
    v_limit = round(v_max, 3)                 # agent_state v_limit at speed_scale 1 (planner rounding)
    assert v_limit < v_max                    # the case the unrounded v_max broke
    text = 'resume at full speed'
    ctx = build_context('rover_0', states={'rover_0': state(v_limit=v_limit)}, v_max_mps=v_max, operator_text=text)
    assert ctx['v_max_mps'] == ctx['speed_limit_mps'] == v_limit
    local = assess_operator_command(text, ctx)
    assert local['intent'] == 'faster' and not local['exceeds_limit'] and not local['unsafe']
    d = deterministic_fallback('operator_command', ctx)
    assert (d.action, d.target_speed_mps) == (SLOW, v_limit)


def test_event_gate_keeps_the_rockslide_over_a_later_slope_warning():                                  # R11
    g = EventGate({'novel_hazard': 10.0})
    slope = EventSpec('novel_hazard', 'rover_0', 'slope_hazard', 'slope 21 deg')
    rock = EventSpec('novel_hazard', 'rover_0', 'path_blocked', 'rockslide')
    mark = EventSpec('novel_hazard', 'rover_0', 'hazard_marked', 'tip zone')
    g.submitted(slope, 100.0)
    g.defer(rock, 103.0)
    g.defer(slope, 105.0)                     # lower priority: does not replace the rockslide
    g.defer(mark, 106.0)                      # same priority: the first deferred stays
    assert g.due(110.1) == [rock]
    g.defer(slope, 111.0)
    g.defer(mark, 112.0)                      # higher priority replaces a deferred slope warning
    assert g.due(130.0) == [mark]


def test_hub_keeps_the_deferred_rockslide_when_a_slope_warning_follows():                                # R11
    hub, out, clock = make_hub()
    hub.on_trace(json.dumps(trace('slope_hazard', input='Slope 21 deg')))
    hub.poll()
    clock.t += 3.0
    hub.on_trace(json.dumps(trace('path_blocked', input='Rockslide 6 m ahead')))
    clock.t += 2.0
    hub.on_trace(json.dumps(trace('slope_hazard', input='Slope 23 deg')))
    clock.t += 5.1
    hub.poll()
    assert hub.poll() == 1 and out['decisions'][-1]['trigger_input'] == 'Rockslide 6 m ahead'


@pytest.mark.parametrize('text', ['stop', 'slow down', 'set speed 0.3 m/s', 'go faster', 'fly through the crater',
                                  'do a barrel roll'])
def test_hub_drone_takes_altitude_commands_only(text):                                                  # R2
    calls = []
    answer = {'action': 'SLOW_AND_REPLAN', 'target_speed_mps': 0.0, 'reasoning_trace': 'x'}
    hub, out, clock = make_hub(GemmaClient(api_key='AIzaTEST-not-a-real-key', transport=gemma_transport(answer, calls)))
    hub.last_error = 'earlier cloud error'
    assert hub.on_command(cmd('cmd-1', text, 'scout_0'))['robot'] == 'scout_0'
    assert len(out['decisions']) == 1                         # answered at once, no immediate stop decision
    d = out['decisions'][0]
    assert (d['action'], d['target_speed_mps'], d['target_altitude_m'], d['immediate']) == (CLARIFY, None, None, False)
    assert d['reasoning_trace'] == bb.DRONE_ONLY_TEXT == ('Drones take altitude commands only; select a rover for '
                                                          'speed or stop commands')
    assert (d['robot'], d['source'], d['command_id'], d['command_seq']) == ('scout_0', 'local-fallback', 'cmd-1', 1)
    assert d['local']['intent'] != 'drone_altitude' and d['target_robot'] is None
    assert out['traces'][0]['robot'] == 'scout_0'
    assert out['traces'][0]['action'] == 'Request operator clarification' + bb.TRACE_SUFFIX
    time.sleep(0.05)
    assert hub.poll() == 0 and calls == [] and hub.brain.stats()['submitted'] == 0      # no Gemma call
    assert hub.last_error == 'earlier cloud error'
    assert BrainCap('rover_0').offer(d, 0.0) is None and BrainAltitude('scout_0').offer(d, 0.0) is None
    hub.brain.close()


@pytest.mark.parametrize('text, alt', [('drone down to 8 m', 8.0), ('climb to 10 m', 10.0), ('descend to 8 m', 8.0),
                                       ('fly at 15 m AGL', 15.0), ('higher', 16.0), ('lower', 8.0)])
def test_hub_drone_altitude_command_still_goes_to_the_brain(text, alt):                                  # R2
    # typed to the selected drone, an altitude command need not name it
    hub, out, clock = make_hub()
    hub.on_command(cmd('cmd-1', text, 'scout_0'))
    assert out['decisions'] == [] and hub.poll() == 1
    d = out['decisions'][0]
    assert (d['action'], d['target_altitude_m'], d['target_robot']) == (ALTITUDE, alt, 'scout_0'), text
    assert d['command_text'] == text                          # the operator's own words are what is reported
    assert BrainAltitude('scout_0').offer(d, 0.0) == alt


def test_huge_json_integers_never_crash_the_bridge():                                                   # R4
    assert bb._num(BIG) is None and bb._num(-BIG) is None and bb._num(json.loads('9' * 400)) is None
    c = BrainCap('rover_0')
    assert c.offer(parse_decision(json.dumps(decision(target_speed_mps=BIG))), 0.0) is None
    assert c.offer(parse_decision(json.dumps(op_decision('stop', target_speed_mps=BIG, command_seq=BIG))),
                   0.0) is None
    a = BrainAltitude('scout_0')
    assert a.offer(parse_decision(json.dumps(decision(action=ALTITUDE, target_altitude_m=BIG,
                                                      target_robot='scout_0'))), 0.0) is None
    assert parse_operator_command(json.dumps({'id': 'c', 'text': 'stop', 't': BIG}), ROBOTS)['t'] is None
    # agent_state / perception / edge / env numbers the bridge reads, plus malformed health blocks
    hub, out, clock = make_hub()
    hub.on_state('rover_0', json.dumps(state(v_limit=BIG, speed=BIG, altitude_agl=BIG,
                                             health={'pct': BIG, 'battery_pct': BIG, 'critical': False,
                                                     'faults': BIG})))
    hub.on_state('rover_1', json.dumps(state('rover_1', health=[BIG])))
    hub.on_state('scout_0', json.dumps(state('scout_0', 'SURVEY', BIG, altitude_agl=BIG)))
    hub.on_perception('rover_0', json.dumps({'uncertainty': {'score': BIG, 'camera_conf': BIG, 'lidar_conf': BIG},
                                             'corridor': {'obstacle_dist_m': BIG}}))
    hub.on_edge('rover_0', json.dumps({'uncertainty': BIG}))
    hub.on_env(json.dumps({'dust_level': BIG}))
    assert hub.on_trace(json.dumps(trace('path_blocked'))).event == 'novel_hazard'
    assert hub.on_trace(json.dumps(trace('caution_enter', robot='rover_1'))).event == 'low_confidence'
    assert hub.on_command(json.dumps({'id': 'cmd-1', 'text': 'stop', 'robot': 'rover_0', 't': BIG}))['t'] is None
    hub.on_command(json.dumps({'id': 'cmd-2', 'text': 'drone down', 'robot': 'scout_0', 't': BIG}))
    hub.poll()
    assert sorted(d['event'] for d in out['decisions']) == ['low_confidence', 'novel_hazard', 'operator_command',
                                                            'operator_command', 'operator_command']
    assert hub.guard.dropped == 0
    rover = next(d for d in out['decisions'] if d['event'] == 'novel_hazard')
    assert rover['target_speed_mps'] == 0.0            # unreadable v_limit → the brain reasons about a stopped robot


def test_hub_entry_points_drop_a_failing_message_with_one_throttled_warning():                          # R4
    warnings, clock = [], Clock()
    brain = AgentBrain(None, min_interval_s=bb.MIN_INTERVAL_S, clock=clock)
    hub = bb.BrainHub(brain, robots=ROBOTS, publish_decision=lambda d: None, publish_trace=lambda t: None,
                      clock=clock, wall=lambda: 1.0, warn=warnings.append)

    def boom(*a, **kw):
        raise OverflowError('int too large to convert to float')
    hub.brain.submit = boom
    assert hub.on_trace(json.dumps(trace('path_blocked'))) is None           # dropped, never raised
    assert hub.on_command(cmd('cmd-1', 'slow down')) is None
    assert hub.on_command('{') is None                                       # malformed: shares the throttle
    assert hub.guard.dropped == 3 and len(warnings) == 1
    assert warnings[0] == 'dropped a /agent_decision_trace message (OverflowError: int too large to convert to float)'
    clock.t += bb.WARN_EVERY_S
    hub.on_trace(json.dumps(trace('path_blocked')))
    assert len(warnings) == 2 and warnings[1].endswith('(+2 more dropped since the last warning)')
    hub.brain.stats = boom
    assert hub.status(True, True) is None and hub.guard.dropped == 5


def test_hub_poll_drops_one_failing_decision_not_the_batch():                                            # R4
    hub, out, clock = make_hub()
    sent, publish = [], hub.publish_decision

    def flaky(d):
        sent.append(d)
        if len(sent) == 1:
            raise ValueError('publisher gone')
        publish(d)
    hub.publish_decision = flaky
    hub.on_trace(json.dumps(trace('path_blocked')))
    hub.on_trace(json.dumps(trace('caution_enter', robot='rover_1')))
    assert hub.poll() == 1 and len(out['decisions']) == 1 and hub.guard.dropped == 1


def test_drop_guard_for_the_planner_callbacks():                                                        # R4
    warnings, clock = [], Clock(0.0)
    g = bb.DropGuard(warnings.append, every_s=5.0, clock=clock)
    assert g.call('x', lambda v: v * 2, 3) == 6 and g.dropped == 0
    assert g.call('/brain/decision', float, BIG) is None                 # OverflowError → dropped
    assert g.wrap('/brain/decision', lambda: 1 / 0)() is None
    assert g.dropped == 2 and warnings == ['dropped a /brain/decision message (OverflowError: int too large to '
                                           'convert to float)']

    def bad_warn(_):
        raise RuntimeError('logger gone')
    assert bb.DropGuard(bad_warn).call('x', float, BIG) is None          # a failing logger does not raise either
