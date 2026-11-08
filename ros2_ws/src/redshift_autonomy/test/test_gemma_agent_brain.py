"""Unit tests for the advisory Gemma brain (no network: every HTTP exchange goes through a fake transport)."""
import json
import math
import random
import threading
import time

import pytest

from redshift_autonomy import gemma_agent_brain as gb
from redshift_autonomy.edge_budget import PROFILES
from redshift_autonomy.gemma_agent_brain import (
    ACTIONS, ALTITUDE, CLARIFY, DEPLOY, EVENTS, REJECT, SLOW, AgentBrain, Decision, GemmaClient, GemmaError,
    allowed_actions, assess_operator_command, build_summary, deterministic_fallback, load_api_key, parse_response,
    project_latency, validate)

KEY = 'AIzaTEST-not-a-real-key-0123456789abcdef'


# ---------------------------------------------------------------------------------------------------- fixtures
def ctx(**kw):
    base = {'robot': 'rover_0', 'robot_type': 'rover', 'state': 'EXPLORING', 'speed_mps': 0.5,
            'speed_limit_mps': 0.8, 'v_max_mps': 1.0, 'health': 0.9, 'battery': 0.7, 'camera_conf': 0.8,
            'lidar_conf': 0.9, 'uncertainty': 0.2, 'nearest_obstacle_m': 6.0,
            'known_hazards': ['boulder field', 'crater rim'],
            'fleet': {'recovery_rover': 'rover_1', 'drone': 'scout_0', 'drone_agl_m': 12.0}}
    fleet = kw.pop('fleet', None)
    base.update(kw)
    if fleet is not None:
        base['fleet'] = fleet
    return base


NO_FLEET = {'recovery_rover': None, 'drone': None}


def gemma_payload(answer, tin=140, tout=56, model='gemma-4-26b-a4b-it', text=None):
    return {'candidates': [{'content': {'role': 'model', 'parts': [{'text': text if text is not None
                                                                     else json.dumps(answer)}]},
                            'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': tin, 'candidatesTokenCount': tout},
            'modelVersion': model}


class FakeTransport:
    """Records every request; answers from a list of (status, headers, body) or a callable."""

    def __init__(self, *responses, delay_s=0.0, gate=None):
        self.responses = list(responses)
        self.calls = []
        self.delay_s = delay_s
        self.gate = gate
        self.lock = threading.Lock()

    def __call__(self, url, headers, body, timeout_s):
        with self.lock:
            self.calls.append({'url': url, 'headers': dict(headers), 'body': json.loads(body), 'timeout': timeout_s})
            r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if self.gate is not None:
            assert self.gate.wait(5.0)
        if self.delay_s:
            time.sleep(self.delay_s)
        if callable(r):
            return r()
        status, hdrs, payload = r
        return status, hdrs, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


def ok(answer, **kw):
    return 200, {}, gemma_payload(answer, **kw)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def wait_for(brain, n, timeout=3.0):
    out, end = [], time.monotonic() + timeout
    while len(out) < n and time.monotonic() < end:
        out += brain.poll()
        time.sleep(0.005)
    return out


def client(*responses, **kw):
    tr = FakeTransport(*responses, delay_s=kw.pop('delay_s', 0.0), gate=kw.pop('gate', None))
    return GemmaClient(api_key=KEY, transport=tr, **kw), tr


# ---------------------------------------------------------------------------------------------------- invariants
def lim_of(c):
    v = c.get('speed_limit_mps')
    return max(0.0, float(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0.0


def assert_safe(d: Decision, event, c):
    assert d.action in ACTIONS
    assert d.source in ('gemma', 'local-fallback')
    assert d.event == event
    assert len(d.reasoning_trace) <= gb.REASONING_MAX_CHARS and d.reasoning_trace
    lim = lim_of(c)
    if d.target_speed_mps is not None:
        assert 0.0 <= d.target_speed_mps <= lim + 1e-9, (d, lim)
    fleet = c.get('fleet') or {}
    if d.action == DEPLOY:
        assert event in ('critical_damage', 'peer_sos')
        rr = fleet.get('recovery_rover')
        disabled = c.get('peer') if event == 'peer_sos' else c.get('robot')
        assert rr and fleet.get('recovery_available', True) and rr != disabled
        assert d.target_robot == (rr if isinstance(rr, str) else 'recovery_rover')
    if d.action == ALTITUDE:
        assert fleet.get('drone') or c.get('robot_type') in ('scout', 'drone')
        assert gb.DRONE_ALT_MIN_M <= d.target_altitude_m <= gb.DRONE_ALT_MAX_M
    else:
        assert d.target_altitude_m is None
    if d.action == REJECT:
        assert event == 'operator_command'
    if event == 'operator_command':
        a = assess_operator_command(c.get('operator_text', ''), c)
        if a['unsafe']:
            assert d.action == REJECT
        if a['intent'] == 'stop':
            assert d.target_speed_mps == 0.0
    json.dumps(d.as_dict(), allow_nan=False)   # HUD-serialisable, no inf / nan


# ---------------------------------------------------------------------------------------------------- vocabulary
def test_vocabulary():
    assert EVENTS == ('novel_hazard', 'low_confidence', 'critical_damage', 'peer_sos', 'operator_command')
    assert ACTIONS == ('SLOW_AND_REPLAN', 'REQUEST_CLARIFICATION', 'DEPLOY_RECOVERY_ROVER', 'ADJUST_DRONE_ALTITUDE',
                       'REJECT_UNSAFE_COMMAND')
    with pytest.raises(ValueError):
        build_summary('teleport', ctx())
    with pytest.raises(ValueError):
        validate({'action': SLOW, 'reasoning_trace': 'x'}, 'teleport', ctx())


def test_allowed_actions_follow_capabilities():
    assert allowed_actions('low_confidence', ctx(fleet=NO_FLEET)) == (SLOW, CLARIFY)
    assert allowed_actions('critical_damage', ctx()) == (SLOW, CLARIFY, DEPLOY, ALTITUDE)
    assert REJECT in allowed_actions('operator_command', ctx())
    assert DEPLOY not in allowed_actions('novel_hazard', ctx())
    # the recovery rover cannot rescue itself
    assert DEPLOY not in allowed_actions('critical_damage', ctx(robot='rover_1'))
    assert DEPLOY in allowed_actions('peer_sos', ctx(robot='rover_1', peer='rover_0'))
    assert DEPLOY not in allowed_actions('peer_sos', ctx(peer='rover_1'))
    busy = {'recovery_rover': 'rover_1', 'recovery_available': False}
    assert DEPLOY not in allowed_actions('critical_damage', ctx(fleet=busy))


# ---------------------------------------------------------------------------------------------------- validation
def test_valid_answer_is_kept_and_attributed_to_gemma():
    d = validate({'action': SLOW, 'target_speed_mps': 0.3, 'reasoning_trace': 'Camera 0.2 in dust; slow.'},
                 'low_confidence', ctx(), model='gemma-4-26b-a4b-it', latency_ms=2210.4, tokens_in=140, tokens_out=56)
    assert (d.action, d.target_speed_mps, d.source, d.model) == (SLOW, 0.3, 'gemma', 'gemma-4-26b-a4b-it')
    assert d.adjusted == [] and d.error is None
    assert (d.latency_ms, d.tokens_in, d.tokens_out) == (2210.4, 140, 56)


@pytest.mark.parametrize('raw,err', [
    ({'action': 'GO_FASTER', 'reasoning_trace': 'x'}, 'not one of ACTIONS'),
    ({'action': 7, 'reasoning_trace': 'x'}, 'not one of ACTIONS'),
    ({'reasoning_trace': 'x'}, 'not one of ACTIONS'),
    ({'action': SLOW}, 'missing reasoning_trace'),
    ({'action': SLOW, 'reasoning_trace': '   '}, 'missing reasoning_trace'),
    (['SLOW_AND_REPLAN'], 'not a JSON object'),
    ('SLOW_AND_REPLAN', 'not a JSON object'),
    (None, 'not a JSON object'),
])
def test_invalid_answers_fall_back_with_error(raw, err):
    d = validate(raw, 'low_confidence', ctx(), latency_ms=900.0, tokens_in=10, tokens_out=5)
    assert d.source == 'local-fallback' and d.model == gb.LOCAL_BRAIN
    assert err in d.error
    assert d.latency_ms == 900.0 and d.tokens_out == 5      # the call still happened and is reported honestly
    assert_safe(d, 'low_confidence', ctx())


def test_action_normalisation_is_recorded():
    d = validate({'action': ' slow and-replan ', 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.action == SLOW and d.source == 'gemma'
    assert any('normalised' in a for a in d.adjusted)


def test_speed_is_clamped_to_local_limit_and_recorded():
    d = validate({'action': SLOW, 'target_speed_mps': 3.0, 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.target_speed_mps == 0.8 and any('local limit' in a for a in d.adjusted)
    d = validate({'action': SLOW, 'target_speed_mps': -2, 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.target_speed_mps == 0.0 and any('→ 0' in a for a in d.adjusted)
    d = validate({'action': SLOW, 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.target_speed_mps == pytest.approx(0.28) and any('missing' in a for a in d.adjusted)
    for bad in ('fast', True, float('nan')):
        d = validate({'action': SLOW, 'target_speed_mps': bad, 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
        assert d.target_speed_mps == pytest.approx(0.28) and any('not a number' in a for a in d.adjusted)
    d = validate({'action': SLOW, 'target_speed_mps': float('inf'), 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.target_speed_mps == 0.8


def test_unknown_or_hold_limit_means_zero():
    c = ctx()
    del c['speed_limit_mps']
    d = validate({'action': SLOW, 'target_speed_mps': 0.5, 'reasoning_trace': 'x'}, 'novel_hazard', c)
    assert d.target_speed_mps == 0.0 and any('unknown' in a for a in d.adjusted)
    d = validate({'action': SLOW, 'target_speed_mps': 0.5, 'reasoning_trace': 'x'}, 'novel_hazard',
                 ctx(state='SAFE_HOLD', speed_limit_mps=0.0))
    assert d.target_speed_mps == 0.0
    # agent_state alias
    d = validate({'action': SLOW, 'target_speed_mps': 0.5, 'reasoning_trace': 'x'}, 'novel_hazard',
                 {'robot': 'rover_0', 'v_limit': 0.4})
    assert d.target_speed_mps == 0.4


def test_reasoning_is_truncated():
    d = validate({'action': SLOW, 'target_speed_mps': 0.2, 'reasoning_trace': 'word ' * 200}, 'novel_hazard', ctx())
    assert len(d.reasoning_trace) <= 300 and any('truncated' in a for a in d.adjusted)


def test_deploy_only_for_damage_or_sos_with_an_available_rover():
    raw = {'action': DEPLOY, 'reasoning_trace': 'send help'}
    d = validate(raw, 'critical_damage', ctx(health=0.1))
    assert d.action == DEPLOY and d.source == 'gemma' and d.target_robot == 'rover_1'
    assert d.target_speed_mps == 0.0            # the damaged robot holds
    d = validate(raw, 'novel_hazard', ctx())
    assert d.action != DEPLOY and d.source == 'local-fallback' and 'not permitted' in d.error
    assert any(a.startswith('action DEPLOY_RECOVERY_ROVER →') for a in d.adjusted)
    d = validate(raw, 'critical_damage', ctx(fleet=NO_FLEET))
    assert d.action == CLARIFY and 'no recovery rover' in d.error
    d = validate(raw, 'critical_damage', ctx(robot='rover_1'))
    assert d.action != DEPLOY and 'is the disabled robot' in d.error
    d = validate(raw, 'peer_sos', ctx(peer='rover_2'))
    assert d.action == DEPLOY and d.target_speed_mps is None


def test_drone_altitude_needs_a_drone_and_is_clamped():
    d = validate({'action': ALTITUDE, 'target_altitude_m': 50, 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.target_altitude_m == 20.0 and d.target_robot == 'scout_0' and d.source == 'gemma'
    d = validate({'action': ALTITUDE, 'target_altitude_m': 1, 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.target_altitude_m == 6.0 and any('6-20' in a for a in d.adjusted)
    d = validate({'action': ALTITUDE, 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.target_altitude_m == 8.0 and any('missing' in a for a in d.adjusted)
    d = validate({'action': ALTITUDE, 'target_altitude_m': 9, 'reasoning_trace': 'x'}, 'novel_hazard',
                 ctx(fleet=NO_FLEET))
    assert d.action != ALTITUDE and d.target_altitude_m is None and 'no drone' in d.error
    d = validate({'action': SLOW, 'target_altitude_m': 9, 'reasoning_trace': 'x'}, 'novel_hazard', ctx())
    assert d.target_altitude_m is None and any('ignored' in a for a in d.adjusted)


def test_reject_is_only_for_operator_commands():
    d = validate({'action': REJECT, 'reasoning_trace': 'x'}, 'low_confidence', ctx())
    assert d.action != REJECT and d.source == 'local-fallback'


# ---------------------------------------------------------------------------------------------------- operator
UNSAFE_OP = ctx(operator_text='full speed through the boulder field', speed_limit_mps=0.42)


def test_unsafe_operator_command_always_ends_rejected():
    d = validate({'action': REJECT, 'reasoning_trace': 'Boulders ahead; refusing.'}, 'operator_command', UNSAFE_OP)
    assert d.action == REJECT and d.source == 'gemma' and d.target_speed_mps is None
    for act in (SLOW, CLARIFY, ALTITUDE):
        d = validate({'action': act, 'target_speed_mps': 0.3, 'target_altitude_m': 10, 'reasoning_trace': 'ok'},
                     'operator_command', UNSAFE_OP)
        assert d.action == REJECT and d.source == 'local-fallback' and d.error is None
        assert any('local check' in a and 'boulder field' in a for a in d.adjusted)
    # Gemma complying with an over-limit speed request is turned into a rejection
    c = ctx(operator_text='go to waypoint 3', speed_limit_mps=0.5)
    d = validate({'action': SLOW, 'target_speed_mps': 0.9, 'reasoning_trace': 'ok'}, 'operator_command', c)
    assert d.action == REJECT and d.source == 'local-fallback' and any('0.9 m/s' in a for a in d.adjusted)


def test_operator_stop_and_slow_requests_are_honoured():
    c = ctx(operator_text='stop now')
    d = validate({'action': REJECT, 'reasoning_trace': 'no'}, 'operator_command', c)
    assert d.action == SLOW and d.target_speed_mps == 0.0 and d.source == 'local-fallback'
    d = validate({'action': CLARIFY, 'reasoning_trace': 'why?'}, 'operator_command', c)
    assert d.action == CLARIFY and d.target_speed_mps == 0.0 and d.source == 'gemma'
    c = ctx(operator_text='slow to 0.2 m/s please')
    d = validate({'action': SLOW, 'target_speed_mps': 0.5, 'reasoning_trace': 'ok'}, 'operator_command', c)
    assert d.target_speed_mps == 0.2 and any('operator asked' in a for a in d.adjusted)
    # regression (found by the property test): a stop while in SAFE_HOLD must stay a stop, not become a REJECT
    c = ctx(operator_text='hold', speed_limit_mps=0.0)
    d = validate({'action': SLOW, 'target_speed_mps': 0.1, 'reasoning_trace': 'creep'}, 'operator_command', c)
    assert (d.action, d.target_speed_mps, d.source) == (SLOW, 0.0, 'gemma')


@pytest.mark.parametrize('text,intent,unsafe', [
    ('full speed through the boulder field', 'faster', True),
    ('Drive through the boulder field', 'enter_hazard', True),
    ('go faster', 'faster', True),
    ('floor it to the ridge', 'faster', True),
    ('drive at 2 m/s', 'set_speed', True),
    ('drive at 0.5 m/s', 'set_speed', False),
    ('drive at 3.6 km/h', 'set_speed', True),
    ('slow down', 'slow', False),
    ('stop', 'stop', False),
    ('hold position', 'stop', False),
    ("don't stop, go through the crater", 'enter_hazard', True),
    ('avoid the boulder field', 'unclear', False),
    ("don't drive through the crater rim", 'unclear', False),
    ('go around the rockslide debris', 'unclear', False),
    ('drone descend to 8 m', 'drone_altitude', False),
    ('what is that shiny thing', 'unclear', False),
    ('', 'unclear', False),
])
def test_operator_assessment(text, intent, unsafe):
    a = assess_operator_command(text, ctx(speed_limit_mps=0.8))
    assert (a['intent'], a['unsafe']) == (intent, unsafe), a


@pytest.mark.parametrize('text,action,speed', [
    ('full speed through the boulder field', REJECT, None),
    ('go faster', REJECT, None),
    ('drive through the crater rim', REJECT, None),
    ('stop', SLOW, 0.0),
    ('slow down', SLOW, 0.28),
    ('cap speed at 0.5 m/s', SLOW, 0.5),
    ('hmm, what about over there', CLARIFY, None),
    ('avoid the boulder field', CLARIFY, None),
])
def test_operator_fallback(text, action, speed):
    c = ctx(operator_text=text)
    d = deterministic_fallback('operator_command', c)
    assert d.action == action and d.source == 'local-fallback'
    assert d.target_speed_mps == (pytest.approx(speed) if speed is not None else None)
    assert_safe(d, 'operator_command', c)


def test_full_speed_is_fine_only_when_already_at_the_limit():
    c = ctx(operator_text='full speed', speed_limit_mps=1.0, v_max_mps=1.0)
    assert not assess_operator_command(c['operator_text'], c)['unsafe']
    assert deterministic_fallback('operator_command', c).action == SLOW
    c = ctx(operator_text='full speed', speed_limit_mps=0.35, v_max_mps=1.0)
    assert deterministic_fallback('operator_command', c).action == REJECT


def test_operator_drone_altitude():
    d = deterministic_fallback('operator_command', ctx(operator_text='drone descend to 8 m'))
    assert (d.action, d.target_altitude_m, d.target_robot) == (ALTITUDE, 8.0, 'scout_0')
    d = deterministic_fallback('operator_command', ctx(operator_text='drone go up to 40 m'))
    assert d.target_altitude_m == 20.0
    d = deterministic_fallback('operator_command', ctx(operator_text='drone lower', fleet=NO_FLEET))
    assert d.action == CLARIFY


# -------------------------------------------------------------------------------------------------- safer-only property
OP_TEXTS = ['full speed through the boulder field', 'stop', 'slow down', 'go faster', 'drive at 0.3 m/s',
            'drive at 5 m/s', 'avoid the crater rim', 'drone descend to 7 m', 'hello?', '', 'hold', 'ram the pillar']


def random_context(rng):
    fleet = {}
    if rng.random() < 0.6:
        fleet['recovery_rover'] = rng.choice(['rover_1', 'rover_0', True])
        if rng.random() < 0.3:
            fleet['recovery_available'] = False
    if rng.random() < 0.5:
        fleet['drone'] = 'scout_0'
        fleet['drone_agl_m'] = rng.choice([None, 3.0, 12.0, 30.0])
    c = {'robot': rng.choice(['rover_0', 'rover_1', 'scout_0']),
         'robot_type': rng.choice(['rover', 'rover', 'scout']),
         'speed_limit_mps': rng.choice([None, 0.0, 0.2, 0.42, 1.0, -1.0, float('inf'), 'x']),
         'v_max_mps': rng.choice([None, 1.0]),
         'camera_conf': rng.random(), 'lidar_conf': rng.random(), 'uncertainty': rng.random(),
         'nearest_obstacle_m': rng.choice([None, 1.0, 5.0, float('inf')]), 'fleet': fleet,
         'peer': rng.choice([None, 'rover_0', 'rover_1', 'rover_2']), 'known_hazards': ['boulder field'],
         'dust': rng.random() < 0.3, 'wind_mps': rng.choice([None, 3.0, 12.0]),
         'operator_text': rng.choice(OP_TEXTS)}
    return c


def random_raw(rng):
    if rng.random() < 0.05:
        return rng.choice([None, [], 'SLOW_AND_REPLAN', 42])
    raw = {}
    if rng.random() < 0.95:
        raw['action'] = rng.choice(list(ACTIONS) * 3 + ['GO_FAST', '', None, 3, 'slow_and_replan',
                                                        'deploy recovery rover'])
    if rng.random() < 0.9:
        raw['reasoning_trace'] = rng.choice(['ok', 'x' * 1000, '', None, 'Camera 0.1, dust; slow.'])
    if rng.random() < 0.8:
        raw['target_speed_mps'] = rng.choice([None, -1, 0, 0.1, 0.5, 2, 1e9, float('inf'), float('nan'), 'fast',
                                              '0.4', True])
    if rng.random() < 0.6:
        raw['target_altitude_m'] = rng.choice([None, -5, 0, 6, 10, 25, 1e6, 'high', float('nan')])
    return raw


def test_property_validation_never_less_safe_than_local_limits():
    rng = random.Random(1234)
    for _ in range(3000):
        event = rng.choice(EVENTS)
        c = random_context(rng)
        d = validate(random_raw(rng), event, c, model='gemma-4-26b-a4b-it', tokens_in=100, tokens_out=40)
        assert_safe(d, event, c)


def test_property_fallback_is_safe_and_deterministic():
    rng = random.Random(99)
    for _ in range(1500):
        event = rng.choice(EVENTS)
        c = random_context(rng)
        d1, d2 = deterministic_fallback(event, c), deterministic_fallback(event, json.loads(json.dumps(
            c, allow_nan=True)))
        assert_safe(d1, event, c)
        assert d1.as_dict() == d2.as_dict()


# --------------------------------------------------------------------------------------------------- fallback per event
def test_fallback_low_confidence():
    d = deterministic_fallback('low_confidence', ctx(camera_conf=0.2, lidar_conf=0.5, uncertainty=0.6))
    assert d.action == SLOW and d.target_speed_mps == pytest.approx(0.35 * 0.8)
    assert d.source == 'local-fallback' and d.latency_ms is None and d.model == 'local-rules'
    d = deterministic_fallback('low_confidence', ctx(camera_conf=0.1, lidar_conf=0.15, uncertainty=0.9))
    assert d.action == CLARIFY and d.target_speed_mps == 0.0


def test_fallback_novel_hazard():
    d = deterministic_fallback('novel_hazard', ctx(hazard='falling pillar', nearest_obstacle_m=8.0))
    assert d.action == SLOW and d.target_speed_mps == pytest.approx(0.28) and 'falling pillar' in d.reasoning_trace
    d = deterministic_fallback('novel_hazard', ctx(hazard='rockslide', nearest_obstacle_m=1.5))
    assert d.action == SLOW and d.target_speed_mps == 0.0


def test_fallback_drone_in_dust_or_wind_descends():
    drone = ctx(robot='scout_0', robot_type='scout', fleet={'drone': 'scout_0', 'drone_agl_m': 12.0},
                hazard='dust storm front')
    d = deterministic_fallback('novel_hazard', drone)
    assert (d.action, d.target_altitude_m, d.target_robot) == (ALTITUDE, 8.0, 'scout_0')
    d = deterministic_fallback('low_confidence', dict(drone, hazard=None, wind_mps=11.0,
                                                      fleet={'drone': 'scout_0', 'drone_agl_m': 7.0}))
    assert d.action == ALTITUDE and d.target_altitude_m == 6.0
    # a rover in the dust slows; it does not move the drone unless told the drone is affected
    rover = ctx(hazard='dust storm', dust=True)
    assert deterministic_fallback('novel_hazard', rover).action == SLOW
    rover['fleet'] = dict(rover['fleet'], drone_affected=True)
    assert deterministic_fallback('novel_hazard', rover).action == ALTITUDE


def test_fallback_critical_damage_and_sos():
    d = deterministic_fallback('critical_damage', ctx(health=0.15, damage='wheel motor failed'))
    assert (d.action, d.target_robot, d.target_speed_mps) == (DEPLOY, 'rover_1', 0.0)
    assert '15%' in d.reasoning_trace
    d = deterministic_fallback('critical_damage', ctx(fleet=NO_FLEET))
    assert d.action == CLARIFY and d.target_speed_mps == 0.0
    d = deterministic_fallback('peer_sos', ctx(robot='rover_1', peer='rover_0'))
    assert d.action == DEPLOY and d.target_speed_mps is None
    d = deterministic_fallback('peer_sos', ctx(peer='rover_1'))
    assert d.action == CLARIFY


def test_fallback_error_and_projection_fields():
    d = deterministic_fallback('low_confidence', ctx(), error='timeout')
    assert d.error == 'timeout' and d.projected_device == 'orin_nano_15w'
    assert d.projected_fits is False and d.projected_edge_ms is None
    assert 'local rules are the on-device brain' in d.projected_note


# ---------------------------------------------------------------------------------------------------- summary
def test_summary_is_compact_and_complete():
    c = ctx(event_extra='ignored', hazard='dust storm, visibility 6 m', operator_text='hold')
    s = build_summary('low_confidence', c)
    txt = json.dumps(s, separators=(',', ':'), ensure_ascii=False)
    assert len(txt) < 500
    assert s['event'] == 'low_confidence' and s['robot'] == 'rover_0' and s['v_limit_mps'] == 0.8
    assert s['fleet'] == {'recovery_rover': 'rover_1', 'recovery_ready': True, 'drone': 'scout_0',
                          'drone_agl_m': 12.0}
    assert s['cam_conf'] == 0.8 and s['operator'] == 'hold' and 'event_extra' not in s
    assert None not in s.values()


@pytest.mark.parametrize('text,want', [
    ('resume at full speed', 1.0),          # "full speed" = v_max_mps
    ('slow down to 0.2 m/s', 0.2),
    ('stop', 0.0),
    ('go faster', 'unbounded'),
    ('drone to 8 m AGL', None),             # asks for no speed: no requested_mps
])
def test_operator_summary_carries_the_requested_speed(text, want):
    s = build_summary('operator_command', ctx(operator_text=text))
    assert s.get('requested_mps') == want and s['v_limit_mps'] == 0.8
    json.dumps(s)
    # only operator commands carry it, even when another event's context has operator text
    assert 'requested_mps' not in build_summary('novel_hazard', ctx(operator_text=text))


def test_summary_size_bound_with_huge_inputs():
    c = ctx(hazard='h' * 5000, damage='d' * 5000, operator_text='o' * 5000, state='S' * 500, robot='r' * 300,
            known_hazards=['k' * 300] * 50, peer='p' * 300)
    s = build_summary('peer_sos', c)
    assert len(json.dumps(s, separators=(',', ':'), ensure_ascii=False)) <= gb.SUMMARY_MAX_CHARS
    assert s['event'] == 'peer_sos' and 'v_limit_mps' in s


def test_summary_handles_numpy_and_percent_values():
    np = pytest.importorskip('numpy')
    s = build_summary('low_confidence', ctx(camera_conf=np.float32(0.25), battery=62, speed_mps=np.float64(0.4)))
    assert s['cam_conf'] == 0.25 and s['battery'] == 0.62 and s['v_mps'] == 0.4
    json.dumps(s)


# ---------------------------------------------------------------------------------------------------- client
def test_request_body_shape_and_header():
    cl, tr = client(ok({'action': SLOW, 'target_speed_mps': 0.2, 'reasoning_trace': 'x'}))
    d = cl.decide('low_confidence', ctx(hazard='dust storm'))
    assert d.source == 'gemma'
    call = tr.calls[0]
    assert call['url'] == ('https://generativelanguage.googleapis.com/v1beta/models/'
                           'gemma-4-26b-a4b-it:generateContent')
    assert call['headers']['x-goog-api-key'] == KEY
    assert call['headers']['Content-Type'] == 'application/json'
    assert call['timeout'] == 8.0
    body = call['body']
    assert set(body) == {'contents', 'generationConfig', 'systemInstruction'}
    assert len(body['contents']) == 1 and body['contents'][0]['role'] == 'user'
    text = body['contents'][0]['parts'][0]['text']
    summary = json.loads(text.split('Event summary (JSON): ', 1)[1])
    assert summary == build_summary('low_confidence', ctx(hazard='dust storm'))
    assert 'REJECT_UNSAFE_COMMAND' in body['systemInstruction']['parts'][0]['text']
    gc = body['generationConfig']
    assert gc['responseMimeType'] == 'application/json'
    assert gc['temperature'] == 0.2 and 100 <= gc['maxOutputTokens'] <= 256
    assert 'thinkingConfig' not in gc and 'thinkingConfig' not in body
    schema = gc['responseSchema']
    assert schema['type'] == 'OBJECT' and schema['required'] == ['action', 'reasoning_trace']
    assert set(schema['properties']) == {'action', 'target_speed_mps', 'target_altitude_m', 'reasoning_trace'}
    assert schema['properties']['action']['type'] == 'STRING'
    assert schema['properties']['target_speed_mps']['type'] == 'NUMBER'
    assert schema['properties']['action']['enum'] == list(allowed_actions('low_confidence', ctx()))
    assert set(schema['properties']['action']['enum']) <= set(ACTIONS)


def test_inline_instructions_when_system_instruction_rejected():
    rejected = (400, {}, {'error': {'code': 400, 'message': 'Developer instruction is not enabled for this model'}})
    cl, tr = client(rejected, ok({'action': SLOW, 'reasoning_trace': 'x'}))
    d = cl.decide('novel_hazard', ctx())
    assert d.source == 'gemma' and len(tr.calls) == 2
    assert 'systemInstruction' in tr.calls[0]['body'] and 'systemInstruction' not in tr.calls[1]['body']
    assert tr.calls[1]['body']['contents'][0]['parts'][0]['text'].startswith(gb.INSTRUCTIONS)
    assert cl.use_system_instruction is False


def test_key_never_leaks():
    leaky = (400, {}, {'error': {'message': f'API key {KEY} not valid'}})
    cl, tr = client(leaky)
    assert KEY not in repr(cl) and KEY not in str(cl) and 'key=set' in repr(cl)
    with pytest.raises(GemmaError) as ei:
        cl.call('novel_hazard', ctx())
    exc = ei.value
    assert KEY not in str(exc) and KEY not in repr(exc) and exc.__context__ is None and exc.__cause__ is None
    assert KEY not in tr.calls[0]['url'] and KEY not in json.dumps(tr.calls[0]['body'])
    d = cl.decide('novel_hazard', ctx())
    assert d.source == 'local-fallback' and 'HTTP 400' in d.error
    assert KEY not in json.dumps(d.as_dict())

    def boom():
        raise OSError(f'connection reset while sending {KEY}')
    cl2, _ = client(boom)
    with pytest.raises(GemmaError) as ei:
        cl2.call('novel_hazard', ctx())
    assert KEY not in str(ei.value) and ei.value.__context__ is None
    assert KEY not in json.dumps(cl2.decide('novel_hazard', ctx()).as_dict())


def test_timeout_becomes_fallback():
    def slow():
        raise TimeoutError('The read operation timed out')
    cl, _ = client(slow)
    d = cl.decide('novel_hazard', ctx())
    assert d.source == 'local-fallback' and 'timeout after 8 s' in d.error and d.latency_ms is None


def test_no_key_means_fallback_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    tr = FakeTransport(ok({'action': SLOW, 'reasoning_trace': 'x'}))
    cl = GemmaClient(transport=tr, env_path=tmp_path / 'missing.env')
    assert not cl.has_key and 'key=missing' in repr(cl)
    d = cl.decide('low_confidence', ctx())
    assert d.source == 'local-fallback' and 'no API key' in d.error and tr.calls == []


def test_response_parsing():
    r = parse_response(gemma_payload(None, text='```json\n{"action": "SLOW_AND_REPLAN", "reasoning_trace": "x"}\n```'),
                       latency_ms=12.0)
    assert r.raw == {'action': SLOW, 'reasoning_trace': 'x'} and (r.tokens_in, r.tokens_out) == (140, 56)
    multi = gemma_payload(None)
    multi['candidates'][0]['content']['parts'] = [{'text': 'thinking...', 'thought': True},
                                                  {'text': '{"action": "REQUEST_CLARIFICATION",'},
                                                  {'text': ' "reasoning_trace": "unclear"}'}]
    assert parse_response(multi).raw == {'action': CLARIFY, 'reasoning_trace': 'unclear'}
    r = parse_response({'promptFeedback': {'blockReason': 'SAFETY'}, 'usageMetadata': {'promptTokenCount': 9}})
    assert r.raw is None and 'blocked: SAFETY' in r.error and r.tokens_in == 9
    trunc = gemma_payload(None, text='{"action": "SLOW_AND_REPLAN", "reasoning_trace": "the dust')
    trunc['candidates'][0]['finishReason'] = 'MAX_TOKENS'
    r = parse_response(trunc)
    assert r.raw is None and 'MAX_TOKENS' in r.error
    assert parse_response('nope').raw is None


def test_decide_uses_reported_model_tokens_and_measured_latency():
    cl, _ = client(ok({'action': SLOW, 'target_speed_mps': 0.1, 'reasoning_trace': 'x'}, tin=141, tout=57,
                      model='gemma-4-26b-a4b-it-001'), delay_s=0.03)
    d = cl.decide('low_confidence', ctx(edge_profile='agx_orin_60w'))
    assert d.source == 'gemma' and d.model == 'gemma-4-26b-a4b-it-001'
    assert (d.tokens_in, d.tokens_out) == (141, 57)
    assert d.latency_ms >= 25.0                        # measured, not hard-coded
    assert d.projected_device == 'agx_orin_60w' and d.projected_fits is True
    p = project_latency(141, 57, 'agx_orin_60w', 'gemma-4-26b-a4b-it')
    assert d.projected_edge_ms == p['total_ms'] and 'roofline estimate, not measured' in d.projected_note
    d = cl.decide('low_confidence', ctx())             # default profile: Orin Nano 8 GB
    assert d.projected_fits is False and d.projected_edge_ms is None
    assert 'local rules are the on-device brain' in d.projected_note


@pytest.mark.parametrize('payload', [b'not json at all', {'candidates': []},
                                     gemma_payload(None, text='Sure! I would slow down.'),
                                     gemma_payload({'action': 'TELEPORT', 'reasoning_trace': 'x'})])
def test_malformed_answers_fall_back(payload):
    cl, _ = client((200, {}, payload))
    d = cl.decide('novel_hazard', ctx())
    assert d.source == 'local-fallback' and d.error
    assert_safe(d, 'novel_hazard', ctx())


def test_http_errors_fall_back():
    for status in (401, 403, 404, 500):
        cl, _ = client((status, {}, {'error': {'message': 'boom'}}))
        d = cl.decide('novel_hazard', ctx())
        assert d.source == 'local-fallback' and f'HTTP {status}' in d.error


def test_rate_limit_raises_with_retry_after():
    cl, _ = client((429, {'Retry-After': '7'}, {'error': {'message': 'quota'}}))
    with pytest.raises(gb.GemmaRateLimited) as ei:
        cl.call('novel_hazard', ctx())
    assert ei.value.status == 429 and ei.value.retry_after_s == 7.0
    body = {'error': {'code': 429, 'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo',
                                                 'retryDelay': '39s'}]}}
    assert gb._retry_after({}, json.dumps(body).encode()) == 39.0
    assert gb._retry_after({}, b'') is None
    d = client((503, {}, b''))[0].decide('novel_hazard', ctx())
    assert d.source == 'local-fallback' and 'rate-limited (HTTP 503)' in d.error


# ---------------------------------------------------------------------------------------------------- API key
def test_load_api_key(monkeypatch, tmp_path):
    env = tmp_path / '.env'
    env.write_text('# secrets\nOTHER=1\nexport GEMINI_API_KEY="abc123"  \n')
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    assert load_api_key(env) == 'abc123'
    env.write_text("GEMINI_API_KEY=xyz # comment\n")
    assert load_api_key(env) == 'xyz'
    env.write_text('GEMINI_API_KEY=\n')
    assert load_api_key(env) is None
    assert load_api_key(tmp_path / 'nope.env') is None
    monkeypatch.setenv('GEMINI_API_KEY', 'from-env')
    assert load_api_key(env) == 'from-env'
    monkeypatch.delenv('GEMINI_API_KEY')
    monkeypatch.setenv('REDSHIFT_ENV_FILE', str(tmp_path / 'custom.env'))
    (tmp_path / 'custom.env').write_text('GEMINI_API_KEY=custom\n')
    assert load_api_key() == 'custom'


# ---------------------------------------------------------------------------------------------------- async brain
GOOD = ok({'action': SLOW, 'target_speed_mps': 0.2, 'reasoning_trace': 'x'})


def test_submit_never_blocks_on_a_slow_call():
    gate = threading.Event()
    cl, tr = client(GOOD, gate=gate)
    got = []
    with AgentBrain(cl, on_decision=got.append) as brain:
        t0 = time.perf_counter()
        rid = brain.submit('low_confidence', ctx())
        rid2 = brain.submit('novel_hazard', ctx())
        assert time.perf_counter() - t0 < 0.05
        assert rid and rid2 and rid != rid2
        assert brain.poll() == [] and brain.stats()['in_flight'] == 2
        gate.set()
        out = wait_for(brain, 2)
    assert {d.request_id for d in out} == {rid, rid2}
    assert all(d.source == 'gemma' and not d.stale and d.robot == 'rover_0' for d in out)
    assert {d.request_id for d in got} == {rid, rid2}


def test_dedupe_inside_min_interval():
    gate = threading.Event()
    clock = FakeClock()
    cl, tr = client(GOOD, gate=gate)
    brain = AgentBrain(cl, clock=clock, min_interval_s={'low_confidence': 3.0, 'novel_hazard': 1.0})
    try:
        rid = brain.submit('low_confidence', ctx())
        assert brain.submit('low_confidence', ctx()) == rid          # still in flight → same id
        other = brain.submit('low_confidence', ctx(robot='rover_1'))
        assert other not in (None, rid)
        gate.set()
        wait_for(brain, 2)
        clock.t += 1.0
        assert brain.submit('low_confidence', ctx()) is None           # finished, still inside 3 s
        clock.t += 2.5
        assert brain.submit('low_confidence', ctx()) not in (None, rid)
        # operator commands with different text are different requests
        a = brain.submit('operator_command', ctx(operator_text='stop'))
        b = brain.submit('operator_command', ctx(operator_text='slow down'))
        assert a and b and a != b
        assert brain.stats()['deduped'] == 2
        wait_for(brain, 3)
        assert len(tr.calls) == 5
    finally:
        brain.close()


def test_context_is_snapshotted():
    gate = threading.Event()
    cl, tr = client(GOOD, gate=gate)
    with AgentBrain(cl) as brain:
        c = ctx(hazard='rockslide')
        brain.submit('novel_hazard', c)
        c['hazard'] = 'changed after submit'
        c['fleet']['drone'] = None
        gate.set()
        wait_for(brain, 1)
    s = json.loads(tr.calls[0]['body']['contents'][0]['parts'][0]['text'].split(': ', 1)[1])
    assert s['hazard'] == 'rockslide' and s['fleet']['drone'] == 'scout_0'


def test_rate_limit_backoff_with_retry_after():
    clock = FakeClock()
    cl, tr = client((429, {'Retry-After': '5'}, {'error': {'message': 'quota'}}), GOOD)
    with AgentBrain(cl, clock=clock, min_interval_s=0.0) as brain:
        brain.submit('novel_hazard', ctx())
        d = wait_for(brain, 1)[0]
        assert d.source == 'local-fallback' and 'rate-limited (HTTP 429)' in d.error and 'backing off 5 s' in d.error
        assert brain.backoff_remaining_s() == pytest.approx(5.0)
        clock.t += 1.0
        t0 = time.perf_counter()
        brain.submit('low_confidence', ctx())
        out = brain.poll()                           # answered synchronously, no HTTP call
        assert time.perf_counter() - t0 < 0.05 and len(out) == 1 and len(tr.calls) == 1
        assert out[0].source == 'local-fallback' and out[0].error == 'rate-limited (backing off)'
        clock.t += 4.5
        brain.submit('low_confidence', ctx())
        d = wait_for(brain, 1)[0]
        assert d.source == 'gemma' and len(tr.calls) == 2


def test_rate_limit_exponential_backoff_caps_at_30s():
    clock = FakeClock()
    cl, tr = client((429, {}, b''))
    delays = []
    with AgentBrain(cl, clock=clock, min_interval_s=0.0) as brain:
        for _ in range(6):
            brain.submit('novel_hazard', ctx())
            wait_for(brain, 1)
            delays.append(round(brain.backoff_remaining_s(), 3))
            clock.t += delays[-1] + 0.01
    assert delays == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_stale_results_are_flagged_but_delivered():
    clock = FakeClock()

    def answer_late():
        clock.t += 12.0                                 # the call "took" 12 s of brain-clock time
        return 200, {}, json.dumps(gemma_payload({'action': SLOW, 'reasoning_trace': 'late'})).encode()
    cl, _ = client(answer_late)
    with AgentBrain(cl, clock=clock, deadline_s=10.0) as brain:
        brain.submit('novel_hazard', ctx())
        d = wait_for(brain, 1)[0]
    assert d.stale is True and d.age_ms == pytest.approx(12000.0) and d.source == 'gemma'


def test_brain_without_client_or_key_answers_locally():
    brain = AgentBrain(None)
    rid = brain.submit('critical_damage', ctx(health=0.1))
    out = brain.poll()
    assert rid and len(out) == 1 and out[0].action == DEPLOY and out[0].source == 'local-fallback'
    brain.close()
    brain = AgentBrain(GemmaClient(api_key='', env_path='/nonexistent/.env', transport=FakeTransport(GOOD)))
    brain.client._key = None
    brain.submit('low_confidence', ctx())
    assert 'no API key' in brain.poll()[0].error
    brain.close()


def test_worker_exception_still_returns_a_decision():
    class Broken(GemmaClient):
        def decide(self, *a, **kw):
            raise ZeroDivisionError
    with AgentBrain(Broken(api_key=KEY, transport=FakeTransport(GOOD))) as brain:
        brain.submit('novel_hazard', ctx())
        d = wait_for(brain, 1)[0]
    assert d.source == 'local-fallback' and 'internal error: ZeroDivisionError' in d.error


def test_bad_callback_does_not_kill_the_worker():
    def bad(_):
        raise RuntimeError('ui gone')
    cl, _ = client(GOOD)
    with AgentBrain(cl, on_decision=bad) as brain:
        brain.submit('novel_hazard', ctx())
        assert len(wait_for(brain, 1)) == 1
        assert brain.stats()['callback_errors'] == 1


def test_close_is_clean_and_idempotent():
    cl, _ = client(GOOD, delay_s=0.05)
    brain = AgentBrain(cl)
    brain.submit('novel_hazard', ctx())
    brain.close()
    brain.close()
    assert brain.submit('low_confidence', ctx()) is None
    assert not any(t.name.startswith('gemma-brain') and t.is_alive() for t in threading.enumerate())


# ---------------------------------------------------------------------------------------------------- projection
def test_device_ids_match_edge_profiles():
    assert set(gb.DEVICES) == set(PROFILES)


def test_projection_fit_rules():
    assert not project_latency(140, 56, 'orin_nano_15w', 'gemma-4-26b-a4b-it')['fits']
    assert not project_latency(140, 56, 'orin_nx_25w', 'gemma-4-26b-a4b-it')['fits']
    assert project_latency(140, 56, 'agx_orin_60w', 'gemma-4-26b-a4b-it')['fits']
    assert not project_latency(140, 56, 'orin_nx_25w', 'gemma-4-31b-it')['fits']
    assert project_latency(140, 56, 'agx_orin_60w', 'gemma-4-31b-it')['fits']
    assert project_latency(140, 56, 'orin_nano_15w', 'gemma-edge-4b')['fits']
    assert project_latency(140, 56, 'rpi5_8w', 'gemma-edge-4b')['fits']
    p = project_latency(140, 56, 'orin_nano_15w', 'gemma-4-26b-a4b-it')
    assert p['weights_gb'] == 13.0 and p['required_gb'] == 14.5 and p['usable_gb'] == 6.4
    with pytest.raises(ValueError):
        project_latency(1, 1, 'jetson_tx1', 'gemma-4-26b-a4b-it')
    with pytest.raises(ValueError):
        project_latency(1, 1, 'agx_orin_60w', 'llama-9000')


def test_projection_math_and_assumptions():
    p = project_latency(140, 56, 'agx_orin_60w', 'gemma-4-26b-a4b-it')
    decode_tok_s = 0.6 * 204.8e9 / (4e9 * 0.5)                # 61.44 tok/s (active params only for MoE)
    prefill_tok_s = 0.3 * 137e12 / (2 * 4e9)                  # 5137.5 tok/s
    assert p['decode_tok_s'] == pytest.approx(decode_tok_s, abs=0.01)
    assert p['prefill_tok_s'] == pytest.approx(prefill_tok_s, abs=0.1)
    assert p['prefill_ms'] == pytest.approx(140 / prefill_tok_s * 1000, abs=0.1)
    assert p['decode_ms'] == pytest.approx(56 / decode_tok_s * 1000, abs=0.1)
    assert p['total_ms'] == pytest.approx(p['prefill_ms'] + p['decode_ms'], abs=0.11)
    assert p['estimate'] is True and p['device'] == 'agx_orin_60w' and p['model'] == 'gemma-4-26b-a4b-it'
    a = p['assumptions']
    for k in ('bytes_per_param', 'runtime_overhead_gb', 'usable_ram_fraction', 'decode_bandwidth_efficiency',
              'prefill_compute_efficiency', 'device_bandwidth_gbs', 'device_int8_dense_tops',
              'model_active_params_b', 'model_total_params_b'):
        assert k in a
    assert 'not measured' in a['note']
    dense = project_latency(140, 56, 'agx_orin_60w', 'gemma-4-31b-it')
    assert dense['decode_ms'] > 7 * p['decode_ms']             # 31 B active vs 4 B active


def test_projection_monotonic_in_tokens():
    for dev in gb.DEVICES:
        for model in gb.MODELS:
            prev = -1.0
            for n in range(0, 2001, 50):
                t = project_latency(n, 56, dev, model)['total_ms']
                assert t >= prev
                prev = t
            prev = -1.0
            for n in range(0, 501, 10):
                t = project_latency(140, n, dev, model)['total_ms']
                assert t >= prev
                prev = t
            assert project_latency(1000, 56, dev, model)['total_ms'] > project_latency(10, 56, dev, model)['total_ms']
            assert project_latency(140, 400, dev, model)['total_ms'] > project_latency(140, 4, dev, model)['total_ms']


def test_projection_shows_edge_reference_is_slower_on_small_devices():
    nano = project_latency(140, 56, 'orin_nano_15w', 'gemma-edge-4b')
    agx = project_latency(140, 56, 'agx_orin_60w', 'gemma-edge-4b')
    rpi = project_latency(140, 56, 'rpi5_8w', 'gemma-edge-4b')
    assert rpi['total_ms'] > nano['total_ms'] > agx['total_ms'] > 0


def test_full_speed_rounded_up_by_gemma_is_clamped_not_rejected():
    # the operator asks for full speed at the full local limit (0.826 m/s at 2.2 g); Gemma answers with two decimals,
    # 0.83: that is its rounding, not a request above the limit, so the cap is clamped to the limit, not rejected
    ctx = {'robot': 'rover_0', 'speed_limit_mps': 0.826, 'v_max_mps': 0.826, 'operator_text': 'resume at full speed'}
    d = gb.validate({'action': 'SLOW_AND_REPLAN', 'target_speed_mps': 0.83, 'reasoning_trace': 'r'},
                    'operator_command', ctx)
    assert d.action == 'SLOW_AND_REPLAN' and d.source == 'gemma' and d.target_speed_mps == 0.826
    # a real excess is still refused
    d = gb.validate({'action': 'SLOW_AND_REPLAN', 'target_speed_mps': 1.2, 'reasoning_trace': 'r'},
                    'operator_command', ctx)
    assert d.action == 'REJECT_UNSAFE_COMMAND' and d.target_speed_mps is None
