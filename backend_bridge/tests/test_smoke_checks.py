"""The smoke test's verdicts (backend_bridge/scripts/smoke_test.py), checked without a backend: the advisory-brain
predicates, the rescue-room map check, and the Phase 3 pillar / storm helpers and sequences (on a scripted backend)."""
import asyncio
import importlib.util
import math
import time
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'redshift_smoke_test', Path(__file__).resolve().parents[1] / 'scripts' / 'smoke_test.py')
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def _nh(**kw):
    return dict({'robot': 'rover_0', 'event': 'novel_hazard', 'action': 'SLOW_AND_REPLAN', 'trigger': 'path_blocked',
                 'trigger_input': 'Planned path crosses new lethal cells'}, **kw)


# ---------------------------------------------------------------------------------------------------- B2 rockslide
@pytest.mark.parametrize('d', [
    _nh(), _nh(trigger='hazard_marked', trigger_input='Permanent hazard'),
    _nh(trigger='slope_hazard', trigger_input='Boulder field 6.8 m ahead'),        # the text names the slide
    _nh(trigger='slope_hazard', trigger_input='LiDAR OBSTACLE in corridor'),
    _nh(trigger=None, trigger_input='rockslide debris'),
])
def test_rockslide_decision_accepts_the_slide_triggers(d):
    assert smoke.rockslide_decision(d, 'rover_0') is True


@pytest.mark.parametrize('d', [
    _nh(trigger='slope_hazard', trigger_input='LiDAR ground slope 24.0° in the corridor ahead'),   # another hazard
    _nh(trigger='slope_hazard', trigger_input=None),
    _nh(robot='rover_1'),                                                             # another rover
    _nh(event='low_confidence'), _nh(event='operator_command', trigger='operator', trigger_input='boulder?'),
    'not a dict', None,
])
def test_rockslide_decision_rejects_unrelated_decisions(d):
    assert smoke.rockslide_decision(d, 'rover_0') is False


def test_rockslide_window_starts_at_the_injection():
    t = 1000.0
    early, late = (t - 0.2, _nh(request_id='early')), (t + 1.0, _nh(request_id='late'))
    pred = lambda d: smoke.rockslide_decision(d, 'rover_0')      # noqa: E731
    # the rockslide check passes no slack: a decision received 0.2 s before the injection is not its answer
    assert smoke.first_decision([early, late], t, pred) == late
    assert smoke.first_decision([early], t, pred) is None
    assert smoke.first_decision([(t, _nh())], t, pred) == (t, _nh())                # "at" the injection counts
    # the other brain checks keep their 0.5 s receipt slack
    assert smoke.first_decision([early, late], t, pred, slack=0.5) == early
    slope = (t + 0.5, _nh(trigger='slope_hazard', trigger_input='LiDAR ground slope 24°'))
    assert smoke.first_decision([slope, late], t, pred) == late


# ---------------------------------------------------------------------------------------------------- B2 cap applied
def _rover(state='EXPLORING', path=10, **cap):
    c = dict({'speed_mps': 0.2, 'action': 'SLOW_AND_REPLAN', 'source': 'gemma', 'operator': True,
              'remaining_s': 59.0, 'applied': True}, **cap) if cap.get('speed_mps', 0.2) is not None else None
    return {'state': state, 'path': [[float(i), 0.0] for i in range(path)], 'brain_cap': c, 'speed': 0.19}


def test_cap_verdict_waits_until_a_low_enough_cap_is_shown():
    assert smoke.cap_verdict({'state': 'EXPLORING', 'brain_cap': None}, 0.2)[0] is None
    assert smoke.cap_verdict(_rover(speed_mps=0.35), 0.2)[0] is None
    assert smoke.cap_verdict(_rover(speed_mps='0.1'), 0.2)[0] is None
    assert smoke.cap_verdict(_rover(speed_mps=True), 0.2)[0] is None


@pytest.mark.parametrize('state', ['EXPLORING', 'CAUTION', 'RESCUE_TRANSIT', 'RESCUE_DOCKING'])
def test_cap_verdict_requires_applied_while_following_or_docking(state):
    ok, why = smoke.cap_verdict(_rover(state), 0.2)
    assert ok is True and 'applied' in why
    ok, why = smoke.cap_verdict(_rover(state, applied=False), 0.2)
    assert ok is False and 'not applied although' in why and state in why
    ok, why = smoke.cap_verdict(_rover(state, applied=None), 0.2)             # the planner must say
    assert ok is False and '"applied"' in why
    c = _rover(state)
    del c['brain_cap']['applied']
    assert smoke.cap_verdict(c, 0.2)[0] is False


@pytest.mark.parametrize('st', [_rover('REPLANNING', applied=False), _rover('SAFE_HOLD', applied=False),
                                _rover('EXPLORING', path=0, applied=False), _rover('REPAIRING', applied=False)])
def test_cap_verdict_accepts_not_applied_only_outside_follow_and_dock(st):
    ok, why = smoke.cap_verdict(st, 0.2)
    assert ok is True and 'not applied: not following / docking' in why and str(st['state']) in why


def test_operator_cap_cleared():
    assert smoke.operator_cap_cleared({'brain_cap': None}) is True
    assert smoke.operator_cap_cleared({}) is True
    assert smoke.operator_cap_cleared(_rover(operator=False)) is True                # an event cap binds
    assert smoke.operator_cap_cleared(_rover(speed_mps=0.0, operator=True)) is False  # the stop is still set


def _meta():
    return {'size_m': 100.0, 'res': 0.5, 'origin': [-50.0, -50.0]}


def test_rescue_room_needs_a_stand_off_the_rescuer_can_reach():
    import numpy as np
    cls = np.ones((200, 200), int)                       # everything mapped FREE
    ok, why = smoke.rescue_room(cls, _meta(), (0.0, 0.0), (20.0, 20.0))
    assert ok and why.endswith('reachable by the rescuer') and not why.startswith('0 ')
    # 2.6 m from the west edge: the whole ring is inside the 3 m edge band + inflation on that side, but not all of it
    assert smoke.rescue_room(cls, _meta(), (-47.4, -24.7), (20.0, 20.0))[0]
    # a wall of boulders between the rover and the rescuer: free stand-off cells exist, none reachable
    walled = cls.copy()
    walled[:, 110] = 6
    ok, why = smoke.rescue_room(walled, _meta(), (0.0, 0.0), (20.0, 20.0))
    assert not ok and ', 0 reachable' in why
    # enclosed by hazards all round
    pocket = cls.copy()
    pocket[90:111, 90:111] = 3
    pocket[95:106, 95:106] = 1
    assert not smoke.rescue_room(pocket, _meta(), (0.25, 0.25), (20.0, 20.0))[0]


def test_map_classes_decodes_the_ws_layer_image():
    import base64

    import cv2
    import numpy as np
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 1] = 40 * np.arange(4)[None, :]             # classes 0..3 along x
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    cls = smoke.map_classes(base64.b64encode(buf.tobytes()).decode())
    assert cls.tolist() == [[0, 1, 2, 3]] * 4


# ---------------------------------------------------------------------------------------------------- phase 3 pillar
def test_pillar_decision_accepts_the_block_and_texts_naming_the_trunk():
    assert smoke.pillar_decision(_nh(), 'rover_0') is True                                   # path_blocked
    assert smoke.pillar_decision(_nh(trigger='hazard_marked'), 'rover_0') is True
    assert smoke.pillar_decision(_nh(trigger='slope_hazard', trigger_input='Fallen PILLAR 7 m ahead'), 'rover_0')
    assert smoke.pillar_decision(_nh(trigger=None, trigger_input='12 obstacle cells ahead'), 'rover_0')
    assert not smoke.pillar_decision(_nh(trigger='slope_hazard', trigger_input='LiDAR ground slope 24°'), 'rover_0')
    assert not smoke.pillar_decision(_nh(robot='rover_1'), 'rover_0')
    assert not smoke.pillar_decision(_nh(event='low_confidence'), 'rover_0')
    assert not smoke.pillar_decision(None, 'rover_0')
    # the rockslide predicate is unchanged by the shared helper
    assert smoke.hazard_decision(_nh(trigger=None, trigger_input='rockslide debris'), 'rover_0',
                                 smoke.ROCKSLIDE_WORDS)
    assert not smoke.hazard_decision(_nh(trigger=None, trigger_input='rockslide debris'), 'rover_0', ('pillar',))


def _blk(t, inp, robot='rover_0', kind='path_blocked'):
    return {'robot': robot, 'kind': kind, 't': t, 'state': 'REPLANNING', 'input': inp}


def test_pillar_block_prefers_the_trace_that_names_the_pillar():
    t0 = 100.0
    early = _blk(t0 - 2.0, '9 obstacle cells 4.0 m ahead; camera confirms a fallen pillar')     # before the inject
    rocks = _blk(t0 + 3.0, '6 obstacle cells 5.1 m ahead across path; camera confirms 2 boulders')
    pillar = _blk(t0 + 6.0, '11 obstacle cells 4.2 m ahead across path; camera confirms a fallen pillar')
    other = _blk(t0 + 1.0, 'camera confirms a fallen pillar', robot='rover_1')
    assert smoke.pillar_block([early, other, rocks, pillar], 'rover_0', t0) is pillar
    assert smoke.pillar_block([early, other, rocks], 'rover_0', t0) is rocks                  # any block, else
    assert smoke.pillar_block([early, other, _blk(t0 + 1, 'x', kind='replanned')], 'rover_0', t0) is None
    assert smoke.names_pillar(pillar) and not smoke.names_pillar(rocks) and not smoke.names_pillar(None)


def _frames(*items):
    return [(t, [{'cls': c, 'conf': 0.86, 'range_m': 7.4, 'world_xy': [1.0, 2.0]} for c in cls]) for t, cls in items]


def test_first_sighting_and_class_counts_since_the_inject():
    fr = _frames((99.0, ['PILLAR']), (100.5, ['ROCK', 'ROCK']), (101.0, ['ROCK', 'PILLAR']), (102.0, ['PILLAR']))
    t, d = smoke.first_sighting(fr, 'PILLAR', 100.0)
    assert t == 101.0 and d['cls'] == 'PILLAR'
    assert smoke.first_sighting(fr, 'CRATER', 100.0) is None
    assert smoke.first_sighting([(101.0, ['junk', None])], 'PILLAR', 100.0) is None              # malformed entries
    assert smoke.class_counts(fr, 100.0) == {'ROCK': 2, 'PILLAR': 2}                             # frames, not boxes


def test_pillar_verdict_needs_a_block_and_the_pillar_identified():
    t0 = 100.0
    named = _blk(t0 + 6.0, '11 obstacle cells; camera confirms a fallen pillar')
    lidar = _blk(t0 + 2.6, '10 obstacle cells 6.4 m ahead across path (LiDAR step > 0.30 m); camera confirms 1 boulder')
    seen = (t0 + 4.5, {'cls': 'PILLAR', 'conf': 0.86, 'range_m': 5.4})
    ok, why = smoke.pillar_verdict(named, None, t0)
    assert ok and 'fallen pillar' in why and '6.0 s' in why
    ok, why = smoke.pillar_verdict(named, seen, t0)
    assert ok and 'PILLAR identified 4.5 s' in why
    ok, why = smoke.pillar_verdict(lidar, seen, t0)               # LiDAR blocks first, the camera names it after
    assert ok and why.index('path_blocked 2.6 s') < why.index('PILLAR identified 4.5 s') and '5.4 m' in why
    later = _blk(t0 + 7.0, '8 obstacle cells 3.9 m ahead across path')
    ok, why = smoke.pillar_verdict(later, seen, t0)               # or the other way round
    assert ok and why.index('PILLAR identified') < why.index('path_blocked 7.0 s')
    ok, why = smoke.pillar_verdict(None, seen, t0)                 # a sighting alone does not block anything
    assert not ok and 'no path_blocked' in why and 'PILLAR identified' in why
    ok, why = smoke.pillar_verdict(lidar, None, t0)                # a block nobody identified
    assert not ok and 'without naming the pillar' in why and 'no PILLAR detection on its footprint' in why
    ok, why = smoke.pillar_verdict(None, None, t0)
    assert not ok and 'no path_blocked' in why


def test_route_ahead_needs_the_route_in_front_of_the_camera():
    east = [[float(i), 0.0] for i in range(1, 12)]
    assert smoke.route_ahead(east, {'x': 0.0, 'y': 0.0, 'yaw': 0.0})
    assert smoke.route_ahead(east, {'x': 0.0, 'y': 0.0, 'yaw': math.radians(20)})
    assert not smoke.route_ahead(east, {'x': 0.0, 'y': 0.0, 'yaw': math.radians(30)})
    assert not smoke.route_ahead(east, {'x': 0.0, 'y': 0.0, 'yaw': math.pi})        # facing away
    back = [[-1.0, 0.0], [-2.0, 0.0], [-2.0, 1.0], [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert not smoke.route_ahead(back, {'x': 0.0, 'y': 0.0, 'yaw': 0.0})            # turns back first
    assert not smoke.route_ahead(east[:5], {'x': 0.0, 'y': 0.0, 'yaw': 0.0})        # shorter than 8 m
    assert not smoke.route_ahead(east, {'x': 0.0, 'y': 0.0}) and not smoke.route_ahead(None, None)


def test_on_footprint_is_the_distance_to_the_fallen_trunk():
    ev = {'base': [0.0, 0.0], 'tip': [6.0, 0.0]}
    assert smoke.on_footprint([3.0, 2.4], ev) and smoke.on_footprint([-2.0, 0.0], ev)
    assert not smoke.on_footprint([3.0, 2.6], ev) and not smoke.on_footprint([9.0, 0.0], ev)
    assert not smoke.on_footprint(None, ev) and not smoke.on_footprint([1.0, 1.0], {})


def test_descent_verdict_needs_the_drone_to_come_down_from_above_the_limit():
    t0 = 100.0
    ok, why = smoke.descent_verdict(12.0, (112.0, 8.1), t0, 9.5)
    assert ok and '12.0 s: 12.0 → 8.1 m AGL' in why
    ok, why = smoke.descent_verdict(8.4, (101.0, 8.4), t0, 9.5)   # already low: nothing to show
    assert not ok and 'not above 9.5 m' in why
    ok, why = smoke.descent_verdict(None, None, t0, 9.5)
    assert not ok and 'AGL None m' in why
    ok, why = smoke.descent_verdict(12.0, None, t0, 9.5)
    assert not ok and 'never ≤ 9.5 m' in why


def test_path_distance():
    assert smoke.path_distance([3.0, 4.0], [[0.0, 0.0], [10.0, 10.0]]) == pytest.approx(5.0)
    assert smoke.path_distance(None, [[0.0, 0.0]]) is None
    assert smoke.path_distance([1.0, 1.0], []) is None


# ---------------------------------------------------------------------------------------------------- phase 3 storm
def test_values_since_and_median():
    s = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]
    assert smoke.values_since(s, 2.0) == [20.0, 30.0, 40.0]
    assert smoke.values_since(s, 2.0, 4.0) == [20.0, 30.0]
    assert smoke.median([3.0, 1.0, 2.0]) == 2.0 and smoke.median([]) is None


def test_solar_drop_compares_medians_so_attitude_swings_cancel():
    before = [182.0, 140.0, 205.0, 190.0, 178.0]           # the panel tilts as the rover drives: one low sample
    drop, why = smoke.solar_drop(before, [27.0, 31.0, 22.0])
    assert drop == pytest.approx(1 - 27.0 / 182.0) and drop >= smoke.SOLAR_DROP_MIN
    assert '182 W' in why and '27 W' in why and '85 %' in why
    drop, _ = smoke.solar_drop(before, [140.0, 60.0, 50.0])                  # still ramping down
    assert drop < smoke.SOLAR_DROP_MIN
    assert smoke.solar_drop([], [20.0])[0] is None
    assert smoke.solar_drop(before, [])[0] is None
    drop, why = smoke.solar_drop([5.0, 6.0], [0.5])                           # no sunlit reference to drop from
    assert drop is None and 'baseline' in why


def _ramp(t0, dur, peak=1.0, dt=0.5, ramp=1.5):
    """mission.env dust level samples: ramps up over `ramp` s, holds `dur` s, ramps down."""
    out, t = [], t0
    while t < t0 + dur + ramp + 3.0:
        if t < t0 + ramp:
            v = peak * (t - t0) / ramp
        elif t < t0 + dur:
            v = peak
        else:
            v = max(0.0, peak * (1 - (t - t0 - dur) / ramp))
        out.append((t, round(v, 3)))
        t += dt
    return out


def test_storm_end_is_read_from_the_falling_dust_level():
    lv = _ramp(100.0, 45.0)
    end = smoke.storm_end(lv, 100.0)
    assert end is not None and 145.0 <= end <= 145.6
    assert smoke.storm_end([x for x in lv if x[0] < 145.0], 100.0) is None                      # still blowing
    assert smoke.storm_end(_ramp(100.0, 45.0, peak=0.3), 100.0) is None                       # never a storm
    # an earlier, weaker anomaly before t0 is not this storm's end
    assert smoke.storm_end(_ramp(0.0, 10.0, peak=0.85) + lv, 100.0) == end


def test_telemetry_values_and_the_recorded_series():
    st = {'health': {'solar_w': 182.4}, 'uncertainty': {'camera_conf': 0.91}, 'altitude_agl': 11.9,
          'wind_mps': None, 'detections': [{'cls': 'PILLAR', 'conf': 0.8, 'range_m': 6.0, 'world_xy': [1, 2]}]}
    assert smoke.telemetry_values(st) == {'solar_w': 182.4, 'camera_conf': 0.91, 'altitude_agl': 11.9,
                                          'wind_mps': None}
    assert smoke.telemetry_values({'health': None, 'uncertainty': {'camera_conf': True}}) == {
        'solar_w': None, 'camera_conf': None, 'altitude_agl': None, 'wind_mps': None}
    s = smoke.Stream()
    s.feed({'type': 'telemetry', 'mission': {'env': {'dust_level': 0.4}},
            'robots': {'rover_0': st, 'scout_0': {'altitude_agl': 12.0, 'detections': []}}})
    s.feed({'type': 'telemetry', 'mission': {'env': {'dust_level': 1.0}}, 'robots': {'rover_0': dict(st, health=None)}})
    assert [v for _, v in s.series[('env', 'dust_level')]] == [0.4, 1.0]
    assert [v for _, v in s.series[('rover_0', 'solar_w')]] == [182.4]                          # None is skipped
    assert [v for _, v in s.series[('scout_0', 'altitude_agl')]] == [12.0]
    assert len(s.dets['rover_0']) == 2 and 'scout_0' not in s.dets                              # empty lists skipped
    assert smoke.first_sighting(s.dets['rover_0'], 'PILLAR', 0.0)[1]['range_m'] == 6.0


def test_remaining():
    assert smoke.remaining(100.0, 25.0, 110.0) == 15.0
    assert smoke.remaining(100.0, 25.0, 130.0) == 0.0


def test_first_at_most_finds_the_first_low_enough_sample_since_t0():
    agl = [(99.0, 6.0), (100.0, 12.1), (104.0, 10.2), (106.0, 9.4), (107.0, 8.1)]
    assert smoke.first_at_most(agl, 100.0, 9.5) == (106.0, 9.4)       # the low sample before t0 does not count
    assert smoke.first_at_most(agl, 100.0, 8.0) is None
    assert smoke.first_at_most([(101.0, 0.0)], 100.0, 9.5) == (101.0, 0.0)  # a 0 m reading is still a hit


# ------------------------------------------------------------------------------- phase 3 sequences, scripted backend
K = 0.01            # timeout stretch: every simulated-second budget becomes 1 % of it, so a sequence runs in ~1 s


class _Resp:
    status_code, headers = 200, {'content-type': 'application/json'}

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


class _FakeHttp:
    """POST /api/inject/* → {ok: true}, running the scripted world's reaction (sync, or a coroutine as a task)."""
    def __init__(self, script):
        self.script, self.tasks = script, []

    async def post(self, path, json=None):
        react = self.script.get(path)
        if react is not None:
            res = react()
            if asyncio.iscoroutine(res):
                self.tasks.append(asyncio.get_running_loop().create_task(res))
        return _Resp({'ok': True, 'detail': f'scripted {path}'})


_ids = iter(range(1, 10 ** 6))


def _trace(robot, kind, inp='', action='', state='EXPLORING'):
    return {'type': 'trace', 'entry': {'id': next(_ids), 'robot': robot, 'kind': kind, 't': time.time(),
                                       'state': state, 'input': inp, 'action': action}}


def _tel(robots, dust=0.0):
    return {'type': 'telemetry', 'mission': {'env': {'dust_level': dust, 'seismic': False,
                                                     'kind': 'dust_storm' if dust else None, 'wind_mps': 14 * dust}},
            'robots': robots}


ROVER = {'state': 'EXPLORING', 'mode': 'FUSED', 'path': [[float(i), 0.0] for i in range(10)],
         'pose': {'x': 0.0, 'y': 0.0, 'yaw': 0.0}, 'health': {'solar_w': 182.0, 'battery_pct': 80.0},
         'uncertainty': {'camera_conf': 0.95}, 'detections': [], 'power_save': False}
SCOUT = {'state': 'SURVEY', 'altitude_agl': 12.0, 'wind_mps': None, 'wind_descent': False}


def _smoke(script):
    sm = smoke.Smoke('http://127.0.0.1:9')
    sm.http = _FakeHttp(script)
    return sm


def _results(sm):
    return [(n, ok) for n, ok, _ in sm.results]


def test_pillar_sequence_passes_when_the_rover_perceives_and_replans():
    def fell():
        sm.s.feed({'type': 'event', 'event': {'op': 'pillar_collapse', 'ok': True, 'robot': 'rover_0', 'base': [1, 2],
                                              'tip': [5, 6], 'cross': [3, 4], 'waited_s': 2.5, 't': time.time()}})
        sm.s.feed(_tel({'rover_0': dict(ROVER, detections=[{'cls': 'PILLAR', 'conf': 0.86, 'range_m': 7.4,
                                                            'world_xy': [3, 4]}])}))
        sm.s.feed(_trace('rover_0', 'path_blocked', '11 obstacle cells 4.2 m ahead across path (LiDAR step > 0.30 m)'
                         '; camera confirms a fallen pillar', state='REPLANNING'))
        sm.s.feed(_trace('rover_0', 'replanned', action='New route 31.0 m (+4.2 m), 1412 nodes, 38 ms'))
        why = '11 obstacle cells; camera confirms a fallen pillar'
        sm.s.feed({'type': 'brain', 'decision': _nh(source='local-fallback', model='local-rules', trigger_input=why)})
    sm = _smoke({'/api/inject/pillar': fell})
    sm.s.feed(_tel({'rover_0': ROVER}))
    asyncio.run(sm.pillar_sequence('rover_0', K))
    assert _results(sm) == [('pillar collapse injected', True), ('pillar_collapse world event', True),
                            ('pillar perceived → path_blocked', True), ('pillar: re-planned route', True),
                            ('brain: novel_hazard decision after the pillar collapse', True)]
    detail = dict((n, d) for n, _, d in sm.results)
    assert 'held 2.5 s' in detail['pillar_collapse world event']
    assert 'fallen pillar' in detail['pillar perceived → path_blocked'] and 'PILLAR identified' in \
        detail['pillar perceived → path_blocked']


def test_pillar_sequence_reports_what_the_rover_saw_when_it_misses_the_pillar():
    def fell():
        sm.s.feed({'type': 'event', 'event': {'op': 'pillar_collapse', 'ok': True, 'robot': 'rover_0', 'base': [1, 2],
                                              'tip': [5, 6], 'cross': [3, 0.5], 't': time.time()}})
        sm.s.feed(_tel({'rover_0': dict(ROVER, detections=[{'cls': 'ROCK', 'conf': 0.7, 'range_m': 6.0}])}))
    sm = _smoke({'/api/inject/pillar': fell})
    sm.s.feed(_tel({'rover_0': ROVER}))
    asyncio.run(sm.pillar_sequence('rover_0', K))
    assert [ok for _, ok in _results(sm)] == [True, True, False, False, False]
    why = sm.results[2][2]
    assert 'no path_blocked' in why and 'no PILLAR detection' in why and "{'ROCK': 1}" in why
    assert '0.5 m from the path' in why
    assert 'no path_blocked caused by the pillar' in sm.results[3][2]
    # no pillar in the world: the event check fails and the rest is not attempted
    sm = _smoke({})
    sm.s.feed(_tel({'rover_0': ROVER}))
    asyncio.run(sm.pillar_sequence('rover_0', K))
    assert _results(sm) == [('pillar collapse injected', True), ('pillar_collapse world event', False)]
    assert 'no event' in sm.results[1][2]


def _clear(scout_agl=11.8):
    return {'rover_0': ROVER, 'rover_1': ROVER, 'scout_0': dict(SCOUT, altitude_agl=scout_agl)}


def test_storm_sequence_passes_when_every_robot_reacts_and_recovers():
    async def blow():
        sm.s.feed({'type': 'event', 'event': {'op': 'anomaly', 'ok': True, 'kind': 'dust_storm', 'dust_level': 1.0,
                                              'duration_s': 45.0, 'wind_mps': 14.0, 't': time.time()}})
        dusty = dict(ROVER, state='CAUTION', mode='LIDAR_ONLY', power_save=True, uncertainty={'camera_conf': 0.31},
                     health={'solar_w': 27.0, 'battery_pct': 79.0})
        sm.s.feed(_tel({'rover_0': dusty, 'rover_1': dusty,
                        'scout_0': dict(SCOUT, altitude_agl=8.1, wind_mps=13.2, wind_descent=True)}, dust=1.0))
        sm.s.feed(_trace('rover_0', 'caution_enter', 'Camera confidence 0.31, LiDAR 0.52 → uncertainty 0.71'))
        sm.s.feed(_trace('rover_0', 'power_save', 'Solar 27 W (was 182 W, −85 %), irradiance 0.15, battery 79 %',
                         'Battery conservation: speed ×0.6'))
        sm.s.feed(_trace('scout_0', 'wind_descent', 'Wind 13.2 m/s from 240° estimated from drift (cmd vs odometry)',
                         'Descending to 8 m AGL', state='SURVEY'))
        await asyncio.sleep(0.5)         # the mid-storm checks (≤ 35 × K s) run meanwhile
        sm.s.feed(_tel(_clear(), dust=0.4))
        sm.s.feed(_trace('rover_0', 'power_restored', 'Irradiance 0.93 for 5 s'))
        sm.s.feed(_trace('scout_0', 'wind_clear', 'Wind 2.1 m/s for 5 s', state='SURVEY'))
        sm.s.feed(_trace('rover_0', 'caution_exit', 'Camera confidence recovered to 0.91 for 2 s'))
        sm.s.feed(_tel(_clear(), dust=0.0))
    sm = _smoke({'/api/inject/storm': blow})
    sm.s.feed(_tel(_clear(12.0)))
    asyncio.run(sm.storm_sequence('rover_0', 'scout_0', ['rover_0', 'rover_1'], K))
    names = [n for n, _ in _results(sm)]
    assert names == ['storm injected', 'storm → CAUTION ≤ 6 s, camera confidence < 0.5',
                     'storm: rover_0 measured solar power drops ≥ 80 %',
                     'storm: rover_0 power_save (battery conservation)',
                     'storm: drone wind_descent ≤ 25 s', 'storm: drone AGL ≤ 9.5 m ≤ 35 s',
                     'storm clears: rover_0 power_restored', 'storm clears: drone wind_clear',
                     'storm clears → fused navigation restored']
    assert all(ok for _, ok in _results(sm)), sm.results
    detail = dict((n, d) for n, _, d in sm.results)
    assert '182 W → 27 W' in detail['storm: rover_0 measured solar power drops ≥ 80 %']
    assert '12.0 → 8.1 m AGL' in detail['storm: drone AGL ≤ 9.5 m ≤ 35 s']
    assert 'after the storm' in detail['storm clears → fused navigation restored']


def test_storm_sequence_fails_each_check_with_its_context_when_nothing_reacts():
    sm = _smoke({})
    sm.s.feed(_tel(_clear(12.0)))
    asyncio.run(sm.storm_sequence('rover_0', 'scout_0', ['rover_0', 'rover_1'], K / 2))
    assert [ok for _, ok in _results(sm)] == [True] + [False] * 8
    detail = dict((n, d) for n, _, d in sm.results)
    assert 'not traced' in detail['storm → CAUTION ≤ 6 s, camera confidence < 0.5']
    assert '182 W before, no samples since' in detail['storm: rover_0 measured solar power drops ≥ 80 %']
    assert 'AGL 12.0 m at the inject, never ≤ 9.5 m' in detail['storm: drone AGL ≤ 9.5 m ≤ 35 s']
    assert 'no storm end seen' in detail['storm clears: rover_0 power_restored']
    # one rover, no drone: the drone checks are left out
    sm = _smoke({})
    sm.s.feed(_tel({'rover_0': ROVER}))
    asyncio.run(sm.storm_sequence('rover_0', None, ['rover_0'], K / 2))
    assert not any('drone' in n for n, _ in _results(sm)) and len(sm.results) == 6
