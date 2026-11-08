"""The smoke test's advisory-brain verdicts (backend_bridge/scripts/smoke_test.py), checked without a backend."""
import importlib.util
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
