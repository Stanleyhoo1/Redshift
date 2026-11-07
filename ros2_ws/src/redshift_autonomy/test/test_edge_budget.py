import pytest

from redshift_autonomy.edge_budget import PROFILES, EdgeBudget


def test_profile_catalog():
    assert set(PROFILES) == {'orin_nano_15w', 'orin_nx_25w', 'agx_orin_60w', 'rpi5_8w'}
    nano = PROFILES['orin_nano_15w']
    assert nano.watts_max == 15 and nano.perf_scale == pytest.approx(2.6)


def test_load_and_power_math():
    b = EdgeBudget('orin_nano_15w')
    s = b.update({'perception': 10.0}, {'perception': 10.0}, now=0.0)
    # 10 ms host * 2.6 perf * 10 Hz = 260 ms of device time per second
    assert s.load_pct == pytest.approx(26.0)
    assert s.device_ms == pytest.approx(26.0)
    assert s.power_w == pytest.approx(5 + 10 * 0.26)
    assert s.throttle_level == 0
    assert s.perception_hz == pytest.approx(10.0)


def test_throttles_under_overload_and_recovers_with_hysteresis():
    b = EdgeBudget('orin_nano_15w')
    t = 0.0
    changed = []
    for _ in range(30):  # 40 ms * 2.6 * 10 Hz = 104 % load
        s = b.update({'perception': 40.0}, {'perception': 10.0}, now=t)
        changed.append(s.changed)
        t += 0.5
    assert s.throttle_level >= 1
    assert s.perception_hz < 10.0
    assert any(changed)
    level_hi = s.throttle_level
    reasons = []
    for _ in range(40):  # 5 ms -> 13 % load
        s = b.update({'perception': 5.0}, {'perception': 10.0}, now=t)
        if s.changed:
            reasons.append(s.change_reason)
        t += 0.5
    assert s.throttle_level < level_hi
    assert reasons and all('restored' in r for r in reasons)


def test_middle_band_holds_level():
    b = EdgeBudget('orin_nano_15w')
    t = 0.0
    for _ in range(30):
        s = b.update({'perception': 40.0}, {'perception': 10.0}, now=t)
        t += 0.5
    lvl = s.throttle_level
    for _ in range(20):  # 75 % load: between LOW and HIGH
        s = b.update({'perception': 28.85}, {'perception': 10.0}, now=t)
        t += 0.5
    assert s.throttle_level == lvl


def test_as_dict_shape():
    d = EdgeBudget('rpi5_8w').update({'perception': 5, 'planner': 2}, {'perception': 5, 'planner': 1}, now=0).as_dict()
    assert set(d) >= {'profile', 'host_ms', 'device_ms', 'load_pct', 'power_w', 'perception_hz', 'throttle_level'}
