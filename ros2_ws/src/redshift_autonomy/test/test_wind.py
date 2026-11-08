"""Unit tests for the drone's wind estimate (odometry vs its own commands) and the local wind-descent rule."""
import math

import pytest

from redshift_autonomy.wind import (WIND_TAU_S, WindDescent, WindEstimator, clear_texts, compass_from_deg,
                                    descent_texts, wind_feedforward)

DT = 0.1                                  # the scout planner's 10 Hz tick


def fly(est, t0, seconds, cmd_body, wind_xy, yaw=0.0, lag=None):
    """Feed `seconds` of ticks: odometry = commanded velocity (world frame) + wind; returns the end time."""
    c, s = math.cos(yaw), math.sin(yaw)
    t = t0
    for _ in range(int(round(seconds / DT))):
        cx, cy = c * cmd_body[0] - s * cmd_body[1], s * cmd_body[0] + c * cmd_body[1]
        v = (cx + wind_xy[0], cy + wind_xy[1]) if lag is None else lag(t)
        est.update(t, v, cmd_body, yaw)
        t += DT
    return t


def test_compass_bearing_is_where_the_wind_comes_from():
    assert compass_from_deg(0.0) == pytest.approx(270.0)               # blowing toward east = from the west
    assert compass_from_deg(math.pi / 2) == pytest.approx(180.0)       # toward north = from the south
    assert compass_from_deg(math.radians(30.0)) == pytest.approx(240.0)


def test_steady_drift_converges_to_the_wind():
    est = WindEstimator()
    assert not est.ready and est.speed == 0.0
    wind = (-6.6, -11.43)                                               # 13.2 m/s from 30° compass
    fly(est, 0.0, 8.0, (2.0, 0.0), wind)
    spd, brg = est.speed, est.bearing
    assert est.ready and spd == pytest.approx(13.2, abs=0.1)
    assert brg == pytest.approx(math.atan2(wind[1], wind[0]), abs=0.01)
    assert est.from_deg == pytest.approx(30.0, abs=1.0)


def test_command_is_rotated_into_the_world_by_yaw():
    # flying north at 2 m/s with the nose pointing north: body +x is world +y. A yaw mistake would read 2.8 m/s.
    est = WindEstimator()
    fly(est, 0.0, 8.0, (2.0, 0.0), (0.0, 0.0), yaw=math.pi / 2)
    assert est.speed < 0.05


def test_ema_time_constant_on_simulated_time():
    est = WindEstimator()
    fly(est, 0.0, WIND_TAU_S, (0.0, 0.0), (10.0, 0.0))
    assert est.speed == pytest.approx(10.0 * (1 - math.exp(-1.0)), rel=0.08)


def test_command_transients_are_ignored():
    """A 0 → 4 m/s command step while the airframe lags behind: the residual is the controller's lag, not wind."""
    est = WindEstimator()
    t = fly(est, 0.0, 3.0, (0.0, 0.0), (0.0, 0.0))
    t_step = t

    def lagging(tt):                    # first-order response, tau 0.25 s
        return (4.0 * (1.0 - math.exp(-(tt - t_step) / 0.25)), 0.0)
    t = fly(est, t, 0.4, (4.0, 0.0), None, lag=lagging)
    assert est.transient and est.speed == pytest.approx(0.0, abs=1e-9)
    fly(est, t, 3.0, (4.0, 0.0), None, lag=lagging)
    assert not est.transient and est.speed < 0.6
    # the same step without the transient gate reads a spurious gust of wind
    naive = WindEstimator(transient_mps=math.inf)
    t = fly(naive, 0.0, 3.0, (0.0, 0.0), (0.0, 0.0))
    t_step = t
    fly(naive, t, 0.4, (4.0, 0.0), None, lag=lagging)
    assert naive.speed > 0.4


def test_gradual_command_changes_still_count():
    est = WindEstimator()
    t = 0.0
    for k in range(80):                  # +0.15 m/s per tick: 0.75 m/s per 0.5 s, under the 1 m/s gate
        cmd = (min(4.0, 0.15 * k), 0.0)
        est.update(t, (cmd[0] + 9.0, 0.0), cmd, 0.0)
        assert not est.transient
        t += DT
    assert est.speed == pytest.approx(9.0, abs=0.2)


def test_non_finite_odometry_is_skipped():
    est = WindEstimator()
    est.update(0.0, (math.nan, 0.0), (0.0, 0.0), 0.0)
    assert not est.ready and est.speed == 0.0


def test_wind_descent_enters_after_two_seconds_and_targets_cruise_minus_four():
    d = WindDescent()
    assert d.target_agl(12.0) is None
    t, ev = 0.0, []
    while t < 1.95:
        ev.append(d.update(t, 9.0))
        t += DT
    assert set(ev) == {None} and not d.active
    assert d.update(2.0, 9.0) == 'descend' and d.active
    assert d.target_agl(12.0) == pytest.approx(8.0)
    assert d.target_agl(9.0) == pytest.approx(6.0)          # never below the 6 m floor
    assert d.update(2.1, 9.0) is None                        # one event per transition


def test_a_short_gust_does_not_descend():
    d = WindDescent()
    for k in range(15):                                      # 1.5 s above 8 m/s, then a lull
        assert d.update(k * DT, 12.0) is None
    assert d.update(1.5, 7.0) is None
    for k in range(15):
        assert d.update(1.6 + k * DT, 12.0) is None          # the timer restarted
    assert not d.active


def test_wind_descent_hysteresis_and_exit_after_five_seconds():
    d = WindDescent()
    t = 0.0
    for _ in range(21):
        d.update(t, 10.0)
        t += DT
    assert d.active
    for _ in range(100):                                     # 6.5 m/s: between the thresholds, stays down
        assert d.update(t, 6.5) is None
        t += DT
    assert d.active
    t0 = t
    out = None
    while out is None and t < t0 + 10.0:
        out = d.update(t, 4.0)
        t += DT
    assert out == 'clear' and not d.active and d.target_agl(12.0) is None
    assert t - DT - t0 == pytest.approx(5.0, abs=1e-6)
    assert d.update(t, None) is None                         # no estimate: nothing changes


def test_trace_texts_carry_the_measured_numbers():
    tx = descent_texts(13.2, math.radians(30.0), 8.0)
    assert tx['input'] == 'Wind 13.2 m/s from 240° estimated from drift (cmd vs odometry)'
    assert tx['reasoning'] == 'Wind shear grows with height; lower is calmer and keeps the footprint on target'
    assert tx['action'] == 'Descending to 8 m AGL' and tx['severity'] == 'warn'
    assert 'once surveying' in descent_texts(13.2, 0.0, 8.0, applies_now=False)['action']
    cl = clear_texts(3.9, 12.0)
    assert '3.9 m/s' in cl['input'] and cl['action'] == 'Climbing back to 12 m AGL'


def test_feedforward_leans_into_the_wind_within_the_airspeed_limit():
    assert wind_feedforward((2.0, 0.0), (-6.0, 0.0), 18.0) == pytest.approx((8.0, 0.0))
    vx, vy = wind_feedforward((4.0, 0.0), (-20.0, 0.0), 18.0)
    assert math.hypot(vx, vy) == pytest.approx(18.0) and vx > 0     # saturates at the airframe limit
    assert wind_feedforward((1.0, 2.0), (0.0, 0.0), 18.0) == pytest.approx((1.0, 2.0))


@pytest.mark.parametrize('gust', [0.0, 0.25])
def test_a_drone_compensating_from_its_own_estimate_holds_station_in_a_storm(gust):
    """Closed loop, like the scout: the world adds the wind to whatever is commanded; the drone commands its guidance
    minus its own estimate. The feed-forward ramps fast as the storm builds, but the transient gate only looks at the
    guidance, so the estimate must not stall: it converges and the drone's ground speed falls back to near zero."""
    est = WindEstimator()
    t, px, py = 0.0, 0.0, 0.0
    track = []
    for k in range(int(30 / DT)):
        ramp = min(1.0, max(0.0, (t - 2.0) / 1.5))            # the storm builds over 1.5 s from t = 2 s
        w = 14.0 * ramp * (1.0 + gust * math.sin(2 * math.pi * 0.3 * t))
        wind = (w * math.cos(0.5), w * math.sin(0.5))
        guide = (0.0, 0.0)                                     # holding station
        cmd = wind_feedforward(guide, est.vector, 18.0) if est.ready else guide
        v = (cmd[0] + wind[0], cmd[1] + wind[1])
        est.update(t, v, cmd, 0.0, guidance_body_xy=guide)
        px, py = px + v[0] * DT, py + v[1] * DT
        track.append((px, py, math.hypot(*v)))
        t += DT
    assert est.speed == pytest.approx(14.0, abs=14.0 * gust + 0.5)
    tail = track[-int(10 / DT):]
    # held station over the last 10 s: gusts sway it (the estimate lags them) but it does not drift downwind
    assert math.hypot(tail[-1][0] - tail[0][0], tail[-1][1] - tail[0][1]) < (6.0 if gust else 0.5)
    if not gust:
        assert max(s for _, _, s in tail) < 0.2
    assert math.hypot(px, py) < 25.0                          # not blown hundreds of metres downwind
