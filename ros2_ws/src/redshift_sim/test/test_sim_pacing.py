import pytest

from redshift_sim.sim_pacing import SimPacer


def run(p: SimPacer, wall_dt: float, ticks: int, render_done=True):
    """Drive the pacer like the mock world's 100 Hz physics thread; returns [(t, due keys)] for every due event."""
    events = []
    for _ in range(ticks):
        for dt, due in p.steps(wall_dt):
            assert dt >= 0.0 and dt <= SimPacer.SUBSTEP + 1e-9
            if due:
                events.append((round(p.t, 6), sorted(due)))
                if render_done:
                    for k in due:
                        p.done(k)
    return events


def test_advances_at_speed_with_bounded_substeps():
    for speed in (0.5, 1.0, 2.0):
        p = SimPacer(speed)
        run(p, 0.01, 100)                          # 1 s of wall time
        assert p.t == pytest.approx(speed, abs=1e-6)


def test_sensors_fire_exactly_on_their_sim_schedule_at_any_speed():
    for speed in (0.5, 2.0):
        p = SimPacer(speed)
        p.add_sensor('cam', 0.1, phase=0.02)
        ev = run(p, 0.01, int(round(1.0 / speed * 100)))     # 1 s of sim time
        times = [t for t, _ in ev]
        assert times[:3] == pytest.approx([0.02, 0.12, 0.22], abs=1e-6)
        assert len(times) == 10                   # 10 Hz per simulated second, independent of speed


def test_lockstep_holds_the_clock_while_a_due_frame_is_still_rendering():
    p = SimPacer(2.0)
    p.add_sensor('lidar', 0.1)
    run(p, 0.01, 2, render_done=False)           # first frame (t=0) queued but never finished
    t_stuck = p.t
    run(p, 0.01, 50, render_done=False)
    assert p.t == pytest.approx(0.1)              # clock stops at the next due time and waits there
    assert t_stuck <= 0.1
    p.done('lidar')
    run(p, 0.01, 1)
    assert p.t > 0.1                              # released: time moves again


def test_catch_up_after_a_short_stall_is_bounded():
    p = SimPacer(1.0)
    run(p, 0.01, 10)
    t0 = p.t
    run(p, 1.5, 1)                                # physics thread stalled 1.5 s of wall time
    assert p.t - t0 == pytest.approx(SimPacer.MAX_LAG, abs=1e-6)   # no 1.5 s burst: excess is dropped


def test_speed_change_and_clamping():
    p = SimPacer(1.0)
    p.set_speed(10.0)
    assert p.speed == SimPacer.MAX_SPEED
    p.set_speed(0.0)
    assert p.speed == SimPacer.MIN_SPEED
    p.set_speed(2.0)
    run(p, 0.01, 50)
    assert p.t == pytest.approx(1.0, abs=1e-6)


def test_many_sensors_due_together_are_reported_once():
    p = SimPacer(1.0)
    p.add_sensor('a', 0.1)
    p.add_sensor('b', 0.1)
    ev = run(p, 0.01, 11)
    assert ev[0] == (0.0, ['a', 'b']) and ev[1] == (0.1, ['a', 'b'])
