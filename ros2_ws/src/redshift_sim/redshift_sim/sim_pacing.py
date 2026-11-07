"""Simulated-time pacing for the mock world (pure; no ROS).

The mock world is a clock source, like Gazebo: simulated time advances at `speed` × wall time, in physics substeps
of at most SUBSTEP, and every substep ends exactly on the next sensor due time so frames are sampled on their
simulated schedule. Like Gazebo's sensors system it runs in lockstep: if a sensor comes due while its previous frame
is still rendering, the clock waits there. Sensor rates per simulated second therefore hold at any speed; when the
renderer can't keep up, the achieved real-time factor drops instead.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass


@dataclass
class _Sensor:
    period: float
    next_due: float
    busy: bool = False


class SimPacer:
    SUBSTEP = 0.01            # s of simulated time per physics step (≤)
    MAX_LAG = 0.2             # s of simulated time we may catch up after a stall; any excess is dropped
    MIN_SPEED, MAX_SPEED = 0.1, 4.0
    EPS = 1e-9

    def __init__(self, speed: float = 1.0):
        self.t = 0.0
        self.speed = self._clamp(speed)
        self._ideal = 0.0
        self._sensors: dict[object, _Sensor] = {}
        self._lock = threading.Lock()      # done() is called from the render threads

    @classmethod
    def _clamp(cls, speed: float) -> float:
        s = float(speed)
        return min(cls.MAX_SPEED, max(cls.MIN_SPEED, s if math.isfinite(s) else 1.0))

    def set_speed(self, speed: float) -> float:
        self.speed = self._clamp(speed)
        return self.speed

    def add_sensor(self, key, period: float, phase: float = 0.0):
        self._sensors[key] = _Sensor(float(period), self.t + float(phase))

    def done(self, key):
        """The frame queued for `key` has been rendered and published."""
        with self._lock:
            s = self._sensors.get(key)
            if s is not None:
                s.busy = False

    def steps(self, wall_dt: float):
        """Advance toward the paced target for `wall_dt` seconds of wall time.

        Yields (dt, due): dt > 0 → step physics by dt (self.t is already the new time); due → keys whose frame must
        be sampled now, at self.t (the caller snapshots state, renders, then calls done(key)).
        """
        self._ideal = min(self._ideal + max(0.0, wall_dt) * self.speed, self.t + self.MAX_LAG)
        while True:
            due = [k for k, s in self._sensors.items() if s.next_due <= self.t + self.EPS]
            if due:
                with self._lock:
                    if any(self._sensors[k].busy for k in due):
                        self._ideal = min(self._ideal, self.t + self.MAX_LAG)
                        return                      # lockstep: wait for the renderer
                    for k in due:
                        s = self._sensors[k]
                        s.busy = True
                        s.next_due = s.next_due + s.period if self.t - s.next_due < s.period else self.t + s.period
                yield 0.0, due
                continue
            nxt = min((s.next_due for s in self._sensors.values()), default=math.inf)
            stop = min(self._ideal, self.t + self.SUBSTEP, nxt)
            if stop <= self.t + self.EPS:
                return
            dt = stop - self.t
            self.t = stop
            yield dt, []
