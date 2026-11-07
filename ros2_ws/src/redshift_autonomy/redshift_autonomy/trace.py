"""Structured perception → reasoning → action decision traces (docs/protocol.md §3.1)."""
from __future__ import annotations

import itertools
import time

_ids = itertools.count(1)
SENSORS = ('CAM', 'DEPTH', 'LIDAR', 'IMU', 'ODOM', 'MAP', 'FLEET', 'EDGE')


class TraceBuilder:
    def __init__(self, robot: str, dedup_s: float = 2.0):
        self.robot = robot
        self.dedup_s = dedup_s
        self._last: dict[str, float] = {}

    def make(self, kind: str, state: str, sensors, input: str, reasoning: str, confidence: float,
             action: str, severity: str = 'info', latency_ms: float | None = None,
             now: float | None = None, force: bool = False) -> dict | None:
        # `now` (simulated seconds in the nodes) only drives de-duplication; `t` is always the wall-clock epoch the
        # UI shows and clients compare against
        now = time.monotonic() if now is None else now
        last = self._last.get(kind)
        if not force and last is not None and now - last < self.dedup_s:
            return None
        self._last[kind] = now
        return {
            'id': next(_ids), 't': round(time.time(), 3), 'robot': self.robot, 'state': state, 'kind': kind,
            'sensors': [s for s in sensors if s in SENSORS], 'input': input, 'reasoning': reasoning,
            'confidence': round(float(max(0.0, min(1.0, confidence))), 2), 'action': action,
            'severity': severity, 'latency_ms': None if latency_ms is None else round(float(latency_ms), 1),
        }
