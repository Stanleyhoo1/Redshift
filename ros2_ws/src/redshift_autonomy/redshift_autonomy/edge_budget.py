"""Simulated edge-compute budget: maps measured host latencies onto an embedded device profile.

load   = Σ_stage host_ms · perf_scale · stage_hz / 1000      (fraction of one device-second)
power  = watts_idle + (watts_max − watts_idle) · min(1, load)
Throttling: when the smoothed load exceeds 90 % the perception rate drops one 25 % step; it is
restored one step at a time once the load falls below 60 % (2 s minimum dwell between changes).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EdgeProfile:
    id: str
    label: str
    device: str
    watts_idle: float
    watts_max: float
    perf_scale: float          # device_ms = host_ms * perf_scale
    perception_hz_max: float
    planner_hz_max: float
    max_expansions: int

    def as_dict(self) -> dict:
        return asdict(self)


PROFILES: dict[str, EdgeProfile] = {p.id: p for p in [
    EdgeProfile('orin_nano_15w', 'Jetson Orin Nano · 15 W', 'NVIDIA Jetson Orin Nano 8GB', 5.0, 15.0, 2.6, 10.0, 2.0, 60_000),
    EdgeProfile('orin_nx_25w', 'Jetson Orin NX · 25 W', 'NVIDIA Jetson Orin NX 16GB', 7.0, 25.0, 1.6, 15.0, 4.0, 120_000),
    EdgeProfile('agx_orin_60w', 'Jetson AGX Orin · 60 W', 'NVIDIA Jetson AGX Orin 64GB', 15.0, 60.0, 0.9, 20.0, 8.0, 250_000),
    EdgeProfile('rpi5_8w', 'Raspberry Pi 5 · 8 W', 'Raspberry Pi 5 8GB (CPU only)', 3.0, 8.0, 5.0, 5.0, 1.0, 30_000),
]}


@dataclass
class EdgeState:
    profile: str
    host_ms: dict = field(default_factory=dict)
    device_ms: float = 0.0
    load_pct: float = 0.0
    power_w: float = 0.0
    perception_hz: float = 0.0
    throttle_level: int = 0
    changed: bool = False
    change_reason: str = ''

    def as_dict(self) -> dict:
        d = asdict(self)
        d['host_ms'] = {k: round(v, 2) for k, v in self.host_ms.items()}
        for k in ('device_ms', 'load_pct', 'power_w', 'perception_hz'):
            d[k] = round(d[k], 2)
        return d


class EdgeBudget:
    HIGH = 0.90
    LOW = 0.60
    STEP = 0.25
    MAX_LEVEL = 3
    DWELL_S = 2.0
    RESTORE_DWELL_S = 6.0
    EMA = 0.35

    def __init__(self, profile_id: str):
        self.profile = PROFILES[profile_id]
        self.level = 0
        self._ema: float | None = None
        self._last_change = -1e9

    @property
    def perception_hz(self) -> float:
        return self.profile.perception_hz_max * (1.0 - self.STEP * self.level)

    def update(self, host_ms: dict, hz: dict, now: float | None = None) -> EdgeState:
        now = time.monotonic() if now is None else now
        p = self.profile
        device_ms = sum(host_ms.values()) * p.perf_scale
        load = sum(ms * p.perf_scale * hz.get(k, 0.0) for k, ms in host_ms.items()) / 1000.0
        self._ema = load if self._ema is None else self.EMA * load + (1 - self.EMA) * self._ema
        changed, reason = False, ''
        since = now - self._last_change
        if since >= self.DWELL_S:
            # restoring is predictive: only step up if the projected load at the higher rate stays under HIGH
            hz_now = self.perception_hz
            hz_up = p.perception_hz_max * (1.0 - self.STEP * max(0, self.level - 1))
            projected = self._ema * hz_up / max(hz_now, 1e-6)
            if self._ema > self.HIGH and self.level < self.MAX_LEVEL:
                self.level += 1
                changed = True
                reason = (f'Edge load {self._ema * 100:.0f}% > {self.HIGH * 100:.0f}% on {p.label}: '
                          f'perception throttled to {self.perception_hz:.1f} Hz')
            elif (self._ema < self.LOW and self.level > 0 and since >= self.RESTORE_DWELL_S
                  and projected < 0.85 * self.HIGH):
                self.level -= 1
                changed = True
                reason = (f'Edge load {self._ema * 100:.0f}% < {self.LOW * 100:.0f}%: '
                          f'perception restored to {self.perception_hz:.1f} Hz')
            if changed:
                self._last_change = now
        power = p.watts_idle + (p.watts_max - p.watts_idle) * min(1.0, load)
        return EdgeState(profile=p.id, host_ms=dict(host_ms), device_ms=device_ms, load_pct=load * 100.0,
                         power_w=power, perception_hz=self.perception_hz, throttle_level=self.level,
                         changed=changed, change_reason=reason)
