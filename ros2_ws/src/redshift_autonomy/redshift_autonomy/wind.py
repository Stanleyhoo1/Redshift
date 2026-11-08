"""Drone wind estimate from its own odometry, and the local wind-descent rule (docs/protocol.md, Phase 3).

The drone never reads the world's wind. Its velocity controller tracks the commanded velocity, and whatever the air
adds on top shows up as drift: odometry velocity minus the commanded velocity (rotated from the body frame into the
world by the yaw). `WindEstimator` smooths that residual (EMA, tau 1.5 s, simulated time) and ignores it while the
command is changing fast, when the residual is the controller's own lag, not the wind.

`WindDescent` is the local, authoritative altitude rule: wind shear grows with height, so in a strong wind the drone
flies lower (cruise − 4 m, never below 6 m) until the wind has eased for a while. The advisory brain may only lower
the drone further (the planner flies the lower of the two set-points).
"""
from __future__ import annotations

import collections
import math

from .gemma_agent_brain import DRONE_ALT_MIN_M

WIND_TAU_S = 1.5                 # EMA time constant of the drift residual (simulated seconds)
TRANSIENT_DV_MPS = 1.0           # commanded velocity changed by more than this ...
TRANSIENT_S = 0.5                # ... within this window → controller transient: the sample is ignored
WIND_ENTER_MPS = 8.0             # estimated wind above this ...
WIND_ENTER_S = 2.0               # ... for this long → descend
WIND_EXIT_MPS = 5.0              # estimated wind below this ...
WIND_EXIT_S = 5.0                # ... for this long → back to cruise
WIND_DESCENT_M = 4.0             # descend this far below cruise (12 m → 8 m) ...
WIND_MIN_AGL_M = DRONE_ALT_MIN_M  # ... never below the drone's 6 m floor
MAX_DT_S = 1.0                   # longest gap one EMA step may cover (a stalled clock is not a long average)


def compass_from_deg(bearing_rad: float) -> float:
    """Compass bearing (degrees clockwise from north) the wind blows FROM, for a world-frame direction the air moves
    TOWARD (`bearing_rad`, counter-clockwise from +x east)."""
    return (90.0 - math.degrees(bearing_rad) + 180.0) % 360.0


class WindEstimator:
    """World-frame wind from the drone's own odometry: EMA (tau 1.5 s) of (v_odom_xy − v_cmd_world_xy), where v_cmd
    is the commanded body velocity rotated by yaw. update(t, v_odom_xy, v_cmd_body_xy, yaw) -> (speed_mps,
    bearing_rad). Ignores samples while the commanded velocity changed by > 1 m/s in the last 0.5 s (transients).
    `bearing_rad` is the direction the air moves toward (atan2 of the world-frame vector, x east, y north)."""

    def __init__(self, tau_s: float = WIND_TAU_S, transient_mps: float = TRANSIENT_DV_MPS,
                 transient_s: float = TRANSIENT_S):
        self.tau_s, self.transient_mps, self.transient_s = float(tau_s), float(transient_mps), float(transient_s)
        self.wx = self.wy = 0.0
        self.samples = 0                 # accepted (non-transient) samples
        self.transient = False           # the last sample was ignored as a transient
        self._t: float | None = None
        self._cmd: collections.deque = collections.deque()   # (t, cx, cy) world-frame commands in the window

    @property
    def ready(self) -> bool:
        return self.samples > 0

    @property
    def speed(self) -> float:
        return math.hypot(self.wx, self.wy)

    @property
    def bearing(self) -> float:
        return math.atan2(self.wy, self.wx)

    @property
    def from_deg(self) -> float:
        return compass_from_deg(self.bearing)

    @property
    def vector(self) -> tuple[float, float]:
        """World-frame wind estimate (m/s, x east, y north): what the drone's feed-forward leans against."""
        return self.wx, self.wy

    def update(self, t: float, v_odom_xy, v_cmd_body_xy, yaw: float, guidance_body_xy=None) -> tuple[float, float]:
        """`v_cmd_body_xy` is the full command published (the residual is measured against it). `guidance_body_xy`,
        when given, is that command without the wind feed-forward: only guidance changes count as controller
        transients, since the feed-forward itself ramps quickly while a storm builds and would otherwise gate out
        the very samples that make the estimate converge."""
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = float(v_cmd_body_xy[0]), float(v_cmd_body_xy[1])
        cx, cy = c * bx - s * by, s * bx + c * by
        gb = v_cmd_body_xy if guidance_body_xy is None else guidance_body_xy
        gx, gy = c * float(gb[0]) - s * float(gb[1]), s * float(gb[0]) + c * float(gb[1])
        dt = 0.0 if self._t is None else min(MAX_DT_S, max(0.0, t - self._t))
        self._t = t
        self._cmd.append((t, gx, gy))
        while t - self._cmd[0][0] > self.transient_s + 1e-9:
            self._cmd.popleft()
        self.transient = any(math.hypot(gx - x, gy - y) > self.transient_mps for _, x, y in self._cmd)
        if not self.transient:
            rx, ry = float(v_odom_xy[0]) - cx, float(v_odom_xy[1]) - cy
            if math.isfinite(rx) and math.isfinite(ry):
                a = 1.0 - math.exp(-dt / self.tau_s) if self.tau_s > 0 else 1.0
                self.wx += a * (rx - self.wx)
                self.wy += a * (ry - self.wy)
                self.samples += 1
        return self.speed, self.bearing


def wind_feedforward(guidance_xy, wind_xy, v_air_max: float) -> tuple[float, float]:
    """World-frame command that flies `guidance_xy` over the ground in the estimated wind: guidance minus wind,
    limited to the airframe's air-relative speed `v_air_max` (a wind stronger than that still drifts the drone)."""
    vx, vy = float(guidance_xy[0]) - float(wind_xy[0]), float(guidance_xy[1]) - float(wind_xy[1])
    sp = math.hypot(vx, vy)
    if sp > v_air_max > 0.0:
        vx, vy = vx * v_air_max / sp, vy * v_air_max / sp
    return vx, vy


class WindDescent:
    """Local, authoritative altitude rule. Enter when the wind estimate > ENTER_MPS (8.0) for 2.0 s → target AGL =
    max(DRONE_ALT_MIN_M 6, cruise − 4) (8 m from 12); exit when < EXIT_MPS (5.0) for 5.0 s → cruise.
    update(t, wind_mps) -> 'descend' | 'clear' | None; .target_agl(cruise) -> float | None (None = no descent)."""
    ENTER_MPS, EXIT_MPS = WIND_ENTER_MPS, WIND_EXIT_MPS

    def __init__(self, enter_mps: float = WIND_ENTER_MPS, enter_s: float = WIND_ENTER_S,
                 exit_mps: float = WIND_EXIT_MPS, exit_s: float = WIND_EXIT_S, descent_m: float = WIND_DESCENT_M,
                 min_agl_m: float = WIND_MIN_AGL_M):
        self.enter_mps, self.enter_s, self.exit_mps, self.exit_s = enter_mps, enter_s, exit_mps, exit_s
        self.descent_m, self.min_agl_m = descent_m, min_agl_m
        self.active = False
        self._since: float | None = None   # start of the current run past the threshold that would flip the state

    def update(self, t: float, wind_mps: float | None) -> str | None:
        if wind_mps is None or not math.isfinite(wind_mps):
            self._since = None
            return None
        past = wind_mps < self.exit_mps if self.active else wind_mps > self.enter_mps
        if not past:
            self._since = None
            return None
        if self._since is None:
            self._since = t
        if t - self._since < (self.exit_s if self.active else self.enter_s) - 1e-9:
            return None
        self.active, self._since = not self.active, None
        return 'descend' if self.active else 'clear'

    def target_agl(self, cruise_agl: float) -> float | None:
        return max(self.min_agl_m, cruise_agl - self.descent_m) if self.active else None


def descent_texts(speed_mps: float, bearing_rad: float, agl_m: float, applies_now: bool = True) -> dict:
    """Texts for the scout's "wind_descent" trace (`agl_m` = the altitude set-point it descends to)."""
    action = f'Descending to {agl_m:.0f} m AGL'
    if not applies_now:
        action += ' once surveying (take-off / scan altitude unchanged)'
    return {'input': f'Wind {speed_mps:.1f} m/s from {compass_from_deg(bearing_rad):.0f}° estimated from drift '
                     '(cmd vs odometry)',
            'reasoning': 'Wind shear grows with height; lower is calmer and keeps the footprint on target',
            'confidence': 0.85, 'action': action, 'severity': 'warn'}


def clear_texts(speed_mps: float, agl_m: float) -> dict:
    """Texts for the scout's "wind_clear" trace (`agl_m` = the cruise set-point it climbs back to)."""
    return {'input': f'Wind {speed_mps:.1f} m/s estimated from drift (cmd vs odometry), below '
                     f'{WIND_EXIT_MPS:.0f} m/s for {WIND_EXIT_S:.0f} s',
            'reasoning': 'The wind has eased; cruise altitude gives the widest nadir footprint per Wh',
            'confidence': 0.85, 'action': f'Climbing back to {agl_m:.0f} m AGL', 'severity': 'info'}
