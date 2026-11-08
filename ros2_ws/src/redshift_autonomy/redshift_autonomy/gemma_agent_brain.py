"""Advisory Gemma "agent brain": event → compact sensor summary → Gemma (cloud) → validated, safer-only decision,
with a deterministic on-device fallback that answers in exactly the same schema.

Pure Python (stdlib only: no rclpy, no SDK, HTTP via urllib.request) so it can be unit-tested and wired into the
ROS nodes later.

Authority model
---------------
The local FSM / A* / safety layer has *already acted* when an event is submitted here. Gemma is consulted
asynchronously and is advisory only. `validate()` lets its answer make things SAFER and nothing else:

* the action must be one of `ACTIONS` and permitted by the context (`allowed_actions`): DEPLOY_RECOVERY_ROVER only
  for critical_damage / peer_sos when a recovery rover exists, is available and is not the disabled robot;
  ADJUST_DRONE_ALTITUDE only when a drone exists; REJECT_UNSAFE_COMMAND only for operator_command;
* `target_speed_mps` is a speed *cap* clamped to [0, local limit]: the integrator applies min(local, Gemma);
* `target_altitude_m` is clamped to [6, 20] m and only kept for ADJUST_DRONE_ALTITUDE;
* operator_command: a command the local check finds unsafe (asks for more than the local limit, or drives into a
  known hazard) always ends as REJECT_UNSAFE_COMMAND; a stop / slow-down request is always honoured;
* anything invalid (bad JSON, unknown action, missing reasoning, forbidden action) → `deterministic_fallback`.

Every clamp is listed in `Decision.adjusted`. `Decision.source` says which brain chose the action: "gemma" (its
action survived validation, possibly with clamped numbers) or "local-fallback" (no key, network error, timeout,
rate limit, invalid answer, or the local safety layer overrode Gemma's action). `error` is set when Gemma's answer
could not be used at all.

Public API
----------
    EVENTS, ACTIONS, DEVICES, MODELS
    build_summary(event, context) -> dict                    compact JSON sensor summary (≤ SUMMARY_MAX_CHARS)
    allowed_actions(event, context) -> tuple[str, ...]        actions the context permits (sent as the schema enum)
    assess_operator_command(text, context) -> dict            local, deterministic safety reading of operator text
    validate(raw, event, context, *, model, latency_ms, tokens_in, tokens_out, device_id) -> Decision
    deterministic_fallback(event, context, *, error=None, device_id=None, cloud_model=DEFAULT_MODEL) -> Decision
    project_latency(tokens_in, tokens_out, device_id, model_id) -> dict      on-device roofline ESTIMATE
    load_api_key(env_path=None) -> str | None
    parse_response(payload, *, requested_model, latency_ms) -> GemmaReply
    GemmaClient(model="gemma-4-26b-a4b-it", api_key=None, timeout_s=8.0, transport=None, *, env_path=None, ...)
        .build_request(event, context) -> dict   exact JSON body that is POSTed
        .call(event, context) -> GemmaReply      blocking; raises GemmaError / GemmaRateLimited
        .decide(event, context) -> Decision      blocking; never raises; validated answer or fallback
    AgentBrain(client, *, max_workers=2, min_interval_s=3.0, deadline_s=10.0, on_decision=None, clock=...)
        .submit(event, context, robot=None) -> request_id | None      never blocks
        .poll() -> list[Decision]       drain finished decisions (thread-safe)
        .stats() -> dict, .backoff_remaining_s() -> float, .close(wait=True)

Context dict (every key optional; aliases used by agent_state messages in brackets)
---------------------------------------------------------------------------------
    robot, robot_type [type] ("rover" | "scout"/"drone"), state, speed_mps [speed],
    speed_limit_mps [v_limit]   the EFFECTIVE local limit right now (v_max × FSM speed_scale; 0 in SAFE_HOLD).
                                Missing → treated as 0 m/s (hold), so always pass it.
    v_max_mps                   vehicle top speed at this gravity (what "full speed" means)
    health, battery             0..1 (values > 1 are read as percent)
    camera_conf, lidar_conf, uncertainty (0..1), nearest_obstacle_m
    hazard (text), damage (text), peer (robot that sent the SOS), known_hazards (list of names)
    dust (bool), wind_mps, operator_text, edge_profile (device id for the projection)
    fleet: {recovery_rover: id | None, recovery_available: bool (default True), drone: id | None,
            drone_agl_m: float, drone_affected: bool}

Thread model
------------
`AgentBrain.submit` snapshots the context and returns at once; the HTTP call runs on a ThreadPoolExecutor.
Results go to the optional `on_decision(decision)` callback (called on a worker thread, or on the caller's
thread for immediate fallbacks: keep it short and thread-safe) and to a bounded queue drained by `poll()`.
In a ROS node, call `poll()` from a timer and act on the decisions in the executor thread. The clock used for
de-duplication, back-off and staleness is injectable and defaults to wall-clock `time.monotonic` (the cloud call
takes wall time, whatever the simulation speed).

On-device projection (an ESTIMATE, not a measurement)
-----------------------------------------------------
`project_latency` is a transparent roofline model: 4-bit weights (0.5 B/param) + 1.5 GB runtime must fit in 80 % of
device RAM; decode tok/s ≈ 0.6 × memory bandwidth / (active params × 0.5 B); prefill tok/s ≈ 0.3 × dense INT8 TOPS /
(2 × active params). Every assumption is a named constant below and is returned with the result. gemma-4-26b-a4b-it
needs ~14.5 GB, so it does NOT fit an 8 GB Jetson Orin Nano (nor a 16 GB Orin NX): on those profiles the local
rules are the on-device brain and the HUD must say so (`Decision.projected_note`). `gemma-edge-4b` is a 4B-dense
edge-class reference so the HUD can show what would fit a small Jetson.
"""
from __future__ import annotations

import copy
import email.utils
import itertools
import json
import math
import numbers
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------------------------------- vocabulary
EVENTS = ('novel_hazard', 'low_confidence', 'critical_damage', 'peer_sos', 'operator_command')
ACTIONS = ('SLOW_AND_REPLAN', 'REQUEST_CLARIFICATION', 'DEPLOY_RECOVERY_ROVER', 'ADJUST_DRONE_ALTITUDE',
           'REJECT_UNSAFE_COMMAND')
SLOW, CLARIFY, DEPLOY, ALTITUDE, REJECT = ACTIONS

DEFAULT_MODEL = 'gemma-4-26b-a4b-it'
DEFAULT_DEVICE = 'orin_nano_15w'
LOCAL_BRAIN = 'local-rules'
SOURCE_GEMMA = 'gemma'
SOURCE_LOCAL = 'local-fallback'
API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
API_KEY_ENV = 'GEMINI_API_KEY'
ENV_FILE_ENV = 'REDSHIFT_ENV_FILE'

# ---------------------------------------------------------------------------------------------------- safety envelope
SLOW_FRACTION = 0.35          # SLOW_AND_REPLAN default cap = 35 % of the local limit (agent_fsm.CAUTION_SPEED)
CLARIFY_CONF = 0.25           # fused perception confidence below this → hold and ask instead of creeping on
STOP_DIST_M = 2.5             # hazard closer than this → cap 0 m/s (agent_fsm.STOP_DIST_M)
DRONE_ALT_MIN_M = 6.0
DRONE_ALT_MAX_M = 20.0
DRONE_DEFAULT_AGL_M = 12.0    # scout cruise AGL when the context does not report one
DRONE_DESCENT_M = 4.0         # dust / wind on the drone → descend this much (never below DRONE_ALT_MIN_M)
DRONE_WIND_LIMIT_MPS = 8.0
REASONING_MAX_CHARS = 300
SUMMARY_MAX_CHARS = 800
SPEED_ROUNDING_MPS = 0.01     # Gemma answers with two decimals: within this of the limit is rounding, not a request
HOLD_EVENTS = ('novel_hazard', 'low_confidence', 'critical_damage')   # REQUEST_CLARIFICATION holds at 0 m/s

# ---------------------------------------------------------------------------------------------------- client / brain
DEFAULT_TIMEOUT_S = 8.0
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 200
DEFAULT_MIN_INTERVAL_S = 3.0
DEFAULT_DEADLINE_S = 10.0
BACKOFF_INITIAL_S = 2.0
BACKOFF_MAX_S = 30.0
RETRY_AFTER_MAX_S = 120.0     # a server-sent Retry-After is honoured up to this sanity bound
RATE_LIMIT_STATUS = (429, 503)

INSTRUCTIONS = (
    'You are the advisory reasoning layer of Redshift, a planetary rover fleet. The on-board safety layer has '
    'already reacted to the event and keeps authority: your answer may only make the robot safer (slower, replan, '
    'hold, ask, refuse), never faster. Actions: SLOW_AND_REPLAN = cap speed and replan around the hazard; '
    'REQUEST_CLARIFICATION = ask the operator (hold if sensing is unreliable); DEPLOY_RECOVERY_ROVER = send the '
    'recovery rover to a disabled robot; ADJUST_DRONE_ALTITUDE = set drone height above ground, 6-20 m; '
    'REJECT_UNSAFE_COMMAND = refuse an operator command that exceeds v_limit_mps or drives into a known hazard. '
    'known_hazards are already on the map and the planner routes around them: a command drives into one only if '
    'it names it. requested_mps is the speed an operator command asks for ("unbounded" is above any limit): refuse '
    'the command if requested_mps is above v_limit_mps or it drives into a known hazard; otherwise answer '
    'SLOW_AND_REPLAN at requested_mps. Always give target_speed_mps: a cap in m/s, never above v_limit_mps (0 = hold); give target_altitude_m with '
    'ADJUST_DRONE_ALTITUDE. Answer in JSON; reasoning_trace is one sentence of at most 30 words citing the values '
    'you used and matching your numbers.')

# ---------------------------------------------------------------------------------------------------- projection
BYTES_PER_PARAM = 0.5               # 4-bit quantised weights
RUNTIME_OVERHEAD_GB = 1.5           # runtime, activations, KV cache for a ~200-token context, CUDA context
USABLE_RAM_FRACTION = 0.8           # unified memory shared with the OS, ROS nodes and perception
DECODE_BW_EFFICIENCY = 0.6          # achieved / peak memory bandwidth while streaming weights (batch 1)
PREFILL_COMPUTE_EFFICIENCY = 0.3    # achieved / peak dense INT8 TOPS during prefill
FLOPS_PER_PARAM_PER_TOKEN = 2       # one multiply-add per active parameter per token


@dataclass(frozen=True)
class DeviceSpec:
    id: str
    label: str
    ram_gb: float
    bandwidth_gbs: float
    int8_tops: float       # DENSE INT8 TOPS (NVIDIA's headline figures are 2:4-sparse, i.e. 2× these)
    note: str = ''


@dataclass(frozen=True)
class ModelSpec:
    id: str
    total_params_b: float  # all weights resident (MoE: every expert)
    active_params_b: float  # weights touched per token
    arch: str
    note: str = ''


# ids match edge_budget.PROFILES
DEVICES: dict[str, DeviceSpec] = {d.id: d for d in [
    DeviceSpec('orin_nano_15w', 'Jetson Orin Nano 8GB · 15 W', 8.0, 68.0, 20.0),
    DeviceSpec('orin_nx_25w', 'Jetson Orin NX 16GB · 25 W', 16.0, 102.4, 50.0),
    DeviceSpec('agx_orin_60w', 'Jetson AGX Orin 64GB · 60 W', 64.0, 204.8, 137.0),
    DeviceSpec('rpi5_8w', 'Raspberry Pi 5 8GB · 8 W (CPU)', 8.0, 17.0, 0.2, 'CPU NEON, no accelerator'),
]}
MODELS: dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec('gemma-4-26b-a4b-it', 26.0, 4.0, 'MoE', 'the cloud model called here (~26B total / ~4B active)'),
    ModelSpec('gemma-4-31b-it', 31.0, 31.0, 'dense'),
    ModelSpec('gemma-edge-4b', 4.0, 4.0, 'dense', 'edge-class 4B reference for comparison, not deployed'),
]}


# ==================================================================================================== decision
@dataclass
class Decision:
    action: str
    target_speed_mps: float | None      # speed cap for the robot (None = leave the local plan's speed alone)
    target_altitude_m: float | None     # drone AGL (only with ADJUST_DRONE_ALTITUDE)
    reasoning_trace: str                # ≤ REASONING_MAX_CHARS
    source: str                         # "gemma" | "local-fallback"
    model: str                          # model that produced the action (API modelVersion) or "local-rules"
    event: str
    latency_ms: float | None = None     # MEASURED cloud round-trip on this host (None when no call completed)
    tokens_in: int | None = None
    tokens_out: int | None = None
    projected_edge_ms: float | None = None   # roofline ESTIMATE on projected_device; None if the model doesn't fit
    projected_device: str = DEFAULT_DEVICE
    projected_fits: bool = False
    projected_note: str = ''            # human-readable, honest HUD line for the projection
    error: str | None = None            # why Gemma's answer could not be used (None if it was)
    adjusted: list = field(default_factory=list)   # every clamp / override the local layer applied
    target_robot: str | None = None     # recovery rover to deploy / drone to move
    robot: str | None = None
    request_id: str | None = None
    stale: bool = False                 # finished later than AgentBrain.deadline_s after submit
    age_ms: float | None = None         # submit → result

    def as_dict(self) -> dict:
        return asdict(self)


class GemmaError(RuntimeError):
    """Cloud call failed. Messages never contain the API key."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class GemmaRateLimited(GemmaError):
    def __init__(self, message: str, status: int, retry_after_s: float | None = None):
        super().__init__(message, status)
        self.retry_after_s = retry_after_s


@dataclass
class GemmaReply:
    raw: dict | None            # parsed JSON answer, None if unusable (see error)
    text: str
    model: str
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None
    error: str | None = None


# ==================================================================================================== context helpers
_ALIASES = {'speed_limit_mps': ('speed_limit_mps', 'v_limit'), 'speed_mps': ('speed_mps', 'speed'),
            'robot_type': ('robot_type', 'type'), 'drone_agl_m': ('drone_agl_m', 'altitude_agl')}


def _check_event(event: str):
    if event not in EVENTS:
        raise ValueError(f'unknown event {event!r}; expected one of {EVENTS}')


def _get(ctx: dict, key: str, default=None):
    for k in _ALIASES.get(key, (key,)):
        v = ctx.get(k)
        if v is not None:
            return v
    return default


def _num(v) -> float | None:
    """float(v) for real numbers / numeric strings; None for missing, bool, non-numeric or NaN."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, numbers.Real):
        f = float(v)
    elif isinstance(v, str):
        try:
            f = float(v.strip())
        except ValueError:
            return None
    else:
        return None
    return None if math.isnan(f) else f


def _r(v, nd: int = 2) -> float | None:
    f = _num(v)
    return round(f, nd) if f is not None and math.isfinite(f) else None


def _frac(v) -> float | None:
    f = _num(v)
    if f is None or not math.isfinite(f):
        return None
    if f > 1.0:
        f /= 100.0
    return max(0.0, min(1.0, f))


def _s(v, n: int) -> str | None:
    if v is None or v == '':
        return None
    t = ' '.join(str(v).split())
    return t if len(t) <= n else t[:max(0, n - 1)] + '…'


def _limit(ctx: dict) -> float | None:
    """Effective local speed limit, or None if missing / not finite (callers then use 0 m/s)."""
    f = _num(_get(ctx, 'speed_limit_mps'))
    return max(0.0, f) if f is not None and math.isfinite(f) else None


def _fleet(ctx: dict) -> dict:
    f = ctx.get('fleet')
    return f if isinstance(f, dict) else {}


def _is_drone(ctx: dict) -> bool:
    return str(_get(ctx, 'robot_type', '')).lower() in ('scout', 'drone', 'uav', 'aerial')


def _drone_id(ctx: dict) -> str | None:
    d = _fleet(ctx).get('drone')
    if d:
        return d if isinstance(d, str) else 'drone'
    return str(ctx.get('robot') or 'drone') if _is_drone(ctx) else None


def _drone_agl(ctx: dict) -> float | None:
    a = _r(_fleet(ctx).get('drone_agl_m'), 1)
    if a is None and _is_drone(ctx):
        a = _r(_get(ctx, 'drone_agl_m'), 1)
    return a


def _disabled_robot(event: str, ctx: dict) -> str | None:
    return ctx.get('peer') if event == 'peer_sos' else ctx.get('robot')


def _recovery(event: str, ctx: dict) -> tuple[str | None, str]:
    """(recovery rover that may be deployed, '') or (None, reason)."""
    f = _fleet(ctx)
    rr = f.get('recovery_rover')
    if not rr:
        return None, 'no recovery rover in the fleet'
    rr = rr if isinstance(rr, str) else 'recovery_rover'
    if not f.get('recovery_available', True):
        return None, f'recovery rover {rr} is not available'
    dis = _disabled_robot(event, ctx)
    if dis and rr == dis:
        return None, f'recovery rover {rr} is the disabled robot'
    return rr, ''


def _not_permitted(action: str, event: str, ctx: dict) -> str | None:
    if action == DEPLOY:
        if event not in ('critical_damage', 'peer_sos'):
            return f'{DEPLOY} is only for critical_damage / peer_sos'
        rr, why = _recovery(event, ctx)
        return None if rr else why
    if action == ALTITUDE:
        return None if _drone_id(ctx) else 'no drone in the fleet'
    if action == REJECT:
        return None if event == 'operator_command' else 'there is no operator command to reject'
    return None


def allowed_actions(event: str, context: dict) -> tuple:
    """Actions the context permits (capabilities only; operator-command safety is judged in `validate`)."""
    _check_event(event)
    return tuple(a for a in ACTIONS if _not_permitted(a, event, context or {}) is None)


def _ctx_device(ctx: dict, default: str = DEFAULT_DEVICE) -> str:
    d = ctx.get('edge_profile')
    return d if d in DEVICES else default


_WEATHER_RE = re.compile(r'\b(dust|wind|gust|storm|sandstorm)', re.I)


def _drone_weather(ctx: dict) -> str | None:
    """Why dust / wind is affecting the drone (the robot is the drone, or fleet.drone_affected), else None."""
    if not (_is_drone(ctx) or _fleet(ctx).get('drone_affected')) or not _drone_id(ctx):
        return None
    wind = _num(ctx.get('wind_mps'))
    if wind is not None and math.isfinite(wind) and wind >= DRONE_WIND_LIMIT_MPS:
        return f'wind {wind:.0f} m/s'
    if ctx.get('dust'):
        return 'dust'
    m = _WEATHER_RE.search(str(ctx.get('hazard') or ''))
    return m.group(1).lower() if m else None


def _confidence(ctx: dict) -> float | None:
    """Best sensor confidence (the FSM can fall back to either sensor), capped by 1 − uncertainty."""
    vals = [v for v in (_frac(ctx.get('camera_conf')), _frac(ctx.get('lidar_conf'))) if v is not None]
    best = max(vals) if vals else None
    u = _frac(ctx.get('uncertainty'))
    if u is not None:
        best = (1.0 - u) if best is None else min(best, 1.0 - u)
    return best


def _sensor_txt(ctx: dict) -> str:
    parts = []
    for key, name in (('camera_conf', 'cam'), ('lidar_conf', 'LiDAR'), ('uncertainty', 'uncertainty')):
        v = _frac(ctx.get(key))
        if v is not None:
            parts.append(f'{name} {v:.2f}')
    return ', '.join(parts) or 'no confidence readings'


def _clamp_alt(a: float) -> float:
    return max(DRONE_ALT_MIN_M, min(DRONE_ALT_MAX_M, a))


def _descend_alt(ctx: dict) -> float:
    agl = _drone_agl(ctx)
    agl = DRONE_DEFAULT_AGL_M if agl is None else agl
    return round(_clamp_alt(min(agl, DRONE_ALT_MAX_M) - DRONE_DESCENT_M), 1)


# ==================================================================================================== summary
def _summary_size(s: dict) -> int:
    return len(summary_json(s))


def summary_json(summary: dict) -> str:
    return json.dumps(summary, separators=(',', ':'), ensure_ascii=False)


def build_summary(event: str, context: dict) -> dict:
    """Compact, JSON-serialisable sensor summary for the prompt (None fields dropped, ≤ SUMMARY_MAX_CHARS)."""
    _check_event(event)
    c = context or {}
    f = _fleet(c)
    rr = f.get('recovery_rover')
    fleet = {'recovery_rover': (rr if isinstance(rr, str) else 'yes') if rr else None,
             'recovery_ready': _recovery(event, c)[0] is not None if rr else None,
             'drone': _drone_id(c), 'drone_agl_m': _drone_agl(c)}
    fleet = {k: v for k, v in fleet.items() if v is not None}
    hz = [h for h in (_s(x, 32) for x in list(c.get('known_hazards') or [])[:6]) if h]
    s = {'event': event, 'robot': _s(c.get('robot'), 24), 'type': _s(_get(c, 'robot_type'), 12),
         'state': _s(c.get('state'), 20), 'v_mps': _r(_get(c, 'speed_mps')), 'v_limit_mps': _r(_limit(c)),
         'v_max_mps': _r(_get(c, 'v_max_mps')), 'health': _r(_frac(c.get('health'))),
         'battery': _r(_frac(c.get('battery'))), 'cam_conf': _r(_frac(c.get('camera_conf'))),
         'lidar_conf': _r(_frac(c.get('lidar_conf'))), 'uncertainty': _r(_frac(c.get('uncertainty'))),
         'obstacle_m': _r(c.get('nearest_obstacle_m'), 1), 'hazard': _s(c.get('hazard'), 120),
         'damage': _s(c.get('damage'), 100), 'peer': _s(c.get('peer'), 24), 'dust': True if c.get('dust') else None,
         'wind_mps': _r(c.get('wind_mps'), 1), 'known_hazards': hz or None, 'fleet': fleet or None,
         'operator': _s(_get(c, 'operator_text'), 160)}
    if event == 'operator_command' and s['operator']:
        # the local parser's reading of the asked-for speed ("full speed" = v_max_mps), so Gemma compares two
        # numbers instead of re-deriving the limit rule; None for commands that ask for no speed
        req = assess_operator_command(_get(c, 'operator_text'), c)['requested_speed_mps']
        s['requested_mps'] = 'unbounded' if req is not None and math.isinf(req) else _r(req)
    s = {k: v for k, v in s.items() if v is not None}
    while _summary_size(s) > SUMMARY_MAX_CHARS:
        if s.get('known_hazards'):
            s['known_hazards'].pop()
            if not s['known_hazards']:
                del s['known_hazards']
            continue
        longest = max((k for k in ('hazard', 'damage', 'operator', 'state', 'robot', 'peer', 'type') if k in s),
                      key=lambda k: len(s[k]), default=None)
        if longest is None or len(s[longest]) <= 8:
            break
        s[longest] = _s(s[longest], max(8, len(s[longest]) - 24))
    return s


# ==================================================================================================== operator text
_NEG_RE = re.compile(r"\b(?:don'?t|do not|never|not|no|avoid|avoiding|around|bypass|skirt|away from|clear of|"
                     r"out of)\b")
_STOP_RE = re.compile(r'\b(stop|halt|hold|freeze|pause|wait|abort|e-?stop|stand ?by|stay put|brake)\b')
_SLOW_RE = re.compile(r'\b(slow(?:er)?|careful(?:ly)?|cautious(?:ly)?|reduce speed|ease off|gently)\b')
_FULL_RE = re.compile(r'\b(full (?:speed|throttle|power)|max(?:imum)? speed|top speed|flat out|floor it|'
                      r'as fast as|gun it|pedal to the metal)\b')
_FAST_RE = re.compile(r'\b(fast(?:er)?|speed up|hurry|quick(?:er|ly)?|rush|sprint|accelerate|boost)\b')
_ENTER_RE = re.compile(r'\b(through|into|across|over|onto|enter|cross|traverse|straight|via|plough|plow|ram)\b')
_SPEED_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(m/s|mps|km/h|kph|kmh|met(?:er|re)s? per second)')
_ALT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)\b(?!\s*/|\s*per)')
_DRONE_WORDS = r'drone|uav|quad(?:copter)?|aerial'
_DOWN_RE = re.compile(r'\b(lower|descend|down|drop)\b')
_UP_RE = re.compile(r'\b(higher|climb|ascend|up|raise)\b')
GENERIC_HAZARDS = ('boulder', 'crater', 'rockslide', 'rock slide', 'cliff', 'ledge', 'pillar', 'debris', 'fissure',
                   'crevasse', 'sinkhole', 'pit', 'rubble', 'dust storm', 'hazard', 'lava tube')


_CLAUSE_RE = re.compile(r"[,;.!?]|\b(?:then|and|but)\b")


def _words_before(t: str, pos: int, n: int) -> str:
    """Last n words before pos, within the same clause ("don't stop, go through X": the negation is not X's)."""
    start = max((m.end() for m in _CLAUSE_RE.finditer(t, 0, pos)), default=0)
    return ' '.join(re.findall(r"[\w'/-]+", t[start:pos])[-n:])


def _live(rx: re.Pattern, t: str):
    """First match of rx that is not negated by the 3 words before it ("don't stop" is not a stop)."""
    for m in rx.finditer(t):
        if not _NEG_RE.search(_words_before(t, m.start(), 3)):
            return m
    return None


def _hazard_entry(t: str, ctx: dict) -> str | None:
    """Name of a hazard the text drives INTO ("through the boulder field"), ignoring "around/avoid/don't ..."."""
    names = [str(h).lower().strip() for h in (ctx.get('known_hazards') or []) if h] + list(GENERIC_HAZARDS)
    for name in (n for n in names if n):
        for m in re.finditer(r'\b' + re.escape(name), t):
            tail = _words_before(t, m.start(), 5)
            if _ENTER_RE.search(tail) and not _NEG_RE.search(tail):
                return name
    return None


def assess_operator_command(text: str, context: dict) -> dict:
    """Deterministic local reading of an operator command.

    Returns {intent, requested_speed_mps, requested_altitude_m, hazard, exceeds_limit, unsafe, reasons} with intent
    one of stop | slow | set_speed | faster | drone_altitude | enter_hazard | unclear. `unsafe` = asks for more than
    the local limit, or drives into a known (or generic) hazard.
    """
    ctx = context or {}
    t = ' '.join(str(text or '').lower().replace('’', "'").split())
    lim = _limit(ctx) or 0.0
    v_max = _num(_get(ctx, 'v_max_mps'))
    drone = _drone_id(ctx)
    drone_rx = re.compile(r'\b(' + _DRONE_WORDS + (('|' + re.escape(drone.lower())) if drone else '') + r')\b')
    sm = _SPEED_RE.search(t)
    num = (float(sm.group(1)) / (3.6 if sm.group(2).startswith('k') else 1.0)) if sm else None
    hazard = _hazard_entry(t, ctx)
    intent, req, alt = 'unclear', None, None
    if _live(_STOP_RE, t):
        intent, req = 'stop', 0.0
    elif _live(_FULL_RE, t):
        intent, req = 'faster', (v_max if v_max is not None and math.isfinite(v_max) else math.inf)
    elif num is not None:
        intent, req = 'set_speed', num
    elif _live(_FAST_RE, t):
        intent, req = 'faster', math.inf          # unbounded increase: the advisory layer never raises speed
    elif _live(_SLOW_RE, t):
        intent, req = 'slow', round(SLOW_FRACTION * lim, 3)
    elif drone_rx.search(t) and (_ALT_RE.search(t) or _DOWN_RE.search(t) or _UP_RE.search(t)):
        intent = 'drone_altitude'
        am = _ALT_RE.search(t)
        agl = _drone_agl(ctx)
        agl = DRONE_DEFAULT_AGL_M if agl is None else agl
        if am:
            alt = float(am.group(1))
        elif _DOWN_RE.search(t):
            alt = agl - DRONE_DESCENT_M
        else:
            alt = agl + DRONE_DESCENT_M
        alt = round(_clamp_alt(alt), 1)
    elif hazard:
        intent = 'enter_hazard'
    exceeds = req is not None and req > lim + 1e-9
    reasons = []
    if exceeds:
        asked = 'an unbounded speed increase' if math.isinf(req) else f'{req:.2f} m/s'
        reasons.append(f'asks for {asked} > local limit {lim:.2f} m/s')
    if hazard:
        reasons.append(f"drives into known hazard '{hazard}'")
    return {'intent': intent, 'requested_speed_mps': req, 'requested_altitude_m': alt, 'hazard': hazard,
            'exceeds_limit': exceeds, 'unsafe': bool(exceeds or hazard), 'reasons': reasons}


# ==================================================================================================== projection
def _model_key(name: str | None) -> str | None:
    if not name:
        return None
    n = str(name).split('/')[-1]
    if n in MODELS:
        return n
    return next((k for k in sorted(MODELS, key=len, reverse=True) if n.startswith(k)), None)


def project_latency(tokens_in: int, tokens_out: int, device_id: str = DEFAULT_DEVICE,
                    model_id: str = DEFAULT_MODEL) -> dict:
    """Roofline ESTIMATE (not a measurement) of running `model_id` on `device_id` for one request.

    fits        = weights (total params × 0.5 B) + 1.5 GB ≤ 0.8 × device RAM
    decode tok/s  = 0.6 × bandwidth / (active params × 0.5 B)     (weight streaming bound, batch 1)
    prefill tok/s = 0.3 × dense INT8 TOPS / (2 × active params)   (compute bound)
    Ignores KV-cache traffic, dequantisation overhead, MoE routing / expert-locality effects, thermal throttling
    and scheduling; the MoE decode figure is optimistic (assumes only the active experts are read per token).
    """
    if device_id not in DEVICES:
        raise ValueError(f'unknown device {device_id!r}; expected one of {tuple(DEVICES)}')
    mk = _model_key(model_id)
    if mk is None:
        raise ValueError(f'unknown model {model_id!r}; expected one of {tuple(MODELS)}')
    dev, m = DEVICES[device_id], MODELS[mk]
    tin, tout = max(0, int(tokens_in or 0)), max(0, int(tokens_out or 0))
    weights_gb = m.total_params_b * BYTES_PER_PARAM
    required_gb = weights_gb + RUNTIME_OVERHEAD_GB
    usable_gb = USABLE_RAM_FRACTION * dev.ram_gb
    active_bytes = m.active_params_b * 1e9 * BYTES_PER_PARAM
    decode_tok_s = DECODE_BW_EFFICIENCY * dev.bandwidth_gbs * 1e9 / active_bytes
    prefill_tok_s = PREFILL_COMPUTE_EFFICIENCY * dev.int8_tops * 1e12 / (FLOPS_PER_PARAM_PER_TOKEN
                                                                         * m.active_params_b * 1e9)
    prefill_ms = 1000.0 * tin / prefill_tok_s
    decode_ms = 1000.0 * tout / decode_tok_s
    return {
        'device': dev.id, 'device_label': dev.label, 'model': m.id, 'estimate': True,
        'fits': required_gb <= usable_gb, 'weights_gb': round(weights_gb, 2), 'required_gb': round(required_gb, 2),
        'usable_gb': round(usable_gb, 2), 'tokens_in': tin, 'tokens_out': tout,
        'prefill_tok_s': round(prefill_tok_s, 1), 'decode_tok_s': round(decode_tok_s, 2),
        'prefill_ms': round(prefill_ms, 1), 'decode_ms': round(decode_ms, 1),
        'total_ms': round(prefill_ms + decode_ms, 1),
        'assumptions': {
            'bytes_per_param': BYTES_PER_PARAM, 'runtime_overhead_gb': RUNTIME_OVERHEAD_GB,
            'usable_ram_fraction': USABLE_RAM_FRACTION, 'decode_bandwidth_efficiency': DECODE_BW_EFFICIENCY,
            'prefill_compute_efficiency': PREFILL_COMPUTE_EFFICIENCY,
            'flops_per_param_per_token': FLOPS_PER_PARAM_PER_TOKEN,
            'device_ram_gb': dev.ram_gb, 'device_bandwidth_gbs': dev.bandwidth_gbs,
            'device_int8_dense_tops': dev.int8_tops, 'model_total_params_b': m.total_params_b,
            'model_active_params_b': m.active_params_b, 'model_arch': m.arch,
            'note': 'roofline estimate, not measured; ignores KV cache, dequantisation, MoE routing and throttling',
        },
    }


def _projection(tokens_in, tokens_out, device_id: str, model_id: str | None) -> dict:
    out = {'projected_edge_ms': None, 'projected_device': device_id, 'projected_fits': False}
    mk = _model_key(model_id)
    if device_id not in DEVICES or mk is None:
        out['projected_note'] = f'no on-device profile for {model_id} on {device_id}'
        return out
    p = project_latency(tokens_in or 0, tokens_out or 0, device_id, mk)
    label = DEVICES[device_id].label
    out['projected_fits'] = p['fits']
    if not p['fits']:
        out['projected_note'] = (f"{mk} needs ~{p['required_gb']:.1f} GB > {p['usable_gb']:.1f} GB usable on {label}: "
                                 f'on this device the local rules are the on-device brain')
    elif tokens_in is None and tokens_out is None:
        out['projected_note'] = f"{mk} would fit {label} (~{p['required_gb']:.1f} GB); no tokens to project"
    else:
        out['projected_edge_ms'] = p['total_ms']
        out['projected_note'] = f"projected on {label}: ~{p['total_ms']:.0f} ms (roofline estimate, not measured)"
    return out


# ==================================================================================================== decisions
def _clean(text, n: int = REASONING_MAX_CHARS) -> str:
    t = ' '.join(str(text or '').split())
    return t if len(t) <= n else t[:n - 1] + '…'


def _decision(event: str, ctx: dict, action: str, speed, alt, reasoning: str, *, source: str, model: str,
              cloud_model: str | None = None, latency_ms=None, tokens_in=None, tokens_out=None,
              device_id: str | None = None, error: str | None = None, adjusted=None) -> Decision:
    dev = device_id or _ctx_device(ctx)
    target = _recovery(event, ctx)[0] if action == DEPLOY else _drone_id(ctx) if action == ALTITUDE else None
    return Decision(
        action=action, target_speed_mps=None if speed is None else round(max(0.0, float(speed)), 3),
        target_altitude_m=None if alt is None else round(float(alt), 1), reasoning_trace=_clean(reasoning),
        source=source, model=model, event=event,
        latency_ms=None if latency_ms is None else round(float(latency_ms), 1),
        tokens_in=tokens_in, tokens_out=tokens_out, error=error, adjusted=list(adjusted or []),
        target_robot=target, robot=ctx.get('robot'),
        **_projection(tokens_in, tokens_out, dev, cloud_model or model))


def _rules(event: str, ctx: dict) -> tuple:
    """The deterministic on-device brain: (action, target_speed_mps, target_altitude_m, reasoning)."""
    lim = _limit(ctx) or 0.0
    slow = round(SLOW_FRACTION * lim, 3)
    if event in ('novel_hazard', 'low_confidence'):
        wx = _drone_weather(ctx)
        if wx:
            agl = _drone_agl(ctx)
            alt = _descend_alt(ctx)
            at = f' at {agl:.0f} m AGL' if agl is not None else ''
            return ALTITUDE, None, alt, (f'Local rules: {wx} affecting the drone{at}; descend to {alt:.0f} m for '
                                         f'ground contrast and calmer air.')
    if event == 'low_confidence':
        conf, sens = _confidence(ctx), _sensor_txt(ctx)
        if conf is not None and conf < CLARIFY_CONF:
            return CLARIFY, 0.0, None, (f'Local rules: perception confidence {conf:.2f} < {CLARIFY_CONF:.2f} '
                                        f'({sens}); hold at 0 m/s and ask the operator.')
        return SLOW, slow, None, (f'Local rules: low perception confidence ({sens}); cap {slow:.2f} m/s '
                                  f'({SLOW_FRACTION:.0%} of {lim:.2f}) and replan.')
    if event == 'novel_hazard':
        haz = _s(ctx.get('hazard'), 80) or 'unidentified hazard'
        obs = _num(ctx.get('nearest_obstacle_m'))
        if obs is not None and obs < STOP_DIST_M:
            return SLOW, 0.0, None, f'Local rules: {haz} at {obs:.1f} m (< {STOP_DIST_M} m); stop and replan around it.'
        at = f' at {obs:.1f} m' if obs is not None and math.isfinite(obs) else ''
        return SLOW, slow, None, f'Local rules: novel hazard ({haz}){at}; cap {slow:.2f} m/s and replan around it.'
    if event in ('critical_damage', 'peer_sos'):
        dis = _disabled_robot(event, ctx) or 'the disabled robot'
        rr, why = _recovery(event, ctx)
        hold = 0.0 if event == 'critical_damage' else None
        if event == 'critical_damage':
            h = _frac(ctx.get('health'))
            what = f"critical damage on {dis} ({_s(ctx.get('damage'), 80) or 'unspecified'}" + (
                f', health {h:.0%})' if h is not None else ')')
        else:
            what = f'SOS from {dis}' + (f" ({_s(ctx.get('hazard') or ctx.get('damage'), 60)})"
                                        if ctx.get('hazard') or ctx.get('damage') else '')
        if rr:
            return DEPLOY, hold, None, f'Local rules: {what}; deploy recovery rover {rr} to {dis}.'
        return CLARIFY, hold, None, f'Local rules: {what}; {why}, so hold and ask the operator.'
    # operator_command
    text = _get(ctx, 'operator_text') or ''
    a = assess_operator_command(text, ctx)
    quoted = _s(text, 60) or '(empty)'
    if a['unsafe']:
        return REJECT, (0.0 if a['intent'] == 'stop' else None), None, (
            f"Local rules: rejected '{quoted}': {'; '.join(a['reasons'])}. Keeping the local plan.")
    if a['intent'] in ('stop', 'slow', 'set_speed', 'faster'):
        cap = min(a['requested_speed_mps'], lim)
        verb = {'stop': 'operator stop', 'slow': 'operator slow-down'}.get(a['intent'], 'operator speed request')
        return SLOW, cap, None, f'Local rules: {verb} accepted; cap {cap:.2f} m/s (local limit {lim:.2f}) and replan.'
    if a['intent'] == 'drone_altitude':
        if _drone_id(ctx):
            alt = a['requested_altitude_m']
            return ALTITUDE, None, alt, f'Local rules: operator drone altitude request; set {alt:.0f} m AGL (6-20 m).'
        return CLARIFY, None, None, f"Local rules: '{quoted}' asks for a drone but none is in the fleet; clarify."
    return CLARIFY, None, None, f"Local rules: command '{quoted}' is unclear; please restate it."


def _fallback(event: str, ctx: dict, *, error=None, adjusted=None, cloud_model=DEFAULT_MODEL, latency_ms=None,
              tokens_in=None, tokens_out=None, device_id=None, override_from: str | None = None,
              why: str = '') -> Decision:
    action, speed, alt, reasoning = _rules(event, ctx)
    adj = list(adjusted or [])
    if override_from:
        adj.append(f'action {override_from} → {action} ({why})')
    return _decision(event, ctx, action, speed, alt, reasoning, source=SOURCE_LOCAL, model=LOCAL_BRAIN,
                     cloud_model=cloud_model, latency_ms=latency_ms, tokens_in=tokens_in, tokens_out=tokens_out,
                     device_id=device_id, error=error, adjusted=adj)


def deterministic_fallback(event: str, context: dict, *, error: str | None = None, device_id: str | None = None,
                           cloud_model: str = DEFAULT_MODEL) -> Decision:
    """Rule-based decision in the same schema (no randomness). `cloud_model` only feeds the projection fields."""
    _check_event(event)
    return _fallback(event, context or {}, error=error, cloud_model=cloud_model, device_id=device_id)


def validate(raw, event: str, context: dict, *, model: str = DEFAULT_MODEL, latency_ms: float | None = None,
             tokens_in: int | None = None, tokens_out: int | None = None,
             device_id: str | None = None) -> Decision:
    """Turn Gemma's raw JSON answer into a Decision that is never less safe than the local layer."""
    _check_event(event)
    ctx = context or {}
    meta = dict(cloud_model=model, latency_ms=latency_ms, tokens_in=tokens_in, tokens_out=tokens_out,
                device_id=device_id)
    if not isinstance(raw, dict):
        return _fallback(event, ctx, error='invalid answer: not a JSON object', **meta)
    adjusted: list[str] = []
    a_raw = raw.get('action')
    action = re.sub(r'[\s-]+', '_', a_raw.strip()).upper() if isinstance(a_raw, str) else None
    if action not in ACTIONS:
        return _fallback(event, ctx, error=f'invalid answer: action {str(a_raw)[:40]!r} is not one of ACTIONS', **meta)
    if action != a_raw:
        adjusted.append(f'action {a_raw!r} normalised to {action}')
    reasoning = raw.get('reasoning_trace')
    if not isinstance(reasoning, str) or not reasoning.strip():
        return _fallback(event, ctx, error='invalid answer: missing reasoning_trace', **meta)
    why = _not_permitted(action, event, ctx)
    if why:
        return _fallback(event, ctx, error=f'gemma proposed {action}: not permitted ({why})', adjusted=adjusted,
                         override_from=action, why=why, **meta)

    lim = _limit(ctx)
    if lim is None:
        adjusted.append('local speed limit unknown → treated as 0 m/s')
        lim = 0.0
    speed, alt = _num(raw.get('target_speed_mps')), _num(raw.get('target_altitude_m'))
    for key, val in (('target_speed_mps', speed), ('target_altitude_m', alt)):
        if raw.get(key) is not None and val is None:
            adjusted.append(f'{key} {str(raw.get(key))[:20]!r} is not a number → ignored')

    op = None
    if event == 'operator_command':
        op = assess_operator_command(_get(ctx, 'operator_text') or '', ctx)
        if action != REJECT and op['unsafe']:
            return _fallback(event, ctx, adjusted=adjusted, override_from=action,
                             why='local check: ' + '; '.join(op['reasons']), **meta)
        honour = op['intent'] in ('stop', 'slow', 'set_speed')     # safe request: clamp below, never reject
        if action != REJECT and not honour and speed is not None and speed > lim + SPEED_ROUNDING_MPS:
            adjusted.append(f'action {action} → {REJECT} (gemma proposed {speed:g} m/s > local limit {lim:.2f})')
            return _decision(event, ctx, REJECT, None, None,
                             f'Local safety layer: operator command implies {speed:g} m/s > local limit {lim:.2f} m/s; '
                             f'rejected, keeping the local plan.', source=SOURCE_LOCAL, model=LOCAL_BRAIN,
                             adjusted=adjusted, **meta)
        if action == REJECT and honour:
            return _fallback(event, ctx, adjusted=adjusted, override_from=action,
                             why='operator stop / slow-down requests are always honoured', **meta)

    # ---- target speed: a cap in [0, local limit]
    if speed is None:
        if action == SLOW:
            speed = round(SLOW_FRACTION * lim, 3)
            adjusted.append(f'target_speed_mps missing → {speed:.2f} ({SLOW_FRACTION:.0%} of local limit)')
        elif (action == CLARIFY and event in HOLD_EVENTS) or (action == DEPLOY and event == 'critical_damage'):
            speed = 0.0
    elif speed < 0.0:
        adjusted.append(f'target_speed_mps {speed:g} → 0')
        speed = 0.0
    elif speed > lim:
        adjusted.append(f'target_speed_mps {speed:g} → {lim:.2f} (local limit)')
        speed = lim
    if op is not None and op['intent'] in ('stop', 'slow', 'set_speed'):
        cap = min(op['requested_speed_mps'], lim)
        if speed is None or speed > cap + 1e-9:
            adjusted.append(f'target_speed_mps {"none" if speed is None else f"{speed:.2f}"} → {cap:.2f} '
                            f'(operator asked for {cap:.2f})')
            speed = cap

    # ---- drone altitude: only with ADJUST_DRONE_ALTITUDE, clamped to [6, 20] m
    if action == ALTITUDE:
        if alt is None:
            alt = _descend_alt(ctx)
            adjusted.append(f'target_altitude_m missing → {alt:.1f}')
        elif not DRONE_ALT_MIN_M <= alt <= DRONE_ALT_MAX_M:
            c = _clamp_alt(alt)
            adjusted.append(f'target_altitude_m {alt:g} → {c:g} ({DRONE_ALT_MIN_M:g}-{DRONE_ALT_MAX_M:g} m)')
            alt = c
    elif alt is not None:
        adjusted.append(f'target_altitude_m ignored for {action}')
        alt = None

    trimmed = ' '.join(reasoning.split())
    if len(trimmed) > REASONING_MAX_CHARS:
        adjusted.append(f'reasoning_trace truncated to {REASONING_MAX_CHARS} chars')
    return _decision(event, ctx, action, speed, alt, trimmed, source=SOURCE_GEMMA, model=model or DEFAULT_MODEL,
                     adjusted=adjusted, **meta)


# ==================================================================================================== API key
def _read_env_value(path: Path, name: str) -> str | None:
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    for line in lines:
        s = line.strip()
        if s.startswith('export '):
            s = s[7:].lstrip()
        if not s or s.startswith('#') or '=' not in s:
            continue
        k, v = s.split('=', 1)
        if k.strip() != name:
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
            v = v[1:-1]
        else:
            v = v.split(' #', 1)[0].strip()
        return v or None
    return None


def _env_candidates():
    if os.environ.get(ENV_FILE_ENV):
        yield Path(os.environ[ENV_FILE_ENV])
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        yield d / '.env'
        if (d / '.git').exists() or (d / 'ros2_ws').is_dir():   # stop at the repo root
            return


def load_api_key(env_path: str | os.PathLike | None = None) -> str | None:
    """GEMINI_API_KEY from the environment, else from `env_path` or the repo's .env (found by walking up from this
    module to the repo root; REDSHIFT_ENV_FILE overrides). The key is returned only to the caller, never logged."""
    v = os.environ.get(API_KEY_ENV, '').strip()
    if v:
        return v
    for p in ([Path(env_path)] if env_path else _env_candidates()):
        v = _read_env_value(p, API_KEY_ENV)
        if v:
            return v
    return None


# ==================================================================================================== HTTP
Transport = Callable[[str, dict, bytes, float], tuple]


def urllib_transport(url: str, headers: dict, body: bytes, timeout_s: float) -> tuple:
    """POST via urllib.request → (status, headers, body). HTTP errors are returned, network errors raised."""
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        try:
            data = e.read()
        except Exception:  # noqa: BLE001 - body is best-effort
            data = b''
        return e.code, dict(e.headers.items()) if e.headers else {}, data


def _header(headers: dict, name: str):
    return next((v for k, v in (headers or {}).items() if str(k).lower() == name.lower()), None)


def _retry_after(headers: dict, data: bytes) -> float | None:
    """Retry-After header (seconds or HTTP date), else google.rpc.RetryInfo.retryDelay ("39s") in the body."""
    h = _header(headers, 'Retry-After')
    if h is not None:
        f = _num(h)
        if f is not None and math.isfinite(f):
            return max(0.0, f)
        try:
            dt = email.utils.parsedate_to_datetime(str(h))
            return max(0.0, dt.timestamp() - time.time())
        except (TypeError, ValueError):
            pass
    try:
        for d in json.loads(data or b'{}').get('error', {}).get('details', []) or []:
            m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)s\s*', str(d.get('retryDelay', '')))
            if m:
                return float(m.group(1))
    except (ValueError, AttributeError, TypeError):
        pass
    return None


def _error_message(data: bytes) -> str:
    try:
        return str(json.loads(data).get('error', {}).get('message', ''))[:200] or 'no message'
    except (ValueError, AttributeError, TypeError):
        return (data or b'')[:200].decode('utf-8', 'replace') or 'no message'


def _extract_json(text: str):
    t = (text or '').strip()
    if t.startswith('```'):
        t = re.sub(r'^```[a-zA-Z]*\s*|\s*```$', '', t).strip()
    for cand in (t, t[t.find('{'):t.rfind('}') + 1] if '{' in t and '}' in t else ''):
        if not cand:
            continue
        try:
            v = json.loads(cand)
        except ValueError:
            continue
        return v if isinstance(v, dict) else None
    return None


def parse_response(payload: dict, *, requested_model: str = DEFAULT_MODEL, latency_ms: float = 0.0) -> GemmaReply:
    """generateContent response → GemmaReply (text of candidates[0].content.parts, usageMetadata token counts)."""
    if not isinstance(payload, dict):
        return GemmaReply(None, '', requested_model, latency_ms, error='response is not a JSON object')
    usage = payload.get('usageMetadata') if isinstance(payload.get('usageMetadata'), dict) else {}
    tin, tout = usage.get('promptTokenCount'), usage.get('candidatesTokenCount')
    tin = tin if isinstance(tin, int) else None
    tout = tout if isinstance(tout, int) else None
    model = str(payload.get('modelVersion') or requested_model)
    cands = payload.get('candidates') if isinstance(payload.get('candidates'), list) else []
    if not cands or not isinstance(cands[0], dict):
        block = (payload.get('promptFeedback') or {}).get('blockReason') if isinstance(
            payload.get('promptFeedback'), dict) else None
        return GemmaReply(None, '', model, latency_ms, tin, tout,
                          error='no candidates' + (f' (blocked: {block})' if block else ''))
    c0 = cands[0]
    finish = c0.get('finishReason')
    content = c0.get('content') if isinstance(c0.get('content'), dict) else {}
    parts = content.get('parts') if isinstance(content.get('parts'), list) else []
    text = ''.join(str(p.get('text', '')) for p in parts if isinstance(p, dict) and not p.get('thought'))
    raw = _extract_json(text)
    err = None
    if raw is None:
        err = 'malformed JSON from model' + (f' (finishReason {finish})' if finish and finish != 'STOP' else '')
    return GemmaReply(raw, text, model, latency_ms, tin, tout, finish, err)


class GemmaClient:
    """Blocking REST client for Gemma on the Gemini API (generateContent, JSON mode + responseSchema).

    The key goes only into the `x-goog-api-key` header: never the URL, repr, logs or exception messages.
    `transport(url, headers, body_bytes, timeout_s) -> (status, headers, body_bytes)` is injectable for tests.
    `systemInstruction` was verified accepted by gemma-4-26b-a4b-it and gemma-4-31b-it (2026-11-07); if a model
    answers HTTP 400 about (system / developer) instructions, the client switches to inline instructions and retries.
    `thinkingConfig` is never sent (HTTP 400 on Gemma).
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S,
                 transport: Transport | None = None, *, env_path=None, device_id: str = DEFAULT_DEVICE,
                 temperature: float = TEMPERATURE, max_output_tokens: int = MAX_OUTPUT_TOKENS,
                 use_system_instruction: bool = True):
        self.model = model
        self.timeout_s = float(timeout_s)
        self.device_id = device_id if device_id in DEVICES else DEFAULT_DEVICE
        self.temperature = temperature
        self.max_output_tokens = int(max_output_tokens)
        self.use_system_instruction = use_system_instruction
        self._transport = transport or urllib_transport
        self._key = api_key or load_api_key(env_path)

    @property
    def has_key(self) -> bool:
        return bool(self._key)

    @property
    def url(self) -> str:
        return API_URL.format(model=urllib.parse.quote(self.model, safe='-._'))

    def __repr__(self) -> str:
        return (f'GemmaClient(model={self.model!r}, key={"set" if self._key else "missing"}, '
                f'timeout_s={self.timeout_s:g})')

    __str__ = __repr__

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, '***') if self._key else text

    def build_request(self, event: str, context: dict, *, system_instruction: bool | None = None) -> dict:
        _check_event(event)
        ctx = context or {}
        sys_inst = self.use_system_instruction if system_instruction is None else system_instruction
        summary = 'Event summary (JSON): ' + summary_json(build_summary(event, ctx))
        schema = {'type': 'OBJECT',
                  'properties': {'action': {'type': 'STRING', 'enum': list(allowed_actions(event, ctx))},
                                 'target_speed_mps': {'type': 'NUMBER'}, 'target_altitude_m': {'type': 'NUMBER'},
                                 'reasoning_trace': {'type': 'STRING'}},
                  'required': ['action', 'reasoning_trace']}
        body = {'contents': [{'role': 'user', 'parts': [{'text': summary if sys_inst else
                                                         INSTRUCTIONS + '\n\n' + summary}]}],
                'generationConfig': {'responseMimeType': 'application/json', 'responseSchema': schema,
                                     'temperature': self.temperature, 'maxOutputTokens': self.max_output_tokens}}
        if sys_inst:
            body['systemInstruction'] = {'parts': [{'text': INSTRUCTIONS}]}
        return body

    def _post(self, body: dict) -> tuple:
        data = json.dumps(body).encode('utf-8')
        t0 = time.monotonic()
        err = None
        try:
            status, hdrs, resp = self._transport(self.url, {'Content-Type': 'application/json',
                                                            'x-goog-api-key': self._key}, data, self.timeout_s)
        except Exception as e:  # noqa: BLE001 - classified below, re-raised outside the handler (no key in context)
            err = self._transport_error(e)
        if err is not None:
            raise err
        return int(status), dict(hdrs or {}), resp or b'', (time.monotonic() - t0) * 1000.0

    def _transport_error(self, e: Exception) -> GemmaError:
        reason = getattr(e, 'reason', None)
        if isinstance(e, TimeoutError) or isinstance(reason, TimeoutError) or 'timed out' in str(e).lower():
            return GemmaError(f'timeout after {self.timeout_s:g} s')
        if isinstance(e, urllib.error.URLError):
            return GemmaError(f'network error: {self._scrub(str(reason))[:160]}')
        return GemmaError(f'transport error: {type(e).__name__}: {self._scrub(str(e))[:160]}')

    def call(self, event: str, context: dict) -> GemmaReply:
        """One blocking generateContent round-trip. Raises GemmaRateLimited (429/503) or GemmaError."""
        if not self._key:
            raise GemmaError(f'no API key (set {API_KEY_ENV} or add it to the repo .env)')
        status, hdrs, data, ms = self._post(self.build_request(event, context))
        if (status == 400 and self.use_system_instruction
                and re.search(r'(system|developer)[ _]?instruction', _error_message(data), re.I)):
            self.use_system_instruction = False
            status, hdrs, data, ms2 = self._post(self.build_request(event, context))
            ms += ms2
        if status in RATE_LIMIT_STATUS:
            raise GemmaRateLimited(f'rate-limited (HTTP {status})', status, _retry_after(hdrs, data))
        if status != 200:
            raise GemmaError(f'HTTP {status}: {self._scrub(_error_message(data))}', status)
        try:
            payload = json.loads(data)
        except ValueError:
            return GemmaReply(None, '', self.model, ms, error='response body is not JSON')
        return parse_response(payload, requested_model=self.model, latency_ms=ms)

    def decide(self, event: str, context: dict, *, device_id: str | None = None,
               raise_rate_limited: bool = False) -> Decision:
        """Blocking: validated Gemma decision, or the deterministic fallback (with `error`) on any failure."""
        _check_event(event)
        ctx = context or {}
        dev = device_id if device_id in DEVICES else _ctx_device(ctx, self.device_id)

        def fb(err: str, reply: GemmaReply | None = None) -> Decision:
            return _fallback(event, ctx, error=err, cloud_model=self.model, device_id=dev,
                             latency_ms=reply.latency_ms if reply else None,
                             tokens_in=reply.tokens_in if reply else None,
                             tokens_out=reply.tokens_out if reply else None)

        if not self._key:
            return fb(f'no API key ({API_KEY_ENV} unset)')
        try:
            reply = self.call(event, ctx)
        except GemmaRateLimited as e:
            if raise_rate_limited:
                raise
            return fb(str(e))
        except GemmaError as e:
            return fb(f'{self.model}: {e}')
        except Exception as e:  # noqa: BLE001 - never let the advisory layer crash the caller
            return fb(f'{self.model}: internal error {type(e).__name__}')
        if reply.raw is None:
            return fb(f'{reply.model}: {reply.error}', reply)
        return validate(reply.raw, event, ctx, model=reply.model, latency_ms=reply.latency_ms,
                        tokens_in=reply.tokens_in, tokens_out=reply.tokens_out, device_id=dev)


# ==================================================================================================== async brain
class AgentBrain:
    """Non-blocking front end: `submit()` returns immediately, decisions arrive via callback and/or `poll()`.

    * De-duplication: the same (robot, event[, operator text]) inside `min_interval_s` (float, or {event: s}) is not
      re-sent; submit() returns the in-flight request id, or None if that request already finished.
    * Rate limits: HTTP 429/503 → back off (Retry-After / retryDelay if given, else 2 s doubling to 30 s). While
      backing off, submit() answers at once with the local fallback (error "rate-limited (backing off)").
    * No client / no key / closed pool: immediate local fallback. Results that took longer than `deadline_s` are
      still delivered, with `stale=True`.
    """

    def __init__(self, client: GemmaClient | None = None, *, max_workers: int = 2,
                 min_interval_s: float | dict = DEFAULT_MIN_INTERVAL_S, deadline_s: float = DEFAULT_DEADLINE_S,
                 on_decision: Callable[[Decision], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 backoff_initial_s: float = BACKOFF_INITIAL_S, backoff_max_s: float = BACKOFF_MAX_S,
                 max_results: int = 256, device_id: str | None = None):
        self.client = client
        self.min_interval_s = min_interval_s
        self.deadline_s = float(deadline_s)
        self.on_decision = on_decision
        self.clock = clock
        self.backoff_initial_s = float(backoff_initial_s)
        self.backoff_max_s = float(backoff_max_s)
        self.device_id = device_id
        self._lock = threading.Lock()
        self._results: deque = deque(maxlen=max_results)
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix='gemma-brain')
        self._last: dict = {}             # dedupe key -> (t_submit, request_id)
        self._inflight: set = set()
        self._ids = itertools.count(1)
        self._backoff_until = -math.inf
        self._backoff_n = 0
        self._closed = False
        self._stats = {k: 0 for k in ('submitted', 'deduped', 'cloud_calls', 'gemma', 'local', 'rate_limited',
                                      'stale', 'callback_errors')}

    # ------------------------------------------------------------------ public
    def interval_for(self, event: str) -> float:
        m = self.min_interval_s
        return float(m.get(event, DEFAULT_MIN_INTERVAL_S) if isinstance(m, dict) else m)

    def backoff_remaining_s(self) -> float:
        with self._lock:
            return max(0.0, self._backoff_until - self.clock())

    def submit(self, event: str, context: dict, robot: str | None = None) -> str | None:
        """Queue an advisory request; never blocks. Returns the request id (None if de-duplicated or closed)."""
        _check_event(event)
        ctx = copy.deepcopy(dict(context or {}))
        robot = str(robot or ctx.get('robot') or 'robot')
        ctx.setdefault('robot', robot)
        key = (robot, event, _get(ctx, 'operator_text') if event == 'operator_command' else None)
        now = self.clock()
        immediate = None
        with self._lock:
            if self._closed:
                return None
            last = self._last.get(key)
            if last is not None and now - last[0] < self.interval_for(event):
                self._stats['deduped'] += 1
                return last[1] if last[1] in self._inflight else None
            rid = f'{robot}:{event}:{next(self._ids)}'
            self._last[key] = (now, rid)
            if len(self._last) > 512:
                horizon = now - max(self.interval_for(e) for e in EVENTS)
                self._last = {k: v for k, v in self._last.items() if v[0] >= horizon}
            self._stats['submitted'] += 1
            if self.client is None:
                immediate = 'no Gemma client configured'
            elif not self.client.has_key:
                immediate = f'no API key ({API_KEY_ENV} unset)'
            elif now < self._backoff_until:
                immediate = 'rate-limited (backing off)'
                self._stats['rate_limited'] += 1
            else:
                try:
                    self._pool.submit(self._work, rid, event, ctx, robot, now)
                    self._inflight.add(rid)
                except RuntimeError:          # pool already shut down
                    immediate = 'brain closed'
        if immediate is not None:
            self._finish(self._local(event, ctx, immediate), rid, robot, now)
        return rid

    def poll(self) -> list:
        """All decisions finished since the last poll (oldest first)."""
        with self._lock:
            out = list(self._results)
            self._results.clear()
        return out

    def stats(self) -> dict:
        with self._lock:
            d = dict(self._stats)
            d['in_flight'] = len(self._inflight)
            d['backoff_remaining_s'] = round(max(0.0, self._backoff_until - self.clock()), 2)
        return d

    def close(self, wait: bool = True):
        """Stop accepting work and shut the pool down (queued requests are cancelled; running ones finish)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ internals
    def _local(self, event: str, ctx: dict, error: str) -> Decision:
        return _fallback(event, ctx, error=error, cloud_model=self.client.model if self.client else DEFAULT_MODEL,
                         device_id=self.device_id if self.device_id in DEVICES else
                         _ctx_device(ctx, self.client.device_id if self.client else DEFAULT_DEVICE))

    def _register_backoff(self, retry_after_s: float | None) -> float:
        with self._lock:
            self._backoff_n += 1
            if retry_after_s is not None:
                delay = min(max(0.0, retry_after_s), RETRY_AFTER_MAX_S)
            else:
                delay = min(self.backoff_initial_s * 2 ** (self._backoff_n - 1), self.backoff_max_s)
            self._backoff_until = max(self._backoff_until, self.clock() + delay)
            self._stats['rate_limited'] += 1
        return delay

    def _work(self, rid: str, event: str, ctx: dict, robot: str, t0: float):
        try:
            if self.backoff_remaining_s() > 0:
                with self._lock:
                    self._stats['rate_limited'] += 1
                d = self._local(event, ctx, 'rate-limited (backing off)')
            else:
                with self._lock:
                    self._stats['cloud_calls'] += 1
                try:
                    d = self.client.decide(event, ctx, device_id=self.device_id, raise_rate_limited=True)
                    with self._lock:
                        self._backoff_n = 0
                except GemmaRateLimited as e:
                    delay = self._register_backoff(e.retry_after_s)
                    d = self._local(event, ctx, f'rate-limited (HTTP {e.status}); backing off {delay:.0f} s')
        except Exception as e:  # noqa: BLE001 - a decision must always come back
            d = self._local(event, ctx, f'internal error: {type(e).__name__}')
        self._finish(d, rid, robot, t0)

    def _finish(self, d: Decision, rid: str, robot: str, t0: float):
        age = max(0.0, self.clock() - t0)
        d.request_id, d.robot = rid, robot
        d.age_ms = round(age * 1000.0, 1)
        d.stale = age > self.deadline_s
        with self._lock:
            self._inflight.discard(rid)
            self._stats['gemma' if d.source == SOURCE_GEMMA else 'local'] += 1
            self._stats['stale'] += int(d.stale)
            self._results.append(d)
            cb = None if self._closed else self.on_decision
        if cb is not None:
            try:
                cb(d)
            except Exception:  # noqa: BLE001 - a faulty callback must not kill the worker
                with self._lock:
                    self._stats['callback_errors'] += 1
