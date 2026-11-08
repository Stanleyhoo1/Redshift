"""Pure glue between the fleet's ROS messages and the advisory Gemma brain (gemma_agent_brain).

`BrainHub` is the whole of fleet_brain_node without rclpy: it turns decision traces and operator commands into
brain events, builds the context dict from the robots' latest agent_state / edge_telemetry / perception, submits to
AgentBrain and formats what the node publishes (/brain/decision and one "brain_advice" trace per decision). The
planners use `BrainCap` / `BrainAltitude` to keep their advisory cap bookkeeping on simulated time.

Authority: the local FSM / A* / safety layer has already acted when an event reaches the brain. A brain decision
can only make a robot safer: a speed cap is min()-ed with the local speed for 'follow' / 'dock' drives only, so it
never raises speed and never touches RECOVERY_REVERSE, safe-hold or back-off motions.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import json
import math
import numbers
import time
from dataclasses import dataclass, field
from typing import Callable

from .agent_fsm import unit_label
from .gemma_agent_brain import (ACTIONS, ALTITUDE, CLARIFY, DEFAULT_MODEL, DEPLOY, DRONE_ALT_MAX_M, DRONE_ALT_MIN_M,
                                REJECT, SLOW, SOURCE_GEMMA, AgentBrain, assess_operator_command,
                                deterministic_fallback)
from .trace import TraceBuilder

# ---------------------------------------------------------------------------------------------------- event mapping
HAZARD_KINDS = ('path_blocked', 'slope_hazard', 'hazard_marked')   # → novel_hazard
LOW_CONFIDENCE_KINDS = ('caution_enter',)                          # the FSM's CAUTION entry → low_confidence
# EventGate: which deferred trigger of the same (robot, event) is kept (higher wins; ties keep the first deferred)
TRIGGER_PRIORITY = {'path_blocked': 2, 'hazard_marked': 2, 'slope_hazard': 1}
DRONE_TYPES = ('scout', 'drone', 'uav', 'aerial')
DRONE_ONLY_TEXT = 'Drones take altitude commands only; select a rover for speed or stop commands'
WARN_EVERY_S = 5.0             # DropGuard: at most one "dropped a message" warning per this many seconds
# per-event de-duplication handed to AgentBrain (seconds, wall clock: the cloud call takes wall time)
MIN_INTERVAL_S = {'novel_hazard': 10.0, 'low_confidence': 10.0, 'critical_damage': 3.0, 'peer_sos': 3.0,
                  'operator_command': 0.5}
TEXT_MAX = 200                 # operator command length (backend validates the same bound)
INPUT_MAX = 160                # trace input carried as hazard / damage / trigger text

# FSM states in which the rover is holding or not under path-following control: the effective local limit is 0
HOLD_STATES = ('BOOT', 'SAFE_HOLD', 'AWAIT_RESCUE', 'REPAIRING', 'AWAIT_SCOUT', 'RECOVERY_REVERSE',
               'MISSION_COMPLETE')
DUST_LEVEL = 0.3               # /world/env_state dust_level at which the context reports dust
STATE_FRESH_S = 10.0           # wall seconds: older agent_state / perception is treated as missing
PERCEPTION_FRESH_S = 5.0

# ---------------------------------------------------------------------------------------------------- caps
OPERATOR_CAP_S = 60.0          # simulated seconds an operator speed cap lasts
EVENT_CAP_S = 15.0             # ... an event (hazard / confidence / damage / SOS) speed cap
EVENT_HOLD_S = 4.0             # ... an event hold (cap 0, e.g. REQUEST_CLARIFICATION)
ALTITUDE_S = 60.0              # ... an ADJUST_DRONE_ALTITUDE cruise AGL
HOLD_FLOOR_MPS = 0.05          # a cap below this is a hold: rounded to 0 (same expiry rules as a 0 cap)
SOS_EVENTS = ('critical_damage', 'peer_sos')       # advice about an SOS never caps a rover (the local rescue runs)
RESCUE_STATES = ('RESCUE_DOCKING', 'REPAIRING')    # event-slot caps are ignored here (operator caps still apply)
RELEASE_INTENTS = ('faster', 'set_speed')          # an operator command that lifts the operator's own cap / stop
CAPPED_DRIVES = ('follow', 'dock')                 # the only drives a cap applies to
TRACE_SUFFIX = ' (advisory; local safety layer keeps authority)'


def _num(v) -> float | None:
    """Finite float for a real number (not bool), else None. Never raises: JSON integers have any number of digits
    and float() of one too large for a double raises OverflowError."""
    if v is None or isinstance(v, bool) or not isinstance(v, numbers.Real):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def _r3(v) -> float | None:
    f = _num(v)
    return None if f is None else round(f, 3)


def _int(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _short(v, n: int) -> str:
    t = ' '.join(str(v or '').split())
    return t if len(t) <= n else t[:n - 1] + '…'


def loads_dict(data) -> dict | None:
    """json.loads that only ever returns a dict (malformed or non-object JSON → None, never raises)."""
    try:
        d = json.loads(data)
    except (TypeError, ValueError, RecursionError):
        return None
    return d if isinstance(d, dict) else None


def finite_json(obj):
    """Copy of obj with every non-finite float replaced by None: browsers' JSON.parse rejects Infinity / NaN
    (assess_operator_command reports an unbounded speed request as math.inf)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    return obj


def dumps(obj) -> str:
    return json.dumps(finite_json(obj), allow_nan=False)


@dataclass(frozen=True)
class EventSpec:
    """One brain event: what to ask about, for which robot, and what raised it."""
    event: str
    robot: str
    trigger: str                  # trace kind that raised it, or "operator"
    trigger_input: str = ''
    hazard: str | None = None
    damage: str | None = None
    peer: str | None = None
    state: str | None = None      # FSM state the trace was raised in (newer than the robot's last agent_state)

    @property
    def key(self) -> tuple:
        return self.robot, self.event


def is_brain_trace(trace: dict) -> bool:
    """Traces the brain path itself produced (brain_advice from fleet_brain_node, brain_cap from the planners)."""
    return 'brain' in trace or str(trace.get('kind', '')).startswith('brain_')


def event_for_trace(trace, robots, recovery_rover: str | None = None) -> EventSpec | None:
    """Map an /agent_decision_trace entry to a brain event, or None.

    path_blocked / slope_hazard / hazard_marked → novel_hazard (hazard = the trace input); caution_enter →
    low_confidence; critical_damage → critical_damage for the damaged rover; sos (active) → peer_sos for the
    recovery rover with the SOS robot as peer (none when there is no recovery rover or it is the one in trouble).
    Brain traces and traces from robots outside the fleet are ignored."""
    if not isinstance(trace, dict) or is_brain_trace(trace):
        return None
    robot, kind = trace.get('robot'), trace.get('kind')
    if not isinstance(robot, str) or robot not in robots or not isinstance(kind, str):
        return None
    text = _short(trace.get('input'), INPUT_MAX)
    fsm = trace.get('state') if isinstance(trace.get('state'), str) and trace.get('state') else None
    if kind in HAZARD_KINDS:
        return EventSpec('novel_hazard', robot, kind, text, hazard=text or None, state=fsm)
    if kind in LOW_CONFIDENCE_KINDS:
        return EventSpec('low_confidence', robot, kind, text, state=fsm)
    if kind == 'critical_damage':
        return EventSpec('critical_damage', robot, kind, text, damage=text or None, state=fsm)
    if kind == 'sos' and trace.get('status', 'active') == 'active':
        if not recovery_rover or recovery_rover == robot or recovery_rover not in robots:
            return None
        return EventSpec('peer_sos', recovery_rover, kind, text, damage=text or None, peer=robot)
    return None


def parse_operator_command(data, robots) -> dict | None:
    """/operator/command JSON → {id, text, robot, t}, or None when malformed. robot defaults to rover_0 (else the
    first rover, else the first robot); a robot outside the fleet invalidates the command."""
    d = loads_dict(data)
    if d is None:
        return None
    cid, text, robot = d.get('id'), d.get('text'), d.get('robot')
    if not isinstance(cid, str) or not cid.strip() or len(cid) > 64:
        return None
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= TEXT_MAX:
        return None
    robots = list(robots)
    if robot is None:
        rovers = [r for r in robots if r.startswith('rover')]
        robot = 'rover_0' if 'rover_0' in robots else (rovers or robots or [None])[0]
    if not isinstance(robot, str) or robot not in robots:
        return None
    return {'id': cid.strip(), 'text': text.strip(), 'robot': robot, 't': _num(d.get('t'))}


def hazard_name(ev: dict, op: str | None = None) -> str | None:
    """Short hazard name from a /world/events or /fleet/hazard_marks message (for the context's known_hazards)."""
    if not isinstance(ev, dict):
        return None
    if op == 'hazard_mark':
        reason = ev.get('reason')
        return f'{_short(reason, 20)} hazard zone' if isinstance(reason, str) and reason else 'hazard zone'
    if ev.get('op') == 'spawn_rockslide' and ev.get('ok'):
        return 'rockslide'
    return None


class KnownHazards:
    """The most recent distinct hazard names (newest last), bounded."""

    def __init__(self, limit: int = 6):
        self.limit = limit
        self._names: list[str] = []

    def add(self, name: str | None):
        if not name:
            return
        if name in self._names:
            self._names.remove(name)
        self._names.append(name)
        del self._names[:-self.limit]

    def names(self) -> list:
        return list(self._names)


# ---------------------------------------------------------------------------------------------------- context
def effective_limit(state: dict | None) -> float:
    """The rover's effective local speed limit right now: agent_state v_limit, but 0 while the FSM holds the robot
    (or the robot is critical, or nothing is known about it: the brain then reasons about a stopped robot)."""
    if not isinstance(state, dict):
        return 0.0
    if state.get('state') in HOLD_STATES or _dict(state.get('health')).get('critical'):
        return 0.0
    v = _num(state.get('v_limit'))
    return max(0.0, v) if v is not None else 0.0


def recovery_available(recovery_rover: str | None, states: dict) -> bool:
    """The recovery rover exists and is not itself waiting for rescue / critically damaged."""
    if not recovery_rover:
        return False
    st = states.get(recovery_rover)
    if not isinstance(st, dict):
        return True                # in the fleet, no state yet: its own FSM decides whether it can go
    return st.get('state') != 'AWAIT_RESCUE' and not _dict(st.get('health')).get('critical')


def _pct_frac(v) -> float | None:
    f = _num(v)
    return None if f is None else max(0.0, min(1.0, f / 100.0))


def build_context(robot: str, *, states: dict, edge: dict | None = None, perception: dict | None = None,
                  env: dict | None = None, recovery_rover: str | None = None, drone: str | None = None,
                  v_max_mps: float | None = None, hazard: str | None = None, damage: str | None = None,
                  peer: str | None = None, known_hazards=(), operator_text: str | None = None,
                  edge_profile: str | None = None) -> dict:
    """Context dict for gemma_agent_brain from the robot's latest messages (`states` = every robot's latest
    agent_state, None/missing when unknown or stale; `perception` = its latest /perception/detections).
    `v_max_mps` is rounded like agent_state v_limit (3 decimals), so full speed at the full local limit is equal to
    the limit, not a hair above it."""
    st = states.get(robot) if isinstance(states.get(robot), dict) else {}
    health = st.get('health') if isinstance(st.get('health'), dict) else {}
    ctx = {'robot': robot, 'robot_type': st.get('type') or ('rover' if robot.startswith('rover') else 'scout'),
           'state': st.get('state'), 'speed_mps': _num(st.get('speed')),
           'speed_limit_mps': effective_limit(st or None), 'v_max_mps': _r3(v_max_mps),
           'health': _pct_frac(health.get('pct')), 'battery': _pct_frac(health.get('battery_pct'))}
    unc = (perception or {}).get('uncertainty') if isinstance((perception or {}).get('uncertainty'), dict) else {}
    ctx['camera_conf'] = _num(unc.get('camera_conf'))
    ctx['lidar_conf'] = _num(unc.get('lidar_conf'))
    ctx['uncertainty'] = _num(unc.get('score'))
    if ctx['uncertainty'] is None:
        ctx['uncertainty'] = _num((edge or {}).get('uncertainty'))
    cor = (perception or {}).get('corridor') if isinstance((perception or {}).get('corridor'), dict) else {}
    ctx['nearest_obstacle_m'] = _num(cor.get('obstacle_dist_m'))
    faults = health.get('faults')
    if damage is None and isinstance(faults, (list, tuple)) and faults:
        damage = ', '.join(str(f) for f in faults[:4])
    ctx.update(hazard=hazard, damage=damage, peer=peer, known_hazards=list(known_hazards) or None,
               dust=bool((_num((env or {}).get('dust_level')) or 0.0) >= DUST_LEVEL),
               operator_text=operator_text, edge_profile=edge_profile)
    dst = states.get(drone) if drone and isinstance(states.get(drone), dict) else {}
    fleet = {'recovery_rover': recovery_rover or None, 'recovery_available': recovery_available(recovery_rover, states),
             'drone': drone or None, 'drone_agl_m': _num(dst.get('altitude_agl'))}
    ctx['fleet'] = {k: v for k, v in fleet.items() if v is not None}
    return {k: v for k, v in ctx.items() if v is not None}


class LatestJson:
    """Latest raw message per name with its receipt time; parsed lazily, once (perception and agent_state arrive at
    5-10 Hz per robot but are only read when an event is raised)."""

    def __init__(self):
        self._raw: dict = {}

    def put(self, name: str, data: str, now: float):
        self._raw[name] = [data, now, None, False]

    def get(self, name: str, now: float, max_age_s: float) -> dict | None:
        e = self._raw.get(name)
        if e is None or now - e[1] > max_age_s:
            return None
        if not e[3]:
            e[2], e[3] = loads_dict(e[0]), True
        return e[2]


def trigger_priority(spec: EventSpec) -> int:
    return TRIGGER_PRIORITY.get(spec.trigger, 0)


class EventGate:
    """Re-offers an event that AgentBrain de-duplicated away once its interval has passed, so a burst (a
    slope_hazard a few seconds before a rockslide) never swallows the rockslide. Of several deferred triggers of the
    same (robot, event) the highest-priority one is kept (path_blocked / hazard_marked over slope_hazard; ties keep
    the first deferred), so a later slope warning cannot replace the rockslide. Same clock as AgentBrain."""

    def __init__(self, intervals: dict = MIN_INTERVAL_S, margin_s: float = 0.05):
        self.intervals = dict(intervals)
        self.margin_s = margin_s
        self._last: dict = {}            # key → time of the last accepted submit
        self._pending: dict = {}         # key → the EventSpec waiting for its slot

    def submitted(self, spec: EventSpec, now: float):
        self._last[spec.key] = now
        self._pending.pop(spec.key, None)

    def defer(self, spec: EventSpec, now: float):
        cur = self._pending.get(spec.key)
        if cur is None or trigger_priority(spec) > trigger_priority(cur):
            self._pending[spec.key] = spec
        self._last.setdefault(spec.key, now)

    def due(self, now: float) -> list:
        out = []
        for key, spec in list(self._pending.items()):
            if now - self._last.get(key, -math.inf) >= self.intervals.get(spec.event, 3.0) + self.margin_s:
                out.append(self._pending.pop(key))
        return out

    def pending(self) -> int:
        return len(self._pending)


class DropGuard:
    """Runs a message handler so one bad message can never take a node down: an exception drops that message and
    logs at most one warning per `every_s` (the others are counted in `dropped` and summarised in the next one)."""

    def __init__(self, warn: Callable[[str], None] | None, every_s: float = WARN_EVERY_S,
                 clock: Callable[[], float] = time.monotonic):
        self.warn, self.every_s, self.clock = warn, every_s, clock
        self.dropped = 0
        self._last = -math.inf
        self._quiet = 0

    def call(self, what: str, fn: Callable, *args, **kw):
        """fn(*args, **kw), or None (message dropped, throttled warning) if it raised."""
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001 - a malformed message must never kill the node
            self.drop(what, e)
            return None

    def wrap(self, what: str, fn: Callable) -> Callable:
        return functools.partial(self.call, what, fn)

    def drop(self, what: str, why):
        """Count one dropped message; warn unless a warning went out less than `every_s` ago."""
        self.dropped += 1
        now = self.clock()
        if now - self._last < self.every_s:
            self._quiet += 1
            return
        self._last = now
        more = f' (+{self._quiet} more dropped since the last warning)' if self._quiet else ''
        self._quiet = 0
        reason = f'{type(why).__name__}: {_short(why, 120)}' if isinstance(why, BaseException) else _short(why, 120)
        if self.warn is not None:
            try:
                self.warn(f'dropped a {what} message ({reason}){more}')
            except Exception:  # noqa: BLE001 - logging must not raise either
                pass


def _guarded(what: str, default=None):
    """BrainHub entry point: an exception drops this one message via `self.guard` and returns `default`."""
    def deco(fn):
        @functools.wraps(fn)
        def entry(self, *args, **kw):
            try:
                return fn(self, *args, **kw)
            except Exception as e:  # noqa: BLE001 - see DropGuard
                self.guard.drop(what, e)
                return default
        return entry
    return deco


# ---------------------------------------------------------------------------------------------------- outputs
def decision_message(d: dict, *, trigger: str, trigger_input: str = '', command_id: str | None = None,
                     command_text: str | None = None, local: dict | None = None, immediate: bool = False,
                     command_seq: int | None = None, t: float = 0.0) -> dict:
    """/brain/decision payload: Decision.as_dict() verbatim plus where it came from (protocol: /brain/decision).
    `command_seq` orders operator commands as the brain node received them, so a late answer to an older command
    never replaces the cap of a newer one."""
    msg = dict(d)
    msg.update(t=round(float(t), 3), trigger=trigger, trigger_input=_short(trigger_input, INPUT_MAX),
               command_id=command_id, command_text=command_text, local=local, immediate=bool(immediate),
               command_seq=command_seq)
    return finite_json(msg)


def _who(source: str) -> str:
    return 'Gemma, validated' if source == SOURCE_GEMMA else 'local rules'


def action_text(msg: dict, peer: str | None = None) -> str:
    """Human text of a decision's action including its cap, for the brain_advice trace (`peer` = the robot in
    trouble for a peer_sos, which the decision itself does not name)."""
    a, sp, alt = msg.get('action'), _num(msg.get('target_speed_mps')), _num(msg.get('target_altitude_m'))
    tgt = msg.get('target_robot')
    subject = msg.get('robot')
    cap = '' if sp is None else ('; hold at 0 m/s' if sp <= 0.0 else f'; cap {sp:.2f} m/s')
    if a == SLOW:
        txt = 'Slow and re-plan' + cap
    elif a == CLARIFY:
        txt = 'Request operator clarification' + cap
    elif a == DEPLOY:                    # no motion change on the planners (the local rescue already runs)
        peer = peer or (subject if subject != tgt else None)
        txt = f'Deploy recovery rover {unit_label(tgt) if tgt else "?"}' + (
            f' to {unit_label(peer)}' if peer else '')
    elif a == ALTITUDE:
        txt = f'Set {unit_label(tgt) if tgt else "drone"} cruise altitude to ' + (
            f'{alt:.0f} m AGL' if alt is not None else 'a safer AGL') + cap
    elif a == REJECT:                    # only a stop riding on a rejection changes motion
        txt = 'Reject unsafe operator command; keep the local plan' + (cap if sp is not None and sp <= 0.0 else '')
    else:
        txt = str(a)
    return txt + TRACE_SUFFIX


BRAIN_TRACE_KEYS = ('source', 'model', 'event', 'action', 'target_speed_mps', 'target_altitude_m', 'latency_ms',
                    'tokens_in', 'tokens_out', 'projected_edge_ms', 'projected_device', 'projected_fits',
                    'projected_note', 'error', 'adjusted', 'command_id')


def advice_trace(msg: dict, state: str | None, builder: TraceBuilder, peer: str | None = None) -> dict:
    """One "brain_advice" decision trace for a /brain/decision message (TraceBuilder fields + "brain")."""
    gemma = msg.get('source') == SOURCE_GEMMA
    event = msg.get('event', '?')
    if msg.get('command_text') is not None:
        inp = f"{event}: '{_short(msg['command_text'], 80)}'" + (' (immediate local stop)' if msg.get('immediate')
                                                                 else '')
    else:
        inp = f"{event} ← {msg.get('trigger', '?')}" + (f": {msg['trigger_input']}" if msg.get('trigger_input') else '')
    sev = 'warn' if msg.get('action') in (REJECT, CLARIFY) else 'info'
    t = builder.make('brain_advice', state or 'UNKNOWN', [], _short(inp, 220), msg.get('reasoning_trace') or '',
                     0.8 if gemma else 0.9, action_text(msg, peer), sev, latency_ms=_num(msg.get('latency_ms')),
                     force=True)
    t['sensors'] = ['GEMMA'] if gemma else ['LOCAL']     # TraceBuilder only keeps on-board sensor names
    t['brain'] = {k: msg.get(k) for k in BRAIN_TRACE_KEYS}
    return t


def trace_robot(msg: dict) -> str:
    return msg.get('robot') or msg.get('target_robot') or 'fleet'


Meta = collections.namedtuple('Meta', 'trigger trigger_input command_id command_text local seq peer')
_UNKNOWN = Meta('unknown', '', None, None, None, None, None)
META_MAX = 512                 # request id → origin bound (entries leave when their decision arrives)


class BrainHub:
    """fleet_brain_node without rclpy. Feed it the raw message strings; it calls `publish_decision(msg)` and
    `publish_trace(trace)` with dicts (serialise with `dumps`). Call `poll()` from a timer on the same thread as the
    inputs: AgentBrain answers on its worker threads, and everything here runs on the caller's thread.

    * events: trace → `event_for_trace` → AgentBrain.submit; one de-duplicated away is re-offered by `EventGate`.
    * operator commands: a stop is answered at once by the local rules ("immediate": true, cap 0) and still
      submitted; its answer replaces the immediate one. `command_seq` numbers commands in arrival order. A command
      to a drone that is not an altitude command is answered at once with REQUEST_CLARIFICATION (no cap, no cloud
      call): drones take altitude commands only.
    * every decision → /brain/decision + one "brain_advice" trace (robot = decision robot, its latest FSM state).
    * every entry point drops a message it cannot handle (one throttled `warn` per WARN_EVERY_S), never raises.
    `clock` must be AgentBrain's clock (monotonic, wall); `wall` stamps the human-facing 't' fields."""

    def __init__(self, brain: AgentBrain, *, robots, publish_decision: Callable[[dict], None],
                 publish_trace: Callable[[dict], None], recovery_rover: str | None = None, drone: str | None = None,
                 edge_profile: str = 'orin_nano_15w', model: str = DEFAULT_MODEL, rover_v_max: float | None = None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 log: Callable[[str], None] | None = None, warn: Callable[[str], None] | None = None):
        self.brain = brain
        self.robots = list(robots)
        self.recovery = recovery_rover or None
        self.drone = drone or None
        self.edge, self.model, self.rover_v_max = edge_profile, model, rover_v_max
        self.publish_decision, self.publish_trace = publish_decision, publish_trace
        self.clock, self.wall, self.log = clock, wall, log
        self.guard = DropGuard(warn if warn is not None else log, clock=clock)
        self.gate = EventGate(MIN_INTERVAL_S)
        self.states, self.edges, self.percep = LatestJson(), LatestJson(), LatestJson()
        self.env: dict = {}
        self.hazards = KnownHazards()
        self.last_error: str | None = None
        self._meta: collections.OrderedDict = collections.OrderedDict()
        self._seq = 0
        self._builders: dict = {}

    # ------------------------------------------------------------------ inputs
    @_guarded('agent_state')
    def on_state(self, robot: str, data: str):
        self.states.put(robot, data, self.clock())

    @_guarded('edge_telemetry')
    def on_edge(self, robot: str, data: str):
        self.edges.put(robot, data, self.clock())

    @_guarded('perception')
    def on_perception(self, robot: str, data: str):
        self.percep.put(robot, data, self.clock())

    @_guarded('/world/env_state')
    def on_env(self, data: str):
        d = loads_dict(data)
        if d is not None:
            self.env = d

    @_guarded('hazard')
    def on_hazard(self, data: str, op: str | None = None):
        d = loads_dict(data)
        if d is not None:
            self.hazards.add(hazard_name(d, op))

    @_guarded('/agent_decision_trace')
    def on_trace(self, data: str) -> EventSpec | None:
        spec = event_for_trace(loads_dict(data), self.robots, self.recovery)
        if spec is not None:
            self.submit_event(spec)
        return spec

    @_guarded('/operator/command')
    def on_command(self, data: str) -> dict | None:
        """Handle one /operator/command; returns the parsed command, or None if it was malformed or could not be
        handled (dropped with one throttled warning)."""
        cmd = parse_operator_command(data, self.robots)
        if cmd is None:
            self.guard.drop('/operator/command', 'malformed command ignored')
            return None
        self._seq += 1
        robot, text = cmd['robot'], cmd['text']
        said = text
        if self.is_drone(robot) and robot.lower() not in text.lower():
            # typed to the selected drone: "climb to 10 m" names no drone, but the local reading needs it to
            said = f'{robot} {text}'
        ctx = self.context(EventSpec('operator_command', robot, 'operator', said), operator_text=said)
        local = assess_operator_command(said, ctx)
        meta = Meta('operator', text, cmd['id'], text, local, self._seq, None)
        if self.is_drone(robot, ctx) and local['intent'] != 'drone_altitude':
            # a speed / stop / unclear command to a drone: answered here, no stop decision and no cloud call
            self._local_answer(ctx, robot, meta, f"local:{cmd['id']}", drone_only=True)
            return cmd
        if local['intent'] == 'stop':
            # the local layer honours a stop at once; Gemma's (or the fallback's) answer replaces it when it arrives
            self._local_answer(ctx, robot, meta, f"immediate:{cmd['id']}", immediate=True)
        rid = self.brain.submit('operator_command', ctx, robot=robot)
        if rid is None:
            # the same text for the same robot was answered < 0.5 s ago: answer this id with the local rules
            self._local_answer(ctx, robot, meta, f"duplicate:{cmd['id']}",
                               error='duplicate of the same command < 0.5 s earlier; local rules answered')
        else:
            self._remember(rid, meta)
        return cmd

    # ------------------------------------------------------------------ brain
    def is_drone(self, robot: str, ctx: dict | None = None) -> bool:
        return robot == self.drone or str((ctx or {}).get('robot_type') or '').lower() in DRONE_TYPES

    def context(self, spec: EventSpec, operator_text: str | None = None) -> dict:
        now = self.clock()
        states = {r: self.states.get(r, now, STATE_FRESH_S) for r in self.robots}
        if spec.state:
            # the trace is newer than the robot's last agent_state: reason about the state it was raised in
            st = dict(states.get(spec.robot) or {}, state=spec.state)
            if spec.state in HOLD_STATES:
                st['speed'] = 0.0
            states[spec.robot] = st
        st = states.get(spec.robot) or {}
        v_max = self.rover_v_max if spec.robot.startswith('rover') else st.get('v_limit')
        return build_context(spec.robot, states=states, edge=self.edges.get(spec.robot, now, STATE_FRESH_S),
                             perception=self.percep.get(spec.robot, now, PERCEPTION_FRESH_S), env=self.env,
                             recovery_rover=self.recovery, drone=self.drone, v_max_mps=v_max, hazard=spec.hazard,
                             damage=spec.damage, peer=spec.peer, known_hazards=self.hazards.names(),
                             operator_text=operator_text, edge_profile=self.edge)

    def submit_event(self, spec: EventSpec):
        now = self.clock()
        rid = self.brain.submit(spec.event, self.context(spec), robot=spec.robot)
        if rid is None or rid in self._meta:
            # de-duplicated (answered or in flight): keep the newest trigger and ask again when its slot opens
            self.gate.defer(spec, now)
            return
        self.gate.submitted(spec, now)
        self._remember(rid, Meta(spec.trigger, spec.trigger_input, None, None, None, None, spec.peer))

    @_guarded('brain poll', default=0)
    def poll(self) -> int:
        """Publish every finished decision and re-offer deferred events that are due; returns decisions published.
        One decision (or deferred event) that cannot be handled is dropped on its own, not with the whole batch."""
        n = 0
        for d in self.brain.poll():
            n += self.guard.call('/brain/decision', self._publish_finished, d) or 0
        for spec in self.gate.due(self.clock()):
            self.guard.call('deferred event', self.submit_event, spec)
        return n

    @_guarded('/brain/status')
    def status(self, enabled: bool, key_present: bool) -> dict | None:
        """/brain/status payload."""
        return {'enabled': bool(enabled), 'key_present': bool(key_present), 'model': self.model, 'device': self.edge,
                'stats': self.brain.stats(), 'backoff_s': round(self.brain.backoff_remaining_s(), 2),
                'last_error': self.last_error, 't': round(self.wall(), 3)}

    # ------------------------------------------------------------------ internals
    def _remember(self, rid: str, meta: Meta):
        self._meta.setdefault(rid, []).append(meta)
        while len(self._meta) > META_MAX:
            self._meta.popitem(last=False)

    def _publish_finished(self, d) -> int:
        dd = d.as_dict()
        metas = self._meta.pop(d.request_id, None) or [_UNKNOWN]
        for meta in metas:
            self._publish(dd, meta)
        return len(metas)

    def _local_answer(self, ctx: dict, robot: str, meta: Meta, request_id: str, *, immediate: bool = False,
                      error: str | None = None, drone_only: bool = False):
        d = deterministic_fallback('operator_command', ctx, device_id=self.edge, cloud_model=self.model, error=error)
        if drone_only:
            d = dataclasses.replace(d, action=CLARIFY, target_speed_mps=None, target_altitude_m=None,
                                    target_robot=None, reasoning_trace=DRONE_ONLY_TEXT, adjusted=[])
        d.robot, d.request_id, d.age_ms = robot, request_id, 0.0
        # no cloud call was made for a drone-only answer: it says nothing about the last Gemma answer
        self._publish(d.as_dict(), meta, immediate=immediate, track_error=not drone_only)

    def _publish(self, dd: dict, meta: Meta, immediate: bool = False, track_error: bool = True):
        msg = decision_message(dd, trigger=meta.trigger, trigger_input=meta.trigger_input, command_id=meta.command_id,
                               command_text=meta.command_text, local=meta.local, immediate=immediate,
                               command_seq=meta.seq, t=self.wall())
        self.publish_decision(msg)
        robot = trace_robot(msg)
        if robot in self.robots:
            state = (self.states.get(robot, self.clock(), STATE_FRESH_S) or {}).get('state')
        else:
            state = 'FLEET'
        tb = self._builders.setdefault(robot, TraceBuilder(robot))
        self.publish_trace(advice_trace(msg, state, tb, meta.peer))
        if not immediate and track_error:
            self.last_error = msg.get('error')
        if self.log is not None:
            lat = msg.get('latency_ms')
            self.log(f"{robot} {msg.get('event')} → {msg.get('action')} cap={msg.get('target_speed_mps')} "
                     f"[{msg.get('source')} / {msg.get('model')}"
                     + (f', measured cloud round-trip {lat:.0f} ms' if lat is not None else '')
                     + (', immediate' if immediate else '') + ']'
                     + (f" ({msg['error']})" if msg.get('error') else ''))


# ---------------------------------------------------------------------------------------------------- planner side
def parse_decision(data) -> dict | None:
    """A /brain/decision message, or None if it is not one (bad JSON, unknown action, wrong types)."""
    d = loads_dict(data)
    if d is None or d.get('action') not in ACTIONS:
        return None
    for k in ('robot', 'target_robot', 'event', 'source'):
        if d.get(k) is not None and not isinstance(d.get(k), str):
            return None
    return d


@dataclass
class Cap:
    speed_mps: float
    action: str
    source: str
    operator: bool                # lives in the operator slot
    event: str
    until: float                  # simulated seconds
    duration_s: float
    model: str = ''
    seq: int | None = None
    command_id: str | None = None
    command_text: str | None = None
    reasoning: str = ''
    extra: dict = field(default_factory=dict)   # immediate: the instant local stop; stop: a real operator stop
    released: bool = False        # not a cap: reports that this decision lifted the operator cap (never stored)

    def remaining_s(self, now: float) -> float:
        return max(0.0, self.until - now)


class BrainCap:
    """A rover's advisory speed caps, on simulated time.

    Two slots, operator and event: a new decision replaces the previous cap of its slot and the binding cap is the
    lower of the live ones, so an event decision can never lift an operator stop (nor an operator command lift an
    event hold). Expiry: operator 60 s; event 15 s; hold (cap 0, or any cap below HOLD_FLOOR_MPS, rounded to 0) 4 s.

    * Events: DEPLOY_RECOVERY_ROVER / REJECT_UNSAFE_COMMAND change no motion, critical_damage / peer_sos advice never
      caps a rover (the local rescue runs) and stale event decisions are dropped (their context is out of date).
      Event caps are ignored while the rover is in RESCUE_DOCKING / REPAIRING.
    * Operator commands: only a real stop (local intent "stop", the immediate stop decision, or a requested speed
      below HOLD_FLOOR_MPS) holds in the operator slot (60 s). A 'faster' / 'set_speed' command naming no hazard
      first lifts the operator slot (the local limits rule again; nothing is ever raised above them): 'faster' sets
      nothing more, 'set_speed' then caps at the validated speed. A SLOW_AND_REPLAN answer with a positive cap sets
      the operator cap. Every other answer (REJECT / CLARIFY / DEPLOY, or a hold answering a non-stop command)
      changes no motion and never touches the event slot, so it cannot shorten a live hazard cap.
      Ordering: an answer with a lower command_seq than the highest one ever applied or released is dropped, and a
      stale answer is applied only if it is a stop."""

    def __init__(self, robot: str):
        self.robot = robot
        self.operator: Cap | None = None
        self.event: Cap | None = None
        self.seq: int | None = None         # highest command_seq applied or released

    def offer(self, d: dict, now: float) -> Cap | None:
        """Apply a parsed /brain/decision; returns the new Cap if it set one (or a `released` Cap when it lifted
        the live operator cap), else None."""
        if not isinstance(d, dict) or d.get('robot') != self.robot:
            return None
        action, event = d.get('action'), d.get('event')
        if action not in ACTIONS:
            return None
        raw = d.get('target_speed_mps')
        sp = _num(raw)
        if raw is not None and (sp is None or sp < 0.0):
            return None                                      # malformed cap: ignore the decision
        if sp is not None and sp < HOLD_FLOOR_MPS:
            sp = 0.0                                         # a crawl this slow is a hold
        stale = d.get('stale') is True
        if event != 'operator_command':
            if stale or sp is None or action in (DEPLOY, REJECT) or event in SOS_EVENTS:
                return None
            return self._set(d, sp, now, operator=False)
        seq = _int(d.get('command_seq'))
        if seq is not None and self.seq is not None and seq < self.seq:
            return None                                      # answers an older command than one already acted on
        local = _dict(d.get('local'))
        intent = local.get('intent')
        req = _num(local.get('requested_speed_mps'))
        stop = (intent == 'stop' or d.get('immediate') is True
                or (intent == 'set_speed' and req is not None and req < HOLD_FLOOR_MPS))   # "set speed 0.03" = stop
        if sp == 0.0 and stop:
            return self._set(d, 0.0, now, operator=True, seq=seq, stop=True)     # operator stop (even if stale)
        if stale:
            return None
        released = None
        if intent in RELEASE_INTENTS and not local.get('hazard'):
            released = self._release(d, now, seq)            # the operator lifts their own stop / cap
            if intent == 'faster':
                return released                              # "full speed": no cap at all, the local limits rule
        if sp is not None and sp > 0.0 and action not in (REJECT, CLARIFY, DEPLOY):
            return self._set(d, sp, now, operator=True, seq=seq)                 # replaces the operator slot
        return released                                      # any other answer changes no motion

    def _note_seq(self, seq: int | None):
        if seq is not None:
            self.seq = seq if self.seq is None else max(self.seq, seq)

    def _cap(self, d: dict, sp: float, now: float, dur: float, *, operator: bool, seq: int | None, **kw) -> Cap:
        return Cap(speed_mps=round(sp, 3), action=d.get('action'), source=str(d.get('source') or ''),
                   operator=operator, event=str(d.get('event') or ''), until=now + dur, duration_s=dur,
                   model=str(d.get('model') or ''), seq=seq,
                   command_id=d.get('command_id') if isinstance(d.get('command_id'), str) else None,
                   command_text=d.get('command_text') if isinstance(d.get('command_text'), str) else None,
                   reasoning=str(d.get('reasoning_trace') or ''), **kw)

    def _set(self, d: dict, sp: float, now: float, *, operator: bool, seq: int | None = None,
             stop: bool = False) -> Cap:
        dur = OPERATOR_CAP_S if operator else (EVENT_HOLD_S if sp == 0.0 else EVENT_CAP_S)
        cap = self._cap(d, sp, now, dur, operator=operator, seq=seq,
                        extra={'immediate': bool(d.get('immediate')), 'stop': stop})
        if operator:
            self.operator = cap
        else:
            self.event = cap
        self._note_seq(seq)
        return cap

    def _release(self, d: dict, now: float, seq: int | None) -> Cap | None:
        self._note_seq(seq)
        cur = self.operator if self.operator is not None and now < self.operator.until else None
        self.operator = None
        if cur is None:
            return None
        return self._cap(d, cur.speed_mps, now, 0.0, operator=True, seq=seq, released=True,
                         extra={'immediate': False, 'stop': bool(cur.extra.get('stop'))})

    def active(self, now: float, state: str | None = None) -> Cap | None:
        """The binding live cap; in RESCUE_DOCKING / REPAIRING only the operator slot counts."""
        slots = (self.operator,) if state in RESCUE_STATES else (self.operator, self.event)
        live = [c for c in slots if c is not None and now < c.until]
        return min(live, key=lambda c: (c.speed_mps, not c.operator)) if live else None

    def limit(self, now: float, state: str | None = None) -> float | None:
        c = self.active(now, state)
        return None if c is None else c.speed_mps

    def holding(self, now: float, state: str | None = None) -> bool:
        """True while the binding cap is a hold (0 m/s)."""
        return self.limit(now, state) == 0.0

    def apply(self, v: float, w: float, now: float, state: str | None = None) -> tuple:
        """(v, w) of a 'follow' / 'dock' drive under the cap: v = min(v, cap) with w scaled to keep the curvature;
        a hold (cap 0) also stops turning on the spot."""
        cap = self.limit(now, state)
        if cap is None:
            return v, w
        if cap <= 0.0:
            return 0.0, 0.0
        if abs(v) <= cap:
            return v, w
        s = cap / abs(v)
        return math.copysign(cap, v), w * s

    def info(self, now: float, state: str | None = None, applied: bool = False) -> dict | None:
        """agent_state "brain_cap"; `applied` = the current drive ('follow' / 'dock') runs under the cap."""
        c = self.active(now, state)
        if c is None:
            return None
        return {'speed_mps': c.speed_mps, 'action': c.action, 'source': c.source, 'operator': c.operator,
                'remaining_s': round(c.remaining_s(now), 1), 'applied': bool(applied)}


def cap_trace(cap: Cap) -> dict:
    """Texts for the planner's "brain_cap" trace when a new cap is applied (or the operator cap is released)."""
    who = _who(cap.source)
    brain = 'Gemma' if cap.source == SOURCE_GEMMA else 'the local rules'
    from_operator = cap.event == 'operator_command'
    what = 'an operator command' if from_operator else f"a {cap.event.replace('_', ' ')} event"
    if cap.released:
        lifted = 'Operator stop' if cap.extra.get('stop') else 'Operator speed cap'
        action = f'{lifted} lifted by a newer operator command ({who}); the local limits apply'
    elif cap.speed_mps <= 0.0:
        if cap.extra.get('stop'):
            why = f'operator stop ({who})'
            if cap.action == REJECT:
                why += '; the rest of the command was rejected'
            if cap.extra.get('immediate'):
                why += ', immediate'
        else:
            why = {CLARIFY: f'{brain} requested clarification', SLOW: f'{brain} requested a stop and re-plan',
                   ALTITUDE: f'{brain} requested a hold'}.get(cap.action, f'{brain} hold')
        action = f'Holding {cap.duration_s:.0f} s: {why}'
    else:
        action = f'Speed capped to {cap.speed_mps:.2f} m/s by {what} ({who}) for {cap.duration_s:.0f} s'
    inp = f'/brain/decision {cap.action} for {cap.event}' + (
        f": '{_short(cap.command_text, 80)}'" if cap.command_text else '')
    hold = cap.speed_mps <= 0.0 and not cap.released
    return {'input': inp, 'reasoning': _short(cap.reasoning, 300) or 'advisory brain decision',
            'confidence': 0.8 if cap.source == SOURCE_GEMMA else 0.9, 'action': action,
            'severity': 'warn' if hold else 'info'}


class BrainAltitude:
    """A drone's advisory cruise AGL (ADJUST_DRONE_ALTITUDE addressed to it), clamped to [6, 20] m, for 60 s."""

    def __init__(self, robot: str):
        self.robot = robot
        self.agl_m: float | None = None
        self.until = -math.inf
        self.last: dict | None = None

    def offer(self, d: dict, now: float) -> float | None:
        if not isinstance(d, dict) or d.get('action') != ALTITUDE:
            return None
        tgt = d.get('target_robot')
        if tgt != self.robot and not (tgt is None and d.get('robot') == self.robot):
            return None
        alt = _num(d.get('target_altitude_m'))
        if alt is None or (d.get('event') != 'operator_command' and d.get('stale') is True):
            return None
        self.agl_m = round(max(DRONE_ALT_MIN_M, min(DRONE_ALT_MAX_M, alt)), 1)
        self.until = now + ALTITUDE_S
        self.last = d
        return self.agl_m

    def target(self, now: float) -> float | None:
        return self.agl_m if self.agl_m is not None and now < self.until else None

    def info(self, now: float, flying_agl_m: float | None = None) -> dict | None:
        """agent_state "brain_agl"; `flying_agl_m` = the altitude-hold set-point in a state that flies the advisory
        AGL (None otherwise): "applied" is true only while that set-point is the advisory AGL."""
        a = self.target(now)
        if a is None:
            return None
        applied = flying_agl_m is not None and abs(flying_agl_m - a) < 0.05
        return {'agl_m': a, 'remaining_s': round(max(0.0, self.until - now), 1), 'applied': applied}


def altitude_trace(agl: float, d: dict, applies_now: bool) -> dict:
    """Texts for the scout's "brain_cap" trace when it accepts an advisory cruise AGL."""
    who = _who(d.get('source', ''))
    event = str(d.get('event') or '?')
    what = 'an operator command' if event == 'operator_command' else f"a {event.replace('_', ' ')} event"
    action = f'Cruise altitude {agl:.0f} m AGL for {ALTITUDE_S:.0f} s by {what} ({who})'
    if not applies_now:
        action += '; applies once surveying (take-off / scan altitude unchanged)'
    quote = f": '{_short(d.get('command_text'), 80)}'" if d.get('command_text') else ''
    return {'input': f'/brain/decision {ALTITUDE} for {event}{quote}',
            'reasoning': _short(d.get('reasoning_trace'), 300) or 'advisory brain decision',
            'confidence': 0.8 if d.get('source') == SOURCE_GEMMA else 0.9, 'action': action, 'severity': 'info'}
