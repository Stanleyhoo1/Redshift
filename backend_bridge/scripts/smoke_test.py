#!/usr/bin/env python3
"""End-to-end smoke test of a running backend (the same REST + WebSocket traffic the web UI produces).

    python3 backend_bridge/scripts/smoke_test.py [--url http://127.0.0.1:8000] [--backend mock|gazebo] [--speed 1.0]
                                                 [--brain on|off] [--skip-phase3]

Deploys a mission, checks telemetry / camera / LiDAR / map streams and exploration progress, then runs the three
unscripted changes (rockslide, dust anomaly, tip/slip jolt) and checks the agents' traced responses, and finally
aborts and checks no simulator process is left. Exit code 0 = every check passed.

Phase 3 (docs/protocol.md §10; --skip-phase3 leaves it out) follows the operator commands:
- a basalt pillar topples across rover_0's path: the world event, then within 25 s a path_blocked that names the
  pillar, or a path_blocked plus a PILLAR detection on the pillar's footprint (either order: LiDAR usually blocks
  the route while the trunk is still falling); a re-plan after that block, and a novel_hazard brain decision;
- a 45 s dust storm: CAUTION within 6 s with the camera confidence below 0.5, rover_0's measured solar power down
  ≥ 80 % against its median just before, power_save; with a drone, wind_descent within 25 s and its measured AGL
  ≤ 9.5 m within 35 s (injected only once the drone is above that). Once the dust level in mission.env falls,
  power_restored, wind_clear and caution_exit; then nothing from either change was read as critical damage.

The advisory brain (docs/protocol.md §9) is checked too: a novel_hazard decision after the rockslide, a
critical_damage / peer_sos decision after the strike, and operator commands at the end (an unsafe one rejected, a
slow-down cap applied, an immediate stop). These checks pass whether Gemma or the local fallback answered (--brain
off, no API key, rate-limited); each detail says which brain decided, the model, the measured cloud round-trip and
the tokens.

--speed runs the mission at that simulation speed and also checks the achieved real-time factor, that mission time
follows simulated time, and a live speed change. Timeouts are behavioural (simulated) seconds, so below 1× they are
stretched to wall time; above 1× they are left as they are.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import statistics
import subprocess
import sys
import time

from collections import deque
from pathlib import Path

import cv2
import httpx
import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend_bridge.mission_manager import CLEANUP_PATTERNS  # noqa: E402

# ---------------------------------------------------------------------------------------------------- brain checks
# what the rockslide raises on the rover (brain_bridge event mapping): its blocked path, or the hazard it marks
ROCKSLIDE_TRIGGERS = ('path_blocked', 'hazard_marked')
ROCKSLIDE_WORDS = ('obstacle', 'boulder', 'rockslide')
PILLAR_WORDS = ('pillar', 'obstacle', 'boulder')     # a trigger text that names the fallen trunk (or its chunks)
# FSM states whose drive is 'follow' along the planned path / 'dock' — the only drives a brain cap limits
FOLLOW_STATES = ('EXPLORING', 'CAUTION', 'RESCUE_TRANSIT')
DOCK_STATES = ('RESCUE_DOCKING',)


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def first_decision(decisions, t_start: float, pred, slack: float = 0.0):
    """First (receipt time, decision) of `decisions` received at or after t_start - slack that satisfies pred."""
    return next(((rx, d) for rx, d in decisions if rx >= t_start - slack and pred(d)), None)


def hazard_decision(d: dict, rover: str, words: tuple) -> bool:
    """A novel_hazard decision for `rover` that an injected obstacle raised: triggered by the blocked path or the
    marked hazard, or whose trigger text names the obstacle — not some other hazard (a slope) that happened
    meanwhile."""
    if not isinstance(d, dict) or d.get('robot') != rover or d.get('event') != 'novel_hazard':
        return False
    if d.get('trigger') in ROCKSLIDE_TRIGGERS:
        return True
    text = str(d.get('trigger_input') or '').lower()
    return any(w in text for w in words)


def rockslide_decision(d: dict, rover: str) -> bool:
    """A novel_hazard decision for `rover` that the rockslide raised."""
    return hazard_decision(d, rover, ROCKSLIDE_WORDS)


def pillar_decision(d: dict, rover: str) -> bool:
    """A novel_hazard decision for `rover` that the fallen pillar raised (same triggers as the rockslide's)."""
    return hazard_decision(d, rover, PILLAR_WORDS)


def drive_of(st: dict) -> tuple[bool, str]:
    """(is the rover in a follow / dock drive, why) from its telemetry state and planned path."""
    state, path = st.get('state'), st.get('path') or []
    if state in DOCK_STATES:
        return True, f'docking ({state})'
    if state in FOLLOW_STATES and len(path) > 1:
        return True, f'following a {len(path)}-point path ({state})'
    return False, f'not following / docking: state {state}, {len(path)}-point path'


def cap_verdict(st: dict, limit: float) -> tuple[bool | None, str]:
    """A rover's telemetry against a cap ≤ limit: (None, why) while no such brain_cap is shown; else (ok, why).
    The cap must be applied (brain_cap "applied": true) while the rover follows its path or docks; "applied": false
    passes only when it is in neither drive at that moment."""
    c = st.get('brain_cap')
    v = _num(c.get('speed_mps')) if isinstance(c, dict) else None
    if v is None or v > limit + 1e-6:
        return None, f'brain_cap={c}'
    driving, why = drive_of(st)
    applied = c.get('applied')
    if applied is True:
        return True, f'cap {v:.2f} m/s applied, {why}'
    if applied is False:
        return (not driving), f'cap {v:.2f} m/s not applied' + (f' although {why}' if driving else f': {why}')
    return False, f'cap {v:.2f} m/s without an "applied" flag: {c}'


def operator_cap_cleared(st: dict) -> bool:
    """No operator cap binds the rover (brain_cap null, or the binding cap is an event cap)."""
    c = st.get('brain_cap')
    return not (isinstance(c, dict) and c.get('operator') is True)


BORDER_M = 3.0             # the planner's sector-edge keep-out (agent_planner_node.BORDER_M)
INFLATION_CELLS = 2        # the planner's 0.6 m inflation at 0.5 m cells (ceil(0.6 / 0.5))
LETHAL_CLASSES = (3, 4, 6)  # HAZARD, PERMANENT, OBSTACLE (grid_map)


def map_classes(png_b64: str) -> np.ndarray:
    """Class per cell from a WS `map` layer image (G = class × 40; row 0 = the sector's south edge)."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(png_b64), np.uint8), cv2.IMREAD_COLOR)
    return np.round(img[:, :, 1] / 40.0).astype(int)


def rescue_room(cls: np.ndarray, meta: dict, target_xy, rescuer_xy) -> tuple[bool, str]:
    """Whether the rescuer could plan to a stand-off beside the target on this map, mirroring the planner: lethal
    classes and the sector-edge band, inflated like the cost map, block; a free cell 2.0–2.45 m from the target
    must lie in the rescuer's own connected region. A target enclosed by steep ground, boulders and hazard zones is
    correctly reported with rescue_noroute instead, and docking cannot be exercised there."""
    res, (ox, oy), n = meta['res'], meta['origin'][:2], cls.shape[0]
    b = int(round(BORDER_M / res))
    core = np.isin(cls, LETHAL_CLASSES).astype(np.uint8)
    k = 2 * INFLATION_CELLS + 1
    blocked = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    blocked[:b, :] = blocked[-b:, :] = blocked[:, :b] = blocked[:, -b:] = True
    ii, jj = np.mgrid[0:n, 0:n]
    d = np.hypot(ox + (jj + 0.5) * res - target_xy[0], oy + (ii + 0.5) * res - target_xy[1])
    cand = (d >= 2.0) & (d <= 2.45) & ~blocked
    rr = int(np.clip((rescuer_xy[1] - oy) / res, 1, n - 2))
    rc = int(np.clip((rescuer_xy[0] - ox) / res, 1, n - 2))
    free = ~blocked
    free[rr - 1:rr + 2, rc - 1:rc + 2] = True           # the planner escapes a ring the rescuer stands in
    _, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    reach = int((cand & (lab == lab[rr, rc])).sum())
    return reach > 0, f'{int(cand.sum())} free stand-off cells beside the target, {reach} reachable by the rescuer'


# ---------------------------------------------------------------------------------------- phase 3: pillar and storm
SERIES_LEN = 1800              # telemetry samples kept per followed reading: ≈ 2 min of the 15 Hz stream
SOLAR_DROP_MIN = 0.80          # storm: the rover's measured solar power must fall at least this much (85 % modelled)
SOLAR_BASELINE_MIN_W = 20.0    # a baseline below this is no sunlit reference to measure a drop against
SOLAR_BEFORE_S = 5.0           # wall seconds of solar_w samples just before the storm: their median is the baseline
SOLAR_AFTER_S = 1.5            # wall seconds of the latest samples whose median is compared with it
STORM_PEAK_MIN = 0.5           # mission.env dust level the storm must reach before its end can be read
STORM_FALL_FRAC = 0.9          # the storm is clearing once the level falls below this fraction of its peak
STORM_CLEAR_LEVEL = 0.02       # mission.env dust level below which sensor_effects passes the raw streams through
STORM_CAM_MAX = 0.5            # storm: the rover's camera confidence must fall below this
DRONE_AGL_MAX = 9.5            # storm: the drone's measured AGL must come down to this (wind descent 12 → 8 m)
STORM_S = 45.0                 # /api/inject/storm's default duration (sim s); the anomaly event's echo overrides it
# timeouts, simulated seconds (stretched to wall time below 1×, like every other timeout here)
PILLAR_EVENT_S = 12.0          # near its goal the world holds the pillar ≤ 8 s for the rover's next route
PILLAR_BLOCK_S = 25.0          # the injection → path_blocked naming the pillar, or a PILLAR detection
PILLAR_REPLAN_S = 20.0         # that block → replanned (as for the rockslide)
PILLAR_FOOTPRINT_M = 2.5       # a PILLAR detection's world_xy this close to the base → tip segment is the fallen trunk
PILLAR_AHEAD_M, PILLAR_VIEW_DEG = 8.0, 25.0   # inject once the route 8 m on lies in the camera's view (hfov 80°)
PILLAR_AIM_S = 60.0            # ... waiting at most this long for that (then inject anyway, and say so)
PILLAR_BRAIN_S = 35.0          # the injection → the novel_hazard decision: the block ≤ 25 s plus a cloud round-trip
STORM_CAUTION_S = 6.0          # the injection → caution_enter with the camera confidence below STORM_CAM_MAX
STORM_SOLAR_S = 10.0           # the injection → solar_w down: 1.5 s ramp, then hw_status and agent_state at 5 Hz
STORM_POWER_S = 12.0           # the injection → power_save: PowerSaver enters after 2 s below irradiance 0.35
WIND_DESCENT_S = 25.0          # the injection → the drone's wind_descent trace
DRONE_AGL_S = 35.0             # the injection → the drone's measured AGL ≤ DRONE_AGL_MAX
STORM_CLEAR_S = 30.0           # the storm's end → power_restored (5 s dwell), wind_clear (5 s), caution_exit
STORM_OVER_S = 20.0            # then at most this long for the air to clear and every rover to leave CAUTION


def telemetry_values(st: dict) -> dict:
    """The readings the Phase 3 checks follow over time, from one WS telemetry robots.<r> entry: the measured solar
    power (health, from hw_status), camera confidence, the drone's AGL and its own wind estimate (None = absent)."""
    return {'solar_w': _num((st.get('health') or {}).get('solar_w')),
            'camera_conf': _num((st.get('uncertainty') or {}).get('camera_conf')),
            'altitude_agl': _num(st.get('altitude_agl')), 'wind_mps': _num(st.get('wind_mps'))}


def values_since(series, t0: float, t1: float = math.inf) -> list:
    """Values of (wall epoch, value) samples with t0 ≤ t < t1."""
    return [v for t, v in series if t0 <= t < t1]


def median(xs) -> float | None:
    return statistics.median(xs) if xs else None


def first_at_most(series, t0: float, limit: float):
    """First (wall epoch, value) sample since t0 whose value is ≤ limit, else None."""
    return next(((t, v) for t, v in series if t >= t0 and v <= limit), None)


def remaining(t0: float, timeout: float, now: float) -> float:
    """Wall seconds left of a timeout that started at t0 (several checks share one deadline from an injection)."""
    return max(0.0, t0 + timeout - now)


def solar_drop(before: list, after: list) -> tuple[float | None, str]:
    """Fractional drop of the rover's measured solar power, median against median: the panel's attitude swings a
    single sample by ±20 % as the rover drives. (None, why) without samples or without a sunlit baseline."""
    b, a = median(before), median(after)
    if b is None:
        return None, 'no solar_w samples just before the storm'
    if b < SOLAR_BASELINE_MIN_W:
        return None, f'baseline only {b:.0f} W (< {SOLAR_BASELINE_MIN_W:.0f} W): no sunlit reference'
    if a is None:
        return None, f'{b:.0f} W before, no samples since'
    drop = 1.0 - a / b
    return drop, f'{b:.0f} W → {a:.0f} W ({-100 * drop:+.0f} %, medians of {len(before)} / {len(after)} samples)'


def storm_end(levels, t0: float) -> float | None:
    """Wall time the storm began to clear, from (wall epoch, mission.env.dust_level) samples: the first one since t0
    below STORM_FALL_FRAC of the running peak once that peak has reached STORM_PEAK_MIN; None while it lasts."""
    peak = 0.0
    for t, v in levels:
        if t < t0:
            continue
        if peak >= STORM_PEAK_MIN and v < STORM_FALL_FRAC * peak:
            return t
        peak = max(peak, v)
    return None


def first_sighting(frames, cls: str, t0: float, where=None):
    """First (wall epoch, detection) of class `cls` in (wall epoch, telemetry detections) frames since t0 (and, with
    `where`, for which where(detection) holds)."""
    for t, dets in frames:
        if t >= t0:
            hit = next((d for d in dets if isinstance(d, dict) and d.get('cls') == cls
                        and (where is None or where(d))), None)
            if hit is not None:
                return t, hit
    return None


def class_counts(frames, t0: float) -> dict:
    """How many telemetry frames since t0 carried each detection class (for the failure report)."""
    out: dict = {}
    for t, dets in frames:
        if t >= t0:
            for c in {d.get('cls') for d in dets if isinstance(d, dict) and isinstance(d.get('cls'), str)}:
                out[c] = out.get(c, 0) + 1
    return out


def names_pillar(tr: dict | None) -> bool:
    return 'pillar' in str((tr or {}).get('input') or '').lower()


def pillar_block(traces, rover: str, t0: float, slack: float = 0.5) -> dict | None:
    """The rover's path_blocked trace since t0 that names the pillar, else its first path_blocked since t0."""
    blocked = [tr for tr in traces if tr.get('robot') == rover and tr.get('kind') == 'path_blocked'
               and tr.get('t', 0) >= t0 - slack]
    return next((tr for tr in blocked if names_pillar(tr)), blocked[0] if blocked else None)


def route_ahead(path, pose: dict, along_m: float = PILLAR_AHEAD_M, fov_deg: float = PILLAR_VIEW_DEG) -> bool:
    """Does the route point along_m along the planned path lie in front of the rover's camera: within fov_deg of its
    heading and at least 0.8 · along_m away in a straight line (a route that turns back first does not count)?"""
    try:
        x0, y0, yaw = float(pose['x']), float(pose['y']), float(pose['yaw'])
    except (KeyError, TypeError, ValueError):
        return False
    acc, (px, py) = 0.0, (x0, y0)
    for q in path or ():
        acc += math.hypot(q[0] - px, q[1] - py)
        px, py = q[0], q[1]
        if acc >= along_m:
            off = (math.atan2(py - y0, px - x0) - yaw + math.pi) % (2 * math.pi) - math.pi
            return abs(math.degrees(off)) <= fov_deg and math.hypot(px - x0, py - y0) >= 0.8 * along_m
    return False


def on_footprint(xy, ev: dict, tol_m: float = PILLAR_FOOTPRINT_M) -> bool:
    """Is world point xy within tol_m of the fallen pillar's footprint (the event's base → tip segment)?"""
    try:
        (px, py), (ax, ay), (bx, by) = xy, ev['base'], ev['tip']
    except (KeyError, TypeError, ValueError):
        return False
    dx, dy = bx - ax, by - ay
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9)))
    return math.hypot(px - (ax + u * dx), py - (ay + u * dy)) <= tol_m


def pillar_verdict(blocked: dict | None, sighting, t0: float) -> tuple[bool, str]:
    """The pillar is perceived when it blocks the path and the camera identifies it: a path_blocked trace that names
    it ("camera confirms a fallen pillar"), or a path_blocked since the injection plus a PILLAR detection on the
    pillar's footprint since then, in either order (LiDAR usually blocks the route while the trunk is still falling,
    the camera names it a moment later). A sighting without a block, or a block nobody identified, is not enough.
    (ok, detail)."""
    blk = f"path_blocked {blocked['t'] - t0:.1f} s: {blocked.get('input')}" if blocked else 'no path_blocked'
    det = None
    if sighting is not None:
        ts, d = sighting
        det = f"PILLAR identified {ts - t0:.1f} s at {d.get('range_m')} m (conf {d.get('conf')})"
    if names_pillar(blocked):
        return True, blk + (f'; {det}' if det else '')
    if det is None:
        return False, (f"path_blocked {blocked['t'] - t0:.1f} s without naming the pillar: {blocked.get('input')}"
                       if blocked else 'no path_blocked') + '; no PILLAR detection on its footprint since the inject'
    if blocked is None:
        return False, f'{det}; but no path_blocked since the inject'
    first, then = (blk, det) if blocked['t'] <= ts else (det, blk)
    return True, f'{first}; {then}'


def descent_verdict(agl0, hit, t0: float, limit: float) -> tuple[bool, str]:
    """The drone came down to ≤ limit from above it: one already that low at the injection shows no descent.
    hit = the first (wall epoch, AGL) ≤ limit since t0, or None. (ok, detail)."""
    if not isinstance(agl0, (int, float)) or agl0 <= limit:
        return False, f'AGL {agl0} m at the inject, not above {limit} m: no descent to show'
    if hit is None:
        return False, f'AGL {agl0:.1f} m at the inject, never ≤ {limit} m since'
    return True, f'{hit[0] - t0:.1f} s: {agl0:.1f} → {hit[1]:.1f} m AGL'


def path_distance(pt, path) -> float | None:
    """Planar distance from pt to the nearest point of a [[x, y], …] path (None without either)."""
    if not pt or not path:
        return None
    return min(math.hypot(pt[0] - q[0], pt[1] - q[1]) for q in path)


class Stream:
    def __init__(self):
        self.status: dict = {}
        self.telemetry: dict | None = None
        self.traces: list = []
        self.events: list = []
        self.brain: list = []             # (wall receipt time, /brain/decision) in arrival order
        self.counts: dict = {}
        self.frames: dict = {}            # (robot, 'rgb' | 'depth') → camera frames received
        self.map_stats: dict = {}
        self.timeline: list = []          # --dump: every trace and event, plus a 1 Hz pose / state sample per robot
        self._last_sample = 0.0
        self.last_map: dict | None = None
        self._map_snaps = 0
        self.series: dict = {}            # (robot | 'env', reading) → (wall epoch, value) per telemetry frame
        self.dets: dict = {}              # robot → (wall epoch, compact detections) per frame that carried any

    def _push(self, key: tuple, t: float, v):
        if v is not None:
            self.series.setdefault(key, deque(maxlen=SERIES_LEN)).append((t, v))

    def _record(self, msg: dict):
        """Follow the readings a single telemetry frame cannot show: a drop, a sighting, the storm's end."""
        t = time.time()
        self._push(('env', 'dust_level'), t, _num(((msg.get('mission') or {}).get('env') or {}).get('dust_level')))
        for r, st in (msg.get('robots') or {}).items():
            for key, v in telemetry_values(st).items():
                self._push((r, key), t, v)
            dets = st.get('detections')
            if isinstance(dets, list) and dets:
                self.dets.setdefault(r, deque(maxlen=SERIES_LEN)).append((t, dets))

    def map_snapshot(self, why: str):
        """Into the timeline: the latest planet map layers (R elevation, G class × 40, B reveal age) with meta."""
        if self.last_map is not None:
            m = self.last_map
            self.timeline.append({'w': time.monotonic(), 'type': 'map', 'why': why, 'png_b64': m.get('png_b64'),
                                  'meta': m.get('meta'), 'stats': m.get('stats')})

    def _sample(self, msg: dict):
        w = time.monotonic()
        if w - self._last_sample < 1.0:
            return
        self._last_sample = w
        robots = {}
        for r, st in (msg.get('robots') or {}).items():
            p = st.get('pose') or {}
            robots[r] = {'state': st.get('state'), 'xy': [p.get('x'), p.get('y')], 'yaw': p.get('yaw'),
                         'rescue': st.get('rescue'), 'health': (st.get('health') or {}).get('pct'),
                         'goal': st.get('goal'), 'sensors': st.get('sensors'), 'brain_cap': st.get('brain_cap'),
                         'brain_agl': st.get('brain_agl'), 'mode': st.get('mode'),
                         'solar_w': (st.get('health') or {}).get('solar_w'), 'power_save': st.get('power_save'),
                         'wind_mps': st.get('wind_mps'), 'wind_descent': st.get('wind_descent')}
        m = msg.get('mission') or {}
        self.timeline.append({'w': w, 'type': 'sample', 'sim_t': m.get('sim_time_s'), 'rtf': m.get('rtf'),
                              'robots': robots, 'brain': m.get('brain'), 'env': m.get('env')})

    def feed(self, msg: dict):
        t = msg.get('type')
        self.counts[t] = self.counts.get(t, 0) + 1
        if t == 'camera':                 # every robot's rgb is fanned out; only the selected one also gets depth
            key = (msg.get('robot'), msg.get('kind') or 'rgb')
            self.frames[key] = self.frames.get(key, 0) + 1
        elif t == 'status':
            self.status = msg
        elif t == 'telemetry':
            self.telemetry = msg
            self._record(msg)
            self._sample(msg)
        elif t == 'trace':
            self.traces.append(msg['entry'])
            self.timeline.append({'w': time.monotonic(), 'type': 'trace', 'entry': msg['entry']})
            kind = msg['entry'].get('kind')
            if kind in ('rescue_start', 'rescue_abort') or (kind == 'rescue_noroute' and self._map_snaps < 3):
                self._map_snaps += kind == 'rescue_noroute'
                self.map_snapshot(kind)              # the map a rescue planned on / gave up on, for a replay
        elif t == 'event':
            self.events.append(msg['event'])
            self.timeline.append({'w': time.monotonic(), 'type': 'event', 'event': msg['event']})
        elif t == 'map':
            self.map_stats = msg['stats']
            self.last_map = msg
        elif t == 'brain':
            d = msg.get('decision')
            if isinstance(d, dict):
                self.brain.append((time.time(), d))
                self.timeline.append({'w': time.monotonic(), 'type': 'brain', 'decision': d})

    def streams_ok(self, camera: str) -> bool:
        """rgb and depth frames from the selected camera robot (not just any robot's rgb), LiDAR and the map."""
        return all(self.frames.get((camera, kind)) for kind in ('rgb', 'depth')) and \
            all(self.counts.get(t) for t in ('lidar', 'map'))

    def frames_summary(self) -> str:
        per: dict = {}
        for (r, kind), n in sorted(self.frames.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] != 'rgb', kv[0][1])):
            per.setdefault(r, []).append(f'{kind} {n}')
        return ', '.join(f'{r} {" + ".join(v)}' for r, v in per.items()) or 'none'


class Smoke:
    def __init__(self, url: str):
        self.url = url.rstrip('/')
        self.http = httpx.AsyncClient(base_url=self.url, timeout=60.0)
        self.s = Stream()
        self.results: list = []
        self.t0 = time.monotonic()

    def check(self, name: str, ok: bool, detail: str = ''):
        self.results.append((name, ok, detail))
        print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f' — {detail}' if detail else ''), flush=True)
        return ok

    async def wait(self, pred, timeout: float, poll: float = 0.1):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            v = pred()
            if v:
                return v, time.monotonic() - t0
            await asyncio.sleep(poll)
        return None, timeout

    def robot(self, r: str) -> dict:
        return ((self.s.telemetry or {}).get('robots') or {}).get(r) or {}

    def rockslide_diag(self, t: float, rover: str, before: dict) -> str:
        """Why no path_blocked trace: rover state/path at injection, where the rocks fell, what it traced since."""
        pose, path = before.get('pose') or {}, before.get('path') or []
        ev = next((e for e in self.s.events if e.get('op') == 'spawn_rockslide' and e.get('t', 0) >= t - 1), {})
        imp = ev.get('impact')
        near = min((((imp[0] - q[0]) ** 2 + (imp[1] - q[1]) ** 2) ** 0.5 for q in path), default=None) if imp else None
        since = [f"{tr['kind']}@{tr.get('state')}" for tr in self.s.traces if tr['robot'] == rover and tr['t'] >= t - 0.5]
        return (f"at inject: state={before.get('state')} path_pts={len(path)} pose=({pose.get('x', 0):.1f},"
                f"{pose.get('y', 0):.1f}); impact={imp} ({near if near is None else round(near, 1)} m from path), "
                f"{len(ev.get('rocks', []))} rocks; now state={self.robot(rover).get('state')}; traces since: {since[:8]}")

    def trace_after(self, t_start: float, robot: str, kinds: tuple):
        for tr in self.s.traces:
            if tr['robot'] == robot and tr['kind'] in kinds and tr['t'] >= t_start - 0.5:
                return tr
        return None

    # ------------------------------------------------------------------ advisory brain
    def brain_after(self, t_start: float, pred, slack: float = 0.5):
        """First /brain/decision received since t_start - slack (wall) that satisfies pred → (receipt time,
        decision)."""
        return first_decision(self.s.brain, t_start, pred, slack)

    @staticmethod
    def brain_desc(d: dict) -> str:
        """Who decided and what it cost: source, real model, measured host→cloud round-trip, tokens."""
        def num(k):                       # a malformed field must not crash the report
            v = d.get(k)
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        cap, alt, lat = num('target_speed_mps'), num('target_altitude_m'), num('latency_ms')
        return (f"{d.get('robot')} {d.get('event')} → {d.get('action')}"
                + ('' if cap is None else f' cap {cap:.2f} m/s') + ('' if alt is None else f' AGL {alt:.1f} m')
                + f" [{d.get('source')} · {d.get('model')} · "
                + ('no cloud call' if lat is None else f'cloud {lat:.0f} ms measured')
                + f" · tokens {d.get('tokens_in')}/{d.get('tokens_out')}]"
                + (f" error: {d['error']}" if d.get('error') else '')
                + (f" (trigger {d.get('trigger')}: {str(d.get('trigger_input') or '')[:60]})"
                   if d.get('trigger') else ''))

    def brain_recent(self, n: int = 4) -> str:
        st = self.mission().get('brain') or {}
        last = ' | '.join(self.brain_desc(d) for _, d in self.s.brain[-n:]) or 'no decisions yet'
        return f"brain status stats={st.get('stats')} last_error={st.get('last_error')}; last decisions: {last}"

    async def wait_brain(self, t_start: float, pred, timeout: float, slack: float = 0.5):
        """Wait until a matching decision has arrived, at most `timeout` s after t_start (wall)."""
        await self.wait(lambda: self.brain_after(t_start, pred, slack), max(0.0, t_start + timeout - time.time()))
        hit = self.brain_after(t_start, pred, slack)
        return hit if hit and hit[0] - t_start <= timeout else None

    async def operator(self, text: str, robot: str):
        """POST /api/command → (command id or None, wall time sent, response)."""
        t = time.time()
        r = await self.http.post('/api/command', json={'text': text, 'robot': robot})
        body = r.json() if r.headers.get('content-type', '').startswith('application/json') else {}
        self.s.timeline.append({'w': time.monotonic(), 'type': 'operator', 'text': text, 'robot': robot,
                                'status': r.status_code, 'response': body})
        return (body.get('id') if r.status_code == 200 and body.get('ok') else None), t, (r.status_code, body)

    async def mapping_rover(self, rovers: list, states: tuple, k: float, what: str) -> str:
        """The first rover (rover_0 first) in one of `states` within 30 s, else rover_0: a rescue that did not finish
        must not take the later checks down with it."""
        x, _ = await self.wait(lambda: next((r for r in rovers if self.robot(r).get('state') in states), None), 30 * k)
        x = x or rovers[0]
        if x != rovers[0]:
            print(f'  [INFO] {what} on {x}: {rovers[0]} is {self.robot(rovers[0]).get("state")}', flush=True)
        return x

    async def operator_sequence(self, rover: str, k: float):
        """Operator commands through the advisory brain: unsafe → rejected, slow-down → capped, stop → immediate."""
        await self.wait(lambda: self.robot(rover).get('state') in ('EXPLORING', 'CAUTION', 'REPLANNING'), 20 * k)
        # 1. unsafe: the local check refuses a command into a known hazard (or above the local limit), whatever
        #    Gemma says
        cid, t, resp = await self.operator('drive at full speed into the rockslide', rover)
        hit = cid and await self.wait_brain(t, lambda d: d.get('command_id') == cid
                                            and d.get('action') == 'REJECT_UNSAFE_COMMAND', 15 * k)
        seen = [self.brain_desc(d) for _, d in self.s.brain if cid and d.get('command_id') == cid]
        self.check('operator: unsafe command rejected', bool(hit),
                   f'{hit[0] - t:.1f} s: {self.brain_desc(hit[1])}' if hit else
                   f'POST → {resp}; decisions for {cid}: {seen or "none"}; {self.brain_recent()}')
        # 2. slow-down: a cap ≤ 0.2 m/s in the decision, then honoured by the rover's planner (agent_state brain_cap)
        cid, t, resp = await self.operator('slow down to 0.2 m/s', rover)
        hit = cid and await self.wait_brain(t, lambda d: d.get('command_id') == cid
                                            and isinstance(d.get('target_speed_mps'), (int, float))
                                            and d['target_speed_mps'] <= 0.2 + 1e-6, 15 * k)
        seen = [self.brain_desc(d) for _, d in self.s.brain if cid and d.get('command_id') == cid]
        self.check('operator: slow-down → cap ≤ 0.2 m/s', bool(hit),
                   f'{hit[0] - t:.1f} s: {self.brain_desc(hit[1])}' if hit else
                   f'POST → {resp}; decisions for {cid}: {seen or "none"}; {self.brain_recent()}')
        if hit:
            seen = {}

            def capped():               # verdict on the same telemetry sample that first shows the cap
                seen['v'], seen['st'] = cap_verdict(self.robot(rover), 0.2), self.robot(rover)
                return seen['v'][0] is not None
            got, dt = await self.wait(capped, 5 * k)
            ok, why = seen.get('v', (None, 'no telemetry'))
            st = seen.get('st') or {}
            self.check(f'operator: {rover} planner applies the cap', bool(got) and ok is True,
                       f"{dt:.1f} s: {why}; brain_cap={st.get('brain_cap')} speed {st.get('speed')}")
        # 3. stop: the brain node answers at once with the local decision (immediate), the rover halts
        v0 = self.ground_speed(rover)
        cid, t, resp = await self.operator('stop', rover)
        hit = cid and await self.wait_brain(t, lambda d: d.get('command_id') == cid and d.get('immediate') is True,
                                            5 * k)
        stopped, dt = await self.wait(lambda: (v := self.ground_speed(rover)) is not None and abs(v) < 0.05,
                                      max(0.0, t + 5 * k - time.time()), poll=0.05)
        self.check('operator: stop → immediate decision, rover halts', bool(hit) and bool(stopped),
                   (f'decision {hit[0] - t:.2f} s: {self.brain_desc(hit[1])}' if hit else
                    f'no immediate decision for {cid} (POST → {resp}); {self.brain_recent()}')
                   + f'; ground speed {v0} → {self.ground_speed(rover)} m/s'
                   + (f' after {time.time() - t:.1f} s' if stopped else ' (still moving after 5 s)'))
        # 4. lift the stop cap so nothing later is slowed; any outcome is acceptable here
        cid, t, resp = await self.operator('resume at full speed', rover)
        hit = cid and await self.wait_brain(t, lambda d: d.get('command_id') == cid and not d.get('immediate'), 10 * k)
        # the planner releases the operator's own stop on a hazard-free faster / set-speed command (§9.6)
        cleared = operator_cap_cleared(self.robot(rover))
        if hit and not cleared:                                  # 5 Hz agent_state lag
            cleared = bool((await self.wait(lambda: operator_cap_cleared(self.robot(rover)), 3 * k))[0])
        print('  [INFO] operator: "resume at full speed" → '
              + (self.brain_desc(hit[1]) if hit else f'no decision within {10 * k:.0f} s (POST → {resp})')
              + f"; {rover} operator cap " + ('cleared' if cleared else 'still set')
              + f" (brain_cap={self.robot(rover).get('brain_cap')})", flush=True)

    def ground_speed(self, r: str):
        st = self.robot(r)
        v = (st.get('sensors') or {}).get('v_ground')
        return v if isinstance(v, (int, float)) else st.get('speed')

    async def ws_reader(self, ws):
        async for raw in ws:
            self.s.feed(json.loads(raw))

    def rescue_room_now(self, target: str, rescuer: str) -> tuple[bool, str]:
        m = self.s.last_map
        p, q = self.robot(target).get('pose') or {}, self.robot(rescuer).get('pose') or {}
        if not m or p.get('x') is None or q.get('x') is None:
            return False, 'no map or pose yet'
        return rescue_room(map_classes(m['png_b64']), m['meta'], (p['x'], p['y']), (q['x'], q['y']))

    async def rescue_sequence(self, target: str, rescuer: str, k: float):
        await self.wait(lambda: self.robot(target).get('state') in ('EXPLORING', 'CAUTION', 'REPLANNING'), 40 * k)
        # like an operator in the demo, strike the target once it stands where a teammate can reach it: the scripted
        # rockslide → tip sequence can leave it in a pocket of steep ground and boulders by the sector edge, where
        # the rescuer correctly declines (rescue_noroute) and docking could not be exercised
        room = {'v': (False, '')}

        def roomy():
            room['v'] = self.rescue_room_now(target, rescuer)
            return room['v'][0]
        _, dt = await self.wait(roomy, 40 * k, poll=0.5)
        print(f"  [INFO] strike site: {room['v'][1]}" + ('' if room['v'][0] else
              f' (no open ground within {40 * k:.0f} s: striking anyway; expect rescue_noroute)'), flush=True)
        t = time.time()
        r = (await self.http.post('/api/inject/damage', json={'robot': target})).json()
        self.check('critical damage injected', r.get('ok'), r.get('detail', ''))
        ev, _ = await self.wait(lambda: next((e for e in self.s.events if e.get('op') == 'critical_damage'
                                              and e.get('t', 0) >= t - 1), None), 5 * k)
        self.check('debris strike world event', bool(ev and ev.get('ok')),
                   f"peak {ev.get('peak_g')} g, wheels {ev.get('wheel_health')}" if ev else 'no event')
        h, _ = await self.wait(lambda: ((self.robot(target).get('health') or {}).get('critical')
                                        and self.robot(target).get('health')), 8 * k)
        self.check('damage perceived: health < 15 %', bool(h) and h.get('pct', 100) < 15.0,
                   f"health {h.get('pct')}%, faults {h.get('faults')}, peak {h.get('peak_g')} g" if h
                   else f"health {(self.robot(target).get('health') or {})}")
        # the local FSM has already acted (safe-hold, SOS); the brain is consulted on the side
        hit = await self.wait_brain(t, lambda d: d.get('event') in ('critical_damage', 'peer_sos'), 20 * k)
        self.check('brain: decision after the strike', bool(hit),
                   f'{hit[0] - t:.1f} s: {self.brain_desc(hit[1])}' if hit else self.brain_recent())
        chain = [(target, 'critical_damage', 6), (target, 'sos', 6), (rescuer, 'rescue_start', 40),
                 (rescuer, 'rescue_dock', 180), (rescuer, 'docked', 50), (target, 'repaired', 30),
                 (rescuer, 'repair_complete', 10), (rescuer, 'resume', 20), (target, 'resume', 20)]
        for robot, kind, tmo in chain:
            tr, _ = await self.wait(lambda robot=robot, kind=kind: self.trace_after(t, robot, (kind,)), tmo * k)
            st = self.robot(robot)
            since = [f"{x['robot']} {x['kind']}@{x['state']}: {x['input'][:70]}" for x in self.s.traces
                     if x['robot'] in (target, rescuer) and x['t'] >= t - 0.5]
            self.check(f'rescue: {robot} {kind}', tr is not None,
                       f'{time.time() - t:.1f} s: {tr["input"]}' if tr else
                       f'{time.time() - t:.1f} s: state={st.get("state")} rescue={st.get("rescue")} '
                       f'health={(st.get("health") or {}).get("pct")}; traces since the strike (last 10): '
                       + ' | '.join(since[-10:]))
            if tr is None:
                return
            if kind == 'docked':
                lock = self.trace_after(t, rescuer, ('rescue_lock',))
                self.check('rescue: visual lock-on perceived', lock is not None and 'PERCEIVED: DAMAGED UNIT' in
                           lock['input'], lock['input'] if lock else 'docked on odometry only')
        ev = next((e for e in self.s.events if e.get('op') == 'repair' and e.get('ok') and e.get('t', 0) >= t), None)
        self.check('repair accepted by the world (≤ 1.6 m)', ev is not None,
                   f"distance {ev.get('distance_m')} m" if ev else str([e for e in self.s.events if e.get('op') == 'repair'][-2:]))
        await self.wait(lambda: self.robot(target).get('state') in ('EXPLORING', 'CAUTION', 'REPLANNING')
                        and self.robot(rescuer).get('state') in ('EXPLORING', 'CAUTION', 'REPLANNING'), 20 * k)
        h = self.robot(target).get('health') or {}
        self.check('both rovers back to mapping', self.robot(target).get('state') in ('EXPLORING', 'CAUTION', 'REPLANNING')
                   and self.robot(rescuer).get('state') in ('EXPLORING', 'CAUTION', 'REPLANNING'),
                   f"{target} {self.robot(target).get('state')} health {h.get('pct')}%, "
                   f"{rescuer} {self.robot(rescuer).get('state')}")

    # ------------------------------------------------------------------ phase 3: pillar collapse, dust storm
    def env(self) -> dict:
        return self.mission().get('env') or {}

    async def wait_by(self, pred, t0: float, timeout: float, poll: float = 0.1):
        """wait() against the deadline t0 + timeout (wall) that several checks of one injection share; pred is tried
        at least once more, so a check whose budget an earlier one used up still sees what has already happened."""
        v, _ = await self.wait(pred, remaining(t0, timeout, time.time()), poll)
        return v if v else pred()

    def robot_diag(self, t: float, robot: str, n: int = 10) -> str:
        """A robot's state now, the environment, and its last n trace kinds since t (the failure report)."""
        st = self.robot(robot)
        h = st.get('health') or {}
        if robot.startswith('scout'):
            now = (f"state={st.get('state')} AGL={st.get('altitude_agl')} wind_mps={st.get('wind_mps')} "
                   f"wind_descent={st.get('wind_descent')} brain_agl={st.get('brain_agl')}")
        else:
            now = (f"state={st.get('state')} mode={st.get('mode')} uncertainty={st.get('uncertainty')} "
                   f"power_save={st.get('power_save')} solar_w={h.get('solar_w')} battery={h.get('battery_pct')}%")
        kinds = [f"{x['kind']}@{x.get('state')}" for x in self.s.traces if x['robot'] == robot and x['t'] >= t - 0.5]
        return f'now {now}; env={self.env()}; {robot} traces since (last {n}): {kinds[-n:]}'

    def pillar_event(self, t: float) -> dict | None:
        return next((e for e in self.s.events if e.get('op') == 'pillar_collapse' and e.get('t', 0) >= t - 1), None)

    def pillar_diag(self, t: float, rover: str, before: dict) -> str:
        """Why the pillar was not perceived: rover state/path at injection, where the pillar fell, what the rover's
        perception reported and what it traced since."""
        pose, path = before.get('pose') or {}, before.get('path') or []
        ev = self.pillar_event(t) or {}
        near = path_distance(ev.get('cross'), path)
        since = [f"{tr['kind']}@{tr.get('state')}" for tr in self.s.traces
                 if tr['robot'] == rover and tr['t'] >= t - 0.5]
        return (f"at inject: state={before.get('state')} path_pts={len(path)} pose=({pose.get('x', 0):.1f},"
                f"{pose.get('y', 0):.1f}); pillar base={ev.get('base')} tip={ev.get('tip')} cross={ev.get('cross')} "
                f"({'?' if near is None else f'{near:.1f}'} m from the path), held {ev.get('waited_s')} s; "
                f"detections since (frames per class): {class_counts(self.s.dets.get(rover, ()), t) or 'none'}; "
                f"now state={self.robot(rover).get('state')}; traces since: {since[:8]}")

    async def pillar_sequence(self, rover: str, k: float):
        """A basalt pillar topples across the rover's planned path: perceived (camera / depth / LiDAR), the path
        blocked and re-planned around it, and a brain decision on the new hazard."""
        # "as the rover approaches": inject when the route ahead is in the camera's view, not behind a turn
        aimed, _ = await self.wait(lambda: self.robot(rover).get('state') == 'EXPLORING'
                                   and route_ahead(self.robot(rover).get('path'), self.robot(rover).get('pose')),
                                   PILLAR_AIM_S * k)
        if not aimed:
            print(f'  [INFO] pillar: no route ahead in the camera view within {PILLAR_AIM_S:.0f} s; injecting anyway',
                  flush=True)
        before = self.robot(rover)
        t = time.time()
        r = (await self.http.post('/api/inject/pillar', json={'robot': rover})).json()
        self.check('pillar collapse injected', r.get('ok'), r.get('detail', ''))
        ev, dt = await self.wait(lambda: self.pillar_event(t), PILLAR_EVENT_S * k)
        ok = bool(ev and ev.get('ok'))
        held = _num((ev or {}).get('waited_s'))
        self.check('pillar_collapse world event', ok,
                   f"{dt:.1f} s: base {ev.get('base')} → tip {ev.get('tip')}, across the route at {ev.get('cross')}"
                   + (f' (held {held:.1f} s for the next route)' if held else '') if ok else
                   f"{(ev or {}).get('detail', 'no event')}; {self.pillar_diag(t, rover, before)}")
        if not ok:                       # no pillar in the world: nothing to perceive, re-plan or advise on
            return
        seen: dict = {'tr': None, 'det': None}

        def perceived():                 # the block naming the pillar, or a block and a PILLAR sighting on its footprint
            seen['det'] = first_sighting(self.s.dets.get(rover, ()), 'PILLAR', t,
                                         where=lambda d: on_footprint(d.get('world_xy'), ev))
            seen['tr'] = pillar_block(self.s.traces, rover, t)
            return pillar_verdict(seen['tr'], seen['det'], t)[0]
        await self.wait_by(perceived, t, PILLAR_BLOCK_S * k)
        ok, why = pillar_verdict(seen['tr'], seen['det'], t)
        self.check('pillar perceived → path_blocked', ok, why if ok else f'{why}; {self.pillar_diag(t, rover, before)}')
        if not ok:                       # a re-plan with no pillar block to answer would prove nothing
            self.check('pillar: re-planned route', False, 'no path_blocked caused by the pillar to re-plan from')
        else:
            t_blk = seen['tr']['t']
            tr, _ = await self.wait(lambda: self.trace_after(t_blk, rover, ('replanned',)), PILLAR_REPLAN_S * k)
            self.check('pillar: re-planned route', tr is not None,
                       f"{tr['t'] - t_blk:.1f} s after the block ({tr['t'] - t:.1f} s): {tr['action']}" if tr
                       else self.robot_diag(t, rover))
        # only decisions received after the injection whose trigger is the fallen pillar itself
        hit = await self.wait_brain(t, lambda d: pillar_decision(d, rover), PILLAR_BRAIN_S * k, slack=0.0)
        other = [f"{d.get('trigger')}: {str(d.get('trigger_input') or '')[:50]}" for rx, d in self.s.brain
                 if rx >= t and d.get('robot') == rover and d.get('event') == 'novel_hazard']
        self.check('brain: novel_hazard decision after the pillar collapse', bool(hit),
                   f"{hit[0] - t:.1f} s, trigger {hit[1].get('trigger')}: {self.brain_desc(hit[1])}" if hit else
                   f'novel_hazard decisions for {rover} since the injection, by trigger: {other or "none"}; '
                   + self.brain_recent())

    async def storm_sequence(self, rover: str, drone: str | None, rovers: list, k: float):
        """A dust storm (solar blackout + high wind) that each robot infers from its own sensors: CAUTION from the
        camera's confidence, battery conservation from the measured solar power, the drone's descent from its own
        drift; then every one of them undone once the storm has passed."""
        await self.wait(lambda: self.robot(rover).get('state') == 'EXPLORING', 30 * k)
        if drone:                        # inject at survey altitude, so the descent has somewhere to come down from
            await self.wait(lambda: (_num(self.robot(drone).get('altitude_agl')) or 0.0) > DRONE_AGL_MAX + 1.0, 30 * k)
        agl0 = _num(self.robot(drone).get('altitude_agl')) if drone else None
        t = time.time()
        before = values_since(self.s.series.get((rover, 'solar_w'), ()), t - SOLAR_BEFORE_S, t)
        r = (await self.http.post('/api/inject/storm', json={})).json()
        self.check('storm injected', r.get('ok'), r.get('detail', ''))

        def echo() -> dict:              # the anomaly event: kind, wind and how long the world will blow
            return next((e for e in self.s.events if e.get('op') == 'anomaly' and e.get('t', 0) >= t - 1), {})
        # 1. CAUTION on the camera's own confidence, which the orange-red haze drives below 0.5
        cam: dict = {'min': None}

        def caution():
            c = values_since(self.s.series.get((rover, 'camera_conf'), ()), t)
            cam['min'] = min(c) if c else None
            tr = self.trace_after(t, rover, ('caution_enter',))
            return tr if tr and cam['min'] is not None and cam['min'] < STORM_CAM_MAX else None
        tr = await self.wait_by(caution, t, STORM_CAUTION_S * k)
        entered = self.trace_after(t, rover, ('caution_enter',))
        self.check(f'storm → CAUTION ≤ {STORM_CAUTION_S:.0f} s, camera confidence < {STORM_CAM_MAX}', tr is not None,
                   f"{tr['t'] - t:.1f} s: {tr['input']}; camera confidence min {cam['min']:.2f}, "
                   f"mode {self.robot(rover).get('mode')}" if tr else
                   f"caution_enter {'%.1f s' % (entered['t'] - t) if entered else 'not traced'}, camera confidence "
                   f"min since the inject {cam['min']}; storm event {echo() or 'none'}; {self.robot_diag(t, rover)}")
        # 2. the rover's measured solar power (hw_status → health.solar_w), median just before vs the latest samples
        sol: dict = {'v': (None, 'no samples yet')}

        def dropped():
            after = values_since(self.s.series.get((rover, 'solar_w'), ()), max(t, time.time() - SOLAR_AFTER_S))
            sol['v'] = solar_drop(before, after)
            return sol['v'][0] is not None and sol['v'][0] >= SOLAR_DROP_MIN
        ok = await self.wait_by(dropped, t, STORM_SOLAR_S * k)
        self.check(f'storm: {rover} measured solar power drops ≥ {100 * SOLAR_DROP_MIN:.0f} %', bool(ok),
                   f"{time.time() - t:.1f} s: {sol['v'][1]}" + ('' if ok else f'; {self.robot_diag(t, rover)}'))
        # 3. battery conservation, from the same power readings
        tr = await self.wait_by(lambda: self.trace_after(t, rover, ('power_save',)), t, STORM_POWER_S * k)
        self.check(f'storm: {rover} power_save (battery conservation)', tr is not None,
                   f"{tr['t'] - t:.1f} s: {tr['input']} → {tr['action']}" if tr else self.robot_diag(t, rover))
        # 4. the drone estimates the wind from its own drift and comes down by itself
        if drone:
            tr = await self.wait_by(lambda: self.trace_after(t, drone, ('wind_descent',)), t, WIND_DESCENT_S * k)
            self.check(f'storm: drone wind_descent ≤ {WIND_DESCENT_S:.0f} s', tr is not None,
                       f"{tr['t'] - t:.1f} s: {tr['input']} → {tr['action']}" if tr else
                       f'storm event {echo() or "none"}; {self.robot_diag(t, drone)}')
            hit = await self.wait_by(lambda: first_at_most(self.s.series.get((drone, 'altitude_agl'), ()), t,
                                                           DRONE_AGL_MAX), t, DRONE_AGL_S * k)
            ok, why = descent_verdict(agl0, hit, t, DRONE_AGL_MAX)
            self.check(f'storm: drone AGL ≤ {DRONE_AGL_MAX} m ≤ {DRONE_AGL_S:.0f} s', ok,
                       why if ok else f'{why}; {self.robot_diag(t, drone)}')
        # 5. after the storm (its end read from the falling dust level, so the checks follow the simulated storm
        #    however the simulation is paced), everything is undone before the next change starts
        dur = _num(echo().get('duration_s')) or STORM_S
        end = await self.wait_by(lambda: storm_end(self.s.series.get(('env', 'dust_level'), ()), t), t,
                                 (2 * dur + 15) * k, poll=0.25)
        t_from, t_dl = (end, end) if end is not None else (t, time.time())
        since = 'after the storm' if end is not None else 'after the inject (no storm end seen in mission.env)'
        chain = [(rover, 'power_restored', f'storm clears: {rover} power_restored')]
        chain += [(drone, 'wind_clear', 'storm clears: drone wind_clear')] if drone else []
        chain += [(rover, 'caution_exit', 'storm clears → fused navigation restored')]
        for robot, kind, name in chain:
            tr = await self.wait_by(lambda robot=robot, kind=kind: self.trace_after(t_from, robot, (kind,)),
                                    t_dl, STORM_CLEAR_S * k)
            self.check(name, tr is not None, f"{tr['t'] - t_from:.1f} s {since}: {tr['input']}" if tr else
                       f'{time.time() - t_from:.1f} s {since}: {self.robot_diag(t, robot)}')

        def over():                      # the speed change and the pacing check must not run in the storm's tail
            return (_num(self.env().get('dust_level')) or 0.0) <= STORM_CLEAR_LEVEL and all(
                self.robot(x).get('state') != 'CAUTION' and self.robot(x).get('mode') != 'LIDAR_ONLY' for x in rovers)
        ok, dt = await self.wait(over, STORM_OVER_S * k)
        print('  [INFO] storm over: ' + (f'air clear and no rover in CAUTION ({dt:.1f} s after the checks)' if ok else
              f"still not clear after {dt:.0f} s: env={self.env()}, "
              + ', '.join(f"{x} {self.robot(x).get('state')}/{self.robot(x).get('mode')}" for x in rovers)
              + '; going on anyway'), flush=True)

    def mission(self) -> dict:
        return (self.s.telemetry or {}).get('mission') or {}

    async def rtf_settled(self, target: float, timeout: float):
        """Wait until the achieved real-time factor is within 15 % of target; returns the last value seen."""
        seen = {'rtf': None}

        def ok():
            seen['rtf'] = self.mission().get('rtf')
            return seen['rtf'] is not None and abs(seen['rtf'] - target) <= 0.15 * target
        await self.wait(ok, timeout, poll=0.25)
        return seen['rtf']

    async def run(self, backend: str, sector: str, rover: str = 'rover_0', speed: float = 1.0, rovers: int = 2,
                  brain: bool = True, phase3: bool = True) -> bool:
        print(f'Redshift smoke test against {self.url} (backend={backend}, sector={sector}, speed={speed:g}×, '
              f'brain={"on" if brain else "off"})', flush=True)
        k = 1.0 / min(speed, 1.0)            # stretch simulated-time timeouts to wall time below 1×
        async with websockets.connect(self.url.replace('http', 'ws', 1) + '/ws', max_size=2**24) as ws:
            reader = asyncio.create_task(self.ws_reader(ws))
            await ws.send(json.dumps({'type': 'select_camera', 'robot': rover}))
            r = await self.http.post('/api/mission/deploy', json={'sector': sector, 'gravity_g': 2.2, 'sim_backend': backend,
                                                                   'fleet': {'rovers': rovers, 'drones': 1}, 'seed': 42,
                                                                   'sim_speed': speed, 'ai_brain': brain})
            self.check('deploy accepted', r.status_code == 200 and r.json().get('ok'), r.text[:120])
            _, dt = await self.wait(lambda: self.s.status.get('status') == 'running', 150 if backend == 'gazebo' else 90)
            st = self.s.status
            self.check('mission running', st.get('status') == 'running', f'{dt:.1f} s, backend={st.get("backend")}')
            if backend == 'gazebo':
                self.check('no fallback to mock', st.get('backend') == 'gazebo')
            if st.get('status') != 'running':
                reader.cancel()
                return False
            f = (st.get('config') or {}).get('fleet') or {'rovers': 1, 'drones': 1}
            robots = [f'rover_{i}' for i in range(f['rovers'])] + [f'scout_{i}' for i in range(f['drones'])]
            _, dt = await self.wait(lambda: all(self.robot(x).get('state') not in (None, 'BOOT') for x in robots), 40 * k)
            self.check('telemetry for every robot', all(self.robot(x) for x in robots),
                       ', '.join(f'{x}:{self.robot(x).get("state")}' for x in robots))
            bs, _ = await self.wait(lambda: self.mission().get('brain'), 5 * k)
            self.check('brain status reported', bool(bs) and bs.get('enabled') is brain,
                       f"enabled={bs.get('enabled')} key_present={bs.get('key_present')} model={bs.get('model')} "
                       f"device={bs.get('device')}" if bs else 'no /brain/status in the telemetry mission block')
            await self.wait(lambda: self.s.streams_ok(rover), 20 * k)
            self.check('camera / lidar / map streams', self.s.streams_ok(rover),
                       str({n: self.s.counts.get(n, 0) for n in ('telemetry', 'camera', 'lidar', 'map', 'trace')})
                       + f'; camera frames ({rover} selected, needs rgb + depth): {self.s.frames_summary()}')
            rtf = await self.rtf_settled(speed, 10)
            self.check('simulation paced at the requested speed', rtf is not None and rtf >= 0.75 * speed,
                       f'target {speed:g}×, achieved RTF {rtf}')
            m0, w0 = self.mission(), time.monotonic()
            await asyncio.sleep(25 * k)
            m1, w1 = self.mission(), time.monotonic()
            p0, p1 = m0['percent_mapped'], m1['percent_mapped']
            self.check('exploration progress', p1 - p0 >= 3.0, f'{p0:.1f}% → {p1:.1f}% in {25 * k:.0f} s')
            # T+ must advance exactly as far as /clock over the window. (Comparing its rate with one instantaneous RTF
            # sample failed whenever Gazebo's RTF wandered, e.g. 1.1 → 0.5 under load, though the clock was right.)
            de, s0, s1 = m1['elapsed_s'] - m0['elapsed_s'], m0.get('sim_time_s'), m1.get('sim_time_s')
            ds = None if s0 is None or s1 is None else s1 - s0
            self.check('mission clock runs on simulated time', ds is not None and ds > 0 and abs(de - ds) <= 0.3 + 0.02 * ds,
                       f'T+ advanced {de:.1f} s, simulated time {"?" if ds is None else f"{ds:.1f}"} s, '
                       f'wall {w1 - w0:.1f} s' + ('' if ds is None else f' (mean RTF {ds / (w1 - w0):.2f})'))

            # ---- unscripted change 1: rockslide across the planned path
            await self.wait(lambda: self.robot(rover).get('state') == 'EXPLORING' and len(self.robot(rover).get('path') or []) > 5, 40 * k)
            before = self.robot(rover)
            t = time.time()
            r = (await self.http.post('/api/inject/rockslide', json={'robot': rover})).json()
            self.check('rockslide injected', r.get('ok'), r.get('detail', ''))
            # near its goal the world holds the slide (≤ 8 s) for the rover's next route: allow for that
            tr, dt = await self.wait(lambda: self.trace_after(t, rover, ('path_blocked',)), 18 * k)
            ev = next((e for e in self.s.events if e.get('op') == 'spawn_rockslide' and e.get('t', 0) >= t - 1), {})
            held = f' (slide held {ev["waited_s"]:.1f} s for the next route)' if ev.get('waited_s') else ''
            self.check('rockslide → REPLANNING trace', tr is not None,
                       f'{dt:.1f} s: {tr["input"]}{held}' if tr else self.rockslide_diag(t, rover, before))
            tr, dt = await self.wait(lambda: self.trace_after(t, rover, ('replanned',)), 20 * k)
            self.check('re-planned route', tr is not None, f'{dt:.1f} s: {tr["action"] if tr else ""}')
            ev = next((e for e in self.s.events if e.get('op') == 'spawn_rockslide' and e.get('t', 0) >= t - 1), None)
            self.check('rockslide world event', ev is not None and len(ev.get('rocks', [])) > 0)
            # only decisions received after the injection whose trigger is the slide itself
            hit = await self.wait_brain(t, lambda d: rockslide_decision(d, rover), 20 * k, slack=0.0)
            other = [f"{d.get('trigger')}: {str(d.get('trigger_input') or '')[:50]}" for rx, d in self.s.brain
                     if rx >= t and d.get('robot') == rover and d.get('event') == 'novel_hazard']
            self.check('brain: novel_hazard decision after the rockslide', bool(hit),
                       f"{hit[0] - t:.1f} s, trigger {hit[1].get('trigger')}: {self.brain_desc(hit[1])}" if hit else
                       f'novel_hazard decisions for {rover} since the injection, by trigger: {other or "none"}; '
                       + self.brain_recent())

            # ---- unscripted change 2: dust/seismic anomaly
            await self.wait(lambda: self.robot(rover).get('state') == 'EXPLORING', 30 * k)
            t = time.time()
            r = (await self.http.post('/api/inject/anomaly', json={'dust_level': 0.85, 'duration_s': 15})).json()
            self.check('anomaly injected', r.get('ok'), r.get('detail', ''))
            tr, dt = await self.wait(lambda: self.trace_after(t, rover, ('caution_enter',)), 10 * k)
            mode, _ = await self.wait(lambda: self.robot(rover).get('mode') == 'LIDAR_ONLY', 3 * k)   # 5 Hz state lag
            self.check('dust → CAUTION / LIDAR_ONLY', tr is not None and bool(mode),
                       f'{dt:.1f} s: {tr["input"] if tr else ""}')
            tr, dt = await self.wait(lambda: self.trace_after(t, rover, ('caution_exit',)), 40 * k)
            rs = self.robot(rover)
            self.check('dust clears → fused navigation restored', tr is not None, f'{dt:.1f} s after inject' + (
                '' if tr else f"; now state={rs.get('state')} mode={rs.get('mode')} uncertainty={rs.get('uncertainty')} "
                f"env={((self.s.telemetry or {}).get('mission') or {}).get('env')}; traces since: "
                f"{[x['kind'] for x in self.s.traces if x['robot'] == rover and x['t'] >= t - 0.5][:10]}"))

            # ---- failure: tip/slip jolt → safe-hold recovery
            await self.wait(lambda: self.robot(rover).get('state') == 'EXPLORING', 30 * k)
            before = self.robot(rover)
            t = time.time()
            r = (await self.http.post('/api/inject/tip', json={'robot': rover})).json()
            self.check('tip/slip jolt injected', r.get('ok'), r.get('detail', ''))
            peak = {'roll': 0.0}

            def safe_hold_seen():       # also tracks the peak roll the 15 Hz telemetry shows, for the failure report
                roll = (self.robot(rover).get('imu') or {}).get('roll_deg')
                if roll is not None:
                    peak['roll'] = max(peak['roll'], abs(roll))
                return self.trace_after(t, rover, ('safe_hold',))
            chain = [('safe_hold', 4), ('recovery_reverse', 10), ('hazard_marked', 30), ('scout_tasked', 2),
                     ('scan_complete', 60), ('resume', 5)]
            for kind, tmo in chain:
                pred = safe_hold_seen if kind == 'safe_hold' else (lambda kind=kind: self.trace_after(t, rover, (kind,)))
                tr, dt = await self.wait(pred, tmo * k, poll=0.05)
                detail = f'{time.time() - t:.1f} s: {tr["input"]}' if tr else f'{time.time() - t:.1f} s: '
                if tr is None and kind == 'safe_hold':
                    ev = next((e for e in self.s.events if e.get('op') == 'jolt' and e.get('t', 0) >= t - 1), None)
                    detail += (f"state before={before.get('state')} speed={before.get('speed')}; jolt event={ev}; "
                               f"peak |roll| seen {peak['roll']:.1f}° (tip alarm {self.robot(rover).get('tip_alarm_deg')}°); "
                               f"now state={self.robot(rover).get('state')}")
                self.check(f'recovery: {kind}', tr is not None, detail)
                if tr is None:
                    break
            perm = self.s.map_stats.get('permanent_hazards', [])
            self.check('permanent hazard on /planet_map', len(perm) > 0, str(perm[:2]))

            # ---- most critical failure: debris strike → SOS → recovery-rover rescue → field repair → both resume
            # nothing so far (boot, rockslide, dust, jolt) may have been read as critical damage
            early = [tr for tr in self.s.traces if tr['kind'] in ('critical_damage', 'sos')]
            self.check('no critical damage before a strike', not early,
                       '; '.join(f"{tr['robot']} {tr['state']} {tr['kind']}: {tr['input'][:80]}" for tr in early[:3]))
            if 'rover_1' in robots:
                await self.rescue_sequence('rover_0', 'rover_1', k)

            # ---- operator commands through the advisory brain (after the rescue / the tip recovery)
            rovers = [x for x in robots if x.startswith('rover')]
            await self.operator_sequence(await self.mapping_rover(rovers, ('EXPLORING', 'CAUTION', 'REPLANNING'), k,
                                                                  'Operator commands'), k)

            # ---- Phase 3 unscripted changes: a pillar topples across the path, then a dust storm (solar blackout +
            #      high wind). They come last: a fallen trunk plus a storm can leave rover_0 in a pocket no rescuer
            #      can reach, which would test the strike site rather than the rescue.
            if phase3:
                t3 = time.time()
                p3 = await self.mapping_rover(rovers, ('EXPLORING',), k, 'Phase 3')
                await self.pillar_sequence(p3, k)
                await self.storm_sequence(p3, next((x for x in robots if x.startswith('scout')), None), rovers, k)
                late = [tr for tr in self.s.traces if tr['kind'] in ('critical_damage', 'sos') and tr['t'] >= t3 - 0.5]
                self.check('no critical damage from the pillar or the storm', not late,
                           '; '.join(f"{tr['robot']} {tr['state']} {tr['kind']}: {tr['input'][:80]}" for tr in late[:3]))
            else:
                print('  [INFO] Phase 3 checks (pillar collapse, dust storm) skipped (--skip-phase3)', flush=True)

            # ---- live speed change mid-mission
            alt = 0.5 * speed if speed > 0.75 else 2.0 * speed     # a target every backend can reach (mock ≈ 1.1× max)
            r = (await self.http.post('/api/sim/speed', json={'speed': alt})).json()
            ev, _ = await self.wait(lambda: next((e for e in self.s.events if e.get('op') == 'set_speed'), None), 5)
            rtf = await self.rtf_settled(alt, 8) if r.get('ok') else None
            self.check('live speed change', bool(r.get('ok')) and bool(ev and ev.get('ok')) and rtf is not None
                       and abs(rtf - alt) <= 0.15 * alt, f"{r.get('detail', r)}; achieved RTF {rtf}")

            ids = [tr['id'] for tr in self.s.traces]
            self.check('trace ids unique', len(ids) == len(set(ids)), f'{len(ids)} traces')
            src = [d.get('source') for _, d in self.s.brain]
            lat = [d['latency_ms'] for _, d in self.s.brain if isinstance(d.get('latency_ms'), (int, float))]
            print(f"  [INFO] brain: {len(src)} decisions "
                  f"(gemma {src.count('gemma')}, local {src.count('local-fallback')})"
                  + (f', measured cloud round-trip median {sorted(lat)[len(lat) // 2]:.0f} ms' if lat else '')
                  + f"; status stats {(self.mission().get('brain') or {}).get('stats')}", flush=True)
            r = (await self.http.post('/api/mission/abort')).json()
            await self.wait(lambda: self.s.status.get('status') == 'stopped', 30)
            self.check('abort → stopped', self.s.status.get('status') == 'stopped')
            reader.cancel()
        left = sorted({pid for pat in CLEANUP_PATTERNS for pid in subprocess.run(
            ['pgrep', '-f', pat], capture_output=True, text=True).stdout.split()})
        self.check('no simulator processes left', not left, ' '.join(left))
        failed = [n for n, ok, _ in self.results if not ok]
        print(f'\n{len(self.results) - len(failed)}/{len(self.results)} checks passed in '
              f'{time.monotonic() - self.t0:.0f} s' + (f'; FAILED: {", ".join(failed)}' if failed else ''), flush=True)
        return not failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--backend', choices=('mock', 'gazebo'), default='mock')
    ap.add_argument('--sector', choices=('small', 'medium', 'large'), default='medium')
    ap.add_argument('--speed', type=float, default=1.0, help='simulation speed (real-time factor), 0.25–3')
    ap.add_argument('--rovers', type=int, choices=(1, 2), default=2,
                    help='fleet rovers (2 adds the critical-damage → rescue sequence)')
    ap.add_argument('--brain', choices=('on', 'off'), default='on',
                    help='MissionConfig ai_brain: Gemma advisory brain (cloud) on, or local rules only')
    ap.add_argument('--dump', type=Path, metavar='JSONL',
                    help='write every trace, event and brain decision plus 1 Hz per-robot pose / state samples here '
                         '(for post-mortems)')
    ap.add_argument('--skip-phase3', action='store_true',
                    help='skip the Phase 3 changes (pillar collapse, dust storm with solar blackout and high wind)')
    a = ap.parse_args()
    smoke = Smoke(a.url)
    try:
        ok = asyncio.run(smoke.run(a.backend, a.sector, speed=a.speed, rovers=a.rovers, brain=a.brain == 'on',
                                   phase3=not a.skip_phase3))
    finally:
        if a.dump:
            smoke.s.map_snapshot('end')
            a.dump.write_text(''.join(json.dumps(x) + '\n' for x in smoke.s.timeline))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
