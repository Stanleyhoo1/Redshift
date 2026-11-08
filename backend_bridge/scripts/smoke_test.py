#!/usr/bin/env python3
"""End-to-end smoke test of a running backend (the same REST + WebSocket traffic the web UI produces).

    python3 backend_bridge/scripts/smoke_test.py [--url http://127.0.0.1:8000] [--backend mock|gazebo] [--speed 1.0]
                                                 [--brain on|off]

Deploys a mission, checks telemetry / camera / LiDAR / map streams and exploration progress, then runs the three
unscripted changes (rockslide, dust anomaly, tip/slip jolt) and checks the agents' traced responses, and finally
aborts and checks no simulator process is left. Exit code 0 = every check passed.

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
import subprocess
import sys
import time

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
# FSM states whose drive is 'follow' along the planned path / 'dock' — the only drives a brain cap limits
FOLLOW_STATES = ('EXPLORING', 'CAUTION', 'RESCUE_TRANSIT')
DOCK_STATES = ('RESCUE_DOCKING',)


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def first_decision(decisions, t_start: float, pred, slack: float = 0.0):
    """First (receipt time, decision) of `decisions` received at or after t_start - slack that satisfies pred."""
    return next(((rx, d) for rx, d in decisions if rx >= t_start - slack and pred(d)), None)


def rockslide_decision(d: dict, rover: str) -> bool:
    """A novel_hazard decision for `rover` that the rockslide raised: triggered by the blocked path or the marked
    hazard, or whose trigger text names the obstacle — not some other hazard (a slope) that happened meanwhile."""
    if not isinstance(d, dict) or d.get('robot') != rover or d.get('event') != 'novel_hazard':
        return False
    if d.get('trigger') in ROCKSLIDE_TRIGGERS:
        return True
    text = str(d.get('trigger_input') or '').lower()
    return any(w in text for w in ROCKSLIDE_WORDS)


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
                         'brain_agl': st.get('brain_agl')}
        m = msg.get('mission') or {}
        self.timeline.append({'w': w, 'type': 'sample', 'sim_t': m.get('sim_time_s'), 'rtf': m.get('rtf'),
                              'robots': robots, 'brain': m.get('brain')})

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
            self._sample(msg)
        elif t == 'trace':
            self.traces.append(msg['entry'])
            self.timeline.append({'w': time.monotonic(), 'type': 'trace', 'entry': msg['entry']})
            if msg['entry'].get('kind') == 'rescue_noroute' and self._map_snaps < 3:
                self._map_snaps += 1
                self.map_snapshot('rescue_noroute')
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
        print(f'  [INFO] operator: "resume at full speed" → '
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
                  brain: bool = True) -> bool:
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
            await self.operator_sequence('rover_0', k)

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
    a = ap.parse_args()
    smoke = Smoke(a.url)
    try:
        ok = asyncio.run(smoke.run(a.backend, a.sector, speed=a.speed, rovers=a.rovers, brain=a.brain == 'on'))
    finally:
        if a.dump:
            smoke.s.map_snapshot('end')
            a.dump.write_text(''.join(json.dumps(x) + '\n' for x in smoke.s.timeline))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
