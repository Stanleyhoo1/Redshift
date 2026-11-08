#!/usr/bin/env python3
"""End-to-end smoke test of a running backend (the same REST + WebSocket traffic the web UI produces).

    python3 backend_bridge/scripts/smoke_test.py [--url http://127.0.0.1:8000] [--backend mock|gazebo] [--speed 1.0]

Deploys a mission, checks telemetry / camera / LiDAR / map streams and exploration progress, then runs the three
unscripted changes (rockslide, dust anomaly, tip/slip jolt) and checks the agents' traced responses, and finally
aborts and checks no simulator process is left. Exit code 0 = every check passed.

--speed runs the mission at that simulation speed and also checks the achieved real-time factor, that mission time
follows simulated time, and a live speed change. Timeouts are behavioural (simulated) seconds, so below 1× they are
stretched to wall time; above 1× they are left as they are.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time

from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend_bridge.mission_manager import CLEANUP_PATTERNS  # noqa: E402


class Stream:
    def __init__(self):
        self.status: dict = {}
        self.telemetry: dict | None = None
        self.traces: list = []
        self.events: list = []
        self.counts: dict = {}
        self.map_stats: dict = {}

    def feed(self, msg: dict):
        t = msg.get('type')
        self.counts[t] = self.counts.get(t, 0) + 1
        if t == 'status':
            self.status = msg
        elif t == 'telemetry':
            self.telemetry = msg
        elif t == 'trace':
            self.traces.append(msg['entry'])
        elif t == 'event':
            self.events.append(msg['event'])
        elif t == 'map':
            self.map_stats = msg['stats']


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

    async def ws_reader(self, ws):
        async for raw in ws:
            self.s.feed(json.loads(raw))

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

    async def run(self, backend: str, sector: str, rover: str = 'rover_0', speed: float = 1.0) -> bool:
        print(f'Redshift smoke test against {self.url} (backend={backend}, sector={sector}, speed={speed:g}×)', flush=True)
        k = 1.0 / min(speed, 1.0)            # stretch simulated-time timeouts to wall time below 1×
        async with websockets.connect(self.url.replace('http', 'ws', 1) + '/ws', max_size=2**24) as ws:
            reader = asyncio.create_task(self.ws_reader(ws))
            await ws.send(json.dumps({'type': 'select_camera', 'robot': rover}))
            r = await self.http.post('/api/mission/deploy', json={'sector': sector, 'gravity_g': 2.2, 'sim_backend': backend,
                                                                   'fleet': {'rovers': 1, 'drones': 1}, 'seed': 42,
                                                                   'sim_speed': speed})
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
            await self.wait(lambda: self.s.counts.get('camera') and self.s.counts.get('lidar') and self.s.counts.get('map'), 20 * k)
            self.check('camera / lidar / map streams', all(self.s.counts.get(k) for k in ('camera', 'lidar', 'map')),
                       str({k: self.s.counts.get(k, 0) for k in ('telemetry', 'camera', 'lidar', 'map', 'trace')}))
            rtf = await self.rtf_settled(speed, 10)
            self.check('simulation paced at the requested speed', rtf is not None and rtf >= 0.75 * speed,
                       f'target {speed:g}×, achieved RTF {rtf}')
            m0, w0 = self.mission(), time.monotonic()
            await asyncio.sleep(25 * k)
            m1, w1 = self.mission(), time.monotonic()
            p0, p1 = m0['percent_mapped'], m1['percent_mapped']
            self.check('exploration progress', p1 - p0 >= 3.0, f'{p0:.1f}% → {p1:.1f}% in {25 * k:.0f} s')
            rate, rtf = (m1['elapsed_s'] - m0['elapsed_s']) / (w1 - w0), m1.get('rtf') or 0.0
            self.check('mission clock runs on simulated time', rtf > 0 and abs(rate - rtf) <= 0.2 * rtf,
                       f'T+ advanced {rate:.2f} s per s, RTF {rtf:.2f}')

            # ---- unscripted change 1: rockslide across the planned path
            await self.wait(lambda: self.robot(rover).get('state') == 'EXPLORING' and len(self.robot(rover).get('path') or []) > 5, 40 * k)
            before = self.robot(rover)
            t = time.time()
            r = (await self.http.post('/api/inject/rockslide', json={'robot': rover})).json()
            self.check('rockslide injected', r.get('ok'), r.get('detail', ''))
            tr, dt = await self.wait(lambda: self.trace_after(t, rover, ('path_blocked',)), 12 * k)
            self.check('rockslide → REPLANNING trace', tr is not None,
                       f'{dt:.1f} s: {tr["input"]}' if tr else self.rockslide_diag(t, rover, before))
            tr, dt = await self.wait(lambda: self.trace_after(t, rover, ('replanned',)), 15 * k)
            self.check('re-planned route', tr is not None, f'{dt:.1f} s: {tr["action"] if tr else ""}')
            ev = next((e for e in self.s.events if e.get('op') == 'spawn_rockslide' and e.get('t', 0) >= t - 1), None)
            self.check('rockslide world event', ev is not None and len(ev.get('rocks', [])) > 0)

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

            # ---- live speed change mid-mission
            alt = 0.5 * speed if speed > 0.75 else 2.0 * speed     # a target every backend can reach (mock ≈ 1.1× max)
            r = (await self.http.post('/api/sim/speed', json={'speed': alt})).json()
            ev, _ = await self.wait(lambda: next((e for e in self.s.events if e.get('op') == 'set_speed'), None), 5)
            rtf = await self.rtf_settled(alt, 8) if r.get('ok') else None
            self.check('live speed change', bool(r.get('ok')) and bool(ev and ev.get('ok')) and rtf is not None
                       and abs(rtf - alt) <= 0.15 * alt, f"{r.get('detail', r)}; achieved RTF {rtf}")

            ids = [tr['id'] for tr in self.s.traces]
            self.check('trace ids unique', len(ids) == len(set(ids)), f'{len(ids)} traces')
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
    a = ap.parse_args()
    ok = asyncio.run(Smoke(a.url).run(a.backend, a.sector, speed=a.speed))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
