"""Mission lifecycle: writes the runtime config, runs `ros2 launch redshift_sim mission.launch.py` in its own
process group, streams its output as the boot log, tracks subsystem readiness, falls back from Gazebo to the mock
world when Gazebo produces no sensor data, and tears everything down on abort."""
from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable

import yaml

from .config_schema import MissionConfig

ROOT = Path(__file__).resolve().parent.parent
ANSI = re.compile(r'\x1b\[[0-9;]*m')
ROS_LINE = re.compile(r'^\[([\w\-]+)-\d+\] (?:\[(INFO|WARN|ERROR|DEBUG)\] \[[\d.]+\] \[([\w/]+)\]: )?(.*)$')
GAZEBO_TIMEOUT_S = 45.0
READY_TIMEOUT_S = 120.0
PIPE_LIMIT = 1 << 20          # longest launch-output line we buffer (asyncio's default is 64 KiB)
# leftovers to reap after the session is gone. Each one names this checkout's runtime/ or install/ tree (or the
# bridge's unique node name), so an unrelated Gazebo or ROS process on the same machine is never touched.
CLEANUP_PATTERNS = (f'^gz sim .*{ROOT}/runtime/', '^/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge .*__node:=redshift_gz_bridge',
                    f'^/usr/bin/python3 {ROOT}/ros2_ws/install/')
# a Gazebo-side process dying while the mission is still launching means the simulator will not come up
GAZEBO_DIED = re.compile(r'\[(gz_sim|parameter_bridge|world_interface_node)-\d+\]:? process has died')  # not the optional GUI
# exec wrapper: background shells start us with SIGINT ignored, and that would be inherited by the launch tree
SIG_RESET = ('import os, signal, sys\n'
             'for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s, signal.SIG_DFL)\n'
             'os.execvp(sys.argv[1], sys.argv[1:])')

LogFn = Callable[[str], Awaitable[None]]
StatusFn = Callable[[], Awaitable[None]]
EventFn = Callable[[dict], Awaitable[None]]


def checklist_keys(cfg: MissionConfig) -> list[str]:
    world = ['gazebo', 'bridge'] if cfg.sim_backend == 'gazebo' else ['mock_world']
    return world + ['sensor_effects', 'perception', 'mapping'] + cfg.robots


def readiness_from_marks(backend: str, robots: dict[str, dict], first_global: dict) -> set:
    """Checklist keys satisfied by first-arrival marks (robot -> {kind: t}, plus fleet-wide kinds) as RosLink records
    them. sensor_effects needs a clean camera frame from every robot: its /world/env_state timer runs whether or not
    the simulator renders anything, so env_state is no evidence that sensor data flows."""
    marks = list(robots.values())
    ready = set()
    if any('odom' in m for m in marks):
        ready |= {'gazebo', 'bridge'} if backend == 'gazebo' else {'mock_world'}
    if marks and all('frame' in m for m in marks):
        ready.add('sensor_effects')
    if marks and all('detections' in m for m in marks):
        ready.add('perception')
    if 'stats' in first_global:
        ready.add('mapping')
    ready |= {r for r, m in robots.items() if 'agent_state' in m and 'odom' in m}
    return ready


def tidy_log_line(raw: str) -> str | None:
    """Condense a ros2-launch output line for the boot terminal; None = drop."""
    line = ANSI.sub('', raw).rstrip()
    if not line.strip():
        return None
    m = ROS_LINE.match(line)
    if m:
        proc, level, node, msg = m.groups()
        tag = node or proc
        if 'RuntimeWarning' in msg or msg.strip().startswith(('pts =', '^')):
            return None
        return f'[{tag}] {"" if level in (None, "INFO") else level + " "}{msg}'
    if line.startswith('[INFO] [launch]') or line.startswith('[INFO] [') and 'process started' in line:
        return line.replace('[INFO] ', '', 1)
    if line.startswith('[ERROR]') or line.startswith('[WARN') or 'process has died' in line:
        return line
    return line if len(line) < 220 else line[:217] + '...'


def default_launch_cmd(config_path: Path) -> list[str]:
    return ['ros2', 'launch', 'redshift_sim', 'mission.launch.py', f'config:={config_path}']


class MissionManager:
    def __init__(self, runtime_dir: Path, on_status: StatusFn, on_log: LogFn, on_event: EventFn,
                 readiness: Callable[[MissionConfig], set] | None = None,
                 prepare: Callable[[MissionConfig], None] | None = None,
                 launch_cmd: Callable[[Path], list[str]] = default_launch_cmd,
                 gazebo_timeout_s: float = GAZEBO_TIMEOUT_S, ready_timeout_s: float = READY_TIMEOUT_S,
                 cleanup_patterns=CLEANUP_PATTERNS):
        self.runtime = Path(runtime_dir)
        self.on_status, self.on_log, self.on_event = on_status, on_log, on_event
        self.readiness = readiness or (lambda cfg: set())
        self.prepare = prepare or (lambda cfg: None)   # e.g. re-subscribe to the new fleet, once the old one is gone
        self.launch_cmd = launch_cmd
        self.gazebo_timeout_s = gazebo_timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.cleanup_patterns = cleanup_patterns
        self.status = 'idle'
        self.detail: str | None = None
        self.config: MissionConfig | None = None
        self.checklist: dict[str, bool] = {}
        self.started_at: float | None = None
        self.running_since: float | None = None
        self.fallback_from: str | None = None
        self.proc: asyncio.subprocess.Process | None = None
        self._pump_task: asyncio.Task | None = None
        self._supervise_task: asyncio.Task | None = None
        self._fallback_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._expect_exit = False
        self._gen = 0                  # bumped per launch; a queued fallback for an older launch is ignored
        self.tail: list[str] = []

    @property
    def backend(self) -> str | None:
        return self.config.sim_backend if self.config else None

    def status_dict(self) -> dict:
        return {'status': self.status, 'backend': self.backend, 'detail': self.detail, 'checklist': dict(self.checklist),
                'config': self.config.model_dump() if self.config else None}

    async def _set(self, status: str, detail: str | None = None):
        self.status, self.detail = status, detail
        if status == 'running' and self.running_since is None:
            self.running_since = time.time()
        try:
            await self.on_status()
        except Exception:  # noqa: BLE001 - a broken listener must not wedge the lifecycle
            pass

    # ------------------------------------------------------------------ lifecycle
    async def deploy(self, cfg: MissionConfig) -> None:
        async with self._lock:
            await self._stop(silent=True)
            self.fallback_from = None
            await self._start(cfg)

    async def _start(self, cfg: MissionConfig) -> None:
        self._gen += 1
        self.config = cfg
        self.checklist = {k: False for k in checklist_keys(cfg)}
        self.started_at = time.time()
        self.running_since = None
        self.tail = []
        try:
            self.prepare(cfg)
        except Exception as e:  # noqa: BLE001
            await self._set('failed', f'Could not prepare the ROS link for {cfg.robots}: {e}')
            return
        self.runtime.mkdir(parents=True, exist_ok=True)
        cfg_path = self.runtime / 'mission_config.yaml'
        data = cfg.model_dump()
        data['runtime_dir'] = str(self.runtime / 'world')
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))
        await self._set('launching')
        await self.on_log(f'[mission] {cfg_path.relative_to(ROOT) if cfg_path.is_relative_to(ROOT) else cfg_path}: '
                          f'sector={cfg.sector} g={cfg.gravity_g:.1f} fleet={cfg.robots} edge={cfg.edge_profile} '
                          f'backend={cfg.sim_backend} seed={cfg.seed}')
        cmd = self.launch_cmd(cfg_path)
        await self.on_log('[mission] $ ' + ' '.join(cmd))
        self._expect_exit = False
        try:   # own session (= process group) so abort can signal the whole launch tree; no preexec_fn (threads)
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable, '-c', SIG_RESET, *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=str(ROOT), start_new_session=True, limit=PIPE_LIMIT,
                env=dict(os.environ, PYTHONUNBUFFERED='1'))
        except OSError as e:
            await self._set('failed', f'Could not start launch: {e}')
            return
        self._pump_task = asyncio.create_task(self._pump(self.proc, cfg, self._gen))
        self._supervise_task = asyncio.create_task(self._supervise(cfg, self._gen))

    async def abort(self) -> None:
        async with self._lock:
            await self._stop(silent=False)

    async def _stop(self, silent: bool) -> None:
        was_active = self.status in ('launching', 'running', 'degraded')
        self._expect_exit = True
        self._gen += 1                 # invalidates any fallback still queued on the lock
        await _cancel(self._supervise_task)
        self._supervise_task = None
        if self.proc is not None:      # the pump keeps draining stdout meanwhile, so a chatty shutdown can't block
            await self._terminate(self.proc)
            self.proc = None
        if self._pump_task is not None and self._pump_task is not asyncio.current_task():
            try:
                await asyncio.wait_for(self._pump_task, 3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                await _cancel(self._pump_task)
        self._pump_task = None
        await self._reap_leftovers()
        if not silent and was_active:
            await self.on_log('[mission] all mission processes stopped')
            await self._set('stopped', 'Mission aborted by operator')

    async def _terminate(self, proc) -> None:
        if proc.returncode is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        for sig, wait_s in ((signal.SIGINT, 8.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(proc.wait(), wait_s)
                break
            except asyncio.TimeoutError:
                continue
        try:   # children may outlive the launch process; finish the group
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _reap_leftovers(self) -> None:
        for pat in self.cleanup_patterns:
            p = await asyncio.create_subprocess_exec('pkill', '-KILL', '-f', pat, stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.DEVNULL)
            await p.wait()

    async def _log(self, line: str) -> None:
        try:
            await self.on_log(line)
        except Exception:  # noqa: BLE001
            pass

    def _schedule_fallback(self, cfg: MissionConfig, gen: int, reason: str) -> None:
        if self._fallback_task is None or self._fallback_task.done():
            self._fallback_task = asyncio.create_task(self._fallback(cfg, gen, reason))

    # ------------------------------------------------------------------ background tasks
    async def _pump(self, proc, cfg: MissionConfig, gen: int) -> None:
        assert proc.stdout is not None
        while True:
            try:
                raw = await proc.stdout.readline()
            except ValueError:         # a single line longer than PIPE_LIMIT: asyncio drops it, keep reading
                continue
            if not raw:
                break
            line = tidy_log_line(raw.decode('utf-8', 'replace'))
            if line is None:
                continue
            self.tail = (self.tail + [line])[-40:]
            if self._expect_exit:
                continue
            if self.status == 'launching' or 'ERROR' in line or 'process has died' in line or 'WARN' in line:
                await self._log(line)
            if cfg.sim_backend == 'gazebo' and self.status == 'launching' and GAZEBO_DIED.search(line):
                self._schedule_fallback(cfg, gen, f'Gazebo process died during start-up ({line[:120]})')
            elif 'process has died' in line and self.status == 'running':
                await self._set('degraded', line[:200])
        rc = await proc.wait()
        if self._expect_exit or gen != self._gen or self.status not in ('launching', 'running', 'degraded'):
            return
        last = next((ln for ln in reversed(self.tail) if 'ERROR' in ln or 'died' in ln), self.tail[-1] if self.tail else '')
        if cfg.sim_backend == 'gazebo' and self.status == 'launching':
            self._schedule_fallback(cfg, gen, f'Gazebo launch exited with code {rc} during start-up: {last[:120]}')
        else:
            await self._set('failed', f'launch exited with code {rc}: {last[:200]}')

    async def _supervise(self, cfg: MissionConfig, gen: int) -> None:
        t0 = time.monotonic()
        not_ready = False              # degraded only because a subsystem was late (recoverable)
        beat = t0
        while True:
            await asyncio.sleep(0.25)
            try:
                ready = self.readiness(cfg)
            except Exception:  # noqa: BLE001
                ready = set()
            changed = False
            for k in self.checklist:
                if not self.checklist[k] and k in ready:
                    self.checklist[k] = True
                    changed = True
                    await self._log(f'[mission] ✓ {k} ready ({time.monotonic() - t0:.1f} s)')
            if changed and (self.status == 'launching' or (self.status == 'degraded' and not_ready)):
                if all(self.checklist.values()):
                    await self._log('[mission] all subsystems ready, handing control to the agents')
                    not_ready = False
                    await self._set('running')
                else:
                    await self._set(self.status, self.detail)
            if self.status == 'launching' and gen == self._gen:
                waited = time.monotonic() - t0
                if cfg.sim_backend == 'gazebo' and waited > self.gazebo_timeout_s and not self.checklist.get('sensor_effects'):
                    self._schedule_fallback(cfg, gen, f'Gazebo produced no camera frames within {self.gazebo_timeout_s:.0f} s')
                    return
                if waited > self.ready_timeout_s:
                    missing = [k for k, v in self.checklist.items() if not v]
                    not_ready = True
                    await self._set('degraded', f'Not ready after {self.ready_timeout_s:.0f} s: {", ".join(missing)}')
                elif not changed and time.monotonic() - beat >= 1.0:
                    beat = time.monotonic()
                    await self._set(self.status, self.detail)    # 1 Hz status heartbeat while launching

    async def _fallback(self, cfg: MissionConfig, gen: int, reason: str) -> None:
        async with self._lock:
            if gen != self._gen or self.status != 'launching':
                return                 # aborted or redeployed while this was queued
            await self._log(f'[watchdog] {reason}: falling back to mock physics/sensors')
            await self._stop(silent=True)
            self.fallback_from = 'gazebo'
            await self.on_event({'op': 'fallback', 'ok': True, 'from': 'gazebo', 'to': 'mock', 'reason': reason,
                                 't': time.time()})
            await self._start(cfg.model_copy(update={'sim_backend': 'mock'}))


async def _cancel(task: asyncio.Task | None) -> None:
    if task is None or task.done() or task is asyncio.current_task():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
