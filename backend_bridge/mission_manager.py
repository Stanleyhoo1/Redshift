"""Mission lifecycle: writes the runtime config, runs `ros2 launch exoswarm_sim mission.launch.py` in its own
process group, streams its output as the boot log, tracks subsystem readiness, falls back from Gazebo to the mock
world when Gazebo produces no sensor data, and tears everything down on abort."""
from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
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
# leftovers to reap after the process group is gone; anchored so they never match an unrelated shell
CLEANUP_PATTERNS = ('^gz sim', '^/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge',
                    f'^/usr/bin/python3 {ROOT}/ros2_ws/install/')

LogFn = Callable[[str], Awaitable[None]]
StatusFn = Callable[[], Awaitable[None]]
EventFn = Callable[[dict], Awaitable[None]]


def checklist_keys(cfg: MissionConfig) -> list[str]:
    world = ['gazebo', 'bridge'] if cfg.sim_backend == 'gazebo' else ['mock_world']
    return world + ['sensor_effects', 'perception', 'mapping'] + cfg.robots


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
    return ['ros2', 'launch', 'exoswarm_sim', 'mission.launch.py', f'config:={config_path}']


def _child_setup():  # runs in the child between fork and exec
    os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # background shells start us with SIGINT ignored
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


class MissionManager:
    def __init__(self, runtime_dir: Path, on_status: StatusFn, on_log: LogFn, on_event: EventFn,
                 readiness: Callable[[MissionConfig], set] | None = None,
                 launch_cmd: Callable[[Path], list[str]] = default_launch_cmd,
                 gazebo_timeout_s: float = GAZEBO_TIMEOUT_S, ready_timeout_s: float = READY_TIMEOUT_S,
                 cleanup_patterns=CLEANUP_PATTERNS):
        self.runtime = Path(runtime_dir)
        self.on_status, self.on_log, self.on_event = on_status, on_log, on_event
        self.readiness = readiness or (lambda cfg: set())
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
        self._tasks: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._expect_exit = False
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
        await self.on_status()

    # ------------------------------------------------------------------ lifecycle
    async def deploy(self, cfg: MissionConfig) -> None:
        async with self._lock:
            await self._stop(silent=True)
            self.fallback_from = None
            await self._start(cfg)

    async def _start(self, cfg: MissionConfig) -> None:
        self.config = cfg
        self.checklist = {k: False for k in checklist_keys(cfg)}
        self.started_at = time.time()
        self.running_since = None
        self.tail = []
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
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, stdin=subprocess.DEVNULL,
                cwd=str(ROOT), preexec_fn=_child_setup, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        except OSError as e:
            await self._set('failed', f'Could not start launch: {e}')
            return
        self._tasks = [asyncio.create_task(self._pump(self.proc)), asyncio.create_task(self._supervise(cfg))]

    async def abort(self) -> None:
        async with self._lock:
            await self._stop(silent=False)

    async def _stop(self, silent: bool) -> None:
        was_active = self.status in ('launching', 'running', 'degraded')
        self._expect_exit = True
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        if self.proc is not None:
            await self._terminate(self.proc)
            self.proc = None
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

    # ------------------------------------------------------------------ background tasks
    async def _pump(self, proc) -> None:
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = tidy_log_line(raw.decode('utf-8', 'replace'))
            if line is None:
                continue
            self.tail = (self.tail + [line])[-40:]
            if self.status == 'launching' or 'ERROR' in line or 'process has died' in line or 'WARN' in line:
                await self.on_log(line)
            if 'process has died' in line and self.status == 'running':
                await self._set('degraded', line[:200])
        rc = await proc.wait()
        if not self._expect_exit and self.status in ('launching', 'running', 'degraded'):
            last = next((ln for ln in reversed(self.tail) if 'ERROR' in ln or 'died' in ln), self.tail[-1] if self.tail else '')
            await self._set('failed', f'launch exited with code {rc}: {last[:200]}')

    async def _supervise(self, cfg: MissionConfig) -> None:
        t0 = time.monotonic()
        while True:
            await asyncio.sleep(0.25)
            ready = self.readiness(cfg)
            changed = False
            for k in self.checklist:
                if not self.checklist[k] and k in ready:
                    self.checklist[k] = True
                    changed = True
                    await self.on_log(f'[mission] ✓ {k} ready ({time.monotonic() - t0:.1f} s)')
            if changed and self.status == 'launching':
                if all(self.checklist.values()):
                    await self.on_log('[mission] all subsystems ready, handing control to the agents')
                    await self._set('running')
                else:
                    await self.on_status()
            if self.status != 'launching':
                continue
            waited = time.monotonic() - t0
            if cfg.sim_backend == 'gazebo' and waited > self.gazebo_timeout_s and not self.checklist.get('sensor_effects'):
                asyncio.create_task(self._fallback(cfg, f'Gazebo produced no sensor data within {self.gazebo_timeout_s:.0f} s'))
                return
            if waited > self.ready_timeout_s:
                missing = [k for k, v in self.checklist.items() if not v]
                await self._set('degraded', f'Not ready after {self.ready_timeout_s:.0f} s: {", ".join(missing)}')

    async def _fallback(self, cfg: MissionConfig, reason: str) -> None:
        async with self._lock:
            await self.on_log(f'[watchdog] {reason}: falling back to mock physics/sensors')
            await self._stop(silent=True)
            self.fallback_from = 'gazebo'
            await self.on_event({'op': 'fallback', 'ok': True, 'from': 'gazebo', 'to': 'mock', 'reason': reason,
                                 't': time.time()})
            await self._start(cfg.model_copy(update={'sim_backend': 'mock'}))
