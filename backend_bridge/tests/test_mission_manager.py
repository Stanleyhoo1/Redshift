import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

from backend_bridge.config_schema import MissionConfig
from backend_bridge.mission_manager import MissionManager, checklist_keys, readiness_from_marks, tidy_log_line

# the manager's exec wrapper keeps the pid, so the parent is the test process: when that dies (pytest killed) we are
# re-parented and exit on our own instead of surviving in our own session; 120 s caps a launch nobody aborted
FAKE = r'''
import os, sys, time, signal
ppid, t_end = os.getppid(), time.monotonic() + 120
print("[INFO] [launch]: All log files can be found below /tmp", flush=True)
print("[agent_planner_node-8] [INFO] [1794064186.07] [rover_0_planner]: rover_0: θmax 16.9°", flush=True)
mode = sys.argv[2]
if mode == "crash":
    print("[ERROR] [mapping_coordinator_node-6]: process has died [pid 1, exit code 1]", flush=True)
    sys.exit(3)
while os.getppid() == ppid and time.monotonic() < t_end:
    time.sleep(0.05)
'''


class Rec:
    def __init__(self):
        self.logs, self.events, self.statuses = [], [], []


def make(tmp_path, mode='run', ready=lambda cfg: set(checklist_keys(cfg)), **kw):
    rec = Rec()

    async def on_status():
        rec.statuses.append(mgr.status)

    async def on_log(line):
        rec.logs.append(line)

    async def on_event(ev):
        rec.events.append(ev)

    mgr = MissionManager(tmp_path / 'runtime', on_status, on_log, on_event, readiness=ready,
                         launch_cmd=lambda p: [sys.executable, '-c', FAKE, str(p), mode], cleanup_patterns=(), **kw)
    return mgr, rec


def proc_stat(pid: int) -> list | None:
    """/proc/<pid>/stat fields after the command name ([state, ppid, ...]); None once the process is reaped."""
    try:
        with open(f'/proc/{pid}/stat') as f:
            return f.read().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None


def gone(pid: int) -> bool:
    st = proc_stat(pid)
    return st is None or st[0] == 'Z'      # a zombie has exited; its new parent just has not reaped it yet


@contextlib.asynccontextmanager
async def aborting(mgr):
    """Abort however the body ends (failed assert, wait_for timeout, cancel), so no fake launch outlives the test."""
    try:
        yield mgr
    finally:
        await mgr.abort()


async def wait_for(pred, timeout=5.0):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            raise AssertionError('timeout')
        await asyncio.sleep(0.02)


def test_deploy_reaches_running_then_abort_kills_group(tmp_path):
    async def go():
        mgr, rec = make(tmp_path)
        async with aborting(mgr):
            await mgr.deploy(MissionConfig(sim_backend='mock'))
            cfg = yaml.safe_load((tmp_path / 'runtime' / 'mission_config.yaml').read_text())
            assert cfg['sector'] == 'medium' and cfg['runtime_dir'].endswith('runtime/world')
            await wait_for(lambda: mgr.status == 'running')
            assert all(mgr.checklist.values()) and 'launching' in rec.statuses
            assert any('[rover_0_planner] rover_0: θmax 16.9°' in ln for ln in rec.logs)
            pid = mgr.proc.pid
            assert int(proc_stat(pid)[1]) == os.getpid()      # what FAKE's orphan check relies on
            await mgr.abort()
            assert mgr.status == 'stopped'
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    asyncio.run(go())


def test_unexpected_exit_marks_failed(tmp_path):
    async def go():
        mgr, rec = make(tmp_path, mode='crash', ready=lambda cfg: set())
        async with aborting(mgr):
            await mgr.deploy(MissionConfig(sim_backend='mock'))
            await wait_for(lambda: mgr.status == 'failed')
            assert 'code 3' in mgr.detail and 'process has died' in mgr.detail
    asyncio.run(go())


def test_gazebo_watchdog_falls_back_to_mock(tmp_path):
    def ready(cfg):   # Gazebo never produces sensor data; the mock world comes up immediately
        return set(checklist_keys(cfg)) if cfg.sim_backend == 'mock' else {'gazebo'}

    async def go():
        mgr, rec = make(tmp_path, ready=ready, gazebo_timeout_s=0.4)
        async with aborting(mgr):
            await mgr.deploy(MissionConfig(sim_backend='gazebo'))
            await wait_for(lambda: mgr.status == 'running' and mgr.backend == 'mock')
            assert mgr.fallback_from == 'gazebo'
            assert rec.events and rec.events[0]['op'] == 'fallback' and rec.events[0]['to'] == 'mock'
            assert any('falling back to mock' in ln for ln in rec.logs)
            assert 'mock_world' in mgr.checklist and 'gazebo' not in mgr.checklist
    asyncio.run(go())


def gazebo_without_rendering(cfg):
    """What RosLink sees when gz-sim runs but its rendering sensors never produce frames: odom/IMU flow, the
    sensor-effects env timer publishes, planners report state - but no camera frame ever arrives."""
    if cfg.sim_backend == 'mock':
        return set(checklist_keys(cfg))
    marks = {r: {'odom': 1.0, 'agent_state': 1.0} for r in cfg.robots}
    return readiness_from_marks('gazebo', marks, {'env': 1.0})


def test_gazebo_watchdog_fires_when_only_env_and_odom_arrive(tmp_path):
    async def go():
        mgr, rec = make(tmp_path, ready=gazebo_without_rendering, gazebo_timeout_s=0.4)
        async with aborting(mgr):
            await mgr.deploy(MissionConfig(sim_backend='gazebo'))
            await wait_for(lambda: mgr.status == 'running' and mgr.backend == 'mock')
            assert rec.events and rec.events[0]['op'] == 'fallback'
    asyncio.run(go())


def test_readiness_from_marks():
    robots = {'rover_0': {'odom': 1, 'agent_state': 1}, 'scout_0': {'odom': 1}}
    assert readiness_from_marks('gazebo', robots, {'env': 1}) == {'gazebo', 'bridge', 'rover_0'}
    robots['rover_0']['frame'] = 1
    assert 'sensor_effects' not in readiness_from_marks('gazebo', robots, {})      # every robot needs a frame
    robots['scout_0'].update(frame=1, detections=1, agent_state=1)
    robots['rover_0']['detections'] = 1
    assert readiness_from_marks('mock', robots, {'stats': 1}) == \
        {'mock_world', 'sensor_effects', 'perception', 'mapping', 'rover_0', 'scout_0'}
    assert readiness_from_marks('mock', {}, {}) == set()


def test_gazebo_launch_exit_while_launching_falls_back(tmp_path):
    def ready(cfg):
        return set(checklist_keys(cfg)) if cfg.sim_backend == 'mock' else set()

    async def go():
        rec = Rec()

        async def noop(*_):
            pass

        async def on_event(ev):
            rec.events.append(ev)

        mgr = MissionManager(tmp_path / 'runtime', noop, noop, on_event, readiness=ready, cleanup_patterns=(),
                             launch_cmd=lambda p: [sys.executable, '-c', FAKE, str(p), 'crash' if 'sim_backend: gazebo' in
                                                   p.read_text() else 'run'], gazebo_timeout_s=30)
        async with aborting(mgr):
            await mgr.deploy(MissionConfig(sim_backend='gazebo'))
            await wait_for(lambda: mgr.status == 'running' and mgr.backend == 'mock')
            assert rec.events[0]['op'] == 'fallback' and 'exited' in rec.events[0]['reason']
    asyncio.run(go())


def test_abort_during_pending_fallback_does_not_start_mock(tmp_path):
    async def go():
        mgr, rec = make(tmp_path, ready=gazebo_without_rendering, gazebo_timeout_s=0.3)
        async with aborting(mgr):
            await mgr.deploy(MissionConfig(sim_backend='gazebo'))
            async with mgr._lock:                  # hold the lock so the watchdog's fallback has to queue behind it
                await wait_for(lambda: mgr._fallback_task is not None)
                await mgr._stop(silent=False)
            await asyncio.sleep(0.5)
            assert mgr.status == 'stopped' and mgr.backend == 'gazebo' and mgr.proc is None
    asyncio.run(go())


def test_degraded_recovers_when_late_subsystem_arrives(tmp_path):
    late = {'ok': False}

    def ready(cfg):
        keys = set(checklist_keys(cfg))
        return keys if late['ok'] else keys - {'mapping'}

    async def go():
        mgr, rec = make(tmp_path, ready=ready, ready_timeout_s=0.3)
        async with aborting(mgr):
            await mgr.deploy(MissionConfig(sim_backend='mock'))
            await wait_for(lambda: mgr.status == 'degraded')
            late['ok'] = True
            await wait_for(lambda: mgr.status == 'running')
            assert mgr.running_since is not None and mgr.detail is None
    asyncio.run(go())


def test_overlong_output_line_does_not_kill_pump(tmp_path):
    long_fake = FAKE.replace('mode = sys.argv[2]', 'print("x" * 300000, flush=True)\nprint("[mission-test] after", flush=True)\nmode = sys.argv[2]')

    async def go():
        rec = Rec()

        async def noop(*_):
            pass

        async def on_log(line):
            rec.logs.append(line)

        mgr = MissionManager(tmp_path / 'runtime', noop, on_log, noop, readiness=lambda cfg: set(),
                             launch_cmd=lambda p: [sys.executable, '-c', long_fake, str(p), 'run'], cleanup_patterns=())
        async with aborting(mgr):
            await mgr.deploy(MissionConfig(sim_backend='mock'))
            await wait_for(lambda: any('after' in ln for ln in rec.logs))
            assert mgr.status == 'launching'
    asyncio.run(go())


ORPHAN = r'''
import subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[2])
with out.open("wb") as f:      # start the fake launch in its own session as the manager does, wait till it is up, go
    p = subprocess.Popen([sys.executable, "-c", sys.argv[1], "cfg", "run"], stdin=subprocess.DEVNULL, stdout=f,
                         stderr=subprocess.STDOUT, start_new_session=True)
while out.read_bytes().count(b"\n") < 2 and p.poll() is None:
    time.sleep(0.01)
print(p.pid)
'''


def test_fake_launch_exits_when_orphaned(tmp_path):
    """A killed test run (no finally, no abort) must not leave the fake launch behind for hours."""
    pid = int(subprocess.run([sys.executable, '-c', ORPHAN, FAKE, str(tmp_path / 'fake.out')], stdout=subprocess.PIPE,
                             text=True, check=True, timeout=10).stdout)
    try:
        t0 = time.monotonic()
        while not gone(pid) and time.monotonic() - t0 < 3.0:
            time.sleep(0.02)
        assert gone(pid), 'fake launch outlived its parent'
    finally:
        if not gone(pid):
            os.kill(pid, signal.SIGKILL)


def test_tidy_log_line():
    assert tidy_log_line('[agent_planner_node-8] [INFO] [1794064186.077] [rover_0_planner]: [EXPLORING] go\n') == \
        '[rover_0_planner] [EXPLORING] go'
    assert tidy_log_line('[perception_node-7] [WARN] [1.0] [rover_0_perception]: slow').startswith(
        '[rover_0_perception] WARN slow')
    assert tidy_log_line('   \n') is None
    assert tidy_log_line('\x1b[1;33m[gz_sim-1] warning\x1b[0m') == '[gz_sim] warning'
