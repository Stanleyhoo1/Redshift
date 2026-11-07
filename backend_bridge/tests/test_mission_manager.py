import asyncio
import os
import sys
import time

import pytest
import yaml

from backend_bridge.config_schema import MissionConfig
from backend_bridge.mission_manager import MissionManager, checklist_keys, tidy_log_line

FAKE = r'''
import sys, time, signal
print("[INFO] [launch]: All log files can be found below /tmp", flush=True)
print("[agent_planner_node-8] [INFO] [1794064186.07] [rover_0_planner]: rover_0: θmax 16.9°", flush=True)
mode = sys.argv[2]
if mode == "crash":
    print("[ERROR] [mapping_coordinator_node-6]: process has died [pid 1, exit code 1]", flush=True)
    sys.exit(3)
while True:
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


async def wait_for(pred, timeout=5.0):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            raise AssertionError('timeout')
        await asyncio.sleep(0.02)


def test_deploy_reaches_running_then_abort_kills_group(tmp_path):
    async def go():
        mgr, rec = make(tmp_path)
        await mgr.deploy(MissionConfig(sim_backend='mock'))
        cfg = yaml.safe_load((tmp_path / 'runtime' / 'mission_config.yaml').read_text())
        assert cfg['sector'] == 'medium' and cfg['runtime_dir'].endswith('runtime/world')
        await wait_for(lambda: mgr.status == 'running')
        assert all(mgr.checklist.values()) and 'launching' in rec.statuses
        assert any('[rover_0_planner] rover_0: θmax 16.9°' in ln for ln in rec.logs)
        pid = mgr.proc.pid
        await mgr.abort()
        assert mgr.status == 'stopped'
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    asyncio.run(go())


def test_unexpected_exit_marks_failed(tmp_path):
    async def go():
        mgr, rec = make(tmp_path, mode='crash', ready=lambda cfg: set())
        await mgr.deploy(MissionConfig(sim_backend='mock'))
        await wait_for(lambda: mgr.status == 'failed')
        assert 'code 3' in mgr.detail and 'process has died' in mgr.detail
        await mgr.abort()
    asyncio.run(go())


def test_gazebo_watchdog_falls_back_to_mock(tmp_path):
    def ready(cfg):   # Gazebo never produces sensor data; the mock world comes up immediately
        return set(checklist_keys(cfg)) if cfg.sim_backend == 'mock' else {'gazebo'}

    async def go():
        mgr, rec = make(tmp_path, ready=ready, gazebo_timeout_s=0.4)
        await mgr.deploy(MissionConfig(sim_backend='gazebo'))
        await wait_for(lambda: mgr.status == 'running' and mgr.backend == 'mock')
        assert mgr.fallback_from == 'gazebo'
        assert rec.events and rec.events[0]['op'] == 'fallback' and rec.events[0]['to'] == 'mock'
        assert any('falling back to mock' in ln for ln in rec.logs)
        assert 'mock_world' in mgr.checklist and 'gazebo' not in mgr.checklist
        await mgr.abort()
    asyncio.run(go())


def test_tidy_log_line():
    assert tidy_log_line('[agent_planner_node-8] [INFO] [1794064186.077] [rover_0_planner]: [EXPLORING] go\n') == \
        '[rover_0_planner] [EXPLORING] go'
    assert tidy_log_line('[perception_node-7] [WARN] [1.0] [rover_0_perception]: slow').startswith(
        '[rover_0_perception] WARN slow')
    assert tidy_log_line('   \n') is None
    assert tidy_log_line('\x1b[1;33m[gz_sim-1] warning\x1b[0m') == '[gz_sim] warning'
