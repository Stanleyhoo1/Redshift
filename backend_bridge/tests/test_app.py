import asyncio
import json
import math

from fastapi.testclient import TestClient

from backend_bridge import app as appmod
from backend_bridge.app import CLIENT_QUEUE, Hub, dumps, origin_ok


def test_dumps_replaces_non_finite_floats():
    out = json.loads(dumps({'a': float('nan'), 'b': [1.0, float('inf'), {'c': -math.inf}], 'd': 'x', 'e': 2}))
    assert out == {'a': None, 'b': [1.0, None, {'c': None}], 'd': 'x', 'e': 2}
    assert 'NaN' not in dumps({'v': float('nan')})


def test_origin_policy():
    assert origin_ok(None, '127.0.0.1:8000')                                   # curl / scripts
    assert origin_ok('http://localhost:5173', '127.0.0.1:8000')                # Vite dev proxy
    assert origin_ok('http://127.0.0.1:8000', '127.0.0.1:8000')
    assert origin_ok('http://192.168.1.20:8000', '192.168.1.20:8000')         # same origin over the LAN
    assert not origin_ok('https://evil.example', '127.0.0.1:8000')
    assert not origin_ok('http://192.168.1.20:8000', '127.0.0.1:8000')


class SlowWS:
    def __init__(self, stall: bool):
        self.stall, self.sent, self.closed = stall, [], None

    async def send_text(self, text):
        if self.stall:
            await asyncio.sleep(3600)
        self.sent.append(text)

    async def close(self, code=1000):
        self.closed = code


def test_slow_client_is_dropped_without_blocking_others():
    async def go():
        hub = Hub()
        fast, slow = SlowWS(False), SlowWS(True)
        cf, cs = hub.attach(fast, 'rover_0'), hub.attach(slow, 'rover_0')
        for i in range(CLIENT_QUEUE + 5):           # broadcast never awaits a client
            hub.broadcast({'type': 'telemetry', 'i': i})
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        assert slow not in hub.clients
        assert cs.task.done()
        assert fast in hub.clients and len(fast.sent) == CLIENT_QUEUE + 5
        await asyncio.sleep(0.05)
        assert slow.closed == 1013
        hub.drop(cf)
    asyncio.run(go())


def test_stalled_send_times_out(monkeypatch):
    monkeypatch.setattr(appmod, 'SEND_TIMEOUT_S', 0.1)

    async def go():
        hub = Hub()
        slow = SlowWS(True)
        hub.attach(slow, 'rover_0')
        hub.broadcast({'type': 'status'})
        await asyncio.sleep(0.4)
        assert slow not in hub.clients and slow.closed == 1013
    asyncio.run(go())


def test_request_validation_and_origin_guard():
    c = TestClient(appmod.app)                      # no lifespan: every request below is refused before the handler
    r = c.post('/api/inject/anomaly', json={'dust_level': 'lots'})
    assert r.status_code == 422 and 'dust_level' in r.json()['detail']
    r = c.post('/api/inject/anomaly', json={'duration_s': 999})
    assert r.status_code == 422 and isinstance(r.json()['detail'], str)
    assert c.post('/api/inject/tip', json=[1, 2]).status_code == 422
    assert c.post('/api/mission/deploy', content=b'{not json', headers={'content-type': 'application/json'}).status_code == 422
    assert c.post('/api/mission/deploy', json={'sector': 'galactic'}).status_code == 422
    r = c.post('/api/mission/abort', headers={'origin': 'https://evil.example'})
    assert r.status_code == 403


class FakeLink:
    def __init__(self):
        self.cmds, self.stats, self.env, self.sim = [], {}, {}, (None, None)

    def publish_command(self, cmd):
        self.cmds.append(cmd)
        return 1

    def sim_clock(self):
        return self.sim


class FakeMgr:
    def __init__(self, status='running'):
        from backend_bridge.config_schema import MissionConfig
        self.status, self.backend, self.running_since = status, 'mock', 1000.0
        self.config = MissionConfig(sim_backend='mock')

    def status_dict(self):
        return {'status': self.status}


def test_sim_speed_endpoint_validates_commands_and_records(monkeypatch):
    link, mgr = FakeLink(), FakeMgr()
    monkeypatch.setattr(appmod.hub, 'link', link)
    monkeypatch.setattr(appmod.hub, 'mgr', mgr)
    c = TestClient(appmod.app)
    assert c.post('/api/sim/speed', json={'speed': 5.0}).status_code == 422
    assert c.post('/api/sim/speed', json={'speed': 'fast'}).status_code == 422
    assert c.post('/api/sim/speed', json={}).status_code == 422
    r = c.post('/api/sim/speed', json={'speed': 2.0}).json()
    assert r['ok'] and r['speed'] == 2.0
    assert link.cmds == [{'op': 'set_speed', 'speed': 2.0}]
    assert appmod.hub.speed == 2.0 and mgr.config.sim_speed == 2.0
    mgr.status = 'launching'
    assert c.post('/api/sim/speed', json={'speed': 1.0}).json()['ok'] is False
    assert len(link.cmds) == 1


def test_mission_elapsed_follows_sim_time_and_relatches_on_restart(monkeypatch):
    link, mgr = FakeLink(), FakeMgr()
    hub = Hub()
    hub.link, hub.mgr = link, mgr
    monkeypatch.setattr(appmod.time, 'time', lambda: 1010.0)
    assert hub.mission_dict()['elapsed_s'] == 10.0          # no /clock yet: wall time since running
    link.sim = (40.0, 2.0)
    d = hub.mission_dict()
    assert d['elapsed_s'] == 0.0 and d['rtf'] == 2.0 and d['sim_time_s'] == 40.0
    link.sim = (52.5, 2.0)
    assert hub.mission_dict()['elapsed_s'] == 12.5          # simulated seconds since it began running
    link.sim = (3.0, 1.0)                                   # world restarted (clock went backwards)
    assert hub.mission_dict()['elapsed_s'] == 0.0
    mgr.running_since = None
    assert hub.mission_dict()['elapsed_s'] == 0.0


def test_mission_clock_latches_when_running_begins_even_without_clients(monkeypatch):
    link, mgr = FakeLink(), FakeMgr()
    hub = Hub()
    hub.link, hub.mgr = link, mgr
    link.sim = (8.0, 1.0)
    asyncio.run(hub.on_status())                            # status → running (no clients connected)
    link.sim = (68.0, 1.0)                                  # first telemetry tick a minute later
    assert hub.mission_dict()['elapsed_s'] == 60.0
