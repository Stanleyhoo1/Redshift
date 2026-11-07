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
