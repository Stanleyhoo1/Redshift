import asyncio
import itertools
import json
import math
import threading
import time

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
        self.op_cmds, self.op_listeners, self.brain = [], 1, None
        self.cmd_listeners = 1

    def publish_command(self, cmd):
        self.cmds.append(cmd)
        return self.cmd_listeners

    def publish_operator_command(self, cmd):
        self.op_cmds.append(cmd)
        return self.op_listeners

    def brain_status(self):
        return self.brain

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


def test_damage_inject_targets_a_running_rover(monkeypatch):
    link, mgr = FakeLink(), FakeMgr()
    monkeypatch.setattr(appmod.hub, 'link', link)
    monkeypatch.setattr(appmod.hub, 'mgr', mgr)
    c = TestClient(appmod.app)
    r = c.post('/api/inject/damage', json={}).json()
    assert r['ok'] and 'Scout Rover 1' in r['detail']
    assert link.cmds == [{'op': 'critical_damage', 'robot': 'rover_0'}]
    assert c.post('/api/inject/damage', json={'robot': 'rover_1'}).json()['ok']
    assert c.post('/api/inject/damage', json={'robot': 'scout_0'}).json()['ok'] is False
    assert len(link.cmds) == 2


def test_camera_fan_out_sends_every_robot_and_depth_for_the_selected(monkeypatch):
    import asyncio
    import threading

    from backend_bridge.encoders import RobotCache

    class CamLink(FakeLink):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.robots = {r: RobotCache(r) for r in ('rover_0', 'rover_1', 'scout_0')}
            self.layers_version, self.layers = 0, None
            for k, c in enumerate(self.robots.values()):
                c.jpeg_b64, c.jpeg_t = f'jpg{k}', 100.0 + k
            self.robots['rover_0'].depth_b64, self.robots['rover_0'].depth_t = 'dep0', 100.0

        def lidar_cloud(self, r):
            return None

    link, mgr = CamLink(), FakeMgr()
    monkeypatch.setattr(appmod.hub, 'link', link)
    monkeypatch.setattr(appmod.hub, 'mgr', mgr)
    sent = []
    cl = appmod.Client.__new__(appmod.Client)
    cl.ws, cl.camera, cl.task, cl.sent_cam, cl.sent_depth = None, 'rover_0', None, {}, None
    monkeypatch.setattr(appmod.hub, 'clients', {1: cl})
    monkeypatch.setattr(appmod.hub, 'send', lambda c, text: sent.append(json.loads(text)))
    monkeypatch.setattr(appmod.hub, 'broadcast', lambda msg: None)
    asyncio.run(appmod.hub._broadcast_tick({'tick': 0, 'map_v': -1}))
    cams = [(m['robot'], m['kind']) for m in sent if m['type'] == 'camera']
    assert set(cams) == {('rover_0', 'rgb'), ('rover_1', 'rgb'), ('scout_0', 'rgb'), ('rover_0', 'depth')}
    sent.clear()
    link.robots['rover_1'].jpeg_t += 0.1          # 10 Hz for a non-selected robot: throttled to ≤ 4 Hz
    link.robots['rover_0'].jpeg_t += 0.1
    asyncio.run(appmod.hub._broadcast_tick({'tick': 1, 'map_v': -1}))
    cams = [(m['robot'], m['kind']) for m in sent if m['type'] == 'camera']
    assert cams == [('rover_0', 'rgb')]


# ---------------------------------------------------------------------------------------------------- advisory brain
FAKE_KEY = 'AIzaFAKE-unit-test-key-0123456789'


def test_options_brain_block_never_leaks_the_key(monkeypatch):
    from redshift_autonomy import gemma_agent_brain as gab
    monkeypatch.setenv(gab.API_KEY_ENV, FAKE_KEY)
    r = TestClient(appmod.app).get('/api/options')
    assert r.status_code == 200 and FAKE_KEY not in r.text
    assert r.json()['brain'] == {'model': 'gemma-4-26b-a4b-it', 'key_present': True, 'default_enabled': True}
    monkeypatch.setattr(gab, 'load_api_key', lambda env_path=None: None)
    assert TestClient(appmod.app).get('/api/options').json()['brain']['key_present'] is False


def test_operator_command_validation_and_409_when_idle(monkeypatch):
    link, mgr = FakeLink(), FakeMgr(status='idle')
    monkeypatch.setattr(appmod.hub, 'link', link)
    monkeypatch.setattr(appmod.hub, 'mgr', mgr)
    c = TestClient(appmod.app)
    r = c.post('/api/command', json={'text': 'slow down to 0.2 m/s'})
    assert r.status_code == 409 and r.json()['ok'] is False
    mgr.status = 'stopped'
    assert c.post('/api/command', json={'text': 'stop'}).status_code == 409
    mgr.status = 'running'
    for bad in ({}, {'text': ''}, {'text': '   '}, {'text': 'x' * 201}, {'text': 42}, {'text': 'stop', 'robot': 7}):
        r = c.post('/api/command', json=bad)
        assert r.status_code == 422 and isinstance(r.json()['detail'], str), bad
    assert c.post('/api/command', json=['stop']).status_code == 422
    r = c.post('/api/command', json={'text': 'stop', 'robot': 'rover_7'})
    assert r.status_code == 422 and 'rover_7' in r.json()['detail']
    assert link.op_cmds == []                                 # nothing invalid reached the brain
    mgr.status = 'launching'
    assert c.post('/api/command', json={'text': 'stop'}).status_code == 409
    monkeypatch.setattr(appmod.hub, 'mgr', None)              # before the lifespan has built a manager
    assert c.post('/api/command', json={'text': 'stop'}).status_code == 409


def test_operator_command_publishes_with_fresh_ids(monkeypatch):
    link, mgr = FakeLink(), FakeMgr()
    hub = Hub()
    hub.link, hub.mgr = link, mgr
    monkeypatch.setattr(appmod, 'hub', hub)
    c = TestClient(appmod.app)
    r1 = c.post('/api/command', json={'text': '  slow down\n to 0.2 m/s ', 'robot': 'rover_0'}).json()
    r2 = c.post('/api/command', json={'text': 'stop', 'robot': None}).json()
    r3 = c.post('/api/command', json={'text': 'climb to 15 m', 'robot': 'scout_0'}).json()
    assert r1 == {'ok': True, 'id': 'cmd-1'} and r2 == {'ok': True, 'id': 'cmd-2'} and r3['id'] == 'cmd-3'
    m1, m2 = link.op_cmds[0], link.op_cmds[1]
    assert set(m1) == {'id', 'text', 'robot', 't'}
    assert m1['id'] == 'cmd-1' and m1['text'] == 'slow down to 0.2 m/s' and m1['robot'] == 'rover_0'
    assert m2['robot'] is None and abs(m2['t'] - time.time()) < 5
    assert json.loads(json.dumps(link.op_cmds)) == link.op_cmds          # plain JSON on the wire
    mgr.status = 'degraded'                                               # still a running mission
    assert c.post('/api/command', json={'text': 'stop'}).json()['ok']
    link.op_listeners = 0                                                 # brain node not up
    r = c.post('/api/command', json={'text': 'stop'}).json()
    assert r['ok'] is False and r['id'] == 'cmd-5' and 'not listening' in r['detail']


def test_brain_decisions_are_pushed_to_every_client_unchanged():
    decision = {'action': 'REJECT_UNSAFE_COMMAND', 'target_speed_mps': 0.0, 'source': 'local-fallback',
                'model': 'local-rules', 'event': 'operator_command', 'latency_ms': None, 'command_id': 'cmd-1',
                'immediate': False, 'robot': 'rover_0', 'projected_note': 'estimate', 'adjusted': ['x'],
                'age_ms': float('nan')}

    async def go():
        hub = Hub()
        hub.loop, hub.queue = asyncio.get_running_loop(), asyncio.Queue()
        a, b = SlowWS(False), SlowWS(False)
        ca, cb = hub.attach(a, 'rover_0'), hub.attach(b, 'scout_0')
        pump = asyncio.create_task(hub.pump_queue())
        t = threading.Thread(target=hub.on_brain, args=(decision,))      # RosLink calls it on the ROS thread
        t.start()
        t.join()
        for _ in range(50):
            if a.sent and b.sent:
                break
            await asyncio.sleep(0.01)
        pump.cancel()
        hub.drop(ca)
        hub.drop(cb)
        return a.sent, b.sent
    sa, sb = asyncio.run(go())
    assert sa == sb and len(sa) == 1
    msg = json.loads(sa[0])
    assert msg['type'] == 'brain' and msg['decision'] == dict(decision, age_ms=None)    # strict JSON for the browser


def test_mission_dict_carries_latest_brain_status():
    link, mgr = FakeLink(), FakeMgr()
    hub = Hub()
    hub.link, hub.mgr = link, mgr
    assert hub.mission_dict()['brain'] is None
    link.brain = {'enabled': True, 'key_present': False, 'model': 'gemma-4-26b-a4b-it', 'device': 'orin_nano_15w',
                  'stats': {'cloud_calls': 0, 'local': 3}, 'backoff_s': 0.0, 'last_error': None, 't': 1.0}
    assert hub.mission_dict()['brain'] == link.brain


def test_ros_link_brain_callbacks_ignore_malformed_messages():
    from types import SimpleNamespace

    from backend_bridge.ros_link import RosLink, parse_json_object
    assert parse_json_object('{"a": 1}') == {'a': 1}
    for bad in ('{not json', '[1, 2]', '3', 'null', None, b'\xff'):
        assert parse_json_object(bad) is None
    link = RosLink.__new__(RosLink)                 # callbacks only: no rclpy node (a live backend may be running)
    link.lock, got, traces = threading.RLock(), [], []
    link.on_brain, link.on_trace = got.append, traces.append
    link._ids = itertools.count(1)
    link.reset()
    assert link.brain_status() is None
    for raw in ('{oops', '["x"]', '{"action": "SLOW_AND_REPLAN", "brain": {"source": "gemma"}}'):
        link._on_brain_decision(SimpleNamespace(data=raw))
        link._on_brain_status(SimpleNamespace(data=raw))
        link._on_trace(SimpleNamespace(data=raw))
    assert got == [{'action': 'SLOW_AND_REPLAN', 'brain': {'source': 'gemma'}}]
    assert link.brain_status() == dict(got[0], stale=False)      # no 't': ages from its arrival, just now
    assert traces == [dict(got[0], id=1)]           # trace entries keep their extra "brain" key
    link.reset()                                    # a new mission starts without the old brain status
    assert link.brain_status() is None


def test_ros_link_brain_status_goes_stale_after_3_s_and_keeps_its_last_fields():
    from types import SimpleNamespace

    from backend_bridge.ros_link import BRAIN_STATUS_STALE_S, RosLink
    assert BRAIN_STATUS_STALE_S == 3.0
    link = RosLink.__new__(RosLink)                 # callbacks only: no rclpy node
    link.lock = threading.RLock()
    link.reset()
    assert link.brain_status(now=1000.0) is None    # null before any status arrives, at any time
    st = {'enabled': True, 'key_present': True, 'model': 'gemma-4-26b-a4b-it', 'device': 'orin_nano_15w',
          'stats': {'cloud_calls': 4, 'stale': 1}, 'backoff_s': 0.0, 'last_error': None, 't': 1000.0}
    link._on_brain_status(SimpleNamespace(data=json.dumps(st)))
    assert link.brain_status(now=1000.5) == dict(st, stale=False)
    assert link.brain_status(now=1003.0)['stale'] is False          # exactly 3 s: still fresh
    old = link.brain_status(now=1003.5)
    assert old == dict(st, stale=True)                               # the last fields are kept, flagged stale
    assert old['stats'] == {'cloud_calls': 4, 'stale': 1}            # AgentBrain's own stale counter untouched
    assert link.brain is not None and 'stale' not in link.brain      # the cache itself is never rewritten
    st2 = dict(st, t=1010.0, stats={'cloud_calls': 5})
    link._on_brain_status(SimpleNamespace(data=json.dumps(st2)))     # the node reports again → fresh
    assert link.brain_status(now=1010.2) == dict(st2, stale=False)
    for bad_t in (None, 'soon', True, float('nan')):                 # no usable 't': ages from its arrival
        link._on_brain_status(SimpleNamespace(data=json.dumps(dict(st, t=bad_t))))
        link._brain_rx = 2000.0
        assert link.brain_status(now=2002.0)['stale'] is False
        assert link.brain_status(now=2003.1)['stale'] is True
    link._on_brain_status(SimpleNamespace(data=json.dumps(st)))      # real wall clock: t=1000.0 is long past
    assert link.brain_status()['stale'] is True
    link._on_brain_status(SimpleNamespace(data=json.dumps(dict(st, t=time.time()))))
    assert link.brain_status()['stale'] is False
    link.reset()
    assert link.brain_status() is None


def test_mission_dict_reports_a_stale_brain_status():
    from types import SimpleNamespace

    from backend_bridge.ros_link import RosLink
    link = RosLink.__new__(RosLink)
    link.lock = threading.RLock()
    link.reset()
    link.stats, link.env = {}, {'dust_level': 0.0, 'seismic': False}
    link.sim_clock = lambda: (None, None)
    hub = Hub()
    hub.link, hub.mgr = link, FakeMgr()
    assert hub.mission_dict()['brain'] is None
    st = {'enabled': True, 'key_present': False, 'model': 'gemma-4-26b-a4b-it', 'device': 'orin_nano_15w',
          'stats': {}, 'backoff_s': 0.0, 'last_error': None}
    link._on_brain_status(SimpleNamespace(data=json.dumps(dict(st, t=time.time()))))
    assert hub.mission_dict()['brain'] == dict(st, t=link.brain['t'], stale=False)
    link._on_brain_status(SimpleNamespace(data=json.dumps(dict(st, t=time.time() - 10.0))))
    m = hub.mission_dict()
    assert m['brain']['stale'] is True and m['brain']['model'] == 'gemma-4-26b-a4b-it'
    json.dumps(m, allow_nan=False)                                   # strict JSON for the browser


def test_ros_link_brain_status_with_a_huge_integer_t_never_breaks_telemetry():
    from types import SimpleNamespace

    from backend_bridge.ros_link import RosLink
    link = RosLink.__new__(RosLink)                 # callbacks only: no rclpy node
    link.lock = threading.RLock()
    link.reset()
    link._on_brain_status(SimpleNamespace(data='{"enabled": true, "t": ' + '9' * 400 + '}'))
    st = link.brain_status()                        # float() of that 't' overflows: it ages from its arrival instead
    assert st is not None and st['stale'] is False and st['enabled'] is True


# ---------------------------------------------------------------------------------------------------- Phase 3
def test_pillar_inject_targets_a_running_rover(monkeypatch):
    link, mgr = FakeLink(), FakeMgr()
    monkeypatch.setattr(appmod.hub, 'link', link)
    monkeypatch.setattr(appmod.hub, 'mgr', mgr)
    c = TestClient(appmod.app)
    r = c.post('/api/inject/pillar', json={}).json()
    assert r['ok'] and set(r) == {'ok', 'detail'} and 'rover_0' in r['detail']
    assert link.cmds == [{'op': 'pillar_collapse', 'robot': 'rover_0'}]
    assert c.post('/api/inject/pillar').json()['ok']                       # no body at all: the first rover
    assert c.post('/api/inject/pillar', json={'robot': 'rover_1'}).json()['ok']
    assert link.cmds[-1] == {'op': 'pillar_collapse', 'robot': 'rover_1'}
    assert c.post('/api/inject/pillar', json={'robot': 'scout_0'}).json() == {'ok': False,
                                                                             'detail': "Unknown rover 'scout_0'"}
    assert c.post('/api/inject/pillar', json=[1, 2]).status_code == 422
    assert len(link.cmds) == 3
    link.cmd_listeners = 0
    r = c.post('/api/inject/pillar', json={}).json()
    assert r == {'ok': False, 'detail': 'World is not listening on /world/command'}
    mgr.status = 'launching'
    assert c.post('/api/inject/pillar', json={}).json() == {'ok': False, 'detail': 'No running mission'}
    assert len(link.cmds) == 4


def test_storm_inject_sends_a_dust_storm_anomaly(monkeypatch):
    link, mgr = FakeLink(), FakeMgr()
    monkeypatch.setattr(appmod.hub, 'link', link)
    monkeypatch.setattr(appmod.hub, 'mgr', mgr)
    c = TestClient(appmod.app)
    r = c.post('/api/inject/storm', json={}).json()
    assert r['ok'] and set(r) == {'ok', 'detail'} and '45 s' in r['detail']
    assert link.cmds == [{'op': 'anomaly', 'kind': 'dust_storm', 'dust_level': 1.0, 'duration_s': 45.0,
                          'seismic_s': 0.0, 'wind_mps': 14.0}]
    assert json.loads(json.dumps(link.cmds)) == link.cmds                   # plain JSON on the wire
    assert c.post('/api/inject/storm').json()['ok']                         # no body: the defaults
    r = c.post('/api/inject/storm', json={'duration_s': 20, 'wind_mps': 10.5}).json()
    assert r['ok'] and link.cmds[-1]['duration_s'] == 20.0 and link.cmds[-1]['wind_mps'] == 10.5
    assert link.cmds[-1]['dust_level'] == 1.0 and link.cmds[-1]['kind'] == 'dust_storm'
    for bad in ({'duration_s': 999}, {'duration_s': 1}, {'wind_mps': -1}, {'wind_mps': 50}, {'wind_mps': 'gale'}):
        r = c.post('/api/inject/storm', json=bad)
        assert r.status_code == 422 and isinstance(r.json()['detail'], str), bad
    assert c.post('/api/inject/storm', json=['x']).status_code == 422
    assert len(link.cmds) == 3                                              # nothing invalid reached the world
    link.cmd_listeners = 0
    assert c.post('/api/inject/storm', json={}).json()['ok'] is False
    mgr.status = 'idle'
    assert c.post('/api/inject/storm', json={}).json() == {'ok': False, 'detail': 'No running mission'}
    assert len(link.cmds) == 4


def test_plain_anomaly_inject_is_unchanged(monkeypatch):
    link, mgr = FakeLink(), FakeMgr()
    monkeypatch.setattr(appmod.hub, 'link', link)
    monkeypatch.setattr(appmod.hub, 'mgr', mgr)
    assert TestClient(appmod.app).post('/api/inject/anomaly', json={}).json()['ok']
    assert link.cmds == [{'op': 'anomaly', 'dust_level': 0.85, 'duration_s': 25.0, 'seismic_s': 4.0}]


def test_mission_env_passes_storm_kind_and_wind_through():
    link, mgr = FakeLink(), FakeMgr()
    hub = Hub()
    hub.link, hub.mgr = link, mgr
    assert hub.mission_dict()['env'] == {'dust_level': 0.0, 'seismic': False, 'kind': None, 'wind_mps': 0.0}
    link.env = {'dust_level': 0.62, 'seismic': False, 'remaining_s': 30.0, 'kind': 'dust_storm', 'wind_mps': 8.7}
    assert hub.mission_dict()['env'] == {'dust_level': 0.62, 'seismic': False, 'kind': 'dust_storm', 'wind_mps': 8.7}


def test_ros_link_env_state_ignores_malformed_messages():
    from types import SimpleNamespace

    from backend_bridge.ros_link import RosLink
    link = RosLink.__new__(RosLink)                 # callbacks only: no rclpy node
    link.lock = threading.RLock()
    link.reset()
    assert link.env == {'dust_level': 0.0, 'seismic': False, 'remaining_s': None, 'kind': None, 'wind_mps': 0.0}
    st = {'dust_level': 0.4, 'seismic': False, 'remaining_s': 40.0, 'kind': 'dust_storm', 'wind_mps': 5.6}
    link._on_env(SimpleNamespace(data=json.dumps(st)))
    assert link.env == st and 'env' in link.first_global
    for bad in ('{oops', '[1, 2]', '3', 'null'):
        link._on_env(SimpleNamespace(data=bad))
        assert link.env == st, bad
    link.reset()                                    # a new mission starts calm
    assert link.env['kind'] is None and link.env['wind_mps'] == 0.0


def test_ros_link_robot_json_cache_ignores_non_objects():
    from types import SimpleNamespace

    from backend_bridge.encoders import RobotCache, robot_telemetry
    from backend_bridge.ros_link import RosLink
    link = RosLink.__new__(RosLink)                 # callbacks only: no rclpy node
    link.lock = threading.RLock()
    c = RobotCache('rover_0')
    link._json(c, 'agent_state', SimpleNamespace(data='{"state": "EXPLORING", "power_save": true}'))
    for bad in ('[1, 2]', '{oops', '7'):
        link._json(c, 'agent_state', SimpleNamespace(data=bad))
        link._json(c, 'detections', SimpleNamespace(data=bad))
    t = robot_telemetry(c)
    assert t['state'] == 'EXPLORING' and t['power_save'] is True and t['detections'] == []
