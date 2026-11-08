"""Redshift Mission Control backend: REST + one WebSocket (docs/protocol.md §6-7) over the ROS 2 fleet.

    uvicorn backend_bridge.app:app --port 8000      (from the repo root, with ROS + ros2_ws sourced)
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import math
import time
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from . import _paths  # noqa: F401
from .config_schema import MissionConfig, options, physics_preview
from .encoders import lidar_points_body, png_b64, robot_telemetry
from .mission_manager import ROOT, MissionManager
from .ros_link import HAVE_ROS, RosLink
from redshift_autonomy.agent_fsm import display_name
from redshift_sim.robot_models import ROVER_SPEC, SCOUT_SPEC

log = logging.getLogger('redshift')
DIST = ROOT / 'web_ui' / 'dist'
RUNTIME = ROOT / 'runtime'
TELEMETRY_HZ = 15.0
LIDAR_HZ = 5.0
OTHER_CAM_HZ = 4.0          # camera tiles of the robots a client is not focused on
CLIENT_QUEUE = 240        # ≈ 7 s of outbound frames; a client further behind than that is disconnected
SEND_TIMEOUT_S = 2.0
LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1'}


def _finite(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


def dumps(msg: dict) -> str:
    """JSON for the browser. JSON.parse rejects NaN/Infinity (and ws.ts would drop the whole frame), so
    non-finite floats become null."""
    try:
        return json.dumps(msg, allow_nan=False)
    except ValueError:
        return json.dumps(_finite(msg), allow_nan=False)


def origin_ok(origin: str | None, host: str | None) -> bool:
    """Browsers attach Origin to cross-site requests: accept none (curl, scripts), a localhost page (also the Vite
    dev server, whose proxy keeps the browser's Origin), or the same origin as this server. Anything else is a
    foreign web page trying to drive the fleet."""
    if not origin:
        return True
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return (parts.hostname or '') in LOCAL_HOSTS or (host is not None and parts.netloc == host)


class Client:
    __slots__ = ('ws', 'camera', 'queue', 'task', 'sent_cam', 'sent_depth')

    def __init__(self, ws: WebSocket, camera: str):
        self.ws, self.camera = ws, camera
        self.queue: asyncio.Queue[str] = asyncio.Queue(CLIENT_QUEUE)
        self.task: asyncio.Task | None = None
        self.sent_cam: dict[str, float] = {}       # robot → timestamp of the last rgb frame sent
        self.sent_depth: float | None = None


class Hub:
    def __init__(self):
        self.clients: dict[WebSocket, Client] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None
        self.link: RosLink | None = None
        self.mgr: MissionManager | None = None
        self.link_error: str | None = None
        self._bg: set[asyncio.Task] = set()
        self.speed = 1.0                     # target sim speed (real-time factor) of the current mission
        self._sim0: tuple[float | None, float | None] = (None, None)   # (running_since, sim time when it began)
        self._cmd_ids = itertools.count(1)   # operator command ids "cmd-<n>", unique for this backend process

    # ------------------------------------------------------------------ outbound (never blocks the caller)
    def attach(self, ws: WebSocket, camera: str) -> Client:
        cl = Client(ws, camera)
        self.clients[ws] = cl
        cl.task = asyncio.create_task(self._writer(cl))
        return cl

    def drop(self, cl: Client, reason: str | None = None):
        if self.clients.get(cl.ws) is cl:
            del self.clients[cl.ws]
        if cl.task is not None and cl.task is not asyncio.current_task():
            cl.task.cancel()
        if reason:                       # we are dropping it (not a disconnect): close so its receive loop ends
            log.warning('dropping WebSocket client: %s', reason)
            self._spawn(self._close(cl.ws))

    async def _close(self, ws: WebSocket):
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ws.close(code=1013), SEND_TIMEOUT_S)

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _writer(self, cl: Client):
        try:
            while True:
                text = await cl.queue.get()
                await asyncio.wait_for(cl.ws.send_text(text), SEND_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self.drop(cl, f'send stalled for {SEND_TIMEOUT_S:.0f} s')
        except Exception:  # noqa: BLE001 - client went away
            self.drop(cl)

    def send(self, cl: Client, text: str):
        try:
            cl.queue.put_nowait(text)
        except asyncio.QueueFull:
            self.drop(cl, f'{CLIENT_QUEUE} messages behind')

    def broadcast(self, msg: dict):
        if not self.clients:
            return
        text = dumps(msg)
        for cl in list(self.clients.values()):
            self.send(cl, text)

    def status_msg(self) -> dict:
        return dict(type='status', **self.mgr.status_dict())

    async def on_status(self):
        m = self.mgr
        if m.status == 'running' and self.link and self._sim0[0] != m.running_since:
            self._sim0 = (m.running_since, self.link.sim_clock()[0])    # mission clock starts now, in sim time
        self.broadcast(self.status_msg())

    async def on_log(self, line: str):
        log.info(line)
        self.broadcast({'type': 'boot_log', 'line': line, 't': round(time.time(), 3)})

    async def on_event(self, ev: dict):
        self.broadcast({'type': 'event', 'event': ev})

    def prepare(self, cfg: MissionConfig):
        """Called by the mission manager after the previous mission is gone and before the new one launches, so no
        stale message from the old fleet lands in (and falsely 'readies') the new caches."""
        self.speed = cfg.sim_speed
        self._sim0 = (None, None)
        if self.link:
            self.link.set_fleet(cfg.robots, self._specs(cfg))

    def from_ros(self, msg: dict):
        """Called on the ROS thread: hand the message to the asyncio loop."""
        if self.loop and self.queue:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    def on_brain(self, decision: dict):
        """A /brain/decision (ROS thread): pushed to every client at once, unchanged (docs/protocol.md §9.8)."""
        self.from_ros({'type': 'brain', 'decision': decision})

    def next_command_id(self) -> str:
        return f'cmd-{next(self._cmd_ids)}'

    @staticmethod
    def _specs(cfg: MissionConfig) -> dict:
        return {r: (ROVER_SPEC if r.startswith('rover') else SCOUT_SPEC) for r in cfg.robots}

    # ------------------------------------------------------------------ loops
    async def pump_queue(self):
        while True:
            msg = await self.queue.get()
            try:
                self.broadcast(msg)
            except Exception:  # noqa: BLE001
                log.exception('failed to forward %s', msg.get('type'))

    def mission_elapsed(self, sim_t: float | None) -> float:
        """Mission time since 'running', in simulated seconds (wall seconds until /clock arrives)."""
        since = self.mgr.running_since
        if not since:
            return 0.0
        if sim_t is not None and (self._sim0[0] != since or self._sim0[1] is None or sim_t < self._sim0[1]):
            self._sim0 = (since, sim_t)          # latch the sim time at which this mission began running
        if sim_t is None or self._sim0[1] is None:
            return time.time() - since
        return sim_t - self._sim0[1]

    def mission_dict(self) -> dict:
        st = self.link.stats if self.link else {}
        env = self.link.env if self.link else {}
        sim_t, rtf = self.link.sim_clock() if self.link else (None, None)
        m = self.mgr
        return {'status': m.status, 'backend': m.backend,
                'elapsed_s': round(self.mission_elapsed(sim_t), 1),
                'sim_time_s': None if sim_t is None else round(sim_t, 2),
                'sim_speed': self.speed, 'rtf': None if rtf is None else round(rtf, 2),
                'percent_mapped': float(st.get('percent', 0.0)), 'science_found': len(st.get('science', [])),
                'hazard_cells': int(st.get('hazard_cells', 0)),
                'env': {'dust_level': float(env.get('dust_level', 0.0)), 'seismic': bool(env.get('seismic', False))},
                'brain': self.link.brain_status() if self.link else None,
                'config': m.config.model_dump() if m.config else None}

    async def broadcaster(self):
        period = 1.0 / TELEMETRY_HZ
        state = {'tick': 0, 'map_v': -1}
        nxt = time.perf_counter()
        while True:
            nxt = max(nxt + period, time.perf_counter() - period)     # don't burst to catch up after a stall
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
            try:
                await self._broadcast_tick(state)
            except Exception:  # noqa: BLE001 - one bad frame must not end telemetry for the session
                log.exception('telemetry broadcast failed')

    async def _broadcast_tick(self, state: dict):
        state['tick'] += 1
        tick = state['tick']
        link, m = self.link, self.mgr
        if link is None or m.status not in ('running', 'degraded') or not self.clients:
            return
        with link.lock:
            robots = {r: robot_telemetry(c) for r, c in link.robots.items()}
            layers_v, layers = link.layers_version, link.layers
            stats = dict(link.stats)
            cams = {r: (c.jpeg_t, c.jpeg_b64, c.depth_t, c.depth_b64) for r, c in link.robots.items()}
        self.broadcast({'type': 'telemetry', 't': round(time.time(), 3), 'mission': self.mission_dict(),
                        'robots': robots})
        if layers is not None and layers_v != state['map_v'] and tick % 4 == 0:
            state['map_v'] = layers_v
            meta = {k: stats.get(k) for k in ('size_m', 'res', 'cells', 'origin', 'elev_min', 'elev_max')}
            self.broadcast({'type': 'map', 'png_b64': await asyncio.to_thread(png_b64, layers),
                            'meta': meta, 'stats': stats})
        lidar_cache: dict = {}
        for cl in list(self.clients.values()):
            r = cl.camera
            # every robot's feed: the selected one at full rate (rgb + depth), the others at ≤ OTHER_CAM_HZ
            for rr, (t_jpg, jpg, t_dep, dep) in cams.items():
                last = cl.sent_cam.get(rr)
                if not jpg or last == t_jpg or (rr != r and last is not None and t_jpg - last < 1.0 / OTHER_CAM_HZ):
                    continue
                cl.sent_cam[rr] = t_jpg
                self.send(cl, dumps({'type': 'camera', 'robot': rr, 'kind': 'rgb', 'jpeg_b64': jpg,
                                     't': round(t_jpg, 3)}))
            t_jpg, jpg, t_dep, dep = cams.get(r, (0.0, None, 0.0, None))
            if dep and cl.sent_depth != t_dep:
                cl.sent_depth = t_dep
                self.send(cl, dumps({'type': 'camera', 'robot': r, 'kind': 'depth', 'jpeg_b64': dep,
                                     't': round(t_dep, 3)}))
            if r in robots and r.startswith('rover') and tick % int(TELEMETRY_HZ / LIDAR_HZ) == 0:
                if r not in lidar_cache:
                    cloud = link.lidar_cloud(r)
                    lidar_cache[r] = None if cloud is None else dumps({
                        'type': 'lidar', 'robot': r, 'max_range': 25,
                        'points': lidar_points_body(cloud, ROVER_SPEC['lidar_xyz'], 1500), 't': round(time.time(), 3)})
                if lidar_cache[r]:
                    self.send(cl, lidar_cache[r])

    # ------------------------------------------------------------------ commands
    def running_rover(self, robot) -> tuple[str | None, str | None]:
        m = self.mgr
        if m.status not in ('running', 'degraded') or not m.config:
            return None, 'No running mission'
        rovers = [r for r in m.config.robots if r.startswith('rover')]
        r = robot or rovers[0]
        return (r, None) if isinstance(r, str) and r in rovers else (None, f'Unknown rover {r!r}')

    def command(self, cmd: dict) -> dict:
        if self.link is None:
            return {'ok': False, 'detail': self.link_error or 'ROS link unavailable'}
        n = self.link.publish_command(cmd)
        if n == 0:
            return {'ok': False, 'detail': 'World is not listening on /world/command'}
        return {'ok': True}


class RobotBody(BaseModel):
    model_config = ConfigDict(extra='ignore')
    robot: str | None = None


class SpeedBody(BaseModel):
    model_config = ConfigDict(extra='ignore')
    speed: float = Field(..., ge=0.25, le=3.0)


class AnomalyBody(BaseModel):
    model_config = ConfigDict(extra='ignore')
    dust_level: float = Field(0.85, ge=0.0, le=1.0)
    duration_s: float = Field(25.0, ge=3.0, le=120.0)


# operator text for the advisory brain: 1-200 characters once trimmed; runs of whitespace (newlines, tabs) collapse
# to one space so a pasted multi-line command reads as one line in traces and the prompt
OperatorText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
                         AfterValidator(lambda v: ' '.join(v.split()))]


class CommandBody(BaseModel):
    model_config = ConfigDict(extra='ignore')
    text: OperatorText
    robot: str | None = None


def invalid(e: ValidationError) -> JSONResponse:
    errs = e.errors(include_url=False, include_context=False)
    msg = '; '.join(f"{'.'.join(str(x) for x in er.get('loc', ())) or 'body'}: {er.get('msg')}" for er in errs)
    return JSONResponse({'ok': False, 'detail': msg or 'invalid request', 'errors': errs}, status_code=422)


async def parse_body(request: Request, model: type[BaseModel]):
    raw = await request.body()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JSONResponse({'ok': False, 'detail': 'body must be JSON'}, status_code=422)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return None, JSONResponse({'ok': False, 'detail': 'body must be a JSON object'}, status_code=422)
    try:
        return model(**data), None
    except ValidationError as e:
        return None, invalid(e)


hub = Hub()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    hub.queue = asyncio.Queue()
    if HAVE_ROS:
        try:
            hub.link = RosLink(on_trace=lambda t: hub.from_ros({'type': 'trace', 'entry': t}),
                               on_event=lambda e: hub.from_ros({'type': 'event', 'event': e}),
                               on_brain=hub.on_brain)
        except Exception as e:  # noqa: BLE001
            hub.link_error = f'ROS link failed: {e}'
    else:
        hub.link_error = 'rclpy not importable (source /opt/ros/jazzy/setup.bash and ros2_ws/install/setup.bash)'
    hub.mgr = MissionManager(RUNTIME, hub.on_status, hub.on_log, hub.on_event, prepare=hub.prepare,
                             readiness=lambda cfg: hub.link.readiness(cfg.sim_backend) if hub.link else set())
    tasks = [asyncio.create_task(hub.pump_queue()), asyncio.create_task(hub.broadcaster())]
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await hub.mgr.abort()
        if hub.link:
            hub.link.shutdown()


app = FastAPI(title='Redshift Mission Control', lifespan=lifespan)


@app.middleware('http')
async def same_origin_writes(request: Request, call_next):
    if request.method not in ('GET', 'HEAD', 'OPTIONS') and not origin_ok(request.headers.get('origin'),
                                                                         request.headers.get('host')):
        return JSONResponse({'ok': False, 'detail': 'cross-origin request refused'}, status_code=403)
    return await call_next(request)


@app.get('/api/options')
async def api_options():
    return options()


@app.get('/api/physics_preview')
async def api_physics_preview(g: float = 2.2):
    return physics_preview(g)


@app.post('/api/mission/deploy')
async def api_deploy(request: Request):
    cfg, err = await parse_body(request, MissionConfig)
    if err:
        return err
    if hub.link is None:
        return JSONResponse({'ok': False, 'detail': hub.link_error}, status_code=503)
    await hub.mgr.deploy(cfg)
    return {'ok': hub.mgr.status != 'failed', 'mission_id': f'm{int(hub.mgr.started_at or time.time())}',
            'status': hub.mgr.status, 'detail': hub.mgr.detail}


@app.post('/api/mission/abort')
async def api_abort():
    await hub.mgr.abort()
    return {'ok': True}


@app.get('/api/mission/status')
async def api_status():
    d = hub.mgr.status_dict()
    d['started_at'] = hub.mgr.started_at
    if hub.link_error:
        d['detail'] = d['detail'] or hub.link_error
    return d


@app.post('/api/inject/rockslide')
async def api_rockslide(request: Request):
    body, err = await parse_body(request, RobotBody)
    if err:
        return err
    r, msg = hub.running_rover(body.robot)
    if msg:
        return {'ok': False, 'detail': msg}
    res = hub.command({'op': 'spawn_rockslide', 'robot': r})
    return res if not res['ok'] else {'ok': True, 'detail': f'Rockslide triggered across {r} planned path'}


@app.post('/api/inject/anomaly')
async def api_anomaly(request: Request):
    body, err = await parse_body(request, AnomalyBody)
    if err:
        return err
    if hub.mgr.status not in ('running', 'degraded'):
        return {'ok': False, 'detail': 'No running mission'}
    res = hub.command({'op': 'anomaly', 'dust_level': body.dust_level, 'duration_s': body.duration_s, 'seismic_s': 4.0})
    return res if not res['ok'] else {'ok': True,
                                      'detail': f'Dust {body.dust_level:.2f} for {body.duration_s:.0f} s with a 4 s tremor'}


@app.post('/api/inject/tip')
async def api_tip(request: Request):
    body, err = await parse_body(request, RobotBody)
    if err:
        return err
    r, msg = hub.running_rover(body.robot)
    if msg:
        return {'ok': False, 'detail': msg}
    res = hub.command({'op': 'jolt', 'robot': r})
    return res if not res['ok'] else {'ok': True, 'detail': f'Subsidence jolt applied to {r} chassis'}


@app.post('/api/inject/damage')
async def api_damage(request: Request):
    body, err = await parse_body(request, RobotBody)
    if err:
        return err
    r, msg = hub.running_rover(body.robot)
    if msg:
        return {'ok': False, 'detail': msg}
    res = hub.command({'op': 'critical_damage', 'robot': r})
    return res if not res['ok'] else {'ok': True, 'detail': f'Debris strike: critical damage on {display_name(r)}'}


@app.post('/api/command')
async def api_command(request: Request):
    """Operator command → /operator/command for fleet_brain_node. The answer (a /brain/decision carrying this
    command_id) arrives over the WebSocket; the local safety layer judges the command whatever Gemma says."""
    body, err = await parse_body(request, CommandBody)
    if err:
        return err
    m = hub.mgr
    if m is None or m.status not in ('running', 'degraded') or not m.config:
        return JSONResponse({'ok': False, 'detail': 'No running mission'}, status_code=409)
    if body.robot is not None and body.robot not in m.config.robots:
        return JSONResponse({'ok': False, 'detail': f'robot: {body.robot!r} is not in the fleet {m.config.robots}'},
                            status_code=422)
    if hub.link is None:
        return JSONResponse({'ok': False, 'detail': hub.link_error or 'ROS link unavailable'}, status_code=503)
    cid = hub.next_command_id()
    n = hub.link.publish_operator_command({'id': cid, 'text': body.text, 'robot': body.robot,
                                           't': round(time.time(), 3)})
    if n == 0:
        return {'ok': False, 'id': cid, 'detail': 'The agent brain is not listening on /operator/command'}
    return {'ok': True, 'id': cid}


@app.post('/api/sim/speed')
async def api_speed(request: Request):
    body, err = await parse_body(request, SpeedBody)
    if err:
        return err
    m = hub.mgr
    if m.status not in ('running', 'degraded') or not m.config:
        return {'ok': False, 'detail': 'No running mission'}
    res = hub.command({'op': 'set_speed', 'speed': body.speed})
    if not res['ok']:
        return res
    hub.speed = body.speed
    m.config = m.config.model_copy(update={'sim_speed': body.speed})
    return {'ok': True, 'speed': body.speed, 'detail': f'Simulation speed set to {body.speed:g}×'}


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    if not origin_ok(ws.headers.get('origin'), ws.headers.get('host')):
        await ws.close(code=1008)
        return
    await ws.accept()
    first = (hub.mgr.config.robots[0] if hub.mgr and hub.mgr.config else 'rover_0')
    cl = hub.attach(ws, first)
    hub.send(cl, dumps(hub.status_msg()))
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get('type') == 'select_camera' and isinstance(msg.get('robot'), str):
                cl.camera = msg['robot']
    except (WebSocketDisconnect, RuntimeError, KeyError):    # KeyError: a binary frame
        pass
    finally:
        hub.drop(cl)


if DIST.is_dir():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='ui')
