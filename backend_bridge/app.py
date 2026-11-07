"""Redshift Mission Control backend: REST + one WebSocket (docs/protocol.md §6-7) over the ROS 2 fleet.

    uvicorn backend_bridge.app:app --port 8000      (from the repo root, with ROS + ros2_ws sourced)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import _paths  # noqa: F401
from .config_schema import MissionConfig, options, physics_preview
from .encoders import lidar_points_body, png_b64, robot_telemetry
from .mission_manager import ROOT, MissionManager
from .ros_link import HAVE_ROS, RosLink
from exoswarm_sim.robot_models import ROVER_SPEC, SCOUT_SPEC

log = logging.getLogger('redshift')
DIST = ROOT / 'web_ui' / 'dist'
RUNTIME = ROOT / 'runtime'
TELEMETRY_HZ = 15.0
LIDAR_HZ = 5.0


class Hub:
    def __init__(self):
        self.clients: dict[WebSocket, dict] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None
        self.link: RosLink | None = None
        self.mgr: MissionManager | None = None
        self.task: asyncio.Task | None = None
        self.link_error: str | None = None

    # ------------------------------------------------------------------ outbound
    async def send(self, ws: WebSocket, text: str):
        try:
            await ws.send_text(text)
        except Exception:  # noqa: BLE001 - client went away
            self.clients.pop(ws, None)

    async def broadcast(self, msg: dict):
        if not self.clients:
            return
        text = json.dumps(msg)
        await asyncio.gather(*(self.send(ws, text) for ws in list(self.clients)))

    def status_msg(self) -> dict:
        return dict(type='status', **self.mgr.status_dict())

    async def on_status(self):
        await self.broadcast(self.status_msg())

    async def on_log(self, line: str):
        log.info(line)
        await self.broadcast({'type': 'boot_log', 'line': line, 't': round(time.time(), 3)})

    async def on_event(self, ev: dict):
        if ev.get('op') == 'fallback' and self.link and self.mgr.config:
            self.link.set_fleet(self.mgr.config.robots, self._specs(self.mgr.config))
        await self.broadcast({'type': 'event', 'event': ev})

    def from_ros(self, msg: dict):
        """Called on the ROS thread: hand the message to the asyncio loop."""
        if self.loop and self.queue:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    @staticmethod
    def _specs(cfg: MissionConfig) -> dict:
        return {r: (ROVER_SPEC if r.startswith('rover') else SCOUT_SPEC) for r in cfg.robots}

    # ------------------------------------------------------------------ loops
    async def pump_queue(self):
        while True:
            msg = await self.queue.get()
            await self.broadcast(msg)

    def mission_dict(self) -> dict:
        st = self.link.stats if self.link else {}
        env = self.link.env if self.link else {}
        m = self.mgr
        return {'status': m.status, 'backend': m.backend,
                'elapsed_s': round(time.time() - m.running_since, 1) if m.running_since else 0.0,
                'percent_mapped': float(st.get('percent', 0.0)), 'science_found': len(st.get('science', [])),
                'hazard_cells': int(st.get('hazard_cells', 0)),
                'env': {'dust_level': float(env.get('dust_level', 0.0)), 'seismic': bool(env.get('seismic', False))},
                'config': m.config.model_dump() if m.config else None}

    async def broadcaster(self):
        period = 1.0 / TELEMETRY_HZ
        tick = 0
        last_map_v = -1
        sent_cam: dict = {}
        nxt = time.perf_counter()
        while True:
            nxt += period
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
            tick += 1
            link, m = self.link, self.mgr
            if link is None or m.status not in ('running', 'degraded') or not self.clients:
                continue
            with link.lock:
                robots = {r: robot_telemetry(c) for r, c in link.robots.items()}
                layers_v, layers = link.layers_version, link.layers
                stats = dict(link.stats)
                cams = {r: (c.jpeg_t, c.jpeg_b64) for r, c in link.robots.items()}
            await self.broadcast({'type': 'telemetry', 't': round(time.time(), 3), 'mission': self.mission_dict(),
                                  'robots': robots})
            if layers is not None and layers_v != last_map_v and tick % 4 == 0:
                last_map_v = layers_v
                meta = {k: stats.get(k) for k in ('size_m', 'res', 'cells', 'origin', 'elev_min', 'elev_max')}
                await self.broadcast({'type': 'map', 'png_b64': await asyncio.to_thread(png_b64, layers),
                                      'meta': meta, 'stats': stats})
            lidar_cache: dict = {}
            for ws, cl in list(self.clients.items()):
                r = cl.get('camera')
                t_jpg, jpg = cams.get(r, (0.0, None))
                if jpg and sent_cam.get(ws) != t_jpg:
                    sent_cam[ws] = t_jpg
                    await self.send(ws, json.dumps({'type': 'camera', 'robot': r, 'jpeg_b64': jpg, 't': round(t_jpg, 3)}))
                if r in link.robots and r.startswith('rover') and tick % int(TELEMETRY_HZ / LIDAR_HZ) == 0:
                    if r not in lidar_cache:
                        cloud = link.lidar_cloud(r)
                        lidar_cache[r] = None if cloud is None else json.dumps({
                            'type': 'lidar', 'robot': r, 'max_range': 25,
                            'points': lidar_points_body(cloud, ROVER_SPEC['lidar_xyz'], 1500), 't': round(time.time(), 3)})
                    if lidar_cache[r]:
                        await self.send(ws, lidar_cache[r])

    # ------------------------------------------------------------------ commands
    def running_rover(self, robot: str | None) -> tuple[str | None, str | None]:
        m = self.mgr
        if m.status not in ('running', 'degraded') or not m.config:
            return None, 'No running mission'
        rovers = [r for r in m.config.robots if r.startswith('rover')]
        r = robot or rovers[0]
        return (r, None) if r in rovers else (None, f'Unknown rover {r!r}')

    def command(self, cmd: dict) -> dict:
        if self.link is None:
            return {'ok': False, 'detail': self.link_error or 'ROS link unavailable'}
        n = self.link.publish_command(cmd)
        if n == 0:
            return {'ok': False, 'detail': 'World is not listening on /world/command'}
        return {'ok': True}


hub = Hub()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    hub.queue = asyncio.Queue()
    if HAVE_ROS:
        try:
            hub.link = RosLink(on_trace=lambda t: hub.from_ros({'type': 'trace', 'entry': t}),
                               on_event=lambda e: hub.from_ros({'type': 'event', 'event': e}))
        except Exception as e:  # noqa: BLE001
            hub.link_error = f'ROS link failed: {e}'
    else:
        hub.link_error = 'rclpy not importable (source /opt/ros/jazzy/setup.bash and ros2_ws/install/setup.bash)'
    hub.mgr = MissionManager(RUNTIME, hub.on_status, hub.on_log, hub.on_event,
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


@app.get('/api/options')
async def api_options():
    return options()


@app.get('/api/physics_preview')
async def api_physics_preview(g: float = 2.2):
    return physics_preview(g)


@app.post('/api/mission/deploy')
async def api_deploy(body: dict):
    try:
        cfg = MissionConfig(**(body or {}))
    except ValidationError as e:
        return JSONResponse({'ok': False, 'detail': e.errors(include_url=False, include_context=False)}, status_code=422)
    if hub.link is None:
        return JSONResponse({'ok': False, 'detail': hub.link_error}, status_code=503)
    hub.link.set_fleet(cfg.robots, hub._specs(cfg))
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
async def api_rockslide(body: dict | None = None):
    r, err = hub.running_rover((body or {}).get('robot'))
    if err:
        return {'ok': False, 'detail': err}
    res = hub.command({'op': 'spawn_rockslide', 'robot': r})
    return res if not res['ok'] else {'ok': True, 'detail': f'Rockslide triggered across {r} planned path'}


@app.post('/api/inject/anomaly')
async def api_anomaly(body: dict | None = None):
    b = body or {}
    if hub.mgr.status not in ('running', 'degraded'):
        return {'ok': False, 'detail': 'No running mission'}
    dust = min(1.0, max(0.0, float(b.get('dust_level') if b.get('dust_level') is not None else 0.85)))
    dur = min(120.0, max(3.0, float(b.get('duration_s') if b.get('duration_s') is not None else 25.0)))
    res = hub.command({'op': 'anomaly', 'dust_level': dust, 'duration_s': dur, 'seismic_s': 4.0})
    return res if not res['ok'] else {'ok': True, 'detail': f'Dust {dust:.2f} for {dur:.0f} s with a 4 s tremor'}


@app.post('/api/inject/tip')
async def api_tip(body: dict | None = None):
    r, err = hub.running_rover((body or {}).get('robot'))
    if err:
        return {'ok': False, 'detail': err}
    res = hub.command({'op': 'jolt', 'robot': r})
    return res if not res['ok'] else {'ok': True, 'detail': f'Subsidence jolt applied to {r} chassis'}


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    first = (hub.mgr.config.robots[0] if hub.mgr and hub.mgr.config else 'rover_0')
    hub.clients[ws] = {'camera': first}
    await hub.send(ws, json.dumps(hub.status_msg()))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get('type') == 'select_camera' and isinstance(msg.get('robot'), str):
                hub.clients[ws]['camera'] = msg['robot']
    except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
        pass
    finally:
        hub.clients.pop(ws, None)


if DIST.is_dir():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='ui')
