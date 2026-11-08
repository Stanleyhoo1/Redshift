"""FastAPI server for the Redshift simulation (see PROTOCOL.md).

The Simulation runs in a background thread paced to SIM_SPEED x real time. An asyncio task sends a
`tick` to every connected WebSocket client about 5 times per wall second; new clients get an `init`
first. Everything that touches the simulation holds one lock. POST /api/hazard changes the world
(rockfall, sand, dust storm, rover fault); GET /api/camera/{rover_id}.jpg comes from camera.py.

Tier 2 (protocol v3): the shared Gemma planner (gemma.get_planner()) is created when the server
starts and loads the model in the background (about 40 s the first time), so it is warm before the
rovers need it. POST /api/chat and POST /api/answer go to sim.advisor (advisor.py). A reset cancels
questions still queued for Gemma about the old world.

Version 7: with AUTO_LAUNCH=0 (the default) the server builds the world from its env vars and waits in the
setup phase (frozen at t = 0; hazards, chat and answers get 409) until POST /api/launch, which may change any
launch setting first (GET /api/config lists them). POST /api/reset and every launch build the new world
before stopping the old one, so a failed build leaves the old world running.

    uvicorn redshift.server:app --port 8000           (what Docker runs)
    python -m redshift.server [--viewer] [--port 8000] (native run, optional MuJoCo viewer)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from .config import PRESETS, SETTINGS, SETTINGS_BY_KEY, Config, physics_table
from .gemma import close_planner, get_planner
from .hazards import TYPES as HAZARD_TYPES
from .hazards import HazardError
from .sim import Simulation

TICK_HZ = 5.0


def _auto_reset_s() -> float:
    """Seconds to wait after exploration ends (all rovers parked) before starting a new planet. 0 = off."""
    try:
        return max(0.0, float(os.environ.get("AUTO_RESET_S", "0") or 0))
    except ValueError:
        return 0.0


AUTO_RESET_S = _auto_reset_s()
NOT_LAUNCHED = "The mission has not launched yet."


def launch_at_start() -> bool:
    """AUTO_LAUNCH=1 (or the AUTO_RESET_S demo loop) starts the mission when the server starts. Otherwise
    the server builds the world and waits in the setup phase for POST /api/launch (v7)."""
    return os.environ.get("AUTO_LAUNCH", "0").strip().lower() in ("1", "true", "yes", "on") or AUTO_RESET_S > 0


CLIENT_QUEUE_MAX = 40          # about 8 s of ticks; a client further behind than that gets a fresh init
MAX_DEBT_S = 2.0               # if the sim falls this far behind real time, stop trying to catch up


def _log(msg: str) -> None:
    print(f"[redshift] {msg}", file=sys.stderr, flush=True)


def _clean(obj):
    """Replace non-finite floats with None (only used if a NaN ever slips through)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def dumps(obj) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), allow_nan=False)
    except ValueError:
        return json.dumps(_clean(obj), separators=(",", ":"), allow_nan=False)


# ====================================================================== simulation thread
class SimRunner:
    """Owns one Simulation and the thread that steps it in real time (times sim_speed).
    planner: the Gemma planner for tier 2 (default: the server's shared one, if it started one;
    otherwise none, so the rovers run on the rules alone)."""

    def __init__(self, cfg: Config, planner=None):
        self.cfg = cfg
        self.lock = threading.Lock()
        planner = planner if planner is not None else state.planner
        self.sim = Simulation(cfg, planner=planner, frame_source=self._frame if planner is not None else None)
        self._wait_lock = threading.Lock()
        self._waiting = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.speed = 0.0               # measured sim seconds per wall second
        self.errors = 0
        self.started = False           # v7: False while the world waits in the setup phase

    @contextmanager
    def locked(self):
        """Hold the simulation lock. The sim thread steps aside quickly when someone is waiting."""
        with self._wait_lock:
            self._waiting += 1
        try:
            self.lock.acquire()
        finally:
            with self._wait_lock:
                self._waiting -= 1
        try:
            yield self.sim
        finally:
            self.lock.release()

    def _frame(self, rover_id: str):
        """Camera frame for Gemma (advisor worker thread only, never the sim thread): rendered on the
        camera render thread, which holds the sim lock just for the scene update."""
        try:
            from . import camera
        except Exception:               # noqa: BLE001 (no OpenGL here: Gemma gets the numbers only)
            return None
        return camera.render_jpeg(self.sim, rover_id, "fpv", 320, 240, runner=self)

    def start(self) -> None:
        self.started = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="redshift-sim", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self.sim.close()

    def _run(self) -> None:
        sim, speed = self.sim, max(self.cfg.sim_speed, 0.01)
        wall0, sim0 = time.perf_counter(), sim.t
        win_wall, win_sim = wall0, sim0
        next_status = wall0 + 60.0
        consecutive_errors = 0
        while not self._stop.is_set():
            now = time.perf_counter()
            ahead = sim.t - (sim0 + (now - wall0) * speed)
            if ahead > 0.0:
                self._stop.wait(min(ahead / speed, 0.05))
                continue
            if -ahead > MAX_DEBT_S:            # machine too slow: run flat out but forget the backlog
                wall0, sim0 = now, sim.t
            with self.lock:
                try:
                    sim.step()
                    consecutive_errors = 0
                except Exception:
                    self.errors += 1
                    consecutive_errors += 1
                    _log("simulation step failed:\n" + traceback.format_exc())
                    if consecutive_errors >= 20:
                        _log("too many errors in a row, the simulation is paused")
                        self._stop.wait(1.0)
            if self._waiting:
                time.sleep(0.001)              # let the broadcaster or the viewer take the lock
            if now - win_wall >= 2.0:
                self.speed = (sim.t - win_sim) / (now - win_wall)
                win_wall, win_sim = now, sim.t
            if now >= next_status:
                next_status = now + 60.0
                st = sim.fleet.stats()
                _log(f"t={sim.t:.0f} s, {self.speed:.2f}x real time, explored {st['explored_pct']:.0f}%, "
                     f"slope limit {st['slope_limit_deg']:.0f} deg, targets {st['targets_found']}/{st['targets_total']}")


# ====================================================================== clients
class Client:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue()

    def send_later(self, text: str) -> bool:
        """Queue a message. Returns False if the client is too far behind (caller resyncs it)."""
        if self.queue.qsize() >= CLIENT_QUEUE_MAX:
            return False
        self.queue.put_nowait(text)
        return True

    def restart_with(self, text: str) -> None:
        """Drop everything still queued and start again from this message (an init)."""
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(text)

    async def pump(self) -> None:
        try:
            while True:
                text = await self.queue.get()
                await self.ws.send_text(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass   # the receive loop notices the disconnect and cleans up


class AppState:
    def __init__(self):
        self.runner: SimRunner | None = None
        self.planner = None            # shared Gemma planner (tier 2), made at server start
        self.clients: set = set()
        self.reset_lock: asyncio.Lock | None = None
        self.broadcaster: asyncio.Task | None = None
        # v7: "setup" while the world is built but frozen until POST /api/launch (only the lifespan sets
        # it; without one, as in the tests, the server behaves as before)
        self.phase = "running"


state = AppState()


def build_init_text() -> str:
    with state.runner.locked() as sim:
        msg = sim.init_message()
        msg["phase"] = state.phase           # v7, server-level fields: not in Simulation.init_message
        msg["config"] = sim.cfg.settings()
    return dumps(msg)


def _describe(cfg: Config) -> str:
    return (f"seed {cfg.seed}, {cfg.num_rovers} rovers, {cfg.gravity_g:g} g, map {cfg.map_size:g} m, "
            f"{cfg.sim_speed:g}x real time")


def _bad(error: str, status: int = 400, **extra) -> JSONResponse:
    return JSONResponse({"ok": False, "error": error, **extra}, status_code=status)


async def broadcast_loop() -> None:
    loop = asyncio.get_running_loop()
    period = 1.0 / TICK_HZ
    next_t = loop.time()
    while True:
        next_t += period
        delay = next_t - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -1.0:
            next_t = loop.time()
        try:
            runner = state.runner
            if runner is None:
                continue
            # Build the tick and hand it to the current clients with no await in between, so a
            # client that connects now gets its init strictly before any later tick.
            with runner.locked() as sim:
                msg = sim.tick_message()
            text = dumps(msg)
            lagging = [c for c in list(state.clients) if not c.send_later(text)]
            if lagging:
                init = build_init_text()
                for c in lagging:
                    c.restart_with(init)
        except Exception:
            _log("broadcast failed:\n" + traceback.format_exc())


def _reset_lock() -> asyncio.Lock:
    """Serialises restarts and launches (made here on first use when no lifespan ran, as in the tests)."""
    if state.reset_lock is None:
        state.reset_lock = asyncio.Lock()
    return state.reset_lock


def _reinit_clients() -> None:
    """Send every client a fresh init (no await in here, so no tick of the new world arrives first)."""
    init = build_init_text()
    for c in list(state.clients):
        c.restart_with(init)


async def _restart(cfg: Config) -> None:
    """Build a world from cfg, then stop the old one, start the new one and re-init every client. The caller
    holds the reset lock. The new world is built first: if that fails (the exception propagates) the old
    world is untouched and keeps running. Order after the build, as before v7: stop the old world, THEN
    cancel. old.stop() joins the old sim thread and its advisor worker, which submit to the shared planner
    outside the sim lock; cancelling earlier would let a question they send in that window take Gemma's one
    slot ahead of the new world's first questions. The new world submits nothing until it steps."""
    new = await asyncio.to_thread(SimRunner, cfg)
    old = state.runner
    if old is not None:
        await asyncio.to_thread(old.stop)
    if state.planner is not None:
        state.planner.cancel()         # drop questions about the old world (nothing of it runs any more)
    # swap and re-init every client in one go (no await), so nobody sees the new sim's ticks first
    state.runner = new
    state.phase = "running"
    new.start()
    _reinit_clients()
    _log(f"new world: {_describe(cfg)}")


class SettingsRefused(Exception):
    """Config.from_settings refused a request's settings (errors: its readable texts)."""

    def __init__(self, errors: list):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


class AlreadyLaunched(Exception):
    """A launch asked for from the setup phase found the mission already started with other settings."""


ALREADY_LAUNCHED = "Another launch started the mission first. Use New mission to change its settings."


def _current_cfg() -> Config:
    return state.runner.cfg if state.runner is not None else Config.from_env()


def _settings_cfg(base: Config, data: dict, seed: int | None = None) -> Config:
    """base with these launch settings (and seed, if given) applied, or SettingsRefused."""
    cfg, errors = Config.from_settings(base, data)
    if errors:
        raise SettingsRefused(errors)
    return cfg if seed is None else cfg.with_seed(seed)


def _resolve(cfg):
    """cfg is a Config, or a function that makes one from the current world's cfg. Called under the reset
    lock, so a request queued behind another restart or launch builds on the settings that one left, not on
    the world it saw when it came in."""
    return cfg(_current_cfg()) if callable(cfg) else cfg


async def restart_sim(cfg) -> Config | None:
    """Replace the current world with a fresh one, running at once. cfg: a Config, or a function of the
    current world's cfg (see _resolve) that may return None to do nothing. Returns the cfg built from."""
    async with _reset_lock():
        cfg = _resolve(cfg)
        if cfg is not None:
            await _restart(cfg)
        return cfg


_NOW = object()


def launch_sim(cfg, *, seen=_NOW, from_setup: bool | None = None):
    """Start the mission (a coroutine to await): the built world itself if cfg is what it was built with and
    it never ran, otherwise a new world built from cfg (a Config, or a function of the current world's cfg).

    seen and from_setup are the world and phase the launch was asked from (default: as they are when
    launch_sim is called, not when the reset lock comes free). A launch asked from the setup phase starts that
    mission once: if another launch (or a reset) started it while this one waited for the lock, nothing is
    rebuilt. The coroutine then returns "already" when its settings match the running world and raises
    AlreadyLaunched when they do not. Otherwise it returns "started" or "rebuilt"."""
    if seen is _NOW:
        seen = state.runner
    if from_setup is None:
        from_setup = state.phase == "setup"
    return _launch(cfg, seen, from_setup)


async def _launch(cfg, seen, from_setup: bool) -> str:
    async with _reset_lock():
        runner = state.runner
        cfg = _resolve(cfg)
        if from_setup and runner is not None and (runner is not seen or runner.started):
            if runner.cfg == cfg:
                _log("launch: the mission had already started with these settings")
                return "already"
            raise AlreadyLaunched(ALREADY_LAUNCHED)
        if runner is not None and not runner.started and runner.cfg == cfg:
            state.phase = "running"
            runner.start()
            _reinit_clients()
            _log(f"launch: {_describe(cfg)}")
            return "started"
        await _restart(cfg)
        return "rebuilt"


def _next_planet(runner: SimRunner, seed: int):
    """auto_reset_loop's new world, for restart_sim: the current settings with a new seed, decided under the
    reset lock. If a reset or launch replaced the finished world meanwhile, nothing happens (the new world
    gets its own countdown)."""
    def make(cur: Config):
        if state.runner is not runner:
            return None
        _log(f"exploration finished, landing on a new planet (seed {seed})")
        return cur.with_seed(seed)
    return make


async def auto_reset_loop() -> None:
    """Keeps a demo running unattended: once the fleet has finished exploring and every rover is
    parked or charging, wait AUTO_RESET_S seconds and land on a new planet."""
    watched, done_since = None, None
    while True:
        await asyncio.sleep(2.0)
        try:
            runner = state.runner
            if runner is None:
                continue
            if runner is not watched:          # a manual reset restarts the countdown
                watched, done_since = runner, None
            with runner.locked() as sim:     # a faulted rover is not parked: no reset while one is down
                done = sim.fleet.complete and all(r.state in ("idle", "charging") for r in sim.rovers)
            if not done:
                done_since = None
                continue
            now = time.monotonic()
            if done_since is None:
                done_since = now
            elif now - done_since >= AUTO_RESET_S:
                await restart_sim(_next_planet(runner, random.randint(1, 99999)))
        except Exception:
            _log("auto reset failed:\n" + traceback.format_exc())


async def planner_watch() -> None:
    """Log the Gemma planner's state changes (warming, ready, offline) so the console shows them."""
    last = None
    while True:
        await asyncio.sleep(1.0)
        p = state.planner
        if p is None:
            continue
        st = p.stats()
        key = (st["state"], st["model"])
        if key != last:
            last = key
            extra = f", warm-up {st['warmup_ms'] / 1000:.1f} s" if st["state"] == "ready" and st["warmup_ms"] else ""
            _log(f"gemma planner {st['state']}: {st['model']}{extra}. {st['detail'] or ''}".rstrip())


# ====================================================================== app
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config.from_env()
    state.reset_lock = asyncio.Lock()
    # Gemma first: its monitor thread finds Ollama and loads the model in the background right away
    state.planner = get_planner()
    st = state.planner.stats()
    _log(f"gemma planner: {st['state']}, model {st['requested_model']} via {st['backend']} {st['url']} "
         f"(budget {st['budget_ms']:.0f} ms per call)")
    state.runner = await asyncio.to_thread(SimRunner, cfg)
    if launch_at_start():
        state.phase = "running"
        state.runner.start()
        _log(f"simulation running: {_describe(cfg)}")
    else:   # v7: the world is built and waits (frozen at t = 0) for POST /api/launch from the setup screen
        state.phase = "setup"
        _log(f"world built, waiting for launch (POST /api/launch, or AUTO_LAUNCH=1 to start at once): "
             f"{_describe(cfg)}")
    watch = asyncio.create_task(planner_watch())
    state.broadcaster = asyncio.create_task(broadcast_loop())
    auto = asyncio.create_task(auto_reset_loop()) if AUTO_RESET_S > 0 else None
    try:
        yield
    finally:
        for task in (state.broadcaster, auto, watch):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        runner, state.runner = state.runner, None
        if runner is not None:
            await asyncio.to_thread(runner.stop)
        if state.planner is not None:
            state.planner = None
            await asyncio.to_thread(close_planner)


app = FastAPI(title="Redshift sim", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


try:   # onboard camera renders (camera.py); the server runs without them if they cannot load
    from .camera import router as camera_router
    app.include_router(camera_router)
except Exception as _e:  # noqa: BLE001
    _log(f"camera endpoint unavailable: {_e}")


@app.get("/")
async def root():
    return PlainTextResponse("Redshift sim. WebSocket on /ws, HTTP on /api/health, /api/truth, /api/config, "
                             "/api/launch, /api/reset, /api/hazard, /api/chat, /api/answer, "
                             "/api/camera/{rover_id}.jpg.\n")


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/truth")
async def truth():
    runner = state.runner
    if runner is None:
        return JSONResponse({"ok": False, "error": "simulation not running"}, status_code=503)
    text = await asyncio.to_thread(runner.sim.truth_json)   # static per terrain, cached after the first call
    return Response(content=text, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/config")
async def get_config():
    """v7 launch settings for the setup screen: the settings of the world the server holds (current), what
    the env vars give (defaults), the settings table (fields), presets, and the predicted climb and tow
    limits per gravity step (physics)."""
    defaults = Config.from_env()
    runner = state.runner
    cfg = runner.cfg if runner is not None else defaults
    return {"ok": True, "phase": state.phase, "current": cfg.settings(), "defaults": defaults.settings(),
            "fields": [s.to_json() for s in SETTINGS], "presets": PRESETS, "physics": physics_table(cfg)}


@app.post("/api/reset")
async def reset(request: Request):
    """Restart now and run (also from the setup phase): {"seed": 7} as before, plus any launch setting (v7).
    Keys left out keep the current world's settings. The new world is built before the old one stops.
    As before v7, keys that are not settings are ignored (listed in `ignored`) and any seed of 0 or more
    works, also above the setup screen's 2147483647."""
    body = (await request.body()).strip()
    data = {}
    if body:
        try:
            data = json.loads(body)
        except ValueError:
            return _bad('Body must be JSON like {"seed": 7}')
        data = dict(data) if isinstance(data, dict) else {}       # anything else was ignored before v7
    seed = data.pop("seed", None)
    if seed is not None:
        try:
            seed = int(seed)                                       # as before: "8" and 7.9 are accepted
        except (TypeError, ValueError, OverflowError):
            return _bad("seed must be an integer")
        if seed < 0:                                               # numpy cannot seed with one
            return _bad("seed must be 0 or more")
    ignored = [k for k in data if k not in SETTINGS_BY_KEY]
    data = {k: v for k, v in data.items() if k in SETTINGS_BY_KEY}
    try:   # checked now, so a bad body gets its 400 at once; applied again to the then current world
        _settings_cfg(_current_cfg(), data)
        cfg = await restart_sim(lambda cur: _settings_cfg(cur, data, seed))
    except SettingsRefused as e:
        return _bad(str(e), errors=e.errors)
    except Exception as e:  # noqa: BLE001 (the old world keeps running)
        _log("reset failed:\n" + traceback.format_exc())
        return _bad(f"Could not build the new world: {e}", 500)
    out = {"ok": True, "seed": cfg.seed, "config": cfg.settings()}
    if ignored:
        out["ignored"] = ignored
    return out


@app.post("/api/launch")
async def launch(request: Request):
    """v7: start the mission with these launch settings, e.g. {"gravity_g": 1.0, "num_rovers": 4}. Keys left
    out keep the built world's values; {} launches the world as built. While running it starts a new mission.
    Asked for from the setup phase it starts the mission once (two setup screens launching together do not
    land twice: see launch_sim)."""
    runner = state.runner
    if runner is None:
        return _bad("simulation not running", 503)
    seen, from_setup = runner, state.phase == "setup"           # what this request was made against
    body = (await request.body()).strip()
    try:
        data = json.loads(body) if body else {}
    except ValueError:
        return _bad('Body must be JSON like {"gravity_g": 1.5, "num_rovers": 4}')
    try:   # checked now, so a bad body gets its 400 at once; applied again to the then current world
        _settings_cfg(_current_cfg(), data)
        await launch_sim(lambda cur: _settings_cfg(cur, data), seen=seen, from_setup=from_setup)
    except SettingsRefused as e:
        return _bad(str(e), errors=e.errors)
    except AlreadyLaunched as e:
        return _bad(str(e), 409)
    except Exception as e:  # noqa: BLE001 (the current world keeps running)
        _log("launch failed:\n" + traceback.format_exc())
        return _bad(f"Could not build the new world: {e}", 500)
    return {"ok": True, "phase": state.phase, "config": state.runner.cfg.settings()}


@app.post("/api/hazard")
async def hazard(request: Request):
    """Change the world now: {"type": "rockfall" | "sand" | "dust_storm" | "rover_fault" | "spire_fall" |
    "lifeform", "rover": "r2", "x": 12.5, "y": -30.0} (rover, x and y optional; x and y place rockfall,
    sand, spire_fall and lifeform at that point). The rovers are not told; they have to notice
    through their sensors."""
    runner = state.runner
    if runner is None:
        return JSONResponse({"ok": False, "error": "simulation not running"}, status_code=503)
    if state.phase == "setup":
        return _bad(NOT_LAUNCHED, 409)
    body = (await request.body()).strip()
    try:
        data = json.loads(body) if body else {}
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": 'Body must be JSON like {"type": "rockfall", "rover": "r2"}'},
                            status_code=400)
    kind, rover = data.get("type"), data.get("rover")
    if kind not in HAZARD_TYPES:
        return JSONResponse({"ok": False, "error": f"unknown hazard type {kind!r} (use one of "
                                                   f"{', '.join(HAZARD_TYPES)})"}, status_code=400)
    if rover is not None and not isinstance(rover, str):
        return JSONResponse({"ok": False, "error": "rover must be a rover id like \"r2\""}, status_code=400)
    x, y = data.get("x"), data.get("y")
    if (x is None) != (y is None):
        return JSONResponse({"ok": False, "error": "give both x and y (world metres), or neither"}, status_code=400)
    if x is not None:
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)) \
                or not (math.isfinite(x) and math.isfinite(y)):
            return JSONResponse({"ok": False, "error": "x and y must be numbers (world metres)"}, status_code=400)

    def trigger():
        with runner.locked() as sim:
            return sim.hazards.trigger(kind, rover, x, y)

    try:
        h = await asyncio.to_thread(trigger)
    except HazardError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    _log(f"hazard {h['id']}: {h['type']}" + (f" on {h['rover']}" if h.get("rover") else "")
         + (f" at ({x:g}, {y:g})" if x is not None else ""))
    return {"ok": True, "hazard": h}


async def _json_body(request: Request):
    body = (await request.body()).strip()
    try:
        data = json.loads(body) if body else {}
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


@app.post("/api/chat")
async def chat(request: Request):
    """Operator message: {"text": "...", "rover": "r2"} (rover optional). Posted to the chat at once;
    Gemma (or the keyword parser) interprets it in the background and the rover answers in the chat."""
    runner = state.runner
    if runner is None:
        return JSONResponse({"ok": False, "error": "simulation not running"}, status_code=503)
    if state.phase == "setup":
        return _bad(NOT_LAUNCHED, 409)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"ok": False, "error": 'Body must be JSON like {"text": "rover 2 explore north"}'},
                            status_code=400)
    text, rover = data.get("text"), data.get("rover")
    if not isinstance(text, str) or not text.strip():
        return JSONResponse({"ok": False, "error": "text must not be empty"}, status_code=400)
    if rover is not None and not isinstance(rover, str):
        return JSONResponse({"ok": False, "error": "rover must be a rover id like \"r2\""}, status_code=400)

    def post():
        with runner.locked() as sim:
            hint = rover if rover in {r.id for r in sim.rovers} else None
            return sim.advisor.operator_message(text, hint)

    cid = await asyncio.to_thread(post)
    return {"ok": True, "id": cid}


@app.post("/api/answer")
async def answer(request: Request):
    """Operator answer to a clarification request: {"id": "q3", "option": "reroute"}."""
    runner = state.runner
    if runner is None:
        return JSONResponse({"ok": False, "error": "simulation not running"}, status_code=503)
    if state.phase == "setup":
        return _bad(NOT_LAUNCHED, 409)
    data = await _json_body(request)
    if data is None or not isinstance(data.get("id"), str) or not isinstance(data.get("option"), str):
        return JSONResponse({"ok": False, "error": 'Body must be JSON like {"id": "q3", "option": "reroute"}'},
                            status_code=400)

    def give():
        with runner.locked() as sim:
            return sim.advisor.answer(data["id"], data["option"])

    status = await asyncio.to_thread(give)
    if status == "ok":
        return {"ok": True}
    if status == "unknown":
        return JSONResponse({"ok": False, "error": f"unknown request {data['id']!r}"}, status_code=404)
    if status == "closed":
        return JSONResponse({"ok": False, "error": f"request {data['id']} already expired or was answered"},
                            status_code=409)
    return JSONResponse({"ok": False, "error": f"option {data['option']!r} is not one of this request's options"},
                        status_code=400)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    client = Client(ws)
    try:
        client.restart_with(build_init_text())
    except Exception:
        _log("could not build init:\n" + traceback.format_exc())
        await ws.close()
        return
    state.clients.add(client)
    pump = asyncio.create_task(client.pump())
    try:
        while True:
            msg = await ws.receive()          # anything the client sends is ignored
            if msg.get("type") == "websocket.disconnect":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        state.clients.discard(client)
        pump.cancel()


# ====================================================================== native run with viewer
def _probe_display() -> str | None:
    """Check that a window with an OpenGL context can be opened. MuJoCo's viewer exits the whole
    process if it cannot, so test first with a hidden window. Returns None if OK, else the reason."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no display found"
    if sys.platform == "darwin":
        return None   # GLFW must be started by mjpython itself on macOS; launch_passive explains if not
    try:
        import warnings

        import glfw
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not glfw.init():
                return "could not start GLFW (no usable display)"
            try:
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
                win = glfw.create_window(16, 16, "probe", None, None)
                if not win:
                    return "could not create an OpenGL window"
                glfw.destroy_window(win)
            finally:
                glfw.terminate()
    except Exception as e:
        return f"display check failed ({e})"
    return None


def _run_viewer(server) -> bool:
    """Open the MuJoCo viewer in this (main) thread and keep it synced with the sim.
    Returns False if no viewer could be opened."""
    reason = _probe_display()
    if reason:
        _log(f"{reason}, so no viewer: running headless")
        return False
    try:
        import mujoco.viewer  # imported lazily: needs OpenGL, which the Docker image does not have
    except Exception as e:
        _log(f"MuJoCo viewer unavailable ({e}): running headless")
        return False
    while state.runner is None and not server.should_exit:
        time.sleep(0.1)
    while not server.should_exit:
        runner = state.runner
        if runner is None:
            time.sleep(0.1)
            continue
        try:
            handle = mujoco.viewer.launch_passive(runner.sim.model, runner.sim.data,
                                                  show_left_ui=False, show_right_ui=False)
        except Exception as e:
            hint = " (on macOS start it with mjpython)" if sys.platform == "darwin" else ""
            _log(f"could not open the MuJoCo viewer{hint}: {e}. Running headless")
            return False
        with handle:
            with runner.locked():
                handle.cam.lookat[:] = (0.0, 0.0, 0.0)
                handle.cam.distance = 0.75 * runner.cfg.map_size
                handle.cam.elevation = -35.0
                handle.cam.azimuth = 135.0
                try:   # the look from visuals.py: sun shadows, sky, horizon haze; rovers, lander, liquid shown
                    for g in (3, 4, 5):
                        handle.opt.geomgroup[g] = 1
                    flags = handle.user_scn.flags
                    flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
                    flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
                    flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
                    flags[mujoco.mjtRndFlag.mjRND_FOG] = 0     # GL fog would also fog the skybox flat
                except Exception as e:      # noqa: BLE001 (an older viewer: keep its defaults)
                    _log(f"viewer render flags not set: {e}")
            while handle.is_running() and not server.should_exit and state.runner is runner:
                with runner.locked():
                    handle.sync()
                time.sleep(1.0 / 30.0)
            closed_by_user = not handle.is_running()
        if closed_by_user and state.runner is runner:
            _log("viewer closed; the server keeps running (Ctrl+C to stop)")
            return True
        # otherwise a reset swapped in a new world: reopen the viewer on it
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Redshift simulation server.")
    ap.add_argument("--viewer", action="store_true", help="open the native MuJoCo viewer (or set VIEWER=1)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args(argv)

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
    if not (args.viewer or Config.from_env().viewer):
        server.run()
        return 0
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    import signal

    signal.signal(signal.SIGTERM, lambda *_: setattr(server, "should_exit", True))
    try:
        _run_viewer(server)
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        thread.join(10.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
