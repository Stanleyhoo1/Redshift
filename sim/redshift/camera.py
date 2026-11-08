"""Onboard cameras: live MuJoCo renders from each rover (PROTOCOL.md, GET /api/camera/{rover_id}.jpg).

Two views, both MuJoCo free cameras computed from the rover pose, so the model needs no cameras:
  fpv    on top of the sensor mast, looking along the heading, tilted down a little, wide angle.
         Like the LiDAR head it is gimballed: it follows the rover's heading but keeps the horizon level.
  chase  about 3 m behind and 1.6 m above the rover, looking at it (rover, wheels and terrain visible).

Threading: an OpenGL context belongs to the thread that made it, and FastAPI runs handlers on many
threads, so ONE render thread owns the GL context and the MuJoCo render context (rebuilt when a
reset brings a new model). Requests reach it through a queue and get a Future back; the async
handler awaits it. The sim lock is held only while mjv_updateScene copies the state into the
scene; rendering, reading pixels and JPEG encoding happen outside the lock.

Look (visuals.py): procedural regolith, hazy sky with a low sun, the sun's shadows and distance haze.
  shadows  the directional light's shadow box (SHADOW_CLIP_M around a point a little ahead of the
           camera) is moved in front of each camera before rendering, so shadows are crisp where the
           camera looks (CAMERA_SHADOWS=0 switches them off: they cost the most on software GL).
  haze     exponential distance fog in the horizon colour, applied to the pixels from the depth buffer
           (MuJoCo's own fog would also fog the skybox, which then turns into one flat colour).
The rules' quick colour check and spot_targets() render without shadows or haze (machine vision
frames, cheaper, and on the sim thread). scripts/camera_bench.py measures the render time per frame.

Backends: MuJoCo picks the GL backend from MUJOCO_GL when it is first imported.
  Docker   MUJOCO_GL=osmesa + PYOPENGL_PLATFORM=osmesa (software rendering, no GPU or display).
  Native   the default (GLFW) on Windows, macOS and Linux with a display. On Linux without a
           display we fall back to EGL, then OSMesa, if MUJOCO_GL is not set.
If no context can be created the endpoint answers 503 with a JSON error; the rest of the server
is not affected.

Wiring (one line in server.py):
    from .camera import router as camera_router; app.include_router(camera_router)

Python helpers (also used by the tests): render_frame(), render_rgb(), render_jpeg(),
renderer_status(), and spot_targets() (camera based target detection, see its docstring; not wired
into the rovers). spot_targets() renders in the calling thread with that thread's own GL context,
because it is meant to run with the sim lock already held.
"""
from __future__ import annotations

import asyncio
import atexit
import contextlib
import importlib
import io
import math
import os
import queue
import sys
import threading
import time
import warnings
from concurrent.futures import Future
from dataclasses import dataclass

import mujoco
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

DEFAULT_W, DEFAULT_H = 320, 240
MAX_W, MAX_H = 1280, 720      # PROTOCOL.md v5 (the model's offscreen buffer is this size too)
MIN_SIDE = 16
JPEG_QUALITY = 80
VIEWS = ("fpv", "chase")
RENDER_TIMEOUT_S = 10.0       # a request waits at most this long for its frame
MAX_QUEUED = 32               # more pending renders than this: answer 503 (busy) instead of queueing
GL_RETRY_S = 30.0             # after a failed GL start, wait this long before trying again

# ---- camera rig (metres, degrees). Rover frame: x forward, z up, origin at the chassis centre.
FPV_FORWARD = 0.26            # just in front of the sensor head (the head box spans x = 0.12 .. 0.24)
FPV_UP = 0.05                 # above lidar_height, where the LiDAR site sits
FPV_PITCH_DEG = -10.0         # tilted down a little: ground ahead in the lower half, horizon above the middle
FPV_FOVY_DEG = 62.0           # vertical field of view (about 78 deg horizontal at 4:3)
CHASE_BACK = 3.0
CHASE_UP = 1.6
CHASE_LOOK_AHEAD = 1.0        # aim a little ahead of the rover: rover in the lower middle, horizon in view
CHASE_LOOK_UP = 0.4
CHASE_FOVY_DEG = 52.0
CHASE_CLEARANCE = 0.8         # keep the chase camera at least this high above the ground
CHASE_SIGHT_CLEARANCE = 0.25  # and keep the ground this far below the line of sight to the rover
CHASE_SMOOTH_S = 0.3          # chase camera heading follows the rover with this time constant (wall s)
NEAR_CLIP = 0.05              # the model's znear is relative to its extent (several metres here)
FAR_CLIP = 600.0              # the skybox is drawn beyond 300 m; terrain is never more than ~170 m away
SKY_DIST_M = 260.0            # pixels farther than this (along the ray) are sky: no distance haze
SHADOW_CLIP_M = 14.0          # sun shadows are drawn within this many metres of a point ...
SHADOW_AHEAD_M = 10.0         # ... this far ahead of the camera (beyond that the haze hides their absence)
SHADOW_BACK_M = 40.0          # the sun's shadow camera sits this far up-sun of that point
SHADOWS = os.environ.get("CAMERA_SHADOWS", "1").strip().lower() not in ("0", "off", "false", "no")

_GEOM_GROUPS = (1, 1, 1, 1, 1, 1)   # terrain, rocks and spires, targets and lifeforms, rovers, lander ring
                                    # (the walls in group 4 have alpha 0 and are never drawn), liquid (5)


class RenderUnavailable(RuntimeError):
    """No OpenGL context could be created on this machine (or rendering is switched off)."""


class RenderBusy(RuntimeError):
    """Too many frames are already queued."""


class UnknownRover(KeyError):
    pass


def _log(msg: str) -> None:
    print(f"[camera] {msg}", file=sys.stderr, flush=True)


# ====================================================================== camera rig
@dataclass
class CamSpec:
    eye: np.ndarray          # world position (3,)
    forward: np.ndarray      # unit view direction (3,)
    fovy: float              # vertical field of view, degrees
    near: float
    far: float

    def basis(self):
        """Unit (forward, right, up) of the camera, as MuJoCo builds it for a free camera."""
        f = self.forward / np.linalg.norm(self.forward)
        right = np.cross(f, (0.0, 0.0, 1.0))
        n = np.linalg.norm(right)
        right = right / n if n > 1e-9 else np.array([0.0, -1.0, 0.0])
        return f, right, np.cross(right, f)


def _find_rover(sim, rover_id: str):
    for r in sim.rovers:
        if r.id == rover_id:
            return r
    raise UnknownRover(rover_id)


def _ground(sim, x: float, y: float):
    ter = getattr(sim, "terrain", None)
    if ter is None or not hasattr(ter, "height_at"):
        return None
    try:
        return float(ter.height_at(float(x), float(y)))
    except Exception:
        return None


def _rover_pose(sim, rover):
    """Chassis position, rotation matrix and heading straight from MjData (exact, not a tick old)."""
    d = sim.data
    pos = np.array(d.xpos[rover.body], dtype=np.float64)
    R = np.array(d.xmat[rover.body], dtype=np.float64).reshape(3, 3)
    if not (np.isfinite(pos).all() and np.isfinite(R).all()):
        pos = np.array([rover.x, rover.y, rover.z], dtype=np.float64)
        R = np.eye(3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return pos, R, yaw


def rig(view: str, sim, rover, heading=None) -> CamSpec:
    """Camera for this view at the current state. heading overrides the chase camera azimuth (rad)."""
    pos, R, yaw = _rover_pose(sim, rover)
    lidar_h = float(getattr(sim.cfg, "lidar_height", 0.75))
    if view == "fpv":
        local = np.array([FPV_FORWARD, 0.0, lidar_h + FPV_UP])
        # mast top in the world (it sways with the chassis); upright offset if the rover is on its side
        eye = pos + (R @ local if R[2, 2] > 0.5 else np.array([0.0, 0.0, local[2]]))
        p = math.radians(FPV_PITCH_DEG)
        fwd = np.array([math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), math.sin(p)])
        return CamSpec(eye, fwd, FPV_FOVY_DEG, NEAR_CLIP, FAR_CLIP)
    if view != "chase":
        raise ValueError(f"unknown view {view!r}")
    az = yaw if heading is None else heading
    look = pos + np.array([CHASE_LOOK_AHEAD * math.cos(az), CHASE_LOOK_AHEAD * math.sin(az), CHASE_LOOK_UP])
    eye = pos + np.array([-CHASE_BACK * math.cos(az), -CHASE_BACK * math.sin(az), CHASE_UP])
    g = _ground(sim, eye[0], eye[1])
    if g is not None:
        eye[2] = max(eye[2], g + CHASE_CLEARANCE)
    # rise over a crest between the camera and the rover so the rover stays in sight
    for frac in (0.25, 0.5, 0.75):
        p = eye + frac * (look - eye)
        g = _ground(sim, p[0], p[1])
        if g is not None and g + CHASE_SIGHT_CLEARANCE > p[2]:
            eye[2] += (g + CHASE_SIGHT_CLEARANCE - p[2]) / (1.0 - frac)
    fwd = look - eye
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    return CamSpec(eye, fwd, CHASE_FOVY_DEG, 0.1, FAR_CLIP)


def _apply_free_camera(cam: mujoco.MjvCamera, spec: CamSpec) -> None:
    """A free camera sits at lookat - distance * forward(azimuth, elevation)."""
    f = spec.forward
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.trackbodyid = -1
    cam.distance = 1.0
    cam.lookat[:] = spec.eye + f
    cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
    cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, float(f[2])))))


def _set_frustum(scn: mujoco.MjvScene, spec: CamSpec) -> None:
    """Our own clip planes and field of view (mjv_updateScene uses the model's, which are made for
    the overview camera). frustum_width stays 0, so mjr_render takes the width from the aspect ratio."""
    top = spec.near * math.tan(math.radians(spec.fovy) / 2.0)
    for i in range(2):
        c = scn.camera[i]
        c.frustum_near = spec.near
        c.frustum_far = spec.far
        c.frustum_top = top
        c.frustum_bottom = -top
        c.frustum_center = 0.0


def _place_sun_shadow(scn: mujoco.MjvScene, spec: CamSpec) -> None:
    """Centre the directional light's shadow box on the ground a little ahead of the camera. MuJoCo
    draws directional shadows in a box of half width shadowClip around the line through the light's
    position, for casters in front of that position."""
    f = spec.forward
    fh = np.array([f[0], f[1], 0.0])
    n = np.linalg.norm(fh)
    fh = fh / n if n > 1e-6 else np.zeros(3)
    focus = spec.eye + fh * SHADOW_AHEAD_M - np.array([0.0, 0.0, 1.0])
    for i in range(scn.nlight):
        lt = scn.lights[i]
        if lt.castshadow and int(lt.type) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL) and not lt.headlight:
            d = np.asarray(lt.dir, dtype=np.float64)
            d /= max(np.linalg.norm(d), 1e-9)
            lt.pos[:] = focus - d * SHADOW_BACK_M


def _drop_markers(scn: mujoco.MjvScene, markers: set) -> None:
    """Leave the science target marker capsules out of a look frame (their natural decoration is drawn)."""
    geom_t = int(mujoco.mjtObj.mjOBJ_GEOM)
    for i in range(scn.ngeom):
        sg = scn.geoms[i]
        if sg.objtype == geom_t and sg.objid in markers:
            sg.size[:] = 1e-5
            sg.pos[2] -= 1000.0


def _vision_lighting(scn: mujoco.MjvScene) -> None:
    """Machine vision frames (look=False): even, near neutral light from the camera (an on-board
    illuminator) instead of the low warm sun, and no glow, so colours stay close to the true surface
    colours without clipping (a clipped channel shifts the hue) or white highlights (violet crystal plus
    a highlight looks lavender, like a mineral marker)."""
    for i in range(scn.nlight):
        lt = scn.lights[i]
        lt.specular[:] = 0.0
        if lt.headlight:
            lt.ambient[:] = (0.32, 0.31, 0.3)
            lt.diffuse[:] = (0.55, 0.54, 0.53)
        else:
            lt.diffuse[:] = np.asarray(lt.diffuse) * 0.35
            lt.ambient[:] = 0.0
    for i in range(scn.ngeom):
        scn.geoms[i].emission = 0.0


_RAY_K: dict = {}


def _ray_factor(h: int, w: int, fovy: float) -> np.ndarray:
    """Per pixel: distance along the ray / depth along the view axis (cached per image size and fov)."""
    key = (h, w, round(fovy, 3))
    k = _RAY_K.get(key)
    if k is None:
        tv = math.tan(math.radians(fovy) / 2.0)
        u = ((np.arange(w) + 0.5) / w * 2.0 - 1.0) * tv * w / h
        v = (1.0 - (np.arange(h) + 0.5) / h * 2.0) * tv
        k = np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2).astype(np.float32)
        if len(_RAY_K) > 16:
            _RAY_K.clear()
        _RAY_K[key] = k
    return k


def _haze(rgb: np.ndarray, dep: np.ndarray, fovy: float) -> np.ndarray:
    """Distance haze of a dense atmosphere: blend toward the horizon colour with distance along the ray
    (exponential, nothing within visuals.FOG_START_M), never on the sky (farther than SKY_DIST_M)."""
    from .visuals import FOG_END_M, FOG_START_M, HORIZON_RGB, HAZE_MAX
    dist = dep * _ray_factor(dep.shape[0], dep.shape[1], fovy)
    d = np.clip(dist - FOG_START_M, 0.0, None)
    f = (1.0 - np.exp(d * (-3.0 / (FOG_END_M - FOG_START_M)))) * HAZE_MAX
    f[dist > SKY_DIST_M] = 0.0
    fog = np.asarray(HORIZON_RGB, np.float32) * 255.0
    out = rgb.astype(np.float32)
    out += (fog - out) * f[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _linear_depth(raw: np.ndarray, near: float, far: float) -> np.ndarray:
    """Depth buffer (reverse Z, as mujoco.Renderer reads it) to metres along the view axis."""
    zfar, znear = np.float32(far), np.float32(near)
    c = -(zfar + znear) / (zfar - znear)
    d = -(np.float32(2) * zfar * znear) / (zfar - znear)
    c = np.float32(-0.5) * c - np.float32(0.5)
    d = np.float32(-0.5) * d
    return (d / (raw.astype(np.float64) + c)).astype(np.float32)


# ====================================================================== render thread
def _make_gl_context(w: int, h: int):
    """GL context for MuJoCo's chosen backend; on headless Linux with MUJOCO_GL unset, try EGL then OSMesa."""
    gl_env = os.environ.get("MUJOCO_GL", "").strip().lower()
    if gl_env in ("disable", "disabled", "off", "false", "0"):
        raise RenderUnavailable(f"rendering is switched off (MUJOCO_GL={gl_env})")
    linux = sys.platform.startswith("linux")
    headless = linux and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    candidates = []
    default = getattr(mujoco, "GLContext", None)
    if default is not None and not (headless and gl_env == ""):    # GLFW cannot work without a display
        candidates.append(default)
    if linux and gl_env in ("", "glfw"):
        candidates += ["egl", "osmesa"]
    tried = []
    for cand in candidates:
        name = cand if isinstance(cand, str) else getattr(cand, "__module__", str(cand))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if isinstance(cand, str):
                    cand = importlib.import_module(f"mujoco.{cand}").GLContext
                ctx = cand(w, h)
                if hasattr(ctx, "_context") and not ctx._context:     # GLFW returns no window instead of raising
                    raise RuntimeError("no OpenGL window could be created (no display?)")
                ctx.make_current()
            return ctx
        except Exception as e:
            tried.append(f"{name}: {type(e).__name__}: {e}")
    hint = ("set MUJOCO_GL=osmesa (apt install libosmesa6) or MUJOCO_GL=egl"
            if sys.platform.startswith("linux") else "check the graphics driver")
    raise RenderUnavailable("no OpenGL context could be created (" + "; ".join(tried) + f"). Hint: {hint}")


class _Gfx:
    """MuJoCo render state for one model: GPU context (textures, height fields), scene, options."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.con)
        self.con.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        self.scn = mujoco.MjvScene(model, maxgeom=int(model.ngeom) + 500)
        flags = self.scn.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0          # the sky texture has its own horizon haze
        flags[mujoco.mjtRndFlag.mjRND_FOG] = 0           # distance haze is applied from depth (_haze)
        self.con.shadowClip = SHADOW_CLIP_M
        # science target marker capsules: what LiDAR, the spectrometer and machine vision see. Look frames
        # that draw the visual-only decoration (group 4) show the natural target instead (world.py)
        self.markers = {g for g in range(model.ngeom)
                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("target_")}
        self.cam = mujoco.MjvCamera()
        self._opts: dict = {}
        self.max_w = int(min(MAX_W, model.vis.global_.offwidth))
        self.max_h = int(min(MAX_H, model.vis.global_.offheight))

    def option(self, hide_groups: tuple = ()) -> mujoco.MjvOption:
        opt = self._opts.get(hide_groups)
        if opt is None:
            opt = mujoco.MjvOption()
            opt.geomgroup[:] = _GEOM_GROUPS
            for g in hide_groups:
                opt.geomgroup[g] = 0
            opt.sitegroup[:] = 0
            self._opts[hide_groups] = opt
        return opt

    def free(self) -> None:
        if self.con is not None:
            self.con.free()
            self.con = None


@dataclass
class Frame:
    rgb: np.ndarray                 # (h, w, 3) uint8, top row first
    depth: np.ndarray | None        # (h, w) float32 metres along the view axis, if asked for
    spec: CamSpec
    t: float                        # sim time of the frame
    rover_pos: np.ndarray           # chassis centre at that time
    rover_yaw: float


class GLRenderer:
    """A GL context plus MuJoCo render state. Use it from ONE thread only: the one that made it."""

    def __init__(self):
        self._ctx = None
        self._gfx: _Gfx | None = None
        self._error: str | None = None
        self._error_t = -1e9
        self._chase: dict = {}                # (id(sim), rover id) -> (smoothed heading, wall time)
        self.backend: str | None = None
        self.frames = 0

    def ensure(self) -> None:
        if self._ctx is not None:
            return
        if self._error is not None and time.monotonic() - self._error_t < GL_RETRY_S:
            raise RenderUnavailable(self._error)
        try:
            self._ctx = _make_gl_context(MAX_W, MAX_H)
        except RenderUnavailable as e:
            self._error, self._error_t = str(e), time.monotonic()
            _log(f"rendering unavailable: {e}")
            raise
        self._error = None
        self.backend = type(self._ctx).__module__
        _log(f"OpenGL context ready ({self.backend}, thread {threading.current_thread().name})")

    def _gfx_for(self, model) -> _Gfx:
        if self._gfx is not None and self._gfx.model is model:
            return self._gfx
        self._ctx.make_current()
        if self._gfx is not None:
            self._gfx.free()
            self._gfx = None
        self._gfx = _Gfx(model)
        return self._gfx

    def release(self) -> None:
        if self._gfx is not None and self._ctx is not None:
            with contextlib.suppress(Exception):
                self._ctx.make_current()
                self._gfx.free()
        self._gfx = None
        if self._ctx is not None:
            with contextlib.suppress(Exception):
                self._ctx.free()
        self._ctx = None

    def _chase_heading(self, sim, rid: str, yaw: float) -> float:
        now = time.monotonic()
        key = (id(sim), rid)
        prev = self._chase.get(key)
        if prev is None or now - prev[1] > 2.0:
            h = yaw
        else:
            a = 1.0 - math.exp(-(now - prev[1]) / CHASE_SMOOTH_S)
            h = prev[0] + a * math.remainder(yaw - prev[0], 2 * math.pi)
        if len(self._chase) > 64:
            self._chase.clear()
        self._chase[key] = (h, now)
        return h

    def frame(self, runner, sim, rover_id: str, view: str, w: int, h: int, *,
              depth: bool = False, hide_groups: tuple = (), smooth: bool = True, look: bool = True) -> Frame:
        """Render one view. runner given: hold runner.locked() for the scene update only.
        runner None: the caller already holds the sim lock (or nothing else touches sim).
        look False: no shadows and no distance haze (machine vision frames: cheaper)."""
        self.ensure()
        try:
            return self._frame(runner, sim, rover_id, view, w, h, depth, tuple(hide_groups), smooth, look)
        except (UnknownRover, ValueError):
            raise
        except Exception:
            self.release()           # GL trouble: start from a fresh context next time
            raise

    def _frame(self, runner, sim, rover_id, view, w, h, depth, hide_groups, smooth, look=True) -> Frame:
        for _attempt in range(3):
            g = self._gfx_for(sim.model)
            hold = runner.locked() if runner is not None else contextlib.nullcontext(sim)
            with hold as s:
                if s.model is not g.model:      # a reset swapped the model in between: rebuild, retry
                    sim = s
                    continue
                rover = _find_rover(s, rover_id)
                pos, _R, yaw = _rover_pose(s, rover)
                heading = self._chase_heading(s, rover_id, yaw) if (view == "chase" and smooth) else None
                spec = rig(view, s, rover, heading)
                _apply_free_camera(g.cam, spec)
                mujoco.mjv_updateScene(g.model, s.data, g.option(hide_groups), None, g.cam,
                                       mujoco.mjtCatBit.mjCAT_ALL.value, g.scn)
                t = float(s.data.time)
            break
        else:
            raise RuntimeError("the model kept changing while rendering")
        w = max(MIN_SIDE, min(int(w), g.max_w))
        h = max(MIN_SIDE, min(int(h), g.max_h))
        _set_frustum(g.scn, spec)
        g.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if (look and SHADOWS) else 0
        if look and 4 not in hide_groups and g.markers:
            _drop_markers(g.scn, g.markers)
        if look and SHADOWS:
            _place_sun_shadow(g.scn, spec)
        elif not look:
            _vision_lighting(g.scn)
        rect = mujoco.MjrRect(0, 0, w, h)
        self._ctx.make_current()
        mujoco.mjr_render(rect, g.scn, g.con)
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        raw = np.empty((h, w), dtype=np.float32) if (depth or look) else None
        mujoco.mjr_readPixels(rgb, raw, rect, g.con)
        self.frames += 1
        rgb = rgb[::-1]                                     # OpenGL rows start at the bottom
        dep = _linear_depth(raw[::-1], spec.near, spec.far) if raw is not None else None
        if look:
            rgb = _haze(rgb, dep, spec.fovy)
        return Frame(np.ascontiguousarray(rgb), dep if depth else None, spec, t, pos, yaw)


class RenderThread:
    """The HTTP camera's render thread: owns one GLRenderer and runs queued jobs one at a time."""

    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self._mu = threading.Lock()
        self._pending: dict = {}              # coalescing key -> Future (identical queued requests share one)
        self._thread: threading.Thread | None = None
        self.gl = GLRenderer()

    def submit(self, fn, *args, key=None) -> Future:
        """Run fn(gl, *args) on the render thread. Jobs with the same key, while still queued, share a result."""
        with self._mu:
            if key is not None:
                fut = self._pending.get(key)
                if fut is not None:
                    return fut
            if self._jobs.qsize() >= MAX_QUEUED:
                raise RenderBusy("too many camera frames queued, try again shortly")
            fut = Future()
            if key is not None:
                self._pending[key] = fut
            self._jobs.put((fut, key, fn, args))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="redshift-camera", daemon=True)
                self._thread.start()
        return fut

    def call(self, fn, *args, timeout: float = RENDER_TIMEOUT_S):
        """Blocking version of submit (runs inline when already on the render thread)."""
        if threading.current_thread() is self._thread:
            return fn(self.gl, *args)
        return self.submit(fn, *args).result(timeout)

    def _run(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                break
            fut, key, fn, args = item
            if key is not None:
                with self._mu:
                    if self._pending.get(key) is fut:
                        del self._pending[key]
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(self.gl, *args))
            except BaseException as e:          # noqa: BLE001 (handed to whoever waits for it)
                fut.set_exception(e)
        self.gl.release()

    def shutdown(self, timeout: float = 2.0) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            self._jobs.put(None)
            t.join(timeout)
        self._thread = None


_RT = RenderThread()
_LOCAL = threading.local()       # GLRenderer per calling thread, for spot_targets()


def _local_renderer() -> GLRenderer:
    gl = getattr(_LOCAL, "gl", None)
    if gl is None:
        gl = _LOCAL.gl = GLRenderer()
    return gl


@atexit.register
def _shutdown_render_thread() -> None:
    _RT.shutdown()
    gl = getattr(_LOCAL, "gl", None)
    if gl is not None:
        gl.release()


# ====================================================================== encoding and helpers
def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb), "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _jpeg_job(gl: GLRenderer, runner, sim, rover_id, view, w, h) -> bytes:
    return encode_jpeg(gl.frame(runner, sim, rover_id, view, w, h).rgb)


def render_frame(sim, rover_id: str, view: str = "fpv", w: int = DEFAULT_W, h: int = DEFAULT_H, *,
                 runner=None, depth: bool = False, hide_groups: tuple = (), smooth: bool = False) -> Frame:
    """Blocking render on the render thread. With runner=None the caller must own the sim (tests,
    scripts); never call it while holding the sim lock if the HTTP endpoint is live (use runner=)."""
    return _RT.call(lambda gl: gl.frame(runner, sim, rover_id, view, w, h,
                                        depth=depth, hide_groups=hide_groups, smooth=smooth))


def render_rgb(sim, rover_id: str, view: str = "fpv", w: int = DEFAULT_W, h: int = DEFAULT_H, **kw) -> np.ndarray:
    return render_frame(sim, rover_id, view, w, h, **kw).rgb


def render_jpeg(sim, rover_id: str, view: str = "fpv", w: int = DEFAULT_W, h: int = DEFAULT_H, **kw) -> bytes:
    return encode_jpeg(render_rgb(sim, rover_id, view, w, h, **kw))


def renderer_status() -> dict:
    """{"ok": bool, "backend": "mujoco.osmesa" | ..., "error": str | None}. Starts GL if needed."""
    def probe(gl: GLRenderer):
        try:
            gl.ensure()
            return {"ok": True, "backend": gl.backend, "error": None}
        except RenderUnavailable as e:
            return {"ok": False, "backend": None, "error": str(e)}
    return _RT.call(probe)


# ====================================================================== camera based target spotting
DETECT_W, DETECT_H = 160, 120
DETECT_MAX_RANGE = 12.0        # beyond this haze washes the colours out
DETECT_MIN_PIXELS = 6
DETECT_TOL = 0.22              # max per channel chroma difference from the marker colour
DETECT_SPLIT = 0.7             # 0 = background colour, 1 = marker colour: a match must be beyond this
DETECT_MIN_VALUE = 50          # ignore pixels darker than this (shadowed edges, dark sky)
DETECT_DEPTH_GAP = 0.75        # split a colour blob where depth jumps by more than this (two markers)
DETECT_MERGE_R = 1.5           # blobs closer than this (m) are one marker (markers are >= 2.5 m apart)
MARKER_RADIUS = 0.28           # target capsule radius in world.py: its axis is this far behind the surface
DETECT_HIDE_GROUPS = (3, 4)    # the fleet's own hardware (rovers, lander ring) is not drawn


def _parse_rgb(v):
    """'r g b a' floats, '#rrggbb', or a sequence (0..1 or 0..255) -> np.array([r, g, b]) in 0..1."""
    try:
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("#") and len(s) >= 7:
                return np.array([int(s[k:k + 2], 16) / 255.0 for k in (1, 3, 5)])
            vals = [float(x) for x in s.replace(",", " ").split()[:3]]
        else:
            vals = [float(x) for x in list(v)[:3]]
        if len(vals) < 3:
            return None
        arr = np.array(vals)
        return arr / 255.0 if arr.max() > 1.0 else arr
    except (TypeError, ValueError):
        return None


def target_palette() -> dict:
    """Marker colour per target kind, read from world.TARGET_RGBA at call time (it may change)."""
    from . import world

    pal = {}
    for kind, v in getattr(world, "TARGET_RGBA", {}).items():
        rgb = _parse_rgb(v)
        if rgb is not None and rgb.max() > 0:
            pal[kind] = rgb
    return pal


def _chroma(rgb: np.ndarray):
    """Colour with the brightness divided out (each pixel scaled so its largest channel is 1)."""
    c = rgb.astype(np.float32)
    v = c.max(axis=-1)
    return c / np.maximum(v, 1.0)[..., None], v


def _blobs(mask: np.ndarray) -> list:
    """8-connected components of a small boolean image, as (rows, cols) index arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, ys, xs = [(y0, x0)], [], []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for yy in (y - 1, y, y + 1):
                if 0 <= yy < h:
                    for xx in (x - 1, x, x + 1):
                        if 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
        out.append((np.array(ys), np.array(xs)))
    return out


def detect_targets(fr: Frame, palette: dict | None = None, max_range: float = DETECT_MAX_RANGE,
                   min_pixels: int = DETECT_MIN_PIXELS) -> list:
    """Find target markers in an fpv frame rendered with depth. See spot_targets()."""
    if fr.depth is None:
        raise ValueError("detect_targets needs a frame rendered with depth=True")
    palette = target_palette() if palette is None else palette
    h, w = fr.rgb.shape[:2]
    chroma, value = _chroma(fr.rgb)
    bright = value >= DETECT_MIN_VALUE
    usable = bright & (fr.depth < max_range + 2.0)
    # "nothing here" references: the typical colour of each sixth of the image (horizontal bands)
    # and of the whole frame, at any distance. Some band is mostly ground even when the white lander
    # pad fills the bottom and sky the top; a marker never fills a whole band.
    bgs = []
    for rows in [slice(k * h // 6, (k + 1) * h // 6) for k in range(6)] + [slice(0, h)]:
        px = chroma[rows][bright[rows]]
        if len(px) >= 20:
            bgs.append(np.median(px, axis=0))
    spec = fr.spec
    f, right, up = spec.basis()
    tan_v = math.tan(math.radians(spec.fovy) / 2.0)
    tan_h = tan_v * w / h
    u = ((np.arange(w) + 0.5) / w * 2.0 - 1.0) * tan_h
    v = (1.0 - (np.arange(h) + 0.5) / h * 2.0) * tan_v
    kinds = list(palette)
    if not kinds:
        return []
    # each pixel goes to the nearest marker colour only (a specular highlight on a purple marker
    # drifts toward light blue). It must also sit well past the middle of the line from every
    # background colour to that marker colour: some marker colours (orange vents) are close to the
    # ground's, so "closer to the marker than to the ground" alone is not enough
    refs = np.stack([palette[k] / palette[k].max() for k in kinds])
    dk = np.abs(chroma[None] - refs[:, None, None, :]).max(axis=-1)
    nearest = dk.argmin(axis=0)
    d_near = dk.min(axis=0)
    ref = refs[nearest]                                  # (h, w, 3) colour of the nearest marker kind
    along = np.full((h, w), np.inf, dtype=np.float32)
    for bg in bgs:
        axis = ref - bg
        n2 = np.maximum((axis * axis).sum(axis=-1), 1e-6)
        along = np.minimum(along, ((chroma - bg) * axis).sum(axis=-1) / n2)
    match = usable & (d_near < DETECT_TOL) & (along > DETECT_SPLIT)
    found = []
    for ki, kind in enumerate(kinds):
        for ys, xs in _blobs(match & (nearest == ki)):
            if len(ys) < min_pixels:
                continue
            dep = fr.depth[ys, xs]
            order = np.argsort(dep)
            cuts = np.nonzero(np.diff(dep[order]) > DETECT_DEPTH_GAP)[0] + 1
            for part in np.split(order, cuts):
                if len(part) < min_pixels:
                    continue
                py, px, pd = ys[part], xs[part], dep[part]
                dirs = f[None, :] + u[px][:, None] * right[None, :] + v[py][:, None] * up[None, :]
                pts = spec.eye[None, :] + pd[:, None] * dirs
                x, y, z = (float(c) for c in np.median(pts, axis=0))   # robust to edge pixels
                ox, oy = x - spec.eye[0], y - spec.eye[1]
                n = math.hypot(ox, oy)
                if n > 1e-6:                    # from the visible surface to the marker's axis
                    x += MARKER_RADIUS * ox / n
                    y += MARKER_RADIUS * oy / n
                dx, dy = x - fr.rover_pos[0], y - fr.rover_pos[1]
                dist = math.hypot(dx, dy)
                if dist > max_range:
                    continue
                bearing = math.remainder(math.atan2(dy, dx) - fr.rover_yaw, 2 * math.pi)
                found.append({"kind": kind, "x": round(float(x), 2), "y": round(float(y), 2),
                              "z": round(float(z), 2), "distance": round(float(dist), 2),
                              "bearing": round(float(bearing), 4),
                              "bearing_deg": round(math.degrees(bearing), 1), "pixels": int(len(part))})
    # one marker, one detection: a smaller blob on the same spot as a bigger one is part of it
    # (markers are at least 2.5 m apart in terrain.py)
    out = []
    for det in sorted(found, key=lambda d_: -d_["pixels"]):
        if all(math.hypot(det["x"] - o["x"], det["y"] - o["y"]) > DETECT_MERGE_R for o in out):
            out.append(det)
    out.sort(key=lambda d_: d_["distance"])
    return out


def spot_targets(sim, rover, *, w: int = DETECT_W, h: int = DETECT_H,
                 max_range: float = DETECT_MAX_RANGE, palette: dict | None = None) -> list:
    """Camera based target detection (not wired into the rovers; the spectrometer rays do that today).

    Renders a small fpv RGB + depth image from the rover's mast camera, finds pixels whose colour
    (brightness divided out) matches a marker colour in world.TARGET_RGBA and is clearly not the
    ground's colour, groups them into blobs (split where depth jumps), and returns one detection per
    blob, nearest first:
        {"kind": "ice", "x": .., "y": .., "z": ..,   estimated world position of the marker axis
         "distance": m, "bearing": rad, "bearing_deg": deg,   from the rover centre, relative to its
         heading (positive = to the left), "pixels": blob size}
    The fleet's own hardware (other rovers, the lander ring) is not drawn, so it is never mistaken
    for a target. Measured with offline sweeps (and checked in tests/test_camera.py): on seeds
    42, 7 and 3 with the current and the old marker colours, every unobstructed marker at 3 to 9 m
    was found, position error median 0.03 m (max 0.24 m), bearing error median 0.1 deg (max 2.7);
    no false detections in those 432 views nor in 1000 random poses (4 seeds, 240 near the lander).
    Markers hidden behind other markers or terrain are not seen (it is a camera). Usable range is
    about 1.5 to 12 m; beyond that haze washes the colours out.

    Call it with the sim lock held (for example from Rover.tick on the sim thread) or on a sim
    nothing else is stepping. It renders in the CALLING thread with that thread's own GL context
    (made on first use, about 0.4 s), never through the HTTP render thread, so it cannot deadlock
    against camera requests waiting for the lock. Cost at 160 x 120 with OSMesa: about 14 ms to
    render plus 14 ms to analyse (25 to 40 ms per call on a busy 2 CPU machine), all of it with the
    caller's lock held, so call it every few seconds, not every tick. Raises RenderUnavailable if
    no GL context can be made.
    """
    fr = _local_renderer().frame(None, sim, rover.id, "fpv", w, h, depth=True,
                                 hide_groups=DETECT_HIDE_GROUPS, smooth=False, look=False)
    return detect_targets(fr, palette, max_range)


# ====================================================================== HTTP
router = APIRouter()


def _error(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg}, status_code=status, headers={"Cache-Control": "no-store"})


@router.get("/api/camera/{rover_id}.jpg")
async def camera_jpg(rover_id: str, view: str = "fpv", w: int = DEFAULT_W, h: int = DEFAULT_H):
    """PROTOCOL.md: live JPEG from a rover's camera. 404 unknown rover, 503 if rendering is unavailable."""
    from . import server   # lazy: server imports this module

    runner = server.state.runner
    if runner is None:
        return _error(503, "simulation not running")
    view = (view or "fpv").strip().lower()
    if view not in VIEWS:
        return _error(400, "view must be 'fpv' or 'chase'")
    w = max(MIN_SIDE, min(int(w), MAX_W))
    h = max(MIN_SIDE, min(int(h), MAX_H))
    sim = runner.sim
    if rover_id not in {r.id for r in sim.rovers}:
        return _error(404, f"unknown rover {rover_id!r}")
    try:
        fut = _RT.submit(_jpeg_job, runner, sim, rover_id, view, w, h, key=(id(runner), rover_id, view, w, h))
        # shield: a timed out request must not cancel a frame that other requests may share
        jpg = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(fut)), RENDER_TIMEOUT_S)
    except UnknownRover:
        return _error(404, f"unknown rover {rover_id!r}")
    except RenderUnavailable as e:
        return _error(503, f"rendering unavailable: {e}")
    except RenderBusy as e:
        return _error(503, str(e))
    except asyncio.TimeoutError:
        return _error(503, "camera render timed out")
    except Exception as e:      # noqa: BLE001 (the camera must never take the server down)
        _log(f"render failed: {type(e).__name__}: {e}")
        return _error(503, f"render failed: {type(e).__name__}: {e}")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
