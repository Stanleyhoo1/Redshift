"""Onboard cameras: live MuJoCo renders from each rover (PROTOCOL.md, GET /api/camera/{rover_id}.jpg).

Two views, both MuJoCo free cameras computed from the rover pose, so the model needs no cameras:
  fpv    on top of the sensor mast, looking along the heading, tilted down a little, wide angle.
         Like the LiDAR head it is gimballed: it follows the rover's heading but keeps the horizon level.
  chase  about 3 m behind and 1.6 m above the rover, looking at it (rover, wheels and terrain visible).

Threading: an OpenGL context belongs to the thread that made it, and FastAPI runs handlers on many
threads, so ONE render thread owns the GL context and the MuJoCo render context (rebuilt when a
reset brings a new model). The sim lock is held only to copy qpos and the mocap poses into a private
MjData (microseconds); kinematics, the scene update, rendering, reading pixels, haze, object
detection and JPEG encoding all happen outside the lock.

Speed (protocol v8, "Faster camera"), all without changing a pixel:
  culling  MuJoCo draws every geom in the scene, three times with shadows (shadow map, base pass,
           sun pass), and the 120 m map has 576 terrain tiles plus 240 bodies parked below it. The
           renderer keeps a private copy of the model whose geom groups it rewrites per frame: only
           geoms inside the camera frustum or inside the sun's shadow prism (casters that can shade
           what the camera sees) are drawn. Checked pixel for pixel against the full scene.
  streams  GET /api/camera answers at once with the newest frame of that view and size; the render
           thread keeps rendering every view asked for in the last STREAM_IDLE_S seconds, timed so a
           fresh frame is ready just before the client's next request (one render per request, at
           most STREAM_MAX_FPS), and drops the rest.
  vision   the same thread runs the onboard object detector (vision.py) on every streamed fpv frame
           and on a small, cheap render per rover (VISION_HZ, no shadows), see "Object detection".

Look (visuals.py): procedural regolith, hazy sky with a low sun, the sun's shadows and distance haze.
  shadows  the directional light's shadow box (SHADOW_CLIP_M around a point a little ahead of the
           camera) is moved in front of each camera before rendering, so shadows are crisp where the
           camera looks (CAMERA_SHADOWS=0 switches them off: they cost the most on software GL).
  haze     exponential distance fog in the horizon colour, applied to the pixels from the depth buffer
           (MuJoCo's own fog would also fog the skybox, which then turns into one flat colour).
  dust     during a dust storm the haze thickens into a brown dust veil (visibility about 20 m),
           ramping in and out over DUST_RAMP_S, so the camera (and the detector) see less.
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
renderer_status(), segment_ids() (offline ground truth for the detector's evaluation), and
spot_targets() (the old marker colour spotting, see its docstring; not wired into the rovers).
spot_targets() renders in the calling thread with that thread's own GL context, because it is meant
to run with the sim lock already held. start_vision(runner) (server.py) starts the rovers' detectors.
"""
from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import importlib
import io
import itertools
import json
import math
import os
import queue
import sys
import threading
import time
import warnings
import weakref
from concurrent.futures import Future
from dataclasses import dataclass, field

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
DUST_RGB = (0.62, 0.47, 0.35)  # dust storm veil
DUST_SCALE_M = 9.0             # dust thickens with distance on this scale (visibility about 20 m) ...
DUST_MAX = 0.93                # ... up to this much
DUST_SKY = 0.8                 # and veils the sky this much
DUST_RAMP_S = 5.0              # sim seconds to thicken at the start of a storm and to clear at its end


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


STREAM_IDLE_S = 4.0            # a view nobody asked for in this long stops rendering (protocol v8)
STREAM_MAX = 6                 # views rendered in the background at once (the least recently asked goes)
STREAM_MAX_FPS = max(1.0, _env_float("CAMERA_FPS", 12.0))   # never more often than this per view
STREAM_LEAD_S = 0.03           # a frame should be ready this long before the client's next request
STREAM_REFRESH_S = 1.0         # a frame nobody fetched yet is still re-rendered after this long
VISION_HZ = max(0.0, _env_float("VISION_HZ", 1.0))           # detector renders per rover per wall second
VISION_W, VISION_H = 256, 192  # the small, cheap detector render (no shadows)
VISION_STATS_EVERY = 5         # one detector frame in this many is also checked against ground truth
CULL_GROUP = 5                 # geom group that means "not drawn this frame" in the private render model

_GEOM_GROUPS = (1, 1, 1, 1, 1, 0)   # terrain, rocks and spires, targets and lifeforms, rovers, lander ring
                                    # and decoration; group 5 is "culled" in the render copy (liquid moves to 2)
_FRAME_SEQ = itertools.count(1)     # frame numbers (X-Frame, vision.frame): one sequence for every renderer


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


def _rover_pose(sim, rover, data=None):
    """Chassis position, rotation matrix and heading straight from MjData (exact, not a tick old).
    data: the renderer's private snapshot (default: the sim's own MjData)."""
    d = sim.data if data is None else data
    pos = np.array(d.xpos[rover.body], dtype=np.float64)
    R = np.array(d.xmat[rover.body], dtype=np.float64).reshape(3, 3)
    if not (np.isfinite(pos).all() and np.isfinite(R).all()):
        pos = np.array([rover.x, rover.y, rover.z], dtype=np.float64)
        R = np.eye(3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return pos, R, yaw


def rig(view: str, sim, rover, heading=None, data=None) -> CamSpec:
    """Camera for this view at the current state. heading overrides the chase camera azimuth (rad).
    data: a private MjData snapshot to read the rover pose from (default sim.data)."""
    pos, R, yaw = _rover_pose(sim, rover, data)
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


def _cull_mask(centres: np.ndarray, rbound: np.ndarray, spec: CamSpec, w: int, h: int,
               shadows: bool) -> np.ndarray:
    """Geoms that can change a pixel of this frame: bounding sphere inside the camera frustum, or (with
    shadows) inside the sun's shadow prism, where it may cast a shadow onto what the camera sees.
    MuJoCo draws directional shadows only for casters within shadowClip of the line through the light's
    position (SHADOW_BACK_M up-sun of the shadow focus), so the prism is that square (its half diagonal,
    to be safe) extended along the sun direction."""
    f, right, up = spec.basis()
    rel = centres - spec.eye
    tv = math.tan(math.radians(spec.fovy) / 2.0)
    th = tv * w / h
    keep = rel @ f >= -rbound
    for n in (th * f - right, th * f + right, tv * f - up, tv * f + up):     # inward side planes
        keep &= rel @ (n / np.linalg.norm(n)) >= -rbound
    if shadows:
        from .visuals import TO_SUN
        fh = np.array([f[0], f[1], 0.0])
        nh = np.linalg.norm(fh)
        fh = fh / nh if nh > 1e-6 else np.zeros(3)
        r2 = centres - (spec.eye + fh * SHADOW_AHEAD_M - np.array([0.0, 0.0, 1.0]))
        along = r2 @ TO_SUN
        perp = np.linalg.norm(r2 - along[:, None] * TO_SUN[None, :], axis=1)
        keep |= (perp <= SHADOW_CLIP_M * 1.415 + rbound) & (along <= SHADOW_BACK_M + rbound)
    return keep


def _dust_level(sim) -> float:
    """How thick the dust storm veil is right now (0 = none, 1 = full), ramping in and out."""
    hz = getattr(sim, "hazards", None)
    st = getattr(hz, "storm", None) if hz is not None else None
    if st is None or not getattr(st, "active", False):
        return 0.0
    t = float(sim.data.time)
    a = (t - float(st.t)) / DUST_RAMP_S
    b = (float(st.until) - t) / DUST_RAMP_S if getattr(st, "until", None) is not None else 1.0
    return float(min(1.0, max(0.0, min(a, b))))


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


def _blend(rgb: np.ndarray, f: np.ndarray, colour) -> np.ndarray:
    """rgb * (1 - f) + colour * f per pixel, in 16 bit integers (f quantised to 1/256: at most half a
    grey level off, about twice as fast as float32 on a full HD frame)."""
    a = np.rint(f * 256.0).astype(np.uint16)
    out = rgb.astype(np.uint16)
    out *= (256 - a)[..., None]
    out += a[..., None] * np.asarray([round(c * 255.0) for c in colour], np.uint16)
    out += 128
    out >>= 8
    return out.astype(np.uint8)


def _haze(rgb: np.ndarray, dep: np.ndarray, fovy: float, dust: float = 0.0) -> np.ndarray:
    """Distance haze of a dense atmosphere: blend toward the horizon colour with distance along the ray
    (exponential, nothing within visuals.FOG_START_M), never on the sky (farther than SKY_DIST_M).
    dust > 0: a dust storm veil on top (thicker with distance, the sky veiled too)."""
    from .visuals import FOG_END_M, FOG_START_M, HORIZON_RGB, HAZE_MAX
    dist = dep * _ray_factor(dep.shape[0], dep.shape[1], fovy)
    sky = dist > SKY_DIST_M
    f = dist - np.float32(FOG_START_M)
    np.maximum(f, 0.0, out=f)
    f *= np.float32(-3.0 / (FOG_END_M - FOG_START_M))
    np.exp(f, out=f)
    f *= np.float32(-HAZE_MAX)
    f += np.float32(HAZE_MAX)
    f[sky] = 0.0
    out = _blend(rgb, f, HORIZON_RGB)
    if dust > 0.0:
        g = np.exp(dist * np.float32(-1.0 / DUST_SCALE_M))
        g *= np.float32(-DUST_MAX * dust)
        g += np.float32(DUST_MAX * dust)
        g[sky] = DUST_SKY * dust
        out = _blend(out, g, DUST_RGB)
    return out


def _linear_depth(raw: np.ndarray, near: float, far: float) -> np.ndarray:
    """Depth buffer (reverse Z, as mujoco.Renderer reads it) to metres along the view axis (float32 is
    plenty here: within 1e-5 relative of a float64 decode out to the far plane)."""
    zfar, znear = np.float32(far), np.float32(near)
    c = -(zfar + znear) / (zfar - znear)
    d = -(np.float32(2) * zfar * znear) / (zfar - znear)
    c = np.float32(-0.5) * c - np.float32(0.5)
    d = np.float32(-0.5) * d
    out = raw + c
    np.divide(d, out, out=out)
    return out


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
    """MuJoCo render state for one model: a private copy of the model (its geom groups are rewritten per
    frame for culling; the sim's model is never touched) and of the state, the GPU context (textures,
    height fields), the scene and its options."""

    def __init__(self, model: mujoco.MjModel):
        self.src = model                                  # the sim's model (a reset brings a new one)
        self.model = m = copy.copy(model)
        self.data = mujoco.MjData(m)
        self.orig_group = np.array(m.geom_group, dtype=np.int32)
        base = self.orig_group.copy()
        base[base == CULL_GROUP] = 2                      # liquid: drawn with the targets in the copy
        self.base_group = base
        self.rbound = np.array(m.geom_rbound, dtype=np.float64)
        self.never = (np.asarray(m.geom_rgba)[:, 3] <= 0.0) & (np.asarray(m.geom_matid) < 0)   # the walls
        hf = np.flatnonzero(np.asarray(m.geom_type) == int(mujoco.mjtGeom.mjGEOM_HFIELD))
        # bodies parked far below the map (spare rockfall boulders and spires) are never drawn
        self.zmin = float(np.asarray(m.geom_pos)[hf, 2].min()) - 5.0 if len(hf) else -1e9
        self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.con)
        self.con.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        self.scn = mujoco.MjvScene(m, maxgeom=int(m.ngeom) + 500)
        flags = self.scn.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0          # the sky texture has its own horizon haze
        flags[mujoco.mjtRndFlag.mjRND_FOG] = 0           # distance haze is applied from depth (_haze)
        self.con.shadowClip = SHADOW_CLIP_M
        # science target marker capsules: what LiDAR, the spectrometer and machine vision see. Look frames
        # that draw the visual-only decoration (group 4) show the natural target instead (world.py)
        self.markers = np.array([g for g in range(m.ngeom)
                                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("target_")],
                                dtype=np.int64)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = _GEOM_GROUPS
        self.opt.sitegroup[:] = 0
        self.max_w = int(min(MAX_W, m.vis.global_.offwidth))
        self.max_h = int(min(MAX_H, m.vis.global_.offheight))

    def snapshot(self, d) -> None:
        """Copy the state the picture needs (sim lock held: microseconds)."""
        self.data.qpos[:] = d.qpos
        if self.model.nmocap:
            self.data.mocap_pos[:] = d.mocap_pos
            self.data.mocap_quat[:] = d.mocap_quat
        self.data.time = d.time

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
    frame: int = 0                  # frame number (one sequence for every renderer)
    view: str = "fpv"
    rover: str = ""
    dust: float = 0.0               # dust storm veil in this frame (0 to 1)


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
        self.last: dict | None = None         # what the last frame drew (segment_ids re-renders it)

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
        if self._gfx is not None and self._gfx.src is model:
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
              depth: bool = False, hide_groups: tuple = (), smooth: bool = True, look: bool = True,
              shadows: bool | None = None) -> Frame:
        """Render one view. runner given: hold runner.locked() only to copy the state.
        runner None: the caller already holds the sim lock (or nothing else touches sim).
        look False: no shadows and no distance haze (machine vision frames: cheaper).
        shadows False: a look frame without the sun's shadows (the detector's cheap frames)."""
        self.ensure()
        try:
            return self._frame(runner, sim, rover_id, view, w, h, depth, tuple(hide_groups), smooth, look, shadows)
        except (UnknownRover, ValueError):
            raise
        except Exception:
            self.release()           # GL trouble: start from a fresh context next time
            raise

    def _frame(self, runner, sim, rover_id, view, w, h, depth, hide_groups, smooth, look=True,
               shadows=None) -> Frame:
        for _attempt in range(3):
            g = self._gfx_for(sim.model)
            hold = runner.locked() if runner is not None else contextlib.nullcontext(sim)
            with hold as s:
                if s.model is not g.src:        # a reset swapped the model in between: rebuild, retry
                    sim = s
                    continue
                rover = _find_rover(s, rover_id)
                g.snapshot(s.data)
                t = float(s.data.time)
                dust = _dust_level(s) if look else 0.0
            break
        else:
            raise RuntimeError("the model kept changing while rendering")
        m, d = g.model, g.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        pos, _R, yaw = _rover_pose(sim, rover, d)
        heading = self._chase_heading(sim, rover_id, yaw) if (view == "chase" and smooth) else None
        spec = rig(view, sim, rover, heading, data=d)
        w = max(MIN_SIDE, min(int(w), g.max_w))
        h = max(MIN_SIDE, min(int(h), g.max_h))
        shadow_on = bool(look and SHADOWS and (shadows is None or shadows))
        keep = _cull_mask(d.geom_xpos, g.rbound, spec, w, h, shadow_on)
        keep &= (d.geom_xpos[:, 2] + g.rbound > g.zmin) & ~g.never
        if hide_groups:
            keep &= ~np.isin(g.orig_group, hide_groups)
        if look and 4 not in hide_groups and len(g.markers):
            keep[g.markers] = False
        m.geom_group[:] = np.where(keep, g.base_group, CULL_GROUP)
        _apply_free_camera(g.cam, spec)
        mujoco.mjv_updateScene(m, d, g.opt, None, g.cam, mujoco.mjtCatBit.mjCAT_ALL.value, g.scn)
        _set_frustum(g.scn, spec)
        g.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadow_on else 0
        if shadow_on:
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
        num = next(_FRAME_SEQ)
        self.last = {"w": w, "h": h, "spec": spec}
        rgb = rgb[::-1]                                     # OpenGL rows start at the bottom
        dep = _linear_depth(raw[::-1], spec.near, spec.far) if raw is not None else None
        if look:
            rgb = _haze(rgb, dep, spec.fovy, dust)
        return Frame(np.ascontiguousarray(rgb), dep if depth else None, spec, t, pos, yaw,
                     frame=num, view=view, rover=rover_id, dust=dust)

    def segment_ids(self) -> np.ndarray:
        """Ground truth for evaluating the detector OFFLINE (never used for decisions): geom id per pixel
        (-1 = sky or nothing) of the frame this renderer drew last, from the same state, camera and
        culling. Uses the non-multisampled window buffer where there is one (OSMesa), so ids are not
        blended at object edges."""
        if self.last is None or self._gfx is None:
            raise RuntimeError("no frame rendered yet")
        g, w, h = self._gfx, self.last["w"], self.last["h"]
        flags = g.scn.flags
        keep = {k: int(flags[k]) for k in (mujoco.mjtRndFlag.mjRND_SEGMENT, mujoco.mjtRndFlag.mjRND_IDCOLOR,
                                           mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_SKYBOX)}
        flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        rect = mujoco.MjrRect(0, 0, w, h)
        self._ctx.make_current()
        window = bool(g.con.windowAvailable)
        try:
            if window:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW.value, g.con)
            mujoco.mjr_render(rect, g.scn, g.con)
            px = np.empty((h, w, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(px, None, rect, g.con)
        finally:
            if window:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, g.con)
            for k, v in keep.items():
                flags[k] = v
        seg = px[::-1].astype(np.int64)
        seg = seg[..., 0] + seg[..., 1] * 256 + seg[..., 2] * 65536          # scene geom index + 1, 0 = none
        geom_t = int(mujoco.mjtObj.mjOBJ_GEOM)
        lut = np.full(g.scn.ngeom + 1, -1, dtype=np.int64)
        for i in range(g.scn.ngeom):
            sg = g.scn.geoms[i]
            if sg.segid >= 0 and sg.objtype == geom_t and sg.segid + 1 < len(lut):
                lut[sg.segid + 1] = sg.objid
        seg[(seg < 0) | (seg >= len(lut))] = 0
        return lut[seg]


@dataclass
class _Stream:
    """One view a client keeps asking for (protocol v8): rendered in the background, newest frame kept."""
    runner: object
    rover: str
    view: str
    w: int
    h: int
    last_req: float                       # wall (monotonic) time of the latest request
    req_dt: float = 0.25                  # smoothed time between requests
    cost: float = 0.1                     # smoothed render + encode time (s)
    due: float = 0.0                      # next render may start then
    started: float = -1e9                 # when the last render started
    served: bool = True                   # the newest frame went out at least once
    jpeg: bytes | None = None
    num: int = 0                          # frame number
    t: float = 0.0                        # sim time of the frame
    wall: float = 0.0                     # when it was rendered (monotonic)
    dets: str = "[]"                      # X-Detections of exactly this frame
    error: str | None = None
    err_unavailable: bool = False         # the error was "no OpenGL here"
    first: Future = field(default_factory=Future)


@dataclass
class _VisionRun:
    """The onboard detectors of one running sim (server.SimRunner): a small render per rover at VISION_HZ."""
    runner: object
    due: dict = field(default_factory=dict)       # rover id -> next detector render (monotonic)
    count: int = 0


_STOP, _WAKE = object(), object()


class RenderThread:
    """The camera's render thread: owns one GLRenderer, runs queued jobs one at a time (they go first),
    and in between renders the streamed views and the rovers' detector frames that are due."""

    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self._mu = threading.Lock()
        self._pending: dict = {}              # coalescing key -> Future (identical queued requests share one)
        self._thread: threading.Thread | None = None
        self.gl = GLRenderer()
        self.streams: dict = {}               # (id(runner), rover, view, w, h) -> _Stream
        self.vision: dict = {}                # id(runner) -> _VisionRun
        self.stream_frames = 0
        self.vision_frames = 0
        self.vision_cost = 0.08               # smoothed time of one detector render + detection (s)

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="redshift-camera", daemon=True)
            self._thread.start()

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
            self._ensure_thread()
        return fut

    def call(self, fn, *args, timeout: float = RENDER_TIMEOUT_S):
        """Blocking version of submit (runs inline when already on the render thread)."""
        if threading.current_thread() is self._thread:
            return fn(self.gl, *args)
        return self.submit(fn, *args).result(timeout)

    # ---------------------------------------------------------------- background work
    def stream(self, runner, rover: str, view: str, w: int, h: int) -> _Stream:
        """A client asked for this view: register (or refresh) its stream and return it."""
        now = time.monotonic()
        key = (id(runner), rover, view, w, h)
        with self._mu:
            st = self.streams.get(key)
            if st is not None and st.runner is not runner:
                st = None
            new = st is None
            if new:
                if len(self.streams) >= STREAM_MAX:
                    old = min(self.streams, key=lambda k: self.streams[k].last_req)
                    self.streams.pop(old)
                st = self.streams[key] = _Stream(runner, rover, view, w, h, last_req=now, due=now)
            else:
                gap = now - st.last_req
                if gap > 0.02:
                    st.req_dt = gap if gap > 3.0 * st.req_dt else 0.7 * st.req_dt + 0.3 * gap
                st.last_req = now
                st.served = True
                if st.jpeg is not None:     # the next frame should be ready just before the next request
                    interval = min(1.0, max(1.0 / STREAM_MAX_FPS, st.req_dt))
                    due = max(st.started + 1.0 / STREAM_MAX_FPS, now + interval - st.cost - STREAM_LEAD_S)
                    if now - st.wall > interval:
                        due = min(due, max(now, st.started + 1.0 / STREAM_MAX_FPS))   # stale already: now
                    st.due = due
            self._ensure_thread()
        self._jobs.put(_WAKE)
        return st

    def status(self) -> dict:
        """What the render thread is doing (GET /api/vision/stats, for transparency)."""
        now = time.monotonic()
        with self._mu:
            streams = [{"rover": st.rover, "view": st.view, "w": st.w, "h": st.h,
                        "requests_per_s": round(1.0 / max(st.req_dt, 1e-3), 1),
                        "render_ms": round(st.cost * 1000.0), "frame": st.num,
                        "age_ms": round((now - st.wall) * 1000.0) if st.jpeg is not None else None}
                       for st in self.streams.values()]
        return {"streams": streams, "stream_frames": self.stream_frames, "detector_frames": self.vision_frames,
                "detector_hz_per_rover": VISION_HZ, "detector_size": [VISION_W, VISION_H]}

    def start_vision(self, runner) -> None:
        if VISION_HZ <= 0:
            return
        with self._mu:
            self.vision[id(runner)] = _VisionRun(runner)
            self._ensure_thread()
        self._jobs.put(_WAKE)

    def stop_vision(self, runner) -> None:
        with self._mu:
            self.vision.pop(id(runner), None)

    def _next_task(self, now: float):
        """(due time, kind, item) of the most urgent background task, or None. Drops idle streams and
        stopped sims."""
        best = None
        with self._mu:
            for key, st in list(self.streams.items()):
                if now - st.last_req > STREAM_IDLE_S or _stopped(st.runner):
                    self.streams.pop(key)
                    continue
                if st.error is not None:
                    continue
                due = st.due if st.served else max(st.due, st.wall + STREAM_REFRESH_S)
                if best is None or due < best[0]:
                    best = (due, "stream", st)
            if best is not None and best[0] <= now:
                return best                    # a view someone is watching goes before the detector renders
            first_stream = best
            for key, vr in list(self.vision.items()):
                if _stopped(vr.runner):
                    self.vision.pop(key)
                    continue
                sim = vr.runner.sim
                for r in sim.rovers:
                    due = vr.due.setdefault(r.id, now + 0.1 * len(vr.due))
                    if best is None or due < best[0]:
                        best = (due, "vision", (vr, r.id))
        if best is not None and best[1] == "vision" and first_stream is not None \
                and first_stream[0] < max(now, best[0]) + 1.2 * self.vision_cost:
            return first_stream                # a detector render now would make that view's next frame late
        return best

    def _render_stream(self, st: _Stream) -> None:
        from . import vision
        t0 = time.monotonic()
        st.started = t0
        try:
            detect = st.view == "fpv" and vision.ENABLED
            fr = self.gl.frame(st.runner, st.runner.sim, st.rover, st.view, st.w, st.h, depth=detect)
            dets = "[]"
            if detect:
                res = vision.detect_frame(fr, source="stream")
                vision.publish(st.runner.sim, st.rover, res)
                dets = res.header()
            jpg = encode_jpeg(fr.rgb)
        except BaseException as e:            # noqa: BLE001 (answered as 503 by the endpoint)
            st.error = f"{type(e).__name__}: {e}" if not isinstance(e, RenderUnavailable) else str(e)
            st.err_unavailable = isinstance(e, RenderUnavailable)
            if not st.first.done():
                st.first.set_exception(e)
            return
        now = time.monotonic()
        with self._mu:
            st.jpeg, st.num, st.t, st.wall, st.dets, st.served = jpg, fr.frame, fr.t, now, dets, False
            st.cost = 0.7 * st.cost + 0.3 * (now - t0)
            # the next one is timed by the next request (stream()); nobody fetching: refresh now and then
            st.due = now + STREAM_REFRESH_S
        self.stream_frames += 1
        if not st.first.done():
            st.first.set_result(True)

    def _render_vision(self, vr: _VisionRun, rid: str) -> None:
        from . import vision
        now = time.monotonic()
        period = 1.0 / VISION_HZ
        vr.due[rid] = now + period
        sim = vr.runner.sim
        rover = next((r for r in sim.rovers if r.id == rid), None)
        brain = getattr(rover, "vision", None)
        last = getattr(brain, "latest", None)
        if last is not None and last.source == "stream" and now - last.wall < 0.9 * period:
            return                             # the streamed fpv view already gave this rover fresh detections
        try:
            fr = self.gl.frame(vr.runner, sim, rid, "fpv", VISION_W, VISION_H, depth=True, smooth=False,
                               shadows=False)
        except UnknownRover:
            vr.due.pop(rid, None)
            return
        except RenderUnavailable:
            vr.due[rid] = now + GL_RETRY_S
            return
        res = vision.detect_frame(fr, source="detector")
        vision.publish(sim, rid, res)
        self.vision_cost = 0.8 * self.vision_cost + 0.2 * (time.monotonic() - now)
        vr.count += 1
        self.vision_frames += 1
        if VISION_STATS_EVERY and vr.count % VISION_STATS_EVERY == 0:
            try:                               # transparency only: compare with the truth, never decide on it
                vision.stats_for(sim).add(res, self.gl.segment_ids(), self.gl._gfx.model)
            except Exception as e:             # noqa: BLE001
                _log(f"vision stats failed: {type(e).__name__}: {e}")

    def _run(self) -> None:
        while True:
            now = time.monotonic()
            task = self._next_task(now)
            wait = 0.5 if task is None else min(0.5, max(0.0, task[0] - now))
            try:
                item = self._jobs.get(timeout=wait) if wait > 0 else self._jobs.get_nowait()
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            if item is _WAKE:
                continue
            if item is not None:
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
                continue
            if task is None or task[0] > time.monotonic():
                continue
            try:
                if task[1] == "stream":
                    self._render_stream(task[2])
                else:
                    self._render_vision(*task[2])
            except Exception as e:                  # noqa: BLE001 (background work must never kill the thread)
                _log(f"background render failed: {type(e).__name__}: {e}")
                if task[1] == "vision":
                    task[2][0].due[task[2][1]] = time.monotonic() + 5.0
        self.gl.release()

    def shutdown(self, timeout: float = 2.0) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            self._jobs.put(_STOP)
            t.join(timeout)
        self._thread = None


def _stopped(runner) -> bool:
    """A SimRunner that was stopped (reset or shutdown). Runners without the attribute count as running."""
    ev = getattr(runner, "_stop", None)
    return bool(ev is not None and ev.is_set())


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
    """One-off JPEG. With a running sim (runner given) an fpv frame also goes through the rover's object
    detector, so whoever asked (Gemma's anomaly check) and the rover's `vision` see the same frame."""
    from . import vision
    detect = runner is not None and view == "fpv" and vision.ENABLED
    fr = gl.frame(runner, sim, rover_id, view, w, h, depth=detect)
    if detect:
        vision.publish(sim, rover_id, vision.detect_frame(fr, source="gemma"))
    return encode_jpeg(fr.rgb)


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


def segment_ids(sim, rover_id: str, view: str = "fpv", w: int = DEFAULT_W, h: int = DEFAULT_H, *,
                shadows: bool | None = None) -> tuple:
    """OFFLINE evaluation only: (Frame with depth, geom id per pixel) of one look frame and its ground
    truth segmentation, rendered from the same state on the render thread (caller owns the sim)."""
    def job(gl: GLRenderer):
        fr = gl.frame(None, sim, rover_id, view, w, h, depth=True, smooth=False, shadows=shadows)
        return fr, gl.segment_ids()
    return _RT.call(job)


def start_vision(runner) -> None:
    """Start the rovers' onboard detectors for a running sim (server.SimRunner.start). They render on the
    camera thread, stop by themselves when the runner stops, and are skipped if rendering is unavailable."""
    _RT.start_vision(runner)


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


FRAME_HEADERS = "X-Frame, X-Frame-Age-Ms, X-Detections"


@router.get("/api/camera/{rover_id}.jpg")
async def camera_jpg(rover_id: str, view: str = "fpv", w: int = DEFAULT_W, h: int = DEFAULT_H):
    """PROTOCOL.md: live JPEG from a rover's camera. 404 unknown rover, 503 if rendering is unavailable.
    Protocol v8: answers at once with the newest frame of this view and size (rendered in the background
    while someone keeps asking), with X-Frame, X-Frame-Age-Ms and X-Detections (fpv only; [] for chase,
    which is not an onboard camera)."""
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
    st = _RT.stream(runner, rover_id, view, w, h)
    try:
        if st.jpeg is None and st.error is None:
            # shield: a timed out request must not cancel the stream's first frame
            await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(st.first)), RENDER_TIMEOUT_S)
    except UnknownRover:
        return _error(404, f"unknown rover {rover_id!r}")
    except asyncio.TimeoutError:
        return _error(503, "camera render timed out")
    except Exception:           # noqa: BLE001 (reported from st.error below)
        pass
    with _RT._mu:
        jpg, num, wall, dets, err = st.jpeg, st.num, st.wall, st.dets, st.error
        if err is not None:
            _RT.streams.pop((id(runner), rover_id, view, w, h), None)     # the next request tries again
    if jpg is None or err is not None:
        if st.err_unavailable:
            return _error(503, f"rendering unavailable: {err}")
        if err is not None and err.startswith("UnknownRover"):
            return _error(404, f"unknown rover {rover_id!r}")
        _log(f"render failed: {err}")
        return _error(503, f"render failed: {err}")
    age = max(0.0, (time.monotonic() - wall) * 1000.0)
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store", "X-Frame": str(num), "X-Frame-Age-Ms": f"{age:.0f}",
                             "X-Detections": dets, "Access-Control-Expose-Headers": FRAME_HEADERS})
