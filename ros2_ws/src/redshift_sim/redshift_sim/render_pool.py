"""Worker processes for the mock world's ray casting.

The numpy ray-marcher holds the GIL for much of each frame, so one process tops out near one core: with two rovers
and the scout (two 10 Hz RGB-D cameras, two 10 Hz LiDARs, a 5 Hz downward camera; ≈1 s of render work per simulated
second) the lockstep clock could no longer hold 1× real time, and extra render threads only thrash the GIL. Each
worker process holds its own copy of the static terrain (heights, normals, texture: sent once at start-up) and
receives the moving part of the scene with every job: the object arrays from `scene_objects` (boulders, falling
rockslide rocks, robot bodies; ~100 rows). Results are bit-identical to rendering in-process.

`procs=0` renders in the calling thread (no pool), and so does any failure to start one.
"""
from __future__ import annotations

import multiprocessing as mp
import signal

import numpy as np

from .mock_render import Scene, camera_frame, lidar_scan

OBJ_FIELDS = ('c', 'radii', 'yaw', 'albedo', 'emissive', 'kind')
JOB_TIMEOUT_S = 5.0       # a frame takes ≤ 0.1 s; past this a worker is gone: render that frame in-process
_SCENE: Scene | None = None


def scene_objects(scene: Scene) -> dict:
    """Snapshot of what moves or appears at run time: every object array the renderers read."""
    return {k: np.array(getattr(scene, k), copy=True) for k in OBJ_FIELDS}


def _init(heights, size_m, texture, tex_size_m, g):
    global _SCENE
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # the launch's Ctrl+C is the parent's to handle
    _SCENE = Scene(heights, size_m, texture=texture, tex_size_m=tex_size_m, g=g)


def _with_objects(scene: Scene, objs: dict) -> Scene:
    for k in OBJ_FIELDS:
        setattr(scene, k, objs[k])
    return scene


def _render(kind: str, objs: dict, args: tuple, kw: dict):
    scene = _with_objects(_SCENE, objs)
    return camera_frame(scene, *args, **kw) if kind == 'cam' else lidar_scan(scene, *args, **kw)


class RenderPool:
    def __init__(self, scene: Scene, procs: int):
        self.scene = scene
        self.pool = None
        self.procs = 0
        self.error = None
        self.timeouts = 0
        if procs > 0:
            try:
                ctx = mp.get_context('spawn')          # never fork a process that runs DDS threads
                self.pool = ctx.Pool(procs, initializer=_init,
                                     initargs=(scene.h, scene.size_m, scene.texture, scene.tex_size_m, scene.g))
                self.procs = procs
            except (OSError, ValueError, RuntimeError) as e:  # no pool: render in-process
                self.error = e
                self.pool = None

    def _local(self, kind: str, objs: dict, args: tuple, kw: dict):
        local = _with_objects(Scene.__new__(Scene), objs)
        local.__dict__.update({k: v for k, v in self.scene.__dict__.items() if k not in OBJ_FIELDS})
        return camera_frame(local, *args, **kw) if kind == 'cam' else lidar_scan(local, *args, **kw)

    def _run(self, kind: str, objs: dict, args: tuple, kw: dict):
        pool = self.pool
        if pool is None:
            return self._local(kind, objs, args, kw)
        try:
            return pool.apply_async(_render, (kind, objs, args, kw)).get(timeout=JOB_TIMEOUT_S)
        except mp.TimeoutError:                       # a lost job would otherwise stall the lockstep clock
            self.timeouts += 1
            return self._local(kind, objs, args, kw)

    def camera_frame(self, objs: dict, origin, R_cam, w: int, h: int, hfov: float, far: float, scale: float = 0.5,
                     near: float = 0.1, exclude=None):
        """mock_render.camera_frame on the scene with these objects."""
        return self._run('cam', objs, (origin, R_cam, w, h, hfov, far),
                         {'scale': scale, 'near': near, 'exclude': exclude})

    def lidar_scan(self, objs: dict, origin, R_body, spec: dict, exclude=None):
        """mock_render.lidar_scan on the scene with these objects."""
        return self._run('lidar', objs, (origin, R_body, spec), {'exclude': exclude})

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
