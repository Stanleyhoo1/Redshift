"""Camera speed (camera.py) and what the camera costs the rovers' brains.

    python scripts/camera_bench.py                 # render time per frame: 320x240 and 1280x720, fpv and chase
    python scripts/camera_bench.py --n 20 --sizes 320x240,640x480,1280x720
    CAMERA_SHADOWS=0 python scripts/camera_bench.py   # without the sun's shadows
    python scripts/camera_bench.py --live          # the server's way: a running sim, the rovers' detectors,
                                                   # and a client polling like the UI (protocol v8 streams)

Default mode: rovers spread over the default 120 m map (seed 42) after a few sim seconds of driving;
prints median and p90 wall ms per frame (render, haze, JPEG; one-off renders, no streaming).

--live: a SimRunner steps the sim at SIM_SPEED with the rovers' detectors on (camera.VISION_HZ), then
three phases of --seconds each: nobody watching, a client polling fpv 320 x 240 every 250 ms (the
sensor panel), and one polling fpv 1280 x 720 every 330 ms (the big viewer). For each phase: request
latency, frame age (X-Frame-Age-Ms), new frames per second, detector time, and the brain time per
decision cycle (avg / p95 over all rovers; the budget is 40 ms per 200 ms cycle).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redshift import camera  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.sim import Simulation  # noqa: E402


def frames(args) -> int:
    sim = Simulation(Config(seed=42, plan_clock="model"))
    sim.run_for(args.seconds)
    st = camera.renderer_status()
    if not st["ok"]:
        print(f"no rendering here: {st['error']}")
        return 1
    print(f"backend {st['backend']}, shadows {'on' if camera.SHADOWS else 'off'}, "
          f"shadow map {int(sim.model.vis.quality.shadowsize)} px over +-{camera.SHADOW_CLIP_M:g} m, "
          f"MSAA {int(sim.model.vis.quality.offsamples)}x")
    rids = [r.id for r in sim.rovers]
    for size in args.sizes.split(","):
        w, h = (int(v) for v in size.lower().split("x"))
        for view in camera.VIEWS:
            camera.render_jpeg(sim, rids[0], view, w, h)          # warm up (context, textures)
            ts = []
            for k in range(args.n):
                t0 = time.perf_counter()
                camera.render_jpeg(sim, rids[k % len(rids)], view, w, h)
                ts.append((time.perf_counter() - t0) * 1000.0)
            a = np.array(ts)
            print(f"  {w}x{h} {view:5s}: median {np.median(a):6.0f} ms, p90 {np.percentile(a, 90):6.0f} ms "
                  f"({len(a)} frames)")
    return 0


def live(args) -> int:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from redshift import server, vision
    st = camera.renderer_status()
    if not st["ok"]:
        print(f"no rendering here: {st['error']}")
        return 1
    runner = server.SimRunner(Config.from_env(seed=42))
    server.state.runner = runner
    runner.start()                                  # also starts the rovers' detectors
    app = FastAPI()
    app.include_router(camera.router)
    print(f"backend {st['backend']}, sim speed {runner.cfg.sim_speed:g}x, detectors {camera.VISION_HZ:g} Hz per "
          f"rover ({camera.VISION_W}x{camera.VISION_H}, no shadows), {args.seconds:g} s per phase")
    time.sleep(3.0)

    def brain_window():
        return {r.id: len(r.brain_all) for r in runner.sim.rovers}

    def brain_since(mark):
        vals = []
        for r in runner.sim.rovers:
            vals += r.brain_all[mark[r.id]:]
        a = np.array(vals) if vals else np.zeros(1)
        return a.mean(), np.percentile(a, 95), a.max(), len(vals)

    try:
        with TestClient(app) as client:
            for label, url, period in (("nobody watching", None, None),
                                       ("fpv 320x240 every 250 ms", "/api/camera/r1.jpg?w=320&h=240", 0.25),
                                       ("fpv 1280x720 every 330 ms", "/api/camera/r1.jpg?w=1280&h=720", 0.33)):
                mark, t_sim0 = brain_window(), runner.sim.t
                lat, ages, nums = [], [], []
                end = time.perf_counter() + args.seconds
                while time.perf_counter() < end:
                    if url is None:
                        time.sleep(0.2)
                        continue
                    t0 = time.perf_counter()
                    r = client.get(url)
                    dt = time.perf_counter() - t0
                    if r.status_code == 200:
                        lat.append(dt * 1000.0)
                        ages.append(int(r.headers["X-Frame-Age-Ms"]))
                        nums.append(int(r.headers["X-Frame"]))
                    time.sleep(max(0.0, period - dt))
                avg, p95, mx, n = brain_since(mark)
                speed = (runner.sim.t - t_sim0) / args.seconds
                vstat = vision.stats_for(runner.sim).summary()["ms"]
                line = f"  {label:27s} brain avg {avg:5.2f} ms, p95 {p95:5.2f} ms, max {mx:5.1f} ms ({n} cycles), " \
                       f"sim {speed:.2f}x"
                if lat:
                    k = len(lat) // 5                       # skip the first requests (the view starting up)
                    la, ag = np.array(lat[k:]), np.array(ages[k:])
                    fps = len(set(nums[k:])) / (args.seconds * (len(lat) - k) / len(lat))
                    line += (f"\n  {'':27s} request {np.median(la):5.0f} ms (p95 {np.percentile(la, 95):4.0f}), frame age "
                             f"{np.median(ag):4.0f} ms (p95 {np.percentile(ag, 95):4.0f}), {fps:4.1f} new frames/s")
                line += f"\n  {'':27s} detector {vstat['avg']:.0f} ms avg per frame (p95 {vstat['p95']:.0f})"
                print(line, flush=True)
            print(f"  render thread: {camera._RT.status()}")
    finally:
        runner.stop()
        server.state.runner = None
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="frames per size and view (default 12)")
    ap.add_argument("--sizes", default="320x240,1280x720")
    ap.add_argument("--seconds", type=float, default=None,
                    help="sim seconds to run first (default 60), or with --live seconds per phase (default 20)")
    ap.add_argument("--live", action="store_true", help="a running sim with streams, detectors and brain times")
    args = ap.parse_args()
    if args.live:
        args.seconds = 20.0 if args.seconds is None else args.seconds
        return live(args)
    args.seconds = 60.0 if args.seconds is None else args.seconds
    return frames(args)


if __name__ == "__main__":
    sys.exit(main())
