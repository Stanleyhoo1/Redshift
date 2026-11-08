"""Camera render time per frame (camera.py through its render thread, plus JPEG encoding).

    python scripts/camera_bench.py                 # 320x240 and 1280x720, fpv and chase
    python scripts/camera_bench.py --n 20 --sizes 320x240,640x480,1280x720
    CAMERA_SHADOWS=0 python scripts/camera_bench.py   # without the sun's shadows

Rovers are spread over the default 120 m map (seed 42) and drive for a few sim seconds first, so
the views show terrain, rocks, spires and creatures. Prints median and p90 wall ms per frame.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redshift import camera  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.sim import Simulation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="frames per size and view (default 12)")
    ap.add_argument("--sizes", default="320x240,1280x720")
    ap.add_argument("--seconds", type=float, default=60.0, help="sim seconds to run first")
    args = ap.parse_args()
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


if __name__ == "__main__":
    sys.exit(main())
