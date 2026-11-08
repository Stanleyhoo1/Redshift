"""Offline evaluation of the onboard object detector (redshift/vision.py) against MuJoCo segmentation.

The detector only ever sees RGB + depth; the segmentation render here is the ground truth for scoring
(same state, same camera). Views: the mast camera facing every kind of object at several distances and
angles, other rovers and dropped boulders in front of it, random poses all over the map, on a few seeds.

    python scripts/vision_eval.py                       # seeds 42 7 3, both frame types
    python scripts/vision_eval.py --seeds 42 --n 40 --draw /tmp/vision   # also save frames with boxes
    python scripts/vision_eval.py --dust                # the same views in a dust storm

Frame types: "detector" (the rovers' own cheap 256 x 192 render without shadows) and "stream" (the UI's
320 x 240 look frame with shadows). Prints precision and recall per class at confidence >= 0.5 and the
detector's time per frame.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from redshift import camera, vision  # noqa: E402
from redshift.config import Config  # noqa: E402
from redshift.sim import Simulation  # noqa: E402
from redshift.world import rover_spawn_height  # noqa: E402

COLOURS = {"rock": (170, 170, 170), "boulder": (255, 255, 255), "spire": (200, 120, 255), "creature": (60, 255, 140),
           "water": (60, 140, 255), "mineral": (230, 180, 255), "ice": (180, 230, 255), "vent": (255, 140, 40),
           "trace": (190, 255, 60), "rover": (255, 230, 0), "lander": (255, 90, 90)}


def place(sim, rover, x, y, yaw):
    q, a = sim.data.qpos, rover.qadr
    q[a:a + 3] = (x, y, sim.terrain.height_at(x, y) + rover_spawn_height(sim.cfg) + 0.02)
    q[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    sim.data.qvel[rover.vadr:rover.vadr + rover.nv] = 0.0


def ok_spot(sim, x, y) -> bool:
    half = sim.cfg.map_size / 2 - 3
    if abs(x) > half or abs(y) > half:
        return False
    if sim.terrain.water_at(x, y) is not None or sim.terrain.wet_near(x, y, 1.2):
        return False
    return sim.terrain.max_slope_near(x, y, 0.6) < 22.0


def drop_boulders(sim, rng, n=6):
    """Put a few rockfall boulders on open ground (as a rockfall would), for the boulder class."""
    pts = []
    for (mid, _gid, (sx, sy, sz)) in sim.world.boulders[:n]:
        for _ in range(50):
            x, y = rng.uniform(-sim.cfg.map_size / 2 + 6, sim.cfg.map_size / 2 - 6, 2)
            if ok_spot(sim, x, y) and math.hypot(x, y) > 8:
                break
        sim.data.mocap_pos[mid] = (x, y, sim.terrain.height_at(x, y) + 0.45 * sz)    # as hazards.py rests them
        pts.append((x, y))
    return pts


def views(sim, rng, n_random: int):
    """(name, x, y, yaw, others) camera poses; others: [(rover index, x, y, yaw)] placed in view."""
    out = []
    things = [("target", t.x, t.y) for t in sim.terrain.targets]
    things += [("spire", s.x, s.y) for s in sim.terrain.spires]
    things += [("creature", c[0], c[1]) for h in sim.terrain.herds for c in h.members]
    things += [("water", w.cx, w.cy, w.radius) for w in sim.terrain.water if w.kind == "lake"]
    things += [("rock", r.x, r.y) for r in list(sim.terrain.rocks)[::4]]
    things += [("lander", 0.0, 0.0)] * 3
    things += [("boulder", x, y) for x, y in getattr(sim, "_eval_boulders", [])]
    for kind, tx, ty, *extra in things:
        base = float(extra[0]) if extra else 0.0
        for d in ((3.5, 6.0, 10.0, 15.0) if kind != "water" else (3.0, 6.0, 10.0, 16.0)):
            d = d + base
            for _k in range(2 if kind == "water" else 1):
                ang = rng.uniform(-math.pi, math.pi)
                x, y = tx - d * math.cos(ang), ty - d * math.sin(ang)
                if not ok_spot(sim, x, y):
                    continue
                yaw = ang + math.radians(rng.uniform(-20, 20))
                out.append((f"{kind}@{d - base:.0f}", x, y, yaw, []))
    for k in range(max(4, n_random // 4)):          # other rovers in front of the camera
        for _ in range(30):
            x, y = rng.uniform(-40, 40, 2)
            yaw = rng.uniform(-math.pi, math.pi)
            d1, d2 = rng.uniform(3.5, 12.0, 2)
            p1 = (x + d1 * math.cos(yaw + 0.3), y + d1 * math.sin(yaw + 0.3))
            p2 = (x + d2 * math.cos(yaw - 0.35), y + d2 * math.sin(yaw - 0.35))
            if ok_spot(sim, x, y) and ok_spot(sim, *p1) and ok_spot(sim, *p2):
                out.append(("rovers", x, y, yaw, [(1, p1[0], p1[1], rng.uniform(-3, 3)),
                                                  (2, p2[0], p2[1], rng.uniform(-3, 3))]))
                break
    for k in range(n_random):
        for _ in range(30):
            x, y = rng.uniform(-sim.cfg.map_size / 2 + 4, sim.cfg.map_size / 2 - 4, 2)
            if ok_spot(sim, x, y):
                out.append(("random", x, y, rng.uniform(-math.pi, math.pi), []))
                break
    return out


def draw(path, rgb, objects, insts):
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb).resize((rgb.shape[1] * 2, rgb.shape[0] * 2), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    W, H = im.size
    sx, sy = W / rgb.shape[1], H / rgb.shape[0]
    for g in insts:
        if g["counts"]:
            b = g["box"]
            dr.rectangle([b[0] * sx, b[1] * sy, (b[2] + 1) * sx, (b[3] + 1) * sy], outline=(255, 255, 255))
    for o in objects:
        b = o["box"]
        c = COLOURS.get(o["label"], (255, 0, 0))
        dr.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=c, width=2)
        dr.text((b[0] * W + 2, max(0, b[1] * H - 11)), f"{o['label']} {o['conf']:.2f} {o['dist']:.0f}m", fill=c)
    im.save(path)


def run(seeds, n_random, kinds, dust, draw_dir, limit, verbose):
    totals = {k: {"tp": {l: 0 for l in vision.LABELS}, "fp": {l: 0 for l in vision.LABELS},
                  "fn": {l: 0 for l in vision.LABELS}, "ms": [], "frames": 0, "conf": {}} for k in kinds}
    nd = 0
    for seed in seeds:
        sim = Simulation(Config(seed=seed, plan_clock="model"))
        rng = np.random.default_rng(1000 + seed)
        sim._eval_boulders = drop_boulders(sim, rng)
        if dust:
            sim.hazards.trigger("dust_storm")
            sim.data.time += camera.DUST_RAMP_S + 1.0
        r1 = sim.rovers[0]
        vs = views(sim, rng, n_random)
        if limit:
            vs = vs[:limit]
        for vi, (name, x, y, yaw, others) in enumerate(vs):
            for r in sim.rovers[1:]:                 # park the others far away unless placed in view
                place(sim, r, 50.0 - 3 * r.idx, -55.0, 0.0)
            for idx, ox, oy, oyaw in others:
                if idx < len(sim.rovers):
                    place(sim, sim.rovers[idx], ox, oy, oyaw)
            place(sim, r1, x, y, yaw)
            mujoco.mj_forward(sim.model, sim.data)
            for kind in kinds:
                w, h, sh = (camera.VISION_W, camera.VISION_H, False) if kind == "detector" else (320, 240, None)
                fr, seg = camera.segment_ids(sim, r1.id, "fpv", w, h, shadows=sh)
                res = vision.detect_frame(fr, source=kind)
                insts = vision.gt_instances(seg, sim.model, fr.depth)
                m = vision.match(res.objects, insts, w, h)
                T = totals[kind]
                T["frames"] += 1
                T["ms"].append(res.ms)
                for k2 in ("tp", "fp", "fn"):
                    for lab in m[k2]:
                        T[k2][lab] += 1
                for pair in m["confusion"]:
                    T["conf"][pair] = T["conf"].get(pair, 0) + 1
                if verbose and (m["fp"] or m["fn"]):
                    print(f"  seed {seed} view {vi} {name} {kind}: fp {m['fp']} fn {m['fn']} conf {m['confusion']}")
                if draw_dir and (m["fp"] or m["fn"] or vi % 7 == 0) and nd < 400:
                    nd += 1
                    draw(f"{draw_dir}/s{seed}_{vi:03d}_{name.replace('@', '_')}_{kind}.png", fr.rgb, res.objects, insts)
        sim.close()
    return totals


def report(totals) -> dict:
    out = {}
    for kind, T in totals.items():
        print(f"\n{kind} frames: {T['frames']}, detector {np.mean(T['ms']):.1f} ms avg, "
              f"p95 {np.percentile(T['ms'], 95):.1f} ms")
        print(f"  {'class':9s} {'prec':>6s} {'recall':>6s} {'tp':>5s} {'fp':>5s} {'fn':>5s}")
        rows = {}
        for lab in vision.LABELS:
            tp, fp, fn = T["tp"][lab], T["fp"][lab], T["fn"][lab]
            p = tp / (tp + fp) if tp + fp else float("nan")
            r = tp / (tp + fn) if tp + fn else float("nan")
            rows[lab] = {"precision": round(p, 3), "recall": round(r, 3), "tp": tp, "fp": fp, "fn": fn}
            print(f"  {lab:9s} {p:6.2f} {r:6.2f} {tp:5d} {fp:5d} {fn:5d}")
        tp, fp, fn = sum(T["tp"].values()), sum(T["fp"].values()), sum(T["fn"].values())
        print(f"  {'all':9s} {tp / max(tp + fp, 1):6.2f} {tp / max(tp + fn, 1):6.2f} {tp:5d} {fp:5d} {fn:5d}")
        top = sorted(T["conf"].items(), key=lambda kv: -kv[1])[:10]
        print("  confusions (detected -> actually):", ", ".join(f"{a}->{b} {c}" for (a, b), c in top))
        out[kind] = {"classes": rows, "frames": T["frames"], "ms_avg": round(float(np.mean(T["ms"])), 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 3])
    ap.add_argument("--n", type=int, default=30, help="random poses per seed")
    ap.add_argument("--kinds", default="detector,stream")
    ap.add_argument("--dust", action="store_true")
    ap.add_argument("--draw", default="", help="save frames with boxes here")
    ap.add_argument("--limit", type=int, default=0, help="at most this many views per seed")
    ap.add_argument("--json", default="", help="write the summary here")
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    st = camera.renderer_status()
    if not st["ok"]:
        print(f"no rendering here: {st['error']}")
        return 1
    if args.draw:
        Path(args.draw).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    totals = run(args.seeds, args.n, args.kinds.split(","), args.dust, args.draw, args.limit, args.v)
    out = report(totals)
    print(f"\n({time.time() - t0:.0f} s)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
