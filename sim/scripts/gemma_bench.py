"""Benchmark the Gemma advisory layer against a real Ollama: latency, validity, sample answers.

Run from sim/ (any machine with Ollama; no sim needed):
    python scripts/gemma_bench.py --url http://localhost:11434 --model gemma4:e2b --n 10
    python scripts/gemma_bench.py --image r1.jpg          # a real rover camera frame (see GEMMA.md)
    python scripts/gemma_bench.py --fake                  # dry run without a model

It sends the same prompts, JSON schemas and validation the rovers use (redshift/gemma.py):
rover choices, anomaly interpretations (with an image) and operator commands, n of each.
Answers are measured even when slower than GEMMA_TIMEOUT_S, so you see the real latency; the
summary says how many would have been dropped by the live timeout. At the end it prints one
block between two "8<" lines: paste that block back.
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redshift.gemma import (DEFAULT_MODEL, FakeBackend, GemmaConfig, GemmaError, OllamaBackend,  # noqa: E402
                            build_prompt, estimate_tokens, latency_summary, make_choice_request,
                            make_command_request, make_interpretation_request, run_request)

# ---------------------------------------------------------------- realistic scenarios (sim sensor names)
BASE = {"battery": 0.72, "state": "exploring", "lidar_range": 8.0, "lidar_hits": 812, "pitch": 3.1, "roll": -1.2,
        "wheel_speed": 0.62, "ground_speed": 0.58, "slip": 0.06, "motor_load": 0.41, "slope_limit_deg": 20.0,
        "home_distance": 31.2}

CHOICES = [
    ("normal", dict(BASE), [
        {"id": "f1", "kind": "frontier", "summary": "frontier 9 m north, 34 cells, slope max 12 deg, battery after return 58%"},
        {"id": "f2", "kind": "frontier", "summary": "frontier 21 m east, 80 cells, slope max 23 deg, battery after return 41%"},
        {"id": "f3", "kind": "frontier", "summary": "frontier 6 m west, 8 cells, slope max 6 deg, battery after return 66%"},
        {"id": "home", "kind": "return", "summary": "lander 31 m south-west, battery on arrival 60%"},
    ], {"f1", "f3"}),
    ("low battery", {**BASE, "battery": 0.24, "home_distance": 22.0}, [
        {"id": "f1", "kind": "frontier", "summary": "frontier 14 m south, 50 cells, slope max 9 deg, battery after return 3%"},
        {"id": "home", "kind": "return", "summary": "lander 22 m east, battery on arrival 17%"},
        {"id": "wait", "kind": "wait", "summary": "stay here, battery drains 1% per minute"},
    ], {"home"}),
    ("target seen", {**BASE, "battery": 0.55, "spectro": {"id": "t3", "dist": 6.2}}, [
        {"id": "t3", "kind": "target", "summary": "ice target 6.2 m north-east, slope max 8 deg, battery after return 47%"},
        {"id": "f1", "kind": "frontier", "summary": "frontier 15 m west, 60 cells, slope max 10 deg, battery after return 40%"},
        {"id": "home", "kind": "return", "summary": "lander 25 m south, battery on arrival 48%"},
    ], {"t3"}),
    ("dust", {**BASE, "lidar_range": 3.1, "lidar_hits": 210}, [
        {"id": "f1", "kind": "frontier", "summary": "frontier 12 m north, 40 cells, slope max 11 deg, battery after return 52%"},
        {"id": "wait", "kind": "wait", "summary": "stop until the LiDAR range is back to 8 m"},
        {"id": "home", "kind": "return", "summary": "lander 18 m south, battery on arrival 62%"},
    ], {"wait", "home", "f1"}),
]

INTERPRETATIONS = [
    ("wheel slip", {**BASE, "wheel_speed": 0.62, "ground_speed": 0.08, "slip": 0.46, "motor_load": 0.71},
     {"type": "wheel_slip", "detail": "wheels spinning, rover not moving", "duration_s": 2.4},
     ["reroute", "back_out", "proceed_slowly", "wait"], {"back_out", "reroute"}),
    ("new obstacle", {**BASE, "ground_speed": 0.0, "motor_load": 0.95, "pitch": 4.0},
     {"type": "new_obstacle", "detail": "LiDAR shows a 0.6 m high steep spot ahead", "distance_m": 1.8},
     ["reroute", "back_out", "proceed_slowly", "wait"], {"reroute", "back_out"}),
    ("low visibility", {**BASE, "lidar_range": 2.4, "lidar_hits": 140},
     {"type": "low_visibility", "detail": "LiDAR returns dropped", "nominal_range_m": 8.0},
     ["proceed_slowly", "wait", "reroute"], {"proceed_slowly", "wait"}),
    ("unclear tilt", {**BASE, "pitch": 14.0, "roll": 9.0, "slip": 0.12, "motor_load": 0.6},
     {"type": "unexpected_tilt", "detail": "tilt rose 9 deg in 1 s on a mapped flat cell"},
     ["reroute", "back_out", "wait"], {"back_out", "wait", "ask_operator"}),
]

FLEET = {"rovers": [{"id": "r1", "name": "Rover 1", "color": "#ff7a45", "x": 3.2, "y": -1.0, "state": "exploring",
                     "battery": 0.64},
                    {"id": "r2", "name": "Rover 2", "color": "#36cfc9", "x": -8.5, "y": 12.1, "state": "returning",
                     "battery": 0.31},
                    {"id": "r3", "name": "Rover 3", "color": "#9254de", "x": 15.0, "y": 4.4,
                     "state": "investigating", "battery": 0.82}],
         "targets": [{"id": "t4", "kind": "ice", "x": 12.0, "y": -4.5, "status": "found"},
                     {"id": "t2", "kind": "basalt", "x": -3.0, "y": 9.0, "status": "sampled"}],
         "lander": [0.0, 0.0], "map_size": 80}

COMMANDS = [
    ("rover 2 go to 10, -5", ("go_to", "r2")),
    ("send the orange one back home", ("return_home", "r1")),
    ("everyone stop", ("hold", "all")),
    ("r3, explore towards the north east", ("explore_direction", "r3")),
    ("can the purple rover check out target 4", ("sample_target", "r3")),
    ("keep away from 12, -4, about 5 m", ("avoid_area", None)),
    ("all rovers carry on", ("resume", "all")),
    ("do the thing over there", ("unclear", None)),
]


def synthetic_jpeg() -> tuple:
    """A 320 x 240 test picture (ground, sky, one rock) if Pillow is installed, else (None, None)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None, None
    img = Image.new("RGB", (320, 240), (150, 98, 70))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 320, 90], fill=(205, 170, 140))
    d.ellipse([125, 120, 205, 185], fill=(70, 62, 58))
    d.ellipse([140, 128, 175, 150], fill=(98, 88, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue(), "synthetic 320x240 (ground, sky, one dark rock)"


def jpeg_size(data: bytes) -> str:
    try:
        from PIL import Image
        w, h = Image.open(io.BytesIO(data)).size
        return f"{w}x{h}"
    except Exception:                         # noqa: BLE001
        return "?x?"


def short(text: str, n: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3] + "..."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=os.environ.get("GEMMA_BENCH_URL", "http://localhost:11434"))
    ap.add_argument("--model", default=os.environ.get("GEMMA_MODEL", DEFAULT_MODEL))
    ap.add_argument("--n", type=int, default=10, help="calls per kind (choice, interpretation, command)")
    ap.add_argument("--image", help="JPEG for the interpretation prompts (default: a synthetic picture)")
    ap.add_argument("--no-image", action="store_true", help="interpretations without an image")
    ap.add_argument("--timeout", type=float, default=60.0, help="per call; long so slow answers are measured")
    ap.add_argument("--live-timeout", type=float, default=float(os.environ.get("GEMMA_TIMEOUT_S", 4.0)),
                    help="GEMMA_TIMEOUT_S of the live system (only for the report)")
    ap.add_argument("--budget-ms", type=float, default=float(os.environ.get("GEMMA_BUDGET_MS", 3000)))
    ap.add_argument("--num-predict", type=int, default=96)
    ap.add_argument("--think", action="store_true", help="let the model think (slower)")
    ap.add_argument("--kinds", default="choice,interpretation,command")
    ap.add_argument("--fake", action="store_true", help="use the fake backend (no Ollama)")
    ap.add_argument("--quiet", action="store_true", help="only the summary block")
    args = ap.parse_args(argv)

    cfg = GemmaConfig(url=args.url, fallback_urls=(), model=args.model, timeout_s=args.timeout,
                      budget_ms=args.budget_ms, num_predict=args.num_predict, think=args.think)
    backend = FakeBackend(0.05) if args.fake else OllamaBackend.from_config(cfg)
    say = (lambda *a, **k: None) if args.quiet else print

    version = "n/a"
    details = ""
    if isinstance(backend, OllamaBackend):
        try:
            version = backend._request(backend.url, "/api/version", None, 3.0).get("version", "?")
        except GemmaError as e:
            print(f"Cannot reach Ollama at {args.url}: {e}")
            print("Is Ollama running? Windows: start it from the Start menu (tray icon). Then: ollama list")
            return 2
    state, detail = backend.check(5.0)
    if state != "ready":
        print(f"Ollama at {args.url}: {state}. {detail}")
        return 2
    if isinstance(backend, OllamaBackend):
        try:
            show = backend._request(backend.url, "/api/show", {"model": backend.model}, 10.0)
            dt = show.get("details") or {}
            caps = show.get("capabilities") or []
            details = " ".join(x for x in (dt.get("parameter_size"), dt.get("quantization_level")) if x)
            if caps:
                details += f"  capabilities: {', '.join(caps)}"
        except GemmaError:
            pass

    say(f"Model {backend.model} at {backend.url} ({detail}). Warming up (first load can take 30 s)...")
    t0 = time.perf_counter()
    err = backend.warmup(300.0)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    if err:
        print(err)
        return 2
    say(f"Warm-up {warm_ms:.0f} ms")

    image, image_note = None, "none"
    if not args.no_image:
        if args.image:
            with open(args.image, "rb") as f:
                image = f.read()
            image_note = f"{os.path.basename(args.image)} {jpeg_size(image)} {len(image) // 1024} KB"
        else:
            image, image_note = synthetic_jpeg()
            image_note = image_note or "none (Pillow not installed)"

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    rows = {k: [] for k in kinds}             # (result, meta, sensible, scenario name)
    ticket = 0
    for kind in kinds:
        for i in range(max(1, args.n)):
            ticket += 1
            if kind == "choice":
                name, perc, cands, good = CHOICES[i % len(CHOICES)]
                req = make_choice_request(ticket, "r1", 100.0 + i, perc, cands, cands[0]["id"])
            elif kind == "interpretation":
                name, perc, anom, opts, good = INTERPRETATIONS[i % len(INTERPRETATIONS)]
                req = make_interpretation_request(ticket, "r2", 100.0 + i, perc, anom, image, opts)
            elif kind == "command":
                name, good = COMMANDS[i % len(COMMANDS)]
                req = make_command_request(ticket, 100.0 + i, name, FLEET)
            else:
                print(f"unknown kind {kind}")
                return 2
            meta = {}
            res, _ = run_request(backend, req, cfg, meta)
            if kind == "choice":
                sensible = res.ok and res.choice in good
                what = f"{res.choice} ({res.confidence:.2f}): {res.reason}"
            elif kind == "interpretation":
                sensible = res.ok and (res.model_action in good or res.action in good)
                what = (f"{res.classification} -> {res.action} (model {res.model_action}, "
                        f"{res.confidence:.2f}): {res.reason}")
            else:
                c = res.command or {}
                sensible = res.ok and c.get("type") == good[0] and (good[1] is None or c.get("rover") == good[1])
                what = f"{c} ({res.confidence:.2f}): {res.question if c.get('type') == 'unclear' else res.ack}"
            rows[kind].append((res, meta, sensible, name))
            status = "ok " if res.ok else "BAD"
            say(f"[{kind} {i + 1}/{args.n}] {res.latency_ms:7.0f} ms {status} {name}: "
                f"{short(what) if res.ok else short(res.error)}")

    # ---------------------------------------------------------------- summary
    sys_p, user_p = build_prompt(make_choice_request(0, "r1", 0.0, CHOICES[0][1], CHOICES[0][2]))
    lines = ["----- 8< ----- Redshift Gemma bench: paste from here -----",
             f"when      {time.strftime('%Y-%m-%d %H:%M')}  python {platform.python_version()} on {platform.platform()}",
             f"ollama    {version} at {backend.url}",
             f"model     {backend.model} (requested {args.model}) {details}".rstrip(),
             f"warm-up   {warm_ms:.0f} ms (model load)",
             f"image     {image_note}",
             f"prompt    choice prompt ~{estimate_tokens(sys_p + user_p)} tokens (estimate), num_predict "
             f"{args.num_predict}, think {'on' if args.think else 'off'}",
             f"budget    {args.budget_ms:.0f} ms per call; live timeout {args.live_timeout:g} s",
             f"{'kind':<15}{'n':>3}{'valid':>8}{'sensible':>10}{'avg_ms':>8}{'p50_ms':>8}{'p95_ms':>8}"
             f"{'max_ms':>8}{'>budget':>9}{'>timeout':>10}{'tok_in':>8}{'tok_out':>8}{'tok/s':>7}"]
    all_lat = []
    for kind in kinds:
        rs = rows[kind]
        lat = [r.latency_ms for r, _, _, _ in rs]
        all_lat += lat
        s = latency_summary(lat, args.budget_ms)
        valid = sum(1 for r, _, _, _ in rs if r.ok)
        sens = sum(1 for _, _, ok, _ in rs if ok)
        late = sum(1 for x in lat if x > args.live_timeout * 1000.0)
        tin = [m.get("prompt_eval_count") for _, m, _, _ in rs if m.get("prompt_eval_count")]
        tout = [m.get("eval_count") for _, m, _, _ in rs if m.get("eval_count")]
        rates = [m["eval_count"] / (m["eval_duration"] / 1e9) for _, m, _, _ in rs
                 if m.get("eval_count") and m.get("eval_duration")]
        avg = (lambda v: f"{sum(v) / len(v):.0f}" if v else "-")
        lines.append(f"{kind:<15}{len(rs):>3}{f'{valid}/{len(rs)}':>8}{f'{sens}/{len(rs)}':>10}{s['avg_ms']:>8.0f}"
                     f"{s['p50_ms']:>8.0f}{s['p95_ms']:>8.0f}{s['max_ms']:>8.0f}{s['over_budget']:>9}{late:>10}"
                     f"{avg(tin):>8}{avg(tout):>8}{(f'{sum(rates) / len(rates):.0f}' if rates else '-'):>7}")
    s = latency_summary(all_lat, args.budget_ms)
    lines.append(f"{'all':<15}{s['n']:>3}{'':>18}{s['avg_ms']:>8.0f}{s['p50_ms']:>8.0f}{s['p95_ms']:>8.0f}"
                 f"{s['max_ms']:>8.0f}{s['over_budget']:>9}")
    lines.append("samples:")
    for kind in kinds:
        shown = 0
        for r, _, _, name in rows[kind]:
            if not r.ok or shown >= 3:
                continue
            shown += 1
            if kind == "choice":
                lines.append(f"  choice/{name}: {r.choice} ({r.confidence:.2f}) \"{short(r.reason, 140)}\"")
            elif kind == "interpretation":
                lines.append(f"  interpretation/{name}: {r.classification} -> {r.action} ({r.confidence:.2f}) "
                             f"\"{short(r.reason, 140)}\"")
            else:
                c = r.command or {}
                say_ = r.question if c.get("type") == "unclear" else r.ack
                lines.append(f"  command/{name!r}: {c} ({r.confidence:.2f}) \"{short(say_, 100)}\"")
    errs = [(k, r.error, m.get("content", "")) for k in kinds for r, m, _, _ in rows[k] if not r.ok]
    if errs:
        lines.append(f"errors ({len(errs)}, first 5):")
        for k, e, raw in errs[:5]:
            lines.append(f"  {k}: {short(e, 140)}" + (f" | raw: {short(raw, 120)}" if raw else ""))
    lines.append("----- 8< ----- to here -----")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
