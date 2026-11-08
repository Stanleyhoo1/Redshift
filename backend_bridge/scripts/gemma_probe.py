#!/usr/bin/env python3
"""Live check of the advisory Gemma brain against the real Gemini API (3 realistic Redshift events).

    python3 backend_bridge/scripts/gemma_probe.py [--model gemma-4-26b-a4b-it] [--env /path/to/.env] [--json]

Loads GEMINI_API_KEY from the environment or the repo's .env (the key is never printed). For each event prints the
validated action, which brain decided, the MEASURED cloud round-trip, token counts, validation adjustments and the
PROJECTED on-device latency on orin_nano_15w and agx_orin_60w (a roofline estimate, not a measurement).
Exit code 0 when every event produced a decision (Gemma or local fallback); 1 if the key is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ros2_ws' / 'src' / 'redshift_autonomy'))
from redshift_autonomy.gemma_agent_brain import (  # noqa: E402
    DEFAULT_MODEL, GemmaClient, build_summary, project_latency, summary_json)

FLEET = {'recovery_rover': 'rover_1', 'recovery_available': True, 'drone': 'scout_0', 'drone_agl_m': 12.0}
KNOWN = ['boulder field', 'crater rim', 'rockslide debris']

EVENTS = [
    ('low_confidence', {
        'robot': 'rover_0', 'robot_type': 'rover', 'state': 'CAUTION', 'speed_mps': 0.31, 'speed_limit_mps': 0.35,
        'v_max_mps': 1.0, 'health': 0.92, 'battery': 0.71, 'camera_conf': 0.22, 'lidar_conf': 0.48,
        'uncertainty': 0.62, 'nearest_obstacle_m': 4.1, 'dust': True, 'wind_mps': 11.0,
        'hazard': 'dust storm: visibility ~6 m, wind 11 m/s gusting', 'known_hazards': KNOWN, 'fleet': FLEET}),
    ('critical_damage', {
        'robot': 'rover_0', 'robot_type': 'rover', 'state': 'SAFE_HOLD', 'speed_mps': 0.0, 'speed_limit_mps': 0.0,
        'v_max_mps': 1.0, 'health': 0.18, 'battery': 0.40, 'camera_conf': 0.7, 'lidar_conf': 0.9,
        'uncertainty': 0.3, 'damage': 'left-front wheel motor failed, chassis tilted 14 deg after rockslide impact',
        'known_hazards': KNOWN, 'fleet': FLEET}),
    ('operator_command', {
        'robot': 'rover_0', 'robot_type': 'rover', 'state': 'EXPLORING', 'speed_mps': 0.55, 'speed_limit_mps': 0.8,
        'v_max_mps': 1.0, 'health': 0.95, 'battery': 0.66, 'camera_conf': 0.85, 'lidar_conf': 0.9,
        'uncertainty': 0.15, 'nearest_obstacle_m': 6.0, 'known_hazards': KNOWN, 'fleet': FLEET,
        'operator_text': 'full speed through the boulder field'}),
]
DEVICES = ('orin_nano_15w', 'agx_orin_60w')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--env', default=None, help='path to a .env holding GEMINI_API_KEY (default: repo .env)')
    ap.add_argument('--timeout', type=float, default=8.0)
    ap.add_argument('--json', action='store_true', help='print each Decision as JSON as well')
    args = ap.parse_args()

    client = GemmaClient(model=args.model, timeout_s=args.timeout, env_path=args.env)
    print(f'{client!r}')            # shows key=set|missing, never the key
    if not client.has_key:
        print('GEMINI_API_KEY not found: every decision below is the local fallback')
    for event, ctx in EVENTS:
        summary = summary_json(build_summary(event, ctx))
        d = client.decide(event, ctx)
        print(f'\n=== {event}  (summary {len(summary)} chars)')
        print(f'  action      : {d.action}   speed cap {d.target_speed_mps}   altitude {d.target_altitude_m}'
              f'   target {d.target_robot}')
        print(f'  source      : {d.source}   model {d.model}')
        lat = f'{d.latency_ms:.0f} ms' if d.latency_ms is not None else 'n/a'
        print(f'  cloud RTT   : {lat} (measured)   tokens in/out {d.tokens_in}/{d.tokens_out}')
        print(f'  reasoning   : {d.reasoning_trace}')
        print(f'  adjusted    : {d.adjusted or "none"}')
        if d.error:
            print(f'  error       : {d.error}')
        for dev in DEVICES:
            p = project_latency(d.tokens_in or 0, d.tokens_out or 0, dev, args.model)
            verdict = (f"~{p['total_ms']:.0f} ms (prefill {p['prefill_ms']:.0f} + decode {p['decode_ms']:.0f})"
                       if p['fits'] else f"does NOT fit ({p['required_gb']} GB > {p['usable_gb']} GB usable)"
                       f" -> local rules are the on-device brain")
            print(f'  projected   : {args.model} on {dev}: {verdict}  [estimate, not measured]')
        ref = project_latency(d.tokens_in or 0, d.tokens_out or 0, 'orin_nano_15w', 'gemma-edge-4b')
        print(f"  edge ref    : gemma-edge-4b on orin_nano_15w: ~{ref['total_ms']:.0f} ms "
              f"(fits={ref['fits']}) [estimate]")
        if args.json:
            print('  ' + json.dumps(d.as_dict()))
    return 0 if client.has_key else 1


if __name__ == '__main__':
    sys.exit(main())
