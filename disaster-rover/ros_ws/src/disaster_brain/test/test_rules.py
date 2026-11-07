"""Rule-set test for the rclpy brain: replay TS-sim frames into BrainCore, then feed deliberately bad tool calls.
usage: python3 test/test_rules.py frames.jsonl   (frames from: npx tsx scripts/dump-frames.ts --seed 3 --duration 75 --disasters flood,fire)"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from disaster_brain.brain_core import BrainCore

br = BrainCore()
for line in open(sys.argv[1]):
    br.tick(json.loads(line)['frame'])
b, rd = br.b, br.rd
water = hazard = safe = wall = -1
for i in range(len(b.risk)):
    d = math.hypot(b.xs[i] - rd.x, b.ys[i] - rd.y)
    if water < 0 and b.water[i] > 0.3: water = i
    if hazard < 0 and b.predicted[i] > 0.6 and b.water[i] <= 0.05 and b.pred_water[i] < 0.3: hazard = i
    if safe < 0 and 8 < d < 25 and b.last_seen[i] >= 0 and b.predicted[i] < 0.1 and b.water[i] == 0 and not b.blocked(i) and b.channels[4][i] < 0.2: safe = i
    if wall < 0 and b.obst[i] > 0.9 and b.risk[i] < 0.1 and b.water[i] <= 0.05 and b.pred_water[i] < 0.3 and b.predicted[i] < 0.3: wall = i
P = lambda i: dict(x=round(float(b.xs[i]), 1), y=round(float(b.ys[i]), 1))
probes = rd.status['probes']
pr0 = probes[0] if probes else dict(x=0, y=0)
cases = [
    ('unknown tool', dict(name='self_destruct', args=dict(reason='t')), 'unknown_tool', None),
    ('missing args', dict(name='go_to_region', args=dict(x='north', reason='t')), 'bad_args', None),
    ('hold too long', dict(name='hold_position', args=dict(seconds=600, reason='t')), 'bad_args', None),
    ('survey radius 500', dict(name='survey_area', args=dict(radius=500, reason='t', **P(safe))), 'bad_args', None),
    ('off the map', dict(name='go_to_region', args=dict(x=150, y=-20, reason='t')), 'out_of_bounds', None),
    ('drive into the river', dict(name='go_to_region', args=dict(reason='t', **P(water))), 'target_in_water', None),
    ('probe in the river', dict(name='place_probe_at', args=dict(reason='t', **P(water))), 'target_in_water', None),
    ('into predicted hazard', dict(name='go_to_region', args=dict(reason='t', **P(max(hazard, 0)))), 'predicted_hazard', 'hazard'),
    ('probe next to a probe', dict(name='place_probe_at', args=dict(x=pr0['x'] + 6, y=pr0['y'] + 4, reason='t')), 'probe_spacing', 'probe'),
    ('into a wall (obstacle cell)', dict(name='go_to_region', args=dict(reason='t', **P(max(wall, 0)))), 'unreachable', 'wall'),
    ('hold inside hazard zone', dict(name='hold_position', args=dict(seconds=20, reason='t')), 'unsafe_hold', 'hold'),
    ('beyond battery reserve', dict(name='go_to_region', args=dict(reason='t', **P(safe))), 'battery_reserve', 'battery'),
    ('GOOD: survey known safe area', dict(name='survey_area', args=dict(radius=20, reason='t', **P(safe))), 'ok', None),
    ('GOOD: retreat to high ground', dict(name='retreat_to_high_ground', args=dict(reason='t')), 'ok', None),
    ('GOOD: hold 10 s', dict(name='hold_position', args=dict(seconds=10, reason='t')), 'ok', None),
]
passed = total = 0
for name, call, expect, special in cases:
    if (special == 'hazard' and hazard < 0) or (special == 'probe' and not probes) or (special == 'wall' and wall < 0):
        print('SKIP  %s' % name); continue
    total += 1
    batt, ri = rd.status['battery'], b.idx(rd.x, rd.y)
    pred = b.predicted[ri]
    if special == 'battery': rd.status['battery'] = 5
    if special == 'hold': b.predicted[ri] = 0.9
    r = br.propose_goal(call)
    rd.status['battery'] = batt; b.predicted[ri] = pred; br.goal = None
    ok = r['rule'] == expect
    passed += ok
    print('%s  %-30s → %s  %s' % ('PASS' if ok else 'FAIL', name, 'ACCEPTED' if r['ok'] else 'REJECTED (%s)' % r['rule'], '' if r['ok'] else r['detail']))
print('\n%d/%d rule-set checks passed' % (passed, total))
sys.exit(0 if passed == total else 1)
