"""Open-loop parity check: replay TS-sim frames through the Python BrainCore and compare with the TS brain.
usage: python3 test/parity_replay.py frames.jsonl"""
import json, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from disaster_brain.brain_core import BrainCore

br = BrainCore()
same = n = 0
worst = 0.0
tt = []
for line in open(sys.argv[1]):
    d = json.loads(line)
    t0 = time.perf_counter()
    out = br.tick(d['frame'])
    tt.append((time.perf_counter() - t0) * 1000)
    n += 1
    same += br.behaviour == d['ts']['behaviour']
    if out['decision_log'] and n % 5 == 0:
        pass
tt.sort()
print('frames %d  behaviour agreement %.1f%%  tick ms: median %.1f  p95 %.1f  max %.1f' % (n, 100 * same / n, tt[len(tt) // 2], tt[int(len(tt) * .95)], tt[-1]))
