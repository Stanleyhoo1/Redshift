"""The rclpy-independent brain. Mirror of web/brain/brain.ts (LocalBrain).

PERCEPTION -> PREDICTION -> UTILITY SELECTION -> PLANNER (A*) -> control, SAFETY REFLEXES override.
Input: one sensor frame (dict with ROS message shapes). Output: {cmd_vel, drop_probe, decision_log}.
It never sees the simulator's world state.
"""
import math
import numpy as np

from .config import CFG, G, N, BEHAVIOURS, compass, wrap
from .grid import Belief
from .perception import perceive, PerceptionState
from .prediction import Predictor
from .planner import astar, reach_map, path_risk, cost_grid
from .utility import build_ctx, score, select
from .safety import Safety, tw
from .rules import validate_tool_call


def _zero():
    return tw(0, 0)


class BrainCore:
    name = 'ros2-py'

    def __init__(self):
        self.b = Belief()
        self.ps = PerceptionState()
        self.pred = Predictor()
        self.safety = Safety()
        self.rd = None
        self.behaviour = 'explore'
        self.since = 0.0
        self.scores = {k: 0.0 for k in BEHAVIOURS}
        self.ctx = None
        self.reach = None
        self.target = None
        self.path = None
        self.last_plan = -1.0
        self.home = None
        self.goal = None
        self.reflex = None
        self.confidence = 0.5
        self.bad_until = np.zeros(N)
        self.events = []
        self._reflex_pos = (0.0, 0.0, 0.0)
        self._last_map = -1.0
        self._last_log = -1.0
        self._last_log_key = ''
        self._stuck_since = -1.0
        self._recover_until = -1.0
        self._recover_w = 1.0
        self._last_probe_drop = -1e9
        self._batt_event = False
        self._path_event = -1e9
        self._grid = None

    # ------------------------------------------------------------------ tick
    def tick(self, f):
        b, P = self.b, self.pred.p
        rd = perceive(b, f, self.ps)
        self.rd = rd
        t = rd.t
        if self.home is None:
            self.home = (rd.x, rd.y)
        self.pred.observe(rd, self.ps)
        if not rd.status['alive']:
            return dict(cmd_vel=_zero(), drop_probe=None, decision_log=None)

        if self._last_map < 0 or t - self._last_map >= CFG['mapPeriod']:
            self._last_map = t
            self.pred.map_update(b, rd, self.ps)
            for k in self.pred.newly_sensed:
                self.events.append(dict(kind='new_hazard', text='new hazard type sensed: %s' % k))
            self._grid = cost_grid(b, False)
            reach = reach_map(b, rd.x, rd.y, 75, grid=self._grid)
            reach[self.bad_until > t] = np.inf
            self.reach = reach
            self.ctx = build_ctx(b, rd, P, reach, self._probes(rd), self.home, self.goal)
            if not self._batt_event and rd.status['battery'] < CFG['batteryAgentEvent']:
                self._batt_event = True
                self.events.append(dict(kind='battery', text='battery %.0f%%' % rd.status['battery']))
        ctx = self.ctx
        if self.goal and t > self.goal['until']:
            self.goal = None

        placed = 6 - rd.status['probes_left']
        self.scores = score(ctx, P, rd, self.goal, placed)
        prev = self.behaviour
        nb = select(self.scores, self.behaviour, self.since, t)
        if nb != self.behaviour:
            self.behaviour, self.since, self.target, self.path = nb, t, None, None
        srt = sorted(self.scores.values(), reverse=True)
        self.confidence = max(0.05, min(0.99, 0.45 + (srt[0] - srt[1]) + 0.35 * b.conf[b.idx(rd.x, rd.y)]))

        self._choose_target(rd, ctx)

        drop = False
        cmd = tw(0, 0)
        if self.target:
            tg = self.target
            dist = math.hypot(tg['x'] - rd.x, tg['y'] - rd.y)
            risky = path_risk(b, self.path['cells'], 40)[0] > 0.6 if self.path else False
            if not self.path or t - self.last_plan > 2.0 or risky:
                if risky and t - self._path_event > 15:
                    self._path_event = t
                    self.events.append(dict(kind='hazard_on_path', text='predicted hazard entered the planned path'))
                self.last_plan = t
                urgent = self.behaviour in ('flee', 'high_ground', 'return_to_safety')
                self.path = astar(b, rd.x, rd.y, tg['x'], tg['y'], False, grid=self._grid)
                if not self.path and urgent:
                    self.path = astar(b, rd.x, rd.y, tg['x'], tg['y'], True)
                if not self.path:
                    self._blacklist(tg['x'], tg['y'], t)
                    self.target = None
            if self.target and dist < 2.5:
                if self.behaviour == 'place_probe' and rd.status['probes_left'] > 0 and t - self._last_probe_drop > 3:
                    drop = True
                    self._last_probe_drop = t
                    self.events.append(dict(kind='probe_placed', text='probe placed at (%.0f, %.0f): %s' % (rd.x, rd.y, tg['why'])))
                    if self.goal and self.goal['call']['name'] == 'place_probe_at':
                        self.goal = None
                    self.since = t - CFG['minDwell']
                if self.behaviour == 'explore' and self.goal and self.goal['call']['name'] == 'go_to_region' and \
                        math.hypot(self.goal['x'] - rd.x, self.goal['y'] - rd.y) < 5:
                    self.goal = None
                self.target = None
                self.path = None
            elif self.path:
                cmd = self._follow(rd, self.path)
        elif self.behaviour in ('explore', 'flee'):
            cmd = tw(0, 0.5)

        # stuck recovery
        if t < self._recover_until:
            cmd = tw(-1.0, self._recover_w)
        elif abs(cmd['linear']['x']) > 0.6 and abs(rd.v) < 0.12:
            if self._stuck_since < 0:
                self._stuck_since = t
            if t - self._stuck_since > 1.5:
                self._recover_until = t + 1.2
                self._recover_w = 1.2 if int(math.floor(t)) % 2 else -1.2
                self._stuck_since = -1.0
                ax, ay = rd.x + 2 * math.cos(rd.yaw), rd.y + 2 * math.sin(rd.yaw)
                if b.inside(ax, ay):
                    b.obst[b.idx(ax, ay)] = 1.0
                self.path = None
        else:
            self._stuck_since = -1.0

        # SAFETY REFLEXES
        reflex = self.safety.check(rd, b, P)
        started = reflex is not None and (self.reflex is None or self.reflex['name'] != reflex['name'])
        self.reflex = reflex
        if reflex:
            cmd = reflex['cmd']
            rx, ry, rt = self._reflex_pos
            moved = math.hypot(rd.x - rx, rd.y - ry)
            if not started and moved < 0.4 and t - rt > 1.5:
                ph = (t - rt) % 3
                cmd = tw(0, 1.4) if ph < 1.5 else tw(1.5, 0)
                reflex['text'] += ' [stalled: wiggle]'
            elif started or moved >= 0.4:
                self._reflex_pos = (rd.x, rd.y, t)
        if cmd['linear']['x'] > 0 and rd.lidar_front < CFG['reflexFront']:
            cmd = tw(0, cmd['angular']['z'] or 0.9)

        log = None
        key = '%s|%s' % (self.behaviour, reflex['name'] if reflex else '')
        if started:
            log = 'REFLEX %s (overrides %s)' % (reflex['text'], self.behaviour)
        elif key != self._last_log_key or t - self._last_log > 4 or drop:
            log = self._log_line(rd, prev, drop)
        if log:
            self._last_log = t
            self._last_log_key = key
        return dict(cmd_vel=cmd, drop_probe=dict(x=rd.x, y=rd.y, z=0.0) if drop else None, decision_log=log)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _probes(rd):
        return [(p['x'], p['y']) for p in rd.status['probes']]

    def _blacklist(self, x, y, t):
        i = self.b.idx(x, y)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                k = i + dy * G + dx
                if 0 <= k < N:
                    self.bad_until[k] = t + 20

    def _choose_target(self, rd, ctx):
        t, b, cur, beh, g = rd.t, self.b, self.target, self.behaviour, self.goal
        want = None
        if beh == 'explore':
            want = dict(x=g['x'], y=g['y'], value=1.0, cost=0.0, why='agent goal: ' + g['text']) if (g and g['call']['name'] == 'go_to_region') else ctx['frontier']
        elif beh == 'flee':
            want = ctx['safe']
        elif beh == 'high_ground':
            want = ctx['high']
        elif beh == 'place_probe':
            want = dict(x=g['x'], y=g['y'], value=1.0, cost=0.0, why='agent goal: ' + g['text']) if (g and g['call']['name'] == 'place_probe_at') else ctx['probe']
        elif beh == 'return_to_safety':
            hi = b.idx(*self.home)
            want = dict(x=self.home[0], y=self.home[1], value=1.0, cost=0.0, why='home base') if b.plan_risk(hi) < 0.3 else ctx['safe']
        if not want:
            self.target = None
            return
        if cur and cur['kind'] == beh:
            ci = b.idx(cur['x'], cur['y'])
            stale = (t - cur['set'] > 15) if beh == 'explore' else (t - cur['set'] > 6)
            bad = b.plan_risk(ci) > 0.45 or b.blocked(ci) or (beh == 'explore' and b.last_seen[ci] >= 0 and self._frontier_gone(ci))
            moved = math.hypot(want['x'] - cur['x'], want['y'] - cur['y']) > 8
            if not bad and not (stale and moved) and not (moved and beh in ('flee', 'high_ground') and want['value'] > cur['value'] + 3):
                return
        self.target = dict(want, kind=beh, set=t)
        self.path = None

    def _frontier_gone(self, i):
        x, y = i % G, i // G
        ls = self.b.last_seen
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                X, Y = x + dx, y + dy
                if 0 <= X < G and 0 <= Y < G and ls[Y * G + X] < 0:
                    return False
        return True

    def _follow(self, rd, p):
        pts = p['path']
        k, best = 0, float('inf')
        for i, (px, py) in enumerate(pts):
            d = math.hypot(px - rd.x, py - rd.y)
            if d < best:
                best, k = d, i
        tx, ty = pts[-1]
        for i in range(k, len(pts)):
            if math.hypot(pts[i][0] - rd.x, pts[i][1] - rd.y) > 3.5:
                tx, ty = pts[i]
                break
        err = wrap(math.atan2(ty - rd.y, tx - rd.x) - rd.yaw)
        urgent = self.behaviour in ('flee', 'high_ground')
        v = (CFG['maxV'] if urgent else CFG['maxV'] * 0.8) * max(0.0, math.cos(err)) ** 2
        if abs(err) > 1.2:
            v = 0.0
        if rd.lidar_front < 4:
            v = min(v, 0.6 + 0.4 * rd.lidar_front)
        w = max(-CFG['maxW'], min(CFG['maxW'], 2.2 * err))
        return tw(v, w)

    def _log_line(self, rd, prev, drop):
        P, s, beh = self.pred.p, self.scores, self.behaviour
        trig = 'LiDAR %.0f%% returns, map %.0f%% known' % (rd.return_frac * 100, 100 * (1 - (self.ctx['unexplored'] if self.ctx else 1)))
        T, d, a = rd.thermal_max
        if T > 80:
            trig = 'Thermal %.0f°C %.0f m %s' % (T, d, compass(a))
        if P['flood']['active'] or rd.depth > 0.05:
            if rd.depth > 0.02:
                w = 'depth %.2f m' % rd.depth
            elif rd.water_near:
                w = 'edge %.0f m %s' % (rd.water_near[0], compass(rd.water_near[1]))
            else:
                w = 'not in view'
            trig = 'Water ' + w
        if rd.wind_speed > 11 or rd.funnel:
            trig = 'Wind %.0f m/s, ΔP −%.0f Pa%s' % (rd.wind_speed, max(0, rd.pressure_drop),
                                                    (', funnel %.0f m %s' % (rd.funnel['dist'], compass(rd.funnel['bearing']))) if rd.funnel else '')
        if rd.shake > CFG['shakeThresh']:
            trig = 'IMU shake %.1f m/s²' % rd.shake
        if rd.flashes:
            trig = 'Impact flash %.0f m %s' % (rd.flashes[0]['dist'], compass(rd.flashes[0]['bearing']))
        if beh == 'place_probe' or drop:
            why = self.target['why'] if self.target else (self.ctx['probe']['why'] if self.ctx and self.ctx['probe'] else '-')
            trig = 'Probe spot: ' + why
        preds = [P[k]['text'] for k in ('fire', 'flood', 'tornado', 'meteor', 'quake') if P[k]['text']]
        sc = ' '.join('%s %.2f' % kv for kv in sorted(s.items(), key=lambda kv: -kv[1])[:3])
        tg = self.target
        if tg:
            via = ' → %.0f m %s (%s)' % (math.hypot(tg['x'] - rd.x, tg['y'] - rd.y), compass(math.atan2(tg['y'] - rd.y, tg['x'] - rd.x)), tg['why'])
        else:
            via = ' (stay put)' if beh == 'hold_still' else ''
        action = 'DROP PROBE' if drop else beh.upper()
        goal = (' [goal: %s]' % self.goal['call']['name']) if self.goal else ''
        return '%s | %s | %s → %s%s (%.2f)%s%s' % (trig, '; '.join(preds) if preds else 'no hazard predicted', sc, action, via,
                                                 self.confidence, (' [was %s]' % prev) if prev != beh else '', goal)

    # ------------------------------------------------------------------ agent interface
    def summary(self):
        rd, ctx, P, b = self.rd, self.ctx, self.pred.p, self.b
        if rd is None or ctx is None:
            return 'no data yet'
        dirs = []
        for k in range(8):
            a = k * math.pi / 4
            unk = tot = 0
            for d in (30, 50, 70):
                x, y = rd.x + d * math.cos(a), rd.y + d * math.sin(a)
                if not b.inside(x, y):
                    continue
                tot += 1
                if b.last_seen[b.idx(x, y)] < 0:
                    unk += 1
            if tot and unk / tot > 0.6:
                dirs.append(compass(a))
        hz = [P[k]['text'] for k in ('fire', 'flood', 'tornado', 'meteor', 'quake') if P[k]['text']]
        probes = ' '.join('(%.0f,%.0f%s)' % (p['x'], p['y'], '' if p['alive'] else ' lost') for p in rd.status['probes']) or 'none'

        def f(s):
            return '(%.0f,%.0f)' % (s['x'], s['y']) if s else 'none'
        return '\n'.join([
            't=%.0fs rover at (%.0f,%.0f) battery %.0f%% health %.0f%% probes_left %d behaviour %s.' % (
                rd.t, rd.x, rd.y, rd.status['battery'], rd.status['health'], rd.status['probes_left'], self.behaviour),
            'Map %.0f%% explored; unexplored directions: %s.' % (100 * (1 - ctx['unexplored']), ','.join(dirs) or 'few'),
            'Hazards sensed: %s.' % ('; '.join(hz) if hz else 'none'),
            'Probes placed: %s. Best probe spot: %s (%s). Safest reachable spot: %s. Highest safe ground: %s%s.' % (
                probes, f(ctx['probe']), ctx['probe']['why'] if ctx['probe'] else '-', f(ctx['safe']), f(ctx['high']),
                (' (%s)' % ctx['high']['why']) if ctx['high'] else ''),
            'Current goal: %s.' % (self.goal['text'] if self.goal else 'none'),
        ])

    def propose_goal(self, call):
        if self.rd is None or self.ctx is None:
            return dict(ok=False, rule='no_state', detail='brain has no sensor data yet')
        res = validate_tool_call(call, self.b, self.rd, self.pred.p, self.home, self._probes(self.rd),
                                 lambda x, y: astar(self.b, self.rd.x, self.rd.y, x, y, False))
        if res['ok']:
            self.goal = res['goal']
        return res

    def hazard_map_msg(self):
        """compact belief for the web UI: 100x100 bytes per layer, base64-free (lists of ints 0..100)."""
        b = self.b
        q = lambda a: np.clip(np.round(np.nan_to_num(a) * 100), 0, 100).astype(int).tolist()
        return dict(n=G, risk=q(b.risk), predicted=q(b.predicted), conf=q(b.conf), obst=q(b.obst),
                    water=q(b.channels[0]), heat=q(b.channels[1]), rubble=q(b.channels[2]), wind=q(b.channels[3]))

    def info(self):
        return dict(behaviour=self.behaviour, reflex=self.reflex['text'] if self.reflex else None, scores=self.scores,
                    confidence=round(self.confidence, 2), goal=self.goal['text'] if self.goal else 'none',
                    goalPt=dict(x=self.goal['x'], y=self.goal['y']) if (self.goal and self.goal['call']['name'] not in ('hold_position', 'retreat_to_high_ground')) else None,
                    path=[dict(x=p[0], y=p[1]) for p in self.path['path']] if self.path else None)
