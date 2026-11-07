"""RULE SET (guardrails) for agent tool calls. Mirror of web/brain/rules.ts.

Every tool call from the LLM is validated against the brain's SENSED map before it can influence the rover.
Rules, in order: unknown_tool, bad_args, out_of_bounds, target_in_water, predicted_hazard, unreachable,
battery_reserve, no_probes_left, probe_spacing, unsafe_hold
"""
import math

from .config import CFG, HALF
from .utility import TOOL_NAMES


def _num(v):
    if isinstance(v, bool):
        return float('nan')
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and v.strip():
        try:
            return float(v)
        except ValueError:
            return float('nan')
    return float('nan')


def _reject(rule, detail):
    return dict(ok=False, rule=rule, detail=detail)


def validate_tool_call(call, b, rd, P, home, probes, plan):
    """call = {'name': str, 'args': dict}; plan(x, y) -> path dict or None. Returns {ok, rule, detail, goal?}"""
    name = call.get('name') if isinstance(call, dict) else None
    if not isinstance(name, str) or name not in TOOL_NAMES:
        return _reject('unknown_tool', 'tool "%s" is not in the allowed set' % name)
    a = call.get('args') or {}
    if not isinstance(a, dict):
        return _reject('bad_args', 'arguments must be an object')
    reason = a.get('reason') if isinstance(a.get('reason'), str) else ''
    reason = reason[:160]
    t = rd.t

    def mk(x, y, radius, until, text):
        return dict(ok=True, rule='ok', detail='all rules passed',
                    goal=dict(call=dict(name=name, args=a), t=t, until=until, x=x, y=y, radius=radius, text=text))

    if name == 'hold_position':
        s = _num(a.get('seconds'))
        if not math.isfinite(s) or s < 1 or s > 60:
            return _reject('bad_args', 'seconds must be 1..60, got %r' % (a.get('seconds'),))
        i = b.idx(rd.x, rd.y)
        if b.risk[i] > 0.4 or b.predicted[i] > 0.5:
            return _reject('unsafe_hold', 'rover position is inside a predicted hazard zone (risk %.2f)' % b.predicted[i])
        return mk(rd.x, rd.y, 0, t + s, 'hold_position %.0f s: "%s"' % (s, reason))
    if name == 'retreat_to_high_ground':
        return mk(rd.x, rd.y, 0, t + 45, 'retreat_to_high_ground: "%s"' % reason)

    x, y = _num(a.get('x')), _num(a.get('y'))
    if not (math.isfinite(x) and math.isfinite(y)):
        return _reject('bad_args', 'x/y must be numbers, got %r' % ([a.get('x'), a.get('y')],))
    radius = 0.0
    if name == 'survey_area':
        radius = _num(a.get('radius'))
        if not math.isfinite(radius) or radius < 5 or radius > 60:
            return _reject('bad_args', 'radius must be 5..60 m, got %r' % (a.get('radius'),))
    if not b.inside(x, y) or abs(x) > HALF - 3 or abs(y) > HALF - 3:
        return _reject('out_of_bounds', '(%.0f,%.0f) is outside the mission area' % (x, y))
    i = b.idx(x, y)
    if b.water[i] > 0.05 or b.pred_water[i] > 0.3:
        return _reject('target_in_water', '(%.0f,%.0f) is water or predicted to flood (sensed depth %.2f m)' % (x, y, max(0.0, b.water[i])))
    if b.predicted[i] > 0.4 or b.risk[i] > 0.4:
        return _reject('predicted_hazard', '(%.0f,%.0f) is inside a predicted hazard zone (risk %.2f)' % (x, y, max(b.predicted[i], b.risk[i])))
    if name == 'place_probe_at':
        if rd.status['probes_left'] <= 0:
            return _reject('no_probes_left', 'no probes left')
        for p in probes:
            if math.hypot(p[0] - x, p[1] - y) < CFG['probeRuleSpacing']:
                return _reject('probe_spacing', 'within %.0f m of the probe at (%.0f,%.0f)' % (CFG['probeRuleSpacing'], p[0], p[1]))
    path = plan(x, y)
    if not path:
        return _reject('unreachable', 'no safe path to (%.0f,%.0f) on the sensed map' % (x, y))
    back = math.hypot(x - home[0], y - home[1]) * 1.3
    need = (path['cost'] + back) * CFG['batteryPerM'] + CFG['batteryReserve']
    if need > rd.status['battery']:
        return _reject('battery_reserve', 'needs ~%.0f%% battery to go and return with reserve, have %.0f%%' % (need, rd.status['battery']))
    return mk(x, y, radius or 12.0, t + 60, '%s(%.0f,%.0f%s): "%s"' % (name, x, y, (',r=%.0f' % radius) if radius else '', reason))
