"""UTILITY SELECTION: score behaviours from the sensed state + the agent's goal. Mirror of web/brain/utility.ts."""
import math
import numpy as np

from .config import CFG, G, N, RES, BEHAVIOURS, WATER, HEAT, RUBBLE, WIND, SLOPE, clamp01
from .grid import shift2

TOOL_NAMES = ['go_to_region', 'place_probe_at', 'retreat_to_high_ground', 'hold_position', 'survey_area']

# offsets (in cells) of the coverage sample points used for probe value: centre + 12 at 11 m + 12 at 20 m
_COV_PTS = [(0.0, 0.0)] + [(r * math.cos(k * math.pi / 6), r * math.sin(k * math.pi / 6)) for r in (11.0, 20.0) for k in range(12)]


def _spot(i, value, cost, why):
    return dict(x=-100 + ((i % G) + 0.5) * RES, y=-100 + ((i // G) + 0.5) * RES, value=float(value), cost=float(cost), why=why)


def build_ctx(b, rd, P, reach, probes, home, goal):
    ri = b.idx(rd.x, rd.y)
    x0, y0 = ri % G, ri // G
    R = 6
    danger_near = dyn_near = pred_water_near = 0.0
    ch = b.channels
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            x, y = x0 + dx, y0 + dy
            if x < 0 or y < 0 or x >= G or y >= G or dx * dx + dy * dy > R * R:
                continue
            i = y * G + x
            fall = 1 - math.sqrt(dx * dx + dy * dy) / (R + 2)
            dyn = max(ch[HEAT][i], ch[RUBBLE][i], ch[WIND][i])
            danger_near = max(danger_near, b.risk[i] * fall)
            dyn_near = max(dyn_near, max(dyn, 0.0 if b.pred_water[i] > 0.05 else b.predicted[i] * 0.8) * fall)
            if dx * dx + dy * dy <= 9:
                pred_water_near = max(pred_water_near, (b.pred_water[i] - ch[WATER][i]) * fall)
    e = b.elev[ri]
    flood = P['flood']
    above_water = e - flood['stage'] if (flood['active'] and not math.isnan(e) and not math.isnan(flood['stage'])) else 99.0

    reach_ok = np.isfinite(reach)
    pr = b.predicted
    pred_stage = flood['predStage']
    away = []
    if P['fire']['active'] and P['fire']['nearest'] < 50:
        away.append(P['fire']['bearing'])
    to = P['tornado']
    if to['active'] and to['dist'] < 50:
        away.append(math.atan2(to['y'] - rd.y, to['x'] - rd.x))
    me = P['meteor']
    if me['active'] and math.hypot(me['cx'] - rd.x, me['cy'] - rd.y) < 50:
        away.append(math.atan2(me['cy'] - rd.y, me['cx'] - rd.x))
    # ---- safest spot ----
    safe = None
    m = reach_ok & (pr < 0.3) & (b.pred_water < 0.15)
    if m.any():
        bonus = np.zeros(N)
        if flood['active']:
            bonus = np.where(~np.isnan(b.elev), np.minimum(4.0, np.nan_to_num(b.elev) - pred_stage) * 2, 0.0)
        toward = np.zeros(N)
        ang = np.arctan2(b.ys - rd.y, b.xs - rd.x)
        for a in away:
            toward = np.maximum(toward, np.cos(ang - a))
        track = np.zeros(N)
        if to['active']:
            ex, ey = to['vx'] * 60, to['vy'] * 60
            L2 = ex * ex + ey * ey
            u = np.clip(((b.xs - to['x']) * ex + (b.ys - to['y']) * ey) / L2, 0, 1) if L2 > 1 else np.zeros(N)
            track = 20 * np.maximum(0, 1 - np.hypot(b.xs - to['x'] - u * ex, b.ys - to['y'] - u * ey) / 35)
        v = np.minimum(b.haz_dist, 45) - 0.2 * np.where(reach_ok, reach, 0) - 40 * pr - 15 * np.maximum(0, toward) - track + bonus
        v = np.where(m, v, -np.inf)
        i = int(np.argmax(v))
        safe = _spot(i, v[i], reach[i], '%.0f m from hazards' % min(b.haz_dist[i], 99))
    # ---- high ground ----
    high = None
    m = reach_ok & ~np.isnan(b.elev) & ~b.struct & (pr < 0.35) & (b.pred_water < 0.05)
    if m.any():
        ps_ = 0.0 if math.isnan(pred_stage) else pred_stage
        above = np.nan_to_num(b.elev) - ps_
        v = np.minimum(above, 6) - 0.035 * np.where(reach_ok, reach, 0) - 15 * pr - np.where(ch[SLOPE] > 0.5, 2.0, 0.0)
        v = np.where(m, v, -np.inf)
        i = int(np.argmax(v))
        high = _spot(i, v[i], reach[i], '+%.1f m above water' % above[i])
    # ---- probe spot ----
    probe = None
    if rd.status['probes_left'] > 0:
        ii = np.arange(N)
        cand = reach_ok & (reach <= 70) & ((ii % G) % 2 == 0) & ((ii // G) % 2 == 0) & (b.last_seen >= 0) & ~b.blocked_arr() \
            & (pr <= 0.2) & (b.water <= 0.02) & (b.pred_water <= 0.1)
        for p in probes:
            cand &= np.hypot(b.xs - p[0], b.ys - p[1]) >= CFG['probeSpacing']
        best_v, best_i, best_why = -1e9, -1, ''
        for i in np.nonzero(cand)[0]:
            val, why = probe_value(b, int(i), P, probes)
            v = val - 0.004 * reach[i]
            if v > best_v:
                best_v, best_i, best_why = v, int(i), why
        if best_i >= 0:
            probe = _spot(best_i, best_v, reach[best_i], best_why)
    # ---- frontier ----
    frontier = None
    unk = (b.last_seen < 0).reshape(G, G).astype(float)
    cnt = np.zeros((G, G))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cnt += shift2(unk, dx, dy, 0.0)
    cnt = cnt.reshape(-1)
    m = reach_ok & (reach >= 6) & (b.plan_risk_arr() <= 0.3) & (cnt > 0)
    if m.any():
        head = np.cos(np.arctan2(b.ys - rd.y, b.xs - rd.x) - rd.yaw)
        v = cnt * 0.4 - 0.05 * np.where(reach_ok, reach, 0) + 1.5 * head - 8 * pr
        v = v - np.where((np.abs(b.xs) > 92) | (np.abs(b.ys) > 92), 3.0, 0.0)
        if goal and goal['call']['name'] in ('survey_area', 'go_to_region'):
            dg = np.hypot(b.xs - goal['x'], b.ys - goal['y'])
            v = v + np.where(dg < goal['radius'], 6.0, -0.05 * dg)
        v = np.where(m, v, -np.inf)
        i = int(np.argmax(v))
        frontier = _spot(i, v[i], reach[i], '%d unknown cells' % cnt[i])
    return dict(
        dangerHere=float(b.risk[ri]), dangerNear=danger_near, dynNear=dyn_near, predHere=float(b.predicted[ri]),
        predWaterNear=pred_water_near, hazDist=float(b.haz_dist[ri]), structNear=bool(b.struct_dist[ri] < 10),
        aboveWater=above_water, unexplored=1 - b.known_frac(), battery=rd.status['battery'], health=rd.status['health'],
        probesLeft=rd.status['probes_left'], probe=probe, safe=safe, high=high, frontier=frontier)


def probe_value(b, i, P, probes):
    x, y = b.cx(i), b.cy(i)
    fresh = 0
    for ox, oy in _COV_PTS:
        px, py = x + ox, y + oy
        if not b.inside(px, py):
            continue
        if not any(math.hypot(p[0] - px, p[1] - py) < CFG['probeRadius'] for p in probes):
            fresh += 1
    cov = fresh / len(_COV_PTS)
    warn, why = 0.2, 'coverage'
    hd = b.haz_dist[i]
    any_dyn = P['fire']['active'] or P['tornado']['active'] or P['meteor']['active'] or P['quake']['recent']
    if any_dyn and 10 < hd < 50:
        w = clamp01(1 - abs(hd - 24) / 26)
        if w > warn:
            warn, why = w, 'early warning %.0f m ahead of hazard' % hd
    wd = b.water_dist[i]
    if wd < 30 and not math.isnan(b.elev[i]):
        st = 0.0 if math.isnan(P['flood']['stage']) else P['flood']['stage']
        above = b.elev[i] - st
        if 0.4 < above < 3:
            w = (0.9 if P['flood']['active'] else 0.6) * (1 - wd / 40)
            if w > warn:
                warn, why = w, 'flood gauge %.0f m from water, +%.1f m' % (wd, above)
    sd = b.struct_dist[i]
    if 5 < sd < 25 and 0.5 > warn:
        warn, why = 0.5, 'watches structures %.0f m away' % sd
    return 0.4 * cov + 0.6 * warn, why


def score(ctx, P, rd, goal, placed):
    s = {k: 0.0 for k in BEHAVIOURS}
    low_batt = ctx['battery'] < CFG['batteryLow']
    s['explore'] = (0.5 if ctx['frontier'] else 0.1) * (1 - ctx['dangerNear']) * min(1.0, ctx['unexplored'] * 2.5) * (0.3 if low_batt else 1.0)
    f = ctx['dynNear']
    fi, to, me = P['fire'], P['tornado'], P['meteor']
    if fi['active'] and fi['eta'] < 45:
        f = max(f, 0.35 + 0.55 * (1 - fi['eta'] / 45))
    if fi['active'] and fi['nearest'] < 35:
        f = max(f, 0.8 + 0.15 * (1 - fi['nearest'] / 35))
    if fi['active'] and fi['eta'] < 30:
        f = max(f, 0.7)
    if to['active'] and to['cpa'] < 32 and to['tcpa'] < 35:
        f = max(f, 0.5 + 0.45 * (1 - to['cpa'] / 32))
    if to['active'] and to['dist'] < 35:
        f = max(f, 0.6 + 0.35 * (1 - to['dist'] / 35))
    if me['active']:
        d = math.hypot(me['cx'] - rd.x, me['cy'] - rd.y)
        if d < me['spread'] * 1.6:
            f = max(f, 0.55 + 0.4 * (1 - d / (me['spread'] * 1.6)))
    if P['quake']['recent'] and ctx['structNear']:
        f = max(f, 0.85 if P['quake']['active'] else 0.45)
    s['flee'] = clamp01(f) * (1.0 if ctx['safe'] else 0.5)
    if P['flood']['active']:
        aw = ctx['aboveWater']
        s['high_ground'] = clamp01(0.3 + 0.7 * ctx['predWaterNear'] + (0.35 * (2 - max(0, aw)) / 2 if aw < 2 else 0))
    else:
        s['high_ground'] = 0.5 * ctx['predWaterNear']
    if not ctx['high']:
        s['high_ground'] *= 0.5
    s['hold_still'] = 0.75 if (P['quake']['active'] and not ctx['structNear']) else 0.0
    if ctx['probesLeft'] > 0 and ctx['probe'] and rd.t > 8:
        behind = 0.12 if placed < min(6, rd.t / 24) else 0.0
        s['place_probe'] = clamp01(ctx['probe']['value'] * (1 - ctx['dangerNear']) * (1.0 if ctx['probe']['cost'] < 45 else 0.7) * 0.9 + behind)
    s['place_probe'] *= 1 - 0.6 * s['flee']
    if low_batt:
        s['return_to_safety'] = 0.6 + (CFG['batteryLow'] - ctx['battery']) / 50
    if ctx['health'] < 35:
        s['return_to_safety'] = max(s['return_to_safety'], 0.55)
    if goal:
        g = goal['call']['name']
        if g in ('go_to_region', 'survey_area'):
            s['explore'] += 0.35
        if g == 'place_probe_at' and ctx['probesLeft'] > 0:
            s['place_probe'] = max(s['place_probe'], 0.62)
        if g == 'retreat_to_high_ground':
            s['high_ground'] = max(s['high_ground'] + 0.4, 0.65)
        if g == 'hold_position':
            s['hold_still'] = max(s['hold_still'], 0.62)
    for k in BEHAVIOURS:
        s[k] = round(clamp01(s[k]), 2)
    return s


def select(s, current, since, t):
    best, best_v = current, -1.0
    for k in BEHAVIOURS:
        v = s[k] + (CFG['hysteresis'] if k == current else 0)
        if v > best_v:
            best_v, best = v, k
    if best != current and t - since < CFG['minDwell'] and s[best] < s[current] + 0.25:
        return current
    return best
