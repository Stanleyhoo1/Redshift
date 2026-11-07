"""PREDICTION: hazard spread direction/speed from the rover's own sensor history -> 30 s risk map.
Mirror of web/brain/prediction.ts (numpy-vectorised)."""
import math
import numpy as np

from .config import CFG, G, N, RES, WATER, HEAT, RUBBLE, WIND, SLOPE, clamp01, compass
from .grid import shift2, dilate_falloff, paint_lin, distance_field


def _slope(ts, vs):
    n = len(ts)
    if n < 2:
        return 0.0
    mt = sum(ts) / n
    mv = sum(vs) / n
    num = sum((a - mt) * (b - mv) for a, b in zip(ts, vs))
    den = sum((a - mt) ** 2 for a in ts)
    return num / den if den > 1e-6 else 0.0


def _empty():
    return dict(
        fire=dict(active=False, vx=0.0, vy=0.0, speed=0.0, growth=0.0, nearest=999.0, eta=999.0, bearing=0.0, text=''),
        flood=dict(active=False, stage=float('nan'), rate=0.0, predStage=float('nan'), text=''),
        tornado=dict(active=False, x=0.0, y=0.0, vx=0.0, vy=0.0, R=6.0, dist=999.0, cpa=999.0, tcpa=999.0, text=''),
        meteor=dict(active=False, cx=0.0, cy=0.0, vx=0.0, vy=0.0, spread=15.0, count=0, text=''),
        quake=dict(active=False, recent=False, level=0.0, text=''),
    )


class Predictor:
    def __init__(self):
        self.p = _empty()
        self.fire_hist = []
        self.stage_hist = []
        self.tor_hist = []
        self.impacts = []
        self.last_shake_t = -1e9
        self.shake_level = 0.0
        self.sensed = set()
        self.newly_sensed = []

    def observe(self, rd, ps):
        t = rd.t
        if not math.isnan(rd.stage_obs):
            self.stage_hist.append((t, rd.stage_obs))
            while self.stage_hist and self.stage_hist[0][0] < t - 40:
                self.stage_hist.pop(0)
        if rd.funnel:
            self.tor_hist.append((t, rd.funnel['x'], rd.funnel['y'], 1.0))
        elif rd.wind_speed > CFG['tornadoWind'] or rd.pressure_drop > CFG['tornadoPressure']:
            rx, ry = rd.wind_x - ps.amb_x, rd.wind_y - ps.amb_y
            a = math.atan2(ry, rx) + math.pi / 2
            r = max(8.0, min(60.0, 6 * math.sqrt(0.6 * 55 * 55 / max(10.0, rd.pressure_drop))))
            self.tor_hist.append((t, rd.x + r * math.cos(a), rd.y + r * math.sin(a), 0.3))
        while self.tor_hist and self.tor_hist[0][0] < t - 12:
            self.tor_hist.pop(0)
        for f in rd.flashes:
            if any(abs(i[0] - t) < 0.4 and math.hypot(i[1] - f['x'], i[2] - f['y']) < 4 for i in self.impacts):
                continue
            self.impacts.append((t, f['x'], f['y'], f['size']))
        while self.impacts and self.impacts[0][0] < t - 30:
            self.impacts.pop(0)
        if rd.shake > CFG['shakeThresh']:
            self.last_shake_t = t
            self.shake_level = rd.shake
        else:
            self.shake_level *= 0.9

    def map_update(self, b, rd, ps):
        t, T = rd.t, CFG['horizon']
        b.update_conf(t)
        cw = np.zeros(N); ch = np.zeros(N); cr = np.zeros(N); cwi = np.zeros(N); cs = np.zeros(N)
        pred = np.zeros(N)
        b.pred_water = np.zeros(N)
        self.newly_sensed = []

        def mark(k):
            if k not in self.sensed:
                self.sensed.add(k)
                self.newly_sensed.append(k)

        # ---- water now ----
        cw = np.where(b.water > 0, np.clip(b.water / 0.35 + 0.15, 0, 1), 0.0)

        # ---- heat now: aged observation, dilated (radiant heat reaches ~8 m) ----
        h = b.heat
        valid = ~np.isnan(h) & (h > 60)
        age = t - b.heat_t
        Te = np.where(valid, CFG['ambient'] + (np.nan_to_num(h) - CFG['ambient']) * np.exp(-np.maximum(0, age - 15) / 40), 0.0)
        r0 = np.where(valid, np.clip((Te - 60) / 250, 0, 1), 0.0)
        r0 = np.where(r0 >= 0.08, r0, 0.0)
        ch = dilate_falloff(r0, 4, lambda d: 1 - d / 5)
        hot = np.nonzero(valid & (Te > CFG['hotTemp']) & (age < 25))[0]
        if len(hot):
            mark('fire')

        # ---- rubble: quake debris near structures, recent impacts ----
        since = t - self.last_shake_t
        if since < 60:
            mark('earthquake')
            q = 0.9 if since < 2 else 0.55 * (1 - since / 60) + 0.15
            src = np.where(b.struct & (b.obst > 0.3), q, 0.0)
            cr = np.maximum(cr, dilate_falloff(src, 3, lambda d: 1 - d / 4))
        for (it, ix, iy, ir) in self.impacts:
            v = 0.85 * (1 - (t - it) / 30)
            paint_lin(cr, b.idx(ix, iy), int(math.ceil((ir + 4) / RES)), v, 1.0, (ir + 6) / RES)
        if self.impacts:
            mark('meteor')

        # ---- slope from elevation (dilated by one cell) ----
        e2 = b.elev.reshape(G, G)
        ok2 = (~np.isnan(e2)) & (~b.struct.reshape(G, G))
        m = np.zeros((G, G))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            en = shift2(e2, -dx, -dy, np.nan)       # neighbour value at (x+dx, y+dy)
            okn = shift2(ok2, -dx, -dy, False)
            diff = np.where(ok2 & okn, np.abs(np.nan_to_num(en) - np.nan_to_num(e2)), 0.0)
            np.maximum(m, diff, out=m)
        deg = np.degrees(np.arctan(m / RES))
        tmp = np.where(ok2, np.clip((deg - 12) / 14, 0, 1), 0.0)
        cs2 = tmp.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                np.maximum(cs2, shift2(tmp, dx, dy, 0.0), out=cs2)
        cs = cs2.reshape(-1)

        P = self.p
        # ---- fire ----
        P['fire']['active'] = len(hot) > 0
        if len(hot):
            hx, hy = b.xs[hot], b.ys[hot]
            cx, cy = float(hx.mean()), float(hy.mean())
            rad = math.sqrt(len(hot) * RES * RES / math.pi)
            self.fire_hist.append((t, cx, cy, rad))
            while self.fire_hist and self.fire_hist[0][0] < t - 20:
                self.fire_hist.pop(0)
            ts = [hh[0] for hh in self.fire_hist]
            span = ts[-1] - ts[0] if ts else 0
            growth = max(0.0, min(1.5, _slope(ts, [hh[3] for hh in self.fire_hist])))
            amb_s = math.hypot(ps.amb_x, ps.amb_y) or 1.0
            wdx, wdy = ps.amb_x / amb_s, ps.amb_y / amb_s
            vx, vy = wdx * 0.35, wdy * 0.35
            if span > 6:
                mvx = _slope(ts, [hh[1] for hh in self.fire_hist])
                mvy = _slope(ts, [hh[2] for hh in self.fire_hist])
                sp = max(growth, 0.15)
                vx = 0.55 * mvx + 0.45 * wdx * sp
                vy = 0.55 * mvy + 0.45 * wdy * sp
            sp = math.hypot(vx, vy)
            if sp > 2:
                vx *= 2 / sp
                vy *= 2 / sp
            is_hot = np.zeros(N, dtype=bool)
            is_hot[hot] = True
            h2 = is_hot.reshape(G, G)
            edge = np.zeros((G, G), dtype=bool)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                edge |= ~shift2(h2, dx, dy, True)
            front = np.nonzero((h2 & edge).reshape(-1))[0]
            if len(front) > 160:
                front = front[np.linspace(0, len(front) - 1, 160).astype(int)]
            rr = int(min(7, math.ceil((3 + growth * T * 0.4) / RES)))
            d_all = np.hypot(hx - rd.x, hy - rd.y)
            k = int(np.argmin(d_all))
            nearest, ni = float(d_all[k]), int(hot[k])
            for i in front:
                fx, fy = b.xs[i], b.ys[i]
                for s in range(4):
                    px, py = fx + vx * T * s / 3, fy + vy * T * s / 3
                    if b.inside(px, py):
                        paint_lin(pred, b.idx(px, py), rr, 0.9, 1.0, rr + 1)
            bx, by = rd.x - b.xs[ni], rd.y - b.ys[ni]
            closing = (vx * bx + vy * by) / max(nearest, 1) + growth * 0.5 if nearest > 0 else 0
            eta = max(0.0, nearest - 6) / closing if closing > 0.02 else 999.0
            bearing = math.atan2(b.ys[ni] - rd.y, b.xs[ni] - rd.x)
            P['fire'].update(vx=vx, vy=vy, speed=math.hypot(vx, vy), growth=growth, nearest=nearest, eta=eta, bearing=bearing)
            P['fire']['text'] = 'fire front %.0f m %s moving %s %.2f m/s%s' % (
                nearest, compass(bearing), compass(math.atan2(vy, vx)), math.hypot(vx, vy),
                (', reaches rover in %.0f s' % eta) if eta < 120 else '')
        else:
            self.fire_hist = []
            P['fire']['text'] = ''

        # ---- flood ----
        if len(self.stage_hist) >= 3:
            recent = sorted(s for _, s in self.stage_hist[-10:])
            stage = recent[len(recent) >> 1]
            rate = _slope([a for a, _ in self.stage_hist], [s for _, s in self.stage_hist])
            span = self.stage_hist[-1][0] - self.stage_hist[0][0]
            t0 = self.stage_hist[0][0]
            old = sorted(s for a, s in self.stage_hist if a < t0 + 6)
            rise = stage - old[len(old) >> 1]
            rising = rate > 0.004 and span > 10 and rise > 0.1
            pred_stage = stage + (rate * T + 0.25 if rising else 0.1)
            P['flood'].update(active=rising, stage=stage, rate=rate, predStage=pred_stage)
            if rising:
                mark('flood')
            # flood fill from seen water through dry ground lower than the predicted stage
            ground = np.where(~np.isnan(b.dry_elev), b.dry_elev, np.where(~np.isnan(b.elev) & ~b.struct, b.elev, np.nan))
            can = (~np.isnan(ground)) & (np.nan_to_num(ground, nan=1e9) < pred_stage)
            seen = b.water > 0.05
            filled = seen.copy()
            can2, fill2 = can.reshape(G, G), filled.reshape(G, G)
            for _ in range(G * 2):
                grow = np.zeros((G, G), dtype=bool)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    grow |= shift2(fill2, dx, dy, False)
                new = grow & can2 & ~fill2
                if not new.any():
                    break
                fill2 |= new
            newly = fill2.reshape(-1) & ~seen
            v = np.where(newly, np.clip((pred_stage - np.nan_to_num(ground)) / 0.35 + 0.2, 0, 1), 0.0)
            np.maximum(pred, v, out=pred)
            np.maximum(b.pred_water, v, out=b.pred_water)
            ri = b.idx(rd.x, rd.y)
            above = b.elev[ri] - stage if not math.isnan(b.elev[ri]) else float('nan')
            P['flood']['text'] = ('water stage %.2f m rising %.1f cm/s (+%.2f m in %d s)%s' % (
                stage, rate * 100, rate * T, T, (', rover %.1f m above water' % above) if not math.isnan(above) else '')) if rising else ''

        # ---- tornado ----
        th = self.tor_hist
        P['tornado']['active'] = bool(th) and t - th[-1][0] < 6
        if P['tornado']['active']:
            mark('tornado')
            good = [hh for hh in th if hh[3] > 0.5]
            if len(good) < 3:
                good = th
            last = good[-5:]
            x = sum(hh[1] for hh in last) / len(last)
            y = sum(hh[2] for hh in last) / len(last)
            vx = vy = 0.0
            if len(good) >= 4 and good[-1][0] - good[0][0] > 2:
                vx = _slope([hh[0] for hh in good], [hh[1] for hh in good])
                vy = _slope([hh[0] for hh in good], [hh[2] for hh in good])
            sp = math.hypot(vx, vy)
            if sp > 7:
                vx *= 7 / sp
                vy *= 7 / sp
            dist = math.hypot(x - rd.x, y - rd.y)
            rx, ry, vv = rd.x - x, rd.y - y, vx * vx + vy * vy
            tcpa = max(0.0, (rx * vx + ry * vy) / vv) if vv > 0.01 else 0.0
            cpa = math.hypot(rx - vx * tcpa, ry - vy * tcpa)
            P['tornado'].update(x=x, y=y, vx=vx, vy=vy, R=7.0, dist=dist, cpa=cpa, tcpa=tcpa)
            if b.inside(x, y):
                paint_lin(cwi, b.idx(x, y), int(math.ceil(26 / RES)), 1.0, 1.15, 26 / RES)
            s = 0
            while s <= T:
                px, py = x + vx * s, y + vy * s
                if b.inside(px, py):
                    paint_lin(pred, b.idx(px, py), int(math.ceil(24 / RES)), 0.95, 1.1, 24 / RES)
                s += 3
            P['tornado']['text'] = 'vortex %.0f m %s moving %s %.1f m/s, passes %.0f m from rover in %.0f s' % (
                dist, compass(math.atan2(y - rd.y, x - rd.x)), compass(math.atan2(vy, vx)), math.hypot(vx, vy), cpa, tcpa)
        else:
            P['tornado']['text'] = ''

        # ---- meteors ----
        im = [i for i in self.impacts if t - i[0] < 25]
        P['meteor']['active'] = len(im) >= 1
        if im:
            sw = sx = sy = 0.0
            for (it, ix, iy, _) in im:
                w = 1 - (t - it) / 30
                sw += w; sx += ix * w; sy += iy * w
            cx, cy = sx / sw, sy / sw
            sp2 = sum((ix - cx) ** 2 + (iy - cy) ** 2 for (_, ix, iy, _) in im)
            spread = max(14.0, min(45.0, math.sqrt(sp2 / len(im)) * 1.4))
            vx = vy = 0.0
            if len(im) >= 5:
                vx = max(-2.0, min(2.0, _slope([i[0] for i in im], [i[1] for i in im])))
                vy = max(-2.0, min(2.0, _slope([i[0] for i in im], [i[2] for i in im])))
            P['meteor'].update(cx=cx, cy=cy, vx=vx, vy=vy, spread=spread, count=len(im))
            level = min(0.75, 0.25 + len(im) * 0.06)
            for s in (0, 5, 10, 15):
                px, py = cx + vx * s, cy + vy * s
                if b.inside(px, py):
                    paint_lin(pred, b.idx(px, py), int(math.ceil(spread / RES)), level, 1.2, spread / RES)
            d = math.hypot(cx - rd.x, cy - rd.y)
            P['meteor']['text'] = '%d impacts in 25 s, zone centre %.0f m %s (radius %.0f m)%s' % (
                len(im), d, compass(math.atan2(cy - rd.y, cx - rd.x)), spread,
                (' drifting %s' % compass(math.atan2(vy, vx))) if math.hypot(vx, vy) > 0.3 else '')
        else:
            P['meteor']['text'] = ''

        # ---- quake ----
        P['quake']['active'] = t - self.last_shake_t < 1.5
        P['quake']['recent'] = since < 60
        P['quake']['level'] = self.shake_level
        P['quake']['text'] = ('ground shaking %.1f m/s², aftershocks likely near structures' % self.shake_level) if P['quake']['active'] else \
            ('aftershock risk near structures (%.0f s since shaking)' % since) if P['quake']['recent'] else ''

        # ---- combine ----
        risk = np.maximum(np.maximum(cw, ch), np.maximum(cr, cwi))
        b.channels = [cw, ch, cr, cwi, cs]
        b.risk = risk
        b.predicted = np.maximum(pred, risk)
        dyn_src = (ch > 0.4) | (cr > 0.4) | (cwi > 0.4) | ((b.predicted > 0.45) & (cw < 0.05))
        b.haz_dist = distance_field(dyn_src)
        b.pred_water = np.maximum(b.pred_water, cw)
        b.water_dist = distance_field(b.water > 0.05)
        b.struct_dist = distance_field(b.struct)
