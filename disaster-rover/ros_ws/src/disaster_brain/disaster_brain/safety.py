"""SAFETY REFLEXES: hard limits that override every behaviour instantly. Mirror of web/brain/safety.ts."""
import math

from .config import CFG, HEAT, SLOPE, compass, wrap


def tw(v, w):
    return dict(linear=dict(x=float(v), y=0.0, z=0.0), angular=dict(x=0.0, y=0.0, z=float(w)))


def escape_heading(rd, heading, speed=None):
    speed = CFG['maxV'] if speed is None else speed
    mw = CFG['maxW']
    err = wrap(heading - rd.yaw)
    if abs(err) > 2.0:
        err_back = wrap(heading - (rd.yaw + math.pi))
        return tw(-speed * 0.5, max(-mw, min(mw, 2.5 * err_back)))
    v = speed * max(0.25, math.cos(err))
    return tw(v, max(-mw, min(mw, 2.5 * err)))


def clearance(rd, heading, half_width=0.3):
    m = float('inf')
    for k, r in enumerate(rd.scan_r):
        a = rd.scan_a0 + k * rd.scan_inc
        if abs(wrap(a - heading)) > half_width:
            continue
        r = rd.scan_max if r > rd.scan_max else r
        if r < m:
            m = r
    return m


def steer_clear(rd, cmd, need=2.6):
    """Local steering: re-aim a forward command at the nearest LiDAR-clear heading."""
    if cmd['linear']['x'] <= 0.05:
        return cmd
    want = rd.yaw + cmd['angular']['z'] * 0.6
    if clearance(rd, want) >= need:
        return cmd
    best, best_cost = None, float('inf')
    for k in range(-12, 13):
        h = want + k * math.pi / 18
        c = clearance(rd, h)
        if c < need:
            continue
        cost = abs(k) - min(c, 10) * 0.05
        if cost < best_cost:
            best_cost, best = cost, h
    if best is None:
        return tw(0, 1.2 if cmd['angular']['z'] >= 0 else -1.2)
    err = wrap(best - rd.yaw)
    v = 0.0 if abs(err) > 1.0 else min(cmd['linear']['x'], CFG['maxV'] * math.cos(err))
    return tw(v, max(-CFG['maxW'], min(CFG['maxW'], 2.5 * err)))


class Safety:
    def __init__(self):
        self.low_conf_since = -1.0
        self.last_rescan = -1e9
        self.last_dry = None

    def check(self, rd, b, P):
        if rd.depth < 0.03:
            self.last_dry = (rd.x, rd.y)
        # 1. incoming meteor
        worst = None
        for s in rd.streaks:
            d = math.hypot(s['x'] - rd.x, s['y'] - rd.y)
            if d < s['size'] + 4 and (worst is None or d - s['size'] < worst[0] - worst[4]):
                worst = (d, s['ttl'], s['x'], s['y'], s['size'])
        if worst and rd.depth <= 0.35:
            d, ttl, wx, wy, _ = worst
            away = self._dodge_heading(rd, b, math.atan2(rd.y - wy, rd.x - wx) if d > 0.5 else rd.yaw)
            return dict(name='meteor_dodge', cmd=escape_heading(rd, away),
                        text='Camera: meteor streak, impact %.1f m %s in %.1f s → DODGE %s' % (d, compass(math.atan2(wy - rd.y, wx - rd.x)), ttl, compass(away)))
        # 2. water
        heat_near = rd.temp_c > 40 or any(s[1] < 14 for s in rd.hot_sectors)
        if rd.depth > (0.28 if heat_near else CFG['reflexWater']):
            ld = self.last_dry
            dd = math.hypot(ld[0] - rd.x, ld[1] - rd.y) if ld else 99.0
            if ld and 0.3 < dd < 15:
                h = math.atan2(ld[1] - rd.y, ld[0] - rd.x)
            else:
                h = self.escape_search(rd, b, 0.22) if heat_near else self._best_dry_heading(rd, b)
            return dict(name='water', cmd=escape_heading(rd, h, 2.2),
                        text='Water depth %.2f m > %s m limit → climb out %s' % (rd.depth, CFG['reflexWater'], compass(h)))
        # 3. heat
        close = [s for s in rd.hot_sectors if s[1] < 9]
        if rd.temp_c > CFG['reflexTemp'] or close:
            h = self.escape_search(rd, b, 0.22)
            if rd.temp_c > CFG['reflexTemp']:
                why = 'Body temp %.0f°C > %.0f°C' % (rd.temp_c, CFG['reflexTemp'])
            else:
                near = min(close, key=lambda c: c[1])
                why = 'Thermal %.0f°C at %.1f m %s' % (near[2], near[1], compass(near[0]))
            return dict(name='heat', cmd=escape_heading(rd, h), text='%s → ESCAPE %s' % (why, compass(h)))
        # 4. tornado wind
        if rd.wind_speed > CFG['reflexWind']:
            to = P['tornado']
            center = math.atan2(to['y'] - rd.y, to['x'] - rd.x) if to['active'] else math.atan2(rd.wind_y, rd.wind_x) + math.pi / 2
            h = center + math.pi
            if to['active'] and math.hypot(to['vx'], to['vy']) > 0.5:
                side = math.copysign(1.0, (rd.x - to['x']) * -to['vy'] + (rd.y - to['y']) * to['vx']) or 1.0
                perp = math.atan2(to['vx'] * side, -to['vy'] * side)
                h = math.atan2(math.sin(h) + math.sin(perp), math.cos(h) + math.cos(perp))
            h = self._avoid_water(rd, b, h)
            return dict(name='wind', cmd=escape_heading(rd, h),
                        text='Wind %.0f m/s > %s, pressure −%.0f Pa → RUN %s away from vortex' % (rd.wind_speed, CFG['reflexWind'], rd.pressure_drop, compass(h)))
        # 5. tipping
        if rd.tilt_deg > CFG['reflexTilt']:
            ax, ay = rd.x + 2.5 * math.cos(rd.yaw), rd.y + 2.5 * math.sin(rd.yaw)
            if b.inside(ax, ay):
                b.obst[b.idx(ax, ay)] = 1.0
            back = (-1.0 if rd.pitch > 0 else 1.0) if abs(rd.pitch) > abs(rd.roll) else -1.0
            bx = rd.x + math.copysign(3, back) * math.cos(rd.yaw)
            by = rd.y + math.copysign(3, back) * math.sin(rd.yaw)
            if b.inside(bx, by) and b.water[b.idx(bx, by)] > 0.05:
                back = -back * 0.8
            w = 0.8 * math.copysign(1.0, rd.roll) if abs(rd.roll) > abs(rd.pitch) else 0.0
            return dict(name='tilt', cmd=tw(back, w), text='IMU tilt %.0f° > %.0f° → back off the slope, mark it impassable' % (rd.tilt_deg, CFG['reflexTilt']))
        # 6. stale data
        if rd.dt > CFG['staleFrame']:
            return dict(name='stale', cmd=tw(0, 0), text='Sensor frame %.2f s old → STOP' % rd.dt)
        # 7. low perception confidence (not while a known fire is close)
        fire_close = P['fire']['active'] and P['fire']['nearest'] < 30
        if rd.return_frac < CFG['lowConfReturnFrac'] and rd.smoke > 0.25 and not fire_close and rd.t - self.last_rescan > 10:
            if self.low_conf_since < 0:
                self.low_conf_since = rd.t
            if rd.t - self.low_conf_since >= 2.5:
                self.last_rescan = rd.t
                self.low_conf_since = -1.0
            else:
                return dict(name='rescan', cmd=tw(0, 0.6),
                            text='LiDAR returns %.0f%%, smoke %.2f → low confidence: STOP & RE-SCAN' % (rd.return_frac * 100, rd.smoke))
        else:
            self.low_conf_since = -1.0
        return None

    def _dodge_heading(self, rd, b, away):
        best, best_s = away, -1e18
        for k in range(16):
            a = away + (k - 8) * math.pi / 16
            s = 2 * math.cos(a - away) + min(clearance(rd, a), 10) / 10
            for d in (3, 6):
                px, py = rd.x + d * math.cos(a), rd.y + d * math.sin(a)
                if not b.inside(px, py):
                    s -= 10
                    continue
                i = b.idx(px, py)
                if b.water[i] > 0.15:
                    s -= 10
                if b.blocked(i):
                    s -= 4
                if b.channels[SLOPE][i] >= 0.6:
                    s -= 6
            if s > best_s:
                best_s, best = s, a
        return best

    def escape_search(self, rd, b, max_depth):
        best, best_s = rd.yaw + math.pi, -1e18
        heat = b.channels[HEAT]
        for k in range(24):
            a = k * math.pi / 12
            s = 0.0
            for (hb, hd, ht) in rd.hot_sectors:
                s -= (ht / max(4.0, hd * hd)) * max(0.0, math.cos(a - hb)) * 3
            for d in (3, 6, 10):
                px, py = rd.x + d * math.cos(a), rd.y + d * math.sin(a)
                if not b.inside(px, py):
                    s -= 30
                    continue
                i = b.idx(px, py)
                if b.blocked(i):
                    s -= 8
                wd = b.water[i]
                if wd > max_depth:
                    s -= 25 * (wd - max_depth + 0.1)
                s -= 6 * heat[i]
                if b.channels[SLOPE][i] >= 0.6:
                    s -= 10
                if b.water_dist[i] < 4 and b.channels[SLOPE][i] > 0.3:
                    s -= 8
            s += 0.5 * clearance(rd, a) / 30
            if s > best_s:
                best_s, best = s, a
        return best

    def _best_dry_heading(self, rd, b):
        best, best_s = rd.yaw + math.pi, -1e18
        for k in range(16):
            a = k * math.pi / 8
            s = 0.0
            for d in (3, 6, 9):
                px, py = rd.x + d * math.cos(a), rd.y + d * math.sin(a)
                if not b.inside(px, py):
                    s -= 50
                    continue
                i = b.idx(px, py)
                e = 0.0 if math.isnan(b.elev[i]) else b.elev[i]
                s += e - (3 * b.water[i] if b.water[i] > 0.05 else 0) - (20 if b.blocked(i) else 0) + (0.5 if b.water[i] == 0 else 0)
            if s > best_s:
                best_s, best = s, a
        return best

    def _avoid_water(self, rd, b, h):
        for off in (0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6):
            a = h + off
            ok = True
            for d in (3, 6, 10):
                px, py = rd.x + d * math.cos(a), rd.y + d * math.sin(a)
                if not b.inside(px, py):
                    ok = False
                    break
                i = b.idx(px, py)
                if b.water[i] > 0.15 or b.blocked(i):
                    ok = False
                    break
            if ok:
                return a
        return h
