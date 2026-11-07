"""SAFETY REFLEXES: hard limits that override every behaviour instantly. Mirror of web/brain/safety.ts."""
import math

from .config import CFG, compass, wrap


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


class Safety:
    def __init__(self):
        self.low_conf_since = -1.0

    def check(self, rd, b, P):
        # 1. incoming meteor
        worst = None
        for s in rd.streaks:
            d = math.hypot(s['x'] - rd.x, s['y'] - rd.y)
            if d < s['size'] + 4 and (worst is None or d - s['size'] < worst[0] - worst[4]):
                worst = (d, s['ttl'], s['x'], s['y'], s['size'])
        if worst:
            d, ttl, wx, wy, _ = worst
            away = math.atan2(rd.y - wy, rd.x - wx) if d > 0.5 else rd.yaw
            return dict(name='meteor_dodge', cmd=escape_heading(rd, away),
                        text='Camera: meteor streak, impact %.1f m %s in %.1f s → DODGE %s' % (d, compass(math.atan2(wy - rd.y, wx - rd.x)), ttl, compass(away)))
        # 2. water
        if rd.depth > CFG['reflexWater']:
            h = self._best_dry_heading(rd, b)
            return dict(name='water', cmd=escape_heading(rd, h, 2.2),
                        text='Water depth %.2f m > %s m limit → climb out %s' % (rd.depth, CFG['reflexWater'], compass(h)))
        # 3. heat
        close = [s for s in rd.hot_sectors if s[1] < 9]
        if rd.temp_c > CFG['reflexTemp'] or close:
            rx = ry = 0.0
            for (a, dist, T) in rd.hot_sectors:
                w = T / max(1.0, dist * dist)
                rx -= w * math.cos(a)
                ry -= w * math.sin(a)
            h = math.atan2(ry, rx) if (rx or ry) else rd.yaw + math.pi
            h = self._avoid_water(rd, b, h)
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
            w = 0.8 * math.copysign(1.0, rd.roll) if abs(rd.roll) > abs(rd.pitch) else 0.0
            return dict(name='tilt', cmd=tw(back, w), text='IMU tilt %.0f° > %.0f° → back off the slope, mark it impassable' % (rd.tilt_deg, CFG['reflexTilt']))
        # 6. stale data
        if rd.dt > CFG['staleFrame']:
            return dict(name='stale', cmd=tw(0, 0), text='Sensor frame %.2f s old → STOP' % rd.dt)
        # 7. low perception confidence
        if rd.return_frac < CFG['lowConfReturnFrac'] and rd.smoke > 0.25:
            if self.low_conf_since < 0:
                self.low_conf_since = rd.t
            if rd.t - self.low_conf_since < 2.5:
                return dict(name='rescan', cmd=tw(0, 0.6),
                            text='LiDAR returns %.0f%%, smoke %.2f → low confidence: STOP & RE-SCAN' % (rd.return_frac * 100, rd.smoke))
        else:
            self.low_conf_since = -1.0
        return None

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
