"""Alien grazers: small herds of slow six-legged creatures (protocol v4 `lifeforms`, ground truth).

Each creature is a MuJoCo mocap body (world.creature_body_xml) moved once per decision cycle:
  - it wanders inside its herd's home area at 0.2 to 0.5 m/s, with pauses to graze,
  - keeps loose herd cohesion (back toward the herd's middle if it strays, apart if too close),
  - avoids steep ground, liquid, rocks and spires (it knows the true terrain: it lives there),
  - shies away from any rover closer than about 3 m (and speeds up to 0.5 m/s).
They are visible to LiDAR and the cameras but nothing collides with them. The rovers know nothing of
this module; they only see LiDAR points that move (tracker.py) and camera pictures.
"""
from __future__ import annotations

import math

import numpy as np

SHY_M = 3.0           # a rover closer than this makes a creature walk away
FLEE_V = 0.5
MAX_V = 0.5
TURN_RATE = 1.2       # rad/s
COHESION_M = 3.5      # beyond this from the herd's middle, steer back
SEPARATION_M = 1.4
PROBE_M = 1.0         # look this far ahead for bad ground
MAX_SLOPE = 15.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Creature:
    def __init__(self, cid: str, herd, mocap: int, x: float, y: float, yaw: float, size: float, cruise: float,
                 rng: np.random.Generator):
        self.id = cid
        self.herd = herd
        self.mocap = mocap
        self.x, self.y, self.yaw = x, y, yaw
        self.z = 0.0
        self.size = size
        self.cruise = cruise
        self.speed = 0.0
        self.goal = (x, y)
        self.pause_until = 0.0
        self.next_goal = 0.0
        self.phase = float(rng.uniform(0, 2 * math.pi))

    def to_msg(self) -> dict:
        return {"id": self.id, "kind": "grazer", "x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2),
                "yaw": round(self.yaw, 3), "speed": round(self.speed, 2), "size": round(self.size, 2)}


class Herd:
    def __init__(self, spec):
        self.id = spec.id
        self.cx, self.cy, self.radius = spec.cx, spec.cy, spec.radius
        self.members: list = []

    def centre(self):
        xs = [c.x for c in self.members]
        ys = [c.y for c in self.members]
        return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else (self.cx, self.cy)


class LifeformManager:
    def __init__(self, sim):
        self.sim = sim
        self.cfg = sim.cfg
        self.ter = sim.terrain
        self.rng = np.random.default_rng(self.cfg.seed * 4253 + 17)
        self.herds: list = []
        self.creatures: list = []
        mocaps = list(sim.world.life_mocap or [])
        k = 0
        for spec in getattr(self.ter, "herds", []):
            h = Herd(spec)
            for n, (x, y, yaw, size, cruise) in enumerate(spec.members):
                if k >= len(mocaps):
                    break
                c = Creature(f"l{k + 1}", h, mocaps[k], x, y, yaw, size, cruise, self.rng)
                h.members.append(c)
                self.creatures.append(c)
                k += 1
            self.herds.append(h)
        self._half = (self.cfg.nx - 1) * self.cfg.res / 2.0
        self._rocky = None
        for c in self.creatures:
            self._place(c, 0.0)

    # ------------------------------------------------------------------ ground checks (truth)
    def _bad(self, x: float, y: float) -> bool:
        ter = self.ter
        if abs(x) > self._half - 2.0 or abs(y) > self._half - 2.0:
            return True
        if math.hypot(x, y) < self.cfg.lander_radius + 2.0:
            return True                     # they keep off the strange thing that landed
        i, j = ter.ij(x, y)
        if ter.slope[j, i] > MAX_SLOPE or ter.wet[j, i]:
            return True
        if self._rocky is None:
            self._rocky = (ter.truth_heights - ter.heights) > 0.15
        return bool(self._rocky[j, i])

    def _place(self, c: Creature, t: float) -> None:
        d = self.sim.data
        c.z = self.ter.height_at(c.x, c.y)
        bob = 0.02 * math.sin(c.phase + 6.0 * t) if c.speed > 0.05 else 0.0
        d.mocap_pos[c.mocap] = (c.x, c.y, c.z + bob)
        d.mocap_quat[c.mocap] = (math.cos(c.yaw / 2), 0.0, 0.0, math.sin(c.yaw / 2))

    # ------------------------------------------------------------------ behaviour
    def step(self, t: float, dt: float) -> None:
        if not self.creatures:
            return
        rovers = [(r.x, r.y) for r in self.sim.rovers]
        rng = self.rng
        for h in self.herds:
            if not h.members:
                continue
            hx, hy = h.centre()
            for c in h.members:
                flee = None
                for rx, ry in rovers:
                    dd = math.hypot(c.x - rx, c.y - ry)
                    if dd < SHY_M and (flee is None or dd < flee[0]):
                        flee = (dd, rx, ry)
                if flee is not None:
                    want = math.atan2(c.y - flee[2], c.x - flee[1])
                    v = FLEE_V
                    c.pause_until = t + float(rng.uniform(1.0, 3.0))
                elif t < c.pause_until:
                    want, v = c.yaw, 0.0
                else:
                    if t >= c.next_goal or math.hypot(c.goal[0] - c.x, c.goal[1] - c.y) < 0.8:
                        rr = h.radius * math.sqrt(float(rng.uniform()))
                        aa = float(rng.uniform(0, 2 * math.pi))
                        c.goal = (h.cx + rr * math.cos(aa), h.cy + rr * math.sin(aa))
                        c.next_goal = t + float(rng.uniform(12.0, 25.0))
                        if rng.uniform() < 0.35:            # stop to graze for a while
                            c.pause_until = t + float(rng.uniform(3.0, 8.0))
                    gx, gy = c.goal[0] - c.x, c.goal[1] - c.y
                    dch = math.hypot(c.x - hx, c.y - hy)
                    if dch > COHESION_M:                    # loose herd cohesion
                        w = min(1.0, (dch - COHESION_M) / 3.0)
                        gx = (1 - w) * gx + w * (hx - c.x) * 2.0
                        gy = (1 - w) * gy + w * (hy - c.y) * 2.0
                    for o in h.members:
                        if o is c:
                            continue
                        do = math.hypot(c.x - o.x, c.y - o.y)
                        if 1e-6 < do < SEPARATION_M:
                            gx += (c.x - o.x) / do * 1.5
                            gy += (c.y - o.y) / do * 1.5
                    want = math.atan2(gy, gx)
                    v = c.cruise
                # avoid steep ground, liquid, rocks and spires: try headings around the wanted one
                if v > 0.0:
                    for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi):
                        a = want + off
                        if not self._bad(c.x + PROBE_M * math.cos(a), c.y + PROBE_M * math.sin(a)):
                            want = a
                            break
                    else:
                        v = 0.0
                turn = _wrap(want - c.yaw)
                c.yaw = _wrap(c.yaw + max(-TURN_RATE * dt, min(TURN_RATE * dt, turn)))
                if abs(turn) > 1.2:
                    v *= 0.3                                # turn on the spot first
                c.speed += max(-0.5 * dt, min(0.5 * dt, min(v, MAX_V) - c.speed))
                nx_, ny_ = c.x + c.speed * dt * math.cos(c.yaw), c.y + c.speed * dt * math.sin(c.yaw)
                if not self._bad(nx_, ny_):
                    c.x, c.y = nx_, ny_
                else:
                    c.speed = 0.0
                self._place(c, t)

    # ------------------------------------------------------------------ demo: bring a herd into view
    def move_herd(self, herd: Herd, x: float, y: float, face: float | None = None) -> None:
        """Teleport a herd around (x, y) and make that its home area."""
        herd.cx, herd.cy = x, y
        rng = self.rng
        placed = []
        for c in herd.members:
            for _ in range(80):
                rr, aa = float(rng.uniform(0.0, 2.2)), float(rng.uniform(0, 2 * math.pi))
                px, py = x + rr * math.cos(aa), y + rr * math.sin(aa)
                if self._bad(px, py) or any(math.hypot(px - qx, py - qy) < 1.3 for qx, qy in placed):
                    continue
                break
            else:
                px, py = x, y
            placed.append((px, py))
            c.x, c.y = px, py
            c.yaw = float(rng.uniform(0, 2 * math.pi)) if face is None else face + float(rng.uniform(-0.6, 0.6))
            c.goal = (px + 2.0 * math.cos(c.yaw), py + 2.0 * math.sin(c.yaw))
            c.next_goal = self.sim.t + float(rng.uniform(6.0, 12.0))
            c.pause_until = 0.0                 # start walking at once, so the rovers see motion
            c.speed = c.cruise
            self._place(c, self.sim.t)

    def spot_ok(self, x: float, y: float) -> bool:
        """Can a herd stand around (x, y)?"""
        return not self._bad(x, y) and sum(not self._bad(x + 1.5 * math.cos(a), y + 1.5 * math.sin(a))
                                           for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)) >= 6

    def to_msgs(self) -> list:
        return [c.to_msg() for c in self.creatures]
