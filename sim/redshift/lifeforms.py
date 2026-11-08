"""Alien grazers: small herds of slow six-legged creatures (protocol v4 `lifeforms`, ground truth).

Each creature is a MuJoCo mocap body (world.creature_body_xml) moved once per decision cycle:
  - it wanders inside its herd's home area at 0.2 to 0.5 m/s, with pauses to graze,
  - keeps loose herd cohesion (back toward the herd's middle if it strays, apart if too close),
  - avoids steep ground, liquid, rocks and spires (it knows the true terrain: it lives there),
  - shies away from any rover closer than about 3 m (and speeds up to 0.5 m/s).
They are visible to LiDAR and the cameras. The rovers know nothing of this module; they only see LiDAR points
that move (tracker.py) and camera pictures.

Protocol v9: the biggest grazer of each herd is territorial. A rover that stays within TERRITORY_M of it for
TERRITORY_S makes it warn (it stops, faces the rover and steps toward it) and then charge at CHARGE_V, homing on
the rover a little, for only as far as it takes to reach where the rover was (a rover that backs away in time is
missed). A creature that reaches a rover rams it: a force on the rover's chassis in MuJoCo for a few substeps
(health.DamageMonitor.push), which the rover's IMU feels like any other shock. A creature that walks into a rover
without charging bumps it gently (usually below any damage). Then it trots back to its herd.
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
TERRITORY_M = 3.5     # protocol v9: a rover this close to the territorial grazer ...
TERRITORY_S = 12.0    # ... for this long provokes a charge (observers keep 4 m or more)
WARN_S = 1.2          # it warns first (stops, faces the rover, steps toward it)
WARN_V = 0.3
CHARGE_V = 2.0        # m/s
CHARGE_TURN = 1.0     # rad/s while charging (homing a little)
CHARGE_COOLDOWN_S = 90.0
CREATURE_MASS = 120.0  # kg (a grazer about 1.3 m long, built for 2 g)
RAM_S = 0.02          # a ram lasts this long (s): the impulse of a partly elastic collision, spread over it
CONTACT_M = 0.55      # body contact: centre distance below this plus half the creature's length
PROBE_M = 1.0         # look this far ahead for bad ground
MAX_SLOPE = 15.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Creature:
    kind = "grazer"           # v7 life variety sets another kind on the creature (v7_life.py)

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
        self.territorial = False             # protocol v9: the herd's biggest charges rovers that stay close
        self.annoy = 0.0
        self.mode = None                     # None, "warn", "charge", "retreat"
        self.mode_t = 0.0
        self.target = None                   # the rover it charges
        self.travel = 0.0                    # charge distance left (m)
        self.cooldown = 0.0
        self.provoked_at = None              # demo: charge this rover at this time
        self.bumped = -1e9

    def to_msg(self) -> dict:
        if self.kind != "grazer":
            from .v7_life import msg_extra
            return {"id": self.id, "kind": self.kind, "x": round(self.x, 2), "y": round(self.y, 2),
                    "z": round(self.z, 2), "yaw": round(self.yaw, 3), "speed": round(self.speed, 2),
                    "size": round(self.size, 2), **msg_extra(self)}
        return {"id": self.id, "kind": "grazer", "x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2),
                "yaw": round(self.yaw, 3), "speed": round(self.speed, 2), "size": round(self.size, 2)}


class Herd:
    def __init__(self, spec):
        self.id = spec.id
        self.cx, self.cy, self.radius = spec.cx, spec.cy, spec.radius
        self.kind = getattr(spec, "kind", "grazer")
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
                if h.kind != "grazer":
                    from . import v7_life
                    c.kind = h.kind
                    v7_life.init_creature(self, h, c)
                h.members.append(c)
                self.creatures.append(c)
                k += 1
            if h.members:
                max(h.members, key=lambda c_: c_.size).territorial = True
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
        if c.kind != "grazer":
            from . import v7_life
            v7_life.place(self, c, t)
            return
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
            if h.kind != "grazer":             # v7 life variety: the other kinds behave their own way
                from . import v7_life
                v7_life.step_herd(self, h, t, dt, rovers)
                continue
            hx, hy = h.centre()
            for c in h.members:
                if c.territorial and self._territory(c, t, dt):
                    self._place(c, t)
                    continue
                self._bump(c, t)
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

    # ------------------------------------------------------------------ protocol v9: territory and charges
    def provoke(self, herd: Herd, rover, at_t: float) -> None:
        """Demo: the herd's territorial member charges this rover at sim time at_t."""
        c = next((c_ for c_ in herd.members if c_.territorial), None)
        if c is not None:
            c.provoked_at = (rover.id, at_t)
            c.cooldown = 0.0

    def _territory(self, c: Creature, t: float, dt: float) -> bool:
        """Territorial behaviour. True while it is warning, charging or trotting back (normal moves skipped)."""
        rovers = self.sim.rovers
        if c.mode is None:
            near = min(rovers, key=lambda r: math.hypot(r.x - c.x, r.y - c.y), default=None)
            if near is None:
                return False
            dd = math.hypot(near.x - c.x, near.y - c.y)
            if dd < TERRITORY_M and near.state != "towed":
                c.annoy += dt
            else:
                c.annoy = max(0.0, c.annoy - 2.0 * dt)
            target = None
            if c.provoked_at is not None and t >= c.provoked_at[1]:
                target = next((r for r in rovers if r.id == c.provoked_at[0]), None)
                c.provoked_at = None
            elif c.annoy >= TERRITORY_S and t >= c.cooldown:
                target = near
            if target is None:
                return False
            c.mode, c.mode_t, c.target, c.annoy = "warn", t, target.id, 0.0
            return True
        tgt = next((r for r in rovers if r.id == c.target), None)
        if tgt is None:
            c.mode = None
            return False
        dx, dy = tgt.x - c.x, tgt.y - c.y
        dist = math.hypot(dx, dy)
        want = math.atan2(dy, dx)
        if c.mode == "warn":
            c.yaw = _wrap(c.yaw + max(-2.0 * dt, min(2.0 * dt, _wrap(want - c.yaw))))
            c.speed = WARN_V if dist > 2.5 else 0.0
            self._move(c, dt)
            if t - c.mode_t >= WARN_S:
                c.mode, c.mode_t = "charge", t
                c.yaw = want
                c.travel = max(0.5, dist - (CONTACT_M + 0.5 * c.size) + 0.1)      # just enough to reach it
            return True
        if c.mode == "charge":
            c.yaw = _wrap(c.yaw + max(-CHARGE_TURN * dt, min(CHARGE_TURN * dt, _wrap(want - c.yaw))))
            c.speed = CHARGE_V
            step = min(CHARGE_V * dt, max(c.travel, 0.0))
            reach = CONTACT_M + 0.5 * c.size
            c.speed = step / dt
            moved = self._move(c, dt)
            c.travel -= step
            if math.hypot(tgt.x - c.x, tgt.y - c.y) <= reach:
                self._ram(c, tgt, t, CHARGE_V, "charge")
                c.mode, c.mode_t = "retreat", t
                c.cooldown = t + CHARGE_COOLDOWN_S
                return True
            if c.travel <= 0.0 or not moved:
                c.mode, c.mode_t = "retreat", t             # it fell short: a bluff
                c.cooldown = t + CHARGE_COOLDOWN_S
                self.sim.fleet.log(t, "hazard", f"A grazer charged {tgt.name} and stopped short.", None, at=(c.x, c.y))
            return True
        # retreat: trot back toward the herd's middle for a few seconds
        hx, hy = c.herd.centre()
        c.yaw = _wrap(c.yaw + max(-TURN_RATE * dt, min(TURN_RATE * dt, _wrap(math.atan2(hy - c.y, hx - c.x) - c.yaw))))
        c.speed = 0.8
        self._move(c, dt)
        if t - c.mode_t > 4.0:
            c.mode, c.target = None, None
            c.speed = 0.3
        return True

    def _move(self, c: Creature, dt: float) -> bool:
        nx_, ny_ = c.x + c.speed * dt * math.cos(c.yaw), c.y + c.speed * dt * math.sin(c.yaw)
        if self._bad(nx_, ny_):
            c.speed = 0.0
            return False
        c.x, c.y = nx_, ny_
        return True

    def _ram(self, c: Creature, rover, t: float, v: float, why: str) -> None:
        """The creature hits the rover: a horizontal force on its chassis (MuJoCo xfrc_applied) for RAM_S."""
        mon = getattr(self.sim, "damage", None)
        ux, uy = math.cos(c.yaw), math.sin(c.yaw)
        mu = CREATURE_MASS * rover.mass / (CREATURE_MASS + rover.mass)      # reduced mass of the two
        f = 1.5 * mu * v / RAM_S                                             # impulse (restitution 0.5) / duration
        steps = max(1, int(round(RAM_S / self.sim.model.opt.timestep)))
        # where the blow came from, in the rover's frame (+x forward, +y left)
        cy_, sy_ = math.cos(rover.yaw), math.sin(rover.yaw)
        bx, by = -ux, -uy
        local = (bx * cy_ + by * sy_, -bx * sy_ + by * cy_)
        if mon is not None:
            mon.push(rover.idx, (f * ux, f * uy, 0.1 * f), steps, "lifeform", local)
        if why == "charge":
            self.sim.fleet.log(t, "hazard", f"A grazer rammed {rover.name} at {v:.1f} m/s.", None, at=(rover.x, rover.y))
        c.bumped = t

    def _bump(self, c: Creature, t: float) -> None:
        """A creature that walks into a rover without charging bumps it gently."""
        if c.speed < 0.15 or t - c.bumped < 3.0:
            return
        reach = CONTACT_M + 0.5 * c.size
        for r in self.sim.rovers:
            if math.hypot(r.x - c.x, r.y - c.y) < reach:
                ang = math.atan2(r.y - c.y, r.x - c.x)
                if abs(_wrap(ang - c.yaw)) < 1.2:
                    self._ram(c, r, t, c.speed, "bump")
                    return

    # ------------------------------------------------------------------ demo: bring a herd into view
    def move_herd(self, herd: Herd, x: float, y: float, face: float | None = None) -> None:
        """Teleport a herd around (x, y) and make that its home area."""
        herd.cx, herd.cy = x, y
        rng = self.rng
        placed = []
        spread, sep = 2.2, 1.3
        if herd.kind != "grazer":
            from .v7_life import KINDS
            spread, sep = KINDS[herd.kind].spread, KINDS[herd.kind].sep
        for c in herd.members:
            for _ in range(80):
                rr, aa = float(rng.uniform(0.0, spread)), float(rng.uniform(0, 2 * math.pi))
                px, py = x + rr * math.cos(aa), y + rr * math.sin(aa)
                if self._bad(px, py) or any(math.hypot(px - qx, py - qy) < sep for qx, qy in placed):
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
        if herd.kind != "grazer":
            from . import v7_life
            v7_life.on_moved(self, herd, self.sim.t)

    def spot_ok(self, x: float, y: float) -> bool:
        """Can a herd stand around (x, y)?"""
        return not self._bad(x, y) and sum(not self._bad(x + 1.5 * math.cos(a), y + 1.5 * math.sin(a))
                                           for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)) >= 6

    def to_msgs(self) -> list:
        return [c.to_msg() for c in self.creatures]
