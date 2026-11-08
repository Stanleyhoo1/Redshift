"""Rover health and damage (protocol v9).

Every rover has six parts, 0 to 100 (100 = like new): drive_l, drive_r, lidar, camera, power, comms. Damage comes
from what happens in the physics, never from a script:

  hard hits     the IMU's accelerometer (the chassis' linear acceleration from MuJoCo, averaged over 20 ms like a
                shock sensor) and the chassis skin (a MuJoCo touch sensor over the body: contact force on the chassis).
                The contact that caused it (a boulder, a spire, a rock, the ground, another rover) and where it
                touched the rover pick the parts: a side hit damages that side's drive, the mast top the LiDAR and
                the camera, the rear deck the power system and the antenna.
  rollover      the IMU's tilt (mast, camera and antenna take the knock).
  immersion     the belly water probe (power and both drives).
  motor strain  each side's motor torque at its limit for a long time (pushing into sand, a long stall) heats the
                motors until the controller trips (a "tripped motor controller": the side stops until it is reset).
  dust          a storm coats the camera window and the solar deck (cleaned at the lander or by a helper).
  lifeforms     a creature that runs into a rover pushes it (a force on the chassis in MuJoCo); the IMU feels the
                shock like any other hit.

Damage has a kind: "broken" (needs a lander repair, and a spare below 30%) or a field-fixable kind: "jam" (a jammed
wheel), "trip" (a tripped motor controller), "dirt" (dust on the optics). A helper rover fixes those on the spot.

Effects go through the existing mechanics (rover.py reads them every cycle): a weak drive side gets a lower torque
cap, top wheel speed and servo gain (the rover pulls toward that side and climbs less), a damaged LiDAR loses range
and rays, a damaged camera adds noise to its frames (the detector sees it), damaged power drains the battery faster
and damaged comms shorten the radio range.

The parts in each tick are the rover's own built-in self tests (motor current against wheel speed per side, LiDAR
return count against what the range should give, camera noise, battery drain, radio link margin). In the sim those
readings follow the damage the physics caused.
"""
from __future__ import annotations

import math
from collections import deque

import mujoco
import numpy as np

PARTS = ("drive_l", "drive_r", "lidar", "camera", "power", "comms")
SPARE_OF = {"drive_l": "drive", "drive_r": "drive", "lidar": "lidar", "camera": "camera", "power": "power"}
SPARE_KINDS = ("drive", "lidar", "camera", "power")
SPARE_NEED_BELOW = 30.0        # protocol: a drive, LiDAR, camera or power part below this needs a spare
FIELD_KINDS = ("jam", "trip", "dirt")
PART_NAME = {"drive_l": "left drive", "drive_r": "right drive", "lidar": "LiDAR", "camera": "camera",
             "power": "power system", "comms": "radio"}
MOBILE_DRIVE_MIN = 15.0        # a skid steer needs both sides: either side below this and it cannot drive
MOBILE_POWER_MIN = 10.0        # ... nor with the power system below this
OK_MIN = 80.0                  # every part at least this: status "ok"
WRECK_HP = 30.0                # immobile and hp below this (or 3 parts broken): beyond field repair, "wrecked"
PATCH_DRIVE = 25.0             # a field patch brings a dead drive side back to this (enough to limp home) ...
PATCH_POWER = 25.0             # ... and the power system to this
REPAIR_HOME_HP = 55.0          # a rover that can still drive heads home for repair below this hp ...
REPAIR_HOME_PART = 35.0        # ... or with a drive, the LiDAR or power below this
COMMS_FULL_M = 220.0           # radio range with comms at 100% (covers the whole 120 m map)
NOMINAL_CLIMB_DEG = 20.5       # measured climb limit at 2 g with healthy motors (scripts/climb_test.py)

# ---- physics thresholds (tuned with a 900 s no-hazard run: see README "Health and damage") ----
AVG_STEPS = 4                  # accelerometer averaging window (4 x 5 ms = 20 ms, like a shock sensor)
SHOCK_G0 = 6.0                 # 20 ms chassis shocks below this are normal driving at 2 g (measured up to 4.5)
SHOCK_K = 9.0                  # damage points per g above that
TOUCH_N0 = 3500.0              # chassis skin force below this is normal (measured up to 2.2 kN on rocks)
TOUCH_K = 25.0                 # damage points per doubling of the skin force above that (a crushing contact)
CAPTURE_G = 4.0                # remember which contact caused a shock above this (cheap: only then)
CAPTURE_N = 1200.0             # ... or a skin force above this
HIT_WINDOW_S = 4.0             # one hit: only a stronger shock within this window adds damage
STRAIN_LOAD = 0.97             # a side's torque this close to its limit heats its motors ...
STRAIN_TRIP_S = 75.0           # ... and this many seconds of it (net of cooling) trip the motor controller
                               #     (long climbs at the limit measured up to 36 s)
STRAIN_COOL = 0.5              # cooling per second below 80% load
DUST_CAMERA = 0.12             # dirt per second of storm on the camera window (the lander's panels: hazards.py)
WET_EVERY_S = 60.0             # water damage at most once a minute (a rover backing out may dip in again)
KIND_CODES = {"terrain": 1, "rock": 2, "boulder": 3, "spire": 4, "rover": 5, "wheel": 6, "chassis": 7, "other": 0}


def _cause_word(cause: str) -> str:
    return {"boulder": "a boulder", "spire": "a spire", "rock": "a rock", "terrain": "the ground",
            "rover": "another rover", "lifeform": "a charging creature", "rollover": "tipping over",
            "water": "water in the chassis", "strain": "motor overheating", "dust": "dust",
            "operator": "a demo hit", "shock": "a hard knock"}.get(cause, cause)


class Health:
    """One rover's parts, status and self diagnosis."""

    def __init__(self, rover):
        self.rover = rover
        self.perm = {p: 0.0 for p in PARTS}       # permanent loss (lander repair, spares)
        self.soft = {p: 0.0 for p in PARTS}       # field-fixable loss
        self.soft_kind = {p: None for p in PARTS}
        self.last_hit = None                      # {"t", "cause", "shock_g"}
        self.log: deque = deque(maxlen=30)        # damage dicts (with "t")
        self.heat = [0.0, 0.0]                    # motor strain, left and right (s)
        self.version = 0                          # bumps on every change
        self._hit_t = -1e9
        self._hit_peak = 0.0
        self.dust_this_storm = 0.0
        self.dust_logged = False
        self.flip_hit = False
        self.wet_hit = False
        self._eff = None
        self._eff_v = -1

    # ------------------------------------------------------------ readings
    def value(self, p: str, field_fixed: bool = False) -> float:
        loss = self.perm[p] + (0.0 if field_fixed else self.soft[p])
        return max(0.0, min(100.0, 100.0 - loss))

    def values(self, field_fixed: bool = False) -> dict:
        return {p: self.value(p, field_fixed) for p in PARTS}

    def parts_msg(self) -> dict:
        return {p: int(round(self.value(p))) for p in PARTS}

    @staticmethod
    def hp_of(vals: dict) -> int:
        v = list(vals.values())
        return int(round(0.4 * min(v) + 0.6 * sum(v) / len(v)))

    def hp(self) -> int:
        return self.hp_of(self.values())

    def mobile(self, field_fixed: bool = False) -> bool:
        v = self.values(field_fixed)
        return min(v["drive_l"], v["drive_r"]) >= MOBILE_DRIVE_MIN and v["power"] >= MOBILE_POWER_MIN

    def broken_parts(self) -> list:
        """Parts with permanent damage below 10%."""
        return [p for p in PARTS if self.value(p, field_fixed=True) < 10.0]

    def beyond_field_repair(self) -> bool:
        """Immobile even after a field repair and badly hurt overall: a wreck."""
        if self.mobile(field_fixed=True):
            return False
        return self.hp_of(self.values(field_fixed=True)) < WRECK_HP or len(self.broken_parts()) >= 3

    def status(self) -> str:
        if not self.mobile():
            return "wrecked" if self.beyond_field_repair() else "immobile"
        if min(self.values().values()) < OK_MIN:
            return "damaged"
        return "ok"

    def fixable(self) -> list:
        """[(part, kind, points)] the helper can fix on the spot."""
        return [(p, self.soft_kind[p] or "jam", self.soft[p]) for p in PARTS if self.soft[p] > 0.5]

    def needs_repair(self) -> bool:
        """Can drive, but should go home for a repair: low hp or a key part low."""
        v = self.values()
        return self.hp() < REPAIR_HOME_HP or min(v["drive_l"], v["drive_r"], v["lidar"], v["power"]) < REPAIR_HOME_PART

    def worn(self) -> bool:
        """Anything a lander service would improve."""
        return any(self.value(p) < 99.5 for p in PARTS)

    # ------------------------------------------------------------ effects (cached per version)
    def effects(self) -> dict:
        if self._eff is not None and self._eff_v == self.version:
            return self._eff
        v = self.values()
        f = {p: v[p] / 100.0 for p in PARTS}
        fl, fr = f["drive_l"], f["drive_r"]
        lo = min(fl, fr)
        e = {
            "drive_l": fl, "drive_r": fr,
            # top speed the rover allows itself: the weak side must keep up (the controller compensates)
            "v_cap": max(0.15, min(1.0, 0.15 + lo)) * (1.0 if f["lidar"] >= 0.6 else 0.5 + 0.5 * f["lidar"])
            * (1.0 if f["power"] >= 0.4 else 0.75),
            "lidar_range": 0.35 + 0.65 * f["lidar"],
            "lidar_drop": 0.6 * (1.0 - f["lidar"]),
            "camera_noise": 70.0 * (1.0 - f["camera"]),
            "drain": 1.0 + 2.5 * (1.0 - f["power"]),
            "comms_m": COMMS_FULL_M * f["comms"],
        }
        frac = 0.5 * lo + 0.25 * (fl + fr)
        if frac < 0.97:
            s = max(0.0, min(1.0, frac * math.sin(math.radians(NOMINAL_CLIMB_DEG))))
            e["slope_cap"] = max(4.0, math.degrees(math.asin(s)) - 1.0)
        else:
            e["slope_cap"] = None
        self._eff, self._eff_v = e, self.version
        return e

    # ------------------------------------------------------------ changes
    def apply(self, t: float, part: str, amount: float, cause: str, kind: str = "broken", shock_g=None,
              log: bool = True, also: str = "") -> dict | None:
        """Damage one part. Returns the protocol damage dict (or None if nothing changed) and logs the event."""
        if part not in PARTS or amount <= 0.0:
            return None
        before = self.value(part)
        if before <= 0.0:
            return None
        if kind in FIELD_KINDS:
            self.soft[part] = min(100.0 - self.perm[part], self.soft[part] + amount)
            self.soft_kind[part] = kind
        else:
            self.perm[part] = min(100.0, self.perm[part] + amount)
            self.soft[part] = min(self.soft[part], 100.0 - self.perm[part])
        after = self.value(part)
        if int(round(after)) == int(round(before)):
            return None
        self.version += 1
        dmg = {"rover": self.rover.id, "part": part, "before": int(round(before)), "after": int(round(after)),
               "cause": cause, "shock_g": None if shock_g is None else round(float(shock_g), 1)}
        self.log.append(dict(dmg, t=round(float(t), 1), kind=kind))
        self.last_hit = {"t": round(float(t), 1), "cause": cause,
                         "shock_g": None if shock_g is None else round(float(shock_g), 1)}
        if log:
            self._log_damage(t, dmg, kind, also)
        return dmg

    def _log_damage(self, t: float, dmg: dict, kind: str, also: str) -> None:
        r = self.rover
        how = {"jam": " (wheel jammed)", "trip": " (motor controller tripped)", "dirt": " (dirty)"}.get(kind, "")
        g = f", IMU shock {dmg['shock_g']:.1f} g" if dmg.get("shock_g") else ""
        msg = (f"{r.id} damaged by {_cause_word(dmg['cause'])}{g}: {PART_NAME[dmg['part']]} {dmg['before']}% -> "
               f"{dmg['after']}%{how}.{(' ' + also) if also else ''} Health {self.hp()} ({self.status()}).")
        r.fleet.log(t, "warn", msg, r.id, at=(r.x, r.y), damage=dmg)
        r.mem.on_hazard("damage", t, (r.x, r.y), f"damage from {_cause_word(dmg['cause'])}: "
                                                   f"{PART_NAME[dmg['part']]} {dmg['before']}% to {dmg['after']}%")

    def hit(self, t: float, parts: list, severity: float, cause: str, shock_g=None, kinds: dict | None = None) -> list:
        """One physical hit: severity points on the main part, shares on the others. parts: [(part, share)]."""
        out = []
        for p, share in parts:
            amt = severity * share
            if amt < 3.0:
                continue
            kind = (kinds or {}).get(p)
            before = self.value(p)
            if kind is None and p in ("drive_l", "drive_r") and amt >= 6.0:
                # a knock on a wheel jams part of it (the helper can free it); the rest is broken
                self.apply(t, p, 0.4 * amt, cause, "jam", shock_g, log=False)
                self.apply(t, p, 0.6 * amt, cause, "broken", shock_g, log=False)
                kind = "jam"
            else:
                self.apply(t, p, amt, cause, kind or "broken", shock_g, log=False)
            after = self.value(p)
            if int(round(after)) != int(round(before)):
                out.append(({"rover": self.rover.id, "part": p, "before": int(round(before)),
                             "after": int(round(after)), "cause": cause,
                             "shock_g": None if shock_g is None else round(float(shock_g), 1)}, kind or "broken"))
        if not out:
            return []
        out.sort(key=lambda e: e[0]["after"] - e[0]["before"])       # the biggest drop is the event's damage
        main, kind = out[0]
        also = "; ".join(f"{PART_NAME[d['part']]} {d['before']}% -> {d['after']}%" for d, _k in out[1:])
        self._log_damage(t, main, kind, f"Also {also}." if also else "")
        return [d for d, _k in out]

    def can_patch(self) -> bool:
        """After the field fix it still can't drive, but it is not beyond field repair: a helper can patch the
        drive and power (bypass a burnt winding, rewire a cell) just enough to limp home for a lander repair."""
        if not getattr(self.rover.cfg, "field_patch", True):
            return False
        return not self.mobile(field_fixed=True) and not self.beyond_field_repair()

    def field_patch(self) -> list:
        """The helper's temporary patch: drives up to PATCH_DRIVE and power up to PATCH_POWER, no more (the
        lander repairs it properly). Returns [(part, "patch", before, after)]."""
        out = []
        for p, floor in (("drive_l", PATCH_DRIVE), ("drive_r", PATCH_DRIVE), ("power", PATCH_POWER)):
            before = self.value(p, field_fixed=True)
            if before < floor:
                self.perm[p] = 100.0 - floor
                out.append((p, "patch", int(round(before)), int(round(self.value(p)))))
        if out:
            self.version += 1
        return out

    def field_fix(self) -> list:
        """The helper's fix: clears every field-fixable fault. Returns [(part, kind, before, after)]."""
        out = []
        for p in PARTS:
            if self.soft[p] > 0.5:
                before = self.value(p)
                kind = self.soft_kind[p] or "jam"
                self.soft[p] = 0.0
                self.soft_kind[p] = None
                out.append((p, kind, int(round(before)), int(round(self.value(p)))))
        if out:
            self.version += 1
            self.heat = [0.0, 0.0]
        return out

    def service(self, dt: float, spares: dict, rate: float = 2.0) -> list:
        """One step of a lander repair. Field faults are cleared at once; parts below 30% are replaced with a
        spare when the fleet has one; the rest is serviced at `rate` points per second. Returns events
        [("spare", part, kind)] / [("missing", part, kind)] for the caller to log."""
        ev = []
        changed = False
        for p in PARTS:
            if self.soft[p] > 0.0:
                self.soft[p] = 0.0
                self.soft_kind[p] = None
                changed = True
            if self.perm[p] <= 0.0:
                continue
            kind = SPARE_OF.get(p)
            if kind is not None and self.value(p) < SPARE_NEED_BELOW:
                if spares.get(kind, 0) > 0:
                    spares[kind] -= 1
                    self.perm[p] = 40.0              # the new part goes in, then it is fitted and tested
                    ev.append(("spare", p, kind))
                    changed = True
                else:
                    ev.append(("missing", p, kind))
                continue
            self.perm[p] = max(0.0, self.perm[p] - rate * dt)
            changed = True
        if changed:
            self.version += 1
            self.heat = [0.0, 0.0]
        return ev

    def strip(self, part: str) -> float:
        """Salvage: the part is taken out of this rover. Returns the condition it was in."""
        v = self.value(part)
        self.perm[part] = 100.0
        self.soft[part] = 0.0
        self.version += 1
        return v

    def install(self, part: str, condition: float) -> None:
        """A salvaged part goes in (field swap)."""
        self.perm[part] = max(0.0, 100.0 - condition)
        self.soft[part] = 0.0
        self.soft_kind[part] = None
        self.version += 1

    def set_parts(self, vals: dict, kinds: dict | None = None) -> None:
        """Tests and demos: set parts directly (permanent unless a field kind is given)."""
        for p, v in vals.items():
            kind = (kinds or {}).get(p, "broken")
            loss = max(0.0, min(100.0, 100.0 - float(v)))
            if kind in FIELD_KINDS:
                self.perm[p], self.soft[p], self.soft_kind[p] = 0.0, loss, kind
            else:
                self.perm[p], self.soft[p], self.soft_kind[p] = loss, 0.0, None
        self.version += 1

    # ------------------------------------------------------------ physics bookkeeping
    def on_physics(self, t: float, shock_g: float, touch_n: float, contact, ext) -> list:
        """Once per decision cycle: the cycle's peak shock (g), peak chassis skin force (N), the strongest contact
        on the rover during it ({"kind", "local", "force", "wheel"}) and an external push ({"cause", "dir"})."""
        sev = SHOCK_K * max(0.0, shock_g - SHOCK_G0)
        if touch_n > TOUCH_N0:
            sev += TOUCH_K * math.log2(touch_n / TOUCH_N0)
        if sev < 2.0:
            return []
        if t - self._hit_t < HIT_WINDOW_S:
            add = sev - self._hit_peak            # the same hit, still ringing: only a stronger knock counts
            if add < 5.0:
                return []
            self._hit_peak = sev
            sev = add
        else:
            self._hit_t, self._hit_peak = t, sev
        cause = "shock"
        parts = None
        if ext is not None:
            cause = ext["cause"]
            parts = self._parts_from_dir(ext.get("dir"))
        elif contact is not None:
            cause = {"boulder": "boulder", "spire": "spire", "rock": "rock", "terrain": "terrain",
                     "rover": "rover"}.get(contact["kind"], "shock")
            parts = self._parts_from_contact(contact)
        if parts is None:
            parts = [("drive_l", 0.5), ("drive_r", 0.5), ("lidar", 0.3)]
        return self.hit(t, parts, min(sev, 100.0), cause, shock_g=shock_g)

    @staticmethod
    def _parts_from_contact(c: dict) -> list:
        if c.get("wheel") is not None:
            side = "drive_l" if c["wheel"] in (0, 1) else "drive_r"
            return [(side, 1.0)]
        x, y, z = c["local"]
        if z > 0.06:                               # on top: the mast (front) or the deck and antenna (rear)
            if x > 0.0:
                return [("lidar", 1.0), ("camera", 0.7)]
            return [("power", 1.0), ("comms", 0.6)]
        if abs(y) > 0.15:                          # a side: that side's wheels and motors
            side = "drive_l" if y > 0 else "drive_r"
            return [(side, 1.0), ("camera" if x > 0 else "power", 0.3)]
        if x > 0.25:
            return [("camera", 1.0), ("lidar", 0.5), ("drive_l", 0.2), ("drive_r", 0.2)]
        if x < -0.25:
            return [("power", 1.0), ("comms", 0.5)]
        return [("drive_l", 0.6), ("drive_r", 0.6), ("power", 0.3)]     # from below: the belly

    def _parts_from_dir(self, d) -> list:
        """A push from direction d (unit vector in the rover's frame pointing where the blow came from)."""
        if d is None:
            return [("drive_l", 0.6), ("drive_r", 0.6)]
        x, y = d
        if abs(y) >= abs(x):
            side = "drive_l" if y > 0 else "drive_r"
            return [(side, 1.0), ("power", 0.25)]
        if x > 0:
            return [("camera", 1.0), ("lidar", 0.6), ("drive_l", 0.3), ("drive_r", 0.3)]
        return [("power", 1.0), ("comms", 0.6)]

    def strain(self, t: float, dt: float, loads, driving: bool) -> list:
        """Motor heating per side from its torque load (0..1 of the nominal cap). A long saturated push trips
        that side's motor controller (field-fixable) and wears it a little."""
        out = []
        for k, side in enumerate(("drive_l", "drive_r")):
            if self.soft_kind[side] == "trip":
                continue
            ld = float(loads[k])
            if driving and ld >= STRAIN_LOAD:
                self.heat[k] += dt
            elif ld < 0.8:
                self.heat[k] = max(0.0, self.heat[k] - STRAIN_COOL * dt)
            if self.heat[k] >= STRAIN_TRIP_S:
                self.heat[k] = 0.0
                self.apply(t, side, 5.0, "strain", "broken", log=False)
                d = self.apply(t, side, 100.0, "strain", "trip",
                               also=f"Torque at its limit for {STRAIN_TRIP_S:.0f} s: the motor controller tripped.")
                if d is not None:
                    out.append(d)
        return out

    def dust(self, t: float, dt: float, storm: bool) -> None:
        """A dust storm coats the camera window and the solar deck (field-fixable dirt). One event per storm."""
        if not storm:
            self.dust_this_storm = 0.0
            self.dust_logged = False
            return
        cam = DUST_CAMERA * dt
        before = int(self.value("camera"))
        self.soft["camera"] = min(100.0 - self.perm["camera"], self.soft["camera"] + cam)
        if self.soft_kind["camera"] is None:
            self.soft_kind["camera"] = "dirt"
        self.dust_this_storm += cam
        if int(self.value("camera")) != before:
            self.version += 1                     # (only on a whole point: effects are cached per version)
        if not self.dust_logged and self.dust_this_storm >= 6.0:
            self.dust_logged = True
            v = self.value("camera")
            dmg = {"rover": self.rover.id, "part": "camera", "before": int(round(v + self.dust_this_storm)),
                   "after": int(round(v)), "cause": "dust", "shock_g": None}
            self.log.append(dict(dmg, t=round(float(t), 1), kind="dirt"))
            self._log_damage(t, dmg, "dirt", "Dust on the camera window.")

    # ------------------------------------------------------------ words
    def _part_phrase(self, p: str, v: float, e: dict) -> str:
        kind = self.soft_kind[p] if self.soft[p] > 0.5 else None
        name = PART_NAME[p][:1].upper() + PART_NAME[p][1:]
        if p in ("drive_l", "drive_r"):
            other = "right" if p == "drive_l" else "left"
            mine = "left" if p == "drive_l" else "right"
            if kind == "trip":
                return f"{name} at {v:.0f}%: motor current but no wheel speed, the motor controller tripped"
            if kind == "jam" and v < MOBILE_DRIVE_MIN:
                return f"{name} at {v:.0f}%: the wheel is jammed (high current, no rotation)"
            if v < MOBILE_DRIVE_MIN:
                return f"{name} motor at {v:.0f}%: no torque at the wheels"
            fo = e["drive_r" if p == "drive_l" else "drive_l"]
            if fo - v / 100.0 > 0.1:
                return (f"{name} motor at {v:.0f}%: pulls {mine}, top speed "
                        f"{self.rover.cfg.v_max * e['v_cap']:.1f} m/s")
            return f"{name} motor at {v:.0f}%" + (f" (the {other} side too)" if abs(fo - v / 100.0) <= 0.1 else "")
        if p == "lidar":
            rng = self.rover.cfg.lidar_range * e["lidar_range"]
            return f"LiDAR at {v:.0f}%: range {rng:.1f} m, {e['lidar_drop'] * 100:.0f}% of rays lost"
        if p == "camera":
            if kind == "dirt":
                return f"Camera at {v:.0f}%: dust on the window"
            return f"Camera at {v:.0f}%: noisy image, the detector is less sure"
        if p == "power":
            if v < MOBILE_POWER_MIN:
                return f"Power system at {v:.0f}%: not enough current to drive"
            return f"Power system at {v:.0f}%: battery drains {e['drain']:.1f} times faster"
        return f"Radio at {v:.0f}%: range about {e['comms_m']:.0f} m"

    def diagnosis(self) -> str:
        """The rover's own reading of its state, from its built-in self tests."""
        v = self.values()
        e = self.effects()
        bad = sorted((p for p in PARTS if v[p] < 95.0), key=lambda p: v[p])
        if not bad:
            return ("All systems normal: drive current matches wheel speed on both sides, LiDAR returns and camera "
                    "clean, battery drain normal, radio link strong.")
        parts = [self._part_phrase(p, v[p], e) for p in bad[:3]]
        if len(bad) > 3:
            parts.append(f"{len(bad) - 3} more part{'s' if len(bad) > 4 else ''} worn")
        tail = ""
        if not self.mobile():
            tail = " Can't drive."
            if self.beyond_field_repair():
                tail = " Can't drive, beyond field repair."
            elif self.mobile(field_fixed=True):
                tail = " Can't drive until that is fixed (fixable in the field)."
        sl = e.get("slope_cap")
        if self.mobile() and sl is not None:
            tail += f" Own slope limit {sl:.0f} deg."
        return "; ".join(parts) + "." + tail

    def short(self) -> str:
        """'right drive 0%, left drive 20%' (the worst parts)."""
        v = self.values()
        bad = sorted((p for p in PARTS if v[p] < 95.0), key=lambda p: v[p])[:3]
        return ", ".join(f"{PART_NAME[p]} {v[p]:.0f}%" for p in bad) or "all parts normal"

    def msg(self) -> dict:
        out = {"hp": self.hp(), "status": self.status(), "parts": self.parts_msg(), "diagnosis": self.diagnosis(),
               "last_hit": dict(self.last_hit) if self.last_hit is not None else None}
        return out


# ====================================================================== physics monitor (sim level)
class DamageMonitor:
    """Watches every physics substep for shocks and hard contacts on the rovers (cheap: a few vector reads per
    substep; the contact list is only walked when something strong happened), and applies creature pushes."""

    def __init__(self, sim):
        self.sim = sim
        m = sim.model
        rv = sim.rovers
        n = len(rv)
        self.n = n
        self.lin = np.array([[r.vadr, r.vadr + 1, r.vadr + 2] for r in rv], dtype=np.int64)
        self.touch = np.array([int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"{r.id}_touch")])
                               for r in rv], dtype=np.int64)
        self.acc_sum = np.zeros((n, 3))
        self.k = 0
        self.peak_acc = np.zeros(n)
        self.peak_touch = np.zeros(n)
        self.contact = [None] * n
        self.ext = [None] * n
        self.pushes: list = []                     # [rover index, force (6,), steps left]
        self._f6 = np.zeros(6)
        # what each geom is, for the cause of a hit
        kind = np.zeros(m.ngeom, dtype=np.int64)
        owner = np.full(m.ngeom, -1, dtype=np.int64)
        wheel = np.full(m.ngeom, -1, dtype=np.int64)
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("g_tile_") or name.startswith("wall_"):
                kind[g] = KIND_CODES["terrain"]
            elif name.startswith("rock_"):
                kind[g] = KIND_CODES["rock"]
            elif name.startswith("boulder_"):
                kind[g] = KIND_CODES["boulder"]
            elif name.startswith("spire_"):
                kind[g] = KIND_CODES["spire"]
        for k, r in enumerate(rv):
            gc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{r.id}_chassis")
            kind[gc] = KIND_CODES["chassis"]
            owner[gc] = k
            for w in range(4):
                gw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{r.id}_wheel{w}")
                kind[gw] = KIND_CODES["wheel"]
                owner[gw] = k
                wheel[gw] = w
        self.kind, self.owner, self.wheel = kind, owner, wheel
        self.names = {v: k for k, v in KIND_CODES.items()}
        self.peaks_log = None                      # optional list for measurements (scripts)

    def push(self, idx: int, force, steps: int, cause: str, local_dir) -> None:
        """A creature runs into rover idx: a horizontal force (world frame, N) for `steps` substeps."""
        f = np.zeros(6)
        f[:3] = force
        self.pushes.append([idx, f, int(steps)])
        self.ext[idx] = {"cause": cause, "dir": local_dir}

    def step(self, d) -> None:
        """Every physics substep (sim.step)."""
        self.acc_sum += d.qacc[self.lin]
        self.k += 1
        if self.k >= AVG_STEPS:
            a = np.sqrt((self.acc_sum * self.acc_sum).sum(axis=1)) / self.k
            self.acc_sum[:] = 0.0
            self.k = 0
            hot = a > self.peak_acc
            if hot.any():
                np.maximum(self.peak_acc, a, out=self.peak_acc)
                if (a[hot] > CAPTURE_G * 9.81).any():
                    for i in np.flatnonzero(hot & (a > CAPTURE_G * 9.81)):
                        self._capture(d, int(i))
        tch = d.sensordata[self.touch]
        hot = tch > self.peak_touch
        if hot.any():
            np.maximum(self.peak_touch, tch, out=self.peak_touch)
            if (tch[hot] > CAPTURE_N).any():
                for i in np.flatnonzero(hot & (tch > CAPTURE_N)):
                    self._capture(d, int(i))
        if self.pushes:
            keep = []
            for rec in self.pushes:
                idx, f, left = rec
                b = self.sim.rovers[idx].body
                if left > 0:
                    d.xfrc_applied[b] = f
                    rec[2] = left - 1
                    keep.append(rec)
                else:
                    d.xfrc_applied[b] = 0.0
            self.pushes = keep

    def _capture(self, d, idx: int) -> None:
        """The strongest contact on rover idx right now (other geom kind, point in the rover's frame)."""
        m = self.sim.model
        nc = d.ncon
        if nc == 0:
            return
        g = d.contact.geom[:nc]
        own = self.owner[g]
        rows = np.flatnonzero((own[:, 0] == idx) | (own[:, 1] == idx))
        best = None
        for row in rows:
            g1, g2 = int(g[row, 0]), int(g[row, 1])
            mine, other = (g1, g2) if self.owner[g1] == idx else (g2, g1)
            ok = self.kind[other]
            if self.kind[mine] == KIND_CODES["wheel"]:
                if ok not in (KIND_CODES["boulder"], KIND_CODES["spire"], KIND_CODES["chassis"]):
                    continue                       # wheels on the ground or on terrain rocks: normal driving
            mujoco.mj_contactForce(m, d, int(row), self._f6)
            fn = abs(float(self._f6[0]))
            if best is None or fn > best[0]:
                best = (fn, int(row), mine, other)
        if best is None:
            return
        fn, row, mine, other = best
        cur = self.contact[idx]
        if cur is not None and cur["force"] >= fn:
            return
        r = self.sim.rovers[idx]
        p = np.array(d.contact.pos[row])
        R = d.xmat[r.body].reshape(3, 3)
        local = R.T @ (p - d.xpos[r.body])
        ok = self.kind[other]
        kname = self.names.get(int(ok), "other")
        if kname in ("chassis", "wheel"):
            kname = "rover"
        self.contact[idx] = {"kind": kname, "local": (float(local[0]), float(local[1]), float(local[2])),
                             "force": fn, "wheel": int(self.wheel[mine]) if self.wheel[mine] >= 0 else None,
                             "at": (float(p[0]), float(p[1]))}

    def after_physics(self, t: float) -> None:
        """Once per decision cycle (sim thread): hand each rover its cycle's peaks, then reset."""
        g = 9.81
        for k, r in enumerate(self.sim.rovers):
            sg = float(self.peak_acc[k]) / g
            tn = float(self.peak_touch[k])
            if self.peaks_log is not None:
                self.peaks_log.append((t, r.id, sg, tn, r.state, None if self.contact[k] is None else
                                       self.contact[k]["kind"]))
            if r.health is not None and r.state not in ("towed",):
                r.health.on_physics(t, sg, tn, self.contact[k], self.ext[k])
            elif r.health is not None and self.ext[k] is not None:
                r.health.on_physics(t, sg, tn, None, self.ext[k])
        self.peak_acc[:] = 0.0
        self.peak_touch[:] = 0.0
        self.contact = [None] * self.n
        self.ext = [None] * self.n
