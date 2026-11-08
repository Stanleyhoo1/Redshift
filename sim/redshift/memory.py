"""What each rover remembers about its own mission, so it can answer the operator's questions
(PROTOCOL.md version 5) and report `stats` in every tick.

Everything here is cheap bookkeeping, updated incrementally where things happen (no per cycle
scans of the map or the event log):

  mapping      cells this rover mapped first, binned by compass sector and distance band around the
               lander (one np.bincount over the few hundred new cells of a sweep)
  driving      distance driven (the rover's odometer)
  science      targets it found (spectrometer), samples it took (with the on-board analysis), tubes
               it delivered to the lander cache
  hazards      what it met (new rocks, a fallen spire, soft ground, dust, water, a motor fault, a rock
               under the bumper, a slope it could not climb, tipping over) and how it handled it
  lifeforms    things it saw moving, and the herds it observed (individuals, how long, speed)
  events       the last few notable log lines about this rover (fed by Fleet.log)
  goal         where it is going and why (rules, Gemma's pick or the operator)

All lists are bounded. The fact sheets and the rule-based answers built from this live in
answers.py.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

SECTORS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")
BANDS = ((0.0, 15.0, "near"), (15.0, 35.0, "middle"), (35.0, math.inf, "far"))   # metres from the lander
MAX_TARGETS = 24
MAX_SAMPLES = 24
MAX_HAZARDS = 12
MAX_LIFE = 6
MAX_EVENTS = 8
MAX_WATER = 6

# hazard kinds: how a rover describes meeting one and what it did about it (first person)
HAZARD_TEXT = {
    "rocks": ("new rocks across my path", "slowed down and replanned around them"),
    "spire": ("a fallen spire across my path", "slowed down and replanned around it"),
    "sand": ("soft ground (wheels spinning)", "stopped and marked it for the fleet"),
    "dust": ("a dust storm (LiDAR range down)", "slowed down and kept to mapped ground"),
    "water": ("liquid where my LiDAR got no returns", "marked it as water for the fleet and routed around it"),
    "dropoff": ("a drop-off where my LiDAR got no returns", "slowed down and kept to mapped ground"),
    "wet": ("liquid under my chassis (water sensor wet)", "stopped and backed out the way I came"),
    "fault": ("a motor controller fault", "locked the brakes and handed my work to the fleet"),
    "rock": ("a rock under my bumper", "marked it as an obstacle and backed off"),
    "slope": ("a slope too steep to climb", "backed off and replanned"),
    "tipped": ("tipping over", "self-righted and replanned"),
}


def compass(dx: float, dy: float) -> str:
    a = math.degrees(math.atan2(dy, dx)) % 360.0
    return SECTORS[int((a + 22.5) // 45) % 8]


def _bins(fmap) -> np.ndarray:
    """Per map sample: sector * 3 + band around the lander (0, 0). Computed once per map."""
    b = getattr(fmap, "_mem_bins", None)
    if b is None:
        xs = fmap.ox + np.arange(fmap.nx) * fmap.res
        ys = fmap.oy + np.arange(fmap.ny) * fmap.res
        X, Y = np.meshgrid(xs, ys)
        ang = np.degrees(np.arctan2(Y, X)) % 360.0
        sec = ((ang + 22.5) // 45).astype(np.int64) % 8
        d = np.hypot(X, Y)
        band = np.where(d < BANDS[0][1], 0, np.where(d < BANDS[1][1], 1, 2))
        b = (sec * 3 + band).astype(np.int8).ravel()
        fmap._mem_bins = b
    return b


class RoverMemory:
    """One rover's mission memory. See the module docstring."""

    def __init__(self, rover):
        self.rid = rover.id
        self.res2 = float(rover.cfg.res) ** 2
        self.mapped_cells = 0
        self.sector_cells = np.zeros(24, np.int64)          # sector * 3 + band
        self.targets: list = []                             # [{"id", "kind", "t", "est"}]
        self.n_targets = 0
        self.samples: list = []                             # [{"id", "kind", "instrument", "value", "analysis", "t"}]
        self.n_samples = 0
        self.cached = 0                                     # samples this rover delivered to the lander cache
        self.cache_trips = 0
        self.hazards: deque = deque(maxlen=MAX_HAZARDS)     # [{"t", "kind", "what", "handled", "at", "outcome"}]
        self.n_hazards = 0
        self.hazard_kinds: dict = {}                        # kind -> count
        self.sightings = 0                                  # things it saw moving (LiDAR tracks it stopped for)
        self.life: deque = deque(maxlen=MAX_LIFE)           # [{"t", "n", "dur", "speed", "label", "at", "life"}]
        self.n_herds = 0                                    # observations that confirmed a lifeform
        self.water: list = []                               # [{"t", "x", "y", "r", "how"}] water it identified
        self.events: deque = deque(maxlen=MAX_EVENTS)       # [(t, level, msg)]
        self.replans = 0
        self.charges = 0
        self.goal_note = None                               # (t, source, text): why the current goal

    # ------------------------------------------------------------------ updates (cheap)
    def on_mapped(self, fmap, new_flat, n_new: int) -> None:
        """A sweep made n_new samples known for the first time; new_flat: flat indices of the samples
        hit directly (hole filling adds a few more, spread in the same proportions)."""
        if n_new <= 0:
            return
        self.mapped_cells += int(n_new)
        if new_flat is not None and len(new_flat):
            c = np.bincount(_bins(fmap)[new_flat], minlength=24)
            k = n_new / float(len(new_flat))
            self.sector_cells += np.rint(c * k).astype(np.int64) if k != 1.0 else c

    def on_target(self, ts, t: float) -> None:
        self.n_targets += 1
        if len(self.targets) < MAX_TARGETS:
            self.targets.append({"id": ts.spec.id, "kind": ts.spec.kind, "t": round(float(t), 1),
                                 "est": int(ts.spec.est_value)})

    def on_sample(self, ts, t: float) -> None:
        self.n_samples += 1
        if len(self.samples) >= MAX_SAMPLES:
            self.samples.pop(0)
        self.samples.append({"id": ts.spec.id, "kind": ts.spec.kind, "instrument": ts.spec.instrument,
                             "value": int(ts.value or 0), "analysis": str(ts.analysis or ""), "t": round(float(t), 1)})

    def on_cache(self, n: int, t: float) -> None:
        if n > 0:
            self.cached += int(n)
            self.cache_trips += 1

    def on_hazard(self, kind: str, t: float, at=None, what: str | None = None, handled: str | None = None) -> None:
        base_what, base_handled = HAZARD_TEXT.get(kind, (kind.replace("_", " "), "handled it"))
        self.n_hazards += 1
        self.hazard_kinds[kind] = self.hazard_kinds.get(kind, 0) + 1
        self.hazards.append({"t": round(float(t), 1), "kind": kind, "what": what or base_what,
                             "handled": handled or base_handled,
                             "at": None if at is None else (round(float(at[0]), 1), round(float(at[1]), 1)),
                             "outcome": None})

    def note_outcome(self, kinds, text: str, t: float, within: float = 180.0) -> None:
        """How the latest hazard of one of these kinds ended ("+6 m detour", "rebooted after 90 s")."""
        kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
        for h in reversed(self.hazards):
            if h["kind"] in kinds and t - h["t"] <= within:
                if h["outcome"] is None:
                    h["outcome"] = text
                return

    def on_sighting(self, t: float) -> None:
        self.sightings += 1

    def on_observed(self, t: float, n: int, dur: float, speed: float, life: bool, label: str | None, at) -> None:
        if life:
            self.n_herds += 1
        self.life.append({"t": round(float(t), 1), "n": int(n), "dur": round(float(dur)), "speed": round(float(speed), 1),
                          "label": label, "life": bool(life),
                          "at": (round(float(at[0]), 1), round(float(at[1]), 1))})

    def on_water(self, t: float, x: float, y: float, r: float, how: str) -> None:
        if len(self.water) < MAX_WATER:
            self.water.append({"t": round(float(t), 1), "x": round(float(x), 1), "y": round(float(y), 1),
                               "r": round(float(r), 1), "how": how})

    def on_event(self, e: dict) -> None:
        """Fleet.log calls this for every event about this rover."""
        if e.get("level") in ("warn", "discovery") or e.get("source"):
            self.events.append((e["t"], e["level"], e["msg"]))

    def on_goal(self, t: float, source: str, text: str) -> None:
        self.goal_note = (round(float(t), 1), source, text)

    # ------------------------------------------------------------------ reading
    def mapped_m2(self) -> float:
        return self.mapped_cells * self.res2

    def sectors(self) -> list:
        """[(sector name, share of this rover's mapped cells)] by sector, biggest first (no zeros)."""
        by_sec = self.sector_cells.reshape(8, 3).sum(axis=1)
        tot = float(by_sec.sum())
        if tot <= 0:
            return []
        order = np.argsort(-by_sec, kind="stable")
        return [(SECTORS[k], float(by_sec[k]) / tot) for k in order if by_sec[k] > 0]

    def bands(self) -> list:
        """[(band name, share)] near / middle / far from the lander (no zeros)."""
        by_band = self.sector_cells.reshape(8, 3).sum(axis=0)
        tot = float(by_band.sum())
        if tot <= 0:
            return []
        return [(BANDS[k][2], float(by_band[k]) / tot) for k in range(3) if by_band[k] > 0]

    def stats(self, rover) -> dict:
        """PROTOCOL.md v5 rover `stats`: running totals since the sim started."""
        return {"distance_m": round(float(rover.odometer), 1), "mapped_m2": round(self.mapped_m2(), 1),
                "targets_found": int(self.n_targets), "samples_taken": int(self.n_samples),
                "hazards_met": int(self.n_hazards), "lifeforms_seen": int(self.n_herds), "replans": int(self.replans)}
