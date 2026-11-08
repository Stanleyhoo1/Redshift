"""Short conversation memory for the operator chat (PROTOCOL.md version 7).

The advisor keeps the last few chat messages and what they talked about: the places, targets and candidate
lists the operator or the rovers mentioned, the rover the operator talked to last, the last command, and the
reports a survey or an area focus wrote. The operator's next message can refer back to them:

    "go back there", "explore there", "investigate it"     the place talked about last
    "the second one", "the last one"                        an entry of the last list (choose_place options,
                                                            or the targets a reply listed)
    "that lake", "the vent you mentioned"                   the newest mention of that kind
    "do that again", "same for rover 3"                     the last command (for another rover)
    "what did you find there?"                              the last place, and what was reported about it

Everything is small and bounded, and only touched when a chat message is posted or a command is resolved
(never in the 200 ms control loop). context() gives Gemma a few compact lines of it.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

MAX_MSGS = 8           # chat messages kept
MAX_PLACES = 10        # places / targets mentioned, newest last
MAX_REPORTS = 6        # survey and focus reports
SAME_M = 3.0           # two mentions this close are the same place
TARGET_KINDS = ("mineral", "ice", "vent", "trace")


@dataclass
class Ref:
    """One thing the chat talked about."""
    kind: str                  # water, spire, mineral, ice, vent, trace, lifeform, hazard, lander, point, area
    x: float
    y: float
    label: str                 # "the pond's shore", "t4 (ice deposit)", "(10, -5)"
    id: str | None = None      # w1, t4, k2, s3 ... when it has one
    rover: str | None = None   # the rover that went there, or talked about it
    t: float = 0.0
    place: object = None       # the places.Place behind it (None for a point or an area)
    r: float | None = None     # an area's radius

    def where(self) -> str:
        return f"{self.label} at ({self.x:.1f}, {self.y:.1f})" if self.kind not in ("point",) else self.label

    def matches(self, kind: str) -> bool:
        """Does a place word of this kind ("water", "vent", "target", ...) name this mention?"""
        if kind == "target":
            return self.kind in TARGET_KINDS
        return self.kind == kind


class ChatMemory:
    def __init__(self):
        self.msgs: deque = deque(maxlen=MAX_MSGS)      # (from, text)
        self.places: deque = deque(maxlen=MAX_PLACES)  # Ref, newest last (single mentions and list entries)
        self.there: Ref | None = None                  # the place talked about last (a single mention)
        self.cands: list = []                          # [Ref] the last list of candidates
        self.rover: str | None = None                  # the rover the operator talked with last
        self.command: dict | None = None               # {"cmd", "rid", "ref", "text"} the last operator command
        self.reports: deque = deque(maxlen=MAX_REPORTS)  # {"ref", "text", "rover", "t", "summary"}

    # ------------------------------------------------------------------ recording
    def add_msg(self, frm: str, text: str) -> None:
        self.msgs.append((str(frm), " ".join(str(text).split())))

    def mention(self, ref: Ref) -> None:
        """A single place the chat talked about: it becomes "there"."""
        self._push(ref)
        self.there = ref

    def mention_list(self, refs: list) -> None:
        """A list of candidates ("I know 3 vents", "I found t4, t7 and t9"): "the second one" picks from it."""
        refs = [r for r in refs if r is not None]
        if not refs:
            return
        self.cands = list(refs)
        for r in refs:
            self._push(r)
        if len(refs) == 1:
            self.there = refs[0]

    def _push(self, ref: Ref) -> None:
        for old in list(self.places):
            if old.kind == ref.kind and (old.id is not None and old.id == ref.id or
                                         math.hypot(old.x - ref.x, old.y - ref.y) < SAME_M):
                self.places.remove(old)
        self.places.append(ref)

    def set_command(self, cmd: dict, rid, ref: Ref | None, text: str) -> None:
        self.command = {"cmd": dict(cmd), "rid": rid, "ref": ref, "text": text}

    def add_report(self, ref: Ref, text: str, rover: str, t: float, summary: dict | None = None) -> None:
        self.reports.append({"ref": ref, "text": text, "rover": rover, "t": t, "summary": summary})
        self.mention(ref)

    # ------------------------------------------------------------------ reading
    def pick(self, ordinal: int) -> Ref | None:
        """Entry `ordinal` (1 = first, -1 = last) of the last list, or None."""
        if not self.cands:
            return None
        if ordinal == -1:
            return self.cands[-1]
        return self.cands[ordinal - 1] if 1 <= ordinal <= len(self.cands) else None

    def last_of(self, kind: str) -> Ref | None:
        """The newest mention of this kind of place ("that lake": water)."""
        for ref in reversed(self.places):
            if ref.matches(kind):
                return ref
        return None

    def report_near(self, x: float, y: float, radius: float = 12.0) -> dict | None:
        """The newest survey or focus report about a place near (x, y)."""
        for rep in reversed(self.reports):
            if math.hypot(rep["ref"].x - x, rep["ref"].y - y) <= radius:
                return rep
        return None

    def context(self) -> dict:
        """A few compact lines for Gemma: the last messages and what "there", "the second one" and "do that
        again" would mean now. Empty when nothing was said yet."""
        lines = [f"{frm}: {text}" for frm, text in list(self.msgs)[-3:]]
        refs = []
        if self.there is not None:
            refs.append(f"there = {self.there.where()}")
        if len(self.cands) > 1:
            refs.append("last list: " + ", ".join(f"{k + 1} {c.label}" for k, c in enumerate(self.cands[:4])))
        if self.rover:
            refs.append(f"last rover {self.rover}")
        if self.command is not None:
            refs.append(f"last command: {self.command['text']!r}" + (f" ({self.command['rid']})"
                                                                     if self.command.get("rid") else ""))
        return {"lines": lines, "refs": "; ".join(refs)} if (lines or refs) else {}
