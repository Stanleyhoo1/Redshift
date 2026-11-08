"""Commanding by place (PROTOCOL.md version 6): "go to the water source", "survey the spires".

A place is resolved from what the FLEET knows, never from ground truth it has not sensed:

  water      water zones the fleet identified (fleet.water_zones), grouped into bodies (size, centre)
  spire      spots where the fleet's map shows something tall standing (map.tall_t), at a spire site
  mineral, ice, vent, trace, target    found targets (by kind, or one id)
  lifeform   live LiDAR tracks (grouped into herds) and the places herds were observed (fleet + rover memory)
  hazard     soft ground found by wheel slip, new rocks and fallen spires the rovers met
  lander     the lander

It also finds safe goals next to a place (for water a mapped, passable spot about 3 m from the edge on the
side facing the rover; for a creature the observation standoff; never in water, soft ground or a no-go
zone, preferring ground the fleet's reachability map connects to the lander), writes short human labels
("Lake about 14 m across, 22 m north-east of me") and reads which option an operator's chat text picks.

Only called when the operator sends a command or answers (never in the 200 ms control loop).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .fleet import compass
from .gemma import _qualifier, near_reference
from .terrain import KIND_LABEL

TARGET_KINDS = ("mineral", "ice", "vent", "trace")
SHORE_GAP = 3.0            # goal distance from the water's edge (m)
GAP_MIN, GAP_MAX = 2.0, 4.5
EDGE_M = 3.0               # no goals this close to the map edge
LIVE_S = 30.0              # a track seen this recently counts as a herd that is there now
HERD_M = 12.0              # live tracks this close together are one herd
SAME_PLACE_M = 15.0        # a remembered herd spot this close to another is the same herd
MAX_OPTIONS = 6            # options in a choose_place request (plus "Never mind")
DIRS = {"north": (0.0, 1.0), "south": (0.0, -1.0), "east": (1.0, 0.0), "west": (-1.0, 0.0),
        "north-east": (0.7071, 0.7071), "north-west": (-0.7071, 0.7071),
        "south-east": (0.7071, -0.7071), "south-west": (-0.7071, -0.7071)}
PLURAL = {"water": ("body of water", "bodies of water"), "spire": ("spire", "spires"),
          "mineral": ("mineral outcrop", "mineral outcrops"), "ice": ("ice deposit", "ice deposits"),
          "vent": ("vent", "vents"), "trace": ("organic trace", "organic traces"), "target": ("target", "targets"),
          "lifeform": ("herd", "herds"), "hazard": ("hazard site", "hazard sites"), "lander": ("lander", "landers")}
UNKNOWN = {"water": "any water", "spire": "any spires", "mineral": "any mineral outcrops", "ice": "any ice",
           "vent": "any vents", "trace": "any organic traces", "target": "any targets",
           "lifeform": "any herds or creatures", "hazard": "any hazard sites", "lander": "the lander"}
SUMMARY_KINDS = ("water", "vent", "ice", "mineral", "trace", "spire", "lifeform", "hazard")
_CANCEL_RE = re.compile(r"\b(?:never ?mind|cancel|none of (?:them|those|these)|neither|forget (?:it|about it)|"
                        r"no thanks|don'?t bother|skip it|nothing)\b")
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
        "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6}
_SUB_WORDS = {"lake": ("lake",), "pond": ("pond", "pool"), "stream": ("stream", "river", "creek"),
              "herd": ("herd",), "creature": ("creature", "animal"), "spire": ("spire", "crystal"),
              "vent": ("vent",), "ice": ("ice",), "mineral": ("mineral", "outcrop"), "trace": ("trace", "organic"),
              "Soft ground": ("soft", "sand"), "New rocks": ("rocks", "rockfall", "boulder"),
              "Fallen spire": ("fallen",), "Rock": ("rock",), "lander": ("lander",)}


@dataclass
class Place:
    """One thing the fleet knows that the operator can send a rover to."""
    id: str                     # "w1" (a body of water), "t4", "s3", "k2" (live herd: its newest track), ...
    kind: str                   # water, spire, mineral, ice, vent, trace, lifeform, hazard, lander
    sub: str                    # lake / pond / stream, the target kind, herd / creature, "Soft ground", ...
    x: float
    y: float
    size: float | None = None   # water: m across (stream: long); spire: m tall; herd: individuals
    t: float | None = None      # sim time the fleet learned of it (newest part)
    zones: list = field(default_factory=list)    # water: [(x, y, r)]
    standoff: float = 3.0       # how far from (x, y) the goal is
    track: str | None = None    # a live herd: the track to follow up
    live: bool = False

    def noun(self) -> str:
        """'Lake about 14 m across', 'Thermal vent t10', 'Herd of 3, moving now'."""
        k, s = self.kind, self.sub
        if k == "water":
            if s == "stream":
                return f"Stream about {self.size:.0f} m long"
            return f"{s.capitalize()} about {self.size:.0f} m across"
        if k == "spire":
            return f"Spire about {self.size:.0f} m tall" if self.size else "Spire"
        if k in TARGET_KINDS:
            return f"{KIND_LABEL[k].capitalize()} {self.id}"
        if k == "lifeform":
            n = int(self.size or 1)
            what = f"Herd of {n}" if n > 1 else "Creature"
            return what + (", moving now" if self.live else ", seen earlier")
        if k == "hazard":
            return s
        return "The lander"

    def dest(self) -> str:
        """Where the rover goes, for acks and reports: 'the lake's shore', 't4 (ice deposit)'."""
        k, s = self.kind, self.sub
        if k == "water":
            return "the stream's bank" if s == "stream" else f"the {s}'s shore"
        if k == "spire":
            return "the spire"
        if k in TARGET_KINDS:
            return f"{self.id} ({KIND_LABEL[k]})"
        if k == "lifeform":
            who = "herd" if int(self.size or 1) > 1 else "creature"
            return f"a viewpoint about {self.standoff:.0f} m from the {who}"
        if k == "hazard":
            return f"a safe spot by the {s.lower()}"
        return "the lander"


def count_noun(kind: str, n: int) -> str:
    one, many = PLURAL.get(kind, ("place", "places"))
    return f"{n} {one if n == 1 else many}"


# ====================================================================== what the fleet knows
def water_bodies(zones: list) -> list:
    """Water zones [(x, y, r, rover, t)] grouped into bodies (zones closer than 2 m edge to edge are one
    body): centre (area weighted), size across, lake / pond / stream (long and thin), newest zone time."""
    n = len(zones)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if math.hypot(zones[a][0] - zones[b][0], zones[a][1] - zones[b][1]) < zones[a][2] + zones[b][2] + 2.0:
                parent[find(a)] = find(b)
    groups: dict = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(zones[k])
    out = []
    for k, zs in enumerate(sorted(groups.values(), key=lambda g: min(z[4] for z in g))):
        xs = np.array([z[0] for z in zs])
        ys = np.array([z[1] for z in zs])
        w = np.array([z[2] ** 2 for z in zs])
        cx, cy = float((w * xs).sum() / w.sum()), float((w * ys).sum() / w.sum())
        across = max(2.0 * z[2] for z in zs)
        for a in zs:
            for b in zs:
                across = max(across, math.hypot(a[0] - b[0], a[1] - b[1]) + a[2] + b[2])
        sub = "lake"
        if len(zs) >= 3:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]), aweights=w))
            if math.sqrt(max(ev[-1], 0.0) / max(ev[0], 1e-6)) >= 3.0 and across >= 12.0:
                sub = "stream"
        if sub == "lake" and across < 8.0:
            sub = "pond"
        out.append(Place(f"w{k + 1}", "water", sub, cx, cy, size=round(across, 1), t=max(z[4] for z in zs),
                         zones=[(float(z[0]), float(z[1]), float(z[2])) for z in zs], standoff=SHORE_GAP))
    return out


def _spires(sim) -> list:
    """Standing spires the fleet's map shows: something tall in the footprint (LiDAR), position and height
    from the map itself (the centre of the tall samples, the spread of the hits there)."""
    hz = getattr(sim, "hazards", None)
    if hz is None:
        return []
    m = sim.fleet.map
    standing = getattr(hz, "spire_standing", {})
    out = []
    for sp in hz.all_spires():
        if not standing.get(sp.id, True):
            continue                       # a fallen spire is a hazard site
        i, j = m.ij(sp.x, sp.y)
        js, is_ = slice(max(j - 2, 0), j + 3), slice(max(i - 2, 0), i + 3)
        tall = np.isfinite(m.tall_t[js, is_])
        if not tall.any():
            continue
        jj, ii = np.nonzero(tall)
        x = float(np.mean([m.xy(int(a) + is_.start, int(b) + js.start)[0] for b, a in zip(jj, ii)]))
        y = float(np.mean([m.xy(int(a) + is_.start, int(b) + js.start)[1] for b, a in zip(jj, ii)]))
        hi, lo = m.fmax[js, is_], m.fmin[js, is_]
        hi, lo = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
        height = float(hi.max() - lo.min()) if hi.size and lo.size else None
        out.append(Place(sp.id, "spire", "spire", x, y, size=round(height, 1) if height else None,
                         t=float(m.tall_t[js, is_][tall].min()), standoff=3.0))
    return out


def _herds(sim, t: float) -> list:
    f = sim.fleet
    clusters: list = []
    for tr in f.tracker.confirmed():
        if t - tr.last_seen > LIVE_S or tr.ignored:
            continue
        px, py = tr.predict(t)
        for c in clusters:
            if math.hypot(px - c["x"], py - c["y"]) < HERD_M:
                n = len(c["trs"])
                c["x"], c["y"] = (c["x"] * n + px) / (n + 1), (c["y"] * n + py) / (n + 1)
                c["trs"].append(tr)
                break
        else:
            clusters.append({"x": px, "y": py, "trs": [tr]})
    sd = float(sim.cfg.observe_standoff)
    out = []
    for c in clusters:
        n = max(1, f.tracker.distinct(c["trs"], t))
        tr0 = max(c["trs"], key=lambda tr: tr.last_seen)
        out.append(Place(tr0.id, "lifeform", "herd" if n > 1 else "creature", c["x"], c["y"], size=n,
                         t=tr0.last_seen, standoff=sd, track=tr0.id, live=True))
    spots = [(hx, hy, cnt, None) for hx, hy, cnt in f.lifeform_herds]
    for r in sim.rovers:
        spots += [(e["at"][0], e["at"][1], e["n"], e["t"]) for e in r.mem.life if e.get("at")]
    for hx, hy, n, tt in spots:
        same = next((p for p in out if math.hypot(hx - p.x, hy - p.y) < SAME_PLACE_M), None)
        if same is not None:
            if not same.live:
                same.size = max(same.size or 1, n)
                if tt is not None:
                    same.t = tt if same.t is None else max(same.t, tt)
            continue
        out.append(Place(f"h{len(out) + 1}", "lifeform", "herd" if n > 1 else "creature", float(hx), float(hy),
                         size=int(n), t=tt, standoff=sd))
    return out


def _hazards(sim) -> list:
    items = [(z[0], z[1], z[2], "Soft ground", z[4]) for z in sim.fleet.soft_zones]
    for r in sim.rovers:
        for h in r.mem.hazards:
            if h["kind"] in ("rocks", "spire", "rock") and h.get("at"):
                sub = {"rocks": "New rocks", "spire": "Fallen spire", "rock": "Rock"}[h["kind"]]
                items.append((h["at"][0], h["at"][1], 1.0, sub, h["t"]))
    out = []
    for x, y, rad, sub, tt in items:
        if any(math.hypot(x - p.x, y - p.y) < 5.0 for p in out):
            continue
        out.append(Place(f"z{len(out) + 1}", "hazard", sub, float(x), float(y), t=tt, standoff=float(rad) + 3.0))
    return out


def candidates(sim, place: str, target_id: str | None = None, t: float | None = None) -> list:
    """Every known place of this kind (fleet knowledge only)."""
    f = sim.fleet
    t = sim.t if t is None else t
    if place == "water":
        return water_bodies(f.water_zones)
    if place == "spire":
        return _spires(sim)
    if place in TARGET_KINDS or place == "target" or target_id:
        out = []
        for ts in f.targets:
            if ts.status is None:
                continue
            if target_id is not None and ts.spec.id != target_id:
                continue
            if target_id is None and place != "target" and ts.spec.kind != place:
                continue
            out.append(Place(ts.spec.id, ts.spec.kind, ts.spec.kind, float(ts.spec.x), float(ts.spec.y),
                             t=ts.found_t, standoff=2.5))
        return out
    if place == "lifeform":
        return _herds(sim, t)
    if place == "hazard":
        return _hazards(sim)
    if place == "lander":
        return [Place("lander", "lander", "lander", 0.0, 0.0, standoff=float(sim.cfg.lander_radius) + 2.5)]
    return []


def refresh(sim, p: Place, t: float) -> Place:
    """A live herd moves: its current position from the tracker (the place is otherwise static)."""
    if p.track:
        tr = sim.fleet.tracker.get(p.track)
        if tr is not None and t - tr.last_seen < LIVE_S:
            p.x, p.y = tr.predict(t)
    return p


def known_summary(sim, t: float | None = None) -> str:
    bits = []
    for kind in SUMMARY_KINDS:
        n = len(candidates(sim, kind, None, t))
        if n:
            bits.append(count_noun(kind, n))
    return ", ".join(bits) if bits else "nothing yet besides the lander"


def nothing_known(sim, place: str, target_id: str | None, t: float, fleet: bool) -> str:
    who = "We don't" if fleet else "I don't"
    head = f"{who} know target {target_id}." if target_id else f"{who} know {UNKNOWN.get(place, 'that place')} yet."
    return f"{head} Known: {known_summary(sim, t)}."


# ====================================================================== choosing
def apply_qualifier(cands: list, qualifier: str | None, ordinal: int | None = None, ref=(0.0, 0.0),
                    near_xy=None) -> tuple:
    """(candidates left, decisive). nearest / farthest / ordinal are measured from ref (the rover that goes,
    or the lander); a compass direction picks the one furthest that way; biggest / smallest / newest need
    the sizes or times."""
    if not cands:
        return [], False
    if len(cands) == 1:
        return list(cands), True
    q = qualifier or "none"

    def dist(p):
        return math.hypot(p.x - ref[0], p.y - ref[1])

    if q == "nearest":
        return [min(cands, key=dist)], True
    if q == "farthest":
        return [max(cands, key=dist)], True
    if q in ("biggest", "smallest"):
        if all(p.size is not None for p in cands):
            return [(max if q == "biggest" else min)(cands, key=lambda p: p.size)], True
        return list(cands), False
    if q == "newest":
        if any(p.t is not None for p in cands):
            return [max(cands, key=lambda p: p.t if p.t is not None else -1e18)], True
        return list(cands), False
    if q in DIRS:
        ux, uy = DIRS[q]
        return [max(cands, key=lambda p: p.x * ux + p.y * uy)], True
    if q == "near_rover" and near_xy is not None:
        return [min(cands, key=lambda p: math.hypot(p.x - near_xy[0], p.y - near_xy[1]))], True
    if q == "ordinal" and ordinal:
        s = sorted(cands, key=dist)
        if 1 <= ordinal <= len(s):
            return [s[ordinal - 1]], True
        return s, False
    return list(cands), False


def _cell_ok(f, cost, x: float, y: float) -> bool:
    m = f.map
    half = (m.nx - 1) * m.res / 2.0
    if abs(x) > half - EDGE_M or abs(y) > half - EDGE_M:
        return False
    i, j = m.ij(x, y)
    if not m.known[j, i]:
        return False
    if not np.isfinite(cost[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2]).all():
        return False                       # passable, with passable neighbours (slope, rocks, soft, no-go, water)
    if f.water_zone_at(x, y, margin=1.5) is not None or f.soft_zone_at(x, y, margin=1.0) is not None:
        return False
    return True


def approach_goals(sim, p: Place, frm: tuple, t: float, k: int = 4) -> list:
    """Up to k safe goals next to the place, best first, at least 2 m apart: mapped, passable ground (with
    passable neighbours), never in or near water or soft ground or a no-go zone; ground the fleet's
    reachability map connects to the lander first. Water: about 3 m from the edge, on the side facing the
    rover. Anything else: at the place's standoff, on the rover's side."""
    f = sim.fleet
    cost = f.cost(t)
    pts = []
    if p.kind == "water":
        for zx, zy, zr in p.zones:
            for a in range(0, 360, 20):
                ang = math.radians(a)
                x, y = zx + (zr + SHORE_GAP) * math.cos(ang), zy + (zr + SHORE_GAP) * math.sin(ang)
                gap = f.water_zone_gap(x, y)
                if not GAP_MIN <= gap <= GAP_MAX or not _cell_ok(f, cost, x, y):
                    continue
                score = math.hypot(x - frm[0], y - frm[1]) + 4.0 * abs(gap - SHORE_GAP)
                pts.append((score + (0.0 if f.in_reach(x, y) else 60.0), x, y))
    else:
        base = math.atan2(frm[1] - p.y, frm[0] - p.x)
        offs = (0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 150, -150, 180)
        for kr, rad in enumerate((p.standoff, p.standoff + 1.0, p.standoff + 2.0)):
            for ko, off in enumerate(offs):
                ang = base + math.radians(off)
                x, y = p.x + rad * math.cos(ang), p.y + rad * math.sin(ang)
                if not _cell_ok(f, cost, x, y):
                    continue
                pts.append((ko + 2.0 * kr + (0.0 if f.in_reach(x, y) else 60.0), x, y))
    pts.sort()
    out = []
    for _s, x, y in pts:
        if all(math.hypot(x - a, y - b) >= 2.0 for a, b in out):
            out.append((round(x, 1), round(y, 1)))
            if len(out) >= k:
                break
    return out


def option_label(p: Place, frm: tuple) -> str:
    """'Lake about 14 m across, 22 m north-east of me' (from the rover that would go)."""
    d = math.hypot(p.x - frm[0], p.y - frm[1])
    where = "right here" if d < 3.0 else f"{d:.0f} m {compass(p.x - frm[0], p.y - frm[1])} of me"
    near = p.kind != "lander" and math.hypot(p.x, p.y) < 12.0 and math.hypot(frm[0], frm[1]) > 20.0
    return f"{p.noun()}, {where}" + (" near the lander" if near else "")


def pick_option(text: str, cands: dict, frm: tuple, rovers: dict | None = None) -> str | None:
    """Which option of a choose_place request an operator's chat text picks: "p2", "the second one", "the
    last one", "the one near the lander", "the closest", "the north one", "the lake", "never mind" (cancel).
    cands: {"p1": Place, ...} in option order. Returns an option id, or None if it is not clearly one."""
    low = " ".join(str(text or "").lower().split())
    if not low or not cands:
        return None
    if _CANCEL_RE.search(low):
        return "cancel"
    ids = list(cands)
    m = (re.search(r"\bp\s?(\d)\b", low) or re.search(r"\b(?:option|number|no\.?|#)\s*(\d)\b", low)
         or re.fullmatch(r"(\d)\s*[.!]?", low))
    if m:
        pid = f"p{m.group(1)}"
        return pid if pid in cands else None
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|1st|2nd|3rd|4th|5th|6th|last)\b", low)
    if m:
        n = len(ids) if m.group(1) == "last" else _ORD[m.group(1)]
        return ids[n - 1] if 1 <= n <= len(ids) else None
    for tok in re.findall(r"\b[a-z]\d+\b", low):
        hit = [pid for pid, p in cands.items() if p.id == tok]
        if len(hit) == 1:
            return hit[0]
    rv = [{"id": rid, "colour": None} for rid in (rovers or {})]
    near, _span = near_reference(low, rv)
    ref = (0.0, 0.0) if near == "lander" else (rovers or {}).get(near)
    if ref is not None:
        best = min(cands.items(), key=lambda kv: math.hypot(kv[1].x - ref[0], kv[1].y - ref[1]))
        return best[0]
    qual, _n = _qualifier(low)
    if qual not in ("none", "ordinal"):
        left, ok = apply_qualifier(list(cands.values()), qual, ref=frm)
        if ok and len(left) == 1:
            return next(pid for pid, p in cands.items() if p is left[0])
    hits = [pid for pid, p in cands.items()
            if any(re.search(rf"\b{w}", low) for w in _SUB_WORDS.get(p.sub, (p.sub.lower(),)))]
    return hits[0] if len(hits) == 1 else None
