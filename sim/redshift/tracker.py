"""Fleet motion tracker: turns LiDAR point clusters that move between sweeps into tracks (protocol v4
`tracks`).

Simplification, stated honestly: the sim routes LiDAR hits on creature geoms here instead of into the
terrain map (rover.py picks them out by geom id). A real rover would separate them by scan-to-map
differencing (returns that stand where the map says there is open ground). Everything after that
split is done the honest way: the tracker only sees anonymous 3D points. It clusters each sweep,
associates clusters with existing tracks (nearest predicted position inside a gate), smooths
position and velocity with an alpha-beta filter, and only calls something a moving object once it
has been seen in several sweeps AND has moved more than MOVE_M at a believable speed. A cluster that
never moves stays a silent candidate.

Tracks are fleet-wide (every rover's sweeps feed the same tracks). A track not seen for LOST_S becomes
"lost"; a new cluster near where a lost track should be picks it up again (same id), so one creature
is not counted twice.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

LINK_M = 0.7          # points closer than this (xy) belong to the same cluster
MIN_POINTS = 2        # smaller clusters are ignored (a single stray return)
GATE_M = 1.4          # a cluster this close to a track's predicted position updates that track
MOVE_M = 0.35         # a candidate must move at least this far ...
MIN_SPEED = 0.08      # ... at least this fast on average ...
MIN_OBS = 3           # ... over at least this many sweeps, to become a moving-object track
LOST_S = 4.0          # not seen for this long: status "lost"
DROP_S = 40.0         # lost for this long: forgotten
CAND_DROP_S = 6.0     # unconfirmed candidates not seen for this long are dropped
REACQUIRE_M = 3.0     # a new cluster this close to a lost track's last position resumes that track
MERGE_M = 0.9         # two live tracks this close are one object
ALPHA, BETA = 0.5, 0.15


class Track:
    def __init__(self, tid: str, x: float, y: float, size: float, t: float, rover: str):
        self.id = tid
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.size = size
        self.t0 = t
        self.last_seen = t
        self.last_t = t
        self.n = 1
        self.hist: deque = deque(maxlen=40)          # (t, x, y) raw cluster centres
        self.hist.append((t, x, y))
        self.confirmed = False
        self.confirm_t = None
        self.status = "tracking"
        self.observer = None                         # rover id observing it
        self.label = "moving object"
        self.lifeform = False                        # confirmed as a lifeform (Gemma or the rules)
        self.seen_by = {rover}
        self.done_until = -1e9                       # no new observation of this one before then
        self.ignored = False                         # operator said: keep exploring

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def predict(self, t: float, cap: float = 2.0):
        dt = min(max(t - self.last_t, 0.0), cap)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def to_msg(self) -> dict:
        return {"id": self.id, "x": round(self.x, 2), "y": round(self.y, 2), "vx": round(self.vx, 2),
                "vy": round(self.vy, 2), "size": round(self.size, 2), "last_seen": round(self.last_seen, 1),
                "observer": self.observer, "label": self.label, "status": self.status}


def cluster(pts: np.ndarray) -> list:
    """Single-link clustering of (N, 3) points in xy. Returns [(cx, cy, size, ztop, n)]."""
    n = len(pts)
    if n == 0:
        return []
    xy = pts[:, :2]
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    adj = d2 < LINK_M * LINK_M
    label = -np.ones(n, np.int64)
    out = []
    for s in range(n):
        if label[s] >= 0:
            continue
        stack = [s]
        label[s] = s
        members = []
        while stack:
            k = stack.pop()
            members.append(k)
            for q in np.flatnonzero(adj[k] & (label < 0)):
                label[q] = s
                stack.append(int(q))
        if len(members) < MIN_POINTS:
            continue
        p = pts[members]
        span = float(np.sqrt(((p[:, None, :2] - p[None, :, :2]) ** 2).sum(-1).max()))
        out.append((float(p[:, 0].mean()), float(p[:, 1].mean()), span + 0.3, float(p[:, 2].max()), len(members)))
    return out


class Tracker:
    def __init__(self):
        self.tracks: list = []          # confirmed and candidate tracks
        self.seq = 0
        self.merged: dict = {}          # ids of tracks merged into another one
        self.new_confirmed: list = []   # tracks confirmed since the fleet last looked

    # ------------------------------------------------------------------ input
    def add_scan(self, rover_id: str, pts: np.ndarray, t: float) -> None:
        """One sweep's returns from moving-object candidates (N, 3) in world metres."""
        clusters = cluster(pts) if len(pts) else []
        claimed = set()
        for (cx, cy, size, _z, _n) in sorted(clusters, key=lambda c: -c[4]):
            best, bd = None, GATE_M
            for tr in self.tracks:
                if tr.id in claimed or tr.status == "lost":
                    continue
                px, py = tr.predict(t)
                d = math.hypot(cx - px, cy - py)
                if d < bd:
                    best, bd = tr, d
            if best is None:            # maybe a lost track coming back into view
                for tr in self.tracks:
                    if tr.id in claimed or tr.status != "lost" or not tr.confirmed:
                        continue
                    px, py = tr.predict(t, cap=3.0)
                    if math.hypot(cx - px, cy - py) < REACQUIRE_M:
                        best = tr
                        tr.status = "tracking"
                        tr.last_t = t
                        tr.x, tr.y = cx, cy
                        break
            if best is None:
                self.seq += 1
                tr = Track(f"k{self.seq}", cx, cy, size, t, rover_id)
                self.tracks.append(tr)
                claimed.add(tr.id)
                continue
            claimed.add(best.id)
            self._update(best, cx, cy, size, t, rover_id)

    def _update(self, tr: Track, cx: float, cy: float, size: float, t: float, rover_id: str) -> None:
        dt = t - tr.last_t
        if dt > 1e-3:
            px, py = tr.x + tr.vx * dt, tr.y + tr.vy * dt
            rx, ry = cx - px, cy - py
            tr.x, tr.y = px + ALPHA * rx, py + ALPHA * ry
            tr.vx += BETA * rx / dt
            tr.vy += BETA * ry / dt
            sp = math.hypot(tr.vx, tr.vy)
            if sp > 1.5:                # a cluster jump, not a creature: keep the speed believable
                tr.vx, tr.vy = tr.vx * 1.5 / sp, tr.vy * 1.5 / sp
            tr.last_t = t
        else:                           # a second rover saw it in the same cycle: average
            tr.x, tr.y = 0.5 * (tr.x + cx), 0.5 * (tr.y + cy)
        tr.size = max(0.8 * tr.size + 0.2 * size, min(size, 2.5)) if size > tr.size else 0.9 * tr.size + 0.1 * size
        tr.last_seen = t
        tr.n += 1
        tr.hist.append((t, cx, cy))
        tr.seen_by.add(rover_id)
        if not tr.confirmed and tr.n >= MIN_OBS:
            t0, x0, y0 = tr.hist[0]
            moved = max(math.hypot(x - x0, y - y0) for _, x, y in tr.hist)
            span = t - t0
            net = math.hypot(cx - x0, cy - y0) / max(span, 1e-6)
            if moved >= MOVE_M and span >= 0.8 and net >= MIN_SPEED:
                tr.confirmed = True
                tr.confirm_t = t
                self.new_confirmed.append(tr)

    # ------------------------------------------------------------------ upkeep
    def update(self, t: float) -> None:
        self._merge_duplicates(t)
        keep = []
        for tr in self.tracks:
            age = t - tr.last_seen
            if not tr.confirmed:
                if age <= CAND_DROP_S:
                    keep.append(tr)
                continue
            if age > LOST_S and tr.status != "lost":
                tr.status = "lost"
                tr.vx *= 0.5
                tr.vy *= 0.5
            if tr.status == "lost" and age > DROP_S and tr.observer is None:
                continue
            keep.append(tr)
        self.tracks = keep

    def _merge_duplicates(self, t: float) -> None:
        """Two live tracks on top of each other (one creature seen as two clusters, say from two rovers
        on opposite sides) become one: the older keeps its id, observer and label."""
        live = [tr for tr in self.tracks if tr.status == "tracking" and t - tr.last_seen < 1.0]
        drop = set()
        for a in range(len(live)):
            ta = live[a]
            if ta.id in drop:
                continue
            for b in range(a + 1, len(live)):
                tb = live[b]
                if tb.id in drop or math.hypot(ta.x - tb.x, ta.y - tb.y) > MERGE_M:
                    continue
                old, new = (ta, tb) if ta.t0 <= tb.t0 else (tb, ta)
                if new.observer is not None and old.observer is None:
                    old.observer = new.observer
                old.confirmed = old.confirmed or new.confirmed
                old.lifeform = old.lifeform or new.lifeform
                old.n += new.n
                old.last_seen = max(old.last_seen, new.last_seen)
                drop.add(new.id)
                self.merged[new.id] = old.id
        if drop:
            self.tracks = [tr for tr in self.tracks if tr.id not in drop]
            if len(self.merged) > 500:
                self.merged = dict(list(self.merged.items())[-200:])

    # ------------------------------------------------------------------ queries
    def get(self, tid: str):
        for _ in range(10):
            tr = next((tr for tr in self.tracks if tr.id == tid), None)
            if tr is not None or tid not in self.merged:
                return tr
            tid = self.merged[tid]                  # merged into a nearby older track: follow that one
        return None

    @staticmethod
    def distinct(tracks: list, t: float, sep: float = 1.0) -> int:
        """How many separate objects these tracks are (tracks closer than sep count once)."""
        pts = []
        for tr in tracks:
            p = tr.predict(t)
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= sep for q in pts):
                pts.append(p)
        return len(pts)

    def confirmed(self) -> list:
        return [tr for tr in self.tracks if tr.confirmed]

    def near(self, x: float, y: float, radius: float, t: float, live_only: bool = True) -> list:
        out = []
        for tr in self.tracks:
            if not tr.confirmed or (live_only and tr.status == "lost"):
                continue
            px, py = tr.predict(t)
            if math.hypot(px - x, py - y) <= radius:
                out.append(tr)
        return out

    def to_msgs(self) -> list:
        return [tr.to_msg() for tr in self.tracks if tr.confirmed]
