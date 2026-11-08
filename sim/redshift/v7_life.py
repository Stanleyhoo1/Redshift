"""v7 lifeform variety: more kinds of creature, chosen by biome (setting "More lifeforms", Config.life_variety).

With life_variety off nothing here runs: the world has today's grazer herds and nothing else. With it on,
each group of creatures gets a kind, drawn from the biome weights at its home spot, and a couple of extra
groups join the grazers' home areas. The kinds:

  grazer     today's herds: six-legged, teal, wander and graze, shy within about 3 m.
  skitterer  a small fast swarm (6 to 14), jittery quick steps, scatters away from a rover within 8 m.
  drifter    a glowing translucent floating bell 1.5 to 3.5 m up, trailing tendrils, drifting with its
             group's wind (2 to 5). The tendrils hang low enough for the LiDAR to see them.
  crawler    a big slow shelled loner or pair (about 1.6 m long), nearly stationary.
  burrower   pops up from the ground for a few seconds, then sinks and moves underground (3 to 6). The
             LiDAR sees it only while it is up (when down the body is under the ground surface).

Ground truth only: the rovers never read the kind. They see LiDAR points that move (tracker.py) and
camera pictures, and may name a kind from their own camera (the existing identification path).

Own random stream: default_rng([seed, 7113]) for kinds, extra groups and members; the behaviour draws
from the lifeform manager's stream like the grazers do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K_LIFE = 7113
KIND_IDS = ("grazer", "skitterer", "drifter", "crawler", "burrower")

# life weights per biome (only with life_variety on; with biomes off every kind weighs 1)
LIFE_WEIGHTS = {
    "basalt": {"crawler": 3, "skitterer": 2, "grazer": 1},
    "verdigris": {"grazer": 3, "drifter": 2, "crawler": 1},
    "frost": {"burrower": 3, "skitterer": 2},
    "ash": {"drifter": 3, "skitterer": 2, "burrower": 1},
    "salt": {"skitterer": 3, "burrower": 2, "grazer": 1},
    "obsidian": {"crawler": 2, "drifter": 2, "burrower": 1},
    "rust": {"grazer": 2, "skitterer": 1},
}


@dataclass(frozen=True)
class KindSpec:
    group: tuple          # individuals per group (lo, hi) inclusive
    size: tuple           # body length (m)
    cruise: tuple         # walking speed (m/s)
    spread: float         # members start this far from the group's middle
    sep: float            # members start at least this far apart
    home: tuple           # home area radius (m)
    one: str              # "skitterer"
    many: str             # "skitterers"
    group_word: str       # "swarm"


KINDS = {
    "grazer": KindSpec((3, 4), (1.0, 1.5), (0.2, 0.4), 3.0, 1.5, (6.0, 8.0), "grazer", "grazers", "herd"),
    "skitterer": KindSpec((6, 14), (0.4, 0.55), (0.6, 1.0), 2.2, 0.55, (5.0, 7.0), "skitterer", "skitterers",
                          "swarm"),
    "drifter": KindSpec((2, 5), (0.7, 1.1), (0.12, 0.22), 3.0, 1.6, (6.0, 8.0), "drifter", "drifters", "drift"),
    "crawler": KindSpec((1, 2), (1.6, 2.0), (0.11, 0.15), 2.0, 2.4, (4.0, 6.0), "crawler", "crawlers", "pair"),
    "burrower": KindSpec((3, 6), (0.5, 0.7), (0.25, 0.35), 3.0, 1.2, (4.0, 6.0), "burrower", "burrowers", "colony"),
}

# behaviour
SKIT_FLEE_M = 8.0       # a rover closer than this scatters a swarm
SKIT_FLEE_V = 1.4
DRIFT_AVOID_M = 5.0     # drifters drift aside from a rover closer than this
CRAWL_SHY_M = 6.0       # a crawler lumbers slowly away from a rover this close (it never stops for long,
                        # so the motion tracker keeps seeing it move)
BURROW_UP_S = (3.0, 6.0)
BURROW_DOWN_S = (6.0, 14.0)
BURROW_RISE_S = 0.6
BURROW_SHY_M = 5.0      # a burrower stays down while a rover is closer than this


def group_phrase(kind: str, n: int) -> str:
    """'a swarm of skitterers', 'a crawler', 'a herd of creatures' (grazers keep today's words)."""
    k = KINDS.get(kind)
    if k is None or kind == "grazer":
        return "a herd of creatures" if n != 1 else "a creature"
    if n == 1:
        return f"a {k.one}"
    return f"a {k.group_word} of {k.many}"


def drifter_hover(size: float) -> float:
    """Float height of a drifter's bell above the ground: bigger bells float higher (1.5 to 3.5 m)."""
    lo, hi = KINDS["drifter"].size
    return 1.5 + 2.0 * min(1.0, max(0.0, (size - lo) / (hi - lo)))


# ====================================================================== which kind (world generation)
def weights_at(terrain, x: float, y: float) -> dict:
    """Kind -> weight at (x, y): by the biome there when biomes are on, else every kind weighs 1."""
    biome = getattr(terrain, "biome", None)
    if not biome:
        return {k: 1.0 for k in KIND_IDS}
    bid = None
    fn = getattr(terrain, "biome_id_at", None)
    if callable(fn):
        try:
            bid = fn(x, y)
        except Exception:               # noqa: BLE001 (an older terrain: use the primary biome)
            bid = None
    bid = bid or biome.get("id")
    w = LIFE_WEIGHTS.get(bid)
    if not w:
        return {k: 1.0 for k in KIND_IDS}
    return {k: float(w.get(k, 0.0)) for k in KIND_IDS}


def pick_kind(rng: np.random.Generator, weights: dict, used: dict | None = None) -> str:
    """A kind drawn by weight; kinds already used weigh less (0.35 per use), so a world mixes kinds."""
    ks = [k for k in KIND_IDS if weights.get(k, 0.0) > 0.0]
    if not ks:
        return "grazer"
    w = np.array([weights[k] * (0.35 ** (used or {}).get(k, 0)) for k in ks], float)
    return ks[int(rng.choice(len(ks), p=w / w.sum()))]


def _members(terrain, rng, kind: str, cx: float, cy: float) -> list:
    k = KINDS[kind]
    want = int(rng.integers(k.group[0], k.group[1] + 1))
    out = []
    for _ in range(40 * want):
        if len(out) >= want:
            break
        rr, aa = float(rng.uniform(0.3, k.spread)), float(rng.uniform(0.0, 2 * math.pi))
        x, y = cx + rr * math.cos(aa), cy + rr * math.sin(aa)
        if kind != "drifter" and (terrain.max_slope_near(x, y, 0.6) > 14.0 or terrain.wet_near(x, y, 1.0)):
            continue
        if any(math.hypot(x - mx, y - my) < k.sep for mx, my, *_r in out):
            continue
        out.append((x, y, float(rng.uniform(0.0, 2 * math.pi)), float(rng.uniform(*k.size)),
                    float(rng.uniform(*k.cruise))))
    return out


def _site_ok(terrain, cx: float, cy: float, others: list, min_sep: float) -> bool:
    i, j = terrain.ij(cx, cy)
    if not terrain.reach[j, i] or terrain.wet_near(cx, cy, 9.0):
        return False
    kk = int(round(6.0 / terrain.res))
    win = terrain.slope[max(0, j - kk):j + kk + 1, max(0, i - kk):i + kk + 1]
    if float((win <= 12.0).mean()) < 0.85:
        return False
    return all(math.hypot(cx - h.cx, cy - h.cy) >= min_sep for h in others)


def vary(terrain, cfg) -> None:
    """Give each herd home area a kind (by the biome weights there) and add a couple of extra groups.
    Herds that stay grazers keep today's members; the others get members of their kind."""
    from .terrain import HerdSpec
    rng = np.random.default_rng([cfg.seed, K_LIFE])
    herds = list(terrain.herds)
    half = cfg.map_size / 2.0
    scale = cfg.map_size / 80.0
    n_extra = 1 if terrain.area < 1.5 else 2
    for k in range(n_extra):
        for _ in range(300):
            dist = float(rng.uniform(16.0 * min(scale, 1.0), half - 10.0))
            ang = float(rng.uniform(0.0, 2 * math.pi))
            cx, cy = dist * math.cos(ang), dist * math.sin(ang)
            if _site_ok(terrain, cx, cy, herds, 18.0 * min(scale, 1.0)):
                herds.append(HerdSpec(f"v{k + 1}", cx, cy, 6.0, []))
                break
    used: dict = {}
    out = []
    for h in herds:
        kind = pick_kind(rng, weights_at(terrain, h.cx, h.cy), used)
        if kind != "grazer" or not h.members:
            if kind == "grazer":
                members = _members(terrain, rng, "grazer", h.cx, h.cy)
            else:
                members = _members(terrain, rng, kind, h.cx, h.cy)
            if len(members) < min(2, KINDS[kind].group[0]):
                continue
            h = HerdSpec(h.id, h.cx, h.cy, float(rng.uniform(*KINDS[kind].home)), members)
        h.kind = kind
        used[kind] = used.get(kind, 0) + 1
        out.append(h)
    terrain.herds = out


# ====================================================================== MuJoCo bodies (world.py)
MATERIALS = (
    '<material name="lf_skit" rgba="0.95 0.62 0.22 1" specular="0.5" shininess="0.6" emission="0.25"/>'
    '<material name="lf_skit_dark" rgba="0.42 0.22 0.08 1" specular="0.3" shininess="0.4" emission="0.1"/>'
    '<material name="lf_drift" rgba="0.45 0.9 0.95 0.45" specular="0.9" shininess="0.9" emission="0.8"/>'
    '<material name="lf_drift_tendril" rgba="0.95 0.55 0.85 0.35" specular="0.5" shininess="0.6" emission="0.7"/>'
    '<material name="lf_crawl" rgba="0.16 0.15 0.19 1" specular="0.6" shininess="0.7" emission="0.05"/>'
    '<material name="lf_crawl_ridge" rgba="0.78 0.76 0.68 1" specular="0.4" shininess="0.5" emission="0.12"/>'
    '<material name="lf_burrow" rgba="0.72 0.48 0.62 1" specular="0.4" shininess="0.5" emission="0.15"/>'
    '<material name="lf_burrow_dark" rgba="0.34 0.2 0.3 1" specular="0.3" shininess="0.4" emission="0.08"/>'
    '<material name="lf_eye" rgba="1.0 0.95 0.6 1" emission="0.9"/>'
)


def materials_xml(terrain) -> str:
    """The extra materials, only when a world has a kind other than grazers."""
    return MATERIALS if any(getattr(h, "kind", "grazer") != "grazer" for h in getattr(terrain, "herds", [])) else ""


def body_xml(kind: str, name: str, x: float, y: float, z: float, yaw: float, size: float, group: int,
             yaw_quat) -> str:
    """A mocap body for one creature of this kind. Body origin at the ground under its middle (a drifter's
    origin is its bell, lifeforms.py lifts it). LiDAR geoms are in `group` (no contacts); looks only in
    group 4. Few LiDAR geoms per creature."""
    g = f'group="{group}" contype="0" conaffinity="0"'
    v = 'group="4" contype="0" conaffinity="0" mass="0"'
    s = size
    p = [f'<body name="{name}" mocap="true" pos="{x:.3f} {y:.3f} {z:.3f}" quat="{yaw_quat(yaw)}">']
    if kind == "skitterer":
        hz = 0.28 * s
        p.append(f'<geom name="{name}_body" type="ellipsoid" size="{0.5 * s:.3f} {0.32 * s:.3f} {0.22 * s:.3f}" '
                 f'pos="0 0 {hz:.3f}" material="lf_skit" {g}/>')
        p.append(f'<geom name="{name}_head" type="sphere" size="{0.17 * s:.3f}" pos="{0.48 * s:.3f} 0 {hz:.3f}" '
                 f'material="lf_skit_dark" {v}/>')
        for px in (-0.25, 0.05, 0.32):
            for side in (1, -1):
                p.append(f'<geom name="{name}_leg{side}{int(px * 100)}" type="capsule" fromto="{px * s:.3f} '
                         f'{side * 0.2 * s:.3f} {hz:.3f} {(px + 0.08) * s:.3f} {side * 0.62 * s:.3f} 0.01" '
                         f'size="{0.035 * s:.3f}" material="lf_skit_dark" {v}/>')
        for side in (1, -1):
            p.append(f'<geom name="{name}_eye{side}" type="sphere" size="{0.05 * s:.3f}" '
                     f'pos="{0.6 * s:.3f} {side * 0.07 * s:.3f} {hz + 0.06 * s:.3f}" material="lf_eye" {v}/>')
    elif kind == "drifter":
        hov = drifter_hover(s)
        r = 0.5 * s
        p.append(f'<geom name="{name}_bell" type="ellipsoid" size="{r:.3f} {r:.3f} {0.6 * r:.3f}" '
                 f'material="lf_drift" {g}/>')
        # the tendril sheaf hangs to about 0.35 m above the ground: what the LiDAR sees at low height
        p.append(f'<geom name="{name}_sheaf" type="capsule" fromto="0 0 {-0.3 * r:.3f} 0 0 {-(hov - 0.35):.3f}" '
                 f'size="{0.12 * s:.3f}" material="lf_drift_tendril" {g}/>')
        for k in range(4):
            a = k * math.pi / 2 + 0.4
            tx, ty = 0.55 * r * math.cos(a), 0.55 * r * math.sin(a)
            p.append(f'<geom name="{name}_tendril{k}" type="capsule" fromto="{tx:.3f} {ty:.3f} {-0.3 * r:.3f} '
                     f'{1.3 * tx:.3f} {1.3 * ty:.3f} {-(0.55 + 0.12 * k) * hov:.3f}" size="{0.025 * s:.3f}" '
                     f'material="lf_drift_tendril" {v}/>')
        p.append(f'<geom name="{name}_core" type="sphere" size="{0.35 * r:.3f}" pos="0 0 {0.1 * r:.3f}" '
                 f'material="lf_eye" {v}/>')
    elif kind == "crawler":
        hz = 0.16 * s
        p.append(f'<geom name="{name}_shell" type="ellipsoid" size="{0.5 * s:.3f} {0.36 * s:.3f} {0.26 * s:.3f}" '
                 f'pos="0 0 {hz:.3f}" material="lf_crawl" {g}/>')
        p.append(f'<geom name="{name}_head" type="ellipsoid" size="{0.12 * s:.3f} {0.14 * s:.3f} {0.08 * s:.3f}" '
                 f'pos="{0.52 * s:.3f} 0 {0.1 * s:.3f}" material="lf_crawl" {g}/>')
        for k, px in enumerate((-0.3, -0.1, 0.1, 0.3)):
            p.append(f'<geom name="{name}_ridge{k}" type="ellipsoid" size="{0.04 * s:.3f} {0.33 * s:.3f} '
                     f'{0.24 * s:.3f}" pos="{px * s:.3f} 0 {hz + 0.025 * s:.3f}" material="lf_crawl_ridge" {v}/>')
        for side in (1, -1):
            p.append(f'<geom name="{name}_eye{side}" type="sphere" size="{0.025 * s:.3f}" '
                     f'pos="{0.62 * s:.3f} {side * 0.06 * s:.3f} {0.14 * s:.3f}" material="lf_eye" {v}/>')
    elif kind == "burrower":
        r = 0.22 * s
        p.append(f'<geom name="{name}_neck" type="capsule" fromto="0 0 0 0 0 {0.75 * s:.3f}" size="{r:.3f}" '
                 f'material="lf_burrow" {g}/>')
        p.append(f'<geom name="{name}_head" type="sphere" size="{1.15 * r:.3f}" pos="{0.08 * s:.3f} 0 {0.85 * s:.3f}" '
                 f'material="lf_burrow" {g}/>')
        for k, hz in enumerate((0.2, 0.42, 0.62)):
            p.append(f'<geom name="{name}_ring{k}" type="ellipsoid" size="{1.12 * r:.3f} {1.12 * r:.3f} {0.25 * r:.3f}" '
                     f'pos="0 0 {hz * s:.3f}" material="lf_burrow_dark" {v}/>')
        for side in (1, -1):
            p.append(f'<geom name="{name}_eye{side}" type="sphere" size="{0.06 * s:.3f}" '
                     f'pos="{0.08 * s + 0.9 * r:.3f} {side * 0.5 * r:.3f} {0.9 * s:.3f}" material="lf_eye" {v}/>')
    else:
        raise ValueError(f"unknown kind {kind!r}")
    p.append('</body>')
    return "".join(p)


def burrow_depth(size: float) -> float:
    """How far below the ground a burrower's body origin sits when it is down (its top under the surface)."""
    return 0.85 * size + 0.3 * size + 0.35


# ====================================================================== behaviour (lifeforms.py calls these)
def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def init_creature(mgr, h, c) -> None:
    """Per-kind state on a new creature."""
    rng = mgr.rng
    c.hover = drifter_hover(c.size) if c.kind == "drifter" else 0.0
    c.up = 1.0
    c.burrow = "up"
    c.burrow_until = float(rng.uniform(*BURROW_UP_S))
    c.offset = (c.x - h.cx, c.y - h.cy)
    if c.kind == "crawler":
        c.pause_until = float(rng.uniform(0.0, 4.0))
    if c.kind == "drifter" and not hasattr(h, "wind"):
        h.wind = float(rng.uniform(0.0, 2 * math.pi))


def _nearest_rover(c, rovers):
    best = None
    for rx, ry in rovers:
        d = math.hypot(c.x - rx, c.y - ry)
        if best is None or d < best[0]:
            best = (d, rx, ry)
    return best


def _walk(mgr, c, want: float, v: float, dt: float, turn_rate: float, accel: float, max_v: float,
          probe: float = 1.0) -> None:
    """Turn toward `want`, keep off bad ground (truth: they live here), move at up to v."""
    if v > 0.0:
        for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi):
            a = want + off
            if not mgr._bad(c.x + probe * math.cos(a), c.y + probe * math.sin(a)):
                want = a
                break
        else:
            v = 0.0
    turn = _wrap(want - c.yaw)
    c.yaw = _wrap(c.yaw + max(-turn_rate * dt, min(turn_rate * dt, turn)))
    if abs(turn) > 1.2:
        v *= 0.3
    c.speed += max(-accel * dt, min(accel * dt, min(v, max_v) - c.speed))
    nx_, ny_ = c.x + c.speed * dt * math.cos(c.yaw), c.y + c.speed * dt * math.sin(c.yaw)
    if not mgr._bad(nx_, ny_):
        c.x, c.y = nx_, ny_
    else:
        c.speed = 0.0


def _new_goal(mgr, h, c, t: float, every: tuple, reach: float | None = None) -> None:
    rng = mgr.rng
    if reach is None:
        rr = h.radius * math.sqrt(float(rng.uniform()))
        aa = float(rng.uniform(0, 2 * math.pi))
        c.goal = (h.cx + rr * math.cos(aa), h.cy + rr * math.sin(aa))
    else:
        aa = float(rng.uniform(0, 2 * math.pi))
        gx, gy = c.x + reach * math.cos(aa), c.y + reach * math.sin(aa)
        if math.hypot(gx - h.cx, gy - h.cy) > h.radius:          # stay near home
            gx, gy = 0.5 * (gx + h.cx), 0.5 * (gy + h.cy)
        c.goal = (gx, gy)
    c.next_goal = t + float(rng.uniform(*every))


def step_herd(mgr, h, t: float, dt: float, rovers: list) -> None:
    fn = {"skitterer": _step_skitterer, "drifter": _step_drifter, "crawler": _step_crawler,
          "burrower": _step_burrower}[h.kind]
    fn(mgr, h, t, dt, rovers)
    for c in h.members:
        place(mgr, c, t)


def _step_skitterer(mgr, h, t, dt, rovers):
    rng = mgr.rng
    hx, hy = h.centre()
    for c in h.members:
        near = _nearest_rover(c, rovers)
        if near is not None and near[0] < SKIT_FLEE_M:
            if not getattr(c, "fleeing", False):
                c.flee_jitter = float(rng.uniform(-0.7, 0.7))      # the swarm scatters, each its own way
            c.fleeing = True
            want = math.atan2(c.y - near[2], c.x - near[1]) + c.flee_jitter
            v = SKIT_FLEE_V
            c.pause_until = t + float(rng.uniform(0.3, 1.0))
        elif t < c.pause_until:
            c.fleeing = False
            c.yaw = _wrap(c.yaw + float(rng.normal(0.0, 0.25)))  # twitchy while it waits
            want, v = c.yaw, 0.0
        else:
            c.fleeing = False
            if t >= c.next_goal or math.hypot(c.goal[0] - c.x, c.goal[1] - c.y) < 0.4:
                _new_goal(mgr, h, c, t, (1.2, 3.5), reach=float(rng.uniform(1.0, 3.0)))
                if rng.uniform() < 0.3:
                    c.pause_until = t + float(rng.uniform(0.4, 1.5))
            gx, gy = c.goal[0] - c.x, c.goal[1] - c.y
            dch = math.hypot(c.x - hx, c.y - hy)
            if dch > 2.5:
                w = min(1.0, (dch - 2.5) / 2.0)
                gx = (1 - w) * gx + w * (hx - c.x) * 2.0
                gy = (1 - w) * gy + w * (hy - c.y) * 2.0
            want = math.atan2(gy, gx) + float(rng.normal(0.0, 0.35))  # quick jittery gait
            v = c.cruise
        _walk(mgr, c, want, v, dt, 5.0, 3.0, SKIT_FLEE_V, probe=0.6)


def _step_drifter(mgr, h, t, dt, rovers):
    rng = mgr.rng
    half = mgr._half
    # the group's wind veers slowly; near the map edge or the lander it turns back toward open ground
    h.wind = _wrap(h.wind + float(rng.normal(0.0, 0.05)) * dt)
    edge = max(abs(h.cx), abs(h.cy)) > half - 14.0
    close = math.hypot(h.cx, h.cy) < mgr.cfg.lander_radius + 14.0
    if edge or close:
        out = math.atan2(-h.cy, -h.cx) if edge else math.atan2(h.cy, h.cx)
        h.wind = _wrap(h.wind + max(-0.3 * dt, min(0.3 * dt, _wrap(out - h.wind))))
    wv = sum(c.cruise for c in h.members) / max(1, len(h.members))
    h.cx += wv * dt * math.cos(h.wind)
    h.cy += wv * dt * math.sin(h.wind)
    for c in h.members:
        ox, oy = c.offset
        ang = 0.05 * dt
        c.offset = (ox * math.cos(ang) - oy * math.sin(ang), ox * math.sin(ang) + oy * math.cos(ang))
        tx, ty = h.cx + c.offset[0], h.cy + c.offset[1]
        vx = wv * math.cos(h.wind) + 0.3 * (tx - c.x)
        vy = wv * math.sin(h.wind) + 0.3 * (ty - c.y)
        near = _nearest_rover(c, rovers)
        if near is not None and near[0] < DRIFT_AVOID_M:
            k = 0.5 * (DRIFT_AVOID_M - near[0])
            vx += k * (c.x - near[1]) / max(near[0], 0.1)
            vy += k * (c.y - near[2]) / max(near[0], 0.1)
        sp = math.hypot(vx, vy)
        if sp > 0.6:
            vx, vy = vx * 0.6 / sp, vy * 0.6 / sp
            sp = 0.6
        nx_, ny_ = c.x + vx * dt, c.y + vy * dt
        if abs(nx_) < half - 2.0 and abs(ny_) < half - 2.0:
            c.x, c.y = nx_, ny_
        c.speed = sp
        c.yaw = _wrap(c.yaw + 0.25 * dt)            # the bell turns slowly as it drifts


def _step_crawler(mgr, h, t, dt, rovers):
    rng = mgr.rng
    for c in h.members:
        near = _nearest_rover(c, rovers)
        if near is not None and near[0] < CRAWL_SHY_M:
            want, v = math.atan2(c.y - near[2], c.x - near[1]), c.cruise     # lumbers off, no faster
        elif t < c.pause_until:
            want, v = c.yaw, 0.0
        else:
            if t >= c.next_goal or math.hypot(c.goal[0] - c.x, c.goal[1] - c.y) < 0.5:
                _new_goal(mgr, h, c, t, (25.0, 45.0), reach=float(rng.uniform(2.5, 4.0)))
                if rng.uniform() < 0.25:
                    c.pause_until = t + float(rng.uniform(2.0, 5.0))
            want = math.atan2(c.goal[1] - c.y, c.goal[0] - c.x)
            v = c.cruise
        _walk(mgr, c, want, v, dt, 0.3, 0.06, 0.15, probe=1.4)


def _step_burrower(mgr, h, t, dt, rovers):
    rng = mgr.rng
    for c in h.members:
        if c.burrow == "up":
            if t >= c.burrow_until:
                c.burrow, c.burrow_until = "sinking", t + BURROW_RISE_S
            else:
                if t >= c.next_goal or math.hypot(c.goal[0] - c.x, c.goal[1] - c.y) < 0.3:
                    _new_goal(mgr, h, c, t, (1.5, 3.0), reach=1.0)
                _walk(mgr, c, math.atan2(c.goal[1] - c.y, c.goal[0] - c.x), c.cruise, dt, 2.0, 0.8, 0.4, probe=0.6)
        elif c.burrow == "sinking":
            c.up = max(0.0, c.up - dt / BURROW_RISE_S)
            c.speed = 0.0
            if c.up <= 0.0:
                c.burrow, c.burrow_until = "down", t + float(rng.uniform(*BURROW_DOWN_S))
                _new_goal(mgr, h, c, t, (30.0, 40.0), reach=float(rng.uniform(1.5, 3.0)))
        elif c.burrow == "down":
            gx, gy = c.goal[0] - c.x, c.goal[1] - c.y          # tunnels toward where it comes up next
            d = math.hypot(gx, gy)
            if d > 0.05:
                step = min(d, c.cruise * dt)
                nx_, ny_ = c.x + gx / d * step, c.y + gy / d * step
                if not mgr._bad(nx_, ny_):
                    c.x, c.y = nx_, ny_
                c.yaw = math.atan2(gy, gx)
            c.speed = 0.0
            if t >= c.burrow_until:
                near = _nearest_rover(c, rovers)
                if near is not None and near[0] < BURROW_SHY_M:      # shy: not up next to a rover
                    c.burrow_until = t + float(rng.uniform(2.0, 4.0))
                    away = math.atan2(c.y - near[2], c.x - near[1])
                    c.goal = (c.x + 2.0 * math.cos(away), c.y + 2.0 * math.sin(away))
                else:
                    c.burrow, c.burrow_until = "rising", t + BURROW_RISE_S
        else:                                                   # rising
            c.up = min(1.0, c.up + dt / BURROW_RISE_S)
            c.speed = 0.0
            if c.up >= 1.0:
                c.burrow, c.burrow_until = "up", t + float(rng.uniform(*BURROW_UP_S))
                c.next_goal = t


def place(mgr, c, t: float) -> None:
    """Put the creature's mocap body where it is (per kind height)."""
    d = mgr.sim.data
    g = mgr.ter.height_at(c.x, c.y)
    if c.kind == "drifter":
        c.z = g + c.hover + 0.15 * math.sin(c.phase + 0.8 * t)
        zb = c.z
    elif c.kind == "burrower":
        c.z = g - burrow_depth(c.size) * (1.0 - c.up)
        zb = c.z
    else:
        c.z = g
        jig = 0.03 if c.kind == "skitterer" else 0.01
        zb = g + (jig * abs(math.sin(c.phase + 14.0 * t)) if c.speed > 0.05 else 0.0)
    d.mocap_pos[c.mocap] = (c.x, c.y, zb)
    d.mocap_quat[c.mocap] = (math.cos(c.yaw / 2), 0.0, 0.0, math.sin(c.yaw / 2))


def on_moved(mgr, herd, t: float) -> None:
    """The lifeform hazard carried this group into view: start them off so the rovers see motion."""
    for c in herd.members:
        c.offset = (c.x - herd.cx, c.y - herd.cy)
        if herd.kind == "burrower":
            c.burrow, c.up, c.burrow_until = "up", 1.0, t + float(mgr.rng.uniform(4.0, 7.0))
            c.next_goal = t
        if herd.kind == "crawler":
            c.pause_until = 0.0
        place(mgr, c, t)


def msg_extra(c) -> dict:
    """Extra lifeform message fields for some kinds (only with life_variety on)."""
    if c.kind == "burrower":
        return {"up": round(c.up, 2)}
    return {}
