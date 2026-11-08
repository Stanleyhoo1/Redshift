"""Alien flora (v7, flag cfg.flora): fungal trees and glowing bulb bushes in a few groves.

Truth side: placement (own stream, K = 7109), the MuJoCo geoms and a contact counter.
Rover side: identification from the rover's own sensors only. LiDAR sees a stalk as an obstacle. When a
LiDAR return within 10 m lands on an object the rover has not classified, the mast camera takes a small
segmentation frame (throttled, camera hardware time through Rover._hw_call) and labels it. Stated
simplification: LiDAR returns are routed by geom id to find the object to look at (like the creature
tracker), and the segmentation frame stands in for a trained classifier. The class comes from what the
camera drew, never from the plant list. A plant that is identified gets a keep-out zone of plant radius
plus observe_min_dist (planetary protection) that the planner treats as impassable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

K_FLORA = 7109
ID_RANGE_M = 10.0              # LiDAR returns closer than this make a plant a candidate
CAMERA_EVERY_S = 2.0           # at most one identification frame per rover per this many seconds
MIN_PIXELS = 10                # stalk pixels needed in the frame
MIN_CONF = 0.6
NOMINAL_RADIUS = {"tree": 2.2, "bush": 0.6}   # what the rover assumes (largest cap, largest cluster)
TARGET_CLEAR_M = 2.2 + 4.0 + 1.5             # targets stay outside any keep-out zone
PROXY_RGBA = "0.42 0.38 0.34 1"              # neutral grey brown: never a marker colour
WORDS = {"tree": "fungal tree", "bush": "glowing bush"}


@dataclass
class Plant:
    id: int
    kind: str                  # "tree" or "bush"
    x: float
    y: float
    z: float                   # ground height at the foot
    height: float
    radius: float              # cap radius (tree) or cluster radius (bush)
    hue: float                 # 0 violet .. 1 teal (trees); bulb tint for bushes
    bulbs: int = 0
    stalk: float = 0.2         # stalk (proxy) radius

    def name(self) -> str:
        return f"flora_{self.kind}_{self.id}"

    def to_msg(self) -> dict:
        out = {"id": self.id, "kind": self.kind, "x": round(self.x, 2), "y": round(self.y, 2),
               "height": round(self.height, 2), "radius": round(self.radius, 2), "hue": round(self.hue, 3)}
        if self.kind == "bush":
            out["bulbs"] = int(self.bulbs)
        return out


# ====================================================================== placement (truth)
def place(cfg, terrain) -> list:
    """Plants in a few groves, on gentle dry ground, clear of the lander, water, spires, rocks, herds and
    targets. Deterministic per seed (own stream)."""
    rng = np.random.default_rng([cfg.seed, K_FLORA])
    area = terrain.area
    half = cfg.map_size / 2.0
    n_trees = max(3, int(round(9 * area)))
    n_bush = max(4, int(round(13 * area)))
    n_groves = max(2, int(round(2 * area)))
    slope = terrain.slope

    def gentle(x, y, r, lim):
        i, j = terrain.ij(x, y)
        k = max(1, int(math.ceil(r / terrain.res)))
        win = slope[max(0, j - k):j + k + 1, max(0, i - k):i + k + 1]
        return win.size > 0 and float(win.max()) <= lim

    groves = []
    for _ in range(400):
        if len(groves) >= n_groves:
            break
        gx, gy = rng.uniform(-half + 10, half - 10, 2)
        if math.hypot(gx, gy) < 22.0 or not gentle(gx, gy, 3.0, 14.0):
            continue
        if any(math.hypot(gx - a, gy - b) < 18.0 for a, b in groves):
            continue
        groves.append((float(gx), float(gy)))
    if not groves:
        return []

    herd_pts = [(m[0], m[1]) for h in getattr(terrain, "herds", []) for m in h.members]
    plants: list = []

    def ok(x, y, r, kind):
        if abs(x) > half - 4 or abs(y) > half - 4:
            return False
        if math.hypot(x, y) < cfg.lander_flat_radius + 12.0 + r:
            return False
        if not gentle(x, y, max(1.0, r), 12.0):
            return False
        if terrain.wet_near(x, y, r + 3.0):
            return False
        for sp in terrain.spires:
            if math.hypot(x - sp.x, y - sp.y) < sp.radius + r + 3.0:
                return False
        for rk in terrain.rocks:
            if math.hypot(x - rk.x, y - rk.y) < max(rk.sx, rk.sy) + r + 0.6:
                return False
        for t in terrain.targets:
            if math.hypot(x - t.x, y - t.y) < TARGET_CLEAR_M:
                return False
        for hx, hy in herd_pts:
            if math.hypot(x - hx, y - hy) < 8.0:
                return False
        if getattr(terrain, "landform_mask", None) is not None:      # not in pits, on mesas or in the canyon
            from . import v7_landforms
            if not v7_landforms.spot_clear(terrain, x, y, r + 1.0):
                return False
        for v in getattr(terrain, "geysers", None) or []:              # clear of the vents' steam
            if math.hypot(x - v.x, y - v.y) < r + 8.0:
                return False
        for a in getattr(terrain, "ice_sheets", None) or []:           # not on ice
            if math.hypot(x - a["x"], y - a["y"]) < a["radius"] + r + 1.0:
                return False
        for p in plants:
            if math.hypot(x - p.x, y - p.y) < r + p.radius + 0.8:
                return False
        return True

    for kind, count in (("tree", n_trees), ("bush", n_bush)):
        made = 0
        for _ in range(count * 40):
            if made >= count:
                break
            gx, gy = groves[int(rng.integers(len(groves)))]
            x = float(gx + rng.normal(0.0, 5.0))
            y = float(gy + rng.normal(0.0, 5.0))
            if kind == "tree":
                height = float(rng.uniform(2.5, 4.5))
                radius = float(rng.uniform(1.2, 2.2))
                hue = float(rng.uniform(0.0, 1.0))
                bulbs, stalk = 0, float(rng.uniform(0.18, 0.28))
            else:
                height = float(rng.uniform(0.4, 0.7))
                radius = float(rng.uniform(0.25, 0.5))
                hue = float(rng.uniform(0.0, 1.0))
                bulbs, stalk = int(rng.integers(5, 13)), radius * 0.8
            if not ok(x, y, radius, kind):
                continue
            plants.append(Plant(len(plants), kind, x, y, float(terrain.height_at(x, y)), height, radius, hue,
                                bulbs, stalk))
            made += 1
    return plants


def _cap_rgba(hue: float) -> str:
    a = np.array([0.56, 0.24, 0.86])     # violet
    b = np.array([0.12, 0.74, 0.70])     # teal
    c = (1 - hue) * a + hue * b
    return f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"


def geoms_xml(plants: list) -> str:
    """Static geoms. Group 2 proxy (LiDAR and machine vision, not the bumper) plus group 4 decor."""
    out = []
    deco = 'group="4" contype="0" conaffinity="0"'
    for p in plants:
        n = p.name()
        if p.kind == "tree":
            top = p.z + p.height
            out.append(f'<geom name="{n}_stalk" type="capsule" size="{p.stalk:.3f}" '
                       f'fromto="{p.x:.3f} {p.y:.3f} {p.z - 0.2:.3f} {p.x:.3f} {p.y:.3f} {top - 0.3:.3f}" '
                       f'rgba="{PROXY_RGBA}" group="2" contype="1" conaffinity="0"/>')
            out.append(f'<geom name="{n}_cap" type="ellipsoid" pos="{p.x:.3f} {p.y:.3f} {top:.3f}" '
                       f'size="{p.radius:.3f} {p.radius:.3f} {0.42 * p.radius:.3f}" rgba="{_cap_rgba(p.hue)}" {deco}/>')
            out.append(f'<geom name="{n}_gills" type="cylinder" pos="{p.x:.3f} {p.y:.3f} {top - 0.1:.3f}" '
                       f'size="{0.85 * p.radius:.3f} 0.04" rgba="0.80 0.95 0.55 1" {deco}/>')
        else:
            out.append(f'<geom name="{n}_stalk" type="ellipsoid" pos="{p.x:.3f} {p.y:.3f} {p.z + 0.5 * p.height:.3f}" '
                       f'size="{p.stalk:.3f} {p.stalk:.3f} {0.5 * p.height:.3f}" rgba="{PROXY_RGBA}" '
                       f'group="2" contype="0" conaffinity="0"/>')
            for k, (ox, oy) in enumerate(((0.35, 0.1), (-0.3, -0.2))):
                out.append(f'<geom name="{n}_bulb{k}" type="sphere" '
                           f'pos="{p.x + ox * p.radius:.3f} {p.y + oy * p.radius:.3f} {p.z + p.height:.3f}" '
                           f'size="{0.12 + 0.1 * p.radius:.3f}" rgba="0.72 1.0 0.28 1" {deco}/>')
    return "".join(out)


# ====================================================================== camera check (rover side)
def _seg_frame(sim, rid: str, w: int = 160, h: int = 120) -> np.ndarray:
    """Segmentation frame from the rover's mast camera: (h, w) geom id per pixel, -1 for sky.
    Same view as the machine-vision frames (fleet hardware and decor hidden). Kept here so it can later
    be swapped for camera.detect_objects."""
    from . import camera
    gl = camera._local_renderer()
    gl.ensure()
    g = gl._gfx_for(sim.model)
    rover = camera._find_rover(sim, rid)
    spec = camera.rig("fpv", sim, rover, None)
    camera._apply_free_camera(g.cam, spec)
    mujoco.mjv_updateScene(g.model, sim.data, g.option((3, 4)), None, g.cam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, g.scn)
    w = max(16, min(w, g.max_w))
    h = max(16, min(h, g.max_h))
    camera._set_frustum(g.scn, spec)
    fl = g.scn.flags
    keep = (int(fl[mujoco.mjtRndFlag.mjRND_SHADOW]), int(fl[mujoco.mjtRndFlag.mjRND_SKYBOX]))
    fl[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    fl[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    fl[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
    fl[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
    try:
        rect = mujoco.MjrRect(0, 0, w, h)
        gl._ctx.make_current()
        mujoco.mjr_render(rect, g.scn, g.con)
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, rect, g.con)
    finally:
        fl[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        fl[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0
        fl[mujoco.mjtRndFlag.mjRND_SHADOW] = keep[0]
        fl[mujoco.mjtRndFlag.mjRND_SKYBOX] = keep[1]
    seg = rgb[::-1].astype(np.int64)
    sid = seg[..., 0] + seg[..., 1] * 256 + seg[..., 2] * 65536 - 1
    out = np.full(sid.shape, -1, dtype=np.int64)
    for s in np.unique(sid):
        if 0 <= s < g.scn.ngeom:
            gg = g.scn.geoms[int(s)]
            if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                out[sid == s] = int(gg.objid)
    return out


def classify(model, seg: np.ndarray, gids) -> dict:
    """gid -> (cls, confidence, pixels) for each asked-about object the frame shows as a plant."""
    out = {}
    for gid in gids:
        px = int((seg == gid).sum())
        if px < MIN_PIXELS:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
        cls = "tree" if name.startswith("flora_tree_") else "bush" if name.startswith("flora_bush_") else None
        if cls is None:
            continue
        out[int(gid)] = (cls, round(min(0.97, 0.6 + 0.37 * min(1.0, px / 300.0)), 2), px)
    return out


# ====================================================================== the system on a Simulation
class FloraSystem:
    def __init__(self, sim):
        self.sim = sim
        self.plants = list(getattr(sim.terrain, "flora", []) or [])
        m = sim.model
        self.gid_plant = {}
        for k, p in enumerate(self.plants):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{p.name()}_stalk")
            if gid >= 0:
                self.gid_plant[int(gid)] = k
        self.gid_arr = np.array(sorted(self.gid_plant), dtype=np.int32)
        self.zones: dict = {}          # gid -> (x, y, keep-out radius, cls, rover id, t, conf, dist)
        self._cand: dict = {}          # rover id -> {gid: [sum x, sum y, n, min dist]}
        self._cam_t: dict = {}
        self.contacts = 0              # truth: rover-plant contact events
        self.closest = float("inf")    # truth: closest rover centre to plant edge (m)
        self._touching: set = set()
        self.p_xy = np.array([[p.x, p.y] for p in self.plants]) if self.plants else np.zeros((0, 2))
        self.p_r = np.array([p.stalk if p.kind == "bush" else p.radius for p in self.plants])

    # ---------------------------------------------------------------- truth
    def truth_step(self) -> None:
        if not self.plants:
            return
        d = self.sim.data
        now = set()
        gp = self.gid_plant
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in gp or g2 in gp:
                other = g2 if g1 in gp else g1
                for rid, (a, b) in self.sim.world.rover_geom_range.items():
                    if a <= other <= b:
                        now.add((rid, g1 if g1 in gp else g2))
        self.contacts += len(now - self._touching)
        self._touching = now
        for r in self.sim.rovers:
            pos = d.xpos[r.body][:2] if hasattr(r, "body") else None
            if pos is None:
                continue
            gap = float((np.hypot(self.p_xy[:, 0] - pos[0], self.p_xy[:, 1] - pos[1]) - self.p_r).min())
            if gap < self.closest:
                self.closest = gap

    # ---------------------------------------------------------------- rover side
    def on_scan(self, rover, pnt, vec, gid, dist, hit, t: float) -> None:
        """LiDAR hook: remember returns on objects not classified yet, then maybe look with the camera."""
        if not len(self.gid_arr):
            return
        is_f = hit & (dist <= ID_RANGE_M) & np.isin(gid, self.gid_arr)
        cand = self._cand.setdefault(rover.id, {})
        if is_f.any():
            ks = np.flatnonzero(is_f)
            pts = pnt[None, :] + vec[ks] * dist[ks, None]
            for k, p in zip(ks, pts):
                g = int(gid[k])
                if g in self.zones:
                    continue
                c = cand.setdefault(g, [0.0, 0.0, 0, 1e9, t])
                c[0] += float(p[0])
                c[1] += float(p[1])
                c[2] += 1
                c[3] = min(c[3], float(dist[k]))
                c[4] = t
        for g in [g for g, c in cand.items() if g in self.zones or t - c[4] > 10.0]:
            del cand[g]
        if not cand or t - self._cam_t.get(rover.id, -1e9) < CAMERA_EVERY_S:
            return
        if getattr(self.sim, "camera_ok", True) is False:
            return
        self._cam_t[rover.id] = t
        try:
            seg = rover._hw_call(_seg_frame, self.sim, rover.id)
        except Exception:                  # noqa: BLE001 (no OpenGL here: LiDAR obstacle only)
            self.sim.camera_ok = False
            return
        for g, (cls, conf, _px) in classify(self.sim.model, seg, list(cand)).items():
            if conf < MIN_CONF:
                continue
            c = cand.pop(g)
            x, y = c[0] / c[2], c[1] / c[2]
            self._identify(rover, g, cls, conf, x, y, c[3], t)

    def _identify(self, rover, g, cls, conf, x, y, rng_m, t) -> None:
        fleet = rover.fleet
        keep = NOMINAL_RADIUS[cls] + float(rover.cfg.observe_min_dist)
        self.zones[g] = (float(x), float(y), keep, cls, rover.id, float(t), conf, float(rng_m))
        fleet.map.mark_avoid(x, y, keep)
        fleet.invalidate_costs()
        fleet.frontiers_t = -1e9
        fleet.refresh_reach()
        for r in fleet.rovers:
            r.plan_t = -1e9
        fleet.log(t, "info", f"{rover.id} sees possible plant life: {WORDS[cls]} (camera {conf:.2f}, "
                  f"{rng_m:.1f} m). Keeping {rover.cfg.observe_min_dist:.0f} m away.", rover.id, at=(x, y))

    # ---------------------------------------------------------------- messages
    def stats(self) -> dict:
        if not self.zones:
            return {}
        return {"flora_zones": [[round(z[0], 2), round(z[1], 2), round(z[2], 2)] for z in self.zones.values()],
                "flora_seen": len(self.zones)}

    def features(self) -> list:
        return [p.to_msg() for p in self.plants]
