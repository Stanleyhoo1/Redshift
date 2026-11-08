"""Onboard object detection (PROTOCOL.md v8, "Object detection (camera perception)").

Each rover runs a small classical detector on its own mast camera frames: RGB and the depth buffer of
the same render, plus the camera's own pose and intrinsics (what a real rover knows about its camera).
Never MuJoCo segmentation, geom ids or any ground truth: those are only used OFFLINE to measure how good
it is (evaluate(), GET /api/vision/stats, scripts/vision_eval.py).

Pipeline (about 10 ms per frame at 320 x 240 on one CPU core, numpy only):
  1. Geometry. Every pixel becomes a 3D point (depth along its ray). Surface normals come from the
     neighbouring points (the smaller step on each side, so they never straddle a silhouette). Steep
     pixels (surface steeper than about 60 deg) are things standing on the ground: rocks, boulders,
     spires, creatures, rovers, the sides of crystal outcrops.
  2. Colour. Hue, saturation and value per pixel, with brightness divided out, sorted into a few
     material groups designed around the sim's materials (visuals.py): liquid blue, pale ice blue,
     violet crystal, teal green (grazers), cyan, lime and olive (organic traces), saturated orange
     (vent rims, the lander's ring, orange paint), very dark (tyres, vent mouths), bright neutral grey
     (the lander pad and metal), and grey-brown basalt.
  3. Blobs. Steep pixels are grouped where neighbouring 3D points touch; colour groups where the depth
     is continuous. Overlapping blobs at the same depth are one object.
  4. Measures from depth: distance and bearing from the rover, width and height above the local ground
     in metres, and how flat it is.
  5. Scores. Each class has a simple rule score from colour fractions, colour strength (saturation,
     which haze and dust wash out), size and shape; the best class wins and its confidence also falls
     with fewer pixels. So distance and dust storms lower the confidence on their own: the colours fade
     toward the haze and the object covers fewer pixels.

Classes: rock, boulder, spire, creature, water, mineral, ice, vent, trace, rover, lander.

The camera thread (camera.py) runs it off the control loop: on every streamed fpv frame and on a small
render per rover (camera.VISION_HZ). publish() hands the result to the rover's VisionBrain, and the
rover's decision cycle only reads the latest result (inside the 40 ms brain budget):
  - water: a LiDAR hole is confirmed as liquid when the camera sees water there
  - lifeforms: a LiDAR track is confirmed as a creature when the camera sees one where the track is
  - new obstacles: the class (rock, boulder or a fallen spire) of what changed in the map, and room
  - science targets seen beyond the spectrometer's 8 m: the rover may go to check (cooldown, limits)
Each decision is logged with the detection, for example "Camera: possible vent 12 m ahead (0.71).
Going to check it." Gemma's anomaly check gets the detections as extra readings (advisor.py).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

LABELS = ("rock", "boulder", "spire", "creature", "water", "mineral", "ice", "vent", "trace", "rover", "lander")
TARGET_LABELS = ("mineral", "ice", "vent", "trace")
ENABLED = os.environ.get("VISION", "1").strip().lower() not in ("0", "off", "false", "no")

WORK_W = 320               # wider frames are reduced by an integer factor before detection
MAX_RANGE_M = 40.0         # nothing is detected farther than this
SKY_M = 250.0              # depth beyond this is sky
CONF_MIN = 0.3             # weaker detections are dropped
CONF_REPORT = 0.5          # precision and recall are measured at this confidence
MAX_OBJECTS = 16
STEEP_NZ = 0.5             # surface normal z below this: steeper than 60 deg, standing on the ground


# ====================================================================== result
@dataclass
class VisionResult:
    frame: int
    t: float
    view: str
    ms: float
    objects: list                  # protocol objects plus "x", "y" (world estimate) for the rover's rules
    source: str = "detector"       # detector (the rover's own small render), stream (the UI's fpv view), gemma
    wall: float = 0.0              # time.monotonic() when it was made
    rover: str = ""
    dust: float = 0.0

    def public_objects(self) -> list:
        return [{k: o[k] for k in ("label", "conf", "box", "dist", "bearing")} for o in self.objects]

    def msg(self) -> dict:
        """Rover `vision` in ticks (protocol v8)."""
        return {"frame": int(self.frame), "t": round(float(self.t), 1), "view": self.view,
                "ms": round(float(self.ms), 1), "objects": self.public_objects()}

    def header(self) -> str:
        """X-Detections: compact JSON, same shape as vision.objects."""
        return json.dumps(self.public_objects(), separators=(",", ":"))


# ====================================================================== colour groups
# Measured on the sim's own renders (hue in degrees, saturation and value 0..1; near pixels, warm low sun,
# with and without shadows): soil hue 20-27, saturation 0.46-0.65 (never above 0.66), value 0.46-0.81;
# basalt rocks and outcrop blocks hue 15-30, saturation 0.25-0.47, value 0.1-0.55; spire crystal hue
# 270-276, saturation 0.6-0.72; mineral veins hue 274-300, saturation 0.17-0.42, value 0.64-1; vent rim
# and the lander ring hue 19-32, saturation 0.70-0.76, value 1; vent crust saturation 0.67-0.69 and mouth
# value 0.03; lander pad and metal hue 16-24, saturation 0.17-0.26, value 0.4-0.95; grazers hue 141-158;
# cyan paint hue 166-168; ice hue 196-240 at saturation 0.04-0.18; trace film hue 61-66; liquid hue
# 217-222 at saturation 0.5-0.86, value 0.31-0.37. Haze shifts colours toward the horizon's apricot:
# grazers 20 m away read hue 125-135 at saturation 0.3, spires at 17 m saturation 0.44.
GROUPS = ("blue", "pale", "violet", "lavender", "teal", "cyan", "olive", "lime", "hot", "crust", "dark", "pad",
          "basalt")
BIT = {g: 1 << k for k, g in enumerate(GROUPS)}
BLOB_GROUPS = ("blue", "pale", "violet", "lavender", "teal", "cyan", "olive", "lime", "hot", "crust", "pad")
BLOB_MIN_PX = {"blue": 6, "pale": 3, "violet": 3, "lavender": 2, "teal": 4, "cyan": 4, "olive": 3, "lime": 2,
               "hot": 2, "crust": 3, "pad": 20}


def _hsv(rgb: np.ndarray):
    """Hue (deg), saturation, value (0..1) per pixel (rgb 0..255, any shape (..., 3))."""
    c = rgb.astype(np.float32) * np.float32(1.0 / 255.0)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    v = c.max(axis=-1)
    mn = c.min(axis=-1)
    ch = v - mn
    s = ch / np.maximum(v, 1e-6)
    safe = np.maximum(ch, 1e-6)
    hue = np.where(v == r, ((g - b) / safe) % 6.0, np.where(v == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    hue[ch < 1e-6] = 0.0
    return hue, s, v


def _in_hue(hue, lo, hi):
    return (hue >= lo) & (hue <= hi) if lo <= hi else (hue >= lo) | (hue <= hi)


def group_bits(hue, sat, val) -> np.ndarray:
    """Material colour groups as bit flags (uint16), designed around visuals.py (see the table above)."""
    out = np.zeros(np.shape(hue), np.uint16)

    def put(name, m):
        out[m] |= BIT[name]
    put("blue", _in_hue(hue, 203, 238) & (sat >= 0.42) & (val >= 0.14))
    put("pale", _in_hue(hue, 188, 252) & (sat >= 0.025) & (sat < 0.32) & (val >= 0.33))
    put("violet", _in_hue(hue, 258, 318) & (sat >= 0.4) & (val >= 0.18))
    put("lavender", _in_hue(hue, 262, 318) & (sat >= 0.11) & (sat < 0.4) & (val >= 0.42))
    put("teal", _in_hue(hue, 105, 163) & (sat >= 0.2) & (val >= 0.08))      # far grazers: hue 125-135, sat 0.3
    put("cyan", _in_hue(hue, 162, 192) & (sat >= 0.5) & (val >= 0.2))
    put("olive", _in_hue(hue, 46, 82) & (sat >= 0.42) & (val >= 0.16) & (val < 0.6))
    put("lime", _in_hue(hue, 74, 120) & (sat >= 0.55) & (val >= 0.42))
    put("hot", _in_hue(hue, 6, 40) & (sat >= 0.69) & (val >= 0.55))
    put("crust", _in_hue(hue, 17, 31) & (sat >= 0.665) & (sat < 0.69) & (val >= 0.4))
    put("dark", val < 0.13)
    put("pad", _in_hue(hue, 4, 42) & (sat >= 0.09) & (sat <= 0.29) & (val >= 0.5))
    put("basalt", _in_hue(hue, 4, 42) & (sat >= 0.16) & (sat < 0.5) & (val >= 0.07) & (val < 0.56))
    return out


def _build_lut():
    q = (np.arange(64, dtype=np.float32) * 4.0 + 2.0)
    r, g, b = np.meshgrid(q, q, q, indexing="ij")
    rgb = np.stack([r, g, b], -1).reshape(-1, 3)
    hue, sat, val = _hsv(rgb)
    return (group_bits(hue, sat, val), np.rint(sat * 255).astype(np.uint8), np.rint(val * 255).astype(np.uint8),
            np.rint(hue / 2.0).astype(np.uint8))


_LUT = None


def colour_codes(rgb: np.ndarray):
    """(group bits, saturation 0..255, value 0..255, hue / 2) per pixel through a 6 bit per channel table."""
    global _LUT
    if _LUT is None:
        _LUT = _build_lut()
    c = rgb if rgb.dtype == np.uint8 else np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    c = (c >> 2).astype(np.int32)
    idx = (c[..., 0] << 12) | (c[..., 1] << 6) | c[..., 2]
    return _LUT[0][idx], _LUT[1][idx], _LUT[2][idx], _LUT[3][idx]


def _reduce(rgb: np.ndarray, dep: np.ndarray, k: int):
    """Integer downscale: colours averaged over k x k blocks, depth sampled at the block centre."""
    if k <= 1:
        return rgb, dep
    h, w = dep.shape
    H, W = h // k, w // k
    blocks = rgb[:H * k, :W * k].reshape(H, k, W, k, 3)
    if rgb.dtype == np.uint8 and k <= 15:          # integer block sums, one strided add per sample: fast
        acc = np.zeros((H, W, 3), np.uint16)
        for i in range(k):
            for j in range(k):
                acc += blocks[:, i, :, j]
        small = ((acc + (k * k) // 2) // (k * k)).astype(np.uint8)
    else:
        small = blocks.mean(axis=(1, 3)).astype(np.float32)
    return small, np.ascontiguousarray(dep[k // 2:H * k:k, k // 2:W * k:k][:H, :W])


def label_components(mask: np.ndarray, conn_h: np.ndarray, conn_v: np.ndarray) -> np.ndarray:
    """Connected components of mask, where conn_h[y, x] says (y, x) and (y, x + 1) touch and conn_v[y, x]
    says (y, x) and (y + 1, x) touch. Returns int64 labels (-1 outside the mask). Horizontal runs first
    (vectorised), then union-find over the runs that touch vertically (min propagation with pointer
    jumping on the small run graph): about 1 ms on a 320 x 240 frame."""
    H, W = mask.shape
    m = mask
    start = m.copy()
    start[:, 1:] &= ~(m[:, :-1] & conn_h)
    rid = np.cumsum(start.reshape(-1)) - 1
    rid = rid.reshape(H, W)
    nr = int(start.sum())
    out = np.full((H, W), -1, dtype=np.int64)
    if nr == 0:
        return out
    ev = m[:-1] & m[1:] & conn_v
    lab = np.arange(nr, dtype=np.int64)
    if ev.any():
        key = np.unique(rid[:-1][ev] * nr + rid[1:][ev])
        a, b = key // nr, key % nr
        for _ in range(64):
            mn = np.minimum(lab[a], lab[b])
            new = lab.copy()
            np.minimum.at(new, a, mn)
            np.minimum.at(new, b, mn)
            new = new[new]
            new = new[new]
            if np.array_equal(new, lab):
                break
            lab = new
    out[m] = lab[rid[m]]
    return out


# ====================================================================== the detector
@dataclass
class _Blob:
    idx: np.ndarray                # flat pixel indices
    kind: str                      # "steep" or a colour group name
    depth: float
    box: tuple = (0, 0, 0, 0)
    xy: np.ndarray | None = None   # world x, y (median of its points)
    split: bool = False            # a piece of a boulder wall (never glued back to its neighbours)
    teal: bool = False             # a standing blob that is mostly grazer green


def _blobs_from(lab: np.ndarray, kinds, dep_flat: np.ndarray, W: int, min_px) -> list:
    """Blobs of a label image. kinds: a name for all, or an array of group index per pixel."""
    flat = lab.reshape(-1)
    sel = np.flatnonzero(flat >= 0)
    if not len(sel):
        return []
    labs = flat[sel]
    order = np.argsort(labs, kind="stable")
    sel, labs = sel[order], labs[order]
    cuts = np.flatnonzero(np.diff(labs)) + 1
    out = []
    for part in np.split(sel, cuts):
        kind = kinds if isinstance(kinds, str) else BLOB_GROUPS[int(kinds[part[0]])]
        need = min_px if isinstance(min_px, int) else min_px[kind]
        if len(part) >= need:
            ys, xs = part // W, part % W
            out.append(_Blob(part, kind, float(np.median(dep_flat[part])),
                             (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    return out


def _split_walls(blobs: list, dist: np.ndarray, W: int, pix: float, bits: np.ndarray) -> list:
    """Rocks that touch (a rockfall's boulder wall) form one wide steep blob: split it where its top
    silhouette (every pixel at its depth, the not-steep caps too) dips between two domes. Only rock
    coloured blobs (a grazer seen side-on is wide too, with dips between its legs)."""
    out = []
    rockish = BIT["basalt"] | BIT["dark"]
    bflat = bits.reshape(-1)
    for b in blobs:
        x0, y0, x1, y1 = b.box
        wid_m = (x1 - x0 + 1) * b.depth * pix
        if wid_m < 1.5 or x1 - x0 < 8 or len(b.idx) < 60 \
                or np.count_nonzero(bflat[b.idx] & rockish) < 0.6 * len(b.idx):
            out.append(b)
            continue
        ys, xs = b.idx // W, b.idx % W
        up = int(min(y0, max(3, 0.6 / max(b.depth * pix, 1e-3))))
        win = dist[y0 - up:y1 + 1, x0:x1 + 1]
        near = np.abs(win - b.depth) < 0.5 + 0.1 * b.depth
        rows = np.arange(y0 - up, y1 + 1)[:, None]
        top = np.where(near, rows, y1 + 1).min(axis=0).astype(np.float64)
        k = max(1, (x1 - x0) // 24)
        sm = np.convolve(top, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
        hpx = max(1.0, y1 - sm.min())
        cuts = []
        n = len(sm)
        for c in range(3, n - 3):
            lo, hi = max(0, c - 4 * k - 3), min(n, c + 4 * k + 4)
            if sm[c] < sm[lo:hi].max():
                continue                   # not the deepest dip around
            left, right = sm[:c].min(), sm[c + 1:].min()
            if sm[c] - max(left, right) > 0.18 * hpx and (not cuts or c - cuts[-1] > 2 * k + 3):
                cuts.append(c)
        edges = [0] + cuts + [n]
        avg = n / max(1, len(edges) - 1)
        while len(edges) > 2:              # no slivers: a piece under 40% of the average width joins its neighbour
            widths = np.diff(edges)
            i = int(np.argmin(widths))
            if widths[i] >= 0.4 * avg:
                break
            if i == 0:
                edges.pop(1)                                   # the first piece joins the second
            elif i == len(widths) - 1:
                edges.pop(i)                                   # the last joins the one before
            else:
                edges.pop(i if widths[i - 1] < widths[i + 1] else i + 1)   # the narrower neighbour
        if len(edges) <= 2:
            out.append(b)
            continue
        for e0, e1 in zip(edges, edges[1:]):
            sel = (xs - x0 >= e0) & (xs - x0 < e1)
            if sel.sum() >= 4:
                part = b.idx[sel]
                pys, pxs = part // W, part % W
                out.append(_Blob(part, b.kind, b.depth,
                                 (int(pxs.min()), int(pys.min()), int(pxs.max()), int(pys.max())), split=True))
    return out


def _overlap(a, b, pad: int = 1) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _inside(a, b, pad: int = 2) -> float:
    """Share of box a's area inside box b (padded)."""
    ix = max(0, min(a[2], b[2] + pad) - max(a[0], b[0] - pad) + 1)
    iy = max(0, min(a[3], b[3] + pad) - max(a[1], b[1] - pad) + 1)
    return ix * iy / max(1, (a[2] - a[0] + 1) * (a[3] - a[1] + 1))


def detect(rgb: np.ndarray, depth: np.ndarray, spec, rover_pos, rover_yaw: float, dust: float = 0.0,
           debug: dict | None = None) -> list:
    """Objects in one frame (RGB uint8 or float, depth in metres along the view axis, same size).
    spec: camera.CamSpec of the frame (eye, forward, fovy). Returns dicts with label, conf, box
    (normalised x0, y0, x1, y1), dist (m, from the rover), bearing (deg, positive to the left), and x, y
    (world estimate, for the rover's own rules). dust is not used: a dust storm lowers the confidence
    through the picture itself."""
    H, W = depth.shape
    f, right, up = spec.basis()
    tv = math.tan(math.radians(spec.fovy) / 2.0)
    th = tv * W / H
    u = ((np.arange(W, dtype=np.float32) + 0.5) / W * 2.0 - 1.0) * np.float32(th)
    v = (1.0 - (np.arange(H, dtype=np.float32) + 0.5) / H * 2.0) * np.float32(tv)
    f32, r32, u32 = f.astype(np.float32), right.astype(np.float32), up.astype(np.float32)
    dirs = f32[None, None, :] + u[None, :, None] * r32[None, None, :] + v[:, None, None] * u32[None, None, :]
    dep = depth.astype(np.float32)
    valid = (dep > 0.05) & (dep < SKY_M)
    dmask = np.where(valid, dep, np.float32(0.0))
    P = spec.eye.astype(np.float32)[None, None, :] + dmask[..., None] * dirs          # world points
    dist = dep * np.sqrt((dirs * dirs).sum(-1))
    near = valid & (dist < MAX_RANGE_M)
    pix = 2.0 * tv / H                                                                  # rad per pixel

    # ---- normals from the smaller neighbouring step on each side
    Dx = P[:, 1:] - P[:, :-1]
    Dy = P[1:] - P[:-1]
    nx2 = (Dx * Dx).sum(-1)
    ny2 = (Dy * Dy).sum(-1)
    inf = np.float32(1e9)
    left = np.concatenate([np.full((H, 1), inf, np.float32), nx2], axis=1)
    rght = np.concatenate([nx2, np.full((H, 1), inf, np.float32)], axis=1)
    upn = np.concatenate([np.full((1, W), inf, np.float32), ny2], axis=0)
    dnn = np.concatenate([ny2, np.full((1, W), inf, np.float32)], axis=0)
    dx = np.where((rght < left)[..., None], np.concatenate([Dx, Dx[:, -1:]], axis=1),
                  np.concatenate([Dx[:, :1], Dx], axis=1))
    dy = np.where((dnn < upn)[..., None], np.concatenate([Dy, Dy[-1:]], axis=0), np.concatenate([Dy[:1], Dy], axis=0))
    n = np.cross(dx, dy)
    nz = np.abs(n[..., 2]) / np.maximum(np.sqrt((n * n).sum(-1)), 1e-9)
    step_ok = np.sqrt(np.minimum(left, rght)) < np.maximum(0.12, 4.0 * dist * pix)
    steep = near & (nz < STEEP_NZ) & step_ok

    # ---- connectivity: steep pixels where 3D points touch; colour pixels of one group where depth is continuous
    tol = np.maximum(0.15, 3.0 * dist * pix)
    touch_h = np.sqrt(nx2) < np.minimum(tol[:, :-1], tol[:, 1:])
    touch_v = np.sqrt(ny2) < np.minimum(tol[:-1, :], tol[1:, :])
    dflat = dist.reshape(-1)
    bits, sat8, val8, _hue2 = colour_codes(rgb)
    blobs = _split_walls(_blobs_from(label_components(steep, touch_h, touch_v), "steep", dflat, W, 4),
                         np.where(valid, dist, np.float32(1e6)), W, pix, bits)
    code = np.full((H, W), -1, np.int16)              # one blob group per pixel (first match in BLOB_GROUPS)
    for k in range(len(BLOB_GROUPS) - 1, -1, -1):
        code[(bits & BIT[BLOB_GROUPS[k]]) != 0] = k
    cmask = (code >= 0) & near
    if cmask.any():
        rel_h = (np.abs(dep[:, 1:] - dep[:, :-1]) < 0.3 + 0.2 * np.minimum(dep[:, 1:], dep[:, :-1])) \
            & (code[:, 1:] == code[:, :-1])
        rel_v = (np.abs(dep[1:] - dep[:-1]) < 0.3 + 0.2 * np.minimum(dep[1:], dep[:-1])) & (code[1:] == code[:-1])
        blobs += _blobs_from(label_components(cmask, rel_h, rel_v), code.reshape(-1), dflat, W, BLOB_MIN_PX)

    # ---- merge: colour blobs into the steep blob they lie on; colour blobs with each other where they overlap
    nb = len(blobs)
    parent = list(range(nb))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    Pf = P.reshape(-1, 3)
    teal_bit = BIT["teal"]
    bflat0 = bits.reshape(-1)
    for bl in blobs:
        bl.xy = np.median(Pf[bl.idx, :2], axis=0)
        bl.teal = bl.kind == "steep" and np.count_nonzero(bflat0[bl.idx] & teal_bit) > 0.5 * len(bl.idx)
    vent_kinds = {"hot", "crust"}
    for a in range(nb):
        A = blobs[a]
        for b in range(a + 1, nb):
            B = blobs[b]
            if A.kind in vent_kinds and B.kind in vent_kinds and float(np.hypot(*(A.xy - B.xy))) < 0.7:
                parent[root(b)] = root(a)      # a vent's rim, crust and stains: one vent
                continue
            if not _overlap(A.box, B.box, 2):
                continue
            if abs(A.depth - B.depth) > 0.6 + 0.15 * min(A.depth, B.depth):
                continue
            if A.kind == "steep" and B.kind == "steep":
                # parts of one thing (a rover's mast over its chassis) stand on the same spot
                gap = float(np.hypot(*(A.xy - B.xy)))
                if A.split or B.split or gap > (0.75 if (A.teal and B.teal) else 0.6):
                    continue          # (a grazer's legs and body segments stand within 0.75 m of each other)
            elif A.kind == "steep" or B.kind == "steep":
                S_, C_ = (A, B) if A.kind == "steep" else (B, A)
                if _inside(C_.box, S_.box) < 0.6:
                    continue          # a big flat patch (the lander pad) under something standing on it
            elif {A.kind, B.kind} & {"pad"} and {A.kind, B.kind} - {"pad", "hot"}:
                continue
            parent[root(b)] = root(a)
    members: dict = {}
    for i in range(nb):
        members.setdefault(root(i), []).append(i)

    nzf = nz.reshape(-1)
    steepf = steep.reshape(-1)
    bf, sf, vf = bits.reshape(-1), sat8.reshape(-1), val8.reshape(-1)
    validf = valid.reshape(-1)
    r64 = right.astype(np.float32)
    rx, ry = float(rover_pos[0]), float(rover_pos[1])
    out = []
    for mem in members.values():
        idx = blobs[mem[0]].idx if len(mem) == 1 else np.unique(np.concatenate([blobs[i].idx for i in mem]))
        kinds = {blobs[i].kind for i in mem}
        ys, xs = idx // W, idx % W
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        npx = len(idx)
        pts = Pf[idx]
        d_med = float(np.median(dflat[idx]))
        # ground height under it: ground-like pixels just below the box (else its own lowest points)
        gy0, gy1 = min(H - 1, y1 + 1), min(H, y1 + 4)
        band = (np.arange(gy0, gy1)[:, None] * W + np.arange(x0, x1 + 1)[None, :]).reshape(-1)
        band = band[validf[band] & (nzf[band] > 0.75) & (np.abs(dflat[band] - d_med) < 0.6 + 0.15 * d_med)]
        z_obj = pts[:, 2]
        z_low = float(np.percentile(z_obj, 3))
        # the ground cannot be above the object's own lowest point (a ridge in front hides its base)
        z_ground = min(float(np.median(Pf[band, 2])), z_low + 0.05) if len(band) >= 3 else z_low
        lat = pts @ r64
        lo, hi = (np.percentile(lat, (1, 99)) if npx > 60 else (lat.min(), lat.max()))
        width = float(hi - lo) + d_med * pix
        # the top: a rock's cap is not steep, so look a little above the steep part, on the same spot
        cxy = np.median(pts[:, :2], axis=0)
        up_rows = int(min(y0, max(2, 0.4 / max(d_med * pix, 1e-3))))
        top_idx = (np.arange(y0 - up_rows, y1 + 1)[:, None] * W + np.arange(x0, x1 + 1)[None, :]).reshape(-1)
        top_idx = top_idx[validf[top_idx] & (np.abs(dflat[top_idx] - d_med) < 0.4 + 0.08 * d_med)]
        if len(top_idx):
            tp_ = Pf[top_idx]
            on = np.hypot(tp_[:, 0] - cxy[0], tp_[:, 1] - cxy[1]) < 0.5 * width + 0.2
            z_top = max(float(np.percentile(z_obj, 98)), float(tp_[on, 2].max()) if on.any() else -1e9)
            if on.any() and "steep" in kinds:
                cap = top_idx[on & (tp_[:, 2] > z_low + 0.05)]
                if len(cap):                       # the box reaches up to the (not steep) cap
                    y0 = min(y0, int((cap // W).min()))
        else:
            z_top = float(np.percentile(z_obj, 98))
        height = z_top - z_ground
        ob = bf[idx]
        fr = {g: float(np.count_nonzero(ob & BIT[g])) / npx for g in GROUPS}
        ss = sf[idx]
        feats = {"npx": npx, "dist": d_med, "height": height, "width": width, "steep": float(steepf[idx].mean()),
                 "kinds": kinds, "box": (x0, y0, x1, y1), "pix": pix, "W": W, "H": H}
        for g in ("violet", "teal", "blue", "hot", "cyan", "lavender", "pale", "olive"):
            sel = (ob & BIT[g]) != 0
            feats["sat_" + g] = float(ss[sel].mean()) / 255.0 if sel.any() else 0.0
        if "hot" in kinds or "crust" in kinds:      # a vent's dark mouth sits right next to its rim
            wy0, wy1, wx0, wx1 = max(0, y0 - 2), min(H, y1 + 3), max(0, x0 - 2), min(W, x1 + 3)
            win = bits[wy0:wy1, wx0:wx1]
            feats["dark_near"] = float(np.count_nonzero(win & BIT["dark"])) / max(1, win.size)
        label, conf, extra_box = _classify(fr, feats)
        if debug is not None:
            debug.setdefault("all", []).append({"label": label, "conf": conf, "fr": fr, "feats": feats})
        if label is None or conf < CONF_MIN:
            continue
        if extra_box is not None:
            x0, y0, x1, y1 = extra_box
        cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        dxy = math.hypot(cx - rx, cy - ry)
        bearing = math.degrees(math.remainder(math.atan2(cy - ry, cx - rx) - rover_yaw, 2 * math.pi))
        out.append({"label": label, "conf": round(conf, 2),
                    "box": [round(x0 / W, 3), round(y0 / H, 3), round((x1 + 1) / W, 3), round((y1 + 1) / H, 3)],
                    "dist": round(dxy, 1), "bearing": round(bearing, 1), "x": round(cx, 2), "y": round(cy, 2),
                    "height": round(height, 2), "width": round(width, 2), "standing": "steep" in kinds})
    # one object, one detection: a weaker detection mostly inside a stronger one of the same label goes, and
    # the lander's own legs and pad edge are not rocks
    out.sort(key=lambda o: (-o["conf"], -(o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])))
    landers = [o["box"] for o in out if o["label"] == "lander"]
    lakes = [o["box"] for o in out if o["label"] == "water"]
    grazers = [o["box"] for o in out if o["label"] == "creature" and o["standing"]]
    kept = []
    for o in out:
        b = o["box"]
        area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        dup = False
        for k in kept:
            if k["label"] != o["label"]:
                continue
            kb = k["box"]
            ix = max(0.0, min(b[2], kb[2]) - max(b[0], kb[0])) * max(0.0, min(b[3], kb[3]) - max(b[1], kb[1]))
            if ix > 0.5 * area:
                dup = True
                break
        if not dup and o["label"] in ("rock", "boulder", "vent"):
            for lb in landers:
                ph = 0.4 * (lb[3] - lb[1]) + 0.01
                pw = 0.08 * (lb[2] - lb[0]) + 0.01
                ix = max(0.0, min(b[2], lb[2] + pw) - max(b[0], lb[0] - pw))
                iy = max(0.0, min(b[3], lb[3] + ph) - max(b[1], lb[1] - ph))
                if ix * iy > 0.5 * area:
                    dup = True
                    break
        if not dup and o["label"] == "creature" and not o["standing"]:
            # a green outline (maybe around several animals) where standing bodies were found: keep the bodies
            dup = any(_box_iou(b, gb) > 0.1 or (gb[0] <= 0.5 * (b[0] + b[2]) <= gb[2]) and gb[1] <= 0.5 * (b[1] + b[3]) <= gb[3]
                      or (b[0] <= 0.5 * (gb[0] + gb[2]) <= b[2] and b[1] <= 0.5 * (gb[1] + gb[3]) <= b[3])
                      for gb in grazers)
        if not dup and o["label"] == "ice":       # pale glints along a lake's shore are the lake
            for lb in lakes:
                ph, pw = 0.3 * (lb[3] - lb[1]) + 0.02, 0.05 * (lb[2] - lb[0]) + 0.02
                if lb[0] - pw <= 0.5 * (b[0] + b[2]) <= lb[2] + pw and lb[1] - ph <= 0.5 * (b[1] + b[3]) <= lb[3] + ph:
                    dup = True
                    break
        if not dup:
            kept.append(o)
    if debug is not None:
        debug["steep"] = steep
        debug["bits"] = bits
    return kept[:MAX_OBJECTS]


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi, linear in between (lo > hi gives a falling ramp)."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _classify(fr: dict, ft: dict):
    """(label, confidence, box override or None) for one merged object: a rule score per class."""
    npx, hgt, wid = ft["npx"], ft["height"], ft["width"]
    steep = ft["steep"]
    kinds = ft["kinds"]
    vis = _ramp(npx, 3, 40)                        # fewer pixels, less sure
    flat = hgt < 0.3
    stand = "steep" in kinds
    s = {}
    # water: liquid blue, flat, a sizeable patch
    # (depth under a semi-transparent surface comes from the basin floor, so no height rule)
    # a far lake seen at a grazing angle mirrors the sky and turns pale: wide pale patches are water too
    wide = _ramp(wid, 1.4, 2.2)
    s["water"] = max(_ramp(fr["blue"], 0.25, 0.65) * _ramp(ft["sat_blue"], 0.4, 0.7),
                     _ramp(fr["blue"] + fr["pale"], 0.4, 0.8) * wide * 0.8) \
        * (1.0 - _ramp(steep, 0.3, 0.6)) * _ramp(npx, 4, 30)
    # ice: a pale blue-white patch on the ground, under about a metre across
    rows = ft["box"][3] - ft["box"][1] + 1
    s["ice"] = _ramp(fr["pale"], 0.25, 0.6) * (1.0 if flat else 0.4) * (1.0 - _ramp(fr["blue"], 0.2, 0.5)) \
        * (1.0 - wide) * (1.0 if rows >= 2 or npx >= 10 else 0.4)     # a one-row pale sliver is a shore glint
    # trace: olive film with lime specks, flat
    s["trace"] = _ramp(fr["olive"] + 1.5 * fr["lime"], 0.3, 0.7) * (1.0 if flat else 0.3) \
        * (1.0 - _ramp(fr["teal"], 0.15, 0.35))
    # vent: a small hot orange rim and crust with a dark mouth, on a low mound
    vent_c = fr["hot"] + fr["crust"]
    hot_px = fr["hot"] * npx                       # the rim itself, hotter than any soil (sunlit slopes reach crust)
    s["vent"] = _ramp(vent_c, 0.3, 0.7) * _ramp(hot_px, 0.5, 3.0) * (1.0 if hgt < 0.55 else 0.15) \
        * (1.0 if wid < 1.4 else 0.2) \
        * (0.6 + 0.4 * _ramp(ft.get("dark_near", 0.0), 0.0, 0.05)) * (1.0 - _ramp(fr["pad"], 0.1, 0.3)) \
        * (0.5 if stand and steep > 0.5 else 1.0)
    # lander: a big flat warm grey pad (with its orange ring)
    s["lander"] = _ramp(fr["pad"] + 0.8 * fr["hot"], 0.4, 0.8) * _ramp(wid, 1.0, 2.2) * (1.0 if hgt < 0.8 else 0.4) \
        * (0.6 if stand and steep > 0.5 else 1.0)
    # spire: saturated violet crystal, tall (or long, fallen)
    s["spire"] = _ramp(fr["violet"], 0.3, 0.65) * max(_ramp(hgt, 0.7, 1.3), _ramp(wid, 1.2, 2.0) * (hgt > 0.2)) \
        * (1.0 - 0.7 * _ramp(fr["dark"], 0.05, 0.15))
    # mineral: pale violet crystals on dark grey-brown blocks, small
    s["mineral"] = _ramp(fr["lavender"] + 0.6 * fr["violet"], 0.02, 0.15) \
        * _ramp(fr["basalt"] + fr["dark"] + fr["lavender"], 0.4, 0.75) * (1.0 if hgt < 0.8 else 0.2) \
        * (1.0 if wid < 1.6 else 0.3) * (1.0 - _ramp(fr["violet"], 0.3, 0.6))
    # creature: teal green body, about knee high
    s["creature"] = _ramp(fr["teal"], 0.2, 0.5) * _ramp(ft["sat_teal"], 0.22, 0.45) * _ramp(hgt, 0.15, 0.4) \
        * (1.0 if hgt < 1.3 else 0.3)
    # rover: about 1 m tall (its mast), dark tyres and deck, paint and light metal
    paint = max(fr["hot"], fr["cyan"], fr["violet"], fr["lime"], fr["blue"])
    s["rover"] = _ramp(hgt, 0.5, 0.8) * (1.0 if hgt < 1.6 else 0.2) * _ramp(fr["dark"] + 0.5 * fr["basalt"], 0.05, 0.2) \
        * _ramp(paint + fr["pad"], 0.08, 0.3) * (1.0 if wid < 2.0 else 0.3) * (1.0 - _ramp(fr["teal"], 0.2, 0.45)) \
        * (1.0 - _ramp(fr["violet"], 0.6, 0.85)) * stand
    # rock / boulder: something standing on the ground, grey-brown, no telling colour
    plain = 1.0 - max(_ramp(fr["violet"], 0.03, 0.12), _ramp(fr["lavender"], 0.02, 0.08), _ramp(fr["teal"], 0.05, 0.2),
                      _ramp(fr["hot"], 0.03, 0.1), _ramp(fr["blue"], 0.05, 0.2), _ramp(fr["cyan"], 0.05, 0.2),
                      _ramp(fr["pad"], 0.12, 0.3), _ramp(fr["pale"], 0.1, 0.3))
    solid = _ramp(steep, 0.3, 0.6) * stand * _ramp(fr["basalt"] + fr["dark"], 0.3, 0.65) * plain
    # rockfall boulders rest half buried like the terrain rocks: 1.0-1.3 m wide, 0.57-0.75 m tall; terrain
    # rocks are at most 1.2 m wide and 0.65 m tall, mostly much smaller. "boulder" means a big rock.
    big = _ramp(wid, 0.82, 0.97) * _ramp(hgt, 0.42, 0.52)
    s["boulder"] = solid * big
    s["rock"] = solid * (1.0 - big) * _ramp(hgt, 0.06, 0.15)
    label = max(s, key=s.get)
    score = s[label]
    if score <= 0.0:
        return None, 0.0, None
    second = max(v for k, v in s.items() if k != label)
    conf = score * (0.55 + 0.45 * vis) * (1.0 - 0.35 * _ramp(second / max(score, 1e-6), 0.6, 1.0))
    box = _expand_box(ft, 1.2, 0.3) if label == "vent" else None
    return label, float(min(0.99, conf)), box


def _expand_box(ft: dict, width_m: float, height_m: float):
    """Box of an object of known physical size centred on the detected part (a vent's small orange rim
    stands for its whole mound)."""
    x0, y0, x1, y1 = ft["box"]
    W, H, d, pix = ft["W"], ft["H"], max(ft["dist"], 0.5), ft["pix"]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw = max(0.5 * (x1 - x0), 0.5 * width_m / (d * pix))
    hh = max(0.5 * (y1 - y0), 0.5 * height_m / (d * pix))
    return (int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(W - 1, cx + hw)), int(min(H - 1, cy + hh)))


def detect_frame(fr, source: str = "detector") -> VisionResult:
    """Run the detector on a camera.Frame rendered with depth (any size: wide frames are reduced)."""
    t0 = time.perf_counter()
    if fr.depth is None:
        raise ValueError("detect_frame needs a frame rendered with depth=True")
    h, w = fr.depth.shape
    k = max(1, int(math.ceil(w / WORK_W)))
    rgb, dep = _reduce(fr.rgb, fr.depth, k)
    objs = detect(rgb, dep, fr.spec, fr.rover_pos, fr.rover_yaw, fr.dust)
    if k > 1:                       # boxes back to the full frame (the reduced image may crop a few pixels)
        sx, sy = (dep.shape[1] * k) / w, (dep.shape[0] * k) / h
        for o in objs:
            b = o["box"]
            o["box"] = [round(b[0] * sx, 3), round(b[1] * sy, 3), round(b[2] * sx, 3), round(b[3] * sy, 3)]
    ms = (time.perf_counter() - t0) * 1000.0
    return VisionResult(frame=int(fr.frame), t=float(fr.t), view=fr.view, ms=ms, objects=objs, source=source,
                        wall=time.monotonic(), rover=fr.rover, dust=float(getattr(fr, "dust", 0.0)))


def publish(sim, rover_id: str, res: VisionResult) -> None:
    """Hand a result to the rover's VisionBrain (camera thread; one reference swap, the rover reads it in
    its next decision cycle). Results older than the one it already has are ignored."""
    for r in getattr(sim, "rovers", ()):
        if r.id == rover_id:
            vb = getattr(r, "vision", None)
            if isinstance(vb, VisionBrain):
                vb.publish(res)
            return


def observe(sim, rover, w: int = 256, h: int = 192, shadows: bool = False) -> VisionResult | None:
    """Render the rover's camera in the CALLING thread (caller owns the sim, as for spot_targets), detect
    and publish. For tests and headless runs; the server does this on the camera thread instead."""
    from . import camera
    try:
        fr = camera._local_renderer().frame(None, sim, rover.id, "fpv", w, h, depth=True, smooth=False,
                                            shadows=shadows)
    except camera.RenderUnavailable:
        return None
    res = detect_frame(fr, source="detector")
    publish(sim, rover.id, res)
    return res


# ====================================================================== decisions (the rover's rules)
STALE_S = 4.0              # a result older than this (sim s) is not acted on
OBSTACLE_WAIT_S = 8.0      # a new obstacle waits this long (sim s) for a camera class
OBSTACLE_MATCH_M = 3.0     # a rock, boulder or spire detection this close to it is that obstacle
OBSTACLE_ROOM = {"rock": 0.0, "boulder": 0.5, "spire": 0.8}   # extra room (m) kept around it
TRACK_MATCH_M = 1.8        # a creature detection this close to a LiDAR track's position is that track
CREATURE_CONF = 0.55
CREATURE_NOTE_M, CREATURE_NOTE_S = 15.0, 90.0   # one log line per herd and place in this long (track ids churn)
WATER_MATCH_M = 4.0        # a water detection this close to a LiDAR hole confirms it as liquid
WATER_CONF = 0.5
CHECK_CONF = 0.65          # a science target seen beyond the spectrometer this sure may be checked ...
CHECK_RANGE_M = (8.5, 18.0)    # ... at this distance ...
CHECK_COOLDOWN_S = 60.0    # ... not more often than this per rover (sim s) ...
CHECK_TIMEOUT_S = 45.0     # ... giving up after this long ...
CHECK_SPOT_M = 6.0         # ... and never again near a spot the fleet already checked
CHECK_SHORT_M = 3.5        # drive to this far short of it (inside the spectrometer's 8 m)


def _side(bearing: float) -> str:
    if abs(bearing) <= 25.0:
        return "ahead"
    return "to the left" if bearing > 0 else "to the right"


def _say_det(o: dict) -> str:
    """'possible vent 12 m ahead (0.71)'"""
    word = "possible " if o["label"] in TARGET_LABELS else ""
    return f"{word}{o['label']} {o['dist']:.0f} m {_side(o['bearing'])} ({o['conf']:.2f})"


class VisionBrain:
    """What a rover does with its detector's results. publish() runs on the camera thread (reference swap);
    tick() runs in the rover's decision cycle and only reads the latest result."""

    def __init__(self, rover):
        self.rover = rover
        self.latest: VisionResult | None = None
        self.seen = -1                       # frame number of the last result acted on
        self.cam_goal: dict | None = None    # camera check in progress
        self.next_check = 0.0
        self.obstacles_done: set = set()     # anomaly times already classified
        self.decisions = 0
        self._mu = threading.Lock()

    # ------------------------------------------------------------ input
    def publish(self, res: VisionResult) -> None:
        with self._mu:
            cur = self.latest
            if cur is not None and res.frame <= cur.frame:
                return
            self.latest = res

    def fresh(self, t: float) -> VisionResult | None:
        res = self.latest
        if res is None or t - res.t > STALE_S or res.t > t + 1.0:
            return None
        return res

    def msg(self) -> dict | None:
        res = self.latest
        return res.msg() if res is not None else None

    # ------------------------------------------------------------ decision cycle
    def tick(self, t: float) -> None:
        cg = self.cam_goal
        if cg is not None and (t - cg["t0"] > CHECK_TIMEOUT_S or not self._may_continue()):
            self.cam_goal = None             # busy with something more important, or too long: let it go
            if t - cg["t0"] > CHECK_TIMEOUT_S:
                self._log(t, "info", f"Camera check of the possible {cg['label']} took too long. Back to exploring.",
                          at=(cg["x"], cg["y"]))
        res = self.fresh(t)
        if res is None or res.frame == self.seen:
            return
        self.seen = res.frame
        if not res.objects:
            return
        self._obstacle(t, res)
        self._creatures(t, res)
        self._targets(t, res)

    def _log(self, t: float, level: str, text: str, at=None) -> None:
        self.decisions += 1
        r = self.rover
        r.fleet.log(t, level, f"{r.id} {text}", r.id, at=at)

    def _obstacle(self, t: float, res: VisionResult) -> None:
        """The class of a new obstacle LiDAR change detection just flagged (rock, boulder or fallen spire)."""
        r = self.rover
        a = r.anomaly
        if a is None or a.get("kind") != "new_obstacle" or a["t"] in self.obstacles_done:
            return
        if t - a["t"] > OBSTACLE_WAIT_S or res.t < a["t"] - 0.5:
            return
        ax, ay = a["at"]
        cands = [o for o in res.objects if o["label"] in OBSTACLE_ROOM
                 and math.hypot(o["x"] - ax, o["y"] - ay) < OBSTACLE_MATCH_M]
        if not cands:
            return
        o = max(cands, key=lambda c: c["conf"])
        self.obstacles_done.add(a["t"])
        a["camera"] = {"label": o["label"], "conf": o["conf"]}
        room = OBSTACLE_ROOM[o["label"]]
        f = r.fleet
        if room > 0.0 and all(math.hypot(q.x - o["x"], q.y - o["y"]) > room + 1.2 for q in f.rovers):
            f.map.mark_blocked(o["x"], o["y"], room)
            f.invalidate_costs()
            r.plan_t = -1e9
            what = "a fallen spire" if o["label"] == "spire" else "a boulder"
            tail = f"It is {what}: keeping {room:.1f} m of extra room around it."
        else:
            tail = "A small rock: the mapped cells are enough to go around it."
        self._log(t, "info", f"Camera: {_say_det(o)} where the map changed. {tail}", at=(o["x"], o["y"]))

    def _creatures(self, t: float, res: VisionResult) -> None:
        """A creature in the picture where a LiDAR track moves confirms the track as a lifeform."""
        r = self.rover
        f = r.fleet
        tracker = f.tracker
        if not tracker.tracks:
            return
        for o in res.objects:
            if o["label"] != "creature" or o["conf"] < CREATURE_CONF:
                continue
            for tr in tracker.near(o["x"], o["y"], TRACK_MATCH_M + 1.0, res.t):
                if tr.lifeform:
                    continue
                px, py = tr.predict(res.t)
                if math.hypot(px - o["x"], py - o["y"]) > TRACK_MATCH_M + tr.speed * max(0.0, t - res.t):
                    continue
                f.confirm_lifeform(tr, t)
                if tr.label == "moving object":
                    tr.label = "creature"
                if r.obs is not None and r.obs.get("track") == tr.id:
                    r.obs["life"] = True
                notes = getattr(f, "camera_creatures", None)
                if notes is None:
                    notes = f.camera_creatures = []
                if any(math.hypot(px - nx, py - ny) < CREATURE_NOTE_M and t - nt < CREATURE_NOTE_S
                       for nx, ny, nt in notes):
                    continue                 # this herd was announced a moment ago (a new track id, same animals)
                notes[:] = [n for n in notes if t - n[2] < CREATURE_NOTE_S] + [(px, py, t)]
                self._log(t, "discovery", f"Camera: {_say_det(o)} where LiDAR track {tr.id} moves "
                                          f"({tr.speed:.1f} m/s). Counting it as a lifeform.", at=(px, py))

    def _targets(self, t: float, res: VisionResult) -> None:
        """A science target seen beyond the spectrometer's range: maybe go and check it."""
        r = self.rover
        if self.cam_goal is not None or t < self.next_check or not self._free(t):
            return
        f = r.fleet
        spots = getattr(f, "camera_spots", None)
        if spots is None:
            spots = f.camera_spots = []
        best = None
        for o in res.objects:
            if o["label"] not in TARGET_LABELS or o["conf"] < CHECK_CONF:
                continue
            if not (CHECK_RANGE_M[0] <= o["dist"] <= CHECK_RANGE_M[1]):
                continue
            if any(ts.status is not None and math.hypot(ts.spec.x - o["x"], ts.spec.y - o["y"]) < 3.0
                   for ts in f.targets):
                continue                     # already known to the fleet
            if any(math.hypot(sx - o["x"], sy - o["y"]) < CHECK_SPOT_M for sx, sy, _t, _r in spots):
                continue
            if best is None or o["conf"] > best["conf"]:
                best = o
        if best is None:
            return
        dx, dy = best["x"] - r.x, best["y"] - r.y
        d = math.hypot(dx, dy)
        need = r.energy_for_distance(d + 1.3 * math.hypot(best["x"], best["y"])) + r.battery_margin + 0.1
        if need > r.battery:
            return
        k = max(0.0, d - CHECK_SHORT_M) / max(d, 1e-6)
        goal = (r.x + dx * k, r.y + dy * k)
        spots.append((best["x"], best["y"], t, r.id))
        self.cam_goal = {"x": best["x"], "y": best["y"], "goal": goal, "label": best["label"], "conf": best["conf"],
                         "t0": t, "fails": 0}
        self.next_check = t + CHECK_COOLDOWN_S
        r.goal, r.path = goal, []
        r.sticky_goal = None
        r.plan_t = -1e9
        r.mem.on_goal(t, "camera", f"the camera saw a {_say_det(best)}")
        self._log(t, "info", f"Camera: {_say_det(best)}. Going to check it.", at=(best["x"], best["y"]))

    def _free(self, t: float) -> bool:
        r = self.rover
        return (r.state == "exploring" and not r.held and r.op_goal is None and r.focus is None
                and r.survey is None and r.obs is None and not r.question_hold and not r.degraded
                and r.target_idx is None and r.pending_interest is None and t >= r.pause_until)

    def _may_continue(self) -> bool:
        r = self.rover
        return (r.state == "exploring" and not r.held and r.op_goal is None and r.focus is None
                and r.survey is None and r.obs is None and not r.question_hold)

    def drive_check(self, t: float) -> bool:
        """Called by Rover._explore: drive the camera check. True while it is in charge of the goal."""
        cg = self.cam_goal
        if cg is None:
            return False
        r = self.rover
        f = r.fleet
        from .planner import PENDING
        from .terrain import KIND_LABEL
        found = next((ts for ts in f.targets if ts.status is not None
                      and math.hypot(ts.spec.x - cg["x"], ts.spec.y - cg["y"]) < 3.0), None)
        gx, gy = cg["goal"]
        why = None
        if t - cg["t0"] > CHECK_TIMEOUT_S:
            why = f"camera check of the possible {cg['label']} took too long. Back to exploring."
        elif found is not None:
            same = "" if found.spec.kind == cg["label"] else f" (a {KIND_LABEL[found.spec.kind]}, not a {cg['label']})"
            why = (f"camera check done: the spectrometer confirms {found.spec.id}{same} where the camera saw a "
                   f"possible {cg['label']}.")
        elif not self._may_continue():
            self.cam_goal = None             # something more important took over (quietly)
            return False
        elif math.hypot(gx - r.x, gy - r.y) < 1.5:
            why = (f"camera check at ({cg['x']:.0f}, {cg['y']:.0f}): the spectrometer finds no target there (the "
                   f"camera's {cg['label']} was something else). Back to exploring.")
        elif cg["fails"] >= 2:
            why = f"no safe path to the possible {cg['label']} the camera saw. Back to exploring."
        if why is not None:
            self.cam_goal = None
            r.goal, r.path = None, []
            r.next_goal_try = t
            self._log(t, "info", why[0].upper() + why[1:], at=(cg["x"], cg["y"]))
            return False
        r.goal = (gx, gy)
        if not r._need_replan(t):
            return True
        path = f.plan(t, r, r.goal)
        r._after_replan(t, path)
        if path is PENDING:
            r._hold_if_blocked()
        elif path is None:
            cg["fails"] += 1
            if r.block_info is not None:
                r.path = []
            r.plan_t = t
        else:
            r._set_path(t, path)
        return True

    # ------------------------------------------------------------ questions from other rules
    def water_near(self, at, t: float) -> dict | None:
        """The strongest fresh water detection within WATER_MATCH_M of at, or None."""
        res = self.fresh(t)
        if res is None:
            return None
        cands = [o for o in res.objects if o["label"] == "water" and o["conf"] >= WATER_CONF
                 and math.hypot(o["x"] - at[0], o["y"] - at[1]) < WATER_MATCH_M + 0.25 * o["dist"]]
        return max(cands, key=lambda o: o["conf"]) if cands else None

    def describe_near(self, at, t: float, radius: float = 5.0) -> str | None:
        """Detections near a place as one reading for Gemma: 'boulder 0.78 at 4.2 m, rock 0.55 at 6.0 m'."""
        res = self.fresh(t)
        if res is None:
            return None
        near = [o for o in res.objects if at is None or math.hypot(o["x"] - at[0], o["y"] - at[1]) < radius]
        if not near:
            return "nothing detected there" if res.objects is not None else None
        return ", ".join(f"{o['label']} {o['conf']:.2f} at {o['dist']:.1f} m" for o in near[:4])


# ====================================================================== ground truth (offline only)
_GT_CACHE: dict = {}


def gt_table(model) -> tuple:
    """(label index per geom or -1, instance key per geom) from the model's names. OFFLINE evaluation
    only: the rovers never see this."""
    key = id(model)
    hit = _GT_CACHE.get(key)
    if hit is not None and hit[0] is model:
        return hit[1], hit[2]
    import mujoco
    n = model.ngeom
    cls = np.full(n, -1, dtype=np.int64)
    inst = np.full(n, -1, dtype=np.int64)
    rover_roots = set()
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if len(nm) >= 2 and nm[0] == "r" and nm[1:].isdigit():
            rover_roots.add(b)
    suffix = {"mound": "vent", "crust": "vent", "rim": "vent", "mouth": "vent", "stain": "vent",
              "block": "mineral", "vein": "mineral", "sheet": "ice", "slab": "ice", "film": "trace", "spot": "trace"}
    lakes: dict = {}
    for g in range(n):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        b = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        root = int(model.body_rootid[b])
        lab = None
        ik = g
        if gname.startswith("rock_"):
            lab = "rock"
        elif gname.startswith("boulder_"):
            lab = "boulder"
        elif bname.startswith("spire_"):
            lab, ik = "spire", 100000 + b
        elif bname.startswith("decor_"):
            part = gname.split("_", 1)[1] if "_" in gname else ""
            part = part.rstrip("0123456789")
            lab, ik = suffix.get(part), 100000 + b
        elif bname.startswith("life_"):
            lab, ik = "creature", 100000 + b
        elif gname.startswith("water_"):
            wid = gname.split("_")[1]
            lab, ik = "water", 200000 + lakes.setdefault(wid, len(lakes))
        elif root in rover_roots:
            lab, ik = "rover", 300000 + root
        elif gname.startswith("lander_"):
            lab, ik = "lander", 400000
        if lab is not None and lab in LABELS:
            cls[g] = LABELS.index(lab)
            inst[g] = ik
    _GT_CACHE[key] = (model, cls, inst)
    return cls, inst


def gt_instances(seg: np.ndarray, model, depth: np.ndarray | None = None, max_dist: float = 30.0) -> list:
    """Ground truth objects in a segmentation frame: label, pixel box, pixel count, median depth, and
    whether it counts for recall (big and near enough to be seen at all)."""
    cls, inst = gt_table(model)
    H, W = seg.shape
    ok = seg >= 0
    ids = np.where(ok, seg, 0)
    c = np.where(ok, cls[ids], -1)
    k = np.where(ok, inst[ids], -1)
    sel = np.flatnonzero(c.reshape(-1) >= 0)
    if not len(sel):
        return []
    kf = k.reshape(-1)[sel]
    order = np.argsort(kf, kind="stable")
    sel, kf = sel[order], kf[order]
    cuts = np.flatnonzero(np.diff(kf)) + 1
    min_px = max(8, int(round(20 * H * W / (320 * 240))))
    out = []
    for part in np.split(sel, cuts):
        lab = LABELS[int(c.reshape(-1)[part[0]])]
        ys, xs = part // W, part % W
        dmed = float(np.median(depth.reshape(-1)[part])) if depth is not None else 0.0
        out.append({"label": lab, "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    "npx": int(len(part)), "dist": dmed,
                    "counts": len(part) >= min_px and (depth is None or dmed <= max_dist)})
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(objects: list, insts: list, W: int, H: int, conf_min: float = CONF_REPORT) -> dict:
    """Match detections to ground truth. A detection is right if it has the label of a ground truth object
    and either overlaps it (IoU >= 0.3) or its box centre lies inside that object's box (padded 10%).
    Detections on objects too small or far to count are neither right nor wrong."""
    res = {"tp": [], "fp": [], "fn": [], "confusion": []}
    gts = [dict(g, box=(g["box"][0], g["box"][1], g["box"][2] + 1, g["box"][3] + 1), used=False) for g in insts]
    dets = sorted([o for o in objects if o["conf"] >= conf_min], key=lambda o: -o["conf"])
    for o in dets:
        b = (o["box"][0] * W, o["box"][1] * H, o["box"][2] * W, o["box"][3] * H)
        cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        best, best_s = None, 0.0
        for g in gts:
            if g["label"] != o["label"] or (g["used"] and g["counts"]):
                continue
            gb = g["box"]
            pw, ph = 0.1 * (gb[2] - gb[0]) + 1, 0.1 * (gb[3] - gb[1]) + 1
            inside = gb[0] - pw <= cx <= gb[2] + pw and gb[1] - ph <= cy <= gb[3] + ph
            s = max(_iou(b, gb), 0.3 if inside else 0.0)
            if s >= 0.3 and s > best_s:
                best, best_s = g, s
        if best is None:
            res["fp"].append(o["label"])
            other = max(gts, key=lambda g: _iou(b, g["box"]), default=None)
            if other is not None and _iou(b, other["box"]) > 0.2:
                res["confusion"].append((o["label"], other["label"]))
            else:
                res["confusion"].append((o["label"], "background"))
        elif best["counts"]:
            best["used"] = True
            res["tp"].append(o["label"])
        # else: a detection of something too small or far to count: neutral
    for g in gts:
        if g["counts"] and not g["used"]:
            res["fn"].append(g["label"])
    return res


class VisionStats:
    """Running precision and recall per class, measured on the detector's own frames against the sim's
    segmentation (GET /api/vision/stats). Transparency only: the rovers never see the truth."""

    def __init__(self):
        self._mu = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._mu:
            self.tp = {k: 0 for k in LABELS}
            self.fp = {k: 0 for k in LABELS}
            self.fn = {k: 0 for k in LABELS}
            self.confusion: dict = {}
            self.frames = 0
            self.ms: list = []
            self.since = time.time()

    def add(self, res: VisionResult, seg: np.ndarray, model, depth: np.ndarray | None = None) -> dict:
        H, W = seg.shape
        m = match(res.objects, gt_instances(seg, model, depth), W, H)
        with self._mu:
            self.frames += 1
            self.ms.append(float(res.ms))
            if len(self.ms) > 500:
                self.ms = self.ms[-500:]
            for k in m["tp"]:
                self.tp[k] += 1
            for k in m["fp"]:
                self.fp[k] += 1
            for k in m["fn"]:
                self.fn[k] += 1
            for pair in m["confusion"]:
                key = f"{pair[0]}->{pair[1]}"
                self.confusion[key] = self.confusion.get(key, 0) + 1
        return m

    def summary(self) -> dict:
        with self._mu:
            classes = {}
            for k in LABELS:
                tp, fp, fn = self.tp[k], self.fp[k], self.fn[k]
                classes[k] = {"tp": tp, "fp": fp, "fn": fn,
                              "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                              "recall": round(tp / (tp + fn), 3) if tp + fn else None}
            tp, fp, fn = sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values())
            ms = np.array(self.ms) if self.ms else np.zeros(1)
            return {"frames": self.frames, "since": round(self.since, 1), "conf_min": CONF_REPORT,
                    "match": "same label and IoU >= 0.3 or box centre inside the object; objects under about "
                             "20 px (at 320 x 240) or beyond 30 m do not count",
                    "classes": classes,
                    "overall": {"tp": tp, "fp": fp, "fn": fn,
                                "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                                "recall": round(tp / (tp + fn), 3) if tp + fn else None},
                    "ms": {"avg": round(float(ms.mean()), 1), "p95": round(float(np.percentile(ms, 95)), 1)},
                    "confusion": dict(sorted(self.confusion.items(), key=lambda kv: -kv[1])[:12])}


def stats_for(sim) -> VisionStats:
    st = getattr(sim, "vision_stats", None)
    if st is None:
        st = sim.vision_stats = VisionStats()
    return st


# ====================================================================== HTTP
try:
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter()

    @router.get("/api/vision/stats")
    async def vision_stats():
        """Protocol v8 (optional): detector precision and recall per class against the sim's ground truth,
        measured on the rovers' own detector frames since the sim started."""
        from . import server
        runner = server.state.runner
        if runner is None:
            return JSONResponse({"ok": False, "error": "simulation not running"}, status_code=503)
        out = {"ok": True, "enabled": ENABLED}
        out.update(stats_for(runner.sim).summary())
        try:
            from . import camera
            out["camera"] = camera._RT.status()
        except Exception:      # noqa: BLE001
            pass
        return out
except Exception:          # noqa: BLE001 (no FastAPI: the detector still works)
    router = None
