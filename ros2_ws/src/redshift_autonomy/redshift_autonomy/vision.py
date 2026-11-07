"""Local, lightweight perception (OpenCV + numpy only — no cloud, no learned weights).

* Geometry first: depth pixels are back-projected into a gravity-levelled body frame; a robust
  ground plane is fitted and per-pixel residual height separates positive obstacles (boulders),
  negative obstacles (craters) and steep surface patches (slopes).
* Colour second: HSV gates confirm boulders (low-saturation grey vs. rust regolith) and find cyan
  science crystals.
* Uncertainty: image sharpness/contrast/noise and LiDAR return statistics are compared with
  baselines learned in clear conditions, so degradation is *measured*, never assumed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

CLS_COLORS = {'ROCK': (255, 150, 40), 'CRATER': (220, 70, 255), 'SCIENCE': (40, 235, 255), 'SLOPE': (255, 60, 60),
              'ROVER': (110, 255, 150)}
DAMAGED_COLOR = (255, 40, 60)
ROVER_BOX_M = (1.0, 0.62, 0.55)        # length, width, height envelope of a rover (wheels included)
TEAMMATE_DEPTH_TOL_M = 0.9
TEAMMATE_MIN_DEPTH_FRAC = 0.25
TEAMMATE_MIN_LIDAR = 8
SURFACE_TO_CENTRE_M = 0.3              # the sensors see the near face; the centre is about this much further


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, w: int, h: int, hfov: float) -> 'CameraModel':
        f = (w / 2.0) / math.tan(hfov / 2.0)
        return cls(w, h, f, f, w / 2.0, h / 2.0)

    @classmethod
    def from_K(cls, w: int, h: int, K) -> 'CameraModel':
        return cls(w, h, float(K[0]), float(K[4]), float(K[2]), float(K[5]))

    def scaled(self, w: int, h: int) -> 'CameraModel':
        sx, sy = w / self.width, h / self.height
        return CameraModel(w, h, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple                      # x, y, w, h in input-image pixels
    range_m: float
    bearing_rad: float               # + left
    world_xy: tuple | None = None
    sources: list = field(default_factory=list)
    slope_deg: float | None = None
    robot: str | None = None             # ROVER: which teammate
    damaged: bool | None = None
    label: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d['conf'] = round(self.conf, 2)
        d['range_m'] = round(self.range_m, 2)
        d['bearing_rad'] = round(self.bearing_rad, 3)
        d['bbox'] = [int(v) for v in self.bbox]
        if self.world_xy is not None:
            d['world_xy'] = [round(float(v), 2) for v in self.world_xy]
        if self.slope_deg is not None:
            d['slope_deg'] = round(self.slope_deg, 1)
        for k in ('robot', 'damaged', 'label'):
            if d[k] is None:
                del d[k]
        return d


# ----------------------------------------------------------------------------- geometry
def camera_points_body(depth, cam: CameraModel, pitch: float, cam_h: float, stride: int = 1,
                       keep_shape: bool = False):
    """Planar depth → points in a levelled frame under the camera (x fwd, y left, z up from ground)."""
    d = depth[::stride, ::stride].astype(np.float64)
    h, w = d.shape
    sx = depth.shape[1] / w
    sy = depth.shape[0] / h
    u, v = np.meshgrid(np.arange(w) * sx + 0.5, np.arange(h) * sy + 0.5)
    xo = (u - cam.cx) / cam.fx * d
    yo = (v - cam.cy) / cam.fy * d
    xf, yl, zu = d, -xo, -yo
    c, s = math.cos(pitch), math.sin(pitch)
    with np.errstate(invalid='ignore'):
        pts = np.stack([xf * c + zu * s, yl, -xf * s + zu * c + cam_h], axis=-1)
    pts[~np.isfinite(d) | (d <= 0)] = np.nan
    return pts if keep_shape else pts.reshape(-1, 3)


def fit_ground_plane(pts: np.ndarray, x_max: float = 10.0, iters: int = 3, inlier_m: float = 0.15):
    """Robust least-squares z = a·x + b·y + c. Returns (a, b, c) or (0, 0, 0)."""
    p = pts[np.isfinite(pts).all(axis=1)]
    p = p[(p[:, 0] > 0.3) & (p[:, 0] < x_max)]
    if len(p) < 30:
        return 0.0, 0.0, 0.0
    sel = p
    coef = np.array([0.0, 0.0, float(np.median(p[:, 2]))])
    for _ in range(iters):
        A = np.c_[sel[:, 0], sel[:, 1], np.ones(len(sel))]
        coef, *_ = np.linalg.lstsq(A, sel[:, 2], rcond=None)
        res = p[:, 2] - (coef[0] * p[:, 0] + coef[1] * p[:, 1] + coef[2])
        nxt = p[np.abs(res) < inlier_m]
        if len(nxt) < 30:
            break
        sel = nxt
    return float(coef[0]), float(coef[1]), float(coef[2])


def _components(mask: np.ndarray, min_area: int):
    if mask.size == 0 or not mask.any():     # OpenCV segfaults on empty inputs
        return
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            yield labels == k, stats[k]


def detect_objects(rgb, depth, cam: CameraModel, cam_pitch: float, cam_h: float, pose=None,
                   cam_conf: float = 1.0, max_range: float = 15.0, slope_limit_deg: float = 20.0,
                   use_camera: bool = True, classes=('ROCK', 'CRATER', 'SCIENCE', 'SLOPE'), rock_h: float = 0.25,
                   tophat_px: int = 0, min_rock_px: int | None = None, rock_gray_min: float = 0.0):
    """Returns a list of Detection sorted by confidence (bbox in rgb pixel coordinates)."""
    dh, dw = depth.shape
    if rgb.shape[:2] != (dh, dw):
        rgb_s = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        rgb_s = rgb
    bx_scale = rgb.shape[1] / dw
    by_scale = rgb.shape[0] / dh
    cam_d = cam if (cam.width, cam.height) == (dw, dh) else cam.scaled(dw, dh)
    P = camera_points_body(depth, cam_d, cam_pitch, cam_h, keep_shape=True)
    rng_h = np.hypot(P[..., 0], P[..., 1])
    valid = np.isfinite(P).all(axis=-1) & (rng_h < max_range)
    a, b, c = fit_ground_plane(P.reshape(-1, 3))
    resid = np.where(valid, P[..., 2] - (a * P[..., 0] + b * P[..., 1] + c), 0.0)
    if tophat_px > 0:
        # nadir views span tens of metres of undulating terrain: height above the grey opening (top-hat)
        # keeps ramps at zero and isolates bumps narrower than the kernel (boulders)
        z = np.where(valid, P[..., 2], np.nan)
        z = np.where(valid, z, np.nanmedian(z) if valid.any() else 0.0).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(tophat_px) + 1,) * 2)
        resid = np.where(valid, cv2.morphologyEx(z, cv2.MORPH_TOPHAT, k), 0.0)
    hsv = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k3 = np.ones((3, 3), np.uint8)

    pos = valid & (resid > rock_h)
    grayish = (S < 75) & (V > 35) & (V < 215)
    rock_mask = pos | (grayish & valid & (resid > 0.12)) if use_camera else pos
    rock_mask = cv2.morphologyEx(rock_mask.astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)
    neg = cv2.morphologyEx((valid & (resid < -0.30)).astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)
    sci = ((H >= 78) & (H <= 102) & (S > 110) & (V > 90)) if use_camera else np.zeros_like(valid)

    area_scale = (dw * dh) / (160 * 120)
    dets: list[Detection] = []

    def make(cls, m, st, conf, sources, slope=None):
        pts = P[m & valid]
        if len(pts) == 0:
            pts_xy = None
        else:
            pts_xy = np.median(pts[:, :2], axis=0)
        if pts_xy is None:
            u = st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH] / 2
            bearing = -math.atan((u - cam_d.cx) / cam_d.fx)
            r = float('nan')
        else:
            r = float(math.hypot(*pts_xy))
            bearing = float(math.atan2(pts_xy[1], pts_xy[0]))
        wxy = None
        if pose is not None and math.isfinite(r):
            x, y, yaw = pose
            wxy = (x + r * math.cos(yaw + bearing), y + r * math.sin(yaw + bearing))
        bbox = (st[cv2.CC_STAT_LEFT] * bx_scale, st[cv2.CC_STAT_TOP] * by_scale,
                st[cv2.CC_STAT_WIDTH] * bx_scale, st[cv2.CC_STAT_HEIGHT] * by_scale)
        dets.append(Detection(cls, float(min(0.99, conf)), bbox, r, bearing, wxy, sources, slope))

    if 'ROCK' not in classes:
        rock_mask = np.zeros_like(rock_mask)
    if 'CRATER' not in classes:
        neg = np.zeros_like(neg)
    if 'SCIENCE' not in classes:
        sci = np.zeros_like(sci)
    for m, st in _components(rock_mask, int(25 * area_scale) if min_rock_px is None else min_rock_px):
        area = st[cv2.CC_STAT_AREA] / area_scale
        frac_gray = float(grayish[m].mean())
        mres = float(resid[m].mean())
        if frac_gray < rock_gray_min and mres < 0.6:
            continue        # low bump with regolith colour: terrain texture, not a boulder
        conf = 0.55 + 0.2 * min(1.0, area / 150) + 0.15 * frac_gray + 0.1 * min(1.0, mres / 0.6)
        conf *= 0.6 + 0.4 * (cam_conf if use_camera else 0.5)
        make('ROCK', m, st, conf, ['DEPTH', 'CAM'] if use_camera and frac_gray > 0.3 else ['DEPTH'])
    for m, st in _components(neg & ~rock_mask, int(60 * area_scale)):
        area = st[cv2.CC_STAT_AREA] / area_scale
        conf = 0.5 + 0.3 * min(1.0, area / 400) + 0.2 * min(1.0, -float(resid[m].mean()) / 0.8)
        make('CRATER', m, st, conf, ['DEPTH'])
    for m, st in _components(sci, max(4, int(8 * area_scale))):
        area = st[cv2.CC_STAT_AREA] / area_scale
        conf = (0.6 + 0.3 * min(1.0, area / 40) + 0.1 * float(S[m].mean()) / 255) * (0.5 + 0.5 * cam_conf)
        make('SCIENCE', m, st, conf, ['CAM'])

    # steep surface patches: local plane fits on blocks, merged
    blk = max(8, dw // 10)
    nby, nbx = (dh // blk, dw // blk) if 'SLOPE' in classes else (0, 0)
    steep = np.zeros((nby, nbx), bool)
    slopes = np.zeros((nby, nbx))
    obstacle_px = rock_mask | neg
    for i in range(nby):
        for j in range(nbx):
            sl = (slice(i * blk, (i + 1) * blk), slice(j * blk, (j + 1) * blk))
            vb = valid[sl]
            if vb.sum() < blk * blk * 0.5 or obstacle_px[sl].mean() > 0.2:
                continue
            q = P[sl][vb]
            A = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
            coef, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
            sdeg = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
            slopes[i, j] = sdeg
            steep[i, j] = sdeg > slope_limit_deg
    for m, st in _components(steep, 2):
        sdeg = float(slopes[m].max())
        mpx = cv2.resize(m.astype(np.uint8), (nbx * blk, nby * blk), interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((dh, dw), bool)
        full[:nby * blk, :nbx * blk] = mpx
        st_px = np.array([st[0] * blk, st[1] * blk, st[2] * blk, st[3] * blk, st[4] * blk * blk])
        conf = 0.6 + 0.35 * min(1.0, (sdeg - slope_limit_deg) / 10.0)
        make('SLOPE', full, st_px, conf, ['DEPTH'], slope=sdeg)

    dets.sort(key=lambda d: -d.conf)
    return dets


def ground_slope_ahead_deg(a: float) -> float:
    return math.degrees(math.atan(a))


# ----------------------------------------------------------------------------- quality
def image_quality(rgb) -> dict:
    """Coarse-scale contrast/edge energy (robust to pixel noise) plus a pixel-noise estimate."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = max(2, gray.shape[1] // 40)
    d = cv2.resize(gray, (gray.shape[1] // k, gray.shape[0] // k), interpolation=cv2.INTER_AREA).astype(np.float32)
    mag = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))
    sharp = float(np.mean(np.sort(mag, axis=None)[-max(1, mag.size // 10):]))
    noise = float(np.median(np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 5).astype(np.int16))))
    return {'sharpness': sharp, 'contrast': float(d.std()), 'noise': noise, 'brightness': float(gray.mean())}


CLEAR_NOISE_MARGIN = 1.5      # grey levels of pixel noise above the baseline still counted as clear optics
CLEAR_AIR_LIDAR = 0.95        # LiDAR confidence that independently confirms clear air (camera baseline re-learning)


class ImageQualityBaseline:
    """Learns clear-condition image statistics; confidence = similarity to that baseline."""

    def __init__(self, max_samples: int = 30):
        self.samples: list[dict] = []
        self.max_samples = max_samples

    @property
    def ready(self) -> bool:
        return len(self.samples) >= 1

    def _base(self, k):
        return float(np.median([s[k] for s in self.samples]))

    def update(self, q: dict, clear_air: bool = False) -> None:
        """Learn a frame as clear-condition. Frames close to the baseline always are. A frame that has lost texture
        is learned too when an independent sensor confirms clear air (`clear_air`: the LiDAR is undegraded) and the
        image carries no added pixel noise: then the scene changed (smoother ground, a teammate leaving the view),
        not the optics, and the baseline must follow or the camera would look degraded for good."""
        if (len(self.samples) < 5 or self.confidence(q) > 0.8
                or (clear_air and q['noise'] <= self._base('noise') + CLEAR_NOISE_MARGIN)):
            self.samples.append(q)
            self.samples = self.samples[-self.max_samples:]

    def confidence(self, q: dict) -> float:
        if not self.samples:
            return 1.0
        s = min(1.0, q['sharpness'] / max(self._base('sharpness'), 1e-6))
        c = min(1.0, q['contrast'] / max(self._base('contrast'), 1e-6))
        n = 1.0 - float(np.clip((q['noise'] - self._base('noise')) / 8.0, 0.0, 1.0))
        return float(np.clip(0.5 * s + 0.3 * c + 0.2 * n, 0.0, 1.0))


NEAR_FIELD_M = 2.0


def _isolated(mask: np.ndarray) -> np.ndarray:
    """Cells of an organized (rings, samples) mask with no set neighbour along the scan (wrapping) or across rings."""
    nb = np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
    nb[1:] |= mask[:-1]
    nb[:-1] |= mask[1:]
    return mask & ~nb


def lidar_quality(points) -> dict:
    """Scan statistics for LidarBaseline. near_ratio counts near-field returns that look like dust backscatter: in an
    organized scan only isolated ones, because a real surface close by (a boulder beside the route) returns a
    contiguous patch, while backscatter is sparse single beams."""
    P = np.asarray(points, dtype=np.float64)
    organized = P.ndim == 3
    flat = P.reshape(-1, 3)
    fin = np.isfinite(flat).all(axis=1)
    r = np.linalg.norm(flat[fin], axis=1)
    rough = 0.0
    near = float((r < NEAR_FIELD_M).mean()) if len(r) else 0.0
    if organized and P.shape[1] >= 3:
        R = np.linalg.norm(P, axis=-1)
        d = np.abs(R[:, 1:-1] - 0.5 * (R[:, :-2] + R[:, 2:]))
        d = d[np.isfinite(d)]
        rough = float(np.median(d)) if len(d) else 0.0
        with np.errstate(invalid='ignore'):
            near_mask = np.isfinite(R) & (R < NEAR_FIELD_M)
        near = float(_isolated(near_mask).sum() / max(1, int(fin.sum())))
    return {'return_ratio': float(fin.mean()) if len(fin) else 0.0, 'near_ratio': near,
            'roughness': rough, 'n': int(fin.sum())}


class LidarBaseline:
    def __init__(self, max_samples: int = 20):
        self.samples: list[dict] = []
        self.max_samples = max_samples

    def _base(self, k):
        return float(np.median([s[k] for s in self.samples]))

    def update(self, q: dict) -> None:
        if len(self.samples) < 3 or self.confidence(q) > 0.85:
            self.samples.append(q)
            self.samples = self.samples[-self.max_samples:]

    def confidence(self, q: dict) -> float:
        if not self.samples:
            return 1.0
        ratio = min(1.0, q['return_ratio'] / max(self._base('return_ratio'), 1e-6))
        near = max(0.0, q['near_ratio'] - self._base('near_ratio'))
        rough = max(0.0, q['roughness'] - self._base('roughness'))
        return float(np.clip(ratio - 3.0 * near - 4.0 * rough, 0.0, 1.0))


def fuse_uncertainty(camera_conf: float, lidar_conf: float, imu_vib: float, mode: str = 'FUSED') -> float:
    imu_pen = float(np.clip(imu_vib / 2.0, 0.0, 1.0))
    if mode == 'LIDAR_ONLY':
        conf = 0.8 * lidar_conf + 0.2 * (1 - imu_pen)
    else:
        conf = 0.45 * camera_conf + 0.45 * lidar_conf + 0.10 * (1 - imu_pen)
    return float(np.clip(1.0 - conf, 0.0, 1.0))


CORRIDOR_CELL_M = 0.25


def lidar_corridor(points_base, half_width: float = 1.0, x_max: float = 10.0, obstacle_h: float = 0.3,
                   prev_cells=None, x_min: float = 1.0) -> dict:
    """Points in a levelled base frame (ground ≈ z 0) → nearest obstacle in the forward corridor.

    Obstacle cells (0.25 m columns) need >= 3 returns (solid surfaces are dense, dust backscatter is sparse) and,
    when `prev_cells` from the previous scan is given, must also have been dense there (±1 cell for ego-motion):
    dust is re-drawn every scan, boulders are not. `out['cells']` feeds the next call."""
    p = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    out = {'obstacle_dist_m': math.inf, 'slope_ahead_deg': 0.0, 'conf': 0.0, 'points': 0, 'cells': set()}
    if len(p) < 20:
        return out
    near = p[np.hypot(p[:, 0], p[:, 1]) < 12.0]
    a, b, c = fit_ground_plane(near, x_max=12.0, inlier_m=0.2)
    res = p[:, 2] - (a * p[:, 0] + b * p[:, 1] + c)
    # x_min: returns inside the rover's own stopping envelope are dominated by dust backscatter; anything solid
    # that close was already seen (and mapped) on approach
    cor = (p[:, 0] > x_min) & (p[:, 0] < x_max) & (np.abs(p[:, 1]) < half_width)
    obs = cor & (res > obstacle_h) & (res < 3.0)
    if obs.sum() >= 3:
        q = p[obs]
        cell = np.floor(q[:, :2] / CORRIDOR_CELL_M).astype(np.int64)
        key = cell[:, 0] * 100_000 + cell[:, 1]
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        dense_keys = set(uk[cnt >= 3].tolist())
        out['cells'] = dense_keys
        if prev_cells is not None:
            near = {k + dx * 100_000 + dy for k in prev_cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            dense_keys &= near
        dense = np.isin(key, list(dense_keys)) if dense_keys else np.zeros(len(q), bool)
        if dense.any():
            out['obstacle_dist_m'] = float(q[dense, 0].min())
    gnd = cor & (np.abs(res) < 0.15) & (p[:, 0] > 1.0) & (p[:, 0] < 8.0)
    if gnd.sum() >= 10:
        A = np.c_[p[gnd, 0], np.ones(int(gnd.sum()))]
        k, *_ = np.linalg.lstsq(A, p[gnd, 2], rcond=None)
        out['slope_ahead_deg'] = math.degrees(math.atan(k[0]))
    n = int(cor.sum())
    out['points'] = n
    out['conf'] = float(min(0.98, 0.5 + n / 200.0))
    return out


# ----------------------------------------------------------------------------- overlay
def _to_cam(pitch: float, X, Y, Z):
    """Levelled camera frame (x fwd, y left, z up, origin at the camera) → optical (right, down, forward)."""
    c, s = math.cos(pitch), math.sin(pitch)
    xf = c * X - s * Z
    zu = s * X + c * Z
    return -np.asarray(Y, float), -zu, xf


def project_box(cam: CameraModel, cam_pitch: float, rel_xyz, rel_yaw: float, box=ROVER_BOX_M):
    """Image box (x, y, w, h) of a box-shaped body centred at `rel_xyz` (levelled camera frame) with heading
    `rel_yaw` relative to the camera, and the optical-axis depth of its centre; None when not in view."""
    L, W, H = box
    sx, sy, sz = np.meshgrid([-L / 2, L / 2], [-W / 2, W / 2], [-H / 2, H / 2], indexing='ij')
    cy_, sy_ = math.cos(rel_yaw), math.sin(rel_yaw)
    X = rel_xyz[0] + cy_ * sx.ravel() - sy_ * sy.ravel()
    Y = rel_xyz[1] + sy_ * sx.ravel() + cy_ * sy.ravel()
    Z = rel_xyz[2] + sz.ravel()
    xo, yo, xf = _to_cam(cam_pitch, X, Y, Z)
    ok = xf > 0.1
    if ok.sum() < 4:
        return None
    u = cam.cx + cam.fx * xo[ok] / xf[ok]
    v = cam.cy + cam.fy * yo[ok] / xf[ok]
    x0, x1 = int(max(0, math.floor(u.min()))), int(min(cam.width, math.ceil(u.max())))
    y0, y1 = int(max(0, math.floor(v.min()))), int(min(cam.height, math.ceil(v.max())))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    _, _, xf_c = _to_cam(cam_pitch, rel_xyz[0], rel_xyz[1], rel_xyz[2])
    return (x0, y0, x1 - x0, y1 - y0), float(xf_c)


def teammate_detection(cam: CameraModel, depth, cam_pitch: float, rel_xyz, rel_yaw: float, rel_base_xy,
                       lidar_pts=None, robot: str = '', damaged: bool = False, label: str | None = None,
                       world_xy=None, box=ROVER_BOX_M):
    """Confirm a teammate at its fleet-reported pose with this robot's own sensors.

    The reported pose only says *where to look*: the box is projected into the image, and the teammate counts as
    perceived only if depth pixels inside the box sit at the expected range (not an occluding boulder, not sky) or
    enough LiDAR returns lie on its footprint. Returns a `ROVER` Detection, or None if not perceived."""
    proj = project_box(cam, cam_pitch, rel_xyz, rel_yaw, box)
    sources, conf = [], 0.55
    bx, by = float(rel_base_xy[0]), float(rel_base_xy[1])
    rng_geo = math.hypot(bx, by)
    rng = None
    bbox = (0, 0, 0, 0)
    if proj is not None and depth is not None:
        bbox, xf = proj
        x, y, w, h = bbox
        dh, dw = depth.shape[:2]
        kx, ky = dw / cam.width, dh / cam.height
        reg = depth[int(y * ky):int((y + h) * ky) + 1, int(x * kx):int((x + w) * kx) + 1]
        if reg.size:
            match = np.isfinite(reg) & (np.abs(reg - xf) <= TEAMMATE_DEPTH_TOL_M)
            frac = float(match.mean())
            if frac >= TEAMMATE_MIN_DEPTH_FRAC:
                sources += ['CAM', 'DEPTH']
                conf += 0.25 * min(1.0, frac / 0.5)
                rng = rng_geo * (float(np.median(reg[match])) + SURFACE_TO_CENTRE_M) / max(xf, 0.1)
    if lidar_pts is not None and len(lidar_pts):
        p = np.asarray(lidar_pts, float)
        on = (np.hypot(p[:, 0] - bx, p[:, 1] - by) < 0.9) & (p[:, 2] > 0.05) & (p[:, 2] < 1.2)
        if int(on.sum()) >= TEAMMATE_MIN_LIDAR:
            sources.append('LIDAR')
            conf += 0.15
            rng = float(np.median(np.hypot(p[on, 0], p[on, 1]))) + SURFACE_TO_CENTRE_M
    if not sources:
        return None
    return Detection('ROVER', float(min(0.97, conf)), bbox, float(rng if rng is not None else rng_geo),
                     float(math.atan2(by, bx)), tuple(world_xy) if world_xy is not None else None, sources,
                     robot=robot, damaged=bool(damaged), label=label)


def depth_colormap(depth, far_m: float = 25.0) -> np.ndarray:
    """Planar depth → RGB uint8 (turbo: near = red, far = blue, no return = black)."""
    d = np.asarray(depth, np.float32)
    ok = np.isfinite(d) & (d > 0)
    norm = np.zeros(d.shape, np.uint8)
    norm[ok] = (255.0 * np.clip(1.0 - d[ok] / far_m, 0.0, 1.0)).astype(np.uint8)
    img = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    img[~ok] = 0
    return img


def annotate(rgb, dets, info: dict):
    """Burn detections + HUD into an RGB frame (returns RGB uint8, same shape)."""
    img = rgb.copy()
    h, w = img.shape[:2]
    lidar_only = info.get('mode') == 'LIDAR_ONLY'
    if lidar_only:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = (0.45 * cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)).astype(np.uint8)
    fs = max(0.3, w / 900)
    th = 1 if w < 500 else 2
    for d in dets:
        col = DAMAGED_COLOR if d.damaged else CLS_COLORS.get(d.cls, (255, 255, 255))
        x, y, bw, bh = (int(v) for v in d.bbox)
        if bw <= 0 or bh <= 0:
            continue
        cv2.rectangle(img, (x, y), (x + bw, y + bh), col, th + (1 if d.cls == 'ROVER' else 0))
        rng = f' {d.range_m:.1f}m' if math.isfinite(d.range_m) else ''
        extra = f' {d.slope_deg:.0f}deg' if d.slope_deg is not None else ''
        if d.label:      # teammate lock-on: one line per ' | ' part, kept inside the frame
            lines = [t.strip() for t in d.label.strip('[]').split('|')]
            lines[-1] += rng
        else:
            lines = [f'{d.cls} {d.conf:.2f}{rng}{extra}']
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0] for t in lines]
        lh = max(sz[1] for sz in sizes) + 4
        ty = max(lh * len(lines), y - 2)
        for k, (t, (tw, tht)) in enumerate(zip(lines, sizes)):
            yy = ty - (len(lines) - 1 - k) * lh
            xx = int(min(max(0, x), max(0, w - tw - 3)))
            cv2.rectangle(img, (xx, yy - tht - 2), (xx + tw + 2, yy + 1), col, -1)
            cv2.putText(img, t, (xx + 1, yy - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (10, 10, 10), 1, cv2.LINE_AA)
    # Mode, uncertainty and frame rate are shown by Mission Control's own badges in the frame corners, so the
    # burned-in text is limited to centred callouts that never sit under them.
    sl = info.get('slope_ahead_deg')
    if sl is not None:
        txt = f'SLOPE AHEAD {sl:+.1f}deg'
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(img, txt, ((w - tw) // 2, h - 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 220, 120), 1, cv2.LINE_AA)
    if lidar_only:
        msg = 'CAM DEGRADED - LIDAR ONLY'
        (tw, tht), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3, th)
        cx, cy = (w - tw) // 2, h // 2
        cv2.rectangle(img, (cx - 6, cy - tht - 6), (cx + tw + 6, cy + 6), (200, 30, 30), -1)
        cv2.putText(img, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3, (255, 255, 255), th, cv2.LINE_AA)
    return img
