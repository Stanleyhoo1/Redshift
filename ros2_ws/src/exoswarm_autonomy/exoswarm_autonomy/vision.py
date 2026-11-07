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

CLS_COLORS = {'ROCK': (255, 150, 40), 'CRATER': (220, 70, 255), 'SCIENCE': (40, 235, 255), 'SLOPE': (255, 60, 60)}


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
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            yield labels == k, stats[k]


def detect_objects(rgb, depth, cam: CameraModel, cam_pitch: float, cam_h: float, pose=None,
                   cam_conf: float = 1.0, max_range: float = 15.0, slope_limit_deg: float = 20.0,
                   use_camera: bool = True):
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
    hsv = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k3 = np.ones((3, 3), np.uint8)

    pos = valid & (resid > 0.25)
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

    for m, st in _components(rock_mask, int(25 * area_scale)):
        area = st[cv2.CC_STAT_AREA] / area_scale
        frac_gray = float(grayish[m].mean())
        mres = float(resid[m].mean())
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
    nby, nbx = dh // blk, dw // blk
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

    def update(self, q: dict) -> None:
        if len(self.samples) < 5 or self.confidence(q) > 0.8:
            self.samples.append(q)
            self.samples = self.samples[-self.max_samples:]

    def confidence(self, q: dict) -> float:
        if not self.samples:
            return 1.0
        s = min(1.0, q['sharpness'] / max(self._base('sharpness'), 1e-6))
        c = min(1.0, q['contrast'] / max(self._base('contrast'), 1e-6))
        n = 1.0 - float(np.clip((q['noise'] - self._base('noise')) / 8.0, 0.0, 1.0))
        return float(np.clip(0.5 * s + 0.3 * c + 0.2 * n, 0.0, 1.0))


def lidar_quality(points) -> dict:
    P = np.asarray(points, dtype=np.float64)
    organized = P.ndim == 3
    flat = P.reshape(-1, 3)
    fin = np.isfinite(flat).all(axis=1)
    r = np.linalg.norm(flat[fin], axis=1)
    rough = 0.0
    if organized and P.shape[1] >= 3:
        R = np.linalg.norm(P, axis=-1)
        d = np.abs(R[:, 1:-1] - 0.5 * (R[:, :-2] + R[:, 2:]))
        d = d[np.isfinite(d)]
        rough = float(np.median(d)) if len(d) else 0.0
    return {'return_ratio': float(fin.mean()) if len(fin) else 0.0,
            'near_ratio': float((r < 2.0).mean()) if len(r) else 0.0,
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


def lidar_corridor(points_base, half_width: float = 1.0, x_max: float = 10.0, obstacle_h: float = 0.3) -> dict:
    """Points in a levelled base frame (ground ≈ z 0) → nearest obstacle in the forward corridor."""
    p = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    out = {'obstacle_dist_m': math.inf, 'slope_ahead_deg': 0.0, 'conf': 0.0, 'points': 0}
    if len(p) < 20:
        return out
    near = p[np.hypot(p[:, 0], p[:, 1]) < 12.0]
    a, b, c = fit_ground_plane(near, x_max=12.0, inlier_m=0.2)
    res = p[:, 2] - (a * p[:, 0] + b * p[:, 1] + c)
    cor = (p[:, 0] > 0.5) & (p[:, 0] < x_max) & (np.abs(p[:, 1]) < half_width)
    obs = cor & (res > obstacle_h) & (res < 3.0)
    if obs.sum() >= 3:
        out['obstacle_dist_m'] = float(np.sort(p[obs, 0])[2])
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
        col = CLS_COLORS.get(d.cls, (255, 255, 255))
        x, y, bw, bh = (int(v) for v in d.bbox)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), col, th)
        rng = f' {d.range_m:.1f}m' if math.isfinite(d.range_m) else ''
        extra = f' {d.slope_deg:.0f}deg' if d.slope_deg is not None else ''
        label = f'{d.cls} {d.conf:.2f}{rng}{extra}'
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(tht + 2, y - 2)
        cv2.rectangle(img, (x, ty - tht - 2), (x + tw + 2, ty + 1), col, -1)
        cv2.putText(img, label, (x + 1, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (10, 10, 10), 1, cv2.LINE_AA)
    bar = img[:int(16 * fs / 0.35)].astype(np.float32) * 0.35
    img[:bar.shape[0]] = bar.astype(np.uint8)
    hud = (f"{str(info.get('robot', '')).upper()}  {info.get('mode', '')}  U {info.get('uncertainty', 0):.2f}  "
           f"{info.get('fps', 0):.1f}Hz")
    cv2.putText(img, hud, (4, int(12 * fs / 0.35)), cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), 1, cv2.LINE_AA)
    sl = info.get('slope_ahead_deg')
    if sl is not None:
        txt = f'SLOPE AHEAD {sl:+.1f}deg'
        cv2.putText(img, txt, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 220, 120), 1, cv2.LINE_AA)
    if lidar_only:
        msg = 'CAM DEGRADED - LIDAR ONLY'
        (tw, tht), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3, th)
        cx, cy = (w - tw) // 2, h // 2
        cv2.rectangle(img, (cx - 6, cy - tht - 6), (cx + tw + 6, cy + 6), (200, 30, 30), -1)
        cv2.putText(img, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3, (255, 255, 255), th, cv2.LINE_AA)
    return img
