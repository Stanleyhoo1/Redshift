"""Terrain generation and fake sensor rendering for the UI-only demo backend.

Everything here is plausible-looking, not physically faithful: it exists so the web UI can be
exercised end-to-end without ROS 2 / Gazebo. numpy + OpenCV only.
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

CLS_UNKNOWN, CLS_FREE, CLS_CAUTION, CLS_HAZARD, CLS_PERMANENT, CLS_SCIENCE, CLS_OBSTACLE = range(7)


# --------------------------------------------------------------------------------------------
# physics (same model as the plan / web_ui/src/lib/defaults.ts)
# --------------------------------------------------------------------------------------------
def physics_preview(g: float) -> dict:
    weight = 60.0 * g * 9.81
    drive = math.degrees(math.asin(min(1.0, 416.0 / weight)))
    slip = math.degrees(math.atan(math.tan(math.radians(32)) + 150.0 / weight))
    tip = math.degrees(math.atan(0.425 / 0.35))
    lim = min(drive, slip, tip)
    factor = "drive_torque" if lim == drive else "slip" if lim == slip else "tip"
    mx = 0.9 * lim
    return {
        "g": g, "max_slope_deg": mx, "factor": factor, "drive_deg": drive, "slip_deg": slip,
        "tip_deg": tip, "tip_alarm_deg": min(0.7 * tip, mx + 12.0),
        "v_max": min(1.0, math.sqrt(1.5 / g)), "energy_j_per_m": weight * 0.08 / 0.75,
    }


# --------------------------------------------------------------------------------------------
# terrain
# --------------------------------------------------------------------------------------------
@dataclass
class Terrain:
    size_m: float
    res: float
    n: int
    heights: np.ndarray  # [row=y, col=x], metres
    slope_deg: np.ndarray
    rocks: list = field(default_factory=list)  # [x, y, r]
    science: list = field(default_factory=list)  # [x, y]
    spawn: tuple = (0.0, 0.0)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        j = int((x + self.size_m / 2) / self.res)
        i = int((y + self.size_m / 2) / self.res)
        return min(self.n - 1, max(0, i)), min(self.n - 1, max(0, j))

    def height_at(self, x: float, y: float) -> float:
        fx = (x + self.size_m / 2) / self.res - 0.5
        fy = (y + self.size_m / 2) / self.res - 0.5
        j0 = int(np.clip(math.floor(fx), 0, self.n - 2))
        i0 = int(np.clip(math.floor(fy), 0, self.n - 2))
        tx = float(np.clip(fx - j0, 0, 1))
        ty = float(np.clip(fy - i0, 0, 1))
        h = self.heights
        top = h[i0, j0] * (1 - tx) + h[i0, j0 + 1] * tx
        bot = h[i0 + 1, j0] * (1 - tx) + h[i0 + 1, j0 + 1] * tx
        return float(top * (1 - ty) + bot * ty)

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        d = self.res
        return ((self.height_at(x + d, y) - self.height_at(x - d, y)) / (2 * d),
                (self.height_at(x, y + d) - self.height_at(x, y - d)) / (2 * d))


def generate_terrain(size_m: float, res: float, seed: int) -> Terrain:
    n = int(round(size_m / res))
    rng = np.random.default_rng(seed)
    relief = {50: 2.5, 100: 5.0, 200: 9.0}.get(int(size_m), 5.0)
    h = np.zeros((n, n), np.float32)
    for sigma_m, amp in ((size_m / 6, 1.0), (size_m / 14, 0.45), (size_m / 35, 0.18), (2.0, 0.05)):
        nz = rng.standard_normal((n, n)).astype(np.float32)
        nz = cv2.GaussianBlur(nz, (0, 0), sigmaX=max(1.0, sigma_m / res))
        nz /= nz.std() + 1e-6
        h += amp * nz
    h = (h - h.min()) / (h.max() - h.min() + 1e-6) * relief
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    X = (xs + 0.5) * res - size_m / 2
    Y = (ys + 0.5) * res - size_m / 2
    # a steep ridge (crossable at 1 g, not at 2.2 g)
    ang = rng.uniform(0, math.pi)
    off = rng.uniform(-0.1, 0.25) * size_m
    dist = X * math.cos(ang) + Y * math.sin(ang) - off
    h += 0.06 * size_m * np.exp(-0.5 * (dist / (0.035 * size_m)) ** 2)
    # craters
    for _ in range(3):
        cx, cy = rng.uniform(-0.35, 0.35, 2) * size_m
        r = rng.uniform(0.04, 0.08) * size_m
        d2 = ((X - cx) ** 2 + (Y - cy) ** 2) / r ** 2
        h += (-0.8 * np.exp(-d2 * 1.6) + 0.35 * np.exp(-((np.sqrt(d2) - 1.0) ** 2) * 8)) * r * 0.25
    # flattened spawn pad
    sx, sy = -0.35 * size_m, -0.35 * size_m
    pad = np.exp(-(((X - sx) ** 2 + (Y - sy) ** 2) / (0.08 * size_m) ** 2))
    h = h * (1 - pad) + float(h[np.unravel_index(np.argmin((X - sx) ** 2 + (Y - sy) ** 2), h.shape)]) * pad
    gy, gx = np.gradient(h, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    rocks = []
    while len(rocks) < int(rng.integers(25, 45)):
        x, y = rng.uniform(-0.47, 0.47, 2) * size_m
        if math.hypot(x - sx, y - sy) < 0.12 * size_m:
            continue
        rocks.append([float(x), float(y), float(rng.uniform(0.3, 1.1))])
    science = []
    while len(science) < int(rng.integers(5, 8)):
        x, y = rng.uniform(-0.44, 0.44, 2) * size_m
        if math.hypot(x - sx, y - sy) > 0.1 * size_m:
            science.append([float(x), float(y)])
    return Terrain(size_m, res, n, h.astype(np.float32), slope, rocks, science, (sx, sy))


def albedo_texture(t: Terrain, px_m: float) -> np.ndarray:
    """Top-down BGR colour texture of the terrain (row 0 = south) used by both cameras."""
    m = int(round(t.size_m / px_m))
    h = cv2.resize(t.heights, (m, m), interpolation=cv2.INTER_LINEAR)
    gy, gx = np.gradient(h, px_m)
    nrm = np.dstack([-gx * 2.2, -gy * 2.2, np.ones_like(h)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    shade = np.clip(nrm @ np.array([-0.5, 0.55, 0.68]) / 0.98, 0, 1)
    e = (h - h.min()) / (h.max() - h.min() + 1e-6)
    rng = np.random.default_rng(7)
    grain = cv2.GaussianBlur(rng.standard_normal((m, m)).astype(np.float32), (0, 0), 1.2)
    lo, mid, hi = np.array([40, 34, 60.0]), np.array([38, 70, 160.0]), np.array([90, 150, 215.0])  # BGR
    col = np.where(e[..., None] < 0.55, lo + (mid - lo) * (e[..., None] / 0.55), mid + (hi - mid) * ((e[..., None] - 0.55) / 0.45))
    col = col * (0.45 + 0.75 * shade[..., None]) + grain[..., None] * 9
    img = np.clip(col, 0, 255).astype(np.uint8)
    w2p = lambda x, y: (int((x + t.size_m / 2) / px_m), int((y + t.size_m / 2) / px_m))
    for x, y, r in t.rocks:
        cv2.circle(img, w2p(x, y), max(2, int(r / px_m)), (78, 72, 84), -1, cv2.LINE_AA)
    for x, y in t.science:
        cv2.circle(img, w2p(x, y), max(2, int(0.5 / px_m)), (230, 240, 92), -1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------------------------
# sensors
# --------------------------------------------------------------------------------------------
def b64_jpeg(img: np.ndarray, q: int = 72) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def rover_camera(t: Terrain, tex: np.ndarray, px_m: float, pose: dict, rocks: list, dust: float,
                 seismic: bool, label: str, mode: str, unc: float, rng: np.random.Generator) -> np.ndarray:
    W, H, hfov, pitch, cam_h = 320, 240, math.radians(80), math.radians(-12), 1.0
    f = (W / 2) / math.tan(hfov / 2)
    u, v = np.meshgrid(np.arange(W, dtype=np.float32) - W / 2, np.arange(H, dtype=np.float32) - H / 2)
    # ray elevation for each row (camera pitched down)
    elev = pitch - np.arctan(v / f)
    az = np.arctan(-u / f)
    yaw = pose["yaw"]
    down = elev < -0.005
    d = np.where(down, cam_h / np.tan(-np.minimum(elev, -0.005)), 1e4)
    d = np.minimum(d, 60.0)
    wx = pose["x"] + d * np.cos(yaw + az)
    wy = pose["y"] + d * np.sin(yaw + az)
    mx = ((wx + t.size_m / 2) / px_m).astype(np.float32)
    my = ((wy + t.size_m / 2) / px_m).astype(np.float32)
    ground = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    sky_top, sky_hor = np.array([150, 170, 214.0]), np.array([92, 128, 196.0])
    k = np.clip((v + H / 2) / H, 0, 1)[..., None]
    sky = sky_top * (1 - k) + sky_hor * k
    haze = np.clip(d / 55.0, 0, 1)[..., None] ** 1.3
    img = np.where(down[..., None], ground * (1 - haze) + sky_hor * haze, sky)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # boulders (3-D-ish ellipses) + perception boxes burned into the frame
    boxes = []
    for x, y, r in rocks:
        dx, dy = x - pose["x"], y - pose["y"]
        fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
        lat = -dx * math.sin(yaw) + dy * math.cos(yaw)
        if fwd < 1.5 or fwd > 22 or abs(math.atan2(lat, fwd)) > hfov / 2:
            continue
        cx = W / 2 - f * lat / fwd
        base = H / 2 - f * math.tan(math.atan(-cam_h / fwd) - pitch)
        rad = f * r / fwd
        cy = base - rad * 0.7
        cv2.ellipse(img, (int(cx), int(cy)), (max(2, int(rad)), max(2, int(rad * 0.75))), 0, 0, 360, (70, 66, 80), -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(cx - rad * 0.25), int(cy - rad * 0.25)), (max(1, int(rad * 0.55)), max(1, int(rad * 0.35))), 0, 0, 360, (110, 104, 122), -1, cv2.LINE_AA)
        boxes.append((cx - rad * 1.1, cy - rad * 0.95, rad * 2.2, rad * 1.8, fwd, 0.78 + 0.2 * min(1, r)))

    if dust > 0:
        tan = np.full_like(img, (92, 140, 196))
        img = cv2.addWeighted(img, 1 - 0.62 * dust, tan, 0.62 * dust, 0)
        img = cv2.GaussianBlur(img, (0, 0), 0.4 + 2.6 * dust)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 14 * dust, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    if seismic:
        img = np.roll(img, int(rng.integers(-4, 5)), axis=1)

    for bx, by, bw, bh, rng_m, conf in boxes:
        if mode == "LIDAR_ONLY":
            conf *= 0.55
        col = (56, 178, 255)
        p0, p1 = (int(bx), int(by)), (int(bx + bw), int(by + bh))
        cv2.rectangle(img, p0, p1, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"ROCK {conf:.2f} {rng_m:.1f}m", (p0[0], max(10, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)
    # corridor overlay + HUD
    cv2.line(img, (W // 2 - 40, H - 1), (W // 2 - 14, int(H * 0.62)), (255, 195, 76), 1, cv2.LINE_AA)
    cv2.line(img, (W // 2 + 40, H - 1), (W // 2 + 14, int(H * 0.62)), (255, 195, 76), 1, cv2.LINE_AA)
    cv2.putText(img, f"{label} {mode} unc {unc:.2f}", (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (235, 231, 236), 1, cv2.LINE_AA)
    return img


def scout_camera(tex: np.ndarray, px_m: float, size_m: float, pose: dict, science: list, perm: list,
                 scanning: bool, dust: float, label: str) -> np.ndarray:
    W, H, foot = 320, 240, 24.0
    s = (W / foot) * px_m  # output px per texture px
    cx_t = (pose["x"] + size_m / 2) / px_m
    cy_t = (pose["y"] + size_m / 2) / px_m
    ang = math.degrees(pose["yaw"]) - 90
    M = cv2.getRotationMatrix2D((cx_t, cy_t), ang, s)
    M[0, 2] += W / 2 - cx_t
    M[1, 2] += H / 2 - cy_t
    img = cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 30, 34))
    img = cv2.flip(img, 0)  # texture row 0 is south; image row 0 must be "ahead"
    if dust > 0:
        tan = np.full_like(img, (92, 140, 196))
        img = cv2.addWeighted(img, 1 - 0.5 * dust, tan, 0.5 * dust, 0)

    def to_px(x: float, y: float) -> tuple[int, int]:
        q = M @ np.array([(x + size_m / 2) / px_m, (y + size_m / 2) / px_m, 1.0])
        return int(q[0]), int(H - 1 - q[1])

    for x, y in science:
        px, py = to_px(x, y)
        if 0 <= px < W and 0 <= py < H:
            cv2.rectangle(img, (px - 12, py - 12), (px + 12, py + 12), (230, 242, 92), 1, cv2.LINE_AA)
            cv2.putText(img, "SCIENCE 0.93", (px - 12, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (230, 242, 92), 1, cv2.LINE_AA)
    for x, y, r in perm:
        px, py = to_px(x, y)
        rr = int(r * W / foot)
        cv2.circle(img, (px, py), rr, (92, 74, 255), 1, cv2.LINE_AA)
    if scanning:
        cv2.drawMarker(img, (W // 2, H // 2), (230, 242, 92), cv2.MARKER_CROSS, 26, 1, cv2.LINE_AA)
        cv2.circle(img, (W // 2, H // 2), 46, (230, 242, 92), 1, cv2.LINE_AA)
        cv2.putText(img, "SLOPE SCAN", (W // 2 - 32, H // 2 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 242, 92), 1, cv2.LINE_AA)
    cv2.putText(img, f"{label} NADIR 12m AGL", (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (235, 231, 236), 1, cv2.LINE_AA)
    return img


def lidar_points(t: Terrain, pose: dict, rocks: list, dust: float, rng: np.random.Generator) -> list:
    """≤1500 body-frame points: 16 rings × 90 azimuths with rock returns and dust backscatter."""
    az = np.linspace(-math.pi, math.pi, 90, endpoint=False)
    el = np.radians(np.linspace(-15, 10, 16))
    A, E = np.meshgrid(az, el)
    sensor_h = 0.8
    rng_g = np.where(E < -0.01, sensor_h / np.tan(-np.minimum(E, -0.01)), np.inf)
    yaw = pose["yaw"]
    r = np.minimum(rng_g, 25.0)
    wx = pose["x"] + r * np.cos(yaw + A)
    wy = pose["y"] + r * np.sin(yaw + A)
    gi = np.clip(((wy + t.size_m / 2) / t.res).astype(int), 0, t.n - 1)
    gj = np.clip(((wx + t.size_m / 2) / t.res).astype(int), 0, t.n - 1)
    gz = t.heights[gi, gj] - pose["z"]
    hit = np.isfinite(rng_g) & (rng_g <= 25.0)
    zb = np.where(hit, gz, np.nan)
    for x, y, rr in rocks:
        dx, dy = x - pose["x"], y - pose["y"]
        dist = math.hypot(dx, dy)
        if dist > 25 or dist < 0.5:
            continue
        b = math.atan2(dy, dx) - yaw
        db = np.angle(np.exp(1j * (A - b)))
        hw = math.atan2(rr, dist)
        on = np.abs(db) < hw
        hz = sensor_h + dist * np.tan(E)
        on &= (hz < rr * 1.4) & ((dist < r) | ~hit)
        r = np.where(on, dist - rr * 0.6, r)
        zb = np.where(on, hz - 0.2, zb)
        hit |= on
    xb = r * np.cos(A)
    yb = r * np.sin(A)
    keep = hit & np.isfinite(zb)
    if dust > 0:
        keep &= rng.random(keep.shape) > 0.35 * dust
    pts = np.stack([xb[keep], yb[keep], zb[keep]], axis=1)
    if dust > 0:
        n = int(260 * dust)
        ra = rng.uniform(0.3, 3.5, n)
        aa = rng.uniform(-math.pi, math.pi, n)
        pts = np.vstack([pts, np.stack([ra * np.cos(aa), ra * np.sin(aa), rng.uniform(-0.3, 1.2, n)], axis=1)])
    return np.round(pts[:1500], 2).tolist()


def map_png_b64(t: Terrain, cls: np.ndarray, seen_t: np.ndarray, now: float) -> str:
    known = cls > 0
    emin, emax = float(t.heights.min()), float(t.heights.max())
    R = np.where(known, np.round(255 * (t.heights - emin) / (emax - emin + 1e-6)), 0)
    G = cls.astype(np.float32) * 40
    B = np.where(known, np.clip(255 * (1 - (now - seen_t) / 6.0), 0, 255), 0)
    rgb = np.dstack([R, G, B]).astype(np.uint8)  # row 0 = south (protocol §5)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("ascii")
