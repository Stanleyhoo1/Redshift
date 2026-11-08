"""The planet's look in the camera renders: procedural textures, sky, sun, haze and materials.

Visual only. Nothing here changes the physics, the contacts, the LiDAR or any sensor: textures and
materials are colours, the extra decoration geoms (world.py) are massless, never collide and sit in
geom groups the LiDAR does not scan (3: rovers, 4: lander decoration).

Made once per seed at startup with numpy (cached, about 0.3 s):
  regolith   tileable 5 m ground texture (one copy per heightfield tile, so seamless across tiles):
             multi-octave value noise for dusty ochre-rust soil with darker and greyer patches, fine
             grain, and scattered pebbles. A few variants share the same border band but differ
             inside, so the 5 m repeat does not read as a pattern.
  rock       grainy basalt for rocks and boulders
  solar      dark blue cells with thin silver grid lines (rover decks)
  sky        cube map for a dense super-Earth atmosphere: bright apricot haze at the horizon, rust
             and deep brown toward the zenith, a low warm sun with a glow around it.
Lighting: a warm low sun (casts shadows), a weak dusty sky fill, a dim headlight; linear distance
fog in the horizon colour (aerial perspective) and MuJoCo's horizon haze. Colours of the science
targets, spires and grazers match the web dashboard (mineral #b48cff, ice #a8d4ff, vent #ff8a3d,
spires violet, grazers greenish and slightly bioluminescent).
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

# ---- sun and atmosphere
SUN_AZ_DEG = 210.0                 # direction TO the sun (0 = east, 90 = north): south-west
SUN_EL_DEG = 30.0                  # low warm sun: long shadows
_az, _el = math.radians(SUN_AZ_DEG), math.radians(SUN_EL_DEG)
TO_SUN = np.array([math.cos(_el) * math.cos(_az), math.cos(_el) * math.sin(_az), math.sin(_el)])
SUN_DIR = -TO_SUN                  # the light's direction of travel
SUN_RGB = (0.9, 0.84, 0.75)
SKY_FILL_RGB = (0.16, 0.12, 0.10)  # weak light from the dusty sky (no shadows)
HEAD_AMBIENT = (0.34, 0.29, 0.26)   # a dense, dusty atmosphere lights the shadows
HEAD_DIFFUSE = (0.10, 0.09, 0.08)
HORIZON_RGB = (0.83, 0.64, 0.51)   # sky at the horizon = haze = distance fog
FOG_START_M = 12.0                 # distance haze starts here ...
FOG_END_M = 140.0                  # ... and reaches about 95% of HAZE_MAX here (exponential)
HAZE_MAX = 0.9
SHADOW_SIZE = 2048

REGOLITH_PX = 1024
REGOLITH_VARIANTS = 4
ROCK_PX = 256
SOLAR_PX = 128
SKY_PX = 512
WATER_PX = 256

TARGET_EMISSION = 0.35
_CACHE: OrderedDict = OrderedDict()


def _hex(rgb) -> str:
    return " ".join(f"{float(c):.3f}" for c in rgb)


# ====================================================================== noise
def _tile_noise(px: int, cells: int, rng) -> np.ndarray:
    """Tileable value noise: a random cells x cells grid, smoothly interpolated to px x px, wrapping."""
    g = rng.random((cells, cells)).astype(np.float32)
    x = np.arange(px, dtype=np.float32) * (cells / px)
    i0 = np.floor(x).astype(np.int64) % cells
    i1 = (i0 + 1) % cells
    f = x - np.floor(x)
    f = f * f * (3.0 - 2.0 * f)
    a = g[i0][:, i0] * (1 - f)[None, :] + g[i0][:, i1] * f[None, :]
    b = g[i1][:, i0] * (1 - f)[None, :] + g[i1][:, i1] * f[None, :]
    return a * (1 - f)[:, None] + b * f[:, None]


def _fbm(px: int, cells0: int, octaves: int, rng, gain: float = 0.55) -> np.ndarray:
    out = np.zeros((px, px), np.float32)
    amp, tot, cells = 1.0, 0.0, cells0
    for _ in range(octaves):
        if cells > px // 2:
            break
        out += amp * (_tile_noise(px, cells, rng) - 0.5)
        tot += amp
        amp *= gain
        cells *= 2
    return out / max(tot, 1e-6)


def _discs(h: int, w: int, xs, ys, rs, alphas):
    """Soft discs (pebbles) at sub-pixel centres, all at once: per pixel the summed coverage and the
    coverage-weighted index sums (wrapping at the edges, so tileable textures stay tileable).
    Returns (coverage (h, w), list of (Y, X, a) per disc-size group for colour accumulation)."""
    acc = np.zeros(h * w, np.float32)
    parts = []
    ks = np.ceil(np.asarray(rs) + 1.0).astype(np.int64)
    for k in np.unique(ks):
        sel = np.flatnonzero(ks == k)
        oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
        cx, cy, r = xs[sel], ys[sel], rs[sel]
        ix, iy = np.floor(cx).astype(np.int64), np.floor(cy).astype(np.int64)
        dx = (ix[:, None, None] + ox[None] + 0.5) - cx[:, None, None]
        dy = (iy[:, None, None] + oy[None] + 0.5) - cy[:, None, None]
        a = (np.clip(r[:, None, None] + 0.5 - np.hypot(dx, dy), 0.0, 1.0) * alphas[sel][:, None, None]).astype(np.float32)
        flat = (((iy[:, None, None] + oy[None]) % h) * w + ((ix[:, None, None] + ox[None]) % w)).ravel()
        av = a.ravel()
        np.add.at(acc, flat, av)
        parts.append((sel, flat, a.reshape(len(sel), -1)))
    return acc.reshape(h, w), parts


def _paint_discs(img: np.ndarray, xs, ys, rs, rgbs, alphas) -> np.ndarray:
    """Composite soft coloured discs over img (h, w, 3) float."""
    h, w = img.shape[:2]
    if len(xs) == 0:
        return img
    cov, parts = _discs(h, w, np.asarray(xs, np.float64), np.asarray(ys, np.float64), np.asarray(rs, np.float64),
                        np.asarray(alphas, np.float32))
    col = np.zeros((h * w, 3), np.float32)
    rgbs = np.asarray(rgbs, np.float32)
    for sel, flat, a in parts:
        for c in range(3):
            np.add.at(col[:, c], flat, (a * rgbs[sel][:, c:c + 1]).ravel())
    col = col.reshape(h, w, 3) / np.maximum(cov, 1e-6)[..., None]
    A = np.clip(cov, 0.0, 1.0)[..., None]
    return img * (1 - A) + col * A


def _shade_discs(img: np.ndarray, xs, ys, rs, alphas, factor: float) -> np.ndarray:
    """Darken (factor < 1) or lighten (factor > 1) img under soft discs."""
    h, w = img.shape[:2]
    if len(xs) == 0:
        return img
    cov, _ = _discs(h, w, np.asarray(xs, np.float64), np.asarray(ys, np.float64), np.asarray(rs, np.float64),
                    np.asarray(alphas, np.float32))
    A = np.clip(cov, 0.0, 1.0)[..., None]
    return img * (1 + (factor - 1) * A)


# ====================================================================== textures
def _regolith(seed: int, px: int, variants: int) -> list:
    rng = np.random.default_rng(1000 + seed)
    tint = 1.0 + 0.04 * (rng.random(3) - 0.5)                  # each planet a slightly different soil
    low = _fbm(px, 2, 3, rng)                                  # patches 1 to 2.5 m
    mid = _fbm(px, 8, 3, rng)                                  # 15 to 60 cm
    grain = _fbm(px, 64, 4, rng, 0.6)                          # 1 to 8 cm
    rust = _fbm(px, 3, 3, rng)                                 # redder vs greyer soil
    dark = np.array([0.35, 0.30, 0.26], np.float32)
    light = np.array([0.72, 0.63, 0.55], np.float32)
    grey = np.array([0.55, 0.53, 0.51], np.float32)
    redd = np.array([0.64, 0.48, 0.39], np.float32)

    def colour(lowf, rustf):
        b = np.clip(0.52 + 1.15 * lowf + 0.55 * mid + 0.45 * grain, 0.0, 1.0)[..., None]
        base = dark + (light - dark) * b
        r = np.clip(0.5 + 1.6 * rustf, 0.0, 1.0)[..., None]
        hue = grey + (redd - grey) * r
        return base * (hue / hue.mean(axis=-1, keepdims=True)) * tint

    base = colour(low, rust)
    # pebbles shared by every variant (they wrap, so the texture stays tileable): mostly tiny grains,
    # a few bigger ones with a soft shadow and a lit top
    n = int(1400 * (px / 1024.0) ** 2)
    px_, py_ = rng.random(n) * px, rng.random(n) * px
    pr = px / 1024.0 * (0.6 + 3.2 * rng.random(n) ** 4)
    c = rng.random(n)
    prgb = np.where((c < 0.55)[:, None], [0.30, 0.26, 0.23], np.where((c < 0.8)[:, None], [0.60, 0.54, 0.47],
                                                                       [0.45, 0.33, 0.25]))
    prgb = prgb * (0.85 + 0.3 * rng.random(n))[:, None]
    pa = 0.3 + 0.35 * rng.random(n)
    big = pr > 1.6
    # inside-only differences per variant: a window that is 0 on the border band, 1 inside
    u = (np.arange(px) + 0.5) / px
    wv = np.clip(np.minimum(u, 1.0 - u) / 0.16, 0.0, 1.0)
    wv = wv * wv * (3 - 2 * wv)
    win = (wv[:, None] * wv[None, :])
    out = []
    for k in range(variants):
        vr = np.random.default_rng(2000 + 97 * seed + k)
        if k == 0:
            img = base.copy()
        else:
            dlow = _fbm(px, 2, 3, vr) * 1.1
            drust = _fbm(px, 3, 2, vr) * 1.2
            img = colour(low + win * dlow, rust + win * drust)
        img = _shade_discs(img, px_[big] + 0.35 * pr[big], py_[big] + 0.35 * pr[big], pr[big] * 1.1, 0.5 * pa[big], 0.7)
        img = _paint_discs(img, px_, py_, pr, prgb, pa)
        img = _paint_discs(img, px_[big] - 0.3 * pr[big], py_[big] - 0.3 * pr[big], 0.45 * pr[big],
                           np.minimum(prgb[big] * 1.35, 1.0), 0.5 * pa[big])
        if k > 0:                                              # a few extra pebbles well inside
            m = n // 3
            ex, ey = (0.2 + 0.6 * vr.random(m)) * px, (0.2 + 0.6 * vr.random(m)) * px
            er = px / 1024.0 * (0.6 + 3.5 * vr.random(m) ** 4)
            ergb = np.array([0.30, 0.26, 0.23]) * (0.85 + 0.3 * vr.random(m))[:, None]
            img = _paint_discs(img, ex, ey, er, ergb, 0.3 + 0.35 * vr.random(m))
        out.append(np.clip(img * 255.0, 0, 255).astype(np.uint8))
    return out


def _rock(seed: int, px: int) -> np.ndarray:
    rng = np.random.default_rng(3000 + seed)
    a = _fbm(px, 4, 5, rng, 0.6)
    spots = _fbm(px, 32, 2, rng)
    b = np.clip(0.55 + 1.2 * a + 0.5 * spots, 0.0, 1.0)[..., None]
    img = np.array([0.20, 0.17, 0.15]) + np.array([0.36, 0.31, 0.27]) * b
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _water(px: int) -> np.ndarray:
    """Deep blue liquid with faint ripples and a few glints (stays clearly blue for the camera check)."""
    rng = np.random.default_rng(7)
    rip = _fbm(px, 6, 4, rng, 0.6)
    streak = _tile_noise(px, 24, rng)
    streak = np.repeat(streak[:, ::8], 8, axis=1)[:, :px]      # stretched: ripples along one direction
    b = np.clip(0.82 + 0.5 * rip + 0.18 * (streak - 0.5), 0.55, 1.15)[..., None]
    img = np.array([0.55, 0.78, 1.0]) * b
    glint = (rip > np.quantile(rip, 0.995))[..., None]
    img = np.where(glint, np.array([0.9, 0.95, 1.0]), img)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _solar(px: int) -> np.ndarray:
    img = np.empty((px, px, 3), np.float32)
    img[:] = (0.05, 0.08, 0.17)
    y = np.arange(px)
    shade = 0.85 + 0.15 * np.sin(y / px * math.pi)[:, None]
    img *= shade[..., None]
    cell = px // 8
    lines = (y % cell) < max(1, px // 64)
    img[lines, :] = (0.55, 0.57, 0.62)
    img[:, lines] = (0.55, 0.57, 0.62)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def sky_colour(d: np.ndarray) -> np.ndarray:
    """Sky colour (0..1) for unit view directions d (..., 3) in world coordinates (z up)."""
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1.0, 1.0)))
    e = np.maximum(el, 0.0)
    zen = np.array([0.17, 0.08, 0.07])
    mid = np.array([0.42, 0.24, 0.17])
    hor = np.array(HORIZON_RGB)
    f1 = np.exp(-e / 6.0)[..., None]                         # bright haze band at the horizon
    f2 = np.exp(-e / 30.0)[..., None]
    c = zen * (1 - f2) + mid * f2 * (1 - f1) + hor * f1
    below = np.clip(-el / 8.0, 0.0, 1.0)[..., None]          # under the horizon: dusty ground haze
    c = c * (1 - below) + np.array([0.46, 0.31, 0.23]) * below
    cosg = np.clip((d * TO_SUN).sum(-1), -1.0, 1.0)
    g = np.degrees(np.arccos(cosg))
    glow = (0.55 * np.exp(-g / 3.0) + 0.22 * np.exp(-g / 14.0))[..., None]
    c = c + glow * np.array([1.0, 0.72, 0.45])
    disk = np.clip((0.75 - g) / 0.15, 0.0, 1.0)[..., None]  # the sun's disc, about 1.5 deg across
    c = c * (1 - disk) + np.array([1.0, 0.95, 0.82]) * disk
    return np.clip(c, 0.0, 1.0)


def _sky(px: int) -> np.ndarray:
    """Cube map in MuJoCo's face order (+x, -x, up, down, -y, +y), as found by rendering a
    direction-coded test cube: side faces have row 0 at the top and column 0 on the right."""
    r = (np.arange(px) + 0.5) / px
    t = (1 - 2 * r)[:, None] * np.ones((1, px))
    s = np.ones((px, 1)) * (-1 + 2 * r)[None, :]
    one = np.ones((px, px))
    faces = [np.stack([one, s, t], -1), np.stack([-one, -s, t], -1), np.stack([s, t, one], -1),
             np.stack([s, -t, -one], -1), np.stack([s, -one, t], -1), np.stack([-s, one, t], -1)]
    out = []
    for f in faces:
        d = f / np.linalg.norm(f, axis=-1, keepdims=True)
        out.append(sky_colour(d))
    return np.clip(np.concatenate(out, 0) * 255.0, 0, 255).astype(np.uint8)


def textures(seed: int) -> dict:
    """{texture name: uint8 array (h, w, 3)} for this seed (cached)."""
    key = int(seed)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    out = {f"regolith{k}": img for k, img in enumerate(_regolith(key, REGOLITH_PX, REGOLITH_VARIANTS))}
    out["rocktex"] = _rock(key, ROCK_PX)
    out["watertex"] = _water(WATER_PX)
    out["solartex"] = _solar(SOLAR_PX)
    sky = _CACHE.get("sky")
    if sky is None:
        sky = _CACHE["sky"] = _sky(SKY_PX)
    out["sky"] = sky
    _CACHE[key] = out
    while len(_CACHE) > 5:
        old = next(k for k in _CACHE if k != "sky")
        del _CACHE[old]
    return out


# ====================================================================== MJCF pieces
def visual_xml(cfg) -> str:
    """The <visual> block. Fog distances are set in metres by paint() (MJCF wants them relative to
    the model's extent, known only after compiling)."""
    span = cfg.map_size
    return (f'<visual>'
            f'<global offwidth="1280" offheight="720" elevation="-25" azimuth="135"/>'
            f'<map znear="0.02" zfar="{span * 3:.0f}" haze="0.12" fogstart="0.2" fogend="1.5" shadowclip="1.0"/>'
            f'<quality shadowsize="{SHADOW_SIZE}" numslices="20" numstacks="12"/>'
            f'<headlight ambient="{_hex(HEAD_AMBIENT)}" diffuse="{_hex(HEAD_DIFFUSE)}" specular="0.04 0.04 0.04"/>'
            f'<rgba haze="{_hex(HORIZON_RGB)} 1" fog="{_hex(HORIZON_RGB)} 1"/>'
            f'</visual>')


def asset_xml(cfg) -> str:
    """Textures (filled in by paint()) and materials."""
    rp, kp, sp, yp = REGOLITH_PX, ROCK_PX, SOLAR_PX, SKY_PX
    parts = [f'<texture name="sky" type="skybox" builtin="flat" rgb1="0.5 0.3 0.2" width="{yp}" height="{yp}"/>']
    for k in range(REGOLITH_VARIANTS):
        parts.append(f'<texture name="regolith{k}" type="2d" builtin="flat" rgb1="0.55 0.38 0.27" '
                     f'width="{rp}" height="{rp}"/>')
        parts.append(f'<material name="ground{k}" texture="regolith{k}" texrepeat="1 1" texuniform="false" '
                     f'specular="0.06" shininess="0.15" reflectance="0"/>')
    parts += [
        f'<texture name="rocktex" type="2d" builtin="flat" rgb1="0.35 0.3 0.27" width="{kp}" height="{kp}"/>',
        f'<texture name="solartex" type="2d" builtin="flat" rgb1="0.05 0.08 0.17" width="{sp}" height="{sp}"/>',
        f'<texture name="watertex" type="2d" builtin="flat" rgb1="0.6 0.8 1" width="{WATER_PX}" height="{WATER_PX}"/>',
        '<material name="ground" texture="regolith0" texrepeat="1 1" texuniform="false" specular="0.06" shininess="0.15"/>',
        '<material name="rock" texture="rocktex" texrepeat="1 1" rgba="1 1 1 1" specular="0.12" shininess="0.2"/>',
        '<material name="solar" texture="solartex" texrepeat="1 1" specular="0.9" shininess="0.95"/>',
        '<material name="metal" rgba="0.74 0.74 0.77 1" specular="0.7" shininess="0.75"/>',
        '<material name="darkmetal" rgba="0.17 0.17 0.19 1" specular="0.5" shininess="0.6"/>',
        '<material name="rubber" rgba="0.13 0.12 0.11 1" specular="0.08" shininess="0.1"/>',
        '<texture name="tyretex" type="2d" builtin="checker" rgb1="0.09 0.09 0.09" rgb2="0.16 0.15 0.14" width="8" height="8"/>',
        '<material name="tyre" texture="tyretex" texrepeat="12 1" specular="0.1" shininess="0.1"/>',
        '<material name="lens" rgba="0.02 0.03 0.05 1" specular="1" shininess="1"/>',
        '<material name="pad" rgba="0.70 0.70 0.72 1" specular="0.5" shininess="0.6"/>',
        '<material name="marking" rgba="1.0 0.48 0.27 1" specular="0.3" shininess="0.4" emission="0.15"/>',
        '<material name="beacon" rgba="1.0 0.35 0.15 1" emission="0.9"/>',
        # the spire capsules (what LiDAR sees) stay magenta, far from every target colour even where their
        # edges blend with the soil (machine vision frames see them); the violet faceted crystal shells
        # around them (group 4) are what the camera shows
        '<material name="spire" rgba="0.88 0.22 0.5 1" specular="0.6" shininess="0.8" emission="0.15"/>',
        '<material name="spire_core" rgba="1.0 0.55 0.8 1" specular="0.6" shininess="0.8" emission="0.4"/>',
        '<material name="life" rgba="0.20 0.58 0.42 1" specular="0.5" shininess="0.75" emission="0.2"/>',
        '<material name="life_dark" rgba="0.07 0.29 0.23 1" specular="0.3" shininess="0.5" emission="0.12"/>',
        '<material name="water" texture="watertex" texrepeat="6 6" texuniform="true" rgba="0.10 0.28 0.55 0.88" '
        'specular="1" shininess="0.98"/>',
        '<material name="crystal" rgba="0.48 0.24 0.92 1" specular="1" shininess="1" emission="0.1"/>',
        # natural looks of the science targets (visual only; the protocol colours as accents)
        '<material name="mound" texture="regolith0" texrepeat="2 2" rgba="0.78 0.70 0.64 1" specular="0.05" shininess="0.1"/>',
        '<material name="vent_mouth" rgba="0.03 0.02 0.02 1" specular="0" shininess="0"/>',
        '<material name="vent_rim" rgba="1.0 0.541 0.239 1" specular="0.2" shininess="0.3" emission="0.3"/>',
        '<material name="vent_crust" rgba="0.62 0.42 0.26 1" specular="0.1" shininess="0.2"/>',
        '<material name="outcrop" texture="rocktex" rgba="0.62 0.6 0.64 1" specular="0.2" shininess="0.3"/>',
        '<material name="mineral_vein" rgba="0.706 0.549 1.0 1" specular="1" shininess="1" emission="0.3"/>',
        '<material name="ice" rgba="0.66 0.78 0.92 1" specular="1" shininess="1" emission="0.04"/>',
        '<material name="ice_chunk" rgba="0.659 0.831 1.0 1" specular="1" shininess="1" emission="0.12"/>',
        '<material name="trace_film" rgba="0.30 0.36 0.14 1" specular="0.7" shininess="0.85" emission="0.05"/>',
        '<material name="trace_spot" rgba="0.55 1.0 0.15 1" specular="0.4" shininess="0.6" emission="0.4"/>',
        '<material name="life_joint" rgba="0.08 0.30 0.22 1" specular="0.3" shininess="0.5" emission="0.1"/>',
        '<material name="life_glow" rgba="0.3 0.9 0.55 1" emission="0.7"/>',
    ]
    return "".join(parts)


def target_material_xml(kind: str, rgba: str) -> str:
    """Science target markers glow a little in their dashboard colour (also keeps the colour even on the
    shaded side, which camera target spotting relies on)."""
    return (f'<material name="target_{kind}" rgba="{rgba}" specular="0.15" shininess="0.3" '
            f'emission="{TARGET_EMISSION}"/>')


def lights_xml() -> str:
    """Sun (casts shadows; its position is upstream of the map so the viewer's shadow box covers it;
    camera.py moves it in front of each camera) and a weak sky fill from the other side."""
    p = -SUN_DIR * 90.0
    fill = np.array([-SUN_DIR[0], -SUN_DIR[1], -0.8])
    fill /= np.linalg.norm(fill)
    return (f'<light name="sun" directional="true" pos="{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}" '
            f'dir="{SUN_DIR[0]:.4f} {SUN_DIR[1]:.4f} {SUN_DIR[2]:.4f}" diffuse="{_hex(SUN_RGB)}" '
            f'specular="0.35 0.3 0.25" ambient="0.06 0.04 0.03" castshadow="true"/>'
            f'<light name="skyfill" directional="true" pos="0 0 60" dir="{fill[0]:.4f} {fill[1]:.4f} {fill[2]:.4f}" '
            f'diffuse="{_hex(SKY_FILL_RGB)}" specular="0 0 0" castshadow="false"/>')


def ground_material(i0: int, j0: int, seed: int) -> str:
    """Which regolith variant a heightfield tile uses (a fixed hash, so the pattern does not repeat)."""
    h = (i0 * 73856093) ^ (j0 * 19349663) ^ (seed * 83492791)
    return f"ground{(h >> 3) % REGOLITH_VARIANTS}"


def crystal_mesh(name: str, length: float, radius: float, tip: float) -> str:
    """A faceted crystal: hexagonal prism (faces just outside a capsule of this radius) along local z
    from 0 to length, with a pyramidal tip of height `tip`. Every face has its own vertices, so the
    facets render sharp."""
    R = radius * 1.08 / math.cos(math.pi / 6)          # circumradius: the flat faces clear the capsule
    ring = [(R * math.cos(k * math.pi / 3), R * math.sin(k * math.pi / 3)) for k in range(6)]
    v, f = [], []

    def tri(a, b, c):
        n = len(v)
        v.extend([a, b, c])
        f.append((n, n + 1, n + 2))

    top, z0 = length, -0.02
    for k in range(6):
        (x0, y0), (x1, y1) = ring[k], ring[(k + 1) % 6]
        tri((x0, y0, z0), (x1, y1, z0), (x1, y1, top))
        tri((x0, y0, z0), (x1, y1, top), (x0, y0, top))
        tri((x0, y0, top), (x1, y1, top), (0.0, 0.0, top + tip))
        tri((0.0, 0.0, z0), (x1, y1, z0), (x0, y0, z0))
    vs = " ".join(f"{a:.4f} {b:.4f} {c:.4f}" for (a, b, c) in v)
    fs = " ".join(f"{a} {b} {c}" for (a, b, c) in f)
    return f'<mesh name="{name}" vertex="{vs}" face="{fs}"/>'


def paint(model, cfg) -> None:
    """Fill the procedural textures into the compiled model and set the fog in metres."""
    import mujoco

    for name, img in textures(cfg.seed).items():
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, name)
        if tid < 0:
            continue
        w, h, nc = int(model.tex_width[tid]), int(model.tex_height[tid]), int(model.tex_nchannel[tid])
        if img.shape[0] != h or img.shape[1] != w or nc != 3:
            continue
        adr = int(model.tex_adr[tid])
        model.tex_data[adr:adr + w * h * 3] = img.reshape(-1)
    ext = max(float(model.stat.extent), 1.0)
    model.vis.map.fogstart = FOG_START_M / ext
    model.vis.map.fogend = FOG_END_M / ext
    # the viewer's shadow box: the whole map around the sun's position line
    model.vis.map.shadowclip = (0.75 * cfg.map_size) / ext
