"""Seeded biomes (v7, `biomes` setting): each world gets a primary ground type, a secondary one in a few
soft regions, its own colours and its own terrain mix.

Ground truth for world generation and looks only. The rovers never read the biome: it changes how many
mesas, dunes, vents or plants a world gets and how the ground looks to the camera and on the map, nothing
else. With `biomes` off nothing here runs: `weight()` is exactly 1.0 and every world is byte for byte the
one it was before.

Streams: the pick and the region field come from default_rng([seed, 7112]) (K_BIOMES), so turning biomes
on never moves a draw of any other stream. Counts change only through `count()`, after the generator's own
draws, so the features that remain keep their places.
"""
from __future__ import annotations

import math

import numpy as np

K_BIOMES = 7112
K_LIFE = 7113                  # the lifeform variety stream (lifeforms track)
PRIMARY_SHARE = 0.75           # feature weights blend 75% primary, 25% secondary
REGION_COVER = (0.25, 0.40)    # share of the sector the secondary biome covers
MAP_CELL_M = 4.0               # features.biome_map cell size (about)

FEATURES = ("mesas", "canyon", "dunes", "boulder_fields", "pits", "geysers", "ice_sheets", "flora")


def _w(mesas, canyon, dunes, boulder_fields, pits, geysers, ice_sheets, flora) -> dict:
    return dict(zip(FEATURES, (mesas, canyon, dunes, boulder_fields, pits, geysers, ice_sheets, flora)))


# Camera colours (0..1, the regolith texture's ramp): dark and light ends, a neutral and a tinted hue that
# noise mixes, three pebble colours, rock (base, span), the light (sun, ambient) and the sky (zenith, mid,
# horizon = haze, below).
# Every one is kept clear of the camera checks: far in chroma from the target markers (violet mineral,
# light blue ice, orange vent, lime trace) and never "blue liquid" (blue at most 1.6 x red).
BIOMES: dict = {
    "basalt": {
        "name": "Basalt fields", "primary": True,
        "weights": _w(1.0, 1.0, 0.3, 1.5, 1.8, 1.2, 0.2, 0.5),
        "life": {"crawler": 3, "skitterer": 2, "grazer": 1},
        "soil": {"dark": (0.15, 0.15, 0.16), "light": (0.48, 0.48, 0.50), "grey": (0.41, 0.42, 0.43),
                 "tint": (0.41, 0.42, 0.45),
                 "pebbles": ((0.10, 0.10, 0.11), (0.44, 0.45, 0.48), (0.30, 0.29, 0.29))},
        "rock": ((0.12, 0.12, 0.125), (0.31, 0.31, 0.33)),
        "light": {"sun": (0.86, 0.86, 0.87), "ambient": (0.29, 0.30, 0.32)},
        "sky": {"zen": (0.10, 0.11, 0.15), "mid": (0.30, 0.31, 0.37), "hor": (0.63, 0.64, 0.69),
                "below": (0.34, 0.35, 0.38)},
    },
    "verdigris": {
        "name": "Verdigris lowlands", "primary": True,
        "weights": _w(0.6, 0.6, 0.3, 0.6, 0.4, 0.4, 0.0, 2.0),
        "life": {"grazer": 3, "drifter": 2, "crawler": 1},
        "soil": {"dark": (0.22, 0.25, 0.18), "light": (0.56, 0.61, 0.54), "grey": (0.49, 0.51, 0.48),
                 "tint": (0.40, 0.52, 0.40),
                 "pebbles": ((0.17, 0.19, 0.14), (0.55, 0.58, 0.52), (0.34, 0.38, 0.32))},
        "rock": ((0.17, 0.19, 0.14), (0.32, 0.36, 0.29)),
        "light": {"sun": (0.88, 0.88, 0.81), "ambient": (0.30, 0.32, 0.30)},
        "sky": {"zen": (0.10, 0.14, 0.14), "mid": (0.30, 0.36, 0.34), "hor": (0.66, 0.72, 0.67),
                "below": (0.36, 0.40, 0.37)},
    },
    "frost": {
        "name": "Frost plains", "primary": True,
        "weights": _w(0.5, 0.7, 0.6, 0.6, 1.0, 0.3, 2.5, 0.0),
        "life": {"burrower": 3, "skitterer": 2},
        "soil": {"dark": (0.58, 0.60, 0.62), "light": (0.95, 0.96, 0.97), "grey": (0.79, 0.80, 0.81),
                 "tint": (0.76, 0.78, 0.82),
                 "pebbles": ((0.36, 0.37, 0.40), (0.92, 0.93, 0.94), (0.55, 0.57, 0.60))},
        "rock": ((0.23, 0.235, 0.25), (0.40, 0.41, 0.43)),
        "light": {"sun": (0.88, 0.90, 0.93), "ambient": (0.34, 0.35, 0.38)},
        "sky": {"zen": (0.18, 0.20, 0.26), "mid": (0.44, 0.48, 0.54), "hor": (0.80, 0.83, 0.86),
                "below": (0.55, 0.57, 0.60)},
    },
    "ash": {
        "name": "Ash wastes", "primary": True,
        "weights": _w(0.6, 1.0, 1.2, 0.8, 1.2, 2.0, 0.0, 0.3),
        "life": {"drifter": 3, "skitterer": 2, "burrower": 1},
        "soil": {"dark": (0.25, 0.24, 0.27), "light": (0.66, 0.64, 0.69), "grey": (0.56, 0.55, 0.57),
                 "tint": (0.57, 0.53, 0.61),
                 "pebbles": ((0.18, 0.17, 0.19), (0.64, 0.62, 0.66), (0.40, 0.38, 0.42))},
        "rock": ((0.15, 0.14, 0.16), (0.32, 0.31, 0.34)),
        "light": {"sun": (0.88, 0.84, 0.85), "ambient": (0.31, 0.29, 0.32)},
        "sky": {"zen": (0.13, 0.11, 0.15), "mid": (0.36, 0.31, 0.38), "hor": (0.71, 0.66, 0.72),
                "below": (0.40, 0.38, 0.41)},
    },
    "salt": {
        "name": "Salt flats", "primary": True,
        "weights": _w(1.6, 1.2, 2.0, 0.5, 0.3, 0.0, 0.5, 0.2),
        "life": {"skitterer": 3, "burrower": 2, "grazer": 1},
        "soil": {"dark": (0.56, 0.55, 0.52), "light": (0.95, 0.93, 0.88), "grey": (0.82, 0.81, 0.79),
                 "tint": (0.88, 0.85, 0.77),
                 "pebbles": ((0.30, 0.31, 0.33), (0.95, 0.93, 0.88), (0.44, 0.45, 0.47))},
        "rock": ((0.19, 0.19, 0.20), (0.32, 0.33, 0.35)),
        "light": {"sun": (0.92, 0.90, 0.85), "ambient": (0.33, 0.32, 0.31)},
        "sky": {"zen": (0.20, 0.17, 0.15), "mid": (0.48, 0.42, 0.36), "hor": (0.86, 0.81, 0.72),
                "below": (0.60, 0.57, 0.52)},
    },
    "obsidian": {
        "name": "Obsidian badlands", "primary": True,
        "weights": _w(1.3, 1.4, 0.2, 1.6, 1.0, 0.8, 0.0, 1.0),
        "life": {"crawler": 2, "drifter": 2, "burrower": 1},
        "soil": {"dark": (0.065, 0.06, 0.065), "light": (0.30, 0.29, 0.33), "grey": (0.22, 0.22, 0.23),
                 "tint": (0.25, 0.23, 0.27),
                 "pebbles": ((0.05, 0.05, 0.05), (0.36, 0.35, 0.39), (0.20, 0.19, 0.22))},
        "rock": ((0.06, 0.06, 0.06), (0.22, 0.21, 0.25)),
        "light": {"sun": (0.88, 0.85, 0.89), "ambient": (0.30, 0.28, 0.32)},
        "sky": {"zen": (0.08, 0.06, 0.10), "mid": (0.25, 0.20, 0.28), "hor": (0.54, 0.49, 0.57),
                "below": (0.28, 0.25, 0.30)},
    },
    "rust": {                                                  # today's look: only ever a secondary region
        "name": "Rust plains", "primary": False,
        "weights": _w(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "life": {"grazer": 2, "skitterer": 1},
        "soil": {"dark": (0.35, 0.30, 0.26), "light": (0.72, 0.63, 0.55), "grey": (0.55, 0.53, 0.51),
                 "tint": (0.64, 0.48, 0.39),
                 "pebbles": ((0.30, 0.26, 0.23), (0.60, 0.54, 0.47), (0.45, 0.33, 0.25))},
        "rock": ((0.20, 0.17, 0.15), (0.36, 0.31, 0.27)),
        "light": {"sun": (0.9, 0.84, 0.75), "ambient": (0.34, 0.29, 0.26)},
        "sky": {"zen": (0.17, 0.08, 0.07), "mid": (0.42, 0.24, 0.17), "hor": (0.83, 0.64, 0.51),
                "below": (0.46, 0.31, 0.23)},
    },
}
IDS = tuple(BIOMES)
PRIMARY_IDS = tuple(k for k, b in BIOMES.items() if b["primary"])


def enabled(cfg) -> bool:
    return bool(getattr(cfg, "biomes", False))


def pick(cfg):
    """(primary, secondary) biome ids for this seed, or None with biomes off. The primary is never rust."""
    if not enabled(cfg):
        return None
    rng = np.random.default_rng([int(cfg.seed), K_BIOMES])
    primary = PRIMARY_IDS[int(rng.integers(len(PRIMARY_IDS)))]
    others = [k for k in IDS if k != primary]
    secondary = others[int(rng.integers(len(others)))]
    return primary, secondary


def _smooth(e0: float, e1: float, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Region:
    """The secondary biome's share of the ground: a smooth, low-frequency field (a few soft blobs warped by
    long waves), 0 in the primary biome and 1 deep in the secondary one, about 25 to 40% of the sector over
    one half. The lander's surroundings stay primary."""

    def __init__(self, cfg):
        rng = np.random.default_rng([int(cfg.seed), K_BIOMES, 1])
        size = float(cfg.map_size)
        half = size / 2.0
        n = int(rng.integers(2, 5))
        self.blobs = [(float(rng.uniform(-half, half)), float(rng.uniform(-half, half)),
                       float(rng.uniform(0.16, 0.28)) * size, float(rng.uniform(0.7, 1.0))) for _ in range(n)]
        self.waves = [(float(rng.uniform(0.0, 2.0 * math.pi)), float(rng.uniform(0.35, 0.7)) * size,
                       float(rng.uniform(0.0, 2.0 * math.pi))) for _ in range(3)]
        self.lander_r = 0.1 * size + 6.0
        cover = float(rng.uniform(*REGION_COVER))
        g = np.linspace(-half, half, 97)
        GX, GY = np.meshgrid(g, g)
        f = self._raw(GX, GY)
        self.t = float(np.quantile(f, 1.0 - cover))
        self.band = 0.08 * float(f.max() - f.min()) + 1e-6

    def _raw(self, X, Y):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        f = np.zeros(np.broadcast(X, Y).shape)
        for cx, cy, r, a in self.blobs:
            f = f + a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * r * r))
        for ang, wl, ph in self.waves:
            f = f + 0.16 * np.sin(2.0 * math.pi * (X * math.cos(ang) + Y * math.sin(ang)) / wl + ph)
        return f - 0.9 * np.exp(-(X * X + Y * Y) / (2.0 * self.lander_r ** 2))

    def weight_at(self, x, y):
        """Secondary share 0..1 at (x, y) (arrays broadcast)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        keep = _smooth(0.5 * self.lander_r, self.lander_r, np.hypot(x, y))      # the lander stays primary
        return _smooth(self.t - self.band, self.t + self.band, self._raw(x, y)) * keep


def attach(terrain) -> None:
    """Terrain hook, before any v7 feature is made: sets terrain.biome and terrain.biome_region."""
    terrain.biome = None
    terrain.biome_region = None
    got = pick(terrain.cfg)
    if got is None:
        return
    primary, secondary = got
    terrain.biome = {"id": primary, "name": BIOMES[primary]["name"], "secondary": secondary}
    terrain.biome_region = Region(terrain.cfg)


def biome_id_at(terrain, x: float, y: float):
    """The biome id at (x, y): the secondary one where its share is half or more. None with biomes off."""
    b = getattr(terrain, "biome", None)
    reg = getattr(terrain, "biome_region", None)
    if not b or reg is None:
        return None
    return b["secondary"] if float(reg.weight_at(x, y)) >= 0.5 else b["id"]


def weight(terrain, feature: str) -> float:
    """Count multiplier for one v7 feature: 1.0 with biomes off. A 0 in the primary biome means none at all;
    otherwise 75% primary, 25% secondary."""
    b = getattr(terrain, "biome", None)
    if not b:
        return 1.0
    wp = float(BIOMES[b["id"]]["weights"].get(feature, 1.0))
    if wp <= 0.0:
        return 0.0
    ws = float(BIOMES[b["secondary"]]["weights"].get(feature, 1.0))
    return PRIMARY_SHARE * wp + (1.0 - PRIMARY_SHARE) * ws


def count(terrain, feature: str, n: int) -> int:
    """n scaled by the feature's weight (rounded; 0 means none). Exactly n when the weight is 1."""
    w = weight(terrain, feature)
    if w == 1.0:
        return n
    return max(0, int(round(n * w)))


def has_canyon(terrain) -> bool:
    """The canyon is single: it is there only if its weight is 0.5 or more."""
    return weight(terrain, "canyon") >= 0.5


def biome_map(terrain):
    """features.biome_map: the secondary share on a coarse grid (about 4 m cells), rounded to 2 decimals,
    row major by j. None with biomes off."""
    reg = getattr(terrain, "biome_region", None)
    if not getattr(terrain, "biome", None) or reg is None:
        return None
    cfg = terrain.cfg
    size = (cfg.nx - 1) * cfg.res
    n = max(2, int(math.ceil(size / MAP_CELL_M)) + 1)
    res = size / (n - 1)
    ox, oy = float(cfg.origin[0]), float(cfg.origin[1])
    xs = ox + np.arange(n) * res
    ys = oy + np.arange(n) * res
    X, Y = np.meshgrid(xs, ys)
    w = np.round(reg.weight_at(X, Y), 2)
    return {"nx": n, "ny": n, "origin": [ox, oy], "res": round(res, 4),
            "w": [float(v) for v in w.ravel()]}


def world_msg(terrain):
    """init world.biome: {id, name, secondary}, or None with biomes off."""
    b = getattr(terrain, "biome", None)
    return dict(b) if b else None


def tile_biome(terrain, x: float, y: float):
    """Which ground a heightfield tile centred at (x, y) shows in the camera: 'primary' or 'secondary'."""
    reg = getattr(terrain, "biome_region", None)
    if reg is None:
        return "primary"
    return "secondary" if float(reg.weight_at(x, y)) >= 0.5 else "primary"


def life_weights(terrain) -> dict:
    """{kind: weight} of the lifeform kinds for this world's primary biome ({} with biomes off)."""
    b = getattr(terrain, "biome", None)
    return dict(BIOMES[b["id"]]["life"]) if b else {}
