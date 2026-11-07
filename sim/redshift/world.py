"""Builds the MuJoCo model: tiled heightfield terrain, lander pad, rocks, target markers, rovers.

Geom groups:
  0 terrain heightfield tiles
  1 rocks and lander pad (seen by LiDAR)
  2 science target markers (no collision, seen by the spectrometer rays)
  3 rovers (excluded from every LiDAR scan)
  5 invisible walls just outside the map edge (keep rovers on the map, not seen by LiDAR)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import Config

GROUP_TERRAIN = 0
GROUP_ROCK = 1
GROUP_TARGET = 2
GROUP_ROVER = 3
GROUP_WALL = 5

TARGET_RGBA = {
    "mineral": "0.95 0.55 0.15 1",
    "ice": "0.55 0.85 1.0 1",
    "vent": "1.0 0.25 0.35 1",
}

WHEEL_LAYOUT = [  # (sx, sy): front-left, rear-left, front-right, rear-right
    (1, 1), (-1, 1), (1, -1), (-1, -1),
]


def _hex_to_rgba(color: str, alpha: float = 1.0) -> str:
    c = color.lstrip("#")
    r, g, b = (int(c[k:k + 2], 16) / 255.0 for k in (0, 2, 4))
    return f"{r:.3f} {g:.3f} {b:.3f} {alpha}"


def _yaw_quat(yaw: float) -> str:
    return f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"


def rover_body_xml(cfg: Config, rid: str, x: float, y: float, z: float, yaw: float, color: str) -> str:
    cx, cy, cz = cfg.chassis_half
    r = cfg.wheel_radius
    rgba = _hex_to_rgba(color)
    wf = cfg.wheel_friction
    parts = [
        f'<body name="{rid}" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{_yaw_quat(yaw)}">',
        f'  <freejoint name="{rid}_root"/>',
        f'  <geom name="{rid}_chassis" type="box" size="{cx} {cy} {cz}" mass="{cfg.chassis_mass}" '
        f'rgba="{rgba}" group="{GROUP_ROVER}" friction="0.5 0.005 0.0001" contype="4" conaffinity="5"/>',
        # visual only parts: deck, mast and sensor head
        f'  <geom name="{rid}_deck" type="box" size="{cx * 0.7:.3f} {cy * 0.8:.3f} 0.015" pos="-0.05 0 {cz + 0.015:.3f}" '
        f'rgba="0.15 0.15 0.18 1" group="{GROUP_ROVER}" contype="0" conaffinity="0" mass="0"/>',
        f'  <geom name="{rid}_mast" type="cylinder" size="0.022 {cfg.lidar_height / 2:.3f}" '
        f'pos="0.18 0 {cfg.lidar_height / 2:.3f}" rgba="0.8 0.8 0.85 1" group="{GROUP_ROVER}" '
        f'contype="0" conaffinity="0" mass="0"/>',
        f'  <geom name="{rid}_head" type="box" size="0.06 0.06 0.035" pos="0.18 0 {cfg.lidar_height:.3f}" '
        f'rgba="{rgba}" group="{GROUP_ROVER}" contype="0" conaffinity="0" mass="0"/>',
        f'  <site name="{rid}_lidar" pos="0.18 0 {cfg.lidar_height + 0.05:.3f}" size="0.03" group="{GROUP_ROVER}"/>',
    ]
    def wheel(k: int, pos: str, indent: str) -> list:
        return [
            f'{indent}<body name="{rid}_wb{k}" pos="{pos}">',
            f'{indent}  <joint name="{rid}_w{k}" type="hinge" axis="0 1 0" damping="0.05" armature="0.02"/>',
            f'{indent}  <geom name="{rid}_wheel{k}" type="sphere" size="{r}" mass="{cfg.wheel_mass}" '
            f'friction="{wf} 0.005 0.0001" rgba="0.12 0.12 0.12 1" group="{GROUP_ROVER}" contype="2" conaffinity="1"/>',
            f'{indent}</body>',
        ]

    # rear wheels are fixed to the chassis; the front pair sits on an oscillating axle (a roll hinge)
    # so all four wheels stay loaded on uneven ground instead of rocking like a wobbly table
    for k, (sx, sy) in enumerate(WHEEL_LAYOUT):
        if sx < 0:
            parts += wheel(k, f"{sx * cfg.wheel_x} {sy * cfg.wheel_y} {cfg.wheel_z}", "  ")
    parts += [
        f'  <body name="{rid}_axle" pos="{cfg.wheel_x} 0 {cfg.wheel_z}">',
        f'    <joint name="{rid}_axle" type="hinge" axis="1 0 0" range="-0.3 0.3" damping="1.0" armature="0.01"/>',
        f'    <geom name="{rid}_axlebar" type="capsule" fromto="0 {-cfg.wheel_y * 0.8:.3f} 0 0 {cfg.wheel_y * 0.8:.3f} 0" '
        f'size="0.03" mass="0.5" rgba="0.3 0.3 0.32 1" group="{GROUP_ROVER}" contype="0" conaffinity="0"/>',
    ]
    for k, (sx, sy) in enumerate(WHEEL_LAYOUT):
        if sx > 0:
            parts += wheel(k, f"0 {sy * cfg.wheel_y} 0", "    ")
    parts.append('  </body>')
    parts.append('</body>')
    return "\n".join(parts)


def rover_actuator_xml(cfg: Config, rid: str) -> str:
    t = cfg.motor_torque
    w = cfg.max_wheel_speed
    return "\n".join(
        f'<velocity name="{rid}_m{k}" joint="{rid}_w{k}" kv="{cfg.motor_kv}" '
        f'forcerange="{-t} {t}" ctrlrange="{-w} {w}"/>'
        for k in range(4)
    )


def rover_spawn_height(cfg: Config) -> float:
    """Chassis centre height above flat ground when resting on the wheels."""
    return cfg.wheel_radius - cfg.wheel_z


def physics_options_xml(cfg: Config) -> str:
    return (
        f'<option timestep="{cfg.timestep}" gravity="0 0 {-cfg.gravity:.4f}" integrator="implicitfast" '
        f'cone="{cfg.cone}" impratio="{cfg.impratio}" iterations="30" ls_iterations="10"/>'
    )


@dataclass
class TileInfo:
    hfield_id: int
    j0: int
    i0: int
    nrow: int
    ncol: int
    lo: float
    rz: float


@dataclass
class World:
    model: mujoco.MjModel
    data: mujoco.MjData
    tiles: list
    rover_ids: list
    target_geom_ids: dict      # geom id -> target index
    rover_geom_range: dict     # rover id -> (first geom id, last geom id)


def _tile_spans(n: int, tile: int) -> list:
    """Split sample indices 0..n-1 into overlapping spans [a, b] (inclusive) of at most tile cells."""
    spans = []
    a = 0
    while a < n - 1:
        b = min(a + tile, n - 1)
        if n - 1 - b < 3 and b != n - 1:
            b = n - 1
        spans.append((a, b))
        a = b
    return spans


def build_world(cfg: Config, terrain, rover_spawns: list) -> World:
    """terrain: terrain.Terrain. rover_spawns: list of (rid, x, y, yaw, color)."""
    H = terrain.heights
    res = cfg.res
    ox, oy = cfg.origin
    spans_i = _tile_spans(cfg.nx, cfg.hfield_tile)
    spans_j = _tile_spans(cfg.ny, cfg.hfield_tile)

    assets, geoms, tile_meta = [], [], []
    for (j0, j1) in spans_j:
        for (i0, i1) in spans_i:
            sub = H[j0:j1 + 1, i0:i1 + 1]
            lo = float(sub.min()) - 0.02
            rz = max(float(sub.max()) - lo + 0.02, 0.06)
            name = f"tile_{i0}_{j0}"
            hx = (i1 - i0) * res / 2.0
            hy = (j1 - j0) * res / 2.0
            cxw = ox + (i0 + i1) * res / 2.0
            cyw = oy + (j0 + j1) * res / 2.0
            assets.append(
                f'<hfield name="{name}" nrow="{j1 - j0 + 1}" ncol="{i1 - i0 + 1}" size="{hx} {hy} {rz:.4f} 0.5"/>'
            )
            geoms.append(
                f'<geom name="g_{name}" type="hfield" hfield="{name}" pos="{cxw:.4f} {cyw:.4f} {lo:.4f}" '
                f'material="ground" group="{GROUP_TERRAIN}" friction="{cfg.terrain_friction} 0.005 0.0001"/>'
            )
            tile_meta.append((j0, i0, j1 - j0 + 1, i1 - i0 + 1, lo, rz))

    # lander pad (visual only) and a little mast
    lh = float(terrain.height_at(0.0, 0.0))
    geoms.append(
        f'<geom name="lander_pad" type="cylinder" size="{cfg.lander_radius} 0.03" pos="0 0 {lh + 0.03:.4f}" '
        f'rgba="0.85 0.85 0.9 1" group="{GROUP_ROCK}" contype="0" conaffinity="0"/>'
    )
    geoms.append(
        f'<geom name="lander_ring" type="cylinder" size="{cfg.lander_radius * 0.55:.3f} 0.035" pos="0 0 {lh + 0.035:.4f}" '
        f'rgba="1.0 0.48 0.27 1" group="4" contype="0" conaffinity="0"/>'
    )

    # invisible boundary walls just outside the map so nothing can drive off the edge
    half = (cfg.nx - 1) * res / 2.0
    wz0, wz1 = terrain.min_h - 3.0, terrain.max_h + 3.0
    wzc, wzh = (wz0 + wz1) / 2.0, (wz1 - wz0) / 2.0
    for k, (px, py, sx, sy) in enumerate([(-half - 0.5, 0.0, 0.5, half + 1.0), (half + 0.5, 0.0, 0.5, half + 1.0),
                                          (0.0, -half - 0.5, half + 1.0, 0.5), (0.0, half + 0.5, half + 1.0, 0.5)]):
        geoms.append(
            f'<geom name="wall_{k}" type="box" size="{sx:.3f} {sy:.3f} {wzh:.3f}" pos="{px:.3f} {py:.3f} {wzc:.3f}" '
            f'rgba="0 0 0 0" group="{GROUP_WALL}" friction="0.5 0.005 0.0001"/>'
        )

    for k, rock in enumerate(terrain.rocks):
        geoms.append(
            f'<geom name="rock_{k}" type="ellipsoid" size="{rock.sx:.3f} {rock.sy:.3f} {rock.sz:.3f}" '
            f'pos="{rock.x:.3f} {rock.y:.3f} {rock.z:.3f}" euler="0 0 {rock.yaw:.3f}" material="rock" '
            f'group="{GROUP_ROCK}" friction="0.8 0.005 0.0001"/>'
        )

    for t in terrain.targets:
        geoms.append(
            f'<geom name="target_{t.id}" type="capsule" size="0.28 0.35" pos="{t.x:.3f} {t.y:.3f} {t.z + 0.6:.3f}" '
            f'rgba="{TARGET_RGBA[t.kind]}" group="{GROUP_TARGET}" contype="0" conaffinity="0"/>'
        )

    bodies, actuators, rover_ids = [], [], []
    z0 = rover_spawn_height(cfg)
    for (rid, x, y, yaw, color) in rover_spawns:
        gz = float(terrain.height_at(x, y))
        bodies.append(rover_body_xml(cfg, rid, x, y, gz + z0 + 0.03, yaw, color))
        actuators.append(rover_actuator_xml(cfg, rid))
        rover_ids.append(rid)

    span = cfg.map_size
    xml = f"""
<mujoco model="redshift">
  <compiler angle="radian" autolimits="true"/>
  {physics_options_xml(cfg)}
  <size memory="64M"/>
  <visual>
    <global offwidth="1280" offheight="720" elevation="-25" azimuth="135"/>
    <map znear="0.02" zfar="{span * 3:.0f}" haze="0.2"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.3 0.3" diffuse="0.6 0.55 0.5" specular="0.1 0.1 0.1"/>
    <rgba haze="0.55 0.3 0.22 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.45 0.22 0.15" rgb2="0.08 0.03 0.03" width="256" height="256"/>
    <texture name="groundtex" type="2d" builtin="checker" rgb1="0.62 0.36 0.24" rgb2="0.58 0.33 0.22" width="256" height="256" mark="random" random="0.08" markrgb="0.5 0.3 0.2"/>
    <material name="ground" texture="groundtex" texrepeat="1 1" texuniform="true" reflectance="0.0"/>
    <material name="rock" rgba="0.35 0.3 0.28 1"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 50" dir="-0.4 0.3 -1" diffuse="0.8 0.7 0.6" castshadow="true"/>
    {''.join(geoms)}
    {''.join(bodies)}
  </worldbody>
  <actuator>
    {''.join(actuators)}
  </actuator>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)

    tiles = []
    for k, (j0, i0, nrow, ncol, lo, rz) in enumerate(tile_meta):
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, f"tile_{i0}_{j0}")
        sub = H[j0:j0 + nrow, i0:i0 + ncol]
        adr = model.hfield_adr[hid]
        # MuJoCo hfield layout: data[r * ncol + c] is the sample at x = -hx + c * dx, y = -hy + r * dy,
        # which matches the protocol grid (row = j along y, col = i along x).
        model.hfield_data[adr:adr + nrow * ncol] = ((sub - lo) / rz).astype(np.float32).ravel()
        tiles.append(TileInfo(hid, j0, i0, nrow, ncol, lo, rz))

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    target_geom_ids = {}
    for idx, t in enumerate(terrain.targets):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"target_{t.id}")
        target_geom_ids[gid] = idx

    rover_geom_range = {}
    for rid in rover_ids:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rid)
        gids = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == bid]
        rover_geom_range[rid] = (min(gids), max(gids))

    return World(model, data, tiles, rover_ids, target_geom_ids, rover_geom_range)
