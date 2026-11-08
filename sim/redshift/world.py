"""Builds the MuJoCo model: tiled heightfield terrain, lander pad, rocks, target markers, rovers.

Geom groups:
  0 terrain heightfield tiles
  1 rocks, rockfall boulders, alien spires and the lander pad (seen by LiDAR, solid)
  2 science target markers and lifeforms (no collision, seen by LiDAR rays)
  3 rovers (excluded from every LiDAR scan)
  4 lander ring, lander decoration and the invisible walls just outside the map edge (not seen by LiDAR)
  5 liquid surfaces of lakes and rivers (semi-transparent, no collision; LiDAR rays that hit them
    get no return)

Rockfall boulders are mocap bodies parked far below the map until hazards.py drops them in. Spires
are mocap bodies too (origin at the base) so a spire_fall can topple one by rotating it, and a few
spare spires wait below the map for "drop a new spire here". Lifeforms are mocap bodies moved by
lifeforms.py. Each rover carries a simulated IMU (framequat + gyro on a site at the chassis centre).

The look (textures, sky, sun, fog, materials) comes from visuals.py and is visual only: decoration
geoms are massless, never collide and sit in groups the LiDAR does not scan (3 rovers, 4 lander).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import visuals
from .config import Config

GROUP_TERRAIN = 0
GROUP_ROCK = 1
GROUP_TARGET = 2
GROUP_ROVER = 3
GROUP_WALL = 4
GROUP_WATER = 5

# Spires (magenta capsules inside violet crystal shells), liquid (deep blue), grazers (green, slightly
# bioluminescent) and every other material are defined in visuals.py. The capsule colour stays far from
# every target colour (camera target spotting) and the liquid stays clearly blue (the rules' camera
# check for a lake).

TARGET_RGBA = {          # same colours as the dashboard: mineral #b48cff, ice #a8d4ff, vent #ff8a3d
    "mineral": "0.706 0.549 1.0 1",
    "ice": "0.659 0.831 1.0 1",
    "vent": "1.0 0.541 0.239 1",
    "trace": "0.55 1.0 0.15 1",     # lime green: organic traces near the herds (far from the teal grazers)
}

BOULDER_PARK_Z = -60.0     # hidden boulders wait this far below the lowest terrain

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
    vis = f'group="{GROUP_ROVER}" contype="0" conaffinity="0" mass="0"'     # visual only: no mass, no contacts
    lh = cfg.lidar_height
    parts = [
        f'<body name="{rid}" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{_yaw_quat(yaw)}">',
        f'  <freejoint name="{rid}_root"/>',
        f'  <geom name="{rid}_chassis" type="box" size="{cx} {cy} {cz}" mass="{cfg.chassis_mass}" '
        f'material="paint" rgba="{rgba}" group="{GROUP_ROVER}" friction="0.5 0.005 0.0001" contype="4" conaffinity="5"/>',
        # visual only parts: solar deck, mast, sensor head with camera lens and LiDAR puck, antenna
        f'  <geom name="{rid}_deck" type="box" size="{cx * 0.7:.3f} {cy * 0.8:.3f} 0.015" pos="-0.05 0 {cz + 0.015:.3f}" '
        f'material="solar" {vis}/>',
        f'  <geom name="{rid}_mast" type="cylinder" size="0.022 {lh / 2:.3f}" pos="0.18 0 {lh / 2:.3f}" '
        f'material="metal" {vis}/>',
        f'  <geom name="{rid}_head" type="box" size="0.06 0.06 0.035" pos="0.18 0 {lh:.3f}" material="darkmetal" {vis}/>',
        f'  <site name="{rid}_lidar" pos="0.18 0 {lh + 0.05:.3f}" size="0.03" group="{GROUP_ROVER}"/>',
        f'  <site name="{rid}_imu" pos="0 0 0" size="0.02" group="{GROUP_ROVER}"/>',
        f'  <geom name="{rid}_lens" type="cylinder" size="0.022 0.008" pos="0.245 0 {lh:.3f}" euler="0 1.5708 0" '
        f'material="lens" {vis}/>',
        f'  <geom name="{rid}_puck" type="cylinder" size="0.045 0.014" pos="0.18 0 {lh + 0.049:.3f}" '
        f'material="paint" rgba="{rgba}" {vis}/>',
        f'  <geom name="{rid}_antenna" type="cylinder" size="0.007 0.22" pos="{-cx + 0.08:.3f} {cy - 0.06:.3f} {cz + 0.22:.3f}" '
        f'material="metal" {vis}/>',
    ]
    def wheel(k: int, pos: str, indent: str) -> list:
        return [
            f'{indent}<body name="{rid}_wb{k}" pos="{pos}">',
            f'{indent}  <joint name="{rid}_w{k}" type="hinge" axis="0 1 0" damping="0.05" armature="0.02"/>',
            # the contact sphere is drawn fully transparent; a visual tyre and hub turn with it
            f'{indent}  <geom name="{rid}_wheel{k}" type="sphere" size="{r}" mass="{cfg.wheel_mass}" '
            f'friction="{wf} 0.005 0.0001" priority="1" rgba="0 0 0 0" group="{GROUP_ROVER}" '
            f'contype="2" conaffinity="1"/>',
            f'{indent}  <geom name="{rid}_tyre{k}" type="cylinder" size="{r:.3f} 0.065" euler="1.5708 0 0" '
            f'material="tyre" {vis}/>',
            f'{indent}  <geom name="{rid}_hub{k}" type="cylinder" size="{0.55 * r:.3f} 0.071" euler="1.5708 0 0" '
            f'material="metal" {vis}/>',
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
        f'size="0.03" mass="0.5" material="darkmetal" group="{GROUP_ROVER}" contype="0" conaffinity="0"/>',
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


def rover_sensor_xml(rid: str) -> str:
    """Simulated IMU: attitude (framequat, what an AHRS chip reports) and body rates (gyro)."""
    return (f'<framequat name="{rid}_imu_quat" objtype="site" objname="{rid}_imu"/>'
            f'<gyro name="{rid}_imu_gyro" site="{rid}_imu"/>')


def boulder_sizes(cfg: Config) -> list:
    """Semi-axes (sx, sy, sz) of the hidden rockfall boulders. Deterministic, a mix of sizes."""
    base = (0.55, 0.62, 0.5, 0.58, 0.66, 0.52, 0.6, 0.56, 0.64, 0.5, 0.6, 0.54)
    out = []
    for k in range(cfg.num_boulders):
        r = base[k % len(base)]
        out.append((r, round(0.9 * r, 3), round(0.78 * r, 3)))
    return out


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
    boulders: list             # [(mocap id, geom id, (sx, sy, sz))] hidden rockfall boulders
    spires: list = None        # [(mocap id, [geom ids])] per terrain spire, then the spare ones
    water_gids: np.ndarray = None   # liquid surface geoms (LiDAR: no return)
    life_mocap: list = None    # mocap id per creature (lifeforms.py order)
    life_gids: np.ndarray = None    # every creature geom (LiDAR hits go to the motion tracker)


def _quat_z_to(u) -> str:
    """Quaternion (w x y z) turning local +z onto the unit vector u."""
    ux, uy, uz = u
    ang = math.acos(max(-1.0, min(1.0, uz)))
    ax, ay = -uy, ux
    n = math.hypot(ax, ay)
    if n < 1e-9:
        return "1 0 0 0" if uz > 0 else "0 1 0 0"
    s = math.sin(ang / 2) / n
    return f"{math.cos(ang / 2):.6f} {ax * s:.6f} {ay * s:.6f} 0"


def spire_body_xml(name: str, x: float, y: float, z: float, height: float, radius: float, sides: list,
                   meshes: list | None = None) -> str:
    """A crystal spire whose body origin is its base (so toppling = rotating the body). With a meshes
    list, each capsule also gets a faceted crystal shell (visual only: group 4, no contacts, so LiDAR
    still sees the capsules); the mesh assets are appended to the list."""
    r = radius
    parts = [f'<body name="{name}" mocap="true" pos="{x:.3f} {y:.3f} {z:.3f}">',
             f'<geom name="{name}_g0" type="capsule" fromto="0 0 {r:.3f} 0 0 {max(height - r, r + 0.05):.3f}" '
             f'size="{r:.3f}" material="spire" group="{GROUP_ROCK}" friction="0.8 0.005 0.0001"/>',
             f'<geom name="{name}_g1" type="capsule" fromto="0 0 {0.55 * height:.3f} 0 0 {max(height - 0.4 * r, 0.6 * height):.3f}" '
             f'size="{0.6 * r:.3f}" material="spire_core" group="{GROUP_ROCK}" contype="0" conaffinity="0"/>']
    for k, (az, lean, length, sr) in enumerate(sides):
        x1 = math.sin(lean) * length * math.cos(az)
        y1 = math.sin(lean) * length * math.sin(az)
        z1 = math.cos(lean) * length
        parts.append(f'<geom name="{name}_g{k + 2}" type="capsule" fromto="0 0 {sr:.3f} {x1:.3f} {y1:.3f} {z1 + sr:.3f}" '
                     f'size="{sr:.3f}" material="spire" group="{GROUP_ROCK}" friction="0.8 0.005 0.0001"/>')
    if meshes is not None:
        vis = 'material="crystal" group="4" contype="0" conaffinity="0" mass="0"'
        meshes.append(visuals.crystal_mesh(f"{name}_cr0", height - 0.45 * r, r, 1.0 * r))
        parts.append(f'<geom name="{name}_crystal0" type="mesh" mesh="{name}_cr0" {vis}/>')
        for k, (az, lean, length, sr) in enumerate(sides):
            u = (math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean))
            meshes.append(visuals.crystal_mesh(f"{name}_cr{k + 1}", length + 1.55 * sr, sr, 1.0 * sr))
            px, py, pz = -u[0] * sr, -u[1] * sr, sr - u[2] * sr
            parts.append(f'<geom name="{name}_crystal{k + 1}" type="mesh" mesh="{name}_cr{k + 1}" '
                         f'pos="{px:.4f} {py:.4f} {pz:.4f}" quat="{_quat_z_to(u)}" {vis}/>')
    parts.append('</body>')
    return "".join(parts)


def rover_materials_xml() -> str:
    """The materials rover_body_xml refers to, for models built outside build_world (climb_test.py)."""
    return ('<texture name="solartex" type="2d" builtin="flat" rgb1="0.05 0.08 0.17" width="8" height="8"/>'
            '<material name="paint" rgba="0.8 0.8 0.8 1" specular="0.45" shininess="0.55"/>'
            '<material name="solar" texture="solartex" specular="0.9" shininess="0.95"/>'
            '<material name="metal" rgba="0.74 0.74 0.77 1" specular="0.7" shininess="0.75"/>'
            '<material name="darkmetal" rgba="0.17 0.17 0.19 1" specular="0.5" shininess="0.6"/>'
            '<material name="rubber" rgba="0.13 0.12 0.11 1" specular="0.08" shininess="0.1"/>'
            '<texture name="tyretex" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.16 0.15 0.14" width="8" height="8"/>'
            '<material name="tyre" texture="tyretex" texrepeat="12 1" specular="0.1" shininess="0.1"/>'
            '<material name="lens" rgba="0.02 0.03 0.05 1" specular="1" shininess="1"/>')


def creature_body_xml(name: str, x: float, y: float, z: float, yaw: float, size: float) -> str:
    """A six-legged grazer, about `size` m long: body, head, three back plates, six legs. Body origin
    at the ground under its middle. Group 2: LiDAR sees it, nothing collides with it."""
    s_ = size / 1.2
    hz = 0.5 * s_
    g = f'group="{GROUP_TARGET}" contype="0" conaffinity="0"'
    v = 'group="4" contype="0" conaffinity="0" mass="0"'          # visual only (LiDAR never scans group 4)
    parts = [f'<body name="{name}" mocap="true" pos="{x:.3f} {y:.3f} {z:.3f}" quat="{_yaw_quat(yaw)}">',
             f'<geom name="{name}_body" type="ellipsoid" size="{0.5 * s_:.3f} {0.26 * s_:.3f} {0.2 * s_:.3f}" '
             f'pos="0 0 {hz:.3f}" material="life_joint" {g}/>',
             f'<geom name="{name}_head" type="sphere" size="{0.15 * s_:.3f}" pos="{0.55 * s_:.3f} 0 {hz + 0.07 * s_:.3f}" '
             f'material="life" {g}/>']
    for k, px in enumerate((-0.28, 0.0, 0.28)):
        parts.append(f'<geom name="{name}_plate{k}" type="ellipsoid" size="{0.13 * s_:.3f} {0.2 * s_:.3f} {0.06 * s_:.3f}" '
                     f'pos="{px * s_:.3f} 0 {hz + 0.17 * s_:.3f}" material="life_dark" {g}/>')
    for px in (-0.3, 0.0, 0.3):
        for side in (1, -1):
            parts.append(f'<geom name="{name}_leg{"l" if side > 0 else "r"}{int((px + 0.3) * 10)}" type="capsule" '
                         f'fromto="{px * s_:.3f} {side * 0.18 * s_:.3f} {hz - 0.05 * s_:.3f} '
                         f'{(px + 0.06) * s_:.3f} {side * 0.38 * s_:.3f} 0.03" size="{0.035 * s_:.3f}" '
                         f'material="life_dark" {g}/>')
    # what the camera shows: three slightly flattened body segments (the dark core shows between them
    # as joints), glowing eyes and a row of bioluminescent spots along each flank
    for k, (px, sx, sy, sz, dz) in enumerate(((-0.31, 0.21, 0.27, 0.19, -0.01), (0.0, 0.23, 0.30, 0.215, 0.01),
                                               (0.3, 0.2, 0.27, 0.19, 0.0))):
        parts.append(f'<geom name="{name}_seg{k}" type="ellipsoid" size="{sx * s_:.3f} {sy * s_:.3f} {sz * s_:.3f}" '
                     f'pos="{px * s_:.3f} 0 {hz + dz * s_:.3f}" material="life" {v}/>')
    for side in (1, -1):
        parts.append(f'<geom name="{name}_eye{"l" if side > 0 else "r"}" type="sphere" size="{0.03 * s_:.3f}" '
                     f'pos="{0.66 * s_:.3f} {side * 0.07 * s_:.3f} {hz + 0.12 * s_:.3f}" material="life_glow" {v}/>')
        for k, px in enumerate((-0.2, 0.15)):
            parts.append(f'<geom name="{name}_spot{"l" if side > 0 else "r"}{k}" type="sphere" size="{0.022 * s_:.3f}" '
                         f'pos="{px * s_:.3f} {side * 0.27 * s_:.3f} {hz + 0.02 * s_:.3f}" material="life_glow" {v}/>')
    parts.append('</body>')
    return "".join(parts)


def _ground_frame(terrain, x: float, y: float) -> tuple:
    """(ground height, quaternion w x y z turning +z onto the ground normal) at (x, y)."""
    e = 0.3
    gz = float(terrain.height_at(x, y))
    dzdx = (float(terrain.height_at(x + e, y)) - float(terrain.height_at(x - e, y))) / (2 * e)
    dzdy = (float(terrain.height_at(x, y + e)) - float(terrain.height_at(x, y - e))) / (2 * e)
    n = np.array([-dzdx, -dzdy, 1.0])
    n /= np.linalg.norm(n)
    return gz, _quat_z_to(n)


def target_decor_xml(t, terrain) -> str:
    """What a science target looks like to the camera (visual only, group 4: no LiDAR, no contacts, no
    mass). The LiDAR, the spectrometer and machine vision keep using the marker capsule, which the
    camera's look frames leave out. Protocol colours are accents: vent orange rim, mineral violet
    crystals, ice white-blue, trace lime spots."""
    v = 'group="4" contype="0" conaffinity="0" mass="0"'
    rng = np.random.default_rng(int(t.id.lstrip("t") or 0) * 7919 + 17)
    gz, q = _ground_frame(terrain, t.x, t.y)
    yaw = float(rng.uniform(0, 2 * math.pi))
    out = [f'<body name="decor_{t.id}" pos="{t.x:.3f} {t.y:.3f} {gz:.4f}" quat="{q}">',
           f'<frame euler="0 0 {yaw:.3f}">']
    if t.kind == "vent":
        out += [f'<geom name="{t.id}_mound" type="ellipsoid" size="0.6 0.52 0.2" pos="0 0 -0.05" material="mound" {v}/>',
                f'<geom name="{t.id}_crust" type="ellipsoid" size="0.27 0.25 0.06" pos="0 0 0.115" material="vent_crust" {v}/>',
                f'<geom name="{t.id}_rim" type="cylinder" size="0.17 0.012" pos="0 0 0.158" material="vent_rim" {v}/>',
                f'<geom name="{t.id}_mouth" type="cylinder" size="0.115 0.02" pos="0 0 0.163" material="vent_mouth" {v}/>']
        for k in range(3):                      # warm mineral stains spilling down the flank
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.22, 0.42)
            out.append(f'<geom name="{t.id}_stain{k}" type="ellipsoid" size="{rng.uniform(0.05, 0.09):.3f} '
                       f'{rng.uniform(0.03, 0.05):.3f} 0.012" pos="{r * math.cos(a):.3f} {r * math.sin(a):.3f} '
                       f'{0.15 - 0.55 * (r - 0.15) ** 1.2:.3f}" euler="0 0 {a:.3f}" material="vent_crust" {v}/>')
    elif t.kind == "mineral":
        for k in range(3):                      # an angular outcrop of tilted blocks
            out.append(f'<geom name="{t.id}_block{k}" type="box" size="{rng.uniform(0.18, 0.3):.3f} '
                       f'{rng.uniform(0.14, 0.22):.3f} {rng.uniform(0.1, 0.17):.3f}" '
                       f'pos="{rng.uniform(-0.2, 0.2):.3f} {rng.uniform(-0.2, 0.2):.3f} 0.04" '
                       f'euler="{rng.uniform(-0.35, 0.35):.3f} {rng.uniform(-0.35, 0.35):.3f} {rng.uniform(0, 3.1):.3f}" '
                       f'material="outcrop" {v}/>')
        for k in range(6):                      # violet crystal veins breaking out of it
            a = rng.uniform(0, 2 * math.pi)
            lean = rng.uniform(0.15, 0.75)
            u = (math.sin(lean) * math.cos(a), math.sin(lean) * math.sin(a), math.cos(lean))
            r0 = rng.uniform(0.0, 0.18)
            out.append(f'<geom name="{t.id}_vein{k}" type="mesh" mesh="mineral_xtal" '
                       f'pos="{r0 * math.cos(a):.3f} {r0 * math.sin(a):.3f} {rng.uniform(0.05, 0.15):.3f}" '
                       f'quat="{_quat_z_to(u)}" material="mineral_vein" {v}/>')
    elif t.kind == "ice":                       # an irregular glossy patch, partly buried, with broken slabs
        for k in range(4):
            out.append(f'<geom name="{t.id}_sheet{k}" type="ellipsoid" size="{rng.uniform(0.22, 0.42):.3f} '
                       f'{rng.uniform(0.15, 0.3):.3f} 0.025" pos="{rng.uniform(-0.25, 0.25):.3f} '
                       f'{rng.uniform(-0.2, 0.2):.3f} {rng.uniform(-0.012, 0.0):.3f}" euler="0 0 {rng.uniform(0, 3.1):.3f}" '
                       f'material="ice" {v}/>')
        for k in range(4):
            out.append(f'<geom name="{t.id}_slab{k}" type="box" size="{rng.uniform(0.05, 0.11):.3f} '
                       f'{rng.uniform(0.04, 0.08):.3f} 0.02" pos="{rng.uniform(-0.3, 0.3):.3f} '
                       f'{rng.uniform(-0.2, 0.2):.3f} 0.02" euler="{rng.uniform(-0.25, 0.25):.3f} '
                       f'{rng.uniform(-0.25, 0.25):.3f} {rng.uniform(0, 3.1):.3f}" material="ice_chunk" {v}/>')
    else:                                       # trace: a thin organic film with lime specks
        for k in range(4):
            out.append(f'<geom name="{t.id}_film{k}" type="ellipsoid" size="{rng.uniform(0.2, 0.36):.3f} '
                       f'{rng.uniform(0.12, 0.22):.3f} 0.012" pos="{rng.uniform(-0.25, 0.25):.3f} '
                       f'{rng.uniform(-0.2, 0.2):.3f} 0.004" euler="0 0 {rng.uniform(0, 3.1):.3f}" '
                       f'material="trace_film" {v}/>')
        for k in range(7):
            r = rng.uniform(0.012, 0.028)
            out.append(f'<geom name="{t.id}_spot{k}" type="ellipsoid" size="{r:.3f} {r * 0.8:.3f} 0.006" '
                       f'pos="{rng.uniform(-0.4, 0.4):.3f} {rng.uniform(-0.28, 0.28):.3f} 0.014" '
                       f'material="trace_spot" {v}/>')
    out.append('</frame></body>')
    return "".join(out)


def lander_decor_xml(cfg: Config, terrain, lh: float) -> str:
    """Visual only lander details (group 4: no LiDAR, no contacts, massless): a low central hub with a
    beacon, and four landing legs with footpads just outside the pad. Kept low so the rovers' chase
    cameras see over them."""
    g = 'group="4" contype="0" conaffinity="0" mass="0"'
    top = lh + 0.06
    parts = [f'<geom name="lander_hub" type="cylinder" size="0.42 0.07" pos="0 0 {top + 0.07:.4f}" material="metal" {g}/>',
             f'<geom name="lander_hub_band" type="cylinder" size="0.44 0.015" pos="0 0 {top + 0.11:.4f}" material="marking" {g}/>',
             f'<geom name="lander_beacon" type="sphere" size="0.05" pos="0 0 {top + 0.16:.4f}" material="beacon" {g}/>']
    r0, r1 = cfg.lander_radius - 0.1, cfg.lander_radius + 0.55
    for k, a in enumerate((0.785, 2.356, 3.927, 5.498)):
        x0, y0 = r0 * math.cos(a), r0 * math.sin(a)
        x1, y1 = r1 * math.cos(a), r1 * math.sin(a)
        gz = float(terrain.height_at(x1, y1))
        parts.append(f'<geom name="lander_leg{k}" type="capsule" fromto="{x0:.3f} {y0:.3f} {top + 0.02:.4f} '
                     f'{x1:.3f} {y1:.3f} {gz + 0.08:.4f}" size="0.035" material="metal" {g}/>')
        parts.append(f'<geom name="lander_foot{k}" type="cylinder" size="0.2 0.025" pos="{x1:.3f} {y1:.3f} {gz + 0.03:.4f}" '
                     f'material="darkmetal" {g}/>')
    return "".join(parts)


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
    from . import v7_biomes

    def _secondary(x: float, y: float) -> bool:          # v7 biomes: tiles in the secondary region
        return v7_biomes.tile_biome(terrain, x, y) == "secondary"

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
                f'material="{visuals.ground_material(i0, j0, cfg.seed, _secondary(cxw, cyw))}" group="{GROUP_TERRAIN}" '
                f'friction="{cfg.terrain_friction} 0.005 0.0001"/>'
            )
            tile_meta.append((j0, i0, j1 - j0 + 1, i1 - i0 + 1, lo, rz))

    # lander pad (visual only) and a little mast
    lh = float(terrain.height_at(0.0, 0.0))
    geoms.append(
        f'<geom name="lander_pad" type="cylinder" size="{cfg.lander_radius} 0.03" pos="0 0 {lh + 0.03:.4f}" '
        f'material="pad" group="{GROUP_ROCK}" contype="0" conaffinity="0"/>'
    )
    geoms.append(
        f'<geom name="lander_ring" type="cylinder" size="{cfg.lander_radius * 0.55:.3f} 0.035" pos="0 0 {lh + 0.035:.4f}" '
        f'material="marking" group="4" contype="0" conaffinity="0"/>'
    )
    geoms.append(lander_decor_xml(cfg, terrain, lh))

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
    for k, rock in enumerate(getattr(terrain, "bfield", None) or []):     # v7 boulder fields
        geoms.append(
            f'<geom name="bfield_{k}" type="ellipsoid" size="{rock.sx:.3f} {rock.sy:.3f} {rock.sz:.3f}" '
            f'pos="{rock.x:.3f} {rock.y:.3f} {rock.z:.3f}" euler="0 0 {rock.yaw:.3f}" material="rock" '
            f'group="{GROUP_ROCK}" friction="0.8 0.005 0.0001"/>'
        )

    decor = []
    for t in terrain.targets:
        geoms.append(
            f'<geom name="target_{t.id}" type="capsule" size="0.28 0.35" pos="{t.x:.3f} {t.y:.3f} {t.z + 0.6:.3f}" '
            f'material="target_{t.kind}" group="{GROUP_TARGET}" contype="0" conaffinity="0"/>'
        )
        decor.append(target_decor_xml(t, terrain))

    # liquid surfaces: a flat disc (lake) or flat boxes (river) at the water level. The guard ring
    # around each one keeps the ground above the level, so only the liquid itself shows.
    water_names = []
    for w in getattr(terrain, "water", []):
        if w.kind == "lake":
            water_names.append(f"water_{w.id}")
            geoms.append(f'<geom name="water_{w.id}" type="cylinder" size="{w.radius:.3f} 0.01" '
                         f'pos="{w.cx:.3f} {w.cy:.3f} {w.level - 0.01:.4f}" material="water" '
                         f'group="{GROUP_WATER}" contype="0" conaffinity="0"/>')
        else:
            for k, (bx, by, hl, hw, yaw) in enumerate(w.boxes or []):
                water_names.append(f"water_{w.id}_{k}")
                geoms.append(f'<geom name="water_{w.id}_{k}" type="box" size="{hl:.3f} {hw:.3f} 0.01" '
                             f'pos="{bx:.3f} {by:.3f} {w.level - 0.01 - 0.0007 * k:.4f}" euler="0 0 {yaw:.4f}" '
                             f'material="water" group="{GROUP_WATER}" contype="0" conaffinity="0"/>')

    if getattr(cfg, "geysers", False):              # v7 geyser vent cones (seen by LiDAR, solid)
        from .v7_geysers import vent_geoms_xml
        geoms += vent_geoms_xml(getattr(terrain, "geysers", []))

    bodies, actuators, sensors, rover_ids = [], [], [], []
    z0 = rover_spawn_height(cfg)
    for (rid, x, y, yaw, color) in rover_spawns:
        gz = float(terrain.height_at(x, y))
        bodies.append(rover_body_xml(cfg, rid, x, y, gz + z0 + 0.03, yaw, color))
        actuators.append(rover_actuator_xml(cfg, rid))
        sensors.append(rover_sensor_xml(rid))
        rover_ids.append(rid)

    # rockfall boulders: kinematic (mocap) bodies parked far below the map, seen by LiDAR (group 1)
    # and solid for rovers once hazards.py moves them into place
    park_z = terrain.min_h + BOULDER_PARK_Z
    bsizes = boulder_sizes(cfg)
    for k, (sx, sy, sz) in enumerate(bsizes):
        bodies.append(
            f'<body name="boulder_{k}" mocap="true" pos="{-20.0 + 3.0 * k:.2f} 0 {park_z:.2f}">'
            f'<geom name="boulder_{k}" type="ellipsoid" size="{sx:.3f} {sy:.3f} {sz:.3f}" material="rock" '
            f'group="{GROUP_ROCK}" friction="0.8 0.005 0.0001"/></body>'
        )

    # alien spires (mocap, origin at the base) and spare ones parked below the map
    spire_names = []
    crystal_meshes: list = []
    spires = list(getattr(terrain, "spires", []))
    for sp in spires:
        spire_names.append(f"spire_{sp.id}")
        bodies.append(spire_body_xml(f"spire_{sp.id}", sp.x, sp.y, sp.z, sp.height, sp.radius, sp.sides,
                                     crystal_meshes))
    if spires or cfg.spires:
        for k in range(cfg.reserve_spires):
            name = f"spire_spare{k}"
            spire_names.append(name)
            bodies.append(spire_body_xml(name, 20.0 + 3.0 * k, 0.0, park_z, 3.0, 0.32,
                                         [(0.5, 0.45, 0.8, 0.18), (2.9, 0.5, 0.6, 0.16)], crystal_meshes))

    bodies += decor                 # what the targets look like to the camera (static, visual only)
    if getattr(terrain, "flora", None):     # v7 alien flora: static proxies and decor
        from . import v7_flora
        geoms.append(v7_flora.geoms_xml(terrain.flora))

    # lifeforms (lifeforms.py moves them)
    life_names = []
    for h in getattr(terrain, "herds", []):
        kind = getattr(h, "kind", "grazer")
        for k, (x, y, yaw, size, _speed) in enumerate(h.members):
            name = f"life_{h.id}_{k}"
            life_names.append(name)
            if kind == "grazer":
                bodies.append(creature_body_xml(name, x, y, float(terrain.height_at(x, y)), yaw, size))
            else:                   # v7 life variety: the other kinds (v7_life.py)
                from . import v7_life
                bodies.append(v7_life.body_xml(kind, name, x, y, float(terrain.height_at(x, y)), yaw, size,
                                               GROUP_TARGET, _yaw_quat))
    if any(getattr(h, "kind", "grazer") != "grazer" for h in getattr(terrain, "herds", [])):
        from . import v7_life
        assets.append(v7_life.materials_xml(terrain))

    span = cfg.map_size
    xml = f"""
<mujoco model="redshift">
  <compiler angle="radian" autolimits="true"/>
  {physics_options_xml(cfg)}
  <size memory="64M"/>
  {visuals.visual_xml(cfg)}
  <asset>
    {visuals.asset_xml(cfg)}
    <material name="paint" rgba="0.8 0.8 0.8 1" specular="0.45" shininess="0.55"/>
    {''.join(visuals.target_material_xml(k, v) for k, v in TARGET_RGBA.items())}
    {''.join(assets)}
    {''.join(crystal_meshes)}
    {visuals.crystal_mesh("mineral_xtal", 0.26, 0.035, 0.06)}
  </asset>
  <worldbody>
    {visuals.lights_xml(cfg)}
    {''.join(geoms)}
    {''.join(bodies)}
  </worldbody>
  <actuator>
    {''.join(actuators)}
  </actuator>
  <sensor>
    {''.join(sensors)}
  </sensor>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    visuals.paint(model, cfg)

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

    boulders = []
    for k, size in enumerate(bsizes):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"boulder_{k}")
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"boulder_{k}")
        boulders.append((int(model.body_mocapid[bid]), int(gid), size))

    def body_geoms(bname):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        return bid, [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]

    spire_info = []
    for name in spire_names:
        bid, gids = body_geoms(name)
        spire_info.append((int(model.body_mocapid[bid]), gids))
    water_gids = np.array(sorted(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in water_names),
                          dtype=np.int32)
    life_mocap, life_g = [], []
    for name in life_names:
        bid, gids = body_geoms(name)
        life_mocap.append(int(model.body_mocapid[bid]))
        life_g += gids
    return World(model, data, tiles, rover_ids, target_geom_ids, rover_geom_range, boulders,
                 spires=spire_info, water_gids=water_gids, life_mocap=life_mocap,
                 life_gids=np.array(sorted(life_g), dtype=np.int32))
