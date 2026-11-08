"""Generates the per-deploy Gazebo world (SDF + heightmap + textures + bridge config).

Everything the mission needs lands in one runtime directory:
  world.sdf · heightmap.png (16-bit) · regolith_diffuse.png · regolith_normal.png · terrain.npz ·
  bridge.yaml · world_meta.json

It also holds the pure Gazebo-side helpers of world_interface_node: the SDF of the toppling basalt pillar it spawns
at run time (pillar_model_sdf) and the body-frame storm-wind drift its scout command relay adds (wind_drift_body).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from .robot_models import ROVER_SPEC, rover_sdf, scout_sdf
from .terrain_gen import generate_terrain

WORLD_NAME = 'redshift'
SECTORS = {'small': (50.0, 0.5), 'medium': (100.0, 0.5), 'large': (200.0, 1.0)}
G_EARTH = 9.81
# Gazebo maps image row 0 to the +y edge of the heightmap; our grid row 0 is the −y edge.
FLIP_HEIGHTMAP_ROWS = True


def robot_names(fleet: dict) -> list[str]:
    return [f'rover_{k}' for k in range(int(fleet.get('rovers', 1)))] + \
           [f'scout_{k}' for k in range(int(fleet.get('drones', 0)))]


def _periodic_noise(rng, n: int, beta: float = 2.2) -> np.ndarray:
    """Seamlessly tileable fractal noise (spectral synthesis: periodic by construction)."""
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spec = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / f ** (beta / 2)
    spec[0, 0] = 0
    out = np.real(np.fft.ifft2(spec))
    return (out - out.min()) / (out.max() - out.min())


def _write_textures(out: Path, seed: int):
    rng = np.random.default_rng(seed + 1000)
    n = 512
    noise = 0.7 * _periodic_noise(rng, n, 2.4) + 0.3 * _periodic_noise(rng, n, 1.2)
    base = np.array([52, 84, 156], np.float64)      # BGR rust regolith
    img = base[None, None, :] * (0.72 + 0.5 * noise[..., None])
    specks = rng.random((n, n)) < 0.003
    img[specks] *= 0.6
    cv2.imwrite(str(out / 'regolith_diffuse.png'), np.clip(img, 0, 255).astype(np.uint8))
    normal = np.zeros((n, n, 3), np.uint8)
    normal[:] = (255, 128, 128)                    # BGR of a flat tangent-space normal (128,128,255)
    cv2.imwrite(str(out / 'regolith_normal.png'), normal)


def _rock_sdf(k, rock, ground_z):
    a, b, c = rock.radius, rock.radius * 0.8, rock.height / 2
    z = ground_z + c * 0.7
    shade = 0.38 + 0.08 * ((k * 37) % 5) / 4
    geom = f'<geometry><ellipsoid><radii>{a:.3f} {b:.3f} {c:.3f}</radii></ellipsoid></geometry>'
    return f'''
  <model name="rock_{k}">
    <static>true</static>
    <pose>{rock.x:.3f} {rock.y:.3f} {z:.3f} 0 0 {rock.yaw:.3f}</pose>
    <link name="link">
      <collision name="c">{geom}</collision>
      <visual name="v">{geom}
        <material><ambient>{shade * 0.7:.3f} {shade * 0.7:.3f} {shade * 0.7:.3f} 1</ambient>
          <diffuse>{shade:.3f} {shade * 0.98:.3f} {shade * 0.96:.3f} 1</diffuse><specular>0.1 0.1 0.1 1</specular></material>
      </visual>
    </link>
  </model>'''


def _crystal_sdf(k, x, y, ground_z):
    shards = []
    for j, (dx, dy, roll, pitch, h) in enumerate(((0, 0, 0.0, 0.0, 0.42), (0.09, 0.04, 0.35, 0.2, 0.3),
                                                  (-0.07, 0.06, -0.3, 0.25, 0.26), (0.02, -0.09, 0.2, -0.35, 0.28))):
        shards.append(f'''
      <visual name="shard_{j}">
        <pose>{dx} {dy} {h / 2:.3f} {roll} {pitch} 0</pose>
        <geometry><box><size>0.07 0.07 {h}</size></box></geometry>
        <material><ambient>0.05 0.5 0.55 1</ambient><diffuse>0.15 0.85 0.9 1</diffuse>
          <specular>0.8 0.9 0.9 1</specular><emissive>0.05 0.55 0.65 1</emissive></material>
      </visual>''')
    return f'''
  <model name="crystal_{k}">
    <static>true</static>
    <pose>{x:.3f} {y:.3f} {ground_z:.3f} 0 0 {k * 0.7:.2f}</pose>
    <link name="link">
      <collision name="c"><pose>0 0 0.15 0 0 0</pose><geometry><cylinder><radius>0.14</radius><length>0.3</length></cylinder></geometry></collision>
      {''.join(shards)}
    </link>
  </model>'''


GZ_CLOCK_TOPIC = '/gz/clock'      # Gazebo's per-step clock; world_interface_node republishes it throttled on /clock


def _e(ros, gz, ros_t, gz_t, direction='GZ_TO_ROS'):
    return {'ros_topic_name': ros, 'gz_topic_name': gz, 'ros_type_name': ros_t, 'gz_type_name': gz_t,
            'direction': direction}


def cmd_topic_ros(name: str) -> str:
    """ROS topic bridged to the robot's gz DiffDrive / VelocityControl `/<name>/cmd_vel`. Every robot drives through
    world_interface_node's relay `cmd_vel` → `cmd_vel_hw`: a rover's is scaled by its wheel health (protocol §8.2),
    the scout's carries the storm-wind drift (§10.1)."""
    return f'/{name}/cmd_vel_hw'


# --- scout command relay: storm wind drift (world_interface_node) -------------------------------------------------
WIND_MIN_AGL_M = 0.3          # below this the scout sits on its skids: the wind does not slide a landed drone


def world_to_body_xy(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """A world-frame horizontal vector in the body frame of a level robot heading `yaw`: the rotation the scout
    planner applies to its own commands, since VelocityControl's linear velocity is body-frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx + s * vy, -s * vx + c * vy


def wind_drift_body(wind_xy_world, yaw: float, h_agl: float) -> tuple[float, float]:
    """Body-frame velocity (m/s) the relay adds to the scout's command: the world wind at its height
    (rover_hw.StormWind.vector), rotated into the body frame; none while it is landed (h_agl < WIND_MIN_AGL_M)."""
    wx, wy = float(wind_xy_world[0]), float(wind_xy_world[1])
    if h_agl < WIND_MIN_AGL_M or (wx == 0.0 and wy == 0.0):
        return 0.0, 0.0
    return world_to_body_xy(wx, wy, yaw)


# --- toppling basalt pillar (world_interface_node `pillar_collapse`, protocol §10.1) --------------------------------
BASALT_RGB = (20, 19, 18)     # scene_events.PILLAR['albedo_rgb']: fresh basalt (albedo ≈ 0.08), diffuse = albedo / 255.
                              # Gazebo's camera renders it at HSV V ≈ 70 (gamma-encoded), the boulders (0.38-0.46) far lighter
PILLAR_MU = 1.2               # ground friction on every pillar collision (≥ 1.0): the base pivots instead of skating
TOPPLE_MARGIN_DEG = 2.5       # spawn tilt past the flat base's balance angle, so gravity topples it, never rights it
KEEL_PROTRUDE_M = 0.03        # anti-roll keel: a flat face this far beyond the trunk on the side that lands down ...
KEEL_THICK_M = 0.06           # ... this thick,
KEEL_HALF_WIDTH = 0.75        # ... half as wide as this × radius (a fallen trunk holds on slopes up to ≈ 35°),
KEEL_END_FRAC = 0.05          # ... and clear of each end by this fraction of the height (off the ground while standing)


def pillar_balance_deg(height: float, radius: float) -> float:
    """Tilt (deg) at which a flat-based solid cylinder standing on its base rim balances, atan(2r / h): released at
    less, its centre of mass is still over the base and it rights itself (6.65° for the 6 m × 0.35 m pillar)."""
    return math.degrees(math.atan2(2.0 * radius, height))


def pillar_spawn_tilt_deg(tilt0_deg: float, height: float, radius: float) -> float:
    """Spawn tilt: the requested one, raised past the balance angle by TOPPLE_MARGIN_DEG when it would not topple.
    (The mock's thin-rod model falls from any tilt; a real 0.7 m-wide base does not.)"""
    return max(float(tilt0_deg), pillar_balance_deg(height, radius) + TOPPLE_MARGIN_DEG)


def disc_ground_z(height_at, x: float, y: float, radius: float, n: int = 12) -> float:
    """Highest ground under a disc of `radius` at (x, y) (centre + n rim points): a base spawned there never starts
    inside a slope. `height_at` is vectorised (terrain_gen.Terrain.height_at)."""
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    xs = np.concatenate([[x], x + radius * np.cos(a)])
    ys = np.concatenate([[y], y + radius * np.sin(a)])
    return float(np.max(height_at(xs, ys)))


def pillar_model_sdf(name: str, base_xyz, fall_dir, tilt0_deg: float, height: float, radius: float, mass: float,
                     albedo_rgb=BASALT_RGB) -> str:
    """A dynamic basalt pillar for EntityFactory: one link, a solid cylinder (radius × height, inertia of a solid
    cylinder) whose model origin is the centre of its base, leaning pillar_spawn_tilt_deg toward `fall_dir` (unit
    world xy) so gravity topples it across the route. `base_xyz` = the ground point under the base (terrain + a
    clearance): the base centre is lifted by r·sin(tilt), so its low rim, the pivot, sits there.

    Pose = (0, tilt, atan2(fall_dir)): the fall turns the trunk about its own y axis, so its +x side lands down. That
    side carries a thin flat keel collision a few cm proud of the trunk, so the fallen cylinder rests on a face instead
    of rolling down a slope and off the route. The keel is collision only; cameras and LiDAR see the cylinder."""
    bx, by, bz = (float(v) for v in base_xyz)
    h, r, m = float(height), float(radius), float(mass)
    tilt = math.radians(pillar_spawn_tilt_deg(tilt0_deg, h, r))
    yaw = math.atan2(float(fall_dir[1]), float(fall_dir[0]))
    z0 = bz + r * math.sin(tilt)
    i_side, i_axis = m * (3.0 * r * r + h * h) / 12.0, m * r * r / 2.0
    keel_x = r + KEEL_PROTRUDE_M - KEEL_THICK_M / 2.0
    keel_len = h * (1.0 - 2.0 * KEEL_END_FRAC)
    dr, dg, db = (c / 255.0 for c in albedo_rgb)
    mid = f'<pose>0 0 {h / 2:.4f} 0 0 0</pose>'
    cylinder = f'<geometry><cylinder><radius>{r:.4f}</radius><length>{h:.4f}</length></cylinder></geometry>'
    friction = f'<surface><friction><ode><mu>{PILLAR_MU}</mu><mu2>{PILLAR_MU}</mu2></ode></friction></surface>'
    inertia = (f'<inertia><ixx>{i_side:.4f}</ixx><iyy>{i_side:.4f}</iyy><izz>{i_axis:.4f}</izz>'
               f'<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>')
    return f'''<sdf version="1.10"><model name="{name}">
  <pose>{bx:.3f} {by:.3f} {z0:.3f} 0 {tilt:.5f} {yaw:.5f}</pose>
  <link name="link">
    <inertial>{mid}<mass>{m:.3f}</mass>{inertia}</inertial>
    <collision name="trunk">{mid}{cylinder}{friction}</collision>
    <collision name="keel"><pose>{keel_x:.4f} 0 {h / 2:.4f} 0 0 0</pose>
      <geometry><box><size>{KEEL_THICK_M:.4f} {2 * KEEL_HALF_WIDTH * r:.4f} {keel_len:.4f}</size></box></geometry>
      {friction}
    </collision>
    <visual name="trunk_visual">{mid}{cylinder}
      <material><ambient>{0.6 * dr:.3f} {0.6 * dg:.3f} {0.6 * db:.3f} 1</ambient>
        <diffuse>{dr:.3f} {dg:.3f} {db:.3f} 1</diffuse><specular>0.05 0.05 0.05 1</specular></material>
    </visual>
  </link>
</model></sdf>'''


def _bridge_entries(name: str) -> list[dict]:
    e = _e
    out = [
        e(cmd_topic_ros(name), f'/{name}/cmd_vel', 'geometry_msgs/msg/Twist', 'gz.msgs.Twist', 'ROS_TO_GZ'),
        e(f'/{name}/odom', f'/{name}/odom', 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry'),
        e(f'/{name}/imu_raw', f'/{name}/imu', 'sensor_msgs/msg/Imu', 'gz.msgs.IMU'),
        e(f'/{name}/camera/image_raw', f'/{name}/camera/image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'),
        e(f'/{name}/camera/depth_raw', f'/{name}/camera/depth_image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'),
        e(f'/{name}/camera/camera_info', f'/{name}/camera/camera_info', 'sensor_msgs/msg/CameraInfo',
          'gz.msgs.CameraInfo'),
    ]
    if name.startswith('rover'):
        out += [
            e(f'/{name}/wheel_odom', f'/{name}/wheel_odom', 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry'),
            e(f'/{name}/lidar/points_raw', f'/{name}/lidar/points', 'sensor_msgs/msg/PointCloud2',
              'gz.msgs.PointCloudPacked'),
        ]
    return out


STEP_S = 0.004               # physics step; the world interface keeps it when changing the speed at run time


def sim_speed_of(cfg: dict) -> float:
    """Target real-time factor from a MissionConfig dict (validated 0.25–3.0 by the backend)."""
    try:
        return min(4.0, max(0.1, float(cfg.get('sim_speed', 1.0))))
    except (TypeError, ValueError):
        return 1.0


def build_world(cfg: dict, out_dir) -> dict:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    size_m, res = SECTORS[cfg.get('sector', 'medium')]
    g_mult = float(cfg.get('gravity_g', 2.2))
    speed = sim_speed_of(cfg)
    seed = int(cfg.get('seed', 42))
    names = robot_names(cfg.get('fleet', {'rovers': 1, 'drones': 1}))
    terr = generate_terrain(size_m, seed)
    terr.save(out / 'terrain.npz')

    hmax = float(terr.heights.max())
    img = np.round(terr.heights / max(hmax, 1e-6) * 65535).astype(np.uint16)
    if FLIP_HEIGHTMAP_ROWS:
        img = np.flipud(img)
    cv2.imwrite(str(out / 'heightmap.png'), img)
    _write_textures(out, seed)

    spawn = {}
    rover_k = 0
    for name in names:
        if name.startswith('rover'):
            x, y = terr.spawn_xy[rover_k]
            rover_k += 1
            yaw = math.atan2(-y, -x)
            spawn[name] = (x, y, terr.height_at(x, y) + ROVER_SPEC['base_height_m'] + 0.05, yaw)
        else:
            x, y = terr.scout_spawn_xy
            spawn[name] = (x, y, terr.height_at(x, y) + 0.12, 0.0)

    robots_sdf = ''.join(rover_sdf(n, spawn[n]) if n.startswith('rover') else scout_sdf(n, spawn[n]) for n in names)
    rocks_sdf = ''.join(_rock_sdf(k, r, terr.height_at(r.x, r.y)) for k, r in enumerate(terr.rocks))
    crystals_sdf = ''.join(_crystal_sdf(k, x, y, terr.height_at(x, y)) for k, (x, y) in enumerate(terr.science))
    tex_size = 14.0
    world = f'''<?xml version="1.0"?>
<sdf version="1.10">
<world name="{WORLD_NAME}">
  <gravity>0 0 {-G_EARTH * g_mult:.4f}</gravity>
  <physics name="4ms" type="ignored">
    <max_step_size>{STEP_S}</max_step_size>
    <real_time_factor>{speed:g}</real_time_factor>
    <dart><collision_detector>bullet</collision_detector></dart>
  </physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
  <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
  <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
  <scene>
    <ambient>0.48 0.38 0.32 1</ambient>
    <background>0.70 0.44 0.30 1</background>
    <shadows>false</shadows>
    <grid>false</grid>
  </scene>
  <light type="directional" name="sun">
    <cast_shadows>false</cast_shadows>
    <pose>0 0 50 0 0 0</pose>
    <diffuse>1.0 0.86 0.72 1</diffuse>
    <specular>0.3 0.25 0.2 1</specular>
    <direction>-0.55 0.35 -0.75</direction>
  </light>
  <model name="terrain">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><heightmap>
          <uri>file://{out / 'heightmap.png'}</uri>
          <size>{size_m} {size_m} {hmax:.4f}</size>
          <pos>0 0 0</pos>
        </heightmap></geometry>
        <surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface>
      </collision>
      <visual name="visual">
        <geometry><heightmap>
          <use_terrain_paging>false</use_terrain_paging>
          <texture>
            <diffuse>file://{out / 'regolith_diffuse.png'}</diffuse>
            <normal>file://{out / 'regolith_normal.png'}</normal>
            <size>{tex_size}</size>
          </texture>
          <uri>file://{out / 'heightmap.png'}</uri>
          <size>{size_m} {size_m} {hmax:.4f}</size>
          <pos>0 0 0</pos>
        </heightmap></geometry>
      </visual>
    </link>
  </model>
{rocks_sdf}
{crystals_sdf}
{robots_sdf}
</world>
</sdf>
'''
    (out / 'world.sdf').write_text(world)
    entries = [_e(GZ_CLOCK_TOPIC, '/clock', 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock')]
    entries += [e for n in names for e in _bridge_entries(n)]
    (out / 'bridge.yaml').write_text(yaml.safe_dump(entries, sort_keys=False))
    meta = {
        'world_name': WORLD_NAME, 'world_path': str(out / 'world.sdf'),
        'heightmap_path': str(out / 'heightmap.png'), 'terrain_path': str(out / 'terrain.npz'),
        'bridge_path': str(out / 'bridge.yaml'), 'size_m': size_m, 'res': res, 'gravity_g': g_mult,
        'sim_speed': speed, 'max_step_s': STEP_S,
        'height_max': hmax, 'robots': names, 'spawn': {k: [round(float(v), 4) for v in p] for k, p in spawn.items()},
        'n_rocks': len(terr.rocks), 'n_science': len(terr.science),
    }
    (out / 'world_meta.json').write_text(json.dumps(meta, indent=2))
    return meta
