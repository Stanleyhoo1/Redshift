import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
import yaml

from redshift_sim.robot_models import ROVER_SPEC, SCOUT_SPEC, rover_sdf, scout_sdf
from redshift_sim.terrain_gen import Terrain
from redshift_sim.scene_events import PILLAR
from redshift_sim.world_builder import (BASALT_RGB, GZ_CLOCK_TOPIC, PILLAR_MU, STEP_S, WIND_MIN_AGL_M, WORLD_NAME,
                                        build_world, cmd_topic_ros, disc_ground_z, pillar_balance_deg, pillar_model_sdf,
                                        pillar_spawn_tilt_deg, robot_names, wind_drift_body, world_to_body_xy)

CFG = {'sector': 'small', 'gravity_g': 2.2, 'fleet': {'rovers': 2, 'drones': 1}, 'seed': 3}


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp('runtime')
    return build_world(CFG, out), out


def test_robot_names():
    assert robot_names({'rovers': 2, 'drones': 1}) == ['rover_0', 'rover_1', 'scout_0']
    assert robot_names({'rovers': 1, 'drones': 0}) == ['rover_0']


def test_world_sdf_parses_with_gravity_and_models(built):
    meta, out = built
    root = ET.parse(meta['world_path']).getroot()
    world = root.find('world')
    assert world.get('name') == WORLD_NAME
    gz = [float(v) for v in world.find('gravity').text.split()]
    assert gz[2] == pytest.approx(-9.81 * 2.2, abs=1e-3)
    names = {m.get('name') for m in world.findall('model')}
    assert {'terrain', 'rover_0', 'rover_1', 'scout_0'} <= names
    terr = Terrain.load(meta['terrain_path'])
    assert sum(n.startswith('rock_') for n in names) == len(terr.rocks)
    assert sum(n.startswith('crystal_') for n in names) == len(terr.science)
    plugins = {p.get('name') for p in world.findall('plugin')}
    assert {'gz::sim::systems::Physics', 'gz::sim::systems::Sensors', 'gz::sim::systems::UserCommands',
            'gz::sim::systems::Imu', 'gz::sim::systems::ApplyLinkWrench'} <= plugins


def test_world_physics_paced_by_sim_speed(tmp_path):
    for speed, expect in ((None, 1.0), (2.0, 2.0), (0.5, 0.5)):
        cfg = dict(CFG) if speed is None else dict(CFG, sim_speed=speed)
        meta = build_world(cfg, tmp_path / f's{speed}')
        phys = ET.parse(meta['world_path']).getroot().find('world/physics')
        assert float(phys.find('real_time_factor').text) == pytest.approx(expect)
        assert float(phys.find('max_step_size').text) == pytest.approx(STEP_S)
        assert meta['sim_speed'] == pytest.approx(expect) and meta['max_step_s'] == pytest.approx(STEP_S)


def test_heightmap_png(built):
    meta, _ = built
    img = cv2.imread(meta['heightmap_path'], cv2.IMREAD_UNCHANGED)
    terr = Terrain.load(meta['terrain_path'])
    assert img.dtype == np.uint16 and img.shape == (terr.n, terr.n)
    assert img.max() == 65535 and img.min() == 0
    assert (terr.n - 1) & (terr.n - 2) == 0          # 2^k + 1


def test_bridge_yaml(built):
    meta, _ = built
    entries = yaml.safe_load(open(meta['bridge_path']))
    by_robot = {}
    for e in entries:
        by_robot.setdefault(e['ros_topic_name'].split('/')[1], []).append(e)
    assert len(by_robot['rover_0']) == 8 and len(by_robot['scout_0']) == 6
    raw = {e['ros_topic_name'] for e in by_robot['rover_0']}
    assert {'/rover_0/camera/image_raw', '/rover_0/camera/depth_raw', '/rover_0/lidar/points_raw',
            '/rover_0/imu_raw', '/rover_0/odom', '/rover_0/wheel_odom', '/rover_0/cmd_vel_hw'} <= raw
    # every robot drives through the world interface's relay: ROS cmd_vel_hw → gz cmd_vel (never ROS cmd_vel); the
    # rovers' relay applies the drivetrain health, the scout's adds the storm wind drift
    for r in ('rover_0', 'rover_1', 'scout_0'):
        cmd = [e for e in entries if e['gz_topic_name'] == f'/{r}/cmd_vel']
        assert len(cmd) == 1 and cmd[0]['ros_topic_name'] == f'/{r}/cmd_vel_hw' == cmd_topic_ros(r)
        assert cmd[0]['direction'] == 'ROS_TO_GZ' and cmd[0]['ros_type_name'] == 'geometry_msgs/msg/Twist'
        assert not [e for e in entries if e['ros_topic_name'] == f'/{r}/cmd_vel']
    assert '/scout_0/cmd_vel_hw' in {e['ros_topic_name'] for e in by_robot['scout_0']}
    clock = [e for e in entries if e['gz_topic_name'] == '/clock']
    assert len(clock) == 1 and clock[0]['ros_topic_name'] == GZ_CLOCK_TOPIC and clock[0]['direction'] == 'GZ_TO_ROS'


def test_spawn_poses_sit_on_terrain(built):
    meta, _ = built
    terr = Terrain.load(meta['terrain_path'])
    for name, (x, y, z, yaw) in meta['spawn'].items():
        ground = terr.height_at(x, y)
        assert ground < z < ground + 0.6, name


def test_rover_sdf_physics_matches_spec():
    root = ET.fromstring('<sdf version="1.10">' + rover_sdf('rover_0', (0, 0, 1, 0)) + '</sdf>')
    m = root.find('model')
    efforts = [float(j.find('axis/limit/effort').text) for j in m.findall('joint')]
    assert len(efforts) == 4 and all(e == pytest.approx(ROVER_SPEC['tau_max_nm']) for e in efforts)
    mass = sum(float(l.find('inertial/mass').text) for l in m.findall('link'))
    assert mass == pytest.approx(ROVER_SPEC['mass_kg'], abs=0.01)
    sensors = {s.get('type') for s in m.iter('sensor')}
    assert sensors == {'imu', 'gpu_lidar', 'rgbd_camera'}


def test_scout_sdf_has_velocity_control_and_down_camera():
    root = ET.fromstring('<sdf version="1.10">' + scout_sdf('scout_0', (0, 0, 1, 0)) + '</sdf>')
    m = root.find('model')
    assert 'gz::sim::systems::VelocityControl' in {p.get('name') for p in m.findall('plugin')}
    cam = [s for s in m.iter('sensor') if s.get('type') == 'rgbd_camera'][0]
    pitch = float(cam.find('pose').text.split()[4])
    assert pitch == pytest.approx(math.pi / 2, abs=1e-3)
    assert SCOUT_SPEC['cruise_agl_m'] == 12.0


# ---------------------------------------------------------------- pillar collapse (Gazebo model, spec §1.1)
PILLAR_ARGS = dict(name='pillar_1', base_xyz=(3.0, -2.0, 1.25), fall_dir=(0.6, 0.8), tilt0_deg=4.0, height=6.0,
                   radius=0.35, mass=1500.0)


def _pillar(**kw):
    return ET.fromstring(pillar_model_sdf(**dict(PILLAR_ARGS, **kw))).find('model')


def _rpy_matrix(r, p, y):
    """SDF / ROS fixed-axis roll-pitch-yaw: R = Rz(y)·Ry(p)·Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _floats(el):
    return [float(v) for v in el.text.split()]


def _collisions(link):
    return {c.get('name'): c for c in link.findall('collision')}


def test_pillar_model_is_one_dynamic_link_with_solid_cylinder_mass_and_inertia():
    m = _pillar()
    assert m.get('name') == 'pillar_1'
    assert m.find('static') is None or m.find('static').text.strip() in ('false', '0')
    links = m.findall('link')
    assert len(links) == 1
    inertial = links[0].find('inertial')
    assert float(inertial.find('mass').text) == pytest.approx(1500.0)
    inertia = {k: float(inertial.find(f'inertia/{k}').text) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')}
    i_side = 1500.0 * (3 * 0.35 ** 2 + 6.0 ** 2) / 12
    assert inertia['ixx'] == pytest.approx(i_side, rel=1e-4) and inertia['iyy'] == pytest.approx(i_side, rel=1e-4)
    assert inertia['izz'] == pytest.approx(1500.0 * 0.35 ** 2 / 2, rel=1e-4)
    assert inertia['ixy'] == inertia['ixz'] == inertia['iyz'] == 0.0
    assert _floats(inertial.find('pose')) == pytest.approx([0, 0, 3.0, 0, 0, 0])      # centre of mass mid-trunk


def test_pillar_trunk_collision_and_visual_are_the_cylinder_in_dark_basalt():
    link = _pillar().find('link')
    names = [e.get('name') for e in link if e.get('name')]
    assert len(names) == len(set(names))                     # SDF rejects a collision and a visual sharing a name
    trunk = _collisions(link)['trunk']
    vis = link.find('visual')
    for el in (trunk, vis):
        cyl = el.find('geometry/cylinder')
        assert float(cyl.find('radius').text) == pytest.approx(0.35)
        assert float(cyl.find('length').text) == pytest.approx(6.0)
        assert _floats(el.find('pose')) == pytest.approx([0, 0, 3.0, 0, 0, 0])      # base at the model origin
    diffuse = _floats(vis.find('material/diffuse'))
    assert diffuse[:3] == pytest.approx([c / 255 for c in BASALT_RGB], abs=2e-3) and BASALT_RGB == PILLAR['albedo_rgb']
    assert max(diffuse[:3]) <= 0.35                                                 # dark: the PILLAR vision cue
    amber = _floats(_pillar(albedo_rgb=(200, 100, 50)).find('link/visual/material/diffuse'))
    assert amber[:3] == pytest.approx([200 / 255, 100 / 255, 50 / 255], abs=2e-3)


def test_pillar_ground_friction_keeps_the_base_from_skating():
    link = _pillar().find('link')
    assert PILLAR_MU >= 1.0
    for c in link.findall('collision'):
        ode = c.find('surface/friction/ode')
        assert float(ode.find('mu').text) >= 1.0 and float(ode.find('mu2').text) >= 1.0, c.get('name')


def test_pillar_spawns_tilted_toward_the_route_with_its_low_rim_on_the_base_point():
    m = _pillar()
    x, y, z, r, p, yw = _floats(m.find('pose'))
    R = _rpy_matrix(r, p, yw)
    axis = R @ np.array([0.0, 0.0, 1.0])
    assert math.degrees(math.acos(axis[2])) == pytest.approx(pillar_spawn_tilt_deg(4.0, 6.0, 0.35), abs=1e-3)
    assert axis[:2] / np.linalg.norm(axis[:2]) == pytest.approx([0.6, 0.8], abs=1e-4)   # leans along fall_dir
    assert (x, y) == pytest.approx((3.0, -2.0))
    rim = [R @ np.array([0.35 * math.cos(a), 0.35 * math.sin(a), 0.0]) + [x, y, z]
           for a in np.linspace(0, 2 * math.pi, 73)]
    assert min(q[2] for q in rim) == pytest.approx(1.25, abs=1e-3)       # never spawned into the ground
    low = min(rim, key=lambda q: q[2])
    assert (low[:2] - [x, y]) @ np.array([0.6, 0.8]) > 0.3                # the pivot is on the route side


def test_pillar_spawn_tilt_is_past_the_flat_base_balance_so_gravity_topples_it():
    assert pillar_balance_deg(6.0, 0.35) == pytest.approx(math.degrees(math.atan(0.7 / 6.0)))      # ≈ 6.65°
    th = math.radians(pillar_spawn_tilt_deg(4.0, 6.0, 0.35))
    assert 3.0 * math.sin(th) > 0.35 * math.cos(th)                  # centre of mass beyond the pivot rim
    assert math.degrees(th) < 15.0                                   # still a standing pillar, not pre-fallen
    assert pillar_spawn_tilt_deg(12.0, 6.0, 0.35) == pytest.approx(12.0)                 # a steeper request is kept
    m = _pillar(tilt0_deg=12.0)
    assert _floats(m.find('pose'))[4] == pytest.approx(math.radians(12.0), abs=1e-4)


def test_fallen_pillar_rests_on_its_keel_instead_of_rolling_off_the_route():
    m = _pillar()
    yw = _floats(m.find('pose'))[5]
    link = m.find('link')
    keel = _collisions(link)['keel']
    kx, ky, kz = _floats(keel.find('pose'))[:3]
    sx, sy, sz = _floats(keel.find('geometry/box/size'))
    # it falls by rotating about its own y axis, so the link +x side ends up facing the ground
    assert (_rpy_matrix(0.0, math.pi / 2, yw) @ np.array([1.0, 0.0, 0.0]))[2] == pytest.approx(-1.0)
    face = kx + sx / 2
    assert ky == 0.0 and face > 0.35                                  # the flat face protrudes beyond the trunk
    assert face - 0.35 <= 0.05                                        # ... by a few cm (visual stays on the ground)
    # resting on a flat face sy wide with the centre of mass `face` above it: holds on slopes beyond 30°
    assert math.degrees(math.atan((sy / 2) / face)) > 30.0
    assert kz == pytest.approx(3.0) and 0.8 * 6.0 <= sz < 6.0         # along the trunk, clear of both ends
    # while the pillar stands at its spawn tilt the keel is off the ground (only the base rim touches)
    x, y, z, r, p, _ = _floats(m.find('pose'))
    R = _rpy_matrix(r, p, yw)
    corners = [R @ np.array([face, sgn * sy / 2, kz - sz / 2]) + [x, y, z] for sgn in (-1, 1)]
    assert min(c[2] for c in corners) > 1.25 + 0.1


def test_disc_ground_z_is_the_highest_ground_under_the_base(built):
    def slope(x, y):
        return 0.2 * np.asarray(x) + 0.1 * np.asarray(y)

    assert disc_ground_z(slope, 1.0, 2.0, 0.35) == pytest.approx(0.2 + 0.2 + 0.35 * math.hypot(0.2, 0.1), abs=2e-3)
    assert disc_ground_z(lambda x, y: np.zeros_like(np.asarray(x, float)), 5.0, 5.0, 0.35) == 0.0
    terr = Terrain.load(built[0]['terrain_path'])
    x, y = terr.spawn_xy[0]
    assert disc_ground_z(terr.height_at, x, y, 0.35) >= terr.height_at(x, y)


# ---------------------------------------------------------------- scout wind relay (spec §1.2)
def test_world_to_body_xy_is_the_scout_planners_world_to_body_rotation():
    assert world_to_body_xy(1.0, 0.0, 0.0) == pytest.approx((1.0, 0.0))
    # heading north, a wind blowing toward the east pushes the drone to its right (body −y)
    assert world_to_body_xy(1.0, 0.0, math.pi / 2) == pytest.approx((0.0, -1.0), abs=1e-12)
    for yaw in np.linspace(-math.pi, math.pi, 9):
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = world_to_body_xy(3.0, -2.0, yaw)
        assert (bx, by) == pytest.approx((c * 3.0 + s * -2.0, -s * 3.0 + c * -2.0))   # agent_planner_node scout
        assert (c * bx - s * by, s * bx + c * by) == pytest.approx((3.0, -2.0))       # body → world restores it


def test_wind_drift_body_rotates_the_world_wind_and_spares_a_landed_drone():
    assert wind_drift_body((0.0, 10.0), math.pi / 2, 12.0) == pytest.approx((10.0, 0.0), abs=1e-9)   # tailwind
    assert wind_drift_body((10.0, 0.0), math.pi, 8.0) == pytest.approx((-10.0, 0.0), abs=1e-9)       # headwind
    assert wind_drift_body((7.0, -3.0), 0.4, WIND_MIN_AGL_M - 0.01) == (0.0, 0.0)                  # on its skids
    assert wind_drift_body((0.0, 0.0), 0.4, 12.0) == (0.0, 0.0)
