import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
import yaml

from redshift_sim.robot_models import ROVER_SPEC, SCOUT_SPEC, rover_sdf, scout_sdf
from redshift_sim.terrain_gen import Terrain
from redshift_sim.world_builder import WORLD_NAME, build_world, robot_names

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
            '/rover_0/imu_raw', '/rover_0/odom', '/rover_0/wheel_odom', '/rover_0/cmd_vel'} <= raw
    cmd = [e for e in entries if e['ros_topic_name'] == '/rover_0/cmd_vel'][0]
    assert cmd['direction'] == 'ROS_TO_GZ'


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
