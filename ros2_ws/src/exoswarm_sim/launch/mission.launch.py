"""Redshift mission bring-up.

    ros2 launch exoswarm_sim mission.launch.py config:=/abs/path/mission_config.yaml [autonomy:=true]

The config is a MissionConfig (docs/protocol.md §6) plus optional `runtime_dir`. The world (SDF,
heightmap, terrain.npz, bridge.yaml) is generated into runtime_dir, then either Gazebo Harmonic
(+ ros_gz bridge + world interface) or the mock world is started, followed by the autonomy stack.
"""
import json
import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from exoswarm_sim.robot_models import ROVER_SPEC, SCOUT_SPEC
from exoswarm_sim.world_builder import SECTORS, build_world


def _setup(context):
    cfg_path = Path(LaunchConfiguration('config').perform(context)).resolve()
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    autonomy = LaunchConfiguration('autonomy').perform(context).lower() in ('1', 'true', 'yes')
    runtime = Path(cfg.get('runtime_dir') or cfg_path.parent / 'world').resolve()
    meta = build_world(cfg, runtime)
    backend = cfg.get('sim_backend', 'gazebo')
    g = float(cfg.get('gravity_g', 2.2))
    size_m, res = SECTORS[cfg.get('sector', 'medium')]
    robots = meta['robots']
    edge = cfg.get('edge_profile', 'orin_nano_15w')
    specs = {r: (ROVER_SPEC if r.startswith('rover') else SCOUT_SPEC) for r in robots}
    acts = [LogInfo(msg=f'[redshift] backend={backend} sector={size_m}m g={g} robots={robots} runtime={runtime}')]

    if backend == 'gazebo':
        acts.append(ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', '--headless-rendering', '-v', '2', meta['world_path']],
                                   name='gz_sim', output='screen'))
        if cfg.get('gazebo_gui'):
            acts.append(ExecuteProcess(cmd=['gz', 'sim', '-g', '-v', '1'], name='gz_gui', output='log'))
        acts.append(Node(package='ros_gz_bridge', executable='parameter_bridge', name='redshift_gz_bridge', output='log',
                         parameters=[{'config_file': meta['bridge_path']}]))
        acts.append(Node(package='exoswarm_sim', executable='world_interface_node', name='world_interface',
                         output='screen', parameters=[{'world_name': meta['world_name'], 'terrain_path': meta['terrain_path'],
                                                       'g_mult': g, 'robots': robots}]))
    else:
        acts.append(Node(package='exoswarm_sim', executable='mock_world_node', name='mock_world', output='screen',
                         parameters=[{'terrain_path': meta['terrain_path'], 'robots': robots, 'g_mult': g,
                                      'spawn_json': json.dumps(meta['spawn']), 'specs_json': json.dumps(specs)}]))

    for r in robots:
        rtype = 'rover' if r.startswith('rover') else 'scout'
        acts.append(Node(package='exoswarm_sim', executable='sensor_effects_node', name=f'{r}_sensor_effects',
                         output='log', parameters=[{'robot': r, 'robot_type': rtype,
                                                    'publish_env': r == robots[0]}]))

    if autonomy:
        common = {'g_mult': g, 'edge_profile': edge, 'size_m': size_m, 'map_res': res}
        acts.append(Node(package='exoswarm_autonomy', executable='mapping_coordinator_node', name='mapping_coordinator',
                         output='screen', parameters=[dict(common, robots=robots, specs_json=json.dumps(specs))]))
        for r in robots:
            rtype = 'rover' if r.startswith('rover') else 'scout'
            p = dict(common, robot=r, robot_type=rtype, spec_json=json.dumps(specs[r]), robots=robots)
            acts.append(Node(package='exoswarm_autonomy', executable='perception_node', name=f'{r}_perception',
                             output='screen', parameters=[p]))
            acts.append(Node(package='exoswarm_autonomy', executable='agent_planner_node', name=f'{r}_planner',
                             output='screen', parameters=[p]))
    return acts


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', description='Path to mission_config.yaml'),
        DeclareLaunchArgument('autonomy', default_value='true'),
        SetEnvironmentVariable('GZ_IP', os.environ.get('GZ_IP', '127.0.0.1')),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')),
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),
        OpaqueFunction(function=_setup),
    ])
