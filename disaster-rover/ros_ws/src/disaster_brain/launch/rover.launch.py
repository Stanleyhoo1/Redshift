"""Starts rosbridge (ws://localhost:9090), the disaster_brain node and (optionally) the Gemma 4 mission agent.

  ros2 launch disaster_brain rover.launch.py                  # brain + agent
  ros2 launch disaster_brain rover.launch.py use_agent:=false # utility brain only
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rb = os.path.join(get_package_share_directory('rosbridge_server'), 'launch', 'rosbridge_websocket_launch.xml')
    return LaunchDescription([
        DeclareLaunchArgument('use_agent', default_value='true', description='start the Gemma 4 mission_agent'),
        DeclareLaunchArgument('model', default_value='gemma4:e2b'),
        DeclareLaunchArgument('ollama_url', default_value='http://localhost:11434'),
        DeclareLaunchArgument('port', default_value='9090'),
        IncludeLaunchDescription(AnyLaunchDescriptionSource(rb), launch_arguments={'port': LaunchConfiguration('port')}.items()),
        Node(package='disaster_brain', executable='brain_node', name='disaster_brain', output='screen'),
        Node(package='disaster_brain', executable='mission_agent', name='mission_agent', output='screen',
             condition=IfCondition(LaunchConfiguration('use_agent')),
             parameters=[{'model': LaunchConfiguration('model'), 'ollama_url': LaunchConfiguration('ollama_url')}]),
    ])
