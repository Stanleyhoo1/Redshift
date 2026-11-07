"""mission_agent: placeholder until milestone 8 (Gemma 4 commander)."""
import rclpy
from rclpy.node import Node


def main():
    rclpy.init()
    node = Node('mission_agent')
    node.get_logger().info('mission_agent stub')
    rclpy.spin(node)


if __name__ == '__main__':
    main()
