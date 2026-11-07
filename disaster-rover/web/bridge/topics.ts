import type { SensorFrame } from '../brain/msgs';

/** web -> ROS: sensor topics. Order matters: /rover_status is published LAST and closes the frame. */
export const SENSOR_TOPICS: { key: keyof SensorFrame; topic: string; type: string }[] = [
  { key: 'scan', topic: '/scan', type: 'sensor_msgs/msg/LaserScan' },
  { key: 'odom', topic: '/odom', type: 'nav_msgs/msg/Odometry' },
  { key: 'imu', topic: '/imu', type: 'sensor_msgs/msg/Imu' },
  { key: 'thermal', topic: '/thermal', type: 'std_msgs/msg/Float32MultiArray' },
  { key: 'water_depth', topic: '/water_depth', type: 'std_msgs/msg/Float32' },
  { key: 'wind', topic: '/wind', type: 'geometry_msgs/msg/Vector3Stamped' },
  { key: 'pressure', topic: '/pressure', type: 'sensor_msgs/msg/FluidPressure' },
  { key: 'elevation', topic: '/elevation', type: 'std_msgs/msg/Float32MultiArray' },
  { key: 'camera_water', topic: '/camera/water', type: 'std_msgs/msg/Float32MultiArray' },
  { key: 'camera_detections', topic: '/camera/detections', type: 'std_msgs/msg/Float32MultiArray' },
  { key: 'rover_status', topic: '/rover_status', type: 'std_msgs/msg/String' },
];
/** ROS -> web */
export const OUT_TOPICS = {
  cmd_vel: { topic: '/cmd_vel', type: 'geometry_msgs/msg/Twist' },
  drop_probe: { topic: '/drop_probe', type: 'geometry_msgs/msg/Point' },
  decision_log: { topic: '/decision_log', type: 'std_msgs/msg/String' },
  agent_log: { topic: '/agent_log', type: 'std_msgs/msg/String' },
  agent_status: { topic: '/agent_status', type: 'std_msgs/msg/String' },
  brain_info: { topic: '/brain_info', type: 'std_msgs/msg/String' },
  hazard_map: { topic: '/hazard_map', type: 'std_msgs/msg/String' },
} as const;
/** web -> agent: UI toggle */
export const AGENT_ENABLE = { topic: '/agent_enable', type: 'std_msgs/msg/Bool' };
