/**
 * ROS 2 message shapes, mirrored 1:1 so the TS brain and the rclpy brain see
 * identical data. Field names follow the .msg definitions exactly
 * (sensor_msgs/LaserScan, nav_msgs/Odometry, ...), so a frame can be sent to
 * rosbridge as-is.
 */
export interface Time { sec: number; nanosec: number }
export interface Header { stamp: Time; frame_id: string }
export interface Vector3 { x: number; y: number; z: number }
export interface Point { x: number; y: number; z: number }
export interface Quaternion { x: number; y: number; z: number; w: number }
export interface Twist { linear: Vector3; angular: Vector3 }

export interface LaserScan {
  header: Header;
  angle_min: number; angle_max: number; angle_increment: number;
  time_increment: number; scan_time: number;
  range_min: number; range_max: number;
  ranges: number[]; intensities: number[];
}
export interface Odometry {
  header: Header; child_frame_id: string;
  pose: { pose: { position: Point; orientation: Quaternion }; covariance: number[] };
  twist: { twist: Twist; covariance: number[] };
}
export interface Imu {
  header: Header;
  orientation: Quaternion; orientation_covariance: number[];
  angular_velocity: Vector3; angular_velocity_covariance: number[];
  linear_acceleration: Vector3; linear_acceleration_covariance: number[];
}
export interface MultiArrayDimension { label: string; size: number; stride: number }
export interface Float32MultiArray { layout: { dim: MultiArrayDimension[]; data_offset: number }; data: number[] }
export interface Float32 { data: number }
export interface Vector3Stamped { header: Header; vector: Vector3 }
export interface FluidPressure { header: Header; fluid_pressure: number; variance: number }
export interface StringMsg { data: string }

/** Contents of /rover_status (JSON inside std_msgs/String). Only things the rover itself knows. */
export interface RoverStatus {
  seq: number; t: number;
  health: number; battery: number; probes_left: number; alive: boolean;
  /** onboard thermometer: air temperature at the rover body (°C) */
  temp_c: number;
  /** positions of probes this rover dropped, with their latest telemetry */
  probes: ProbeReading[];
}
export interface ProbeReading {
  id: number; x: number; y: number; alive: boolean;
  water: number; temp: number; wind: number; shake: number;
  /** coarse local samples in the probe's sensing radius: [x, y, waterDepth, tempC, elev] */
  samples: number[];
}

/** One full sensor frame = everything the brain is allowed to see. */
export interface SensorFrame {
  scan: LaserScan;
  odom: Odometry;
  imu: Imu;
  /** 36 sectors x [maxTempC, distance m]; dims "sector","field" */
  thermal: Float32MultiArray;
  water_depth: Float32;
  wind: Vector3Stamped;
  pressure: FluidPressure;
  /** depth camera elevation patch: [originX, originY, res, n, ...n*n heights (NaN=unseen)] */
  elevation: Float32MultiArray;
  /** camera water detector: 36 sectors x distance to visible water (-1 none) */
  camera_water: Float32MultiArray;
  /** camera object detections: [class, bearing(rad, world), distance, size] * k. class 1=impact flash 2=funnel 3=incoming streak 4=smoke */
  camera_detections: Float32MultiArray;
  rover_status: StringMsg;
}

export interface BrainOutput {
  cmd_vel: Twist;
  drop_probe: Point | null;
  decision_log: string | null;
  agent_log?: string | null;
}

export const zeroTwist = (): Twist => ({ linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0 } });
export const stampOf = (t: number): Time => ({ sec: Math.floor(t), nanosec: Math.floor((t % 1) * 1e9) });
export const header = (t: number, frame = 'base_link'): Header => ({ stamp: stampOf(t), frame_id: frame });
export const yawToQuat = (yaw: number, roll = 0, pitch = 0): Quaternion => {
  const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2), cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2), cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
  return { w: cr * cp * cy + sr * sp * sy, x: sr * cp * cy - cr * sp * sy, y: cr * sp * cy + sr * cp * sy, z: cr * cp * sy - sr * sp * cy };
};
export const quatToYaw = (q: Quaternion): number => Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
export const quatToRollPitch = (q: Quaternion): [number, number] => {
  const roll = Math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));
  const s = 2 * (q.w * q.y - q.z * q.x);
  const pitch = Math.abs(s) >= 1 ? Math.sign(s) * Math.PI / 2 : Math.asin(s);
  return [roll, pitch];
};
