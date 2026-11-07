import type { SensorFrame, BrainOutput } from './msgs';

/**
 * The single interface both brains implement.
 * Input: one SensorFrame (sensor readings only, ROS message shapes).
 * Output: {cmd_vel, drop_probe, decision_log}.
 * The TS brain (web/brain) and the rclpy brain (ros_ws/src/disaster_brain) are interchangeable behind it.
 */
export interface Brain {
  readonly name: string;
  tick(frame: SensorFrame): BrainOutput;
}

/** Minimal placeholder used in milestone 1: drive forward, turn away from close LiDAR returns. */
export class WanderBrain implements Brain {
  readonly name = 'wander';
  tick(f: SensorFrame): BrainOutput {
    const r = f.scan.ranges, n = r.length;
    let front = Infinity; for (let k = n / 2 - 6; k <= n / 2 + 6; k++) front = Math.min(front, r[k]);
    const left = Math.min(...r.slice(n / 2 + 6, n / 2 + 18)), right = Math.min(...r.slice(n / 2 - 18, n / 2 - 6));
    const v = front < 3 ? 0 : 2;
    const w = front < 5 ? (left > right ? 1.2 : -1.2) : 0;
    return { cmd_vel: { linear: { x: v, y: 0, z: 0 }, angular: { x: 0, y: 0, z: w } }, drop_probe: null, decision_log: null };
  }
}
