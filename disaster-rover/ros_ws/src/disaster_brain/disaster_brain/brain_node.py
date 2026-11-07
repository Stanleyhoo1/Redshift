"""brain_node: the ROS 2 brain. Subscribes to the rover's sensor topics (published by the web sim through
rosbridge), runs BrainCore once per sensor frame (/rover_status closes a frame), publishes /cmd_vel,
/drop_probe and /decision_log. Also validates Gemma's tool calls (/agent_proposal) with the rule set.
"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float32, Float32MultiArray
from geometry_msgs.msg import Twist, Point, Vector3Stamped
from sensor_msgs.msg import LaserScan, Imu, FluidPressure
from nav_msgs.msg import Odometry

from .brain_core import BrainCore


def _hdr(h):
    return {'stamp': {'sec': h.stamp.sec, 'nanosec': h.stamp.nanosec}, 'frame_id': h.frame_id}


def _v3(v):
    return {'x': v.x, 'y': v.y, 'z': v.z}


def _q(q):
    return {'x': q.x, 'y': q.y, 'z': q.z, 'w': q.w}


CONVERT = {
    'scan': lambda m: {'header': _hdr(m.header), 'angle_min': m.angle_min, 'angle_max': m.angle_max, 'angle_increment': m.angle_increment,
                       'range_min': m.range_min, 'range_max': m.range_max, 'ranges': list(m.ranges), 'intensities': list(m.intensities)},
    'odom': lambda m: {'header': _hdr(m.header), 'pose': {'pose': {'position': _v3(m.pose.pose.position), 'orientation': _q(m.pose.pose.orientation)}},
                       'twist': {'twist': {'linear': _v3(m.twist.twist.linear), 'angular': _v3(m.twist.twist.angular)}}},
    'imu': lambda m: {'header': _hdr(m.header), 'orientation': _q(m.orientation), 'angular_velocity': _v3(m.angular_velocity),
                      'linear_acceleration': _v3(m.linear_acceleration)},
    'thermal': lambda m: {'data': list(m.data)},
    'water_depth': lambda m: {'data': m.data},
    'wind': lambda m: {'header': _hdr(m.header), 'vector': _v3(m.vector)},
    'pressure': lambda m: {'header': _hdr(m.header), 'fluid_pressure': m.fluid_pressure},
    'elevation': lambda m: {'data': list(m.data)},
    'camera_water': lambda m: {'data': list(m.data)},
    'camera_detections': lambda m: {'data': list(m.data)},
}

TOPICS = [
    ('scan', '/scan', LaserScan), ('odom', '/odom', Odometry), ('imu', '/imu', Imu), ('thermal', '/thermal', Float32MultiArray),
    ('water_depth', '/water_depth', Float32), ('wind', '/wind', Vector3Stamped), ('pressure', '/pressure', FluidPressure),
    ('elevation', '/elevation', Float32MultiArray), ('camera_water', '/camera/water', Float32MultiArray),
    ('camera_detections', '/camera/detections', Float32MultiArray),
]


class BrainNode(Node):
    def __init__(self):
        super().__init__('disaster_brain')
        self.core = BrainCore()
        self.latest = {}
        self.last_t = -1.0
        self.tick_ms = []
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        for key, topic, typ in TOPICS:
            self.create_subscription(typ, topic, lambda m, k=key: self.latest.__setitem__(k, CONVERT[k](m)), qos)
        self.create_subscription(String, '/rover_status', self.on_status, qos)
        self.create_subscription(String, '/agent_proposal', self.on_proposal, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_probe = self.create_publisher(Point, '/drop_probe', 10)
        self.pub_log = self.create_publisher(String, '/decision_log', 50)
        self.pub_agent_log = self.create_publisher(String, '/agent_log', 20)
        self.pub_summary = self.create_publisher(String, '/brain_summary', 5)
        self.pub_info = self.create_publisher(String, '/brain_info', 5)
        self.pub_map = self.create_publisher(String, '/hazard_map', 2)
        self.last_summary = -1e9
        self.last_info = -1e9
        self.last_map_pub = -1e9
        self.get_logger().info('disaster_brain up: waiting for sensor frames from rosbridge')

    def on_status(self, msg):
        frame = dict(self.latest)
        frame['rover_status'] = {'data': msg.data}
        if any(k not in frame for k, _, _ in TOPICS):
            return
        st = json.loads(msg.data)
        if st['t'] < self.last_t - 1.0:
            self.get_logger().info('sim time went backwards (reset): new belief')
            self.core = BrainCore()
        self.last_t = st['t']
        # the scan stamp defines frame time: keep the status-time if the scan lags (topics arrive independently)
        t0 = time.perf_counter()
        out = self.core.tick(frame)
        self.tick_ms.append((time.perf_counter() - t0) * 1000)
        c = out['cmd_vel']
        tw = Twist()
        tw.linear.x, tw.angular.z = float(c['linear']['x']), float(c['angular']['z'])
        self.pub_cmd.publish(tw)
        if out['drop_probe']:
            p = Point()
            p.x, p.y = float(out['drop_probe']['x']), float(out['drop_probe']['y'])
            self.pub_probe.publish(p)
        if out['decision_log']:
            self.pub_log.publish(String(data=out['decision_log']))
        t = self.core.rd.t if self.core.rd else 0.0
        events = self.core.events
        if events or t - self.last_summary > 2.0:
            self.core.events = []
            self.last_summary = t
            self.pub_summary.publish(String(data=json.dumps({'t': t, 'summary': self.core.summary(), 'events': events})))
        if t - self.last_info > 0.25:
            self.last_info = t
            info = self.core.info()
            if len(self.tick_ms) > 0:
                s = sorted(self.tick_ms[-200:])
                info['tick_ms'] = round(s[len(s) // 2], 2)
            self.pub_info.publish(String(data=json.dumps(info)))
        if t - self.last_map_pub > 1.0:
            self.last_map_pub = t
            self.pub_map.publish(String(data=json.dumps(self.core.hazard_map_msg())))

    def on_proposal(self, msg):
        """Gemma proposed a tool call: validate with the rule set, log ACCEPTED/REJECTED."""
        try:
            p = json.loads(msg.data)
        except ValueError:
            return
        lat = p.get('latency_s', 0.0)
        if p.get('error'):
            self.pub_agent_log.publish(String(data='[AGENT] %s → IGNORED (%s) | %.1f s' % (p.get('summary_short', ''), p['error'], lat)))
            return
        call = p.get('tool_call')
        res = self.core.propose_goal(call) if call else dict(ok=False, rule='no_tool_call', detail='no parsable tool call')
        args = call.get('args', {}) if isinstance(call, dict) else {}
        reason = args.get('reason', '') if isinstance(args, dict) else ''
        verdict = 'ACCEPTED' if res['ok'] else 'REJECTED (%s: %s)' % (res['rule'], res['detail'])
        argtxt = ', '.join('%s=%s' % (k, v) for k, v in args.items() if k != 'reason') if isinstance(args, dict) else ''
        line = '[AGENT] %s → %s(%s) → "%s" → %s | %.1f s' % (p.get('summary_short', ''), call.get('name') if isinstance(call, dict) else '?', argtxt, reason, verdict, lat)
        self.pub_agent_log.publish(String(data=line))
        self.get_logger().info(line)


def main():
    rclpy.init()
    node = BrainNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception) as e:  # noqa: BLE001
        if not isinstance(e, KeyboardInterrupt):
            node.get_logger().error('brain_node crashed: %r' % e)
            raise
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
