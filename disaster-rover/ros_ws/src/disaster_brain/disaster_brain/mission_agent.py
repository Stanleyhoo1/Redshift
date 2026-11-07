"""mission_agent: the slow strategic layer. Gemma 4 (gemma4:e2b via local Ollama) reads the brain's sensed
situation summary (/brain_summary) and proposes ONE tool call on /agent_proposal. The brain node validates it
with the rule set and logs ACCEPTED / REJECTED on /agent_log. The LLM call runs in a background thread so it
never blocks anything; if it is off, slow or failing, the utility brain simply carries on.
"""
import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

from .agent_core import call_ollama, Trigger


class MissionAgent(Node):
    def __init__(self):
        super().__init__('mission_agent')
        self.declare_parameter('model', 'gemma4:e2b')
        self.declare_parameter('ollama_url', 'http://localhost:11434')
        self.model = self.get_parameter('model').value
        self.url = self.get_parameter('ollama_url').value
        self.enabled = True
        self.trigger = Trigger()
        self.lock = threading.Lock()
        self.in_flight = False
        self.state = 'idle'
        self.last_latency = 0.0
        self.last_text = ''
        self.latencies = []
        self.create_subscription(String, '/brain_summary', self.on_summary, 10)
        self.create_subscription(Bool, '/agent_enable', self.on_enable, 10)
        self.pub = self.create_publisher(String, '/agent_proposal', 10)
        self.pub_status = self.create_publisher(String, '/agent_status', 10)
        self.create_timer(1.0, self.heartbeat)
        self.get_logger().info('mission_agent up: model %s at %s' % (self.model, self.url))

    def on_enable(self, msg):
        self.enabled = bool(msg.data)
        self.state = 'idle' if self.enabled else 'off'

    def heartbeat(self):
        s = {'state': self.state, 'latency_s': round(self.last_latency, 1), 'text': self.last_text, 'model': self.model}
        if self.latencies:
            ls = sorted(self.latencies)
            s['median_latency_s'] = round(ls[len(ls) // 2], 1)
        self.pub_status.publish(String(data=json.dumps(s)))

    def on_summary(self, msg):
        if not self.enabled:
            return
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            why = self.trigger.should_call(d.get('t', 0.0), d.get('events', []), self.in_flight)
            if not why:
                return
            self.in_flight = True
        self.state = 'thinking (%s)' % why
        threading.Thread(target=self.ask, args=(d['summary'], why), daemon=True).start()

    def ask(self, summary, why):
        try:
            call, err, lat, _raw = call_ollama(self.url, self.model, summary)
            self.last_latency = lat
            if not err:
                self.latencies.append(lat)
            short = 'trigger: %s; %s' % (why, summary.split('\n')[2][:90] if summary.count('\n') >= 2 else summary[:90])
            prop = {'tool_call': call, 'error': err, 'latency_s': round(lat, 1), 'summary_short': short}
            self.pub.publish(String(data=json.dumps(prop)))
            self.state = 'error' if err else 'idle'
            self.last_text = err or '%s %s' % (call['name'], json.dumps(call['args'])[:120])
        finally:
            with self.lock:
                self.in_flight = False


def main():
    rclpy.init()
    node = MissionAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
