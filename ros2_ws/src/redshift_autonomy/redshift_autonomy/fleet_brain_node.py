"""Fleet "agent brain" node: decision traces + operator commands → advisory Gemma (or local-rules) decisions.

One per mission (mission.launch.py). A thin rclpy wrapper around brain_bridge.BrainHub: subscriptions feed it raw
messages, a 0.2 s timer drains AgentBrain's finished decisions onto /brain/decision (plus one "brain_advice" trace
each), and /brain/status reports the brain at 1 Hz. AgentBrain never blocks: the cloud call runs on its worker
threads; everything published here is published from the executor thread. The planners apply the decisions
(brain_bridge.BrainCap): Gemma is advisory, the local FSM / A* / safety layer has already acted and keeps authority.
Every callback goes through a guarded BrainHub entry point: a message that cannot be handled is dropped with one
throttled warning, it never takes the node down.

Runs on wall-clock time, not /clock: the cloud round-trip, AgentBrain's de-duplication / back-off and the UI's 't'
fields are wall time, and polling must neither stall nor race with the simulation speed.

The Gemini API key is read once into the client and never logged, published or kept on this node.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .brain_bridge import MIN_INTERVAL_S, BrainHub, dumps
from .gemma_agent_brain import DEFAULT_MODEL, AgentBrain, GemmaClient, load_api_key
from .physics_model import RoverPhysics

POLL_S = 0.2
STATUS_S = 1.0


class FleetBrainNode(Node):
    def __init__(self):
        super().__init__('fleet_brain')
        for k, v in (('enabled', True), ('model', DEFAULT_MODEL), ('edge_profile', 'orin_nano_15w'),
                     ('robots', ['rover_0']), ('recovery_rover', ''), ('drone', ''), ('env_file', ''),
                     ('g_mult', 2.2)):
            self.declare_parameter(k, v)
        gp = lambda k: self.get_parameter(k).value  # noqa: E731
        self.enabled = bool(gp('enabled'))
        model = str(gp('model') or DEFAULT_MODEL)
        edge = str(gp('edge_profile') or 'orin_nano_15w')
        robots = [str(r) for r in gp('robots')]
        key = load_api_key(str(gp('env_file') or '') or None)
        self.key_present = key is not None
        client = GemmaClient(model=model, api_key=key, device_id=edge) if self.enabled and key else None
        del key                                   # the client holds the only copy
        self.brain = AgentBrain(client, min_interval_s=MIN_INTERVAL_S, device_id=edge)
        self.pub_decision = self.create_publisher(String, '/brain/decision', 50)
        self.pub_status = self.create_publisher(String, '/brain/status', 10)
        self.pub_trace = self.create_publisher(String, '/agent_decision_trace', 50)
        self.hub = BrainHub(self.brain, robots=robots, recovery_rover=str(gp('recovery_rover') or ''),
                            drone=str(gp('drone') or ''), edge_profile=edge, model=model,
                            rover_v_max=RoverPhysics().limits(float(gp('g_mult'))).v_max,
                            publish_decision=lambda d: self.pub_decision.publish(String(data=dumps(d))),
                            publish_trace=lambda t: self.pub_trace.publish(String(data=dumps(t))),
                            log=self.get_logger().info, warn=self.get_logger().warning)
        h = self.hub
        # each callback only forwards m.data to a guarded hub entry point (brain_bridge._guarded)
        self.create_subscription(String, '/agent_decision_trace', lambda m: h.on_trace(m.data), 100)
        self.create_subscription(String, '/operator/command', lambda m: h.on_command(m.data), 10)
        self.create_subscription(String, '/world/env_state', lambda m: h.on_env(m.data), 10)
        self.create_subscription(String, '/world/events', lambda m: h.on_hazard(m.data), 20)
        self.create_subscription(String, '/fleet/hazard_marks', lambda m: h.on_hazard(m.data, 'hazard_mark'), 10)
        for r in robots:
            self.create_subscription(String, f'/{r}/agent_state', lambda m, r=r: h.on_state(r, m.data), 10)
            self.create_subscription(String, f'/{r}/edge_telemetry', lambda m, r=r: h.on_edge(r, m.data), 10)
            self.create_subscription(String, f'/{r}/perception/detections',
                                     lambda m, r=r: h.on_perception(r, m.data), 10)
        self.create_timer(POLL_S, h.poll)
        self.create_timer(STATUS_S, h.guard.wrap('/brain/status', self.publish_status))
        self.get_logger().info(
            f'fleet brain: enabled={self.enabled}, API key {"present" if self.key_present else "missing"}, '
            f'advisor={"Gemma " + model + " (Google AI cloud)" if client else "local rules"}, device={edge}, '
            f'robots={robots}, recovery={h.recovery}, drone={h.drone}')

    def publish_status(self):
        st = self.hub.status(self.enabled, self.key_present)
        if st is not None:
            self.pub_status.publish(String(data=dumps(st)))


def main():
    rclpy.init()
    node = FleetBrainNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.brain.close(wait=False)      # running cloud calls end by their own timeout; queued ones are cancelled
        node.destroy_node()
        rclpy.try_shutdown()
