import ROSLIB from 'roslib';
import type { SensorFrame, Twist } from '../brain/msgs';
import { zeroTwist } from '../brain/msgs';
import type { BeliefView } from '../brain/view';
import { SENSOR_TOPICS, OUT_TOPICS, AGENT_ENABLE } from './topics';

/** Decoded /hazard_map (the ROS brain's belief) so the minimap / overlay can show it. */
class RemoteBelief implements BeliefView {
  n = 100; res = 2;
  risk = new Float32Array(10000); predicted = new Float32Array(10000); conf = new Float32Array(10000); obst = new Float32Array(10000);
  channels = [0, 1, 2, 3, 4].map(() => new Float32Array(10000));
  cellOfWorld(i: number) { return i; }
  load(m: Record<string, number[]>) {
    const put = (dst: Float32Array, src?: number[]) => { if (src) for (let i = 0; i < src.length; i++) dst[i] = src[i] / 100; };
    put(this.risk, m.risk); put(this.predicted, m.predicted); put(this.conf, m.conf); put(this.obst, m.obst);
    put(this.channels[0], m.water); put(this.channels[1], m.heat); put(this.channels[2], m.rubble); put(this.channels[3], m.wind);
  }
}

/**
 * Browser <-> rosbridge link (roslib). Publishes each sensor frame on the ROS topics, keeps the latest
 * /cmd_vel, queues /drop_probe and log lines for the app to consume.
 */
export class RosLink {
  ros: any = null;
  connected = false;
  status = 'not connected';
  lastCmd: Twist = zeroTwist();
  lastCmdAt = 0;
  dropPending = false;
  logs: string[] = [];
  agentLogs: string[] = [];
  agentStatus: { state: string; text?: string; latency_s?: number } | null = null;
  agentStatusAt = 0;
  info: any = null;
  belief = new RemoteBelief();
  hasBelief = false;
  private pubs = new Map<string, any>();
  private agentPub: any = null;

  connect(url: string): Promise<boolean> {
    return new Promise(resolve => {
      this.close();
      this.status = `connecting to ${url}…`;
      const ros = new ROSLIB.Ros({ url });
      this.ros = ros;
      let settled = false;
      const done = (ok: boolean) => { if (!settled) { settled = true; resolve(ok); } };
      ros.on('connection', () => {
        this.connected = true; this.status = `connected ${url}`;
        for (const s of SENSOR_TOPICS) this.pubs.set(s.topic, new ROSLIB.Topic({ ros, name: s.topic, messageType: s.type, queue_size: 1 }));
        for (const p of this.pubs.values()) p.advertise();
        this.agentPub = new ROSLIB.Topic({ ros, name: AGENT_ENABLE.topic, messageType: AGENT_ENABLE.type }); this.agentPub.advertise();
        const sub = (t: { topic: string; type: string }, cb: (m: any) => void) => new ROSLIB.Topic({ ros, name: t.topic, messageType: t.type, queue_length: 20 }).subscribe(cb);
        sub(OUT_TOPICS.cmd_vel, (m: Twist) => { this.lastCmd = m; this.lastCmdAt = performance.now(); });
        sub(OUT_TOPICS.drop_probe, () => { this.dropPending = true; });
        sub(OUT_TOPICS.decision_log, (m: { data: string }) => this.logs.push(m.data));
        sub(OUT_TOPICS.agent_log, (m: { data: string }) => this.agentLogs.push(m.data));
        sub(OUT_TOPICS.agent_status, (m: { data: string }) => { try { this.agentStatus = JSON.parse(m.data); this.agentStatusAt = performance.now(); } catch { /* */ } });
        sub(OUT_TOPICS.brain_info, (m: { data: string }) => { try { this.info = JSON.parse(m.data); } catch { /* */ } });
        sub(OUT_TOPICS.hazard_map, (m: { data: string }) => { try { this.belief.load(JSON.parse(m.data)); this.hasBelief = true; } catch { /* */ } });
        done(true);
      });
      ros.on('error', () => { this.connected = false; this.status = `rosbridge not reachable at ${url}`; done(false); });
      ros.on('close', () => { if (this.connected) this.status = 'rosbridge connection closed'; this.connected = false; done(false); });
      setTimeout(() => { if (!this.connected) { this.status = `rosbridge not reachable at ${url}`; done(false); } }, 3000);
    });
  }
  publishFrame(f: SensorFrame) {
    if (!this.connected) return;
    for (const s of SENSOR_TOPICS) this.pubs.get(s.topic)?.publish(new ROSLIB.Message(f[s.key]));
  }
  setAgent(on: boolean) { if (this.connected && this.agentPub) this.agentPub.publish(new ROSLIB.Message({ data: on })); }
  /** ms since the last /cmd_vel */
  cmdAge() { return this.lastCmdAt ? performance.now() - this.lastCmdAt : Infinity; }
  close() { try { this.ros?.close(); } catch { /* */ } this.ros = null; this.connected = false; this.pubs.clear(); this.hasBelief = false; }
}
