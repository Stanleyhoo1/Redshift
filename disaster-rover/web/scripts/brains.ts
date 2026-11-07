import type { SensorFrame, BrainOutput, Twist } from '../brain/msgs';
import { zeroTwist } from '../brain/msgs';
import { RosLite } from '../bridge/roslite';
import { SENSOR_TOPICS, OUT_TOPICS } from '../bridge/topics';
import { WanderBrain } from '../brain/types';
import { LocalBrain } from '../brain/brain';
import type { Sim } from '../sim/sim';

export interface RunnerBrain {
  name: string;
  tick(f: SensorFrame): BrainOutput | Promise<BrainOutput>;
  close?(): Promise<void> | void;
  stats?(): Record<string, unknown>;
}

export async function makeBrain(kind: string, args: Record<string, string>, _sim: Sim): Promise<RunnerBrain> {
  if (kind === 'wander') return new WanderBrain();
  if (kind === 'local') {
    const b = new LocalBrain(); b.noProbes = args['no-probes'] === 'true';
    return Object.assign(b, { stats: () => ({ probeWarnings: b.probeWarnings }) });
  }
  if (kind === 'ros') { const b = new RosBrain(); await b.connect(args.url ?? 'ws://localhost:9090'); return b; }
  throw new Error(`unknown brain ${kind}`);
}

/**
 * Drives the sim from the ROS 2 brain over rosbridge, in lockstep: publish one sensor frame,
 * wait for the /cmd_vel it produces (timeout -> keep the last command, counted as a miss).
 */
export class RosBrain implements RunnerBrain {
  name = 'ros2-py (rosbridge)';
  private ros = new RosLite();
  private waiting: ((t: Twist) => void) | null = null;
  private lastCmd: Twist = zeroTwist();
  private logs: string[] = []; private agentLogs: string[] = []; private drop = false;
  misses = 0; ticks = 0; rtt: number[] = [];
  async connect(url: string) {
    await this.ros.connect(url);
    for (const s of SENSOR_TOPICS) this.ros.advertise(s.topic, s.type);
    this.ros.subscribe(OUT_TOPICS.cmd_vel.topic, OUT_TOPICS.cmd_vel.type, (m: Twist) => { this.lastCmd = m; const w = this.waiting; this.waiting = null; w?.(m); });
    this.ros.subscribe(OUT_TOPICS.drop_probe.topic, OUT_TOPICS.drop_probe.type, () => { this.drop = true; });
    this.ros.subscribe(OUT_TOPICS.decision_log.topic, OUT_TOPICS.decision_log.type, (m: { data: string }) => this.logs.push(m.data));
    this.ros.subscribe(OUT_TOPICS.agent_log.topic, OUT_TOPICS.agent_log.type, (m: { data: string }) => this.agentLogs.push(m.data));
    // let DDS discovery match the new publishers before the first frame
    await new Promise(r => setTimeout(r, 1500));
  }
  async tick(f: SensorFrame): Promise<BrainOutput> {
    this.ticks++;
    const t0 = performance.now();
    const got = new Promise<Twist | null>(res => { this.waiting = res; setTimeout(() => { if (this.waiting === res) { this.waiting = null; res(null); } }, 2000); });
    for (const s of SENSOR_TOPICS) this.ros.publish(s.topic, f[s.key]);
    const cmd = await got;
    if (!cmd) this.misses++; else this.rtt.push(performance.now() - t0);
    // give the probe / log messages published right after /cmd_vel a moment to arrive
    await new Promise(r => setTimeout(r, 2));
    const out: BrainOutput = { cmd_vel: cmd ?? this.lastCmd, drop_probe: this.drop ? { x: 0, y: 0, z: 0 } : null, decision_log: this.logs.length ? this.logs.join('\n') : null, agent_log: this.agentLogs.length ? this.agentLogs.join('\n') : null };
    this.logs = []; this.agentLogs = []; this.drop = false;
    return out;
  }
  close() { this.ros.close(); }
  stats() { const s = [...this.rtt].sort((a, b) => a - b); return { ticks: this.ticks, missedFrames: this.misses, rttMedianMs: +(s[s.length >> 1] ?? 0).toFixed(1), rttP95Ms: +(s[Math.floor(s.length * 0.95)] ?? 0).toFixed(1) }; }
}
