import type { LocalBrain } from './brain';
import type { ToolCall } from './utility';

/**
 * Gemma 4 mission commander for the LOCAL TS brain. Mirror of disaster_brain/agent_core.py + mission_agent.py:
 * same prompt, same five tools, same trigger policy, same rule-set validation (brain.proposeGoal).
 * The request runs asynchronously (fetch), so it never blocks the 10 Hz brain loop.
 */
export const SYSTEM_PROMPT =
  "You are the mission commander of an autonomous disaster-response rover. The rover's own safety layer " +
  'handles immediate danger; you set the strategic goal. Mission: survey the area and place early-warning ' +
  'probes where they warn people about hazards, while keeping the rover alive. Coordinates are metres, map ' +
  'is -97..97 on both axes. Use only positions mentioned in the situation report. Call exactly one tool.';

const R = { type: 'string', description: 'one short sentence: why' };
const XY = { x: { type: 'number' }, y: { type: 'number' } };
const fn = (name: string, description: string, properties: object, required: string[]) =>
  ({ type: 'function', function: { name, description, parameters: { type: 'object', properties, required } } });
export const TOOLS = [
  fn('go_to_region', 'Drive to a point (e.g. towards unexplored ground).', { ...XY, reason: R }, ['x', 'y', 'reason']),
  fn('place_probe_at', 'Drive to a point and drop an early-warning probe there.', { ...XY, reason: R }, ['x', 'y', 'reason']),
  fn('retreat_to_high_ground', 'Climb to the highest safe ground (floods).', { reason: R }, ['reason']),
  fn('hold_position', 'Stay still for some seconds (e.g. while the ground shakes).', { seconds: { type: 'number' }, reason: R }, ['seconds', 'reason']),
  fn('survey_area', 'Explore within a radius of a point.', { ...XY, radius: { type: 'number' }, reason: R }, ['x', 'y', 'radius', 'reason']),
];
export const TRIGGER_EVENTS = ['new_hazard', 'hazard_on_path', 'battery', 'probe_placed', 'probe_warning'];
export const PERIOD = 30, TIMEOUT_MS = 60000;

export function parseResponse(d: any): ToolCall {
  const msg = d?.message ?? {};
  const calls = msg.tool_calls ?? [];
  if (calls.length) {
    const f = calls[0].function ?? {};
    const args = typeof f.arguments === 'string' ? JSON.parse(f.arguments) : f.arguments;
    if (typeof f.name !== 'string' || !args || typeof args !== 'object') throw new Error('malformed tool call');
    return { name: f.name, args } as ToolCall;
  }
  const text: string = (msg.content ?? '').trim();
  const i = text.indexOf('{'), j = text.lastIndexOf('}');
  if (i >= 0 && j > i) {
    const o = JSON.parse(text.slice(i, j + 1));
    const name = o.name ?? o.tool, args = o.arguments ?? o.args ?? o.parameters ?? {};
    if (typeof name === 'string' && typeof args === 'object') return { name, args } as ToolCall;
  }
  throw new Error(`no tool call in output: ${JSON.stringify(text.slice(0, 80))}`);
}

interface Result { call: ToolCall | null; error: string | null; latency: number; why: string; summary: string }

export class AgentClient {
  enabled = false;
  url = 'http://localhost:11434';
  model = 'gemma4:e2b';
  inFlight: Promise<Result> | null = null;
  private result: Result | null = null;
  private lastCallT = -1e9;
  private pendingEvents: string[] = [];
  private reqSimT = 0; private reqWall = 0;
  state = 'idle'; lastLatency = 0; lastError: string | null = null; lastOkWall = 0;
  latencies: number[] = []; accepted = 0; rejected = 0; errors = 0; calls = 0;
  rejections: Record<string, number> = {};

  reset() { this.inFlight = null; this.result = null; this.lastCallT = -1e9; this.pendingEvents = []; this.state = 'idle'; }

  /** Called every brain tick. Returns an /agent_log line when a proposal was judged, else null. */
  step(brain: LocalBrain, t: number): string | null {
    const ev = brain.events.splice(0);
    if (!this.enabled) return null;
    for (const e of ev) if (TRIGGER_EVENTS.includes(e.kind)) this.pendingEvents.push(e.kind);
    let line: string | null = null;
    if (this.result) {
      const r = this.result; this.result = null;
      if (r.error) {
        this.errors++;
        line = `[AGENT] ${r.why} → IGNORED (${r.error}) | ${r.latency.toFixed(1)} s`;
      } else {
        const res = brain.proposeGoal(r.call!);
        const args = Object.entries(r.call!.args).filter(([k]) => k !== 'reason').map(([k, v]) => `${k}=${v}`).join(', ');
        if (res.ok) this.accepted++; else { this.rejected++; this.rejections[res.rule] = (this.rejections[res.rule] ?? 0) + 1; }
        line = `[AGENT] ${r.why}; ${r.summary} → ${r.call!.name}(${args}) → "${String(r.call!.args.reason ?? '')}" → ${res.ok ? 'ACCEPTED' : `REJECTED (${res.rule}: ${res.detail})`} | ${r.latency.toFixed(1)} s`;
      }
    }
    if (!this.inFlight && (this.pendingEvents.length || t - this.lastCallT >= PERIOD)) {
      const why = 'trigger: ' + ([...new Set(this.pendingEvents)].join(', ') || 'periodic');
      this.pendingEvents = []; this.lastCallT = t;
      const summary = brain.summary();
      this.reqSimT = t; this.reqWall = performance.now(); this.calls++;
      this.state = `thinking (${why.slice(9)})`;
      this.inFlight = this.ask(summary, why).then(r => { this.result = r; this.inFlight = null; return r; });
    }
    return line;
  }

  private async ask(summary: string, why: string): Promise<Result> {
    const t0 = performance.now();
    const short = summary.split('\n')[2]?.slice(0, 90) ?? '';
    const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), TIMEOUT_MS);
    try {
      const r = await fetch(this.url + '/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ctl.signal,
        body: JSON.stringify({ model: this.model, stream: false, think: false, keep_alive: '30m', options: { num_predict: 200, temperature: 0.2 }, tools: TOOLS, messages: [{ role: 'system', content: SYSTEM_PROMPT }, { role: 'user', content: 'Situation report:\n' + summary }] }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      const latency = (performance.now() - t0) / 1000;
      this.lastLatency = latency;
      try {
        const call = parseResponse(d);
        this.latencies.push(latency); this.state = 'idle'; this.lastError = null; this.lastOkWall = performance.now();
        return { call, error: null, latency, why, summary: short };
      } catch (e) { this.state = 'error'; this.lastError = 'unparsable output'; return { call: null, error: `unparsable output (${(e as Error).message})`, latency, why, summary: short }; }
    } catch (e) {
      const latency = (performance.now() - t0) / 1000;
      this.state = 'error'; this.lastError = `ollama unreachable: ${(e as Error).message}`;
      return { call: null, error: this.lastError, latency, why, summary: short };
    } finally { clearTimeout(to); }
  }

  /**
   * Headless runs only: never let sim time run ahead of wall time while a request is in flight,
   * so the LLM's real CPU latency costs real sim time (the rover keeps driving on the utility brain meanwhile).
   */
  async syncWait(simT: number) {
    if (!this.inFlight) return;
    const ahead = (simT - this.reqSimT) - (performance.now() - this.reqWall) / 1000;
    if (ahead > 0) await Promise.race([this.inFlight, new Promise(r => setTimeout(r, ahead * 1000))]);
  }

  statusText(): { text: string; ok: boolean } {
    if (!this.enabled) return { text: 'Agent off: utility brain only', ok: false };
    if (this.state === 'error') return { text: `Agent offline, utility fallback (${this.lastError})`, ok: false };
    return { text: `Gemma 4 (local): ${this.state}${this.lastLatency ? ` · last ${this.lastLatency.toFixed(1)} s` : ''}`, ok: true };
  }
  stats() {
    const s = [...this.latencies].sort((a, b) => a - b);
    return { agentCalls: this.calls, accepted: this.accepted, rejected: this.rejected, agentErrors: this.errors, rejections: this.rejections, latencyMedianS: +(s[s.length >> 1] ?? 0).toFixed(1), latencyP95S: +(s[Math.floor(s.length * 0.95)] ?? 0).toFixed(1) };
  }
}
