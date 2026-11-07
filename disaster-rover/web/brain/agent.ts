import type { LocalBrain } from './brain';
/** Gemma 4 mission commander client for the local TS brain. (milestone 8: stub) */
export class AgentClient {
  enabled = false;
  reset() {}
  step(_b: LocalBrain, _t: number): string | null { return null; }
  statusText() { return { text: this.enabled ? 'Agent offline, utility fallback' : 'Agent off: utility brain only', ok: false }; }
}
