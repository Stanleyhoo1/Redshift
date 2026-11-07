/**
 * Minimal rosbridge v2 client over the standard WebSocket API (works in Node 22 and browsers).
 * Used by the headless ROS runner; the browser app uses roslib (bridge/rosbridge.ts).
 */
export class RosLite {
  private ws: WebSocket | null = null;
  private subs = new Map<string, ((m: any) => void)[]>();
  private advertised = new Set<string>();
  connected = false;

  connect(url: string, timeoutMs = 3000): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      const to = setTimeout(() => { try { ws.close(); } catch { /* */ } reject(new Error(`rosbridge not reachable at ${url}`)); }, timeoutMs);
      ws.onopen = () => { clearTimeout(to); this.ws = ws; this.connected = true; resolve(); };
      ws.onerror = () => { clearTimeout(to); reject(new Error(`rosbridge not reachable at ${url}`)); };
      ws.onclose = () => { this.connected = false; };
      ws.onmessage = ev => {
        const m = JSON.parse(typeof ev.data === 'string' ? ev.data : '');
        if (m.op === 'publish') for (const cb of this.subs.get(m.topic) ?? []) cb(m.msg);
      };
    });
  }
  private send(o: unknown) { if (this.ws && this.connected) this.ws.send(JSON.stringify(o)); }
  advertise(topic: string, type: string) { if (!this.advertised.has(topic)) { this.advertised.add(topic); this.send({ op: 'advertise', topic, type }); } }
  publish(topic: string, msg: unknown) { this.send({ op: 'publish', topic, msg }); }
  subscribe(topic: string, type: string, cb: (m: any) => void) {
    if (!this.subs.has(topic)) { this.subs.set(topic, []); this.send({ op: 'subscribe', topic, type, queue_length: 50 }); }
    this.subs.get(topic)!.push(cb);
  }
  close() { try { this.ws?.close(); } catch { /* */ } this.connected = false; }
}
