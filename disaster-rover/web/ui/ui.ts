import type { Sim } from '../sim/sim';
import { DISASTER_KINDS, DisasterKind, DisasterConfig, defaultDisasterConfig } from '../sim/disasters';
import type { RenderToggles } from '../render/scene';
import type { BeliefView } from '../brain/view';
import { N, CELL, HALF } from '../sim/world';
import { SENSOR } from '../sim/sensors';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const LABEL: Record<DisasterKind, string> = { flood: 'Flood', fire: 'Wildfire', quake: 'Earthquake', tornado: 'Tornado', meteor: 'Meteor shower' };

export type BrainMode = 'ros' | 'local' | 'manual';
export interface BrainInfo {
  behaviour?: string; goal?: string; reflex?: string | null; scores?: Record<string, number>; confidence?: number;
  view?: BeliefView | null; path?: { x: number; y: number }[] | null; goalPt?: { x: number; y: number } | null;
  roverEst?: { x: number; y: number } | null;
}

export class UI {
  onStart = () => {}; onPause = () => {}; onReset = () => {};
  onTrigger: (k: DisasterKind) => void = () => {};
  onBrainChange: (m: BrainMode) => void = () => {};
  onAgentToggle: (on: boolean) => void = () => {};
  private logEl = $('log'); private evEl = $('events'); private agentEl = $('agentPanel');
  private lastEvents = 0; private mini = ($('minimap') as HTMLCanvasElement).getContext('2d')!;
  private img = this.mini.createImageData(N, N);
  private miniAcc = 0;

  constructor() {
    const host = $('disasters'), chaos = $('chaos');
    for (const k of DISASTER_KINDS) {
      const row = document.createElement('div'); row.className = 'dis';
      row.innerHTML = `<label><input type="checkbox" id="d_${k}" /> ${LABEL[k]}</label><input type="range" id="i_${k}" min="0.1" max="1" step="0.05" value="0.6" title="intensity" />`;
      host.appendChild(row);
      const b = document.createElement('button'); b.textContent = LABEL[k]; b.onclick = () => this.onTrigger(k); chaos.appendChild(b);
    }
    $('start').onclick = () => this.onStart();
    $('pause').onclick = () => this.onPause();
    $('reset').onclick = () => this.onReset();
    document.querySelectorAll<HTMLInputElement>('input[name=brain]').forEach(r => r.onchange = () => this.onBrainChange(this.brainMode()));
    ($('agent') as HTMLInputElement).onchange = () => this.onAgentToggle(this.agentOn());
  }

  seed() { return +($('seed') as HTMLInputElement).value || 1; }
  duration() { return +($('duration') as HTMLInputElement).value || 180; }
  speed() { return +($('speed') as HTMLSelectElement).value || 1; }
  rosUrl() { return ($('rosUrl') as HTMLInputElement).value; }
  brainMode(): BrainMode { return (document.querySelector<HTMLInputElement>('input[name=brain]:checked')?.value ?? 'local') as BrainMode; }
  setBrainMode(m: BrainMode) { const r = document.querySelector<HTMLInputElement>(`input[name=brain][value=${m}]`); if (r) r.checked = true; }
  agentOn() { return ($('agent') as HTMLInputElement).checked; }
  disasterConfig(): DisasterConfig {
    const c = defaultDisasterConfig();
    for (const k of DISASTER_KINDS) c[k] = { enabled: ($(`d_${k}`) as HTMLInputElement).checked, intensity: +($(`i_${k}`) as HTMLInputElement).value };
    c.random = ($('random') as HTMLInputElement).checked;
    return c;
  }
  toggles(): RenderToggles {
    const ck = (id: string) => ($(id) as HTMLInputElement).checked;
    return { lidar: ck('tLidar'), hazards: ck('tHaz'), fog: ck('tFog'), camera: ck('tCam'), path: ck('tPath') };
  }
  setRosStatus(text: string, cls: '' | 'ok' | 'warn' | 'bad') { const e = $('rosStatus'); e.textContent = text; e.className = 'status ' + cls; }
  setAgentStatus(text: string, cls: '' | 'ok' | 'warn' | 'bad') { const e = $('agentStatus'); e.textContent = text; e.className = 'status ' + cls; }

  clearLogs() { this.logEl.innerHTML = ''; this.evEl.innerHTML = ''; this.agentEl.innerHTML = ''; this.lastEvents = 0; $('banner').className = 'hidden'; }
  private append(el: HTMLElement, t: number, text: string, cls = '', max = 150) {
    const d = document.createElement('div');
    d.innerHTML = `<span class="t">[${t.toFixed(1)}]</span> <span class="${cls}"></span>`;
    (d.lastChild as HTMLElement).textContent = text;
    el.appendChild(d);
    while (el.childElementCount > max) el.firstChild!.remove();
    el.scrollTop = el.scrollHeight;
  }
  log(t: number, text: string) { this.append(this.logEl, t, text); }
  agentLog(t: number, text: string) { this.append(this.agentEl, t, text, /ACCEPTED/.test(text) ? 'acc' : /REJECTED|IGNORED|error|offline/i.test(text) ? 'rej' : '', 40); }

  update(sim: Sim, info: BrainInfo | null) {
    const r = sim.rover;
    const bar = (v: number, c = 'var(--ok)') => `<div class="bar"><div style="width:${Math.max(0, Math.min(100, v))}%;background:${c}"></div></div>`;
    $('status').innerHTML = `
      <span class="k">time</span><span>${sim.t.toFixed(1)} / ${sim.duration} s</span>
      <span class="k">health</span><span>${r.health.toFixed(0)}%${bar(r.health, r.health < 40 ? 'var(--bad)' : 'var(--ok)')}</span>
      <span class="k">battery</span><span>${r.battery.toFixed(0)}%${bar(r.battery, r.battery < 30 ? 'var(--warn)' : '#4aa3ff')}</span>
      <span class="k">probes left</span><span>${r.probesLeft} / 6</span>
      <span class="k">behaviour</span><span><b>${info?.behaviour ?? '-'}</b>${info?.reflex ? ` <span style="color:var(--bad)">REFLEX: ${info.reflex}</span>` : ''}</span>
      <span class="k">mission goal</span><span>${info?.goal ?? 'none'}</span>
      <span class="k">surveyed</span><span>${sim.metrics().surveyedPct}%</span>`;
    for (; this.lastEvents < sim.dis.events.length; this.lastEvents++) { const e = sim.dis.events[this.lastEvents]; this.append(this.evEl, e.t, e.text, '', 60); }
    const ban = $('banner');
    if (sim.done) {
      const m = sim.metrics();
      ban.className = m.survived ? 'ok' : 'bad';
      ban.textContent = m.survived ? `SURVIVED · ${m.probesPlaced} probes · ${m.surveyedPct}% surveyed` : `FAILED: ${m.cause}`;
    }
    // overlay with the live decision
    const ov = $('overlay');
    if (info?.behaviour) {
      const sc = info.scores ? Object.entries(info.scores).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<span>${k} ${v.toFixed(2)}</span>`).join('') : '';
      ov.innerHTML = `<div class="beh">${info.behaviour.toUpperCase()}${info.confidence !== undefined ? ` <small>(${info.confidence.toFixed(2)})</small>` : ''}</div>${info.reflex ? `<div class="reflex">SAFETY REFLEX: ${info.reflex}</div>` : ''}<div class="scores">${sc}</div>${info.goal && info.goal !== 'none' ? `<div class="agent">GOAL: ${info.goal}</div>` : ''}`;
    } else ov.innerHTML = '';
    this.miniAcc++;
    if (this.miniAcc % 6 === 0) this.drawMini(sim, info);
  }

  private drawMini(sim: Sim, info: BrainInfo | null) {
    const v = info?.view, d = this.img.data;
    for (let cy = 0; cy < N; cy++) for (let cx = 0; cx < N; cx++) {
      const i = cy * N + cx, o = ((N - 1 - cy) * N + cx) * 4;
      let r = 18, g = 20, b = 26;
      if (v) {
        const bi = v.cellOfWorld(i);
        const c = v.conf[bi];
        if (c > 0) {
          const k = 0.35 + 0.65 * Math.min(1, c);
          r = 60 * k; g = 70 * k; b = 64 * k;
          if (v.obst[bi] > 0.5) { r = 170; g = 170; b = 170; }
          const [w, h, ru, wi] = [v.channels[0][bi], v.channels[1][bi], v.channels[2][bi], v.channels[3][bi]];
          if (v.predicted[bi] > 0.2) { r = r * 0.4 + 255 * 0.6 * v.predicted[bi]; g = g * 0.4 + 210 * 0.6 * v.predicted[bi]; b *= 0.5; }
          if (w > 0.05) { r = r * (1 - w) + 58 * w; g = g * (1 - w) + 123 * w; b = b * (1 - w) + 213 * w; }
          if (h > 0.05) { r = r * (1 - h) + 255 * h; g = g * (1 - h) + 90 * h; b = b * (1 - h) + 31 * h; }
          if (ru > 0.05) { r = r * (1 - ru) + 181 * ru; g = g * (1 - ru) + 139 * ru; b = b * (1 - ru) + 90 * ru; }
          if (wi > 0.05) { r = r * (1 - wi) + 176 * wi; g = g * (1 - wi) + 77 * wi; b = b * (1 - wi) + 255 * wi; }
        }
      } else if (sim.revealed[i]) { r = 70; g = 90; b = 70; }
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
    const ctx = this.mini;
    ctx.putImageData(this.img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(ctx.canvas, 0, 0, N, N, 0, 0, ctx.canvas.width, ctx.canvas.height);
    const s = ctx.canvas.width / (N * CELL);
    const P = (x: number, y: number): [number, number] => [(x + HALF) * s, (HALF - y) * s];
    if (info?.path && info.path.length > 1) {
      ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 1.5; ctx.beginPath();
      info.path.forEach((p, k) => { const [x, y] = P(p.x, p.y); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    }
    for (const p of sim.probes) {
      const [x, y] = P(p.x, p.y);
      ctx.strokeStyle = p.alive ? 'rgba(51,255,136,.6)' : 'rgba(255,60,60,.5)'; ctx.beginPath(); ctx.arc(x, y, SENSOR.probeRadius * s, 0, 7); ctx.stroke();
      ctx.fillStyle = p.alive ? '#33ff88' : '#ff3c3c'; ctx.fillRect(x - 2, y - 2, 4, 4);
    }
    if (info?.goalPt) { const [x, y] = P(info.goalPt.x, info.goalPt.y); ctx.strokeStyle = '#c89bff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.stroke(); }
    const r = sim.rover, [rx, ry] = P(r.x, r.y);
    ctx.fillStyle = '#f0b030'; ctx.beginPath(); ctx.arc(rx, ry, 3.5, 0, 7); ctx.fill();
    ctx.strokeStyle = '#f0b030'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(rx + 8 * Math.cos(r.yaw), ry - 8 * Math.sin(r.yaw)); ctx.stroke();
  }
}
