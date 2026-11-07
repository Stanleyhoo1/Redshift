import type { Sim } from '../sim/sim';
import { DISASTER_KINDS, DisasterKind, DisasterConfig, defaultDisasterConfig } from '../sim/disasters';
import type { RenderToggles } from '../render/scene';
import type { BeliefView } from '../brain/view';
import { N, CELL, HALF } from '../sim/world';
import { SENSOR } from '../sim/sensors';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** Small inline SVG icons (no external assets). */
export const ICON: Record<string, string> = {
  flood: '<svg viewBox="0 0 24 24"><path d="M2 15c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/><path d="M2 20c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/><path d="M12 3c-2 3-4 5-4 7a4 4 0 0 0 8 0c0-2-2-4-4-7z"/></svg>',
  fire: '<svg viewBox="0 0 24 24"><path d="M12 2c1 4 6 6 6 12a6 6 0 0 1-12 0c0-3 2-5 3-6 0 2 1 3 2 3 0-3-1-6 1-9z"/></svg>',
  quake: '<svg viewBox="0 0 24 24"><path d="M2 12h4l2-5 3 10 3-12 3 9 2-2h3"/></svg>',
  tornado: '<svg viewBox="0 0 24 24"><path d="M3 4h18M5 8h14M7 12h9M9 16h5M11 20h2"/></svg>',
  meteor: '<svg viewBox="0 0 24 24"><circle cx="15" cy="15" r="5"/><path d="M3 3l8 8M7 3l6 6M3 7l6 6"/></svg>',
  probe: '<svg viewBox="0 0 24 24"><path d="M12 3v12M8 21l4-6 4 6"/><circle cx="12" cy="4" r="2"/></svg>',
  ok: '<svg viewBox="0 0 24 24"><path d="M4 12l5 5L20 6"/></svg>',
  warn: '<svg viewBox="0 0 24 24"><path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.5"/></svg>',
};
const LABEL: Record<DisasterKind, string> = { flood: 'Flood', fire: 'Wildfire', quake: 'Earthquake', tornado: 'Tornado', meteor: 'Meteor shower' };
const COLOR: Record<string, string> = { flood: 'var(--flood)', fire: 'var(--fire)', quake: 'var(--quake)', tornado: 'var(--tornado)', meteor: 'var(--meteor)', sim: 'var(--ok)' };
export const BEH_COLOR: Record<string, string> = { explore: '#34d17a', flee: '#ff5147', high_ground: '#4aa3ff', hold_still: '#ffd60a', place_probe: '#c58bff', return_to_safety: '#ff9f0a' };
const BEH_LABEL: Record<string, string> = { explore: 'EXPLORE', flee: 'FLEE', high_ground: 'HIGH GROUND', hold_still: 'HOLD STILL', place_probe: 'PLACE PROBE', return_to_safety: 'RETURN TO SAFETY' };

export type BrainMode = 'ros' | 'local' | 'manual';
export interface BrainInfo {
  behaviour?: string; goal?: string; reflex?: string | null; scores?: Record<string, number>; confidence?: number;
  view?: BeliefView | null; path?: { x: number; y: number }[] | null; goalPt?: { x: number; y: number } | null;
  hazards?: string[];
}

export class UI {
  onStart = () => {}; onPause = () => {}; onReset = () => {};
  onTrigger: (k: DisasterKind) => void = () => {};
  onBrainChange: (m: BrainMode) => void = () => {};
  onAgentToggle: (on: boolean) => void = () => {};
  private logEl = $('log'); private evEl = $('events'); private agentEl = $('agentPanel');
  private lastEvents = 0; private mini = ($('minimap') as HTMLCanvasElement).getContext('2d')!;
  private img = new ImageData(N, N); private off = document.createElement('canvas');
  private miniAcc = 0; private lastAgent = ''; private agentState = ''; private lastHud = '';

  constructor() {
    this.off.width = N; this.off.height = N;
    const host = $('disasters'), chaos = $('chaos');
    for (const k of DISASTER_KINDS) {
      const row = document.createElement('label'); row.className = 'dis'; row.style.setProperty('--c', COLOR[k]);
      row.innerHTML = `<span class="ic">${ICON[k]}</span><span class="nm">${LABEL[k]}</span><input type="range" id="i_${k}" min="0.1" max="1" step="0.05" value="0.6" title="intensity" /><span class="iv" id="iv_${k}">0.60</span><input type="checkbox" id="d_${k}" />`;
      host.appendChild(row);
      const cb = row.querySelector('input[type=checkbox]') as HTMLInputElement, rg = row.querySelector('input[type=range]') as HTMLInputElement;
      const sync = () => row.classList.toggle('on', cb.checked);
      cb.addEventListener('change', sync);
      // programmatic .checked = true (tests) has no event: poll the visual state cheaply
      setInterval(sync, 300);
      rg.addEventListener('click', e => e.stopPropagation());
      rg.addEventListener('input', () => { $(`iv_${k}`).textContent = (+rg.value).toFixed(2); });
      const b = document.createElement('button'); b.style.setProperty('--c', COLOR[k]); b.innerHTML = `${ICON[k]}${LABEL[k]}`; b.onclick = () => this.onTrigger(k); chaos.appendChild(b);
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
  setRunning(on: boolean) { $('pause').innerHTML = on ? '❚❚ Pause' : '▶ Resume'; }
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
  setAgentStatus(text: string, cls: '' | 'ok' | 'warn' | 'bad') { this.agentState = text; const e = $('agentStatus'); e.textContent = text; e.className = 'status ' + cls; }

  clearLogs() { this.lastAgent = ''; this.logEl.innerHTML = ''; this.evEl.innerHTML = ''; this.agentEl.innerHTML = ''; this.lastEvents = 0; $('banner').className = 'hidden'; $('toasts').innerHTML = ''; }
  private append(el: HTMLElement, t: number, text: string, cls = '', max = 150) {
    const d = document.createElement('div'); d.className = cls;
    d.innerHTML = `<span class="t">${t.toFixed(1).padStart(5)}</span> <span></span>`;
    (d.lastChild as HTMLElement).textContent = text;
    el.appendChild(d);
    while (el.childElementCount > max) el.firstChild!.remove();
    el.scrollTop = el.scrollHeight;
  }
  log(t: number, text: string) {
    const cls = text.startsWith('REFLEX') ? 'reflex' : text.startsWith('PROBE') ? 'probe' : text.includes('DROP PROBE') ? 'drop' : '';
    this.append(this.logEl, t, text, cls);
    if (text.startsWith('PROBE')) this.toast(text.replace(/ at .*/, ''), 'var(--ok)', ICON.probe);
  }
  agentLog(t: number, text: string) {
    const m = text.match(/→ (\w+\([^)]*\)) → .*→ (ACCEPTED|REJECTED \([a-z_]+)[^|]*\|\s*([\d.]+ s)/);
    this.lastAgent = m ? `${m[1]} → ${m[2]}${m[2].startsWith('REJECTED') ? ')' : ''} · ${m[3]}` : text.slice(0, 90);
    this.append(this.agentEl, t, text, /ACCEPTED/.test(text) ? 'acc' : /REJECTED|IGNORED|error|offline/i.test(text) ? 'rej' : '', 40);
  }
  toast(text: string, color: string, icon = ICON.warn) {
    const host = $('toasts'), d = document.createElement('div');
    d.className = 'toast'; d.style.setProperty('--c', color); d.innerHTML = `${icon}<span></span>`;
    (d.lastChild as HTMLElement).textContent = text;
    host.appendChild(d);
    while (host.childElementCount > 4) host.firstChild!.remove();
    setTimeout(() => d.classList.add('out'), 3800); setTimeout(() => d.remove(), 4300);
  }

  update(sim: Sim, info: BrainInfo | null, mode: BrainMode = 'local') {
    const r = sim.rover;
    const bar = (v: number, c: string) => `<div class="bar"><div style="width:${Math.max(0, Math.min(100, v))}%;background:${c}"></div></div>`;
    const m = sim.metrics();
    const hc = r.health < 40 ? 'var(--bad)' : r.health < 70 ? 'var(--warn)' : 'var(--ok)';
    $('status').innerHTML = `
      <div class="stat"><div class="k">Health</div><div class="v">${r.health.toFixed(0)}%</div>${bar(r.health, hc)}</div>
      <div class="stat"><div class="k">Battery</div><div class="v">${r.battery.toFixed(0)}%</div>${bar(r.battery, r.battery < 30 ? 'var(--warn)' : 'var(--info)')}</div>
      <div class="stat"><div class="k">Probes left</div><div class="pips">${Array.from({ length: 6 }, (_, k) => `<i class="${k < r.probesLeft ? 'on' : ''}"></i>`).join('')}</div></div>
      <div class="stat"><div class="k">Surveyed</div><div class="v">${m.surveyedPct}%</div>${bar(m.surveyedPct * 2, 'var(--accent)')}</div>
      <div class="stat full"><div class="k">Mission goal</div><div class="v" style="font-size:12px;font-weight:500">${esc(info?.goal && info.goal !== 'none' ? info.goal : 'none: utility brain decides')}</div></div>`;
    // behaviour scores
    if (info?.scores) {
      const best = Object.entries(info.scores).sort((a, b) => b[1] - a[1])[0]?.[0];
      $('scores').innerHTML = Object.entries(info.scores).map(([k, v]) =>
        `<div class="score ${k === info.behaviour ? 'win' : ''}" style="--c:${BEH_COLOR[k] ?? '#888'}"><span class="n">${k === info.behaviour ? '▸ ' : ''}${k.replace(/_/g, ' ')}</span><span class="t"><div style="width:${v * 100}%"></div></span><span class="p">${v.toFixed(2)}</span></div>`).join('');
      void best;
    } else $('scores').innerHTML = `<div style="color:var(--muted);font-size:12px">${mode === 'manual' ? 'manual driving' : 'waiting for the brain…'}</div>`;
    // HUD
    $('hudTime').textContent = `${sim.t.toFixed(1)} s / ${sim.duration}`;
    $('hudBrain').textContent = mode === 'ros' ? 'ROS 2 BRAIN' : mode === 'local' ? 'LOCAL TS BRAIN' : 'MANUAL';
    const beh = info?.behaviour ?? (mode === 'manual' ? 'manual' : 'idle');
    const hb = $('hudBeh'); hb.textContent = BEH_LABEL[beh] ?? beh.toUpperCase();
    hb.style.borderColor = BEH_COLOR[beh] ?? '#555'; hb.style.color = BEH_COLOR[beh] ?? '#ccc';
    const hz = (info?.hazards ?? []).join(',');
    if (hz !== this.lastHud) {
      this.lastHud = hz;
      $('hudHaz').innerHTML = (info?.hazards ?? []).map(h => `<span class="chip haz" style="--c:${COLOR[h] ?? '#ccc'}">${ICON[h] ?? ''}${h === 'quake' ? 'SHAKING' : h.toUpperCase()} SENSED</span>`).join('');
    }
    for (; this.lastEvents < sim.dis.events.length; this.lastEvents++) {
      const e = sim.dis.events[this.lastEvents];
      this.append(this.evEl, e.t, e.text, 'ev', 60);
      if (e.kind !== 'sim' && /ignited|rising|touched|incoming|Earthquake|Aftershock|surge|shifted|started a fire/.test(e.text)) this.toast(e.text, COLOR[e.kind], ICON[e.kind]);
      if (e.kind === 'sim' && /lost/.test(e.text)) this.toast(e.text, 'var(--bad)', ICON.probe);
    }
    const ban = $('banner');
    if (sim.done) {
      ban.className = m.survived ? 'ok' : 'bad';
      ban.innerHTML = m.survived ? `MISSION COMPLETE<small>survived ${m.t} s · ${m.probesPlaced} probes · ${m.surveyedPct}% surveyed</small>` : `ROVER LOST<small>${esc(m.cause)} at ${m.t} s</small>`;
    }
    // live decision overlay
    const ov = $('overlay');
    const rows: string[] = [];
    if (info?.behaviour) rows.push(`<div class="row" style="--c:${BEH_COLOR[info.behaviour] ?? '#f2a33a'}"><b>${BEH_LABEL[info.behaviour] ?? info.behaviour}</b>${info.confidence !== undefined ? ` · confidence ${info.confidence.toFixed(2)}` : ''}</div>`);
    if (info?.reflex) rows.push(`<div class="row reflex">⚠ ${esc(info.reflex)}</div>`);
    if (info?.goal && info.goal !== 'none') rows.push(`<div class="row goal">🎯 ${esc(info.goal)}</div>`);
    if (/offline/i.test(this.agentState)) rows.push(`<div class="row offline">Agent offline, utility fallback</div>`);
    else if (this.lastAgent && !/^Agent off/.test(this.agentState)) rows.push(`<div class="row gemma">GEMMA · ${esc(this.lastAgent)}</div>`);
    ov.innerHTML = rows.join('');
    if (++this.miniAcc % 6 === 0) this.drawMini(sim, info);
  }

  private drawMini(sim: Sim, info: BrainInfo | null) {
    const v = info?.view, d = this.img.data;
    for (let cy = 0; cy < N; cy++) for (let cx = 0; cx < N; cx++) {
      const i = cy * N + cx, o = ((N - 1 - cy) * N + cx) * 4;
      let r = 14, g = 18, b = 24;
      if (v) {
        const bi = v.cellOfWorld(i), c = v.conf[bi];
        if (c > 0) {
          const k = 0.4 + 0.6 * Math.min(1, c);
          r = 52 * k; g = 66 * k; b = 60 * k;
          if (v.obst[bi] > 0.5) { r = 175; g = 178; b = 182; }
          const [w, h, ru, wi] = [v.channels[0][bi], v.channels[1][bi], v.channels[2][bi], v.channels[3][bi]];
          const p = v.predicted[bi];
          if (p > 0.2) { r = r * 0.4 + 255 * 0.6 * p; g = g * 0.4 + 210 * 0.6 * p; b *= 0.5; }
          if (w > 0.05) { r = r * (1 - w) + 58 * w; g = g * (1 - w) + 123 * w; b = b * (1 - w) + 213 * w; }
          if (h > 0.05) { r = r * (1 - h) + 255 * h; g = g * (1 - h) + 90 * h; b = b * (1 - h) + 31 * h; }
          if (ru > 0.05) { r = r * (1 - ru) + 181 * ru; g = g * (1 - ru) + 139 * ru; b = b * (1 - ru) + 90 * ru; }
          if (wi > 0.05) { r = r * (1 - wi) + 176 * wi; g = g * (1 - wi) + 77 * wi; b = b * (1 - wi) + 255 * wi; }
        }
      } else if (sim.revealed[i]) { r = 60; g = 82; b = 64; }
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
    const ctx = this.mini, W = ctx.canvas.width;
    this.off.getContext('2d')!.putImageData(this.img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.off, 0, 0, W, W);
    const s = W / (N * CELL);
    const P = (x: number, y: number): [number, number] => [(x + HALF) * s, (HALF - y) * s];
    // grid every 50 m + compass
    ctx.strokeStyle = 'rgba(255,255,255,.06)'; ctx.lineWidth = 1;
    for (let k = -100; k <= 100; k += 50) { const [a] = P(k, 0), [, bb] = P(0, k); ctx.beginPath(); ctx.moveTo(a, 0); ctx.lineTo(a, W); ctx.moveTo(0, bb); ctx.lineTo(W, bb); ctx.stroke(); }
    ctx.fillStyle = 'rgba(255,255,255,.55)'; ctx.font = '600 10px Inter, sans-serif'; ctx.fillText('N', W - 14, 13);
    if (info?.path && info.path.length > 1) {
      ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.beginPath();
      info.path.forEach((p, k) => { const [x, y] = P(p.x, p.y); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.setLineDash([]);
    }
    for (const p of sim.probes) {
      const [x, y] = P(p.x, p.y);
      ctx.strokeStyle = p.alive ? 'rgba(52,209,122,.7)' : 'rgba(255,81,71,.5)'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, SENSOR.probeRadius * s, 0, 7); ctx.stroke();
      ctx.fillStyle = p.alive ? '#34d17a' : '#ff5147'; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
    }
    if (info?.goalPt) { const [x, y] = P(info.goalPt.x, info.goalPt.y); ctx.strokeStyle = '#c58bff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.stroke(); }
    const r = sim.rover, [rx, ry] = P(r.x, r.y);
    // sensor range + heading wedge
    ctx.fillStyle = 'rgba(242,163,58,.12)'; ctx.beginPath(); ctx.moveTo(rx, ry); ctx.arc(rx, ry, 28 * s, -r.yaw - 0.6, -r.yaw + 0.6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#f2a33a'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(rx + 8 * Math.cos(r.yaw), ry - 8 * Math.sin(r.yaw)); ctx.lineTo(rx + 5 * Math.cos(r.yaw + 2.4), ry - 5 * Math.sin(r.yaw + 2.4)); ctx.lineTo(rx + 5 * Math.cos(r.yaw - 2.4), ry - 5 * Math.sin(r.yaw - 2.4)); ctx.closePath(); ctx.fill(); ctx.stroke();
  }
}

const esc = (s: string) => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]!));
