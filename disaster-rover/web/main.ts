import { Sim } from './sim/sim';
import { SceneView } from './render/scene';
import { UI, BrainMode, BrainInfo } from './ui/ui';
import { LocalBrain } from './brain/brain';
import { AgentClient } from './brain/agent';
import { RosLink } from './bridge/rosbridge';
import type { BrainOutput, SensorFrame } from './brain/msgs';
import { zeroTwist } from './brain/msgs';

const keys = new Set<string>();
window.addEventListener('keydown', e => { if ((e.target as HTMLElement).tagName === 'INPUT') return; keys.add(e.key.toLowerCase()); });
window.addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));

function manual(): BrainOutput {
  const t = zeroTwist();
  if (keys.has('w')) t.linear.x = 3; if (keys.has('s')) t.linear.x = -1.5;
  if (keys.has('a')) t.angular.z = 1.4; if (keys.has('d')) t.angular.z = -1.4;
  const drop = keys.has('p'); keys.delete('p');
  return { cmd_vel: t, drop_probe: drop ? { x: 0, y: 0, z: 0 } : null, decision_log: null };
}

const ui = new UI();
const view = new SceneView(document.getElementById('c') as HTMLCanvasElement);
const ros = new RosLink();
let sim = new Sim({ seed: ui.seed(), duration: ui.duration(), disasters: ui.disasterConfig() });
let local = new LocalBrain();
let agent = new AgentClient();
let mode: BrainMode = ui.brainMode();
view.load(sim);
let running = false, acc = 0, last = performance.now();

function newRun() {
  sim = new Sim({ seed: ui.seed(), duration: ui.duration(), disasters: ui.disasterConfig() });
  local = new LocalBrain();
  agent.reset();
  view.load(sim); running = false; acc = 0; ui.clearLogs();
}
ui.onStart = () => { if (sim.done || sim.t > 0) newRun(); running = true; };
ui.onPause = () => { running = !running; };
ui.onReset = () => newRun();
ui.onTrigger = k => { sim.trigger(k); };
let connectToken = 0;
ui.onBrainChange = async m => {
  const token = ++connectToken;
  if (m === 'ros') {
    ui.setRosStatus('connecting…', 'warn');
    const ok = await ros.connect(ui.rosUrl());
    if (token !== connectToken) return; // a newer selection superseded this attempt
    if (!ok) {
      ui.setRosStatus(`⚠ ${ros.status}: falling back to Local TS brain`, 'bad');
      ui.setBrainMode('local'); mode = 'local'; return;
    }
    ui.setRosStatus(`ROS 2 brain: ${ros.status}`, 'ok');
    ui.setBrainMode('ros');
    ros.setAgent(ui.agentOn());
  } else { ros.close(); ui.setRosStatus(m === 'local' ? 'Local TS brain (in browser)' : 'Manual driving', ''); }
  mode = m;
};
ui.onAgentToggle = on => { if (mode === 'ros') ros.setAgent(on); agent.enabled = on; };
agent.enabled = ui.agentOn();

/** One brain tick (10 Hz sim time). */
function brainTick(f: SensorFrame): BrainOutput {
  if (mode === 'manual') return manual();
  if (mode === 'ros') {
    if (!ros.connected) {
      ui.setRosStatus(`⚠ ${ros.status}: falling back to Local TS brain`, 'bad');
      ui.setBrainMode('local'); mode = 'local';
    } else {
      ros.publishFrame(f);
      const drop = ros.dropPending; ros.dropPending = false;
      const logs = ros.logs.splice(0); const alogs = ros.agentLogs.splice(0);
      // stale ROS commands are not trusted
      const cmd = ros.cmdAge() < 1000 ? ros.lastCmd : zeroTwist();
      return { cmd_vel: cmd, drop_probe: drop ? { x: 0, y: 0, z: 0 } : null, decision_log: logs.join('\n') || null, agent_log: alogs.join('\n') || null };
    }
  }
  const out = local.tick(f);
  const alog = agent.step(local, sim.t);
  if (alog) out.agent_log = alog;
  return out;
}

function brainInfo(): BrainInfo | null {
  if (mode === 'local') return local.info();
  if (mode === 'ros' && ros.info) return { ...ros.info, view: ros.hasBelief ? ros.belief : null };
  return null;
}

function agentStatus() {
  if (!ui.agentOn()) { ui.setAgentStatus('Agent off: utility brain only', ''); return; }
  if (mode === 'ros') {
    const st = ros.agentStatus, age = performance.now() - ros.agentStatusAt;
    if (!st || age > 5000) ui.setAgentStatus('Agent offline, utility fallback', 'bad');
    else ui.setAgentStatus(`Gemma 4 (ROS): ${st.state}${st.latency_s ? ` · last ${st.latency_s.toFixed(1)} s` : ''}`, st.state === 'error' ? 'bad' : 'ok');
  } else if (mode === 'local') {
    const s = agent.statusText();
    ui.setAgentStatus(s.text, s.ok ? 'ok' : 'bad');
  } else ui.setAgentStatus('Agent idle in manual mode', '');
}

let statusAcc = 0;
function loop(now: number) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (running && !sim.done) {
    acc += dt * ui.speed();
    while (acc >= Sim.DT && !sim.done) {
      if (sim.steps % Sim.BRAIN_EVERY === 0) {
        const f = sim.frame();
        const out = brainTick(f);
        sim.apply(out);
        if (out.decision_log) for (const l of out.decision_log.split('\n')) ui.log(sim.t, l);
        if (out.agent_log) for (const l of out.agent_log.split('\n')) ui.agentLog(sim.t, l);
      }
      sim.step(); acc -= Sim.DT;
    }
  }
  const info = brainInfo();
  view.toggles = ui.toggles();
  view.render(info?.view ?? null, info?.path ?? null, info?.goalPt ?? null);
  ui.update(sim, info);
  if ((statusAcc += dt) > 0.5) { statusAcc = 0; agentStatus(); }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
