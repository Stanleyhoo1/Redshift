import { Sim } from './sim/sim';
import { SceneView } from './render/scene';
import { UI } from './ui/ui';
import { LocalBrain } from './brain/brain';
import type { BrainOutput, SensorFrame } from './brain/msgs';
import { zeroTwist } from './brain/msgs';

const keys = new Set<string>();
window.addEventListener('keydown', e => { if ((e.target as HTMLElement).tagName === 'INPUT') return; keys.add(e.key.toLowerCase()); });
window.addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));

function manual(_f: SensorFrame): BrainOutput {
  const t = zeroTwist();
  if (keys.has('w')) t.linear.x = 3; if (keys.has('s')) t.linear.x = -1.5;
  if (keys.has('a')) t.angular.z = 1.4; if (keys.has('d')) t.angular.z = -1.4;
  const drop = keys.has('p'); keys.delete('p');
  return { cmd_vel: t, drop_probe: drop ? { x: 0, y: 0, z: 0 } : null, decision_log: null };
}

const ui = new UI();
const view = new SceneView(document.getElementById('c') as HTMLCanvasElement);
let sim = new Sim({ seed: ui.seed(), duration: ui.duration(), disasters: ui.disasterConfig() });
view.load(sim);
let running = false, acc = 0, last = performance.now();
let local = new LocalBrain();

ui.onStart = () => { if (sim.done || sim.t > 0) reset(); running = true; };
ui.onPause = () => { running = !running; };
ui.onReset = () => reset();
ui.onTrigger = k => sim.trigger(k);
function reset() { sim = new Sim({ seed: ui.seed(), duration: ui.duration(), disasters: ui.disasterConfig() }); local = new LocalBrain(); view.load(sim); running = false; ui.clearLogs(); }

function loop(now: number) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (running && !sim.done) {
    acc += dt * ui.speed();
    while (acc >= Sim.DT && !sim.done) {
      if (sim.steps % Sim.BRAIN_EVERY === 0) {
        const f = sim.frame();
        const out = ui.brainMode() === 'manual' ? manual(f) : local.tick(f);
        sim.apply(out);
        if (out.decision_log) ui.log(sim.t, out.decision_log);
      }
      sim.step(); acc -= Sim.DT;
    }
  }
  view.toggles = ui.toggles();
  const info = ui.brainMode() === 'local' ? local.info() : null;
  view.render(info?.view ?? null, info?.path ?? null, info?.goalPt ?? null);
  ui.update(sim, info);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
