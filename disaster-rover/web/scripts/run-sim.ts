/**
 * Headless runner: sim + local TS brain, no rendering.
 *   npm run sim -- --disasters flood,fire --seed 3 --duration 180 [--intensity 0.6] [--random] [--brain local|wander] [--agent] [--quiet] [--json]
 */
import { Sim } from '../sim/sim';
import { parseArgs, disasterConfig } from './args';
import { makeBrain } from './brains';

const a = parseArgs(process.argv.slice(2));
const seed = +(a.seed ?? 1), duration = +(a.duration ?? 180), intensity = +(a.intensity ?? 0.6);
const sim = new Sim({ seed, duration, disasters: disasterConfig(a.disasters, intensity, a.random === 'true'), flat: a.flat === 'true' });
const brain = await makeBrain(a.brain ?? 'local', a, sim);
const quiet = a.quiet === 'true' || a.json === 'true';
const t0 = performance.now();
let brainMs = 0, ticks = 0, lastEvent = 0;

while (!sim.done) {
  if (sim.steps % Sim.BRAIN_EVERY === 0) {
    const f = sim.frame();
    const tb = performance.now();
    const out = await brain.tick(f);
    brainMs += performance.now() - tb; ticks++;
    sim.apply(out);
    if (!quiet && out.decision_log) console.log(`[${sim.t.toFixed(1).padStart(6)}] ${out.decision_log}`);
    if (!quiet && out.agent_log) console.log(`[${sim.t.toFixed(1).padStart(6)}] ${out.agent_log}`);
  }
  sim.step();
  if (!quiet) for (; lastEvent < sim.dis.events.length; lastEvent++) { const e = sim.dis.events[lastEvent]; console.log(`[${e.t.toFixed(1).padStart(6)}] *** ${e.text}`); }
}
await brain.close?.();
const m = sim.metrics();
const wall = (performance.now() - t0) / 1000;
if (a.json === 'true') console.log(JSON.stringify({ seed, disasters: a.disasters ?? '', brain: brain.name, ...m, brainTickMs: +(brainMs / Math.max(1, ticks)).toFixed(2), wallS: +wall.toFixed(1), ...(brain.stats?.() ?? {}) }));
else {
  console.log('\n================ RESULT ================');
  console.log(`seed ${seed} | disasters: ${a.disasters ?? 'none'} | brain: ${brain.name}`);
  console.log(m.survived ? `SURVIVED (${m.t}s, health ${m.health}%, battery ${m.battery}%)` : `FAILED at ${m.t}s: ${m.cause}`);
  console.log(`probes placed: ${m.probesPlaced} (alive ${m.probesAlive}) | area surveyed: ${m.surveyedPct}% | hazard coverage by probes: ${m.hazardCoveragePct}%`);
  console.log(`brain tick: ${(brainMs / Math.max(1, ticks)).toFixed(2)} ms avg over ${ticks} ticks | wall ${wall.toFixed(1)} s`);
  const st = brain.stats?.(); if (st) console.log('agent:', JSON.stringify(st));
}
process.exit(0);
