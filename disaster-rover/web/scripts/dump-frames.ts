/** Dump sensor frames + TS brain outputs to JSONL for the Python parity test. */
import { writeFileSync } from 'node:fs';
import { Sim } from '../sim/sim';
import { LocalBrain } from '../brain/brain';
import { parseArgs, disasterConfig } from './args';
const a = parseArgs(process.argv.slice(2));
const sim = new Sim({ seed: +(a.seed ?? 3), duration: +(a.duration ?? 60), disasters: disasterConfig(a.disasters, +(a.intensity ?? 0.6), false) });
const br = new LocalBrain();
const lines: string[] = [];
while (!sim.done) {
  if (sim.steps % Sim.BRAIN_EVERY === 0) {
    const f = sim.frame(); const o = br.tick(f); sim.apply(o);
    lines.push(JSON.stringify({ frame: f, ts: { behaviour: br.behaviour, cmd: o.cmd_vel, reflex: br.reflex?.name ?? null } }));
  }
  sim.step();
}
writeFileSync(a.out ?? 'frames.jsonl', lines.join('\n'));
console.log('wrote', lines.length, 'frames', JSON.stringify(sim.metrics()));
