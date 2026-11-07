/**
 * Survival benchmark: N seeds x scenarios, run in parallel child processes.
 *   npm run bench -- --scenarios flood,fire,flood+fire --seeds 1-10 [--brain local|ros] [--agent] [--duration 180] [--jobs 8] [--out results/x.json]
 */
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, appendFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseArgs } from './args';

const a = parseArgs(process.argv.slice(2));
const scenarios = (a.scenarios ?? 'flood,fire,quake,tornado,meteor,all').split(',');
const [s0, s1] = (a.seeds ?? '1-10').split('-').map(Number);
const seeds = Array.from({ length: (s1 ?? s0) - s0 + 1 }, (_, i) => s0 + i);
const jobs = +(a.jobs ?? (a.brain === 'ros' ? 1 : 8));
const extra = ['--duration', a.duration ?? '180', '--intensity', a.intensity ?? '0.6', '--brain', a.brain ?? 'local'];
if (a.agent === 'true') extra.push('--agent');
if (a.random === 'true') extra.push('--random');
if (a['no-probes'] === 'true') extra.push('--no-probes');

type Row = Record<string, any>;
const queue: { sc: string; seed: number }[] = [];
// --seed-major: all scenarios for seed 1, then seed 2, ... (partial runs still cover every scenario)
if (a['seed-major'] === 'true') for (const seed of seeds) for (const sc of scenarios) queue.push({ sc, seed });
else for (const sc of scenarios) for (const seed of seeds) queue.push({ sc, seed });
const rows: Row[] = [];
const runOne = (sc: string, seed: number) => new Promise<Row>(res => {
  const p = spawn(process.execPath, ['--import', 'tsx', 'scripts/run-sim.ts', '--disasters', sc.replace(/\+/g, ','), '--seed', String(seed), '--json', ...extra], { stdio: ['ignore', 'pipe', 'pipe'] });
  let out = '', err = '';
  const killer = setTimeout(() => p.kill('SIGKILL'), +(a['run-timeout'] ?? 300) * 1000);
  p.on('close', () => clearTimeout(killer));
  p.stdout.on('data', d => (out += d)); p.stderr.on('data', d => (err += d));
  p.on('close', () => {
    try { res({ scenario: sc, ...JSON.parse(out.trim().split('\n').pop()!) }); }
    catch { res({ scenario: sc, seed, survived: false, cause: 'runner error: ' + (err || out).slice(0, 200) }); }
  });
});
let done = 0;
await Promise.all(Array.from({ length: jobs }, async () => {
  while (queue.length) {
    const { sc, seed } = queue.shift()!;
    const r = await runOne(sc, seed); rows.push(r); done++;
    if (a.jsonl) appendFileSync(a.jsonl, JSON.stringify(r) + '\n');
    process.stderr.write(`\r${done}/${scenarios.length * seeds.length} ${sc} seed ${seed}: ${r.survived ? 'ok' : 'FAIL ' + r.cause}                    `);
  }
}));
process.stderr.write('\n');
console.log(`\n| scenario | survived | probes placed (avg) | area surveyed (avg %) | hazard coverage (avg %) | probe early warnings (avg) | failures |`);
console.log(`|---|---|---|---|---|---|---|`);
for (const sc of scenarios) {
  const rs = rows.filter(r => r.scenario === sc).sort((x, y) => x.seed - y.seed);
  const ok = rs.filter(r => r.survived).length;
  const avg = (k: string) => (rs.reduce((s, r) => s + (r[k] ?? 0), 0) / rs.length).toFixed(1);
  const fails = rs.filter(r => !r.survived).map(r => `s${r.seed}: ${r.cause}`).join('; ');
  console.log(`| ${sc} | ${ok}/${rs.length} | ${avg('probesPlaced')} | ${avg('surveyedPct')} | ${avg('hazardCoveragePct')} | ${avg('probeWarnings')} | ${fails || '-'} |`);
}
if (a.out) { mkdirSync(dirname(a.out), { recursive: true }); writeFileSync(a.out, JSON.stringify(rows, null, 1)); }
process.exit(0);
