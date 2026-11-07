import { defaultDisasterConfig, DisasterConfig, DISASTER_KINDS, DisasterKind } from '../sim/disasters';

export function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const k = a.slice(2), v = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
    out[k] = v;
  }
  return out;
}

const ALIASES: Record<string, DisasterKind> = { flood: 'flood', fire: 'fire', wildfire: 'fire', quake: 'quake', earthquake: 'quake', tornado: 'tornado', meteor: 'meteor', meteors: 'meteor', rockfall: 'meteor' };

export function disasterConfig(list: string | undefined, intensity: number, random: boolean): DisasterConfig {
  const cfg = defaultDisasterConfig();
  cfg.random = random;
  for (const raw of (list ?? '').split(',').map(s => s.trim().toLowerCase()).filter(Boolean)) {
    if (raw === 'all') { for (const k of DISASTER_KINDS) cfg[k] = { enabled: true, intensity }; continue; }
    const k = ALIASES[raw];
    if (!k) throw new Error(`unknown disaster "${raw}" (use ${Object.keys(ALIASES).join(', ')}, all)`);
    cfg[k] = { enabled: true, intensity };
  }
  return cfg;
}
