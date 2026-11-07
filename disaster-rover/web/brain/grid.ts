import { CFG, NCH } from './config';
import type { BeliefView } from './view';

/**
 * The brain's belief about the world: a 2 m grid built ONLY from sensor readings.
 * Raw observations (elev, water, heat, obstacle evidence, ...) are written by perception;
 * per-hazard risk channels + predictions are recomputed from them by prediction.
 */
export class Belief implements BeliefView {
  readonly n = CFG.G; readonly res = CFG.RES;
  // raw observations
  elev = new Float32Array(CFG.G * CFG.G).fill(NaN);
  /** ground elevation seen while the cell was dry (for flood prediction) */
  dryElev = new Float32Array(CFG.G * CFG.G).fill(NaN);
  struct = new Uint8Array(CFG.G * CFG.G);
  obst = new Float32Array(CFG.G * CFG.G);
  /** observed water depth; -1 = never seen */
  water = new Float32Array(CFG.G * CFG.G).fill(-1);
  waterT = new Float32Array(CFG.G * CFG.G).fill(-1);
  /** observed temperature (°C); NaN never */
  heat = new Float32Array(CFG.G * CFG.G).fill(NaN);
  heatT = new Float32Array(CFG.G * CFG.G).fill(-1);
  /** rubble / impact evidence 0..1 (decays) */
  rubbleObs = new Float32Array(CFG.G * CFG.G);
  lastSeen = new Float32Array(CFG.G * CFG.G).fill(-1);
  // derived (prediction.ts)
  channels: Float32Array[] = Array.from({ length: NCH }, () => new Float32Array(CFG.G * CFG.G));
  risk = new Float32Array(CFG.G * CFG.G);
  predicted = new Float32Array(CFG.G * CFG.G);
  /** distance (m) to the nearest predicted hazard cell */
  hazDist = new Float32Array(CFG.G * CFG.G).fill(999);
  conf = new Float32Array(CFG.G * CFG.G);
  /** predicted (30 s) water risk alone, for high-ground decisions */
  predWater = new Float32Array(CFG.G * CFG.G);
  /** distance (m) to nearest seen water / nearest structure */
  waterDist = new Float32Array(CFG.G * CFG.G).fill(999);
  structDist = new Float32Array(CFG.G * CFG.G).fill(999);
  t = 0;

  idx(x: number, y: number): number {
    let cx = Math.floor((x + CFG.HALF) / CFG.RES), cy = Math.floor((y + CFG.HALF) / CFG.RES);
    if (cx < 0) cx = 0; if (cy < 0) cy = 0; if (cx >= CFG.G) cx = CFG.G - 1; if (cy >= CFG.G) cy = CFG.G - 1;
    return cy * CFG.G + cx;
  }
  inside(x: number, y: number) { return x > -CFG.HALF && x < CFG.HALF && y > -CFG.HALF && y < CFG.HALF; }
  cx(i: number) { return -CFG.HALF + ((i % CFG.G) + 0.5) * CFG.RES; }
  cy(i: number) { return -CFG.HALF + (Math.floor(i / CFG.G) + 0.5) * CFG.RES; }
  /** both grids share geometry (2 m) so this is the identity; kept so the view can be swapped */
  cellOfWorld(i: number) { return i; }

  see(i: number, t: number) { this.lastSeen[i] = t; }
  confidence(i: number, t: number) { const s = this.lastSeen[i]; return s < 0 ? 0 : Math.exp(-(t - s) / CFG.confTau); }
  updateConf(t: number) { this.t = t; for (let i = 0; i < this.conf.length; i++) this.conf[i] = this.confidence(i, t); }
  knownFrac() { let k = 0; for (let i = 0; i < this.lastSeen.length; i++) if (this.lastSeen[i] >= 0) k++; return k / this.lastSeen.length; }
  blocked(i: number) { return this.obst[i] > 0.55; }
  /** planning risk: current risk, or most of the predicted risk */
  planRisk(i: number) { const p = this.predicted[i] * 0.85; return this.risk[i] > p ? this.risk[i] : p; }
}
