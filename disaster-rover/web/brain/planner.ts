import { CFG, CH } from './config';
import { Belief } from './grid';

/** Binary min-heap keyed by number (shared by A* and Dijkstra). */
class Heap {
  k: number[] = []; v: number[] = [];
  get size() { return this.k.length; }
  push(key: number, val: number) {
    const k = this.k, v = this.v; k.push(key); v.push(val);
    let i = k.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (k[p] <= k[i]) break; [k[p], k[i]] = [k[i], k[p]]; [v[p], v[i]] = [v[i], v[p]]; i = p; }
  }
  pop(): number {
    const k = this.k, v = this.v, top = v[0];
    const lk = k.pop()!, lv = v.pop()!;
    if (k.length) {
      k[0] = lk; v[0] = lv; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < k.length && k[l] < k[m]) m = l; if (r < k.length && k[r] < k[m]) m = r;
        if (m === i) break; [k[m], k[i]] = [k[i], k[m]]; [v[m], v[i]] = [v[i], v[m]]; i = m;
      }
    }
    return top;
  }
}

const NB: [number, number, number][] = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, 1.4142], [1, -1, 1.4142], [-1, 1, 1.4142], [-1, -1, 1.4142]];

/**
 * Cost of entering a cell. Infinity = impassable.
 * Weighted by predicted risk so the route avoids where the hazard WILL be, not just where it is.
 */
export function cellCost(b: Belief, i: number, startI: number, relaxed = false): number {
  if (i === startI) return 1;
  if (b.blocked(i)) return Infinity;
  const r = b.planRisk(i), sl = b.channels[CH.slope][i], w = b.channels[CH.water][i];
  if ((r > 0.85 && !relaxed) || sl >= 0.6 || w > 0.7) return Infinity;
  let c = 1 + 9 * r * r + 6 * sl + (b.lastSeen[i] < 0 ? 0.6 : 0);
  // keep a margin from obstacles (the 2 m grid is coarse vs the 0.9 m rover radius)
  const G = CFG.G, x = i % G, y = (i / G) | 0;
  if ((x > 0 && b.blocked(i - 1)) || (x < G - 1 && b.blocked(i + 1)) || (y > 0 && b.blocked(i - G)) || (y < G - 1 && b.blocked(i + G))) c += 2.5;
  return c;
}

export interface PathResult { path: { x: number; y: number }[]; cost: number; cells: number[] }

/** A* from (sx,sy) to (gx,gy) on the belief grid. Returns null if unreachable. */
export function astar(b: Belief, sx: number, sy: number, gx: number, gy: number, relaxed = false, maxExpand = 12000): PathResult | null {
  const G = CFG.G, s = b.idx(sx, sy), g = b.idx(gx, gy);
  if (cellCost(b, g, s, relaxed) === Infinity && g !== s) return null;
  const gs = new Float32Array(G * G).fill(Infinity), from = new Int32Array(G * G).fill(-1), closed = new Uint8Array(G * G);
  const h = (i: number) => Math.hypot((i % G) - (g % G), ((i / G) | 0) - ((g / G) | 0));
  const open = new Heap(); gs[s] = 0; open.push(h(s), s);
  let exp = 0;
  while (open.size) {
    const i = open.pop(); if (closed[i]) continue; closed[i] = 1;
    if (i === g) break;
    if (++exp > maxExpand) return null;
    const x = i % G, y = (i / G) | 0;
    for (const [dx, dy, dl] of NB) {
      const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= G || ny >= G) continue;
      const j = ny * G + nx; if (closed[j]) continue;
      // no corner cutting past obstacles
      if (dx && dy && (b.blocked(y * G + nx) || b.blocked(ny * G + x))) continue;
      const c = cellCost(b, j, s, relaxed); if (c === Infinity) continue;
      const ng = gs[i] + dl * c;
      if (ng < gs[j]) { gs[j] = ng; from[j] = i; open.push(ng + h(j), j); }
    }
  }
  if (!closed[g]) return null;
  const cells: number[] = []; for (let i = g; i !== -1; i = from[i]) cells.push(i);
  cells.reverse();
  const path = cells.map(i => ({ x: b.cx(i), y: b.cy(i) }));
  if (path.length) path[path.length - 1] = { x: gx, y: gy };
  return { path, cost: gs[g] * CFG.RES, cells };
}

/** Dijkstra from the rover out to maxDist metres: travel cost (m-equivalent) per reachable cell. */
export function reachMap(b: Belief, sx: number, sy: number, maxDist: number, relaxed = false): Float32Array {
  const G = CFG.G, s = b.idx(sx, sy);
  const gs = new Float32Array(G * G).fill(Infinity);
  const open = new Heap(); gs[s] = 0; open.push(0, s);
  const lim = maxDist / CFG.RES * 3;
  while (open.size) {
    const i = open.pop(); const gi = gs[i]; if (gi > lim) continue;
    const x = i % G, y = (i / G) | 0;
    for (const [dx, dy, dl] of NB) {
      const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= G || ny >= G) continue;
      const j = ny * G + nx;
      if (Math.hypot(nx - (s % G), ny - ((s / G) | 0)) * CFG.RES > maxDist) continue;
      if (dx && dy && (b.blocked(y * G + nx) || b.blocked(ny * G + x))) continue;
      const c = cellCost(b, j, s, relaxed); if (c === Infinity) continue;
      const ng = gi + dl * c;
      if (ng < gs[j]) { gs[j] = ng; open.push(ng, j); }
    }
  }
  for (let i = 0; i < gs.length; i++) if (gs[i] !== Infinity) gs[i] *= CFG.RES;
  return gs;
}

/** max planning risk along the first `metres` of a path */
export function pathRisk(b: Belief, cells: number[], metres: number): { risk: number; at: number } {
  let worst = 0, at = -1;
  const lim = Math.ceil(metres / CFG.RES);
  for (let k = 1; k < Math.min(cells.length, lim); k++) { const r = b.planRisk(cells[k]); if (r > worst) { worst = r; at = k * CFG.RES; } }
  return { risk: worst, at };
}
