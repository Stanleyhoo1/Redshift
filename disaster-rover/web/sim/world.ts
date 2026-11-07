import { Rng } from './rng';

/** World is 200 m x 200 m, x and y in [-100, 100], z up. */
export const WORLD = 200;
export const HALF = WORLD / 2;
/** Coarse environment grid (terrain, water, fire, smoke): 2 m cells. */
export const CELL = 2;
export const N = WORLD / CELL;
/** Fine occupancy grid for LiDAR raycasts and collisions: 0.5 m cells. */
export const OCC_RES = 0.5;
export const ON = WORLD / OCC_RES;

export enum Land { Grass = 0, Field = 1, Forest = 2, Town = 3, Water = 4, Rock = 5, Road = 6 }
export enum Occ { Free = 0, Tree = 1, Building = 2, Rubble = 3, Rock = 4, Crack = 5, Crater = 6 }

export interface Building { id: number; x: number; y: number; w: number; d: number; h: number; collapse: number }
export interface Tree { x: number; y: number; r: number; burnt: boolean }
export interface Crater { x: number; y: number; r: number; t: number }
export interface Crack { x0: number; y0: number; x1: number; y1: number; w: number }

export interface WorldOptions { seed: number; flat?: boolean }

const smooth = (t: number) => t * t * (3 - 2 * t);
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

/** Seeded 2D value noise with fbm. */
class Noise {
  private p: Float32Array;
  constructor(rng: Rng) { this.p = new Float32Array(256 * 256); for (let i = 0; i < this.p.length; i++) this.p[i] = rng.next(); }
  private v(ix: number, iy: number) { return this.p[((ix & 255) << 8) | (iy & 255)]; }
  value(x: number, y: number) {
    const ix = Math.floor(x), iy = Math.floor(y), fx = smooth(x - ix), fy = smooth(y - iy);
    const a = this.v(ix, iy), b = this.v(ix + 1, iy), c = this.v(ix, iy + 1), d = this.v(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
  }
  fbm(x: number, y: number, oct = 4) {
    let s = 0, amp = 0.5, f = 1;
    for (let i = 0; i < oct; i++) { s += amp * this.value(x * f, y * f); f *= 2; amp *= 0.5; }
    return s / (1 - Math.pow(0.5, oct));
  }
}

export class World {
  readonly seed: number;
  readonly flat: boolean;
  /** ground elevation (m), per coarse cell */
  height = new Float32Array(N * N);
  /** land cover per coarse cell */
  land = new Uint8Array(N * N);
  /** permanent river / lake cells */
  river = new Uint8Array(N * N);
  /** dynamic water depth above ground (m) */
  water = new Float32Array(N * N);
  /** wildfire state */
  fuel = new Float32Array(N * N);
  burning = new Float32Array(N * N);
  burnt = new Uint8Array(N * N);
  smoke = new Float32Array(N * N);
  /** rubble cover 0..1 (earthquake debris) */
  rubble = new Float32Array(N * N);
  /** fine occupancy grid */
  occ = new Uint8Array(ON * ON);
  buildings: Building[] = [];
  trees: Tree[] = [];
  craters: Crater[] = [];
  cracks: Crack[] = [];
  /** flood stage (water surface elevation, m). River surface is 0. */
  stage = 0;
  start = { x: 0, y: 0, yaw: 0 };
  town = { x: 0, y: 0, r: 0 };

  constructor(opts: WorldOptions) {
    this.seed = opts.seed;
    this.flat = !!opts.flat;
    const rng = new Rng(opts.seed);
    if (this.flat) this.genFlat(); else this.generate(rng);
    this.rebuildOcc();
  }

  // ---------- index helpers ----------
  static idx(cx: number, cy: number) { return cy * N + cx; }
  static cellOf(x: number, y: number): [number, number] {
    return [clamp(Math.floor((x + HALF) / CELL), 0, N - 1), clamp(Math.floor((y + HALF) / CELL), 0, N - 1)];
  }
  static cellCenter(cx: number, cy: number): [number, number] { return [-HALF + (cx + 0.5) * CELL, -HALF + (cy + 0.5) * CELL]; }
  cellIndexAt(x: number, y: number) { const [cx, cy] = World.cellOf(x, y); return World.idx(cx, cy); }
  inBounds(x: number, y: number) { return x > -HALF && x < HALF && y > -HALF && y < HALF; }

  /** Bilinear ground height at a world point. */
  heightAt(x: number, y: number): number {
    const gx = clamp((x + HALF) / CELL - 0.5, 0, N - 1.001), gy = clamp((y + HALF) / CELL - 0.5, 0, N - 1.001);
    const ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy;
    const h = this.height;
    const a = h[iy * N + ix], b = h[iy * N + ix + 1], c = h[(iy + 1) * N + ix], d = h[(iy + 1) * N + ix + 1];
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
  }
  /** Ground gradient (dz/dx, dz/dy). */
  gradAt(x: number, y: number): [number, number] {
    const e = 1.0;
    return [(this.heightAt(x + e, y) - this.heightAt(x - e, y)) / (2 * e), (this.heightAt(x, y + e) - this.heightAt(x, y - e)) / (2 * e)];
  }
  /** depth at a point: the cell's water surface minus the (bilinear) ground there, so depth varies smoothly at shorelines */
  waterAt(x: number, y: number) {
    const i = this.cellIndexAt(x, y), w = this.water[i];
    if (w <= 0) return 0;
    return Math.max(0, this.height[i] + w - this.heightAt(x, y));
  }
  /** Height including structures, as a depth camera would see it (DSM). */
  surfaceAt(x: number, y: number): number {
    const o = this.occAt(x, y);
    const g = this.heightAt(x, y) + this.water[this.cellIndexAt(x, y)];
    if (o === Occ.Building) { const b = this.buildingAt(x, y); return g + (b ? b.h * (1 - 0.7 * b.collapse) : 6); }
    if (o === Occ.Tree) return g + 8;
    if (o === Occ.Rubble) return g + 1.2;
    if (o === Occ.Rock) return g + 1.0;
    if (o === Occ.Crater) return g;
    return g;
  }
  buildingAt(x: number, y: number): Building | undefined {
    return this.buildings.find(b => Math.abs(x - b.x) <= b.w / 2 && Math.abs(y - b.y) <= b.d / 2);
  }
  occAt(x: number, y: number): number {
    const ox = Math.floor((x + HALF) / OCC_RES), oy = Math.floor((y + HALF) / OCC_RES);
    if (ox < 0 || oy < 0 || ox >= ON || oy >= ON) return Occ.Rock;
    return this.occ[oy * ON + ox];
  }
  setOccDisk(x: number, y: number, r: number, v: Occ, onlyFree = false) {
    const r2 = r * r;
    const x0 = Math.floor((x - r + HALF) / OCC_RES), x1 = Math.ceil((x + r + HALF) / OCC_RES);
    const y0 = Math.floor((y - r + HALF) / OCC_RES), y1 = Math.ceil((y + r + HALF) / OCC_RES);
    for (let oy = Math.max(0, y0); oy < Math.min(ON, y1); oy++) for (let ox = Math.max(0, x0); ox < Math.min(ON, x1); ox++) {
      const wx = -HALF + (ox + 0.5) * OCC_RES, wy = -HALF + (oy + 0.5) * OCC_RES;
      if ((wx - x) ** 2 + (wy - y) ** 2 <= r2) { const i = oy * ON + ox; if (!onlyFree || this.occ[i] === Occ.Free) this.occ[i] = v; }
    }
  }
  setOccRect(x: number, y: number, w: number, d: number, v: Occ) {
    const x0 = Math.floor((x - w / 2 + HALF) / OCC_RES), x1 = Math.ceil((x + w / 2 + HALF) / OCC_RES);
    const y0 = Math.floor((y - d / 2 + HALF) / OCC_RES), y1 = Math.ceil((y + d / 2 + HALF) / OCC_RES);
    for (let oy = Math.max(0, y0); oy < Math.min(ON, y1); oy++) for (let ox = Math.max(0, x0); ox < Math.min(ON, x1); ox++) this.occ[oy * ON + ox] = v;
  }
  rebuildOcc() {
    this.occ.fill(0);
    for (const t of this.trees) if (!t.burnt) this.setOccDisk(t.x, t.y, t.r, Occ.Tree);
    for (const b of this.buildings) this.setOccRect(b.x, b.y, b.w, b.d, Occ.Building);
  }

  // ---------- generation ----------
  private genFlat() {
    this.height.fill(1);
    this.land.fill(Land.Grass);
    for (let i = 0; i < N * N; i++) this.fuel[i] = 0.4;
    // a few boxes so obstacle avoidance has something to do
    this.buildings.push({ id: 0, x: 20, y: 10, w: 8, d: 8, h: 6, collapse: 0 }, { id: 1, x: -15, y: -25, w: 10, d: 6, h: 5, collapse: 0 });
    this.start = { x: 0, y: 0, yaw: 0 };
  }

  private generate(rng: Rng) {
    const noise = new Noise(rng);
    // River: meanders west->east across the map, with a lake at one end.
    const rA = rng.range(12, 22), rL = rng.range(35, 55), rP = rng.range(0, 6.28), rOff = rng.range(-35, 35);
    const riverY = (x: number) => rOff + rA * Math.sin(x / rL + rP) + 6 * Math.sin(x / 17 + 2 * rP);
    const lakeSide = rng.next() < 0.5 ? -1 : 1;
    const lake = { x: lakeSide * rng.range(55, 70), y: riverY(lakeSide * 62) + rng.range(-8, 8), r: rng.range(14, 20) };
    // One big hill/ridge on the side away from the river offset, for high ground.
    const hillSide = rOff > 0 ? -1 : 1;
    const hills = [
      { x: rng.range(-60, 60), y: hillSide * rng.range(45, 70), r: rng.range(30, 40), h: rng.range(12, 18) },
      { x: rng.range(-70, 70), y: -hillSide * rng.range(65, 85), r: rng.range(18, 26), h: rng.range(7, 11) },
      { x: rng.range(-80, 80), y: rng.range(-30, 30) + hillSide * 20, r: rng.range(12, 18), h: rng.range(4, 7) },
    ];
    for (let cy = 0; cy < N; cy++) for (let cx = 0; cx < N; cx++) {
      const [x, y] = World.cellCenter(cx, cy);
      const dr = Math.abs(y - riverY(x));
      // floodplain profile: low near the river, rising away from it
      let h = 0.6 + 3.2 * smooth(clamp(dr / 45, 0, 1)) + 2.2 * noise.fbm(x / 40 + 10, y / 40 + 10);
      for (const hl of hills) { const d = Math.hypot(x - hl.x, y - hl.y) / hl.r; if (d < 1.6) h += hl.h * Math.exp(-d * d * 1.4); }
      h += 1.2 * (noise.fbm(x / 15, y / 15) - 0.5);
      const halfW = 4 + 1.5 * noise.value(x / 20, 3);
      let isRiver = false;
      if (dr < halfW) { h = -1.4 * (1 - dr / halfW) - 0.3; isRiver = true; }
      // banks ramp up gently (max ~19°): no cliffs where hills meet the floodplain
      else h = Math.min(h, 0.25 + (dr - halfW) * 0.34);
      const dl = Math.hypot(x - lake.x, y - lake.y) / lake.r;
      if (dl < 1) { h = -2 * (1 - dl) - 0.3; isRiver = true; }
      else h = Math.min(h, 0.3 + (dl - 1) * lake.r * 0.34);
      const i = World.idx(cx, cy);
      this.height[i] = h;
      this.river[i] = isRiver ? 1 : 0;
    }
    // land cover
    for (let i = 0; i < N * N; i++) {
      const cx = i % N, cy = (i / N) | 0; const [x, y] = World.cellCenter(cx, cy);
      if (this.river[i]) { this.land[i] = Land.Water; this.fuel[i] = 0; continue; }
      const [gx, gy] = this.gradAtCell(cx, cy);
      const slope = Math.hypot(gx, gy);
      const f = noise.fbm(x / 35 + 50, y / 35 - 20);
      if (slope > 0.9) { this.land[i] = Land.Rock; this.fuel[i] = 0.05; }
      else if (f > 0.56) { this.land[i] = Land.Forest; this.fuel[i] = 1; }
      else if (f < 0.38) { this.land[i] = Land.Field; this.fuel[i] = 0.25; }
      else { this.land[i] = Land.Grass; this.fuel[i] = 0.5; }
    }
    // town: flattish dry spot, moderately low (so floods threaten it)
    let best = { x: 0, y: 0, s: -1e9 };
    for (let k = 0; k < 300; k++) {
      const x = rng.range(-70, 70), y = rng.range(-70, 70);
      const dr = Math.abs(y - riverY(x));
      if (dr < 22 || Math.hypot(x - lake.x, y - lake.y) < lake.r + 15) continue;
      const h = this.heightAt(x, y);
      let rough = 0; for (let a = 0; a < 8; a++) rough += Math.abs(this.heightAt(x + 14 * Math.cos(a), y + 14 * Math.sin(a)) - h);
      const s = -rough - Math.abs(h - 2.5) * 1.5 - Math.abs(dr - 35) * 0.05;
      if (s > best.s) best = { x, y, s };
    }
    this.town = { x: best.x, y: best.y, r: 22 };
    let bid = 0;
    const tRot = 0;
    for (let gy = -2; gy <= 2; gy++) for (let gx = -2; gx <= 2; gx++) {
      if (rng.next() < 0.25) continue;
      const bx = best.x + gx * 11 + rng.range(-1, 1), by = best.y + gy * 11 + rng.range(-1, 1);
      if (Math.hypot(bx - best.x, by - best.y) > 26) continue;
      const i = this.cellIndexAt(bx, by); if (this.river[i]) continue;
      this.buildings.push({ id: bid++, x: bx, y: by, w: rng.range(5, 8), d: rng.range(5, 8), h: rng.range(4, 11), collapse: 0 });
    }
    void tRot;
    for (let i = 0; i < N * N; i++) {
      const [x, y] = World.cellCenter(i % N, (i / N) | 0);
      if (!this.river[i] && Math.hypot(x - best.x, y - best.y) < 30) {
        const onRoad = Math.abs(((x - best.x + 5.5) % 11 + 11) % 11 - 5.5) < 1.2 || Math.abs(((y - best.y + 5.5) % 11 + 11) % 11 - 5.5) < 1.2;
        this.land[i] = onRoad ? Land.Road : Land.Town; this.fuel[i] = onRoad ? 0 : 0.15;
      }
    }
    // trees in forest cells
    for (let k = 0; k < 30000 && this.trees.length < 1400; k++) {
      const x = rng.range(-HALF + 2, HALF - 2), y = rng.range(-HALF + 2, HALF - 2);
      const i = this.cellIndexAt(x, y);
      const dens = this.land[i] === Land.Forest ? 0.22 : this.land[i] === Land.Grass ? 0.006 : 0;
      if (rng.next() > dens) continue;
      if (this.trees.some(t => (t.x - x) ** 2 + (t.y - y) ** 2 < 16)) continue;
      this.trees.push({ x, y, r: rng.range(0.4, 0.7), burnt: false });
    }
    // start: open, dry, gentle, not too close to river or town, near the middle
    let sb = { x: 0, y: 0, s: -1e9 };
    for (let k = 0; k < 400; k++) {
      const x = rng.range(-45, 45), y = rng.range(-45, 45); const i = this.cellIndexAt(x, y);
      if (this.river[i] || this.land[i] === Land.Forest || this.land[i] === Land.Town || this.land[i] === Land.Rock) continue;
      const [gx, gy] = this.gradAt(x, y); const dr = Math.abs(y - riverY(x));
      if (this.trees.some(t => (t.x - x) ** 2 + (t.y - y) ** 2 < 25)) continue;
      if (this.buildings.some(b => Math.abs(b.x - x) < b.w / 2 + 6 && Math.abs(b.y - y) < b.d / 2 + 6)) continue;
      const s = -Math.hypot(gx, gy) * 10 - Math.abs(dr - 28) * 0.1 - Math.hypot(x, y) * 0.03 - Math.max(0, 25 - Math.hypot(x - best.x, y - best.y));
      if (s > sb.s) sb = { x, y, s };
    }
    this.start = { x: sb.x, y: sb.y, yaw: rng.range(-Math.PI, Math.PI) };
    for (let i = 0; i < N * N; i++) this.water[i] = this.river[i] ? Math.max(0, -this.height[i]) : 0;
  }
  private gradAtCell(cx: number, cy: number): [number, number] {
    const h = (x: number, y: number) => this.height[World.idx(clamp(x, 0, N - 1), clamp(y, 0, N - 1))];
    return [(h(cx + 1, cy) - h(cx - 1, cy)) / (2 * CELL), (h(cx, cy + 1) - h(cx, cy - 1)) / (2 * CELL)];
  }
}
