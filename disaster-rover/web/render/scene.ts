import * as THREE from 'three';
import { Sim } from '../sim/sim';
import { World, N, CELL, HALF, Land } from '../sim/world';
import { SENSOR } from '../sim/sensors';
import type { BeliefView } from '../brain/view';
import { buildRover, buildProbe, buildBuilding, softDot, streakTex, RoverModel, ProbeModel, BuildingModel } from './models';
import { waterMaterial, skyDome } from './water';

/** sim (x, y, z-up) -> three (x, y-up, -y) */
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y);
const hash = (n: number) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };

const LAND: Record<number, [number, number, number]> = {
  [Land.Grass]: [0.36, 0.56, 0.25], [Land.Field]: [0.70, 0.64, 0.36], [Land.Forest]: [0.20, 0.38, 0.17],
  [Land.Town]: [0.52, 0.52, 0.48], [Land.Water]: [0.42, 0.38, 0.28], [Land.Rock]: [0.50, 0.47, 0.44], [Land.Road]: [0.30, 0.30, 0.32],
};
export const BEHAVIOUR_COLOR: Record<string, number> = {
  explore: 0x33ff88, flee: 0xff3b30, high_ground: 0x3aa0ff, hold_still: 0xffd60a, place_probe: 0xbf5af2, return_to_safety: 0xff9f0a,
};

export interface RenderToggles { lidar: boolean; hazards: boolean; fog: boolean; camera: boolean; path: boolean }

/** terrain mesh resolution: 1 m vertices (201 x 201) for smooth hills and shorelines */
const TR = 201, TS = 1;

export class SceneView {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  roverCam: THREE.PerspectiveCamera;
  toggles: RenderToggles = { lidar: true, hazards: false, fog: true, camera: true, path: true };
  private sim!: Sim;
  private sun = new THREE.DirectionalLight(0xfff1dc, 2.4);
  private hemi = new THREE.HemisphereLight(0xcfe4ff, 0x4a5a3a, 1.0);
  private sky = skyDome();
  private terrain!: THREE.Mesh; private tPos!: THREE.BufferAttribute; private tCol!: THREE.BufferAttribute;
  private tBase!: Float32Array; private tCell!: Int32Array; private shoreDist!: Float32Array;
  private water!: THREE.Mesh; private wPos!: THREE.BufferAttribute; private wDepth!: THREE.BufferAttribute; private wFlood!: THREE.BufferAttribute;
  private waterMat = waterMaterial();
  private conLow!: THREE.InstancedMesh; private conTop!: THREE.InstancedMesh; private broad!: THREE.InstancedMesh; private trunks!: THREE.InstancedMesh;
  private treeKind: Uint8Array = new Uint8Array(0); private treeScale: Float32Array = new Float32Array(0);
  private buildings: BuildingModel[] = [];
  private flamesO!: THREE.InstancedMesh; private flamesI!: THREE.InstancedMesh;
  private embers!: THREE.Points; private emberData: { x: number; y: number; z: number; vz: number; life: number }[] = [];
  private smoke!: THREE.Points; private rain!: THREE.LineSegments;
  private rubble!: THREE.InstancedMesh; private debris!: THREE.InstancedMesh;
  private meteors!: THREE.InstancedMesh; private trails!: THREE.LineSegments;
  private craterGroup = new THREE.Group(); private crackGroup = new THREE.Group(); private probeGroup = new THREE.Group();
  private hotCraters: { mesh: THREE.Mesh; t: number }[] = [];
  private tornado = new THREE.Group(); private funnels: THREE.Mesh[] = []; private dustCloud!: THREE.Points;
  private rover!: RoverModel;
  private probes: ProbeModel[] = [];
  private lidar!: THREE.LineSegments; private lidarHits!: THREE.Points;
  private pathLine!: THREE.Line; private goalMarker = new THREE.Group();
  private flashLight = new THREE.PointLight(0xffcc88, 0, 140, 1.5);
  private fireLight = new THREE.PointLight(0xff7a2a, 0, 90, 1.6);
  private camYaw = 0.6; private camPitch = 0.72; private camDist = 38;
  private camTarget = new THREE.Vector3();
  private lastSlow = -1; private nCraters = 0; private nCracks = 0; private nProbes = -1;
  private dummy = new THREE.Object3D();
  private frameT = 0;

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.camera = new THREE.PerspectiveCamera(52, 1, 0.3, 1000);
    this.roverCam = new THREE.PerspectiveCamera(70, 4 / 3, 0.2, 900);
    this.scene.fog = new THREE.Fog(0xb8cde0, 160, 420);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const sc = this.sun.shadow.camera; sc.left = -45; sc.right = 45; sc.top = 45; sc.bottom = -45; sc.near = 1; sc.far = 260;
    this.sun.shadow.bias = -0.0006; this.sun.shadow.normalBias = 0.04;
    this.installControls();
    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  private installControls() {
    let drag = false, lx = 0, ly = 0;
    this.canvas.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; ly = e.clientY; });
    window.addEventListener('pointerup', () => { drag = false; });
    window.addEventListener('pointermove', e => {
      if (!drag) return;
      this.camYaw -= (e.clientX - lx) * 0.006; this.camPitch = Math.max(0.08, Math.min(1.45, this.camPitch + (e.clientY - ly) * 0.005));
      lx = e.clientX; ly = e.clientY;
    });
    this.canvas.addEventListener('wheel', e => { this.camDist = Math.max(6, Math.min(220, this.camDist * (1 + Math.sign(e.deltaY) * 0.1))); e.preventDefault(); }, { passive: false });
  }

  resize() {
    const w = this.canvas.clientWidth || 800, h = this.canvas.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  /** (Re)build every mesh for a new sim. */
  load(sim: Sim) {
    this.sim = sim;
    this.scene.clear();
    this.buildings = []; this.probes = []; this.hotCraters = []; this.emberData = [];
    this.lastSlow = -1; this.nCraters = 0; this.nCracks = 0; this.nProbes = -1;
    this.craterGroup = new THREE.Group(); this.crackGroup = new THREE.Group(); this.probeGroup = new THREE.Group();
    const w = sim.world;
    this.scene.add(this.sky.mesh, this.hemi, this.sun, this.sun.target, this.flashLight, this.fireLight);

    // ---------- terrain: 1 m grid, bilinear heights ----------
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(TR * TR * 3), col = new Float32Array(TR * TR * 3), idx: number[] = [];
    this.tBase = new Float32Array(TR * TR * 3); this.tCell = new Int32Array(TR * TR);
    // distance to permanent water (for sandy banks)
    this.shoreDist = new Float32Array(N * N).fill(99);
    for (let i = 0; i < N * N; i++) if (w.river[i]) this.shoreDist[i] = 0;
    for (let pass = 0; pass < 4; pass++) for (let i = 0; i < N * N; i++) {
      const x = i % N, y = (i / N) | 0; let m = this.shoreDist[i];
      if (x > 0) m = Math.min(m, this.shoreDist[i - 1] + 1); if (x < N - 1) m = Math.min(m, this.shoreDist[i + 1] + 1);
      if (y > 0) m = Math.min(m, this.shoreDist[i - N] + 1); if (y < N - 1) m = Math.min(m, this.shoreDist[i + N] + 1);
      this.shoreDist[i] = m;
    }
    for (let j = 0; j < TR; j++) for (let i = 0; i < TR; i++) {
      const k = j * TR + i, x = -HALF + i * TS, y = -HALF + j * TS;
      const cx = Math.min(N - 1, Math.floor((x + HALF) / CELL)), cy = Math.min(N - 1, Math.floor((y + HALF) / CELL));
      const c = cy * N + cx; this.tCell[k] = c;
      pos[k * 3] = x; pos[k * 3 + 1] = w.heightAt(x, y); pos[k * 3 + 2] = -y;
      // base colour: land + noise + slope rock + sandy banks
      let [r, g, b] = LAND[w.land[c]];
      if (w.land[c] === Land.Water) [r, g, b] = [0.45, 0.40, 0.30];
      const n = (hash(k) - 0.5) * 0.08 + (hash(Math.floor(x / 7) * 31 + Math.floor(y / 7)) - 0.5) * 0.07;
      r += n; g += n * 1.2; b += n * 0.6;
      const [gx, gy] = w.gradAt(x, y), sl = Math.hypot(gx, gy);
      const rock = Math.min(1, Math.max(0, (sl - 0.35) / 0.4));
      r = r * (1 - rock) + 0.48 * rock; g = g * (1 - rock) + 0.45 * rock; b = b * (1 - rock) + 0.42 * rock;
      if (this.shoreDist[c] <= 2 && !w.river[c] && w.height[c] < 1.2) { const s = 1 - this.shoreDist[c] / 3; r = r * (1 - s) + 0.78 * s; g = g * (1 - s) + 0.70 * s; b = b * (1 - s) + 0.50 * s; }
      const hl = Math.min(1, Math.max(0, w.height[c] / 18)); r += hl * 0.06; g += hl * 0.04; b += hl * 0.06;
      this.tBase.set([r, g, b], k * 3);
      if (i < TR - 1 && j < TR - 1) idx.push(k, k + 1, k + TR, k + 1, k + TR + 1, k + TR);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx); geo.computeVertexNormals();
    this.tPos = geo.getAttribute('position') as THREE.BufferAttribute;
    this.tCol = geo.getAttribute('color') as THREE.BufferAttribute;
    this.terrain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 }));
    this.terrain.receiveShadow = true;
    this.scene.add(this.terrain);
    // map border walls (a low dark skirt so the world doesn't end in a knife edge)
    const skirt = new THREE.Mesh(new THREE.BoxGeometry(WORLD_SIZE + 4, 6, WORLD_SIZE + 4), new THREE.MeshStandardMaterial({ color: 0x3b3428, roughness: 1 }));
    skirt.position.y = -3.4; this.scene.add(skirt);

    // ---------- water: one vertex per 2 m cell ----------
    const wgeo = new THREE.BufferGeometry();
    const wp = new Float32Array(N * N * 3), widx: number[] = [];
    for (let cy = 0; cy < N; cy++) for (let cx = 0; cx < N; cx++) {
      const i = cy * N + cx, [x, y] = World.cellCenter(cx, cy);
      wp[i * 3] = x; wp[i * 3 + 1] = -5; wp[i * 3 + 2] = -y;
      if (cx < N - 1 && cy < N - 1) widx.push(i, i + 1, i + N, i + 1, i + N + 1, i + N);
    }
    wgeo.setAttribute('position', new THREE.BufferAttribute(wp, 3));
    wgeo.setAttribute('aDepth', new THREE.BufferAttribute(new Float32Array(N * N).fill(-1), 1));
    wgeo.setAttribute('aFlood', new THREE.BufferAttribute(new Float32Array(N * N), 1));
    wgeo.setIndex(widx);
    this.wPos = wgeo.getAttribute('position') as THREE.BufferAttribute;
    this.wDepth = wgeo.getAttribute('aDepth') as THREE.BufferAttribute;
    this.wFlood = wgeo.getAttribute('aFlood') as THREE.BufferAttribute;
    this.water = new THREE.Mesh(wgeo, this.waterMat); this.water.renderOrder = 2; this.water.frustumCulled = false;
    this.scene.add(this.water);

    // ---------- vegetation: conifers (2 cones) + broadleaf (low-poly crown) ----------
    const nT = Math.max(1, w.trees.length);
    const leaf = (c: number) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.9, flatShading: true });
    this.conLow = new THREE.InstancedMesh(new THREE.ConeGeometry(1.9, 4.2, 7), leaf(0xffffff), nT);
    this.conTop = new THREE.InstancedMesh(new THREE.ConeGeometry(1.25, 3.2, 7), leaf(0xffffff), nT);
    this.broad = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(2.1, 0), leaf(0xffffff), nT);
    this.trunks = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.22, 0.34, 3, 6), new THREE.MeshStandardMaterial({ color: 0x5a3b22, roughness: 1 }), nT);
    this.treeKind = new Uint8Array(w.trees.length); this.treeScale = new Float32Array(w.trees.length);
    w.trees.forEach((t, k) => {
      const forest = w.land[w.cellIndexAt(t.x, t.y)] === Land.Forest;
      this.treeKind[k] = hash(k * 3.1) < (forest ? 0.8 : 0.25) ? 0 : 1;
      this.treeScale[k] = 0.75 + hash(k * 7.7) * 0.55;
    });
    for (const m of [this.conLow, this.conTop, this.broad, this.trunks]) { m.castShadow = true; m.receiveShadow = true; this.scene.add(m); }

    // ---------- buildings ----------
    for (const b of w.buildings) {
      const m = buildBuilding(b.id, b.w, b.d, b.h);
      m.group.position.copy(V(b.x, b.y, w.heightAt(b.x, b.y)));
      this.buildings.push(m); this.scene.add(m.group);
    }

    // ---------- fire, smoke, embers, rain ----------
    const add = THREE.AdditiveBlending;
    this.flamesO = new THREE.InstancedMesh(new THREE.ConeGeometry(1.0, 3.0, 6), new THREE.MeshBasicMaterial({ color: 0xff5a14, transparent: true, opacity: 0.85, blending: add, depthWrite: false }), 4000);
    this.flamesI = new THREE.InstancedMesh(new THREE.ConeGeometry(0.55, 2.0, 6), new THREE.MeshBasicMaterial({ color: 0xffd23a, transparent: true, opacity: 0.9, blending: add, depthWrite: false }), 4000);
    this.flamesO.count = 0; this.flamesI.count = 0;
    const eg = new THREE.BufferGeometry(); eg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(900 * 3), 3));
    this.embers = new THREE.Points(eg, new THREE.PointsMaterial({ color: 0xffa040, size: 0.45, map: softDot(), transparent: true, blending: add, depthWrite: false }));
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3000 * 3), 3));
    sg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(3000 * 3), 3));
    this.smoke = new THREE.Points(sg, new THREE.PointsMaterial({ size: 9, map: softDot(), vertexColors: true, transparent: true, opacity: 0.42, depthWrite: false }));
    this.smoke.frustumCulled = false; this.embers.frustumCulled = false;
    const rg = new THREE.BufferGeometry(); rg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(1600 * 6), 3));
    this.rain = new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ color: 0xaac4dd, transparent: true, opacity: 0.45 }));
    this.rain.frustumCulled = false; this.rain.visible = false;
    for (let k = 0; k < 1600; k++) (this.rain.userData.p ??= []).push([(hash(k) - 0.5) * 80, hash(k * 2.3) * 30, (hash(k * 5.7) - 0.5) * 80]);

    // ---------- debris, rubble, meteors ----------
    this.rubble = new THREE.InstancedMesh(new THREE.DodecahedronGeometry(0.7, 0), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, flatShading: true }), 3000); this.rubble.count = 0;
    this.rubble.castShadow = true;
    this.debris = new THREE.InstancedMesh(new THREE.BoxGeometry(0.6, 0.15, 0.4), new THREE.MeshStandardMaterial({ color: 0x5a4632, roughness: 1 }), 400); this.debris.count = 0;
    this.meteors = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.9, 1), new THREE.MeshBasicMaterial({ color: 0xffe2a0 }), 200); this.meteors.count = 0;
    const tg = new THREE.BufferGeometry(); tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(200 * 6), 3));
    tg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(200 * 6), 3));
    this.trails = new THREE.LineSegments(tg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: add, depthWrite: false }));
    this.trails.frustumCulled = false;
    this.scene.add(this.flamesO, this.flamesI, this.embers, this.smoke, this.rain, this.rubble, this.debris, this.meteors, this.trails, this.craterGroup, this.crackGroup, this.probeGroup);

    // ---------- tornado: three counter-rotating streaked shells + dust ring + debris cloud ----------
    this.tornado = new THREE.Group(); this.funnels = [];
    for (const [rt, rb, op, h] of [[11, 1.4, 0.55, 52], [8, 1.0, 0.4, 50], [14, 2.5, 0.25, 55]] as const) {
      const tex = streakTex().clone(); tex.needsUpdate = true; tex.repeat.set(3, 1);
      const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, 32, 8, true), new THREE.MeshLambertMaterial({ map: tex, color: 0x9a9aa0, transparent: true, opacity: op, side: THREE.DoubleSide, depthWrite: false }));
      m.position.y = h / 2; this.tornado.add(m); this.funnels.push(m);
    }
    const dust = new THREE.Mesh(new THREE.TorusGeometry(7, 2.5, 8, 32), new THREE.MeshLambertMaterial({ color: 0x8b7a60, transparent: true, opacity: 0.45, depthWrite: false }));
    dust.rotation.x = Math.PI / 2; dust.position.y = 1.5; this.tornado.add(dust);
    const dg = new THREE.BufferGeometry(); const dp = new Float32Array(400 * 3);
    for (let k = 0; k < 400; k++) { const a = hash(k) * 6.28, r = 3 + hash(k * 3) * 14; dp.set([Math.cos(a) * r, hash(k * 7) * 20, Math.sin(a) * r], k * 3); }
    dg.setAttribute('position', new THREE.BufferAttribute(dp, 3));
    this.dustCloud = new THREE.Points(dg, new THREE.PointsMaterial({ color: 0x6b5a45, size: 1.6, map: softDot(), transparent: true, opacity: 0.7, depthWrite: false }));
    this.tornado.add(this.dustCloud); this.tornado.visible = false; this.scene.add(this.tornado);

    // ---------- rover ----------
    this.rover = buildRover(); this.scene.add(this.rover.group);

    // ---------- LiDAR: fading rays + hit points ----------
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SENSOR.lidarBeams * 6), 3));
    const lc = new Float32Array(SENSOR.lidarBeams * 6);
    for (let k = 0; k < SENSOR.lidarBeams; k++) lc.set([1, 0.25, 0.35, 0.25, 0.05, 0.1], k * 6);
    lg.setAttribute('color', new THREE.BufferAttribute(lc, 3));
    this.lidar = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6, blending: add, depthWrite: false }));
    const hg = new THREE.BufferGeometry(); hg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SENSOR.lidarBeams * 3), 3));
    this.lidarHits = new THREE.Points(hg, new THREE.PointsMaterial({ color: 0xff3355, size: 0.55, map: softDot(), transparent: true, blending: add, depthWrite: false }));
    this.lidar.frustumCulled = false; this.lidarHits.frustumCulled = false;
    this.scene.add(this.lidar, this.lidarHits);

    // ---------- planned path + goal marker ----------
    const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(600 * 3), 3));
    this.pathLine = new THREE.Line(pg, new THREE.LineBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.9 }));
    this.pathLine.frustumCulled = false; this.scene.add(this.pathLine);
    this.goalMarker = new THREE.Group();
    const gr = new THREE.Mesh(new THREE.TorusGeometry(2.2, 0.2, 8, 32), new THREE.MeshBasicMaterial({ color: 0xc58bff }));
    gr.rotation.x = Math.PI / 2; this.goalMarker.add(gr);
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 30, 12, 1, true), new THREE.MeshBasicMaterial({ color: 0xc58bff, transparent: true, opacity: 0.25, blending: add, depthWrite: false }));
    beam.position.y = 15; this.goalMarker.add(beam);
    this.goalMarker.visible = false; this.scene.add(this.goalMarker);

    this.camYaw = -sim.rover.yaw + Math.PI + 0.5;
    this.camTarget.copy(V(sim.rover.x, sim.rover.y, sim.rover.z));
    this.updateSlow(true, null);
  }

  /** Expensive updates (colours, water, instances), ~4 Hz. */
  private updateSlow(force: boolean, belief: BeliefView | null) {
    const sim = this.sim, w = sim.world, t = sim.t;
    if (!force && t - this.lastSlow < 0.25) return;
    this.lastSlow = t;
    const fog = this.toggles.fog;

    // ----- terrain colours (+ heights after craters) -----
    const col = this.tCol.array as Float32Array, pos = this.tPos.array as Float32Array, base = this.tBase;
    const reshape = w.craters.length !== this.nCraters;
    for (let k = 0; k < TR * TR; k++) {
      const c = this.tCell[k];
      let r = base[k * 3], g = base[k * 3 + 1], b = base[k * 3 + 2];
      if (w.burnt[c]) { const ash = 0.08 + hash(k) * 0.05; r = ash; g = ash * 0.95; b = ash * 0.9; }
      else if (w.burning[c] > 0) { r = 0.35 + 0.5 * w.burning[c]; g = 0.12; b = 0.04; }
      if (w.rubble[c] > 0.2) { r = r * 0.5 + 0.28; g = g * 0.5 + 0.25; b = b * 0.5 + 0.22; }
      if (w.water[c] > 0.02 && !w.river[c]) { r *= 0.6; g *= 0.55; b *= 0.45; } // wet mud under floodwater
      if (this.toggles.hazards && belief) {
        const risk = belief.risk[c], pr = belief.predicted[c];
        if (pr > 0.15) { const a = 0.45 * pr; r = r * (1 - a) + a; g = g * (1 - a) + 0.82 * a; b *= 1 - a; }
        if (risk > 0.1) { r = r * (1 - risk) + risk; g *= 1 - risk * 0.8; b *= 1 - risk * 0.8; }
      }
      if (fog) {
        // soft fog edge: average 'revealed' over the 3x3 cells around this vertex
        const cx = c % N, cy = (c / N) | 0; let rv = 0, cnt = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = cx + dx, Y = cy + dy; if (X >= 0 && Y >= 0 && X < N && Y < N) { rv += sim.revealed[Y * N + X]; cnt++; } }
        const f = rv / cnt, dk = 0.18 + 0.82 * f;
        r = r * dk + 0.03 * (1 - f); g = g * dk + 0.05 * (1 - f); b = b * dk + 0.09 * (1 - f);
      }
      col[k * 3] = r; col[k * 3 + 1] = g; col[k * 3 + 2] = b;
      if (reshape || force) { const x = pos[k * 3], y = -pos[k * 3 + 2]; pos[k * 3 + 1] = w.heightAt(x, y); }
    }
    this.tCol.needsUpdate = true;
    if (reshape || force) { this.tPos.needsUpdate = true; this.terrain.geometry.computeVertexNormals(); }

    // ----- water surface: smooth shoreline via negative depth on dry neighbours -----
    const wp = this.wPos.array as Float32Array, wd = this.wDepth.array as Float32Array, wf = this.wFlood.array as Float32Array;
    for (let i = 0; i < N * N; i++) {
      const hidden = fog && !sim.revealed[i];
      if (w.water[i] > 0.02 && !hidden) {
        wp[i * 3 + 1] = w.height[i] + w.water[i]; wd[i] = w.water[i]; wf[i] = w.river[i] ? 0 : 1; continue;
      }
      let surf = -1e9, fl = 0;
      const cx = i % N, cy = (i / N) | 0;
      if (!hidden) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const X = cx + dx, Y = cy + dy; if (X < 0 || Y < 0 || X >= N || Y >= N) continue;
        const j = Y * N + X; if (w.water[j] > 0.02) { const s = w.height[j] + w.water[j]; if (s > surf) { surf = s; fl = w.river[j] ? 0 : 1; } }
      }
      if (surf > -1e8) { wp[i * 3 + 1] = surf; wd[i] = Math.min(-0.01, surf - w.height[i]); wf[i] = fl; }
      else { wp[i * 3 + 1] = w.height[i] - 2; wd[i] = -1; wf[i] = 0; }
    }
    this.wPos.needsUpdate = true; this.wDepth.needsUpdate = true; this.wFlood.needsUpdate = true;

    // ----- trees -----
    const fogOK = (x: number, y: number) => !fog || sim.revealed[w.cellIndexAt(x, y)] === 1;
    const d = this.dummy, c = new THREE.Color();
    w.trees.forEach((tr, k) => {
      const ci = w.cellIndexAt(tr.x, tr.y);
      const gone = tr.burnt && w.burnt[ci] === 0 && w.burning[ci] === 0; // ripped out by the tornado
      const show = fogOK(tr.x, tr.y) && !gone;
      const s = this.treeScale[k] * (show ? 1 : 0), h = w.heightAt(tr.x, tr.y);
      const kind = this.treeKind[k], burnt = tr.burnt, crown = burnt ? 0.55 : 1;
      const tint = hash(k * 1.7);
      d.rotation.set(0, hash(k) * 6.28, 0);
      d.position.copy(V(tr.x, tr.y, h + 1.5 * s)); d.scale.set(s, s, s); d.updateMatrix(); this.trunks.setMatrixAt(k, d.matrix);
      if (kind === 0) {
        d.position.copy(V(tr.x, tr.y, h + 4.2 * s)); d.scale.setScalar(s * crown); d.updateMatrix(); this.conLow.setMatrixAt(k, d.matrix);
        d.position.copy(V(tr.x, tr.y, h + 6.6 * s)); d.updateMatrix(); this.conTop.setMatrixAt(k, d.matrix);
        d.scale.setScalar(0); d.updateMatrix(); this.broad.setMatrixAt(k, d.matrix);
        c.setRGB(0.13 + tint * 0.06, 0.32 + tint * 0.1, 0.14 + tint * 0.05);
      } else {
        d.position.copy(V(tr.x, tr.y, h + 4.3 * s)); d.scale.set(s * crown, s * crown * 0.85, s * crown); d.updateMatrix(); this.broad.setMatrixAt(k, d.matrix);
        d.scale.setScalar(0); d.updateMatrix(); this.conLow.setMatrixAt(k, d.matrix); this.conTop.setMatrixAt(k, d.matrix);
        c.setRGB(0.28 + tint * 0.12, 0.48 + tint * 0.12, 0.16 + tint * 0.06);
      }
      if (burnt) c.setRGB(0.07, 0.065, 0.06);
      this.conLow.setColorAt(k, c); this.broad.setColorAt(k, c);
      this.conTop.setColorAt(k, burnt ? c : c.clone().offsetHSL(0, 0, 0.05));
    });
    for (const m of [this.conLow, this.conTop, this.broad, this.trunks]) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; }

    // ----- buildings (collapse = squashed, tilted, roof slumped, darker) -----
    w.buildings.forEach((b, k) => {
      const m = this.buildings[k];
      m.group.visible = fogOK(b.x, b.y);
      const s = 1 - 0.7 * b.collapse;
      m.walls.scale.y = s; m.walls.position.y = m.h * s / 2;
      m.roof.position.y = m.h * s + (m.roof.scale.y / 2) * (1 - b.collapse);
      m.roof.rotation.z = b.collapse * 0.5; m.group.rotation.z = b.collapse * 0.12 * ((b.id % 2) * 2 - 1);
      (m.walls.material as THREE.MeshStandardMaterial).emissive.setRGB(0, 0, 0);
      if (b.collapse > 0) (m.walls.material as THREE.MeshStandardMaterial).color.multiplyScalar(1); // keep tint
    });

    // ----- flames, smoke, rubble -----
    let nf = 0;
    for (let i = 0; i < N * N && nf < 4000; i++) {
      if (w.burning[i] <= 0) continue;
      const [x, y] = World.cellCenter(i % N, (i / N) | 0);
      const s = 0.45 + w.burning[i] * (0.8 + 0.35 * hash(i + Math.floor(t * 8)));
      d.rotation.set(0, hash(i) * 6, 0);
      d.position.copy(V(x + (hash(i) - 0.5), y + (hash(i * 3) - 0.5), w.height[i] + 1.4 * s)); d.scale.set(s, s * (0.9 + 0.5 * hash(i * 7 + Math.floor(t * 10))), s); d.updateMatrix();
      this.flamesO.setMatrixAt(nf, d.matrix);
      d.scale.multiplyScalar(0.8); d.position.y -= 0.3 * s; d.updateMatrix(); this.flamesI.setMatrixAt(nf, d.matrix);
      nf++;
    }
    this.flamesO.count = nf; this.flamesI.count = nf;
    this.flamesO.instanceMatrix.needsUpdate = true; this.flamesI.instanceMatrix.needsUpdate = true;
    const sp = this.smoke.geometry.getAttribute('position') as THREE.BufferAttribute, sc = this.smoke.geometry.getAttribute('color') as THREE.BufferAttribute;
    let ns = 0;
    for (let cy = 0; cy < N; cy += 2) for (let cx = 0; cx < N; cx += 2) {
      const i = cy * N + cx, s = w.smoke[i]; if (s < 0.06 || ns >= 3000) continue;
      const [x, y] = World.cellCenter(cx, cy);
      for (let layer = 0; layer < 2 && ns < 3000; layer++) {
        const jx = (hash(i + layer * 17) - 0.5) * 3, jy = (hash(i * 5 + layer) - 0.5) * 3;
        sp.setXYZ(ns, x + jx, w.height[i] + 4 + layer * 6 + 5 * s + Math.sin(t * 0.5 + i) * 0.8, -(y + jy));
        const g = 0.16 + 0.14 * (1 - Math.min(1, s)) + layer * 0.06;
        sc.setXYZ(ns, g, g * 0.97, g * 0.93); ns++;
      }
    }
    this.smoke.geometry.setDrawRange(0, ns); sp.needsUpdate = true; sc.needsUpdate = true;
    let nr = 0;
    for (let i = 0; i < N * N && nr < 3000; i++) {
      if (w.rubble[i] < 0.3) continue;
      const [x, y] = World.cellCenter(i % N, (i / N) | 0);
      for (let k = 0; k < 3 && nr < 3000; k++) {
        d.position.copy(V(x + (hash(i * 3.1 + k) - 0.5) * 1.8, y + (hash(i * 1.7 + k) - 0.5) * 1.8, w.height[i] + 0.25)); d.scale.setScalar(0.4 + 0.7 * w.rubble[i] * hash(i + k)); d.rotation.set(i, k, i * k); d.updateMatrix();
        this.rubble.setMatrixAt(nr, d.matrix); this.rubble.setColorAt(nr, c.setHSL(0.08, 0.1, 0.35 + hash(i * k) * 0.2)); nr++;
      }
    }
    d.rotation.set(0, 0, 0);
    this.rubble.count = nr; this.rubble.instanceMatrix.needsUpdate = true; if (this.rubble.instanceColor) this.rubble.instanceColor.needsUpdate = true;

    // ----- craters (fresh ones glow), cracks, probes -----
    for (; this.nCraters < w.craters.length; this.nCraters++) {
      const cr = w.craters[this.nCraters], h = w.heightAt(cr.x, cr.y);
      const disc = new THREE.Mesh(new THREE.CircleGeometry(cr.r + 0.5, 20), new THREE.MeshStandardMaterial({ color: 0x241d18, emissive: 0xff5a10, emissiveIntensity: 2, roughness: 1 }));
      disc.rotation.x = -Math.PI / 2; disc.position.copy(V(cr.x, cr.y, h + 0.06)); this.craterGroup.add(disc);
      this.hotCraters.push({ mesh: disc, t: cr.t });
      const rim = new THREE.Mesh(new THREE.TorusGeometry(cr.r + 0.7, 0.55, 6, 20), new THREE.MeshStandardMaterial({ color: 0x5b4c40, roughness: 1, flatShading: true }));
      rim.rotation.x = Math.PI / 2; rim.position.copy(disc.position); rim.castShadow = true; this.craterGroup.add(rim);
    }
    for (; this.nCracks < w.cracks.length; this.nCracks++) {
      const ck = w.cracks[this.nCracks]; const L = Math.hypot(ck.x1 - ck.x0, ck.y1 - ck.y0);
      const m = new THREE.Mesh(new THREE.BoxGeometry(L, 0.2, ck.w), new THREE.MeshBasicMaterial({ color: 0x0d0907 }));
      const mx = (ck.x0 + ck.x1) / 2, my = (ck.y0 + ck.y1) / 2;
      m.position.copy(V(mx, my, w.heightAt(mx, my) + 0.08)); m.rotation.y = Math.atan2(ck.y1 - ck.y0, ck.x1 - ck.x0); this.crackGroup.add(m);
      const lip = new THREE.Mesh(new THREE.BoxGeometry(L, 0.25, ck.w + 0.6), new THREE.MeshStandardMaterial({ color: 0x6a5846, roughness: 1 }));
      lip.position.copy(m.position); lip.position.y -= 0.05; lip.rotation.y = m.rotation.y; this.crackGroup.add(lip);
    }
    if (sim.probes.length !== this.nProbes || force) {
      this.nProbes = sim.probes.length;
      this.probeGroup.clear(); this.probes = [];
      for (const p of sim.probes) {
        const pm = buildProbe(p.id);
        const ring = new THREE.Mesh(new THREE.RingGeometry(SENSOR.probeRadius - 0.25, SENSOR.probeRadius, 64), new THREE.MeshBasicMaterial({ color: 0x33ff88, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }));
        ring.rotation.x = -Math.PI / 2; ring.position.y = 0.4; ring.name = 'ring';
        const pulse = new THREE.Mesh(new THREE.RingGeometry(0.8, 1.0, 32), new THREE.MeshBasicMaterial({ color: 0x33ff88, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false }));
        pulse.rotation.x = -Math.PI / 2; pulse.position.y = 0.45; pulse.name = 'pulse';
        pm.group.add(ring, pulse);
        pm.group.position.copy(V(p.x, p.y, w.heightAt(p.x, p.y))); pm.group.userData.id = p.id;
        this.probeGroup.add(pm.group); this.probes.push(pm);
      }
    }
  }

  render(belief: BeliefView | null, path: { x: number; y: number }[] | null, goal: { x: number; y: number } | null, behaviour = 'explore') {
    const sim = this.sim; if (!sim) return;
    const w = sim.world, r = sim.rover, t = sim.t;
    this.frameT += 1 / 60;
    const ft = this.frameT;
    this.updateSlow(false, belief);
    this.waterMat.uniforms.uTime.value = ft;

    // rover
    this.rover.group.position.copy(V(r.x, r.y, r.z));
    this.rover.group.rotation.set(0, 0, 0);
    this.rover.group.rotateY(r.yaw); this.rover.group.rotateX(-r.roll); this.rover.group.rotateZ(r.pitch);
    if (!r.alive) { this.rover.group.rotateX(Math.PI * 0.55); }
    this.rover.update(ft, r.v, r.w, r.probesLeft, BEHAVIOUR_COLOR[behaviour] ?? 0x33ff88, r.alive);
    sim.probes.forEach((p, k) => {
      const pm = this.probes[k]; if (!pm) return;
      pm.update(ft, p.alive);
      const ring = pm.group.getObjectByName('ring')!; ring.visible = p.alive;
      const pulse = pm.group.getObjectByName('pulse') as THREE.Mesh;
      const ph = (ft * 0.35 + p.id * 0.3) % 1;
      pulse.visible = p.alive; pulse.scale.setScalar(1 + ph * (SENSOR.probeRadius - 1)); (pulse.material as THREE.MeshBasicMaterial).opacity = 0.45 * (1 - ph);
    });

    // LiDAR rays + hit points
    const f = sim.lastFrame;
    const showL = this.toggles.lidar && !!f;
    this.lidar.visible = showL; this.lidarHits.visible = showL;
    if (f && showL) {
      const a = this.lidar.geometry.getAttribute('position') as THREE.BufferAttribute, arr = a.array as Float32Array;
      const hp = this.lidarHits.geometry.getAttribute('position') as THREE.BufferAttribute;
      const sz = r.z + 1.15;
      let nh = 0;
      f.scan.ranges.forEach((rg, k) => {
        const ang = r.yaw + f.scan.angle_min + k * f.scan.angle_increment, dd = Math.min(rg, f.scan.range_max);
        const ex = r.x + dd * Math.cos(ang), ey = r.y + dd * Math.sin(ang);
        arr.set([r.x, sz, -r.y, ex, sz, -ey], k * 6);
        if (rg <= f.scan.range_max) hp.setXYZ(nh++, ex, sz, -ey);
      });
      a.needsUpdate = true; hp.needsUpdate = true; this.lidarHits.geometry.setDrawRange(0, nh);
    }
    // path + goal
    this.pathLine.visible = this.toggles.path && !!path && path.length > 1;
    if (path && this.pathLine.visible) {
      const a = this.pathLine.geometry.getAttribute('position') as THREE.BufferAttribute, arr = a.array as Float32Array;
      const n = Math.min(path.length, 600);
      for (let k = 0; k < n; k++) arr.set([path[k].x, w.heightAt(path[k].x, path[k].y) + 0.5, -path[k].y], k * 3);
      a.needsUpdate = true; this.pathLine.geometry.setDrawRange(0, n);
    }
    this.goalMarker.visible = !!goal;
    if (goal) { this.goalMarker.position.copy(V(goal.x, goal.y, w.heightAt(goal.x, goal.y) + 0.5)); this.goalMarker.rotation.y = ft; }

    // fire light + embers
    let fx = 0, fy = 0, fn = 0;
    for (let i = 0; i < N * N; i += 3) if (w.burning[i] > 0.5) { const [x, y] = World.cellCenter(i % N, (i / N) | 0); if (Math.hypot(x - r.x, y - r.y) < 70) { fx += x; fy += y; fn++; } }
    this.fireLight.intensity = fn ? Math.min(900, 120 + fn * 18) * (0.85 + 0.15 * Math.sin(ft * 17)) : 0;
    if (fn) this.fireLight.position.copy(V(fx / fn, fy / fn, w.heightAt(fx / fn, fy / fn) + 6));
    if (this.flamesO.count && this.emberData.length < 900) for (let k = 0; k < 6; k++) {
      const i = Math.floor(Math.random() * N * N); if (w.burning[i] <= 0) continue;
      const [x, y] = World.cellCenter(i % N, (i / N) | 0);
      this.emberData.push({ x: x + (Math.random() - 0.5) * 2, y: y + (Math.random() - 0.5) * 2, z: w.height[i] + 2, vz: 2 + Math.random() * 3, life: 2 + Math.random() * 2 });
    }
    const ep = this.embers.geometry.getAttribute('position') as THREE.BufferAttribute;
    const [wx, wy] = [sim.dis.windX, sim.dis.windY];
    this.emberData = this.emberData.filter(e => (e.life -= 1 / 60) > 0);
    this.emberData.forEach((e, k) => { e.z += e.vz / 60; e.x += wx * 0.3 / 60; e.y += wy * 0.3 / 60; ep.setXYZ(k, e.x + Math.sin(ft * 3 + k) * 0.2, e.z, -e.y); });
    ep.needsUpdate = true; this.embers.geometry.setDrawRange(0, this.emberData.length);

    // rain while a flood is underway (storm)
    const storm = sim.dis.flood.active ? Math.min(1, (t - sim.dis.flood.t0) / 8) : 0;
    this.rain.visible = storm > 0.05;
    if (this.rain.visible) {
      const rp = this.rain.geometry.getAttribute('position') as THREE.BufferAttribute, P: number[][] = this.rain.userData.p;
      const cx = this.camTarget.x, cz = this.camTarget.z, cy0 = this.camTarget.y;
      P.forEach((p, k) => {
        p[1] -= 0.55; if (p[1] < -5) p[1] += 35;
        const x = cx + p[0], y = cy0 + p[1], z = cz + p[2];
        rp.setXYZ(k * 2, x, y, z); rp.setXYZ(k * 2 + 1, x + 0.05, y - 0.9, z + 0.05);
      });
      rp.needsUpdate = true; (this.rain.material as THREE.LineBasicMaterial).opacity = 0.45 * storm;
    }
    this.waterMat.uniforms.uRain.value = storm;

    // tornado
    const tor = sim.dis.tornadoes.find(tt => tt.active);
    this.tornado.visible = !!tor;
    if (tor) {
      this.tornado.position.copy(V(tor.x, tor.y, w.heightAt(tor.x, tor.y)));
      this.tornado.scale.set(tor.R / 6, 1, tor.R / 6);
      this.funnels.forEach((m, k) => { m.rotation.y = ft * (2.5 + k) * (k === 1 ? -1 : 1); ((m.material as THREE.MeshLambertMaterial).map as THREE.Texture).offset.y = ft * 0.4 * (k + 1); });
      this.dustCloud.rotation.y = ft * 2.2;
    }
    const dm = this.dummy;
    this.debris.count = Math.min(400, sim.dis.debris.length);
    sim.dis.debris.slice(0, 400).forEach((db, k) => { dm.position.copy(V(db.x, db.y, w.heightAt(db.x, db.y) + db.z)); dm.rotation.set(ft * 5 + k, ft * 3, k); dm.scale.setScalar(1); dm.updateMatrix(); this.debris.setMatrixAt(k, dm.matrix); });
    this.debris.instanceMatrix.needsUpdate = true;

    // meteors with glowing trails
    const ms = sim.dis.meteors.slice(0, 200);
    this.meteors.count = ms.length;
    const tp = this.trails.geometry.getAttribute('position') as THREE.BufferAttribute, tc = this.trails.geometry.getAttribute('color') as THREE.BufferAttribute;
    ms.forEach((m, k) => {
      const ttl = Math.max(0, m.tImpact - t), hgt = ttl * 30;
      const p = V(m.x + ttl * 12, m.y + ttl * 5, w.heightAt(m.x, m.y) + hgt);
      dm.position.copy(p); dm.rotation.set(0, 0, 0); dm.scale.setScalar(0.8 + ttl * 0.3); dm.updateMatrix(); this.meteors.setMatrixAt(k, dm.matrix);
      const back = V(m.x + (ttl + 0.6) * 12, m.y + (ttl + 0.6) * 5, w.heightAt(m.x, m.y) + hgt + 18);
      tp.setXYZ(k * 2, p.x, p.y, p.z); tp.setXYZ(k * 2 + 1, back.x, back.y, back.z);
      tc.setXYZ(k * 2, 1, 0.85, 0.5); tc.setXYZ(k * 2 + 1, 0.2, 0.05, 0);
    });
    this.meteors.instanceMatrix.needsUpdate = true; tp.needsUpdate = true; tc.needsUpdate = true; this.trails.geometry.setDrawRange(0, ms.length * 2);
    for (const hc of this.hotCraters) { const age = t - hc.t; (hc.mesh.material as THREE.MeshStandardMaterial).emissiveIntensity = Math.max(0, 2.5 * Math.exp(-age / 6)); }
    const fl = sim.dis.flashes[sim.dis.flashes.length - 1];
    if (fl && t - fl.t < 0.6) { this.flashLight.position.copy(V(fl.x, fl.y, w.heightAt(fl.x, fl.y) + 4)); this.flashLight.intensity = 5000 * (1 - (t - fl.t) / 0.6); } else this.flashLight.intensity = 0;

    // atmosphere: smoke haze + storm darkening
    const sm = Math.min(1, w.smoke[w.cellIndexAt(r.x, r.y)] * 1.5);
    this.sky.set(sm * 0.85, storm * 0.7);
    const fogCol = new THREE.Color(0xb8cde0).lerp(new THREE.Color(0x7d8690), storm * 0.7).lerp(new THREE.Color(0x6e5a48), sm);
    const fogObj = this.scene.fog as THREE.Fog;
    fogObj.color.copy(fogCol); fogObj.near = 160 * (1 - Math.min(0.85, sm + storm * 0.3)); fogObj.far = 420 * (1 - Math.min(0.75, sm * 0.9 + storm * 0.35));
    this.sun.intensity = 2.4 * (1 - 0.55 * storm) * (1 - 0.5 * sm);
    this.hemi.intensity = 1.0 * (1 - 0.3 * storm);

    // follow camera (orbit around a smoothed target); sun + shadow box follow the rover
    const target = V(r.x, r.y, r.z + 1.5);
    this.camTarget.lerp(target, 0.12);
    const cy = this.camYaw, cp = this.camPitch, dist = this.camDist;
    this.camera.position.set(this.camTarget.x + dist * Math.cos(cp) * Math.cos(cy), this.camTarget.y + dist * Math.sin(cp), this.camTarget.z + dist * Math.cos(cp) * Math.sin(cy));
    const ground = w.inBounds(this.camera.position.x, -this.camera.position.z) ? w.heightAt(this.camera.position.x, -this.camera.position.z) + 1.5 : -10;
    if (this.camera.position.y < ground) this.camera.position.y = ground;
    this.camera.lookAt(this.camTarget);
    this.sun.position.set(this.camTarget.x + 60, this.camTarget.y + 110, this.camTarget.z + 40);
    this.sun.target.position.copy(this.camTarget); this.sun.target.updateMatrixWorld();
    this.sky.mesh.position.copy(this.camera.position);

    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, W, H);
    this.renderer.render(this.scene, this.camera);
    const pip = document.getElementById('pip');
    if (this.toggles.camera) {
      const cw = Math.round(Math.min(320, W * 0.28)), ch = Math.round(cw * 0.75);
      this.roverCam.position.copy(V(r.x + 0.6 * Math.cos(r.yaw), r.y + 0.6 * Math.sin(r.yaw), r.z + 2.05));
      this.roverCam.lookAt(V(r.x + 10 * Math.cos(r.yaw), r.y + 10 * Math.sin(r.yaw), r.z + 1.0));
      this.lidar.visible = false; this.lidarHits.visible = false; this.rover.group.visible = false;
      this.sky.mesh.position.copy(this.roverCam.position);
      this.renderer.setScissorTest(true);
      this.renderer.setViewport(W - cw - 14, 14, cw, ch); this.renderer.setScissor(W - cw - 14, 14, cw, ch);
      this.renderer.render(this.scene, this.roverCam);
      this.renderer.setScissorTest(false);
      this.rover.group.visible = true; this.lidar.visible = showL; this.lidarHits.visible = showL;
      if (pip) { pip.style.display = 'block'; pip.style.width = cw + 'px'; pip.style.height = ch + 'px'; pip.style.right = '14px'; pip.style.bottom = '14px'; }
    } else if (pip) pip.style.display = 'none';
  }
}

const WORLD_SIZE = HALF * 2;
