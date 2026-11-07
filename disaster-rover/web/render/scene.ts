import * as THREE from 'three';
import { Sim } from '../sim/sim';
import { World, N, CELL, HALF, Land } from '../sim/world';
import { SENSOR } from '../sim/sensors';
import type { BeliefView } from '../brain/view';

/** sim (x, y, z-up) -> three (x, y-up, -y) */
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y);

const LAND_COLORS: Record<number, [number, number, number]> = {
  [Land.Grass]: [0.42, 0.62, 0.30], [Land.Field]: [0.70, 0.66, 0.38], [Land.Forest]: [0.22, 0.42, 0.20],
  [Land.Town]: [0.55, 0.53, 0.50], [Land.Water]: [0.35, 0.35, 0.30], [Land.Rock]: [0.50, 0.47, 0.44], [Land.Road]: [0.32, 0.32, 0.34],
};

export interface RenderToggles { lidar: boolean; hazards: boolean; fog: boolean; camera: boolean; path: boolean }

export class SceneView {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  roverCam: THREE.PerspectiveCamera;
  toggles: RenderToggles = { lidar: true, hazards: false, fog: true, camera: true, path: true };
  private sim!: Sim;
  private terrain!: THREE.Mesh; private terrainColors!: THREE.BufferAttribute; private terrainPos!: THREE.BufferAttribute;
  private water!: THREE.Mesh; private waterPos!: THREE.BufferAttribute;
  private trees!: THREE.InstancedMesh; private trunks!: THREE.InstancedMesh;
  private buildings: THREE.Mesh[] = [];
  private flames!: THREE.InstancedMesh; private smoke!: THREE.InstancedMesh; private rubble!: THREE.InstancedMesh;
  private debris!: THREE.InstancedMesh; private meteors!: THREE.InstancedMesh;
  private craterGroup = new THREE.Group(); private crackGroup = new THREE.Group(); private probeGroup = new THREE.Group();
  private tornadoGroup = new THREE.Group();
  private rover = new THREE.Group();
  private lidar!: THREE.LineSegments;
  private pathLine!: THREE.Line; private goalMarker!: THREE.Mesh;
  private flashLight = new THREE.PointLight(0xffcc88, 0, 120);
  private camYaw = 0; private camPitch = 0.62; private camDist = 34;
  private camTarget = new THREE.Vector3();
  private lastSlow = -1; private nCraters = 0; private nCracks = 0; private nProbes = -1;
  private dummy = new THREE.Object3D();

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.shadowMap.enabled = false;
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.5, 600);
    this.roverCam = new THREE.PerspectiveCamera(70, 4 / 3, 0.3, 200);
    this.scene.background = new THREE.Color(0x9cc4e4);
    this.scene.fog = new THREE.Fog(0x9cc4e4, 120, 320);
    this.scene.add(new THREE.HemisphereLight(0xdfefff, 0x445533, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(60, 120, 40); this.scene.add(sun);
    this.scene.add(this.flashLight);
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
    this.canvas.addEventListener('wheel', e => { this.camDist = Math.max(6, Math.min(160, this.camDist * (1 + Math.sign(e.deltaY) * 0.1))); e.preventDefault(); }, { passive: false });
  }

  resize() {
    const w = this.canvas.clientWidth || 800, h = this.canvas.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  /** (Re)build every mesh for a new sim. */
  load(sim: Sim) {
    this.sim = sim;
    for (const o of [...this.scene.children]) if (!(o instanceof THREE.Light)) this.scene.remove(o);
    this.buildings = []; this.lastSlow = -1; this.nCraters = 0; this.nCracks = 0; this.nProbes = -1;
    this.craterGroup = new THREE.Group(); this.crackGroup = new THREE.Group(); this.probeGroup = new THREE.Group(); this.tornadoGroup = new THREE.Group();
    const w = sim.world;

    // terrain grid: one vertex per cell centre
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(N * N * 3), col = new Float32Array(N * N * 3), idx: number[] = [];
    for (let cy = 0; cy < N; cy++) for (let cx = 0; cx < N; cx++) {
      const i = cy * N + cx, [x, y] = World.cellCenter(cx, cy);
      pos[i * 3] = x; pos[i * 3 + 1] = w.height[i]; pos[i * 3 + 2] = -y;
      if (cx < N - 1 && cy < N - 1) idx.push(i, i + 1, i + N, i + 1, i + N + 1, i + N);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx); geo.computeVertexNormals();
    this.terrainPos = geo.getAttribute('position') as THREE.BufferAttribute;
    this.terrainColors = geo.getAttribute('color') as THREE.BufferAttribute;
    this.terrain = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }));
    this.scene.add(this.terrain);

    const wgeo = geo.clone();
    wgeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    this.waterPos = wgeo.getAttribute('position') as THREE.BufferAttribute;
    this.water = new THREE.Mesh(wgeo, new THREE.MeshPhongMaterial({ color: 0x2f6fa8, transparent: true, opacity: 0.78, shininess: 80 }));
    this.scene.add(this.water);

    // trees
    const nT = w.trees.length;
    this.trees = new THREE.InstancedMesh(new THREE.ConeGeometry(1.8, 6, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }), Math.max(1, nT));
    this.trunks = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.25, 0.35, 2.5, 5), new THREE.MeshLambertMaterial({ color: 0x5a3b22 }), Math.max(1, nT));
    this.trees.count = nT; this.trunks.count = nT;
    this.scene.add(this.trees, this.trunks);

    for (const b of w.buildings) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(b.w, b.h, b.d), new THREE.MeshLambertMaterial({ color: new THREE.Color().setHSL(0.08, 0.15, 0.55 + (b.id % 4) * 0.06) }));
      m.position.copy(V(b.x, b.y, w.heightAt(b.x, b.y) + b.h / 2));
      this.buildings.push(m); this.scene.add(m);
    }

    const flameGeo = new THREE.ConeGeometry(0.9, 2.6, 5);
    this.flames = new THREE.InstancedMesh(flameGeo, new THREE.MeshBasicMaterial({ color: 0xff6a1a }), 4000); this.flames.count = 0;
    this.smoke = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(2.4, 0), new THREE.MeshLambertMaterial({ color: 0x555555, transparent: true, opacity: 0.35, depthWrite: false }), 2500); this.smoke.count = 0;
    this.rubble = new THREE.InstancedMesh(new THREE.DodecahedronGeometry(0.7, 0), new THREE.MeshLambertMaterial({ color: 0x7a6f66 }), 3000); this.rubble.count = 0;
    this.debris = new THREE.InstancedMesh(new THREE.BoxGeometry(0.5, 0.3, 0.4), new THREE.MeshLambertMaterial({ color: 0x4a3a2a }), 400); this.debris.count = 0;
    this.meteors = new THREE.InstancedMesh(new THREE.SphereGeometry(0.9, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffd27a }), 200); this.meteors.count = 0;
    this.scene.add(this.flames, this.smoke, this.rubble, this.debris, this.meteors, this.craterGroup, this.crackGroup, this.probeGroup, this.tornadoGroup);

    // tornado funnel
    const fun = new THREE.Mesh(new THREE.CylinderGeometry(9, 1.5, 45, 20, 8, true), new THREE.MeshLambertMaterial({ color: 0x6b6b70, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
    fun.position.y = 22; this.tornadoGroup.add(fun); this.tornadoGroup.visible = false;

    // rover
    this.rover = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.6, 1.2), new THREE.MeshLambertMaterial({ color: 0xf0b030 }));
    body.position.y = 0.6; this.rover.add(body);
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.9), new THREE.MeshLambertMaterial({ color: 0x333333 })); mast.position.set(0.5, 1.3, 0); this.rover.add(mast);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.2, 0.4), new THREE.MeshLambertMaterial({ color: 0x222222 })); head.position.set(0.55, 1.8, 0); this.rover.add(head);
    const lid = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.15, 12), new THREE.MeshLambertMaterial({ color: 0x1144aa })); lid.position.set(0, 1.0, 0); this.rover.add(lid);
    for (const [wx, wz] of [[0.6, 0.7], [-0.6, 0.7], [0.6, -0.7], [-0.6, -0.7]]) {
      const wh = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 0.25, 12), new THREE.MeshLambertMaterial({ color: 0x222222 }));
      wh.rotation.x = Math.PI / 2; wh.position.set(wx, 0.35, wz); this.rover.add(wh);
    }
    this.scene.add(this.rover);

    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SENSOR.lidarBeams * 6), 3));
    this.lidar = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xff2255, transparent: true, opacity: 0.55 }));
    this.scene.add(this.lidar);
    const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(600 * 3), 3));
    this.pathLine = new THREE.Line(pg, new THREE.LineBasicMaterial({ color: 0x00e5ff })); this.scene.add(this.pathLine);
    this.goalMarker = new THREE.Mesh(new THREE.TorusGeometry(2.2, 0.25, 6, 24), new THREE.MeshBasicMaterial({ color: 0xb04dff }));
    this.goalMarker.rotation.x = Math.PI / 2; this.goalMarker.visible = false; this.scene.add(this.goalMarker);
    this.scene.add(this.flashLight);
    this.camYaw = -sim.rover.yaw + Math.PI;
    this.updateSlow(true, null);
  }

  /** Expensive updates (colours, instances), ~4 Hz. */
  private updateSlow(force: boolean, belief: BeliefView | null) {
    const sim = this.sim, w = sim.world, t = sim.t;
    if (!force && t - this.lastSlow < 0.25) return;
    this.lastSlow = t;
    const col = this.terrainColors.array as Float32Array, pos = this.terrainPos.array as Float32Array, wp = this.waterPos.array as Float32Array;
    for (let i = 0; i < N * N; i++) {
      let [r, g, b] = LAND_COLORS[w.land[i]];
      if (w.burnt[i]) { r = 0.12; g = 0.11; b = 0.10; }
      else if (w.burning[i] > 0) { r = 0.9; g = 0.35; b = 0.08; }
      if (w.rubble[i] > 0.2) { r = r * 0.6 + 0.25; g = g * 0.6 + 0.22; b = b * 0.6 + 0.2; }
      if (this.toggles.hazards && belief) {
        const bi = belief.cellOfWorld(i);
        const risk = belief.risk[bi], pr = belief.predicted[bi];
        if (pr > 0.15) { r = r * 0.5 + 0.5 * 1; g = g * 0.5 + 0.5 * 0.8; b *= 0.5; }
        if (risk > 0.1) { r = r * (1 - risk) + risk; g *= 1 - risk * 0.8; b *= 1 - risk * 0.8; }
      }
      if (this.toggles.fog && !sim.revealed[i]) { r *= 0.22; g *= 0.24; b *= 0.3; }
      col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = b;
      pos[i * 3 + 1] = w.height[i];
      const hidden = this.toggles.fog && !sim.revealed[i];
      wp[i * 3 + 1] = w.water[i] > 0.02 && !hidden ? w.height[i] + w.water[i] : w.height[i] - 0.6;
    }
    this.terrainColors.needsUpdate = true; this.terrainPos.needsUpdate = true; this.waterPos.needsUpdate = true;
    if (w.craters.length !== this.nCraters) this.terrain.geometry.computeVertexNormals();
    (this.water.material as THREE.MeshPhongMaterial).color.setHex(w.stage > 0.3 ? 0x5a6a50 : 0x2f6fa8);

    const fogOK = (x: number, y: number) => !this.toggles.fog || sim.revealed[w.cellIndexAt(x, y)] === 1;
    const d = this.dummy, c = new THREE.Color();
    w.trees.forEach((tr, k) => {
      const show = fogOK(tr.x, tr.y);
      const gone = tr.burnt && w.burnt[w.cellIndexAt(tr.x, tr.y)] === 0 && w.burning[w.cellIndexAt(tr.x, tr.y)] === 0; // ripped out by tornado
      const h = w.heightAt(tr.x, tr.y);
      d.position.copy(V(tr.x, tr.y, h + 5)); d.scale.setScalar(show && !gone ? (tr.burnt ? 0.6 : 1) : 0); d.rotation.set(0, 0, 0); d.updateMatrix();
      this.trees.setMatrixAt(k, d.matrix);
      this.trees.setColorAt(k, c.setHex(tr.burnt ? 0x1c1a18 : 0x2f6a2c));
      d.position.copy(V(tr.x, tr.y, h + 1.2)); d.scale.setScalar(show && !gone ? 1 : 0); d.updateMatrix(); this.trunks.setMatrixAt(k, d.matrix);
    });
    this.trees.instanceMatrix.needsUpdate = true; if (this.trees.instanceColor) this.trees.instanceColor.needsUpdate = true; this.trunks.instanceMatrix.needsUpdate = true;

    w.buildings.forEach((b, k) => {
      const m = this.buildings[k];
      m.visible = fogOK(b.x, b.y);
      const s = 1 - 0.7 * b.collapse;
      m.scale.set(1, s, 1); m.position.y = w.heightAt(b.x, b.y) + b.h * s / 2;
      m.rotation.z = b.collapse * 0.25 * ((b.id % 2) * 2 - 1);
    });

    let nf = 0, ns = 0, nr = 0;
    for (let i = 0; i < N * N && nf < 4000; i++) {
      const [x, y] = World.cellCenter(i % N, (i / N) | 0);
      if (w.burning[i] > 0) {
        const s = 0.5 + w.burning[i] * (0.8 + 0.4 * Math.sin(t * 9 + i));
        d.position.copy(V(x, y, w.height[i] + 1.2 * s)); d.scale.set(s, s * (1 + 0.3 * Math.sin(t * 13 + i * 0.7)), s); d.updateMatrix();
        this.flames.setMatrixAt(nf++, d.matrix);
      }
    }
    for (let cy = 0; cy < N; cy += 2) for (let cx = 0; cx < N; cx += 2) {
      const i = cy * N + cx; const s = w.smoke[i]; if (s < 0.08 || ns >= 2500) continue;
      const [x, y] = World.cellCenter(cx, cy);
      d.position.copy(V(x, y, w.height[i] + 5 + 6 * s + Math.sin(i) * 2)); d.scale.setScalar(0.6 + 1.6 * s); d.updateMatrix();
      this.smoke.setMatrixAt(ns++, d.matrix);
    }
    for (let i = 0; i < N * N && nr < 3000; i++) {
      if (w.rubble[i] < 0.3) continue;
      const [x, y] = World.cellCenter(i % N, (i / N) | 0);
      for (let k = 0; k < 2 && nr < 3000; k++) {
        d.position.copy(V(x + Math.sin(i * 3.1 + k) * 0.8, y + Math.cos(i * 1.7 + k) * 0.8, w.height[i] + 0.3)); d.scale.setScalar(0.6 + 0.6 * w.rubble[i]); d.rotation.set(i, k, i * k); d.updateMatrix();
        this.rubble.setMatrixAt(nr++, d.matrix);
      }
    }
    d.rotation.set(0, 0, 0);
    this.flames.count = nf; this.smoke.count = ns; this.rubble.count = nr;
    this.flames.instanceMatrix.needsUpdate = true; this.smoke.instanceMatrix.needsUpdate = true; this.rubble.instanceMatrix.needsUpdate = true;

    for (; this.nCraters < w.craters.length; this.nCraters++) {
      const cr = w.craters[this.nCraters];
      const m = new THREE.Mesh(new THREE.CircleGeometry(cr.r + 0.6, 16), new THREE.MeshLambertMaterial({ color: 0x2a2420 }));
      m.rotation.x = -Math.PI / 2; m.position.copy(V(cr.x, cr.y, w.heightAt(cr.x, cr.y) + 0.08)); this.craterGroup.add(m);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(cr.r + 0.7, 0.45, 5, 16), new THREE.MeshLambertMaterial({ color: 0x5b4c40 }));
      ring.rotation.x = Math.PI / 2; ring.position.copy(m.position); this.craterGroup.add(ring);
    }
    for (; this.nCracks < w.cracks.length; this.nCracks++) {
      const ck = w.cracks[this.nCracks]; const L = Math.hypot(ck.x1 - ck.x0, ck.y1 - ck.y0);
      const m = new THREE.Mesh(new THREE.BoxGeometry(L, 0.15, ck.w), new THREE.MeshBasicMaterial({ color: 0x110d0a }));
      const mx = (ck.x0 + ck.x1) / 2, my = (ck.y0 + ck.y1) / 2;
      m.position.copy(V(mx, my, w.heightAt(mx, my) + 0.1)); m.rotation.y = Math.atan2(ck.y1 - ck.y0, ck.x1 - ck.x0); this.crackGroup.add(m);
    }
    if (sim.probes.length !== this.nProbes || force) {
      this.nProbes = sim.probes.length;
      this.probeGroup.clear();
      for (const p of sim.probes) {
        const g = new THREE.Group();
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 2.2), new THREE.MeshLambertMaterial({ color: 0xdddddd })); pole.position.y = 1.1; g.add(pole);
        const top = new THREE.Mesh(new THREE.SphereGeometry(0.32, 10, 8), new THREE.MeshBasicMaterial({ color: 0x33ff88 })); top.position.y = 2.3; top.name = 'top'; g.add(top);
        const ring = new THREE.Mesh(new THREE.RingGeometry(SENSOR.probeRadius - 0.3, SENSOR.probeRadius, 48), new THREE.MeshBasicMaterial({ color: 0x33ff88, transparent: true, opacity: 0.35, side: THREE.DoubleSide }));
        ring.rotation.x = -Math.PI / 2; ring.position.y = 0.3; ring.name = 'ring'; g.add(ring);
        g.position.copy(V(p.x, p.y, w.heightAt(p.x, p.y))); g.userData.id = p.id; this.probeGroup.add(g);
      }
    }
    for (const g of this.probeGroup.children) {
      const p = sim.probes[g.userData.id]; if (!p) continue;
      ((g.getObjectByName('top') as THREE.Mesh).material as THREE.MeshBasicMaterial).color.setHex(p.alive ? 0x33ff88 : 0xff3333);
      g.getObjectByName('ring')!.visible = p.alive;
    }
  }

  render(belief: BeliefView | null, path: { x: number; y: number }[] | null, goal: { x: number; y: number } | null) {
    const sim = this.sim; if (!sim) return;
    const w = sim.world, r = sim.rover, t = sim.t;
    this.updateSlow(false, belief);

    this.rover.position.copy(V(r.x, r.y, r.z));
    this.rover.rotation.set(0, 0, 0);
    this.rover.rotateY(r.yaw); this.rover.rotateX(-r.roll); this.rover.rotateZ(r.pitch);
    if (!r.alive) this.rover.rotation.x = Math.PI * 0.6;

    // LiDAR rays from the last scan the brain saw
    const f = sim.lastFrame;
    this.lidar.visible = this.toggles.lidar && !!f;
    if (f && this.toggles.lidar) {
      const a = this.lidar.geometry.getAttribute('position') as THREE.BufferAttribute, arr = a.array as Float32Array;
      const sz = r.z + 0.9;
      f.scan.ranges.forEach((rg, k) => {
        const ang = r.yaw + f.scan.angle_min + k * f.scan.angle_increment, d = Math.min(rg, f.scan.range_max);
        arr.set([r.x, sz, -r.y, r.x + d * Math.cos(ang), sz, -(r.y + d * Math.sin(ang))], k * 6);
      });
      a.needsUpdate = true;
    }
    this.pathLine.visible = this.toggles.path && !!path && path.length > 1;
    if (path && this.pathLine.visible) {
      const a = this.pathLine.geometry.getAttribute('position') as THREE.BufferAttribute, arr = a.array as Float32Array;
      const n = Math.min(path.length, 600);
      for (let k = 0; k < n; k++) arr.set([path[k].x, w.heightAt(path[k].x, path[k].y) + 0.4, -path[k].y], k * 3);
      a.needsUpdate = true; this.pathLine.geometry.setDrawRange(0, n);
    }
    this.goalMarker.visible = !!goal;
    if (goal) { this.goalMarker.position.copy(V(goal.x, goal.y, w.heightAt(goal.x, goal.y) + 0.5)); this.goalMarker.rotation.z = t; }

    // tornado
    const tor = sim.dis.tornadoes.find(tt => tt.active);
    this.tornadoGroup.visible = !!tor;
    if (tor) { this.tornadoGroup.position.copy(V(tor.x, tor.y, w.heightAt(tor.x, tor.y))); this.tornadoGroup.rotation.y = t * 3; this.tornadoGroup.scale.set(tor.R / 6, 1, tor.R / 6); }
    const dm = this.dummy;
    this.debris.count = Math.min(400, sim.dis.debris.length);
    sim.dis.debris.slice(0, 400).forEach((db, k) => { dm.position.copy(V(db.x, db.y, w.heightAt(db.x, db.y) + db.z)); dm.rotation.set(t * 5 + k, t * 3, 0); dm.scale.setScalar(1); dm.updateMatrix(); this.debris.setMatrixAt(k, dm.matrix); });
    this.debris.instanceMatrix.needsUpdate = true;
    this.meteors.count = Math.min(200, sim.dis.meteors.length);
    sim.dis.meteors.slice(0, 200).forEach((m, k) => {
      const ttl = Math.max(0, m.tImpact - t), hgt = ttl * 30;
      dm.position.copy(V(m.x + ttl * 12, m.y + ttl * 5, w.heightAt(m.x, m.y) + hgt)); dm.rotation.set(0, 0, 0); dm.scale.setScalar(1 + ttl * 0.3); dm.updateMatrix(); this.meteors.setMatrixAt(k, dm.matrix);
    });
    this.meteors.instanceMatrix.needsUpdate = true;
    const fl = sim.dis.flashes[sim.dis.flashes.length - 1];
    if (fl && t - fl.t < 0.6) { this.flashLight.position.copy(V(fl.x, fl.y, w.heightAt(fl.x, fl.y) + 4)); this.flashLight.intensity = 3000 * (1 - (t - fl.t) / 0.6); } else this.flashLight.intensity = 0;

    // smoke darkens sky locally
    const sm = w.smoke[w.cellIndexAt(r.x, r.y)];
    const sky = new THREE.Color(0x9cc4e4).lerp(new THREE.Color(0x6e6460), Math.min(1, sm * 1.5));
    (this.scene.background as THREE.Color).copy(sky); (this.scene.fog as THREE.Fog).color.copy(sky);
    (this.scene.fog as THREE.Fog).near = 120 * (1 - Math.min(0.85, sm)); (this.scene.fog as THREE.Fog).far = 320 * (1 - Math.min(0.8, sm));

    // follow camera
    const target = V(r.x, r.y, r.z + 1.5);
    this.camTarget.lerp(target, 0.15);
    const followYaw = -r.yaw + Math.PI + this.camYaw * 0;
    void followYaw;
    const cy = this.camYaw, cp = this.camPitch, dist = this.camDist;
    this.camera.position.set(this.camTarget.x + dist * Math.cos(cp) * Math.cos(cy), this.camTarget.y + dist * Math.sin(cp), this.camTarget.z + dist * Math.cos(cp) * Math.sin(cy));
    this.camera.lookAt(this.camTarget);

    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, W, H);
    this.renderer.render(this.scene, this.camera);
    if (this.toggles.camera) {
      // onboard camera picture-in-picture (bottom-right)
      const cw = Math.round(Math.min(320, W * 0.28)), ch = Math.round(cw * 0.75);
      this.roverCam.position.copy(V(r.x + 0.55 * Math.cos(r.yaw), r.y + 0.55 * Math.sin(r.yaw), r.z + 1.8));
      this.roverCam.lookAt(V(r.x + 10 * Math.cos(r.yaw), r.y + 10 * Math.sin(r.yaw), r.z + 1.0));
      this.lidar.visible = false; this.rover.visible = false;
      this.renderer.setScissorTest(true);
      this.renderer.setViewport(W - cw - 10, 10, cw, ch); this.renderer.setScissor(W - cw - 10, 10, cw, ch);
      this.renderer.render(this.scene, this.roverCam);
      this.renderer.setScissorTest(false);
      this.rover.visible = true; this.lidar.visible = this.toggles.lidar && !!f;
    }
    void CELL; void HALF;
  }
}
