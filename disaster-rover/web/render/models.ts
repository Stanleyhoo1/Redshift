import * as THREE from 'three';

/** Shared procedural textures (canvas-generated, no external assets). */
const texCache = new Map<string, THREE.Texture>();
function canvasTex(key: string, w: number, h: number, draw: (g: CanvasRenderingContext2D) => void, repeat = true): THREE.Texture {
  const hit = texCache.get(key); if (hit) return hit;
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d')!);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
  t.anisotropy = 4;
  texCache.set(key, t);
  return t;
}

/** Solar panel: dark blue cells with a silver grid. */
export const solarTex = () => canvasTex('solar', 128, 64, g => {
  g.fillStyle = '#0c1d3a'; g.fillRect(0, 0, 128, 64);
  for (let y = 0; y < 4; y++) for (let x = 0; x < 8; x++) {
    const gr = g.createLinearGradient(x * 16, y * 16, x * 16 + 16, y * 16 + 16);
    gr.addColorStop(0, '#1d3f7a'); gr.addColorStop(1, '#0f2550');
    g.fillStyle = gr; g.fillRect(x * 16 + 1, y * 16 + 1, 14, 14);
  }
}, false);

/** House facade with windows and a door band; tinted per building via material colour. */
export const facadeTex = () => canvasTex('facade', 128, 128, g => {
  g.fillStyle = '#ffffff'; g.fillRect(0, 0, 128, 128);
  g.fillStyle = 'rgba(0,0,0,0.06)'; for (let y = 0; y < 128; y += 8) g.fillRect(0, y, 128, 1);
  for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
    const X = 16 + x * 64, Y = 18 + y * 60;
    g.fillStyle = '#5a4a3a'; g.fillRect(X - 3, Y - 3, 38, 34);
    const gr = g.createLinearGradient(X, Y, X + 32, Y + 28); gr.addColorStop(0, '#9fc7e8'); gr.addColorStop(1, '#3d5f80');
    g.fillStyle = gr; g.fillRect(X, Y, 32, 28);
    g.fillStyle = '#5a4a3a'; g.fillRect(X + 15, Y, 2, 28); g.fillRect(X, Y + 13, 32, 2);
  }
});

/** Radial soft blob, for smoke / embers / glow sprites. */
export const softDot = () => canvasTex('dot', 64, 64, g => {
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.4, 'rgba(255,255,255,0.55)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
}, false);

/** Streaky noise for the tornado funnel. */
export const streakTex = () => canvasTex('streak', 128, 256, g => {
  g.fillStyle = '#808080'; g.fillRect(0, 0, 128, 256);
  for (let i = 0; i < 260; i++) {
    const x = Math.random() * 128, w = 1 + Math.random() * 4, a = Math.random() * 0.35;
    g.fillStyle = Math.random() < 0.5 ? `rgba(40,40,45,${a})` : `rgba(210,210,215,${a})`;
    g.fillRect(x, 0, w, 256);
  }
});

const std = (color: number, o: Partial<THREE.MeshStandardMaterialParameters> = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.7, metalness: 0.1, ...o });

export interface RoverModel {
  group: THREE.Group;
  /** call every frame */
  update(t: number, speed: number, yawRate: number, probesLeft: number, beaconColor: number, alive: boolean): void;
}

/**
 * Six-wheel rocker-bogie rover. Model frame: +x forward, +y up, +z left.
 * Includes chassis, solar deck, mast + pan/tilt camera head, spinning LiDAR puck, antenna,
 * probe rack (shows probes remaining), headlights and a behaviour-coloured status beacon.
 */
export function buildRover(): RoverModel {
  const g = new THREE.Group();
  const orange = std(0xf2a33a, { roughness: 0.45, metalness: 0.25 });
  const dark = std(0x2b2f36, { roughness: 0.6, metalness: 0.4 });
  const metal = std(0xb9bec6, { roughness: 0.35, metalness: 0.8 });
  const rubber = std(0x161616, { roughness: 0.95, metalness: 0 });
  const glass = std(0x0a0f18, { roughness: 0.1, metalness: 0.9 });

  const body = new THREE.Group(); body.position.y = 0.72; g.add(body);
  // chassis: main tub + bevelled top + side skirts
  const tub = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.42, 1.0), orange); body.add(tub);
  const top = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.08, 0.9), dark); top.position.y = 0.25; body.add(top);
  for (const s of [-1, 1]) { const sk = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.18, 0.05), dark); sk.position.set(0, -0.12, s * 0.52); body.add(sk); }
  const nose = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.3, 0.95), dark); nose.position.set(0.88, 0, 0); body.add(nose);
  // headlights
  const lightMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff2c0, emissiveIntensity: 2 });
  for (const s of [-1, 1]) { const hl = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.08, 0.16), lightMat); hl.position.set(0.95, 0.04, s * 0.3); body.add(hl); }
  // solar deck
  const solar = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.03, 0.82), new THREE.MeshStandardMaterial({ map: solarTex(), roughness: 0.25, metalness: 0.6 }));
  solar.position.set(-0.22, 0.31, 0); body.add(solar);
  // probe rack at the rear: 6 slots
  const rack = new THREE.Group(); rack.position.set(-0.72, 0.36, 0); body.add(rack);
  const rackBase = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.06, 0.8), dark); rack.add(rackBase);
  const probeMat = std(0xe8eef4, { roughness: 0.4 }); const probeCap = std(0x33ff88, { emissive: 0x115533 });
  const slots: THREE.Group[] = [];
  for (let k = 0; k < 6; k++) {
    const p = new THREE.Group(); p.position.set(((k % 2) - 0.5) * 0.12, 0.14, (Math.floor(k / 2) - 1) * 0.25);
    const c = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.24, 10), probeMat); p.add(c);
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), probeCap); cap.position.y = 0.13; p.add(cap);
    rack.add(p); slots.push(p);
  }
  // mast + pan/tilt camera head
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.06, 1.0, 10), metal); mast.position.set(0.55, 0.78, 0.25); body.add(mast);
  const head = new THREE.Group(); head.position.set(0.55, 1.3, 0.25); body.add(head);
  const hb = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.16, 0.42), dark); head.add(hb);
  for (const s of [-1, 1]) { const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.05, 14), glass); lens.rotation.z = Math.PI / 2; lens.position.set(0.12, 0, s * 0.12); head.add(lens); }
  // beacon
  const beaconMat = new THREE.MeshStandardMaterial({ color: 0x33ff88, emissive: 0x33ff88, emissiveIntensity: 1.5 });
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 10), beaconMat); beacon.position.set(0, 0.14, 0); head.add(beacon);
  // LiDAR puck
  const lidar = new THREE.Group(); lidar.position.set(0.25, 0.42, -0.2); body.add(lidar);
  const lb = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.15, 0.14, 20), dark); lidar.add(lb);
  const ring = new THREE.Mesh(new THREE.CylinderGeometry(0.132, 0.132, 0.04, 20), new THREE.MeshStandardMaterial({ color: 0x2266ff, emissive: 0x3388ff, emissiveIntensity: 1.2 }));
  ring.position.y = 0.02; lidar.add(ring);
  const win = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.06), glass); win.position.set(0.13, 0.02, 0); lidar.add(win);
  // antenna
  const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.015, 0.9, 6), metal); ant.position.set(-0.5, 0.75, -0.38); body.add(ant);
  const antTip = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 6), std(0xff3333, { emissive: 0xaa0000 })); antTip.position.set(-0.5, 1.2, -0.38); body.add(antTip);
  const dish = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 8, 0, Math.PI * 2, 0, Math.PI / 3), metal);
  dish.rotation.set(0, 0, -Math.PI / 2.6); dish.position.set(-0.25, 0.55, -0.32); body.add(dish);

  // rocker-bogie suspension + 6 wheels
  const wheels: THREE.Group[] = [];
  const wheelGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.22, 20);
  const hubGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.24, 10);
  for (const s of [-1, 1]) {
    const rocker = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.06, 0.06), metal); rocker.position.set(0.1, 0.55, s * 0.68); rocker.rotation.z = -0.12; g.add(rocker);
    const bogie = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.06, 0.06), metal); bogie.position.set(-0.35, 0.45, s * 0.68); bogie.rotation.z = 0.15; g.add(bogie);
    for (const x of [0.72, 0, -0.72]) {
      const w = new THREE.Group(); w.position.set(x, 0.28, s * 0.72);
      const tire = new THREE.Mesh(wheelGeo, rubber); tire.rotation.x = Math.PI / 2; w.add(tire);
      const hub = new THREE.Mesh(hubGeo, metal); hub.rotation.x = Math.PI / 2; w.add(hub);
      // treads
      for (let k = 0; k < 10; k++) {
        const tr = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.035, 0.23), rubber);
        const a = k / 10 * Math.PI * 2; tr.position.set(Math.cos(a) * 0.285, Math.sin(a) * 0.285, 0); tr.rotation.z = a; w.add(tr);
      }
      g.add(w); wheels.push(w);
    }
  }
  g.traverse(o => { if ((o as THREE.Mesh).isMesh) { o.castShadow = true; o.receiveShadow = true; } });

  let spin = 0;
  return {
    group: g,
    update(t, speed, yawRate, probesLeft, beaconColor, alive) {
      spin += speed * 0.05 / 0.28;
      wheels.forEach((w, k) => { w.rotation.z = -spin * (1 + (k % 3 === 1 ? 0 : 0)) - yawRate * 0.02 * (k < 3 ? -1 : 1); });
      lidar.rotation.y = alive ? t * 12 : lidar.rotation.y;
      head.rotation.y = Math.sin(t * 0.7) * 0.5;
      slots.forEach((p, k) => (p.visible = k < probesLeft));
      beaconMat.color.setHex(beaconColor); beaconMat.emissive.setHex(beaconColor);
      beaconMat.emissiveIntensity = alive ? 1 + Math.sin(t * 6) * 0.8 : 0;
      ring.visible = alive;
      lightMat.emissiveIntensity = alive ? 2 : 0;
    },
  };
}

export interface ProbeModel { group: THREE.Group; update(t: number, alive: boolean): void }

/** Early-warning probe: tripod, sensor body, solar cap, whip antenna, blinking beacon. */
export function buildProbe(id: number): ProbeModel {
  const g = new THREE.Group();
  const white = std(0xe9edf2, { roughness: 0.45 }), metal = std(0x9aa3ad, { metalness: 0.8, roughness: 0.3 }), dark = std(0x2b2f36);
  for (let k = 0; k < 3; k++) {
    const a = k / 3 * Math.PI * 2 + id;
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.03, 1.1, 6), metal);
    leg.position.set(Math.cos(a) * 0.3, 0.5, Math.sin(a) * 0.3);
    leg.rotation.set(Math.sin(a) * 0.35, 0, -Math.cos(a) * 0.35); g.add(leg);
  }
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 0.55, 14), white); body.position.y = 1.15; g.add(body);
  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.185, 0.185, 0.06, 14), std(0xf2a33a)); band.position.y = 1.05; g.add(band);
  const sensor = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.2, 10), dark); sensor.position.y = 0.78; g.add(sensor);
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.22, 0.04, 16), new THREE.MeshStandardMaterial({ map: solarTex(), roughness: 0.3, metalness: 0.5 }));
  cap.position.y = 1.45; g.add(cap);
  const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.012, 1.0, 5), metal); ant.position.set(0.12, 1.95, 0); g.add(ant);
  const beaconMat = new THREE.MeshStandardMaterial({ color: 0x33ff88, emissive: 0x33ff88, emissiveIntensity: 2 });
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 10), beaconMat); beacon.position.y = 1.56; g.add(beacon);
  g.traverse(o => { if ((o as THREE.Mesh).isMesh) o.castShadow = true; });
  return {
    group: g,
    update(t, alive) {
      const c = alive ? 0x33ff88 : 0xff3333;
      beaconMat.color.setHex(c); beaconMat.emissive.setHex(c);
      beaconMat.emissiveIntensity = alive ? (Math.sin(t * 4 + id) > 0.6 ? 3 : 0.4) : 0.3;
      g.rotation.z = alive ? 0 : 0.6;
    },
  };
}

export interface BuildingModel { group: THREE.Group; walls: THREE.Mesh; roof: THREE.Mesh; h: number }

const WALL_COLORS = [0xe9dcc3, 0xd7c9b0, 0xc9d4d9, 0xe3c7a8, 0xd8d0c4, 0xbfc9b8];
const ROOF_COLORS = [0x9b3d2b, 0x6b4a3a, 0x4a4f57, 0x8a4a35];

/** House: textured walls (windows), pyramid roof, chimney. */
export function buildBuilding(id: number, w: number, d: number, h: number): BuildingModel {
  const g = new THREE.Group();
  const tex = facadeTex().clone(); tex.needsUpdate = true;
  tex.repeat.set(Math.max(1, Math.round(w / 3.5)), Math.max(1, Math.round(h / 3.2)));
  const walls = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({ map: tex, color: WALL_COLORS[id % WALL_COLORS.length], roughness: 0.85 }));
  walls.position.y = h / 2; g.add(walls);
  const roofH = Math.min(w, d) * 0.45;
  const roof = new THREE.Mesh(new THREE.ConeGeometry(Math.SQRT1_2 * 1.08, 1, 4, 1), std(ROOF_COLORS[id % ROOF_COLORS.length], { roughness: 0.8, flatShading: true }));
  roof.rotation.y = Math.PI / 4; roof.scale.set(w, roofH, d); roof.position.y = h + roofH / 2; g.add(roof);
  const chim = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1.4, 0.6), std(0x7b5a4a)); chim.position.set(w * 0.25, h + roofH * 0.6, d * 0.2); g.add(chim);
  const base = new THREE.Mesh(new THREE.BoxGeometry(w + 0.4, 0.3, d + 0.4), std(0x8a8580)); base.position.y = 0.15; g.add(base);
  g.traverse(o => { if ((o as THREE.Mesh).isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return { group: g, walls, roof, h };
}
