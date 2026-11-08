import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore, useSim, useApp, nowSeconds, roverPoses } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { SIGNAL, SLOPE, REGOLITH, VOID } from '../lib/theme.js'
import { reducedMotion } from './sceneConstants.js'
import { regolithTextures, blackTexture } from './proceduralTextures.js'
import { waterShared } from './waterShared.js'
import { biomeMapOf, worldPalette } from '../lib/biomes.js'

// World (x, y, z-up) maps to three.js (x, z, -y) everywhere in the scene.
// The terrain is one indexed grid mesh with nx * ny vertices and smooth normals, drawn with
// a physically based material whose surface is generated in the shader: procedural regolith
// (multi-scale albedo, pebbles, a crack network, a detail normal map, roughness), rockier
// and darker on steep ground and ridges with hints of strata on cliffs, fine dust settled
// in hollows, wet dark ground along identified water. Unknown ground is discarded (the dark
// floor below shows through); the explored edge wobbles and fades softly into the dark.
// On top, in display colours (the HUD layer): the optional slope tint with contour lines,
// sensor pulses round each rover, the reveal flash and a faint scan rim on the edge.

const display3 = (hex) => {
  const c = new THREE.Color(hex).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace)
  return new THREE.Vector3(c.r, c.g, c.b)
}
const linear3 = (hex) => new THREE.Color(hex)
const DUNE_SAND = '#c99a6c'
// Ground palette uniforms: the primary biome's (A) and the secondary biome's (2), blended per pixel by
// the secondary share from features.biome_map. Without a biome both are today's rust palette.
const PALETTE_UNIFORMS = [
  ['uDustA', 'dustLight'],
  ['uDustB', 'dust'],
  ['uSoil', 'soil'],
  ['uRock', 'rock'],
  ['uRockLight', 'rockLight'],
  ['uStrataA', 'strataA'],
  ['uStrataB', 'strataB'],
  ['uPebble', 'pebble'],
  ['uDuneSand', 'dune'],
]

// v7 biomes: palettes from init world.biome, the blend grid from features.biome_map (a small float
// texture, linear filtered). Old sims, the mock and biomes off: today's look (uBiomeOn 0).
function syncBiome(u, world, features) {
  const pal = worldPalette(world)
  for (const [name, key] of PALETTE_UNIFORMS) {
    u[name].value.set(pal.ground[key])
    u[name + '2'].value.set(pal.ground2[key])
  }
  const map = pal.biome ? biomeMapOf(features) : null
  const old = u.uBiome.value
  if (!map) {
    u.uBiomeOn.value = 0
    if (old && old !== blackTexture()) old.dispose()
    u.uBiome.value = blackTexture()
    return
  }
  // 8 bit (R8): linear filtering works everywhere, unlike float textures without OES_texture_float_linear
  const bytes = new Uint8Array(map.w.length)
  for (let k = 0; k < bytes.length; k++) bytes[k] = Math.round(map.w[k] * 255)
  const tex = new THREE.DataTexture(bytes, map.nx, map.ny, THREE.RedFormat, THREE.UnsignedByteType)
  tex.magFilter = THREE.LinearFilter
  tex.minFilter = THREE.LinearFilter
  tex.needsUpdate = true
  if (old && old !== blackTexture()) old.dispose()
  u.uBiome.value = tex
  u.uBiomeMap.value.set(map.ox, map.oy, 1 / map.res, 0)
  u.uBiomeSize.value.set(map.nx, map.ny)
  u.uBiomeOn.value = 1
}
const V7_LANDFORM_KEYS = ['mesas', 'pits', 'canyon', 'dunes']

// v7 landform uniforms from features (on only when the sim sends any landform).
function syncLandforms(u, features) {
  const on = !!features && V7_LANDFORM_KEYS.some((k) => (Array.isArray(features[k]) ? features[k].length > 0 : !!features[k]))
  u.uV7.value = on ? 1 : 0
  let c = 0
  const dunes = on && Array.isArray(features.dunes) ? features.dunes : []
  for (const d of dunes) {
    if (c >= 8 || !d || ![d.x, d.y, d.radius].every(Number.isFinite)) continue
    u.uDunes.value[c].set(d.x, -d.y, d.radius, ((Number.isFinite(d.wind_deg) ? d.wind_deg : 0) * Math.PI) / 180)
    c++
  }
  u.uDuneCount.value = c
}

const vertexCommon = /* glsl */ `
attribute float aSlope;
attribute float aKnown;
attribute float aReveal;
attribute float aEdge;
attribute float aCurv;
uniform float uTime;
uniform float uMotion;
varying float vSlope;
varying float vFlash;
varying float vEdge;
varying float vCurv;
varying vec3 vWorld;
varying vec3 vGeoN;
`

const vertexBody = /* glsl */ `
  float tAge = uTime - aReveal;
  float tRise = aKnown > 0.5 ? smoothstep(0.0, 0.7, tAge) : 1.0;
  transformed.y -= (1.0 - tRise) * 1.1 * uMotion;
  vSlope = aSlope;
  vEdge = aEdge;
  vCurv = aCurv;
  vFlash = aKnown > 0.5 ? exp(-max(tAge, 0.0) * 1.3) : 0.0;
  vWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;
  vGeoN = normalize(mat3(modelMatrix) * objectNormal);
`

const fragmentCommon = /* glsl */ `
uniform sampler2D uDetail;
uniform sampler2D uDetailN;
uniform sampler2D uShore;
uniform vec4 uShoreMap;
uniform vec2 uShoreSize;
uniform vec3 uDustA;
uniform vec3 uDustB;
uniform vec3 uSoil;
uniform vec3 uRock;
uniform vec3 uRockLight;
uniform vec3 uStrataA;
uniform vec3 uStrataB;
uniform vec3 uPebble;
uniform vec3 uDustA2;
uniform vec3 uDustB2;
uniform vec3 uSoil2;
uniform vec3 uRock2;
uniform vec3 uRockLight2;
uniform vec3 uStrataA2;
uniform vec3 uStrataB2;
uniform vec3 uPebble2;
uniform vec3 uDuneSand2;
uniform sampler2D uBiome;
uniform vec4 uBiomeMap;
uniform vec2 uBiomeSize;
uniform float uBiomeOn;
uniform float uTime;
uniform float uMotion;
uniform float uLimit;
uniform float uSlopeMode;
uniform vec3 uSignal;
uniform vec3 uOk;
uniform vec3 uCaution;
uniform vec3 uBad;
uniform vec3 uVoid;
uniform vec2 uRovers[8];
uniform int uRoverCount;
uniform float uV7;
uniform vec4 uDunes[8];
uniform int uDuneCount;
uniform vec3 uDuneSand;
varying float vSlope;
varying float vFlash;
varying float vEdge;
varying float vCurv;
varying vec3 vWorld;
varying vec3 vGeoN;
const mat2 ROT1 = mat2(0.8, -0.6, 0.6, 0.8);
const mat2 ROT2 = mat2(-0.28, 0.96, -0.96, -0.28);
`

// Replaces <map_fragment>: everything about the surface, computed once.
const surface = /* glsl */ `
  if (vEdge < 0.26) discard; // unknown ground: out before any texture work
  vec2 wp = vWorld.xz;
  vec4 mA = texture2D(uDetail, wp * 0.0131 + vec2(0.13, 0.71));
  vec4 mB = texture2D(uDetail, ROT1 * wp * 0.0417 + vec2(0.31, 0.07));

  // The edge of the known area, wobbled over a metre or two so it never shows the grid.
  float edgeN = mB.a * 0.55 + texture2D(uDetail, wp * 0.19).a * 0.45;
  float thr = mix(0.26, 0.8, smoothstep(0.32, 0.68, edgeN));
  if (vEdge < thr) discard;
  float edgeFade = smoothstep(thr, thr + 0.3, vEdge);
  float edgeRim = 1.0 - smoothstep(thr, thr + 0.045, vEdge);

  float macro = mA.a * 0.6 + mB.a * 0.4;

  // v7 biomes: the secondary biome's share here, its border frayed a little by the macro noise.
  float bW = 0.0;
  if (uBiomeOn > 0.5) {
    vec2 buv = (vec2((wp.x - uBiomeMap.x) * uBiomeMap.z, (-wp.y - uBiomeMap.y) * uBiomeMap.z) + 0.5) / uBiomeSize;
    bW = smoothstep(0.2, 0.8, texture2D(uBiome, buv).r + (macro - 0.5) * 0.45);
  }
  vec3 cDustA = mix(uDustA, uDustA2, bW);
  vec3 cDustB = mix(uDustB, uDustB2, bW);
  vec3 cSoil = mix(uSoil, uSoil2, bW);
  vec3 cRock = mix(uRock, uRock2, bW);
  vec3 cRockLight = mix(uRockLight, uRockLight2, bW);
  vec3 cStrataA = mix(uStrataA, uStrataA2, bW);
  vec3 cStrataB = mix(uStrataB, uStrataB2, bW);
  vec3 cPebble = mix(uPebble, uPebble2, bW);
  vec3 cDuneSand = mix(uDuneSand, uDuneSand2, bW);
#ifdef TERRAIN_HQ
  vec4 d1 = texture2D(uDetail, wp * 0.27);
  vec4 d2 = texture2D(uDetail, ROT2 * wp * 0.173 + vec2(0.57, 0.21));
  float bw = smoothstep(0.38, 0.62, mB.r);
  vec4 det = mix(d1, d2, bw);
#else
  vec4 det = texture2D(uDetail, wp * 0.27);
#endif
  vec3 Ng = normalize(vGeoN);
  float cosS = Ng.y;
  float curv = clamp(vCurv * 25.0, -1.0, 1.0); // + hollow, - ridge
  float steep = smoothstep(0.95, 0.8, cosS); // about 18 to 37 deg
  float cliff = smoothstep(0.85, 0.66, cosS); // about 32 to 49 deg

  // Dust on the flats, varied at several scales.
  float tone = det.r * 0.35 + macro * 0.65;
  vec3 dust = mix(cDustB, cDustA, smoothstep(0.25, 0.8, tone));
  // Large albedo features, as from orbit: dark basaltic sand sheets and redder, iron-rich
  // dust in places, greyer in others.
  float sheet = smoothstep(0.48, 0.7, mA.r * 0.65 + mB.g * 0.15 + (1.0 - mA.a) * 0.35) * (1.0 - smoothstep(0.0, 0.5, -curv));
  dust = mix(dust, cSoil * (0.85 + 0.25 * det.r), sheet * 0.78);
  float hue = mB.r - 0.5;
  dust *= vec3(1.0 + hue * 0.16, 1.0 - hue * 0.02, 1.0 - hue * 0.14);
  dust = mix(dust, cDustA * 1.06, smoothstep(0.1, 0.7, curv) * 0.4);
  float peb = det.g;
  dust = mix(dust, mix(cPebble, cRock, step(0.8, fract(det.r * 9.0))), peb * 0.38);
  // Dried crust with fine cracks, only in patches.
  float crust = smoothstep(0.56, 0.72, mB.a);
  dust *= 1.0 - det.b * 0.16 * crust;

  // Exposed rock on steep ground and ridges, banded on cliffs.
  vec3 rock = mix(cRock, cRockLight, smoothstep(0.35, 0.8, det.r * 0.7 + mA.r * 0.3));
  float bands = sin(vWorld.y * 4.2 + (mB.r - 0.5) * 5.0 + (det.r - 0.5) * 1.4);
  vec3 strata = mix(cStrataA, cStrataB, smoothstep(-0.35, 0.55, bands));
  strata *= 0.9 + 0.1 * sin(vWorld.y * 11.0 + (mA.r - 0.5) * 6.0);
  rock = mix(rock, strata, cliff * 0.75);
  float rockAmt = steep + max(-curv, 0.0) * 0.45 + (det.r - 0.5) * 0.35 + (macro - 0.5) * 0.3;
  float rockMask = smoothstep(0.42, 0.62, rockAmt);
  vec3 albedo = mix(dust, rock, rockMask);

  // v7 landforms (only when the sim sends any): mesa and butte walls, canyon walls and pit
  // rims in darker, boldly banded rock; dune fields in a warmer, lighter sand that fades out
  // at the edge, with long crests across the wind.
  float duneAmt = 0.0;
  float duneCrest = 1.0;
  if (uV7 > 0.5) {
    float wall = smoothstep(0.82, 0.58, cosS); // about 35 to 55 deg
    float wb = sin(vWorld.y * 2.4 + (mB.r - 0.5) * 2.5 + (det.r - 0.5) * 0.8);
    vec3 wallCol = mix(cStrataA * 0.5, cStrataB * 0.82, smoothstep(-0.45, 0.5, wb));
    wallCol *= 0.86 + 0.14 * sin(vWorld.y * 8.5 + (mA.r - 0.5) * 5.0);
    albedo = mix(albedo, wallCol, wall * 0.85);
    for (int i = 0; i < 8; i++) {
      if (i >= uDuneCount) break;
      vec4 dz = uDunes[i];
      vec2 dd = wp - dz.xy;
      float rr = length(dd) / max(dz.z, 1.0);
      float a = 1.0 - smoothstep(0.5, 1.0, rr + (macro - 0.5) * 0.25);
      if (a <= 0.0) continue;
      vec2 wd = vec2(cos(dz.w), -sin(dz.w));
      // Long crests across the wind: a gentle windward rise, a short darker slip face.
      float ph = fract(dot(dd, wd) * 0.14 + sin(dot(dd, vec2(-wd.y, wd.x)) * 0.09) * 0.35);
      float face = smoothstep(0.72, 0.8, ph) * (1.0 - smoothstep(0.9, 1.0, ph));
      if (a > duneAmt) duneCrest = mix(1.04, 0.8, face) * (0.97 + 0.06 * ph);
      duneAmt = max(duneAmt, a);
    }
    duneAmt *= 1.0 - wall;
    vec3 sand = cDuneSand * (0.9 + 0.12 * det.r + 0.06 * macro) * mix(1.0, duneCrest, duneAmt);
    albedo = mix(albedo, sand, duneAmt * 0.8);
  }

  // Wet, darker ground along water the fleet has identified.
  vec2 suv = (vec2((wp.x - uShoreMap.x) * uShoreMap.z, (-wp.y - uShoreMap.y) * uShoreMap.z) + 0.5) / uShoreSize;
  float wet = texture2D(uShore, suv).r;

  float tRough = mix(0.96, 0.85, rockMask) - peb * 0.08;
#ifdef TERRAIN_HQ
  vec4 nA = texture2D(uDetailN, wp * 0.27);
  vec4 nB = texture2D(uDetailN, ROT2 * wp * 0.173 + vec2(0.57, 0.21));
  vec4 nF = texture2D(uDetailN, ROT1 * wp * 1.13 + vec2(0.4, 0.9));
  vec2 gA = (nA.xy - 0.5) * 2.0;
  vec2 gB = ((nB.xy - 0.5) * 2.0) * ROT2;
  vec2 gF = ((nF.xy - 0.5) * 2.0) * ROT1;
  vec2 dn = mix(gA, gB, bw) * 0.7 + gF * 0.35;
  // Wind ripples in the sand sheets.
  vec2 wdir = vec2(0.83, 0.56);
  float rph = dot(wp, wdir) * 9.5 + (mB.r - 0.5) * 9.0 + (det.r - 0.5) * 2.0;
  dn += wdir * cos(rph) * 0.22 * sheet * (1.0 - rockMask);
  dn *= mix(0.7, 1.2, rockMask) * (1.0 - 0.6 * wet);
  float cav = mix(nA.z, nB.z, bw);
  albedo *= mix(1.0, 0.8 + 0.2 * cav, crust);
  vec3 tNormalW = normalize(vec3(Ng.x + dn.x, Ng.y, Ng.z + dn.y));
#else
  vec3 tNormalW = Ng;
#endif
  albedo *= mix(1.0, 0.48, wet);
  tRough = mix(tRough, 0.38, wet);
  diffuseColor.rgb = albedo;
`

// After <fog_fragment>: the HUD layer, in display colours.
const overlay = /* glsl */ `
  gl_FragColor.rgb = mix(uVoid, gl_FragColor.rgb, 0.1 + 0.9 * edgeFade);
  if (uSlopeMode > 0.5) {
    // Slope relative to the fleet's current limit: comfortable below 60 %, caution from
    // 60 % up to the limit, impassable at or above it (antialiased just below each step).
    float r = vSlope / max(uLimit, 1.0);
    float aa = clamp(fwidth(r), 1e-4, 0.04);
    vec3 sc = mix(uOk, uCaution, smoothstep(0.6 - aa, 0.6, r));
    sc = mix(sc, uBad, smoothstep(1.0 - aa, 1.0, r));
    float lum = dot(gl_FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 tint = sc * clamp(0.62 + lum * 1.25, 0.6, 1.3);
    gl_FragColor.rgb = mix(gl_FragColor.rgb, tint, 0.68 * edgeFade);
    // Contour lines: thin every 1 m, stronger every 5 m, faded where they would crowd.
    float hgt = vWorld.y;
    float w = max(fwidth(hgt), 1e-4);
    float c1 = abs(fract(hgt + 0.5) - 0.5);
    float c5 = abs(fract(hgt / 5.0 + 0.5) - 0.5) * 5.0;
    float crowd = 1.0 - smoothstep(0.1, 0.32, w);
    float lines = max((1.0 - smoothstep(0.0, 1.2 * w, c1)) * 0.16, (1.0 - smoothstep(0.0, 1.6 * w, c5)) * 0.3) * crowd;
    gl_FragColor.rgb *= 1.0 - lines;
  }
  // Sensor pulses spreading from each rover.
  float pulse = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= uRoverCount) break;
    float dist = length(vWorld.xz - uRovers[i]);
    float ph = fract(uTime * 0.4 + float(i) * 0.37);
    float rad = ph * 10.0;
    float fall = 1.0 - ph;
    pulse += exp(-pow((dist - rad) / 0.22, 2.0)) * fall * fall;
  }
  gl_FragColor.rgb = mix(gl_FragColor.rgb, uSignal, clamp(pulse * uMotion, 0.0, 1.0) * 0.28);
  gl_FragColor.rgb = mix(gl_FragColor.rgb, uSignal, clamp(vFlash, 0.0, 1.0) * 0.5);
  gl_FragColor.rgb = mix(gl_FragColor.rgb, uSignal, edgeRim * 0.4);
`

function createMaterial(high) {
  const tex = regolithTextures()
  const m = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.95, metalness: 0 })
  m.userData.realistic = true
  m.defines = high ? { TERRAIN_HQ: '' } : {}
  const uniforms = {
    uDetail: { value: tex.detail },
    uDetailN: { value: tex.normal },
    uShore: { value: blackTexture() },
    uShoreMap: { value: new THREE.Vector4(0, 0, 1, 0) },
    uShoreSize: { value: new THREE.Vector2(1, 1) },
    uDustA: { value: linear3(REGOLITH.dustLight) },
    uDustB: { value: linear3(REGOLITH.dust) },
    uSoil: { value: linear3(REGOLITH.soil) },
    uRock: { value: linear3(REGOLITH.rock) },
    uRockLight: { value: linear3(REGOLITH.rockLight) },
    uStrataA: { value: linear3(REGOLITH.strataA) },
    uStrataB: { value: linear3(REGOLITH.strataB) },
    uPebble: { value: linear3(REGOLITH.pebble) },
    uTime: { value: 0 },
    uMotion: { value: 1 },
    uLimit: { value: 30 },
    uSlopeMode: { value: 0 },
    uSignal: { value: display3(SIGNAL) },
    uOk: { value: display3(SLOPE.ok) },
    uCaution: { value: display3(SLOPE.caution) },
    uBad: { value: display3(SLOPE.bad) },
    uVoid: { value: display3(VOID) },
    uRovers: { value: Array.from({ length: 8 }, () => new THREE.Vector2(1e5, 1e5)) },
    uRoverCount: { value: 0 },
    uV7: { value: 0 },
    uDunes: { value: Array.from({ length: 8 }, () => new THREE.Vector4(0, 0, 0, 0)) },
    uDuneCount: { value: 0 },
    uDuneSand: { value: linear3(DUNE_SAND) },
    uDustA2: { value: linear3(REGOLITH.dustLight) },
    uDustB2: { value: linear3(REGOLITH.dust) },
    uSoil2: { value: linear3(REGOLITH.soil) },
    uRock2: { value: linear3(REGOLITH.rock) },
    uRockLight2: { value: linear3(REGOLITH.rockLight) },
    uStrataA2: { value: linear3(REGOLITH.strataA) },
    uStrataB2: { value: linear3(REGOLITH.strataB) },
    uPebble2: { value: linear3(REGOLITH.pebble) },
    uDuneSand2: { value: linear3(DUNE_SAND) },
    uBiome: { value: blackTexture() },
    uBiomeMap: { value: new THREE.Vector4(0, 0, 1, 0) },
    uBiomeSize: { value: new THREE.Vector2(1, 1) },
    uBiomeOn: { value: 0 },
  }
  m.userData.uniforms = uniforms
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${vertexCommon}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${vertexBody}`)
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${fragmentCommon}`)
      .replace('#include <map_fragment>', surface)
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = tRough;')
      .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\n  normal = normalize((viewMatrix * vec4(tNormalW, 0.0)).xyz);')
      .replace('#include <fog_fragment>', `#include <fog_fragment>\n${overlay}`)
  }
  m.customProgramCacheKey = () => 'redshift-terrain-3'
  return m
}

// Shadow pass: only the ground that is drawn casts (no walls down to the unknown floor).
function createDepthMaterial() {
  const m = new THREE.MeshDepthMaterial()
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float aEdge;\nvarying float vEdge;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vEdge = aEdge;')
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vEdge;')
      .replace('void main() {', 'void main() {\n  if (vEdge < 0.5) discard;')
  }
  m.customProgramCacheKey = () => 'redshift-terrain-depth-1'
  return m
}

// Owns the geometry for one terrain (one init) and keeps it in sync with the
// sample arrays, processing only the samples that changed.
class TerrainMesh {
  constructor(T) {
    this.T = T
    const { nx, ny, res, ox, oy } = T
    const n = nx * ny
    this.fogY = T.minH - 0.8
    this.pos = new Float32Array(n * 3)
    this.nor = new Float32Array(n * 3)
    this.slope = new Float32Array(n)
    this.known = new Float32Array(n)
    this.reveal = new Float32Array(n)
    this.edge = new Float32Array(n)
    this.curv = new Float32Array(n)
    this.stamp = new Uint32Array(n)
    this.touched = new Int32Array(n)
    this.frame = 0
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i
        this.pos[k * 3] = ox + i * res
        this.pos[k * 3 + 1] = this.fogY
        this.pos[k * 3 + 2] = -(oy + j * res)
        this.nor[k * 3 + 1] = 1
      }
    }
    const quads = (nx - 1) * (ny - 1)
    const index = n > 65535 ? new Uint32Array(quads * 6) : new Uint16Array(quads * 6)
    let q = 0
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = j * nx + i
        const b = a + 1
        const c = a + nx
        const d = c + 1
        index[q++] = a
        index[q++] = b
        index[q++] = c
        index[q++] = b
        index[q++] = d
        index[q++] = c
      }
    }
    const geo = new THREE.BufferGeometry()
    const dyn = (arr, size) => new THREE.BufferAttribute(arr, size).setUsage(THREE.DynamicDrawUsage)
    this.attrs = {
      position: dyn(this.pos, 3),
      normal: dyn(this.nor, 3),
      aSlope: dyn(this.slope, 1),
      aKnown: dyn(this.known, 1),
      aReveal: dyn(this.reveal, 1),
      aEdge: dyn(this.edge, 1),
      aCurv: dyn(this.curv, 1),
    }
    for (const [name, attr] of Object.entries(this.attrs)) geo.setAttribute(name, attr)
    geo.setIndex(new THREE.BufferAttribute(index, 1))
    const half = Math.hypot(T.width, T.depth) / 2
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(T.cx, (T.minH + T.maxH) / 2, -T.cy), half + 10)
    this.geometry = geo
    this.syncAll()
  }

  provisional(k) {
    // Height for an unknown vertex next to known ones: the mean of its known neighbours,
    // so triangles along the edge of the known area stay flat instead of forming walls.
    const { nx, ny, known, h } = this.T
    const i = k % nx
    const j = (k / nx) | 0
    let sum = 0
    let count = 0
    for (let dj = -1; dj <= 1; dj++) {
      const jj = j + dj
      if (jj < 0 || jj >= ny) continue
      for (let di = -1; di <= 1; di++) {
        const ii = i + di
        if (ii < 0 || ii >= nx || (!di && !dj)) continue
        const m = jj * nx + ii
        if (known[m]) {
          sum += h[m]
          count++
        }
      }
    }
    return count ? sum / count : this.fogY
  }

  // Normal and curvature (discrete Laplacian of height: + in hollows, - on ridges).
  normalAt(k) {
    const { nx, ny, res } = this.T
    const i = k % nx
    const j = (k / nx) | 0
    const il = i > 0 ? k - 1 : k
    const ir = i < nx - 1 ? k + 1 : k
    const jd = j > 0 ? k - nx : k
    const ju = j < ny - 1 ? k + nx : k
    const p = this.pos
    const hx = (p[ir * 3 + 1] - p[il * 3 + 1]) / (((ir - il) || 1) * res)
    const hy = (p[ju * 3 + 1] - p[jd * 3 + 1]) / ((((ju - jd) / nx) || 1) * res)
    // world normal (-hx, -hy, 1) in three.js coordinates is (-hx, 1, hy)
    const len = Math.hypot(hx, 1, hy)
    this.nor[k * 3] = -hx / len
    this.nor[k * 3 + 1] = 1 / len
    this.nor[k * 3 + 2] = hy / len
    const h0 = p[k * 3 + 1]
    this.curv[k] = this.known[k] ? (p[il * 3 + 1] + p[ir * 3 + 1] + p[jd * 3 + 1] + p[ju * 3 + 1] - 4 * h0) : 0
  }

  // How surrounded by known ground a sample is (5 x 5 neighbourhood), 0 when unknown: the
  // shader cuts and fades the edge on this, so it is soft instead of following the grid.
  edgeAt(k) {
    const { nx, ny, known } = this.T
    if (!known[k]) {
      this.edge[k] = 0
      return
    }
    const i = k % nx
    const j = (k / nx) | 0
    let s = 0
    for (let dj = -2; dj <= 2; dj++) {
      const jj = j + dj
      if (jj < 0 || jj >= ny) continue
      const row = jj * nx
      for (let di = -2; di <= 2; di++) {
        const ii = i + di
        if (ii >= 0 && ii < nx && known[row + ii]) s++
      }
    }
    this.edge[k] = s / 25
  }

  writeKnown(k) {
    const T = this.T
    this.pos[k * 3 + 1] = T.h[k]
    this.slope[k] = T.slope[k]
    this.known[k] = 1
    this.reveal[k] = T.revealAt[k]
  }

  flagAll() {
    for (const attr of Object.values(this.attrs)) attr.needsUpdate = true
  }

  syncAll() {
    const T = this.T
    const n = T.nx * T.ny
    for (let k = 0; k < n; k++) if (T.known[k]) this.writeKnown(k)
    for (let k = 0; k < n; k++) if (!T.known[k]) this.pos[k * 3 + 1] = this.provisional(k)
    for (let k = 0; k < n; k++) this.normalAt(k)
    for (let k = 0; k < n; k++) this.edgeAt(k)
    for (let q = 0; q < T.dirtyCount; q++) T.dirtyFlag[T.dirty[q]] = 0
    T.dirtyCount = 0
    this.flagAll()
  }

  drain() {
    const T = this.T
    if (!T.dirtyCount) return false
    const { nx, ny } = T
    this.frame = (this.frame + 1) >>> 0 || 1
    const frame = this.frame
    let tc = 0
    for (let q = 0; q < T.dirtyCount; q++) {
      const k = T.dirty[q]
      T.dirtyFlag[k] = 0
      this.writeKnown(k)
      const i = k % nx
      const j = (k / nx) | 0
      for (let dj = -2; dj <= 2; dj++) {
        const jj = j + dj
        if (jj < 0 || jj >= ny) continue
        for (let di = -2; di <= 2; di++) {
          const ii = i + di
          if (ii < 0 || ii >= nx) continue
          const m = jj * nx + ii
          if (Math.abs(di) <= 1 && Math.abs(dj) <= 1 && !T.known[m]) this.pos[m * 3 + 1] = this.provisional(m)
          if (this.stamp[m] !== frame) {
            this.stamp[m] = frame
            this.touched[tc++] = m
          }
        }
      }
    }
    T.dirtyCount = 0
    for (let t = 0; t < tc; t++) {
      this.normalAt(this.touched[t])
      this.edgeAt(this.touched[t])
    }
    this.flagAll()
    return true
  }

  dispose() {
    this.geometry.dispose()
  }
}

// v7, Reality on with landforms present: the true ground as a terrain of its own (all of it
// known), drawn a little below the measured terrain, so mesas, the canyon, pits and dunes show
// in 3D before the fleet has mapped them.
function truthTerrain(truth) {
  const { nx, ny, res, ox, oy, heights } = truth
  const n = nx * ny
  const slope = new Float32Array(n)
  let minH = Infinity
  let maxH = -Infinity
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i
      const h = heights[k]
      if (h < minH) minH = h
      if (h > maxH) maxH = h
      const hx = (heights[j * nx + Math.min(i + 1, nx - 1)] - heights[j * nx + Math.max(i - 1, 0)]) / (2 * res)
      const hy = (heights[Math.min(j + 1, ny - 1) * nx + i] - heights[Math.max(j - 1, 0) * nx + i]) / (2 * res)
      slope[k] = (Math.atan(Math.hypot(hx, hy)) * 180) / Math.PI
    }
  }
  const width = (nx - 1) * res
  const depth = (ny - 1) * res
  return {
    nx, ny, res, ox, oy, width, depth, cx: ox + width / 2, cy: oy + depth / 2, minH, maxH,
    h: heights, slope, known: new Uint8Array(n).fill(1), revealAt: new Float32Array(n).fill(-1e6),
    dirty: new Int32Array(1), dirtyFlag: new Uint8Array(n), dirtyCount: 0,
  }
}

function TruthGround({ material }) {
  const truth = useApp((a) => a.truth)
  const gen = useSim((s) => s.gen)
  const mesh = useMemo(() => (truth && truth.heights && truth.gen === gen ? new TerrainMesh(truthTerrain(truth)) : null), [truth, gen])
  useEffect(() => () => mesh?.dispose(), [mesh])
  if (!mesh) return null
  return <mesh geometry={mesh.geometry} material={material} position={[0, -0.12, 0]} frustumCulled={false} renderOrder={0} receiveShadow />
}

export function Terrain({ slopeColors, reality = false }) {
  const landforms = useSim((s) => !!s.features && V7_LANDFORM_KEYS.some((k) => (Array.isArray(s.features[k]) ? s.features[k].length > 0 : !!s.features[k])))
  const terrain = useSim((s) => s.terrain)
  const roverMeta = useSim((s) => s.roverMeta)
  const graphics = useGraphics()
  const material = useMemo(() => createMaterial(graphics === 'high'), []) // eslint-disable-line react-hooks/exhaustive-deps
  const depthMaterial = useMemo(createDepthMaterial, [])
  const mesh = useMemo(() => (terrain ? new TerrainMesh(terrain) : null), [terrain])

  useEffect(() => () => mesh?.dispose(), [mesh])
  useEffect(() => () => (material.dispose(), depthMaterial.dispose()), [material, depthMaterial])
  useEffect(() => {
    const hq = graphics === 'high'
    if (('TERRAIN_HQ' in material.defines) === hq) return
    material.defines = hq ? { TERRAIN_HQ: '' } : {}
    material.needsUpdate = true
  }, [graphics, material])

  useFrame(() => {
    if (!mesh) return
    const s = simStore.get()
    if (s.terrain !== mesh.T) return
    mesh.drain()
    const u = material.userData.uniforms
    u.uTime.value = nowSeconds()
    u.uLimit.value = s.limit.current ?? 30
    u.uSlopeMode.value = slopeColors ? 1 : 0
    u.uMotion.value = reducedMotion ? 0 : 1
    let c = 0
    for (const m of roverMeta) {
      const p = roverPoses.get(m.id)
      if (!p || c >= 8) continue
      u.uRovers.value[c].set(p.x, p.z)
      c++
    }
    u.uRoverCount.value = c
    if (material.userData.features !== s.features || material.userData.world !== s.world) {
      if (material.userData.features !== s.features) syncLandforms(u, s.features)
      const was = material.userData.biomeKey
      const bm = s.features?.biome_map
      const key = `${s.world?.biome?.id}|${s.world?.biome?.secondary}|${bm ? bm.nx + ',' + bm.ny + ',' + bm.origin + ',' + bm.w?.length : ''}`
      if (key !== was || (bm && material.userData.biomeMap !== bm)) {
        syncBiome(u, s.world, s.features)
        material.userData.biomeKey = key
        material.userData.biomeMap = bm
      }
      material.userData.features = s.features
      material.userData.world = s.world
    }
    // Wet shoreline from the water layer (identified water only).
    const w = waterShared
    if (w.shore && w.map && u.uShore.value !== w.shore) {
      u.uShore.value = w.shore
      u.uShoreMap.value.set(w.map.ox, w.map.oy, 1 / w.map.res, 0)
      u.uShoreSize.value.set(w.map.nx, w.map.ny)
    } else if (!w.shore && u.uShore.value !== blackTexture()) {
      u.uShore.value = blackTexture()
      u.uShoreSize.value.set(1, 1)
    }
  })

  if (!mesh) return null
  return (
    <>
      <mesh geometry={mesh.geometry} material={material} customDepthMaterial={depthMaterial} frustumCulled={false} renderOrder={0} castShadow receiveShadow />
      {reality && landforms && <TruthGround material={material} />}
    </>
  )
}
