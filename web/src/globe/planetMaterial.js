import * as THREE from 'three'
import { GLOBE } from './globePalette.js'

// The planet's surface and its air as three.js ShaderMaterials (WebGL2, GLSL ES 3.0; three
// adds the precision, the matrices and cameraPosition). Both end with three's tone mapping
// and colour space chunks, so the planet gets the sector's ACES exposure.
// Noise: gnoised() is Inigo Quilez's 3D gradient noise with analytic derivatives (MIT,
// iquilezles.org/articles/gradientnoise); hash33() is Dave Hoskins' "Hash without Sine"
// (MIT). The rest is ours.

// Look knobs (Task 15 tunes these on screenshots).
export const LOOK = {
  sunI: 1.95, // sunlight on the day side
  lommel: 0.36, // share of Lommel-Seeliger (dusty ground, bright to the limb) against Lambert
  relief: 0.03, // bump strength of the fine fractal noise
  craterRelief: 0.3, // bump strength of the craters
  haze: 0.45, // air over the day side of the disc, strongest at the limb
  night: 1.4, // starlight on the night side
  airH: 0.016, // air scale height, planet radii (a thick, hazy atmosphere)
  airI: 1.1, // brightness of the glow on the limb
  gridI: 0.06, // the unsurveyed planet's graticule
  revealS: 0.6, // seconds to fade between the unsurveyed and the true planet
}
export const AIR_R = 1.06 // the air shell's radius

// Shared by the planet and its air: the position on the sphere and the camera, both in
// the planet's own frame (the planet group spins, so modelMatrix is a pure rotation).
export const GLOBE_VERT = /* glsl */ `
varying vec3 vPos;
varying vec3 vCam;
void main() {
  vPos = position;
  vCam = transpose(mat3(modelMatrix)) * (cameraPosition - modelMatrix[3].xyz);
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}
`

const NOISE = /* glsl */ `
vec3 hash33(vec3 p3) {
  p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// Gradient noise, about -1 to 1: x is the value, yzw its gradient.
vec4 gnoised(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec3 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  vec3 ga = hash33(i + vec3(0.0, 0.0, 0.0)) * 2.0 - 1.0;
  vec3 gb = hash33(i + vec3(1.0, 0.0, 0.0)) * 2.0 - 1.0;
  vec3 gc = hash33(i + vec3(0.0, 1.0, 0.0)) * 2.0 - 1.0;
  vec3 gd = hash33(i + vec3(1.0, 1.0, 0.0)) * 2.0 - 1.0;
  vec3 ge = hash33(i + vec3(0.0, 0.0, 1.0)) * 2.0 - 1.0;
  vec3 gf = hash33(i + vec3(1.0, 0.0, 1.0)) * 2.0 - 1.0;
  vec3 gg = hash33(i + vec3(0.0, 1.0, 1.0)) * 2.0 - 1.0;
  vec3 gh = hash33(i + vec3(1.0, 1.0, 1.0)) * 2.0 - 1.0;
  float va = dot(ga, f - vec3(0.0, 0.0, 0.0));
  float vb = dot(gb, f - vec3(1.0, 0.0, 0.0));
  float vc = dot(gc, f - vec3(0.0, 1.0, 0.0));
  float vd = dot(gd, f - vec3(1.0, 1.0, 0.0));
  float ve = dot(ge, f - vec3(0.0, 0.0, 1.0));
  float vf = dot(gf, f - vec3(1.0, 0.0, 1.0));
  float vg = dot(gg, f - vec3(0.0, 1.0, 1.0));
  float vh = dot(gh, f - vec3(1.0, 1.0, 1.0));
  float v = va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
    + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg)
    + u.z * u.x * (va - vb - ve + vf) + (-va + vb + vc - vd + ve - vf - vg + vh) * u.x * u.y * u.z;
  vec3 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
    + u.x * u.y * (ga - gb - gc + gd) + u.y * u.z * (ga - gc - ge + gg)
    + u.z * u.x * (ga - gb - ge + gf) + (-ga + gb + gc - gd + ge - gf - gg + gh) * u.x * u.y * u.z
    + du * (vec3(vb - va, vc - va, ve - va)
      + u.yzx * vec3(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
      + u.zxy * vec3(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg)
      + u.yzx * u.zxy * (-va + vb + vc - vd + ve - vf - vg + vh));
  return vec4(v, d);
}

// Fractal noise with its gradient; each octave rotated so the lattice never lines up.
const mat3 OCT = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);
vec4 fbmd(vec3 p, int octaves) {
  float v = 0.0;
  vec3 d = vec3(0.0);
  float a = 0.5;
  mat3 m = mat3(1.0); // transpose of d(octave coordinates) / dp so far
  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    vec4 n = gnoised(p);
    v += a * n.x;
    d += a * (m * n.yzw);
    a *= 0.5;
    p = OCT * p * 2.02;
    m = 2.02 * m * transpose(OCT);
  }
  return vec4(v, d);
}

// One layer of craters: at most one per cell of a 3D lattice (scale cells per planet
// radius), radius up to a third of a cell. Adds the height gradient (a bowl and a raised
// rim) to g; returns how much of p lies on a crater floor (x) and on a rim (y).
vec2 craters(vec3 p, float scale, float density, float salt, inout vec3 g) {
  vec3 q = p * scale;
  vec3 base = floor(q - 0.5);
  vec2 marks = vec2(0.0);
  for (int k = 0; k < 8; k++) {
    vec3 cell = base + vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1));
    vec3 jit = hash33(cell + salt);
    vec3 rnd = hash33(cell.zxy + salt * 1.7 + 41.3);
    if (rnd.x > density) continue;
    float R = mix(0.12, 0.34, rnd.y * rnd.y);
    vec3 dv = q - (cell + 0.25 + 0.5 * jit);
    float dist = length(dv);
    float x = dist / R;
    if (x >= 1.32) continue;
    float depth = 0.42 * R / scale;
    float slope = 0.0;
    if (x < 1.0) {
      slope += depth * 2.0 * x;
      marks.x = max(marks.x, 1.0 - x * x);
    }
    float s = (x - 1.0) / 0.32;
    if (abs(s) < 1.0) {
      float w = 1.0 - s * s;
      slope += 0.3 * depth * 2.0 * w * (-2.0 * s) / 0.32;
      marks.y = max(marks.y, w * w);
    }
    g += slope * dv * (scale / (R * max(dist, 1e-4)));
  }
  return marks;
}
`

export const PLANET_FRAG = /* glsl */ `
uniform vec3 uSun;
uniform vec3 uOffset;
uniform float uMaria;
uniform float uCraters;
uniform float uCapN;
uniform float uCapS;
uniform float uRust;
uniform float uRelief;
uniform float uCraterRelief;
uniform float uSunI;
uniform float uLommel;
uniform float uHazeI;
uniform float uNightI;
uniform vec3 uDust;
uniform vec3 uDustLight;
uniform vec3 uRustCol;
uniform vec3 uMare;
uniform vec3 uMareDark;
uniform vec3 uIce;
uniform vec3 uSunCol;
uniform vec3 uTwilight;
uniform vec3 uHaze;
uniform vec3 uNight;
uniform float uReveal;
uniform vec3 uBlank;
uniform vec3 uGrid;
uniform float uGridI;
varying vec3 vPos;
varying vec3 vCam;

#ifdef GLOBE_HQ
  #define OCTAVES 7
#else
  #define OCTAVES 4
#endif
#define FREQ 2.2
${NOISE}
// The surveyed (true) planet: the full procedural surface, lit.
vec3 revealed(vec3 n0, vec3 L, vec3 V) {
  // Continents: a slow noise sets the dark plains (maria) and where rust stains the dust.
  vec4 low = fbmd(n0 * 1.1 + uOffset * 0.37, 3);
  float mare = smoothstep(uMaria, uMaria + 0.12, low.x);
  float rough = 1.0 - 0.65 * mare; // the plains are smooth lava

  // Relief, as a height gradient for the lighting: fine noise plus craters (fewer on the
  // younger maria). Graphics Low keeps four octaves and the biggest craters only.
  vec4 hi = fbmd(n0 * FREQ + uOffset, OCTAVES);
  vec3 g = hi.yzw * (FREQ * uRelief * rough);
  vec3 cg = vec3(0.0);
  vec2 crater = craters(n0, 7.0, uCraters * (1.0 - 0.6 * mare), 3.17, cg);
#ifdef GLOBE_HQ
  crater = max(crater, craters(n0, 19.0, uCraters * (1.0 - 0.4 * mare), 7.73, cg));
  crater = max(crater, craters(n0, 47.0, uCraters, 13.31, cg));
#endif
  g += cg * uCraterRelief;

  // Albedo: dust and paler highland dust, rust stains, the dark plains, crater marks.
  float tone = smoothstep(-0.35, 0.45, hi.x * 1.4 + gnoised(n0 * 5.3 + uOffset.zxy).x * 0.35);
  vec3 alb = mix(uDust, uDustLight, tone);
  float rust = uRust * smoothstep(-0.05, 0.35, gnoised(n0 * 2.6 + uOffset.yzx).x * 0.4 - low.x);
  alb = mix(alb, uRustCol, rust * 0.75);
  alb = mix(alb, mix(uMare, uMareDark, smoothstep(uMaria, uMaria + 0.35, low.x)), mare);
  alb *= 1.0 - 0.16 * crater.x + 0.1 * crater.y * (1.0 - mare);

  // Polar ice with ragged edges, and a frost fringe on the bright ground around it.
  float rag = 0.05 * low.x + 0.02 * hi.x + 0.025 * gnoised(n0 * 13.0 + uOffset.yzx).x;
  float edgeN = uCapN + rag;
  float edgeS = uCapS - rag;
  float cap = max(smoothstep(edgeN, edgeN + 0.018, n0.y), smoothstep(edgeS, edgeS + 0.018, -n0.y));
  float frost = max(smoothstep(edgeN - 0.08, edgeN, n0.y), smoothstep(edgeS - 0.08, edgeS, -n0.y)) * (1.0 - cap);
  alb = mix(alb, mix(alb, uIce, 0.5), frost * tone);
  alb = mix(alb, uIce * (0.86 + 0.14 * tone), cap);
  g *= 1.0 - 0.75 * cap;

  // Light. The bumped normal shades the relief; the geometric day side sets the terminator.
  vec3 N = normalize(n0 - (g - dot(g, n0) * n0));
  float geo = dot(n0, L);
  float mu0 = max(dot(N, L), 0.0);
  float mu = max(dot(N, V), 0.02);
  float brdf = mix(mu0, 2.0 * mu0 / (mu0 + mu), uLommel);
  float day = smoothstep(-0.10, 0.22, geo); // soft: the thick air carries light past the edge
  vec3 col = alb * uSunCol * (uSunI * brdf * day);
  // A warm band at the terminator, and faint starlight on the night side.
  float dusk = smoothstep(-0.18, 0.02, geo) * (1.0 - smoothstep(0.02, 0.26, geo));
  col += alb * uTwilight * (0.16 * dusk);
  col += alb * uNight * uNightI;
  // The air over the disc: haze builds towards the day limb.
  float limb = pow(1.0 - max(dot(n0, V), 0.0), 3.0);
  col += uHaze * (uHazeI * limb * smoothstep(-0.2, 0.4, geo));
  // A faint glint on the ice.
  vec3 H = normalize(L + V);
  col += uSunCol * (0.1 * cap * day * pow(max(dot(N, H), 0.0), 36.0));
  return col;
}

void main() {
  vec3 n0 = normalize(vPos);
  vec3 L = normalize(uSun);
  vec3 V = normalize(vCam - n0);

  // Unsurveyed (Reality off): no surface detail, only a very dark, smooth grey ball with a
  // soft terminator, the air's haze on the day limb, and a faint graticule every 15 degrees.
  float geo = dot(n0, L);
  float day = smoothstep(-0.10, 0.22, geo);
  float mu0 = max(geo, 0.0);
  float mu = max(dot(n0, V), 0.02);
  float brdf = mix(mu0, 2.0 * mu0 / (mu0 + mu), uLommel);
  vec3 blank = uBlank * uSunCol * (uSunI * brdf * day) + uBlank * uNight * uNightI;
  float limb = pow(1.0 - max(dot(n0, V), 0.0), 3.0);
  blank += uHaze * (uHazeI * 0.3 * limb * smoothstep(-0.2, 0.4, geo));
  float lat = asin(clamp(n0.y, -1.0, 1.0)) * (12.0 / 3.14159265); // in 15 degree steps
  float lon = atan(n0.x, n0.z) * (12.0 / 3.14159265);
  float gLat = abs(fract(lat + 0.5) - 0.5) / max(fwidth(lat), 1e-4);
  // fwidth jumps where atan wraps (on a meridian anyway): fall back to the latitude's.
  float wLon = fwidth(lon);
  wLon = wLon > 6.0 ? fwidth(lat) / max(sqrt(1.0 - n0.y * n0.y), 0.05) : wLon;
  float gLon = abs(fract(lon + 0.5) - 0.5) / max(wLon, 1e-4);
  float lineLat = 1.0 - smoothstep(0.4, 1.3, gLat);
  float lineLon = (1.0 - smoothstep(0.4, 1.3, gLon)) * (1.0 - smoothstep(0.86, 0.97, abs(n0.y)));
  float grid = max(lineLat, lineLon) * (0.45 + 0.55 * day) * (1.0 - 0.7 * limb);
  blank += uGrid * (uGridI * grid);

  vec3 col = blank;
  if (uReveal > 0.001) col = mix(blank, revealed(n0, L, V), uReveal);

  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

// The glow of the air around the limb: for each view ray past the planet, the air density
// at its closest approach (exponential with height), lit where the sun is up there, with a
// little forward scattering when the sun is behind the planet. Peach low, mauve higher,
// dusk past the terminator, a trace of night glow on the dark limb.
export const AIR_FRAG = /* glsl */ `
uniform vec3 uSun;
uniform vec3 uLow;
uniform vec3 uHigh;
uniform vec3 uDusk;
uniform float uH;
uniform float uR;
uniform float uI;
varying vec3 vPos;
varying vec3 vCam;
void main() {
  vec3 rd = normalize(vPos - vCam);
  vec3 q = vCam + rd * max(-dot(vCam, rd), 0.0);
  float b = length(q);
  float alt = max(b - 1.0, 0.0);
  float dens = exp(-alt / uH) * smoothstep(uR, uR - 0.012, b);
  vec3 L = normalize(uSun);
  float sunUp = dot(q / max(b, 1e-4), L);
  float lit = smoothstep(-0.32, 0.28, sunUp);
  float fwd = 1.0 + 2.5 * pow(max(dot(rd, L), 0.0), 8.0);
  vec3 air = mix(uLow, uHigh, smoothstep(0.0, 3.5 * uH, alt));
  air = mix(uDusk, air, smoothstep(-0.12, 0.3, sunUp));
  vec3 c = air * (dens * lit * fwd * uI) + uDusk * (dens * 0.05);
  gl_FragColor = vec4(c, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

const col = (hex) => new THREE.Color(hex)

export function makePlanetMaterial(high) {
  return new THREE.ShaderMaterial({
    vertexShader: GLOBE_VERT,
    fragmentShader: PLANET_FRAG,
    defines: high ? { GLOBE_HQ: '' } : {},
    uniforms: {
      uSun: { value: new THREE.Vector3(0, 0, 1) },
      uOffset: { value: new THREE.Vector3() },
      uMaria: { value: 0.12 },
      uCraters: { value: 0.6 },
      uCapN: { value: 0.88 },
      uCapS: { value: 0.87 },
      uRust: { value: 0.5 },
      uRelief: { value: LOOK.relief },
      uCraterRelief: { value: LOOK.craterRelief },
      uSunI: { value: LOOK.sunI },
      uLommel: { value: LOOK.lommel },
      uHazeI: { value: LOOK.haze },
      uNightI: { value: LOOK.night },
      uDust: { value: col(GLOBE.dust) },
      uDustLight: { value: col(GLOBE.dustLight) },
      uRustCol: { value: col(GLOBE.rust) },
      uMare: { value: col(GLOBE.mare) },
      uMareDark: { value: col(GLOBE.mareDark) },
      uIce: { value: col(GLOBE.ice) },
      uSunCol: { value: col(GLOBE.sunLight) },
      uTwilight: { value: col(GLOBE.twilight) },
      uHaze: { value: col(GLOBE.haze) },
      uNight: { value: col(GLOBE.night) },
      uReveal: { value: 0 }, // 0 unsurveyed (the fleet's view), 1 the true planet (Reality)
      uBlank: { value: col(GLOBE.blank) },
      uGrid: { value: col(GLOBE.grid) },
      uGridI: { value: LOOK.gridI },
    },
  })
}

export function makeAirMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader: GLOBE_VERT,
    fragmentShader: AIR_FRAG,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uSun: { value: new THREE.Vector3(0, 0, 1) },
      uLow: { value: col(GLOBE.airLow) },
      uHigh: { value: col(GLOBE.airHigh) },
      uDusk: { value: col(GLOBE.airDusk) },
      uH: { value: LOOK.airH },
      uR: { value: AIR_R },
      uI: { value: LOOK.airI },
    },
  })
}

// Per-world values from planetLook() (globeMath.js).
export function applyLook(planet, air, look) {
  const u = planet.uniforms
  const s = look.surface
  u.uSun.value.set(look.sun[0], look.sun[1], look.sun[2]).normalize()
  u.uOffset.value.set(s.offset[0], s.offset[1], s.offset[2])
  u.uMaria.value = s.maria
  u.uCraters.value = s.craters
  u.uCapN.value = s.capN
  u.uCapS.value = s.capS
  u.uRust.value = s.rust
  air.uniforms.uSun.value.copy(u.uSun.value)
}
