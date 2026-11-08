// GLSL for the fog-of-war planet and the mission-sector patch. The sphere mapping in
// PATCH_VERTEX must match lib/geo.ts `sectorToSphere` exactly (lon = (u−½)·90°, lat = (v−½)·90°).

export const NOISE_GLSL = /* glsl */ `
float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}
float vnoise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
    mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y),
    f.z);
}
float fbm(vec3 p) {
  float a = 0.5, s = 0.0;
  for (int k = 0; k < 5; k++) { s += a * vnoise(p); p *= 2.07; a *= 0.5; }
  return s;
}
`

/** Grey noise + 15° graticule + scan sweep, lit from the upper left. `dir` = unit sphere normal. */
export const FOG_GLSL = /* glsl */ `
const float PI = 3.141592653589793;
const vec3 SUN = normalize(vec3(-0.55, 0.5, 0.68));
vec3 fogColor(vec3 dir, float time) {
  float n = fbm(dir * 3.2 + vec3(0.0, 0.0, time * 0.004));
  float n2 = fbm(dir * 22.0);
  vec3 base = mix(vec3(0.115, 0.108, 0.165), vec3(0.235, 0.225, 0.29), smoothstep(0.25, 0.8, n));
  base += (n2 - 0.5) * 0.045;
  float lat = asin(clamp(dir.y, -1.0, 1.0));
  float lon = atan(dir.x, dir.z);
  vec2 g = vec2(lon, lat) / (PI / 12.0);
  vec2 fw = max(fwidth(g), vec2(1e-4));
  vec2 gl = abs(fract(g - 0.5) - 0.5) / fw;
  float line = 1.0 - min(min(gl.x, gl.y), 1.0);
  base += line * vec3(0.07, 0.07, 0.12);
  float lambert = max(dot(dir, SUN), 0.0);
  base *= 0.42 + 0.78 * lambert;
  // scan sweep: a meridian band circling the planet with a fading wake
  float sweepLon = mod(time * 0.35, 2.0 * PI) - PI;
  float d = mod(sweepLon - lon + 2.0 * PI, 2.0 * PI);
  float wake = exp(-d * 5.0) * 0.55 + exp(-d * d * 900.0);
  base += wake * vec3(0.05, 0.11, 0.17) * (0.35 + 0.65 * lambert);
  return base;
}
`

export const PLANET_VERTEX = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

export const PLANET_FRAGMENT = /* glsl */ `
uniform float uTime;
varying vec3 vDir;
${NOISE_GLSL}
${FOG_GLSL}
void main() {
  gl_FragColor = vec4(fogColor(normalize(vDir), uTime), 1.0);
}
`

/** Fresnel rim; used front-side (inner rim) and back-side (outer halo). */
export const ATMO_VERTEX = /* glsl */ `
varying vec3 vNormalV;
varying vec3 vViewV;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vNormalV = normalize(normalMatrix * normal);
  vViewV = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}
`
export const ATMO_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uPower;
uniform float uStrength;
uniform float uBack;
varying vec3 vNormalV;
varying vec3 vViewV;
void main() {
  float f = dot(normalize(vNormalV), normalize(vViewV));
  // back side (outer halo): brightest at the planet limb (f ≈ −0.55 for a 1.2 shell), 0 at its edge
  float rim = uBack > 0.5 ? pow(clamp(-f * 1.8, 0.0, 1.0), uPower) : pow(1.0 - clamp(f, 0.0, 1.0), uPower);
  gl_FragColor = vec4(uColor * rim * uStrength, rim);
}
`

// ---- sector patch ---------------------------------------------------------------------------

const PATCH_MAPPING = /* glsl */ `
const float HALF_PI = 1.5707963267948966;
vec3 patchDir(vec2 uv) {
  float lon = (uv.x - 0.5) * HALF_PI;
  float lat = (uv.y - 0.5) * HALF_PI;
  return vec3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
}
float decodeClass(float g) { return floor(g * 255.0 / 40.0 + 0.5); }
`

export const PATCH_VERTEX = /* glsl */ `
uniform sampler2D uMap;
uniform float uHasMap;
uniform float uExag;
uniform float uLift;
varying vec2 vUv;
varying vec3 vDir;
${PATCH_MAPPING}
void main() {
  vUv = uv;
  vec4 m = texture2D(uMap, uv);
  float known = uHasMap * step(0.5, decodeClass(m.g));
  vec3 dir = patchDir(uv);
  vDir = dir;
  vec3 p = dir * (1.0 + uLift + m.r * known * uExag);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}
`

export const PATCH_FRAGMENT = /* glsl */ `
uniform sampler2D uMap;
uniform float uHasMap;
uniform float uTime;
uniform vec2 uCells;
uniform float uElevRange;
uniform float uRes;
varying vec2 vUv;
varying vec3 vDir;
${PATCH_MAPPING}
${NOISE_GLSL}
${FOG_GLSL}

vec3 elevRamp(float e) {
  vec3 basalt = vec3(0.17, 0.12, 0.13);
  vec3 rust = vec3(0.56, 0.22, 0.12);
  vec3 ochre = vec3(0.86, 0.6, 0.3);
  return e < 0.55 ? mix(basalt, rust, smoothstep(0.0, 0.55, e)) : mix(rust, ochre, smoothstep(0.55, 1.0, e));
}

float sampleH(vec2 uv, float fallback) {
  vec4 s = texture2D(uMap, uv);
  return decodeClass(s.g) > 0.5 ? s.r : fallback;
}

void main() {
  vec3 dir = normalize(vDir);
  vec2 cellCoord = vUv * uCells;
  vec2 fw = max(fwidth(cellCoord), vec2(1e-4));
  // faint cell grid, fading out when cells shrink below ~3 px; stronger every 10 cells
  vec2 gd = abs(fract(cellCoord - 0.5) - 0.5) / fw;
  float cellLine = (1.0 - min(min(gd.x, gd.y), 1.0)) * (1.0 - smoothstep(0.18, 0.4, max(fw.x, fw.y)));
  vec2 c10 = cellCoord / 10.0;
  vec2 gd10 = abs(fract(c10 - 0.5) - 0.5) / max(fwidth(c10), vec2(1e-4));
  float majorLine = 1.0 - min(min(gd10.x, gd10.y), 1.0);

  vec4 m = texture2D(uMap, vUv);
  float cls = uHasMap * decodeClass(m.g);
  vec3 col;

  if (cls < 0.5) {
    // unexplored sector: the planet's fog with a cool tint and survey grid, so the target region reads
    col = fogColor(dir, uTime) * vec3(0.96, 1.0, 1.12) + vec3(0.012, 0.018, 0.04);
    col += vec3(0.05, 0.065, 0.11) * cellLine + vec3(0.07, 0.1, 0.17) * majorLine;
  } else {
    float e = m.r;
    vec2 tx = 1.0 / uCells;
    float hl = sampleH(vUv - vec2(tx.x, 0.0), e);
    float hr = sampleH(vUv + vec2(tx.x, 0.0), e);
    float hd = sampleH(vUv - vec2(0.0, tx.y), e);
    float hu = sampleH(vUv + vec2(0.0, tx.y), e);
    float k = uElevRange / (2.0 * uRes) * 2.2; // vertical exaggeration for readable relief
    vec3 n = normalize(vec3((hl - hr) * k, (hd - hu) * k, 1.0));
    float shade = clamp(dot(n, normalize(vec3(-0.5, 0.55, 0.68))), 0.0, 1.0);
    float lambert = max(dot(dir, SUN), 0.0);
    col = elevRamp(e) * (0.3 + 0.95 * shade) * (0.55 + 0.6 * lambert);

    float pulse = 0.5 + 0.5 * sin(uTime * 5.0);
    if (cls > 1.5 && cls < 2.5) {            // caution
      col = mix(col, vec3(1.0, 0.7, 0.22), 0.34);
    } else if (cls > 2.5 && cls < 3.5) {     // hazard (slope / step)
      col = mix(col, vec3(0.95, 0.24, 0.32), 0.34 + 0.16 * pulse);
    } else if (cls > 3.5 && cls < 4.5) {     // permanent hazard, hatched
      float hatch = step(0.5, fract((cellCoord.x + cellCoord.y) * 0.5));
      col = mix(vec3(0.42, 0.04, 0.1), vec3(0.85, 0.12, 0.22), hatch * (0.6 + 0.4 * pulse));
    } else if (cls > 4.5 && cls < 5.5) {     // science target
      col = mix(col, vec3(0.36, 0.95, 0.9), 0.75) + vec3(0.1, 0.35, 0.35) * pulse;
    } else if (cls > 5.5) {                  // obstacle / rock
      col = mix(col, vec3(0.36, 0.2, 0.22), 0.8);
    }
    col += vec3(0.04, 0.03, 0.03) * cellLine;
    // fresh reveal glow (B channel decays 255 → 0 over 6 s)
    col += vec3(0.25, 0.85, 1.0) * pow(m.b, 1.6) * 0.75;
  }

  // sector border
  vec2 edge = min(vUv, 1.0 - vUv);
  float ed = min(edge.x, edge.y);
  float border = 1.0 - smoothstep(0.0, 0.004, ed);
  float halo = (1.0 - smoothstep(0.0, 0.03, ed)) * 0.12;
  col = mix(col, vec3(0.3, 0.76, 1.0), border * 0.85) + vec3(0.1, 0.25, 0.4) * halo;
  gl_FragColor = vec4(col, 1.0);
}
`

// ---- dust storm shell -------------------------------------------------------------------------
export const DUST_VERTEX = /* glsl */ `
uniform float uLift;
varying vec2 vUv;
varying vec3 vDir;
${PATCH_MAPPING}
void main() {
  vUv = uv;
  vec3 dir = patchDir(uv);
  vDir = dir;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(dir * (1.0 + uLift), 1.0);
}
`
/**
 * uStorm 0 = brown dust haze, 1 = the dust storm's orange-red (sensor_effects STORM_RGB 205, 88, 48), denser.
 * uTime is the haze's own clock (DustShell runs it faster in a storm).
 */
export const DUST_FRAGMENT = /* glsl */ `
uniform float uTime;
uniform float uLevel;
uniform float uStorm;
varying vec2 vUv;
varying vec3 vDir;
${NOISE_GLSL}
void main() {
  vec3 p = vDir * 9.0 + vec3(uTime * 0.18, uTime * 0.05, -uTime * 0.11);
  float n = fbm(p) * 0.75 + fbm(p * 2.7 - uTime * 0.3) * 0.25;
  vec2 edge = min(vUv, 1.0 - vUv);
  float fade = smoothstep(0.0, 0.12, min(edge.x, edge.y));
  float a = clamp(uLevel * (0.2 + 0.8 * smoothstep(0.35, 0.8, n)) * fade, 0.0, 1.0) * mix(0.55, 0.68, uStorm);
  vec3 dust = mix(vec3(0.62, 0.42, 0.24), vec3(0.85, 0.62, 0.38), n);
  vec3 storm = mix(vec3(0.62, 0.24, 0.12), vec3(0.95, 0.42, 0.22), n);
  gl_FragColor = vec4(mix(dust, storm, uStorm), a);
}
`
