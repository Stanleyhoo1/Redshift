import * as THREE from 'three'
import { FLORA, TERRAIN } from '../lib/terrainTheme.js'

// Map markers for the sector view: one small camera-facing glyph per plant, geyser vent and
// boulder field, a constant size on screen, so they read from the overview where a 3 m tree is
// a few pixels. They fade out as the camera closes in (gone under MARKER_NEAR metres, full beyond
// MARKER_FAR), so close-ups show the real models. All of them are one instanced draw call.
// Pure helpers and the shader live here (no React, no store) so node tests can load them.

export const MARKER_NEAR = 24
export const MARKER_FAR = 52
export const KIND = { tree: 0, bush: 1, geyser: 2, boulders: 3 }
// Glyph size on screen (CSS pixels, the glyph itself; the quad adds a margin for the glow).
export const MARKER_PX = { tree: 18, bush: 14, geyser: 19, boulders: 17 }
export const QUAD_MARGIN = 1.35

const smooth = (t) => t * t * (3 - 2 * t)

// 0 when the camera is within MARKER_NEAR of the marker, 1 beyond MARKER_FAR, smooth between.
export function markerFade(dist, near = MARKER_NEAR, far = MARKER_FAR) {
  if (!Number.isFinite(dist)) return 0
  return smooth(Math.min(1, Math.max(0, (dist - near) / (far - near))))
}

// Keep-out rings (plant groups, geyser vents) only belong to a close look: from the overview a
// dashed circle round every group is clutter. 1 when the camera is within RING_NEAR metres of the
// ring's centre, 0 beyond RING_FAR, smooth between (the reverse of markerFade).
export const RING_NEAR = 45
export const RING_FAR = 80
export function ringFade(dist, near = RING_NEAR, far = RING_FAR) {
  if (!Number.isFinite(dist)) return 0
  return 1 - markerFade(dist, near, far)
}

// A '#rrggbb' colour as [r, g, b] in 0..1, as the screen shows it (the canvas is linear and
// unconverted, so these match the legend's CSS swatches exactly).
export function hexRgb(hex) {
  const n = parseInt(String(hex).replace('#', ''), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

const VIOLET = hexRgb(FLORA.violet)
const TEAL = hexRgb(FLORA.teal)

// A fungal tree's cap colour from its hue (0 violet .. 1 teal), as the 3D caps blend it.
export function floraGlyphColor(hue) {
  const h = Number.isFinite(hue) ? Math.min(1, Math.max(0, hue)) : 0
  return VIOLET.map((v, k) => v + (TEAL[k] - v) * h)
}

export const GLYPH_COLOR = {
  bush: hexRgb(FLORA.bulb),
  geyser: hexRgb(TERRAIN.geyser),
  boulders: hexRgb(TERRAIN.boulder),
}

// How bright a vent glyph pulses: 0 while the vent is quiet, 0.35..1 while it erupts.
export function ventPulse(active, t, still = false) {
  if (!active) return 0
  if (still) return 1
  return 0.675 + 0.325 * Math.sin(t * 4.2)
}

export const MAP_MARKER_VERT = /* glsl */ `
attribute vec3 aCenter;
attribute vec3 aColor;
attribute vec4 aData;
uniform vec2 uViewport;
uniform float uNear;
uniform float uFar;
uniform float uPull;
varying vec2 vUv;
varying vec3 vColor;
varying float vKind;
varying float vFade;
varying float vPulse;
void main() {
  vUv = position.xy * 2.0;
  vColor = aColor;
  vKind = aData.x;
  vPulse = aData.w;
  vec4 mv = viewMatrix * vec4(aCenter, 1.0);
  float dist = length(mv.xyz);
  float t = clamp((dist - uNear) / (uFar - uNear), 0.0, 1.0);
  vFade = t * t * (3.0 - 2.0 * t) * aData.z;
  // Slide the anchor a little towards the eye along its own ray: the same spot on screen, but
  // the ground right round it no longer cuts the glyph. Mesas and ridges still hide it.
  mv.xyz *= max(0.05, dist - uPull) / max(dist, 1e-3);
  vec4 clip = projectionMatrix * mv;
  clip.xy += position.xy * aData.y * 2.0 / uViewport * clip.w;
  gl_Position = clip;
}
`

export const MAP_MARKER_FRAG = /* glsl */ `
uniform float uTime;
varying vec2 vUv;
varying vec3 vColor;
varying float vKind;
varying float vFade;
varying float vPulse;

float sdCircle(vec2 p, vec2 c, float r) { return length(p - c) - r; }
float sdBox(vec2 p, vec2 c, vec2 b) {
  vec2 d = abs(p - c) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
// A dome: the top half of an ellipse, flat underneath.
float sdDome(vec2 p, vec2 c, vec2 r) {
  vec2 q = (p - c) / r;
  return max((length(q) - 1.0) * min(r.x, r.y), c.y - p.y);
}

void main() {
  if (vFade < 0.003) discard;
  // Glyph space: the glyph spans about -1..1, the quad a little more (for the glow).
  vec2 p = vUv * ${QUAD_MARGIN.toFixed(2)};
  float aa = fwidth(p.x) * 1.2;
  vec3 ink = vec3(0.06, 0.04, 0.035);
  vec3 col = vColor;
  float d;
  float glowK = 0.55;
  if (vKind < 0.5) {
    // Fungal tree: a wide cap on a pale stalk.
    float cap = sdDome(p, vec2(0.0, -0.12), vec2(0.92, 0.7));
    float stalk = sdBox(p, vec2(0.0, -0.5), vec2(0.15, 0.4));
    d = min(cap, stalk);
    float hi = clamp((p.y + 0.1) / 0.75, 0.0, 1.0);
    vec3 capCol = mix(vColor * 1.05, mix(vColor, vec3(1.0), 0.45), hi * hi);
    capCol = mix(capCol, vec3(1.0), 0.5 * (1.0 - smoothstep(0.0, 0.1, abs(p.y + 0.12)) ) * step(-0.12, p.y));
    col = cap < stalk ? capCol : vec3(0.93, 0.89, 0.8);
  } else if (vKind < 1.5) {
    // Bush: a clump of three glowing bulbs.
    float a = sdCircle(p, vec2(-0.42, -0.3), 0.42);
    float b = sdCircle(p, vec2(0.42, -0.3), 0.42);
    float c = sdCircle(p, vec2(0.0, 0.36), 0.42);
    d = min(a, min(b, c));
    float core = min(length(p - vec2(-0.5, -0.2)), min(length(p - vec2(0.34, -0.2)), length(p - vec2(-0.08, 0.46))));
    col = mix(vec3(1.0, 1.0, 0.85), vColor, smoothstep(0.0, 0.3, core));
    glowK = 0.8;
  } else if (vKind < 2.5) {
    // Geyser vent from above: an ochre sinter ring round a white throat, a halo while it erupts.
    float r = length(p);
    d = r - 0.78;
    col = mix(vec3(1.0, 0.98, 0.93), vColor, smoothstep(0.3, 0.38, r));
    col = mix(col, vColor * 0.55, smoothstep(0.66, 0.78, r));
    col *= 1.0 + 0.35 * vPulse;
    glowK = 0.45 + 0.9 * vPulse;
  } else {
    // Boulder field: three blocky stones.
    float a = sdBox(p, vec2(-0.42, -0.36), vec2(0.36, 0.28)) - 0.06;
    float b = sdBox(p, vec2(0.44, -0.4), vec2(0.3, 0.24)) - 0.06;
    float c = sdBox(p, vec2(0.02, 0.28), vec2(0.4, 0.3)) - 0.06;
    d = min(a, min(b, c));
    float top = clamp(p.y + 0.6, 0.0, 1.0);
    col = mix(vColor * 0.85, mix(vColor, vec3(1.0), 0.35), top);
    glowK = 0.0;
  }
  float fill = 1.0 - smoothstep(-aa, aa, d);
  float edge = 1.0 - smoothstep(0.13 - aa, 0.13 + aa, d);
  float glow = exp(-max(d, 0.0) * 9.0) * glowK * (1.0 - edge);
  vec3 c = mix(ink, col, fill);
  float alpha = max(edge * 0.92, glow * 0.6);
  vec3 outCol = mix(vColor, c, edge);
  if (vKind > 1.5 && vKind < 2.5 && vPulse > 0.0) {
    // An eruption ring that runs outwards.
    float t = fract(uTime * 0.9);
    float ring = exp(-pow((length(p) - mix(0.85, 1.3, t)) / 0.07, 2.0)) * (1.0 - t) * vPulse;
    outCol = mix(outCol, vec3(1.0, 0.96, 0.88), ring * (1.0 - edge));
    alpha = max(alpha, ring * 0.9);
  }
  alpha *= vFade;
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(outCol, alpha);
}
`

export function makeMapMarkerMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader: MAP_MARKER_VERT,
    fragmentShader: MAP_MARKER_FRAG,
    uniforms: {
      uViewport: { value: new THREE.Vector2(1280, 720) },
      uNear: { value: MARKER_NEAR },
      uFar: { value: MARKER_FAR },
      uPull: { value: 3 },
      uTime: { value: 0 },
    },
    transparent: true,
    depthWrite: false,
    depthTest: true,
    toneMapped: false,
  })
}

// The instanced quad all markers share, with room for `max` markers.
export function makeMapMarkerGeometry(max) {
  const g = new THREE.InstancedBufferGeometry()
  const quad = new THREE.PlaneGeometry(1, 1)
  g.index = quad.index
  g.setAttribute('position', quad.getAttribute('position'))
  g.setAttribute('aCenter', new THREE.InstancedBufferAttribute(new Float32Array(max * 3), 3).setUsage(THREE.DynamicDrawUsage))
  g.setAttribute('aColor', new THREE.InstancedBufferAttribute(new Float32Array(max * 3), 3).setUsage(THREE.DynamicDrawUsage))
  g.setAttribute('aData', new THREE.InstancedBufferAttribute(new Float32Array(max * 4), 4).setUsage(THREE.DynamicDrawUsage))
  g.instanceCount = 0
  return g
}
