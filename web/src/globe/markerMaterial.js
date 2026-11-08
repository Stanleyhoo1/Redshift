import * as THREE from 'three'
import { GLOBE } from './globePalette.js'

// The landing marker, drawn on a square in the ground's plane: a crisp ring (radius half
// the square), a soft glow around it, a faint fill, the sector as a small square outline
// at the centre, and a pulse that runs outwards every few seconds. The ring adds light over
// a soft dark backing (premultiplied alpha), so it reads on bright dust and dark plains
// alike. HUD-like: no tone mapping, so it is the signal cyan.

export const MARKER_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv * 2.0 - 1.0;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

export const MARKER_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uTime;
uniform float uFade;
uniform float uHover;
uniform float uPulse;
varying vec2 vUv;
void main() {
  float r = length(vUv);
  float aa = fwidth(r) * 1.5;
  float ring = 1.0 - smoothstep(0.03, 0.03 + aa, abs(r - 0.5));
  float glow = exp(-pow((r - 0.5) / 0.13, 2.0)) * 0.5;
  float fill = (1.0 - smoothstep(0.0, 0.5, r)) * 0.08;
  vec2 a = abs(vUv);
  float box = 1.0 - smoothstep(0.012, 0.012 + fwidth(a.x + a.y) * 1.5, abs(max(a.x, a.y) - 0.1));
  float t = fract(uTime * 0.3);
  float pulse = uPulse * (1.0 - t) * exp(-pow((r - mix(0.5, 0.95, t)) / 0.035, 2.0)) * 0.55;
  float lum = (ring * 0.95 + glow + fill + box * 0.9 + pulse) * (1.0 + 0.5 * uHover);
  float shade = exp(-pow((r - 0.5) / 0.22, 2.0)) * 0.45 + (1.0 - smoothstep(0.0, 0.55, r)) * 0.15;
  float alpha = clamp(lum + shade, 0.0, 1.0) * uFade;
  if (alpha < 0.002) discard;
  gl_FragColor = vec4(uColor * (lum * uFade), alpha);
  #include <colorspace_fragment>
}
`

export function makeMarkerMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader: MARKER_VERT,
    fragmentShader: MARKER_FRAG,
    uniforms: {
      uColor: { value: new THREE.Color(GLOBE.signal) },
      uTime: { value: 0 },
      uFade: { value: 1 },
      uHover: { value: 0 },
      uPulse: { value: 1 },
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending, // with premultipliedAlpha: src ONE, dst ONE_MINUS_SRC_ALPHA
    premultipliedAlpha: true,
    toneMapped: false,
  })
}
