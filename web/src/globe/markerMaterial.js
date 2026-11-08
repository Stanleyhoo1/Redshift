import * as THREE from 'three'
import { GLOBE } from './globePalette.js'

// The landing marker, drawn on a square in the ground's plane: a crisp ring (radius half
// the square), a soft glow around it, a faint fill, the sector as a small square outline
// at the centre, and a pulse that runs outwards every few seconds. The ring adds light over
// a soft dark backing (premultiplied alpha), so it reads on bright dust and dark plains
// alike. HUD-like: no tone mapping, so it is the signal cyan. The square is draped on the
// sphere (radius uDrape), so a ring wide enough to frame the sector map hugs the ground.
// uInner scales the fill, the centre square and the dark backing inside the ring: 0 when
// the fleet's map (SectorMap.jsx) is drawn there instead. uThin narrows the line and glow
// when the ring is drawn larger than its usual size.

export const MARKER_VERT = /* glsl */ `
uniform float uDrape;
varying vec2 vUv;
void main() {
  vUv = uv * 2.0 - 1.0;
  vec4 wp = modelMatrix * vec4(position, 1.0);
  // The planet's centre is the world origin.
  if (uDrape > 0.0) wp.xyz = normalize(wp.xyz) * uDrape;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`

export const MARKER_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uTime;
uniform float uFade;
uniform float uHover;
uniform float uPulse;
uniform float uInner;
uniform float uThin;
varying vec2 vUv;
void main() {
  float r = length(vUv);
  float aa = fwidth(r) * 1.5;
  float ring = 1.0 - smoothstep(0.03 * uThin, 0.03 * uThin + aa, abs(r - 0.5));
  float glow = exp(-pow((r - 0.5) / (0.13 * uThin), 2.0)) * 0.5 * uThin;
  float fill = (1.0 - smoothstep(0.0, 0.5, r)) * 0.08;
  vec2 a = abs(vUv);
  float box = 1.0 - smoothstep(0.012, 0.012 + fwidth(a.x + a.y) * 1.5, abs(max(a.x, a.y) - 0.1));
  float t = fract(uTime * 0.3);
  float pulse = uPulse * (1.0 - t) * exp(-pow((r - mix(0.5, 0.95, t)) / 0.035, 2.0)) * 0.55;
  float inside = mix(1.0, uInner, 1.0 - smoothstep(0.36, 0.47, r));
  float lum = (ring * 0.95 + glow * mix(0.4, 1.0, inside) + (fill + box * 0.9) * uInner + pulse) * (1.0 + 0.5 * uHover);
  float shade = exp(-pow((r - 0.5) / (0.22 * uThin), 2.0)) * 0.45 * inside + (1.0 - smoothstep(0.0, 0.55, r)) * 0.15 * uInner;
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
      uInner: { value: 1 },
      uThin: { value: 1 }, // scales the ring's line, glow and backing widths (a big ring stays fine)
      uDrape: { value: 0 }, // 0: flat; otherwise the radius to drape the square on
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending, // with premultipliedAlpha: src ONE, dst ONE_MINUS_SRC_ALPHA
    premultipliedAlpha: true,
    toneMapped: false,
  })
}
