import * as THREE from 'three'

// Stars as round, soft points with a bright core. HUD-like: no tone mapping, so a star is
// the colour it says. Twinkle (High graphics, no reduced motion) is a slow, small flicker.

export const STAR_VERT = /* glsl */ `
attribute float aMag;
attribute vec3 aTint;
attribute float aPhase;
uniform float uTime;
uniform float uPx;
uniform float uTwinkle;
varying vec3 vCol;
void main() {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  float tw = 1.0 + uTwinkle * 0.2 * sin(uTime * (0.9 + aPhase * 0.25) + aPhase * 5.0);
  gl_PointSize = uPx * mix(1.2, 3.6, aMag);
  vCol = aTint * mix(0.14, 1.0, aMag) * tw;
}
`

export const STAR_FRAG = /* glsl */ `
varying vec3 vCol;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  gl_FragColor = vec4(vCol * exp(-r2 * 5.0), 1.0);
  #include <colorspace_fragment>
}
`

export function makeStarMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader: STAR_VERT,
    fragmentShader: STAR_FRAG,
    uniforms: { uTime: { value: 0 }, uPx: { value: 1 }, uTwinkle: { value: 0 } },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  })
}
