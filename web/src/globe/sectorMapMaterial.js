import * as THREE from 'three'
import { GLOBE } from './globePalette.js'

// The fleet's map on the planet (SectorMap.jsx): the map texture (sRGB, transparent where
// the ground is unknown) on the enlarged patch, dimmed past the terminator, inside a thin
// cyan frame that shows the sector's extent even before anything is mapped. Map-like: no
// tone mapping, so the ground keeps the map legend's colours.

export const SECTOR_VERT = /* glsl */ `
attribute vec2 aLocal;
varying vec2 vUv;
varying vec2 vLocal;
varying vec3 vN;
void main() {
  vUv = uv;
  vLocal = aLocal;
  vN = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

export const SECTOR_FRAG = /* glsl */ `
uniform sampler2D uMap;
uniform vec3 uSun;
uniform vec3 uEdge;
uniform float uEdgeI;
varying vec2 vUv;
varying vec2 vLocal;
varying vec3 vN;
void main() {
  // Texels are opaque or transparent black, so a filtered sample is premultiplied: divide
  // the alpha out, or the edge of the mapped area would darken into a rim.
  vec4 m = texture2D(uMap, vUv);
  float day = smoothstep(-0.12, 0.25, dot(normalize(vN), normalize(uSun)));
  vec3 col = m.rgb / max(m.a, 1e-3) * mix(0.2, 1.0, day);
  vec2 e = min(vLocal, 1.0 - vLocal);
  float d = min(e.x, e.y);
  float aa = max(fwidth(d), 1e-5);
  // The frame over the map (a little weaker over mapped ground).
  float ea = (1.0 - smoothstep(aa * 0.8, aa * 1.8, d)) * uEdgeI * (1.0 - 0.4 * m.a);
  float a = ea + m.a * (1.0 - ea);
  if (a < 0.003) discard;
  gl_FragColor = vec4((uEdge * ea + col * m.a * (1.0 - ea)) / a, a);
  #include <colorspace_fragment>
}
`

export function makeSectorMapMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader: SECTOR_VERT,
    fragmentShader: SECTOR_FRAG,
    uniforms: {
      uMap: { value: null },
      uSun: { value: new THREE.Vector3(0, 0, 1) },
      uEdge: { value: new THREE.Color(GLOBE.signal) },
      uEdgeI: { value: 0.55 },
    },
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    toneMapped: false,
  })
}

// An RGBA map texture nx by ny over `data` (sRGB bytes, row 0 to the south), mipmapped so
// it stays calm when the patch is small on screen.
export function makeSectorTexture(data, nx, ny) {
  const tex = new THREE.DataTexture(data, nx, ny, THREE.RGBAFormat, THREE.UnsignedByteType)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.magFilter = THREE.LinearFilter
  tex.minFilter = THREE.LinearMipmapLinearFilter
  tex.generateMipmaps = true
  tex.wrapS = THREE.ClampToEdgeWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  tex.needsUpdate = true
  return tex
}
