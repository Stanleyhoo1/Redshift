import * as THREE from 'three';

/**
 * Water surface shader.
 * Per vertex: `aDepth` = water surface minus ground (negative on dry vertices that border water, so the
 * shoreline is the depth = 0 contour, interpolated smoothly across each triangle instead of a 2 m staircase),
 * `aFlood` = 1 where the water is floodwater (muddy) rather than the river/lake.
 * Fragment: shallow->deep colour, mud tint, animated ripples (procedural normal), sun glint, shoreline foam.
 */
export function waterMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    fog: true,
    uniforms: THREE.UniformsUtils.merge([
      THREE.UniformsLib.fog,
      { uTime: { value: 0 }, uSun: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() }, uRain: { value: 0 } },
    ]),
    vertexShader: /* glsl */ `
      attribute float aDepth;
      attribute float aFlood;
      uniform float uTime;
      varying float vDepth;
      varying float vFlood;
      varying vec3 vWorld;
      #include <fog_pars_vertex>
      void main() {
        vDepth = aDepth; vFlood = aFlood;
        vec3 p = position;
        float wet = clamp(aDepth * 4.0, 0.0, 1.0);
        p.y += wet * (sin(p.x * 0.35 + uTime * 1.3) * 0.04 + sin(p.z * 0.5 - uTime * 1.1) * 0.03);
        vec4 wp = modelMatrix * vec4(p, 1.0);
        vWorld = wp.xyz;
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform vec3 uSun;
      uniform float uRain;
      varying float vDepth;
      varying float vFlood;
      varying vec3 vWorld;
      #include <fog_pars_fragment>
      float h(vec2 p) {
        return sin(p.x * 0.9 + uTime * 1.7) * 0.5 + sin(p.y * 1.3 - uTime * 1.3) * 0.5
             + sin((p.x + p.y) * 2.3 + uTime * 2.9) * 0.25 + sin((p.x - p.y) * 3.7 - uTime * 3.3) * 0.15;
      }
      void main() {
        if (vDepth <= 0.0) discard;
        vec2 q = vWorld.xz;
        float e = 0.15;
        vec3 n = normalize(vec3(-(h(q + vec2(e, 0.0)) - h(q - vec2(e, 0.0))) * (0.08 + uRain * 0.08), 1.0,
                                -(h(q + vec2(0.0, e)) - h(q - vec2(0.0, e))) * (0.08 + uRain * 0.08)));
        vec3 v = normalize(cameraPosition - vWorld);
        float d = clamp(vDepth / 1.6, 0.0, 1.0);
        vec3 shallow = vec3(0.30, 0.62, 0.66), deep = vec3(0.03, 0.17, 0.30);
        vec3 col = mix(shallow, deep, d);
        vec3 mud = mix(vec3(0.36, 0.29, 0.18), vec3(0.20, 0.15, 0.09), clamp(vDepth / 0.8, 0.0, 1.0));
        col = mix(col, mud, vFlood * 0.85);
        float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
        col = mix(col, vec3(0.70, 0.80, 0.90), fres * (0.45 + 0.2 * vFlood));
        vec3 r = reflect(-uSun, n);
        col += vec3(1.0, 0.95, 0.85) * pow(max(dot(r, v), 0.0), 90.0) * 0.9;
        // foam along the shoreline (and a moving flood front)
        float foam = smoothstep(0.045, 0.0, vDepth) * (0.55 + 0.45 * sin(q.x * 3.0 + q.y * 2.0 + uTime * 4.0));
        col = mix(col, mix(vec3(0.92, 0.93, 0.90), vec3(0.78, 0.70, 0.55), vFlood), foam * 0.6);
        float a = mix(0.62 + 0.3 * vFlood, 0.95, smoothstep(0.0, 0.6, vDepth));
        a = max(a, foam * 0.8);
        gl_FragColor = vec4(col, a);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
        #include <fog_fragment>
      }`,
  });
}

/** Gradient sky dome with a sun disk; `setSmoke` darkens it towards an orange-brown haze. */
export function skyDome(): { mesh: THREE.Mesh; set(smoke: number, storm: number): void } {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { uSmoke: { value: 0 }, uStorm: { value: 0 }, uSun: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() } },
    vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `
      uniform float uSmoke; uniform float uStorm; uniform vec3 uSun; varying vec3 vDir;
      void main(){
        float y = clamp(vDir.y, -0.1, 1.0);
        vec3 top = vec3(0.23, 0.48, 0.80), hor = vec3(0.72, 0.84, 0.94);
        vec3 col = mix(hor, top, pow(max(y, 0.0), 0.6));
        float s = max(dot(normalize(vDir), uSun), 0.0);
        col += vec3(1.0, 0.9, 0.7) * (pow(s, 600.0) * 2.0 + pow(s, 12.0) * 0.25);
        vec3 storm = mix(vec3(0.42, 0.46, 0.50), vec3(0.25, 0.28, 0.32), y);
        col = mix(col, storm, uStorm);
        vec3 smoke = mix(vec3(0.55, 0.42, 0.30), vec3(0.28, 0.24, 0.22), y);
        col = mix(col, smoke, uSmoke);
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(450, 32, 16), mat);
  mesh.renderOrder = -1;
  return { mesh, set(smoke, storm) { mat.uniforms.uSmoke.value = smoke; mat.uniforms.uStorm.value = storm; } };
}
