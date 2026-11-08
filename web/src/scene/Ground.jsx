import { useEffect, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Grid } from '@react-three/drei'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { ATMOSPHERE, VOID } from '../lib/theme.js'
import { FatPolyline } from './fatLine.js'
import { atmo, fogYFor } from './sceneConstants.js'

// The uncharted planet: a dark floor with a faint survey grid, sitting just below the
// lowest terrain so revealed ground always rises out of it. It stays dark near the map
// (unexplored) and only far away fades into the haze over the horizon, so a low camera
// sees the plain meet the sky instead of a black edge.

const srgb = (hex) => {
  const c = new THREE.Color(hex).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace)
  return new THREE.Vector3(c.r, c.g, c.b)
}

function floorMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uVoid: { value: srgb(VOID) },
      uHaze: { value: srgb(ATMOSPHERE.groundHaze) },
      uDust: { value: srgb(ATMOSPHERE.dust) },
      uDustAmt: { value: 0 },
      uNear: { value: 170 },
      uFar: { value: 900 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vW;
      void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uVoid;
      uniform vec3 uHaze;
      uniform vec3 uDust;
      uniform float uDustAmt;
      uniform float uNear;
      uniform float uFar;
      varying vec3 vW;
      void main() {
        float d = length(vW - cameraPosition);
        float f = smoothstep(uNear, uFar, d);
        vec3 haze = mix(uHaze, uDust, uDustAmt);
        vec3 col = mix(uVoid, haze, f * f * (1.0 - 0.3 * uDustAmt) + uDustAmt * 0.25 * smoothstep(0.0, uNear, d));
        gl_FragColor = vec4(col, 1.0);
      }
    `,
    toneMapped: false,
    fog: false,
  })
}

export function Ground() {
  const map = useSim((s) => s.map)
  const size = useThree((s) => s.size)
  const fogY = fogYFor(map)
  const cx = map ? map.cx : 0
  const cz = map ? -map.cy : 0
  const floor = useMemo(floorMaterial, [])
  const plane = useMemo(() => new THREE.PlaneGeometry(4000, 4000).rotateX(-Math.PI / 2), [])
  useEffect(() => () => (floor.dispose(), plane.dispose()), [floor, plane])

  const outline = useMemo(() => {
    if (!map) return null
    const line = new FatPolyline(5, { color: '#36506e', width: 1.5, opacity: 0.9 })
    const x0 = map.ox
    const x1 = map.ox + map.width
    const z0 = -map.oy
    const z1 = -(map.oy + map.depth)
    const y = fogY + 0.03
    line.set([x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1, x0, y, z0], 5)
    return line
  }, [map, fogY])

  useEffect(() => () => outline?.dispose(), [outline])
  useEffect(() => {
    outline?.setResolution(size.width, size.height)
  }, [outline, size])

  useFrame(() => {
    floor.uniforms.uDustAmt.value = atmo.dust
  })

  return (
    <group>
      <mesh geometry={plane} material={floor} position={[cx, fogY - 0.06, cz]} renderOrder={50} />
      <Grid
        position={[cx, fogY, cz]}
        args={[200, 200]}
        cellSize={1}
        cellThickness={0.6}
        cellColor="#141b25"
        sectionSize={5}
        sectionThickness={1.1}
        sectionColor="#1f2b3a"
        fadeDistance={170}
        fadeStrength={1.6}
        infiniteGrid
        renderOrder={-1}
      />
      {outline && <primitive object={outline.object} />}
    </group>
  )
}
