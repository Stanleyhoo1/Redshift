import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useSim } from '../state/store.js'
import { useGraphics } from '../state/visualPrefs.js'
import { ATMOSPHERE, VOID } from '../lib/theme.js'
import { SUN, createSkyMaterial } from './sky.js'
import { atmo } from './sceneConstants.js'

// Light, sky and air for the realistic world:
//  - renderer: ACES filmic tone mapping, sRGB output, soft shadows (High graphics only);
//  - a sky dome that follows the camera, a low warm sun (the key light, its shadow camera
//    fitted to what the camera sees) and a hemisphere fill;
//  - distance haze (exponential fog matched to the horizon), which a dust storm thickens
//    and browns out, dimming the sun;
//  - HUD policy: unlit materials (the data overlays: rings, glows, markers, labels in 3D)
//    skip tone mapping and haze, so they keep the exact theme colours and stay crisp on
//    top of the realistic world.

const SHADOW_SIZE = 2048
const HAZE_DENSITY = 0.0021 // FogExp2: about 7 % haze at 130 m (the overview), 80 % at 600 m
const EXPOSURE = 1.08
const SUN_INTENSITY = 3.5
const FILL_INTENSITY = 1.1

const hazeColor = new THREE.Color(ATMOSPHERE.haze)
const dustColor = new THREE.Color(ATMOSPHERE.dust)
const sunColor = new THREE.Color(ATMOSPHERE.sunLight)
const sunDust = new THREE.Color('#c48a5c')
const fillSky = new THREE.Color(ATMOSPHERE.fillSky)
const fillGround = new THREE.Color(ATMOSPHERE.fillGround)
const fillDust = new THREE.Color('#8a6a55')

const _corner = new THREE.Vector3()
const _dir = new THREE.Vector3()
const _c = new THREE.Vector3()
const _l = new THREE.Vector3()
const lightRot = new THREE.Matrix4().lookAt(SUN, new THREE.Vector3(), new THREE.Vector3(0, 1, 0))
const lightRotInv = lightRot.clone().invert()
const NDC = [
  [-1, -1],
  [1, -1],
  [-1, 1],
  [1, 1],
  [0, 0],
]

// How far from the orbit target the visible ground reaches (metres), from the frustum
// corner rays against the ground plane through the target.
function footprint(camera, target, cap) {
  let r = 8
  for (const [x, y] of NDC) {
    _corner.set(x, y, 0.5).unproject(camera)
    _dir.copy(_corner).sub(camera.position).normalize()
    let t = _dir.y < -1e-3 ? (target.y - camera.position.y) / _dir.y : Infinity
    if (!(t > 0) || t > cap * 3) t = cap * 3
    _corner.copy(camera.position).addScaledVector(_dir, t)
    _corner.y = target.y
    r = Math.max(r, _corner.distanceTo(target))
  }
  return Math.min(r, cap)
}

function isHudMaterial(m) {
  return !!(m && !m.userData.realistic && (m.isMeshBasicMaterial || m.isSpriteMaterial || m.isPointsMaterial || m.isLineBasicMaterial || m.isLineDashedMaterial || m.isLineMaterial))
}

function hudOne(m) {
  if (m.userData.hudSet || !isHudMaterial(m)) return
  m.userData.hudSet = true
  m.toneMapped = false
  if (m.fog) {
    m.fog = false
    m.needsUpdate = true
  }
}

function hudVisit(o) {
  const m = o.material
  if (!m) return
  if (Array.isArray(m)) m.forEach(hudOne)
  else hudOne(m)
}

export function Atmosphere() {
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  const setDpr = useThree((s) => s.setDpr)
  const map = useSim((s) => s.map)
  const dusting = useSim((s) => s.hazards.some((h) => h.type === 'dust_storm' && h.active))
  const graphics = useGraphics()
  const high = graphics === 'high'
  const sun = useRef()
  const hemi = useRef()
  const sky = useRef()
  const st = useRef({ r: 0, frame: 0 })
  const skyMat = useMemo(createSkyMaterial, [])
  const skyGeo = useMemo(() => new THREE.SphereGeometry(1000, 48, 24), [])
  const fog = useMemo(() => new THREE.FogExp2(hazeColor.clone(), HAZE_DENSITY), [])
  const target = useMemo(() => new THREE.Object3D(), [])
  useEffect(() => () => (skyMat.dispose(), skyGeo.dispose()), [skyMat, skyGeo])

  // Renderer and scene setup, before the first frame so materials compile once.
  useLayoutEffect(() => {
    const prev = { tm: gl.toneMapping, exp: gl.toneMappingExposure, cs: gl.outputColorSpace, fog: scene.fog, bg: scene.background }
    gl.toneMapping = THREE.ACESFilmicToneMapping
    gl.toneMappingExposure = EXPOSURE
    gl.outputColorSpace = THREE.SRGBColorSpace
    gl.shadowMap.type = THREE.PCFShadowMap
    scene.fog = fog
    scene.background = new THREE.Color(VOID)
    scene.add(target)
    return () => {
      gl.toneMapping = prev.tm
      gl.toneMappingExposure = prev.exp
      gl.outputColorSpace = prev.cs
      scene.fog = prev.fog
      scene.background = prev.bg
      scene.remove(target)
    }
  }, [gl, scene, fog, target])

  // Graphics quality: shadows and pixel ratio.
  useEffect(() => {
    gl.shadowMap.enabled = high
    gl.shadowMap.needsUpdate = true
    if (sun.current) sun.current.castShadow = high
    setDpr(high ? [1, 1.5] : 1)
  }, [high, gl, setDpr])

  useFrame(({ camera, controls }, dt) => {
    const s = st.current
    // Dust storm amount, eased in and out (read by the particles and shaders).
    const goal = dusting ? 1 : 0
    const k = 1 - Math.exp(-Math.min(dt, 0.1) * (dusting ? 1.4 : 0.9))
    atmo.dust += (goal - atmo.dust) * k
    if (Math.abs(goal - atmo.dust) < 0.002) atmo.dust = goal
    const a = atmo.dust
    const center = controls?.target
    const D = center ? camera.position.distanceTo(center) : 60

    // Haze: absolute distance (zooming in clears it), plus the storm: about 40 % haze at
    // the point the camera looks at, whatever the zoom.
    fog.color.copy(hazeColor).lerp(dustColor, a)
    const dustDensity = 0.72 / Math.max(8, D)
    fog.density = HAZE_DENSITY + (Math.max(HAZE_DENSITY, dustDensity) - HAZE_DENSITY) * a
    if (scene.background?.isColor) scene.background.set(VOID)

    // Sky dome follows the camera.
    if (sky.current) sky.current.position.copy(camera.position)
    skyMat.uniforms.uDust.value = a
    skyMat.uniforms.uSunDisk.value = 1 - a * 0.85

    // Lights dim and warm in the dust.
    const L = sun.current
    if (L) {
      L.intensity = SUN_INTENSITY * (1 - 0.6 * a)
      L.color.copy(sunColor).lerp(sunDust, a)
    }
    if (hemi.current) {
      hemi.current.intensity = FILL_INTENSITY * (1 - 0.15 * a)
      hemi.current.color.copy(fillSky).lerp(fillDust, a)
    }

    // Fit the shadow camera to the visible ground, snapped to whole texels so shadows
    // do not shimmer as the camera glides.
    if (L) {
      const span = map ? Math.max(map.width, map.depth) : 120
      const c = center || _c.set(0, 0, 0)
      const want = footprint(camera, c, span * 0.75 + 10)
      // Quantise the radius (steps of about 9 %) so zooming does not resize it every frame.
      const r = Math.pow(2, Math.ceil(Math.log2(want) * 8) / 8)
      s.r = r
      const texel = (2 * r) / SHADOW_SIZE
      _l.copy(c).applyMatrix4(lightRotInv)
      _l.x = Math.round(_l.x / texel) * texel
      _l.y = Math.round(_l.y / texel) * texel
      _l.applyMatrix4(lightRot)
      target.position.copy(_l)
      target.updateMatrixWorld()
      L.position.copy(_l).addScaledVector(SUN, r + 60)
      const cam = L.shadow.camera
      if (cam.right !== r) {
        cam.left = -r
        cam.right = r
        cam.top = r
        cam.bottom = -r
        cam.near = 1
        cam.far = 2 * r + 130
        cam.updateProjectionMatrix()
      }
      L.shadow.normalBias = texel * 1.6
      L.shadow.bias = -0.00025
      L.shadow.radius = 2 + 3 * a
    }

    // HUD policy, checked twice a second for new objects (cheap: a few hundred nodes).
    if (s.frame++ % 30 === 0) scene.traverse(hudVisit)
  })

  return (
    <>
      <mesh ref={sky} geometry={skyGeo} material={skyMat} renderOrder={100} frustumCulled={false} />
      <hemisphereLight ref={hemi} args={[fillSky, fillGround, FILL_INTENSITY]} />
      <directionalLight
        ref={sun}
        color={sunColor}
        intensity={SUN_INTENSITY}
        target={target}
        castShadow={high}
        shadow-mapSize={[SHADOW_SIZE, SHADOW_SIZE]}
        shadow-bias={-0.00025}
      />
    </>
  )
}
