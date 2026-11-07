import { OrbitControls, Stars } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef, type ComponentRef } from 'react'
import * as THREE from 'three'
import { useStore } from '../lib/store'
import { Effects } from './Effects'
import { PathLines } from './PathLines'
import { PlanetFog } from './PlanetFog'
import { markerPositions, RobotMarkers } from './RobotMarkers'
import { SectorPatch } from './SectorPatch'

const HOME_TARGET = new THREE.Vector3(0, 0.02, 0.62)
const HOME_CAMERA = new THREE.Vector3(0, 0.62, 3.0)
const IDLE_AFTER_MS = 7000
const SWAY_RAD = 0.32

export default function Globe() {
  return (
    <Canvas
      camera={{ position: HOME_CAMERA.toArray(), fov: 36, near: 0.01, far: 200 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
      aria-label="Planet globe with the mission sector"
    >
      <Stars radius={60} depth={40} count={2600} factor={2.4} saturation={0.2} fade speed={0.25} />
      <PlanetFog />
      <SectorPatch />
      <PathLines />
      <RobotMarkers />
      <Effects />
      <CameraRig />
    </Canvas>
  )
}

function CameraRig() {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null)
  const camera = useThree((s) => s.camera)
  const lastInput = useRef(0)
  const swayPhase = useRef(0)
  const swayApplied = useRef(0)
  const shake = useMemo(() => new THREE.Vector3(), [])
  const tmp = useMemo(() => new THREE.Vector3(), [])
  const axis = useMemo(() => new THREE.Vector3(0, 1, 0), [])

  useEffect(() => {
    const c = controls.current
    if (!c) return
    c.target.copy(HOME_TARGET)
    c.update()
    const onStart = () => {
      lastInput.current = performance.now()
      swayApplied.current = 0 // user takes over from wherever the sway left the camera
    }
    c.addEventListener('start', onStart)
    return () => c.removeEventListener('start', onStart)
  }, [])

  // double-click resets the framing
  const gl = useThree((s) => s.gl)
  useEffect(() => {
    const el = gl.domElement
    const reset = () => {
      const c = controls.current
      if (!c) return
      useStore.getState().setFollow(false)
      c.target.copy(HOME_TARGET)
      camera.position.copy(HOME_CAMERA)
      swayApplied.current = 0
      c.update()
    }
    el.addEventListener('dblclick', reset)
    return () => el.removeEventListener('dblclick', reset)
  }, [gl, camera])

  useFrame((_, dt) => {
    const c = controls.current
    if (!c) return
    const st = useStore.getState()
    camera.position.sub(shake)
    shake.set(0, 0, 0)

    const follow = st.followRobot && st.selectedRobot ? markerPositions.get(st.selectedRobot) : undefined
    if (follow) {
      // glide the orbit target onto the robot and pull in to a close orbit
      const k = 1 - Math.exp(-dt * 2.2)
      tmp.copy(follow).sub(c.target).multiplyScalar(k)
      c.target.add(tmp)
      camera.position.add(tmp)
      const off = camera.position.clone().sub(c.target)
      const want = 1.05
      off.setLength(off.length() + (want - off.length()) * k)
      camera.position.copy(c.target).add(off)
    } else if (performance.now() - lastInput.current > IDLE_AFTER_MS) {
      // idle sway around the patch (not a full spin, so the sector stays in view)
      swayPhase.current += dt * 0.11
      const want = SWAY_RAD * Math.sin(swayPhase.current)
      const d = want - swayApplied.current
      swayApplied.current = want
      tmp.copy(camera.position).sub(c.target).applyAxisAngle(axis, d)
      camera.position.copy(c.target).add(tmp)
    }

    if (st.telemetry?.mission.env.seismic) {
      const a = 0.006
      shake.set((Math.random() - 0.5) * a, (Math.random() - 0.5) * a, (Math.random() - 0.5) * a)
      camera.position.add(shake)
    }
    c.update()
  })

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      rotateSpeed={0.5}
      zoomSpeed={0.7}
      minDistance={1.25}
      maxDistance={7}
      enablePan
    />
  )
}
