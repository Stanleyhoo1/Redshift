import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useRobotIds } from '../lib/hooks'
import { metresToUnits } from '../lib/geo'
import { useStore } from '../lib/store'
import { C, robotColor, stateColor } from '../lib/theme'
import { glowTexture, surfacePoint, surfaceQuaternion } from './globeUtil'

const ROVER_SIZE = 0.026
const SCOUT_HOVER = 0.075
const SCOUT_FOOTPRINT_M = 12

/** Displayed marker positions (world space), read by the camera rig for follow mode. */
export const markerPositions = new Map<string, THREE.Vector3>()

function chevronGeometry(): THREE.BufferGeometry {
  // in the local XZ plane: +Z forward, +Y up
  const v = new Float32Array([0, 0, 1, 0.72, 0, -0.8, 0, 0, -0.32, 0, 0, 1, 0, 0, -0.32, -0.72, 0, -0.8])
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(v, 3))
  return g
}

export function RobotMarkers() {
  const ids = useRobotIds()
  return (
    <group>
      {ids.map((id) => (id.startsWith('scout') ? <ScoutMarker key={id} id={id} /> : <RoverMarker key={id} id={id} />))}
    </group>
  )
}

function useSmoothedPose(id: string, hover: number) {
  const pos = useRef(new THREE.Vector3())
  const quat = useRef(new THREE.Quaternion())
  const init = useRef(false)
  const tp = useMemo(() => new THREE.Vector3(), [])
  const tq = useMemo(() => new THREE.Quaternion(), [])
  return (dt: number) => {
    const st = useStore.getState()
    const r = st.telemetry?.robots[id]
    const L = st.map?.meta.size_m ?? sizeFromConfig(st)
    if (!r) return null
    const z = r.type === 'scout' ? undefined : r.pose.z
    surfacePoint(st.map, r.pose.x, r.pose.y, L, 0.004 + hover, z, tp)
    surfaceQuaternion(r.pose.x, r.pose.y, r.pose.yaw, L, tq)
    const k = init.current ? 1 - Math.exp(-dt * 9) : 1
    pos.current.lerp(tp, k)
    quat.current.slerp(tq, k)
    init.current = true
    let mp = markerPositions.get(id)
    if (!mp) markerPositions.set(id, (mp = new THREE.Vector3()))
    mp.copy(pos.current)
    return { r, L, pos: pos.current, quat: quat.current, selected: st.selectedRobot === id, map: st.map }
  }
}

function sizeFromConfig(st: ReturnType<typeof useStore.getState>): number {
  const id = st.telemetry?.mission.config?.sector ?? st.lastConfig?.sector
  return st.options.sectors.find((s) => s.id === id)?.size_m ?? 100
}

function RoverMarker({ id }: { id: string }) {
  const group = useRef<THREE.Group>(null)
  const ring = useRef<THREE.Mesh>(null)
  const glow = useRef<THREE.Sprite>(null)
  const safe = useRef<THREE.Group>(null)
  const geom = useMemo(chevronGeometry, [])
  const rimMat = useMemo(() => new THREE.MeshBasicMaterial({ color: C.o3, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthWrite: false }), [])
  const glowMat = useMemo(() => new THREE.SpriteMaterial({ map: glowTexture(), color: C.o3, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false }), [])
  const pose = useSmoothedPose(id, 0)
  const tq = useMemo(() => new THREE.Quaternion(), [])

  useFrame(({ clock }, dt) => {
    const p = pose(dt)
    const g = group.current
    if (!g) return
    g.visible = !!p
    if (!p) return
    g.position.copy(p.pos)
    g.quaternion.copy(p.quat)
    const sc = stateColor(p.r.state)
    rimMat.color.set(sc)
    glowMat.color.set(sc)
    const alarm = p.r.state === 'SAFE_HOLD' || p.r.state === 'RECOVERY_REVERSE'
    const t = clock.elapsedTime
    if (glow.current) glow.current.scale.setScalar((alarm ? 0.11 : 0.075) * (1 + 0.12 * Math.sin(t * (alarm ? 9 : 2.5))))
    if (ring.current) {
      ring.current.visible = p.selected
      const s = 1 + 0.25 * ((t * 0.9) % 1)
      ring.current.scale.setScalar(s)
      ;(ring.current.material as THREE.MeshBasicMaterial).opacity = 0.8 * (1 - ((t * 0.9) % 1))
    }
    // last safe cell beacon while recovering
    const sm = safe.current
    if (sm) {
      const show = !!p.r.last_safe && ['SAFE_HOLD', 'RECOVERY_REVERSE', 'AWAIT_SCOUT'].includes(p.r.state)
      sm.visible = show
      if (show && p.r.last_safe) {
        surfacePoint(p.map, p.r.last_safe[0], p.r.last_safe[1], p.L, 0.005, p.r.pose.z, sm.position)
        surfaceQuaternion(p.r.last_safe[0], p.r.last_safe[1], t, p.L, tq)
        sm.quaternion.copy(tq)
      }
    }
  })

  return (
    <>
    <group ref={group}>
      <sprite ref={glow} material={glowMat} renderOrder={20} />
      <mesh geometry={geom} material={rimMat} scale={ROVER_SIZE * 1.35} position={[0, 0.0008, 0]} renderOrder={21} />
      <mesh geometry={geom} scale={ROVER_SIZE} position={[0, 0.0016, 0]} renderOrder={22}>
        <meshBasicMaterial color={robotColor(id)} side={THREE.DoubleSide} depthWrite={false} transparent />
      </mesh>
      <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]} renderOrder={19}>
        <ringGeometry args={[0.036, 0.04, 48]} />
        <meshBasicMaterial color={C.bone} transparent depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
    </group>
    {/* last safe cell: a slowly turning diamond the rover reverses toward */}
    <group ref={safe}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} renderOrder={19}>
        <ringGeometry args={[0.009, 0.013, 4]} />
        <meshBasicMaterial color={C.o3} transparent opacity={0.95} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
    </group>
    </>
  )
}

function ScoutMarker({ id }: { id: string }) {
  const group = useRef<THREE.Group>(null)
  const body = useRef<THREE.Mesh>(null)
  const cone = useRef<THREE.Mesh>(null)
  const foot = useRef<THREE.Mesh>(null)
  const ring = useRef<THREE.Mesh>(null)
  const pose = useSmoothedPose(id, SCOUT_HOVER)
  const tether = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, -SCOUT_HOVER, 0)])
    const m = new THREE.LineDashedMaterial({ color: '#c9b8ff', dashSize: 0.006, gapSize: 0.005, transparent: true, opacity: 0.7, depthWrite: false })
    const l = new THREE.Line(g, m)
    l.computeLineDistances()
    return l
  }, [])
  const coneGeom = useMemo(() => {
    const g = new THREE.ConeGeometry(1, 1, 40, 1, true)
    g.translate(0, -0.5, 0) // apex at the origin, base one unit below
    return g
  }, [])
  const glowMat = useMemo(() => new THREE.SpriteMaterial({ map: glowTexture(), color: '#b9a6ff', blending: THREE.AdditiveBlending, transparent: true, depthWrite: false }), [])

  useFrame(({ clock }, dt) => {
    const p = pose(dt)
    const g = group.current
    if (!g) return
    g.visible = !!p
    if (!p) return
    const t = clock.elapsedTime
    g.position.copy(p.pos)
    g.quaternion.copy(p.quat)
    if (body.current) {
      body.current.position.y = 0.004 * Math.sin(t * 2.2)
      body.current.rotation.y = t * 1.4
    }
    const scanning = p.r.state === 'SCANNING' || p.r.state === 'FLY_TO_SCAN'
    const radius = metresToUnits(SCOUT_FOOTPRINT_M, p.L)
    if (cone.current) {
      cone.current.visible = scanning
      cone.current.scale.set(radius, SCOUT_HOVER, radius)
      ;(cone.current.material as THREE.MeshBasicMaterial).opacity = (p.r.state === 'SCANNING' ? 0.22 : 0.1) * (0.75 + 0.25 * Math.sin(t * 6))
    }
    if (foot.current) {
      foot.current.scale.setScalar(radius)
      ;(foot.current.material as THREE.MeshBasicMaterial).opacity = scanning ? 0.55 : 0.18
      foot.current.rotation.z = t * 0.8
    }
    if (ring.current) ring.current.visible = p.selected
  })

  return (
    <group ref={group}>
      <primitive object={tether} renderOrder={18} />
      <mesh ref={foot} position={[0, -SCOUT_HOVER + 0.001, 0]} rotation={[-Math.PI / 2, 0, 0]} renderOrder={18}>
        <ringGeometry args={[0.92, 1, 64, 1, 0, Math.PI * 1.6]} />
        <meshBasicMaterial color={C.sci} transparent depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={cone} geometry={coneGeom} renderOrder={17}>
        <meshBasicMaterial color={C.sci} transparent depthWrite={false} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} />
      </mesh>
      <sprite material={glowMat} scale={0.07} renderOrder={20} />
      <mesh ref={body} renderOrder={22}>
        <octahedronGeometry args={[0.014, 0]} />
        <meshBasicMaterial color={robotColor(id)} wireframe={false} />
      </mesh>
      <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]} renderOrder={19}>
        <ringGeometry args={[0.026, 0.029, 48]} />
        <meshBasicMaterial color={C.bone} transparent opacity={0.7} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
    </group>
  )
}
