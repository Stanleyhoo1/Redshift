import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { localFrame, metresToUnits } from '../lib/geo'
import { useStore, type EventRecord } from '../lib/store'
import { C } from '../lib/theme'
import type { RockslideEvent, XY, XYZ } from '../lib/types'
import { glowTexture, surfacePoint, surfaceQuaternion, useFatLine } from './globeUtil'
import { DUST_FRAGMENT, DUST_VERTEX } from './sectorShader'

const FALL_S = 1.1
const WAVE_S = 1.6
const ROCK_TTL_S = 30

function useSectorSize(): number {
  return useStore((s) => s.map?.meta.size_m ?? s.options.sectors.find((o) => o.id === (s.telemetry?.mission.config?.sector ?? s.lastConfig?.sector))?.size_m ?? 100)
}

/** New events since mount (ignores history so a reload does not replay old meteors). */
function useNewEvents(filter: (e: EventRecord) => boolean): EventRecord[] {
  const events = useStore((s) => s.events)
  const [mountedAt] = useState(() => performance.now())
  return useMemo(() => events.filter((e) => e.receivedAt >= mountedAt - 500 && filter(e)), [events, mountedAt, filter])
}

const isRockslide = (e: EventRecord) => e.event.op === 'spawn_rockslide' && (e.event as RockslideEvent).ok !== false && Array.isArray((e.event as RockslideEvent).impact)
const isPing = (e: EventRecord) => ['scan_request', 'scan_complete', 'hazard_mark'].includes(e.event.op) && Array.isArray((e.event as { xy?: XY }).xy)

export function Effects() {
  const slides = useNewEvents(isRockslide)
  const pings = useNewEvents(isPing)
  return (
    <group>
      {slides.slice(-4).map((e) => (
        <Meteor key={e.seq} rec={e} />
      ))}
      {pings.slice(-8).map((e) => (
        <Ping key={e.seq} rec={e} />
      ))}
      <HazardPulses />
      <DustShell />
    </group>
  )
}

function Meteor({ rec }: { rec: EventRecord }) {
  const ev = rec.event as RockslideEvent
  const L = useSectorSize()
  const map = useStore((s) => s.map)
  const born = rec.receivedAt
  const camera = useThree((s) => s.camera)
  const { impact, start } = useMemo(() => {
    const impact = surfacePoint(map, ev.impact[0], ev.impact[1], L, 0.003)
    const { up } = localFrame(ev.impact[0], ev.impact[1], L)
    // streak in from the upper left of the current view, out of the sky above the impact
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0)
    const camUp = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1)
    const start = impact
      .clone()
      .add(new THREE.Vector3(...up).multiplyScalar(0.25))
      .add(camUp.multiplyScalar(0.5))
      .add(right.multiplyScalar(-0.42))
    return { impact, start }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ev, L])
  const quat = useMemo(() => surfaceQuaternion(ev.impact[0], ev.impact[1], 0, L), [ev, L])
  const trailPos = useMemo(() => new Float32Array(6), [])
  const [trailArr, setTrailArr] = useState<Float32Array | null>(null)
  const trail = useFatLine(trailArr, null, { color: '#ffd9a8', width: 5, additive: true, depthTest: false })
  const head = useRef<THREE.Sprite>(null)
  const wave = useRef<THREE.Mesh>(null)
  const wave2 = useRef<THREE.Mesh>(null)
  const flash = useRef<THREE.Sprite>(null)
  const rocks = useRef<THREE.Group>(null)
  const headMat = useMemo(() => new THREE.SpriteMaterial({ map: glowTexture(), color: '#ffe0b0', blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false }), [])
  const flashMat = useMemo(() => new THREE.SpriteMaterial({ map: glowTexture(), color: '#ffb238', blending: THREE.AdditiveBlending, transparent: true, depthWrite: false }), [])
  const cur = useMemo(() => new THREE.Vector3(), [])
  const tail = useMemo(() => new THREE.Vector3(), [])

  useFrame(() => {
    const t = (performance.now() - born) / 1000
    const fall = Math.min(1, t / FALL_S)
    const e = fall * fall * fall // accelerating
    cur.lerpVectors(start, impact, e)
    tail.lerpVectors(start, impact, Math.max(0, e - 0.4))
    if (head.current) {
      head.current.visible = fall < 1
      head.current.position.copy(cur)
      head.current.scale.setScalar(0.08 + 0.06 * e)
    }
    if (fall < 1) {
      trailPos.set([tail.x, tail.y, tail.z, cur.x, cur.y, cur.z])
      setTrailArr(new Float32Array(trailPos))
    } else if (trailArr) setTrailArr(null)
    const w = (t - FALL_S) / WAVE_S
    const showWave = w >= 0 && w <= 1
    for (const [ref, delay, maxR] of [[wave, 0, 0.17], [wave2, 0.18, 0.11]] as const) {
      const m = ref.current
      if (!m) continue
      const ww = w - delay
      m.visible = showWave && ww >= 0
      if (m.visible) {
        m.scale.setScalar(0.01 + maxR * (1 - Math.pow(1 - ww, 3)))
        ;(m.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 1 - ww) * 0.9
      }
    }
    if (flash.current) {
      const f = (t - FALL_S) / 0.6
      flash.current.visible = f >= 0 && f <= 1
      flash.current.scale.setScalar(0.1 + 0.3 * f)
      flashMat.opacity = Math.max(0, 1 - f)
    }
    if (rocks.current) rocks.current.visible = t > FALL_S && t < ROCK_TTL_S
  })

  return (
    <group>
      <primitive object={trail} renderOrder={30} />
      <sprite ref={head} material={headMat} renderOrder={31} />
      <sprite ref={flash} material={flashMat} position={impact} renderOrder={31} />
      <group position={impact} quaternion={quat}>
        <mesh ref={wave} rotation={[-Math.PI / 2, 0, 0]} visible={false} renderOrder={29}>
          <ringGeometry args={[0.9, 1, 72]} />
          <meshBasicMaterial color={C.nad} transparent depthWrite={false} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} />
        </mesh>
        <mesh ref={wave2} rotation={[-Math.PI / 2, 0, 0]} visible={false} renderOrder={29}>
          <ringGeometry args={[0.8, 1, 72]} />
          <meshBasicMaterial color="#ffe0b0" transparent depthWrite={false} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} />
        </mesh>
      </group>
      <group ref={rocks} visible={false}>
        {(ev.rocks ?? []).slice(0, 24).map((r, k) => (
          <Rock key={k} r={r} L={L} />
        ))}
      </group>
    </group>
  )
}

function Rock({ r, L }: { r: XYZ; L: number }) {
  const map = useStore((s) => s.map)
  const pos = useMemo(() => surfacePoint(map, r[0], r[1], L, 0.002), [map, r, L])
  const size = Math.max(0.004, metresToUnits(Math.max(0.4, r[2]), L))
  return (
    <mesh position={pos} renderOrder={15}>
      <icosahedronGeometry args={[size, 0]} />
      <meshBasicMaterial color="#6d4a4f" />
    </mesh>
  )
}

/** Short ring burst for fleet coordination events (scan request/complete, hazard marks). */
function Ping({ rec }: { rec: EventRecord }) {
  const ev = rec.event as { op: string; xy: XY; radius_m?: number }
  const L = useSectorSize()
  const map = useStore((s) => s.map)
  const pos = useMemo(() => surfacePoint(map, ev.xy[0], ev.xy[1], L, 0.004), [map, ev, L])
  const quat = useMemo(() => surfaceQuaternion(ev.xy[0], ev.xy[1], 0, L), [ev, L])
  const color = ev.op === 'hazard_mark' ? C.ha : C.sci
  const ref = useRef<THREE.Mesh>(null)
  const base = metresToUnits(ev.radius_m ?? 4, L)
  useFrame(() => {
    const m = ref.current
    if (!m) return
    const t = ((performance.now() - rec.receivedAt) / 1000) % 1.4
    const alive = performance.now() - rec.receivedAt < 4200
    m.visible = alive
    m.scale.setScalar(base * (0.6 + 1.6 * t))
    ;(m.material as THREE.MeshBasicMaterial).opacity = (1 - t / 1.4) * 0.9
  })
  return (
    <group position={pos} quaternion={quat}>
      <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]} renderOrder={28}>
        <ringGeometry args={[0.85, 1, 64]} />
        <meshBasicMaterial color={color} transparent depthWrite={false} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} />
      </mesh>
    </group>
  )
}

/** Repeating rings over permanent no-go zones from map stats. */
function HazardPulses() {
  const hz = useStore((s) => s.map?.stats.permanent_hazards)
  const L = useSectorSize()
  const map = useStore((s) => s.map)
  const list = useMemo(() => (hz ?? []).slice(0, 16), [hz])
  return (
    <>
      {list.map((h, k) => (
        <HazardPulse key={`${h[0].toFixed(1)},${h[1].toFixed(1)}`} h={h} L={L} map={map} phase={k * 0.37} />
      ))}
    </>
  )
}

function HazardPulse({ h, L, map, phase }: { h: XYZ; L: number; map: ReturnType<typeof useStore.getState>['map']; phase: number }) {
  const pos = useMemo(() => surfacePoint(map, h[0], h[1], L, 0.004), [map, h, L])
  const quat = useMemo(() => surfaceQuaternion(h[0], h[1], 0, L), [h, L])
  const r = metresToUnits(h[2], L)
  const ref = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => {
    const m = ref.current
    if (!m) return
    const t = (clock.elapsedTime * 0.55 + phase) % 1
    m.scale.setScalar(r * (0.9 + 1.4 * t))
    ;(m.material as THREE.MeshBasicMaterial).opacity = (1 - t) * 0.85
  })
  return (
    <group position={pos} quaternion={quat}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} scale={r} renderOrder={27}>
        <ringGeometry args={[0.93, 1, 64]} />
        <meshBasicMaterial color="#ff2f4f" transparent opacity={0.9} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]} renderOrder={27}>
        <ringGeometry args={[0.88, 1, 64]} />
        <meshBasicMaterial color={C.ha} transparent depthWrite={false} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} />
      </mesh>
    </group>
  )
}

/** Animated haze over the sector while the dust anomaly is active. */
function DustShell() {
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: DUST_VERTEX,
        fragmentShader: DUST_FRAGMENT,
        uniforms: { uTime: { value: 0 }, uLevel: { value: 0 }, uLift: { value: 0.05 } },
        transparent: true,
        depthWrite: false,
      }),
    [],
  )
  const level = useRef(0)
  useEffect(() => () => mat.dispose(), [mat])
  useFrame(({ clock }, dt) => {
    const target = useStore.getState().telemetry?.mission.env.dust_level ?? 0
    level.current += (target - level.current) * (1 - Math.exp(-dt * 2.5))
    mat.uniforms.uTime.value = clock.elapsedTime
    mat.uniforms.uLevel.value = level.current
  })
  return (
    <mesh material={mat} renderOrder={40} frustumCulled={false}>
      <planeGeometry args={[1, 1, 64, 64]} />
    </mesh>
  )
}
