import { useFrame } from '@react-three/fiber'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { metresToUnits } from '../lib/geo'
import { GHOST_LIFETIME_MS, ghosts, type PathGhost } from '../lib/history'
import { useRescue, useRobotIds } from '../lib/hooks'
import { useStore } from '../lib/store'
import { C, robotColor } from '../lib/theme'
import { drape, surfacePoint, surfaceQuaternion, useFatLine } from './globeUtil'

const EMPTY: [number, number][] = []

function useSectorSize(): number {
  return useStore((s) => s.map?.meta.size_m ?? s.options.sectors.find((o) => o.id === (s.telemetry?.mission.config?.sector ?? s.lastConfig?.sector))?.size_m ?? 100)
}

export function PathLines() {
  const ids = useRobotIds()
  return (
    <group>
      {ids.map((id) => (
        <RobotPath key={id} id={id} />
      ))}
      <GhostPaths />
      <RescueTether />
    </group>
  )
}

const TETHER_N = 40

/** Hi-vis arc from the recovery rover to the damaged one while a rescue is under way, flowing toward the target. */
function RescueTether() {
  const v = useRescue()
  const active = !!v && !!v.rescuer && !v.resolved && !v.aborted
  const from = useStore((s) => (active && v?.rescuer ? s.telemetry?.robots[v.rescuer]?.pose : undefined))
  const to = useStore((s) => (active && v ? s.telemetry?.robots[v.target]?.pose : undefined))
  const map = useStore((s) => s.map)
  const L = useSectorSize()
  const locked = !!v?.locked
  const pos = useMemo(() => {
    if (!from || !to) return null
    const d = Math.hypot(to.x - from.x, to.y - from.y)
    const arc = Math.min(0.06, metresToUnits(d, L) * 0.3)
    const out = new Float32Array((TETHER_N + 1) * 3)
    const p = new THREE.Vector3()
    for (let k = 0; k <= TETHER_N; k++) {
      const f = k / TETHER_N
      surfacePoint(map, from.x + (to.x - from.x) * f, from.y + (to.y - from.y) * f, L, 0.012 + arc * 4 * f * (1 - f), from.z + (to.z - from.z) * f, p)
      out.set([p.x, p.y, p.z], k * 3)
    }
    return out
  }, [from, to, map, L])
  const line = useFatLine(pos, null, { color: C.rescue, width: 3.2, dashed: true, dashSize: 0.009, gapSize: 0.006 })
  useFrame((_, dt) => {
    line.material.dashOffset -= dt * (locked ? 0.06 : 0.035)
    line.material.opacity = locked ? 1 : 0.85
  })
  return <primitive object={line} renderOrder={14} />
}

function RobotPath({ id }: { id: string }) {
  const path = useStore((s) => s.telemetry?.robots[id]?.path ?? EMPTY)
  const trail = useStore((s) => s.telemetry?.robots[id]?.trail ?? EMPTY)
  const z = useStore((s) => s.telemetry?.robots[id]?.pose.z)
  const goal = useStore((s) => s.telemetry?.robots[id]?.goal ?? null)
  const map = useStore((s) => s.map)
  const L = useSectorSize()
  const isScout = id.startsWith('scout')
  const color = isScout ? '#b9a6ff' : C.hb

  const pathPos = useMemo(() => (path.length > 1 ? drape(path, map, L, 0.007, isScout ? undefined : z) : null), [path, map, L, z, isScout])
  const trailPos = useMemo(() => (trail.length > 1 ? drape(trail, map, L, 0.005, isScout ? undefined : z) : null), [trail, map, L, z, isScout])
  const trailCol = useMemo(() => {
    if (trail.length < 2) return null
    const c = new THREE.Color(robotColor(id))
    const out = new Float32Array(trail.length * 3)
    for (let k = 0; k < trail.length; k++) {
      const a = Math.pow(k / (trail.length - 1), 1.6) * 0.8 // additive: black = invisible
      out[k * 3] = c.r * a
      out[k * 3 + 1] = c.g * a
      out[k * 3 + 2] = c.b * a
    }
    return out
  }, [trail, id])

  const pathLine = useFatLine(pathPos, null, { color, width: isScout ? 1.6 : 2.6, dashed: true, dashSize: 0.012, gapSize: 0.008 })
  const trailLine = useFatLine(trailPos, trailCol, { color: '#ffffff', width: isScout ? 1.2 : 2, additive: true })

  const goalRef = useRef<THREE.Mesh>(null)
  const q = useMemo(() => new THREE.Quaternion(), [])
  useFrame(({ clock }, dt) => {
    pathLine.material.dashOffset -= dt * 0.03
    const g = goalRef.current
    if (!g) return
    g.visible = !!goal && !isScout
    if (goal) {
      surfacePoint(map, goal[0], goal[1], L, 0.006, undefined, g.position)
      surfaceQuaternion(goal[0], goal[1], clock.elapsedTime * 0.6, L, q)
      g.quaternion.copy(q)
    }
  })

  return (
    <>
      <primitive object={trailLine} renderOrder={10} />
      <primitive object={pathLine} renderOrder={11} />
      <group ref={goalRef}>
        <mesh rotation={[-Math.PI / 2, 0, 0]} renderOrder={12}>
          <ringGeometry args={[0.011, 0.0145, 4]} />
          <meshBasicMaterial color={color} transparent opacity={0.9} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
      </group>
    </>
  )
}

/** Abandoned plans, drawn red and fading out after a drastic re-plan. */
function GhostPaths() {
  const [list, setList] = useState<PathGhost[]>([])
  const sig = useRef('')
  useFrame(() => {
    const now = performance.now()
    const alive = ghosts.filter((g) => now - g.born < GHOST_LIFETIME_MS)
    const s = alive.map((g) => `${g.robot}${g.born}`).join('|')
    if (s !== sig.current) {
      sig.current = s
      setList(alive)
    }
  })
  return (
    <>
      {list.map((g) => (
        <Ghost key={`${g.robot}${g.born}`} ghost={g} />
      ))}
    </>
  )
}

function Ghost({ ghost }: { ghost: PathGhost }) {
  const map = useStore((s) => s.map)
  const L = useSectorSize()
  const pos = useMemo(() => drape(ghost.points, map, L, 0.009), [ghost, map, L])
  const line = useFatLine(pos, null, { color: C.ha, width: 3.2, dashed: true, dashSize: 0.008, gapSize: 0.006 })
  useFrame(() => {
    const age = (performance.now() - ghost.born) / GHOST_LIFETIME_MS
    line.material.opacity = Math.max(0, 1 - age) * (0.75 + 0.25 * Math.sin(age * 40))
  })
  return <primitive object={line} renderOrder={13} />
}
