import { useEffect, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { sortRobots, useStore } from './store'
import type { RobotTelemetry } from './types'

export function useRobotIds(): string[] {
  return useStore(useShallow((s) => (s.telemetry ? sortRobots(Object.keys(s.telemetry.robots)) : [])))
}

export function useSelected(): [string | null, RobotTelemetry | undefined] {
  const id = useStore((s) => s.selectedRobot)
  const r = useStore((s) => (id ? s.telemetry?.robots[id] : undefined))
  return [id, r]
}

/** Re-render every `ms` (for staleness badges and countdowns). */
export function useTick(ms = 500): number {
  const [n, setN] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setN((v) => v + 1), ms)
    return () => clearInterval(t)
  }, [ms])
  return n
}

/** Canvas sized to its container in device pixels. */
export function useCanvasSize(ref: React.RefObject<HTMLElement | null>): { w: number; h: number; dpr: number } {
  const [size, setSize] = useState({ w: 0, h: 0, dpr: 1 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => {
      const r = e.contentRect
      setSize({ w: Math.floor(r.width), h: Math.floor(r.height), dpr: Math.min(2, window.devicePixelRatio || 1) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}
