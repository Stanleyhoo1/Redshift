import { useEffect, useRef } from 'react'
import { useCanvasSize, useSelected } from '../lib/hooks'
import { useStore } from '../lib/store'
import { C } from '../lib/theme'
import { Panel } from '../ui/Panel'

const RINGS = [5, 10, 15, 20, 25]
const CORRIDOR_HALF_W = 0.75 // m, rover half-width + margin
const CORRIDOR_LEN = 8 // m

// height (body frame z, m) → colour: low violet → mint at ground-ish → bone for tall returns
function heightColor(z: number): string {
  const t = Math.max(0, Math.min(1, (z + 1.2) / 2.6))
  const stops: [number, [number, number, number]][] = [
    [0, [92, 72, 190]],
    [0.4, [94, 235, 192]],
    [0.7, [255, 178, 56]],
    [1, [255, 240, 225]],
  ]
  let k = 0
  while (k < stops.length - 2 && t > stops[k + 1][0]) k++
  const [t0, c0] = stops[k]
  const [t1, c1] = stops[k + 1]
  const f = (t - t0) / (t1 - t0 || 1)
  const c = c0.map((v, i) => Math.round(v + (c1[i] - v) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

export function LidarRadar() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { w, h, dpr } = useCanvasSize(wrapRef)
  const [sel, robot] = useSelected()
  const pts = useStore((s) => (sel ? s.lidar[sel]?.points.length : 0)) ?? 0

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || w === 0 || h === 0) return
    cv.width = w * dpr
    cv.height = h * dpr
    let raf = 0
    let lastKey = ''
    const draw = () => {
      raf = requestAnimationFrame(draw)
      const st = useStore.getState()
      const id = st.selectedRobot
      const r = id ? st.telemetry?.robots[id] : undefined
      const lidar = id ? st.lidar[id] : undefined
      const map = st.map
      const key = `${id}|${st.telemetryAt}|${lidar?.receivedAt}|${map?.version}|${w}|${h}`
      if (key === lastKey) return
      lastKey = key
      const ctx = cv.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      const cx = w / 2
      const cy = h / 2
      const maxR = lidar?.max_range ?? 25
      const radius = Math.min(w, h) / 2 - 6
      const s = radius / maxR // px per metre

      ctx.save()
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
      ctx.clip()
      ctx.fillStyle = '#120f24'
      ctx.fillRect(0, 0, w, h)

      // local costmap crop from the global map, rotated so the robot's heading points up
      if (map && r) {
        const yaw = r.pose.yaw
        const sn = Math.sin(yaw)
        const cs = Math.cos(yaw)
        const res = map.meta.res
        const [x0, y0] = map.meta.origin
        const dx = x0 - r.pose.x
        const dy = y0 - r.pose.y
        ctx.save()
        ctx.imageSmoothingEnabled = false
        ctx.globalAlpha = 0.9
        ctx.setTransform(
          dpr * s * res * sn,
          dpr * -s * res * cs,
          dpr * -s * res * cs,
          dpr * -s * res * sn,
          dpr * (cx + s * (sn * dx - cs * dy)),
          dpr * (cy + s * (-cs * dx - sn * dy)),
        )
        ctx.drawImage(map.costmap, 0, 0)
        ctx.restore()
      }

      // range rings
      ctx.lineWidth = 1
      for (const m of RINGS) {
        if (m > maxR) continue
        ctx.strokeStyle = m === maxR ? 'rgba(163,156,200,0.45)' : 'rgba(163,156,200,0.18)'
        ctx.beginPath()
        ctx.arc(cx, cy, m * s, 0, Math.PI * 2)
        ctx.stroke()
      }
      ctx.strokeStyle = 'rgba(163,156,200,0.12)'
      ctx.beginPath()
      ctx.moveTo(cx - radius, cy)
      ctx.lineTo(cx + radius, cy)
      ctx.moveTo(cx, cy - radius)
      ctx.lineTo(cx, cy + radius)
      ctx.stroke()

      // forward corridor
      ctx.fillStyle = 'rgba(76,195,255,0.10)'
      ctx.strokeStyle = 'rgba(76,195,255,0.55)'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.rect(cx - CORRIDOR_HALF_W * s, cy - CORRIDOR_LEN * s, 2 * CORRIDOR_HALF_W * s, CORRIDOR_LEN * s)
      ctx.fill()
      ctx.stroke()
      ctx.setLineDash([])

      // points (body frame: x forward → up, y left → left)
      if (lidar) {
        const sz = Math.max(1.5, 2 * (radius / 120))
        for (const p of lidar.points) {
          const px = cx - p[1] * s
          const py = cy - p[0] * s
          ctx.fillStyle = heightColor(p[2])
          ctx.fillRect(px - sz / 2, py - sz / 2, sz, sz)
        }
      }
      ctx.restore()

      // north marker on the rim
      if (r) {
        // world north (0, 1) in screen space is (−cos yaw, −sin yaw)
        const nx = cx - Math.cos(r.pose.yaw) * (radius - 9)
        const ny = cy - Math.sin(r.pose.yaw) * (radius - 9)
        ctx.fillStyle = C.haze
        ctx.font = '600 10px "Archivo Variable", sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText('N', nx, ny)
      }
      // ring labels
      ctx.fillStyle = 'rgba(163,156,200,0.7)'
      ctx.font = '500 9.5px "Archivo Variable", sans-serif'
      ctx.textAlign = 'left'
      for (const m of [10, 20]) if (m <= maxR) ctx.fillText(`${m} m`, cx + 3, cy - m * s + 9)

      // robot
      ctx.fillStyle = C.bone
      ctx.beginPath()
      ctx.moveTo(cx, cy - 7)
      ctx.lineTo(cx + 5, cy + 5)
      ctx.lineTo(cx, cy + 2.5)
      ctx.lineTo(cx - 5, cy + 5)
      ctx.closePath()
      ctx.fill()
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [w, h, dpr])

  return (
    <Panel title="LiDAR" sensors={['LIDAR', 'MAP']} chipsFrom={205} right={<span className="num text-[11.5px] text-dim">{robot?.type === 'scout' ? 'map only' : `${pts} pts`}</span>} bodyClass="p-1.5">
      <div ref={wrapRef} className="relative h-full min-h-[150px] w-full">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-label={`LiDAR returns around ${robot ? robot.type : 'robot'}`} />
        {robot?.type === 'scout' && (
          <p className="absolute inset-x-0 bottom-0 text-center text-[11px] text-dim">The scout carries no LiDAR: local map only</p>
        )}
      </div>
    </Panel>
  )
}
