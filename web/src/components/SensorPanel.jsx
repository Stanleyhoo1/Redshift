import { useEffect, useRef, useState } from 'react'
import { simStore, useSim } from '../state/store.js'
import { sampleGround } from '../state/simState.js'
import { camDown, fetchCameraFrame } from '../net/connection.js'
import { fixed } from '../lib/format.js'
import { HAZARD, SIGNAL, instrumentStyle, kindStyle } from '../lib/theme.js'
import { useHudRect } from '../lib/useHudRect.js'
import { Battery, StatePill, brainLevel, fmtMs } from './RoverBits.jsx'
import { InstrumentIcon, Tubes } from './SampleBits.jsx'
import { PlannerRow } from './Planner.jsx'
import { reducedMotion } from '../scene/sceneConstants.js'

const num = (v) => typeof v === 'number' && Number.isFinite(v)
const CAM_W = 320
const CAM_H = 240
const CAM_PERIOD = 250 // ms between frame starts: about 4 per second
const CAM_RETRY_NET = 4000 // no connection: try again soon
export const CAM_RETRY_DOWN = 30000 // 503 / 404: rendering is off on that machine, rarely comes back

// ---------------------------------------------------------------- camera

export function cameraProblem(err) {
  if (err.status === 404) return { title: 'Camera unavailable', sub: 'This rover has no camera feed' }
  if (err.status === 0) return { title: 'Camera unavailable', sub: 'No connection to the sim' }
  if (err.status === 503) return { title: 'Camera unavailable', sub: err.message && !/^HTTP/.test(err.message) ? err.message : 'Rendering is off on this machine' }
  return { title: 'Camera unavailable', sub: err.message || 'The sim did not send a picture' }
}

// Polls GET /api/camera/{id}.jpg while mounted (panel open, this rover only). A new
// request starts only after the previous one finished, so slow renders never pile up.
// Frames are decoded before they are swapped in, so the picture never flashes blank.
function CameraFeed({ roverId }) {
  const [view, setView] = useState('fpv')
  const [frame, setFrame] = useState(null)
  const [problem, setProblem] = useState(() => (Date.now() < camDown.until ? camDown.problem : null))
  const [waiting, setWaiting] = useState(true)
  const [attempt, setAttempt] = useState(0)
  const current = useRef(null)

  useEffect(() => {
    let stopped = false
    let timer = null
    let ctrl = null
    setWaiting(true)
    const loop = async () => {
      if (stopped) return
      const down = camDown.until - Date.now()
      if (down > 0) {
        setProblem(camDown.problem)
        setWaiting(false)
        timer = setTimeout(loop, down)
        return
      }
      const started = performance.now()
      ctrl = new AbortController()
      try {
        const blob = await fetchCameraFrame(roverId, view, CAM_W, CAM_H, ctrl.signal)
        const url = URL.createObjectURL(blob)
        try {
          const img = new Image()
          img.src = url
          await img.decode()
        } catch {
          URL.revokeObjectURL(url)
          throw Object.assign(new Error('The picture could not be decoded'), { status: -1 })
        }
        if (stopped) {
          URL.revokeObjectURL(url)
          return
        }
        const old = current.current
        current.current = url
        setFrame(url)
        setProblem(null)
        setWaiting(false)
        if (old) setTimeout(() => URL.revokeObjectURL(old), 500)
        timer = setTimeout(loop, Math.max(0, CAM_PERIOD - (performance.now() - started)))
      } catch (err) {
        if (stopped || err?.name === 'AbortError') return
        const p = cameraProblem(err)
        setProblem(p)
        setWaiting(false)
        if (err.status === 503 || err.status === 404) {
          camDown.until = Date.now() + CAM_RETRY_DOWN
          camDown.problem = p
        }
        timer = setTimeout(loop, err.status === 503 || err.status === 404 ? CAM_RETRY_DOWN : CAM_RETRY_NET)
      }
    }
    loop()
    return () => {
      stopped = true
      clearTimeout(timer)
      ctrl?.abort()
    }
  }, [roverId, view, attempt])

  const retry = () => {
    camDown.until = 0
    setProblem(null)
    setAttempt((n) => n + 1)
  }

  // Drop the last frame when the panel closes (the parent remounts this per rover).
  useEffect(
    () => () => {
      if (current.current) URL.revokeObjectURL(current.current)
      current.current = null
    },
    [],
  )
  const live = frame && !problem
  return (
    <figure className="sp-cam" data-state={live ? 'live' : problem ? 'off' : 'wait'}>
      {frame && <img src={frame} alt={`${view === 'fpv' ? 'Onboard' : 'Chase'} camera view`} width={CAM_W} height={CAM_H} />}
      {!live && (
        <div className="sp-cam-off">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 7.5h3l1.6-2.3h6.8L16 7.5h3A1.5 1.5 0 0 1 20.5 9v8.5A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5V9A1.5 1.5 0 0 1 5 7.5" />
            <circle cx="12" cy="13" r="3.4" />
            {problem && <path d="M3.5 3.5l17 17" />}
          </svg>
          <b>{problem ? problem.title : waiting ? 'Connecting camera' : 'No picture yet'}</b>
          {problem && <span>{problem.sub}</span>}
          {problem && (
            <button type="button" className="sp-cam-retry" onClick={retry}>
              Retry
            </button>
          )}
        </div>
      )}
      <figcaption className="sp-cam-bar">
        <span className="sp-cam-live" data-live={live || undefined}>
          <i aria-hidden="true" />
          {live ? 'Live' : 'Camera'}
        </span>
        <span className="seg seg-xs" role="group" aria-label="Camera view">
          <button type="button" aria-pressed={view === 'fpv'} onClick={() => setView('fpv')} title="Onboard camera on the mast">
            FPV
          </button>
          <button type="button" aria-pressed={view === 'chase'} onClick={() => setView('chase')} title="Behind and above the rover">
            Chase
          </button>
        </span>
      </figcaption>
    </figure>
  )
}

// ---------------------------------------------------------------- LiDAR radar

// Height of a return relative to the ground under the rover: below, level, above, high.
const HEIGHT_BANDS = [
  { max: -0.6, color: '#6f8cff', label: 'Below' },
  { max: 0.25, color: SIGNAL, label: 'Level' },
  { max: 0.7, color: '#f3b24a', label: 'Above' },
  { max: Infinity, color: HAZARD, label: 'High' },
]
const bandOf = (dz) => {
  for (let b = 0; b < HEIGHT_BANDS.length; b++) if (dz <= HEIGHT_BANDS[b].max) return b
  return HEIGHT_BANDS.length - 1
}

// Top-down view of the latest sweep, rover at the centre facing up. The dashed ring is
// the current effective range (sensors.lidar_range): it shrinks in a dust storm.
// Also shown on a clarification card when the rover's camera is unavailable.
export function LidarRadar({ roverId, size, showRange = true }) {
  const canvas = useRef(null)
  const st = useRef({ ring: null, nominal: 8, scan: null, prev: null, prevPose: null, pose: null, last: 0 })

  useEffect(() => {
    const cv = canvas.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    cv.width = Math.round(size * dpr)
    cv.height = Math.round(size * dpr)
    const S = st.current
    S.ring = null
    S.scan = null
    S.prev = null
    let raf = 0
    const draw = (now) => {
      raf = requestAnimationFrame(draw)
      if (now - S.last < 33) return // about 30 fps is plenty for a 5 Hz sensor
      const dt = Math.min(0.1, (now - S.last) / 1000)
      S.last = now
      const s = simStore.get()
      const r = s.rovers[roverId]
      const range = num(r?.sensors?.lidar_range) && r.sensors.lidar_range > 0 ? r.sensors.lidar_range : null
      if (range) S.nominal = Math.max(S.nominal, range)
      if (r && r.scan !== S.scan) {
        S.prev = S.scan
        S.scan = r.scan
      }
      const target = range ?? S.nominal
      S.ring = S.ring === null ? target : S.ring + (target - S.ring) * (1 - Math.exp(-dt * 5))

      const W = size
      const c = W / 2
      const R = W / 2 - 6
      const scale = R / (S.nominal * 1.08)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, W)

      // Scope face
      ctx.fillStyle = 'rgba(6, 12, 18, 0.9)'
      ctx.beginPath()
      ctx.arc(c, c, R, 0, Math.PI * 2)
      ctx.fill()
      ctx.save()
      ctx.beginPath()
      ctx.arc(c, c, R, 0, Math.PI * 2)
      ctx.clip()

      // Outside the current range: shaded, so a shrinking ring reads at a glance.
      const reduced = range !== null && range < S.nominal * 0.8
      const ringPx = S.ring * scale
      if (reduced) {
        ctx.fillStyle = 'rgba(185, 138, 92, 0.16)'
        ctx.beginPath()
        ctx.arc(c, c, R, 0, Math.PI * 2)
        ctx.arc(c, c, ringPx, 0, Math.PI * 2, true)
        ctx.fill()
      }

      // Range rings every 2 m and the cross
      ctx.strokeStyle = 'rgba(120, 150, 190, 0.18)'
      ctx.lineWidth = 1
      for (let m = 2; m < S.nominal * 1.08; m += 2) {
        ctx.beginPath()
        ctx.arc(c, c, m * scale, 0, Math.PI * 2)
        ctx.stroke()
      }
      ctx.beginPath()
      ctx.moveTo(c, c - R)
      ctx.lineTo(c, c + R)
      ctx.moveTo(c - R, c)
      ctx.lineTo(c + R, c)
      ctx.stroke()

      // Sweep
      if (!reducedMotion && r?.scan) {
        const a = ((now / 1000) * Math.PI * 2 * 0.6) % (Math.PI * 2)
        const grad = ctx.createConicGradient ? ctx.createConicGradient(a - Math.PI / 2 - 0.9, c, c) : null
        if (grad) {
          grad.addColorStop(0, 'rgba(79, 224, 239, 0)')
          grad.addColorStop(0.143, 'rgba(79, 224, 239, 0.16)')
          grad.addColorStop(0.144, 'rgba(79, 224, 239, 0)')
          ctx.fillStyle = grad
          ctx.beginPath()
          ctx.arc(c, c, Math.min(R, ringPx), 0, Math.PI * 2)
          ctx.fill()
        }
      }

      // Points, relative to the rover pose (rover faces up)
      if (r) {
        const cy = Math.cos(r.yaw)
        const sy = Math.sin(r.yaw)
        const g = sampleGround(s.terrain, r.x, r.y)
        const gz = g ?? r.z - 0.3
        const plot = (scan, alpha, sz) => {
          if (!scan) return
          for (let b = 0; b < HEIGHT_BANDS.length; b++) {
            ctx.fillStyle = HEIGHT_BANDS[b].color
            ctx.globalAlpha = alpha
            for (let p = 0; p + 2 < scan.length; p += 3) {
              const dx = scan[p] - r.x
              const dy = scan[p + 1] - r.y
              if (bandOf(scan[p + 2] - gz) !== b) continue
              const fwd = dx * cy + dy * sy
              const left = -dx * sy + dy * cy
              const px = c - left * scale
              const py = c - fwd * scale
              ctx.fillRect(px - sz / 2, py - sz / 2, sz, sz)
            }
          }
          ctx.globalAlpha = 1
        }
        plot(S.prev, 0.28, 2)
        plot(S.scan, 0.95, 2.6)
      }

      // Effective range ring (only with a reading: no fake ring from an older sim)
      const hasData = range !== null || !!r?.scan
      if (hasData) ctx.setLineDash([4, 4])
      if (hasData) {
        ctx.lineWidth = reduced ? 2 : 1.5
        ctx.strokeStyle = reduced ? '#f3b24a' : 'rgba(79, 224, 239, 0.75)'
        ctx.beginPath()
        ctx.arc(c, c, ringPx, 0, Math.PI * 2)
        ctx.stroke()
        ctx.setLineDash([])
      }
      ctx.restore()

      // Bezel and heading tick
      ctx.strokeStyle = 'rgba(120, 150, 190, 0.45)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(c, c, R, 0, Math.PI * 2)
      ctx.stroke()

      // The rover itself
      const meta = s.roverMeta.find((m) => m.id === roverId)
      ctx.fillStyle = meta?.color || '#ffffff'
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(c, c - 7)
      ctx.lineTo(c + 5, c + 5)
      ctx.lineTo(c, c + 2.5)
      ctx.lineTo(c - 5, c + 5)
      ctx.closePath()
      ctx.stroke()
      ctx.fill()

      if (!hasData || !showRange) return
      // Range label on the ring
      ctx.font = '600 11px "Archivo Variable", system-ui, sans-serif'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      const label = `${fixed(range ?? S.nominal, 1)} m`
      const lx = c + ringPx * 0.7071 + 3
      const ly = c - ringPx * 0.7071 - 4
      ctx.fillStyle = 'rgba(6, 12, 18, 0.85)'
      const tw = ctx.measureText(label).width
      ctx.fillRect(lx - 2, ly - 7, tw + 4, 14)
      ctx.fillStyle = reduced ? '#f3b24a' : '#93f0fa'
      ctx.fillText(label, lx, ly)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [roverId, size, showRange])

  return <canvas ref={canvas} className="sp-radar-canvas" style={{ width: size, height: size }} aria-hidden="true" />
}

function RadarBlock({ rover, size }) {
  const s = rover?.sensors
  const range = num(s?.lidar_range) ? s.lidar_range : null
  const hits = num(s?.lidar_hits) ? s.lidar_hits : rover?.scan ? Math.floor(rover.scan.length / 3) : null
  const nominal = useRef(8)
  if (range) nominal.current = Math.max(nominal.current, range)
  const reduced = range !== null && range < nominal.current * 0.8
  return (
    <div className="sp-radar" role="img" aria-label={`LiDAR: ${range ? `${fixed(range, 1)} m range` : 'range unknown'}, ${hits ?? 'no'} hits`}>
      <div className="sp-radar-scope" style={{ width: size, height: size }}>
        <LidarRadar roverId={rover.id} size={size} />
        {!rover.scan && <span className="sp-radar-none">No LiDAR data</span>}
        {/* High at the top corner, low at the bottom one: the corners are free of the scope. */}
        <ul className="sp-radar-key" aria-hidden="true">
          {[HEIGHT_BANDS[3], HEIGHT_BANDS[2]].map((b) => (
            <li key={b.label}>
              <i style={{ background: b.color }} />
              {b.label}
            </li>
          ))}
        </ul>
        <ul className="sp-radar-key sp-radar-key-low" aria-hidden="true">
          {[HEIGHT_BANDS[1], HEIGHT_BANDS[0]].map((b) => (
            <li key={b.label}>
              <i style={{ background: b.color }} />
              {b.label}
            </li>
          ))}
        </ul>
      </div>
      <div className="sp-radar-foot">
        <span className="sp-k">LiDAR</span>
        <span className="sp-v" data-warn={reduced || undefined}>
          {range === null ? '--' : `${fixed(range, 1)} m`}
        </span>
        <span className="sp-v sp-dim">{hits === null ? '' : `${hits} hits`}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- IMU, wheels, spectrometer

function Attitude({ pitch, roll }) {
  const p = num(pitch) ? Math.max(-40, Math.min(40, pitch)) : 0
  const r = num(roll) ? Math.max(-60, Math.min(60, roll)) : 0
  const k = 1.3 // px per degree of pitch
  return (
    <svg className="sp-att" viewBox="-30 -30 60 60" aria-hidden="true">
      <defs>
        <clipPath id="sp-att-clip">
          <circle r="27" />
        </clipPath>
      </defs>
      <g clipPath="url(#sp-att-clip)">
        <g transform={`rotate(${-r}) translate(0 ${p * k})`} className="sp-att-world">
          <rect x="-70" y="-70" width="140" height="70" fill="#163454" />
          <rect x="-70" y="0" width="140" height="70" fill="#4a3424" />
          <line x1="-70" x2="70" y1="0" y2="0" stroke="#e8eef5" strokeWidth="1.2" />
          {[-20, -10, 10, 20].map((d) => (
            <line key={d} x1={-Math.abs(d) / 2 - 3} x2={Math.abs(d) / 2 + 3} y1={-d * k} y2={-d * k} stroke="rgba(232,238,245,0.55)" strokeWidth="0.8" />
          ))}
        </g>
      </g>
      <circle r="27" fill="none" stroke="rgba(120,150,190,0.55)" strokeWidth="1.2" />
      <path d="M-15 0h8l3 4 3-4 3 4 3-4h8" fill="none" stroke="#f3b24a" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      <circle r="1.6" fill="#f3b24a" />
    </svg>
  )
}

function Bar({ value, max = 1, level, marks, peak, peakLevel }) {
  const pct = num(value) ? Math.max(0, Math.min(100, (value / max) * 100)) : 0
  return (
    <span className="sp-bar" data-level={level}>
      {num(peak) && <u className="sp-bar-peak" data-level={peakLevel} style={{ width: `${Math.min(100, (peak / max) * 100)}%` }} />}
      <i style={{ width: `${pct}%` }} />
      {marks?.map((m) => (
        <b key={m} style={{ left: `${(m / max) * 100}%` }} />
      ))}
    </span>
  )
}

const slipLevel = (v) => (!num(v) ? 'none' : v >= 0.45 ? 'bad' : v >= 0.2 ? 'warn' : 'ok')
const HOLD_MS = 4500

// Peak hold, like an audio meter: a slip spike lasts a second or two of wall time at
// 3x sim speed, so the meter keeps its peak (and colour) on screen for a few seconds.
function usePeak(value, roverId) {
  const peak = useRef({ v: 0, at: 0, id: roverId })
  const now = performance.now()
  const p = peak.current
  if (p.id !== roverId) Object.assign(p, { v: 0, at: 0, id: roverId })
  if (num(value) && (value >= p.v || now - p.at > HOLD_MS)) {
    p.v = value
    p.at = now
  }
  return p.v
}
const loadLevel = (v) => (!num(v) ? 'none' : v >= 0.9 ? 'bad' : v >= 0.7 ? 'warn' : 'ok')

function Readouts({ rover, limit }) {
  const s = rover.sensors
  const targets = useSim((st) => st.targets)
  const pitch = s?.pitch
  const roll = s?.roll
  const tilt = Math.max(Math.abs(num(pitch) ? pitch : 0), Math.abs(num(roll) ? roll : 0))
  const lim = num(limit) ? limit : 30
  const tiltLevel = !s ? 'none' : tilt >= lim ? 'bad' : tilt >= lim * 0.6 ? 'warn' : 'ok'
  const wheel = s?.wheel_speed
  const ground = s?.ground_speed
  const vmax = Math.max(0.6, num(wheel) ? wheel : 0, num(ground) ? ground : 0)
  const spectro = s?.spectro && typeof s.spectro === 'object' ? s.spectro : null
  const tgt = spectro ? targets.find((t) => t.id === spectro.id) : null
  const ks = tgt ? kindStyle(tgt.kind) : null
  const slip = s?.slip
  const slipPeak = usePeak(slip, rover.id)
  const shownLevel = slipLevel(Math.max(num(slip) ? slip : 0, slipPeak))
  const held = num(slip) && slipPeak >= 0.2 && slipPeak > slip + 0.08
  return (
    <div className="sp-read">
      <div className="sp-imu">
        <Attitude pitch={pitch} roll={roll} />
        <dl className="sp-dl" data-level={tiltLevel}>
          <div className="sp-dl-head">IMU</div>
          <div>
            <dt>Pitch</dt>
            <dd>{num(pitch) ? `${fixed(pitch, 1)} deg` : '--'}</dd>
          </div>
          <div>
            <dt>Roll</dt>
            <dd>{num(roll) ? `${fixed(roll, 1)} deg` : '--'}</dd>
          </div>
        </dl>
      </div>
      <div className="sp-wheels">
        <div className="sp-row">
          <span className="sp-k">Wheels</span>
          <span className="sp-v">{num(wheel) ? `${fixed(wheel, 2)} m/s` : '--'}</span>
          <span className="sp-k sp-k2">Ground</span>
          <span className="sp-v">{num(ground) ? `${fixed(ground, 2)} m/s` : '--'}</span>
        </div>
        {/* Ground speed filled, wheel speed outlined: the gap between them is slip. */}
        <span className="sp-speed" aria-hidden="true">
          <i className="sp-speed-wheel" style={{ width: `${num(wheel) ? Math.min(100, (wheel / vmax) * 100) : 0}%` }} />
          <i className="sp-speed-ground" style={{ width: `${num(ground) ? Math.min(100, (ground / vmax) * 100) : 0}%` }} />
        </span>
        <div className="sp-row sp-row-bar" data-level={num(slip) ? shownLevel : 'none'} title={held ? `Peak ${Math.round(slipPeak * 100)}% in the last few seconds` : undefined}>
          <span className="sp-k">Slip</span>
          <Bar value={slip} level={slipLevel(slip)} marks={[0.2, 0.45]} peak={held ? slipPeak : null} peakLevel={slipLevel(slipPeak)} />
          <span className="sp-v sp-num">{num(slip) ? `${Math.round((held ? slipPeak : slip) * 100)}%` : '--'}</span>
        </div>
        <div className="sp-row sp-row-bar" data-level={loadLevel(s?.motor_load)}>
          <span className="sp-k">Load</span>
          <Bar value={s?.motor_load} level={loadLevel(s?.motor_load)} />
          <span className="sp-v sp-num">{num(s?.motor_load) ? `${Math.round(s.motor_load * 100)}%` : '--'}</span>
        </div>
      </div>
      {rover.sampling ? (
        <InstrumentRow sampling={rover.sampling} targets={targets} tubes={rover.samples} capacity={rover.capacity} />
      ) : (
      <div className="sp-row sp-spectro">
        <span className="sp-k">Spectro</span>
        {spectro ? (
          <span className="sp-v">
            <i className="sw-diamond" style={{ '--c': ks?.color || '#e6edf5' }} />
            {ks ? ks.label : 'Target'} {spectro.id}
            {num(spectro.dist) && <span className="sp-dim">, {fixed(spectro.dist, 1)} m</span>}
          </span>
        ) : (
          <span className="sp-v sp-dim">{s ? 'Nothing in range' : '--'}</span>
        )}
      </div>
      )}
    </div>
  )
}

// v4b: the instrument running on a target: progress and what it is doing.
function InstrumentRow({ sampling, targets, tubes, capacity }) {
  const ins = instrumentStyle(sampling.instrument)
  const t = targets.find((x) => x.id === sampling.target)
  const ks = t ? kindStyle(t.kind) : null
  const pct = Math.round(sampling.progress * 100)
  const tube = Array.isArray(tubes) && Number.isFinite(capacity) ? `, tube ${Math.min(capacity, tubes.length + 1)} of ${capacity}` : ''
  return (
    <div className="sp-instr" style={{ '--c': ks?.color || 'var(--signal)' }}>
      <div className="sp-row sp-row-bar">
        <span className="sp-k sp-instr-k">
          <InstrumentIcon kind={sampling.instrument} />
          {ins.label}
        </span>
        <Bar value={sampling.progress} level="instr" />
        <span className="sp-v sp-num">{pct}%</span>
      </div>
      <div className="sp-instr-sub">
        {ins.verb}
        {t ? ` at the ${ks.label.toLowerCase()} ${t.id}` : ''}
        {tube}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- brain budget

function BrainRow({ brain, budget, cycle }) {
  const hasBudget = num(budget) && budget > 0
  const level = brainLevel(brain, budget)
  if (!brain || !num(brain.avg_ms)) {
    return (
      <div className="sp-brain" data-level="none">
        <span className="sp-k">Brain</span>
        <span className="sp-v sp-dim">No compute data from this sim</span>
      </div>
    )
  }
  const max = Math.max(hasBudget ? budget * 1.2 : 0, (brain.peak_ms || 0) * 1.1, 1)
  const pct = (v) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`
  const over = num(brain.over) ? brain.over : 0
  return (
    <div className="sp-brain" data-level={level}>
      <span className="sp-k">Brain</span>
      <span className="sp-brain-nums">
        <b>{fmtMs(brain.avg_ms)} ms</b> avg, <b>{fmtMs(brain.peak_ms)} ms</b> peak
      </span>
      <span className="sp-brain-bar" aria-hidden="true">
        <i className="sp-brain-avg" style={{ width: pct(brain.avg_ms) }} />
        {num(brain.peak_ms) && <i className="sp-brain-peak" style={{ left: pct(brain.peak_ms) }} />}
        {num(brain.last_ms) && <i className="sp-brain-last" style={{ left: pct(brain.last_ms) }} />}
        {hasBudget && (
          <i className="sp-brain-budget" style={{ left: pct(budget) }}>
            <span>{fmtMs(budget)} ms</span>
          </i>
        )}
      </span>
      <span className="sp-brain-budget-text">{hasBudget ? `budget ${fmtMs(budget)} ms${num(cycle) ? ` per ${Math.round(cycle)} ms cycle` : ' per cycle'}` : 'no stated budget'}</span>
      {hasBudget && (
        <span className="sp-brain-verdict" data-over={over > 0 || undefined}>
          {over > 0 ? `${over} over` : 'Within budget'}
        </span>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- panel

// Matches the breakpoints in styles.css (camera 320 x 240, 256 x 192, 200 x 150).
function useRadarSize() {
  const pick = () => (window.matchMedia('(max-width: 1100px)').matches ? 126 : window.matchMedia('(max-height: 820px), (max-width: 1240px)').matches ? 168 : 214)
  const [size, setSize] = useState(pick)
  useEffect(() => {
    const update = () => setSize(pick())
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])
  return size
}

// Everything one rover's sensors read right now, plus how hard its brain is working.
export function SensorPanel({ roverId, onClose }) {
  const meta = useSim((s) => s.roverMeta.find((m) => m.id === roverId) || null)
  const rover = useSim((s) => s.rovers[roverId] || null)
  const hazards = useSim((s) => s.hazards)
  const budget = useSim((s) => s.fleet.budget_ms)
  const cycle = useSim((s) => s.fleet.cycle_ms)
  const limit = useSim((s) => s.limit.current)
  const planner = useSim((s) => s.planner)
  const targets = useSim((s) => s.targets)
  const ref = useHudRect('sensors')
  const radarSize = useRadarSize()
  if (!meta) return null
  return (
    <section className="hud hud-sensors" ref={ref} style={{ '--rover': meta.color }} aria-label={`${meta.name} live sensors`} data-fault={rover?.state === 'fault' || undefined}>
      <header className="sp-head">
        <span className="rover-dot" aria-hidden="true" />
        <h2 className="sp-name">{meta.name}</h2>
        <StatePill rover={rover} hazards={hazards} />
        <Battery value={rover?.battery} charging={rover?.state === 'charging'} />
        {rover?.capacity && <Tubes samples={rover.samples} capacity={rover.capacity} targets={targets} />}
        <span className="sp-title">Live sensors</span>
        <button type="button" className="sp-close" onClick={onClose} title="Close (Esc)" aria-label="Close the sensor panel">
          <svg viewBox="0 0 12 12" aria-hidden="true">
            <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" />
          </svg>
        </button>
      </header>
      {rover ? (
        <>
          <div className="sp-body">
            <CameraFeed key={`cam:${roverId}`} roverId={roverId} />
            <RadarBlock key={`radar:${roverId}`} rover={rover} size={radarSize} />
            <Readouts rover={rover} limit={limit} />
          </div>
          <div className="sp-brains">
            <BrainRow brain={rover.brain} budget={budget} cycle={cycle} />
            {planner && <PlannerRow planner={planner} />}
          </div>
        </>
      ) : (
        <p className="sp-empty">Waiting for telemetry from {meta.name}.</p>
      )}
    </section>
  )
}
