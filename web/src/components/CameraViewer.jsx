import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { appStore, useSim } from '../state/store.js'
import { fixed, mmss } from '../lib/format.js'
import { stateStyle } from '../lib/theme.js'
import { useNow } from '../lib/useNow.js'
import { useCameraStream } from './useCameraStream.js'
import { CloseButton, Modal, RoverSwitch, roverFromKey } from './Modal.jsx'

// Large camera viewer (PROTOCOL.md version 5 allows frames up to 1280 x 720). Polls only
// this view, about 3 frames a second, one request at a time; the sensor panel's small
// feed pauses meanwhile. Scroll zooms (digitally) around the cursor, drag pans, a
// double-click resets. A camera HUD sits on top: heading, attitude, speed, battery.

const MAX_W = 1280
const MAX_H = 720
const PERIOD = 330 // ms between frame starts
const STALE_S = 2.5 // older than this, the picture is called stale
const ZMAX = 8
const FPV_VFOV = 48 // assumed vertical field of view (deg) for the pitch ladder
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
const num = (v) => typeof v === 'number' && Number.isFinite(v)

// Compass heading in degrees (0 = north, 90 = east) from the protocol's yaw (0 = +x east, CCW).
export const headingOf = (yaw) => ((((90 - (yaw * 180) / Math.PI) % 360) + 360) % 360)

function useBoxSize(ref) {
  const [size, setSize] = useState({ w: 0, h: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth
      const h = el.clientHeight
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}

// ---------------------------------------------------------------- HUD

const COMPASS = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' }

function HeadingTape({ hdg }) {
  const W = 300
  const span = 40 // degrees either side
  const k = W / 2 / span
  const ticks = []
  const first = Math.ceil((hdg - span) / 5) * 5
  for (let d = first; d <= hdg + span; d += 5) {
    const x = W / 2 + (d - hdg) * k
    const n = ((d % 360) + 360) % 360
    const major = n % 15 === 0
    ticks.push(
      <g key={d}>
        <line x1={x} x2={x} y1={major ? 20 : 24} y2={30} />
        {major && (
          <text x={x} y={15} data-card={COMPASS[n] ? true : undefined}>
            {COMPASS[n] || String(n).padStart(3, '0')}
          </text>
        )}
      </g>,
    )
  }
  return (
    <div className="hc-tape">
      <svg viewBox={`0 0 ${W} 34`} width={W} height={34} aria-hidden="true">
        <defs>
          <linearGradient id="hc-fade" x1="0" x2="1">
            <stop offset="0" stopColor="#fff" stopOpacity="0" />
            <stop offset="0.18" stopColor="#fff" stopOpacity="1" />
            <stop offset="0.82" stopColor="#fff" stopOpacity="1" />
            <stop offset="1" stopColor="#fff" stopOpacity="0" />
          </linearGradient>
          <mask id="hc-mask">
            <rect width={W} height="34" fill="url(#hc-fade)" />
          </mask>
        </defs>
        <g mask="url(#hc-mask)">{ticks}</g>
        <path d={`M${W / 2 - 5} 34 L${W / 2} 28 L${W / 2 + 5} 34 Z`} className="hc-caret" />
      </svg>
      <b className="hc-hdg">{String(Math.round(hdg) % 360).padStart(3, '0')}°</b>
    </div>
  )
}

// Pitch ladder and reticle in stage pixels, so the lines stay crisp at any size.
function Reticle({ W, H, pitch, roll, ladder, fpv }) {
  const cx = W / 2
  const cy = H / 2
  const pxDeg = H / FPV_VFOV
  const p = num(pitch) ? clamp(pitch, -30, 30) : 0
  const r = num(roll) ? clamp(roll, -60, 60) : 0
  const gap = Math.max(18, W * 0.02)
  const arm = Math.max(26, W * 0.035)
  const br = Math.min(W, H) * 0.09 // corner brackets of the centre box
  return (
    <svg className="hc-svg" viewBox={`0 0 ${W} ${H}`} width={W} height={H} aria-hidden="true">
      <defs>
        {/* The ladder stays in the middle of the picture, clear of the corner readouts and the tape. */}
        <clipPath id="hc-ladder-clip">
          <rect x={W * 0.22} y={H * 0.2} width={W * 0.56} height={H * 0.62} />
        </clipPath>
      </defs>
      {ladder && (
        <g clipPath="url(#hc-ladder-clip)">
        <g transform={`rotate(${-r} ${cx} ${cy}) translate(0 ${p * pxDeg})`} className="hc-ladder">
          <line x1={cx - W * 0.2} x2={cx - gap * 2.6} y1={cy} y2={cy} />
          <line x1={cx + gap * 2.6} x2={cx + W * 0.2} y1={cy} y2={cy} />
          {[-20, -10, 10, 20].map((d) => {
            const y = cy - d * pxDeg
            const w = W * (Math.abs(d) === 10 ? 0.06 : 0.045)
            return (
              <g key={d} className="hc-rung" data-neg={d < 0 || undefined}>
                <line x1={cx - gap * 2.6 - w} x2={cx - gap * 2.6} y1={y} y2={y} />
                <line x1={cx + gap * 2.6} x2={cx + gap * 2.6 + w} y1={y} y2={y} />
                <text x={cx + gap * 2.6 + w + 6} y={y + 4}>
                  {d > 0 ? `+${d}` : d}
                </text>
              </g>
            )
          })}
        </g>
        </g>
      )}
      {fpv && (
      <g className="hc-ret">
        <line x1={cx - gap - arm} x2={cx - gap} y1={cy} y2={cy} />
        <line x1={cx + gap} x2={cx + gap + arm} y1={cy} y2={cy} />
        <line x1={cx} x2={cx} y1={cy + gap} y2={cy + gap + arm * 0.6} />
        <circle cx={cx} cy={cy} r={1.6} />
        <path
          className="hc-box"
          d={`M${cx - br * 1.6} ${cy - br + 10}V${cy - br}H${cx - br * 1.6 + 14} M${cx + br * 1.6 - 14} ${cy - br}H${cx + br * 1.6}V${cy - br + 10} M${cx - br * 1.6} ${cy + br - 10}V${cy + br}H${cx - br * 1.6 + 14} M${cx + br * 1.6 - 14} ${cy + br}H${cx + br * 1.6}V${cy + br - 10}`}
        />
      </g>
      )}
    </svg>
  )
}

function CamHud({ roverId, view, W, H, zoom, live, age, frameSize }) {
  const meta = useSim((s) => s.roverMeta.find((m) => m.id === roverId) || null)
  const rover = useSim((s) => s.rovers[roverId] || null)
  const t = useSim((s) => s.t)
  if (!W || !H) return null
  const hdg = headingOf(rover?.yaw ?? 0)
  const sn = rover?.sensors
  const speed = num(sn?.ground_speed) ? sn.ground_speed : rover?.speed
  const st = stateStyle(rover?.state)
  const bat = num(rover?.battery) ? Math.round(rover.battery * 100) : null
  const sgn = (v) => (num(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(1)}°` : '--')
  return (
    <div className="hud-cam" data-view={view} style={{ '--rover': meta?.color || '#fff' }}>
      <div className="hc-corner hc-tl">
        <span className="hc-rec" data-live={live || undefined}>
          <i aria-hidden="true" />
          {live ? 'LIVE' : age === null ? 'NO SIGNAL' : `STALE ${Math.round(age)} S`}
        </span>
        <span className="hc-name">{(meta?.name || roverId).toUpperCase()}</span>
        <span className="hc-dim">{view === 'fpv' ? 'CAM 1 MAST · FPV' : 'CAM 2 · CHASE'}</span>
      </div>
      <div className="hc-corner hc-tr">
        <span>T+ {mmss(t)}</span>
        <span className="hc-dim">{frameSize ? `${frameSize.w}×${frameSize.h}` : ''}</span>
        {zoom > 1.01 && <span className="hc-zoom">ZOOM {zoom.toFixed(1)}×</span>}
      </div>
      <HeadingTape hdg={hdg} />
      <Reticle W={W} H={H} pitch={sn?.pitch} roll={sn?.roll} ladder={view === 'fpv' && zoom < 1.01} fpv={view === 'fpv'} />
      <div className="hc-corner hc-bl">
        <span>
          <em>SPD</em> {num(speed) ? `${speed.toFixed(2)} m/s` : '--'}
        </span>
        <span>
          <em>POS</em> {rover ? `${fixed(rover.x, 1)}, ${fixed(rover.y, 1)}` : '--'}
        </span>
        <span className="hc-state" style={{ '--st': st.color }}>
          {rover ? st.label.toUpperCase() : 'NO DATA'}
        </span>
      </div>
      <div className="hc-corner hc-br">
        <span>
          <em>PITCH</em> {sgn(sn?.pitch)} <em>ROLL</em> {sgn(sn?.roll)}
        </span>
        <span className="hc-bat" data-low={(bat !== null && bat < 20) || undefined}>
          <em>BAT</em> {bat === null ? '--' : `${bat}%`}
          <i aria-hidden="true">
            <b style={{ width: `${bat ?? 0}%` }} />
          </i>
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- viewer

export function CameraViewer({ roverId: initialRover, view: initialView = 'fpv' }) {
  const meta = useSim((s) => s.roverMeta)
  const [roverId, setRoverId] = useState(initialRover)
  const [view, setView] = useState(initialView === 'chase' ? 'chase' : 'fpv')
  const wrap = useRef(null)
  const stage = useRef(null)
  const img = useRef(null)
  const area = useBoxSize(wrap)
  const z = useRef({ k: 1, x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const drag = useRef(null)
  const close = useCallback(() => appStore.patch({ modal: null }), [])
  const known = meta.some((m) => m.id === roverId)

  // After a reset the rover may be gone: fall back to the first one, or close.
  useEffect(() => {
    if (known) return
    if (meta.length) setRoverId(meta[0].id)
    else close()
  }, [known, meta, close])

  // Request a frame that fills the stage at the screen's pixel density, capped at the
  // protocol's 1280 x 720 (the sim clamps anything larger anyway). Quantised, so a small
  // resize does not restart the stream.
  const [natural, setNaturalAr] = useState(16 / 9)
  const ar = natural
  const sw = Math.max(0, Math.floor(Math.min(area.w, area.h * ar)))
  const sh = Math.max(0, Math.floor(sw / ar))
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  const reqW = clamp(Math.ceil((sw * dpr) / 128) * 128, 320, MAX_W)
  const reqH = Math.min(MAX_H, Math.round((reqW * 9) / 16))
  const cam = useCameraStream({ roverId: known ? roverId : null, view, w: reqW, h: reqH, period: PERIOD, enabled: sw > 0 })
  useEffect(() => {
    if (cam.natural?.w && cam.natural?.h) setNaturalAr(cam.natural.w / cam.natural.h)
  }, [cam.natural])
  const now = useNow(500)
  const age = cam.frame ? Math.max(0, now - cam.at / 1000) : null
  const live = !!cam.frame && !cam.problem && age < STALE_S

  const apply = useCallback((animate = false) => {
    const el = img.current
    if (!el) return
    const { k, x, y } = z.current
    el.style.transition = animate ? 'transform 200ms cubic-bezier(0.2, 0.8, 0.2, 1)' : 'none'
    el.style.transform = `translate(${x}px, ${y}px) scale(${k})`
  }, [])
  const clampPan = useCallback(() => {
    const q = z.current
    q.x = clamp(q.x, sw * (1 - q.k), 0)
    q.y = clamp(q.y, sh * (1 - q.k), 0)
  }, [sw, sh])
  const reset = useCallback(() => {
    z.current = { k: 1, x: 0, y: 0 }
    apply(true)
    setZoom(1)
  }, [apply])
  const zoomAt = useCallback(
    (mx, my, k2, animate = false) => {
      const q = z.current
      k2 = clamp(k2, 1, ZMAX)
      q.x = mx - ((mx - q.x) * k2) / q.k
      q.y = my - ((my - q.y) * k2) / q.k
      q.k = k2
      clampPan()
      apply(animate)
      setZoom(k2)
    },
    [apply, clampPan],
  )

  // A new rover or stage size starts unzoomed.
  useEffect(() => {
    z.current = { k: 1, x: 0, y: 0 }
    apply(false)
    setZoom(1)
  }, [roverId, sw, sh, apply])

  // Wheel zoom needs a non-passive listener to keep the page from scrolling.
  useEffect(() => {
    const el = stage.current
    if (!el) return
    const onWheel = (e) => {
      e.preventDefault()
      const r = el.getBoundingClientRect()
      const d = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY
      zoomAt(e.clientX - r.left, e.clientY - r.top, z.current.k * Math.exp(-d * 0.0016))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoomAt])

  const onPointerDown = (e) => {
    if (e.button !== 0 || z.current.k <= 1.001) return
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY }
    e.currentTarget.dataset.dragging = 'true'
  }
  const onPointerMove = (e) => {
    const d = drag.current
    if (!d || d.id !== e.pointerId) return
    z.current.x += e.clientX - d.x
    z.current.y += e.clientY - d.y
    d.x = e.clientX
    d.y = e.clientY
    clampPan()
    apply(false)
  }
  const endDrag = (e) => {
    if (drag.current?.id !== e.pointerId) return
    drag.current = null
    delete e.currentTarget.dataset.dragging
  }

  const onKey = (e) => {
    const id = roverFromKey(e, meta)
    if (id) return setRoverId(id)
    const k = e.key.toLowerCase()
    if (k === 'v') setView((v) => (v === 'fpv' ? 'chase' : 'fpv'))
    else if (k === '+' || k === '=') zoomAt(sw / 2, sh / 2, z.current.k * 1.4, true)
    else if (k === '-' || k === '_') zoomAt(sw / 2, sh / 2, z.current.k / 1.4, true)
    else if (k === '0') reset()
  }

  const name = meta.find((m) => m.id === roverId)?.name || roverId
  const zoomed = zoom > 1.001
  return (
    <Modal label={`${name} camera`} onClose={close} className="cv" onKey={onKey}>
      <header className="ix-head">
        <h2 className="ix-title">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 7.5h3l1.6-2.3h6.8L16 7.5h3A1.5 1.5 0 0 1 20.5 9v8.5A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5V9A1.5 1.5 0 0 1 5 7.5" />
            <circle cx="12" cy="13" r="3.4" />
          </svg>
          Camera
        </h2>
        <RoverSwitch value={roverId} onChange={setRoverId} />
        <span className="seg seg-sm ix-views" role="group" aria-label="Camera view">
          <button type="button" aria-pressed={view === 'fpv'} onClick={() => setView('fpv')} title="Onboard camera on the mast (V)">
            FPV
          </button>
          <button type="button" aria-pressed={view === 'chase'} onClick={() => setView('chase')} title="Behind and above the rover (V)">
            Chase
          </button>
        </span>
        <span className="ix-live" data-state={live ? 'live' : cam.frame ? 'stale' : 'off'} role="status">
          <i aria-hidden="true" />
          {live ? 'Live' : cam.frame ? `Stale, ${Math.round(age)} s old` : cam.problem ? 'No picture' : 'Connecting'}
        </span>
        <CloseButton onClick={close} />
      </header>
      <div className="cv-wrap" ref={wrap}>
        <div
          className="cv-stage"
          ref={stage}
          style={{ width: sw, height: sh }}
          data-zoomed={zoomed || undefined}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onDoubleClick={reset}
        >
          {cam.frame && <img ref={img} src={cam.frame} alt={`${name}, ${view === 'fpv' ? 'onboard' : 'chase'} camera`} draggable={false} />}
          {cam.frame && <CamHud roverId={roverId} view={view} W={sw} H={sh} zoom={zoom} live={live} age={age} frameSize={cam.natural} />}
          {(!cam.frame || cam.problem) && (
            <div className="cv-off" data-over={!!cam.frame || undefined}>
              <b>{cam.problem ? cam.problem.title : 'Connecting camera'}</b>
              {cam.problem && <span>{cam.problem.sub}</span>}
              {cam.problem && (
                <button type="button" className="sp-cam-retry" onClick={cam.retry}>
                  Retry
                </button>
              )}
            </div>
          )}
        </div>
      </div>
      <footer className="ix-foot">
        <span>
          <kbd>Scroll</kbd> zoom <kbd>Drag</kbd> pan <kbd>Double-click</kbd> reset <kbd>V</kbd> FPV or Chase <kbd>1</kbd>-<kbd>{Math.min(9, meta.length) || 1}</kbd> rover <kbd>Esc</kbd> close
        </span>
        <span className="ix-foot-r">
          {zoomed && (
            <button type="button" className="ix-link" onClick={reset}>
              Reset zoom
            </button>
          )}
          <b>{zoom.toFixed(1)}×</b>
          {cam.natural && (
            <span className="ix-dim">
              {cam.natural.w}×{cam.natural.h} px
            </span>
          )}
        </span>
      </footer>
    </Modal>
  )
}
