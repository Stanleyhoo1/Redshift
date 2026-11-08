import { useEffect, useRef } from 'react'
import { TELEMETRY_CAP, oldestIndex } from '../state/telemetry.js'
import { nowSeconds, simStore } from '../state/store.js'
import { simTimeNow } from '../state/simState.js'

// One time-series chart on a canvas, drawn from a telemetry ring buffer (see
// state/telemetry.js) about 5 times a second while mounted. X is sim time, spanning the
// last `win` wall-clock seconds (times the sim speed), so a busy browser that gets ticks
// in bursts still draws an even line; Y fits the data (and the reference lines) unless
// the spec fixes it. The draw loop allocates nothing: it walks the buffer in place.
//
// spec: {
//   title, unit, k (multiplier, e.g. 100 for %), digits,
//   lines: [{ s: series key, color, width, dash, fill, label }],
//   min, max (fixed bounds), minSpan (smallest y span), symmetric (around 0),
//   refs: [{ v, color, label }] (horizontal reference lines, in display units)
// }

const L = 40
const R = 10
const T = 8
const B = 18
const GAP_S = 2.5 // no samples for this long (wall-clock seconds at the sim speed): break the line

function niceStep(span) {
  const raw = span / 3
  const p = Math.pow(10, Math.floor(Math.log10(raw || 1)))
  const f = raw / p
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p
}

// As many decimals as the step needs (0.25 -> 2, 0.5 -> 1, 5 -> 0).
function stepDigits(step) {
  let d = 0
  while (d < 3 && Math.abs(step * 10 ** d - Math.round(step * 10 ** d)) > 1e-6) d++
  return d
}
const fmtTick = (v, step) => v.toFixed(stepDigits(step))
const fmtAgo = (s) => (s >= 60 ? `-${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}` : `-${Math.round(s)} s`)

export function TimeChart({ spec, getBuf, win, focus }) {
  const wrap = useRef(null)
  const canvas = useRef(null)
  const vals = useRef([])
  const live = useRef({ spec, getBuf, win })
  live.current.spec = spec
  live.current.getBuf = getBuf
  live.current.win = win

  useEffect(() => {
    const cv = canvas.current
    const box = wrap.current
    if (!cv || !box) return
    const ctx = cv.getContext('2d')
    let W = 0
    let H = 0
    let dpr = 1
    let raf = 0
    let last = -1e9
    const resize = () => {
      W = box.clientWidth
      H = box.clientHeight
      dpr = Math.min(2, window.devicePixelRatio || 1)
      cv.width = Math.max(1, Math.round(W * dpr))
      cv.height = Math.max(1, Math.round(H * dpr))
      cv.style.width = `${W}px`
      cv.style.height = `${H}px`
      last = -1e9
    }
    const ro = new ResizeObserver(resize)
    ro.observe(box)
    resize()

    const render = () => {
      const { spec: sp, getBuf: gb, win: span } = live.current
      const buf = gb()
      const k = sp.k ?? 1
      const lines = sp.lines
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, H)
      const pw = W - L - R
      const ph = H - T - B
      if (pw < 20 || ph < 20) return
      const sim = simStore.get()
      const speed = sim.fleet?.sim_speed > 0 ? sim.fleet.sim_speed : 1
      const simSpan = span * speed
      const t0 = simTimeNow(sim, nowSeconds()) - simSpan
      const gap = GAP_S * speed
      const cap = TELEMETRY_CAP
      const start = buf ? oldestIndex(buf) : 0
      const count = buf ? buf.count : 0

      // Y range over what is on screen
      let lo = Infinity
      let hi = -Infinity
      for (let q = 0; q < count; q++) {
        const i = (start + q) % cap
        if (buf.t[i] < t0 - speed) continue
        for (let l = 0; l < lines.length; l++) {
          const v = buf.series[lines[l].s][i] * k
          if (v === v) {
            if (v < lo) lo = v
            if (v > hi) hi = v
          }
        }
      }
      const hasData = lo <= hi
      if (!hasData) {
        lo = sp.min ?? 0
        hi = sp.max ?? lo + (sp.minSpan ?? 1)
      }
      if (sp.symmetric) {
        const m = Math.max(Math.abs(lo), Math.abs(hi), (sp.minSpan ?? 2) / 2)
        lo = -m
        hi = m
      } else {
        if (sp.min != null) lo = Math.min(sp.min, lo)
        if (sp.max != null) hi = sp.max
        else hi = Math.max(hi, lo + (sp.minSpan ?? 1))
      }
      const refs = sp.refs
      if (refs && sp.max == null && !sp.symmetric) {
        for (let n = 0; n < refs.length; n++) if (Number.isFinite(refs[n].v) && refs[n].v > hi * 0.98) hi = refs[n].v * 1.12
      }
      const step = niceStep(hi - lo)
      hi = Math.ceil(hi / step - 1e-9) * step
      lo = Math.floor(lo / step + 1e-9) * step
      if (hi - lo < step) hi = lo + step
      const yOf = (v) => T + ph - ((v - lo) / (hi - lo)) * ph
      const xOf = (t) => L + ((t - t0) / simSpan) * pw

      // Grid and labels
      ctx.font = '500 11px "Archivo Variable", system-ui, sans-serif'
      ctx.lineWidth = 1
      ctx.textBaseline = 'middle'
      ctx.textAlign = 'right'
      for (let v = lo; v <= hi + step * 0.01; v += step) {
        const y = Math.round(yOf(v)) + 0.5
        ctx.strokeStyle = Math.abs(v) < step * 0.01 && lo < 0 ? 'rgba(170, 183, 199, 0.32)' : 'rgba(120, 150, 190, 0.12)'
        ctx.beginPath()
        ctx.moveTo(L, y)
        ctx.lineTo(L + pw, y)
        ctx.stroke()
        ctx.fillStyle = 'rgba(123, 138, 160, 0.95)'
        ctx.fillText(fmtTick(v, step), L - 6, y)
      }
      const gx = span <= 60 ? 15 : span <= 120 ? 30 : 60
      ctx.textAlign = 'center'
      ctx.textBaseline = 'alphabetic'
      for (let s = 0; s <= span + 0.01; s += gx) {
        const x = Math.round(L + pw - (s / span) * pw) + 0.5
        ctx.strokeStyle = 'rgba(120, 150, 190, 0.08)'
        ctx.beginPath()
        ctx.moveTo(x, T)
        ctx.lineTo(x, T + ph)
        ctx.stroke()
        ctx.fillStyle = 'rgba(123, 138, 160, 0.95)'
        ctx.textAlign = s === 0 ? 'right' : s >= span - 0.01 ? 'left' : 'center'
        ctx.fillText(s === 0 ? 'now' : fmtAgo(s), x, H - 4)
      }

      ctx.save()
      ctx.beginPath()
      ctx.rect(L, T - 2, pw, ph + 4)
      ctx.clip()

      // Reference lines (budgets, thresholds)
      if (refs) {
        for (let n = 0; n < refs.length; n++) {
          const r = refs[n]
          if (!Number.isFinite(r.v) || r.v < lo || r.v > hi) continue
          const y = Math.round(yOf(r.v)) + 0.5
          ctx.setLineDash([5, 4])
          ctx.strokeStyle = r.color || 'rgba(232, 238, 245, 0.7)'
          ctx.lineWidth = 1.2
          ctx.beginPath()
          ctx.moveTo(L, y)
          ctx.lineTo(L + pw, y)
          ctx.stroke()
          ctx.setLineDash([])
          if (r.label) {
            ctx.font = '600 11px "Archivo Variable", system-ui, sans-serif'
            ctx.textAlign = 'left'
            ctx.textBaseline = 'bottom'
            const tw = ctx.measureText(r.label).width
            const lx = L + 6
            const ly = y - 2 < T + 12 ? y + 14 : y - 2
            ctx.fillStyle = 'rgba(9, 13, 20, 0.85)'
            ctx.fillRect(lx - 3, ly - 12, tw + 6, 13)
            ctx.fillStyle = r.color || 'rgba(232, 238, 245, 0.85)'
            ctx.fillText(r.label, lx, ly)
          }
        }
      }

      // Series: optional area fill first, then the lines on top
      const base = yOf(Math.max(lo, Math.min(hi, 0)))
      for (let l = 0; l < lines.length; l++) {
        const ln = lines[l]
        const arr = buf ? buf.series[ln.s] : null
        if (!arr) continue
        for (let pass = ln.fill ? 0 : 1; pass < 2; pass++) {
          ctx.beginPath()
          let pen = false
          let x0 = 0
          let px = 0
          let pt = -1e12
          for (let q = 0; q < count; q++) {
            const i = (start + q) % cap
            const w = buf.t[i]
            if (w < t0 - speed) continue
            const v = arr[i] * k
            const broken = v !== v || w - pt > gap
            pt = w
            if (broken && pen) {
              if (pass === 0) {
                ctx.lineTo(px, base)
                ctx.lineTo(x0, base)
                ctx.closePath()
              }
              pen = false
            }
            if (v !== v) continue
            const x = xOf(w)
            const y = yOf(v)
            if (!pen) {
              if (pass === 0) {
                ctx.moveTo(x, base)
                ctx.lineTo(x, y)
              } else ctx.moveTo(x, y)
              x0 = x
              pen = true
            } else ctx.lineTo(x, y)
            px = x
          }
          if (pen && pass === 0) {
            ctx.lineTo(px, base)
            ctx.lineTo(x0, base)
            ctx.closePath()
          }
          if (pass === 0) {
            ctx.globalAlpha = 0.16
            ctx.fillStyle = ln.color
            ctx.fill()
            ctx.globalAlpha = 1
          } else {
            ctx.strokeStyle = ln.color
            ctx.lineWidth = ln.width ?? 1.8
            ctx.lineJoin = 'round'
            if (ln.dash) ctx.setLineDash(ln.dash)
            ctx.stroke()
            ctx.setLineDash([])
          }
        }
      }
      ctx.restore()

      if (!hasData) {
        ctx.font = '500 12.5px "Archivo Variable", system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillStyle = 'rgba(123, 138, 160, 0.95)'
        ctx.fillText(sp.empty || 'No readings from this sim yet', L + pw / 2, T + ph / 2)
      }

      // Latest values into the legend (plain text, no React re-render)
      const lastI = buf && count ? (buf.head - 1 + cap) % cap : -1
      for (let l = 0; l < lines.length; l++) {
        const el = vals.current[l]
        if (!el) continue
        const v = lastI >= 0 ? buf.series[lines[l].s][lastI] * k : NaN
        const txt = v === v ? (sp.format ? sp.format(v) : v.toFixed(sp.digits ?? 1)) : '--'
        if (el.textContent !== txt) el.textContent = txt
      }
    }

    const loop = (t) => {
      raf = requestAnimationFrame(loop)
      if (t - last < 200) return
      last = t
      render()
    }
    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  return (
    <figure className="ch" data-focus={focus || undefined}>
      <figcaption className="ch-head">
        <span className="ch-title">{spec.title}</span>
        <span className="ch-keys">
          {spec.lines.map((ln, i) => (
            <span key={ln.s} className="ch-key">
              <i style={{ background: ln.color }} data-dash={ln.dash ? true : undefined} aria-hidden="true" />
              {ln.label && <span className="ch-key-label">{ln.label}</span>}
              <b
                ref={(el) => {
                  vals.current[i] = el
                }}
              >
                --
              </b>
            </span>
          ))}
          <span className="ch-unit">{spec.unit}</span>
        </span>
      </figcaption>
      <div className="ch-plot" ref={wrap}>
        <canvas ref={canvas} aria-hidden="true" />
      </div>
    </figure>
  )
}
