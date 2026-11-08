import { KIND_STYLE, LIFE, SPIRE } from './theme.js'

// Version 8 object detection: what each rover's onboard detector reports for its camera
// frames (rover `vision`, camera header X-Detections). Class colours follow the legend:
// target kinds as on the map, creatures lime, spires violet, water blue, rocks neutral.

export const DET_COLORS = {
  mineral: KIND_STYLE.mineral.color,
  ice: KIND_STYLE.ice.color,
  vent: KIND_STYLE.vent.color,
  trace: KIND_STYLE.trace.color,
  creature: LIFE,
  spire: SPIRE.line,
  water: '#4b9be8',
  rock: '#c9c0b3',
  boulder: '#e3d8c6',
  rover: '#e8eef5',
  lander: '#e8eef5',
}
const DARK_TEXT = '#071019'
// Label text on the class colour: dark on light colours, white on the darker blue.
const TEXT_ON = { water: '#ffffff' }

export const detColor = (label) => DET_COLORS[label] || '#e8eef5'

export const fmtConf = (c) => (Number.isFinite(c) ? c.toFixed(2) : '--')

// "creature 0.83 at 6 m"
export function detPhrase(o) {
  const d = Number.isFinite(o.dist) ? ` at ${o.dist < 10 ? o.dist.toFixed(1).replace(/\.0$/, '') : Math.round(o.dist)} m` : ''
  return `${o.label} ${fmtConf(o.conf)}${d}`
}

// Draws the boxes onto a 2D canvas context. Normalised box coordinates map to canvas pixels
// as x = ox + nx * sw, y = oy + ny * sh (the caller folds in any zoom and pan). `px` is the
// canvas pixels per CSS pixel, so lines and type keep their on-screen size. `long` adds the
// distance to each label. Called once per camera frame (and on zoom or resize), never in a
// render loop.
export function drawDetections(ctx, dets, ox, oy, sw, sh, px, long, cw, ch) {
  if (!dets || !dets.length) return
  const lw = 1.5 * px
  const fs = (long ? 12 : 10.5) * px
  const padX = 4 * px
  const lh = fs + 4 * px
  ctx.save()
  ctx.font = `650 ${fs}px "Archivo Variable", system-ui, sans-serif`
  ctx.textBaseline = 'middle'
  ctx.lineJoin = 'round'
  // Far (small) boxes first, so the labels of near objects stay on top.
  const order = dets.length > 1 ? dets.slice().sort((a, b) => (a.box[3] - a.box[1]) * (a.box[2] - a.box[0]) - (b.box[3] - b.box[1]) * (b.box[2] - b.box[0])) : dets
  for (let k = 0; k < order.length; k++) {
    const o = order[k]
    const x0 = ox + o.box[0] * sw
    const y0 = oy + o.box[1] * sh
    const x1 = ox + o.box[2] * sw
    const y1 = oy + o.box[3] * sh
    if (x1 < 0 || y1 < 0 || x0 > cw || y0 > ch) continue
    const w = Math.max(2 * px, x1 - x0)
    const h = Math.max(2 * px, y1 - y0)
    const color = detColor(o.label)
    // A light tint inside small boxes; big ones (a spire close up) stay clear to see through.
    const share = (w * h) / (cw * ch)
    if (share < 0.2) {
      ctx.globalAlpha = 0.09
      ctx.fillStyle = color
      ctx.fillRect(x0, y0, w, h)
      ctx.globalAlpha = 1
    }
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)'
    ctx.lineWidth = lw + 2 * px
    ctx.strokeRect(x0, y0, w, h)
    ctx.strokeStyle = color
    ctx.lineWidth = lw
    ctx.strokeRect(x0, y0, w, h)
    // Corner accents
    const a = Math.min(10 * px, w / 3, h / 3)
    ctx.lineWidth = lw * 1.9
    ctx.beginPath()
    ctx.moveTo(x0, y0 + a)
    ctx.lineTo(x0, y0)
    ctx.lineTo(x0 + a, y0)
    ctx.moveTo(x1 - a, y0)
    ctx.lineTo(x1, y0)
    ctx.lineTo(x1, y0 + a)
    ctx.moveTo(x0, y1 - a)
    ctx.lineTo(x0, y1)
    ctx.lineTo(x0 + a, y1)
    ctx.moveTo(x1 - a, y1)
    ctx.lineTo(x1, y1)
    ctx.lineTo(x1, y1 - a)
    ctx.stroke()
    // Label chip above the box (inside it when there is no room above)
    let text = `${o.label} ${fmtConf(o.conf)}`
    if (long && Number.isFinite(o.dist)) text += ` · ${o.dist.toFixed(1)} m`
    const tw = ctx.measureText(text).width + padX * 2
    const lx = Math.max(0, Math.min(cw - tw, x0 - lw / 2))
    const ly = y0 - lh >= 0 ? y0 - lh : y0
    ctx.fillStyle = color
    ctx.fillRect(lx, ly, tw, lh)
    ctx.fillStyle = TEXT_ON[o.label] || DARK_TEXT
    ctx.fillText(text, lx + padX, ly + lh / 2 + 0.5 * px)
  }
  ctx.restore()
}
