import * as THREE from 'three'

// Procedural, tileable textures for the realistic ground and models, generated once on
// the CPU (deterministic, no image files). All data textures are linear (NoColorSpace)
// and repeat, with mipmaps and anisotropy so they hold up at grazing angles.

export function mulberry(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Periodic 2D gradient noise: period P lattice cells, so it tiles.
function gradientNoise(P, seed) {
  const rand = mulberry(seed)
  const gx = new Float32Array(P * P)
  const gy = new Float32Array(P * P)
  for (let k = 0; k < P * P; k++) {
    const a = rand() * Math.PI * 2
    gx[k] = Math.cos(a)
    gy[k] = Math.sin(a)
  }
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10)
  return (x, y) => {
    const xi = Math.floor(x)
    const yi = Math.floor(y)
    const xf = x - xi
    const yf = y - yi
    const i0 = ((xi % P) + P) % P
    const j0 = ((yi % P) + P) % P
    const i1 = (i0 + 1) % P
    const j1 = (j0 + 1) % P
    const a = j0 * P + i0
    const b = j0 * P + i1
    const c = j1 * P + i0
    const d = j1 * P + i1
    const va = gx[a] * xf + gy[a] * yf
    const vb = gx[b] * (xf - 1) + gy[b] * yf
    const vc = gx[c] * xf + gy[c] * (yf - 1)
    const vd = gx[d] * (xf - 1) + gy[d] * (yf - 1)
    const u = fade(xf)
    const v = fade(yf)
    return (va + (vb - va) * u + (vc - va) * v + (va - vb - vc + vd) * u * v) * 1.4
  }
}

// Tileable fractal noise in [-1, 1] over the unit square (u, v in [0, 1)).
function fbm(base, octaves, seed, gain = 0.5) {
  const layers = Array.from({ length: octaves }, (_, o) => gradientNoise(base << o, seed + o * 101))
  let norm = 0
  for (let o = 0; o < octaves; o++) norm += Math.pow(gain, o)
  return (u, v) => {
    let s = 0
    let amp = 1
    for (let o = 0; o < octaves; o++) {
      const f = base << o
      s += layers[o](u * f, v * f) * amp
      amp *= gain
    }
    return s / norm
  }
}

// Tileable cellular noise: per cell one feature point. Returns F1, F2 (in cell units) and
// a random value of the nearest cell.
function cellular(C, seed) {
  const rand = mulberry(seed)
  const px = new Float32Array(C * C)
  const py = new Float32Array(C * C)
  const pv = new Float32Array(C * C)
  for (let k = 0; k < C * C; k++) {
    px[k] = 0.15 + rand() * 0.7
    py[k] = 0.15 + rand() * 0.7
    pv[k] = rand()
  }
  const out = { f1: 0, f2: 0, v: 0 }
  return (u, v) => {
    const x = u * C
    const y = v * C
    const xi = Math.floor(x)
    const yi = Math.floor(y)
    let f1 = 9
    let f2 = 9
    let val = 0
    for (let dj = -1; dj <= 1; dj++) {
      for (let di = -1; di <= 1; di++) {
        const ci = xi + di
        const cj = yi + dj
        const k = (((cj % C) + C) % C) * C + (((ci % C) + C) % C)
        const dx = ci + px[k] - x
        const dy = cj + py[k] - y
        const d = Math.sqrt(dx * dx + dy * dy)
        if (d < f1) {
          f2 = f1
          f1 = d
          val = pv[k]
        } else if (d < f2) f2 = d
      }
    }
    out.f1 = f1
    out.f2 = f2
    out.v = val
    return out
  }
}

const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

function dataTexture(data, size) {
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  tex.magFilter = THREE.LinearFilter
  tex.minFilter = THREE.LinearMipmapLinearFilter
  tex.generateMipmaps = true
  tex.anisotropy = 4
  tex.colorSpace = THREE.NoColorSpace
  tex.needsUpdate = true
  return tex
}

let regolith = null

// The ground's detail textures (256 px, tileable):
//   detail: R fine albedo grain, G pebbles (1 on a stone), B crack network, A soft blotches
//   normal: RG tangent normal (xy, 0.5 = flat), B cavity (0 deep, 1 exposed), A height
export function regolithTextures() {
  if (regolith) return regolith
  const S = 256
  const grain = fbm(8, 5, 11, 0.55)
  const blotch = fbm(3, 4, 23, 0.5)
  const pebbles = cellular(22, 37)
  const stones = cellular(7, 41)
  const cracks = cellular(5, 53)
  const height = new Float32Array(S * S)
  const det = new Uint8Array(S * S * 4)
  for (let j = 0; j < S; j++) {
    for (let i = 0; i < S; i++) {
      const u = i / S
      const v = j / S
      const g = grain(u, v)
      const b = blotch(u, v)
      const p = pebbles(u, v)
      // Small pebbles: present in about 60 % of cells, round, size varies per cell.
      const pr = 0.18 + p.v * 0.22
      const peb = p.v > 0.4 ? smooth(pr, pr * 0.55, p.f1) : 0
      const pebV = p.v
      // Larger stones, sparser.
      const s = stones(u, v)
      const sr = 0.16 + s.v * 0.2
      const stone = s.v > 0.62 ? smooth(sr, sr * 0.6, s.f1) : 0
      // Polygonal crack network (dried crust), wobbled by the grain.
      const c = cracks(u + g * 0.01, v + g * 0.01)
      const crack = 1 - smooth(0.0, 0.06, c.f2 - c.f1)
      const k = j * S + i
      height[k] = g * 0.35 + peb * (0.45 + 0.3 * pebV) + stone * 0.8 - crack * 0.16
      det[k * 4] = Math.round((0.5 + 0.5 * g) * 255)
      det[k * 4 + 1] = Math.round(Math.min(1, peb * (0.6 + 0.4 * pebV) + stone) * 255)
      det[k * 4 + 2] = Math.round(crack * 255)
      det[k * 4 + 3] = Math.round((0.5 + 0.5 * b) * 255)
    }
  }
  const nor = new Uint8Array(S * S * 4)
  const at = (i, j) => height[(((j % S) + S) % S) * S + (((i % S) + S) % S)]
  const strength = 2.2
  for (let j = 0; j < S; j++) {
    for (let i = 0; i < S; i++) {
      const dx = (at(i + 1, j) - at(i - 1, j)) * strength
      const dy = (at(i, j + 1) - at(i, j - 1)) * strength
      const len = Math.hypot(dx, dy, 1)
      // Cavity: darker only in the cracks (a ring round each pebble would read as a crater).
      const cav = 1 - (det[(j * S + i) * 4 + 2] / 255) * 0.55
      const k = (j * S + i) * 4
      nor[k] = Math.round((0.5 - (0.5 * dx) / len) * 255)
      nor[k + 1] = Math.round((0.5 - (0.5 * dy) / len) * 255)
      nor[k + 2] = Math.round(cav * 255)
      nor[k + 3] = Math.round(Math.min(1, Math.max(0, 0.5 + height[j * S + i] * 0.4)) * 255)
    }
  }
  regolith = { detail: dataTexture(det, S), normal: dataTexture(nor, S) }
  return regolith
}

// ---------------------------------------------------------------- canvas textures

function canvasTexture(size, draw, { srgb = true, repeat = false } = {}) {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')
  draw(ctx, size)
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace
  tex.anisotropy = 8
  if (repeat) tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  return tex
}

const cache = {}

// Painted hull panels for rovers and the lander (tiles every 0.4 m): seams, a few rivets,
// small vents and a hint of grime. Multiplied with the material colour.
export function panelTexture() {
  if (cache.panel) return cache.panel
  cache.panel = canvasTexture(
    256,
    (ctx, S) => {
      const rand = mulberry(7)
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, S, S)
      // Subtle mottling.
      for (let k = 0; k < 900; k++) {
        const v = 235 + Math.floor(rand() * 20)
        ctx.fillStyle = `rgba(${v},${v},${v - 4},0.35)`
        ctx.fillRect(rand() * S, rand() * S, 2 + rand() * 6, 2 + rand() * 6)
      }
      // Panel seams (a cross at the tile edges and one off-centre split).
      ctx.strokeStyle = 'rgba(70,70,74,0.85)'
      ctx.lineWidth = 3
      ctx.strokeRect(1.5, 1.5, S - 3, S - 3)
      ctx.beginPath()
      ctx.moveTo(S * 0.62, 0)
      ctx.lineTo(S * 0.62, S)
      ctx.stroke()
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'
      ctx.lineWidth = 1
      ctx.strokeRect(5, 5, S - 10, S - 10)
      // Rivets along the seams.
      ctx.fillStyle = 'rgba(90,90,95,0.9)'
      for (let k = 0; k < 8; k++) {
        const t = 14 + (k * (S - 28)) / 7
        for (const [x, y] of [
          [9, t],
          [S - 9, t],
          [t, 9],
          [t, S - 9],
        ]) {
          ctx.beginPath()
          ctx.arc(x, y, 1.8, 0, Math.PI * 2)
          ctx.fill()
        }
      }
      // A small vent grille.
      ctx.fillStyle = 'rgba(60,60,64,0.8)'
      for (let k = 0; k < 6; k++) ctx.fillRect(S * 0.72, S * 0.18 + k * 9, S * 0.2, 4)
      // An access hatch.
      ctx.strokeStyle = 'rgba(80,80,84,0.7)'
      ctx.lineWidth = 2
      ctx.strokeRect(S * 0.14, S * 0.5, S * 0.34, S * 0.3)
    },
    { repeat: true },
  )
  return cache.panel
}

// Solar cells: dark blue-black cells with thin silver bus lines (tiles every cell block).
export function solarTexture() {
  if (cache.solar) return cache.solar
  cache.solar = canvasTexture(
    256,
    (ctx, S) => {
      ctx.fillStyle = '#c9ccd2'
      ctx.fillRect(0, 0, S, S)
      const n = 4
      const gap = 4
      const cell = (S - gap * (n + 1)) / n
      const rand = mulberry(3)
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const x = gap + i * (cell + gap)
          const y = gap + j * (cell + gap)
          const v = 18 + Math.floor(rand() * 10)
          const g = ctx.createLinearGradient(x, y, x + cell, y + cell)
          g.addColorStop(0, `rgb(${v},${v + 8},${v + 34})`)
          g.addColorStop(1, `rgb(${v + 6},${v + 14},${v + 48})`)
          ctx.fillStyle = g
          ctx.fillRect(x, y, cell, cell)
          ctx.fillStyle = 'rgba(190,200,215,0.55)'
          for (let b = 1; b < 4; b++) ctx.fillRect(x + (cell * b) / 4 - 0.5, y, 1, cell)
        }
      }
    },
    { repeat: true },
  )
  return cache.solar
}

// Charging pad markings for the lander (one texture over the whole disc, alpha outside).
export function padTexture() {
  if (cache.pad) return cache.pad
  cache.pad = canvasTexture(1024, (ctx, S) => {
    const c = S / 2
    const R = S / 2 - 4
    const rand = mulberry(19)
    ctx.clearRect(0, 0, S, S)
    // The mat: dark grey composite, dusty towards the rim.
    const g = ctx.createRadialGradient(c, c, R * 0.2, c, c, R)
    g.addColorStop(0, '#5d5a57')
    g.addColorStop(0.8, '#55504b')
    g.addColorStop(1, '#6e6156')
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.arc(c, c, R, 0, Math.PI * 2)
    ctx.fill()
    // Dust speckle.
    for (let k = 0; k < 5000; k++) {
      const a = rand() * Math.PI * 2
      const r = Math.sqrt(rand()) * R
      const d = rand()
      ctx.fillStyle = d > 0.5 ? 'rgba(150,120,95,0.25)' : 'rgba(40,38,36,0.25)'
      ctx.fillRect(c + Math.cos(a) * r, c + Math.sin(a) * r, 2 + rand() * 4, 2 + rand() * 4)
    }
    // Tile seams.
    ctx.save()
    ctx.beginPath()
    ctx.arc(c, c, R, 0, Math.PI * 2)
    ctx.clip()
    ctx.strokeStyle = 'rgba(25,25,25,0.35)'
    ctx.lineWidth = 2
    for (let x = c % 128; x < S; x += 128) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, S)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, x)
      ctx.lineTo(S, x)
      ctx.stroke()
    }
    ctx.restore()
    // Outer hazard ring: amber and dark segments.
    const seg = 48
    for (let k = 0; k < seg; k++) {
      ctx.beginPath()
      ctx.arc(c, c, R * 0.965, (k / seg) * Math.PI * 2, ((k + 0.5) / seg) * Math.PI * 2)
      ctx.strokeStyle = k % 2 ? 'rgba(232,170,62,0.95)' : 'rgba(30,28,26,0.9)'
      ctx.lineWidth = R * 0.05
      ctx.stroke()
    }
    // Dock circle (dashed), centre ring and cross hairs.
    ctx.setLineDash([R * 0.06, R * 0.04])
    ctx.strokeStyle = 'rgba(235,232,225,0.8)'
    ctx.lineWidth = R * 0.012
    ctx.beginPath()
    ctx.arc(c, c, R * 0.72, 0, Math.PI * 2)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.lineWidth = R * 0.018
    ctx.beginPath()
    ctx.arc(c, c, R * 0.42, 0, Math.PI * 2)
    ctx.stroke()
    for (let k = 0; k < 12; k++) {
      const a = (k / 12) * Math.PI * 2
      const r0 = k % 3 === 0 ? R * 0.46 : R * 0.84
      ctx.beginPath()
      ctx.moveTo(c + Math.cos(a) * r0, c + Math.sin(a) * r0)
      ctx.lineTo(c + Math.cos(a) * R * 0.9, c + Math.sin(a) * R * 0.9)
      ctx.stroke()
    }
    // Charging coils: faint rounded squares around the dock circle.
    ctx.strokeStyle = 'rgba(120,200,215,0.45)'
    ctx.lineWidth = R * 0.008
    for (let k = 0; k < 6; k++) {
      const a = ((k + 0.5) / 6) * Math.PI * 2
      ctx.save()
      ctx.translate(c + Math.cos(a) * R * 0.72, c + Math.sin(a) * R * 0.72)
      ctx.rotate(a)
      ctx.strokeRect(-R * 0.07, -R * 0.07, R * 0.14, R * 0.14)
      ctx.restore()
    }
  })
  return cache.pad
}

// Organic trace film: mottled magenta-brown microbial mat, transparent at its edge.
export function filmTexture() {
  if (cache.film) return cache.film
  cache.film = canvasTexture(256, (ctx, S) => {
    const c = S / 2
    const rand = mulberry(29)
    ctx.clearRect(0, 0, S, S)
    for (let k = 0; k < 700; k++) {
      const a = rand() * Math.PI * 2
      const r = Math.pow(rand(), 0.7) * S * 0.46
      const x = c + Math.cos(a) * r
      const y = c + Math.sin(a) * r
      const big = rand() < 0.12
      const rad = big ? 6 + rand() * 12 : 2 + rand() * 5
      const t = rand()
      const fall = 1 - r / (S * 0.48)
      const col = t < 0.5 ? `rgba(196,92,140,${0.35 * fall + 0.1})` : t < 0.8 ? `rgba(122,58,84,${0.4 * fall + 0.1})` : `rgba(232,170,120,${0.3 * fall})`
      ctx.fillStyle = col
      ctx.beginPath()
      ctx.ellipse(x, y, rad, rad * (0.6 + rand() * 0.4), rand() * 3, 0, Math.PI * 2)
      ctx.fill()
    }
  })
  return cache.film
}

// Sulphur crust around a vent: yellow and orange deposits fading out, alpha outside.
export function crustTexture() {
  if (cache.crust) return cache.crust
  cache.crust = canvasTexture(256, (ctx, S) => {
    const c = S / 2
    const rand = mulberry(31)
    ctx.clearRect(0, 0, S, S)
    for (let k = 0; k < 900; k++) {
      const a = rand() * Math.PI * 2
      const r = (0.12 + Math.pow(rand(), 0.9) * 0.36) * S
      const x = c + Math.cos(a) * r
      const y = c + Math.sin(a) * r
      const fall = 1 - Math.min(1, r / (S * 0.48))
      const t = rand()
      ctx.fillStyle = t < 0.45 ? `rgba(226,190,70,${0.5 * fall})` : t < 0.8 ? `rgba(214,120,48,${0.45 * fall})` : `rgba(250,236,190,${0.45 * fall})`
      ctx.beginPath()
      ctx.arc(x, y, 2 + rand() * 7 * fall, 0, Math.PI * 2)
      ctx.fill()
    }
  })
  return cache.crust
}

// Ice patch: bluish-white, glassy, with darker cracks and frost, soft at the edge.
export function iceTexture() {
  if (cache.ice) return cache.ice
  cache.ice = canvasTexture(512, (ctx, S) => {
    const c = S / 2
    const rand = mulberry(43)
    ctx.clearRect(0, 0, S, S)
    const g = ctx.createRadialGradient(c, c, S * 0.05, c, c, S * 0.5)
    g.addColorStop(0, 'rgba(214,232,246,1)')
    g.addColorStop(0.62, 'rgba(196,220,240,0.97)')
    g.addColorStop(0.82, 'rgba(186,206,222,0.75)')
    g.addColorStop(1, 'rgba(170,180,190,0)')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, S, S)
    // Frost and dust speckle.
    for (let k = 0; k < 2600; k++) {
      const a = rand() * Math.PI * 2
      const r = Math.sqrt(rand()) * S * 0.46
      const t = rand()
      ctx.fillStyle = t < 0.6 ? 'rgba(255,255,255,0.35)' : t < 0.85 ? 'rgba(150,175,200,0.3)' : 'rgba(150,120,95,0.28)'
      ctx.fillRect(c + Math.cos(a) * r, c + Math.sin(a) * r, 1 + rand() * 3, 1 + rand() * 3)
    }
    // Pressure cracks.
    ctx.strokeStyle = 'rgba(80,120,160,0.55)'
    ctx.lineCap = 'round'
    for (let k = 0; k < 14; k++) {
      let x = c + (rand() - 0.5) * S * 0.6
      let y = c + (rand() - 0.5) * S * 0.6
      let a = rand() * Math.PI * 2
      ctx.lineWidth = 1 + rand() * 2
      ctx.beginPath()
      ctx.moveTo(x, y)
      for (let q = 0; q < 8; q++) {
        a += (rand() - 0.5) * 1.1
        x += Math.cos(a) * S * 0.035
        y += Math.sin(a) * S * 0.035
        ctx.lineTo(x, y)
      }
      ctx.stroke()
    }
  })
  return cache.ice
}

let flat = null
// A 1x1 black texture (no shoreline yet).
export function blackTexture() {
  if (flat) return flat
  flat = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1, THREE.RGBAFormat)
  flat.needsUpdate = true
  return flat
}
