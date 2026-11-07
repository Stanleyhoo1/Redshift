import * as THREE from 'three'

let glow = null

// Soft radial gradient used for additive glows (lander, frontiers, beacons, rover halos).
export function glowTexture() {
  if (glow) return glow
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  g.addColorStop(0, 'rgba(255,255,255,1)')
  g.addColorStop(0.25, 'rgba(255,255,255,0.55)')
  g.addColorStop(0.6, 'rgba(255,255,255,0.12)')
  g.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, size, size)
  glow = new THREE.CanvasTexture(canvas)
  return glow
}

let dot = null

// Small soft-edged disc for LiDAR points.
export function dotTexture() {
  if (dot) return dot
  const size = 32
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  g.addColorStop(0, 'rgba(255,255,255,1)')
  g.addColorStop(0.55, 'rgba(255,255,255,0.9)')
  g.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, size, size)
  dot = new THREE.CanvasTexture(canvas)
  return dot
}

let hatch = null

// Diagonal stripes, tiled in world units, for soft ground the fleet has marked.
export function hatchTexture() {
  if (hatch) return hatch
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = 'rgba(255,255,255,0.22)'
  ctx.fillRect(0, 0, size, size)
  ctx.strokeStyle = 'rgba(255,255,255,1)'
  ctx.lineWidth = 9
  ctx.lineCap = 'square'
  for (let k = -1; k <= 1; k++) {
    ctx.beginPath()
    ctx.moveTo(k * size, size)
    ctx.lineTo(k * size + size, 0)
    ctx.stroke()
  }
  hatch = new THREE.CanvasTexture(canvas)
  hatch.wrapS = hatch.wrapT = THREE.RepeatWrapping
  hatch.anisotropy = 4
  return hatch
}

let warning = null

// Red warning triangle with an exclamation mark, for a rover with a motor fault.
export function warningTexture() {
  if (warning) return warning
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')
  ctx.lineJoin = 'round'
  ctx.beginPath()
  ctx.moveTo(size / 2, 12)
  ctx.lineTo(size - 10, size - 16)
  ctx.lineTo(10, size - 16)
  ctx.closePath()
  ctx.fillStyle = '#ff3b5c'
  ctx.fill()
  ctx.lineWidth = 7
  ctx.strokeStyle = '#ffffff'
  ctx.stroke()
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(size / 2 - 6, 44, 12, 40)
  ctx.beginPath()
  ctx.arc(size / 2, 96, 7, 0, Math.PI * 2)
  ctx.fill()
  warning = new THREE.CanvasTexture(canvas)
  return warning
}
