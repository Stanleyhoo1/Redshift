// Screenshots of the 3D look (realistic terrain, rovers, lander, spires, water, life,
// targets, overlays, dust storm, graphics quality) for before/after comparisons.
// Needs the mock started with MOCK_FIRST_QUESTION_S=99999 and a preview of the build.
// Usage: node scripts/visual-shots.mjs [url] [outDir] [prefix]   Env: CHROME, WIDTH, HEIGHT, ONLY
// Uses the mock-only /api/_mock/teleport hook and the page's ?debug hook (window.__redshift).
import { chromium } from 'playwright-core'
import WebSocket from 'ws'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:4174'
const outDir = process.argv[3] || 'screenshots/visual'
const prefix = process.argv[4] || 'after'
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
const only = process.env.ONLY ? new Set(process.env.ONLY.split(',')) : null
const tag = `${width}x${height}`
const candidates = [process.env.CHROME, '/opt/pw-browsers/chromium-1243/chrome-linux64/chrome', '/usr/bin/chromium'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))
fs.mkdirSync(outDir, { recursive: true })

const browser = await chromium.launch({
  executablePath,
  args: ['--use-gl=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'],
})
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
page.setDefaultTimeout(120000) // a software-rendered page answers slowly
const problems = []
page.on('console', (msg) => {
  if (msg.type() !== 'error' && msg.type() !== 'warning') return
  const t = msg.text()
  if (/status of 503|status of 404|status of 400|GPU stall due to ReadPixels|THREE\.Clock: This module has been deprecated|camera/i.test(t)) return
  problems.push(`[console.${msg.type()}] ${t}`)
})
page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`))

const t0 = Date.now()
const want = (name) => !only || only.has(name)
// The sim's latest time, from our own socket: a software-rendered page can fall seconds
// behind the message stream, so each shot first waits for the page to catch up.
let simT = 0
const sock = new WebSocket(url.replace(/^http/, 'ws') + '/ws')
sock.on('message', (d) => {
  try {
    const m = JSON.parse(d)
    if (Number.isFinite(m.t)) simT = m.t
  } catch {}
})
sock.on('error', () => {})
const catchUp = async (limit = 20000) => {
  const target = simT - 0.6
  const until = Date.now() + limit
  while (Date.now() < until) {
    const t = await page.evaluate(() => window.__redshift.state().t).catch(() => 0)
    if (t >= target) return
    await page.waitForTimeout(250)
  }
}
const shot = async (name, sync = true) => {
  if (sync) await catchUp()
  await page.waitForTimeout(600)
  const file = path.join(outDir, `${prefix}-${tag}-${name}.png`)
  await page.screenshot({ path: file, timeout: 120000 })
  console.log(`saved ${file} at ${((Date.now() - t0) / 1000).toFixed(1)} s`)
}
const wait = (ms) => page.waitForTimeout(ms)
const origin = new URL(url).origin
const api = async (p, body) => {
  const r = await fetch(`${origin}${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
  return { status: r.status, ...(await r.json().catch(() => ({}))) }
}
const st = () => page.evaluate(() => window.__redshift.state())
const view = (x, y, az, el, dist) => page.evaluate((a) => window.__redshift.view(...a), [x, y, az, el, dist])
const orbit = (az, el, dist) => page.evaluate((a) => window.__redshift.orbit(...a), [az, el, dist])
const teleport = (rover, x, y, hold = true, yaw) => api('/api/_mock/teleport', { rover, x, y, hold, ...(yaw !== undefined && { yaw }) })
const follow = async (id) => {
  const n = Number(String(id).replace(/\D/g, ''))
  await page.locator('.hud-controls .seg button').nth(n).click()
  await wait(300)
  await page.keyboard.press('Escape')
}
const legendSlopeOn = async () => (await page.locator('.hud-legend .legend-title').first().textContent().catch(() => '')).startsWith('Ground slope,')
const setSlope = async (on) => {
  if ((await legendSlopeOn()) !== on) await page.keyboard.press('s')
}
const inPoly = (x, y, poly) => {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

// ---------------------------------------------------------------- setup
await api('/api/reset', { seed: 42 })
await api('/api/_mock/rearm', { armed: false })
await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift && window.__redshift.state().rovers.r1, null, { timeout: 30000 })
await page.evaluate(() => document.activeElement?.blur?.())
const truth = await (await fetch(`${origin}/api/truth`)).json()
let s = await st()
const water = s.features.water
const wet = (x, y, pad = 1.5) => water.some((w) => inPoly(x, y, w.outline) || w.outline.some(([px, py]) => Math.hypot(px - x, py - y) < pad))

// Reveal a large irregular region (same points every run): sweep r1 over a grid, skipping
// a few cells so the edge of the explored area stays ragged.
const pts = []
for (let gy = -48; gy <= 48; gy += 12) {
  for (let gx = -48; gx <= 48; gx += 12) {
    const h = Math.abs(Math.sin(gx * 12.9898 + gy * 78.233) * 43758.5453) % 1
    const r = Math.hypot(gx, gy)
    if (r > 50 || (r > 30 && h < 0.4) || wet(gx, gy)) continue
    pts.push([gx + (h - 0.5) * 4, gy + ((h * 7) % 1 - 0.5) * 4])
  }
}
for (const [x, y] of pts) await teleport('r1', x, y, true)
console.log(`revealed around ${pts.length} points`)
await wait(1500)
await catchUp()

// Stations: r1 drives (for dust and wheels), r2 by the lake, r3 near spires.
s = await st()
const lake = water.filter((w) => w.kind === 'lake').map((w) => {
  const cx = w.outline.reduce((t, p) => t + p[0], 0) / w.outline.length
  const cy = w.outline.reduce((t, p) => t + p[1], 0) / w.outline.length
  return { ...w, cx, cy }
}).sort((a, b) => Math.hypot(a.cx, a.cy) - Math.hypot(b.cx, b.cy))[0]
const spires = s.features.spires.filter((p) => p.standing)
const nearSpire = spires.slice().sort((a, b) => Math.hypot(a.x - lake.cx, a.y - lake.cy) - Math.hypot(b.x - lake.cx, b.y - lake.cy))[0]
// Identify the lake: walk r2 round its shore 3 m off the water.
const shore = (ang, off) => {
  let d = 0
  while (inPoly(lake.cx + Math.cos(ang) * d, lake.cy + Math.sin(ang) * d, lake.outline) && d < 60) d += 0.25
  return [lake.cx + Math.cos(ang) * (d + off), lake.cy + Math.sin(ang) * (d + off)]
}
for (let k = 0; k < 10; k++) {
  const [x, y] = shore((k / 10) * Math.PI * 2, 3)
  await teleport('r2', x, y, true)
  await wait(150)
}
// r3 beside the nearest spire grove; reveal it.
await teleport('r3', nearSpire.x - 4, nearSpire.y - 4, true)
await wait(1200)
await catchUp()
console.log(`lake ${lake.id} at (${lake.cx.toFixed(1)}, ${lake.cy.toFixed(1)}), spire ${nearSpire.id} at (${nearSpire.x}, ${nearSpire.y})`)

// ---------------------------------------------------------------- shots
const hasGraphics = (await page.locator('.gfx-seg').count()) > 0
if (hasGraphics) {
  // Always start from High.
  const hi = page.locator('.gfx-seg button', { hasText: 'High' })
  if ((await hi.getAttribute('aria-pressed')) !== 'true') await hi.click()
}
await page.keyboard.press('0')
await wait(3500)
if (want('overview')) await shot('01-overview')

if (want('slope')) {
  const was = await legendSlopeOn()
  await setSlope(true)
  await wait(1500)
  await shot('02-slope-on')
  await setSlope(was)
}

// Rover close-up: r1 driving on open ground, camera low behind and to the side.
if (want('rover')) {
  await api('/api/_mock/battery', { rover: 'r1', battery: 0.95 })
  await teleport('r1', 6, -14, false, 0.6)
  await api('/api/chat', { text: 'resume', rover: 'r1' })
  await follow('r1')
  await wait(2500)
  await orbit(-35, 24, 7.5)
  // Wait until it is driving (wheels turning, dust behind it).
  for (let k = 0; k < 60; k++) {
    if (((await st()).rovers.r1?.speed ?? 0) > 0.15) break
    await wait(500)
  }
  await wait(2500)
  await shot('03-rover-closeup')
  await orbit(150, 38, 11)
  await wait(1800)
  await shot('04-rover-closeup-2')
  await page.keyboard.press('0')
  await wait(800)
}

// Low, towards the sun: the sky, the sun's glow and the haze over the horizon.
if (want('horizon')) {
  s = await st()
  await view(s.lander.x + 6, s.lander.y + 4, 64, 10, 16)
  await wait(2500)
  await shot('15-horizon')
}

if (want('lander')) {
  s = await st()
  await view(s.lander.x, s.lander.y, 28, 34, 14)
  await wait(2200)
  await shot('05-lander')
}

if (want('spires')) {
  // Camera on the spire side, the lake beyond it.
  const mx = nearSpire.x * 0.6 + lake.cx * 0.4
  const my = nearSpire.y * 0.6 + lake.cy * 0.4
  const span = Math.hypot(lake.cx - nearSpire.x, lake.cy - nearSpire.y)
  const az = (Math.atan2(nearSpire.x - lake.cx, -(nearSpire.y - lake.cy)) * 180) / Math.PI
  await view(mx, my, az, 30, Math.max(24, span * 1.1 + 8))
  await wait(2500)
  await shot('06-spires-lake')
  await view(nearSpire.x, nearSpire.y, az + 20, 20, 11)
  await wait(2000)
  await shot('07-spires-close')
}

if (want('grazers')) {
  s = await st()
  const r3 = s.rovers.r3
  let placed = null
  for (const [dx, dy] of [[8, 3], [-8, 3], [3, 8], [3, -8], [8, -6]]) {
    const res = await api('/api/hazard', { type: 'lifeform', x: r3.x + dx, y: r3.y + dy })
    if (res.status === 200) {
      placed = [r3.x + dx, r3.y + dy]
      break
    }
  }
  console.log(`herd ${placed ? 'placed' : 'not placed'}`)
  let track = null
  for (let k = 0; k < 30 && !track; k++) {
    await wait(500)
    s = await st()
    track = s.tracks.find((t) => t.status === 'tracking')
  }
  if (track || placed) {
    await catchUp()
    s = await st()
    // Only creatures near a live fleet track are drawn: frame the creature closest to one.
    const live = s.tracks.filter((t) => t.status === 'tracking')
    const near = (l) => Math.min(...live.map((t) => Math.hypot(t.x - l.x, t.y - l.y)), 1e9)
    const lf = s.lifeforms.slice().sort((a, b) => near(a) - near(b))[0]
    const tx = lf && near(lf) < 3 ? lf.x : live[0] ? live[0].x : placed[0]
    const ty = lf && near(lf) < 3 ? lf.y : live[0] ? live[0].y : placed[1]
    await view(tx, ty, 30, 28, 9)
    await wait(2200)
    await shot('08-grazers')
  }
}

if (want('targets')) {
  for (const kind of ['vent', 'mineral', 'ice']) {
    const t = truth.targets.filter((q) => q.kind === kind && !wet(q.x, q.y, 2)).sort((a, b) => Math.hypot(a.x, a.y) - Math.hypot(b.x, b.y))[0]
    if (!t) continue
    let ok = false
    for (const [dx, dy] of [[2.5, -2], [-2.5, -2], [2.5, 2], [-2.5, 2], [0, -3]]) {
      if ((await teleport('r1', t.x + dx, t.y + dy, true)).status === 200) {
        ok = true
        break
      }
    }
    await wait(1600)
    await catchUp()
    console.log(`${kind} ${t.id} at (${t.x.toFixed(1)}, ${t.y.toFixed(1)}) ${ok ? '' : '(no teleport)'}`)
    await view(t.x, t.y, 20, 30, 8)
    await wait(2200)
    await shot(`09-target-${kind}`)
  }
}

if (want('reality')) {
  await page.keyboard.press('0')
  await page.keyboard.press('r')
  await wait(3500)
  await shot('10-reality')
  await page.keyboard.press('r')
  await wait(500)
}

// A storm lasts 30 wall seconds and a software-rendered page lags, so start one for each
// shot and wait until the page has it.
const storm = async () => {
  const dusty = () => page.evaluate(() => window.__redshift.state().hazards.some((h) => h.type === 'dust_storm' && h.active))
  for (let k = 0; k < 120 && (await dusty()); k++) await wait(500) // let a running one end
  await catchUp()
  await api('/api/hazard', { type: 'dust_storm' })
  for (let k = 0; k < 240 && !(await dusty()); k++) await wait(250)
}
if (want('dust')) {
  await storm()
  s = await st()
  await view(s.rovers.r3.x, s.rovers.r3.y, 20, 32, 22)
  await wait(4500)
  await shot('11-dust-storm', false)
  await storm()
  await page.keyboard.press('0')
  await wait(4500)
  await shot('12-dust-overview', false)
}

if (want('low') && hasGraphics) {
  await page.locator('.gfx-seg button', { hasText: 'Low' }).click()
  await page.keyboard.press('0')
  await wait(3500)
  await shot('13-low-overview')
  s = await st()
  await view(s.rovers.r1.x, s.rovers.r1.y, -35, 26, 9)
  await wait(2000)
  await shot('14-low-closeup')
  await page.locator('.gfx-seg button', { hasText: 'High' }).click()
}

console.log(problems.length ? `console problems:\n${problems.join('\n')}` : 'no console errors or warnings (camera lines ignored)')
sock.close()
await browser.close()
