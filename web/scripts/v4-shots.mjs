// Drives the version 4 and 4b features through the real UI and saves screenshots:
// point-and-click hazards (ghost marker, a rockfall placed by clicking, a refused spot),
// a herd brought in with the Lifeform button, the first-sighting question and a rover
// observing the herd, a spire toppling, a lake (LiDAR hole, then identified, then with
// Reality), sampling (3D ring, rover card, sensor panel), the Samples tab and the legend.
// Needs the mock on 8000 started with MOCK_FIRST_QUESTION_S=99999 (no unrelated question
// in the way) and MOCK_SAMPLE_SCALE=4 (sampling slow enough to catch on a software GPU),
// and `npm run preview`. Uses the mock-only /api/_mock/teleport hook and the page's ?debug
// hook (window.__redshift) to find world points on screen and to hold a spire mid-fall.
// Usage: node scripts/v4-shots.mjs [url] [outDir]   Env: CHROME, WIDTH, HEIGHT
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:4173'
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
const outDir = process.argv[3] || 'screenshots'
const tag = `${width}x${height}`
const candidates = [process.env.CHROME, '/opt/pw-browsers/chromium-1243/chrome-linux64/chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))
if (!executablePath) {
  console.error('No Chrome or Chromium found. Set CHROME=/path/to/chrome.')
  process.exit(2)
}
fs.mkdirSync(outDir, { recursive: true })

const browser = await chromium.launch({
  executablePath,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'],
})
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
const problems = []
let refused = 0
page.on('console', (msg) => {
  if (msg.type() !== 'error' && msg.type() !== 'warning') return
  const t = msg.text()
  // The browser logs every HTTP 400; step 2 provokes one on purpose (a refused spot).
  if (/status of 400/.test(t)) return void refused++
  // The mock answers camera requests with 503 on purpose; the rest are swiftshader and
  // three.js / r3f notices that also appear without this change.
  if (/status of 503|GPU stall due to ReadPixels|THREE\.Clock: This module has been deprecated/.test(t)) return
  problems.push(`[console.${msg.type()}] ${t}`)
})
page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`))

const t0 = Date.now()
const shot = async (name, clip) => {
  const file = path.join(outDir, `v4-${tag}-${name}.png`)
  await page.screenshot({ path: file, ...(clip && { clip }) })
  console.log(`saved ${file} at ${((Date.now() - t0) / 1000).toFixed(1)} s`)
}
const wait = (ms) => page.waitForTimeout(ms)
const origin = new URL(url).origin
const api = async (p, body) => {
  const r = await fetch(`${origin}${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
  return { status: r.status, ...(await r.json().catch(() => ({}))) }
}
const st = () => page.evaluate(() => window.__redshift.state())
const scr = (x, y, lift = 0) => page.evaluate(([a, b, l]) => window.__redshift.screenOf(a, b, l), [x, y, lift])
const teleport = (rover, x, y, hold = false) => api('/api/_mock/teleport', { rover, x, y, hold })
const hz = (type) => page.locator(`.hz-btn[data-type="${type}"]`)
const hover = async (x, y, lift = 0) => {
  const p = await scr(x, y, lift)
  await page.mouse.move(p.x, p.y, { steps: 5 })
  return p
}
const placeAt = async (x, y) => {
  const p = await scr(x, y)
  await page.mouse.move(p.x, p.y, { steps: 3 })
  const resp = page.waitForResponse((r) => r.url().includes('/api/hazard'), { timeout: 8000 }).catch(() => null)
  await page.mouse.click(p.x, p.y)
  const r = await resp
  if (!r) {
    console.log(`place at (${x.toFixed(1)}, ${y.toFixed(1)}): no request (the spot is under a panel?)`)
    return { status: 0 }
  }
  const j = await r.json().catch(() => ({}))
  console.log(`place at (${x.toFixed(1)}, ${y.toFixed(1)}): HTTP ${r.status()} ${j.error || j.hazard?.id || ''}`)
  return { status: r.status(), ...j }
}
// Follow a rover without its sensor panel (the follow button opens it, Esc closes it).
const follow = async (id) => {
  const n = Number(String(id).replace(/\D/g, ''))
  await page.locator('.hud-controls .seg button').nth(n).click()
  await wait(250)
  await page.keyboard.press('Escape')
}
const zoom = async (steps) => {
  const box = await page.locator('.view > div canvas, main canvas').first().boundingBox()
  await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.62)
  for (let k = 0; k < Math.abs(steps); k++) {
    await page.mouse.wheel(0, steps < 0 ? -120 : 120)
    await wait(60)
  }
}
const answerOpen = async (label) => {
  const card = page.locator('.hud-request[data-kind="open"]')
  if (!(await card.count())) return false
  await card.locator('.rq-opt', { hasText: label }).first().click().catch(() => {})
  return true
}
// The point `dist` metres along a rover's route (or straight ahead).
function ahead(r, dist) {
  let px = r.x
  let py = r.y
  let left = dist
  for (const [x, y] of r.path.slice(1)) {
    const seg = Math.hypot(x - px, y - py)
    if (seg >= left) return [px + ((x - px) * left) / seg, py + ((y - py) * left) / seg]
    left -= seg
    px = x
    py = y
  }
  return [r.x + Math.cos(r.yaw) * dist, r.y + Math.sin(r.yaw) * dist]
}
const charge = async () => {
  for (const id of ['r1', 'r2', 'r3']) await api('/api/_mock/battery', { rover: id, battery: 0.95 })
}
// Is a screen point over the open 3D view (not under a HUD panel)?
const clear = async (p) => {
  if (!p.visible || p.x < 0 || p.y < 0 || p.x > width || p.y > height) return false
  for (const sel of ['.hud-hazards', '.hud-controls', '.hud-legend', '.hud-request', '.side', '.banner-place']) {
    for (const b of await page.locator(sel).evaluateAll((els) => els.map((e) => e.getBoundingClientRect().toJSON()))) {
      if (p.x >= b.left - 6 && p.x <= b.right + 6 && p.y >= b.top - 6 && p.y <= b.bottom + 6) return false
    }
  }
  return true
}
const farFromRovers = (s, x, y, d) => Object.values(s.rovers).every((r) => Math.hypot(r.x - x, r.y - y) > d)

// ---------------------------------------------------------------- start
await api('/api/reset', { seed: 42 })
await api('/api/_mock/rearm', { armed: false }) // no first-sighting question until step 3
await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift && window.__redshift.state().rovers.r1, null, { timeout: 30000 })
await wait(5000)

// 0. A lake (first, while the fleet has not been near any water): the LiDAR hole seen
//    from a distance, identified up close, then with Reality.
await charge()
let s = await st()
const zones = Array.isArray(s.fleet.water_zones) ? s.fleet.water_zones : []
const inPoly = (x, y, poly) => {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}
// Water the fleet has not identified yet (no zone inside it or touching its shore).
const nearPoly = (x, y, poly, d) => poly.some(([px, py]) => Math.hypot(px - x, py - y) < d)
const lakes = s.features.water.filter((w) => w.kind === 'lake' && !zones.some((z) => inPoly(z[0], z[1], w.outline) || nearPoly(z[0], z[1], w.outline, z[2] + 1)))
const lake = lakes.sort((p, q) => Math.hypot(...q.outline[0]) - Math.hypot(...p.outline[0]))[0] || s.features.water[0]
const cx = lake.outline.reduce((t, p) => t + p[0], 0) / lake.outline.length
const cy = lake.outline.reduce((t, p) => t + p[1], 0) / lake.outline.length
// Out from the centre along a bearing until `off` metres past the shore.
const shore = (ang, off) => {
  let d = 0
  while (inPoly(cx + Math.cos(ang) * d, cy + Math.sin(ang) * d, lake.outline) && d < 60) d += 0.25
  return [cx + Math.cos(ang) * (d + off), cy + Math.sin(ang) * (d + off)]
}
// From the south-west, so the lake sits up and to the right of the followed rover.
const side = Math.atan2(-13, -8)
const lr = s.rovers.r3 || Object.values(s.rovers)[0]
// Drive (teleport) right round it 6.5 m off the shore: the LiDAR maps the banks but gets
// no returns from the water, so the lake is left as a dark hole in the map. That is beyond
// the 4.5 m at which a rover works out the hole is liquid. End on the south-west side.
for (let k = 1; k <= 9; k++) {
  const [x, y] = shore(side + (k / 9) * Math.PI * 2, 6.5)
  const res = await teleport(lr.id, x, y, true)
  if (res.status !== 200) console.log('teleport', res.error)
  await wait(350)
}
await follow(lr.id)
await wait(2400)
await zoom(12) // the whole lake and its far shore in view
await wait(1200)
await shot('01-lake-hole')
{
  const [x, y] = shore(side, 3)
  await teleport(lr.id, x, y, true)
}
await wait(2800)
await shot('02-lake-identified')
await page.keyboard.press('r')
await wait(1800)
await shot('03-lake-reality')
await page.keyboard.press('r')
await api('/api/chat', { text: 'resume', rover: lr.id })

await page.keyboard.press('0')
await wait(1500)

// 1. Rockfall, following a rover: placement mode with the ghost on its route, red right
//    next to it, then placed by clicking.
s = await st()
const driver = Object.values(s.rovers).find((r) => r.state === 'exploring' && r.path.length > 4) || s.rovers.r1
await follow(driver.id)
await wait(2400)
await hz('rockfall').click()
s = await st()
let [rx, ry] = ahead(s.rovers[driver.id], 7.5)
await hover(rx, ry)
await wait(700)
await shot('04-placement-ghost')
// Right next to the rover (its displayed pose lags the sim a little, so aim close).
for (let k = 0; k < 4; k++) {
  s = await st()
  await hover(s.rovers[driver.id].x + 0.7, s.rovers[driver.id].y - 0.4)
  await wait(500)
  if (await page.locator('.ghost-tag[data-bad]').count()) break
}
await shot('05-placement-invalid')
s = await st()
;[rx, ry] = ahead(s.rovers[driver.id], 7.5)
const rock = await placeAt(rx, ry)
await page.mouse.move(40, height - 40)
await wait(900)
await shot('06-rockfall-placed')

// 2. A spot the sim refuses (on the lander): the error shows inline, placement stays on.
//    (First answer the rover's question about the boulders, so its card is out of the way.)
await page.locator('.hud-request[data-kind="open"]').waitFor({ timeout: 6000 }).catch(() => {})
if (await answerOpen('Go around')) console.log('answered the rockfall question')
await page.keyboard.press('0')
await wait(2600)
s = await st()
for (let k = 0; k < 12 && !(await clear(await scr(s.lander.x + 0.5, s.lander.y - 0.3))); k++) await wait(1000)
await hz('sand').click()
await placeAt(s.lander.x + 0.5, s.lander.y - 0.3)
await wait(500)
await shot('07-placement-error')
await page.keyboard.press('Escape')
await wait(1500)

// 3. Lifeform: a herd brought in by clicking ahead of a rover (held still, so the herd
//    comes to it), the first-sighting question, then the rover following the herd.
await api('/api/_mock/rearm')
await charge()
s = await st()
const okRover = (r) => r.state !== 'fault' && r.state !== 'charging' && !s.requests.some((q) => q.rover === r.id) && r.id !== rock?.hazard?.rover
const watcher = Object.values(s.rovers).filter(okRover).sort((p, q) => Math.hypot(q.x, q.y) - Math.hypot(p.x, p.y))[0] || s.rovers.r3
// Stopped where it is (teleport in place, held), so the herd walks up to it.
await teleport(watcher.id, s.rovers[watcher.id].x, s.rovers[watcher.id].y, true)
await follow(watcher.id)
await wait(2200)
await zoom(-2)
await wait(800)
s = await st()
const wr = s.rovers[watcher.id]
// On gentle mapped ground about 10 m away, so the rover can follow the herd.
const gentle = async (x, y) => {
  for (const f of [1, 0.66, 0.33]) {
    const sl = await page.evaluate(([a, b]) => window.__redshift.slopeAt(a, b), [wr.x + (x - wr.x) * f, wr.y + (y - wr.y) * f])
    if (sl === null || sl > 14) return false
  }
  return true
}
let lp = null
for (const da of [0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8, Math.PI]) {
  const x = wr.x + Math.cos(wr.yaw + da) * 10
  const y = wr.y + Math.sin(wr.yaw + da) * 10
  if (farFromRovers(s, x, y, 4) && Math.hypot(x, y) > 10 && Math.abs(x) < 56 && Math.abs(y) < 56 && (await gentle(x, y))) {
    lp = [x, y]
    break
  }
}
lp ??= [wr.x + 9, wr.y]
await hz('lifeform').click()
await hover(lp[0], lp[1])
await wait(600)
await shot('08-lifeform-ghost')
let herd = await placeAt(lp[0], lp[1])
if (herd.status !== 200) herd = await placeAt(wr.x - 9, wr.y)
await page.mouse.move(40, height - 40)
const asked = await page.locator('.hud-request[data-kind="open"]').waitFor({ timeout: 30000 }).then(() => true).catch(() => false)
console.log(asked ? 'first-sighting question open' : 'no first-sighting question')
await wait(1800)
await shot('09-first-sighting')
await answerOpen('Follow it')
await wait(7000)
await shot('10-observing-herd')

// 4. A spire topples: a rover 6 m south-west of it watches it fall to the east. The
//    page's debug hook holds the fall at two moments (the software GPU in this test
//    draws only a few frames a second).
await charge()
s = await st()
const spires = s.features.spires.filter((p) => p.standing && s.spiresKnown.includes(p.id) && farFromRovers(s, p.x, p.y, 8) && Math.hypot(p.x, p.y) > 10)
const by = Object.values(s.rovers).find((r) => r.id !== watcher.id && r.state !== 'fault') || s.rovers.r2
let sp = null
for (const cand of spires.length ? spires : s.features.spires) {
  for (const [ox, oy] of [[-4, -4], [-4.5, -3], [-3, -4.5], [4, -4]]) {
    const res = await teleport(by.id, cand.x + ox, cand.y + oy, true)
    if (res.status === 200) {
      sp = { ...cand, side: Math.sign(ox) }
      break
    }
  }
  if (sp) break
}
console.log(`spire ${sp?.id} at (${sp?.x}, ${sp?.y}), ${by.id} beside it`)
await follow(by.id)
await wait(2600)
let fall = null
// Toppled sideways (seen side on from the camera), away from the rover.
for (const a of [0, 0.4, -0.4, 0.8].map((d) => (sp.side < 0 ? d : Math.PI - d))) {
  const x = sp.x + Math.cos(a) * 2.2
  const y = sp.y + Math.sin(a) * 2.2
  if (!(await clear(await scr(x, y)))) continue
  await hz('spire_fall').click()
  await hover(x, y)
  await wait(900)
  await shot('11-spire-preview')
  const res = await placeAt(x, y)
  if (res.status === 200) {
    fall = res
    break
  }
  await page.keyboard.press('Escape')
}
await page.mouse.move(40, height - 40)
if (fall) {
  await page.evaluate(() => window.__redshift.holdFalls(0.75))
  await wait(1500)
  await shot('12-spire-falling')
  await page.evaluate(() => window.__redshift.holdFalls(1.85))
  await wait(1500)
  await shot('13-spire-dust')
  await page.evaluate(() => window.__redshift.holdFalls(null))
  await wait(1200)
  await shot('14-spire-fallen')
} else console.log('no spire fall placed')
await api('/api/chat', { text: 'resume', rover: by.id })

// 6. Sampling: wait for a rover to start sampling (sending one to a target it found if
//    none does), follow it (3D ring and chip, its card), then its sensor panel.
await charge()
for (const id of ['r1', 'r2', 'r3']) await api('/api/chat', { text: 'resume', rover: id })
const samplingNow = async (lo, hi) => Object.values((await st()).rovers).find((r) => r.sampling && r.sampling.progress >= lo && r.sampling.progress <= hi)
let sampler = null
for (let k = 0; k < 120 && !sampler; k++) {
  sampler = await samplingNow(0.02, 0.35)
  if (!sampler) await wait(500)
  if (k === 40) {
    // Nudge: put the emptiest rover next to the closest found target on gentle ground.
    s = await st()
    const helper = Object.values(s.rovers)
      .filter((r) => r.state !== 'fault' && r.capacity && r.samples.length < r.capacity)
      .sort((p, q) => p.samples.length - q.samples.length)[0]
    const tgt = helper && s.targets.filter((t) => t.status === 'found' && farFromRovers(s, t.x, t.y, 3)).sort((p, q) => Math.hypot(p.x - helper.x, p.y - helper.y) - Math.hypot(q.x - helper.x, q.y - helper.y))[0]
    if (tgt) {
      console.log(`sending ${helper.id} next to ${tgt.id} (${tgt.kind})`)
      await teleport(helper.id, tgt.x + 2.2, tgt.y + 0.8, false)
    }
  }
}
if (sampler) {
  console.log(`${sampler.id} is sampling ${sampler.sampling.target} with the ${sampler.sampling.instrument}`)
  // Its card opens the sensor panel and follows it; Esc then closes the panel only.
  await page.locator('.rover').nth(Number(sampler.id.replace(/\D/g, '')) - 1).click()
  await wait(2200)
  await shot('15-sensors-instrument')
  await page.keyboard.press('Escape')
  await wait(900)
  await shot('16-sampling')
} else console.log('no rover sampling')

// 7. The Samples tab, and a sample picked from it (camera flies there, value tag up).
await page.locator('#tab-samples').click()
await wait(700)
await shot('17-samples-tab')
const rows = page.locator('.smp-btn')
if (await rows.count()) {
  await rows.first().click()
  await wait(2600)
  await shot('18-sample-picked')
}
await page.keyboard.press('Escape')
await page.locator('#tab-events').click()

// 8. Overview with everything (once a rover has cached samples at the lander), and the legend.
for (const id of ['r1', 'r2', 'r3']) await api('/api/chat', { text: 'resume', rover: id })
for (let k = 0; k < 80; k++) {
  if (((await st()).fleet.samples_cached ?? 0) > 0) break
  await wait(500)
}
await page.evaluate(() => document.activeElement?.blur?.())
await page.keyboard.press('0')
await wait(3000)
await shot('19-overview')
const lg = await page.locator('.hud-legend').boundingBox()
if (lg) await shot('20-legend', { x: Math.max(0, lg.x - 8), y: Math.max(0, lg.y - 8), width: lg.width + 16, height: lg.height + 16 })

// 9. Motor fault: pick the rover by clicking it on the map.
s = await st()
const victim = Object.values(s.rovers).find((r) => r.state !== 'fault' && r.state !== 'charging') || s.rovers.r1
await follow(victim.id)
await wait(2200)
await hz('rover_fault').click()
await hover(victim.x, victim.y, 0.8)
for (let k = 0; k < 4; k++) {
  s = await st()
  await hover(s.rovers[victim.id].x, s.rovers[victim.id].y, 0.8)
  await wait(500)
  if (await page.locator('.ghost-tag[data-rover]').count()) break
}
await shot('21-fault-pick-rover')
{
  const p = await scr((await st()).rovers[victim.id].x, (await st()).rovers[victim.id].y, 0.8)
  const resp = page.waitForResponse((r) => r.url().includes('/api/hazard'))
  await page.mouse.click(p.x, p.y)
  const r = await resp
  console.log(`motor fault by click: HTTP ${r.status()}`)
}
await page.mouse.move(40, height - 40)
await wait(1500)
await shot('22-fault-placed')

console.log(`${refused} expected HTTP 400 console line(s) from the refused spot`)
console.log(problems.length ? `console problems:\n${problems.join('\n')}` : 'no other console errors or warnings (camera 503 lines from the mock and swiftshader notices ignored)')
await browser.close()
