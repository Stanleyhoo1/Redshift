// Drives the version 5 interaction features through the real UI and saves screenshots:
// following a rover, panning away (free camera with the "Back to" pill), the large
// camera viewer with its HUD (zoomed in, and the chase view), the large sensor view
// with charts, and Comms with question chips, a question waiting for its answer, and
// replies from one rover, the fleet and every rover.
// Needs the mock on 8000 (start it with MOCK_ANSWER_DELAY_MS=11000 so the "thinking"
// state lasts long enough to catch on a software GPU) and `npm run preview`.
// Usage: node scripts/v5-shots.mjs [url] [outDir]
// Env: CHROME, WIDTH, HEIGHT, STEPS (comma list of follow, camera, sensors, comms; default all)
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:4173'
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
const outDir = process.argv[3] || 'screenshots'
const tag = `${width}x${height}`
const steps = new Set((process.env.STEPS || 'follow,camera,sensors,comms').split(',').map((x) => x.trim()))
const candidates = [process.env.CHROME, '/opt/pw-browsers/chromium-1243/chrome-linux64/chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))
if (!executablePath) {
  console.error('No Chrome or Chromium found. Set CHROME=/path/to/chrome.')
  process.exit(2)
}
fs.mkdirSync(outDir, { recursive: true })

const browser = await chromium.launch({
  executablePath,
  args: ['--use-gl=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'],
})
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
page.setDefaultTimeout(180000) // a software GPU on a busy machine can take a while per frame
const problems = []
page.on('console', (msg) => {
  if (msg.type() !== 'error' && msg.type() !== 'warning') return
  const t = msg.text()
  // swiftshader and three.js / r3f notices that also appear without this change
  if (/GPU stall due to ReadPixels|THREE\.Clock: This module has been deprecated/.test(t)) return
  problems.push(`[console.${msg.type()}] ${t}`)
})
page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`))

const t0 = Date.now()
const shot = async (name) => {
  const file = path.join(outDir, `v5-${tag}-${name}.png`)
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
const scr = (x, y, lift = 0) => page.evaluate(([a, b, l]) => window.__redshift.screenOf(a, b, l), [x, y, lift])

// ---------------------------------------------------------------- start
await api('/api/reset', { seed: 42 })
await api('/api/_mock/rearm', { armed: false })
await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift && window.__redshift.state().rovers.r2, null, { timeout: 30000 })
await wait(9000) // some telemetry history for the charts, and Gemma warmed up
for (const id of ['r1', 'r2', 'r3']) await api('/api/_mock/battery', { rover: id, battery: 0.9 })

// 1. Follow Rover 2 (its sensor panel opens), then pan away with a right-drag.
await page.keyboard.press('2')
await wait(3200)
if (steps.has('follow')) {
await shot('1-follow')
const band = await page.evaluate(() => {
  const sp = document.querySelector('.hud-sensors').getBoundingClientRect()
  const c = document.querySelector('.hud-controls').getBoundingClientRect()
  const v = document.querySelector('.view').getBoundingClientRect()
  return { x: (c.right + v.right) / 2, y: (v.top + sp.top) / 2 + 30 }
})
await page.mouse.move(band.x + 120, band.y)
await page.mouse.down({ button: 'right' })
for (let k = 1; k <= 14; k++) {
  await page.mouse.move(band.x + 120 - k * 18, band.y + k * 6)
  await wait(16)
}
await page.mouse.up({ button: 'right' })
// The rover drives on; the camera stays where it was put. (Waits on the pill, not a
// timer: a software GPU on a busy machine may draw well under one frame a second.)
await page.waitForSelector('.ix-follow', { timeout: 60000 }).catch(() => {})
await wait(1500)
const pill = await page.locator('.ix-follow').count()
console.log(`after the pan: "Back to" pill ${pill ? 'shown' : 'MISSING'}`)
if (!pill) problems.push('the Back to pill did not appear after a pan')
await shot('2-panned')

// Orbiting (left-drag) and zooming while following must not detach.
await page.locator('.ix-follow-btn').click()
await page.waitForSelector('.ix-follow', { state: 'detached', timeout: 60000 })
await wait(1800)
await page.mouse.move(band.x, band.y)
await page.mouse.down()
await page.mouse.move(band.x + 80, band.y + 10, { steps: 8 })
await page.mouse.up()
await page.mouse.wheel(0, -240)
await wait(800)
const stillFollowing = !(await page.locator('.ix-follow').count())
console.log(`orbit and zoom while following: ${stillFollowing ? 'still following' : 'DETACHED'}`)
if (!stillFollowing) problems.push('orbit or zoom detached the follow')

// An event with a location, clicked while following: flies there and pauses the follow.
const ev = page.locator('.ev-btn').first()
if (await ev.count()) {
  // New events keep sliding in, so the row is never "stable": click it regardless.
  await ev.click({ force: true, timeout: 8000 })
  await page.waitForSelector('.ix-follow', { timeout: 60000 }).catch(() => {})
  const paused = await page.locator('.ix-follow').count()
  console.log(`event click while following: ${paused ? 'paused, pill shown' : 'NO PILL'}`)
  if (!paused) problems.push('an event click did not pause the follow')
  await page.keyboard.press('f') // F resumes
  await page.waitForSelector('.ix-follow', { state: 'detached', timeout: 60000 }).catch(() => problems.push('F did not resume following'))
}
}

// 2. The large camera viewer, from a click on the small camera picture.
if (steps.has('camera')) {
await page.waitForSelector('.sp-cam img', { timeout: 10000 })
await page.locator('.ix-cam-hit').click()
await page.waitForSelector('.cv-stage img', { timeout: 10000 })
await wait(2200)
await shot('3-camera')
const stage = await page.locator('.cv-stage').boundingBox()
await page.mouse.move(stage.x + stage.width * 0.6, stage.y + stage.height * 0.62)
for (let k = 0; k < 6; k++) {
  await page.mouse.wheel(0, -120)
  await wait(40)
}
await page.mouse.down()
await page.mouse.move(stage.x + stage.width * 0.5, stage.y + stage.height * 0.5, { steps: 6 })
await page.mouse.up()
await wait(900)
await shot('4-camera-zoom')
await page.mouse.dblclick(stage.x + stage.width / 2, stage.y + stage.height / 2)
await page.keyboard.press('v')
await wait(1800)
await shot('5-camera-chase')
const reqSize = await page.evaluate(() => {
  const e = performance.getEntriesByType('resource').filter((r) => r.name.includes('/api/camera/')).pop()
  return e ? new URL(e.name).search : null
})
console.log(`last camera request: ${reqSize}`)
await page.keyboard.press('Escape')
await wait(400)
if (await page.locator('.ix-modal').count()) problems.push('Esc did not close the camera viewer')
}

// 3. The large sensor view, from the IMU section's expand button.
if (steps.has('sensors')) {
await page.locator('.ix-expand-read').click()
await page.waitForSelector('.sd .ch canvas')
await wait(1500)
const radarBox = await page.locator('.sd-radar-box').boundingBox()
await page.mouse.move(radarBox.x + radarBox.width / 2, radarBox.y + radarBox.height / 2)
await page.mouse.wheel(0, -120)
await wait(1200)
await shot('6-sensors')
// Click outside the box closes it.
await page.mouse.click(6, 6)
await wait(400)
if (await page.locator('.ix-modal').count()) problems.push('a click outside did not close the sensor view')
}

// 4. Comms: question chips for Rover 2 (the selected rover), a question waiting for
//    its answer, then answers from the rover, the fleet and every rover.
if (steps.has('comms')) {
await page.locator('#tab-comms').click()
await wait(500)
await shot('7-comms-chips')
await page.locator('.ix-ask-chip', { hasText: 'What have you found?' }).click()
await page.waitForSelector('.ix-think', { timeout: 30000 }).catch(() => problems.push('no thinking indicator'))
await shot('8-comms-thinking')
await page.waitForSelector('.msg[data-kind="reply"]', { timeout: 60000 })
await page.locator('.cm-to select').selectOption('all')
await page.locator('.ix-ask-chip', { hasText: 'What have you explored?' }).click()
await wait(400)
await page.locator('.cm-input').fill('Everyone, report battery')
await page.keyboard.press('Enter')
await page.waitForFunction(() => document.querySelectorAll('.msg[data-kind="reply"]').length >= 5, null, { timeout: 60000 }).catch(() => problems.push('not all replies arrived'))
await wait(600)
await page.locator('.msg[data-kind="reply"] .msg-bubble').last().hover()
await wait(200)
await shot('9-comms-replies')
}

console.log(problems.length ? `\nPROBLEMS:\n${problems.join('\n')}` : '\nNo console errors or problems.')
await browser.close()
process.exit(problems.length ? 1 : 0)
