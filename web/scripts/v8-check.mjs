// Short check of the version 8 UI against the mock: the per-rover event log filter and a
// decision's details, the Autonomous / Supervised switch in both states, a supervised
// request with Gemma's suggestion marked, the camera with detection boxes (sensor panel and
// the large viewer, fps readout), the priority chip when the mock has gaps, and no console
// errors. Saves a few screenshots.
// Needs the mock on 8000 (start it with MOCK_FIRST_QUESTION_S=99999) and `npm run preview`.
// Usage: node scripts/v8-check.mjs [url] [outDir]
// Env: CHROME, WIDTH, HEIGHT, GRAPHICS (low by default)
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:4173'
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
const outDir = process.argv[3] || 'screenshots'
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
// Software GL is slow: the low graphics setting keeps this check quick (GRAPHICS=high to override).
await page.addInitScript((g) => localStorage.setItem('redshift.graphics', g), process.env.GRAPHICS || 'low')
page.setDefaultTimeout(180000)
const problems = []
page.on('console', (msg) => {
  if (msg.type() !== 'error' && msg.type() !== 'warning') return
  const t = msg.text()
  if (/GPU stall due to ReadPixels|THREE\.Clock: This module has been deprecated/.test(t)) return
  problems.push(`[console.${msg.type()}] ${t}`)
})
page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`))

const t0 = Date.now()
const log = (m) => console.log(`${((Date.now() - t0) / 1000).toFixed(1)} s  ${m}`)
const wait = (ms) => page.waitForTimeout(ms)
const origin = new URL(url).origin
const api = async (p, body) => {
  const r = await fetch(`${origin}${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
  return { status: r.status, ...(await r.json().catch(() => ({}))) }
}
const st = () => page.evaluate(() => window.__redshift.state())
let failed = false
const fail = (m) => {
  failed = true
  console.error(`FAIL: ${m}`)
}
const shot = async (name, clip) => {
  const file = path.join(outDir, `v8-${width}x${height}-${name}.png`)
  await page.screenshot({ path: file, clip, timeout: 120000 })
  log(`saved ${file}`)
}

await api('/api/reset', { seed: 42 })
await api('/api/_mock/rearm', { armed: false })
await api('/api/mode', { mode: 'autonomous' })
await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift && window.__redshift.state().rovers.r2, null)
for (const id of ['r1', 'r2', 'r3']) await api('/api/_mock/battery', { rover: id, battery: 0.95 })
await wait(2500)

// 1. Autonomous: a decision lands in the log; the Rover 1 filter shows only Rover 1.
await page.waitForSelector('.mode-seg button[data-mode="autonomous"][aria-checked="true"]')
log(`ask r1 (autonomous): ${JSON.stringify(await api('/api/_mock/ask', { rover: 'r1' }))}`)
await page.waitForFunction(() => window.__redshift.state().events.some((e) => e.decision && e.rover === 'r1'), null)
await page.click('#tab-events')
await page.click('.evf[aria-label^="Rover 1"]')
await wait(400)
const whos = await page.$$eval('.log-list .ev .ev-who', (els) => [...new Set(els.map((e) => e.textContent))])
log(`Rover 1 filter shows rows from: ${whos.join(', ')}`)
if (whos.length !== 1 || whos[0] !== 'Rover 1') fail('the Rover 1 filter shows other rows')
const decRow = page.locator('.log-list .ev[data-decision] .dec-sum').first()
if (!(await decRow.count())) fail('no decision row in the Rover 1 log')
else {
  await decRow.click()
  await page.waitForSelector('.dec-body .dec-opts li[data-chosen]')
}
await shot('autonomous-log-rover1')

// 2. Supervised: the switch, then a request with Gemma's suggestion marked.
await page.click('.mode-seg button[data-mode="supervised"]')
await page.waitForFunction(() => window.__redshift.state().fleet.mode === 'supervised', null)
await page.waitForSelector('.mode-seg button[data-mode="supervised"][aria-checked="true"]:not([data-pending])')
log('mode switched to supervised')
await api('/api/_mock/battery', { rover: 'r2', battery: 0.95 }) // slow pages: keep it from heading home meanwhile
log(`ask r2 (supervised): ${JSON.stringify(await api('/api/_mock/ask', { rover: 'r2' }))}`)
await page.waitForSelector('.hud-request .rq-opt[data-suggested]')
await page.click('.evf', { position: { x: 5, y: 5 } }) // back to All
await page.hover('.mode-seg button[data-mode="supervised"]')
await wait(700)
await shot('supervised-request')
await page.click('.hud-request .rq-opt[data-suggested]')
await page.waitForFunction(() => !window.__redshift.state().requests.length, null)
log('answered the suggested option')

// 3. Camera: Rover 2 looks at a herd among spires; the panel draws boxes, then the viewer.
await api('/api/mode', { mode: 'autonomous' })
await api('/api/_mock/teleport', { rover: 'r2', x: -27, y: -8.5, yaw: 3.14, hold: true })
await page.keyboard.press('2')
await page.waitForSelector('.sp-cam[data-state="live"]')
await page.waitForFunction(() => (window.__redshift.state().rovers.r2?.vision?.objects?.length || 0) > 0, null)
await wait(1500)
const rate = await page.locator('.sp-cam-rate').textContent().catch(() => '')
const sees = await page.locator('.sp-sees').textContent().catch(() => '')
log(`panel camera: "${rate}", ${sees}`)
if (!sees) fail('no "Sees" line in the sensor panel')
const panel = await page.locator('.hud-sensors').boundingBox()
if (panel) await shot('sensor-panel', { x: Math.max(0, panel.x - 4), y: Math.max(0, panel.y - 4), width: Math.min(width, panel.width + 8), height: Math.min(height - panel.y + 4, panel.height + 8) })
await page.click('.sp-cam .ix-cam-hit')
await page.waitForSelector('.cv .ix-live[data-state="live"]')
await wait(2500)
const foot = await page.locator('.cv .ix-foot-r').textContent()
const live = await page.locator('.cv .ix-live').textContent()
log(`viewer: ${live} | ${foot}`)
if (!/\d+ detected/.test(foot)) fail('the viewer reports no detections')
const inked = await page.evaluate(() => {
  const c = document.querySelector('.cv .cv-boxes')
  if (!c || !c.width) return 0
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data
  let n = 0
  for (let k = 3; k < d.length; k += 16) if (d[k] > 0) n++
  return n
})
log(`box overlay pixels: ${inked}`)
if (!inked) fail('no boxes drawn on the viewer overlay')
await shot('camera-viewer')
await page.keyboard.press('Escape')
await wait(500)

// 4. Priority chip, when the mock already has unexplored gaps.
const gaps = (await api('/api/_mock/gaps')).gaps || []
if (gaps.length) {
  await page.click('#tab-comms')
  await page.fill('.cm-input', 'prioritize the gaps')
  await page.press('.cm-input', 'Enter')
  await page.waitForSelector('.prio')
  log(`priority chip: ${await page.locator('.prio-text').textContent()}`)
  await page.click('.prio-x')
  await page.waitForSelector('.prio', { state: "detached" })
  log('priority cleared from the chip')
} else log('no gaps yet in the mock: priority chip not checked')

if (problems.length) {
  console.log('\nConsole problems:')
  for (const p of problems) console.log(`  ${p}`)
} else console.log('\nNo console errors or warnings.')
await browser.close()
process.exit(failed || problems.length ? 1 : 0)
