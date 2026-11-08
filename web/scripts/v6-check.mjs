// Short check of the version 6 place choice through the real UI: sends an ambiguous
// place order ("go to the water source") in Comms, waits for the waiting request card and
// the numbered candidate markers (the camera glides to show them), saves one screenshot,
// clicks a marker on the map, and checks the request closes with no console errors.
// Needs the mock on 8000 (start it with MOCK_FIRST_QUESTION_S=99999 so no other question
// gets in the way) and `npm run preview`.
// Usage: node scripts/v6-check.mjs [url] [outDir]
// Env: CHROME, WIDTH, HEIGHT
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
page.setDefaultTimeout(120000)
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

await api('/api/reset', { seed: 42 })
await api('/api/_mock/rearm', { armed: false })
await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift && window.__redshift.state().rovers.r2, null, { timeout: 30000 })
await wait(3000)
// The fleet learns every lake and the stream, so "the water source" is ambiguous.
log(`reveal ${JSON.stringify(await api('/api/_mock/reveal', { water: true }))}`)
for (const id of ['r1', 'r2', 'r3']) await api('/api/_mock/battery', { rover: id, battery: 0.95 })

await page.click('#tab-comms')
await page.fill('.cm-input', 'go to the water source')
await page.press('.cm-input', 'Enter')
log('sent "go to the water source"')
await page.waitForSelector('.hud-request[data-wait] .rq-wait', { timeout: 30000 })
await page.waitForFunction(() => document.querySelectorAll('.place-mark').length >= 2, null, { timeout: 30000 })
const s1 = await st()
const q = s1.requests.find((r) => r.kind === 'choose_place')
if (!q) fail('no choose_place request in the state')
else log(`request ${q.id}: ${q.question} (${q.options.length} options, deadline ${q.deadline})`)
if (q && q.deadline !== null) fail('the request should wait (no deadline)')
if (await page.locator('.hud-request .rq-left').count()) fail('a countdown is shown for a waiting request')
if (await page.locator('.hud-request .rq-safe').count()) fail('a safe default is marked on a waiting request')
const chips = await page.locator('.msg-opt[data-place]').count()
log(`inline chat chips with places: ${chips}`)
if (!chips) fail('no numbered option chips in the chat bubble')
await wait(4500) // the camera's glide to the candidates
const shotFile = path.join(outDir, `v6-${width}x${height}-choose-place.png`)
await page.screenshot({ path: shotFile, timeout: 120000 })
log(`saved ${shotFile}`)

// Click a marker badge that nothing covers (the request card may sit over one).
const target = await page.evaluate(() => {
  for (const el of document.querySelectorAll('.place-mark')) {
    const r = el.getBoundingClientRect()
    const x = r.left + r.width / 2
    const y = r.top + r.height / 2
    const hit = document.elementFromPoint(x, y)
    if (hit && (hit === el || el.contains(hit))) return { x, y, n: el.querySelector('.place-num')?.textContent }
  }
  return null
})
if (!target) fail('no marker badge is clickable')
else {
  await page.mouse.move(target.x, target.y)
  await wait(300)
  const lit = await page.locator('.rq-opt[data-hl]').count()
  log(`hover marker ${target.n}: ${lit} card chip lit`)
  if (!lit) fail('hovering a marker did not light its chip in the card')
  await page.mouse.click(target.x, target.y)
  log(`clicked marker ${target.n}`)
  try {
    await page.waitForFunction((id) => !window.__redshift.state().requests.some((r) => r.id === id), q?.id, { timeout: 20000 })
    log('request closed')
  } catch {
    fail('the request did not close after clicking a marker')
  }
  await wait(1500)
  const markers = await page.locator('.place-mark').count()
  if (markers) fail(`${markers} markers still shown after the request closed`)
  const acks = await page.evaluate((id) => [...document.querySelectorAll('.msg[data-kind="ack"] .msg-bubble')].map((e) => e.textContent).slice(-1)[0], q?.id)
  log(`rover says: ${acks}`)
}

if (problems.length) {
  console.log('\nConsole problems:')
  for (const p of problems) console.log(`  ${p}`)
} else console.log('\nNo console errors or warnings.')
await browser.close()
process.exit(failed || problems.length ? 1 : 0)
