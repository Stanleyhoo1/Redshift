// Drives the v3 features through the real UI (Gemma pill in every state, the Comms tab
// with messages typed into its input, clarification cards: open, answered, expired,
// stacked, 409, and the planner bar in the sensor panel) and saves screenshots.
// Needs the mock on 8000 (it uses the mock-only /api/_mock/* hooks) and `npm run preview`.
// Usage: node scripts/v3-shots.mjs [url] [outDir]   Env: CHROME, WIDTH, HEIGHT
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
page.on('console', (msg) => {
  if (msg.type() !== 'error' && msg.type() !== 'warning') return
  // The mock answers camera requests with 503 on purpose.
  if (/status of 503/.test(msg.text())) return
  problems.push(`[console.${msg.type()}] ${msg.text()}`)
})
page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`))

const t0 = Date.now()
const shot = async (name) => {
  const file = path.join(outDir, `v3-${tag}-${name}.png`)
  await page.screenshot({ path: file })
  console.log(`saved ${file} at ${((Date.now() - t0) / 1000).toFixed(1)} s`)
}
const wait = (ms) => page.waitForTimeout(ms)
const origin = new URL(url).origin
const api = async (p, body) => {
  const r = await fetch(`${origin}${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
  return { status: r.status, ...(await r.json().catch(() => ({}))) }
}
const planner = (state) => api('/api/_mock/planner', { state })
const input = page.locator('.cm-input')
async function type(text, to = 'all') {
  await page.locator('.cm-to select').selectOption(to)
  await input.click()
  await input.fill(text)
  const resp = page.waitForResponse((r) => r.url().includes('/api/chat'))
  await input.press('Enter')
  const r = await resp
  console.log(`chat "${text.slice(0, 40)}" -> HTTP ${r.status()}`)
  await wait(1300)
}

// Fresh world, model warming.
await planner('warming')
await api('/api/reset', { seed: 42 })
await page.goto(url, { waitUntil: 'domcontentloaded' })
await wait(4500)
await shot('01-pill-warming')
await planner('warming') // the page load can take most of the 10 s warm-up in a software-rendered browser
await page.locator('.gm-pill').hover()
await wait(500)
await shot('02-pill-warming-details')
await page.mouse.move(400, 500)

// Ready, then a conversation through the input and a chip.
await planner('ready')
await page.locator('#tab-comms').click()
await wait(600)
await shot('03-comms-empty')
await type('Rover 2 explore north')
await page.locator('.cm-chip', { hasText: 'Hold all' }).click()
await wait(1300)
await type('Resume all')
await type('blorp the zorp quickly')
await type('go to 38, 38', 'r3')
await type('rover 1, go to 120, 4')
await wait(800)
await shot('04-comms-conversation')

await page.locator('.gm-pill').click()
await wait(500)
await shot('05-pill-ready-details')
await page.locator('.gm-pill').click()
await page.mouse.move(400, 500)

// A clarification request: card, countdown, inline question in the thread.
const asked = await api('/api/_mock/ask', { rover: 'r1' })
console.log('ask', asked)
await page.locator('.hud-request[data-kind="open"]').waitFor({ timeout: 5000 })
await wait(1200)
await shot('06-request-open')
await wait(4000)
await shot('07-request-countdown')
await page.locator('.hud-request .rq-opt', { hasText: 'Wait here' }).click()
await wait(350)
await shot('08-request-answered')
await wait(1500)
await shot('09-thread-after-answer')

// Rockfall through the hazard bar: the rover asks; read the events meanwhile.
await page.locator('#tab-events').click()
let rock = { status: 0 }
for (let k = 0; k < 6 && rock.status !== 200; k++) {
  const resp = page.waitForResponse((r) => r.url().includes('/api/hazard'))
  await page.locator('.hz-btn', { hasText: 'Rockfall' }).click()
  const r = await resp
  rock = { status: r.status(), ...(await r.json().catch(() => ({}))) }
  console.log(`rockfall: HTTP ${rock.status} ${rock.error || rock.hazard?.rover || ''}`)
  if (rock.status !== 200) await wait(4500)
}
await page.locator('.hud-request[data-kind="open"]').waitFor({ timeout: 8000 }).catch(() => console.log('no rockfall question'))
await wait(1500)
await shot('10-rockfall-question-events')

// Let it expire: the card says what the rover did.
await page.locator('.hud-request[data-kind="expired"]').waitFor({ timeout: 30000 }).catch(() => console.log('no expiry seen'))
await wait(300)
await shot('11-request-expired')
await page.locator('#tab-comms').click()
await wait(3500)
await shot('12-comms-after-expiry')

// Several open questions at once: a card and strips.
for (const r of ['r1', 'r2', 'r3']) console.log('ask', r, (await api('/api/_mock/ask', { rover: r })).status)
await wait(2000)
await shot('13-request-stack')

// A 409 (the question closed before the answer arrived), simulated at the network level.
await page.route('**/api/answer', (route) =>
  route.fulfill({ status: 409, contentType: 'application/json', body: JSON.stringify({ ok: false, error: 'Too late: that question already expired and the rover took the safe default.' }) }),
)
await page.locator('.hud-request .rq-opt').nth(1).click()
await wait(500)
await shot('14-answer-409')
await page.unroute('**/api/answer')

// The sim refuses an over-long message: its error text shows under the input.
await input.click()
await input.fill('please '.repeat(80))
await input.press('Enter')
await wait(700)
await shot('15-chat-400')
await input.fill('')

// Sensor panel with the Gemma bar, then the other planner states.
await page.locator('.rover').nth(1).click()
await wait(3000)
await shot('16-sensors-planner')
await planner('offline')
await wait(1200)
await shot('17-pill-offline')
await planner('disabled')
await wait(1200)
await shot('18-pill-disabled')
await planner('ready')

console.log(problems.length ? `console problems:\n${problems.join('\n')}` : 'no console errors or warnings (camera 503 lines from the mock ignored)')
await browser.close()
