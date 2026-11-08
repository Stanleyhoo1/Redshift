// Headless check of the v7 setup screen through the real UI. The sim must be in its setup phase (the real
// sim with AUTO_LAUNCH=0, or the mock with MOCK_PRELAUNCH=1). Checks that the screen opens by itself, fills
// the window and cannot be dismissed (not by Esc, the backdrop, Tab into the dashboard behind it or New
// mission behind it), that the world waits, picks Earth gravity and a 4th rover, switches one
// feature, checks the layout at phone width, launches (a new world with 4 rovers at 1 g that moves), then
// reopens it from "New mission" (a centred dialog that starts from the launched settings and Esc closes).
// Saves 5 screenshots.
// Usage: node scripts/setup-check.mjs [url] [outDir]
// Env: CHROME, WIDTH, HEIGHT
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:5181'
const outDir = process.argv[3] || 'screenshots'
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
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
page.setDefaultTimeout(60000)
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
const st = () => page.evaluate(() => window.__redshift.state())
let failed = false
const check = (ok, m) => {
  if (ok) log(`ok: ${m}`)
  else {
    failed = true
    console.error(`FAIL: ${m}`)
  }
}
const shot = async (name) => {
  await page.screenshot({ path: path.join(outDir, name) })
  log(`saved ${name}`)
}
const dialogs = () => page.getByRole('dialog').count()

await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift?.state().phase === 'setup', null, { timeout: 60000 })
const forced = page.getByRole('dialog', { name: 'Plan the landing' })
await forced.waitFor()
await page.waitForSelector('.su-g-num') // the settings arrived
const s0 = await st()
log(`setup phase: gen ${s0.gen}, ${s0.roverMeta.length} rovers, config ${JSON.stringify(s0.config)}`)
await wait(800) // the fade-in
await shot('1-setup.png')
const box = await forced.boundingBox()
check(!!box && box.x === 0 && box.y === 0 && Math.round(box.width) === width && Math.round(box.height) === height,
  `the forced screen fills the window (${JSON.stringify(box)})`)

await page.keyboard.press('Escape')
await page.mouse.click(6, 6) // the backdrop
await wait(400)
check(await forced.isVisible(), 'Esc and a backdrop click leave the forced screen open')
check((await forced.locator('.ix-close').count()) === 0, 'the forced screen has no close button')
// The dashboard behind is inert: Tab never leaves the screen, so Reset sim and New mission cannot be reached.
const behind = await page.evaluate(() => [document.querySelector('main.view')?.inert, document.querySelector('aside.side')?.inert])
check(behind[0] === true && behind[1] === true, `the dashboard behind the forced screen is inert (${behind})`)
await page.locator('#su-seed').focus()
let strayed = ''
for (let i = 0; i < 32 && !strayed; i++) {
  await page.keyboard.press('Tab')
  strayed = await page.evaluate(() => {
    const a = document.activeElement
    return a && a !== document.body && !a.closest('[role="dialog"]') ? (a.textContent || a.tagName).trim().slice(0, 30) : ''
  })
}
check(!strayed, `Tab stays on the forced screen${strayed ? ` (reached ${strayed})` : ''}`)
await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'New mission')?.click())
await page.keyboard.press('Escape')
await wait(400)
check((await forced.isVisible()) && (await dialogs()) === 1, 'New mission and Esc during setup leave the forced screen as it is')
const tA = (await st()).t
await wait(1500)
check((await st()).t === tA, `the world waits in the setup phase (t stays ${tA})`)

await forced.getByRole('button', { name: /^Earth\b/ }).click()
check((await page.locator('.su-g-num').textContent()) === '1.0', 'Earth sets 1.0 g')
check(/9\.8/.test(await page.locator('.su-g-ms').textContent()), 'and reads 9.8 m/s2')
const count = async () => Number(await page.locator('.su-count').textContent())
while ((await count()) < 4) await forced.getByRole('button', { name: 'More rovers' }).click()
while ((await count()) > 4) await forced.getByRole('button', { name: 'Fewer rovers' }).click()
check((await count()) === 4, '4 rovers picked')
const tile = forced.locator('.su-tile').first()
const tileKey = (await tile.locator('.su-tile-name').getAttribute('id')).replace(/^su-t-/, '')
const wasOn = (await tile.getAttribute('aria-pressed')) === 'true'
await tile.click()
check((await tile.getAttribute('aria-pressed')) === String(!wasOn), `${tileKey} switched ${wasOn ? 'off' : 'on'}`)
const said = await page.locator('.su-summary').textContent()
check(said.startsWith('4 rovers land') && said.includes('1.0 g'), `summary: ${said}`)
await wait(400)
await shot('2-setup-edited.png')

await page.setViewportSize({ width: 390, height: 844 })
await wait(600)
const over = await page.evaluate(() => ['.su', '.su-body', '.su-foot'].map((sel) => {
  const e = document.querySelector(sel)
  return e ? e.scrollWidth - e.clientWidth : -1
}))
check(over.every((d) => d === 0), `no sideways scroll at 390 px (${over.join(', ')})`)
check(await forced.getByRole('button', { name: 'Launch', exact: true }).isVisible(), 'Launch is on screen at 390 px')
await shot('3-setup-phone.png')
await page.setViewportSize({ width, height })
await wait(600)

const gen0 = (await st()).gen
await forced.getByRole('button', { name: 'Launch', exact: true }).click()
log('launch pressed')
await page.waitForFunction((g) => {
  const s = window.__redshift?.state()
  return !!s && s.gen > g && s.phase === 'running'
}, gen0, { timeout: 90000 })
await wait(1500)
check((await dialogs()) === 0, 'the setup screen closed after the launch')
const s1 = await st()
check(s1.roverMeta.length === 4, `4 rovers landed (${s1.roverMeta.length})`)
check(s1.world.gravity_g === 1, `the world is at 1 g (${s1.world.gravity_g})`)
check(s1.config?.gravity_g === 1 && s1.config?.num_rovers === 4, `init config matches (${JSON.stringify(s1.config)})`)
check(s1.config?.[tileKey] === !wasOn, `${tileKey} is ${!wasOn} in the new world's config`)
const tB = s1.t
await wait(2500)
check((await st()).t > tB, 'the world moves after the launch')
await shot('4-launched.png')

await page.getByRole('button', { name: 'New mission' }).click()
const again = page.getByRole('dialog', { name: 'New mission' })
await again.waitFor()
await page.waitForSelector('.su-g-num')
await wait(400) // the pop-in animation
const box2 = await again.boundingBox()
check(!!box2 && box2.x > 0 && box2.width < width, `New mission is a centred dialog, not full-screen (${JSON.stringify(box2)})`)
check((await page.locator('.su-g-num').textContent()) === '1.0', 'New mission starts from the launched gravity')
check((await count()) === 4, 'and the launched fleet size')
check((await again.locator('.ix-close').count()) === 1, 'New mission has a close button')
await wait(600)
await shot('5-new-mission.png')
await page.keyboard.press('Escape')
await wait(500)
check((await dialogs()) === 0, 'Esc closes New mission')

if (problems.length) {
  console.log('\nConsole problems:')
  for (const p of problems) console.log(`  ${p}`)
} else console.log('\nNo console errors or warnings.')
await browser.close()
process.exit(failed || problems.length ? 1 : 0)
