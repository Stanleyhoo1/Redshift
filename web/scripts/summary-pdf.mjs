// Renders submission/technical-summary.html to a one-page A4 PDF with a local Chromium.
// Usage: node scripts/summary-pdf.mjs [in.html] [out.pdf]   (CHROME=/path/to/chrome to pick a browser)
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'
import url from 'node:url'

const here = path.dirname(url.fileURLToPath(import.meta.url))
const input = path.resolve(process.argv[2] || path.join(here, '../../submission/technical-summary.html'))
const output = path.resolve(process.argv[3] || input.replace(/\.html$/, '.pdf'))
const candidates = [process.env.CHROME, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))

const browser = await chromium.launch(executablePath ? { executablePath } : {})
const page = await browser.newPage()
await page.goto(url.pathToFileURL(input).href)
const html = await page.content()
const left = html.match(/\{\{[A-Z]+\}\}/g)
if (left) console.warn(`Unfilled placeholders: ${[...new Set(left)].join(', ')}`)
await page.pdf({ path: output, format: 'A4', preferCSSPageSize: true, printBackground: true })
const pages = (fs.readFileSync(output, 'latin1').match(/\/Type\s*\/Page[^s]/g) || []).length
await browser.close()
console.log(`${output}: ${pages} page${pages === 1 ? '' : 's'}`)
if (pages !== 1) process.exit(1)
