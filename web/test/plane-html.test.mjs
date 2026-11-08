import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// drei <Html> labels live in the page, so they do not hide with the sector scene. Every
// label outside src/globe must come through src/globe/PlaneHtml.jsx, which renders nothing
// while the planet is up. A namespace import of drei is refused too, since `drei.Html`
// could not be told apart here.
const src = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'src')
const NAMED = /import\s*\{[^}]*\bHtml\b[^}]*\}\s*from\s*['"]@react-three\/drei['"]/
const NAMESPACE = /import\s+\*\s+as\s+\w+\s+from\s*['"]@react-three\/drei['"]/

test('sector labels use PlaneHtml, never drei Html directly', () => {
  const hits = []
  const walk = (dir) => {
    for (const f of fs.readdirSync(dir)) {
      const p = path.join(dir, f)
      if (fs.statSync(p).isDirectory()) {
        if (path.relative(src, p) !== 'globe') walk(p)
      } else if (/\.(jsx?|mjs)$/.test(f)) {
        const text = fs.readFileSync(p, 'utf8')
        if (NAMED.test(text) || NAMESPACE.test(text)) hits.push(path.relative(src, p))
      }
    }
  }
  walk(src)
  assert.deepEqual(hits.sort(), [])
})
