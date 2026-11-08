import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// The project rule: no en or em dashes anywhere. Covers every file the globe step writes
// or touches.
const web = path.join(path.dirname(fileURLToPath(import.meta.url)), '..')
const BAD = /[\u2013\u2014]/
const TOUCHED = [
  'scripts/v7-globe-check.mjs',
  'src/styles/globe.css',
  'src/App.jsx',
  'src/scene/Scene.jsx',
  'src/scene/CameraRig.jsx',
  'src/scene/Placement.jsx',
  'src/scene/Rovers.jsx',
  'src/scene/Life.jsx',
  'src/scene/Targets.jsx',
  'src/scene/Lander.jsx',
  'src/scene/PlaceCandidates.jsx',
  'src/components/ViewControls.jsx',
  'src/components/HazardPanel.jsx',
  'README.md',
]

function files() {
  const out = []
  const walk = (dir) => {
    if (!fs.existsSync(dir)) return
    for (const f of fs.readdirSync(dir)) {
      const p = path.join(dir, f)
      if (fs.statSync(p).isDirectory()) walk(p)
      else out.push(p)
    }
  }
  walk(path.join(web, 'src/globe'))
  walk(path.join(web, 'test'))
  for (const f of TOUCHED) if (fs.existsSync(path.join(web, f))) out.push(path.join(web, f))
  return out
}

test('no en or em dashes in the globe code, its tests or the files it touches', () => {
  const hits = []
  for (const f of files()) {
    fs.readFileSync(f, 'utf8')
      .split('\n')
      .forEach((line, i) => {
        if (BAD.test(line)) hits.push(`${path.relative(web, f)}:${i + 1}`)
      })
  }
  assert.deepEqual(hits, [])
})
