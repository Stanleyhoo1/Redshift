import test from 'node:test'
import assert from 'node:assert/strict'
import { EXTRA_KINDS, KIND_LOOK, burrowLift, drifterPulse, legendKinds, skitterGait } from '../src/scene/lifeKinds.js'

test('every kind has a look, and the extra kinds are the four new ones', () => {
  assert.deepEqual(Object.keys(KIND_LOOK), ['grazer', 'skitterer', 'drifter', 'crawler', 'burrower'])
  assert.deepEqual(EXTRA_KINDS, ['skitterer', 'drifter', 'crawler', 'burrower'])
  for (const v of Object.values(KIND_LOOK)) {
    assert.match(v.color, /^#[0-9a-f]{6}$/)
    assert.ok(v.label && v.hint && !v.hint.includes(String.fromCharCode(0x2014)))
  }
})

test('legend lists a kind once it is noticed, or every kind with Reality on', () => {
  const life = [
    { kind: 'grazer', x: 0, y: 0 },
    { kind: 'drifter', x: 10, y: 0 },
    { kind: 'drifter', x: 11, y: 0 },
    { kind: 'burrower', x: 30, y: 0 },
    { kind: 'mystery', x: 40, y: 0 },
  ]
  const tracks = [
    { status: 'tracking', x: 10.5, y: 0.5 },
    { status: 'lost', x: 30, y: 0 },
  ]
  assert.deepEqual(legendKinds(life, tracks, false), ['drifter'])
  assert.deepEqual(legendKinds(life, tracks, true), ['grazer', 'drifter', 'burrower'])
  assert.deepEqual(legendKinds(null, tracks, true), [])
  assert.deepEqual(legendKinds([{ x: 0, y: 0 }], [], true), ['grazer'])
})

test('burrow lift eases from down to up and treats missing as up', () => {
  assert.equal(burrowLift(0), 0)
  assert.equal(burrowLift(1), 1)
  assert.equal(burrowLift(undefined), 1)
  assert.equal(burrowLift(-2), 0)
  assert.equal(burrowLift(5), 1)
  assert.ok(Math.abs(burrowLift(0.5) - 0.5) < 1e-9)
  assert.ok(burrowLift(0.25) < 0.25 && burrowLift(0.75) > 0.75)
})

test('gait and pulse stay in their ranges', () => {
  for (let ph = 0; ph < 20; ph += 0.37) {
    const g = skitterGait(ph, 1.3)
    assert.ok(Math.abs(g.swing) <= 1.0001 && Math.abs(g.jitter) <= 0.02)
    const p = drifterPulse(ph, 2.1)
    assert.ok(p.squeeze >= 0.9 && p.squeeze <= 1.06 && Math.abs(p.bob) <= 0.06 && Math.abs(p.sway) <= 0.18)
  }
})
