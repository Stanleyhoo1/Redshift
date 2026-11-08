import test from 'node:test'
import assert from 'node:assert/strict'
import { RETURN_OUT, RISE_RATE, settleControls, topDownPose } from '../src/globe/planeTransition.js'

const near = (a, b, eps = 1e-9) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < eps, `${a} vs ${b}`))

test('top-down pose: above the middle, the whole sector in view, a hair south', () => {
  const map = { ox: -60, oy: -60, width: 120, depth: 120, minH: -2, maxH: 6 }
  const p = topDownPose(map, 42)
  near(p.target, [0, 2, 0])
  const d = p.position[1] - p.target[1]
  assert.ok(Math.abs(d - 252.105) < 0.01, `height ${d}`)
  assert.ok(Math.abs(p.position[0] - p.target[0]) < 1e-9)
  const south = p.position[2] - p.target[2]
  assert.ok(south > 0 && south < 1, `south offset ${south}`)
  assert.ok(topDownPose(map, 42, 0.8).position[1] < p.position[1])
})

test('an off-centre, oblong map is centred on its middle', () => {
  const p = topDownPose({ ox: 0, oy: 10, width: 40, depth: 80, minH: 0, maxH: 2 }, 42)
  near(p.target, [20, 1, -50])
})

test('settleControls applies the damped remainder once and restores damping', () => {
  const calls = []
  const c = {
    enableDamping: true,
    update() {
      calls.push(this.enableDamping)
    },
  }
  settleControls(c)
  assert.deepEqual(calls, [false])
  assert.equal(c.enableDamping, true)
  settleControls(null)
})

test('rates are sane', () => {
  assert.ok(RISE_RATE > 2 && RISE_RATE < 8)
  assert.ok(RETURN_OUT > 1 && RETURN_OUT < 3)
})
