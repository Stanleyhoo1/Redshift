import test from 'node:test'
import assert from 'node:assert/strict'
import {
  DUR,
  advance,
  fading,
  frameHome,
  globeCardShown,
  globeSettled,
  globeShown,
  initialView,
  planeLabelsShown,
  progress,
  requestView,
} from '../src/globe/viewMachine.js'

test('starts on the sector, idle', () => {
  const s = initialView()
  assert.deepEqual([s.view, s.render, s.phase, s.seq, s.home], ['plane', 'plane', 'idle', 0, 0])
  assert.equal(planeLabelsShown(s), true)
  assert.equal(globeShown(s), false)
})

test('going up: out, fade, in, idle', () => {
  let s = requestView(initialView(), 'globe', 10)
  assert.deepEqual([s.view, s.render, s.phase, s.since, s.seq], ['globe', 'plane', 'out', 10, 1])
  assert.equal(planeLabelsShown(s), false)
  assert.equal(advance(s, 10.5), s)
  s = advance(s, 10 + DUR.out)
  assert.equal(s.phase, 'fade')
  assert.equal(fading(s), true)
  s = advance(s, s.since + DUR.fade)
  assert.deepEqual([s.render, s.phase, s.via], ['globe', 'in', 'fade'])
  assert.equal(globeShown(s), true)
  assert.equal(globeCardShown(s), true)
  assert.equal(globeSettled(s), false)
  s = advance(s, s.since + DUR.in)
  assert.equal(s.phase, 'idle')
  assert.equal(globeSettled(s), true)
})

test('asking for the view already asked for changes nothing', () => {
  const s = requestView(initialView(), 'globe', 3)
  assert.equal(requestView(s, 'globe', 3.1), s)
  const p = initialView()
  assert.equal(requestView(p, 'moon', 1), p)
})

test('turning back during the rise eases back in on the sector', () => {
  const up = requestView(initialView(), 'globe', 10)
  const back = requestView(up, 'plane', 10.3)
  assert.deepEqual([back.view, back.render, back.phase, back.via, back.seq], ['plane', 'plane', 'in', 'turn', 2])
  assert.equal(planeLabelsShown(back), true)
})

test('leaving the planet dives out from it first, and can turn back', () => {
  let s = requestView(initialView(), 'globe', 0, true)
  s = requestView(s, 'plane', 5)
  assert.deepEqual([s.view, s.render, s.phase], ['plane', 'globe', 'out'])
  assert.equal(planeLabelsShown(s), false)
  assert.equal(globeCardShown(s), false)
  const again = requestView(s, 'globe', 5.2)
  assert.deepEqual([again.view, again.render, again.phase, again.via], ['globe', 'globe', 'in', 'turn'])
})

test('instant switches skip the motion', () => {
  const s = requestView(initialView(), 'globe', 5, true)
  assert.deepEqual([s.view, s.render, s.phase, s.via], ['globe', 'globe', 'idle', 'instant'])
  assert.equal(requestView(s, 'globe', 6, true), s)
  const mid = requestView(initialView(), 'globe', 1)
  assert.equal(requestView(mid, 'globe', 1.2, true).render, 'globe')
})

test('progress through a phase, and frameHome', () => {
  const s = requestView(initialView(), 'globe', 10)
  assert.equal(progress(s, 10 + DUR.out / 2), 0.5)
  assert.equal(progress(s, 99), 1)
  assert.equal(progress(initialView(), 5), 1)
  assert.equal(frameHome(s).home, s.home + 1)
})
