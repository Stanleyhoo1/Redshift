import test from 'node:test'
import assert from 'node:assert/strict'
import { appStore } from '../src/state/store.js'
import { DUR } from '../src/globe/viewMachine.js'
import { globeShown, leaveGlobe, viewMode } from '../src/globe/viewMode.js'

const reset = () => viewMode.request('plane', { instant: true })

test('a request walks out, fade and in on the store clock', () => {
  reset()
  assert.equal(leaveGlobe(), false)
  viewMode.request('globe')
  let s = viewMode.get()
  assert.deepEqual([s.view, s.phase], ['globe', 'out'])
  viewMode.tick(s.since + DUR.out)
  s = viewMode.get()
  assert.equal(s.phase, 'fade')
  viewMode.tick(s.since + DUR.fade)
  s = viewMode.get()
  assert.equal(s.render, 'globe')
  assert.equal(globeShown(s), true)
  assert.equal(leaveGlobe(), true)
  assert.deepEqual([viewMode.get().view, viewMode.get().phase], ['plane', 'out'])
})

test('subscribers hear changes, not idle ticks', () => {
  reset()
  let calls = 0
  const off = viewMode.subscribe(() => calls++)
  viewMode.tick()
  assert.equal(calls, 0)
  viewMode.request('globe')
  assert.equal(calls, 1)
  viewMode.request('globe')
  assert.equal(calls, 1)
  off()
})

test('toggle flips the asked view; frameHome only counts on the planet', () => {
  reset()
  viewMode.toggle({ instant: true })
  assert.equal(viewMode.get().render, 'globe')
  const h = viewMode.get().home
  viewMode.frameHome()
  assert.equal(viewMode.get().home, h + 1)
  viewMode.toggle({ instant: true })
  assert.equal(viewMode.get().render, 'plane')
  viewMode.frameHome()
  assert.equal(viewMode.get().home, h + 1)
})

test('asking for the planet cancels hazard placement', () => {
  reset()
  appStore.patch({ placing: { type: 'rockfall', mode: 'point' } })
  viewMode.request('globe')
  assert.equal(appStore.get().placing, null)
  reset()
})
