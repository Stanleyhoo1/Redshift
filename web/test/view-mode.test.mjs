import test from 'node:test'
import assert from 'node:assert/strict'
import { appStore, simStore } from '../src/state/store.js'
import { DUR } from '../src/globe/viewMachine.js'
import { globeShown, installViewGuards, leaveGlobe, viewMode } from '../src/globe/viewMode.js'

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

const placeQuestion = (id) => ({ id, kind: 'choose_place', rover: 'r2', options: [{ id: 'p1', x: 1, y: 2 }, { id: 'p2', x: 5, y: 6 }] })

test('a new place question or a placement started elsewhere brings the sector back', () => {
  reset()
  simStore.set({ ...simStore.get(), gen: 1, requests: [placeQuestion('q0')] })
  const off = installViewGuards()
  // A question open before the planet was asked for does not bounce the planet.
  viewMode.request('globe', { instant: true })
  simStore.set({ ...simStore.get(), requests: [placeQuestion('q0')] })
  assert.equal(viewMode.get().view, 'globe')
  // A plain question (no places) does not either.
  simStore.set({ ...simStore.get(), requests: [placeQuestion('q0'), { id: 'q1', kind: 'confirm', options: [{ id: 'yes', label: 'Yes' }] }] })
  assert.equal(viewMode.get().view, 'globe')
  // A new place question does.
  simStore.set({ ...simStore.get(), requests: [placeQuestion('q0'), placeQuestion('q2')] })
  assert.equal(viewMode.get().view, 'plane')
  // Placement started from anywhere while on the planet does too.
  viewMode.request('globe', { instant: true })
  appStore.patch({ placing: { type: 'sand', mode: 'point' } })
  assert.equal(viewMode.get().view, 'plane')
  appStore.patch({ placing: null })
  // A new init (Reset sim, or a reconnect that re-sends the open questions with the same
  // ids) bounces nothing by itself, even with a question this guard has not seen before.
  viewMode.request('globe', { instant: true })
  simStore.set({ ...simStore.get(), gen: 2, requests: [placeQuestion('q2'), placeQuestion('q4')] })
  assert.equal(viewMode.get().view, 'globe')
  // A place question that arrives after it does.
  simStore.set({ ...simStore.get(), requests: [placeQuestion('q2'), placeQuestion('q4'), placeQuestion('q5')] })
  assert.equal(viewMode.get().view, 'plane')
  off()
  viewMode.request('globe', { instant: true })
  simStore.set({ ...simStore.get(), requests: [placeQuestion('q3')] })
  assert.equal(viewMode.get().view, 'globe', 'no guard after uninstall')
  reset()
})
