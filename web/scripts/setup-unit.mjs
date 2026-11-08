// Node checks (no browser) for the v7 setup screen: the pure helpers in src/lib/setup.js, the init
// fields in src/state/simState.js and the HTTP helpers in src/net/connection.js (with a stand-in fetch).
// Usage: node scripts/setup-unit.mjs
import assert from 'node:assert/strict'
import { SEED_FIELD, diffSettings, fieldsByGroup, formatValue, gravityLook, gravityText, metersPerSecond2, parseSeed, physicsAt, presetName, readableError, snap, summary } from '../src/lib/setup.js'

const GRAV = { key: 'gravity_g', type: 'float', label: 'Gravity', group: 'planet', help: '', min: 0.5, max: 3, step: 0.1, unit: 'g' }
const MAP = { key: 'map_size', type: 'int', label: 'Sector size', group: 'sector', help: '', min: 60, max: 200, step: 10, unit: 'm' }
const SPEED = { key: 'sim_speed', type: 'float', label: 'Speed', group: 'simulation', help: '', min: 0.5, max: 10, step: 0.5, unit: 'x' }
const ROVERS = { key: 'num_rovers', type: 'int', label: 'Rovers', group: 'fleet', help: '', min: 1, max: 8, step: 1, unit: '' }
const WATER = { key: 'water', type: 'bool', label: 'Water', group: 'features', help: '' }
const MESAS = { key: 'mesas', type: 'bool', label: 'Mesas', group: 'features', help: '' }
const WEAR = { key: 'wear', type: 'choice', label: 'Wear', group: 'wear', help: '', choices: [{ value: 'off' }, { value: 'normal' }, { value: 'harsh' }] }
const FIELDS = [GRAV, SEED_FIELD, ROVERS, MAP, WATER, MESAS, WEAR, SPEED]

// snap: on the step grid, inside the range, no float noise
assert.equal(snap(GRAV, '1.2300001'), 1.2)
assert.equal(snap(GRAV, 0.1 + 0.2), 0.5)
assert.equal(snap(GRAV, 2.96), 3)
assert.equal(snap(GRAV, 7), 3)
assert.equal(snap(GRAV, 'abc'), 0.5)
assert.equal(snap(MAP, 124), 120)
assert.equal(snap(MAP, 126), 130)
assert.equal(snap(ROVERS, 0), 1)
assert.equal(snap(SPEED, 3.3), 3.5)
assert.equal(snap({ ...SPEED, step: 0.25, min: 0.25 }, 0.8), 0.75)

// diffSettings: only what changed (numbers within a tiny tolerance count as unchanged)
assert.deepEqual(diffSettings({ gravity_g: 2.0000000001, num_rovers: 4, water: true, wear: 'harsh' }, { gravity_g: 2, num_rovers: 3, water: true, wear: 'normal' }), { num_rovers: 4, wear: 'harsh' })
assert.deepEqual(diffSettings({ seed: 42 }, { seed: 42 }), {})

// groups (an unknown group lands in Simulation), seeds, physics rows, presets
const groups = fieldsByGroup([...FIELDS, { key: 'future', type: 'bool', group: 'somewhere' }, null])
assert.deepEqual(groups.planet.map((f) => f.key), ['gravity_g', 'seed'])
assert.deepEqual(groups.features.map((f) => f.key), ['water', 'mesas'])
assert.deepEqual(groups.simulation.map((f) => f.key), ['sim_speed', 'future'])
assert.equal(parseSeed(' 42 '), 42)
assert.equal(parseSeed('0'), 0)
for (const bad of ['', '-1', '4.5', 'abc', '2147483648', '99999999999', null]) assert.equal(parseSeed(bad), null, String(bad))
// the held world's seed is fine as it is, even above the range (a reset or SEED may give one)
assert.equal(parseSeed('3000000000', SEED_FIELD, 3000000000), 3000000000)
assert.equal(parseSeed(' 3000000000 ', SEED_FIELD, 3000000000), 3000000000)
assert.equal(parseSeed('3000000001', SEED_FIELD, 3000000000), null)
assert.equal(parseSeed('7', SEED_FIELD, 3000000000), 7)
assert.equal(parseSeed('-1', SEED_FIELD, -1), null)
const rows = [{ gravity_g: 1.9, climb_deg: 22.6, tow_deg: 8.8 }, { gravity_g: 2, climb_deg: 21.3, tow_deg: 8.4 }]
assert.equal(physicsAt(rows, 2.04).climb_deg, 21.3)
assert.equal(physicsAt([], 2), null)
const presets = [{ label: 'Earth', value: 1 }, { label: 'Redshift-b', value: 2 }]
assert.equal(presetName(presets, 2), 'Redshift-b')
assert.equal(presetName(presets, 2.1), '')
assert.equal(metersPerSecond2(1), '9.8')
assert.equal(metersPerSecond2(2), '19.6')

// the gravity numeral narrows, gets heavier and shifts toward red as gravity grows
const light = gravityLook(0.5)
const mid = gravityLook(2)
const heavy = gravityLook(3)
assert.ok(light.stretch > mid.stretch && mid.stretch > heavy.stretch, 'stretch')
assert.ok(light.weight < mid.weight && mid.weight < heavy.weight, 'weight')
assert.equal(light.color, 'rgb(232, 238, 245)')
assert.notEqual(heavy.color, mid.color)

// value text and the summary sentence under the form
assert.equal(formatValue(GRAV, 2), '2.0 g')
assert.equal(formatValue(GRAV, 0.5 + 12 * 0.1), '1.7 g')
assert.equal(formatValue(GRAV, 2.04), '2.04 g') // a world an env var started off the 0.1 g grid
assert.equal(gravityText(3), '3.0')
assert.equal(gravityText(1.23456), '1.235')
assert.equal(gravityText('x'), '')

// a refused launch reads with the screen's labels, one sentence per problem
assert.equal(readableError('gravity_g must be between 0.5 and 3 g; num_rovers must be between 1 and 8', FIELDS), 'Gravity must be between 0.5 and 3 g. Rovers must be between 1 and 8.')
assert.equal(readableError('map_size must be in steps of 10 m', FIELDS), 'Sector size must be in steps of 10 m.')
assert.equal(readableError("unknown setting 'warp' (use one of gravity_g, seed)", FIELDS), "unknown setting 'warp' (use one of gravity_g, seed).")
assert.equal(readableError('Another launch started the mission first. Use New mission to change its settings.', FIELDS), 'Another launch started the mission first. Use New mission to change its settings.')
assert.equal(readableError('warp_drive must be a number', FIELDS), 'warp_drive must be a number.') // a key this table does not know
assert.equal(readableError('', FIELDS), '')
assert.equal(formatValue(MAP, 120), '120 m')
assert.equal(formatValue(SPEED, 3), '3x real time')
assert.equal(formatValue(ROVERS, 4), '4')
const values = { gravity_g: 2, seed: 42, num_rovers: 3, map_size: 120, water: true, mesas: false, wear: 'normal', sim_speed: 3 }
assert.equal(summary(values, FIELDS), '3 rovers land on a 120 m sector at 2.0 g. 1 of 2 features on, normal wear, 3x real time.')
assert.equal(summary({ ...values, num_rovers: 1, wear: 'off' }, FIELDS), '1 rover lands on a 120 m sector at 2.0 g. 1 of 2 features on, no wear, 3x real time.')
assert.equal(summary({ seed: 7 }, [SEED_FIELD]), 'Seed 7.')
console.log('setup-unit: helpers ok')

// ---- simState: init `phase` and `config` (v7), derived from init when an older sim sends none
const { applyInit, applyTick, emptyState } = await import('../src/state/simState.js')
const INIT = {
  type: 'init',
  world: { name: 'Redshift-b', gravity: 19.62, gravity_g: 2, seed: 42 },
  map: { nx: 241, ny: 241, res: 0.5, origin: [-60, -60] },
  rovers: [{ id: 'r1', name: 'Rover 1', color: '#ff7a45' }, { id: 'r2', name: 'Rover 2', color: '#36cfc9' }],
  fleet: { sim_speed: 3 },
}
assert.equal(emptyState().phase, 'running')
let st = applyInit(emptyState(), INIT, 0)
assert.equal(st.phase, 'running') // an older sim: no phase means running
assert.equal(st.configDerived, true)
assert.deepEqual(st.config, { seed: 42, gravity_g: 2, map_size: 120, num_rovers: 2, sim_speed: 3 })
st = applyInit(st, { ...INIT, phase: 'setup', config: { gravity_g: 1.5, seed: 7, water: false, wear: 'harsh', bad: { x: 1 }, gone: null } }, 1)
assert.equal(st.phase, 'setup')
assert.equal(st.configDerived, false)
assert.deepEqual(st.config, { gravity_g: 1.5, seed: 7, water: false, wear: 'harsh' })
st = applyTick(st, { type: 'tick', t: 1, rovers: [] }, 2)
assert.equal(st.phase, 'setup') // only an init changes the phase
assert.equal(applyInit(st, { ...INIT, phase: 'weird' }, 3).phase, 'running')
console.log('setup-unit: init fields ok')

// ---- net/connection.js with a stand-in fetch: the server's own error text reaches the UI
const { getConfig, launch, resetSim } = await import('../src/net/connection.js')
const calls = []
let reply = { status: 200, text: '{}' }
globalThis.fetch = async (url, opts = {}) => {
  calls.push({ url, method: opts.method || 'GET', body: opts.body ? JSON.parse(opts.body) : null })
  return new Response(reply.text, { status: reply.status, headers: { 'Content-Type': 'application/json' } })
}
const answer = (status, body) => {
  reply = { status, text: typeof body === 'string' ? body : JSON.stringify(body) }
}
answer(404, { detail: 'Not Found' })
assert.equal(await getConfig(), null) // an older sim: the seed-only form
answer(200, { ok: true, phase: 'setup', current: { seed: 1 }, defaults: {}, fields: [SEED_FIELD], presets: {}, physics: [] })
assert.equal((await getConfig()).phase, 'setup')
assert.equal(calls.at(-1).url, '/api/config')
answer(500, { ok: false, error: 'boom' })
await assert.rejects(getConfig(), /boom/)
answer(400, { ok: false, error: 'gravity_g must be between 0.5 and 3 g', errors: ['gravity_g must be between 0.5 and 3 g'] })
await assert.rejects(launch({ gravity_g: 9 }), /gravity_g must be between 0.5 and 3 g/)
assert.deepEqual(calls.at(-1), { url: '/api/launch', method: 'POST', body: { gravity_g: 9 } })
answer(200, { ok: true, phase: 'running', config: { seed: 1 } })
assert.equal((await launch({})).phase, 'running')
answer(200, { ok: true, seed: 7 })
assert.equal((await resetSim(7)).seed, 7)
assert.deepEqual(calls.at(-1).body, { seed: 7 })
answer(400, { ok: false, error: 'seed must be between 0 and 2147483647' })
await assert.rejects(resetSim(-1), /seed must be between/)
answer(502, '')
await assert.rejects(resetSim(7), /Reset failed \(HTTP 502\)/)
console.log('setup-unit: connection ok')
