// Colours shared by the 3D scene and the sidebar. Keep in sync with styles.css.

export const SIGNAL = '#4fe0ef' // the one accent: discovery, frontiers, scan edge
export const VOID = '#05070b'

// Terrain colours, exactly as drawn on flat ground (the legend uses the same values).
export const SLOPE = {
  ok: '#1d6a55',
  caution: '#c9691b',
  bad: '#bf2a1d',
}

// Height palette for "Slope colours" off: neutral rock (dark basalt to pale stone), low to high.
export const GROUND_RAMP = ['#38332f', '#565049', '#7a7068', '#a39889', '#d3c9bb']

export const STATE_STYLE = {
  exploring: { label: 'Exploring', color: '#62d6a8' },
  investigating: { label: 'Investigating', color: SIGNAL },
  returning: { label: 'Returning', color: '#93a6ff' },
  charging: { label: 'Charging', color: '#f2c14c' },
  recovering: { label: 'Recovering', color: '#f46a5b' },
  idle: { label: 'Idle', color: '#8d99aa' },
  fault: { label: 'Fault', color: '#ff3b5c' },
  observing: { label: 'Observing', color: '#c8f25a' },
  // v9: health, rescue and repair
  disabled: { label: 'Disabled', color: '#ff4d4d', alert: true },
  rescuing: { label: 'Rescuing', color: '#ffb547' },
  towing: { label: 'Towing', color: '#ffb547' },
  towed: { label: 'In tow', color: '#f0a05a' },
  repairing: { label: 'Repairing', color: '#5fd4c8' },
  wrecked: { label: 'Wrecked', color: '#8a93a3', wreck: true },
}

// Environment changes injected into the sim (PROTOCOL.md version 2). Crimson, so a
// hazard never reads as a rover warning (amber) or a discovery (cyan).
export const HAZARD = '#ff3b5c'
export const SAND = '#e2bf7d'
export const DUST = '#b98a5c'
// `place`: how the button works with a version 4 sim. 'point' = click the map (x, y),
// 'rover' = click a rover, null = one click. `radius` sizes the ghost marker (metres).
// `v4`: only offered when the sim speaks version 4. `discovery`: styled as a find, not a hazard.
// `v9`: only offered when the sim speaks version 9 (rover health). Damage picks a rover, then
// a severity (and optionally a part) in a small popover.
export const HAZARD_TYPES = [
  { type: 'rockfall', label: 'Rockfall', place: 'point', radius: 2.8, onRover: true, hint: 'Boulders land on a rover route. LiDAR sees them, the planner steers around.' },
  { type: 'sand', label: 'Soft sand', place: 'point', radius: 2.3, hint: 'Hidden soft ground on a route. Only wheel slip gives it away.' },
  { type: 'spire_fall', label: 'Falling spire', place: 'point', radius: 0.8, v4: true, onRover: true, hint: 'A crystal spire topples across a route. LiDAR sees it, the planner steers around.' },
  { type: 'dust_storm', label: 'Dust storm', place: null, hint: 'LiDAR range collapses to a few metres for a while. Rovers slow down.' },
  { type: 'rover_fault', label: 'Motor fault', place: 'rover', hint: 'One rover loses its motors and reboots. The fleet routes around it.' },
  { type: 'damage', label: 'Damage', place: 'rover', v9: true, hint: 'Damage a rover (light, heavy, disable or wreck). Watch its diagnosis, and the rescue if it cannot drive.' },
  { type: 'lifeform', label: 'Lifeform', place: 'point', radius: 3.2, v4: true, discovery: true, hint: 'A small herd of creatures wanders in. The rovers still have to notice them.' },
]

// Version 4: liquid water, alien crystal spires and lifeforms.
export const WATER = { deep: '#0b3260', mid: '#2f7fd0', glint: '#e8fbff', line: '#9fd6ff' }
export const SPIRE = { body: '#cbbcff', glow: '#8a63ff', line: '#b79cff' }
export const LIFE = '#c8f25a' // creatures, fleet tracks and the "Observing" state

// Version 4b: the instrument each target kind is sampled with.
export const INSTRUMENTS = {
  drill: { label: 'Drill', verb: 'Drilling a core', unit: 'core' },
  scoop: { label: 'Scoop', verb: 'Scooping', unit: 'scoop' },
  sniffer: { label: 'Sniffer', verb: 'Sniffing gas', unit: 'gas sample' },
  swab: { label: 'Swab', verb: 'Swabbing', unit: 'swab' },
}
export const instrumentStyle = (k) => INSTRUMENTS[k] || { label: k ? k[0].toUpperCase() + k.slice(1) : 'Instrument', verb: 'Sampling', unit: 'sample' }

export const KIND_STYLE = {
  mineral: { label: 'Mineral', long: 'mineral deposit', color: '#b48cff' },
  ice: { label: 'Ice', long: 'ice patch', color: '#a8d4ff' },
  vent: { label: 'Vent', long: 'thermal vent', color: '#ff8a3d' },
  trace: { label: 'Trace', long: 'organic trace', color: '#ff9ec7' },
}

// The realistic 3D world (sRGB hex). A dense, hazy super-Earth atmosphere lit by a low,
// warm red-dwarf sun: peach haze at the horizon, dusky mauve at the zenith.
export const ATMOSPHERE = {
  zenith: '#3d3747',
  sky: '#8a6863',
  horizon: '#d6a586',
  haze: '#b48a75', // distance haze, matched to the sky low over the horizon
  groundHaze: '#5a4639', // the sky dome below the horizon
  sun: '#fff1d6', // sun disk
  sunGlow: '#ffb27a',
  sunLight: '#ffdab4', // key light
  fillSky: '#b9a3a4', // hemisphere fill from the sky
  fillGround: '#4a372b', // and bounced from the ground
  dust: '#7d5a40', // dust storm haze
  dustSky: '#5a4030',
}

// Ground materials: fine ochre dust on flats, darker basalt on steep ground and ridges.
export const REGOLITH = {
  dustLight: '#ad9a88',
  dust: '#86735f',
  soil: '#4f4741',
  rock: '#4a4440',
  rockLight: '#7a6f67',
  strataA: '#83695a',
  strataB: '#3b3431',
  pebble: '#a09385',
}

export const stateStyle = (s) => STATE_STYLE[s] || { label: s ? s[0].toUpperCase() + s.slice(1) : 'Unknown', color: '#8d99aa' }
export const kindStyle = (k) => KIND_STYLE[k] || { label: k || 'Target', long: k || 'target', color: '#e6edf5' }

// ---------------------------------------------------------------- version 9: rover health

// Health status colours (the hp bar), and part levels (the part indicators).
export const HEALTH = { ok: '#62d6a8', damaged: '#f3b24a', immobile: '#ff4d4d', wrecked: '#8a93a3' }
export const healthColor = (status, hp) => HEALTH[status] || (!Number.isFinite(hp) ? '#8d99aa' : hp >= 85 ? HEALTH.ok : hp >= 40 ? HEALTH.damaged : HEALTH.immobile)
export const partLevel = (v) => (!Number.isFinite(v) ? 'none' : v >= 70 ? 'ok' : v >= 30 ? 'warn' : v > 0 ? 'bad' : 'dead')
export const PARTS = [
  { key: 'drive_l', label: 'Left drive', short: 'L drive' },
  { key: 'drive_r', label: 'Right drive', short: 'R drive' },
  { key: 'lidar', label: 'LiDAR', short: 'LiDAR' },
  { key: 'camera', label: 'Camera', short: 'Camera' },
  { key: 'power', label: 'Power', short: 'Power' },
  { key: 'comms', label: 'Comms', short: 'Comms' },
]
export const partLabel = (k) => PARTS.find((p) => p.key === k)?.label || (k ? k[0].toUpperCase() + k.slice(1).replace(/_/g, ' ') : 'Part')
// Spare kinds at the lander (fleet `spares`).
export const SPARES = [
  { key: 'drive', label: 'Drive motor' },
  { key: 'lidar', label: 'LiDAR' },
  { key: 'camera', label: 'Camera' },
  { key: 'power', label: 'Power pack' },
]
// "tip_over" -> "tip over"; the sim's own words otherwise.
export const causeText = (c) => (c ? c.replace(/_/g, ' ') : 'unknown')
export const SEVERITIES = [
  { key: 'light', label: 'Light', hint: 'A knock: one part loses a little' },
  { key: 'heavy', label: 'Heavy', hint: 'A hard hit: one part badly damaged, the rover still drives' },
  { key: 'disable', label: 'Disable', hint: 'It can no longer drive: distress, a helper, a diagnosis' },
  { key: 'wreck', label: 'Wreck', hint: 'Beyond field repair: tow home or salvage' },
]
