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
}

// Environment changes injected into the sim (PROTOCOL.md version 2). Crimson, so a
// hazard never reads as a rover warning (amber) or a discovery (cyan).
export const HAZARD = '#ff3b5c'
export const SAND = '#e2bf7d'
export const DUST = '#b98a5c'
// `place`: how the button works with a version 4 sim. 'point' = click the map (x, y),
// 'rover' = click a rover, null = one click. `radius` sizes the ghost marker (metres).
// `v4`: only offered when the sim speaks version 4. `discovery`: styled as a find, not a hazard.
export const HAZARD_TYPES = [
  { type: 'rockfall', label: 'Rockfall', place: 'point', radius: 2.8, hint: 'Boulders land on a rover route. LiDAR sees them, the planner steers around.' },
  { type: 'sand', label: 'Soft sand', place: 'point', radius: 2.3, hint: 'Hidden soft ground on a route. Only wheel slip gives it away.' },
  { type: 'spire_fall', label: 'Falling spire', place: 'point', radius: 0.8, v4: true, hint: 'A crystal spire topples across a route. LiDAR sees it, the planner steers around.' },
  { type: 'dust_storm', label: 'Dust storm', place: null, hint: 'LiDAR range collapses to a few metres for a while. Rovers slow down.' },
  { type: 'rover_fault', label: 'Motor fault', place: 'rover', hint: 'One rover loses its motors and reboots. The fleet routes around it.' },
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

export const stateStyle = (s) => STATE_STYLE[s] || { label: s ? s[0].toUpperCase() + s.slice(1) : 'Unknown', color: '#8d99aa' }
export const kindStyle = (k) => KIND_STYLE[k] || { label: k || 'Target', long: k || 'target', color: '#e6edf5' }
