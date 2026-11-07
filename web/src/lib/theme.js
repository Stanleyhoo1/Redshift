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
}

export const KIND_STYLE = {
  mineral: { label: 'Mineral', long: 'mineral deposit', color: '#b48cff' },
  ice: { label: 'Ice', long: 'ice patch', color: '#a8d4ff' },
  vent: { label: 'Vent', long: 'thermal vent', color: '#ff8a3d' },
}

export const stateStyle = (s) => STATE_STYLE[s] || { label: s ? s[0].toUpperCase() + s.slice(1) : 'Unknown', color: '#8d99aa' }
export const kindStyle = (k) => KIND_STYLE[k] || { label: k || 'Target', long: k || 'target', color: '#e6edf5' }
