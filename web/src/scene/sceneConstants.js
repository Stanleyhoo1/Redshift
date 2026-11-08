// Sun direction in three.js coordinates (x east, y up, z south): a low, warm key light from
// the west-north-west, about 34 degrees up, so relief reads from the default (southern)
// camera like a classic shaded-relief map (light from the upper left of the screen).
export const SUN_DIR = [-0.746, 0.559, -0.357]

// World (x east, y north, z up) to three.js (x, y up, z = -north).
export const w2t = (x, y, z = 0) => [x, z, -y]

export const fogYFor = (map) => (map ? map.minH - 0.8 : -4)

export const reducedMotion = typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// Shared, per-frame atmosphere state (no React): written by Atmosphere.jsx, read by the
// dust storm particles and the custom shaders.
//   dust: 0..1, how far an active dust storm has rolled in (eased).
export const atmo = { dust: 0 }
