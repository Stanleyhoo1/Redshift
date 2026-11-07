// Sun direction in three.js coordinates (x east, y up, z south).
export const SUN_DIR = [0.42, 0.86, 0.3]

// World (x east, y north, z up) to three.js (x, y up, z = -north).
export const w2t = (x, y, z = 0) => [x, z, -y]

export const fogYFor = (map) => (map ? map.minH - 0.8 : -4)

export const reducedMotion = typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
