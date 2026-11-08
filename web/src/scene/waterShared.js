// Shared between the water layer and the terrain (no React): the wet-shore mask, one texel
// per map sample (0 dry, 1 at the water's edge), from the water the fleet has identified.
// Written by Water.jsx, read by Terrain.jsx.
export const waterShared = { shore: null, map: null }
