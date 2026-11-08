// Lifeform kinds (PROTOCOL.md version 7, "More lifeforms"): looks, legend text and the small
// pure helpers the instanced models use. Plain data and maths, no three.js, so node tests can
// load it. Grazers keep their own model in Life.jsx; the other kinds are drawn by LifeKinds.jsx.

export const KIND_LOOK = {
  grazer: { label: 'Grazers', color: '#c8f25a', hint: 'Six-legged herds that wander and graze' },
  skitterer: { label: 'Skitterers', color: '#f0a23c', hint: 'Small fast swarms that scatter from a rover within about 8 m' },
  drifter: { label: 'Drifters', color: '#7fe6f0', glow: '#f08ad6', hint: 'Glowing bells that float 1.5 to 3.5 m up and drift with the wind' },
  crawler: { label: 'Crawlers', color: '#d8d0b8', shell: '#2a2731', hint: 'Big slow shelled creatures. Rovers keep 6 m away' },
  burrower: { label: 'Burrowers', color: '#c48aa8', mound: '#6e645b', hint: 'Pop up from the ground for a few seconds, then sink' },
}

// Kinds this module draws (everything but grazers).
export const EXTRA_KINDS = ['skitterer', 'drifter', 'crawler', 'burrower']

// Which kinds to list in the legend: a kind shows once a creature of it is noticed (a live
// fleet track near it), or at once with Reality on. Returns kind ids in a stable order.
export function legendKinds(lifeforms, tracks, reality, near = 3) {
  const out = new Set()
  if (!Array.isArray(lifeforms)) return []
  for (const l of lifeforms) {
    const kind = l && typeof l.kind === 'string' ? l.kind : 'grazer'
    if (!KIND_LOOK[kind] || out.has(kind)) continue
    if (reality) {
      out.add(kind)
      continue
    }
    for (const k of tracks || []) {
      if (k.status === 'tracking' && Math.hypot(k.x - l.x, k.y - l.y) < near) {
        out.add(kind)
        break
      }
    }
  }
  return Object.keys(KIND_LOOK).filter((k) => out.has(k))
}

// A burrower's head height above its mound, 0 (down) to 1 (all the way up), eased so it
// pops up quickly and settles. `up` comes from the sim (0 to 1); missing means up.
export function burrowLift(up) {
  const u = Number.isFinite(up) ? Math.min(1, Math.max(0, up)) : 1
  return u * u * (3 - 2 * u)
}

// Skitterer gait: a quick, slightly irregular tripod step. Returns the leg swing (-1 to 1)
// for a leg with this phase offset, and a small body jitter in metres per metre of size.
export function skitterGait(phase, legOffset) {
  const p = phase * 2.6 + legOffset
  const swing = Math.sin(p) * 0.8 + Math.sin(p * 2.3 + 1.1) * 0.2
  return { swing, jitter: 0.02 * Math.sin(phase * 7.1) * Math.sin(phase * 3.3 + 0.7) }
}

// Drifter bell pulse: the bell squeezes and relaxes slowly (radius scale 0.9 to 1.06) and
// bobs a little; tendrils sway. t in seconds, seed per creature.
export function drifterPulse(t, seed) {
  const s = Math.sin(t * 1.3 + seed)
  return { squeeze: 0.98 + 0.08 * s, bob: 0.06 * Math.sin(t * 0.7 + seed * 1.7), sway: 0.18 * Math.sin(t * 0.9 + seed) }
}
