// Version 9 rescues (fleet `rescues`): who is helping whom, and plain words for each phase.
// Shared by the rover cards, the sensor panel and the map tags.

export const V9_STATES = new Set(['disabled', 'rescuing', 'towing', 'towed', 'repairing', 'wrecked'])

// The open rescue of a rover in trouble, or the one a helper works on (null when none).
export const rescueOf = (rescues, id) => rescues.find((r) => r.rover === id && r.phase !== 'done') || null
export const helpingOf = (rescues, id) => rescues.find((r) => r.helper === id && r.phase !== 'done') || null

const pctText = (p) => (Number.isFinite(p) ? ` (${Math.round(p * 100)}%)` : '')

// One line for the rover card / sensor panel. `name(id)` gives a display name, `dist(id)`
// the distance in metres between this rover and another (or null).
export function rescueLine(rover, rescues, name, dist, asking) {
  if (!rover) return null
  const st = rover.state
  if (st === 'disabled') {
    const rs = rescueOf(rescues, rover.id)
    if (!rs || !rs.helper) return rover.health?.status === 'wrecked' ? 'Wrecked, waiting for help' : 'Immobile, waiting for help'
    const h = name(rs.helper)
    if (rs.phase === 'en_route') {
      const d = dist(rs.helper)
      return `${h} on the way${d !== null ? `, ${Math.round(d)} m` : ''}`
    }
    if (rs.phase === 'diagnosing') return asking ? `Diagnosed by ${h}: your decision` : `${h} is diagnosing it${pctText(rs.progress)}`
    if (rs.phase === 'repairing') return `${h} is fixing it in the field${pctText(rs.progress)}`
    if (rs.phase === 'salvaging') return `${h} is salvaging its parts${pctText(rs.progress)}`
    return `${h} is helping`
  }
  if (st === 'rescuing' || st === 'towing') {
    const rs = helpingOf(rescues, rover.id)
    const who = rs ? name(rs.rover) : 'a rover'
    if (st === 'towing') return `Towing ${who} to the lander${pctText(rs?.progress)}`
    if (!rs || rs.phase === 'en_route') {
      const d = rs ? dist(rs.rover) : null
      return `On the way to help ${who}${d !== null ? `, ${Math.round(d)} m` : ''}`
    }
    if (rs.phase === 'diagnosing') return asking ? `Diagnosed ${who}: asking you tow, salvage or leave` : `Diagnosing ${who}${pctText(rs.progress)}`
    if (rs.phase === 'repairing') return `Fixing ${who} in the field${pctText(rs.progress)}`
    if (rs.phase === 'salvaging') return `Salvaging parts from ${who}${pctText(rs.progress)}`
    return `Helping ${who}`
  }
  if (st === 'towed') {
    const rs = rescueOf(rescues, rover.id)
    return rs?.helper ? `In tow behind ${name(rs.helper)}${pctText(rs.progress)}` : 'In tow to the lander'
  }
  if (st === 'repairing') {
    const rs = rescueOf(rescues, rover.id)
    return `Being repaired at the lander${pctText(rs?.progress)}`
  }
  if (st === 'wrecked') return `Wreck left at (${Math.round(rover.x)}, ${Math.round(rover.y)})`
  return null
}

// The map tag at a rover in trouble: title and sub line (null when there is nothing to say).
export function distressTag(rover, rescues, name, dist, asking) {
  const rs = rescueOf(rescues, rover.id)
  const who = name(rover.id)
  if (rs?.helper && rs.phase !== 'en_route') {
    const h = name(rs.helper)
    if (rs.phase === 'diagnosing') return { kind: 'diag', title: asking ? `${who}: your decision` : `${h} diagnosing ${who}`, sub: asking ? 'Tow, salvage or leave? See the request' : null, progress: asking ? null : rs.progress, busy: !asking }
    if (rs.phase === 'repairing' && rover.state !== 'repairing') return { kind: 'fix', title: `Field repair on ${who}`, sub: `by ${h}`, progress: rs.progress, busy: true }
    if (rs.phase === 'salvaging') return { kind: 'salvage', title: `Salvaging ${who}`, sub: `by ${h}`, progress: rs.progress, busy: true }
    return null
  }
  if (rover.state !== 'disabled') return null
  const wreck = rover.health?.status === 'wrecked'
  if (!rs?.helper) return { kind: 'sos', title: `${who} needs help`, sub: wreck ? 'Wrecked. Waiting for a helper' : 'Distress sent. Waiting for a helper', progress: null, busy: false }
  const d = dist(rs.helper)
  return { kind: 'sos', title: `${who} needs help`, sub: `${name(rs.helper)} on the way${d !== null ? ` · ${Math.round(d)} m` : ''}`, progress: null, busy: false }
}
