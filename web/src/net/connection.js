import { appStore, dispatchSim, simStore } from '../state/store.js'

export function wsUrl() {
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`
}

// Python's json module writes NaN and Infinity as bare words, which JSON.parse
// rejects. Rather than drop the whole message, retry with those values as null.
export function parseMessage(text) {
  try {
    return JSON.parse(text)
  } catch {
    try {
      return JSON.parse(text.replace(/-?\bInfinity\b|\bNaN\b/g, 'null'))
    } catch {
      return null
    }
  }
}

// Opens the sim WebSocket and keeps it open (exponential backoff with jitter).
// Returns a function that closes it for good.
export function startConnection() {
  let ws = null
  let timer = null
  let attempt = 0
  let stopped = false

  const connect = () => {
    appStore.patch({ conn: attempt === 0 ? 'connecting' : 'reconnecting' })
    let sock
    try {
      sock = new WebSocket(wsUrl())
    } catch {
      scheduleReconnect()
      return
    }
    ws = sock
    sock.onopen = () => {
      attempt = 0
      appStore.patch({ conn: 'open' })
    }
    sock.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return
      const msg = parseMessage(ev.data)
      if (!msg) return
      dispatchSim(msg)
      if (msg && msg.type === 'init') loadTruth(simStore.get().gen)
    }
    sock.onclose = () => {
      if (ws !== sock) return
      ws = null
      if (!stopped) scheduleReconnect()
    }
    // onerror is always followed by onclose, which handles the retry.
    sock.onerror = () => {}
  }

  const scheduleReconnect = () => {
    appStore.patch({ conn: 'reconnecting' })
    const delay = Math.min(8000, 500 * 2 ** attempt) * (0.8 + Math.random() * 0.4)
    attempt = Math.min(attempt + 1, 10)
    clearTimeout(timer)
    timer = setTimeout(connect, delay)
  }

  connect()

  return () => {
    stopped = true
    clearTimeout(timer)
    const sock = ws
    ws = null
    if (!sock) return
    // Closing a socket that is still connecting logs a browser warning, so wait for it.
    if (sock.readyState === WebSocket.CONNECTING) sock.onopen = () => sock.close()
    else sock.close()
  }
}

// Ground truth for the Reality overlay, fetched once per init.
export async function loadTruth(gen) {
  try {
    const res = await fetch('/api/truth', { cache: 'no-store' })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const j = parseMessage(await res.text())
    if (!j) throw new Error('Truth is not valid JSON')
    if (simStore.get().gen !== gen) return // a newer init arrived meanwhile
    const nx = j.nx | 0
    const ny = j.ny | 0
    // The protocol says a flat row-major list; accept a list of rows as well.
    const heights = Array.isArray(j.heights) && Array.isArray(j.heights[0]) ? j.heights.flat() : j.heights
    if (!(nx > 1 && ny > 1) || !Array.isArray(heights) || heights.length < nx * ny) throw new Error('Unexpected truth shape')
    const origin = Array.isArray(j.origin) ? j.origin : [0, 0]
    appStore.patch({
      truth: {
        gen,
        nx,
        ny,
        res: +j.res || 0.5,
        ox: +origin[0] || 0,
        oy: +origin[1] || 0,
        heights: Float32Array.from(heights, (v) => (Number.isFinite(v) ? v : 0)),
        targets: Array.isArray(j.targets) ? j.targets : [],
      },
    })
  } catch (err) {
    if (simStore.get().gen === gen) appStore.patch({ truth: { gen, error: String(err?.message || err) } })
  }
}

export async function resetSim(seed) {
  const res = await fetch('/api/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed }),
  })
  if (!res.ok) throw new Error(`Reset failed (HTTP ${res.status})`)
  return res.json().catch(() => ({ ok: true, seed }))
}
