import { appStore, dispatchSim, simStore } from '../state/store.js'
import { sanitizeDetections } from '../state/simState.js'

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
      if (msg && msg.type === 'init') {
        loadTruth(simStore.get().gen)
        // Request ids may start again after a reset: forget answers to the old ones.
        if (Object.keys(appStore.get().answers).length) appStore.patch({ answers: {} })
      }
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

// POST /api/hazard (PROTOCOL.md, version 2; `at` = [x, y] for version 4 point and click).
// Resolves with the server's hazard, or throws an Error carrying the server's own error
// text (HTTP 400) when it refuses.
// `extra`: more body fields, e.g. version 9 damage { severity, part } (null values dropped).
export async function postHazard(type, rover, at = null, extra = null) {
  const body = rover ? { type, rover } : { type }
  if (at) {
    body.x = at[0]
    body.y = at[1]
  }
  if (extra) for (const [k, v] of Object.entries(extra)) if (v != null) body[k] = v
  let res
  try {
    res = await fetch('/api/hazard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new Error('Could not reach the sim')
  }
  const j = parseMessage(await res.text().catch(() => '')) || {}
  if (!res.ok || j.ok === false) {
    if (res.status === 404 || res.status === 405) throw new Error('This sim does not support hazards yet')
    // FastAPI validation errors carry a list in `detail`.
    const detail = Array.isArray(j.detail) ? j.detail[0]?.msg : j.detail
    throw new Error(j.error || (typeof detail === 'string' ? detail : '') || `Hazard refused (HTTP ${res.status})`)
  }
  return j.hazard || null
}

// Shared by every camera view (sensor panel, clarification cards): when the sim says it
// cannot render, do not ask again for a while (each refused request is also a console
// line in the browser).
export const camDown = { until: 0, problem: null }

// The same URL with a cache-busting parameter (kept relative when it is same-origin).
export function freshUrl(url) {
  const u = new URL(url, location.href)
  u.searchParams.set('_', String(Date.now()))
  return u.origin === location.origin ? u.pathname + u.search : u.href
}

// One frame from GET /api/camera/{id}.jpg. Resolves with { blob, frame, ageMs, dets }:
// version 8 headers X-Frame (frame number), X-Frame-Age-Ms (how old the frame was when the
// sim sent it) and X-Detections (the objects detected in exactly that frame); each null when
// the sim does not send it. Throws an Error whose `status` is the HTTP status (0 for a
// network failure).
export async function fetchCameraFrame(id, view, w, h, signal) {
  const res = await fetchImageResponse(`/api/camera/${encodeURIComponent(id)}.jpg?view=${view}&w=${w}&h=${h}&_=${Date.now()}`, signal)
  const frame = Number.parseInt(res.headers.get('X-Frame') ?? '', 10)
  const age = Number.parseFloat(res.headers.get('X-Frame-Age-Ms') ?? '')
  const rawDets = res.headers.get('X-Detections')
  let dets = null
  if (rawDets != null) dets = sanitizeDetections(parseMessage(rawDets)) || []
  return { blob: await res.blob(), frame: Number.isFinite(frame) ? frame : null, ageMs: Number.isFinite(age) && age >= 0 ? age : null, dets }
}

export async function fetchImage(url, signal) {
  return (await fetchImageResponse(url, signal)).blob()
}

async function fetchImageResponse(url, signal) {
  let res
  try {
    res = await fetch(url, { cache: 'no-store', signal })
  } catch (err) {
    if (err?.name === 'AbortError') throw err
    throw Object.assign(new Error('No connection to the sim'), { status: 0 })
  }
  const type = res.headers.get('content-type') || ''
  if (!res.ok || !type.startsWith('image/')) {
    const j = parseMessage(await res.text().catch(() => '')) || {}
    throw Object.assign(new Error(j.error || j.detail || `HTTP ${res.status}`), { status: res.status })
  }
  return res
}

// POST helper for the v3 endpoints. Resolves with the JSON body, or throws an Error with
// the server's own error text and the HTTP `status` (0 when the sim is unreachable).
async function postJson(path, body, fallbackMsg) {
  let res
  try {
    res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  } catch {
    throw Object.assign(new Error('Could not reach the sim'), { status: 0 })
  }
  const j = parseMessage(await res.text().catch(() => '')) || {}
  if (!res.ok || j.ok === false) {
    const detail = Array.isArray(j.detail) ? j.detail[0]?.msg : j.detail
    let text = j.error || (typeof detail === 'string' ? detail : '')
    if (/^not found$/i.test(text)) text = '' // a route the sim does not have (older protocol)
    throw Object.assign(new Error(text || fallbackMsg(res.status)), { status: res.status, own: !!text })
  }
  return j
}

// POST /api/chat (PROTOCOL.md, version 3). The server echoes the message in `chat`.
export function postChat(text, rover) {
  return postJson('/api/chat', rover ? { text, rover } : { text }, (status) =>
    status === 404 || status === 405 ? 'This sim does not support operator chat yet' : `Message refused (HTTP ${status})`,
  )
}

// POST /api/answer: 404 unknown request, 409 already expired or answered.
export function postAnswer(id, option) {
  return postJson('/api/answer', { id, option }, (status) =>
    status === 409 ? 'Too late: this question already closed' : status === 404 ? 'The sim no longer has this question' : `Answer refused (HTTP ${status})`,
  )
}

// POST /api/mode (version 8): "autonomous" or "supervised". Resolves with the JSON body.
export function postMode(mode) {
  return postJson('/api/mode', { mode }, (status) => (status === 404 || status === 405 ? 'This sim has no autonomy modes' : `Mode change refused (HTTP ${status})`))
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
