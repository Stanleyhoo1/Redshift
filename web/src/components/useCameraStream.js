import { useEffect, useRef, useState } from 'react'
import { camDown, fetchCameraFrame } from '../net/connection.js'
import { useNow } from '../lib/useNow.js'

const RETRY_NET = 4000 // no connection: try again soon
export const RETRY_DOWN = 30000 // 503 / 404: rendering is off on that machine, rarely comes back
const DUP_WAIT = 40 // ms: the sim had no newer frame yet, ask again a moment later
const RATE_WINDOW = 2000 // ms of frames counted for the fps readout
const RING = 64

export function cameraProblem(err) {
  if (err.status === 404) return { title: 'Camera unavailable', sub: 'This rover has no camera feed' }
  if (err.status === 0) return { title: 'Camera unavailable', sub: 'No connection to the sim' }
  if (err.status === 503) return { title: 'Camera unavailable', sub: err.message && !/^HTTP/.test(err.message) ? err.message : 'Rendering is off on this machine' }
  return { title: 'Camera unavailable', sub: err.message || 'The sim did not send a picture' }
}

// Decoded off the main thread where the browser can (createImageBitmap), else through an
// object URL. Either way the frame is fully decoded before it is shown: no blank flashes.
async function decodeImage(blob) {
  if (typeof createImageBitmap === 'function') {
    try {
      return await createImageBitmap(blob)
    } catch {
      // fall through to the <img> path (some browsers refuse certain encodings here)
    }
  }
  const url = URL.createObjectURL(blob)
  try {
    const img = new Image()
    img.src = url
    await img.decode()
    return img
  } finally {
    // The decoded image keeps its pixels; the URL is not needed once decode() resolved.
    URL.revokeObjectURL(url)
  }
}

export function closeImage(image) {
  if (image && typeof image.close === 'function') image.close()
}

const isHidden = () => typeof document !== 'undefined' && document.visibilityState === 'hidden'

// Streams GET /api/camera/{id}.jpg while enabled (and the page is visible). Exactly one
// request is in flight: the next one starts as soon as the previous frame has arrived and
// decoded (version 8 sims answer at once with their newest frame), no sooner than `minGap`
// ms after the previous start, and a little later when the sim had nothing newer (same
// X-Frame). Each new frame goes to `onFrame({ image, w, h, no, dets, ageMs })`, which draws
// it and from then on owns `image` (close it with closeImage when replaced). Disabled or
// hidden: polling stops and the caller keeps its last frame.
// Returns { hasFrame, natural: { w, h } | null, problem, waiting, retry, stats } where
// `stats` is a ref read by useCamRate (fps, frame age, detections).
export function useCameraStream({ roverId, view, w, h, minGap = 40, enabled = true, onFrame }) {
  const [hasFrame, setHasFrame] = useState(false)
  const [natural, setNatural] = useState(null)
  const [problem, setProblem] = useState(() => (Date.now() < camDown.until ? camDown.problem : null))
  const [waiting, setWaiting] = useState(true)
  const [attempt, setAttempt] = useState(0)
  const [hidden, setHidden] = useState(isHidden)
  const stats = useRef({ times: new Float64Array(RING), head: 0, count: 0, at: 0, ageMs: null, frame: null, dets: null })
  const cb = useRef(onFrame)
  cb.current = onFrame

  useEffect(() => {
    const onVis = () => setHidden(isHidden())
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [])

  useEffect(() => {
    if (!enabled || !roverId || hidden) return
    let stopped = false
    let timer = null
    let ctrl = null
    const S = stats.current
    S.frame = null // frame numbers belong to one rover, view and size
    const loop = async () => {
      if (stopped) return
      const down = camDown.until - Date.now()
      if (down > 0) {
        setProblem(camDown.problem)
        setWaiting(false)
        timer = setTimeout(loop, down)
        return
      }
      const started = performance.now()
      ctrl = new AbortController()
      try {
        const r = await fetchCameraFrame(roverId, view, w, h, ctrl.signal)
        if (stopped) return
        if (r.frame !== null && r.frame === S.frame) {
          timer = setTimeout(loop, DUP_WAIT) // the same picture again: nothing to decode
          return
        }
        let image
        try {
          image = await decodeImage(r.blob)
        } catch {
          throw Object.assign(new Error('The picture could not be decoded'), { status: -1 })
        }
        if (stopped) {
          closeImage(image)
          return
        }
        const now = performance.now()
        const iw = image.naturalWidth || image.width
        const ih = image.naturalHeight || image.height
        S.frame = r.frame
        S.at = now
        S.ageMs = (r.ageMs ?? 0) + (now - started)
        S.dets = r.dets
        S.times[S.head] = now
        S.head = (S.head + 1) % RING
        S.count = Math.min(RING, S.count + 1)
        cb.current?.({ image, w: iw, h: ih, no: r.frame, dets: r.dets, ageMs: S.ageMs })
        setHasFrame(true)
        setNatural((n) => (n && n.w === iw && n.h === ih ? n : { w: iw, h: ih }))
        setProblem(null)
        setWaiting(false)
        timer = setTimeout(loop, Math.max(0, minGap - (performance.now() - started)))
      } catch (err) {
        if (stopped || err?.name === 'AbortError') return
        const p = cameraProblem(err)
        setProblem(p)
        setWaiting(false)
        const isDown = err.status === 503 || err.status === 404
        if (isDown) {
          camDown.until = Date.now() + RETRY_DOWN
          camDown.problem = p
        }
        timer = setTimeout(loop, isDown ? RETRY_DOWN : RETRY_NET)
      }
    }
    loop()
    return () => {
      stopped = true
      clearTimeout(timer)
      ctrl?.abort()
    }
  }, [roverId, view, w, h, minGap, enabled, attempt, hidden])

  const retry = () => {
    camDown.until = 0
    setProblem(null)
    setWaiting(true)
    setAttempt((n) => n + 1)
  }
  return { hasFrame, natural, problem, waiting, retry, stats, paused: hidden || !enabled }
}

// Frames per second over the last two seconds, the age of the newest frame (ms, as the sim
// reported it plus the trip here), seconds since it arrived, and its detections (null when
// the sim sends none). Re-renders twice a second, not per frame.
export function useCamRate(stats, active = true) {
  useNow(500, active)
  const S = stats.current
  const now = performance.now()
  let n = 0
  for (let k = 0; k < S.count; k++) if (now - S.times[k] <= RATE_WINDOW) n++
  return {
    fps: (n * 1000) / RATE_WINDOW,
    ageMs: S.ageMs,
    since: S.at ? (now - S.at) / 1000 : null,
    dets: S.dets,
  }
}

// "14 fps · 90 ms"
export function fmtRate(r) {
  if (!r || r.since === null) return ''
  if (r.since > 3) return `no new frame for ${Math.round(r.since)} s`
  const fps = r.fps >= 9.5 ? Math.round(r.fps) : Math.round(r.fps * 10) / 10
  const age = r.ageMs === null ? '' : r.ageMs < 1000 ? ` · ${Math.round(r.ageMs)} ms` : ` · ${(r.ageMs / 1000).toFixed(1)} s`
  return `${fps} fps${age}`
}
