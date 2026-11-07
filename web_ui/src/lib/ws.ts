// Single WebSocket to the backend (same host, path /ws) with exponential-backoff reconnect.
import { useStore } from './store'
import type { SelectCameraMsg, ServerMsg } from './types'

const MIN_DELAY = 400
const MAX_DELAY = 8000
let dropped = 0 // unparsable frames (the backend sends strict JSON, so this should stay 0)

class WsClient {
  private sock: WebSocket | null = null
  private attempt = 0
  private timer: ReturnType<typeof setTimeout> | null = null
  private stopped = false

  start() {
    this.stopped = false
    this.connect()
    // keep the server's camera selection in sync with the UI
    useStore.subscribe((s, prev) => {
      if (s.selectedRobot && s.selectedRobot !== prev.selectedRobot) this.selectCamera(s.selectedRobot)
    })
  }

  stop() {
    this.stopped = true
    if (this.timer) clearTimeout(this.timer)
    this.sock?.close()
  }

  url(): string {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${location.host}/ws`
  }

  private connect() {
    if (this.stopped) return
    useStore.getState().setConn('connecting')
    let sock: WebSocket
    try {
      sock = new WebSocket(this.url())
    } catch {
      this.scheduleReconnect()
      return
    }
    this.sock = sock
    sock.onopen = () => {
      this.attempt = 0
      useStore.getState().setConn('open')
      const sel = useStore.getState().selectedRobot
      if (sel) this.selectCamera(sel)
    }
    sock.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return
      let msg: ServerMsg
      try {
        msg = JSON.parse(ev.data)
      } catch (e) {
        dropped += 1
        if (dropped === 1 || dropped % 100 === 0) console.warn(`[ws] dropped ${dropped} unparsable frame(s)`, e)
        return
      }
      try {
        useStore.getState().ingest(msg)
      } catch (e) {
        console.warn('[ws] failed to ingest', msg?.type, e)
      }
    }
    sock.onclose = () => {
      if (this.sock === sock) this.sock = null
      useStore.getState().setConn('closed')
      this.scheduleReconnect()
    }
    sock.onerror = () => sock.close()
  }

  private scheduleReconnect() {
    if (this.stopped || this.timer) return
    const base = Math.min(MAX_DELAY, MIN_DELAY * 2 ** this.attempt)
    const delay = base * (0.75 + Math.random() * 0.5)
    this.attempt++
    this.timer = setTimeout(() => {
      this.timer = null
      this.connect()
    }, delay)
  }

  send(obj: SelectCameraMsg) {
    if (this.sock?.readyState === WebSocket.OPEN) this.sock.send(JSON.stringify(obj))
  }

  selectCamera(robot: string) {
    this.send({ type: 'select_camera', robot })
  }
}

export const ws = new WsClient()
