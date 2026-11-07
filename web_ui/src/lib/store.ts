// Global client state. The WebSocket client (ws.ts) pushes every server message into `ingest`.
import { create } from 'zustand'
import { DEFAULT_OPTIONS } from './defaults'
import { clearHistory, notePathUpdate, noteFrame, pushImu, pushSensors } from './history'
import { decodeMap, type DecodedMap } from './mapDecode'
import { clearDistressClock, type Received } from './rescue'
import { sampleOf } from './sensors'
import type {
  BootLogMsg,
  CameraKind,
  LidarMsg,
  MapMsg,
  MissionConfig,
  Options,
  RescueEvent,
  ServerMsg,
  SosEvent,
  StatusMsg,
  TelemetryMsg,
  TraceEntry,
  WorldEvent,
} from './types'

export type Screen = 'prelaunch' | 'boot' | 'mission'
export type ConnState = 'connecting' | 'open' | 'closed'

export interface CameraFrame {
  jpeg_b64: string
  t: number
  receivedAt: number
  fps: number
}
export type RobotCameras = Partial<Record<CameraKind, CameraFrame>>
export interface EventRecord {
  seq: number
  receivedAt: number
  event: WorldEvent
}
export interface LidarFrame extends LidarMsg {
  receivedAt: number
}

export const TRACE_CAP = 200
const BOOT_CAP = 600
const EVENT_CAP = 60

interface State {
  screen: Screen
  conn: ConnState
  options: Options
  optionsLive: boolean
  status: StatusMsg | null
  bootLog: BootLogMsg[]
  telemetry: TelemetryMsg | null
  telemetryAt: number
  /** latest frame per robot and kind (§8.9: rgb for every robot, depth for the selected one) */
  cameras: Record<string, RobotCameras>
  /** depth vs rgb in the main camera view */
  cameraKind: CameraKind
  /** latest `sos` event per SOS id */
  sos: Record<string, Received<SosEvent>>
  /** latest `rescue` event per SOS id */
  rescues: Record<string, Received<RescueEvent>>
  lidar: Record<string, LidarFrame>
  map: DecodedMap | null
  traces: TraceEntry[]
  events: EventRecord[]
  selectedRobot: string | null
  followRobot: boolean
  lastConfig: MissionConfig | null

  setScreen: (s: Screen) => void
  setConn: (c: ConnState) => void
  setOptions: (o: Options, live: boolean) => void
  selectRobot: (r: string) => void
  setFollow: (f: boolean) => void
  setCameraKind: (k: CameraKind) => void
  beginDeploy: (cfg: MissionConfig) => void
  resetMission: () => void
  ingest: (msg: ServerMsg) => void
}

let eventSeq = 0
let mapBusy = false
let mapPending: MapMsg | null = null

export const useStore = create<State>()((set, get) => ({
  screen: 'prelaunch',
  conn: 'connecting',
  options: DEFAULT_OPTIONS,
  optionsLive: false,
  status: null,
  bootLog: [],
  telemetry: null,
  telemetryAt: 0,
  cameras: {},
  cameraKind: 'rgb',
  sos: {},
  rescues: {},
  lidar: {},
  map: null,
  traces: [],
  events: [],
  selectedRobot: null,
  followRobot: false,
  lastConfig: null,

  setScreen: (screen) => set({ screen }),
  setConn: (conn) => set({ conn }),
  setOptions: (options, optionsLive) => set({ options, optionsLive }),
  selectRobot: (selectedRobot) => set({ selectedRobot }),
  setFollow: (followRobot) => set({ followRobot }),
  setCameraKind: (cameraKind) => set({ cameraKind }),

  beginDeploy: (cfg) => {
    get().resetMission()
    const st = get().status
    set({
      lastConfig: cfg,
      screen: 'boot',
      // optimistic until the server's first status frame arrives
      status: { type: 'status', status: 'launching', backend: cfg.sim_backend, detail: null, checklist: st?.status === 'launching' ? st.checklist : {}, config: cfg },
    })
  },

  resetMission: () => {
    clearHistory()
    clearDistressClock()
    set({
      bootLog: [],
      telemetry: null,
      telemetryAt: 0,
      cameras: {},
      sos: {},
      rescues: {},
      lidar: {},
      map: null,
      traces: [],
      events: [],
    })
  },

  ingest: (msg) => {
    switch (msg.type) {
      case 'status': {
        const prev = get().status
        set({ status: msg })
        routeOnStatus(msg, prev)
        break
      }
      case 'boot_log': {
        const log = get().bootLog
        const next = log.length >= BOOT_CAP ? log.slice(log.length - BOOT_CAP + 1) : log.slice()
        next.push(msg)
        set({ bootLog: next })
        break
      }
      case 'telemetry': {
        const prev = get().telemetry
        const now = performance.now()
        for (const [id, r] of Object.entries(msg.robots)) {
          pushImu(id, { t: msg.t, roll: r.imu.roll_deg, pitch: r.imu.pitch_deg })
          pushSensors(id, sampleOf(r, msg.t))
          notePathUpdate(id, prev?.robots[id]?.path, r.path, [r.pose.x, r.pose.y], now)
        }
        const patch: Partial<State> = { telemetry: msg, telemetryAt: now }
        const sel = get().selectedRobot
        if (!sel || !msg.robots[sel]) {
          const ids = sortRobots(Object.keys(msg.robots))
          if (ids.length) patch.selectedRobot = ids[0]
        }
        set(patch)
        break
      }
      case 'camera': {
        const now = performance.now()
        const kind: CameraKind = msg.kind === 'depth' ? 'depth' : 'rgb'
        const fps = noteFrame(`${msg.robot}:${kind}`, now)
        const cams = get().cameras
        set({ cameras: { ...cams, [msg.robot]: { ...cams[msg.robot], [kind]: { jpeg_b64: msg.jpeg_b64, t: msg.t, receivedAt: now, fps } } } })
        break
      }
      case 'lidar':
        set({ lidar: { ...get().lidar, [msg.robot]: { ...msg, receivedAt: performance.now() } } })
        break
      case 'map':
        queueMap(msg)
        break
      case 'trace': {
        const traces = get().traces
        if (traces.length && traces[0].id === msg.entry.id && traces[0].t === msg.entry.t) break
        const next = [msg.entry, ...traces]
        if (next.length > TRACE_CAP) next.length = TRACE_CAP
        set({ traces: next })
        break
      }
      case 'event': {
        const ev = msg.event
        const now = performance.now()
        // sos / rescue repeat at 1 Hz while active: keep only the latest per SOS id, out of the event log
        if (ev.op === 'sos' && typeof (ev as SosEvent).robot === 'string') {
          const e = ev as SosEvent
          set({ sos: { ...get().sos, [e.id ?? e.robot]: { ev: e, receivedAt: now } } })
          break
        }
        if (ev.op === 'rescue' && typeof (ev as RescueEvent).target === 'string') {
          const e = ev as RescueEvent
          set({ rescues: { ...get().rescues, [e.id ?? e.target]: { ev: e, receivedAt: now } } })
          break
        }
        const next = [...get().events, { seq: ++eventSeq, receivedAt: now, event: ev }]
        if (next.length > EVENT_CAP) next.splice(0, next.length - EVENT_CAP)
        set({ events: next })
        break
      }
    }
  },
}))

function queueMap(msg: MapMsg) {
  if (mapBusy) {
    mapPending = msg // latest wins
    return
  }
  mapBusy = true
  decodeMap(msg.png_b64, msg.meta, msg.stats)
    .then((map) => {
      const old = useStore.getState().map
      useStore.setState({ map })
      // keep the old bitmap alive one frame so the texture upload never sees a closed bitmap
      if (old) setTimeout(() => old.bitmap.close?.(), 1000)
    })
    .catch((e) => console.warn('[map] decode failed', e))
    .finally(() => {
      mapBusy = false
      const next = mapPending
      mapPending = null
      if (next) queueMap(next)
    })
}

/** Screen routing driven by mission status (handles reloads mid-mission too). */
function routeOnStatus(msg: StatusMsg, prev: StatusMsg | null) {
  const { screen, setScreen } = useStore.getState()
  const live = msg.status === 'launching' || msg.status === 'running' || msg.status === 'degraded'
  if (screen === 'prelaunch' && live && (!prev || prev.status !== msg.status)) {
    if (msg.config) useStore.setState({ lastConfig: msg.config })
    setScreen('boot')
  }
  if (screen === 'mission' && (msg.status === 'stopped' || msg.status === 'idle')) {
    setScreen('prelaunch')
  }
}

export function sortRobots(ids: string[]): string[] {
  const rank = (id: string) => (id.startsWith('rover') ? 0 : id.startsWith('scout') ? 1 : 2)
  return [...ids].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
}

export const isLiveStatus = (s: string | undefined | null) => s === 'running' || s === 'degraded'
