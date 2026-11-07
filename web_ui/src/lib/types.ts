// Mirrors docs/protocol.md (v1). Keep in sync with that file — it is the source of truth.

export type XY = [number, number]
export type XYZ = [number, number, number]

export type RobotType = 'rover' | 'scout'
export type Severity = 'info' | 'warn' | 'critical'
export type SensorTag = 'CAM' | 'DEPTH' | 'LIDAR' | 'IMU' | 'ODOM' | 'MAP' | 'FLEET' | 'EDGE'
export type PerceptionMode = 'FUSED' | 'LIDAR_ONLY'
export type MissionStatus = 'idle' | 'launching' | 'running' | 'degraded' | 'failed' | 'stopped'

export type RoverState =
  | 'BOOT'
  | 'EXPLORING'
  | 'REPLANNING'
  | 'CAUTION'
  | 'SAFE_HOLD'
  | 'RECOVERY_REVERSE'
  | 'AWAIT_SCOUT'
  | 'RESUMING'
  | 'MISSION_COMPLETE'
export type ScoutState = 'BOOT' | 'TAKEOFF' | 'SURVEY' | 'FLY_TO_SCAN' | 'SCANNING' | 'MISSION_COMPLETE'
export type AgentState = RoverState | ScoutState

// §6 MissionConfig
export type SectorId = 'small' | 'medium' | 'large'
export type EdgeProfileId = 'orin_nano_15w' | 'orin_nx_25w' | 'agx_orin_60w' | 'rpi5_8w'
export type SimBackend = 'gazebo' | 'mock'

export interface MissionConfig {
  sector: SectorId
  gravity_g: number
  fleet: { rovers: number; drones: number }
  edge_profile: EdgeProfileId
  sim_backend: SimBackend
  seed: number
  gazebo_gui: boolean
  /** target simulated seconds per wall second; changeable live via POST /api/sim/speed */
  sim_speed: number
}

// §6 GET /api/options
export interface SectorOption {
  id: SectorId
  label: string
  size_m: number
  res_m: number
  cells: number
}
export interface RangeOption {
  min: number
  max: number
  default: number
}
export interface EdgeProfileOption {
  id: EdgeProfileId
  label: string
  device: string
  watts_idle: number
  watts_max: number
  perf_scale: number
  perception_hz_max: number
  max_expansions: number
}
export interface BackendOption {
  id: SimBackend
  label: string
  available: boolean
}
export interface Options {
  sectors: SectorOption[]
  gravity: RangeOption & { step: number }
  fleet: { rovers: RangeOption; drones: RangeOption }
  edge_profiles: EdgeProfileOption[]
  backends: BackendOption[]
  sim_speed: { choices: number[]; min: number; max: number; default: number }
  defaults: MissionConfig
}

// §6 GET /api/physics_preview
export type LimitingFactor = 'slip' | 'drive_torque' | 'tip'
export interface PhysicsPreview {
  g: number
  max_slope_deg: number
  factor: LimitingFactor
  drive_deg: number
  slip_deg: number
  tip_deg: number
  tip_alarm_deg: number
  v_max: number
  energy_j_per_m: number
}

export interface DeployResponse {
  ok: boolean
  mission_id?: string
  status?: MissionStatus
  detail?: string
}
export interface OkResponse {
  ok: boolean
  detail?: string
}
export type Checklist = Record<string, boolean>
export interface MissionStatusResponse {
  status: MissionStatus
  backend: SimBackend | string | null
  config: MissionConfig | null
  started_at: number | null
  checklist: Checklist
  detail: string | null
}

// §3.1 Decision trace
export interface TraceEntry {
  id: number
  t: number
  robot: string
  state: string
  kind: string
  sensors: SensorTag[]
  input: string
  reasoning: string
  confidence: number
  action: string
  severity: Severity
  latency_ms?: number | null
}

// §3.4 Edge telemetry
export interface EdgeTelemetry {
  robot: string
  profile: EdgeProfileId | string
  host_ms: { perception: number; planner: number; mapping: number }
  device_ms: number
  loop_hz: number
  load_pct: number
  power_w: number
  perception_hz: number
  throttle_level: number
  uncertainty: number
}

export interface Pose {
  x: number
  y: number
  z: number
  roll: number
  pitch: number
  yaw: number
}

export interface Uncertainty {
  score: number
  camera_conf: number
  lidar_conf: number
  imu_vib: number
}

// §7 telemetry.robots[r]
export interface RobotTelemetry {
  type: RobotType
  pose: Pose
  speed: number
  state: AgentState | string
  mode: PerceptionMode | string
  goal: XY | null
  v_limit: number
  slip: number
  tip_alarm_deg: number
  max_slope_deg: number
  altitude_agl: number | null
  imu: { roll_deg: number; pitch_deg: number; gyro: XYZ }
  uncertainty: Uncertainty
  edge: EdgeTelemetry
  path: XY[]
  trail: XY[]
  last_safe: XY | null
}

export interface MissionEnv {
  dust_level: number
  seismic: number | boolean
}

export interface MissionTelemetry {
  status: MissionStatus
  backend: SimBackend | string
  /** mission time since 'running', in simulated seconds */
  elapsed_s: number
  /** simulator clock (s); null until /clock is seen */
  sim_time_s?: number | null
  /** target speed (real-time factor) */
  sim_speed?: number
  /** achieved real-time factor over the last ~2 s; 0 when the clock has stalled */
  rtf?: number | null
  percent_mapped: number
  science_found: number
  hazard_cells: number
  env: MissionEnv
  config: MissionConfig | null
}

// /planet_map/stats
export interface MapStats {
  percent: number
  known_cells: number
  total_cells: number
  hazard_cells: number
  science: XY[]
  permanent_hazards: XYZ[]
  elev_min: number
  elev_max: number
}

export interface MapMeta {
  size_m: number
  res: number
  cells: number
  origin: XY
  elev_min: number
  elev_max: number
}

// §4 events + backend-originated fleet events
export interface RockslideEvent {
  op: 'spawn_rockslide'
  ok: boolean
  robot: string
  impact: XY
  rocks: XYZ[]
  t: number
}
export interface JoltEvent {
  op: 'jolt'
  ok: boolean
  robot: string
  t: number
}
export interface AnomalyEvent {
  op: 'anomaly'
  ok: boolean
  dust_level: number
  until_t: number
  t: number
}
export interface ScanRequestEvent {
  op: 'scan_request'
  id?: number | string
  robot: string
  xy: XY
  reason?: string
  t: number
}
export interface ScanCompleteEvent {
  op: 'scan_complete'
  id?: number | string
  scout: string
  xy: XY
  max_slope_deg?: number
  mean_slope_deg?: number
  verdict?: string
  t: number
}
export interface HazardMarkEvent {
  op: 'hazard_mark'
  robot: string
  xy: XY
  radius_m: number
  reason?: string
  t: number
}
export interface GenericEvent {
  op: string
  ok?: boolean
  t?: number
  [k: string]: unknown
}
export type WorldEvent =
  | RockslideEvent
  | JoltEvent
  | AnomalyEvent
  | ScanRequestEvent
  | ScanCompleteEvent
  | HazardMarkEvent
  | GenericEvent

// §7 WebSocket server → client
export interface StatusMsg {
  type: 'status'
  status: MissionStatus
  backend: SimBackend | string | null
  detail: string | null
  checklist: Checklist
  config: MissionConfig | null
}
export interface BootLogMsg {
  type: 'boot_log'
  line: string
  t: number
}
export interface TelemetryMsg {
  type: 'telemetry'
  t: number
  mission: MissionTelemetry
  robots: Record<string, RobotTelemetry>
}
export interface CameraMsg {
  type: 'camera'
  robot: string
  jpeg_b64: string
  t: number
}
export interface LidarMsg {
  type: 'lidar'
  robot: string
  points: XYZ[]
  max_range: number
  t: number
}
export interface MapMsg {
  type: 'map'
  png_b64: string
  meta: MapMeta
  stats: MapStats
}
export interface TraceMsg {
  type: 'trace'
  entry: TraceEntry
}
export interface EventMsg {
  type: 'event'
  event: WorldEvent
}
export type ServerMsg = StatusMsg | BootLogMsg | TelemetryMsg | CameraMsg | LidarMsg | MapMsg | TraceMsg | EventMsg

// client → server
export interface SelectCameraMsg {
  type: 'select_camera'
  robot: string
}

// Map class codes (§5, G channel = code·40)
export const CLS = {
  UNKNOWN: 0,
  FREE: 1,
  CAUTION: 2,
  HAZARD: 3,
  PERMANENT: 4,
  SCIENCE: 5,
  OBSTACLE: 6,
} as const
