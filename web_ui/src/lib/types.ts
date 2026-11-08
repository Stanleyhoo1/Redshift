// Mirrors docs/protocol.md (v1 + §8 fleet health / SOS / rescue + the advisory Gemma brain). Keep in sync with that
// file — it is the source of truth. Every §8 / brain field is optional at runtime: older backends omit it.

export type XY = [number, number]
export type XYZ = [number, number, number]

export type RobotType = 'rover' | 'scout'
export type Severity = 'info' | 'warn' | 'critical'
/**
 * DIAG = motor-controller diagnostics (§8.3 hw_status). GEMMA / LOCAL tag `brain_advice` traces: which brain decided
 * (the cloud Gemma advisor, or the deterministic on-board rules).
 */
export type SensorTag = 'CAM' | 'DEPTH' | 'LIDAR' | 'IMU' | 'ODOM' | 'MAP' | 'FLEET' | 'EDGE' | 'DIAG' | 'GEMMA' | 'LOCAL'
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
  // §8.6: damaged rover
  | 'AWAIT_RESCUE'
  // §8.6: recovery rover
  | 'RESCUE_TRANSIT'
  | 'RESCUE_DOCKING'
  | 'REPAIRING'
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
  /** advisory Gemma brain on/off (the launch passes it to fleet_brain_node as `enabled`); absent on older backends */
  ai_brain?: boolean
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
/** GET /api/options `brain`: never carries the key itself, only whether one is configured. */
export interface BrainOption {
  model: string
  key_present: boolean
  default_enabled: boolean
}
export interface Options {
  sectors: SectorOption[]
  gravity: RangeOption & { step: number }
  fleet: { rovers: RangeOption; drones: RangeOption }
  edge_profiles: EdgeProfileOption[]
  backends: BackendOption[]
  sim_speed: { choices: number[]; min: number; max: number; default: number }
  defaults: MissionConfig
  /** absent on older backends (and in the offline fallback options) */
  brain?: BrainOption
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
  /** `brain_advice` traces only: the decision behind the row */
  brain?: BrainTraceInfo
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

// §8.1 roles
export type RobotRole = 'scout' | 'recovery' | 'drone'

// §8.7 health block (rovers and drone)
export interface RobotHealth {
  /** structure_pct × drive_pct / 100 */
  pct: number
  structure_pct: number
  drive_pct: number
  battery_pct: number
  battery_wh: number
  capacity_wh: number
  solar_w: number
  load_w: number
  /** solar_w − load_w; negative = draining */
  net_w: number
  faults: string[]
  /** pct < 15 */
  critical: boolean
  shocks: number
  peak_g: number
}

// §8.4 / §8.7
export type RescuePhase = 'TRANSIT' | 'DOCKING' | 'REPAIRING' | 'DONE' | 'ABORTED'
export interface RescueBlock {
  id: string
  /** 'rescuer' on the recovery rover; the damaged rover carries the other side of the link */
  role: 'rescuer' | 'target' | string
  /** the other robot; null on the damaged rover until a rescuer has taken its SOS */
  peer: string | null
  phase: RescuePhase | string
  range_m: number | null
  locked: boolean
}

// §8.7 sensors block: the raw instrument readings behind the agent's decisions
export interface RobotSensors {
  /** m/s² in the IMU frame */
  accel: XYZ | null
  /** |accel| / 9.80665 */
  accel_g: number | null
  /** rad/s */
  gyro: XYZ | null
  roll_deg: number | null
  pitch_deg: number | null
  /** null on the drone */
  lidar_nearest_m: number | null
  lidar_returns_pct: number | null
  camera_conf: number | null
  v_cmd: number | null
  v_wheel: number | null
  v_ground: number | null
  slip: number | null
  /** drone only */
  altitude_agl: number | null
}

// Advisory brain blocks in agent_state (forwarded in telemetry.robots[r])
/** A speed cap the rover applied from a /brain/decision: v = min(local, cap) on follow / dock drives only. */
export interface BrainCap {
  speed_mps: number
  action: BrainAction | string
  source: BrainSource | string
  /** true when the cap came from an operator command */
  operator: boolean
  /** simulated seconds until the cap expires */
  remaining_s: number
  /**
   * true while the rover's drive is actually limited by the cap (it governs path-following / docking only); false when
   * it is set but the current motion is one it does not touch. Absent on older backends: the HUD then never claims it.
   */
  applied?: boolean
}
/** Scout cruise altitude set by ADJUST_DRONE_ALTITUDE (clamped 6–20 m by the scout's own safety logic). */
export interface BrainAgl {
  agl_m: number
  remaining_s: number
  /** true while the scout flies at it (surveying); false until then (take-off / scan altitude unchanged) */
  applied?: boolean
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
  // §8.7 (absent on older backends; health / rescue are null before the first agent_state)
  role?: RobotRole | string
  name?: string
  health?: RobotHealth | null
  rescue?: RescueBlock | null
  sensors?: RobotSensors | null
  /** rovers: the active advisory speed cap, null when none */
  brain_cap?: BrainCap | null
  /** scout: the active advisory cruise altitude, null when none */
  brain_agl?: BrainAgl | null
}

// §3.2 + §8.8 detections (not forwarded on /ws today; documented for completeness)
export type DetectionClass = 'ROCK' | 'CRATER' | 'SCIENCE' | 'SLOPE' | 'ROVER'
export interface Detection {
  cls: DetectionClass | string
  conf: number
  bbox: [number, number, number, number]
  range_m: number
  bearing_rad: number
  world_xy: XY
  sources: SensorTag[]
  /** ROVER only: which teammate, whether it is damaged, and the lock-on label */
  robot?: string
  damaged?: boolean
  label?: string
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
  /** latest /brain/status (1 Hz); null before the brain node reports or when none runs */
  brain?: BrainStatus | null
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
// §8.4 / §8.5 / §8.9
export interface SosEvent {
  op: 'sos'
  id: string
  robot: string
  xy: XY
  yaw?: number
  health_pct: number
  faults: string[]
  status: 'active' | 'resolved'
  t: number
}
export interface RescueEvent {
  op: 'rescue'
  id: string
  rescuer: string
  target: string
  phase: RescuePhase | string
  range_m: number | null
  locked: boolean
  t: number
}
export interface CriticalDamageEvent {
  op: 'critical_damage'
  ok: boolean
  robot: string
  impact: XY
  peak_g: number
  wheel_health: number[]
  t: number
}
export interface RepairEvent {
  op: 'repair'
  ok: boolean
  robot: string
  by: string
  distance_m: number
  detail?: string
  /** true on the second event, when health is restored */
  done?: boolean
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
  | SosEvent
  | RescueEvent
  | CriticalDamageEvent
  | RepairEvent
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
export type CameraKind = 'rgb' | 'depth'
export interface CameraMsg {
  type: 'camera'
  robot: string
  /** §8.9: absent = rgb. Depth is colour-mapped (turbo, near = red) and sent for the selected robot only. */
  kind?: CameraKind
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
/** every /brain/decision, pushed as soon as the backend sees it */
export interface BrainMsg {
  type: 'brain'
  decision: BrainDecision
}
export type ServerMsg = StatusMsg | BootLogMsg | TelemetryMsg | CameraMsg | LidarMsg | MapMsg | TraceMsg | EventMsg | BrainMsg

// client → server
export interface SelectCameraMsg {
  type: 'select_camera'
  robot: string
}

// Advisory Gemma brain (fleet_brain_node). Gemma only ever makes things safer; the local FSM / safety layer keeps authority.
export type BrainEvent = 'novel_hazard' | 'low_confidence' | 'critical_damage' | 'peer_sos' | 'operator_command'
export type BrainAction =
  | 'SLOW_AND_REPLAN'
  | 'REQUEST_CLARIFICATION'
  | 'DEPLOY_RECOVERY_ROVER'
  | 'ADJUST_DRONE_ALTITUDE'
  | 'REJECT_UNSAFE_COMMAND'
/** which brain chose the action: the cloud model (validated) or the deterministic on-board rules */
export type BrainSource = 'gemma' | 'local-fallback'

/** gemma_agent_brain.assess_operator_command: the local, deterministic safety reading of operator text. */
export interface OperatorAssessment {
  intent: 'stop' | 'slow' | 'set_speed' | 'faster' | 'drone_altitude' | 'enter_hazard' | 'unclear' | string
  /** null when unbounded ("faster" without a top speed; the backend turns ∞ into null) */
  requested_speed_mps: number | null
  requested_altitude_m: number | null
  hazard: string | null
  exceeds_limit: boolean
  unsafe: boolean
  reasons: string[]
}

/** /brain/decision: Decision.as_dict() plus the node's routing fields. */
export interface BrainDecision {
  action: BrainAction | string
  /** speed cap (m/s); null = leave the local plan's speed alone */
  target_speed_mps: number | null
  /** drone AGL (m), only with ADJUST_DRONE_ALTITUDE */
  target_altitude_m: number | null
  reasoning_trace: string
  source: BrainSource | string
  /** model that produced the action (API modelVersion) or "local-rules" */
  model: string
  event: BrainEvent | string
  /** MEASURED host → Google AI cloud round-trip; null when no cloud call completed */
  latency_ms: number | null
  tokens_in: number | null
  tokens_out: number | null
  /** roofline ESTIMATE on projected_device; null when the model does not fit */
  projected_edge_ms: number | null
  projected_device: string
  projected_fits: boolean
  /** honest one-line description of the projection, shown verbatim */
  projected_note: string
  /** why Gemma's answer could not be used (null when it was) */
  error: string | null
  /** every clamp / override the local layer applied */
  adjusted: string[]
  /** recovery rover to deploy / drone to move */
  target_robot: string | null
  robot: string | null
  request_id: string | null
  stale: boolean
  age_ms: number | null
  /** wall epoch s */
  t: number
  /** trace kind that raised the event, or "operator" */
  trigger: string
  trigger_input: string
  command_id: string | null
  command_text: string | null
  /** operator commands only */
  local: OperatorAssessment | null
  /** the instant local decision on an operator STOP; Gemma's later answer for the same command replaces it */
  immediate: boolean
}

/** AgentBrain.stats() */
export interface BrainStats {
  submitted: number
  deduped: number
  cloud_calls: number
  /** answers whose action survived validation */
  gemma: number
  /** local-fallback decisions */
  local: number
  rate_limited: number
  stale: number
  callback_errors: number
  in_flight: number
  backoff_remaining_s: number
}

/** /brain/status (1 Hz) */
export interface BrainStatus {
  enabled: boolean
  key_present: boolean
  /** configured cloud model */
  model: string
  /** edge profile id used for the on-device projection */
  device: string
  stats: Partial<BrainStats>
  backoff_s: number
  last_error: string | null
  t: number
  /** set by the backend when the brain node stopped reporting: the other fields are its last report, not now */
  stale?: boolean
}

/** `brain` key of a `brain_advice` trace entry */
export interface BrainTraceInfo {
  source: BrainSource | string
  model: string
  event: BrainEvent | string
  action: BrainAction | string
  target_speed_mps: number | null
  target_altitude_m: number | null
  latency_ms: number | null
  tokens_in: number | null
  tokens_out: number | null
  projected_edge_ms: number | null
  projected_device: string
  projected_fits: boolean
  projected_note: string
  error: string | null
  adjusted: string[]
  command_id: string | null
}

/** POST /api/command */
export interface CommandResponse {
  ok: boolean
  id: string
  detail?: string
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
