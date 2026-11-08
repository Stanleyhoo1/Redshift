// Built-in fallbacks so PreLaunch renders even when /api/options or /api/physics_preview are
// unreachable. Values match docs/protocol.md §6 and the physics model in the build plan.
import type { MissionConfig, Options, PhysicsPreview, LimitingFactor } from './types'

export const DEFAULT_CONFIG: MissionConfig = {
  sector: 'medium',
  gravity_g: 2.2,
  fleet: { rovers: 2, drones: 1 },
  edge_profile: 'orin_nano_15w',
  sim_backend: 'gazebo',
  seed: 42,
  gazebo_gui: false,
  sim_speed: 1.0,
  ai_brain: true,
}

export const DEFAULT_OPTIONS: Options = {
  sectors: [
    { id: 'small', label: 'Small', size_m: 50, res_m: 0.5, cells: 100 },
    { id: 'medium', label: 'Medium', size_m: 100, res_m: 0.5, cells: 200 },
    { id: 'large', label: 'Large', size_m: 200, res_m: 1.0, cells: 200 },
  ],
  gravity: { min: 1.0, max: 3.0, step: 0.1, default: 2.2 },
  fleet: { rovers: { min: 1, max: 2, default: 2 }, drones: { min: 0, max: 1, default: 1 } },
  edge_profiles: [
    { id: 'orin_nano_15w', label: 'Orin Nano 15 W', device: 'NVIDIA Jetson Orin Nano', watts_idle: 5, watts_max: 15, perf_scale: 2.6, perception_hz_max: 10, max_expansions: 60000 },
    { id: 'orin_nx_25w', label: 'Orin NX 25 W', device: 'NVIDIA Jetson Orin NX', watts_idle: 7, watts_max: 25, perf_scale: 1.6, perception_hz_max: 15, max_expansions: 120000 },
    { id: 'agx_orin_60w', label: 'AGX Orin 60 W', device: 'NVIDIA Jetson AGX Orin', watts_idle: 15, watts_max: 60, perf_scale: 0.9, perception_hz_max: 20, max_expansions: 250000 },
    { id: 'rpi5_8w', label: 'Raspberry Pi 5', device: 'Raspberry Pi 5 (8 GB)', watts_idle: 3, watts_max: 8, perf_scale: 5.0, perception_hz_max: 5, max_expansions: 30000 },
  ],
  backends: [
    { id: 'gazebo', label: 'Gazebo Harmonic', available: true },
    { id: 'mock', label: 'Mock physics', available: true },
  ],
  sim_speed: { choices: [0.5, 1.0, 1.5, 2.0], min: 0.25, max: 3.0, default: 1.0 },
  defaults: DEFAULT_CONFIG,
}

// Rover physics (plan §"Physics & autonomy models"): 60 kg, F_max 416 N, μ = tan 32°, cohesion 150 N,
// tip angle atan(0.425 / 0.35).
const G0 = 9.81
const MASS = 60
const F_MAX = 416
const DEG = 180 / Math.PI

export function localPhysicsPreview(g: number): PhysicsPreview {
  const weight = MASS * g * G0
  const drive = Math.asin(Math.min(1, F_MAX / weight)) * DEG
  const slip = Math.atan(Math.tan(32 / DEG) + 150 / weight) * DEG
  const tip = Math.atan(0.425 / 0.35) * DEG
  const lim = Math.min(drive, slip, tip)
  const factor: LimitingFactor = lim === drive ? 'drive_torque' : lim === slip ? 'slip' : 'tip'
  const max = 0.9 * lim
  return {
    g,
    max_slope_deg: max,
    factor,
    drive_deg: drive,
    slip_deg: slip,
    tip_deg: tip,
    tip_alarm_deg: Math.min(0.7 * tip, max + 12),
    v_max: Math.min(1, Math.sqrt(1.5 / g)),
    energy_j_per_m: (weight * 0.08) / 0.75,
  }
}
