/** Brain constants. Mirrored in ros_ws/src/disaster_brain/disaster_brain/config.py: keep in sync. */
export const CFG = {
  /** belief grid: 100 x 100 cells of 2 m over the 200 m map */
  G: 100, RES: 2, HALF: 100,
  /** confidence time constant: a cell sensed tau seconds ago has conf exp(-1) */
  confTau: 40,
  /** prediction horizon (s) */
  horizon: 30,
  /** how often the hazard map + predictions are recomputed (s) */
  mapPeriod: 0.5,
  maxV: 3.0, maxW: 1.6,
  ambient: 22,
  lidarMax: 30,
  thermalRange: 45,
  cameraRange: 28,
  // ---- safety reflex limits (rover hard limits: water 0.5 m, air 160 °C, tilt 34°, wind 46 m/s) ----
  reflexWater: 0.2,
  reflexTemp: 52,
  reflexTilt: 22,
  reflexWind: 26,
  reflexFront: 1.3,
  lowConfReturnFrac: 0.12,
  staleFrame: 0.6,
  // ---- perception ----
  hotTemp: 120,
  shakeThresh: 1.8,
  tornadoWind: 13,
  tornadoPressure: 35,
  // ---- behaviour selection ----
  hysteresis: 0.08,
  minDwell: 2.0,
  probeSpacing: 26,
  probeRuleSpacing: 20,
  probeRadius: 22,
  batteryLow: 25,
  batteryAgentEvent: 30,
  /** battery % used per metre driven + per second idle (estimates for the return-reserve rule) */
  batteryPerM: 0.075, batteryPerS: 0.1, batteryReserve: 12,
};

export const CH = { water: 0, heat: 1, rubble: 2, wind: 3, slope: 4 } as const;
export const CH_NAMES = ['water', 'heat', 'rubble', 'wind', 'slope'];
export const NCH = 5;
export const BEHAVIOURS = ['explore', 'flee', 'high_ground', 'hold_still', 'place_probe', 'return_to_safety'] as const;
export type Behaviour = typeof BEHAVIOURS[number];

export const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);
export const clamp01 = (v: number) => clamp(v, 0, 1);
export const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
const DIRS = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
/** compass word for a world-frame bearing (rad, 0 = +x = east) */
export const compass = (a: number) => DIRS[((Math.round(a / (Math.PI / 4)) % 8) + 8) % 8];
