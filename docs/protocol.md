# Redshift Data Contracts (v1)

Single source of truth for every interface between Gazebo/mock world, ROS 2 nodes, the
FastAPI backend and the web UI. If code disagrees with this file, fix the code (or update
this file in the same commit).

## 1. Naming & frames

- Robots: `rover_0`, `rover_1` (type `rover`), `scout_0` (type `scout`). Gazebo world name: `redshift`.
- World frame: x east, y north, z up, metres. Sector spans x, y ∈ [−L/2, L/2] (L = sector size).
- Global grid: origin (−L/2, −L/2), resolution `res`; cell (row i, col j) covers
  x ∈ [−L/2 + j·res, …+res), y ∈ [−L/2 + i·res, …+res). **Row index grows with +y.**
- Robot body frame: x forward, y left, z up. Angles in radians on ROS topics, degrees only in
  human-readable strings and fields explicitly suffixed `_deg`.

## 2. ROS 2 topics

Per robot (`/<r>/…`):

| Topic | Type | Producer → consumer |
|---|---|---|
| `cmd_vel` | geometry_msgs/Twist | planner → world |
| `odom` | nav_msgs/Odometry (world-frame pose) | world → all |
| `wheel_odom` | nav_msgs/Odometry (rover only, wheel-encoder twist) | world → planner (slip) |
| `imu_raw` → `imu` | sensor_msgs/Imu | world → sensor_effects → perception/planner |
| `camera/image_raw` → `camera/image` | sensor_msgs/Image rgb8 | world → sensor_effects → perception |
| `camera/depth_raw` → `camera/depth` | sensor_msgs/Image 32FC1 (m, NaN/inf = no return) | world → sensor_effects → perception/mapping |
| `camera/camera_info` | sensor_msgs/CameraInfo | world → perception/mapping |
| `lidar/points_raw` → `lidar/points` | sensor_msgs/PointCloud2 (x,y,z float32, sensor frame) | world → sensor_effects → perception/mapping |
| `perception/annotated/compressed` | sensor_msgs/CompressedImage (jpeg) | perception → backend |
| `perception/detections` | std_msgs/String (JSON §3.2) | perception → planner/mapping/backend |
| `perception/uncertainty` | std_msgs/Float32 (0 = certain, 1 = blind) | perception → planner/backend |
| `perception/scan2d` | sensor_msgs/LaserScan (body frame, near-horizontal slice) | perception → backend |
| `perception/mode_request` | std_msgs/String `FUSED` \| `LIDAR_ONLY` | planner → perception |
| `planned_path` | nav_msgs/Path (frame `world`) | planner → backend/world |
| `agent_state` | std_msgs/String (JSON §3.3) | planner → backend |
| `edge_telemetry` | std_msgs/String (JSON §3.4) | planner → backend |

Global:

| Topic | Type | Notes |
|---|---|---|
| `/planet_map` | nav_msgs/OccupancyGrid | −1 unknown, 0 free, 50 caution, 100 hazard/obstacle |
| `/planet_map/elevation` | sensor_msgs/Image 32FC1 | NaN = unknown, row = grid row |
| `/planet_map/layers` | sensor_msgs/Image rgb8 | PNG layer encoding §5 |
| `/planet_map/stats` | std_msgs/String JSON | `{percent, known_cells, total_cells, hazard_cells, science:[[x,y]], permanent_hazards:[[x,y,r]], elev_min, elev_max}` |
| `/agent_decision_trace` | std_msgs/String JSON §3.1 | all robots |
| `/fleet/claims` | std_msgs/String | `{robot, goal:[x,y], t}` |
| `/fleet/hazard_marks` | std_msgs/String | `{robot, xy:[x,y], radius_m, reason, t}` |
| `/fleet/scan_request` | std_msgs/String | `{id, robot, xy:[x,y], reason, t}` |
| `/fleet/scan_complete` | std_msgs/String | `{id, scout, xy, max_slope_deg, mean_slope_deg, verdict, t}` |
| `/world/command` | std_msgs/String | §4 |
| `/world/events` | std_msgs/String | §4 |
| `/world/env_state` | std_msgs/String | `{dust_level, seismic, remaining_s}` (`remaining_s` = simulated s of dust left, null when clear) |
| `/clock` | rosgraph_msgs/Clock | simulated time, ≤100 Hz (every 0.01 s of sim time): `world_interface_node` (republished from Gazebo via `/gz/clock`) or `mock_world_node` |
| `/gz/clock` | rosgraph_msgs/Clock | Gazebo backend only: gz-sim's per-step clock (250 Hz at 1×) via ros_gz_bridge → `world_interface_node` |

### 2.1 Simulated time

Every autonomy node and `sensor_effects_node` runs with `use_sim_time: true`: timers, hold/settle durations, dedup
windows and sensor stamps follow `/clock`, so the agents behave identically at any simulation speed. The two clock
sources keep their own clocks: Gazebo (paced by its physics `real_time_factor`) and the mock world (a lockstep clock:
time advances at `speed` × wall time but waits for any sensor frame still rendering, so sensor rates hold per
simulated second and an overloaded renderer lowers the achieved real-time factor instead of dropping frames).
Human-facing `t` fields in JSON payloads (traces, events, fleet messages) stay **wall-clock epoch** seconds, because
the UI shows them as times of day.

## 3. JSON payloads

### 3.1 Decision trace
```json
{"id":17,"t":1762550000.12,"robot":"rover_0","state":"REPLANNING","kind":"path_blocked",
 "sensors":["LIDAR","CAM"],"input":"Boulder field 6.8 m ahead across path (5 objects)",
 "reasoning":"Corridor occupancy 0.82; step height 0.9 m > 0.30 m clearance","confidence":0.91,
 "action":"Re-planning A* around obstacle (+7.4 m, 1412 nodes, 38 ms)","severity":"warn","latency_ms":41.7}
```
`severity` ∈ `info | warn | critical`. `sensors` ⊆ `CAM, DEPTH, LIDAR, IMU, ODOM, MAP, FLEET, EDGE`.
`kind` examples: `mission_start, frontier_selected, path_planned, path_blocked, replanned, slope_hazard,
science_target, caution_enter, caution_exit, safe_hold, recovery_reverse, hazard_marked, scout_tasked,
scan_complete, resume, edge_throttle, mission_complete`.

### 3.2 Detections
```json
{"robot":"rover_0","t":0.0,"mode":"FUSED",
 "detections":[{"cls":"ROCK","conf":0.88,"bbox":[x,y,w,h],"range_m":6.2,"bearing_rad":0.1,
                "world_xy":[3.1,-2.0],"sources":["CAM","DEPTH"]}],
 "corridor":{"obstacle_dist_m":6.4,"slope_ahead_deg":11.2,"conf":0.9,"points":42},
 "uncertainty":{"score":0.14,"camera_conf":0.91,"lidar_conf":0.95,"imu_vib":0.05},
 "stage_ms":{"quality":1.2,"detect":9.8,"lidar":2.1,"annotate":1.5}}
```
`cls` ∈ `ROCK | CRATER | SCIENCE | SLOPE`.

### 3.3 Agent state
```json
{"robot":"rover_0","type":"rover","state":"EXPLORING","mode":"FUSED","goal":[12.0,4.5],
 "speed":0.62,"v_limit":0.83,"tilt":{"roll_deg":2.1,"pitch_deg":-4.0},"slip":0.04,
 "breadcrumbs":38,"last_safe":[1.0,2.0],"tip_alarm_deg":28.8,"max_slope_deg":16.8,"altitude_agl":null}
```
Rover states: `BOOT, EXPLORING, REPLANNING, CAUTION, SAFE_HOLD, RECOVERY_REVERSE, AWAIT_SCOUT, RESUMING, MISSION_COMPLETE`.
The tip/slip permanent hazard is a 2.5 m disc on the failure point once `RECOVERY_REVERSE` has backed out of it. If the
reverse times out short (traced as such, not as "reached"), the disc shrinks to stay 0.6 m clear of the rover (min
1.0 m), or a 1.0 m disc is placed just beyond the rover towards the failure point: a rover inside its own lethal zone
could not plan out of it and no rescuer could reach it. A route never halts on cells it was planned past on purpose
(escaping an inflation ring, or a 0.3 m tight-clearance route); only cells that became lethal since do.
Scout states: `BOOT, TAKEOFF, SURVEY, FLY_TO_SCAN, SCANNING, MISSION_COMPLETE`.

### 3.4 Edge telemetry
```json
{"robot":"rover_0","profile":"orin_nano_15w","host_ms":{"perception":11.2,"planner":3.4,"mapping":6.0},
 "device_ms":53.7,"loop_hz":9.8,"load_pct":61.0,"power_w":11.1,"perception_hz":10.0,
 "throttle_level":0,"uncertainty":0.14}
```

## 4. World commands & events (`/world/command`, `/world/events`)

Commands (published by the backend; executed by `world_interface_node` (Gazebo) or `mock_world_node`;
`anomaly`/`clear_anomaly` are executed by `sensor_effects_node` in both modes):
```json
{"op":"spawn_rockslide","robot":"rover_0"}
{"op":"jolt","robot":"rover_0"}
{"op":"anomaly","dust_level":0.85,"duration_s":25,"seismic_s":4}
{"op":"clear_anomaly"}
{"op":"set_speed","speed":2.0}
```
`set_speed` changes the target real-time factor live (clamped to 0.1–4): Gazebo via `/world/<w>/set_physics`
(`max_step_size` unchanged), the mock via its pacer. `duration_s` / `seismic_s` are simulated seconds.
`spawn_rockslide` drops a ~7 m-wide line of boulders across the robot's planned path (`scene_events.rockslide_layout`):
7 m ahead along the route, or 60 % of the remaining route when that is shorter (never under 3 m), so the line never
lands beyond the path's end. With under 4 m of route left (the rover is about to reach its goal) the slide is armed
and waits, up to 8 simulated seconds, for the rover's next planned route and lands across that
(`scene_events.rockslide_due`; the event then carries `waited_s`); after that, or with no path, along the heading.
A second request while one is armed is ignored.

Events (one per executed command):
```json
{"op":"spawn_rockslide","ok":true,"robot":"rover_0","impact":[x,y],"rocks":[[x,y,r]],"waited_s":0.0,"t":0.0}
{"op":"jolt","ok":true,"robot":"rover_0","t":0.0}
{"op":"anomaly","ok":true,"dust_level":0.85,"duration_s":25,"t":0.0}
{"op":"set_speed","ok":true,"speed":2.0,"t":0.0}
```

## 5. Map layer image (`/planet_map/layers`, WS `map.png_b64`)

RGB8, height = width = grid cells, **pixel row i = grid row i (row 0 = southmost, y = −L/2)**.
- **R** elevation: `round(255·(z − elev_min)/(elev_max − elev_min))`, 0 where unknown.
- **G** class: `code·40` with codes `0 unknown, 1 free, 2 caution, 3 hazard (slope/step), 4 permanent hazard,
  5 science target, 6 obstacle (rock)`. Decode with `round(G/40)`.
- **B** reveal age: 255 when a cell is first observed, decays linearly to 0 over 6 s (UI glow).

## 6. Backend REST (`http://localhost:8000`)

- `GET /api/options` →
  `{sectors:[{id,label,size_m,res_m,cells}], gravity:{min,max,step,default}, fleet:{rovers:{min,max,default}, drones:{min,max,default}},
    edge_profiles:[{id,label,device,watts_idle,watts_max,perf_scale,perception_hz_max,max_expansions}],
    backends:[{id,label,available}], sim_speed:{choices:[0.5,1,1.5,2], min:0.25, max:3, default:1}, defaults:MissionConfig}`
- `GET /api/physics_preview?g=2.2` →
  `{g, max_slope_deg, factor:"slip"|"drive_torque"|"tip", drive_deg, slip_deg, tip_deg, tip_alarm_deg, v_max, energy_j_per_m}`
- `POST /api/mission/deploy` body MissionConfig → `{ok, mission_id, status}`
- `POST /api/mission/abort` → `{ok}`
- `GET /api/mission/status` → `{status, backend, config, started_at, checklist:{name:bool}, detail}`
- `POST /api/inject/rockslide` `{robot?}` · `POST /api/inject/anomaly` `{dust_level?, duration_s?}` ·
  `POST /api/inject/tip` `{robot?}` → `{ok, detail}`
- `POST /api/sim/speed` `{speed}` (0.25–3) → `{ok, speed, detail}` — live speed change of the running mission (also
  recorded in the mission config that telemetry and status report); `{ok:false}` when no mission is running.

**MissionConfig**
```json
{"sector":"medium","gravity_g":2.2,"fleet":{"rovers":1,"drones":1},
 "edge_profile":"orin_nano_15w","sim_backend":"gazebo","seed":42,"gazebo_gui":false,"sim_speed":1.0}
```
sector ∈ `small (50 m @ 0.5 m) | medium (100 m @ 0.5 m) | large (200 m @ 1.0 m)`;
edge_profile ∈ `orin_nano_15w | orin_nx_25w | agx_orin_60w | rpi5_8w`; sim_backend ∈ `gazebo | mock`;
sim_speed ∈ 0.25–3.0 = target simulated seconds per wall second (the world's `real_time_factor`). Measured on the
dev machine with 1 rover + 1 scout and the full autonomy stack: Gazebo holds 2× (1.8–2.05), the mock world tops out
near 1.1–1.5× (its numpy sensor rendering is CPU/GIL-bound); the achieved factor is reported as `rtf`.

Mission status values: `idle | launching | running | degraded | failed | stopped`. A mission that is `degraded`
only because a subsystem was late returns to `running` when it arrives.

Checklist readiness (first arrival per mission): `gazebo`/`bridge`/`mock_world` = any robot's odom ·
`sensor_effects` = a clean `/<r>/camera/image` frame from **every** robot · `perception` = detections from every robot ·
`mapping` = `/planet_map/stats` · `<robot>` = its odom + agent_state. On the Gazebo backend the watchdog falls back to the
mock world if `sensor_effects` is still missing after 45 s, or if the launch (or `gz_sim` / the bridge / the world
interface) exits while launching.

Errors: an invalid body → HTTP 422 `{ok:false, detail:"field: message; …", errors:[pydantic errors]}`. A state-changing
request (POST) or a `/ws` upgrade whose `Origin` is neither localhost nor this server's own origin → 403 / close 1008.

## 7. WebSocket `/ws`

Server → client (JSON text frames):
- `{"type":"status", status, backend, detail, checklist:{gazebo,bridge,mapping,rover_0,scout_0,...}, config}` — on change and 1 Hz while launching.
- `{"type":"boot_log","line":"…","t":…}`
- `{"type":"telemetry","t":…,"mission":{status, backend, elapsed_s, sim_time_s, sim_speed, rtf, percent_mapped, science_found, hazard_cells, env:{dust_level, seismic}, config},
   "robots":{"rover_0":{type, pose:{x,y,z,roll,pitch,yaw}, speed, state, mode, goal, v_limit, slip, tip_alarm_deg, max_slope_deg,
   altitude_agl, imu:{roll_deg,pitch_deg,gyro:[x,y,z]}, uncertainty:{score,camera_conf,lidar_conf,imu_vib}, edge:{…§3.4},
   path:[[x,y],…≤150], trail:[[x,y],…≤300], last_safe:[x,y]|null}}}` — 15 Hz.
  `elapsed_s` = mission time since `running` in **simulated** seconds (wall seconds until `/clock` is seen);
  `sim_time_s` = simulator clock; `sim_speed` = target; `rtf` = achieved real-time factor over ~2 s (null until
  measured, 0 when the clock has stalled).
- `{"type":"camera","robot":"rover_0","jpeg_b64":"…","t":…}` — ≤10 Hz, only the client's selected robot.
- `{"type":"lidar","robot":"rover_0","points":[[x,y,z],…≤1500],"max_range":25,"t":…}` — 5 Hz, body frame, selected robot.
- `{"type":"map","png_b64":"…","meta":{size_m,res,cells,origin:[x0,y0],elev_min,elev_max},"stats":{…/planet_map/stats}}` — 2–4 Hz.
- `{"type":"trace","entry":{…§3.1}}` — immediately.
- `{"type":"event","event":{…§4 event, or {op:"scan_request"|"scan_complete"|"hazard_mark"|"fallback", …}}}` — immediately.

Client → server: `{"type":"select_camera","robot":"scout_0"}`.

Frames are strict JSON: non-finite numbers are sent as `null`. Each client has its own bounded queue; a client more than
240 frames behind, or whose socket blocks a send for 2 s, is closed with code 1013 and is expected to reconnect.

## 8. Fleet health, SOS and rescue (v2, Phase 1)

### 8.1 Fleet, roles and display names
Default fleet is **2 rovers + 1 drone** (`fleet.rovers` 1–2, default 2; `fleet.drones` 0–1, default 1).

| id | role | display name | short label |
|---|---|---|---|
| `rover_0` | `scout` | Scout Rover 1 | `ROVER_1` |
| `rover_1` | `recovery` | Heavy Recovery Rover 2 | `ROVER_2` |
| `scout_0` | `drone` | Survey Drone | `DRONE` |

Both rovers use `ROVER_SPEC` (identical physics). `role` comes from the index: `rover_0` = scout, any other rover = recovery.

### 8.2 New ROS topics
| topic | type | publisher | rate (sim) | payload |
|---|---|---|---|---|
| `/<rover>/hw_status` | String JSON | world (mock_world_node / world_interface_node) | 5 Hz | §8.3 |
| `/<rover>/cmd_vel_hw` | Twist | world_interface_node (Gazebo only) | as cmd_vel | `/<rover>/cmd_vel` scaled by drivetrain health, bridged to gz `/<rover>/cmd_vel` |
| `/<r>/perception/depth/compressed` | CompressedImage (jpeg) | perception_node | perception rate | colour-mapped depth (turbo, near = red), same size as the RGB frame |
| `/fleet/sos` | String JSON | damaged rover agent | 1 Hz while active, once on resolve | §8.4 |
| `/fleet/rescue` | String JSON | recovery rover agent | on phase change + 1 Hz | §8.4 |

The drivetrain fault is applied by the world *before* the wheels (mock: inside the dynamics; Gazebo: the
`cmd_vel` → `cmd_vel_hw` relay), so the rover only learns about it from its sensors: `hw_status` (motor-controller
diagnostics) and commanded vs. measured wheel speed.

### 8.3 `hw_status` (simulated motor controllers + solar panel)
```json
{"robot":"rover_0","t":1762550000.1,"wheel_health":[1.0,1.0,1.0,1.0],"faults":[],
 "motor_current_a":[3.1,3.0,3.2,3.1],"solar_w":182.4,"irradiance":0.93}
```
`wheel_health` 0–1 per wheel (FL, FR, RL, RR): fraction of commanded torque the actuator delivers.
`faults` e.g. `"DRIVE_FL_OVERCURRENT"`, `"DRIVE_RL_ENCODER_LOST"`. `irradiance` = dust transmittance
`1 − 0.85·dust_level`. `solar_w` = 220 W peak × irradiance × max(0, n̂·ŝ), where n̂ is the panel normal from the chassis
attitude and ŝ is the sun direction (55° elevation, the rendered sun's azimuth). A level rover gets ≈ 180 W.
`motor_current_a` is computed from the raw commanded `cmd_vel`, before the fault is applied: a faulty wheel draws
current with little output. `t` is wall-clock epoch time (§2.1). In Gazebo, rovers move only while
`world_interface_node` relays `cmd_vel` → `cmd_vel_hw`.

### 8.4 SOS / rescue payloads
```json
{"id":"sos-rover_0-1","robot":"rover_0","xy":[12.3,-4.0],"yaw":1.2,"health_pct":7.5,"faults":["DRIVE_FL_OVERCURRENT"],
 "status":"active","t":1762550000.1}
{"id":"sos-rover_0-1","rescuer":"rover_1","target":"rover_0","phase":"DOCKING","range_m":2.4,"locked":true,"t":…}
```
`status`: `active` | `resolved`. `phase`: `TRANSIT` | `DOCKING` | `REPAIRING` | `DONE` | `ABORTED`. Every message,
including the final `DONE` / `ABORTED` one, carries the SOS `id` and `target`; after `DONE` / `ABORTED` the damaged
rover reports its rescue block as `AWAITING` again (no rescuer) while its SOS is still active.

### 8.5 World commands / events (additions to §4)
- `{"op":"critical_damage","robot":"rover_0"}` — debris strike plus a drivetrain fault (wheel_health →
  `[0.08, 0.35, 0.05, 0.30]`, faults `DRIVE_FL_OVERCURRENT`, `DRIVE_FR_TORQUE_DERATED`, `DRIVE_RL_ENCODER_LOST`,
  `DRIVE_RR_TORQUE_DERATED`). The strike reads ≈ 5.1 g above gravity on the IMU in both backends:
  - mock: a 0.12 s vertical + lateral half-sine and a 5° roll kick (gyro ≈ 0.6 rad/s, below the tip and gyro gates);
  - Gazebo: 40 ms of force along the chassis normal (92 % of the weight) plus a lateral push through the centre of mass
    that reverses half-way (net Δv ≈ 0), so no wheel lifts. A hop would read random 5–30 g touchdown spikes, because
    DART resolves the landing in one 4 ms step. No roll kick. Persistent wrenches accumulate in gz-sim, so the
    mid-pulse reversal is published as a delta.

  Event: `{"op":"critical_damage","ok":true,"robot":"rover_0","impact":[x,y],"peak_g":5.11,"wheel_health":[…],"t":…}`.
  `peak_g` = measured |a| peak / g0 − mission gravity. Rejections: unknown or non-rover robot, a strike already
  under way, and (Gazebo only) no odometry yet or a jolt under way.
- `{"op":"repair","robot":"rover_0","by":"rover_1"}` — the world checks the true centre distance between the
  two robots is ≤ 1.6 m; if so it ramps wheel_health back to 1.0 over 4 s (sim) and clears faults. Event:
  `{"op":"repair","ok":true|false,"robot":"rover_0","by":"rover_1","distance_m":1.34,"detail":"…","t":…}`
  (sent when the repair starts with `"done":false`, and again with `"done":true` plus `wheel_health` when health is
  restored).
  - The distance is planar.
  - `by` must be a different rover.
  - A repeat request during a repair answers `ok:true`, "repair already in progress".
  - A new strike cancels a repair, and no done event follows.
  - Faults clear only when the repair completes.

### 8.6 Rover FSM additions
Damaged rover: `… → SAFE_HOLD (critical, 1.0 s) → AWAIT_RESCUE` (SOS broadcast) `→ RESUMING → EXPLORING` once its
estimated health is ≥ 60 % again. Recovery rover: `EXPLORING → RESCUE_TRANSIT` (A* to a stand-off cell 2.0–2.45 m
from the teammate, preferring 2.2 m on the side facing the approach; the teammate is planned as an uninflated 1.5 m
keep-out, and cells outside the rescuer's reachable region are skipped) `→ RESCUE_DOCKING` (perception-locked creep to ≤ 1.5 m centre distance) `→ REPAIRING → RESUMING → EXPLORING`.
With one rover there is no rescuer: the damaged rover stays in `AWAIT_RESCUE` and says so in its trace.
New trace kinds: `critical_damage`, `sos`, `rescue_start`, `rescue_route`, `rescue_dock`, `rescue_lock`, `docked`,
`repair_start`, `repair_complete`, `repaired`, `rescue_abort`, `rescue_noroute` (no reachable stand-off on
the current map). New trace sensor tag `DIAG` (motor-controller diagnostics).

Docking is entered when the A* stand-off is reached within 3.0 m (or anywhere within 2.4 m); during transit a LiDAR
corridor return closer than 4 m to the target is the target itself and does not trigger a re-plan. Docked = centre
distance ≤ 1.45 m with a perception lock (or after 5 s without one, traced as odometry-only). The rescuer actuates the
repair with `/world/command {"op":"repair"}` (the simulated repair tool), retries every 8 s, and once the target
reports its SOS resolved backs off 2 m and resumes exploring. The docking creep stops while a LiDAR corridor return is
more than 0.9 m short of the target's centre (something between the two rovers); the corridor ignores returns within
1.0 m of the sensor, so the last ≈0.45 m of the creep relies on the perception lock and odometry. Teammates are lethal 0.9 m discs in each rover's
planning cost map (not in the shared map, which filters robot returns).

### 8.7 Agent state / telemetry additions
`agent_state` (rovers) and WS telemetry `robots.<r>` gain:
```json
"role":"scout", "name":"Scout Rover 1",
"health":{"pct":92.0,"structure_pct":96.0,"drive_pct":96.0,"battery_pct":81.4,"battery_wh":488.4,"capacity_wh":600,
          "solar_w":182.4,"load_w":131.0,"net_w":51.4,"faults":[],"critical":false,"shocks":2,"peak_g":1.9,
          "last_shock_g":0.4},
"rescue":{"id":"sos-rover_0-1","role":"rescuer","peer":"rover_0","phase":"DOCKING","range_m":2.4,"locked":true} | null,
"sensors":{"accel":[0.1,-0.2,21.6],"accel_g":2.20,"gyro":[0.0,0.01,0.2],"roll_deg":1.2,"pitch_deg":-3.4,
           "lidar_nearest_m":4.2,"lidar_returns_pct":78.0,"camera_conf":0.97,"v_cmd":0.8,"v_wheel":0.79,
           "v_ground":0.76,"slip":0.04,"altitude_agl":null}
```
Drone: `health` has `structure_pct = drive_pct = 100`, its battery drains with hover power, `solar_w = 0`;
`sensors.lidar_* = null`, `altitude_agl` set. `health`, `rescue` may be `null` before the first agent_state.
`accel` is m/s² in the IMU frame; `accel_g` = |accel| / 9.80665 (≈ the mission gravity at rest).

Health model (rover, `redshift_autonomy/rover_health.py`): `pct = structure_pct × drive_pct / 100`. Structure
loses damage from measured IMU shocks (|a| more than 1.2 g *above* the mission gravity for ≥ 2 *consecutive*
samples, so a partly airborne chassis and the one-sample contact spikes when its wheels land again, even several
in a row a sample apart, are not impacts, and never set an impact's peak; armed only after
50 consecutive at-gravity samples, so an IMU's start-up transient is not one either) and over-tilt.
`peak_g` is the largest shock of the mission, `last_shock_g` the most recent one (both g above gravity);
drive health = mean `wheel_health` from `hw_status`, cross-checked against commanded vs. measured wheel speed.
`pct < 15` → critical. Battery: 600 Wh, starts at the deploy value (rover_0 82 %, rover_1 91 %), integrates
`solar_w − load_w` where load = 25 W avionics + edge compute power + drive power (physics energy/m × speed).

### 8.8 Detections addition (§3.2)
Class `ROVER` (a perceived teammate): `{"cls":"ROVER","robot":"rover_0","damaged":true,"conf":0.93,"bbox":[…],
"range_m":2.4,"bearing_rad":0.05,"world_xy":[x,y],"sources":["CAM","DEPTH","LIDAR"],
"label":"PERCEIVED: DAMAGED UNIT ROVER_1 | DOCKING FOR REPAIR"}`. The detections JSON also gains
`"lidar":{"returns_pct":78.0,"nearest_m":4.2}` (rovers; `null` for the drone).

### 8.9 Backend additions
- `POST /api/inject/damage {robot?}` (default first rover) → `{"ok":true,"detail":"Debris strike: critical damage on Scout Rover 1"}`.
- WS `camera` frames now carry `"kind":"rgb"|"depth"` (absent = rgb) and are sent for **every** robot: the client's
  selected robot at full rate (rgb + depth), every other robot's rgb at ≤ 4 Hz.
- WS events forward `/fleet/sos` as `{op:"sos", …}` and `/fleet/rescue` as `{op:"rescue", …}`.
