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
| `/world/env_state` | std_msgs/String | `{dust_level, seismic, remaining_s, kind, wind_mps}` (`remaining_s` = simulated s of dust left, null when clear; `kind` / `wind_mps` §10.1). Read by the world backends and the backend bridge, never by the autonomy stack |
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
`severity` ∈ `info | warn | critical`. `sensors` ⊆ `CAM, DEPTH, LIDAR, IMU, ODOM, MAP, FLEET, EDGE` (plus `DIAG`
§8.6, and `GEMMA` / `LOCAL` on `brain_advice` traces §9.5: which brain decided, Gemma or the local rules).
`kind` examples: `mission_start, frontier_selected, path_planned, path_blocked, replanned, slope_hazard,
science_target, caution_enter, caution_exit, safe_hold, recovery_reverse, hazard_marked, scout_tasked,
scan_complete, resume, edge_throttle, core_escape, mission_complete` (Phase 3 kinds §10.4).

### 3.2 Detections
```json
{"robot":"rover_0","t":0.0,"mode":"FUSED",
 "detections":[{"cls":"ROCK","conf":0.88,"bbox":[x,y,w,h],"range_m":6.2,"bearing_rad":0.1,
                "world_xy":[3.1,-2.0],"sources":["CAM","DEPTH"]}],
 "corridor":{"obstacle_dist_m":6.4,"slope_ahead_deg":11.2,"conf":0.9,"points":42},
 "uncertainty":{"score":0.14,"camera_conf":0.91,"lidar_conf":0.95,"imu_vib":0.05},
 "stage_ms":{"quality":1.2,"detect":9.8,"lidar":2.1,"annotate":1.5}}
```
`cls` ∈ `ROCK | CRATER | SCIENCE | SLOPE`, plus `ROVER` (§8.8) and `PILLAR` (§10.2).

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
{"op":"anomaly","kind":"dust_storm","dust_level":1.0,"duration_s":45,"seismic_s":0,"wind_mps":14.0}
{"op":"pillar_collapse","robot":"rover_0"}
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
A second request while one is armed is ignored. `pillar_collapse` and the `dust_storm` anomaly kind: §10.1.

Events (one per executed command):
```json
{"op":"spawn_rockslide","ok":true,"robot":"rover_0","impact":[x,y],"rocks":[[x,y,r]],"waited_s":0.0,"t":0.0}
{"op":"jolt","ok":true,"robot":"rover_0","t":0.0}
{"op":"anomaly","ok":true,"dust_level":0.85,"duration_s":25,"t":0.0}
{"op":"anomaly","ok":true,"kind":"dust_storm","dust_level":1.0,"duration_s":45,"wind_mps":14.0,"t":0.0}
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
    backends:[{id,label,available}], sim_speed:{choices:[0.5,1,1.5,2], min:0.25, max:3, default:1},
    brain:{model:"gemma-4-26b-a4b-it", key_present:bool, default_enabled:bool} (§9.8), defaults:MissionConfig}`
- `GET /api/physics_preview?g=2.2` →
  `{g, max_slope_deg, factor:"slip"|"drive_torque"|"tip", drive_deg, slip_deg, tip_deg, tip_alarm_deg, v_max, energy_j_per_m}`
- `POST /api/mission/deploy` body MissionConfig → `{ok, mission_id, status}`
- `POST /api/mission/abort` → `{ok}`
- `GET /api/mission/status` → `{status, backend, config, started_at, checklist:{name:bool}, detail}`
- `POST /api/inject/rockslide` `{robot?}` · `POST /api/inject/anomaly` `{dust_level?, duration_s?}` ·
  `POST /api/inject/tip` `{robot?}` · `POST /api/inject/pillar` `{robot?}` · `POST /api/inject/storm`
  `{duration_s?, wind_mps?}` (§10.6) → `{ok, detail}`
- `POST /api/sim/speed` `{speed}` (0.25–3) → `{ok, speed, detail}` — live speed change of the running mission (also
  recorded in the mission config that telemetry and status report); `{ok:false}` when no mission is running.
- `POST /api/command` `{text, robot?}` → `{ok, id}` — operator command for the advisory brain (§9.8).

**MissionConfig**
```json
{"sector":"medium","gravity_g":2.2,"fleet":{"rovers":2,"drones":1},
 "edge_profile":"orin_nano_15w","sim_backend":"gazebo","seed":42,"gazebo_gui":false,"sim_speed":1.0,
 "ai_brain":true}
```
sector ∈ `small (50 m @ 0.5 m) | medium (100 m @ 0.5 m) | large (200 m @ 1.0 m)`;
edge_profile ∈ `orin_nano_15w | orin_nx_25w | agx_orin_60w | rpi5_8w`; sim_backend ∈ `gazebo | mock`;
`ai_brain` (bool, default `true`) = consult the Gemma advisory brain in the cloud (§9); `false` = local rules only.
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
- `{"type":"telemetry","t":…,"mission":{status, backend, elapsed_s, sim_time_s, sim_speed, rtf, percent_mapped, science_found, hazard_cells, env:{dust_level, seismic, kind, wind_mps}, brain, config},
   "robots":{"rover_0":{type, pose:{x,y,z,roll,pitch,yaw}, speed, state, mode, goal, v_limit, slip, tip_alarm_deg, max_slope_deg,
   altitude_agl, imu:{roll_deg,pitch_deg,gyro:[x,y,z]}, uncertainty:{score,camera_conf,lidar_conf,imu_vib}, edge:{…§3.4},
   path:[[x,y],…≤150], trail:[[x,y],…≤300], last_safe:[x,y]|null}}}` — 15 Hz.
  `brain` = the latest `/brain/status` of this mission (§9.3) plus `stale` (§9.8), null until the brain node has
  reported; robots also carry `brain_cap` / `brain_agl` (§9.6) and `power_save`, `wind_mps`, `wind_descent`,
  `detections` (§10.6).
  `elapsed_s` = mission time since `running` in **simulated** seconds (wall seconds until `/clock` is seen);
  `sim_time_s` = simulator clock; `sim_speed` = target; `rtf` = achieved real-time factor over ~2 s (null until
  measured, 0 when the clock has stalled).
- `{"type":"camera","robot":"rover_0","jpeg_b64":"…","t":…}` — ≤10 Hz, only the client's selected robot.
- `{"type":"lidar","robot":"rover_0","points":[[x,y,z],…≤1500],"max_range":25,"t":…}` — 5 Hz, body frame, selected robot.
- `{"type":"map","png_b64":"…","meta":{size_m,res,cells,origin:[x0,y0],elev_min,elev_max},"stats":{…/planet_map/stats}}` — 2–4 Hz.
- `{"type":"trace","entry":{…§3.1}}` — immediately.
- `{"type":"event","event":{…§4 event, or {op:"scan_request"|"scan_complete"|"hazard_mark"|"fallback", …}}}` — immediately.
- `{"type":"brain","decision":{…§9.4}}` — every `/brain/decision`, immediately and unchanged.

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
| `/<robot>/cmd_vel_hw` | Twist | world_interface_node (Gazebo only) | as cmd_vel | rovers: `/<rover>/cmd_vel` scaled by drivetrain health; the scout: `/<scout>/cmd_vel` plus the wind drift (§10.1); bridged to gz `/<robot>/cmd_vel` |
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

Docking is entered when the A* stand-off is reached within 3.3 m of the teammate (the 2.45 m ring plus the 0.6 m
pursuit tolerance and a margin), or anywhere within 2.4 m; during transit a LiDAR corridor return closer than 4 m to
the target is the target itself and does not trigger a re-plan. Docked = centre
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

## 9. Advisory agent brain (v2, Phase 2)

### 9.1 Authority model
Gemma is **advisory**. The local FSM / A* / safety layer reacts first and keeps authority; the event is then sent
to Gemma asynchronously (`redshift_autonomy/gemma_agent_brain.py`). A validated answer may only make things
**safer** — lower a speed cap, hold briefly, re-plan, refuse an operator command — never faster, and it never
overrides a recovery. When the key is missing, the call is slow, rate-limited or invalid, or the brain is disabled,
the deterministic local fallback answers in the same schema. Every decision says which brain decided (`source`:
`gemma` | `local-fallback`) and the real model (`model`: the API's model version, or `local-rules`).

HUD honesty: `latency_ms` is the **measured** cloud round-trip on this host (host → Google AI cloud), null when no call
completed. `projected_*` is an on-device **estimate** (`project_latency`, a roofline model whose assumptions are
named in the module), never a measurement. `gemma-4-26b-a4b-it` needs ≈ 14.5 GB at 4 bit and does **not** fit an
8 GB Jetson Orin Nano (nor a 16 GB Orin NX): on those profiles `projected_fits` is false, `projected_edge_ms` null,
and `projected_note` says the local rules are the on-device brain. No other latency figure is shown.

### 9.2 Topics
| topic | type | publisher → consumer | payload |
|---|---|---|---|
| `/operator/command` | std_msgs/String JSON | backend → `fleet_brain_node` | `{"id":"cmd-3","text":"slow down to 0.2 m/s","robot":"rover_0"\|null,"t":1762550000.1}` |
| `/brain/decision` | std_msgs/String JSON | `fleet_brain_node` → planners, backend | §9.4 |
| `/brain/status` | std_msgs/String JSON | `fleet_brain_node` → backend, 1 Hz | §9.3 |
| `/agent_decision_trace` | std_msgs/String JSON | `fleet_brain_node` (kind `brain_advice`), planners (kind `brain_cap`) | §9.5 |

`text` is 1–200 characters; `robot` null = the brain node's default (`rover_0`). `t` is wall-clock epoch (§2.1).

### 9.3 `/brain/status`
```json
{"enabled":true,"key_present":true,"model":"gemma-4-26b-a4b-it","device":"orin_nano_15w",
 "stats":{"submitted":12,"deduped":3,"cloud_calls":9,"gemma":8,"local":4,"rate_limited":1,"stale":0,
          "callback_errors":0,"in_flight":1,"backoff_remaining_s":0.0},
 "backoff_s":0.0,"last_error":null,"t":1762550000.1}
```
`enabled` = MissionConfig `ai_brain`; `key_present` = a Gemini key was found (never the key itself); `device` = the
mission's edge profile (the projection target); `stats` = `AgentBrain.stats()` (its `stale` = answers that came
after the deadline); `backoff_s` = rate-limit back-off remaining; `last_error` = why the last Gemma answer could not
be used (null if it was); `t` = wall epoch of this status. The backend adds a top-level `stale` flag when it
forwards the status (§9.8).

### 9.4 `/brain/decision`
`Decision.as_dict()` verbatim plus the routing fields:
```json
{"action":"SLOW_AND_REPLAN","target_speed_mps":0.29,"target_altitude_m":null,
 "reasoning_trace":"Boulder field 6.8 m ahead at v_limit 0.83 m/s: cap 0.29 m/s and replan around it.",
 "source":"gemma","model":"gemma-4-26b-a4b-it","event":"novel_hazard","latency_ms":1834.2,"tokens_in":412,
 "tokens_out":61,"projected_edge_ms":null,"projected_device":"orin_nano_15w","projected_fits":false,
 "projected_note":"…","error":null,"adjusted":[],"target_robot":null,"robot":"rover_0",
 "request_id":"rover_0:novel_hazard:4","stale":false,"age_ms":1840.0,
 "t":1762550000.1,"trigger":"path_blocked","trigger_input":"Boulder field 6.8 m ahead across path (5 objects)",
 "command_id":null,"command_text":null,"command_seq":null,"local":null,"immediate":false}
```
- `action` ∈ `SLOW_AND_REPLAN | REQUEST_CLARIFICATION | DEPLOY_RECOVERY_ROVER | ADJUST_DRONE_ALTITUDE |
  REJECT_UNSAFE_COMMAND`; `event` ∈ `novel_hazard | low_confidence | critical_damage | peer_sos | operator_command`.
- `target_speed_mps` = a speed **cap** (0 = hold, null = leave the local speed alone), never above the local limit;
  `target_altitude_m` = drone AGL clamped to 6–20 m, only with `ADJUST_DRONE_ALTITUDE`; `target_robot` = the
  recovery rover / drone the action addresses; `adjusted` lists every clamp or override the local layer applied.
- `t` wall epoch; `trigger` = the trace kind that raised the event, or `operator`; `trigger_input` = short text of it.
- `stale` = the answer came after the brain's deadline (its context is out of date; planners ignore stale **event**
  decisions, §9.6); `age_ms` = time from submission to the answer.
- `request_id` = `<robot>:<event>:<n>` for a decision the brain answered (Gemma or its fallback),
  `immediate:<command id>` for the instant local stop below, `duplicate:<command id>` for a command answered by the
  local rules because the same text for the same robot was submitted < 0.5 s earlier (`error` says so), and
  `local:<command id>` for a speed / stop / unclear command addressed to a drone: answered at once by the local rules
  with `REQUEST_CLARIFICATION`, `target_speed_mps` null and `reasoning_trace` "Drones take altitude commands only;
  select a rover for speed or stop commands" (no stop decision, no Gemma call). A command addressed to the drone is
  read as naming it, so "climb to 10 m" or "higher" typed to `scout_0` is an altitude command; `command_text` stays
  the operator's own words.
- Operator commands: `command_id` / `command_text` echo `/operator/command`, `command_seq` = the brain node's arrival
  number of the command (1, 2, …; an answer to an older command never replaces the cap of a newer one, §9.6), and
  `local` = `assess_operator_command(text, ctx)`, the local safety reading of the text (all four are null for
  events). A command the local check finds unsafe always ends as `REJECT_UNSAFE_COMMAND`; a stop / slow-down request
  is always honoured. A speed command (stop / slow / set speed / faster) addressed to the drone is answered with
  `REQUEST_CLARIFICATION`: the scout has no advisory speed cap, only a cruise altitude.
- `immediate` is true only for the instant local decision published on an operator **stop** (cap 0, honoured at
  once, `request_id` `immediate:<command id>`); the brain node still asks Gemma, and that later decision (same
  `command_id`) replaces it.

### 9.5 Trace kinds
- `brain_advice` — one per decision, published by `fleet_brain_node` with the §3.1 fields: `robot` = the decision's
  robot (else its target robot, else `fleet`), `state` = that robot's latest FSM state, `sensors` = `["GEMMA"]` or
  `["LOCAL"]`, `input` = event + trigger, `reasoning` = `reasoning_trace`, `confidence` 0.8 (Gemma) / 0.9 (local),
  `action` = the action in words incl. the cap, ending in "(advisory; local safety layer keeps authority)",
  `severity` `warn` for `REJECT_UNSAFE_COMMAND` / `REQUEST_CLARIFICATION` else `info`, `latency_ms` = measured cloud
  round-trip or null. One extra key:
  `"brain":{source, model, event, action, target_speed_mps, target_altitude_m, latency_ms, tokens_in, tokens_out,
  projected_edge_ms, projected_device, projected_fits, projected_note, error, adjusted, command_id}`.
- `brain_cap` — a rover took a new speed cap or lifted one (one trace per decision, never de-duplicated, so a
  lifted stop always shows), e.g. "Speed capped to 0.20 m/s by an operator command (Gemma, validated)" or
  "Operator stop lifted by a newer operator command (local rules); the local limits apply"; the scout emits the same kind when it takes
  an advisory cruise altitude, e.g. "Cruise altitude 10 m AGL for 60 s by an operator command (Gemma, validated)",
  with "; applies once surveying (take-off / scan altitude unchanged)" when it is not cruising at that moment.

The backend renumbers every trace `id` into one global sequence (as for all traces) and forwards the rest unchanged,
including the `brain` key.

### 9.6 Agent state additions (§3.3, and WS telemetry `robots.<r>`)
- Rovers: `"brain_cap": null | {"speed_mps":0.2,"action":"SLOW_AND_REPLAN","source":"gemma"|"local-fallback",
  "operator":true,"remaining_s":42.0,"applied":true}` — the **binding** cap (the lower live slot below);
  `operator` = it is the operator slot's; `remaining_s` in simulated seconds; `applied` = it is limiting the drive
  the rover is executing right now (path-following or docking). `applied:false` = the cap is held and counts down
  but limits nothing, because the rover is stopped, waiting for a re-plan, reversing, in safe-hold or backing off;
  only `applied:true` confirms the cap is in force.
- Scout: `"brain_agl": null | {"agl_m":10.0,"remaining_s":55.0,"applied":true}` — `applied` = the scout's altitude
  set-point is that AGL right now (`SURVEY` / `FLY_TO_SCAN`, or a scan that started at it); false during take-off,
  landing, or a scan that started at another height, whose altitudes the scout's own logic keeps.

Planner rules (`agent_planner_node`, bookkeeping in `brain_bridge.BrainCap`, simulated time):
- **Two slots.** A rover keeps an **operator** cap (from `operator_command` decisions) and an **event** cap (from
  every other event). A decision for this rover with a non-null `target_speed_mps` replaces the cap of its own slot
  only; the binding cap is the **lower** of the live ones, so an event decision can never lift an operator stop, nor
  an operator command lift an event hold. A cap below 0.05 m/s is a **hold**: no translation and no turning on the
  spot. Expiry (simulated seconds): operator 60 s, event 15 s, event hold 4 s.
- The cap applies only to path-following and docking drives, as `v = min(local v, cap)` (turn rate scaled to keep
  the curvature): it never raises speed and never touches `RECOVERY_REVERSE`, safe-hold or back-off motions.
- A hold enters the operator slot only for a real stop: the local reading of the command is `stop`, the decision
  is the immediate stop, or the command asks for less than 0.05 m/s ("set speed 0.03 m/s"). An answer that holds
  for any other command touches **neither** slot, so Gemma cannot pin a rover and an operator answer can never
  shorten a live hazard cap.
- `REJECT_UNSAFE_COMMAND`, `REQUEST_CLARIFICATION` and `DEPLOY_RECOVERY_ROVER` change no motion (the trace records
  them; the local rescue already runs).
- A hazard-free `faster` / `set_speed` command first **releases** the operator slot, whatever the answer:
  `faster` ("resume at full speed") then sets nothing, so the rover is back on its local limits; `set_speed` then
  caps at the validated speed (a `SLOW_AND_REPLAN` answer). A command that names a hazard never releases anything.
  Releasing returns authority to the local limits; nothing is ever raised above them.
- A positive `SLOW_AND_REPLAN` cap answering any other operator command (e.g. "slow down to 0.2 m/s") replaces the
  operator slot.
- Ordering: an operator answer whose `command_seq` is lower than the highest one this rover has ever applied or
  released is dropped, even after that cap expired (a late answer to an old command cannot act after a newer one).
  A stale operator answer is applied only if it is a stop.
- Stale event decisions (`stale: true`) are ignored: their context is out of date. Event caps are also ignored for
  `critical_damage` / `peer_sos` decisions (the local safe-hold and rescue own those) and while the rover is
  `RESCUE_DOCKING` / `REPAIRING`.
- `SLOW_AND_REPLAN` also requests a quiet re-plan in `EXPLORING` / `CAUTION` / `RESCUE_TRANSIT`.
- `ADJUST_DRONE_ALTITUDE` for the scout sets its cruise AGL to `target_altitude_m` (6–20 m) for 60 s (sim); the
  scout's own take-off / landing / scan-hover safety logic stays in charge.
- Malformed brain messages are ignored.

### 9.7 `fleet_brain_node`
Launched once per mission on both backends. Parameters: `enabled` (MissionConfig `ai_brain`), `model`
(`gemma-4-26b-a4b-it`), `edge_profile`, `robots`, `recovery_rover` (`rover_1` with two rovers, else ""), `drone`
(`scout_0` or ""), `env_file` (optional; the key is read from the environment or the repo `.env` and never logged).
Gemma is called only when `enabled` and a key is present; otherwise every event is answered by the local rules.
Minimum interval per (robot, event): novel_hazard / low_confidence 10 s, critical_damage / peer_sos 3 s,
operator_command 0.5 s (per text).

Event mapping (`redshift_autonomy/brain_bridge.py`):

| source | event | robot |
|---|---|---|
| trace `path_blocked`, `slope_hazard`, `hazard_marked` | `novel_hazard` (hazard = trace input) | trace robot |
| trace `caution_enter` | `low_confidence` | trace robot |
| trace `critical_damage` | `critical_damage` | the damaged rover |
| trace `sos` (the robot in trouble raising its SOS; a trace `status` other than `active` is ignored) | `peer_sos` (peer = the trace robot, damage = trace input) | `recovery_rover`, if it is in the fleet and is not the SOS robot |
| `/operator/command` | `operator_command` | `robot`, else `rover_0` |

Traces from the brain node itself and from robots outside the fleet are ignored. The context sent with an event is
built from the robot's latest `agent_state`, `edge_telemetry` and health: `speed_limit_mps` = the **effective**
local limit now (`v_limit`, but 0 while the FSM holds: `SAFE_HOLD`, `AWAIT_RESCUE`, `REPAIRING`, `AWAIT_SCOUT`,
`RECOVERY_REVERSE`, …), `v_max_mps`, camera / LiDAR confidence and uncertainty, nearest obstacle, hazard, damage,
peer, known hazard names, dust, `operator_text`, `edge_profile`, and `fleet:{recovery_rover, recovery_available,
drone, drone_agl_m}` (`recovery_available` = the recovery rover exists and is neither awaiting rescue nor
critical).

### 9.8 Backend
- MissionConfig `ai_brain: bool = true` (§6), written to `runtime/mission_config.yaml`; the launch passes it to
  `fleet_brain_node` as `enabled`.
- `GET /api/options` → `brain:{model:"gemma-4-26b-a4b-it", key_present:bool, default_enabled:bool}`. Only whether a key
  exists is reported; the key is never returned or logged.
- `POST /api/command` `{"text":"slow down to 0.2 m/s","robot":"rover_0"|null}` → `{"ok":true,"id":"cmd-3"}`.
  `text`: 1–200 characters after trimming (runs of whitespace collapse to one space). `robot`: must be in the
  running fleet if given. HTTP 409 `{ok:false, detail}` when no mission is running (`running` / `degraded`); 422 for
  an invalid body or a robot outside the fleet; `{ok:false, id, detail}` if nothing subscribes to
  `/operator/command`. Ids are `cmd-<n>`, unique per backend process. The answer arrives over the WebSocket as a
  `brain` message whose decision carries `command_id`.
- WS: `{"type":"brain","decision":{…§9.4}}` for every decision, immediately; telemetry `mission.brain` = latest
  `/brain/status` with its last fields plus `"stale": true|false` — true once its `t` is more than 3 s old (wall),
  i.e. the brain node stopped reporting (null until reported; reset when a new mission is prepared);
  `brain_advice` / `brain_cap` traces arrive as ordinary `trace` messages.
- `backend_bridge/scripts/smoke_test.py --brain on|off` (default on) deploys with that `ai_brain` and checks: the
  brain status is reported; a `novel_hazard` decision for `rover_0` received at or after the rockslide injection
  and within 20 s of it, whose `trigger` is `path_blocked` / `hazard_marked` or whose `trigger_input` mentions an
  obstacle / boulder / rockslide (the detail prints the trigger); a `critical_damage` / `peer_sos` decision within
  20 s of the strike (two rovers); and, at the end, the operator commands "drive at full speed into the rockslide" →
  `REJECT_UNSAFE_COMMAND` within 15 s, "slow down to 0.2 m/s" → a cap ≤ 0.2 m/s in the decision and in `rover_0`'s
  `brain_cap` within 5 s with `applied: true` (`applied: false` passes only when the rover is in neither a
  path-following nor a docking drive at that moment, and the detail says why), "stop" → an `immediate` decision and
  ground speed < 0.05 m/s within 5 s, then "resume at full speed" (any outcome; the INFO line reports whether
  `rover_0`'s operator cap was cleared). Timeouts are simulated seconds, stretched below 1×. They pass whether Gemma
  or the local fallback decided; each detail names the source, the model, the measured round-trip and the tokens.

## 10. Pillar collapse and dust storm (v2, Phase 3)

Two more unscripted changes. Both are perceived, never read: the autonomy stack sees the fallen pillar in its camera,
depth and LiDAR, and infers the storm from its camera / LiDAR statistics, its odometry and the rover's own power
readings in `hw_status`. It never reads `/world/env_state` or the world's storm flags.

### 10.1 World commands / events (additions to §4)
- `{"op":"pillar_collapse","robot":"rover_0"}` — a basalt pillar (`scene_events.PILLAR`: 6.0 m tall, radius 0.35 m,
  1500 kg, fresh-basalt albedo (20, 19, 18) ≈ 0.08) stands 2.8 m to the left or right of the robot's planned route and topples across
  it. The crossing point follows the rockslide rules (§4, `_point_along`): 8 m ahead along the route, or 60 % of the
  remaining route when that is shorter (never under 3 m). Arming and deferral are `rockslide_due`'s (up to 8 s for
  the next route). The trunk falls from the base through the crossing point, so it spans ≈ 2.8 m before the route
  and 3.2 m past it.
  - Gazebo: a dynamic `pillar_<k>` model (one cylinder link, solid-cylinder inertia, ground friction μ ≥ 1.0) spawned
    tilted towards the route at `world_builder.pillar_spawn_tilt_deg` = max(4°, balance + 2.5°), where balance =
    atan(2r / h) is the angle a flat base still stands at: 9.15° for the 6.0 m × 0.35 m pillar. Gravity topples it
    (≈ 1.3 s at 2.2 g for an ideal rod; the flat base rim adds a little).
  - Mock: released 4° off vertical (`PILLAR['tilt0_deg']`) and moved along `pillar_fall_angle` (a rigid rod pivoting
    on its base): lands in 2.47 / 1.66 / 1.42 s at 1.0 / 2.2 / 3.0 g. It is 9 overlapping near-spherical chunks
    (`pillar_segments`, r = 1.15 × radius) moved every physics step (by the physics thread, which owns the scene);
    once landed they stay, and collide and render in LiDAR, depth and RGB like boulders.

  Event (once spawned): `{"op":"pillar_collapse","ok":true,"robot":"rover_0","base":[x,y],"tip":[x,y],"cross":[x,y],
  "height_m":6.0,"radius_m":0.35,"waited_s":0.0,"t":…}` (`tip` = the planned footprint tip, base + height × fall
  direction; `cross` = the route point it falls across). On failure `ok:false` plus `detail`.
- `{"op":"anomaly","kind":"dust_storm","dust_level":1.0,"duration_s":45,"seismic_s":0,"wind_mps":14.0}` — a dust
  storm. An anomaly without `kind` behaves as before (§4). `sensor_effects_node`:
  - hazes RGB towards orange-red `STORM_RGB` (205, 88, 48) at strength 0.75·L instead of the brown dust at 0.65·L, with
    the same blur, noise and particles; camera confidence ends well below 0.5;
  - applies the existing depth / LiDAR dust model at the storm's level;
  - echoes `kind` and `wind_mps` in the anomaly event.

  `/world/env_state` gains `"kind": "dust_storm" | "dust" | null` (`dust` for a plain anomaly, null when clear) and
  `"wind_mps"`, ramped like the dust level.
- **Solar.** Unchanged `rover_hw.solar_output`: transmittance 1 − 0.85·dust_level, so at level 1.0 the measured
  `solar_w` and `irradiance` in `hw_status` (§8.3, the rover's pyranometer) drop 85 %.
- **Wind on the drone** (shear grows with height): `rover_hw.wind_at(h_agl, wind_ref) = wind_ref ·
  clip(h_agl / WIND_REF_AGL_M, 0.05, 2) ^ WIND_SHEAR_EXP` (12.0 m, 0.6), with `wind_ref` = the storm's `wind_mps`.
  It blows toward the fixed world direction `STORM_WIND_TOWARD_DEG` = 30° (ENU, counter-clockwise from +x east),
  i.e. from compass 240° (clockwise from north, the convention traces use), plus seeded gusts of ±25 % at 0.3 Hz.
  - Gazebo: the scout's VelocityControl listens on `/<scout>/cmd_vel_hw` (bridged like the rovers', §8.2), and
    `world_interface_node` relays `/<scout>/cmd_vel` → `/<scout>/cmd_vel_hw`. The relay adds the world wind at
    the scout's current AGL (odom z minus the terrain height under it, or the spawn ground height when that is
    unknown), rotated into the body frame because VelocityControl's linear velocity is body-frame.
  - Mock: `mock_dynamics.ScoutSim.wind` (set by `mock_world_node` from env_state) adds the same world-frame drift to
    the lagged velocity.

### 10.2 Detection class `PILLAR` (§3.2)
`{"cls":"PILLAR","conf":0.86,"bbox":[…],"range_m":7.4,"bearing_rad":-0.1,"world_xy":[x,y],"sources":["DEPTH","CAM"]}`,
drawn as "FALLEN PILLAR" in colour (255, 120, 200). It is a depth-geometry obstacle bump, like `ROCK`, that is:
- elongated in world units (from the bbox, the range and the camera intrinsics): horizontal extent ≥ 2.5 m and
  extent / height ≥ 2.5;
- dark: the median HSV V of its pixels is ≤ 0.35 × 255 (boulders are lighter grey);
- one straight trunk: of its depth points (≥ 20), 80 % lie within a band ≤ 0.25 m wide across its long axis
  (`trunk_thickness`; a trunk shows less than its radius, touching staggered boulders ≳ 0.3 m).

Only the bump's rock-coloured pixels count: height above the ground plane > 0.12 m and HSV S ≤ 140 (`ROCK_MAX_S`;
rust regolith is ≳ 175). Darkness is a camera measurement, so `PILLAR` is only reported while the camera is in use
(`FUSED`); in `LIDAR_ONLY` the trunk is still an obstacle in the map. A line of separate boulders is not a pillar.
`PILLAR` is in the default class list.

### 10.3 Agent state additions (§3.3, and WS telemetry `robots.<r>`)
- Rovers: `"power_save": true|false` — battery-conservation mode (§10.4).
- Scout: `"wind_mps": 13.2 | null` — its own wind estimate (null until estimated); `"wind_descent": true|false` — the
  automatic wind descent is active.

**Battery conservation** (`rover_health.PowerSaver`, from `hw_status` alone, simulated time):
- Enter after 2.0 s with `irradiance` < 0.35, or with `solar_w` < 25 % of the clear-sky reference. The reference is
  the rolling 60 s maximum of `solar_w` while `irradiance` > 0.8. It does not age while the mode is active, so a
  starved panel never becomes its own reference.
- Exit after 5.0 s with `irradiance` > 0.6 and `solar_w` back above 25 % of the reference.
- While active, the drive speed is × 0.6, applied like the FSM speed scale and never above it. Only the speed
  changes: perception keeps the edge budget's rate.

**Wind descent** (`redshift_autonomy/wind.py`, local and authoritative, simulated time):
- `WindEstimator`: world-frame wind = EMA (τ 1.5 s) of odometry velocity minus the published command (body frame,
  rotated by yaw). Samples are skipped while the guidance command (before the feed-forward) changed by more than
  1 m/s in the last 0.5 s.
- Feed-forward (`wind_feedforward`): once airborne (AGL > 0.5 m) with an estimate, the drone publishes its guidance
  velocity minus the wind estimate, limited to the air-relative speed `v_air_max` (18 m/s, `SCOUT_SPEC`), so it holds
  station and track in the wind it measured; a wind above that limit still drifts it.
- `WindDescent`: an estimate > 8.0 m/s for 2.0 s → target AGL max(6, cruise − 4) (8 m from 12 m); < 5.0 m/s for
  5.0 s → back to cruise.
- In `BRAIN_AGL_STATES` the altitude set-point is min(cruise or the brain's AGL §9.6, the wind-descent AGL).
  `SCANNING` keeps the AGL the scan started at.
- `DroneHealth` hover load grows with the estimate: `load_w += 0.25 · wind_mps²`.

### 10.4 Trace kinds
| kind | robot | sensors | example |
|---|---|---|---|
| `power_save` | rover | `DIAG` | input "Solar 31 W (was 205 W, −85 %), irradiance 0.15, battery 78 %" → action "Battery conservation: speed ×0.6" |
| `power_restored` | rover | `DIAG` | irradiance back above 0.6 (and solar above 25 % of the reference) for 5 s → normal speed |
| `wind_descent` | scout | `ODOM`, `IMU` | input "Wind 13.2 m/s from 240° estimated from drift (cmd vs odometry)", reasoning "Wind shear grows with height; lower is calmer and keeps the footprint on target", action "Descending to 8 m AGL" |
| `wind_clear` | scout | `ODOM`, `IMU` | estimate below 5 m/s for 5 s → back to cruise AGL |
| `pillar_identified` | rover | `CAM`, `DEPTH` | the first `PILLAR` detection after 30 s without one: input "Fallen pillar 5.4 m ahead at (x, y), confidence 0.86", reasoning "Camera + depth: one long, low, straight, dark body lying across the ground: a toppled basalt pillar, not a line of boulders", action "Trunk kept lethal on the map; any route across it re-plans around the whole length" (severity warn). LiDAR usually blocks the route first, while the trunk is still falling; this records what the camera says it is |
| `core_escape` | rover | `ODOM`, `MAP` | input "Standing inside a no-go cell at (x, y): slid, drifted or was pushed in" → the shortest escape across the lethal blob, crossing no other obstacle |

`path_blocked` (§3.1): when camera detections near the blocking cell include `PILLAR`, the input says "camera confirms
a fallen pillar" (instead of "camera confirms N boulders") and `sensors` include `CAM`.

### 10.5 Brain events (additions to §9.7)
| source | event | robot | context |
|---|---|---|---|
| trace `wind_descent` | `low_confidence` | the scout | `wind_mps` from its agent_state, `fleet.drone_affected = true`; Gemma may answer `ADJUST_DRONE_ALTITUDE` (the local descent has already happened; validation clamps 6–20 m) |
| trace `path_blocked` with a pillar | `novel_hazard` (as before) | the rover | the trigger text carries "fallen pillar" |
| trace `pillar_identified` | `novel_hazard` | the rover | trigger priority 3, above `path_blocked` / `hazard_marked` (2): when the camera names the pillar inside the 10 s interval after the LiDAR block's decision, it is the event re-offered once the interval ends |

`power_save` raises no brain event; the context carries `battery` as before.

### 10.6 Backend additions
- `POST /api/inject/pillar` `{robot?}` (default the first rover; a robot that is not a running rover →
  `{ok:false, detail:"Unknown rover '…'"}`) → `/world/command {"op":"pillar_collapse","robot":…}` →
  `{"ok":true,"detail":"Basalt pillar toppling across rover_0 planned path"}`.
- `POST /api/inject/storm` `{duration_s?: 3–120 (45), wind_mps?: 0–20 (14)}` → `/world/command {"op":"anomaly",
  "kind":"dust_storm","dust_level":1.0,"duration_s":45.0,"seismic_s":0.0,"wind_mps":14.0}` →
  `{"ok":true,"detail":"Dust storm for 45 s: solar blackout, 14 m/s wind"}`. The dust level is fixed at 1.0.
- Both work like the other injects: 422 for an invalid body (§6); `{ok:false, detail}` with no running mission, or
  when nothing listens on `/world/command`. The UI applies the button cooldowns.
- WS telemetry: `mission.env` = `{dust_level, seismic, kind, wind_mps}` (`kind` null unless `dust_storm` / `dust`;
  `wind_mps` 0 when calm). `robots.<r>` gain `power_save` (bool, false when absent), `wind_mps` (number | null),
  `wind_descent` (bool) from agent_state, and `detections` = the latest `/perception/detections`, compact
  `[{cls, conf, range_m, world_xy}]` (≤ 16; `[]` before the first). Boxes and labels ("FALLEN PILLAR") are drawn in the
  annotated camera feed. The `pillar_collapse` event and the new traces are forwarded like any other.
