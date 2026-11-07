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
lands beyond the path's end; along the heading when there is no path.

Events (one per executed command):
```json
{"op":"spawn_rockslide","ok":true,"robot":"rover_0","impact":[x,y],"rocks":[[x,y,r]],"t":0.0}
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
