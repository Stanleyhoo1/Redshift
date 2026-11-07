# Redshift — Autonomous Super-Earth Planetary Mapping Fleet · Implementation Plan

> **For agentic workers:** Execute task-by-task with superpowers:executing-plans (inline) — optionally hand Phase 6 (web UI) to a parallel subagent once `docs/protocol.md` (Task 1) is frozen. Steps use `- [ ]` checkboxes. TDD for every pure-Python core module.

**Goal:** A single-command (`./start_all.sh`) embodied multi-agent robotics demo: a configurable Super-Earth world in Gazebo Harmonic, a Ground Rover + Aerial Scout running 100% local perception → mapping → gravity-aware planning → failure recovery in ROS 2 Jazzy, streamed through a FastAPI/WebSocket bridge into a React + Three.js Mission Control with a fog-of-war globe.

**Architecture:** Gazebo (or a drop-in mock world node publishing *identical* topics) → `ros_gz_bridge` → a sensor-effects layer (dust/seismic) → per-robot perception + planner nodes and one global mapping coordinator → FastAPI backend (rclpy node in a background thread) → one WebSocket to the browser. All algorithmic logic lives in pure, unit-tested Python modules; ROS nodes are thin wrappers.

**Tech stack:** Ubuntu 24.04 (WSL2) · ROS 2 Jazzy · Gazebo Harmonic (gz-sim 8.15) + ros_gz · Python 3.12 (numpy 1.26, OpenCV 4.6, scipy, FastAPI 0.115, uvicorn, gz.transport13 bindings) · Node 25 (nvm) · React 19.3 · Vite 8 · Tailwind 4 · three 0.187 · @react-three/fiber 9.8 · drei 10.7 · zustand 5.

---

## Context

The repo `/home/yubow/Redshift` is empty (README + LICENSE, 1 commit). We're building a hackathon entry for the Agentverse 2026 Robotics Track ("Give an agent a body"). Judging weights drive priorities: **operational autonomy + edge budget + failure recovery (35%)**, **perception feed + traceable decisions (25%)**, **unscripted mid-run change + re-planning (25%)**. The deliverable must survive a live 1-minute single-take demo video, so determinism, visible cause→effect, and a robust fallback mode matter more than feature breadth.

**User decisions (confirmed):** globe shows the sector as a ~90°×90° patch on the facing hemisphere (rest = fog); fleet selector allows 1–2 rovers + 0–1 drone (default 1+1); the tip/slip failure is caused by a **real physical jolt** (force/torque pulse) the IMU genuinely measures.

## Verified environment (probed 2026-11-07)

| Item | Finding |
|---|---|
| OS / kernel | Ubuntu 24.04.3, WSL2 6.6, WSLg (`DISPLAY=:0`, Wayland), 8 cores, 11 GB RAM |
| GPU | RTX 5050 Laptop via `/dev/dxg` (D3D12) |
| ROS 2 | Jazzy at `/opt/ros/jazzy`; colcon present; **no** rosbridge/nav2 (not needed) |
| Gazebo | gz-sim **8.15** (Harmonic) via vendor pkgs; `ros_gz_bridge`, `ros_gz_image`, `ros_gz_sim`, `ros_gz_interfaces`, `cv_bridge`, `sensor_msgs_py` installed |
| Rendering sensors | **Verified working headless**: `gz sim -s -r --headless-rendering sensors_demo.sdf` publishes `/camera` 320×240 RGB and `/lidar` 640 rays |
| gz systems | diff-drive, velocity-control, apply-link-wrench, user-commands, sensors, imu, odometry-publisher, physics, scene-broadcaster all present |
| gz Python | `gz.transport13`, `gz.msgs10` import OK → backend/world nodes can call `/world/<w>/create` and publish `EntityWrench` directly |
| Python | 3.12.3 with numpy, cv2, scipy, PIL, fastapi, uvicorn, websockets, pydantic 2, httpx, pyyaml, pytest 9 → **zero new Python deps** |
| Node | nvm Node v25.2.1 / npm 11.6 (must stay ahead of Windows `/mnt/c/.../npm` on PATH); npm registry reachable |

Reference snippets for plugin syntax: `/opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/worlds/{diff_drive_skid,velocity_control,apply_link_wrench,heightmap,sensors_demo,depth_camera_sensor,gpu_lidar_sensor}.sdf`. Heightmaps need DART with `<collision_detector>bullet</collision_detector>` (per `heightmap.sdf`). Wrench topics: `/world/<w>/wrench`, `/wrench/persistent`, `/wrench/clear` (EntityWrench / Entity).

## Key design decisions

1. **One topic contract, two world backends.** `sim_backend: gazebo | mock`. The mock world node publishes exactly the raw topics the gz bridge would (`image_raw`, `depth_raw`, `lidar/points_raw`, `imu_raw`, `odom`, `wheel_odom`) and obeys the same `/world/command` JSON ops. Autonomy and backend never know which is running. Backend auto-falls back to mock if Gazebo yields no camera frames within 45 s (logged as a trace).
2. **World is generated per deploy**, not mutated at runtime: DEPLOY writes `runtime/world.sdf` with `<gravity>0 0 -9.81·g</gravity>`, a procedurally generated 16-bit heightmap PNG, regolith texture, boulders, craters, science crystals, and the robots embedded at spawn poses (no spawn race). The same seeded `Terrain` drives the mock world.
3. **No ground truth in autonomy.** Perception, mapping and planning consume only (degraded) sensor topics + odometry pose. Uncertainty is *computed from sensor data statistics*, never read from the dust flag. (Pose = OdometryPublisher, presented as "VIO/odometry"; optional drift noise param, default 0.)
4. **Dust/seismic is an environment model** (`sensor_effects_node`, in exoswarm_sim) sitting between raw sensors and autonomy: haze/blur/noise on RGB, dropout + range clipping on depth, dropout + range noise + dust backscatter on LiDAR, vibration on IMU.
5. **Gravity is physically grounded and shared.** `physics_model.py` computes the traversability limit from drive torque, Mohr-Coulomb soil traction and static tip angle; the same τ_max is written into the Gazebo wheel joint `<effort>` limit so the simulated rover physically stalls where the planner says it can't climb.
6. **JSON-over-`std_msgs/String`** for traces, detections, agent state, fleet coordination and world commands (schemas in `docs/protocol.md` + `trace.py`); standard typed msgs for sensors, maps (`nav_msgs/OccupancyGrid`), paths (`nav_msgs/Path`), cmd_vel (`geometry_msgs/Twist`). Avoids a CMake interface package; the backend forwards JSON straight to the UI.
7. **Map to browser as a PNG layer image** (R = normalized elevation, G = class, B = reveal age) at 2–4 Hz; the globe shader decodes it. Camera frames are JPEGs encoded once in the perception node.
8. **Everything local**: fonts bundled via `@fontsource`, no CDNs, no cloud calls.
9. **Dependency direction:** `exoswarm_sim` may import `exoswarm_autonomy.physics_model` (mock stall model); `exoswarm_autonomy` never imports `exoswarm_sim` — robot geometry/extrinsics (`ROBOT_SPECS` in `robot_models.py`) reach autonomy nodes as ROS params set by the launch file. The backend imports both (physics preview, option catalogs).

## Architecture & data flow

```
 Web UI (React/R3F) ──REST /api/*──►  FastAPI backend ──spawns──► ros2 launch exoswarm_sim mission.launch.py config:=runtime/mission_config.yaml
        ▲  one WebSocket /ws               │ rclpy RosLink (bg thread)                 │
        └──────────── telemetry/camera/lidar/map/trace/event ◄────────────────────────┤
                                                                                       ▼
   ┌─ gazebo branch: gz sim -s --headless-rendering world.sdf + parameter_bridge + world_interface_node
   └─ mock branch:   mock_world_node (kinematics + numpy raycast lidar/depth/RGB)
            │ raw topics  /<r>/{camera/image_raw,camera/depth_raw,lidar/points_raw,imu_raw,odom,wheel_odom}
            ▼
   sensor_effects_node (per robot; dust/seismic)  →  /<r>/{camera/image,camera/depth,lidar/points,imu}
            ▼                                   ▼
   perception_node (per robot)          mapping_coordinator_node (global) → /planet_map (+elevation, layers, stats)
            │ detections/uncertainty/annotated JPEG          ▲ hazard marks
            ▼                                                │
   agent_planner_node (per robot: rover | scout) ── cmd_vel, planned_path, agent_state, /agent_decision_trace,
                                                     edge_telemetry, /fleet/{claims,hazard_marks,scan_request,scan_complete}
```

## Contracts (frozen in Task 1 as `docs/protocol.md`)

**Robot names:** `rover_0`, `rover_1`, `scout_0`. World name `redshift`. Frame: world x/y ∈ [−L/2, L/2], z up; grid origin (−L/2, −L/2), row = y, col = x.

**ROS topics (per robot `/<r>/…`):** `cmd_vel` Twist · `odom` Odometry (world-frame pose) · `wheel_odom` Odometry (rover, DiffDrive) · `imu_raw`→`imu` Imu · `camera/image_raw`→`camera/image` Image rgb8 · `camera/depth_raw`→`camera/depth` Image 32FC1 · `camera/camera_info` · `lidar/points_raw`→`lidar/points` PointCloud2 · `perception/annotated/compressed` CompressedImage JPEG · `perception/detections` String(JSON) · `perception/uncertainty` Float32 · `perception/scan2d` LaserScan · `perception/mode_request` String (`FUSED`|`LIDAR_ONLY`) · `planned_path` Path · `agent_state` String(JSON) · `edge_telemetry` String(JSON).
**Global:** `/planet_map` OccupancyGrid (−1/0/50/100) · `/planet_map/elevation` Image 32FC1 (NaN unknown) · `/planet_map/layers` Image rgb8 · `/planet_map/stats` String · `/agent_decision_trace` String · `/fleet/claims|hazard_marks|scan_request|scan_complete` String · `/world/command` String · `/world/events` String · `/world/env_state` String.

**`/world/command` ops:** `{"op":"spawn_rockslide","robot":"rover_0"}` (world node computes layout from that robot's `planned_path` + `odom` via `scene_events.rockslide_layout`) · `{"op":"jolt","robot":"rover_0"}` · `{"op":"anomaly","dust_level":0.85,"duration_s":25,"seismic_s":4}` · `{"op":"clear_anomaly"}`. `/world/events` echoes `{"op":…,"ok":bool,"impact":[x,y],"rocks":[[x,y,r],…],"t":…}`.

**Decision trace (`trace.py`):**
```json
{"id":17,"t":1762550000.12,"robot":"rover_0","state":"REPLANNING","kind":"path_blocked",
 "sensors":["LIDAR","CAM"],"input":"Boulder field 6.8 m ahead across path (5 objects)",
 "reasoning":"Corridor occupancy 0.82; step height 0.9 m > 0.30 m clearance","confidence":0.91,
 "action":"Re-planning A* around obstacle (+7.4 m, 1412 nodes, 38 ms)","severity":"warn","latency_ms":41.7}
```
Dedup same (robot, kind) within 2 s unless content changes.

**Detections JSON:** `{robot, t, mode, detections:[{cls: ROCK|CRATER|SCIENCE|SLOPE, conf, bbox:[x,y,w,h], range_m, bearing_rad, world_xy, sources:[…]}], corridor:{obstacle_dist_m, slope_ahead_deg, conf}, uncertainty:{score, camera_conf, lidar_conf, imu_vib}, stage_ms:{…}}`.
**agent_state JSON:** `{robot, type, state, mode, goal:[x,y], speed, v_limit, tilt:{roll,pitch}, slip, breadcrumbs, last_safe:[x,y]}`. **edge_telemetry JSON:** `{robot, profile, host_ms:{perception,planner,mapping}, device_ms, loop_hz, load_pct, power_w, perception_hz, throttle_level, uncertainty}`.

**REST:** `GET /api/options` · `GET /api/physics_preview?g=2.2` · `POST /api/mission/deploy` (MissionConfig) · `POST /api/mission/abort` · `GET /api/mission/status` · `POST /api/inject/rockslide {robot}` · `POST /api/inject/anomaly {dust_level?,duration_s?}` · `POST /api/inject/tip {robot}`.
**MissionConfig:** `sector: small|medium|large` (50 m@0.5 m / 100 m@0.5 m / 200 m@1.0 m) · `gravity_g: 1.0–3.0` (default 2.2) · `fleet: {rovers: 1–2, drones: 0–1}` · `edge_profile: orin_nano_15w|orin_nx_25w|agx_orin_60w|rpi5_8w` · `sim_backend: gazebo|mock` · `seed` · `gazebo_gui: bool`.
**WS server→client:** `telemetry` (15 Hz: per-robot pose/state/imu/edge/path, mission status/elapsed/percent/env) · `camera` (≤10 Hz, selected robot, `jpeg_b64`) · `lidar` (5 Hz, ≤1500 robot-frame points) · `map` (2–4 Hz, `png_b64` + meta + stats) · `trace` (immediate) · `event` (immediate) · `boot_log`. **client→server:** `{"type":"select_camera","robot":…}`.

## Physics & autonomy models (numbers to implement)

- **Rover:** 60 kg, wheel r 0.20 m, track 0.85 m, CoG 0.35 m, τ_max 20.8 N·m/wheel (F_max 416 N). Drive limit θ_d = asin(F_max/(m·g)); slip limit θ_s = atan(tan32° + 150 N/(m·g)); tip θ_t = atan(0.425/0.35) = 50.5°. **θ_max(g) = 0.9·min(θ_d, θ_s, θ_t)** → ≈37° @1.0g (slip-bound), ≈16.8° @2.2g (torque-bound), ≈12.3° @3.0g. Speed limit v = 1.0·min(1, √(1.5/g)) m/s. Tip alarm = min(0.7·θ_t, θ_max + 12°) (≈28.8° @2.2g). Energy/m = m·g·(0.08 + max(0, Δh/d))/0.75.
- **Terrain:** fBm relief 2.5/5/9 m (S/M/L); mostly < 12°; 1–2 ridges at 22–30° (crossable at 1g, impassable at 2.2g → visible gravity effect); craters with rims; 25–60 boulders (0.3–1.2 m ellipsoids); 4–8 cyan science crystals; flattened spawn pad at (−0.35L, −0.35L).
- **Sensors:** rover RGBD 320×240 hfov 80° pitched −12° @10 Hz; gpu_lidar 360×16 (−15°…+10°), 0.3–25 m @10 Hz; IMU 100 Hz. Scout: downward RGBD 160×120 hfov 90° @5 Hz, flies 12 m AGL (≈24 m footprint), VelocityControl (no PX4).
- **Uncertainty:** camera_conf from Laplacian sharpness, RMS contrast, noise vs. a clear-frame baseline; lidar_conf from return ratio, near-field spurious ratio, range noise; imu vibration RMS. FUSED score = 1 − (0.45·cam + 0.45·lidar + 0.10·(1−imu_pen)); LIDAR_ONLY score = 1 − (0.8·lidar + 0.2·(1−imu_pen)). Dust 0.85 ⇒ ≈0.55 (CAUTION, not safe-hold).
- **FSM (rover):** EXPLORING ↔ CAUTION (enter: score > 0.45 for 0.5 s or cam < 0.4; exit: cam > 0.7 for 2 s; speed ×0.35, LIDAR_ONLY, extra inflation) · → REPLANNING on path blocked by new lethal cells or corridor obstacle < stop distance · ANY → SAFE_HOLD on |roll|/|pitch| > tip alarm for 100 ms, or slip > 0.6 for 0.8 s, or score > 0.8 for 1.5 s → (halt 1.5 s) RECOVERY_REVERSE along breadcrumbs (0.3 m/s, ≥3 m back to last safe cell) → publish permanent hazard (r 2.5 m) + `scan_request` → AWAIT_SCOUT (timeout 25 s; skip if no scout) → RESUMING → EXPLORING · MISSION_COMPLETE when no frontiers or ≥95 % known. Breadcrumb every 0.5 m when tilt < 0.5·alarm and score < 0.4.
- **Scout FSM:** TAKEOFF → SURVEY (frontier/lawnmower over unknown at 12 m AGL) ⇄ FLY_TO_SCAN → SCANNING (4 s hover + 3 m orbit) → publish `scan_complete` with measured slope stats.
- **Planner:** 8-connected A* (heapq, octile heuristic), cost = d·(1 + slope ramp) + w_e·g·max(0, Δh) + unknown 2.5 + caution ×3; lethal = slope > θ_max | step > 0.30 m | permanent; lethal dilated by robot radius (0.6 m); max expansions from edge profile. Frontiers = known-free cells adjacent to unknown, clustered with `cv2.connectedComponentsWithStats`, score = size − λ·path_cost − claim penalty (8 m radius around other robots' goals). Pure-pursuit control at 10 Hz.
- **Edge profiles:** `orin_nano_15w` (perf ×2.6, 5–15 W, perception ≤10 Hz, 60k expansions) · `orin_nx_25w` (×1.6, 7–25 W, 15 Hz, 120k) · `agx_orin_60w` (×0.9, 15–60 W, 20 Hz, 250k) · `rpi5_8w` (×5.0, 3–8 W, 5 Hz, 30k). load = Σ(host_ms·perf·hz)/1000; power = idle + (max−idle)·min(1, load); throttle perception −25 % when load > 90 %, restore below 60 % (each change emits a trace).

## File structure

```
start_all.sh · stop_all.sh · README.md · .gitignore · runtime/ (generated, ignored)
docs/protocol.md · docs/superpowers/plans/2026-11-07-redshift-fleet.md (copy of this plan)
ros2_ws/src/exoswarm_sim/            (ament_python)
  exoswarm_sim/terrain_gen.py         seeded heightfield + features (pure)
  exoswarm_sim/robot_models.py        rover/scout SDF + ROBOT_SPECS (geometry, extrinsics, τ_max) (pure)
  exoswarm_sim/world_builder.py       world.sdf + heightmap/texture PNGs + bridge YAML from config (pure)
  exoswarm_sim/scene_events.py        rockslide layout, jolt profile (pure)
  exoswarm_sim/mock_render.py         vectorized raycast lidar/depth/RGB on heightfield+rocks (pure)
  exoswarm_sim/mock_world_node.py     kinematics, roll dynamics, sensor publishing, /world/command
  exoswarm_sim/world_interface_node.py  Gazebo /world/command impl via gz.transport13
  exoswarm_sim/sensor_effects_node.py dust/seismic layer
  launch/mission.launch.py            config yaml → gazebo|mock branch + bridges + autonomy nodes
  test/test_{terrain_gen,world_builder,scene_events,mock_render}.py
ros2_ws/src/exoswarm_autonomy/       (ament_python)
  exoswarm_autonomy/{physics_model,edge_budget,grid_map,planning,agent_fsm,vision,trace}.py   (pure)
  exoswarm_autonomy/{perception_node,mapping_coordinator_node,agent_planner_node}.py
  test/test_{physics_model,edge_budget,grid_map,planning,agent_fsm,vision,trace}.py
backend_bridge/  app.py · ros_link.py · mission_manager.py · encoders.py · config_schema.py
                 tests/test_{config_schema,encoders,mission_manager}.py · scripts/smoke_test.py
web_ui/          package.json (exact pins) · vite.config.ts (proxy /api,/ws → :8000) · index.html
  src/lib/{api,ws,store,geo,types}.ts · src/lib/geo.test.ts
  src/screens/{PreLaunch,BootSequence,MissionControl}.tsx
  src/globe/{Globe,PlanetFog,SectorPatch,sectorShader,RobotMarkers,PathLines,Effects}.tsx|ts
  src/panels/{CameraFeed,LidarRadar,ImuAttitude,DecisionTrace,EdgeHud,InjectBar,FleetStatus,MissionProgress}.tsx
```

## Tasks

Work on branch `feat/redshift-fleet`; commit at the end of every task (message ends with the Co-Authored-By trailer). Run pure tests with `python3 -m pytest ros2_ws/src/<pkg>/test -q`.

### Phase 0 — Scaffold
- [ ] **T1 Scaffold & contracts.** Create branch, dir tree, both `package.xml`/`setup.py`/`setup.cfg`/`resource` markers (entry points for every node), `.gitignore` (build/ install/ log/ runtime/ node_modules/ dist/), `docs/protocol.md` (the Contracts section verbatim), plan copy, README stub. Verify: `cd ros2_ws && colcon build --symlink-install` succeeds; `ros2 pkg list | grep exoswarm` shows both.

### Phase 1 — Pure core (TDD: write tests → see fail → implement → pass → commit)
- [ ] **T2 `physics_model.py`.** Tests: θ_max(1.0) ∈ [35, 39] and limiting factor `slip`; θ_max(2.2) ∈ [15.5, 18] and factor `drive_torque`; monotonic decrease 1.0→3.0; tip_alarm(2.2) ≈ 28.8 ± 1; speed_limit(2.2) < speed_limit(1.0); energy uphill > flat. API: `RoverPhysics.from_spec(spec)`, `.limits(g) -> Limits(max_slope_deg, factor, drive, slip, tip, tip_alarm_deg, v_max)`, `.energy_per_m(g, dh, d)`.
- [ ] **T3 `edge_budget.py`.** Tests: profile catalog keys; load/power math; throttle steps down at load 0.95 and recovers below 0.6 with hysteresis; returns trace-worthy change flag. API: `EdgeBudget(profile).update(host_ms: dict, hz: dict) -> EdgeState`.
- [ ] **T4 `terrain_gen.py`.** Tests: same seed ⇒ identical arrays; spawn pad slope < 3°; at least one ridge cell slope ≥ 22°; rocks don't overlap spawn pad; `height_at(x,y)` bilinear matches grid. API: `generate_terrain(size_m, seed, n) -> Terrain(heights, size_m, n, rocks, craters, science, spawn_xy)`.
- [ ] **T5 `planning.py`.** Tests: A* routes around a lethal wall; at g=2.2 prefers a longer flat detour over a 20° ramp that it takes at g=1.0; returns None when enclosed; frontier extraction finds two clusters on a synthetic known/unknown grid; `path_blocked` index; pure pursuit turns toward the path. API: `build_cost(layers, limits, inflation_m, caution_mode)`, `astar(cost, elev, start, goal, g, max_exp) -> (path, stats)`, `frontiers(known, free, min_size)`, `select_frontier(...)`, `pure_pursuit(pose, path, lookahead, v_max)`.
- [ ] **T6 `grid_map.py`.** Tests: integrating a flat point patch marks cells known/free and ray cells between origin and hits known; a 25° synthetic ramp is HAZARD at 2.2 g but FREE at 1.0 g; a 0.8 m box is lethal via step height; `mark_permanent` persists after re-observation; `layers_rgb()` encodes classes; `percent_known()`. API: `GlobalMap(size_m, res)`, `.integrate(points_world, origin, robot_idx, t)`, `.recompute(limits)`, `.to_occupancy()`, `.layers_rgb(t)`, `.mark_permanent(xy, r)`, `.mark_science(xy)`, `.stats()`.
- [ ] **T7 `agent_fsm.py` + `trace.py`.** Tests: tilt 31° for 120 ms ⇒ SAFE_HOLD → (1.5 s) RECOVERY_REVERSE → arrival ⇒ hazard_mark + scan_request actions → AWAIT_SCOUT → scan_complete ⇒ RESUMING → EXPLORING; caution enter/exit hysteresis; blocked ⇒ REPLANNING; every transition yields a trace dict with sensors/input/reasoning/confidence/action; dedup within 2 s. API: `RoverFSM(limits).step(Inputs, now) -> Output(state, commands, traces)`, `ScoutFSM`, `TraceBuilder`.
- [ ] **T8 `vision.py`.** Tests on synthetic frames: gray blob on rust background + depth bump ⇒ ROCK conf > 0.6; cyan blob ⇒ SCIENCE; depth pit ⇒ CRATER; hazed/blurred/noisy version raises uncertainty score ≥ 0.2 over clear; LiDAR corridor finds obstacle distance on synthetic cloud; `annotate()` returns same-shape BGR. API: `ImageQualityBaseline`, `image_quality`, `lidar_quality`, `fuse_uncertainty`, `detect_objects(rgb, depth, K, cam_pitch, cam_h)`, `lidar_corridor(points_base)`, `annotate(...)`.
- [ ] **T9 `scene_events.py`.** Tests: rockslide center ≈ 7 m along path from robot, rocks span ±3.5 m perpendicular to path tangent, falls back to heading when path empty; jolt profile (force/torque/duration) constants.

### Phase 2 — Gazebo world
- [ ] **T10 `robot_models.py` + `world_builder.py`.** Rover: box chassis, 4 wheels (cylinder visual, sphere collision, μ 0.9, joint `<effort>` = τ_max), DiffDrive (4 joints, topics `/<r>/cmd_vel`, `/<r>/wheel_odom`), OdometryPublisher (`/<r>/odom`, dimensions 3), rgbd_camera `/<r>/camera`, gpu_lidar `/<r>/lidar`, imu `/<r>/imu`. Scout: VelocityControl (`/<r>/cmd_vel`), OdometryPublisher, downward rgbd `/<r>/camera`, imu. World: DART + bullet collision detector, 4 ms step, gravity from config, Physics/UserCommands/SceneBroadcaster/Sensors(ogre2)/Imu/ApplyLinkWrench systems, rust background/haze, low sun, 16-bit heightmap (2ⁿ+1 px) + regolith texture, boulders (static ellipsoids), crystals (emissive cyan), robots at spawn heights. Also emits `runtime/bridge.yaml` (parameter_bridge entries with ros/gz topic remaps → `*_raw`). Tests: XML parses, gravity value, model names, heightmap PNG dims/dtype, bridge YAML has 6 entries/robot.
- [ ] **T11 `mission.launch.py` (gazebo branch) — bring-up.** Reads `config:=`, calls world_builder, launches `gz sim -s -r --headless-rendering` (+ `gz sim -g` if `gazebo_gui`), `parameter_bridge --ros-args -p config_file:=runtime/bridge.yaml`; sets `GZ_SIM_RESOURCE_PATH=runtime/`, `GZ_IP=127.0.0.1`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`. Verify: `ros2 topic hz /rover_0/camera/image_raw` ≈ 10, `/rover_0/lidar/points_raw` ≈ 10, `/scout_0/odom` present; `gz stats` RTF ≥ 0.8 with 1+1 fleet (else drop lidar to 8 rings / camera to 5 Hz); **heightmap orientation check**: compare `/rover_0/odom` z at spawn vs `terrain.height_at` (±0.3 m) and probe a second known point by teleport (`/world/redshift/set_pose`); odom frame check (odom pose == spawn pose at t0, else add offset param).
- [ ] **T12 Gazebo physics tuning.** Teleop via `ros2 topic pub /rover_0/cmd_vel`: rover drives on terrain without jitter; stalls on > ~19° slope at 2.2 g (matches model) ; scout holds altitude ±5 cm over 10 s with zero command (if not, command z-velocity via altitude PID in planner). Tune jolt (persistent wrench on `rover_0::base_link`, then clear) so peak |roll| ∈ [32°, 45°] and the rover self-rights; record final numbers in `scene_events.py`.
- [ ] **T13 `world_interface_node.py`.** gz.transport13: `spawn_rockslide` → `EntityFactory` SDF (dynamic rocks, box collision/ellipsoid visual, 400 kg, dropped from 6 m) via `/world/redshift/create`; `jolt` → `/wrench/persistent` + timed `/wrench/clear`; publishes `/world/events`. Verify: rocks settle within 2 s within 1 m of layout; event JSON emitted.

### Phase 3 — Mock world (fallback)
- [ ] **T14 `mock_render.py`** (TDD): batched ray-march (vectorized numpy) against heightfield + rock ellipsoids for LiDAR (360×16) and depth (160×120 upsampled to 320×240), Lambert+haze RGB shading with regolith/rock/crystal albedo and sky gradient. Tests: ray to a known wall hits at expected range ±0.1 m; rock pixels gray, crystal pixels cyan; full frame < 60 ms.
- [ ] **T15 `mock_world_node.py`.** Unicycle rover with terrain-following z, pitch/roll from terrain normal, slip ∝ excess slope·g, torque-limit stall (same physics_model), 2nd-order roll dynamics for jolt; scout 3-D velocity with first-order lag; publishes all raw topics at the Gazebo rates + `/world/events`; handles rockslide/jolt. Add `scripts/check_topics.py` that asserts topic names/types/rates match between backends. Verify: parity script passes for both.

### Phase 4 — Autonomy nodes (verify in mock first, then Gazebo)
- [ ] **T16 `sensor_effects_node.py`.** Passthrough when clear; dust/seismic degradation per Decision 4; `/world/env_state`. Verify: `ros2 topic hz` unchanged; publishing an `anomaly` op visibly hazes `/rover_0/camera/image` (rqt or saved frame).
- [ ] **T17 `perception_node.py`.** Wraps vision.py; honours `mode_request`; publishes annotated JPEG (≤ edge-profile Hz), detections, uncertainty, scan2d; records stage timings. Scout variant: downward science/hazard detection. Verify: annotated frames show boxes on boulders/crystals; uncertainty ≈ 0.1–0.2 clear, ≥ 0.45 under dust.
- [ ] **T18 `mapping_coordinator_node.py`.** Fuses lidar points (+ subsampled depth) using odom pose + ROBOT_SPECS extrinsics (passed as params), science detections, fleet hazard marks; publishes map topics at 2–4 Hz. Verify: `percent` rises steadily; `/planet_map` viewable via `ros2 topic echo --once` stats; ridge cells classed hazard at 2.2 g.
- [ ] **T19 `agent_planner_node.py`.** Rover: FSM + frontier selection + A* + pure pursuit + breadcrumbs + slip from wheel_odom vs odom + EdgeBudget + traces; scout: ScoutFSM + altitude hold + scan servicing. Verify in mock: rover explores autonomously; publishing `spawn_rockslide` ⇒ REPLANNING trace within 8 s and new path avoids rocks; `anomaly` ⇒ CAUTION + LIDAR_ONLY; `jolt` ⇒ full recovery chain incl. scout flyover. Repeat in Gazebo.

### Phase 5 — Backend bridge
- [ ] **T20 `config_schema.py` + `mission_manager.py`** (TDD): pydantic validation/defaults/option catalog; manager writes `runtime/mission_config.yaml`, launches `ros2 launch …` with `start_new_session=True`, streams stdout to boot log, status machine (idle/launching/running/degraded/failed/stopped), readiness (odom + first frame per robot), Gazebo watchdog → auto mock fallback, abort = SIGINT group → 5 s → SIGKILL + `pkill -f "gz sim"`. Tests use a fake launch command.
- [ ] **T21 `ros_link.py` + `encoders.py`** (encoder TDD): background `MultiThreadedExecutor`; per-fleet subscriptions recreated on deploy; thread-safe cache; traces/events pushed into asyncio via `call_soon_threadsafe`; map layers → PNG base64; lidar → robot-frame downsample ≤ 1500.
- [ ] **T22 `app.py` + `scripts/smoke_test.py`.** REST + `/ws` broadcaster (15 Hz telemetry, per-client camera selection), static mount of `web_ui/dist`. Smoke test (httpx + websockets): deploy → telemetry for all robots (mock 30 s / gazebo 60 s) → percent +5 % in 30 s → rockslide ⇒ REPLANNING ≤ 8 s → anomaly ⇒ CAUTION ≤ 5 s and back to EXPLORING after it ends → tip ⇒ SAFE_HOLD ≤ 2 s … RESUMING ≤ 40 s → abort leaves no `gz` processes. Verify: `python3 backend_bridge/scripts/smoke_test.py --backend mock` exits 0.

### Phase 6 — Web UI (load `frontend-design:frontend-design` before T24)
- [ ] **T23 Scaffold + data layer.** Vite React-TS, Tailwind 4 via `@tailwindcss/vite`, exact-pinned deps (react 19.3.1, three 0.187.0, @react-three/fiber 9.8.2, drei 10.7.9, zustand 5.1.0, typescript ~5.9, vitest), `@fontsource` fonts; `ws.ts` (auto-reconnect), `store.ts`, `api.ts`, `geo.ts` (`sectorToSphere(x, y, elevNorm, L)`, span 90°, R 1, elevation exaggeration 0.04) + vitest for geo (corners/center, round-trip).
- [ ] **T24 PreLaunch + BootSequence.** Config cards (sector, gravity slider with live `/api/physics_preview`: max slope, limiting factor, tip alarm, speed), fleet steppers, edge profile cards, backend toggle, seed, **DEPLOY FLEET** → boot log terminal + subsystem checklist → auto-transition on first telemetry.
- [ ] **T25 Globe.** Fog planet (gray noise, graticule, scan sweep, fresnel rim); `SectorPatch` (256² plane, vertex shader → sphere + displacement from map texture; fragment: unknown ⇒ fog + cell grid, known ⇒ elevation ramp + hillshade, caution amber, hazard red pulse, permanent hazard hatched, science cyan, reveal glow from B channel); texture updated from `createImageBitmap(blob, {premultiplyAlpha:'none', colorSpaceConversion:'none'})`, NearestFilter; robot markers (rover chevrons, drone diamond + tether + scan cone), planned paths (animated dashed), trails, old-path red fade on replan, meteor streak + shockwave on rockslide event, hazard pulse rings, dust overlay; OrbitControls framed on the patch, follow-robot toggle.
- [ ] **T26 Sensor panels.** CameraFeed (robot tabs, LIVE/fps badge, mode badge), LidarRadar (canvas: points by height, range rings, corridor, local costmap crop rotated by yaw), ImuAttitude (artificial horizon, roll/pitch with tip-alarm bands, sparkline).
- [ ] **T27 Trace / HUD / controls.** DecisionTrace (Sensor → Reasoning → Action three-column rows, sensor chips, confidence bar, severity color, state pill), EdgeHud (host vs device latency, load bar, power W, perception Hz, uncertainty gauge with 0.45/0.8 marks), InjectBar (Spawn Meteor/Rockslide · Seismic/Dust Anomaly · Trigger Tip/Slip Failure, with cooldown and live result), FleetStatus, MissionProgress (% mapped, elapsed, science found, hazards). Verify: `npm run build` + `npx tsc --noEmit` + `npx vitest run` pass.

### Phase 7 — Integration & demo hardening
- [ ] **T28 `start_all.sh` / `stop_all.sh`.** Source ROS, force nvm Node first on PATH, colcon build if stale, `npm ci && npm run build` if `dist` stale, export GZ/ROS isolation vars, run uvicorn on :8000, print URL, trap → abort mission + kill gz. Flags: `--dev` (Vite on :5173 too), `--rebuild`. Verify: clean clone → `./start_all.sh` → UI at http://localhost:8000 from the Windows browser.
- [ ] **T29 End-to-end tuning + README.** Run smoke tests on both backends; screenshot the UI headlessly (`npx playwright screenshot`) at config screen and mid-mission to confirm rendering; tune so the 60-s demo beats land (deploy-to-moving < 20 s on Small/Medium, visible globe fill by 0:20); README with architecture diagram, judging-criteria mapping, demo runbook, troubleshooting (WSL rendering, mock fallback). Final `superpowers:requesting-code-review` pass.

## Verification (end-to-end)

1. `python3 -m pytest ros2_ws/src/exoswarm_sim/test ros2_ws/src/exoswarm_autonomy/test backend_bridge/tests -q` → all pass.
2. `cd ros2_ws && colcon build --symlink-install` clean; `cd web_ui && npm run build && npx tsc --noEmit && npx vitest run` clean.
3. `./start_all.sh` then `python3 backend_bridge/scripts/smoke_test.py --backend mock` and `--backend gazebo` → both exit 0 (assertion list in T22).
4. Manual demo pass in the browser at http://localhost:8000: config → deploy → globe fog lifts → rockslide reroute visible in camera bbox + trace + path line → dust ⇒ CAUTION/LIDAR-ONLY with uncertainty gauge rising → tip ⇒ SAFE_HOLD, reverse, red permanent hazard on globe, drone scan cone, RESUMING.
5. `./stop_all.sh` → `pgrep -f "gz sim"` empty.

## Demo runbook (60-s single take)

0:00 Config: Medium · 2.2 g (preview shows max slope 16.8°, torque-bound) · 1 rover + 1 drone · Orin Nano 15 W → DEPLOY · 0:10 globe fog lifts, traces stream · 0:20 **Spawn Rockslide** → meteor streak, boulder boxes in camera, "Re-planning" trace, path reroutes · 0:32 **Dust Anomaly** → haze, uncertainty ↑, CAUTION / LiDAR-only at 35 % speed · 0:42 **Tip/Slip** → IMU roll spike → SAFE-HOLD → reverse → hazard marked → drone scans → RESUMING · 0:58 % mapped + edge budget summary.

## Risks & mitigations

- **Gazebo RTF / WSL rendering** → headless-rendering verified; knobs: 4 ms step, sensor rates/resolution, shadows off; mock fallback (manual toggle + 45 s watchdog).
- **Heightmap/wheel physics jitter** → sphere wheel collisions, bullet detector; if still unstable, chassis-only box collision with VelocityControl + terrain-following z correction for the rover (documented fallback, decided in T12).
- **Jolt flips rover** → tune in T12 with bounded torque; FSM also triggers on gyro-rate spike so a smaller jolt still demonstrates recovery.
- **Scout altitude drift under 2.2 g with VelocityControl** → planner altitude PID on z velocity.
- **Odometry frame semantics** → verified in T11; offset param.
- **Python A* speed on 200² grids** → expansion cap per profile, coarse frontier pre-filter; measured latency feeds the edge HUD (honestly).
