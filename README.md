# Redshift: Autonomous Super-Earth Planetary Mapping Fleet

Embodied multi-agent robotics for the **Agentverse 2026 Robotics Track** ("Give an agent a body").

A Ground Rover and an Aerial Scout map an unexplored sector of a 2.2 g Super-Earth in **Gazebo Harmonic**. Everything they do runs locally on **ROS 2 Jazzy**, with no cloud calls:

- **Perception:** RGB-D + LiDAR + IMU, with uncertainty estimated from the sensor data itself.
- **Mapping:** one shared, gravity-aware fog-of-war map.
- **Planning:** A* that knows what slope the rover can climb at the selected gravity.
- **Recovery:** a tip/slip safe-hold routine that retreats, marks the hazard, and sends the drone to inspect it.

Each robot's onboard computer is simulated against a real edge-compute power budget. A FastAPI + rclpy bridge streams the fleet into a React + Three.js **Mission Control**. Its 3D globe reveals the sector as it is mapped. It also shows live sensor feeds and a Sense → Reason → Act trace for every decision. Inject buttons let you change the world in the middle of a run.

![Mission Control during a rockslide re-plan](docs/media/mission.jpg)

## Quick start

```bash
./start_all.sh            # builds what is stale, then serves Mission Control
# open http://localhost:8000 → configure → DEPLOY FLEET
./stop_all.sh             # or Ctrl+C in the start_all terminal
```

`start_all.sh`:
- sources ROS 2;
- puts the nvm Node ahead of any Windows `npm` that WSL exposes;
- rebuilds the colcon workspace and the web UI only when their sources changed;
- keeps Gazebo and DDS traffic on localhost;
- runs the backend on `:8000`.

Missions are launched from the UI. Ctrl+C aborts the running mission and reaps every simulator process.

| Flag | Effect |
|---|---|
| `--dev` | also run the Vite dev server with hot reload on `:5173` (proxied to the backend) |
| `--rebuild` | force the colcon and web builds |
| `--ui-demo` | no ROS needed: serve the UI against the scripted fake backend in `web_ui/dev/` |
| `--port N`, `--host H` | bind elsewhere (`--host 0.0.0.0` to reach it from another machine) |

## The 60-second demo

| Time | Action | What you see |
|---|---|---|
| 0:00 | Medium sector · **2.2 g** · 1 rover + 1 drone · Orin Nano 15 W → **Deploy fleet** | Gravity preview: max climbable slope 16.9° (limited by wheel torque), tip alarm 28.9°, 0.83 m/s |
| 0:10 | (autonomous) | Boot checklist goes green in ~4 s; fog lifts on the globe as both robots explore; traces stream |
| 0:20 | **Spawn Meteor / Rockslide Ahead** | Meteor streak, then boulder boxes in the camera; `Path blocked` trace about 1 s later; the route re-plans around the slide |
| 0:32 | **Trigger High-G Seismic / Dust Anomaly** | Camera hazes; uncertainty passes 0.45 → **CAUTION**: LiDAR-only at 35 % speed; fused navigation returns when the dust clears |
| 0:42 | **Trigger Tip/Slip Failure** | A real roll jolt (the IMU measures it) → **SAFE HOLD** → reverse along breadcrumbs → permanent no-go zone on the map → drone flies over to scan it → **RESUMING** |
| 0:58 | (wrap-up) | % mapped, science targets, and the edge budget (latency, load, watts, throttle) |

## How it maps to the judging criteria

**Operational autonomy, edge budget, failure recovery (35 %)**
- *Gravity-aware physics.* `physics_model.py` derives the climbable slope from wheel torque, soil traction (Mohr–Coulomb) and the static tip angle at the selected g.
  - The same torque limit is written into the Gazebo wheel joints, so the simulated rover stalls exactly where the planner says it cannot climb.
  - At 1 g the limit is about 37° (traction-bound). At 2.2 g it is 16.9° (torque-bound).
- *Edge budget.* `edge_budget.py` scales the measured host latency by an edge profile: Jetson Orin Nano/NX/AGX or Raspberry Pi 5.
  - It reports the device latency, load and power.
  - It throttles the perception rate by 25 % above 90 % load and restores it below 60 %, with dwell times and prediction so it doesn't oscillate.
- *Safe-hold recovery (`agent_fsm.py`).*
  - **Triggers:** tip (roll or pitch past the alarm angle), slip (wheel odometry vs. measured motion), or sustained high uncertainty.
  - **Response:** halt, then reverse along verified-safe breadcrumbs, then mark a permanent hazard on the shared map, then send the scout to scan it, then resume.

**Perception feed + traceable decisions (25 %)**
- *Local OpenCV perception (`vision.py`).*
  - Boulders: depth-geometry bumps confirmed by colour.
  - Craters: depth pits.
  - Science crystals: cyan in HSV.
  - Steep patches: local plane fits.
  - Rover only: a LiDAR corridor check that rejects sparse dust backscatter.
- *Uncertainty* is computed from the data, never read from a dust flag:
  - camera: sharpness, contrast and noise vs. a clear-sky baseline;
  - LiDAR: return ratio, near-field spurious returns, range noise;
  - IMU: vibration.
- *Decision traces.* Every decision is a trace (`trace.py`) with the sensors used, the input, the reasoning, a confidence and the action. Mission Control shows them as Sense → Reason → Act rows.

**Unscripted mid-run change + re-planning (25 %)**
- *Rockslide.* Boulders drop across the rover's *current* planned path, 7 m ahead (closer when less route remains, so they never land past its end). LiDAR obstacle evidence blocks the path → visible `REPLANNING` → a new A* route (+Δ m, nodes, ms).
- *Dust / seismic anomaly.* A sensor-effects layer hazes, blurs and adds noise to RGB; drops returns and adds backscatter to LiDAR; vibrates the IMU. The rover's FSM switches to LiDAR-only, with slower speed and extra inflation.

**Pre-launch configuration and fog-of-war globe**
- *Configuration:* sector size, gravity 1–3 g (feeds both the Gazebo `<gravity>` and the planner), fleet of 1–2 rovers + 0–1 drone, edge profile, simulator backend, seed.
- *Globe:* the sector is a patch on a fogged planet. A shader decodes the live `/planet_map` layers: elevation, hazard / caution / no-go, science, and newly revealed cells. Robot markers, planned paths, trails and scan cones are drawn on top.

## Architecture

```
 Web UI (React 19 / R3F) ──REST /api/*──► FastAPI backend ──spawns──► ros2 launch redshift_sim mission.launch.py
        ▲   one WebSocket /ws               │ rclpy RosLink (background thread)            │
        └──── telemetry · camera · lidar · map · trace · event ◄──────────────────────────┤
                                                                                           ▼
   ┌─ gazebo: gz sim (headless rendering) + ros_gz parameter_bridge + world_interface_node
   └─ mock:   mock_world_node (kinematics + numpy ray-cast LiDAR / depth / RGB, same topics)
          │ raw: /<r>/{camera/image_raw, camera/depth_raw, lidar/points_raw, imu_raw, odom, wheel_odom}
          ▼
   sensor_effects_node (dust / seismic) → /<r>/{camera/image, camera/depth, lidar/points, imu}
          ▼                                      ▼
   perception_node (per robot)        mapping_coordinator_node → /planet_map (+ elevation, layers, stats)
          │ detections · uncertainty · annotated JPEG           ▲ /fleet/hazard_marks
          ▼                                                     │
   agent_planner_node (rover | scout) → cmd_vel · planned_path · agent_state · edge_telemetry ·
                                        /agent_decision_trace · /fleet/{claims, scan_request, scan_complete}
```

**Design rules**
- *No ground truth reaches the autonomy stack.* It sees only degraded sensor topics and odometry.
- *The pure algorithms are unit-tested, and the ROS nodes are thin wrappers around them.* The algorithms live in `physics_model`, `edge_budget`, `grid_map`, `planning`, `agent_fsm`, `vision`, `trace`, `terrain_gen`, `scene_events`, `mock_render` and `mock_dynamics`.
- *The world is regenerated per deploy from a seed:* heightmap, regolith texture, boulders, craters, ridges and science crystals. Robots are embedded at their spawn poses.
- *Messages:* JSON over `std_msgs/String` for traces, detections, agent state and fleet coordination; typed messages for sensors, maps, paths and `cmd_vel`. The full contract is in [`docs/protocol.md`](docs/protocol.md).

### Two world backends, one topic contract

**Gazebo Harmonic** is the primary backend: DART physics with Bullet collision, `gpu_lidar`, RGB-D and IMU sensors, DiffDrive and VelocityControl. Rockslide and jolt commands run through gz-transport.

**The mock world** is the fallback. The backend switches to it automatically if Gazebo has not delivered a camera frame from every robot within 45 s, or if a Gazebo process exits during start-up. You can also pick it in the UI. It:
- renders the same terrain with vectorized numpy ray casting (camera ≈ 30 ms, LiDAR ≈ 11 ms per frame);
- reuses the same physics model for torque stalls;
- runs the jolt as a closed-loop roll mode.

`ros2_ws/src/redshift_sim/scripts/check_topics.py` checks the mock against a reference capture from Gazebo (`gz_signature.json`). It compares names, types, rates, frame ids, encodings, camera intrinsics, point-cloud layout, ring order and the no-return encoding. The mock currently passes (`PARITY OK`).

## Requirements

Tested on Ubuntu 24.04 under WSL2 with WSLg:
- ROS 2 Jazzy, with `ros_gz_bridge`, `ros_gz_sim` and `cv_bridge`;
- Gazebo Harmonic (gz-sim 8.15), whose `gz.transport13` / `gz.msgs10` Python bindings ship with it;
- Python 3.12 with numpy, OpenCV 4.6, SciPy, FastAPI, uvicorn, pydantic 2 and PyYAML. The smoke test also needs `httpx` and `websockets`;
- Node.js ≥ 20 (developed on v25 via nvm). Web dependencies are exact-pinned in `web_ui/package.json`, and fonts are bundled, so there are no CDN calls.

Without Gazebo, `start_all.sh` warns and only the mock backend works. Without ROS, use `./start_all.sh --ui-demo`.

## Testing

```bash
python3 -m pytest -q                                   # 129 unit tests: autonomy, sim, backend
(cd web_ui && npm run typecheck && npm test)           # TypeScript + 20 vitest tests
python3 backend_bridge/scripts/smoke_test.py --backend mock     # end-to-end against a running backend
python3 backend_bridge/scripts/smoke_test.py --backend gazebo
```

The smoke test drives the same REST and WebSocket traffic as the UI:
1. deploy, then confirm telemetry, camera, LiDAR and map streams and exploration progress;
2. rockslide → `path_blocked` + `replanned`;
3. dust → CAUTION / LiDAR-only, then recovery;
4. tip → `safe_hold`, `recovery_reverse`, `hazard_marked`, `scout_tasked`, `scan_complete`, `resume`, and a permanent hazard on the map;
5. unique trace ids;
6. abort, with no simulator process left.

Latest runs: 23/23 on mock in five consecutive runs, and 24/24 on Gazebo, which has one extra check (no fallback to mock). Typical Gazebo timings: running in about 4–5 s; rockslide → `path_blocked` in about 1 s; tip → `safe_hold` in 0.3 s and `resume` after about 31 s.

One earlier Gazebo run missed `safe_hold` after the jolt. The same jolt then passed 6/6 when repeated on its own. When a rockslide, dust or jolt check fails, the smoke test now prints the rover's state, the world event and what it traced, so a repeat can be diagnosed.

## Repository layout

```
start_all.sh · stop_all.sh            one-command bring-up / teardown
docs/protocol.md                      topic, JSON, REST and WebSocket contracts
ros2_ws/src/redshift_sim/             world generation, Gazebo models + launch, world interface, sensor effects,
                                      mock world (mock_render, mock_dynamics, mock_world_node), check_topics.py
ros2_ws/src/redshift_autonomy/        physics_model, edge_budget, grid_map, planning, agent_fsm, vision, trace
                                      + perception_node, mapping_coordinator_node, agent_planner_node
backend_bridge/                       FastAPI app, rclpy link, mission manager (launch, watchdog, fallback, abort),
                                      encoders, config schema, tests, scripts/smoke_test.py
web_ui/                               React + Three.js Mission Control; dev/ holds the scripted fake backend
```

## Troubleshooting

- **The page doesn't load from the Windows browser.** WSL2 forwards `localhost` by default. If yours doesn't, run `./start_all.sh --host 0.0.0.0` and use the WSL IP address.
- **"port 8000 is already in use".** A previous run is still up. Run `./stop_all.sh`, or pass `--port`.
- **Gazebo is slow or never produces frames.** The backend falls back to the mock world after 45 s without camera frames (or at once if Gazebo exits during start-up), and says so in the boot log and as an event. Choose *Mock physics* in the UI to skip Gazebo entirely.
- **`npm` resolves to `/mnt/c/...`.** That is the Windows npm. `start_all.sh` prefers the newest nvm Node; install one with `nvm install 22`.
- **Simulator processes are left over after a crash.** Run `./stop_all.sh`. Its patterns only match this checkout's `runtime/` world and `install/` nodes, the `redshift_sim` launch and the bridge's `redshift_gz_bridge` node, so other Gazebo or ROS work on the machine is left alone.

## License

See [LICENSE](LICENSE).
