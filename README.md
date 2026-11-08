# Redshift: Autonomous Super-Earth Planetary Mapping Fleet

Embodied multi-agent robotics for the **Agentverse 2026 Robotics Track** ("Give an agent a body").

A fleet of two rovers and an aerial drone maps an unexplored sector of a 2.2 g Super-Earth in **Gazebo Harmonic**: *Scout Rover 1* (`rover_0`), *Heavy Recovery Rover 2* (`rover_1`) and a *Survey Drone* (`scout_0`). Everything that senses, decides and moves runs locally on **ROS 2 Jazzy**:

- **Perception:** RGB-D + LiDAR + IMU, with uncertainty estimated from the sensor data itself.
- **Mapping:** one shared, gravity-aware fog-of-war map.
- **Planning:** A* that knows what slope the rover can climb at the selected gravity.
- **Self-diagnosis:** each rover estimates its own health from IMU shocks, motor-controller diagnostics and a commanded-vs-measured wheel-speed check, and integrates its battery from solar input and its drive and compute load.
- **Recovery:**
  - *Tip/slip:* a safe-hold routine retreats, marks the hazard and sends the drone to inspect it.
  - *Critical damage:* a crippled rover holds safe and broadcasts an SOS. The recovery rover drives over and docks once its own camera, depth and LiDAR confirm the damaged unit. It runs a field repair, and both rovers go back to mapping.
- **Advisory agent brain:** after the local layer has acted, Google's **Gemma** (`gemma-4-26b-a4b-it`, in the cloud) reviews each hazard, low-confidence spell, damage report, SOS and typed operator command. It may only make a robot safer: slow it, hold it, refuse a command, or move the drone. Without a key or a network, local rules answer in the same format, and the HUD says which one decided.

Each robot's onboard computer is simulated against a real edge-compute power budget. A FastAPI + rclpy bridge streams the fleet into a React + Three.js **Mission Control**. Its 3D globe reveals the sector as it is mapped. It also shows live sensor feeds, a Sense → Reason → Act trace for every decision, and the agent brain panel with an operator command box. Inject buttons let you change the world in the middle of a run.

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

### Simulation speed

Pick **0.5× / 1× / 1.5× / 2×** on the pre-launch screen, or switch it live from the Mission Control top bar, which
also shows the achieved real-time factor (`RTF`, amber when the simulator can't keep up). Every autonomy node runs on
simulated time (`use_sim_time`, `/clock`). So holds, timeouts, dedup windows and sensor rates are counted in simulated
seconds, and a mission plays out the same way at any speed, only faster or slower. 2× is handy for recording; 0.5× for
watching a recovery closely.

- **Gazebo** paces physics with its `real_time_factor`; a live change goes through `/world/redshift/set_physics`.
- **The mock world** is a lockstep clock: time waits for any sensor frame still rendering, so sensor rates hold per
  simulated second and an overloaded renderer slows the clock instead of dropping frames. Its ray casting runs in
  worker processes.

Measured on the development laptop (8 cores, RTX 5050), with the full autonomy stack:

| Fleet | Gazebo | Mock |
|---|---|---|
| 2 rovers + drone (default) | 1× holds (0.99–1.03); 2× requested → 1.24× | 1× holds (1.0); 2× requested → 1.29× (the world alone holds 2×) |
| 1 rover + drone | 2× holds (1.8–2.05) | 1× holds |

Above what the machine can render, the mission still runs correctly, only slower than requested.

## The 60-second demo

The brief asks for one continuous take of at most a minute, with perception on screen next to the actions and at least one unscripted change. At 1× the full rescue does not fit in the same minute as the other beats, so there are two takes.

**Take A: adaptation, from deploy to wrap-up**

| Time | Action | What you see |
|---|---|---|
| 0:00 | Medium sector · **2.2 g** · 2 rovers + 1 drone · Orin Nano 15 W → **Deploy fleet** | Gravity preview: max climbable slope 16.9° (limited by wheel torque), tip alarm 28.9°, 0.83 m/s |
| 0:10 | (autonomous) | Boot checklist goes green in ~5 s; fog lifts on the globe as the three robots explore; traces stream |
| 0:20 | **Spawn Meteor / Rockslide Ahead** | Meteor streak, then boulder boxes in the camera; `Path blocked` trace about 1 s later; the route re-plans around the slide |
| 0:32 | **Trigger High-G Seismic / Dust Anomaly** | Camera hazes; uncertainty passes 0.45 → **CAUTION**: LiDAR-only at 35 % speed; fused navigation returns when the dust clears |
| 0:42 | **Trigger Tip/Slip Failure** | A real roll jolt (the IMU measures it) → **SAFE HOLD** → reverse along breadcrumbs → permanent no-go zone on the map → drone flies over to scan it → **RESUMING** |
| 0:58 | (wrap-up) | % mapped, science targets, and the edge budget (latency, load, watts, throttle) |

**Take B: the critical failure mode, fleet self-rescue.** Start recording with the fleet already exploring. Trigger the strike while Rover 2 is near Rover 1: most of the rescue is the drive over, about 57 s at 1× from 22 m away, and 150 s from 45 m.

| Time (22 m gap, Gazebo 1×) | Action | What you see |
|---|---|---|
| 0:00 | Select ROVER_2's camera; **Trigger Critical Damage on Rover 1** | Debris strike: Rover 1's IMU shock and motor faults in the live sensor panel; health below 15 % → SAFE-HOLD → **SOS** banner |
| 0:01 | (autonomous) | Rover 2 drops its frontier: AUTONOMOUS RESCUE MODE and a route to a stand-off beside Rover 1 |
| 0:40 | (autonomous) | Stand-off reached. Rover 2's camera boxes the damaged unit, **[PERCEIVED: DAMAGED UNIT ROVER_1 \| DOCKING FOR REPAIR]**, confirmed by depth and LiDAR; it creeps in to 1.45 m |
| 0:48 | (autonomous) | Docked → REPAIRING: Rover 2's repair tool works on Rover 1's drivetrain (4 s, only within 1.6 m) |
| 0:52 | (autonomous) | Rover 1's own motor diagnostics read nominal → health back to ~90 % |
| 0:57 | (autonomous) | Rover 2 backs off 2 m; both rovers resume mapping |

**Operator commands (fit into either take).** Type into the *Agent brain* panel, or use its presets:
- *drive at full speed into the rockslide* → **REJECT_UNSAFE_COMMAND** about 2 s later, with Gemma's one-sentence reason;
- *slow down to 0.2 m/s* → the planner applies a 0.20 m/s cap for 60 s, shown as *applied* in the panel;
- *stop* → the rover stops at once: a stop never waits for the cloud;
- *resume at full speed* → the operator's cap is lifted and the local limits apply again.

## How it maps to the judging criteria

Weights are the official Agentverse 2026 Robotics Track ones.

**Perception-driven action (25 %): every decision traces to what was perceived**
- *Local OpenCV perception (`vision.py`).*
  - Boulders: depth-geometry bumps confirmed by colour.
  - Craters: depth pits.
  - Science crystals: cyan in HSV.
  - Steep patches: local plane fits.
  - Rover only: a LiDAR corridor check that rejects sparse dust backscatter.
- *Teammates are perceived, not assumed.* The fleet link says where the damaged rover should be. The rescuer only switches to perception-guided docking once its own camera box is confirmed by depth (or by LiDAR returns) at the expected range: `[PERCEIVED: DAMAGED UNIT ROVER_1 | DOCKING FOR REPAIR]`.
- *Uncertainty* is computed from the data, never read from a dust flag:
  - camera: sharpness, contrast and noise vs. a clear-sky baseline, which follows a change of scenery only while the LiDAR independently confirms clear air;
  - LiDAR: return ratio, near-field spurious returns, range noise;
  - IMU: vibration.
- *Health from the rover's own sensors* (`rover_health.py`), never from the world's damage state:
  - IMU shocks (armed only once the accelerometer reads steady gravity, so sensor start-up is not an impact);
  - motor-controller diagnostics, cross-checked against commanded vs. measured wheel speed.
- *Decision traces.* Every decision is a trace (`trace.py`) with the sensors used (`CAM`, `DEPTH`, `LIDAR`, `IMU`, `ODOM`, `MAP`, `FLEET`, `DIAG`), the input, the reasoning, a confidence and the action. Mission Control shows them as Sense → Reason → Act rows next to the live camera, depth, LiDAR and IMU readings. Each brain decision adds a row tagged `GEMMA` or `LOCAL`, with the event that triggered it, and a `brain_cap` row when a planner applies or drops its cap.

**Adaptation to change (25 %): scene changes mid-run, the fleet visibly re-plans**
- *Rockslide.* Boulders drop across the rover's *current* planned path, 7 m ahead. With less route left they land closer, so never past its end; within 4 m of its goal they wait for its next route. LiDAR obstacle evidence blocks the path → visible `REPLANNING` → a new A* route (+Δ m, nodes, ms).
- *Dust / seismic anomaly.* A sensor-effects layer hazes, blurs and adds noise to RGB; drops returns and adds backscatter to LiDAR; vibrates the IMU. The rover's FSM switches to LiDAR-only, with slower speed and extra inflation, and returns to fused navigation when its sensors say the air has cleared.
- *Debris strike.* A real force on the chassis (the IMU measures it) plus drivetrain faults drops health below 15 %. The damaged rover goes SAFE-HOLD → SOS. The recovery rover drops its frontier, re-plans to a stand-off beside it, and docks on perception.

**Operational autonomy within an edge budget (15 %)**
- *Edge budget.* `edge_budget.py` scales the measured host latency by an edge profile: Jetson Orin Nano/NX/AGX or Raspberry Pi 5.
  - It reports the device latency, load and power, and that power feeds the battery model.
  - It throttles the perception rate by 25 % above 90 % load and restores it below 60 %, with dwell times and prediction so it doesn't oscillate.
- *Gravity-aware physics.* `physics_model.py` derives the climbable slope from wheel torque, soil traction (Mohr–Coulomb) and the static tip angle at the selected g.
  - The same torque limit is written into the Gazebo wheel joints, so the simulated rover stalls exactly where the planner says it cannot climb.
  - At 1 g the limit is about 37° (traction-bound). At 2.2 g it is 16.9° (torque-bound).
- *Sensible when unsure (`agent_fsm.py`).* High uncertainty → CAUTION; tip (roll or pitch past the alarm angle), slip (wheel odometry vs. measured motion) or critical health → SAFE-HOLD. No safety reaction waits on the network.
- *An advisory brain that knows its place* (`gemma_agent_brain.py`, `brain_bridge.py`, `fleet_brain_node.py`):
  - **Authority stays on board.** By the time an event reaches the brain, the FSM, A* and safety layer have already acted. `validate()` keeps an answer only if it makes things safer:
    - the action must be one the context allows;
    - speeds are caps, clamped to the local limit, and the planner takes the lower of its own speed and the cap, only while following a route or docking;
    - the drone's height is kept between 6 and 20 m;
    - a command the local check finds unsafe always ends as a rejection, and a stop is always honoured.
  - **Where it runs, said plainly.** Gemma is called over REST on Google AI, measured at a 1.6–2.2 s round-trip for ~400 tokens in and ~60 out. The 26B model needs ~14.5 GB, so it does not fit the 8 GB Orin Nano. On that profile the local rules *are* the on-device brain. The HUD shows a labelled roofline estimate for each profile, e.g. ~1.1 s on an AGX Orin; it is never presented as a measurement.
  - **Honest HUD.** For each decision the panel shows which brain decided (Gemma, or the local fallback and why), the measured round-trip and tokens, what validation clamped, the caps in force with *applied* / *not applied*, and the result of every operator command.
  - **Degrades, never blocks.** The cloud call runs on worker threads, with de-duplication, back-off on rate limits and an answer deadline. Missing key, network error, timeout or an invalid answer → the local rules decide. `ai_brain: false` turns the advisor off entirely.

**Relevance & impact (20 %): the most critical failure mode, handled on screen**
- On a 2.2 g world with no one to send, a crippled rover means the mission is over. Here the fleet rescues itself:
  1. **SOS** (1 Hz heartbeat with position, health and fault codes);
  2. **rescue transit** (A* to a stand-off cell, with the teammate as an obstacle);
  3. **perception-guided docking** (creeps in on the fused range and stops at 1.45 m);
  4. **field repair**, verified by the damaged rover's own motor diagnostics;
  5. both rovers **resume** mapping.
- Mission Control shows the whole sequence: the SOS banner, the transit / docking / repairing / done track, the lock-on box in the rescuer's camera, and the damaged rover's health, faults and battery live.
- The tip/slip failure ends with the same honesty: reverse along breadcrumbs, mark a permanent no-go zone, send the drone to verify, then resume.

**Ambition & technical quality (15 %)**
- Two simulator backends behind one topic contract (Gazebo Harmonic, and a numpy mock world rendered in worker processes), checked against each other by `check_topics.py`.
- Everything runs on simulated time at a selectable speed.
- 428 Python and 98 TypeScript unit tests. The end-to-end smoke test drives every injected change, the full rescue and the operator commands on both backends, with the live Gemma advisor.

**Pre-launch configuration and fog-of-war globe**
- *Configuration:* sector size, gravity 1–3 g (feeds both the Gazebo `<gravity>` and the planner), fleet of 1–2 rovers + 0–1 drone (default 2 + 1), edge profile, simulator backend, seed, simulation speed, agent brain on/off (it warns when no API key is set).
- *Globe:* the sector is a patch on a fogged planet. A shader decodes the live `/planet_map` layers: elevation, hazard / caution / no-go, science, and newly revealed cells. Robot markers, planned paths, trails, scan cones, SOS distress rings and the rescue tether are drawn on top.

## Architecture

```
 Web UI (React 19 / R3F) ──REST /api/*──► FastAPI backend ──spawns──► ros2 launch redshift_sim mission.launch.py
        ▲   one WebSocket /ws               │ rclpy RosLink (background thread)            │
        └──── telemetry · camera · lidar · map · trace · event ◄──────────────────────────┤
                                                                                           ▼
   ┌─ gazebo: gz sim (headless rendering) + ros_gz parameter_bridge + world_interface_node (commands, rover hardware)
   └─ mock:   mock_world_node (kinematics + numpy ray-cast LiDAR / depth / RGB in worker processes, same topics)
          │ raw: /<r>/{camera/image_raw, camera/depth_raw, lidar/points_raw, imu_raw, odom, wheel_odom, hw_status}
          ▼
   sensor_effects_node (dust / seismic) → /<r>/{camera/image, camera/depth, lidar/points, imu}
          ▼                                      ▼
   perception_node (per robot)        mapping_coordinator_node → /planet_map (+ elevation, layers, stats)
          │ detections (incl. perceived teammates) · uncertainty  ▲ /fleet/hazard_marks
          │ annotated JPEG · depth colormap JPEG                  │
          ▼                                                       │
   agent_planner_node (rover | scout) → cmd_vel · planned_path · agent_state (health, rescue, brain cap) · edge_telemetry ·
        rover_health + agent_fsm        /agent_decision_trace · /fleet/{claims, scan_request, scan_complete, sos, rescue}
          ▲ /brain/decision (caps: min with the local speed)          │ traces · agent_state
          │                                                           ▼
   fleet_brain_node (one per mission) ◄── /operator/command (POST /api/command from the UI)
        brain_bridge + gemma_agent_brain → Gemma on Google AI (advisory, worker threads) or local rules → /brain/status
```

**Design rules**
- *No ground truth reaches the autonomy stack.* It sees only degraded sensor topics and odometry.
- *The pure algorithms are unit-tested, and the ROS nodes are thin wrappers around them.* The algorithms live in `physics_model`, `edge_budget`, `grid_map`, `planning`, `agent_fsm`, `rover_health`, `vision`, `trace`, `gemma_agent_brain`, `brain_bridge`, `terrain_gen`, `scene_events`, `rover_hw`, `mock_render`, `render_pool` and `mock_dynamics`.
- *The world is regenerated per deploy from a seed:* heightmap, regolith texture, boulders, craters, ridges and science crystals. Robots are embedded at their spawn poses.
- *Messages:* JSON over `std_msgs/String` for traces, detections, agent state and fleet coordination; typed messages for sensors, maps, paths and `cmd_vel`. The full contract is in [`docs/protocol.md`](docs/protocol.md).

### Two world backends, one topic contract

**Gazebo Harmonic** is the primary backend: DART physics with Bullet collision, `gpu_lidar`, RGB-D and IMU sensors, DiffDrive and VelocityControl. Rockslide, jolt and debris-strike commands run through gz-transport. The simulated rover hardware (`rover_hw.py`: per-wheel drivetrain health and fault codes, motor current, solar output) scales each rover's wheel commands, so a damaged rover crawls and veers.

**The mock world** is the fallback. The backend switches to it automatically if Gazebo has not delivered a camera frame from every robot within 45 s, or if a Gazebo process exits during start-up. You can also pick it in the UI. It:
- renders the same terrain, boulders, crystals and robot bodies with vectorized numpy ray casting (camera ≈ 30 ms, LiDAR ≈ 12 ms per frame). Rendering runs in a pool of worker processes (`render_pool.py`), because one process could not hold 1× for the two-rover fleet. Output is bit-identical to rendering in-process;
- reuses the same physics model for torque stalls;
- runs the jolt as a closed-loop roll mode.

`ros2_ws/src/redshift_sim/scripts/check_topics.py` checks the mock against a reference capture from Gazebo (`gz_signature.json`). It compares names, types, rates, frame ids, encodings, camera intrinsics, point-cloud layout, ring order and the no-return encoding. The mock currently passes (`PARITY OK`).

## Requirements

Tested on Ubuntu 24.04 under WSL2 with WSLg:
- ROS 2 Jazzy, with `ros_gz_bridge`, `ros_gz_sim` and `cv_bridge`;
- Gazebo Harmonic (gz-sim 8.15), whose `gz.transport13` / `gz.msgs10` Python bindings ship with it;
- Python 3.12 with numpy, OpenCV 4.6, SciPy, FastAPI, uvicorn, pydantic 2 and PyYAML. The smoke test also needs `httpx` and `websockets`;
- Node.js ≥ 20 (developed on v25 via nvm). Web dependencies are exact-pinned in `web_ui/package.json`, and fonts are bundled, so there are no CDN calls.

Optional: a Gemini API key, as `GEMINI_API_KEY=...` in `.env` at the repo root (git-ignored) or in the environment, for the cloud advisor. The client is plain `urllib`, with no SDK. The key is read once and never logged or published. Without a key, the local rules answer and the HUD says so. `python3 backend_bridge/scripts/gemma_probe.py` checks the key and model with three sample events.

Without Gazebo, `start_all.sh` warns and only the mock backend works. Without ROS, use `./start_all.sh --ui-demo`.

## Testing

```bash
python3 -m pytest -q                                   # 428 unit tests: autonomy, brain, sim, backend
(cd web_ui && npm run typecheck && npm test)           # TypeScript + 98 vitest tests
python3 backend_bridge/scripts/smoke_test.py --backend mock     # end-to-end against a running backend
python3 backend_bridge/scripts/smoke_test.py --backend gazebo
python3 backend_bridge/scripts/smoke_test.py --backend gazebo --rovers 1 --speed 2   # 1 rover, 2× simulation speed
python3 backend_bridge/scripts/smoke_test.py --backend mock --brain off               # local rules only, no cloud
```

The smoke test drives the same REST and WebSocket traffic as the UI (default fleet: 2 rovers + drone; `--rovers 1` for one):
1. deploy, then confirm telemetry, camera, LiDAR and map streams, exploration progress and the brain status;
2. rockslide → `path_blocked` + `replanned`, then a brain decision on the new hazard;
3. dust → CAUTION / LiDAR-only, then recovery;
4. tip → `safe_hold`, `recovery_reverse`, `hazard_marked`, `scout_tasked`, `scan_complete`, `resume`, and a permanent hazard on the map;
5. nothing so far was read as critical damage;
6. with two rovers, a debris strike on Rover 1:
   - the world event and the perceived health (< 15 %, fault codes);
   - `critical_damage` → `sos` → Rover 2 `rescue_start` → `rescue_dock` → `docked`, with the perception lock-on label, and a brain decision on the damage;
   - Rover 1 `repaired` → `repair_complete` → both `resume`;
   - the world accepted the repair within 1.6 m, and both rovers are mapping again;
7. operator commands: an unsafe one is rejected; a slow-down sets a cap of at most 0.2 m/s that the rover's planner reports as applied; a stop is decided immediately and the rover halts; *resume at full speed* clears the cap;
8. the achieved real-time factor matches `--speed`, mission time follows simulated time, and a live speed change takes effect;
9. unique trace ids;
10. abort, with no simulator process left.

The brain checks pass whichever brain answered (Gemma, or the local rules with `--brain off`, no key or a rate limit). Each check says which brain decided, the model, the measured round-trip and the tokens.

Timeouts are in simulated seconds: below 1× they are stretched to wall time.

Latest runs (2 rovers + drone, 1×, with the live Gemma advisor):
- **Gazebo 50/50.**
  - Rockslide → `path_blocked` in 1.5 s, and Gemma's `SLOW_AND_REPLAN` (cap 0.30 m/s) after its 2.1 s round-trip.
  - Dust → CAUTION in 3.0 s, back to fused navigation 24 s after the inject. Tip → `resume` after 55 s.
  - Strike → `sos` in 2.5 s and Gemma's `DEPLOY_RECOVERY_ROVER` at 2.3 s. The rescuer drove 37 m, reached its stand-off after 84 s, docked at 1.45 m after 99 s, and both rovers were mapping again after 112 s.
  - Operator commands: the unsafe one was rejected in 1.7 s; the 0.20 m/s cap was applied; the stop was decided in 0.01 s and the rover was at 0.03 m/s 0.3 s later.
  - 12 brain decisions (10 by Gemma), median measured round-trip 1.94 s. Pacing: RTF 0.77 at 1× (the check needs ≥ 0.75).
- **Mock 49/49** (Gazebo has one extra check: no fallback to mock). Rescue docked after 85 s, both rovers back after 96 s. *Resume at full speed* → Gemma `SLOW_AND_REPLAN` at 0.83 m/s and the operator cap lifted. 12 decisions (11 by Gemma), median 1.96 s. Also 33/33 with one rover.

Known intermittent issue: the following Gazebo run failed 2 of 41 checks, for one reason. When the strike landed, Rover 2 was itself recovering from a slip 1.3 m from the sector's south edge. It reversed only 1.4 m up a 9° slope in 15 s, marked a 1 m no-go zone and waited for the drone. It then took the SOS and began to turn toward Rover 1. Turning in place on that slope at 2.2 g, it skidded and slid 1.5 m back downhill, with its wheels stopped, into the centre of its own zone. A planner whose start lies inside a lethal core finds no way out, so every retry ended in `rescue_noroute`, and Rover 1 was never repaired. The operator slow-down check failed as a consequence: Rover 1 was still disabled, so its limit was 0 m/s and the local check correctly refused 0.2 m/s. The smoke test already strikes only when a stand-off is reachable from where the rescuer is (`rescue_room`; 15 such cells here), and that gate cannot foresee the slide.

RTF with the brain on was 0.77–0.87 at 1× on Gazebo and 0.88–0.91 on the mock, against 0.99–1.03 in earlier runs. Whether the brain node or other load on the machine caused the drop has not been isolated.

At 2× with 2 rovers + drone, every behaviour check passes, but the pacing check fails on both backends: Gazebo reaches 1.24× and the mock 1.29× (see *Simulation speed*).

When a check fails, the smoke test prints the rover's state, the world event and what the robots traced. `--dump run.jsonl` also writes every trace and event, a 1 Hz pose / state sample per robot, and planet-map snapshots, so a failed repeat can be diagnosed afterwards. Repeat harnesses found and fixed these intermittent failures (Gazebo-only unless marked):
- **A false critical-damage report at start-up.** The Gazebo IMU starts at 0 and then spikes to 40 g. Shock detection now arms only once the IMU reads steady gravity.
- **A false critical-damage report after the tip jolt.** The chassis is briefly airborne, then one 65 g contact spike as it lands. Only acceleration above gravity, held for at least 2 samples, now counts as an impact.
- **A tip jolt the 10 Hz FSM missed.** The FSM now sees the peak attitude and rate of each 100 ms IMU window; 6/6 repeats at 2× now trigger.
- **A rescuer stuck re-planning.** A route that leaves through the rover's own inflation ring was flagged as blocked by that ring on every map update. Only cells that became lethal after the route was planned now block it.
- **A stand-off the planner always rejected.** The fixed 2.2 m stand-off fell inside the damaged rover's own peer disc and inflation ring (≈ 2.0–2.5 m). The teammate is now planned as an uninflated keep-out, and the rescuer searches the whole 2.0–2.45 m ring for a free cell it can reach, approach side first.
- **A rover sealed inside its own hazard zone.** When the reverse timed out short of the 2.5 m zone, the zone covered the rover and every way out. The zone now shrinks to stay 0.6 m clear of the rover (never below 1 m).
- **False slip safe-holds under CPU load.** Ground speed was differentiated over message arrival times, so bunched odometry deliveries read as a stall. It now uses the odometry's simulated timestamps.
- **Mock only: a gentle stall read as a 263 g impact.** The pacer splits physics steps at sensor due times, so a step can last microseconds; backing into a rock zeroed the ground speed in one step, and the accelerometer read that velocity step divided by the step length. The health model wrote the rover off after an ordinary recovery reverse. The mock accelerometer now follows the stall through ≈ 50 ms of tyre compliance.
- **A lost rescue re-plan.** `rescue_start` arrived 0.1 s after RESUMING's re-plan, and the 1 s re-plan rate limit dropped it, so the rescuer first drove its old frontier route away from its teammate. A request inside the gap is now held and served when the gap ends.
- **Test only: a strike where no rescue was possible.** The smoke test struck Rover 1 wherever it happened to be. Once, that was a pocket enclosed by steep cells, rockslide boulders, hazard zones and the 3 m sector-edge band: no stand-off cell was reachable even at 0.3 m clearance out to 6 m, so `rescue_noroute` was the correct answer. The smoke test now decodes the live map and waits until a free stand-off 2.0–2.45 m from Rover 1 is reachable from Rover 2 before it strikes.

## Repository layout

```
start_all.sh · stop_all.sh            one-command bring-up / teardown
docs/protocol.md                      topic, JSON, REST and WebSocket contracts
ros2_ws/src/redshift_sim/             world generation, Gazebo models + launch, world interface, rover hardware
                                      (rover_hw), sensor effects, mock world (mock_render, render_pool,
                                      mock_dynamics, mock_world_node), check_topics.py
ros2_ws/src/redshift_autonomy/        physics_model, edge_budget, grid_map, planning, agent_fsm, rover_health,
                                      vision, trace + perception_node, mapping_coordinator_node, agent_planner_node
backend_bridge/                       FastAPI app, rclpy link, mission manager (launch, watchdog, fallback, abort),
                                      encoders, config schema, tests, scripts/smoke_test.py
web_ui/                               React + Three.js Mission Control; dev/ holds the scripted fake backend
```

## Troubleshooting

- **The page doesn't load from the Windows browser.** WSL2 forwards `localhost` by default. If yours doesn't, run `./start_all.sh --host 0.0.0.0` and use the WSL IP address.
- **"port 8000 is already in use".** A previous run is still up. Run `./stop_all.sh`, or pass `--port`.
- **The RTF readout is amber / below the chosen speed.** The simulator can't render that fast on this machine (see *Simulation speed*). The mission still runs correctly, only slower than requested.
- **Gazebo is slow or never produces frames.** The backend falls back to the mock world after 45 s without camera frames (or at once if Gazebo exits during start-up), and says so in the boot log and as an event. Choose *Mock physics* in the UI to skip Gazebo entirely.
- **`npm` resolves to `/mnt/c/...`.** That is the Windows npm. `start_all.sh` prefers the newest nvm Node; install one with `nvm install 22`.
- **Simulator processes are left over after a crash.** Run `./stop_all.sh`. Its patterns only match this checkout's `runtime/` world and `install/` nodes, the `redshift_sim` launch and the bridge's `redshift_gz_bridge` node, so other Gazebo or ROS work on the machine is left alone.

## License

See [LICENSE](LICENSE).
