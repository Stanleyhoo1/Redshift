# Disaster Rover

**Agentverse 2026 · Robotics track · "Give an agent a body"**

An autonomous rover in a 3D disaster sim. You tick disasters in the GUI (flood, wildfire, earthquake, tornado, meteor shower), and the rover has to survey the area, survive, and drop early-warning probes in useful spots. It decides only from its own sensors (LiDAR, thermal camera, depth camera, water gauge, IMU, wind/pressure, camera detections) and runs entirely on the local machine:

- **ROS 2 brain** (`ros_ws/src/disaster_brain`, rclpy): perception → prediction → utility selection → risk-weighted A*, with safety reflexes on top. Runs at 10 Hz.
- **Gemma 4 E2B mission commander** (local Ollama): a slow strategic layer that proposes one tool call at a time. Every call passes a rule set before it can influence the rover.
- **Web sim** (Vite + TypeScript + Three.js): the world, the disasters, the rover body and its sensors. The sensors are the only thing the brain can see.

There are no cloud calls and no scripted paths. The brain never reads the true world state.

---

## Run it

### Web app (sim + 3D view + GUI)

```bash
source ~/.nvm/nvm.sh            # Node 22 via nvm
cd ~/redshift/disaster-rover/web && npm install && npm run dev
# open http://localhost:5173
```

Tick disasters, set intensity and seed, then press **Start**. The **Chaos panel** buttons trigger a disaster immediately. Under **Brain**, pick **ROS 2**, **Local TS** or **Manual (WASD)**. If rosbridge isn't reachable, the app shows a warning and falls back to the local TS brain. The **Agent** checkbox toggles the Gemma 4 commander.

### ROS 2 side (brain + rosbridge + agent)

```bash
source /opt/ros/jazzy/setup.bash
cd ~/redshift/disaster-rover/ros_ws
colcon build --symlink-install
source install/setup.bash

ros2 launch disaster_brain rover.launch.py                    # rosbridge :9090 + brain_node + mission_agent (Gemma 4)
ros2 launch disaster_brain rover.launch.py use_agent:=false   # utility brain only
# optional: model:=gemma4:e2b ollama_url:=http://localhost:11434 port:=9090
```

Then select **Brain: ROS 2** in the web app. The agent needs Ollama running with `gemma4:e2b` pulled (`ollama pull gemma4:e2b`).

### Headless (no browser)

```bash
cd ~/redshift/disaster-rover/web
npm run sim -- --disasters flood,fire --seed 3 --duration 180            # local TS brain
npm run sim -- --disasters flood --seed 3 --agent                         # + Gemma 4 (latency-faithful)
timeout 300 npm run sim:ros -- --disasters flood,fire --seed 3            # ROS 2 brain over rosbridge (launch it first)
timeout 300 npm run sim:ros -- --disasters flood --seed 3 --realtime      # ROS 2 brain + mission_agent, real-time pacing
npm run bench -- --scenarios flood,fire,quake,tornado,meteor,all --seeds 1-10   # survival table
npm run test:rules                                                        # rule set vs deliberately bad tool calls
```

Options: `--intensity 0.1..1`, `--random` (random disaster timing), `--no-probes` (ablation), `--json`. Each run prints survived/failed, the cause, probes placed, area surveyed, and the probe hazard coverage.

---

## Architecture

```
 ┌──────────────────────────── web (browser or Node, TypeScript) ───────────────────────────┐
 │  sim/      seeded world 200x200 m · disasters · rover physics · SENSORS (only output)    │
 │  render/   Three.js follow cam, LiDAR rays, hazard overlay, fog of war, onboard camera   │
 │  ui/       disasters + chaos panel, brain/agent toggles, status, decision log, minimap   │
 │  brain/    local TS brain (same pipeline as ROS) + TS agent client      ← fallback/tests │
 └──────────────┬───────────────────────────────────────────────────▲────────────────────────┘
   /scan /odom /imu /thermal /water_depth /wind /pressure            │ /cmd_vel /drop_probe
   /elevation /camera/water /camera/detections /rover_status         │ /decision_log /agent_log
                │  rosbridge ws://localhost:9090 (roslib)             │ /brain_info /hazard_map /agent_status
 ┌──────────────▼────────────────────── ROS 2 Jazzy ────────────────┴────────────────────────┐
 │  brain_node (rclpy, 10 Hz, one tick per sensor frame)                                     │
 │    PERCEPTION  sensor frame → 2 m belief grid: obstacles, elevation/structures, water,    │
 │                temperature, rubble/impacts, probe telemetry; confidence decays with age   │
 │    PREDICTION  fire front velocity, flood stage + rise rate, tornado track + CPA,         │
 │                meteor impact zone, aftershock risk → hazard map 30 s ahead                │
 │    UTILITY     explore / flee / high_ground / hold_still / place_probe / return_to_safety │
 │                scored from the sensed state + mission goal, hysteresis + min dwell        │
 │    PLANNER     A* weighted by predicted risk, re-planned when the map changes             │
 │    SAFETY      reflexes override everything: water, heat, meteor streak, tornado wind,    │
 │                tilt, stale data, smoke-blind LiDAR → stop & re-scan; LiDAR local steering │
 │    RULE SET    validates every agent tool call against the sensed map                     │
 │        ▲ /agent_proposal                    │ /brain_summary (text built from the belief) │
 │  mission_agent (rclpy, background thread) ◄─┘                                             │
 │    Gemma 4 E2B via Ollama /api/chat (tool calling) → ONE of: go_to_region, place_probe_at,│
 │    retreat_to_high_ground, hold_position, survey_area                                     │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
```

Both brains implement one interface. The input is one sensor frame in ROS message shapes (`brain/msgs.ts` mirrors the `.msg` files 1:1); the output is `{cmd_vel, drop_probe, decision_log}`. The Python package mirrors the TS brain module for module (`perception`, `prediction`, `utility`, `safety`, `planner`, `rules`). Open-loop parity test: the same 601 frames fed into both brains give 100 % behaviour agreement (`ros_ws/src/disaster_brain/test/parity_replay.py`).

**The brain never reads world state.** `sim/sensors.ts` is the only bridge from ground truth to the brain, and it outputs range-limited, noisy sensor messages. The agent sees only a text summary built from the brain's belief.

### Sensors (simulated, range-limited, noisy)

| Sensor | Topic | Model |
|---|---|---|
| 360° LiDAR, 72 beams, 30 m | `/scan` | raycast through a 0.5 m occupancy grid; smoke attenuates (no return) |
| Odometry | `/odom` | pose + twist, Gaussian noise |
| IMU | `/imu` | attitude from terrain, accel incl. seismic shaking |
| Thermal camera, 36 sectors, 45 m | `/thermal` | max °C + distance + nearest >100 °C pixel per sector; sees through smoke |
| Water depth gauge | `/water_depth` | depth under the rover |
| Wind | `/wind` | ambient wind + tornado vortex (Rankine) |
| Barometer | `/pressure` | tornado pressure deficit |
| Depth camera, 20 m | `/elevation` | 21×21 surface-elevation patch (DSM: buildings, trees, water surface) |
| Camera (water) | `/camera/water` | distance to visible water per sector, range cut by smoke |
| Camera (detections) | `/camera/detections` | impact flashes, tornado funnel, incoming meteor streaks, local smoke |
| Status | `/rover_status` | health, battery, probes, onboard thermometer, probe telemetry |

### Disasters (each changes the world over time and shows up on at least one sensor)

| Disaster | World effect | Sensed by |
|---|---|---|
| Flood | river stage rises; water spreads ring by ring to connected low ground | water gauge, camera water edges, depth camera (surface elevation), probes |
| Wildfire | cellular spread through fuel, driven by wind and slope; smoke advects downwind; the wind shifts once | thermal camera, thermometer, LiDAR/camera range loss in smoke, probes |
| Earthquake | shaking, buildings partly collapse into rubble, ground cracks, aftershock | IMU residual, LiDAR (new obstacles) |
| Tornado | moving vortex pushes the rover and throws debris; strips trees | wind, pressure, camera funnel |
| Meteor shower | drifting impact zone, craters + rocks block paths, impacts can start fires | camera streaks (≈3 s warning) and flashes, LiDAR |

The rover fails if: water > 0.5 m, air > 160 °C (damage above 70 °C), tilt > 34°, wind > 46 m/s (flipped), a direct meteor hit, a collapsing building, or health reaching 0 from debris/blast/heat.

---

## Where each model runs, and how fast

All numbers were measured on the dev machine: **Intel Core Ultra 5 125U (14 threads), 7.6 GB RAM, WSL2 Ubuntu 24.04, CPU only, no GPU.**

| Component | Runs on | Measured |
|---|---|---|
| Utility brain, ROS 2 (rclpy + numpy) | CPU, 1 core | tick median **2.5 ms** (7 ms during a large wildfire), p95 **~33 ms** (up to ~100 ms in a large wildfire: the 2 Hz map/prediction update), max ~145 ms. Runs at the 10 Hz sensor rate. Headless lockstep over rosbridge: 0 missed frames, round trip median 7–9 ms, p95 ~30–40 ms |
| Utility brain, local TS (Node / browser) | CPU, 1 core | tick average 1.5–5 ms at 10 Hz |
| Gemma 4 E2B (`gemma4:e2b`, Q4_K_M, 4.6 B params) via Ollama 0.40 | CPU | first call **~21 s** (cold). Warm calls **4–7 s**, median **5.5 s** (~510 prompt tokens with the prompt prefix cached, ~45 output tokens). Called on events, else at most every 30 s; timeout 60 s |

**Edge budget statement.** The whole stack (sim excluded) fits a laptop-class CPU with no GPU: the brain uses roughly 3–7 % of one core (more during a large wildfire); Gemma uses the CPU for ~4–7 s per call, with calls every ~10–30 s depending on how many events occur. If Gemma is slow, offline or erroring, the utility brain carries on alone and the UI shows **"Agent offline, utility fallback"**.

---

## How each judging criterion is covered

**Perception-driven action (25 %).** Every decision is made from sensor messages, and each decision-log line names the triggering reading, the prediction, the behaviour scores, the chosen action and the confidence:

```
Thermal 698°C 23 m S | fire front 21 m SE moving NW 0.35 m/s, reaches rover in 33 s | flee 0.82 explore 0.02 → FLEE → 47 m N (44 m from hazards) (0.99)
Wind 21 m/s, ΔP −162 Pa, funnel 27 m SW | vortex 26 m SW moving NE 4.0 m/s, passes 0 m from rover in 7 s | flee 0.95 … → FLEE → 57 m NW
REFLEX Camera: meteor streak, impact 7.0 m NW in 2.8 s → DODGE SE (overrides flee)
PROBE 0 early warning: water 0.27 m at (-27,-5), 19 m W of rover
```

Hazard detection is real perception: fire from thermal hotspots, flood stage from where the camera sees the waterline and what the depth camera measures there, tornado position from funnel sightings or from the wind vector (the centre lies 90° left of the residual wind), quakes from the IMU residual after subtracting the rover's own motion. The minimap shows the brain's belief (what it knows), not the world.

**Adaptation to change (25 %).** The chaos panel injects unscripted disasters mid-run, and the wind shifts during wildfires. The brain re-estimates each hazard continuously, predicts it 30 s ahead, re-plans A* when predicted risk enters the path (and raises an agent event), switches behaviour with hysteresis, and lets reflexes override instantly. Combined cases are handled jointly: one escape search trades off heat, water depth and slope (fire beside a river), and refuges are scored to stay off a tornado's projected track.

**Operational autonomy (15 %).** Everything runs locally within the budget above. When unsure, the brain behaves conservatively:
- Confidence decays per cell with time since last sensed.
- Smoke-blinded LiDAR triggers stop & re-scan (unless fire is close).
- Unseen ground next to water is expensive to plan through.
- Frames older than 0.6 s stop the rover, and stale `/cmd_vel` (> 1 s) is not trusted.
- The ROS link falls back to the local brain.
- The LLM is optional, and its output is constrained (tool calls only) and validated by a rule set.

**Relevance and impact (20 %).** The mission isn't only survival: it builds an early-warning sensor network. Probes are placed for coverage gain + warning value (ahead of a sensed hazard's path, as flood gauges near water, or watching the town). Each probe streams water/temperature/elevation samples within 22 m, extends the rover's fog-of-war reveal, and raises **probe early warnings** when a cell turns wet or hot before the rover sees it. In flood runs that's ~10 warnings per run on average.

**Ambition and technical quality (15 %).** Two interchangeable brains (TS and ROS 2) behind one ROS-shaped interface, with an open-loop parity test. A headless lockstep ROS test. A parallel survival benchmark. A latency-faithful LLM-in-the-loop benchmark. Rule-set tests with deliberately bad tool calls (15/15 in both languages). A probe ablation (`--no-probes`). The Three.js view shows LiDAR rays, the planned path, a brain-belief hazard overlay, fog of war and the onboard camera.

---

## Gemma 4 mission commander and rule set

- **Input:** a short text summary built only from the brain's belief: position, battery, health, probes, explored %, unexplored directions, every hazard prediction in words, best probe spot, safest spot, highest safe ground, current goal.
- **Output:** exactly one tool call from `go_to_region(x, y, reason)`, `place_probe_at(x, y, reason)`, `retreat_to_high_ground(reason)`, `hold_position(seconds, reason)`, `survey_area(x, y, radius, reason)`. Native Ollama tool calling, `stream:false, think:false, keep_alive:30m, num_predict:200, temperature:0.2`, standard library only. Output that doesn't parse is logged as IGNORED and changes nothing.
- **Triggers:** new hazard type sensed, predicted hazard entering the planned path, battery < 30 %, probe placed, probe early warning; otherwise at most every 30 s. One request in flight, in a background thread.
- **Rule set** (in the brain, which owns the sensed map; `rules.py` / `rules.ts`), checked in order: `unknown_tool`, `bad_args`, `out_of_bounds`, `target_in_water` (sensed water or predicted to flood), `predicted_hazard`, `unreachable` (no A* path on the sensed map), `battery_reserve` (go + return + 12 % reserve), `no_probes_left`, `probe_spacing` (< 20 m from a probe), `unsafe_hold` (holding inside a predicted hazard zone).
- **Accepted** calls become the mission goal, which biases the utility scores. The planner carries it out; the goal expires after 60 s (hold: its seconds); safety reflexes always override.
- `/agent_log` format: `[AGENT] trigger + summary → tool(args) → "reason" → ACCEPTED | REJECTED (rule: detail) → latency`.

Example from a live flood run on the ROS 2 stack:

```
[AGENT] trigger: probe_placed … → place_probe_at(x=-31, y=-27) → "Place probe at best spot to watch structures" → REJECTED (probe_spacing: within 20 m of the probe at (-29,-25)) | 4.0 s
[AGENT] trigger: probe_placed, probe_warning; water stage 0.31 m rising 1.5 cm/s … → place_probe_at(x=29, y=21) → "Place early-warning probe at best spot to warn about rising water." → ACCEPTED | 5.8 s
```

---

## Survival results

### Utility brain only: 10 seeds per scenario (local TS brain; the ROS brain is the same algorithm, 100 % open-loop parity)

`npm run bench -- --scenarios <s> --seeds 1-10` · duration 180 s · intensity 0.6 · fixed timing (first disaster at 12 s, then every 22 s).

| scenario | survived | probes placed (avg) | area surveyed (avg %) | hazard coverage by probes (avg %) | probe early warnings (avg) | failures |
|---|---|---|---|---|---|---|
| flood | 10/10 | 3.8 | 24.7 | 23.6 | 8.3 | - |
| wildfire | 5/10 | 3.5 | 24.9 | 11.0 | 12.2 | s2, s5, s6, s10 overheated (trapped by the front); s9 drowned (fled heat down a bank into the river) |
| earthquake | 10/10 | 4.5 | 25.3 | 6.4 | 5.8 | - |
| tornado | 10/10 | 4.8 | 32.0 | 30.7 | 4.2 | - |
| meteor shower | 7/10 | 3.8 | 25.2 | 17.1 | 9.5 | s1 direct meteor hit; s2 drowned; s10 overheated (impact-started fire) |
| all five at once | 7/10 | 2.9 | 26.6 | 10.1 | 8.1 | s1 flipped by tornado; s4 overheated; s8 tornado debris |
| **total** | **49/60** | | | | | |

Raw per-run JSON: `utility-10seeds-a.json`, `utility-10seeds-b.json` (wildfire/meteor rows re-run after the last heat-reflex change; same seeds).

### Utility brain + Gemma 4 commander

| scenario | utility only | utility + Gemma 4 | agent calls | accepted / rejected by rule set | failures with agent |
|---|---|---|---|---|---|
| flood | 3/3 | 3/3 | 33 | 23 / 9 | - |
| wildfire | 2/3 | 2/3 | 42 | 28 / 11 | s2 overheated |
| earthquake | 3/3 | 3/3 | 40 | 29 / 10 | - |
| tornado | 3/3 | 3/3 | 33 | 22 / 10 | - |
| meteor shower | 1/3 | 1/3 | 23 | 12 / 10 | s1 meteor hit; s3 overheated |
| all five | 2/3 | 3/3 | 42 | 31 / 10 | - |
| **total** | **14/18** | **15/18** | **213** | **145 / 60** (8 still in flight at run end) | |

- 0 agent errors / unparsable outputs. Per-run median latency 4.8-9.2 s (median of medians 6.0 s).
- Rejections by rule: probe_spacing 28, predicted_hazard 15, target_in_water 10, out_of_bounds 3, no_probes_left 2, unreachable 2.
- Caveats: 3 seeds is a small sample (one extra survival is not significant). The run was paused midway for UI work and resumed; the seed 1 and part of seed 2 runs predate the last heat-escape tweak (which made no measured difference on fire). The agent mainly changes WHERE probes go and what gets surveyed; survival is owned by the utility brain and reflexes by design.

---

## Limits and the critical failure mode

**The critical failure modes are entering deep water and getting caught by fire.** The safety reflexes handle them as follows:

- **Water.** A reflex fires at 0.2 m depth (hard limit 0.5 m) and drives back to the last position where the rover's own gauge read dry. Behind that, the planner refuses cells with sensed water or predicted flooding, makes unseen ground next to water expensive, and the high-ground behaviour climbs above the predicted 30 s stage. The tilt reflex never reverses blindly into water, and a meteor dodge never heads into water. Water deeper than 0.35 m beats a dodge, because drowning is certain and a meteor hit is not.
- **Heat.** A reflex fires at 52 °C body temperature or any >100 °C hotspot within 9 m (hard limit 160 °C). The escape heading weights every hot thermal sector by T/d², avoids water deeper than 0.22 m, steep ground and obstacles, and is LiDAR-steered. With fire close, the water tolerance rises to 0.28 m, because shallow water is a refuge from fire. The flee behaviour starts when the front is within 35 m or predicted to arrive within 30–45 s.

Known limits:
- The most common remaining death is being trapped between a fire front and the river or steep banks (the escape search has no good heading left).
- Dense meteor showers can score a direct hit: the streak warning is ~3 s and the rover turns at 1.6 rad/s.
- Probes trade survey coverage for warning coverage.
- The world is 2.5D (a heightmap plus occupancy grid). LiDAR is a single horizontal plane, and negative obstacles (craters, cracks) are returned as hits.
- Odometry is near-perfect (σ 5 cm): there is no SLAM drift.
- Gemma E2B sometimes proposes duplicate probe spots or riverside spots that are about to flood; the rule set rejects those.
- The ROS brain's map update takes ~30–100 ms in Python (2 Hz); per-frame ticks are a few ms, so 10 Hz control is fine, but the map update could not be pushed much faster.

More detail: `PROGRESS.md` (what's done) and `NOTES.md` (every design decision and the bugs found by testing).
