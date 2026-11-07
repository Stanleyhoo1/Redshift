# Disaster Rover: decisions and why

## Repo
- `gh repo list` showed no `redshift` under kayaclaridge-sys. `gh api user/repos` found `Stanleyhoo1/Redshift` (collaborator, WRITE, pushed today, only LICENSE + README). Assumed this is the team hackathon repo. Work is in `disaster-rover/` on branch `naturaldisasterdemo`; nothing outside that folder is touched.
- Git identity was not configured; set repo-local `user.name=kayaclaridge-sys`, `user.email` = the account email.

## Tooling
- Node 22 via nvm (`~/.nvm`), the PATH `npm` was the Windows one. Run `source ~/.nvm/nvm.sh` first.
- `tsx` runs the TS sim headless in Node; no bundling needed for tests.
- Headless browser screenshots via Playwright Chromium (software WebGL) kept in the scratchpad, used only to catch runtime errors since there is no visible browser.

## Sim
- World 200 x 200 m. Two grids: 2 m environment grid (terrain, water, fire, smoke, rubble), 0.5 m occupancy grid (LiDAR raycasts + collisions).
- Physics 20 Hz (`Sim.DT = 0.05`), sensors + brain 10 Hz.
- Everything seeded (mulberry32). Separate RNG streams for world, disasters and sensor noise so a brain change does not change the world.
- Failure rules live only in the sim: water > 0.5 m, air > 160 °C (damage above 70 °C), tilt > 34°, tornado wind > 46 m/s, meteor direct hit, collapsing building, debris, leaving the map, health 0.

## Brain (M3)
- One belief grid at 2 m (100 x 100) for both hazards and planning. A 4 m grid would make the forest impassable (every 4 m cell has a tree); 2 m is still fine for A* in pure Python (10k cells).
- Confidence is computed lazily from `lastSeen` (exp(-age/40 s)) instead of decaying every cell every tick, so the Python port stays cheap.
- Map + predictions are recomputed at 2 Hz; reflexes and control run every 10 Hz tick on raw readings.
- Thermal camera reports, per 10° sector, the max temperature, its distance AND the distance of the nearest >100 °C pixel (hotspot segmentation). Only the max is not enough: cells in front of a hot max aren't necessarily cool.
- Depth camera is a DSM (includes buildings, trees, water surface). Flood stage = median DSM elevation at camera-seen water edges within depth range; dry ground elevation is remembered per cell, so predicted flooding = cells whose remembered dry elevation is below stage + rate x 30 s, connected to seen water.
- IMU quake detection uses the residual after subtracting the acceleration explained by odometry + gravity; raw accel jumps from the rover's own motion gave false quakes.
- Flood "rising" requires both a regression slope > 0.4 cm/s and > 10 cm rise across the window; single-sample regression was fooled by shoreline noise.
- Hysteresis: +0.08 for the current behaviour, 2 s minimum dwell unless the challenger wins by 0.25.
- Slope risk is dilated by one cell and blocks planning above ~20°; the rover rolls at 34°. Tilt reflex at 22° backs straight down and marks the cell ahead impassable.
- A reflex that produces no motion for 1.5 s (wedged) switches to a turn/push wiggle.
- Sim fixes found by testing: hard river-bank clamp created cliffs (now a 0.34 gradient ramp); rover could spawn touching a building; collision now only checks the leading edge so side contact slides instead of wedging.

## ROS 2 (M4)
- Python brain uses numpy (installed with ROS Jazzy via apt as `python3-numpy`; no pip). Pure-Python loops over 10k cells (distance transforms, dilations) were ~0.5 s per map update; vectorised it is 2.5 ms median / ~35 ms p95 per tick.
- Distance fields in Python are an iterative 8-neighbour min-plus capped at 40 cells (80 m); beyond that 999. Every consumer clamps below 50 m, so results match the TS chamfer.
- Frame time is the `/rover_status` time (JSON `t`): it is published last and closes the frame; topics can arrive slightly out of order through rosbridge.
- Brain node is event-driven (one tick per `/rover_status`), not timer-driven. That makes a lockstep headless test possible (publish frame -> wait for its `/cmd_vel`) and costs nothing in the real-time browser.
- Extra topics beyond the spec: `/elevation`, `/camera/water`, `/camera/detections`, `/pressure` (sensor inputs), `/brain_info`, `/hazard_map`, `/brain_summary`, `/agent_proposal`, `/agent_status`, `/agent_enable` (UI + agent plumbing).
- Bug found by the ROS test: a Float32MultiArray with a 0-size dimension produced stride 0/0 = NaN -> JSON null -> rosbridge rejected the message.
- Browser uses roslib; headless uses a 40-line rosbridge client over Node 22's built-in WebSocket (no extra deps).
- A `/cmd_vel` older than 1 s is not trusted (rover stops); if the socket drops mid-run the app switches to the local brain.

## Flood + wildfire (M5)
- Session crashed mid-M5 while a long benchmark ran; the in-flight edits had all applied (checked marker by marker). Benchmarks are now short (5-8 seeds, `timeout 170`).
- Real cause of the slowdown: TS Dijkstra stored costs in a Float32Array; rounded-up float32 values let `ng < gs[j]` keep succeeding and stale heap entries re-expanded in a cascade (6 s per call). Float64 + closed set: 5 ms. Python was unaffected (float64, stale-entry skip).
- Local steering (`steerClear`): every forward command is checked against the live LiDAR scan; if blocked within 2.6 m it re-aims at the nearest clear heading. The 2 m map lags behind reality in smoke.
- Smoke "stop & re-scan" is suppressed when a fire is within 30 m and has a 10 s cooldown. Standing still in smoke next to a fire was the main cause of early deaths.
- Fire + water: one combined escape search scores heat repulsion, water depth, slope and banks together. With fire close, shallow water up to 0.36 m (hard limit 0.5) is acceptable refuge.
- Refuges in the direction of a sensed fire/tornado/meteor zone are penalised; otherwise the flee target flipped along the fire front.
- Slope risk relaxed to (deg-15)/15 (blocks at 24°) and the tilt reflex to 26°. The old 20° block made the generated 19° river banks a wall and boxed the rover in for a minute at a time.
- Flee with no reachable refuge now drives the escape heading instead of spinning in place.
- Fire is "close" (flee >= 0.8) within 35 m.

## Earthquake, tornado, meteors (M6)
- Sim fairness fixes found by testing (the brain was not at fault):
  - tornado spawn: a far point clamped to the map edge could put touchdown ~10 m from a rover near the edge; now samples a point 60-90 m away inside the map.
  - water depth was a per-cell step: crossing into a river cell went from 0 to 0.5 m instantly (no reflex can react). Depth is now the cell's water surface minus the bilinear ground at the exact point.
  - collision: a contact only blocks a move that brings the rover closer to it, so the rover slides past trees instead of wedging for 20 s.
  - meteor impacts ignite forest 10 % of the time (was 25 %: a heavy shower turned into a dozen wildfires).
- Brain: meteor dodge never heads into water/obstacles/steep ground, and water deeper than 0.35 m beats a dodge. Tilt reflex checks for water before reversing. Refuges near the projected tornado track are penalised. Probe placement is damped by the flee score.
- Unseen cells within 6 m of seen water cost 5 (not 0.6) to plan through: a fleeing rover in smoke drove into an unseen river strip. Side effect: survey coverage with no disaster dropped a little (~27 % -> ~23 % in 180 s on 8 seeds).
- Known limits: dense meteor showers can still score a direct hit (the streak warning is ~3 s and the rover turns at 1.6 rad/s); being caught between a fire and the river remains the most common wildfire failure.

## Probes (M7)
- A probe early warning fires only when a probe sample shows a CHANGE the brain didn't already know: a cell seen dry turning wet, or seen cool turning hot. Samples repeat every frame between 1 Hz updates and river cells are always wet, so "value above threshold" alone would spam.
- Throttled to one warning per probe per hazard kind per 10 s.
- Probes standing in water double as flood gauges (their DSM sample is the water surface).
- Survival is unchanged with/without probes: they are an early-warning network for people (the mission), not a survival aid for the rover. Their cost is time spent driving to good spots instead of exploring.
- Water reflex escape = back to last dry fix. Near fire, shallow water tolerance is 0.28 m (hard limit 0.5), escape planning avoids > 0.22 m.

## Agent (M8)
- Gemma supports native Ollama tool calling (`message.tool_calls`); a JSON-in-text fallback parser handles models that answer in prose+JSON. Anything else is logged as IGNORED (unparsable) and changes nothing.
- The brain validates (it owns the sensed map and the planner); the agent node only proposes. Same rule code path for both brains (rules.ts / rules.py).
- Accepted calls become a goal that biases utility scores (+0.35 explore for go_to/survey, >=0.62 place_probe / hold, +0.4 high_ground). They never bypass safety reflexes, and goals expire (60 s; hold = its seconds).
- Measured latency is much lower than the 23 s in the brief once warm: Ollama caches the long, identical system-prompt + tools prefix (prompt ~510 tokens, ~45 output tokens). Cold first call ~21 s.
- Headless agent runs are latency-faithful: sim time is not allowed to run ahead of wall time while a request is in flight, so the rover keeps driving on the utility brain for exactly as long as Gemma really takes.
- Observed LLM mistakes the rule set catches: re-proposing a probe 3-5 m from one just placed; placing a probe on a riverside spot the brain predicts will flood within 30 s.
- `pkill -f <pattern>` inside a command that also contains the pattern kills its own shell (exit 144). Stop and start ROS in separate commands.

## Visual pass + flood fix (after M9 part 1)
- Bug: pressing Start the first time reused the sim built at page load, so disasters ticked before the first Start were ignored. A flood "never happened" in the browser while headless runs were fine. Start now always builds a fresh run from the current settings; Pause/Resume is separate.
- Water: per-vertex `aDepth` = surface - ground, negative on dry vertices bordering water; the fragment shader discards depth <= 0, so the shoreline is the smooth depth = 0 contour instead of a 2 m staircase. Floodwater (wet, not river) is tinted mud; thin foam line at the edge; animated ripple normals; rain + darker sky while a flood is active.
- Terrain is now a 1 m mesh (201 x 201) with colour noise, rock on slopes, sandy banks and a soft fog-of-war edge (3x3 blur).
- Models are procedural (no assets): rocker-bogie rover (6 wheels, spinning LiDAR, pan/tilt mast camera, probe rack showing probes left, behaviour-coloured beacon); tripod probes with blinking beacon + expanding sensing pulse; conifers and broadleaf trees; houses with window textures and pitched roofs; three-shell tornado; meteors with trails; craters that glow while fresh; fire as flame wall + embers + fire light.
- UI: HUD (time, brain, brain-sensed hazards, behaviour), score bars, stat tiles, toasts for world events and probe warnings, colour-coded logs, framed onboard-camera inset.
