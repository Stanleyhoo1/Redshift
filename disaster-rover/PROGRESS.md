# Disaster Rover: progress

## Done
- M1: Vite + TS + Three.js scaffold, pure-TS headless sim core (`web/sim`), rover diff-drive physics, manual WASD driving, headless runner (`npm run sim`).
- M2: seeded terrain (fbm hills + ridge, meandering river, lake, forest with trees, town grid with buildings and roads, fields). Fog of war: cells are revealed within 28 m of the rover and 22 m of each probe; unrevealed terrain is darkened and objects/water there are hidden. Checked visually with a headless screenshot.

- M3: local TS brain (`web/brain`): perception (LiDAR free/occupied, depth-camera elevation + structure detection, camera water edges, thermal hotspots, IMU shake residual, wind/pressure, probe telemetry), prediction (fire front velocity, flood stage + rise rate, tornado track + CPA, meteor impact zone, quake/aftershock), utility selection with hysteresis, safety reflexes, risk-weighted A*. 12/12 seeds survive 150 s with no disasters, 17–43 % surveyed, 1–6 probes placed, ~3.5 ms per brain tick.

- M4: `ros_ws/src/disaster_brain` (ament_python) mirrors the TS brain module-for-module (numpy for grid math). `brain_node` ticks once per sensor frame (closed by `/rover_status`). `launch/rover.launch.py` starts rosbridge + brain (+ agent). Browser: "Brain: ROS 2" connects via roslib, falls back to Local TS with a warning if rosbridge is unreachable. Headless: `npm run sim:ros` runs lockstep over rosbridge: 0 missed frames, ~8-9 ms round trip. Open-loop parity (same frames into both brains): 100 % behaviour agreement over 601 frames.

- M5: flood + wildfire tuned with `npm run bench`. Local TS, 8 seeds, intensity 0.6: no disaster 8/8, flood 8/8, fire 6/8, flood+fire 7/8. ROS 2 brain via rosbridge: flood seed 3, fire seed 3, flood+fire seeds 2/4/5 all survived, 0 missed frames.

- M6: earthquake (IMU residual -> hold still in the open / flee structures; rubble + cracks via LiDAR), tornado (funnel sightings + wind/pressure -> track, closest-approach prediction, refuges off its 60 s track, wind reflex), meteors (impact flashes -> drifting impact zone; incoming-streak dodge reflex). Local TS, 8 seeds: quake 8/8, tornado 8/8, meteor 6/8, all five at once 5/8. ROS 2 brain seed 4: quake, tornado, meteor all survived, 0 missed frames. Browser smoke test with every chaos button: no console errors.

- M7: probes. Placement = coverage gain + warning value (early-warning distance ahead of a sensed hazard, flood gauge near water, watching structures), >= 26 m spacing, paced over the mission, damped while fleeing. Each probe is a sensor station: 1 Hz samples (water, temperature, elevation) every 5 m within 22 m feed the belief grid, reveal fog of war, act as flood gauges for the stage estimate, and raise "PROBE n early warning" lines + agent events when a cell turns wet or hot. 8 seeds: flood 9.5, fire 10.8, meteor 9.9 early warnings per run on average. Ablation (`--no-probes`, 6 seeds): same survival, 0 warnings; cost is survey time (quake 25.5 % vs 32.6 % surveyed).
- Water reflex now drives back to the last spot the depth gauge read dry (elevation search was fooled by the water surface). 8 seeds: flood 8/8, flood+fire 8/8, fire 5/8, meteor 6/8.

- M8: Gemma 4 (`gemma4:e2b` via local Ollama) mission commander. ROS: `mission_agent` node (background thread, one request in flight, triggers on new hazard / hazard on path / battery < 30 % / probe placed / probe warning, else every 30 s, 60 s timeout, stdlib urllib, native Ollama tool calling) -> `/agent_proposal` -> rule set in `brain_node` -> `/agent_log` + mission goal. Launch flag `use_agent:=false` turns it off. Local TS mirror (`web/brain/agent.ts`) for the browser and headless runs (`--agent`, latency-faithful). Rule-set test with deliberately bad tool calls: 15/15 in TS (`npm run test:rules`) and 15/15 in Python (`test/test_rules.py`), every rule exercised. Live: flood seed 3, 90 s, ROS + agent in real time: 8 calls, 5 accepted, 3 rejected (probe_spacing x3); TS + agent: 5 calls, 2 accepted, 3 rejected (probe_spacing, target_in_water x2). Warm latency 4-7 s per call (median 5.5 s); first call ~21 s.

- M9 (in progress): 3D overlay shows behaviour, scores, confidence, active reflex, mission goal and Gemma's latest verdict (or "Agent offline, utility fallback"). README written. Utility-only survival over 10 seeds x 6 scenarios: 49/60 (`results/utility-10seeds.md`). Heat escape slows near water / steeply downhill.

## Next
- Utility + agent comparison (3 seeds x 6 scenarios, latency-faithful, sequential, ~4 min per run) is running in the background into `results/agent-runs.jsonl`; fill README "AGENT_TABLE" when done.

## Blockers
- (resolved) Pushing was blocked earlier; The repo `redshift` is not under `kayaclaridge-sys`; the only match is `Stanleyhoo1/Redshift` (you have WRITE access, found via `gh api user/repos`). Cloning and the local branch `naturaldisasterdemo` worked, but `git push -u origin naturaldisasterdemo` was refused by the Claude Code permission guard ("create public surface" on a repo owned by another account). All milestone commits are on the local `naturaldisasterdemo` branch in `~/redshift`. The owner pushed the branch; pushes to `naturaldisasterdemo` now go after each milestone.
- Remaining fire failures are mostly "escaped the heat into the river" or getting caught between front and steep river banks; see NOTES.
