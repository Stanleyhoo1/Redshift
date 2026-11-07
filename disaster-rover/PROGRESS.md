# Disaster Rover: progress

## Done
- M1: Vite + TS + Three.js scaffold, pure-TS headless sim core (`web/sim`), rover diff-drive physics, manual WASD driving, headless runner (`npm run sim`).
- M2: seeded terrain (fbm hills + ridge, meandering river, lake, forest with trees, town grid with buildings and roads, fields). Fog of war: cells are revealed within 28 m of the rover and 22 m of each probe; unrevealed terrain is darkened and objects/water there are hidden. Checked visually with a headless screenshot.

- M3: local TS brain (`web/brain`): perception (LiDAR free/occupied, depth-camera elevation + structure detection, camera water edges, thermal hotspots, IMU shake residual, wind/pressure, probe telemetry), prediction (fire front velocity, flood stage + rise rate, tornado track + CPA, meteor impact zone, quake/aftershock), utility selection with hysteresis, safety reflexes, risk-weighted A*. 12/12 seeds survive 150 s with no disasters, 17–43 % surveyed, 1–6 probes placed, ~3.5 ms per brain tick.

- M4: `ros_ws/src/disaster_brain` (ament_python) mirrors the TS brain module-for-module (numpy for grid math). `brain_node` ticks once per sensor frame (closed by `/rover_status`). `launch/rover.launch.py` starts rosbridge + brain (+ agent). Browser: "Brain: ROS 2" connects via roslib, falls back to Local TS with a warning if rosbridge is unreachable. Headless: `npm run sim:ros` runs lockstep over rosbridge: 0 missed frames, ~8-9 ms round trip. Open-loop parity (same frames into both brains): 100 % behaviour agreement over 601 frames.

- M5: flood + wildfire tuned with `npm run bench`. Local TS, 8 seeds, intensity 0.6: no disaster 8/8, flood 8/8, fire 6/8, flood+fire 7/8. ROS 2 brain via rosbridge: flood seed 3, fire seed 3, flood+fire seeds 2/4/5 all survived, 0 missed frames.

## Next
- M6 earthquake, tornado, meteors.

## Blockers
- (resolved) Pushing was blocked earlier; The repo `redshift` is not under `kayaclaridge-sys`; the only match is `Stanleyhoo1/Redshift` (you have WRITE access, found via `gh api user/repos`). Cloning and the local branch `naturaldisasterdemo` worked, but `git push -u origin naturaldisasterdemo` was refused by the Claude Code permission guard ("create public surface" on a repo owned by another account). All milestone commits are on the local `naturaldisasterdemo` branch in `~/redshift`. The owner pushed the branch; pushes to `naturaldisasterdemo` now go after each milestone.
- Remaining fire failures are mostly "escaped the heat into the river" or getting caught between front and steep river banks; see NOTES.
