# Disaster Rover: progress

## Done
- M1: Vite + TS + Three.js scaffold, pure-TS headless sim core (`web/sim`), rover diff-drive physics, manual WASD driving, headless runner (`npm run sim`).
- M2: seeded terrain (fbm hills + ridge, meandering river, lake, forest with trees, town grid with buildings and roads, fields). Fog of war: cells are revealed within 28 m of the rover and 22 m of each probe; unrevealed terrain is darkened and objects/water there are hidden. Checked visually with a headless screenshot.

- M3: local TS brain (`web/brain`): perception (LiDAR free/occupied, depth-camera elevation + structure detection, camera water edges, thermal hotspots, IMU shake residual, wind/pressure, probe telemetry), prediction (fire front velocity, flood stage + rise rate, tornado track + CPA, meteor impact zone, quake/aftershock), utility selection with hysteresis, safety reflexes, risk-weighted A*. 12/12 seeds survive 150 s with no disasters, 17–43 % surveyed, 1–6 probes placed, ~3.5 ms per brain tick.

## Next
- M4 ROS 2 bridge + rclpy brain.

## Blockers
- **Pushing to GitHub was blocked.** The repo `redshift` is not under `kayaclaridge-sys`; the only match is `Stanleyhoo1/Redshift` (you have WRITE access, found via `gh api user/repos`). Cloning and the local branch `naturaldisasterdemo` worked, but `git push -u origin naturaldisasterdemo` was refused by the Claude Code permission guard ("create public surface" on a repo owned by another account). All milestone commits are on the local `naturaldisasterdemo` branch in `~/redshift`. To publish: `cd ~/redshift && git push -u origin naturaldisasterdemo`.
