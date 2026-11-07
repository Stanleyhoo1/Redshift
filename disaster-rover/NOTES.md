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
