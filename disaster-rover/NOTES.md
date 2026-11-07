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
