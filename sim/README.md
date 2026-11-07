# Redshift simulation backend

A fleet of rovers explores an unknown super-Earth at 2 g. Each rover only knows what its own
sensors tell it. The rovers build a shared map from LiDAR, pick where to go next, learn how steep
a slope they can really climb, find and sample science targets, and drive home to charge. They
make every decision on their own; nothing is scripted. The web UI (in `../web`) shows the world
filling in. The message format is in `../PROTOCOL.md`.

## What each module does

| File | What it does |
|---|---|
| `redshift/config.py` | All settings in one place. Reads the env vars below. |
| `redshift/terrain.py` | Makes the ground: plains, hills, ridges, craters (each with a gentle way in), rocks, and 6 to 9 science targets in clusters. Same seed, same world. |
| `redshift/world.py` | Builds the MuJoCo physics model: the ground as height-map tiles, rocks, the lander pad, target markers, and the rovers (4 wheels, skid steering, front axle that rocks so all wheels touch). |
| `redshift/mapping.py` | The shared fleet map. Turns LiDAR hits into heights and slopes for each grid cell, and finds frontiers (edges of the known area). |
| `redshift/planner.py` | A* path planning on the map. Cells steeper than the current slope limit are off limits. |
| `redshift/rover.py` | One rover's brain: LiDAR scan, spectrometer, driving, battery, and its state (exploring, investigating, returning, charging, recovering, idle). |
| `redshift/fleet.py` | Team decisions: who goes to which frontier, the shared slope limit, the target list, and the event log. |
| `redshift/sim.py` | Runs it all: steps the physics, runs the rovers' decisions 5 times per sim second, and builds the messages for the UI. Also a command line runner. |
| `redshift/server.py` | The web server (FastAPI): WebSocket `/ws`, `GET /api/health`, `GET /api/truth`, `POST /api/reset`. Runs the sim in the background at `SIM_SPEED` times real time. |
| `scripts/climb_test.py` | Drives a rover up ramps of known angle to check how steep it can climb. |
| `tests/test_smoke.py` | Runs 400 sim seconds and checks the results and every message. |

## How to run

Install (Python 3.10 to 3.13):

```bash
cd sim
pip install -r requirements.txt
```

Server with the 3D MuJoCo viewer (on your own machine, not in Docker):

```bash
python -m redshift.server --viewer            # or VIEWER=1 python -m redshift.server
mjpython -m redshift.server --viewer          # on macOS the viewer needs mjpython
```

If no display is found, the server prints a warning and runs without the viewer. Open the web UI
(`cd ../web && npm run dev`) to watch the map fill in.

Server only (this is what Docker runs):

```bash
uvicorn redshift.server:app --host 0.0.0.0 --port 8000
```

Headless, as fast as possible, with a summary and the event log at the end:

```bash
python -m redshift.sim --seconds 900 --seed 7
```

Tests:

```bash
python -m pytest -q tests/        # add -s to see the event log
python tests/test_smoke.py        # same checks without pytest
python scripts/climb_test.py      # ramp test for the rover tuning
```

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `SEED` | 42 | terrain and target layout |
| `NUM_ROVERS` | 3 | rovers in the fleet (1 to 8) |
| `SIM_SPEED` | 3.0 | sim seconds per wall second |
| `MAP_SIZE` | 80 | map width in metres |
| `GRAVITY_G` | 2.0 | gravity in Earth g |
| `VIEWER` | 0 | 1 = open the native MuJoCo viewer |
| `AUTO_RESET_S` | 60 | once exploration is complete and every rover is parked, wait this many seconds, then start a new random planet (server only, 0 = off) |

## How the rovers decide

- **Seeing.** Each rover has a spinning LiDAR on a mast (8 m range, about 1000 rays, 5 scans a
  second). Hits go into one shared map of heights and slopes. Rocks show up as steep spots. Rovers
  know their own position exactly (no SLAM).
- **Where to go.** A frontier is the edge between known, drivable ground and the unknown. Each
  rover scores the frontiers: big and close is good, turning around is bad, and a frontier another
  rover is already heading for is much worse. So the rovers spread out. It also checks it has
  enough battery to get there and back. Then A* plans a path around steep ground.
- **Learning the slope limit.** The fleet starts by guessing it can climb 30 deg. If a rover
  stalls nose up on a steady slope, with its motors maxed out and gravity alone needing most of
  their torque, the fleet lowers the limit to 2 deg under that slope. Every rover replans with
  the new limit straight away. The limit never drops below the steepest climb a rover has already
  managed. A stall caused by a rock or by losing grip does not count.
  The real limit at 2 g is about 20.5 deg (see `scripts/climb_test.py`). It is set by motor
  torque, so it gets worse as gravity goes up.
- **Science.** If a LiDAR ray hits a target within 8 m, the rover reports it. If it is free and has
  the battery, it drives over, waits a few seconds to take a sample, and marks it sampled. Busy
  rovers log the find and another rover can pick it up. Targets come in clusters, so frontiers near
  a sampled target get a bonus.
- **Battery.** Driving uses power (more on slopes and in tight turns), and the electronics use a
  little all the time. Each rover keeps checking how much charge it needs to get home and goes
  back with a 15% margin. It charges on the lander pad, then heads out again.
- **Trouble.** If a rover stops moving it works out why: a rock (mark it and go round), a slope
  (learn), lost grip (back off and try another line), or a rover in the way (back off). If it tips
  over it rights itself. If steep ground boxes it in, it heads for the lander, allowing slightly
  steeper ground, until a frontier is reachable again.
- **Done.** When no frontier can be reached from the lander, exploration is complete and every
  rover parks at the lander. The server then waits `AUTO_RESET_S` seconds and lands the fleet on a
  new random planet, so an unattended demo keeps going.

Every decision is logged as a short event that the UI shows live.
