# Redshift

Autonomous rovers exploring an unknown super-Earth. Built for the Agentverse 2026 Robotics track.

Three rovers land on a planet with twice Earth's gravity. There is no map and no link home: a message to Earth would take years to arrive. Each rover senses the ground with a simulated LiDAR, decides where to go, and drives there in a physics simulation (MuJoCo). The fleet shares one map, splits up the unexplored ground, learns how steep a slope it can really climb, and changes plans when it finds something interesting. A web dashboard shows the planet appearing as the rovers explore.

## How it fits the challenge

| Brief | Redshift |
|---|---|
| Perceive | 360 degree LiDAR on every rover (MuJoCo ray casting), plus a short range "spectrometer" that spots science targets |
| Decide | Frontier exploration, fleet task splitting, A* path planning, battery planning and slope limit learning. Everything runs locally: no cloud services, no LLM calls |
| Act | Four motor driven wheels per rover, simulated at 2 g with real torque limits |

Where to aim: coordinated fleets, adaptive autonomy, learning on the fly.

## Quick start (Docker)

You need Docker with Compose (Docker Desktop with WSL integration, or Docker Engine inside WSL).

```bash
docker compose up --build
```

Open http://localhost:8080. The web container talks to the sim over Docker's private network, so only port 8080 is used on your machine.

If the page stays on "Waiting for the simulation", check the sim container with `docker compose ps -a` and `docker compose logs --tail=60 sim`.

Change the planet or speed:

```bash
SEED=7 SIM_SPEED=5 docker compose up --build
```

Stop with Ctrl+C, then `docker compose down`.

| Variable | Default | Meaning |
|---|---|---|
| `SEED` | 42 | terrain and target layout |
| `NUM_ROVERS` | 3 | rovers in the fleet |
| `SIM_SPEED` | 3.0 | sim seconds per real second |
| `MAP_SIZE` | 80 | map width in metres |
| `GRAVITY_G` | 2.0 | gravity in Earth g |
| `AUTO_RESET_S` | 60 | once exploration is complete and the rovers have parked, wait this many seconds and land on a new random planet (0 = off) |

## Run natively with the MuJoCo viewer

Docker runs the sim without a window. To watch the real 3D world in MuJoCo's own viewer (WSL with WSLg, or any desktop), run the sim directly. On WSL, do this from a copy of the repo in your Linux home folder, since Python virtual environments on `/mnt/c` are slow and fragile.

```bash
cd sim
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m redshift.server --viewer
```

Then, in a second terminal:

```bash
cd web
npm install
npm run dev      # http://localhost:5173
```

If no display is found, the server prints a warning and keeps running without the window.

Run the tests with `cd sim && python -m pytest -q tests/`. For a fast headless run with a summary and the full event log: `python -m redshift.sim --seconds 900 --seed 7`.

## Work on the UI with fake data

```bash
cd web
npm install
npm run mock    # fake sim on port 8000
npm run dev     # http://localhost:5173
```

## Project layout

```
PROTOCOL.md          message format between the sim and the UI
docker-compose.yml   runs both services
sim/                 MuJoCo world, rover agents, FastAPI + WebSocket server
web/                 React + Three.js mission control dashboard
```

## How the rovers decide

1. **Sense.** Each rover scans with its LiDAR 5 times a second. Hits go into one shared height map, and every cell gets a slope.
2. **Pick a goal.** Frontiers are the edges between explored and unexplored ground. Each rover scores frontiers by size, distance and how far it has to turn, and avoids frontiers another rover has already claimed, so the fleet spreads out.
3. **Plan a path.** A* over the map, avoiding cells steeper than the fleet's slope limit. Unknown cells are allowed, so paths push into the unknown and get replanned as new ground is seen.
4. **Learn on the fly.** The fleet starts out assuming it can climb 30 degrees. When a rover stalls on a slope, it measures the angle and lowers the limit for the whole fleet, and every rover replans straight away.
5. **React to discoveries.** When the spectrometer spots a target, a free rover drives over and samples it. Deposits come in clusters, so frontiers near sampled targets get priority.
6. **Manage power.** Each rover keeps enough battery to get home plus a 15% margin, returns to the lander to charge, then heads back out.

## Two minute demo

1. Start on the overview: the map is black and three rovers leave the lander.
2. Turn on Paths: the rovers split up toward different frontiers.
3. Watch the event log for a stall on a slope: the fleet slope limit drops and the slope colours update for everyone. With the default seed this happens about 45 seconds in.
4. A discovery: a rover diverts, samples the target, and nearby frontiers get priority.
5. Turn on Reality: what the fleet knows next to the real planet.
6. A rover runs low and drives home to charge on its own.
