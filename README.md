# Redshift

Autonomous rovers exploring an unknown super-Earth. Built for the Agentverse 2026 Robotics track.

Three rovers land on a planet with twice Earth's gravity. There is no map and no link home: a message to Earth would take years to arrive. Each rover senses the ground with a simulated LiDAR, IMU, wheel encoders and a camera, decides where to go, and drives there in a physics simulation (MuJoCo). The fleet shares one map, splits up the unexplored ground, learns how steep a slope it can really climb, and replans when the world changes under it: rockfalls, hidden soft sand, dust storms and breakdowns. A small Gemma model running locally advises the rovers and talks with mission control. A web dashboard shows the planet appearing as the rovers explore.

## How it fits the judging

| Criterion | Redshift |
|---|---|
| Perception-driven action | Every decision comes from the rover's own sensors and every event says which reading caused it ("slip 82%", "slope 21 deg with motors at 87%"). Click an event to fly to where it happened. |
| Adaptation to change | Judges press a hazard button and the scene changes mid-run. The rovers only find out through their sensors (LiDAR sees new rocks, wheel slip reveals sand, LiDAR returns shrink in dust, a teammate reports a fault) and visibly replan. |
| Operational autonomy | Two stated budgets, measured live: each rover's rule brain gets 40 ms per 200 ms cycle, and Gemma calls get 3 s (about 0.3 to 1.5 s on an RTX 4070). When unsure, a rover slows down, and if it still cannot tell what it sees it asks mission control, taking the safe option if nobody answers in 20 s. |
| Relevance and impact | Planetary rovers cannot wait minutes or years for instructions. The failure that ended NASA's Spirit rover, getting stuck in soft soil, is detected and handled on screen. |
| Ambition and quality | Real physics at 2 g, a coordinated fleet, online learning, a two-tier brain (fast rules plus a local Gemma model), and a polished mission control. |

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
| `GEMMA_ENABLED` | 1 | 0 = rules only |
| `GEMMA_MODEL` | gemma4:e2b | falls back to gemma4:e4b, then gemma3:4b if not installed |
| `GEMMA_URL` | http://host.docker.internal:11434 | where Ollama runs |
| `GEMMA_FAKE` | 0 | 1 = built-in fake model, for demos without Ollama |
| `OPERATOR_TIMEOUT_S` | 20 | seconds a rover waits for an answer before taking the safe option |

## Gemma (local AI)

The rovers run fine on rules alone. With Gemma they also get a second opinion on their goals, an interpretation of anything odd they see (from the camera image plus sensor numbers), and the ability to understand typed instructions.

1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull gemma4:e2b`
2. Start Redshift with `docker compose up --build`. The Gemma pill in the dashboard header shows `warming` while the model loads (about 40 s the first time), then `ready`.
3. If the pill says `offline`, the container cannot reach Ollama. If Ollama runs inside WSL, make it listen beyond localhost: run `sudo systemctl edit ollama`, add the two lines `[Service]` and `Environment="OLLAMA_HOST=0.0.0.0"`, then `sudo systemctl restart ollama`. See `sim/GEMMA.md` for details and other setups.

The rules always keep control of safety: Gemma's picks are only taken from options the rules already checked, and if Gemma is slow, offline or gives a bad answer, the rules decide.

## Using the dashboard

- **Sensor panel:** click a rover card, the rover itself, or press 1 to 3. Live camera (onboard or chase view, rendered by MuJoCo), LiDAR radar, IMU, wheel slip, motor load, spectrometer, and brain time against the budget.
- **Hazards bar:** Rockfall, Soft sand, Dust storm, Motor fault, aimed at a chosen rover or Auto.
- **Comms tab:** type instructions ("rover 2, explore north", "everyone come home", "orange one, sample the vent"). Gemma turns them into commands, the rules check they are safe, and the rover replies or refuses with its reasons. Rover questions appear here and as a card at the top, with the rover's camera view and a countdown.
- **Reality:** overlays the true terrain and every real target, so you can compare it with what the fleet knows.

## Run natively with the MuJoCo viewer

Docker runs the sim without a window. To watch the real 3D world in MuJoCo's own viewer, run the sim directly. On WSL, do this from a copy of the repo in your Linux home folder, since Python virtual environments on `/mnt/c` are slow and fragile.

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

Run the tests with `cd sim && python -m pytest -q tests/`. For a fast headless run with scripted hazards: `python -m redshift.sim --seconds 400 --hazard rockfall@60 --hazard sand@120:r1 --hazard dust_storm@200`.

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
sim/                 MuJoCo world, rover agents, hazards, cameras, Gemma advisor, FastAPI + WebSocket server
web/                 React + Three.js mission control dashboard
```

## How the rovers decide

Fast tier (rules, 5 times per sim second, inside a 40 ms budget):

1. **Sense.** LiDAR scans go into one shared height map; the IMU gives tilt; wheel encoders and motor feedback give slip and load.
2. **Pick a goal.** Frontiers are the edges of explored ground. Rovers score them by size, distance and turning, and avoid ones another rover has claimed, so the fleet spreads out.
3. **Plan a path.** A* over the map, avoiding ground steeper than the fleet's slope limit, soft zones and rocks.
4. **Learn on the fly.** The fleet starts out assuming it can climb 30 degrees. A stall on a slope (with the motors near their limit) lowers the limit for everyone.
5. **Stay safe.** Anything unexpected makes the rover slow down before it replans. It backs out of soft ground, routes around new rocks, crawls in dust, and keeps enough battery to get home plus a 15% margin (more when dust covers the lander's solar panels).

Slow tier (local Gemma, a few times a minute, inside a 3 s budget):

6. **Second opinion on goals.** Gemma picks between the safe options the rules prepared and explains why, quoting readings.
7. **Interpret anomalies.** Gemma looks at the camera image and sensor numbers and says what it thinks it is. If it is unsure, the rover asks mission control.
8. **Understand the operator.** Typed instructions become structured commands that the rules check before acting.

## Two minute demo

1. Start on the overview: the map is black and three rovers leave the lander. Point at the Gemma pill (`ready`).
2. Turn on Paths: the rovers split up toward different frontiers.
3. About 45 seconds in, a rover stalls on a slope: the fleet slope limit drops and the colours update for everyone.
4. Press **Rockfall**: boulders drop on a rover's route, its LiDAR sees them, it slows, Gemma looks at the camera, and the rover asks you what to do. Answer, or let the countdown take the safe option.
5. Press **Soft sand**: the rover's wheels spin (open its sensor panel to see slip jump), it backs out, and the whole fleet avoids the patch. This is how NASA's Spirit rover was lost.
6. Press **Dust storm**: LiDAR range collapses on the radar, rovers slow down and stick to mapped ground, and charging at the lander drops.
7. In Comms, type "everyone come home": Gemma turns it into a command and each rover replies.
8. Turn on Reality: what the fleet knows next to the real planet.
