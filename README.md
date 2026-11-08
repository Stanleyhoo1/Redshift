# Redshift

Autonomous rovers exploring an unknown super-Earth. Built for the Agentverse 2026 Robotics track.

Three rovers land on a planet with twice Earth's gravity. There is no map and no link home: a message to Earth would take years to arrive, so everything runs on board, including the AI. Each rover senses the ground with a simulated LiDAR, IMU, wheel encoders and a camera, decides where to go, and drives there in a physics simulation (MuJoCo). The fleet shares one map of a 120 m region, splits up the unexplored ground, learns how steep a slope it can really climb, and replans when the world changes under it: rockfalls, hidden soft sand, dust storms, toppling crystal spires and breakdowns. Along the way it finds lakes (LiDAR gets no echo off liquid), meets slow alien grazers that it stops to observe from a safe distance, and collects samples into a limited set of tubes that it caches at the lander. Each rover also runs an onboard object detector on its own camera (rocks, spires, creatures, water, science targets) and acts on what it sees. A small Gemma model running locally makes the rovers' judgement calls. If a human crew member is on the mission (for example in the lander or an orbiting habitat), they can talk to the rovers in plain language, give orders, and switch to supervised mode to approve decisions themselves. Nothing ever goes back to Earth. A web dashboard shows the planet appearing as the rovers explore.

## How it fits the judging

| Criterion | Redshift |
|---|---|
| Perception-driven action | Every decision comes from the rover's own sensors and every event says which reading caused it ("slip 82%", "slope 21 deg with motors at 87%", "Camera: creature 4 m ahead (0.99) where LiDAR track k1 moves"). The onboard detector boxes what each camera sees, live on the dashboard. Click an event to fly to where it happened. |
| Adaptation to change | Judges pick a hazard and click where it should happen, and the scene changes mid-run. The rovers only find out through their sensors (LiDAR sees new rocks or a fallen spire across the path, wheel slip reveals sand, LiDAR returns shrink in dust, LiDAR points that move become a tracked creature, a teammate reports a fault) and visibly replan. |
| Operational autonomy | Two stated budgets, measured live: each rover's rule brain gets 40 ms per 200 ms cycle, and Gemma calls get 3 s (about 0.3 to 1.5 s on an RTX 4070). Autonomous by default: Gemma makes the judgement calls and every decision is logged with its confidence and reason; below 0.5 confidence the safest option wins. Flip to Supervised (for when a crew member is on site) and the rovers ask the crew before every decision, holding still (unless staying is the danger) until answered, and heading home on their own if the battery runs low while waiting. |
| Relevance and impact | Planetary rovers cannot wait minutes or years for instructions. The failure that ended NASA's Spirit rover, getting stuck in soft soil, is detected and handled on screen. Rovers fill a small number of sample tubes and cache them at the lander, the way Perseverance does for Mars Sample Return, so they have to judge which finds are worth a tube. Possible life is observed from at least 4 m and never touched (planetary protection). |
| Ambition and quality | Real physics at 2 g, a coordinated fleet, online learning, creature tracking from LiDAR, lake detection from missing returns, a science sampling loop, an onboard camera object detector, a two-tier brain (fast rules plus a local Gemma model), natural-language command by place with chat memory, and a polished, realistic crew dashboard. |

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
| `MAP_SIZE` | 120 | map width in metres |
| `GRAVITY_G` | 2.0 | gravity in Earth g |
| `AUTO_RESET_S` | 0 | off: the planet only changes when you press Reset sim. Set it (e.g. 60) to land on a new random planet that many seconds after exploration finishes |
| `GEMMA_ENABLED` | 1 | 0 = rules only |
| `GEMMA_MODEL` | gemma4:e2b | falls back to gemma4:e4b, then gemma3:4b if not installed |
| `GEMMA_URL` | http://host.docker.internal:11434 | where Ollama runs |
| `GEMMA_FAKE` | 0 | 1 = built-in fake model, for demos without Ollama |
| `AUTONOMY_MODE` | autonomous | `autonomous` (Gemma decides, every decision logged) or `supervised` (you are asked before every decision); also a switch in the dashboard |
| `OPERATOR_TIMEOUT_S` | 0 | 0 = questions wait for you; above 0 = seconds before the rover takes the safe option |
| `CAMERA_SHADOWS` | 1 | 0 = camera renders without sun shadows (faster on slow CPUs) |

## Gemma (local AI)

The rovers run fine on rules alone. With Gemma they also get a second opinion on their goals, an interpretation of anything odd they see (from the camera image plus sensor numbers), and the ability to understand typed instructions.

1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull gemma4:e2b`
2. Start Redshift with `docker compose up --build`. The Gemma pill in the dashboard header shows `warming` while the model loads (about 40 s the first time), then `ready`.
3. If the pill says `offline`, the container cannot reach Ollama. If Ollama runs inside WSL, make it listen beyond localhost: run `sudo systemctl edit ollama`, add the two lines `[Service]` and `Environment="OLLAMA_HOST=0.0.0.0"`, then `sudo systemctl restart ollama`. See `sim/GEMMA.md` for details and other setups.

The rules always keep control of safety: Gemma's picks are only taken from options the rules already checked, and if Gemma is slow, offline or gives a bad answer, the rules decide.

## Using the dashboard

The dashboard plays two roles: the judges' window into what the rovers sense and decide, and the console of a crew member on the mission. The hazard buttons are for the demo (the rovers are never told about them); Comms, the request cards and the Autonomous / Supervised switch are what a crew member on site would use.

- **Sensor panel:** click a rover card, the rover itself, or press 1 to 3. Live camera (onboard or chase view, rendered by MuJoCo), LiDAR radar, IMU, wheel slip, motor load, spectrometer, and brain time against the budget.
- **Hazards bar:** pick Rockfall, Soft sand, Falling spire or Lifeform, then click the map where it should happen (a ghost marker shows the spot; Esc cancels; right-click lets the sim choose a spot on a rover's route). If a spot is unsafe (on the lander, in water, too steep, would trap a rover) the bar says why and you can click somewhere else. Dust storm is one click for the whole map. Motor fault: press it, then click a rover. Each run has enough boulders for about 25 rockfalls and 20 spare spires to drop on open ground (any standing spire can also be toppled once); Reset sim restores them.
- **Samples tab:** every sample collected so far, sorted by science value, with the instrument used (drill, scoop, gas sniffer, swab), the lab result, and whether it is still on a rover or cached at the lander. Rover cards show their 3 tube slots and a progress bar while sampling.
- **Comms tab:** talk to the fleet in plain language. Commands by place ("go to the water source", "can we investigate the lake for signs of life", "explore around the vent", "explore one of those holes on the map", "prioritize the gaps", "back to normal"), questions ("rover 2, what have you done so far?", "what did we find there?", "seen any life?"), and references to earlier messages ("go back there", "same for rover 3"). The rover picked in the box answers unless the text names another. If several places match, they are numbered on the map: click one, or type "the second one". Rover questions appear here and as a card at the top with the rover's camera view.
- **Autonomous / Supervised switch** (under the title): who makes the rovers' judgement calls. Decisions show in the event log with who decided, the confidence and why; in supervised mode the cards mark Gemma's suggestion.
- **Event log filters:** All, one per rover, Fleet, and Decisions.
- **Cameras:** click the camera image for a large view (scroll to zoom, drag to pan, V for onboard or chase, D to show or hide detection boxes). The sensor panel's expand buttons open a big radar and history charts.
- **Area exploration:** "explore around X" draws a dashed circle with a countdown while the rover maps it (or re-surveys it if it is already mapped), then it reports what it found. Rovers also stay curious on their own for about a minute after a new find.
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

Run the tests with `cd sim && python -m pytest -q tests/`. For a fast headless run with scripted hazards: `python -m redshift.sim --seconds 400 --hazard rockfall@60 --hazard sand@120:r1 --hazard dust_storm@200 --hazard spire_fall@250:12,-30` (`@time`, then `:rover` or `:x,y`).

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
web/                 React + Three.js crew dashboard
```

## How the rovers decide

Fast tier (rules, 5 times per sim second, inside a 40 ms budget):

1. **Sense.** LiDAR scans go into one shared height map; the IMU gives tilt; wheel encoders and motor feedback give slip and load.
2. **Pick a goal.** Frontiers are the edges of explored ground. Rovers score them by size, distance and turning, and avoid ones another rover has claimed, so the fleet spreads out.
3. **Plan a path.** A* over the map, avoiding ground steeper than the fleet's slope limit, soft zones and rocks.
4. **Learn on the fly.** The fleet starts out assuming it can climb 30 degrees. A stall on a slope (with the motors near their limit) lowers the limit for everyone.
5. **Stay safe.** Anything unexpected makes the rover slow down before it replans. It backs out of soft ground, routes around new rocks, crawls in dust, and keeps enough battery to get home plus a 15% margin (more when dust covers the lander's solar panels).
6. **Spot water.** Liquid gives no LiDAR echo. A patch of missing returns that the camera shows as liquid becomes a lake on the shared map and everyone routes around it; a hole that is not liquid is treated as a drop-off.
7. **Watch for life.** LiDAR points that move between scans become a track, and the camera detector confirms it is a creature. The rover follows at about 6 m (never closer than 4 m) and Gemma (or the crew, in supervised mode) decides whether to keep following; it logs the herd's size and speed.
8. **Collect samples.** The spectrometer estimates each target's science value. A rover samples with the right tool (drill for minerals, scoop for ice, sniffer for vents, swab for organic traces), but skips low-value targets when only one tube is left, and caches full tubes at the lander.
9. **See with the camera.** An onboard detector (RGB and depth pixels only) labels rocks, boulders, spires, creatures, water, minerals, ice, vents and traces. It confirms water and creatures, tells a fallen spire from rocks, and spots targets before the spectrometer reaches them. Its accuracy against the truth is measured in the sim (`GET /api/vision/stats`); the rovers never see the truth.

Slow tier (local Gemma, a few times a minute, inside a 3 s budget):

10. **Second opinion on goals.** Gemma picks between the safe options the rules prepared (frontiers, samples, going home) and explains why, quoting readings.
11. **Judgement calls.** Gemma looks at the camera image, the detector's boxes and the sensor numbers and decides what to do about anything unexpected (autonomous), or suggests it to the crew (supervised).
12. **Understand the crew.** Typed instructions become structured commands that the rules check before acting, with a short memory of the conversation.

## Two minute demo

1. Start on the overview: the map is black and three rovers leave the lander. Point at the Gemma pill (`ready`).
2. Turn on Paths: the rovers split up toward different frontiers.
3. About 45 seconds in, a rover stalls on a slope: the fleet slope limit drops and the colours update for everyone.
4. Press **Rockfall** and click just ahead of a rover: boulders drop, its LiDAR sees them, the camera detector labels them, it slows, and Gemma decides to go around (see the Decisions filter). Flip to **Supervised** and do it again: now the rover asks you, with Gemma's suggestion marked.
5. Press **Soft sand** and click on a rover's path: its wheels spin (open its sensor panel to see slip jump), it backs out, and the whole fleet avoids the patch. This is how NASA's Spirit rover was lost.
6. Press **Falling spire** and click next to a crystal spire on a rover's route: it topples across the path and the rover reroutes around the new long obstacle.
7. Press **Lifeform** and click near a rover: a herd of grazers wanders in, LiDAR picks up moving points, the camera boxes them as creatures, and the rover follows at a safe distance. Open its camera large to show the detection boxes.
8. Press **Dust storm**: LiDAR range collapses on the radar, rovers slow down and stick to mapped ground, and charging at the lander drops.
9. Open the **Samples** tab: what has been collected, its science value, and what is cached at the lander. Point out a rover skipping a low-value target because it has one tube left.
10. In Comms: "can we investigate the water source for signs of life" (the closest rover goes, surveys and reports), then "what did we find there?", then "explore one of those holes on the map".
11. Turn on Reality: what the fleet knows next to the real planet, including lakes it has not reached yet.
