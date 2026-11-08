# RedShift

A fleet of autonomous rovers exploring an unknown super-Earth, in a MuJoCo physics simulation with a live 3D dashboard. Built for the Agentverse 2026 Robotics track.

A group of rovers land and are deployed on a planet with twice Earth's gravity. There is no map, and a message to Earth would take years, so everything runs on board, including the AI. Each rover senses the world with a simulated LiDAR, IMU, wheel encoders and a camera, decides where to go, and drives there. The fleet shares one map, splits up the unexplored ground, learns how steep it can really climb, and replans when the world changes: rockfalls, soft sand, dust storms, falling crystal spires, breakdowns. It finds lakes, watches alien life from a safe distance, samples what is worth a tube, and looks after its own: a broken rover calls for help and its teammates come to diagnose, patch, tow or salvage it. A small Gemma model running locally makes the judgement calls. A crew member on the mission can talk to the rovers in plain language and approve decisions in supervised mode.

## How it fits the brief

| Criterion | Redshift |
|---|---|
| Perception-driven action | Every decision cites the reading behind it ("slip 82%", "slope 21 deg with motors at 87%", "camera: creature 4 m ahead (0.99) where LiDAR track k1 moves"). An onboard detector boxes what each camera sees. Click an event to fly to where it happened. |
| Adaptation to change | Hazards are clicked onto the map mid-run. The rovers only find out through their sensors (new rocks or a fallen spire in the LiDAR, wheel slip in sand, shrinking returns in dust, moving points that become a creature track) and replan. One rover's discovery changes everyone's plan, and a breakdown pulls the best placed teammate onto a rescue. |
| Operational autonomy | Measured budgets: 40 ms per 200 ms cycle for each rover's rules, 3 s per Gemma call. Autonomous by default: Gemma decides, every decision is logged with confidence and reason, and below 0.5 confidence the safest option wins. Supervised mode asks the crew instead. Rovers slow down when unsure and head home on low battery, even mid-question. |
| Relevance and impact | Rovers cannot wait for instructions from Earth. The failure that ended NASA's Spirit rover (stuck in soft soil) is detected and handled on screen. Samples go into a few tubes cached at the lander, as on Perseverance. Life is observed from at least 4 m, never touched. |
| Ambition and quality | Real 2 g physics, a coordinated fleet, online learning, creature tracking from LiDAR, lake detection, a camera object detector, rover health with self-diagnosis, cooperative rescue with a physical tow hitch, seeded biomes, natural-language commands with chat memory, and a realistic dashboard. |

## Quick start

You need Docker with Compose (Docker Desktop with WSL integration, or Docker Engine inside WSL).

```bash
docker compose up --build
```

Open http://localhost:8080, choose the mission settings on the setup screen and press Launch. If the page stays on "Waiting for the simulation", check `docker compose logs --tail=60 sim`.

Settings are environment variables, for example `SEED=7 SIM_SPEED=5 docker compose up --build`:

| Variable | Default | Meaning |
|---|---|---|
| `SEED` | 42 | the planet (terrain, biomes, targets) |
| `NUM_ROVERS` | 3 | rovers in the fleet (1 to 8) |
| `SIM_SPEED` | 3.0 | sim seconds per real second |
| `MAP_SIZE` | 120 | sector width in metres (60 to 200) |
| `GRAVITY_G` | 2.0 | gravity in Earth g (0.5 to 3) |
| `AUTO_LAUNCH` | 0 | 1 = skip the setup screen |
| `AUTONOMY_MODE` | autonomous | or `supervised` (the crew is asked every decision); also a switch in the dashboard |
| `OPERATOR_TIMEOUT_S` | 0 | 0 = questions wait for an answer; above 0 = seconds before the safe default |
| `GEMMA_ENABLED`, `GEMMA_MODEL`, `GEMMA_URL` | 1, gemma4:e2b, host Ollama | the local model (`GEMMA_FAKE=1` for a built-in stand-in) |
| `BIOMES`, `LIFE_VARIETY`, `MESAS`, `CANYON`, `DUNES`, `BOULDER_FIELDS`, `PITS`, `GEYSERS`, `ICE_SHEETS`, `FLORA`, `GROUND_DETAIL` | 1 | world features (0 leaves one out) |
| `CAMERA_SHADOWS` | 1 | 0 = faster camera rendering on slow CPUs |

The full list is in `sim/README.md`.

## Gemma (local AI)

The rovers run on rules alone; Gemma adds a second opinion on goals, judgement calls about anything odd (from the camera image, the detector and sensor numbers), and plain-language chat.

1. Install [Ollama](https://ollama.com) and run `ollama pull gemma4:e2b`.
2. Start Redshift. The Gemma pill in the header shows `warming` (about 40 s the first time), then `ready`.
3. If it says `offline` and Ollama runs in WSL, make it listen beyond localhost: `sudo systemctl edit ollama`, add `[Service]` and `Environment="OLLAMA_HOST=0.0.0.0"`, then `sudo systemctl restart ollama`. More in `sim/GEMMA.md`.

The rules always keep control of safety: Gemma only picks from options the rules have checked, and if it is slow, offline or wrong, the rules decide.

## How the rovers work

Fast tier (rules, 5 times per sim second, inside 40 ms):

- **Map and explore.** LiDAR builds one shared height map. Rovers pick frontiers (edges of the known ground) by size, distance and turning, avoid ones a teammate claimed, and plan paths with A* around steep ground, soft zones and rocks.
- **Learn.** The fleet assumes it can climb 30 degrees; a stall with the motors near their limit lowers the limit for everyone.
- **Stay safe.** Anything unexpected slows a rover before it replans. It backs out of soft ground, routes around new rocks, crawls in dust, and keeps enough battery to get home.
- **Water.** Liquid gives no LiDAR echo; a gap in the returns that the camera shows as liquid becomes a lake everyone avoids.
- **Life.** Moving LiDAR points become a track, confirmed by the camera. The rover follows at about 6 m (never closer than 4 m) and reports the group's size and speed.
- **Samples.** The spectrometer estimates a target's science value; the rover uses the right tool (drill, scoop, gas sniffer, swab), skips weak finds when one tube is left, and caches full tubes at the lander.
- **Camera detector.** From RGB and depth pixels only, it labels rocks, boulders, spires, creatures, water, minerals, ice, vents and traces, confirms water and creatures, tells a fallen spire from rocks, and spots targets before the spectrometer reaches them. Its accuracy against the truth is measured (`GET /api/vision/stats`); the rovers never see the truth.
- **Health and rescue.** Shocks, hard contacts, rollovers, water and motor strain damage individual parts (drives, LiDAR, camera, power, comms), which changes how a rover drives and senses. It diagnoses itself and heads home for repair if it can still drive. If not, it sends a distress signal (or the others notice it went silent). The best placed teammate drops everything else, diagnoses it, fixes simple faults on the spot or patches the drive so it can limp home for a repair with spare parts. A wreck is towed home on a hitch, salvaged, or left. A helper that has to give up says why.

Slow tier (local Gemma, inside 3 s):

- **Judgement calls.** What an anomaly is and what to do about it, whether to follow a creature, which of several places an order means, and tow, salvage or leave. In supervised mode these become questions to the crew with Gemma's pick marked.
- **Goals.** A second opinion on the rules' goal, stated with real numbers ("a gentler route, steepest 15 vs 19 deg").
- **Chat.** Orders and questions in plain language ("go to the water source", "explore one of those holes", "what did we find there?", "rover 2, are you damaged?"), answered from the rover's own memory with every number checked.

## The world

The seed decides the planet. Each feature can be switched off on the setup screen or with its variable.

- **Terrain:** mesas with cliff walls, a canyon with two ramps, pits and lava-tube skylights, boulder fields, rippled dunes (more slip), ice sheets (almost no grip; rovers detect low traction and mark the zone), geysers whose steam scatters the LiDAR, lakes, and crystal spires that can topple.
- **Life:** slow grazers plus four more kinds picked by biome: skitterer swarms that scatter, drifters floating with the wind, big shelled crawlers (kept at 6 m) and burrowers that pop up from mounds. Alien flora too: fungal trees and glowing bulb bushes, kept at 4 m.
- **Biomes:** the seed picks a primary biome for the sector and a secondary one for a few regions. Each sets the ground and sky colours and scales the terrain mix; the rovers never read the biome, they only see the colours.

| Biome | Ground | More of | Less of |
|---|---|---|---|
| Basalt fields | dark blue-grey volcanic rock | pits, boulder fields, vents | dunes, ice |
| Verdigris lowlands | grey-green mineral | flora | pits, vents, ice |
| Frost plains | pale blue-grey and white | ice sheets | vents, flora |
| Ash wastes | lavender-grey ash | geysers, dunes, pits | ice |
| Salt flats | pale bone and cream | dunes, mesas, canyon | pits, geysers |
| Obsidian badlands | glassy near-black | canyon, boulder fields, mesas | dunes, ice |

<table>
<tr>
<td width="50%"><img src="docs/images/biomes/basalt.jpg" alt="Basalt fields"><br><b>Basalt fields</b>, seed 4</td>
<td width="50%"><img src="docs/images/biomes/verdigris.jpg" alt="Verdigris lowlands"><br><b>Verdigris lowlands</b>, seed 31</td>
</tr>
<tr>
<td><img src="docs/images/biomes/frost.jpg" alt="Frost plains"><br><b>Frost plains</b>, seed 1</td>
<td><img src="docs/images/biomes/ash.jpg" alt="Ash wastes"><br><b>Ash wastes</b>, seed 10</td>
</tr>
<tr>
<td><img src="docs/images/biomes/salt.jpg" alt="Salt flats"><br><b>Salt flats</b>, seed 6</td>
<td><img src="docs/images/biomes/obsidian.jpg" alt="Obsidian badlands"><br><b>Obsidian badlands</b>, seed 42 (the default)</td>
</tr>
</table>

Terrain features, close up with Reality on (the rovers still have to find each one with their own sensors):

<table>
<tr>
<td width="50%"><img src="docs/images/terrain/mesa.jpg" alt="Mesas"><br><b>Mesas and buttes</b>: cliff walls no rover can climb</td>
<td width="50%"><img src="docs/images/terrain/canyon.jpg" alt="Canyon"><br><b>Canyon</b>: a winding trench, crossable only at its two ramps</td>
</tr>
<tr>
<td><img src="docs/images/terrain/pits.jpg" alt="Pit"><br><b>Pits and skylights</b>: sheer drops the planner treats as impassable</td>
<td><img src="docs/images/terrain/boulders.jpg" alt="Boulder field"><br><b>Boulder fields</b>: dense rocks to steer around</td>
</tr>
<tr>
<td><img src="docs/images/terrain/dunes.jpg" alt="Dunes"><br><b>Dunes</b>: rippled loose sand where the wheels slip more</td>
<td><img src="docs/images/terrain/ice.jpg" alt="Ice sheet"><br><b>Ice sheets</b>: level ground with almost no grip</td>
</tr>
<tr>
<td><img src="docs/images/terrain/geyser.jpg" alt="Geyser"><br><b>Geysers</b>: vents on a schedule whose steam scatters the LiDAR</td>
<td><img src="docs/images/terrain/flora.jpg" alt="Alien flora"><br><b>Alien flora</b>: fungal trees and glowing bulb bushes, kept at 4 m</td>
</tr>
<tr>
<td><img src="docs/images/terrain/spires.jpg" alt="Crystal spires"><br><b>Crystal spires</b>: tall and brittle, they can topple</td>
<td><img src="docs/images/terrain/water.jpg" alt="Lake"><br><b>Lakes</b>: liquid the rovers must spot and keep out of</td>
</tr>
</table>

Lifeforms. The mix follows the biome, and the rovers never read the true kind:

<table>
<tr>
<td width="50%"><img src="docs/images/life/grazer.jpg" alt="Grazers"><br><b>Grazers</b>: six-legged herds that wander and graze</td>
<td width="50%"><img src="docs/images/life/skitterer.jpg" alt="Skitterers"><br><b>Skitterers</b>: small fast swarms that scatter from a rover</td>
</tr>
<tr>
<td><img src="docs/images/life/drifter.jpg" alt="Drifters"><br><b>Drifters</b>: glowing bells drifting with the wind, trailing tendrils</td>
<td><img src="docs/images/life/crawler.jpg" alt="Crawler"><br><b>Crawlers</b>: big, slow and shelled, kept at 6 m</td>
</tr>
<tr>
<td><img src="docs/images/life/burrower.jpg" alt="Burrower"><br><b>Burrowers</b>: pop up from a mound for a few seconds, then sink</td>
<td><img src="docs/images/life/observe.jpg" alt="Rover observing a crawler"><br><b>Observing</b>: a rover holds its distance while it watches a crawler</td>
</tr>
</table>

To retake the pictures: start the sim with `AUTO_LAUNCH=1` and a vite pointed at it, then run `node scripts/biome-shots.mjs`, `node scripts/feature-shots.mjs` or `node scripts/life-shots.mjs` from `web/` (each explains its arguments at the top).

## The dashboard

It is both the judges' window into what the rovers sense and decide, and the console of a crew member on the mission.

- **Map:** the planet appears as the rovers map it. Toggles for Reality (the true world), frontiers, paths, slope colours, LiDAR and life. Follow a rover with 1 to 3 (drag away to pan freely, F to return). G opens the planet view from orbit.
- **Rover cards and sensor panel:** state, battery, health bar and part icons, tubes; live camera with detection boxes (click for a large zoomable view), LiDAR radar, IMU, slip, load, the rover's own diagnosis, and brain time against the budget.
- **Hazards:** Rockfall, Soft sand, Falling spire and Lifeform are clicked onto the map (unsafe spots are refused with a reason). Dust storm covers the map. Motor fault and Damage pick a rover; Damage also picks a part and Small (25 points), Huge (60), Disable, Wreck or Jam. The rovers are never told; they find out through their sensors.
- **Event log:** Highlights by default (decisions, discoveries, hazards, damage), or All, per rover, Fleet, Decisions and Damage.
- **Comms:** chat with the fleet or one rover. Ambiguous places are numbered on the map to pick from. Rover questions also appear as cards with the rover's camera view.
- **Autonomous / Supervised:** who makes the judgement calls.
- **Samples:** everything collected, by science value, and what is cached at the lander.
- **Folding:** every panel folds and remembers it: the controls (Layers, Camera, Mission), the map key, the Hazards pill, the question cards (to an "N questions waiting" chip) and each sidebar section (with a one-line summary). The sidebar itself collapses to a thin rail that still shows each rover's state and any waiting questions. Events are one line each until clicked.

## Development

Run the sim outside Docker to watch MuJoCo's own 3D viewer (on WSL, from a copy in your Linux home folder):

```bash
cd sim && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m redshift.server --viewer
cd web && npm install && npm run dev      # second terminal, http://localhost:5173
```

- Tests: `cd sim && python -m pytest -q tests/`
- Headless run with scripted hazards: `python -m redshift.sim --seconds 400 --hazard rockfall@60 --hazard sand@120:r1` (`@time`, then `:rover` or `:x,y`)
- UI with fake data: `cd web && npm run mock` (port 8000) and `npm run dev`

```
PROTOCOL.md          message format between the sim and the dashboard
docker-compose.yml   runs both services
sim/                 MuJoCo world, rover agents, hazards, cameras, detector, health and rescue, Gemma, FastAPI server
web/                 React + Three.js dashboard
docs/                images
submission/          technical summary, video plan and pitch
```
