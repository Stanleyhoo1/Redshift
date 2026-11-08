# Redshift simulation backend

A fleet of rovers explores an unknown super-Earth at 2 g. Each rover only knows what its own
sensors tell it. The rovers build a shared map from LiDAR, pick where to go next, learn how steep
a slope they can really climb, find science targets, take samples with the right instrument, carry
them in a few sample tubes and cache them at the lander, and drive home to charge. They make every
decision on their own; nothing is scripted. The planet has lakes and rivers (LiDAR gets no echo from
liquid), tall alien crystal spires, and herds of slow six-legged grazers that the rovers watch from
a safe distance. The world can also change while they work (rockfalls, soft sand, dust storms, motor
faults, a spire falling across a route) and they have to notice and cope. The web UI (in `../web`)
shows the world filling in. The message format is in `../PROTOCOL.md`.

The brain has two tiers. Fast hand-written rules decide everything, 5 times a second, and keep the
rovers safe. A small Gemma model running on your own machine gives advice beside them: which goal
to pick, what an odd sensor reading means (with the camera picture), what the operator's typed
messages mean, and short answers to the operator's questions written from each rover's own memory.
The rules check every piece of advice. See "Two-tier brain" below and `GEMMA.md`.

## What each module does

| File | What it does |
|---|---|
| `redshift/config.py` | All settings in one place. Reads the env vars below. `SETTINGS` is the table of launch settings (key, range, label, help) behind the setup screen, `GET /api/config` and `POST /api/launch`; the slope thresholds tuned for 2 g follow the launch gravity (see "Learning the slope limit"). |
| `redshift/terrain.py` | Makes the ground: plains, hills, ridges, craters (each with a gentle way in), rocks, science targets in clusters, one or two lakes (often with a short river), alien spires, the home areas of the grazer herds, organic "trace" targets near them, and each target's science value and analysis. Feature counts grow with the map area. Same seed, same world. |
| `redshift/world.py` | Builds the MuJoCo physics model: the ground as height-map tiles, rocks, the lander pad, target markers, liquid surfaces, spires, creatures, hidden rockfall boulders and spare spires, and the rovers (4 wheels, skid steering, front axle that rocks so all wheels touch, an IMU). |
| `redshift/mapping.py` | The shared fleet map. Turns LiDAR hits into heights and slopes for each grid cell, notices when mapped ground changes, keeps layers for soft ground, identified water and "no echo here" evidence, and finds frontiers (edges of the known area). |
| `redshift/planner.py` | A* path planning on the map. Steep cells, obstacles, soft ground and water are off limits. The search can stop when its time is up and carry on in the next cycle. |
| `redshift/rover.py` | One rover's brain: sensors, driving, battery, samples, and its state (exploring, investigating, returning, charging, recovering, idle, fault, observing; v9: disabled, rescuing, towing, towed, repairing, wrecked). Applies its health to the motors and sensors, and backs away from a charging creature. |
| `redshift/fleet.py` | Team decisions: who goes to which frontier, the shared slope limit, the target list and the sample cache, soft ground and water reports, who observes a lifeform, who takes over from a broken rover, and the event log. |
| `redshift/tracker.py` | Motion tracker: LiDAR point clusters that move between sweeps become tracks (position, speed, size). |
| `redshift/lifeforms.py` | The grazers: they wander, graze, keep together, avoid bad ground and shy away from rovers. The rovers never read this. |
| `redshift/v7_life.py` | "More lifeforms" (`LIFE_VARIETY`): the kinds (skitterer, drifter, crawler, burrower), their biome weights, which kind each group is (own stream `default_rng([seed, 7113])`), their MuJoCo bodies (few LiDAR geoms; drifter tendrils hang low enough for the LiDAR, a burrower sinks below the ground) and their behaviour. Ground truth only. |
| `redshift/hazards.py` | Changes the world on request: rockfall, sand, dust storm, rover fault, spire fall, and (demo) a herd walking into view. Rockfall, sand, spire fall and lifeform can be put at a clicked point. Only the UI sees this list; the rovers must find out for themselves. |
| `redshift/sim.py` | Runs it all: steps the physics, runs the rovers' decisions 5 times per sim second, and builds the messages for the UI. Also a command line runner that can script hazards. |
| `redshift/camera.py` | Live pictures from each rover's mast camera (MuJoCo renders, up to 1280 x 720), for the UI and for Gemma: sun shadows placed in front of each camera, distance haze from the depth buffer, a dust veil in storms. Renders the views clients keep asking for in the background and answers with the newest frame at once (protocol v8), culls what cannot change a pixel, and runs the rovers' detector renders. |
| `redshift/vision.py` | Onboard object detection (protocol v8): a classical detector on each rover's own camera RGB + depth (11 classes), what the rover decides from it (`VisionBrain`), and the ground truth scoring used only offline and by `GET /api/vision/stats`. |
| `redshift/visuals.py` | The look (visual only): procedural regolith texture, rock grain, solar cells, a hazy sky cube with a low sun, lights, haze colours and materials. In the camera, science targets look like natural features with the dashboard colours as accents (a vent is a low mound with a dark mouth and an orange rim, a mineral an angular outcrop with violet crystal veins, ice a glossy white-blue patch, a trace a thin film with lime specks), spires are faceted violet crystals and grazers have segmented, softly glowing green bodies. These are massless, non-colliding geoms in group 4: LiDAR, the spectrometer, the tracker and machine vision still use the original marker capsules and bodies (the camera's look frames leave the target capsules out). |
| `redshift/memory.py` | Each rover's mission memory, updated where things happen: where it mapped (compass sector and distance band around the lander), distance, targets found and sampled, hazards met and how it handled them, lifeforms observed, notable events, its goal and why. Gives the `stats` in every tick. |
| `redshift/answers.py` | Answers to the operator's questions: a fact sheet per rover (and for the fleet), the facts Gemma may use, and rule-based answers for every topic. |
| `redshift/places.py` | Commands that name a place (protocol v6): finds the known places of a kind from fleet knowledge only (water bodies, found targets, mapped spires, herds, hazard sites, the lander), applies qualifiers (nearest, biggest, north, the second one, near rover 2), finds a safe goal next to a place, writes short labels and reads which option a chat answer picks. |
| `redshift/gemma.py` | Talks to the local Gemma model (Ollama), or a fake model for demos and tests: prompts, answer checks, timeouts, a keyword parser for typed commands. |
| `redshift/advisor.py` | Tier 2: asks Gemma at the right moments, applies its advice when it is safe, runs the operator chat (commands and questions) and the questions to the operator. |
| `redshift/chatmem.py` | Chat memory (protocol v7): the last messages and what they talked about (places, targets, candidate lists, the rover, the last command, survey and focus reports), so "go back there", "the second one", "that lake" and "do that again" resolve. |
| `redshift/health.py` | Protocol v9: each rover's parts (drive_l, drive_r, lidar, camera, power, comms), hp, status and self diagnosis, how damage changes the rover (torque and top speed per drive side, LiDAR range and rays, camera noise, battery drain, radio range), and the physics monitor that turns shocks and contact forces into damage. |
| `redshift/rescue.py` | Protocol v9: distress and the silent-rover heartbeat, picking the helper, diagnosis, field repair, the tow / salvage / leave decision (Gemma or a `rescue` request), physical towing with a kinematic fallback, salvage, wrecks, repair at the lander with spares. |
| `redshift/gaps.py` | Unexplored gaps (protocol v8): connected pockets of unknown ground inside the map with known ground around most of their edge (narrow channels sealed first), with the frontiers on their edge. A few ms on the 120 m map, cached by the fleet and refreshed every 3 s only while needed. |
| `redshift/server.py` | The web server (FastAPI): WebSocket `/ws`, `GET /api/health`, `GET /api/truth`, `GET /api/config`, `POST /api/launch`, `POST /api/reset`, `POST /api/hazard`, `POST /api/chat`, `POST /api/answer`, and the camera images from `camera.py`. Starts Gemma and builds the world; with `AUTO_LAUNCH=0` (the default) it waits in the setup phase until `POST /api/launch`, then runs the sim in the background at `SIM_SPEED` times real time. A reset or launch builds the new world before it stops the old one. |
| `scripts/climb_test.py` | Drives a rover up ramps of known angle to check how steep it can climb. |
| `scripts/gemma_bench.py` | Measures Gemma's speed and answers against a real Ollama. |
| `scripts/camera_bench.py` | Camera render time per frame (320 x 240 and 1280 x 720 by default); `--live`: request latency, frame age, frames per second and brain time with a client polling like the UI. |
| `scripts/vision_eval.py` | Detector precision and recall per class against MuJoCo segmentation (seeds 42, 7, 3; `--draw` saves frames with boxes, `--dust` in a storm). |
| `tests/test_smoke.py` | Runs 400 sim seconds on the default 120 m map with no hazards and checks the results and every message. |
| `tests/test_hazards.py` | Runs every hazard in turn and checks how the rovers react, plus the v2 message fields and `/api/hazard`. |
| `tests/test_camera.py` | Camera renders and the camera endpoint. |
| `tests/test_vision.py` | Protocol v8: the detector on rendered frames of every class, dust lowering confidence, culling changing no pixel, the `X-Frame` / `X-Frame-Age-Ms` / `X-Detections` headers and the rover's `vision` for exactly that frame, the stream cache, and every decision taken from detections. |
| `tests/test_gemma.py` | `gemma.py` against a stub Ollama server. |
| `tests/test_brain.py` | Tier 2 with the fake model: goal switches, anomaly checks, questions, every chat command, Gemma offline, v3 messages, speed and budget. |
| `tests/test_v4.py` | Water (a hole in the map, identified, avoided, the water sensor failsafe), spire fall and replan, motion tracks, observing a lifeform at a safe distance, the first-sighting question, point-and-click hazards and every 400, sampling, tubes and the cache, and the v4 message fields. |
| `tests/test_v5.py` | Questions: intent (questions vs commands), Gemma's answer checks (a bad number or id falls back to the rules), grounded rule answers for every topic, fleet vs per-rover replies, `reply_to`, `stats`, camera sizes up to 1280 x 720. |
| `tests/test_v6.py` | Place commands: polite orders vs questions, place parsing (keyword parser and checked Gemma answers), the resolver and its qualifiers, "nothing known", the closest free rover, `choose_place` answered by `POST /api/answer` and by chat, the survey and its report, requests that never expire by default (and the countdown when `OPERATOR_TIMEOUT_S` is above 0). |
| `tests/test_v7.py` | Area focus (frontier choice inside the circle, claims, the stay and its report, "explore north of the lander", survey then focus), curiosity (triggers, cooldown, limits), chat references (keyword parser and fake Gemma), following a new herd while asking, and leaving for the lander when waiting would drain the battery. |
| `tests/test_v9.py` | Protocol v9: boulder and spire hits damage, normal driving does not; drive pull, LiDAR range, power and camera effects; the `damage` hazard (every severity and every 400); distress, helper choice and field repair; the silent rover; tow home and a lander repair with a spare; waiting for a missing spare; Gemma salvaging a wreck; a supervised `rescue` request (safe / note / about); health questions and rescue orders (keyword parser, fake Gemma, in the sim); a creature charge. |
| `tests/test_v8.py` | Autonomy mode (env, `POST /api/mode`, chat; decisions by Gemma, the rules and the 0.5 floor, logged with a `decision` object; supervised asks with Gemma's suggestion; switching decides open questions), gap detection and gaps as places and as a priority, the other priorities, the re-survey and its report, area questions, and a table of operator messages through the keyword parser and the fake Gemma. |
| `tests/test_v7_biomes.py` | Biomes: the pick (deterministic, never rust as primary, every biome reachable over seeds 0 to 400), the region field and `terrain.biome_id_at`, the feature weights (frost has no flora, salt no geysers, ash no ice), init `world.biome` and `features.biome_map`, flags off byte-identical to the world before biomes, and camera safety: every biome's ground colours stay outside the target spotting tolerance, and from open ground on every biome the target spotting finds only real markers, the liquid check sees no liquid and the plant check labels no ground. |
| `tests/test_v7_settings.py` | Launch settings (protocol v7): the settings table and its parser (every bad value and its message), the round trip, `from_env` clamps and feature switches, the slope thresholds derived from gravity (identical at 2 g), Gemma's gravity text, the setup phase, `GET /api/config`, `POST /api/launch`, settings in `POST /api/reset`, the 409s, and no em dash in the files this step touched. |
| `tests/test_v7_life.py` | More lifeforms: flags off gives exactly the old herds and messages (a digest), kinds by biome weights, each kind's behaviour, LiDAR sees burrowers only while up and drifters at their height, rovers keep 4 m (6 m from crawlers), the hazard follows the biome, and Gemma's kind hint. |

`test_hazards.py` and `test_brain.py` replay scripted scenarios (seed 42, set times, exact sensor
numbers) written for the old 80 m world, so they run with `MAP_SIZE` 80 and the v4 features (water,
spires, lifeforms, slower instruments, tube limit) switched off (`test_hazards.py` also switches off the
v7 curiosity, which changes where the rovers are when a hazard is placed); `test_smoke.py` and `test_v4.py`
run on the 120 m default.

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

Server only (this is what Docker runs):

```bash
uvicorn redshift.server:app --host 0.0.0.0 --port 8000
GEMMA_FAKE=1 uvicorn redshift.server:app --port 8000      # fake Gemma: the whole pipeline without Ollama
```

Talk to the rovers (the UI's chat box does the same), ask them questions, and answer theirs:

```bash
curl -X POST localhost:8000/api/chat -d '{"text": "rover 2, check out the area north of the lander"}'
curl -X POST localhost:8000/api/chat -d '{"text": "go to 10, -5", "rover": "r1"}'
curl -X POST localhost:8000/api/chat -d '{"text": "rover 2, what have you found?"}'
curl -X POST localhost:8000/api/chat -d '{"text": "everyone, report battery"}'
curl -X POST localhost:8000/api/chat -d '{"text": "can we investigate the water source again for signs of life"}'
curl -X POST localhost:8000/api/answer -d '{"id": "q1", "option": "reroute"}'     # or "p2" for a choose_place
curl -X POST localhost:8000/api/mode -d '{"mode": "supervised"}'                   # or "autonomous" (the default)
curl -X POST localhost:8000/api/chat -d '{"text": "prioritize the unexplored gaps"}'    # "back to normal" ends it
```

Change the world while it runs:

```bash
curl -X POST localhost:8000/api/hazard -d '{"type": "rockfall"}'
curl -X POST localhost:8000/api/hazard -d '{"type": "sand", "rover": "r1"}'
curl -X POST localhost:8000/api/hazard -d '{"type": "dust_storm"}'
curl -X POST localhost:8000/api/hazard -d '{"type": "rover_fault", "rover": "r3"}'
curl -X POST localhost:8000/api/hazard -d '{"type": "spire_fall"}'                 # across a rover's route
curl -X POST localhost:8000/api/hazard -d '{"type": "lifeform", "rover": "r2"}'    # a herd walks into view
curl -X POST localhost:8000/api/hazard -d '{"type": "rockfall", "x": 12.5, "y": -30}'   # at a clicked point
curl -X POST localhost:8000/api/hazard -d '{"type": "damage", "rover": "r2", "severity": "disable"}'   # v9 demo
curl -X POST localhost:8000/api/hazard -d '{"type": "damage", "rover": "r3", "severity": "wreck"}'     # salvage demo
curl -X POST localhost:8000/api/hazard -d '{"type": "damage", "rover": "r1", "severity": "jam"}'       # field repair
curl -X POST localhost:8000/api/hazard -d '{"type": "lifeform", "rover": "r2", "charge": true}'       # a charge
curl -X POST localhost:8000/api/chat -d '{"text": "rover 1, go help rover 2"}'      # also: tow rover 2 home, leave it
```

The answer is `{"ok": true, "hazard": {...}}`, or HTTP 400 with a reason (unknown type, nowhere
sensible to put it right now, or it would leave no working rover). Without `"rover"` the sim
picks the exploring rover with the longest route ahead (or a random working one for a fault).
With `"x"` and `"y"` (world metres), rockfall, sand, spire fall and lifeform go to that point (or
as close as is safe). A rockfall lands across the route of a rover heading that way, or as a
cluster there; a spire fall topples the nearest standing spire within 6 m toward the point, or
drops a new spire there. 400 if the point is off the map, on the lander, within 2 m of a rover, or
unsuitable (water, steep ground for sand, no safe room). Dust storm and rover fault ignore the point.

Headless, as fast as possible, with a summary and the event log at the end. Hazards can be
scripted as `TYPE@SECONDS[:ROVER]` (retried for up to 60 s if there is nowhere to put it yet):

```bash
python -m redshift.sim --seconds 900 --seed 7
python -m redshift.sim --seconds 400 --hazard rockfall@60 --hazard sand@120:r1 \
       --hazard dust_storm@200 --hazard rover_fault@260:r3 --ticks
python -m redshift.sim --seconds 900 --hazard lifeform@60 --hazard spire_fall@150 --hazard rockfall@300:12,-30
```

`--ticks` also builds the UI messages and reports their size. `--model-clock` makes runs repeat
exactly (see "Edge compute budget"). `--gemma` attaches Gemma (from the `GEMMA_*` variables) and
`--say` scripts operator messages; the chat is printed at the end:

```bash
GEMMA_FAKE=1 python -m redshift.sim --gemma --seconds 200 --hazard rockfall@60 \
       --say "30:rover 2, go check out the area north of the lander" --say "45:r1 go to 10, -5"
```

Tests (about 5 minutes):

```bash
python -m pytest -q tests/        # add -s to see the event log
python tests/test_smoke.py        # same checks without pytest
python scripts/climb_test.py      # ramp test for the rover tuning
```

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `SEED` | 42 | terrain and target layout (0 or more; the setup screen offers 0 to 2147483647) |
| `NUM_ROVERS` | 3 | rovers in the fleet (1 to 8) |
| `SIM_SPEED` | 3.0 | sim seconds per wall second (0.5 to 10) |
| `MAP_SIZE` | 120 | map width in metres, 60 to 200 (feature counts grow with the area; 80 gives the old layout) |
| `GRAVITY_G` | 2.0 | gravity in Earth g (0.5 to 3); out of range values are clamped |
| `VIEWER` | 0 | 1 = open the native MuJoCo viewer |
| `BRAIN_BUDGET_MS` | 40 | compute budget per rover per decision cycle (ms) |
| `AUTO_RESET_S` | 0 | off by default. If set, once exploration is complete and every rover is parked, wait this many seconds, then start a new random planet (server only) |
| `GEMMA_ENABLED` | 1 | 0 = rules only (no Gemma, no questions; typed commands still work through the keyword parser) |
| `GEMMA_URL` | `http://host.docker.internal:11434` | Ollama; `http://localhost:11434` is tried next |
| `GEMMA_MODEL` | `gemma4:e2b` | falls back to `gemma4:e4b`, then `gemma3:4b`, if not installed |
| `GEMMA_FAKE` | 0 | 1 = fake model (fixed rules, 0.5 s delay): demos and tests without Ollama |
| `GEMMA_BUDGET_MS` | 3000 | stated latency budget per Gemma call (answers after `GEMMA_TIMEOUT_S`, 4 s, are dropped) |
| `GEMMA_ASK_EVERY_S` | 20 | a rover asks Gemma about its goal at most this often (sim s), or when it reaches a goal |
| `OPERATOR_TIMEOUT_S` | 0 | 0 = questions never expire: the rover holds until the operator answers. Above 0: wall seconds before it takes the safe default |
| `AUTONOMY_MODE` | autonomous | `autonomous`: the rovers decide (Gemma, or the rules) and log every decision; `supervised`: the operator is asked every decision. Also `POST /api/mode` and the chat |
| `OPERATOR_ASK_BELOW` | 0.75 | before v8: Gemma confidence below this asked the operator (now the mode decides; still the bar for confirming a lifeform) |
| `OPERATOR_ASK_FIRST` | 1 | before v8: the first anomaly always asked (now the mode decides) |
| `CAMERA_SHADOWS` | 1 | 0 = camera renders without the sun's shadows (about 3 times faster with software OpenGL) |
| `CAMERA_FPS` | 12 | most frames per second rendered for one streamed view (it follows the client's request rate below that) |
| `VISION` | 1 | 0 = no onboard object detection (the rovers decide from LiDAR and the old colour check only) |
| `VISION_HZ` | 1.0 | detector renders per rover per wall second (small 256 x 192 frames without shadows; a streamed fpv view counts too) |
| `AUTO_LAUNCH` | 0 | 0 = build the world and wait in the setup phase until `POST /api/launch` (the setup screen); 1 = start at once. `AUTO_RESET_S` above 0 also starts at once |
| `MESAS`, `CANYON`, `DUNES`, `BOULDER_FIELDS`, `PITS`, `GEYSERS`, `ICE_SHEETS`, `FLORA`, `GROUND_DETAIL` | 1 | version 7 world features (0 leaves one out). Off in a bare `Config()`, on in `Config.from_env()`. Code: `v7_landforms.py` (mesas, canyon, pits, boulder fields), `v7_surface.py` (dunes, ice sheets, wheel friction composer, low traction rule), `v7_geysers.py`, `v7_flora.py`; ground detail is drawn by the web only |
| `BIOMES` | 1 | v7 biomes (`v7_biomes.py`): the seed picks a primary biome and a secondary one in soft regions; they scale the v7 feature counts and set the camera colours. Off in a bare `Config()` |
| `LIFE_VARIETY` | 1 | more lifeform kinds than the grazers (stream `[seed, 7113]`). Off in a bare `Config()` |
| `WEAR` | normal | `off`, `normal` or `harsh` (parts wear out about 4 times sooner). Accepted and echoed in `config`; wear itself is not part of this build yet |

More Gemma settings (timeouts, tokens, model loading) are in `GEMMA.md`.

## Biomes (v7)

`v7_biomes.py` holds the `BIOMES` table: per biome a name, the feature weights, the lifeform weights, and the camera colours (soil ramp, pebbles, rock, sun and ambient light, sky). `Terrain.__init__` calls `v7_biomes.attach` right after the height field and before any v7 feature, so every generator can scale its count:

- `pick(cfg)`: `(primary, secondary)` from `default_rng([seed, 7112])`; the primary is never `rust`, the secondary is any other id (rust allowed). None with biomes off.
- `Region`: the secondary share 0..1, a few soft Gaussian blobs warped by long waves, thresholded at a seeded quantile so it covers 25 to 40 percent of the sector, and kept at 0 round the lander.
- `terrain.biome` (`{id, name, secondary}` or None), `terrain.biome_id_at(x, y)`, `weight(terrain, feature)` (1.0 with biomes off; else 0 when the primary weight is 0, otherwise 0.75 primary + 0.25 secondary) and `count(terrain, feature, n)` (exactly n when the weight is 1). The canyon is built only when `weight >= 0.5`. Counts change after each generator's own draws, so with biomes off the worlds are byte for byte the old ones (the test checks hashes of the heights, the features message, the materials and the textures).
- Camera look: `visuals.textures(seed, biome)` paints the regolith (four primary variants plus two of the secondary, used on tiles whose centre lies in the secondary region), rocks and sky in the biome's colours, and the sun, ambient and haze follow it. Every colour is checked to stay outside the target spotting tolerance of every marker colour and off the blue liquid rule, so no biome's ground, nor one biome's patch seen against another's, can trip the camera checks.

## Sensors

Everything a rover knows comes from these. Each tick message carries what they read (`sensors`)
and up to 200 points of the latest LiDAR sweep (`scan`).

- **LiDAR** on a 0.75 m mast: 1008 rays, 8 m range, 5 sweeps a second. Hits build the shared map.
  `lidar_range` is how far the rover believes it can see, `lidar_hits` the returns in the sweep.
  Liquid gives no echo at all (the beam is absorbed or glances off), so a lake is a hole in the map.
  Hits on creatures go to the motion tracker instead of the map (see "Lifeforms").
- **Camera** on the mast (`camera.py`): an onboard object detector (`vision.py`) reads every frame the
  rover's camera renders and the rules act on what it finds (see "Object detection" below); a LiDAR hole
  without a fresh water detection still gets the old blue-pixel check. Gemma gets the camera picture and
  the detections for anomalies and for anything that moves. The UI's pictures (up to 1280 x 720) show a
  low warm sun with shadows, a hazy sky, distance haze and a dust veil in storms (`visuals.py`).
- **Water sensor**: a wetness probe on the belly. Only matters if a rover ever drives into liquid.
- **Spectrometer**: LiDAR hits on science targets within 8 m. `spectro` is the nearest one.
- **IMU**: a MuJoCo `framequat` (attitude, like an AHRS chip) and `gyro` on the chassis, with a
  little noise. Pitch, roll, tilt, heading and yaw rate all come from it.
- **Wheel encoders**: wheel joint speeds give `wheel_speed`. Body speed over the ground gives
  `ground_speed`. `slip` = 1 - ground speed / wheel speed (when the wheels turn faster than
  0.2 m/s), smoothed over about a second.
- **Motor feedback**: torque per motor gives `motor_load`; the motor controller's status shows a
  fault. **Wheel load** (contact force per wheel, like strain gauges) is read only while slip is
  suspected.
- **Bumper**: three short rays at the nose.
- **Position** is taken from the simulator, standing in for GPS-like localisation (no SLAM).

## Hazards and how the rovers handle them

The hazard list (`hazards` in every message) is for the UI only. The rovers never read it.

| Hazard | What happens in the world | How a rover finds out | What it does |
|---|---|---|---|
| Rockfall | 3 to 5 boulders drop from 5 m onto a rover's route 4 to 7 m ahead, in a wall with gaps too narrow to pass. The rover can always still get round and home. | LiDAR. The map watches mapped ground: when the highest return near a cell stands well above what was mapped, in 3 separate sweeps, the cell is rewritten. One noisy ray changes nothing; rough ground and rock edges are ignored so the map does not flicker. | Slows to 0.3 m/s at once, says what it saw ("r2 sees new rocks blocking its path 5 m ahead (LiDAR, 6 map cells changed)"), plans round them and says how much longer the new route is. The boulders become impassable for everyone. |
| Sand | A hidden soft patch (2.5 m radius) on gentle ground 3 to 5 m ahead. Inside it the wheels lose grip and the rover sinks in: low wheel friction, extra rolling resistance, and drag on the body (high when pushing into fresh sand, low when backing out along its own ruts). | Wheel slip: wheels turning, rover hardly moving, slip over 60% for 1.5 s, on gentle ground, with at least two wheels spinning that are on the ground and working hard. That rules out slopes (own stall logic), turns, and a rover hung up on a rock (its spinning wheels hang in the air). | Stops, reports "wheels spinning but not moving (slip 85%). Soft ground", marks a generous soft zone that every rover now avoids, backs out along its own trail, and plans round. Near known soft ground it trusts weaker slip evidence. Failsafe: if still stuck after 30 s it digs itself out. |
| Dust storm | 90 s. LiDAR range drops from 8 m to 3 m with some lost returns and noise. Dust also covers the lander's solar panels: charging runs at 30%. | Its own scans: no returns from beyond about 3 m for 3 sweeps in a row. Docked rovers see their battery filling slowly. | Becomes "unsure": slows to 0.3 m/s briefly, then drives at 0.4 m/s with gentle turns, stays on mapped ground (unknown ground costs much more to plan through), prefers near frontiers and makes no far trips to targets. A rover charging slowly raises its return margin from 15% to 25%. Each change is logged once, and again when it clears. |
| Rover fault | One rover's motor controller drops out for 90 s. The brakes lock the wheels. State `fault`. | Motor feedback (controller status). | It tells the fleet: its frontier and any target it was going to sample are released at once, the others re-plan, and one may take over its area. Then "r3 rebooted. Back online." and it carries on. A request that would leave no working rover is refused. |

| Spire fall | An alien spire topples across a rover's route 4 to 9 m ahead (about 1.5 s, rotating about its base), then lies there: a long, low obstacle. If no spire stands near the route, a spare one comes up beside it and falls. Never onto a rover or the lander, and nobody gets trapped. | LiDAR change detection, as for a rockfall. The changed cells form one narrow continuous strip, so it is called a long, low obstacle (a boulder row has gaps). | "r2 sees a new long, low obstacle across its path 7 m ahead (LiDAR: about 2.0 m long, 0.8 m tall, 3 map cells changed)", slows, replans ("rerouting around the fallen obstacle"). The UI gets `standing: false` for that spire in an updated `features` object in the next tick (and in every later `init`). |
| Lifeform (demo) | Moves a herd of grazers 6 to 7 m in front of a rover (or to the clicked point), walking across its view. | The rovers must notice them themselves: moving LiDAR clusters (tracker). | See "Lifeforms". |

| Damage (v9 demo) | Takes a rover's parts down at once: `light`, `heavy`, `disable` (always immobile), `wreck` (beyond field repair), `jam` (immobile, fixable in the field); optional `part`. | Its own self tests (health). | See "Health, damage and rescue". |
| Rockfall or spire fall on a rover (v9) | Clicked on or within 2 m of a rover: a boulder lands on it (aimed where it will be), presses on it and rolls off beside it; or a spire topples onto it and slides off. | IMU shock and the chassis skin (contact force), then its self tests. | Damage from the contact, then the usual LiDAR reaction to the new obstacle; distress if it can't drive. |

Hazards stay until `/api/reset`. Exploration-complete and the auto reset keep working (a faulted
rover is not parked, so there is no reset while one is down).

## Health, damage and rescue (protocol v9)

Every rover has six parts, 0 to 100: `drive_l`, `drive_r`, `lidar`, `camera`, `power`, `comms` (`health` in every
tick, with `hp`, `status`, the rover's own `diagnosis` and its `last_hit`). Damage only comes from the physics and
what the rover senses (`health.py`):

| Source | What measures it | Parts |
|---|---|---|
| Hard contacts: a boulder or spire landing on it, a rock, another rover | IMU accelerometer (the chassis' MuJoCo acceleration over 20 ms, a shock sensor) and a MuJoCo touch sensor over the chassis (the skin). The contact list is only read when something strong happened, to find what hit and where | where it touched: a side = that side's drive, mast top = LiDAR and camera, rear deck = power and antenna, a wheel = its drive (partly a jammed wheel, fixable in the field) |
| Tipping over | IMU tilt | LiDAR, camera, antenna, power |
| Driving into water | belly water probe (chassis 5 cm or more under, once a minute) | power, both drives |
| Motor strain | each side's torque at its limit for 75 s net of cooling (pushing into sand, a long stall) | that side's motor controller trips (fixable in the field) |
| Dust storm | exposure | dirt on the camera window (fixable) |
| Creatures | a grazer that runs into a rover pushes it (a force on the chassis for 20 ms, the impulse of the collision); the IMU feels it like any other shock | where the blow came from |

Thresholds come from a 900 s run of seed 42 without hazards: chassis shocks peak at 4.5 g and skin forces at
2.2 kN (driving over rocks with the belly), so damage starts above 6 g (9 points per g) and 3.5 kN (25 points per
doubling); the longest saturated climb heated a motor for 36 s (the trip is at 75 s). A boulder landing on a rover
gives about 33 kN and takes 60 to 100 points off the parts under it; a charging grazer gives about 8 to 9 g.

**Effects, through the existing mechanics.** A weak drive side gets less torque, a lower top wheel speed and a
softer servo, so the rover physically pulls toward it; the controller compensates by lowering its own top speed
(0.4 m/s with a side at 35%) and its own slope limit (from the torque left), which the planner uses; it never lowers
the fleet's limit when it stalls. A damaged LiDAR loses range and rays (visibility checks use its own range, so this
is not mistaken for dust); a damaged camera adds noise to its frames, so the detector's confidence drops by itself;
damaged power drains the battery up to 3.5 times faster; damaged comms shorten the radio range (about 220 m at
100%). Below 55 hp, or with a drive, the LiDAR or power under 35% (comms under 40%), a rover that can still drive goes
home for a repair.

**A charge.** The biggest grazer of each herd is territorial: a rover within 3.5 m of it for 12 s makes it warn,
then charge at 2 m/s for only as far as it takes to reach where the rover was. A rover sees a LiDAR track closing
at 0.65 m/s or more, straight at it, within 4 s (the camera confirms it when it sees a creature there), logs it and
backs away from it for 2.5 s if the ground there is safe (even when parked); a rover that gets far enough back is
missed. A disabled rover can't move and gets rammed.

**Rescue** (`rescue.py`). A rover that can't drive (either drive side under 15%, or power under 10%) stops, drops
every claim (frontier, target, focus, questions), logs its diagnosis and sends a distress signal (event and chat) if
its radio reaches the lander or a rover; otherwise the others notice after 15 s without a heartbeat and go to where
it was last heard. The closest rover with the battery for the trip, a tow home (2.2 times its usual cost per metre)
and its margin becomes the helper, drives over, diagnoses for 20 s and fixes field faults (a jammed wheel, a tripped
motor controller, dirt) for 15 s. Still immobile: tow, salvage or leave, with the rules marking which are safe
(tow: an A* route home within the fleet's slope limit minus 4 deg, and the battery; salvage: a part of 60% or more
left and the battery home; leave: always). Autonomous: Gemma picks among the safe ones (the rules' order otherwise:
tow a repairable rover, salvage a wreck the spares can't fix) and the decision is logged with a `decision` object;
supervised: a `rescue` request with `safe` and `note` on every option and Gemma's pick marked.

Towing is physical: a MuJoCo `connect` constraint (declared at build time for every pair, switched on with
`data.eq_active`) joins the helper's rear hitch to the towed rover's front hitch, 0.35 m beyond each bumper. The
helper drives to a point ahead of the rover, turns to face away and backs up until the hitch points meet; the bar is
soft for 1.5 s, then stiff. The towed rover's drives go passive (free rolling, and its tyres skid sideways more
easily); the helper tows at 0.3 m/s, turns at most 0.3 rad/s in arcs (no pivots, which jackknifed the first
version), with the stricter slope limit. If the tow makes no progress for 12 s, the bar pulls apart or the towed
rover tips, the fallback takes over: the towed rover's collisions go off and it follows the hitch like a trailer,
set every physics substep ("on the rigid bar" in the log). In the tests the 12 m tow arrives on the physical bar.
At the lander the rover is unhitched and repaired (state `repairing`, charging meanwhile): field faults clear at
once, a part under 30% gets a spare (fitted and tested in about 20 s; `spares` starts at drive 2, LiDAR, camera and
power 1), the rest is serviced at 2 points per second; with no spare it waits there. Salvage takes up to two parts
(the kinds the fleet has fewest of first) and the wreck's sample tubes; a part the helper needs is fitted at once,
the others count when it docks. Left and salvaged rovers stay where they are (`wrecks`).

The operator: "rover 2, are you damaged?", "how is rover 3?" (topic `health`), "rover 1, go help rover 2", "tow rover
2 home", "salvage rover 3", "leave it" (the keyword parser reads who helps whom from the word order; Gemma's schema
has `rescue` with `plan` and `target_rover`, and the parser's reading wins). Commands to a rover that is out of
action, being repaired, towing or helping get a refusal that says so.

The decision cycle's share is small: the health bookkeeping is a few dictionary reads per cycle, the shock and
contact monitor a few vector reads per physics substep (the contact list only after a strong hit), and the tow route
is an anytime A* inside the helper's budget. In a 900 s run the sim still runs at about 9.5x and brain p95 stays
around 4 ms.

## Water

One or two lakes (and on most 120 m maps a short river flowing into one) are carved into gentle
ground away from the lander. Each has a flat liquid surface at its `level` (`features.water` in
`init` has the outlines). The basin has a shallow shelf at the shore and a guard ring of ground just
above the level, and a lake or river is only kept if the ground the rovers could reach before is
still reachable around it.

1. **No echo.** Steep LiDAR rays straight ahead that should meet the ground inside the 8 m range
   come back empty. The rover extends the ground line from the last returns in each ray column, so
   a slope that falls away from the beam is not mistaken for a hole. These missing returns are
   recorded as "no echo here" evidence in the map. A hole in two sweeps in a row is checked.
2. **Look.** If the onboard detector saw water there in the last few seconds, that confirms it at once
   ("r2 Camera: water ahead (0.99) where LiDAR gets no returns 6 m ahead. Confirmed liquid: ...").
   Otherwise the rover slows to 0.3 m/s and takes a small camera picture. Blue liquid in the lower
   part of the picture: "r2 LiDAR gets no returns 5 m ahead and the camera shows a liquid surface
   (10% of the lower view is blue). Marking a lake and routing around it." The anomaly
   (`no_returns`) also goes to Gemma with the picture, which can confirm "liquid". No blue: it is a
   drop-off (a crater edge), logged once, and the rover keeps to mapped ground.
3. **Avoid.** Identified water (`fleet.water_zones`) is impassable for every rover. As rovers drive
   along the shore, more of the lake is added quietly. Unknown cells that are water, or that keep
   giving no echo, no longer count as unexplored, so the rovers do not chase frontiers into a lake.
4. **Failsafe.** If a rover still drives into liquid (its belly probe gets wet), it stops, marks the
   water ahead for the fleet and backs out along its own tracks.

Without a working camera (no OpenGL) the rules fall back to the hole's shape: a flat, level edge
(height spread under 8 cm) is probably liquid.

## Lifeforms

Two or three herds of grazers (3 or 4 each, 1 to 1.5 m long, six legs, teal green) live in open
ground. They wander at 0.2 to 0.5 m/s, graze, keep loosely together, avoid steep ground, water,
rocks and spires, and walk away from any rover closer than about 3 m. `lifeforms` (truth) is in
every message for rendering.

- **Seeing them.** LiDAR hits on creatures go to a motion tracker instead of the terrain map. This
  is a stated simplification: a real rover would split them off by comparing each scan with the
  map. After that split the tracker only sees anonymous points: it clusters each sweep, matches
  clusters to tracks, smooths position and speed, and only calls something a moving object after
  several sweeps in which it moved more than 0.35 m. `tracks` in every message is what the fleet
  believes.
- **Curious but careful.** A new track: the nearest free rover with enough battery switches to
  `observing` (one observer per herd, at most two at a time; the others keep exploring): "r3 sees
  something moving 9 m north-west (LiDAR track k2, 0.3 m/s, about 1.2 m). Stopping to look." Gemma
  gets the camera picture and the track numbers and says what it is ("possible lifeform, six legs,
  about 1.2 m"). Then the rover approaches to about 6 m and follows at 0.35 m/s at most, turning to
  keep the camera on it, never closer than 4 m (planetary protection: it backs off if a creature
  comes toward it), never more than 25 m from where it started, and stops after 75 s or earlier for
  battery, terrain, water or if it loses sight: "r3 observed the herd for 70 s: 3 individuals,
  average speed 0.3 m/s. Resuming exploration." `fleet.lifeforms_seen` counts confirmed lifeforms
  (by herd: the most individuals seen at the same time, so a creature that is lost and found again
  is not counted twice).
- **The operator decides, while the rover follows (protocol v7).** A herd nobody asked about yet:
  the rover starts following at about 6 m at once (the creatures may wander off) and asks in the same
  moment: "Possible lifeform 8 m north (...). I'm following at about 6 m (never closer than 4 m) while
  I wait, in case it wanders off. Keep following, observe from here, or keep exploring?" It does not
  hold still for this question and the 75 s limit waits for the answer (the 25 m leash still holds: it
  watches from the edge). Gemma's look adds its label to the question. Each herd is asked about once
  (track ids and re-sightings nearby); "keep exploring" also leaves it alone when it is seen again.
  Soft ground, new rocks, liquid, dust, a fault or a low battery end the follow, close the question and
  say why in the chat. Herds asked about before: Gemma decides. With tier 2 switched off the rules
  follow carefully.
- Any other rover that finds a tracked creature within 3.5 m ahead stops and waits, and the planner
  routes around tracks.

## Samples

Each rover has 3 sample tubes. Sampling uses the instrument for the target kind: mineral = drill
(about 10 s), ice = scoop (8 s), vent = gas sniffer (6 s), trace = swab (6 s; organic traces near
the herds, never the creatures). `sampling.progress` shows the instrument running. Then the rover
analyses the sample on board: a short result and a science value from 1 to 10 (deterministic per
seed; most finds are ordinary, two per map are jackpots, traces tend to score high): "r2 drilled a
core from t3: olivine-rich basalt with water-bearing clays (value 8/10). 2 of 3 tubes used."

- The spectrometer gives an estimate from afar ("estimated value 7/10"), so the rovers weigh value
  against distance, battery and free tubes. With one tube left, finds estimated under 6/10 are
  skipped (logged once: "r2 skips t5 (estimated value 3/10 and only 1 of 3 tubes left, keeping it
  for a better find)").
- Full tubes go to the lander cache on the next return, or on a dedicated trip if a find estimated
  at 8/10 or more is waiting: "r2 cached 3 samples at the lander (fleet total 7, science score
  41)." After a cache trip with charge left the rover heads straight back out.
- Gemma's goal options quote value and tubes ("value 8/10 (estimate), 2 tubes free"), and the
  operator's "sample t4" checks there is a free tube.

**Uncertainty.** Anything perception did not expect (new obstacles on the path, sudden wheel slip,
LiDAR returns collapsing) goes through one method, `Rover.on_anomaly(t, kind, details)`. It stores
the anomaly in `rover.anomaly` (and `rover.anomaly_log`), slows the rover at once, before any
re-planning, and logs what was seen. Other code can listen by adding a function to
`rover.anomaly_hooks`; it is called with `(rover, anomaly)`. The anomaly is a dict like
`{"t": 61.4, "kind": "new_obstacle", "type": "new_obstacle", "at": [2.0, -28.0], "dist": 4.0, "cells": 5}`
(`kind` is `new_obstacle`, `slip`, `low_visibility` or `no_returns`; `type` uses the names
`new_obstacle`, `wheel_slip`, `low_visibility`, `no_returns`). A new obstacle may carry
`"shape": "long"` with its `length` and `height` (a fallen spire).

## Two-tier brain (rules + Gemma)

| | Tier 1: rules | Tier 2: Gemma (advice) |
|---|---|---|
| Runs | 5 times per sim second, every rover | one model call at a time, on its own thread |
| Budget | 40 ms per rover per cycle | 3000 ms per call (timeout 4 s) |
| Decides | everything: map, A*, slope limit, battery and return, soft ground, stalls, faults, every safety reaction | nothing on its own: it suggests, the rules check |

Where Gemma helps:

1. **Goal choice.** When a rover picks a new frontier (at most every 20 s, or when it reaches one),
   it drives to the rules' pick straight away. Meanwhile the rules plan two other frontiers (A*
   path length, steepest slope on the way, battery left after the trip home), maybe a found science
   target and the trip home, and Gemma picks one with a short reason that quotes those numbers. If it
   picks another option that is still valid, the rover switches and logs one event, for example
   `r2 switching to frontier 14 m east (Gemma, 0.3 s): 'f2: slope max 9 deg, battery after return 51%'`.
   If it agrees, that is logged at most once a minute per rover.
2. **Anomalies.** New rocks, wheel slip, LiDAR range collapsing: the rules react at once (slow down,
   back out, reroute). Then the rover's camera picture and sensor numbers go to Gemma, which says what
   it is, how sure it is and what to do. Safe extras are applied (wait, go slowly, back off); unsafe
   ones are refused, for example "proceed slowly" on soft ground (`Safety rule wins: ...`).
3. **Questions to the operator** (supervised mode since v8; autonomous mode decides instead, see 9). Before v8:
   if Gemma is unsure (confidence under 0.75, or it says
   `ask_operator`), or it is the first anomaly since start or reset, the rover asks in the chat with
   2 or 3 safe options and its camera picture, and holds position while it waits (it only moves if
   staying is the danger: backing out of soft ground or liquid, heading home on a low battery, or
   making room for a creature that came too close). Questions wait for the answer (with
   `OPERATOR_TIMEOUT_S` above 0 the rover takes the safest option after that many seconds and says so).
4. **Lifeforms.** When a rover starts observing something that moves, Gemma gets the camera picture
   and the track numbers and says whether it is a lifeform, with a short label and what to do
   (approach to observe, observe from here, ignore). See "Lifeforms".
5. **Typed commands.** Gemma (or the keyword parser if Gemma is offline) turns the operator's text
   into a command; the rules check it and run it or refuse with numbers ("r1: can't, battery 22%
   would not get me there and back"). One reply per command. Commands: go to x, y; explore a
   direction (for 3 minutes); sample a target; return home; hold; resume; avoid an area (a circle that
   is off limits for the whole fleet); anything unclear gets a question back. After a command is done
   the rover carries on on its own.
6. **Questions.** "rover 2, what have you explored?", "any problems?", "everyone, report battery": the
   message is a question (Gemma's reading, or the keyword rules: a question word, a trailing "?",
   "tell me", "report", "status"). The rover (or the fleet, when nobody or everybody is asked; each
   rover for "everyone, report battery") builds a fact sheet from its own memory and Gemma writes a
   1 to 3 sentence reply from it: "I've mapped about 1,904 m2 (13.2% of the 14,400 m2 map), mostly
   north-east of the lander, out past 35 m. I've driven 410 m." A number or id in the reply that is not
   in the fact sheet, no reply within the 3 s budget, or Gemma offline: the rule-based answer is posted
   instead (topics: status, exploration, finds, samples, battery, hazards, lifeforms, water, location,
   why; anything else: "I can't answer that: I only know what my own sensors and memory recorded").
   Replies are `kind: "reply"` with `reply_to` and `source` gemma or rules.

7. **Places (protocol v6).** "go to the water source", "can we investigate the water source again for any
   signs of life around there", "rover 3, survey the spires", "take a look at t7", "go to the biggest
   lake", "check out the vent near rover 2". Polite forms with an action ("can we", "could you", "let's",
   "I want you to", "please") are commands; "can you see any water?" is still a question. The place comes
   from fleet knowledge only (`places.py`). One match, or a qualifier that picks one: the named rover, or
   the closest free one ("I'm the closest free rover (18 m). Heading to the lake's shore at ..."), drives
   to a safe spot next to it through the normal go-to checks (for water about 3 m from the edge on its
   side, for a herd the 6 m observation standoff). With investigate ("investigate", "survey", "look
   around", "signs of life") it then stops, turns slowly through 360 deg for about 30 s with the LiDAR
   tracker on, and reports what moved and what it saw (`kind: "info"`, `reply_to` the command); something
   moving starts the normal lifeform observation. Several matches: a `choose_place` request with numbered
   options (with x, y for the map) and "Never mind"; the rover holds until you pick one in the UI or type
   "the second one", "p2", "the one near the lander" or "the lake". Nothing known: the fleet says so and
   lists what it knows.

8. **Areas, curiosity and chat memory (protocol v7).** "explore around the lake", "go to the vent and look
   around", "explore there", "explore north of the lander" (a circle 20 m that way), "explore around 10,
   -5", and "investigate X" (the 30 s survey first): the rover gets a `focus`, a 15 m circle (bigger for a
   big lake). Its frontier choice takes unclaimed unknown ground inside the circle first (then the
   frontiers nearest its edge); frontiers another rover is heading for stay theirs, and the other rovers
   leave the inside of someone's circle alone. It stays about 2 min from reaching the circle, or until
   nothing unknown is left nearby (or it maps nothing new for 30 s), then reports in chat what it mapped
   there (m2, how much of the circle is now mapped), what it found (targets, samples, water, spires,
   herds) and how long, from its own memory: "Explored around the pond's shore for 118 s: mapped 640 m2 of
   new ground; the area within 15 m is now 94% mapped. Found t4 (ice deposit). Back to normal
   exploration." Curiosity: when a rover finds something new (a target, a new body of water, a cluster of
   spires, a herd it finished watching, or ground whose median slope or height suddenly differs from what
   it has been driving through) it sticks around for about 60 s ("r2 found water here. Sticking around for
   about 60 s to look for more."), only where at least 20% of a 10 m circle is unknown, ending early when
   less than 8% is left or nothing new is mapped for 15 s; never while returning, on a low battery, during
   another focus, or where another rover's focus covers the spot, and then not again for 150 s nor within
   15 m of an earlier spot. The chat remembers the last 8 messages and what they named, so "go back there",
   "what did you find there?", "the second one", "that lake", "the vent you mentioned", "investigate it",
   "do that again" and "same for rover 3" resolve (an unclear reference gets a short question); Gemma gets
   a few lines of that context.

9. **Autonomy, priorities, re-survey, area questions (protocol v8).** By default (`AUTONOMY_MODE=autonomous`)
   a rover never waits for the operator: where it used to ask (an anomaly, a lifeform sighting, several places
   matching a command) Gemma decides, the rules decide when Gemma is offline, and under 0.5 confidence the safest
   option wins. Each decision is an event with a `decision` object and a chat line: "r2 decided (Gemma 0.82): go
   around. New rocks 4 m ahead, 0.6 m tall." `supervised` (toggle in the UI, `POST /api/mode`, or "switch to
   supervised mode") asks every one of those decisions, with Gemma's pick marked as a suggestion; switching back
   decides the open questions at once. Unexplored gaps (holes inside the map, `gaps.py`) are places: "explore one
   of those holes" sends the closest free rover to the best one. A fleet priority ("prioritize the gaps",
   "prioritize the north", "prioritize water", "prioritize sampling") adds a strong bonus to every rover's frontier
   choice until nothing is left (announced in chat) or "back to normal"; the ack names the plan per rover. An
   operator focus on an already mapped area is a re-survey: a loop through the circle for the whole time, rescanning
   with LiDAR and the spectrometer, then a report of what changed ("No changes since it was mapped" is a report too).
   "what did we find there?" and "anything near the lake?" get a fact sheet of that place only. Messages that ask
   for a task the parser cannot place get a short question back instead of a generic status reply.

**Hold** shows as state `idle` with goal `null` and a chat note ("r1: holding at (12, -4), say resume
to continue"). A held rover still watches its battery: if it would not get home otherwise, it leaves
the hold and says so; held on the lander pad it keeps charging. Any rover waiting for an answer keeps
the same battery check as when driving: when it must go home it closes its open questions ("Battery
21%, I can't wait any longer: heading back to the lander. I'll ask again if it still matters."),
drops a focus or a follow and leaves at once.

**Demo a decision and a question:** start the sim (with Ollama, or `GEMMA_FAKE=1`), wait until the rovers
are driving, then trigger a rockfall (UI hazard button, or
`curl -X POST localhost:8000/api/hazard -d '{"type": "rockfall"}'`). In the default autonomous mode the rover
decides and logs it: "r2 decided (Gemma 0.80): go around. New rocks 4 m ahead, 0.6 m tall." Switch to
supervised (`curl -X POST localhost:8000/api/mode -d '{"mode": "supervised"}'`) and trigger another one: now
the rover asks "New rocks block my path 3 m ahead ... Gemma thinks: rock (confidence 0.80). Gemma suggests: go
around (0.80). What should I do?" with Go around (suggested) / Approach slowly / Wait and its camera picture.
Answer in the UI (or `POST /api/answer`); with `OPERATOR_TIMEOUT_S` above 0 it takes the safe default after that
long. Every anomaly asks (one open question per rover).

## Camera speed and object detection (protocol v8)

**Faster camera, same pixels.** With software OpenGL (OSMesa in Docker) MuJoCo draws every geom of the
scene three times when there are shadows (shadow map, base pass, sun pass), and the 120 m map has 576
terrain tiles plus 240 spare boulders and spires parked under it. `camera.py` now:

- renders from a private copy of the model whose geom groups it rewrites per frame, so only geoms inside
  the camera frustum or the sun's shadow prism (casters that can shade what the camera sees) are drawn.
  It changes no pixel (checked in `tests/test_vision.py`);
- holds the sim lock only to copy `qpos` and the mocap poses into its own MjData (microseconds), then does
  kinematics, the scene, rendering, haze, detection and JPEG outside it;
- keeps rendering every view a client asked for in the last 4 s, timed so a fresh frame is ready just
  before the next request, and `GET /api/camera` answers at once with the newest one (`X-Frame`,
  `X-Frame-Age-Ms`, `X-Detections`);
- keeps the look: 2048 px shadow map (1024 adds visible jagged shadow edges around the rover in the chase
  view at 640 x 480 and up), 4x MSAA, JPEG quality 80.

Measured on the dev machine (2 shared CPUs, OSMesa / llvmpipe, other jobs running, seed 42; before = the
pre-v8 code, runs interleaved; `scripts/camera_bench.py` and `--live`):

| | before | after |
|---|---|---|
| one frame 320 x 240 fpv / chase (render, haze, JPEG; median) | 154 / 167 ms | 119 / 127 ms |
| one frame 640 x 480 fpv / chase | 184 / 198 ms | 135 / 150 ms |
| one frame 1280 x 720 fpv / chase | 265 / 271 ms | 233 / 254 ms |
| UI polling fpv 320 x 240 every 250 ms: wait per request | 157 ms (p95 226) | 2 ms (p95 8) |
| ... age of the picture when it arrives | 157 ms | 36 ms median (p95 249) |
| ... new pictures per second | 4.0 | 3.8 |
| UI polling fpv 1280 x 720 every 330 ms: wait / picture age | 297 ms / 297 ms | 2 ms / 58 ms median (p95 590) |
| ... new pictures per second | 3.0 | 2.4 |
| brain ms per cycle, avg / p95: nobody watching | 1.9 / 4.0 | 2.4 / 5.6 (detectors on) |
| ... polling 320 x 240 / 1280 x 720 | 2.5 / 7.3, 2.9 / 8.1 | 2.8 / 7.6, 3.2 / 8.8 |

The budget is 40 ms per 200 ms cycle. With more CPU cores llvmpipe rasterises in parallel, so all of these
are faster on a desktop. In a 420 s headless run with hazards and the detectors on: brain avg 1.7 to 2.1
ms, p95 4.1 to 4.9 ms, max 15 ms.

**Object detection.** Each rover runs a small classical detector (`vision.py`, numpy only) on its own
camera frames: RGB plus the depth buffer of the same render and the camera's pose, never segmentation or
any ground truth. Depth gives every pixel a 3D point; steep surfaces (things standing on the ground) and a
few material colour groups measured on the sim's own renders (liquid blue, ice, violet crystal, grazer
green, the vent's hot orange rim, the lander's grey pad, basalt, tyres) are grouped into blobs; depth gives
each blob a distance, bearing, width and height above the ground; simple per-class rules turn colour
fractions, colour strength, size and shape into a confidence. Classes: rock, boulder, spire, creature,
water, mineral, ice, vent, trace, rover, lander. Haze and dust wash out the colours and far objects cover
fewer pixels, so distance and dust storms lower the confidence on their own (the camera shows the dust
veil during a storm).

It runs on the camera thread, never in the 5 Hz loop: on a small render per rover (256 x 192, no shadows,
`VISION_HZ` 1 per second), on every frame of a streamed fpv view, and on the frame Gemma looks at. The
newest result is the rover's `vision` in each tick; the rover's decision cycle only reads it (under 0.1 ms):

- **Water:** a LiDAR hole with a water detection there is confirmed liquid at once ("r2 Camera: water 5 m
  ahead (0.86) where LiDAR gets no returns 5 m ahead. Confirmed liquid: marking a lake and routing around it.").
- **Lifeforms:** a creature detection where a LiDAR track moves confirms it ("r3 Camera: creature 8 m ahead
  (0.99) where LiDAR track k1 moves (0.2 m/s). Counting it as a lifeform."), once per herd and place.
- **New obstacles:** the class of what LiDAR change detection flagged, and room to match ("r2 Camera: spire
  4 m ahead (0.99) where the map changed. It is a fallen spire: keeping 0.8 m of extra room around it.").
- **Targets beyond the spectrometer:** a mineral, ice, vent or trace seen 8.5 to 18 m away (confidence 0.65
  or more, not known yet) may send a free exploring rover to look, at most once a minute per rover, never to
  a spot the fleet already checked, giving up after 45 s ("r3 Camera: possible mineral 13 m ahead (0.84).
  Going to check it." ... "r3 Camera check done: the spectrometer confirms t10 (a thermal vent, not a
  mineral) where the camera saw a possible mineral.").
- **Gemma** gets the detections near an anomaly or a moving track as a reading ("camera sees: boulder 0.78
  at 4.1 m"), and operator questions quote them.

Measured with `scripts/vision_eval.py` (seeds 42, 7 and 3; 855 views per frame type: the camera facing
every kind of object at 3.5 to 15 m, other rovers and rockfall boulders in front of it, and random poses;
scored against MuJoCo segmentation at confidence 0.5 or more; a detection is right if it has the label of a
ground truth object and overlaps it (IoU 0.3) or has its box centre inside it; objects under about 20 px
at 320 x 240 or beyond 30 m do not count):

| class | UI frame 320 x 240 (shadows): precision / recall | detector frame 256 x 192 (no shadows) |
|---|---|---|
| rock | 0.93 / 0.87 | 0.94 / 0.88 |
| boulder | 0.66 / 0.58 | 0.70 / 0.57 |
| spire | 1.00 / 0.90 | 1.00 / 0.89 |
| creature | 1.00 / 0.71 | 0.99 / 0.70 |
| water | 0.97 / 0.86 | 0.94 / 0.77 |
| mineral | 0.91 / 0.89 | 0.89 / 0.87 |
| ice | 0.95 / 0.98 | 0.96 / 0.83 |
| vent | 1.00 / 0.28 | 1.00 / 0.23 |
| trace | 1.00 / 1.00 | 0.99 / 1.00 |
| rover | 1.00 / 0.58 | 1.00 / 0.58 |
| lander | 0.96 / 0.65 | 0.96 / 0.57 |
| all | 0.94 / 0.79 | 0.94 / 0.79 |
| time per frame | 16.1 ms avg (p95 23.9) | 11.5 ms avg (p95 17.8) |

In a full dust storm (seed 42) recall falls to 0.31: beyond about 8 m things disappear into the veil, and
nearby objects that lose their colour tend to come out as rocks (precision 0.66).

`GET /api/vision/stats` reports the same measures live (one detector frame in five, against the sim's
segmentation; transparency only, the rovers never see it). Limits: vents farther than about 10 m (the
orange rim is a few pixels; the mound alone looks like soil), rock vs boulder is size only (rockfall
boulders rest half buried like terrain rocks, and the biggest terrain rocks look the same), grazers that
overlap in the picture can come out as one, and rocks smaller than about 10 pixels.

## Edge compute budget

Each rover states a budget: 40 ms of compute per 200 ms decision cycle (`BRAIN_BUDGET_MS`). The
"brain" is what a real rover computer would run: turning LiDAR hits into the map, frontier logic,
path planning and the state machine. It does not include the simulated sensor hardware (the ray
casts) or the physics. Work shared by the fleet (frontiers, reachability) is done by one rover
per cycle, taking turns, and counts toward that rover's time. The rules' share of tier 2 (planning
Gemma's candidate goals and operator trips) counts too and also plans in slices. Gemma's own time
does not: the model runs on another thread with its own budget (3000 ms per call).

Planning is anytime: when a plan uses up the cycle's share of the budget, A* stops and carries on
next cycle. Meanwhile the rover keeps its current route, or waits if something blocks it just
ahead. A rover that goes over budget several times in 10 s logs a warning and plans in smaller
slices for a while. Each tick reports `brain` (last, average and peak ms over 10 s, and cycles
over budget).

Measured on the dev machine (2 shared CPUs, 3 rovers, 120 m map, 900 sim s, wall clock, with
`--hazard lifeform@60 --hazard spire_fall@150 --hazard rockfall@300 --hazard sand@400
--hazard dust_storm@500 --hazard rover_fault@650:r3 --ticks`):

| Seed | avg ms | p95 ms | max ms | over 40 ms | sim speed | tick median / p95 / max |
|---|---|---|---|---|---|---|
| 42 | 1.7 to 1.9 | 4.1 to 4.9 | 18 to 45 | 1 of 13500 cycles | 9.4x | 17.2 / 18.8 / 32.9 KB |
| 7 | 1.6 to 1.7 | 3.9 | 12 to 30 | 0 | 9.9x | 17.3 / 18.7 / 33.1 KB |
| 123 | 1.7 | 3.9 to 4.3 | 30 to 32 | 0 | 9.5x | 16.8 / 20.0 / 30.8 KB |

A maximum of about 30 ms is the planner using its whole share of a cycle (0.75 x 40 ms). The
reachability flood fill on the bigger grid is anytime too (at most 80 growth steps per cycle). The
whole sim runs about 9.5 to 10 times faster than real time. A tick message is about 17 KB (most of
it the LiDAR points; `lifeforms` and `tracks` add about 1 KB). An `init` carries the whole known map:
about 1 MB once the 120 m map is explored (56,000 cells), about 150 to 250 KB early on.

Full exploration of the 120 m map (seed 42, no hazards) finishes at about 1590 sim s (97% of the
map, all 20 targets found, 17 samples cached, science score 91, no rover ran out of battery);
at 900 s the hazard runs above had explored 65 to 82%.

The budget is wall clock time, so on a busy machine a plan may be split at a different moment and
two runs can drift apart. `--model-clock` (or `plan_clock="model"` in `Config`) counts A*
expansions instead (3 us each, measured here), which makes runs repeat exactly; the tests use it.

## How the rovers decide

- **Seeing.** Each rover's LiDAR hits go into one shared map of heights and slopes. Rocks show up
  as steep spots. If mapped ground changes (a rockfall), the map notices and updates (hits closer than
  3 m only fill in around a change seen from further away: up close the beams graze rock faces).
- **Where to go.** A frontier is the edge between known, drivable ground and the unknown. Each
  rover scores the frontiers: big and close is good, turning around is bad, and a frontier another
  rover is already heading for is much worse. So the rovers spread out. It also checks it has
  enough battery to get there and back. Then A* plans a path around steep ground. While driving it
  keeps checking the next 12 m of its route against the map and re-plans if anything blocks it.
- **Learning the slope limit.** The fleet starts by guessing it can climb 30 deg. If a rover
  stalls nose up on a steady slope, with its motors maxed out and gravity alone needing most of
  their torque, the fleet lowers the limit to 2 deg under that slope ("slope 21 deg with motors at
  96%"). Every rover replans with the new limit straight away. The limit never drops below the
  steepest climb a rover has already managed. A stall caused by a rock or by losing grip does not
  count. The real limit at 2 g is about 20.5 deg (see `scripts/climb_test.py`).
  At another launch gravity the predicted limit is `asin(4 T / r / (m g))`, capped at 30 deg where grip
  takes over (about 14 deg at 3 g), and the thresholds tuned for 2 g follow it: target paths 15 deg, the
  no-trap flood 18 deg, hazard placement 20 deg, the slope floor 12 deg and the sand wheel load 0.2 (see
  PROTOCOL.md, "Version 7 additions", Sim notes). At 2 g they are unchanged.
- **Science.** If a LiDAR ray hits a target within 8 m, the rover reports it with the
  spectrometer's value estimate. If it is free, has the battery and a tube to spare, it drives over,
  runs the instrument, analyses the sample and keeps it in a tube (see "Samples"). Busy rovers log
  the find and another rover can pick it up. Targets come in clusters, so frontiers near a sampled
  target get a bonus.
- **Battery.** Driving uses power (more on slopes and in tight turns), and the electronics use a
  little all the time. Each rover keeps checking how much charge it needs to get home and goes
  back with a 15% margin (25% while the lander charges slowly). It charges on the lander pad, then
  heads out again. The battery is sized for the map (1.5 times bigger on the 120 m map than on the
  80 m one), so a trip to the far edge and back takes about the same share of a charge.
- **Trouble.** If a rover stops moving it works out why: a rock (mark it and go round), a slope
  (learn), lost grip (back off and try another line), soft ground (back out and mark it), or a
  rover in the way (back off). If it tips over it rights itself. If steep ground boxes it in, it
  heads for the lander, allowing slightly steeper ground, until a frontier is reachable again.
- **Done.** When no frontier can be reached from the lander, exploration is complete and every
  rover parks at the lander and the sim stays as it is until someone presses Reset. (Optional:
  set `AUTO_RESET_S`, e.g. 60, to land the fleet on a new random planet that long after
  exploration finishes, so an unattended demo keeps going.)

Every decision is logged as a short event that the UI shows live, with the sensor reading behind
it and, where it makes sense, the place (`at`) so the UI can fly the camera there.
