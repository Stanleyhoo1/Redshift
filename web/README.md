# Redshift mission control (web UI)

Live 3D view of the rover fleet exploring the super-Earth: the map fills in as the
rovers sense it, with each rover's state, battery and goal, and the fleet's decisions
in the event log. It talks to the sim over the protocol in [`../PROTOCOL.md`](../PROTOCOL.md).

Vite + React + three.js (@react-three/fiber, drei). Node 22.

## Run with the mock sim (no MuJoCo needed)

```sh
npm install
npm run mock      # fake sim on port 8000 (env: SEED, NUM_ROVERS, SIM_SPEED, MAP_SIZE, GRAVITY_G, BRAIN_BUDGET_MS,
                  #   GEMMA_ENABLED, GEMMA_BUDGET_MS, OPERATOR_TIMEOUT_S, MOCK_GEMMA=offline, MOCK_PROTOCOL=2|3,
                  #   MOCK_FIRST_QUESTION_S, MOCK_SAMPLE_SCALE, MOCK_ANSWER_DELAY_MS, MOCK_CAMERA=off)
npm run dev       # in a second terminal, then open http://localhost:5173
```

## Run with the real sim

Start the sim server on port 8000, then `npm run dev`. The dev server proxies `/ws`
and `/api` to `http://localhost:8000`; point it elsewhere with `SIM_URL`, for example
`SIM_URL=http://localhost:8100 npm run dev`.

Everything together in Docker, from the repo root: `docker compose up --build`, then
open http://localhost:8080.

## Build

```sh
npm run build     # static files in dist/ (relative URLs)
npm run preview   # serves dist/ on port 4173 with the same proxy
```

## Using it

- Toggles: Reality (ground truth overlay), Frontiers, Paths, Slope colours, LiDAR (each
  rover's latest sweep as points in its colour), Life (version 4 sims).
- Select a rover (its card, its 3D model or name tag, a Follow button, or `1` to `9`): the
  camera follows it and its live sensor panel opens over the lower left of the view:
  camera (FPV or Chase, about 4 frames a second), a LiDAR scope (rover at the centre facing
  up, returns coloured by height, the dashed ring is the current range), IMU attitude,
  wheel speed against ground speed with slip and motor load (slip holds its peak for a few
  seconds), spectrometer, battery, and brain time against the stated budget. `Esc` or the
  close button hides the panel; clicking the selected card again returns to the overview.
- Hazards (top right): Rockfall, Soft sand, Dust storm, Motor fault, aimed at a rover or
  Auto. With a version 4 sim, Rockfall, Soft sand, Falling spire and Lifeform (styled as a
  find, not a hazard) are placed by clicking the map: the button stays lit, a hint says what
  to do, and a ghost marker with the hazard's radius follows the cursor on the terrain (red
  where the sim would refuse: off the map, on the lander, within 2 m of a rover, in
  identified water, sand on steep ground; near a spire it shows which way that spire would
  fall). A click places it (an orbit drag never does), right-click lets the sim pick, `Esc`
  cancels. A refused spot shows the sim's error and stays in placement mode. Motor fault:
  click it, then click a rover (or pick one under Target). Dust storm stays one click. The sim's own error text shows inline if it refuses. Rovers are never told: they
  find out through LiDAR (boulders, dust), wheel slip (sand) or fleet status (faults).
  On screen: boulders drop in, the fleet's soft zones are hatched (the true patch is a
  dashed outline with Reality on), a dust storm hazes the view with a banner, a faulted
  rover blinks red and its card counts down.
- Water (version 4): the fleet's identified water (`fleet.water_zones`) is drawn as an
  animated surface; until a rover identifies it a lake is a dark hole in the map (LiDAR gets
  no returns from liquid). Reality shows every true body, unidentified parts fainter, with a
  dashed outline. Crystal spires appear once the ground under them is mapped (or with
  Reality) and topple with a dust puff when a `spire_fall` hits.
- Life (version 4, `B` or the Life toggle): grazers are drawn where a fleet track is near
  them, or faded with Reality. Each track has a dashed ring, a velocity arrow and Gemma's
  label (one tag per herd, "+2" for the rest; lost tracks fade). An observing rover gets a
  dashed line of sight and a lime "Observing" pill; the stats count lifeforms seen.
- Samples (version 4b): rover cards show tube slots and, while sampling, the instrument with
  its progress (also in the sensor panel, and as a ring with an instrument chip at the target
  in 3D). Analysed targets show their value on hover. The lander shows its cache count, the
  stats the science score. The Samples tab lists every sample, best first: kind, instrument,
  analysis, value, and who carries it or "cached at lander". Rows fly the camera there.
- Event log rows with a target icon fly the camera to where it happened. Hazards are red.
- Planet view (version 7, web only): `G` or View: Planet in the controls (the row above
  Graphics) lifts the camera off the map, fades, and shows the whole planet, drawn from the
  world's seed: dusty plains, darker maria, craters, polar ice and a glow of thick air on
  the limb, under the same low sun as the sector, with the landing site in daylight. The
  legend gives the planet's name, its radius (gravity x 6371 km, assuming Earth's density)
  and the site's latitude and longitude (from the seed). A cyan ring marks the landing
  sector (a symbol: the sector is far too small to see at this scale), with a live card:
  sector size, explored share, gravity, sim time and every rover's state. Drag to turn the
  planet, scroll to zoom, `0` frames the site again. Left alone for a few seconds it sways
  slowly, about 20 degrees towards the sun over a minute and back, so the site and its card
  stay in view. Reset sim on the planet glides to the new world's site. Clicking the ring,
  Fly down to the sector, `G` or View: Sector dives back to the map, and so does anything
  that acts on the sector: selecting a rover (cards, keys, the card's rover rows), clicking
  an event, a hazard button, or a rover asking you to pick a place. Hazard placement never
  carries over to the planet. An open sensor panel hides on the planet and comes back as
  it was. With reduced motion the switch is instant and the planet stays still. Graphics
  Low draws a simpler planet (fewer noise layers, one crater size, fewer stars).
- Keys: `R` `F` `P` `S` `L` `B` toggle layers, `1` to `9` select a rover, `G` switches
  between the sector and the planet, `Esc` cancels placement, closes the sensor panel (then
  returns to the overview), `0` for the overview (on the planet: frames the site), `/` for
  the message box.
- Reset sim asks for confirmation, then restarts the sim with a random seed.
- Following: orbit (left drag) and zoom keep following the rover. A pan (right drag, or a
  modifier with the left button, or two fingers) or a click on an event that flies the
  camera elsewhere pauses the follow right where you put the camera: the rover stays
  selected with its panel open, its Follow button turns dashed with a pause mark, and a
  "Back to Rover 2" pill appears (click it, press `F`, click the rover or its Follow button)
  to glide back. While paused, `F` resumes instead of toggling Frontiers.
- Camera viewer: click the camera picture (or its expand icon) for a large view, about 80%
  of the screen. It asks for frames sized to the viewer and the screen's pixel density (up
  to 1280 x 720), about 3 a second, one at a time, and the panel's small feed pauses
  meanwhile. FPV or Chase, rover buttons (`1` to `9`), scroll to zoom around the cursor,
  drag to pan, double-click to reset, `V` switches view, `Esc` or a click outside closes.
  A camera HUD shows rover, heading tape, pitch ladder (FPV), speed, position, state,
  pitch and roll, battery, sim time, and whether the picture is live or how stale it is.
- Sensor detail: the expand buttons on the LiDAR scope, the IMU section and the compute
  rows open a large view: a big scope (scroll or the minus and plus buttons change the range
  shown; earlier sweeps fade out), the rover's running totals (version 5 `stats`), and
  charts of the last 1, 2 or 5 minutes: speed, wheel slip, motor load, pitch and roll,
  brain time against its budget, Gemma latency against its budget, battery, LiDAR range.
  The history is recorded in the browser from every tick for every rover (6 minutes,
  fixed-size buffers), so switching rover shows its past straight away.
- The overview keeps the explored area in frame, in the space the panels leave free.
  Dragging or zooming takes over the camera until you press Off (or the sim resets).
- Gemma (top of the sidebar): the local model's state and average latency ("Gemma 4 E2B,
  Ready · 0.4 s"; amber while warming up, "rules only" when offline or off). Hover or click
  it for calls, fallbacks, p95 and the budget. The sensor panel's compute section has a
  second bar for it under the brain bar: average filled, p95 tick, budget line.
- Asking the rovers (version 5): Comms has a row of question chips for the addressee (the
  selected rover, else All): "What have you explored?", "What have you found?", "How's your
  battery?", "Why are you going there?", "Seen any life?", "Any problems?", or type your own.
  Until something answers, the question shows "Rover 2 is thinking" (dropped quietly after
  15 s). Replies (`kind: "reply"`) quote the question they answer (`reply_to`; click to jump
  to it, hover to light both), in the rover's colour or as the fleet; several rovers
  answering the same question are tied together under the first.
- Events and Comms (tabs under the rover cards): the event log tags decisions Gemma made
  ("Gemma") and ones the operator asked for ("Operator"). Comms is the chat with the fleet:
  pick All or a rover, type and press Enter (or `/` from anywhere to jump there; Up recalls
  what you sent), or click a suggestion. Rovers confirm (green), refuse with their reason
  (amber) or ask (cyan), tagged Gemma or Rules. The sim's error text shows under the box.
  With Comms open the fleet stats fold into one row. The tab counts unread replies and
  lights up while a rover is waiting for an answer.
- Clarification requests: when a rover cannot tell what it senses it asks, and a card opens
  under the hazard bar with its camera (its LiDAR sweep if the sim cannot render), the
  question, the options (the safe default marked) and a countdown. Answer there or in the
  thread; the card confirms, or says which safe default it took when time ran out. More
  questions queue as strips under the card. The asking rover's name tag shows a "?".
- Place choices and questions that wait (version 6): orders can name a place ("go to the
  water source", "can we investigate the lake for signs of life"). When several places
  match, the rover asks with a `choose_place` request: every candidate gets a numbered
  marker on the map in the rover's colour (pulsing ring, badge with a short label, faint
  line from the rover), the card shows numbered chips and a small north-up map instead of
  the camera, and the question in Comms has the same chips. Click a marker, a chip, or
  type "the second one". Hovering a marker lights its chips and the other way round. When
  such a request opens the camera glides once to show the rover and all candidates (not
  while you are moving the camera, and not if they are already in view; a follow pauses
  with "Back to"), and goes back afterwards if you did not touch it. A request with
  `expires_in` null shows "Waiting for you" (no countdown, no safe default): the rover
  holds until you answer. Older sims with a countdown work as before.
- Lost tracks (version 4): a faded, frozen copy of the creature stays where the fleet last
  saw it, and the tag says "last seen 12 s ago", until the sim forgets the track.
- Every version 2, 3 and 4 field is optional: with an older sim the panel shows dashes, the
  camera says it is unavailable, the hazard buttons say the sim does not support them, the
  Gemma pill and bar stay hidden, Comms says the sim has no operator chat, and the version 4
  buttons, Life toggle, Samples tab and legend rows are not shown. The map size always comes
  from `init.map` (120 m by default now).

The mock implements all of version 2 (sensors from a ray-marched LiDAR over its heightmap,
brain timings, `POST /api/hazard` with plausible reactions). With `MOCK_PROTOCOL=4` or
lower (or `MOCK_CAMERA=off`) its camera endpoint answers 503 on purpose. `node scripts/v2-shots.mjs` (mock and preview running) clicks through every
hazard and saves screenshots; set `WIDTH` and `HEIGHT` for other sizes.

It also implements version 3: Gemma warms up for about 10 s, then answers in 300 to 900 ms;
Gemma-sourced events quote numbers; `POST /api/chat` understands return, hold, resume,
explore a direction and go to x, y (refusing outside the map, on steep or soft ground, or
when the battery would not cover the trip back) and asks back when it cannot parse; rovers
ask after a rockfall or soft sand and once about 40 s in, with a 20 s countdown;
`POST /api/answer` answers 404 and 409 like the real sim. `MOCK_PROTOCOL=2` makes it a
version 2 sim. Test hooks, mock only: `POST /api/_mock/planner {"state": "offline"}`,
`POST /api/_mock/ask {"rover": "r2"}`, `POST /api/_mock/battery {"rover": "r1", "battery": 0.3}`.
`node scripts/v3-shots.mjs` drives all of it through the UI and saves screenshots.

Version 5 in the mock (the default): questions are told apart from orders (question words,
a trailing "?", "tell me", "report", "status"; "can you go to 10, 5?" stays an order) and
answered after a short think with `kind: "reply"` and `reply_to`, from the mock's own state:
explored area and where, finds, samples, battery and range, why it is going where it is,
lifeforms, water, hazards met, location, status, and "I haven't sensed that" for anything
else. One rover answers when addressed, the fleet answers otherwise, and every rover answers
"everyone, report battery". Acks and refusals carry `reply_to` too. Rovers send `stats`. The
camera answers a PNG drawn on the fly at the requested size (up to 1280 x 720): hazy sky,
noisy ground in perspective that moves with the rover, horizon tilted by roll and pitch, the
rover itself in the chase view. `MOCK_ANSWER_DELAY_MS` adds thinking time (for screenshots).
`node scripts/v5-shots.mjs` (mock started with `MOCK_ANSWER_DELAY_MS=11000`, preview running)
drives follow and pan, the camera viewer, the sensor view and Comms, and saves screenshots.

Version 6 in the mock (the default; `MOCK_PROTOCOL=5` for the previous behaviour): place
orders for water, vents, minerals, ice, traces, spires, herds and the lander, from what the
fleet knows, with qualifiers (nearest, farthest, biggest, smallest, newest, a direction,
near the lander or a rover, an id like t7). The closest free rover goes when none is named;
two or more matches give a `choose_place` request with up to 3 candidates and "Never
mind", answered by `/api/answer` or chat text ("p2", "the second one", "the one near the
lander", "the stream"). The rover drives to a safe spot next to the place (3 m from a
shore), and "investigate", "survey" or "look for life" make it stop, turn slowly for 30 sim
seconds and report in chat (`reply_to` the order). Every request has `kind`, and
`OPERATOR_TIMEOUT_S` defaults to 0 (requests wait, `expires_in` null; set it above 0 for the
countdown). Test hook: `POST /api/_mock/reveal {"water"?, "targets"?, "spires"?}` (the fleet
learns them all). `node scripts/v6-check.mjs` (mock started with
`MOCK_FIRST_QUESTION_S=99999`, preview running) sends an ambiguous order, saves one
screenshot and clicks a marker.

Version 7 planet view: `node scripts/v7-globe-check.mjs http://localhost:5182 <outDir>` (the
mock on 8102 started with `MOCK_FIRST_QUESTION_S=99999`, vite on 5182 with
`SIM_URL=http://localhost:8102`; `ONLY=toggle-key,marker-click` runs a few) switches both
ways and checks the fade, the card, the legend, the kept sensor panel, orbit, scroll, `0`,
the idle sway over more than a minute, Reset sim on the planet, Graphics Low, reduced
motion, every way back to the sector, and click-to-place after a trip to the planet, and
saves screenshots; it fails on any console error or warning. Run it once more with
`WIDTH=1280 HEIGHT=720 ONLY=hud-fit,toggle-key` to check that the controls still end above
the map legend on a short screen. The pure parts (planet numbers and the idle sway, the
view state machine, the star, planet and marker materials) have Node tests, and one test
fails if a file outside `src/globe` imports drei's `Html` directly (sector labels must use
`src/globe/PlaneHtml.jsx`, which hides them on the planet): `node --test test/*.test.mjs`.
With `?debug`, `window.__redshift` also has `setView('globe' | 'plane', instant)`,
`globeState()` and `globeSiteScreen()`, and `state()` reports `view`, `viewRender` and
`viewPhase`.

Version 4 and 4b in the mock: a 120 m map with two lakes and a stream (LiDAR holes until a
rover comes within 4.5 m, then `water_zones` grows as it sees more), groves of spires
(`spire_fall` tips the nearest one within 6 m of the click or drops a new one, and the next
tick resends `features`), wandering grazer herds with fleet tracks, a first-sighting question
("Possible lifeform 8 m north. Follow it or keep exploring?", default observe from here),
rovers that observe or follow at about 6 m, point and click with the protocol's 400 answers,
and sampling (drill, scoop, sniffer, swab; 3 tubes; analyses with values; deliveries to the
lander). More test hooks: `POST /api/_mock/teleport {"rover", "x", "y", "hold"}`,
`POST /api/_mock/rearm {"armed"}` (first-sighting question). `node scripts/v4-shots.mjs`
(mock started with `MOCK_FIRST_QUESTION_S=99999 MOCK_SAMPLE_SCALE=4`, preview running) drives
all of it and saves screenshots; it uses the page's `?debug` hook (`window.__redshift`).
