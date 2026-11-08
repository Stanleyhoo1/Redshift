# Redshift mission control (web UI)

Live 3D view of the rover fleet exploring the super-Earth: the map fills in as the
rovers sense it, with each rover's state, battery and goal, and the fleet's decisions
in the event log. It talks to the sim over the protocol in [`../PROTOCOL.md`](../PROTOCOL.md).

Vite + React + three.js (@react-three/fiber, drei). Node 22.

## Run with the mock sim (no MuJoCo needed)

```sh
npm install
npm run mock      # fake sim on port 8000 (env: SEED, NUM_ROVERS, SIM_SPEED, MAP_SIZE, GRAVITY_G, WEAR, MESAS ... FLORA,
                  #   BRAIN_BUDGET_MS, GEMMA_ENABLED, GEMMA_BUDGET_MS, OPERATOR_TIMEOUT_S, AUTONOMY_MODE, MOCK_GEMMA=offline,
                  #   MOCK_PROTOCOL=2..9, MOCK_PRELAUNCH=1, MOCK_FIRST_QUESTION_S, MOCK_SAMPLE_SCALE,
                  #   MOCK_ANSWER_DELAY_MS, MOCK_CAMERA=off, MOCK_KNOCK_S)
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
  camera (FPV or Chase, as fast as the sim serves new frames), a LiDAR scope (rover at the centre facing
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
- Reset sim asks for confirmation, then restarts the sim with a random seed and the same settings. New
  mission opens the setup screen.
- Setup screen (version 7 sims): opens by itself while the sim waits for a launch (it cannot be closed
  then) and from New mission. Gravity (slider and presets; the big numeral narrows, gets heavier and redder
  as gravity grows, and a figure shows the predicted steepest climb and towing slope), seed, rovers, sector
  size, speed, the world features and wear, all drawn from `GET /api/config`. Launch sends only what
  changed. An older sim gets a seed-only form that resets.
- Planet view (version 7, web only): `G` or View: Planet in the controls (the row above
  Graphics) lifts the camera off the map, fades, and shows the whole planet under the same
  low sun as the sector, with the landing site in daylight and a glow of thick air on the
  limb. Like the sector, it starts blank: the fleet has not surveyed the planet, so it is a
  dark grey ball with faint latitude and longitude lines every 15 degrees. The landing
  sector shows the fleet's own map, live: a square patch at the site, map north to the
  planet's north, empty right after launch and filling in as the rovers map the ground
  (mapped ground in the map legend's colours with shaded relief, the fleet's water, ice,
  plant and soft sand zones on top). It is enlarged to about 14 degrees across, since the
  real sector is far too small to see from orbit. It repaints at most twice a second, and
  only when the map changed (once a second on Graphics Low). Reality (`R`) also works on the
  planet: it fades in the true planet, drawn from the world's seed (dusty plains, darker
  maria, craters, polar ice), and the patch keeps showing the fleet's map. The legend gives
  the planet's name, its radius (gravity x 6371 km, assuming Earth's density) and the
  site's latitude and longitude (from the seed). A cyan ring frames the landing sector, with
  a live card: sector size, explored share, gravity, sim time and every rover's state.
  Drag to turn the
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
- Following: orbit (left drag) and zoom keep following the rover. A pan (right drag, or a
  modifier with the left button, or two fingers) or a click on an event that flies the
  camera elsewhere pauses the follow right where you put the camera: the rover stays
  selected with its panel open, its Follow button turns dashed with a pause mark, and a
  "Back to Rover 2" pill appears (click it, press `F`, click the rover or its Follow button)
  to glide back. While paused, `F` resumes instead of toggling Frontiers.
- Camera viewer: click the camera picture (or its expand icon) for a large view, about 80%
  of the screen. It asks for frames sized to the viewer and the screen's pixel density (up
  to 1280 x 720), one at a time and as fast as the sim has new ones, and the panel's small feed pauses
  meanwhile. FPV or Chase, rover buttons (`1` to `9`), scroll to zoom around the cursor,
  drag to pan, double-click to reset, `V` switches view, `D` hides the detection boxes, `Esc` or a click outside closes.
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
- Focus (version 7): while a rover concentrates on an area (you asked, "explore around the
  lake", or its own curiosity after a find) a dashed circle in its colour lies on the map
  (softer for curiosity) with a tag like "Exploring the pond's shore · 64 s"; its card and
  sensor panel say "Exploring an area (64 s left)" or "Curious: sticking around (40 s left)".
  A rover that sights a herd follows it while its lifeform question waits for you.
- Lost tracks (version 4): a faded, frozen copy of the creature stays where the fleet last
  saw it, and the tag says "last seen 12 s ago", until the sim forgets the track.
- Autonomy (version 8): under the wordmark, Autonomous / Supervised (hover for what each
  means) shows `fleet.mode` and posts `/api/mode`; the choice blinks until the sim confirms
  it. Autonomous: Gemma decides and every decision lands in the event log with a compact
  line (Gemma or Rules, confidence, the choice, "of 3"); click it for the question, every
  option considered (the chosen one ticked) and why. Supervised: request cards and the chat
  chips mark Gemma's pick ("Gemma suggests"). A fleet `priority` shows as an amber chip
  ("Priority: the unexplored gaps (4 left)") whose cross sends "back to normal" over chat;
  if a sim also lists the places (`places`, `gaps` or `areas`, not in the protocol) the map
  marks them. A re-survey focus reads "Re-surveying the lake's shore · 64 s".
- Event log filters: All, one chip per rover (its colour), Fleet (events without a rover)
  and Decisions, each with its count. The browser keeps the last 1000 events; the log
  renders 120 rows and "Show older" adds more.
- Object detection (version 8): both camera views stream with `fetch` (one request in
  flight, the next as soon as a frame has decoded, a short pause when the sim had nothing
  newer), decode with `createImageBitmap`, and draw on a canvas, so frames swap without
  flicker. The `X-Detections` boxes of exactly that frame are drawn on top in class colours
  (mineral, ice, vent and trace as on the map, creatures lime, spires violet, water blue,
  rocks neutral) with label and confidence (and distance in the large view, where they
  follow zoom and pan at a constant size). The box button (or `D` in the viewer) hides them,
  remembered across reloads. A small readout gives fps and frame age; polling stops while
  the tab is hidden or the viewer covers the panel. The sensor panel adds "Sees: creature
  0.83 at 6 m, rock 0.64 at 4 m" from the rover's `vision` and the detector's time.
- Rover health (version 9): each rover card has an hp bar coloured by status (green, amber,
  red when immobile, grey for a wreck) and six part indicators (left and right drive, LiDAR,
  camera, power, comms; hover for the percentage, hover the row for the rover's own
  diagnosis). The sensor panel adds a Health section: every part with its percentage, the
  diagnosis line and the last hit (cause, shock in g, how long ago). New states have their
  own pills and card lines: Disabled (solid red, blinking), Rescuing, Towing, In tow,
  Repairing, Wrecked (greyed). Each hit (`damage` event) shows in the log with the part,
  before and after on a small bar, cause and shock; the Damage filter lists only hits (each
  rover's filter shows its own). On the map a hit flashes with sparks; a disabled rover gets a
  pulsing red and amber distress beacon with a light pillar and "Rover 2 needs help" (calmer
  once help arrives), its helper a dashed line to it while on the way, then a tag with
  progress while it diagnoses, fixes or salvages; towing draws a tow line; a wreck stays as a
  darkened rover with a ground marker and a "Wreck" chip; the lander tag shows the fleet's
  spares (icon and count each) and "Repairing Rover 2" while its pad pulses. Rescue requests
  (`kind: "rescue"`) list tow, salvage and leave with the rules' verdict and note under each
  (unsafe options are locked) and Gemma's suggestion; the UI reads option `safe` and `note`
  (or `why`, `reason`) and request `about` when a sim sends them. Hazard bar: Damage, then
  click a rover (or pick it under Target), then choose the part (Any by default) and how bad
  (light, heavy, disable, wreck) in the small panel under the bar; the sim's refusal shows
  there. Rockfall and Falling spire may be dropped on a rover ("Lands on Rover 2"); the lander
  stays refused.
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

Version 7 in the mock (the default; `MOCK_PROTOCOL=6` for the previous behaviour): `focus` for
"explore around the lake", "go to the vent and look around", "explore around 10, -5", "explore
north of the lander", "explore here/there" and "investigate ..." (survey, then focus), about
120 sim s; curiosity focuses (about 60 s) now and then after a find; a chat report when a focus
ends; and every new herd is followed at once while the rover asks about it.

Version 8 in the mock (the default; `MOCK_PROTOCOL=7` for the previous behaviour; mode from
`AUTONOMY_MODE`, kept across resets): `POST /api/mode` and chat ("switch to supervised mode",
"you decide from now on"). Autonomous rovers decide anomalies, herds and ambiguous places
after a fake Gemma call (rules when it is offline, the safe option under 0.5) and log each
decision; frontier and sampling choices carry a `decision` too. Supervised rovers ask with
the suggestion marked; switching back decides open questions. LiDAR shadows leave pockets of
unknown ground, so "explore one of those holes" and "prioritize the gaps" have targets (the
gaps priority also lists `places`, a mock extra) until "back to normal". An operator focus on
mapped ground becomes a re-survey. Every tick carries each rover's `vision`; the camera
draws rocks, boulders, targets, creatures, spires, water, rovers and the lander as simple
shapes exactly where its `X-Detections` boxes them, keeps the newest frame per view and size
(served at once while fresh) and renders large frames at half resolution. Test hook:
`POST /api/_mock/gaps` lists the current gaps. `node scripts/v8-check.mjs` (mock started
with `MOCK_FIRST_QUESTION_S=99999`, preview running) checks the log filter, both modes, a
suggested option, the camera boxes and the priority chip, and saves screenshots (low
graphics by default, `GRAPHICS=high` to override). `node scripts/cam-rate.mjs [url] [w] [h]`
measures the camera stream on its own (new frames a second, repeats skipped).

Version 9 in the mock (the default; `MOCK_PROTOCOL=8` for the previous behaviour): rover
`health` that changes behaviour (a weak drive side pulls and slows, a damaged LiDAR loses range,
damaged power drains faster), damage from the demo control (`{"type": "damage", "rover",
"severity", "part"}`, with 400s for an unknown severity or part, a wreck, a rover in tow or
under repair), a rockfall or spire dropped on a rover, strain in soft sand and a knock now and
then (`MOCK_KNOCK_S`). An immobile rover sends a distress signal (silent with dead comms: the
fleet notices after 15 s), the closest rover with the battery drives over, diagnoses for 20 s,
then fixes it in the field (sometimes) or the fleet tows it home (it follows behind the
helper) for a repair with spares, salvages it (spares go up, the wreck stays) or leaves it:
Gemma decides in autonomous mode (logged), supervised mode asks with a `rescue` request.
Fleet `spares`, `rescues` (mock extra: `progress`) and `wrecks`. Chat: "rover 1, go help rover
2", "tow it", "salvage rover 3", "leave it", "how is rover 2?". Test hook: `POST
/api/_mock/heal {"rover"?}`.
Version 7 biomes (`src/lib/biomes.js`): each sim biome has a palette (ground dust, soil, rock, strata, pebbles and dune sand for the terrain shader; zenith, sky, horizon, haze and light for the sky dome and fog; a three step ramp for the sector map patch and the legend swatch). The store keeps init `world.biome` and `features.biome_map`; the terrain shader gets both palettes and the blend grid as a small 8 bit texture and mixes them per pixel (the border frayed by noise), and the atmosphere, the planet view (Reality on), the mapped sector patch, the legends and the World line (with the biome name, also on the sector card) follow the world. An older sim, the mock, or biomes off send neither: the `rust` palette is exactly the old theme, so nothing changes. `node --test test/biomes.test.mjs` covers the palettes and the blend helpers.

Version 7 terrain visuals: map markers keep the small features readable from the sector overview. Each fungal tree (a cap glyph in its violet to teal cap colour), bush (a lime bulb cluster), geyser vent (an ochre ring round a white throat that pulses while it erupts) and boulder field (three grey stones) gets one constant-size glyph (one instanced draw call), shown like the models (Reality on, or once the ground there is mapped) and faded out as the camera comes within about 50 m, gone under 24 m, so close-ups show the models. Boulder fields are drawn as real rocks from `boulder_fields[].rocks`.

Version 7 in the mock (the default; `MOCK_PROTOCOL=6` for the previous behaviour): launch settings with
the sim's table, presets and physics rows (`GET /api/config`), `POST /api/launch` (a second launch sent
from the setup phase lands nothing new, as in the sim) and settings in `POST /api/reset` (same error texts as
the sim; other keys are ignored and listed in `ignored`), and init `phase` and `config`. `MOCK_PRELAUNCH=1` starts in
the setup phase like the sim's default `AUTO_LAUNCH=0` (frozen at t = 0, hazards, chat and answers get 409);
without it the mock runs at once, so the older scripts are unaffected. The v7 world features are only stored
and echoed. `node scripts/setup-unit.mjs` checks the setup helpers, the init fields and the HTTP helpers
without a browser; `node scripts/setup-check.mjs [url] [outDir]` (mock started with `MOCK_PRELAUNCH=1`, or
the sim with `AUTO_LAUNCH=0`) drives the setup screen (also that neither Tab nor New mission gets past the
forced screen), launches and saves 5 screenshots.
`scripts/screenshot.mjs` launches a sim that waits in its setup phase (`POST /api/launch {}`) before its
shots, so it works with the sim's default `AUTO_LAUNCH=0`; a running sim or an older one is left alone.
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
