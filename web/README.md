# Redshift mission control (web UI)

Live 3D view of the rover fleet exploring the super-Earth: the map fills in as the
rovers sense it, with each rover's state, battery and goal, and the fleet's decisions
in the event log. It talks to the sim over the protocol in [`../PROTOCOL.md`](../PROTOCOL.md).

Vite + React + three.js (@react-three/fiber, drei). Node 22.

## Run with the mock sim (no MuJoCo needed)

```sh
npm install
npm run mock      # fake sim on port 8000 (env: SEED, NUM_ROVERS, SIM_SPEED, MAP_SIZE, GRAVITY_G, BRAIN_BUDGET_MS,
                  #   GEMMA_ENABLED, GEMMA_BUDGET_MS, OPERATOR_TIMEOUT_S, MOCK_GEMMA=offline, MOCK_PROTOCOL=2)
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
  rover's latest sweep as points in its colour).
- Select a rover (its card, its 3D model or name tag, a Follow button, or `1` to `9`): the
  camera follows it and its live sensor panel opens over the lower left of the view:
  camera (FPV or Chase, about 4 frames a second), a LiDAR scope (rover at the centre facing
  up, returns coloured by height, the dashed ring is the current range), IMU attitude,
  wheel speed against ground speed with slip and motor load (slip holds its peak for a few
  seconds), spectrometer, battery, and brain time against the stated budget. `Esc` or the
  close button hides the panel; clicking the selected card again returns to the overview.
- Hazards (top right): Rockfall, Soft sand, Dust storm, Motor fault, aimed at a rover or
  Auto. The sim's own error text shows inline if it refuses. Rovers are never told: they
  find out through LiDAR (boulders, dust), wheel slip (sand) or fleet status (faults).
  On screen: boulders drop in, the fleet's soft zones are hatched (the true patch is a
  dashed outline with Reality on), a dust storm hazes the view with a banner, a faulted
  rover blinks red and its card counts down.
- Event log rows with a target icon fly the camera to where it happened. Hazards are red.
- Keys: `R` `F` `P` `S` `L` toggle layers, `1` to `9` select a rover, `Esc` closes the
  sensor panel (then returns to the overview), `0` for the overview, `/` for the message box.
- Reset sim asks for confirmation, then restarts the sim with a random seed.
- The overview keeps the explored area in frame, in the space the panels leave free.
  Dragging or zooming takes over the camera until you press Off (or the sim resets).
- Gemma (top of the sidebar): the local model's state and average latency ("Gemma 4 E2B,
  Ready · 0.4 s"; amber while warming up, "rules only" when offline or off). Hover or click
  it for calls, fallbacks, p95 and the budget. The sensor panel's compute section has a
  second bar for it under the brain bar: average filled, p95 tick, budget line.
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
- Every version 2 and 3 field is optional: with an older sim the panel shows dashes, the
  camera says it is unavailable, the hazard buttons say the sim does not support them, the
  Gemma pill and bar stay hidden and Comms says the sim has no operator chat.

The mock implements all of version 2 (sensors from a ray-marched LiDAR over its heightmap,
brain timings, `POST /api/hazard` with plausible reactions). Its camera endpoint answers
503 on purpose. `node scripts/v2-shots.mjs` (mock and preview running) clicks through every
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
