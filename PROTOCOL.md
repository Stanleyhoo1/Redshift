# Redshift protocol (sim <-> web)

This is the contract between the simulation server (`sim/`) and the web UI (`web/`).
Both sides must follow it exactly. Do not use em dash characters anywhere in this project.

## Ports and URLs

| What | Where |
|---|---|
| Sim server (FastAPI) | port 8000: `GET /api/health`, `GET /api/truth`, `POST /api/reset`, `WS /ws` |
| Web UI in Docker | nginx on port 80 inside the container (published as 8080), proxies `/ws` and `/api` to `sim:8000` |
| Web UI dev | Vite on 5173, proxies `/ws` and `/api` to `http://localhost:8000` |
| Mock server (web dev only) | `node web/mock/mock-server.mjs` on port 8000, same protocol as the real sim |

The browser always connects to a relative URL: `ws(s)://<location.host>/ws` and `/api/...`.

## Coordinates and the map grid

- World frame: x east, y north, z up, metres. The map is centred on the origin.
- The map is a grid of height samples (vertices): `nx` by `ny`, spacing `res` metres.
- Sample `(i, j)`: `i` indexes x, `j` indexes y. World position: `x = origin[0] + i * res`, `y = origin[1] + j * res`.
- Flat arrays are row major by `j`: `index = j * nx + i`.
- Default: map size 120 m, `res = 0.5`, so `nx = ny = 241`, `origin = [-60, -60]` (older builds used 80 m, 161, -40). Clients must read the size from `init.map`, never assume it.
- A "cell" in messages is `[i, j, h, slope_deg]`: grid indices (ints), measured height in metres (float, 2 decimals), local slope in degrees (float, 1 decimal).

## Server -> client messages (JSON text frames on /ws)

### `init`
Sent once right after a client connects, and again to all clients after a reset.

```json
{
  "type": "init",
  "world": {"name": "Redshift-b", "gravity": 19.62, "gravity_g": 2.0, "seed": 42},
  "map": {"nx": 161, "ny": 161, "res": 0.5, "origin": [-40.0, -40.0], "min_h": -3.0, "max_h": 8.0},
  "lander": {"x": 0.0, "y": 0.0, "radius": 2.5},
  "rovers": [{"id": "r1", "name": "Rover 1", "color": "#ff7a45"}],
  "known": {"cells": [[80, 80, 0.12, 3.4]]},
  "targets": [],
  "events": [],
  "fleet": {}
}
```

- `known.cells` is the full map known so far (lets a late joiner catch up).
- `targets`, `events`, `fleet` use the same shapes as in `tick` (events: last 50).
- `min_h` / `max_h` are the true terrain height range (useful for colour scales).

### `tick`
Sent about 5 times per wall-clock second.

```json
{
  "type": "tick",
  "t": 123.4,
  "rovers": [
    {"id": "r1", "x": 3.2, "y": -7.5, "z": 0.4, "yaw": 1.57, "speed": 0.8,
     "battery": 0.64, "state": "exploring", "goal": [10.0, -4.0],
     "path": [[3.2, -7.5], [6.0, -6.0], [10.0, -4.0]], "target_id": null}
  ],
  "cells": [[120, 88, 0.42, 12.3]],
  "frontiers": [[12.0, -8.5, 34]],
  "targets": [
    {"id": "t3", "x": 8.1, "y": -2.0, "kind": "mineral", "status": "found", "found_by": "r2", "found_t": 98.2}
  ],
  "events": [
    {"t": 120.1, "rover": "r2", "level": "warn", "msg": "Stalled on a 24 deg slope. Lowering fleet slope limit to 22 deg."}
  ],
  "fleet": {
    "explored_pct": 12.3, "slope_limit_deg": 22.0, "slope_limit_initial_deg": 30.0,
    "targets_found": 2, "targets_total": 6, "sim_speed": 3.0
  }
}
```

- `t`: sim time in seconds.
- `yaw`: radians, 0 = facing +x, counter-clockwise positive.
- `battery`: 0 to 1.
- `state`: one of `"exploring" | "investigating" | "returning" | "charging" | "recovering" | "idle"`.
- `goal`: `[x, y]` or `null`. `path`: planned waypoints, at most 30 points, may be empty.
- `cells`: only cells newly observed or whose value changed since the last tick.
- `frontiers`: current frontier cluster centres `[x, y, size_in_cells]`.
- `targets`: full list of targets discovered so far (never undiscovered ones). `status`: `"found" | "investigating" | "sampled"`. `kind`: `"mineral" | "ice" | "vent"`.
- `events`: only new events since the last tick. `level`: `"info" | "warn" | "discovery"`. `rover` may be `null` for fleet-wide events.

## Version 2 additions (hazards, sensors, edge budget)

All fields below are additions. Clients must tolerate them being missing (older sims).

### Rover additions (in every `tick.rovers[]` entry)

```json
{
  "state": "fault",
  "sensors": {
    "lidar_range": 8.0, "lidar_hits": 812,
    "pitch": 3.1, "roll": -1.2,
    "wheel_speed": 0.62, "ground_speed": 0.58, "slip": 0.06,
    "motor_load": 0.41,
    "spectro": {"id": "t3", "dist": 6.2}
  },
  "scan": [1.2, 3.4, 0.1, 1.3, 3.6, 0.2],
  "brain": {"last_ms": 4.2, "avg_ms": 5.1, "peak_ms": 17.8, "over": 0}
}
```

- New state `"fault"`: motors disabled by a hazard; the rover is a parked obstacle until it reboots.
- `sensors` (what the rover's own sensors read this cycle):
  - `lidar_range`: current effective LiDAR range in m (drops in a dust storm). `lidar_hits`: returns in the last sweep.
  - `pitch`, `roll`: degrees, from the simulated IMU (nose up = positive pitch, right side down = positive roll).
  - `wheel_speed`: average wheel surface speed from the wheel encoders, m/s. `ground_speed`: actual body speed over ground, m/s.
  - `slip`: 0 to 1, how much the wheels spin without moving the rover (0 = full grip).
  - `motor_load`: 0 to 1, average absolute motor torque as a fraction of the torque cap.
  - `spectro`: nearest target the spectrometer currently sees, or `null`.
- `scan`: points from the rover's most recent LiDAR sweep, flat `[x, y, z, x, y, z, ...]` in world metres, 1 decimal, at most 300 points (downsampled).
- `brain`: compute used by this rover's decision making (perception processing, mapping, planning, decisions; NOT the simulated sensor hardware), in ms: `last_ms` last cycle, `avg_ms` and `peak_ms` over the last 10 sim seconds, `over` = cycles over budget in that window.

### Fleet additions (`fleet` in `init` and `tick`)

```json
{"budget_ms": 40.0, "cycle_ms": 200.0, "soft_zones": [[12.0, -4.5, 2.5]]}
```

- `budget_ms`: stated per-rover compute budget per decision cycle. `cycle_ms`: decision cycle length (sim time).
- `soft_zones`: soft ground the fleet has detected through wheel slip, `[x, y, radius]` (what the fleet believes, not ground truth).

### Hazards (`hazards` in `init` and `tick`)

Full list of hazards in the current sim (small, sent every tick):

```json
[
  {"id": "h1", "type": "rockfall", "t": 120.0, "x": 5.0, "y": 8.0, "radius": 3.0, "rover": "r2",
   "rocks": [[4.2, 7.5, 0.6, 0.5], [5.6, 8.4, 0.7, 0.6]], "active": true},
  {"id": "h2", "type": "sand", "t": 150.0, "x": -6.0, "y": 3.0, "radius": 2.5, "rover": "r1", "known": false, "active": true},
  {"id": "h3", "type": "dust_storm", "t": 200.0, "remaining": 45.0, "active": true},
  {"id": "h4", "type": "rover_fault", "t": 260.0, "rover": "r3", "remaining": 60.0, "active": true}
]
```

- `type`: `"rockfall" | "sand" | "dust_storm" | "rover_fault"`. `t`: sim time it started. `rover`: the rover it was aimed at, or null.
- `rockfall.rocks`: true boulder positions `[x, y, z_centre, radius]`. Boulders drop in over about 1 s from `t`.
- `sand`: hidden soft ground (LiDAR cannot see it). `known`: true once any rover has detected it by wheel slip.
- `dust_storm` and `rover_fault` have `remaining` sim seconds and become `"active": false` when over (they may then be dropped from the list).
- Rovers never read this list. They only learn about hazards through their sensors (LiDAR sees boulders, wheel slip reveals sand, LiDAR returns shrink in dust, a faulted rover's status is shared with the fleet).

### Event additions

- New level `"hazard"`: an environment change (always `rover: null`), e.g. "Rockfall on Rover 2's route."
- Optional `"at": [x, y]`: where the event happened, so the UI can fly the camera there.

### `POST /api/hazard`

Body: `{"type": "rockfall" | "sand" | "dust_storm" | "rover_fault", "rover": "r2"}` (`rover` optional: the sim picks a good one).
Returns `{"ok": true, "hazard": {...}}` (same shape as in `hazards`), or HTTP 400 with `{"ok": false, "error": "..."}` (unknown type, no valid placement, every rover already faulted).

## Version 3 additions (local Gemma planner, operator chat, clarification requests)

All optional for clients (older sims omit them).

### `planner` (in `init` and every `tick`)

```json
{"model": "gemma4:e2b", "state": "ready", "online": true, "calls": 42, "ok": 39, "fallbacks": 3,
 "avg_ms": 410.0, "p95_ms": 980.0, "last_ms": 350.0, "budget_ms": 3000.0, "over_budget": 0}
```

`state`: `"ready" | "warming" | "offline" | "disabled"`. When not ready, the rovers run on rules alone.

### Event addition

Optional `"source": "rules" | "gemma" | "operator"` on events (missing means rules). Gemma-sourced messages quote the sensor readings Gemma cited.

### `chat` (in `tick`: new messages since the last tick; in `init`: the last 50)

```json
[{"id": "c12", "t": 123.4, "from": "operator", "to": "r2", "text": "rover 2, check the crater east of the lander", "kind": "command", "source": null, "request_id": null},
 {"id": "c13", "t": 124.1, "from": "r2", "to": "operator", "text": "Heading to (14, 2), 18 m. Max slope on the way 12 deg.", "kind": "ack", "source": "gemma", "request_id": null}]
```

- `from`: `"operator"`, a rover id, or `"fleet"`. `to`: a rover id, `"all"`, `"operator"` or null.
- `kind`: `"command" | "ack" | "refuse" | "question" | "answer" | "info"`.
- `source`: who interpreted or produced it (`"gemma"` or `"rules"`), null for operator text.
- `request_id`: links questions and answers to a request below.

### `requests` (in `init` and every `tick`: full list of OPEN clarification questions)

```json
[{"id": "q3", "rover": "r2", "t": 130.0, "question": "Something I can't identify blocks my path 4 m ahead (LiDAR: 0.8 m tall). What should I do?",
  "options": [{"id": "reroute", "label": "Go around"}, {"id": "proceed_slowly", "label": "Approach slowly"}, {"id": "wait", "label": "Wait"}],
  "default": "reroute", "expires_in": 14.2, "image": "/api/camera/r2.jpg?view=fpv"}]
```

- A rover asks when it (or Gemma) cannot interpret what it senses. It acts cautiously (slows or holds) while waiting.
- `expires_in`: WALL-CLOCK seconds left (env `OPERATOR_TIMEOUT_S`). On expiry the rover takes `default` (always the safest option) and says so in the chat. Since version 6 the default is 0: `expires_in` is null and the request waits for the operator (see Version 6).

### `POST /api/chat`

Body `{"text": "...", "rover": "r2"}` (`rover` optional). Returns `{"ok": true, "id": "c12"}` immediately; Gemma (or the rule-based fallback parser) interprets it asynchronously, the rules check it is safe, and the rover answers in `chat` with `ack`, `refuse` (with the reason and numbers) or `question` (when unclear). 400 for empty text.

### `POST /api/answer`

Body `{"id": "q3", "option": "reroute"}`. Returns `{"ok": true}`, 404 for an unknown request, 409 if it already expired or was answered.

## Version 4 additions (lifeforms, water, alien spires)

All optional for clients.

### `features` (in `init`; ground truth of static features, for rendering and the Reality overlay)

```json
{
  "water": [{"id": "w1", "kind": "lake", "level": -1.4, "outline": [[10.0, 20.0], [14.5, 22.0], [12.0, 26.5]]}],
  "spires": [{"id": "s1", "x": -12.0, "y": 8.5, "height": 3.2, "radius": 0.35, "standing": true}]
}
```

- `water`: liquid bodies (`kind`: `"lake" | "river"`), flat surface at height `level`, `outline` a closed polygon in world metres. LiDAR gets no returns from liquid, so these show up as holes in the rovers' map until a rover identifies them.
- `spires`: tall alien crystal spires (LiDAR sees them like rocks). `standing` false once one has fallen (see the `spire_fall` hazard).

### Rover additions

- New state `"observing"`: the rover is watching a lifeform from a safe distance.
- Optional `"track_id"` on the rover: the lifeform track it is observing, or null.

### `lifeforms` (in `init` and every `tick`; ground truth, full list, small)

```json
[{"id": "l1", "kind": "grazer", "x": 5.2, "y": -14.0, "z": 0.3, "yaw": 0.8, "speed": 0.3, "size": 1.2}]
```

For rendering creatures (the UI decides when to show them: e.g. when a rover is tracking them, or with Reality on).

### `tracks` (in `init` and every `tick`; what the fleet believes from its sensors)

```json
[{"id": "k1", "x": 5.0, "y": -13.8, "vx": 0.2, "vy": 0.1, "size": 1.1, "last_seen": 412.3,
  "observer": "r2", "label": "possible lifeform, six legs, about 1.2 m", "status": "tracking"}]
```

- A track starts when LiDAR returns move between scans (something that moves on its own). `label` comes from Gemma (camera image) or the rules ("moving object").
- `status`: `"tracking" | "lost"`. `observer`: the rover currently observing it, or null.

### Fleet additions

- `water_zones`: water the fleet has identified, `[[x, y, radius], ...]` (impassable for every rover).
- `lifeforms_seen`: count of distinct lifeforms the fleet has confirmed.

### Hazard additions

- New hazard type `"spire_fall"`: a spire topples across a rover's route. Shape: `{"id", "type": "spire_fall", "t", "rover", "spire": "s3", "x0", "y0", "x1", "y1", "active": true}` (base and tip of the fallen spire). Falls over about 1.5 s from `t`.
- New request type for `POST /api/hazard`: `"lifeform"` moves a small herd of creatures into view of the target rover (a demo convenience; the rovers still have to notice them themselves). Its hazard entry: `{"id", "type": "lifeform", "t", "rover", "x", "y", "active": true}`.

### Point and click placement

`POST /api/hazard` accepts optional `"x"` and `"y"` (world metres, e.g. where the user clicked on the map):

```json
{"type": "rockfall", "x": 12.5, "y": -30.0}
```

- `rockfall`, `sand`, `spire_fall` and `lifeform` are placed at (or as close as safely possible to) that point. A rockfall at a point lays its boulders across the route of the nearest rover heading that way if there is one, otherwise as a cluster there. A spire fall topples the nearest standing spire within about 6 m toward the point, or drops a new spire there.
- `dust_storm` is global and ignores the point. `rover_fault` ignores the point and uses `rover`.
- 400 with a clear `error` if the point is off the map, on the lander, within 2 m of a rover, or unsuitable (e.g. sand on a steep slope).

## Version 4b additions (sample collection and caching)

All optional for clients.

- Each rover carries a few sample tubes (default 3). Sampling uses an instrument that depends on the target kind: mineral = `"drill"` (core), ice = `"scoop"`, vent = `"sniffer"` (gas), trace = `"swab"`. It takes several sim seconds, then the rover analyses the sample on board.
- New target kind `"trace"`: organic traces left near lifeform herds (sampling the trace is allowed; the creatures themselves are never sampled).
- Full tubes must be delivered to the lander's sample cache before more can be taken. Rovers weigh science value against distance, battery and free tubes (and ask Gemma for a second opinion) when deciding what to sample.

Target additions (in `targets[]`):

```json
{"id": "t3", "kind": "mineral", "status": "sampled", "instrument": "drill", "value": 8,
 "analysis": "Olivine-rich basalt with water-bearing clays", "carried_by": "r2", "cached": false}
```

- `value`: science value 1 to 10 once analysed (null before). `analysis`: one short line (null before). `carried_by`: rover carrying the tube, or null. `cached`: true once delivered to the lander.

Rover additions:

```json
{"samples": ["t3", "t6"], "capacity": 3,
 "sampling": {"target": "t3", "instrument": "drill", "progress": 0.4}}
```

- `samples`: target ids in the rover's tubes. `sampling`: present only while an instrument is running.

Fleet additions: `"samples_cached"`, `"samples_carried"` (counts) and `"science_score"` (sum of `value` of cached samples).

## Version 5 additions (asking rovers questions, rover stats, large camera frames)

All optional for clients.

### Questions in chat

`POST /api/chat` is unchanged. The sim decides whether operator text is a command or a question (Gemma's interpretation, or the rule-based parser: question words, a trailing "?", "tell me", "report", "status"). A question gets an answer instead of an action:

```json
{"id": "c40", "t": 512.0, "from": "r2", "to": "operator", "kind": "reply", "source": "gemma", "reply_to": "c39",
 "text": "I've mapped about 1,900 m2 (14% of the region), mostly north-east of the lander, and driven 410 m. Found 3 targets: t4 ice, t7 mineral, t9 ice. Sampled 2 of them."}
```

- New `kind`: `"reply"` (an answer to an operator question). New optional field `reply_to`: the id of the operator message it answers (also allowed on `ack` and `refuse`).
- Addressed to one rover ("rover 2, what have you found?", or `rover` in the POST body): that rover replies. Addressed to everyone or nobody ("what has the fleet explored?"): one reply `from: "fleet"`, unless the question is clearly per rover ("everyone, report battery"), then each rover replies.
- Answers are grounded: the sim builds a fact sheet from the rover's own state and memory (what it mapped, where, what it found and sampled, hazards met, lifeforms seen, current goal and why, battery). Gemma writes the reply from the fact sheet only; any number in the reply that is not in the fact sheet makes the sim use its rule-based answer instead. Without Gemma the rule-based answers cover status, exploration, finds, samples, battery, hazards, lifeforms, water, location and "why".
- A rover never makes up what it has not sensed: if asked about something outside its knowledge, it says so.
- Clarification (sim implementation): every chat message carries `reply_to` (null when it answers nothing, e.g. operator messages). Besides `reply`, `ack` and `refuse`, it is also set on the `question` sent back when a command was unclear and on the `info` line when a pending trip is dropped. The reply arrives within about 3 s (Gemma's latency budget), otherwise the rule-based answer is posted.

### Rover addition: `stats` (in every `tick.rovers[]` entry, optional)

```json
"stats": {"distance_m": 410.2, "mapped_m2": 1904.0, "targets_found": 3, "samples_taken": 2, "hazards_met": 4, "lifeforms_seen": 1, "replans": 17}
```

Running totals for this rover since the sim started: `distance_m` driven, `mapped_m2` of ground first mapped by this rover, targets it found, samples it collected, hazards it detected (rocks, sand, dust, spire), herds it observed, and replans.

### Camera size

`GET /api/camera/{rover_id}.jpg` now accepts `w` and `h` up to 1280 x 720 (for an enlarged view in the UI). Larger values are clamped. Default stays 320 x 240.

## Version 6 additions (commanding by place, choosing between candidates, questions that wait)

All optional for clients.

### Commands that name a place

Operator messages can name things the fleet has sensed instead of coordinates: "go to the water source", "can we investigate the lake again for signs of life", "check out the vent", "go back to the herd", "rover 3, survey the spires", "take a look at t7". Polite forms ("can we ...", "could you ...", "let's ...", "I want you to ...") followed by an action are commands, not questions. The sim resolves the place from fleet knowledge only (identified water, found targets by kind or id, lifeform tracks and the places herds were observed, spires on the fleet's map, hazard sites, the lander), never from ground truth the fleet has not sensed.

- One candidate, or a qualifier that picks one (nearest, farthest, biggest, smallest, newest, a compass direction, "the one near rover 2", "the second one", an id): the rover goes. The goal is a safe reachable point next to the place (for water, a shoreline point about 3 m from the edge; for a creature, the observation standoff; never inside water or a no-go zone). The ack says where and how far.
- "investigate", "survey", "look around", "look for life", "check for signs of ...": after arriving the rover surveys: it stops, turns slowly through 360 deg so the camera sweeps the area while LiDAR watches for movement, for about 30 sim seconds, then reports in chat (`kind: "info"`, `reply_to` the command): what moved and what it saw. A moving track during the survey starts the normal lifeform observation.
- No rover named or picked in the UI (or "all"): the closest rover that is free goes, and says so ("r2 is closest (18 m), heading there.").
- Nothing known matches: the fleet says so and lists what it does know ("I don't know any water yet. Known: 2 vents, 1 ice deposit, 1 herd.").
- More than one candidate and nothing to choose between them: a `choose_place` request (below). The rover that would go holds position while it waits.

### Request additions

```json
{"id": "q7", "rover": "r2", "kind": "choose_place", "t": 2655.0,
 "question": "I know 3 bodies of water. Which one?",
 "options": [{"id": "p1", "label": "Lake 14 m across, 22 m north-east of me", "x": 31.0, "y": -12.5},
             {"id": "p2", "label": "Stream near (-40, 8), 61 m west", "x": -40.0, "y": 8.0},
             {"id": "p3", "label": "Pond 6 m across near the lander", "x": 4.0, "y": 9.5},
             {"id": "cancel", "label": "Never mind"}],
 "default": null, "expires_in": null, "image": null}
```

- `kind` (optional on every request): `"anomaly" | "lifeform" | "choose_place"`.
- Options may carry `x`, `y`: the UI highlights every candidate on the map (numbered markers) and the operator answers by clicking a marker, an option chip, or typing in chat ("the second one", "p2", "the one near the lander"). An operator chat message that clearly picks an option of an open `choose_place` request answers it (the chat reply carries that `request_id`).
- `default` may be null (no automatic choice). `expires_in` may be null: the request never expires and waits for the operator.

### Questions wait for the operator

`OPERATOR_TIMEOUT_S` now defaults to 0: no request expires and `expires_in` is null. A rover with an open request holds position until the operator answers; it still moves if staying is the danger (backing out of soft ground or liquid, heading home on a low battery, making room for a creature that came too close). Set `OPERATOR_TIMEOUT_S` above 0 to bring back the countdown and the automatic safe default.

Sim notes (clarifications):
- Anomaly and lifeform requests keep a non-null `default` (the safe option, applied only when `OPERATOR_TIMEOUT_S` is above 0). A `choose_place` request has `default: null`, or `"cancel"` when `OPERATOR_TIMEOUT_S` is above 0.
- A new operator command for a rover (go to, a place, explore, sample, return) closes that rover's open request without a reply (it disappears from `requests`; `POST /api/answer` then gets 409).
- The rover state stays `"exploring"` during a survey (goal null). Chat for a place command: one `ack` (or `refuse`, `question`), an `info` on arrival, and with investigate the survey `info`; all carry `reply_to` the command, and `request_id` when they follow a `choose_place` answer.
- "go to the lander" without investigate means return home (the closest free rover when none is named).

## Version 7 additions (exploring an area, curiosity, chat memory, following lifeforms)

All optional for clients.

### Rover addition: `focus` (present only while a rover concentrates on an area)

```json
"focus": {"x": 12.0, "y": -30.0, "r": 15.0, "reason": "operator", "label": "the pond's shore", "ends_in": 64.0}
```

- While focused, the rover explores the unknown ground within `r` of the centre first (frontiers inside the circle win, then the nearest edge of the circle), and stays until `ends_in` (sim seconds) runs out or nothing unknown is left nearby. Then it reports in chat (`kind: "info"`) what it mapped and found there and goes back to normal exploration.
- `reason: "operator"`: "explore around the lake", "go to the vent and look around", "explore there", "explore north of the lander", and "investigate ..." (survey first, then the focus). About 120 sim s. A plain "go to" still just goes.
- `reason: "interest"`: the rover's own curiosity. When it finds something new (a science target, water, a cluster of spires, a herd it finished watching, or ground that looks different), it sticks around for about 60 sim s to look for more, and says so in the event log ("r2 found water here. Sticking around for about 60 s to look for more."). Never while returning, on low battery, during an operator focus, or where another rover already has a focus.

### Chat memory

The sim keeps a short conversation context: the last 8 or so messages, plus the places, targets, candidates and rovers most recently mentioned by the operator or the rovers. References resolve against it: "go back there", "what did you find there?", "the second one", "do that again", "same for rover 3", "that lake", "the vent you mentioned". Unclear references get a short question or a `choose_place` request as usual. Answers and acks may mention what was resolved ("Back to the pond's shore at (15.7, 1.7), 22 m.").

### Lifeform sightings

When a rover sees a herd it has not asked about before, it starts following at the observation standoff (never closer than the planetary protection distance) at once, because the creatures may wander off, and asks the operator in the same moment (request `kind: "lifeform"`, options: keep following, observe from here, keep exploring). It keeps following until the operator answers (no expiry by default). Each herd is asked about once. Safety still wins: low battery, hazards and water end the follow, and the rover says so.

Sim notes (clarifications):
- `focus.ends_in` counts from when the rover reaches the circle (within `r` plus about 4 m); until then it shows the full duration. An area the rover cannot get into within about 180 sim s is given up (reported).
- An operator focus reports in chat with `reply_to` the command (and `request_id` after a `choose_place` answer). An interest focus reports in the event log only: it is the rover's own initiative, so the chat stays the operator's conversation.
- Lifeform option ids stay `follow`, `observe_here`, `keep_exploring` (labels "Keep following", "Observe from here", "Keep exploring"); `default` stays `observe_here`. When the follow ends for safety before an answer, the request is closed (it leaves `requests`, `POST /api/answer` gets 409) with a chat `info` carrying its `request_id`.
- Any rover waiting for the operator (any request kind) keeps its normal battery check; when it must go home it closes its open requests the same way ("Battery 21%, I can't wait any longer: heading back to the lander. ..."), drops a focus or a follow, and leaves at once. A rover held on the lander pad keeps charging.

### `GET /api/camera/{rover_id}.jpg`

Live image rendered by MuJoCo from the rover's point of view, at the current sim state.
Query: `view=fpv` (onboard camera on the mast, looking forward, default) or `view=chase` (behind and above the rover, showing the rover itself), `w` and `h` (pixels, default 320 x 240, max 640 x 480).
Returns `image/jpeg`. 404 for an unknown rover, 503 with a JSON error if rendering is unavailable on this machine. The UI polls it a few times a second for the selected rover only.

## HTTP

- `GET /api/health` -> `{"ok": true}`
- `GET /api/truth` -> ground truth for the "reality" toggle (not what the rovers know):
  `{"nx": 161, "ny": 161, "res": 0.5, "origin": [-40, -40], "heights": [...nx*ny floats, row major by j...], "targets": [{"id": "t1", "x": 1.0, "y": 2.0, "kind": "mineral"}]}`
- `POST /api/reset` with optional JSON body `{"seed": 7}` -> restarts the sim (new terrain if a new seed), then broadcasts a fresh `init`. Returns `{"ok": true, "seed": 7}`.

## Client -> server

Nothing required. The server must ignore any text the client sends.

## Sim environment variables

| Var | Default | Meaning |
|---|---|---|
| `SEED` | 42 | terrain and target layout |
| `NUM_ROVERS` | 3 | rovers in the fleet |
| `SIM_SPEED` | 3.0 | sim seconds per wall second |
| `MAP_SIZE` | 120 | map width in metres |
| `GRAVITY_G` | 2.0 | gravity in Earth g |
| `VIEWER` | 0 | 1 = open the native MuJoCo viewer (only when running outside Docker) |
| `BRAIN_BUDGET_MS` | 40 | stated compute budget per rover per decision cycle |
| `GEMMA_ENABLED` | 1 | 0 = rules only |
| `GEMMA_URL` | http://host.docker.internal:11434 | Ollama server (falls back to localhost:11434) |
| `GEMMA_MODEL` | gemma4:e2b | preferred model; falls back to gemma4:e4b, then gemma3:4b if not installed |
| `GEMMA_FAKE` | 0 | 1 = deterministic fake model (demo and tests without Ollama) |
| `OPERATOR_TIMEOUT_S` | 0 | 0 = operator requests never expire (the rover waits); above 0 = wall seconds before the rover takes the safe default |
| `AUTO_RESET_S` | 0 | off by default (the sim only resets on `POST /api/reset`); if set, after exploration completes and all rovers park, wait this long, then reset with a random seed and broadcast `init` |
