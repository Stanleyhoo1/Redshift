# Redshift protocol (sim <-> web)

This is the contract between the simulation server (`sim/`) and the web UI (`web/`).
Both sides must follow it exactly. Do not use em dash characters anywhere in this project.

## Ports and URLs

| What | Where |
|---|---|
| Sim server (FastAPI) | port 8000: `GET /api/health`, `GET /api/truth`, `GET /api/config`, `POST /api/launch`, `POST /api/reset`, `WS /ws` |
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

## Version 8 additions (autonomy mode, unexplored gaps and priorities, re-survey, area questions, object detection, faster camera)

All optional for clients.

### Autonomy mode

- Fleet field `mode`: `"autonomous"` (default, env `AUTONOMY_MODE`) or `"supervised"`. `POST /api/mode` with `{"mode": "supervised"}` returns `{"ok": true, "mode": "supervised"}`; 400 for any other value. Chat also works ("switch to supervised mode", "you decide from now on").
- **autonomous:** the rovers never wait for the operator on their own decisions. Where they would ask (an anomaly they cannot identify, a lifeform sighting, several places matching an operator command), Gemma decides (the rules when Gemma is offline; below 0.5 confidence the safest option). Each decision is logged as an event with a `decision` object and said in chat (`kind: "info"`). Operator commands still override at any time.
- **supervised:** every such decision becomes a request to the operator: every anomaly (not only the first or uncertain ones), every lifeform sighting, every ambiguous place. Gemma's pick is shown as a suggestion (option field `"suggested": true`, and the question says "Gemma suggests: go around (0.82)"). Requests wait (version 6 rules).
- Switching to autonomous with requests open: each rover decides them at once and logs the decision.
- Event addition `decision` (optional): `{"question": "New rocks 4 m ahead, 0.6 m tall", "options": ["reroute", "proceed_slowly", "wait"], "choice": "reroute", "by": "gemma", "confidence": 0.82, "why": "LiDAR 0.6 m tall, 3 cells changed: rocks"}`.

### Unexplored gaps and fleet priorities

- New place kind for commands: unexplored gaps ("holes", "gaps", "blank spots", "dark patches", "unexplored areas", "unmapped pockets"): connected areas of unknown ground of at least about 4 m2 with known ground around most of their edge, ranked by size and distance. "explore one of those holes" sends the closest free rover to the best one (autonomous) or opens a `choose_place` request with every gap marked (supervised, when there are several).
- Fleet field `priority` (present while set): `{"kind": "gaps", "label": "the unexplored gaps inside the map", "count": 4, "since": 1834.0, "places": [{"x": 12.0, "y": -8.0, "r": 3.5}]}` (`places`: the gaps still left, for the map, gaps priority only). Set by "prioritize the gaps", "focus on the unexplored holes first", "prioritize the north", "prioritize water", "prioritize sampling". Every rover's goal choice weights the priority strongly until nothing is left or the operator says "back to normal" or "cancel the priority". The ack states the plan ("Prioritizing 4 unexplored gaps: r1 takes the one at (12, -8), r2 the one at (-30, 4)."). Clearing it is announced in chat.

### Re-survey

- Focus addition `mode`: `"explore"` (unknown ground first) or `"resurvey"`. An operator focus on an area that is already mostly mapped becomes a re-survey: the rover drives a loop through the circle for the full time, rescanning with LiDAR, camera and spectrometer and stopping for anything interesting (targets the spectrometer missed, creatures, changes), then reports what changed or what it found ("No changes since it was mapped" is a valid report). Curiosity focuses still skip mapped areas.

### Area questions

- "what did we find there?", "anything near the lake?", "what's at the vent?": the answer covers the referenced place only (within about 15 m): targets, samples, water, spires, herds and hazards there, from fleet knowledge. "We" questions about a place answer for the fleet at that place.

Sim notes (clarifications for the four parts above):
- `decision.by` is `"gemma"` or `"rules"` (also when a safety rule overrides Gemma's pick; `why` says so); `confidence` is null for the rules. The same decision by the same rover within 30 sim s is logged again but said in chat once. An autonomous place decision is said in the command's `ack` ("I know 2 bodies of water; going for the lake ... (Gemma picked it, 0.80)."), so a command still gets one reply.
- Supervised: a suggestion appears once Gemma has answered (none while Gemma is offline). A rover already waiting on a question decides a new one itself and logs it. "you decide" with questions open decides those now and stays supervised; with none open it switches to autonomous. Requests settled by a decision close like answered ones (`POST /api/answer` then gets 409).
- Gaps count (places, `priority.count`) only when a reachable frontier touches them; exploring one is an operator focus around it. Priority `kind`: `"gaps"`, `"direction"` (plus `"direction": "north"`), `"water"` (no water known yet: low ground first, the label says so), `"sampling"` (free rovers start on found targets and fetch them from up to 90 m). A rover with its own operator task finishes it first; curiosity pauses while a priority is set.
- Re-survey when less than 15% of the circle is unknown, or when an operator focus finds nothing to explore within 10 s of arriving. `focus.mode` is on every focus.

### Object detection (camera perception)

- Each rover runs an onboard detector on its own camera frames (RGB and depth pixels only, never ground truth). Classes: `rock`, `boulder`, `spire`, `creature`, `water`, `mineral`, `ice`, `vent`, `trace`, `rover`, `lander`.
- Rover addition `vision` (latest detection for that rover):
  `{"frame": 1234, "t": 512.3, "view": "fpv", "ms": 14.2, "objects": [{"label": "creature", "conf": 0.83, "box": [0.41, 0.38, 0.58, 0.62], "dist": 6.2, "bearing": 12.0}]}`
  `box` is normalised `[x0, y0, x1, y1]` (0 to 1, origin top left). `dist` in metres from depth. `bearing` in degrees, positive to the left of the camera axis.
- `GET /api/camera/{rover_id}.jpg` adds headers `X-Frame` (frame number), `X-Frame-Age-Ms`, and `X-Detections` (compact JSON of the objects detected in exactly that frame, same shape as `vision.objects`). Clients read them with `fetch`.
- Decisions use detections and log them: water confirmation, creature confirmation for LiDAR tracks, the class of a new obstacle, and targets spotted by camera before the spectrometer reaches them ("Camera: possible vent 12 m ahead (0.71). Going to check it.").
- `GET /api/vision/stats` (optional): detector precision and recall per class against ground truth, measured in the sim for transparency (the rovers never see the truth).

### Faster camera

- The server renders the requested views continuously in the background and keeps the newest frame; `GET /api/camera/{rover_id}.jpg` answers at once with the newest frame for that view and size (`X-Frame-Age-Ms` says how old it is). Views nobody has asked for in the last few seconds stop rendering.

Clarifications (sim implementation notes):
- Only the onboard `fpv` camera runs the detector: for `view=chase` (not a rover camera) `X-Detections` is `[]`. The first request for a new view and size waits for its first frame (as before); later ones answer at once. `X-Frame` numbers increase across all frames the server renders (streams, detector frames, Gemma's frames), so a client can tell a repeated frame from a new one. The three headers are listed in `Access-Control-Expose-Headers`.
- Rover `vision` is absent until the rover's first detection. It comes from the newest of: the rover's own small detector render (about 1 per second, 256 x 192, no shadows), the streamed fpv view the UI is polling, or the frame Gemma looked at; `vision.frame` equals `X-Frame` when it came from that frame. Weak detections (confidence below 0.3) are dropped; decisions use about 0.5 and up.
- `GET /api/vision/stats` answers `{"ok": true, "frames", "conf_min", "match", "classes": {label: {"tp", "fp", "fn", "precision", "recall"}}, "overall", "ms": {"avg", "p95"}, "confusion", "camera": {...}}`, measured on one detector frame in five since the sim started (`precision` / `recall` are `null` until there is data).

## Version 9 additions (rover health and damage, distress, rescue, towing, salvage, repair)

All optional for clients.

### Rover addition: `health` (every tick)

```json
"health": {"hp": 64, "status": "damaged",
           "parts": {"drive_l": 100, "drive_r": 35, "lidar": 90, "camera": 70, "power": 100, "comms": 100},
           "diagnosis": "Right drive motor at 35%: pulls left, top speed 0.4 m/s.",
           "last_hit": {"t": 812.4, "cause": "boulder", "shock_g": 7.4}}
```

- `parts` are 0 to 100 (100 = like new). `hp` is the overall figure the UI shows (worst parts weigh most).
- `status`: `"ok" | "damaged" | "immobile" | "wrecked"`. Immobile: cannot drive (both drive sides too weak, or power gone) but can be repaired or towed. Wrecked: beyond field repair; it can only be towed home for a lander repair or salvaged for parts.
- `diagnosis`: the rover's own plain-language reading of its state, from its sensors (motor current against wheel speed, IMU shocks, LiDAR return counts, camera noise, battery behaviour), never from ground truth.
- Damage changes behaviour through the sensors and the physics: a weak drive side makes the rover pull to one side and lowers its top speed and climbing ability (its own slope limit drops); a damaged LiDAR loses range and rays; a damaged camera gets noisy and the detector's confidence drops; damaged power drains the battery faster; damaged comms shortens how far its messages reach.

### Damage sources

Damage comes from what happens in the physics: hard contacts and shocks from rocks and boulders (including a rockfall landing on or next to a rover), a spire falling onto a rover, tipping over, driving into water, long stalls and hard pulling in soft sand (motor strain), dust on the camera and panels, and lifeforms: a herd wandering into a rover, or a territorial creature charging a rover that stays too close too long. Rovers see a charge coming (a fast LiDAR track heading at them, confirmed by the camera) and back away when they can. Each hit is an event with a `damage` object:

```json
"damage": {"rover": "r2", "part": "drive_r", "before": 100, "after": 35, "cause": "boulder", "shock_g": 7.4}
```

Demo control: `POST /api/hazard` with `{"type": "damage", "rover": "r2", "severity": "light" | "heavy" | "disable" | "wreck"}` (optional `"part"`). Damage is also reachable by clicking: a rockfall or falling spire placed on a rover hits it.

### Distress, rescue, towing, salvage, repair

- A rover that becomes immobile stops, logs its diagnosis, and sends a **distress signal** (event plus chat, kind `"info"`): "r2: immobile. Right drive at 0%, left at 20%. Requesting help at (14.2, -30.5)." If its comms are too damaged to reach the others, they notice it went silent (no heartbeat for about 15 s) and go to its last known position.
- The best placed healthy rover becomes the **helper** (closest, enough battery for the trip and back, plus towing if needed) and says so. It drives to the disabled rover, then **diagnoses** it for about 20 s (camera and close inspection), and reports what it found. Some damage can be fixed in the field (a jammed wheel, a tripped motor controller, a dirty camera): the helper fixes it and both carry on.
- Otherwise there are three options, decided by Gemma in autonomous mode or asked as a request of `kind: "rescue"` in supervised mode, with the rules marking which are safe:
  - **tow**: the helper hitches up and tows the rover back to the lander slowly (stricter slope limit, extra battery) for repair. Only offered when a safe route exists and the helper's battery covers it.
  - **salvage**: the helper takes the most useful working parts (adds them to the fleet's spares) and leaves the wreck.
  - **leave**: the wreck stays where it is, marked on the map, and the fleet carries on without it.
- Rescue request options carry the rules' verdict: `{"id": "tow", "label": "Tow it home (180 m)", "safe": true, "note": "Route at most 12 deg, helper battery after tow about 41%"}`; unsafe options have `"safe": false` and the reason in `note` (the UI locks them). The request may carry `about` (the disabled rover's id).
- **Repair at the lander**: a rover at the lander gets repaired over time, using spare parts for broken components. A drive, LiDAR, camera or power part below 30% needs a spare; better parts are serviced without one.
- Fleet additions: `spares` `{"drive": 2, "lidar": 1, "camera": 1, "power": 1}`; `rescues` `[{"rover": "r2", "helper": "r1", "phase": "en_route" | "diagnosing" | "repairing" | "towing" | "salvaging" | "done", "plan": "tow" | "salvage" | "leave" | null}]`; `wrecks` `[{"rover": "r2", "x": 14.2, "y": -30.5}]`.
- New rover states: `"disabled"` (immobile, waiting for help), `"rescuing"` (helper on its way or working), `"towing"`, `"towed"`, `"repairing"` (at the lander), `"wrecked"` (left behind).
- Rovers answer questions about their health ("rover 2, are you damaged?", "how is rover 3?") from `health`, and the operator can order a rescue ("rover 1, go help rover 2", "tow rover 2 home", "salvage rover 3", "leave it").

Sim notes (clarifications):
- `status` "immobile" can still need a tow (a burnt-out motor needs a spare at the lander); "wrecked" means immobile even after a field repair and badly hurt overall (hp under 30, or three or more parts under 10%). A rover pulls toward its weak drive side ("Right drive motor at 35%: pulls right, top speed 0.4 m/s."). A skid steer needs both sides: either drive side under 15%, or power under 10%, is immobile.
- `damage` hazard: `light` takes 15 to 25 points off one part (the named one, or a random one); `heavy` takes a part (default a drive side) to 35 to 45% and never immobilises; `disable` without a part burns out the right drive (0%) and jams the left wheel (20%, fixable in the field, not enough on its own); with `drive_l` / `drive_r` / `power` that part is knocked out; with `lidar`, `camera` or `comms` that part drops to 10% and the right drive burns out too, so the rover is always immobile; `wreck` takes both drives and the power system out (beyond field repair, the sensors stay worth salvaging; with a part, that part is lost too). Extra severity `"jam"`: a jammed left wheel and a tripped right motor controller (immobile, fully fixable in the field, for demoing a field repair). The rover reacts at once (distress in the same call). The response adds `"health"` (the rover's health after the hit). 400 for an unknown severity or part, an unknown rover, a rover already wrecked or being towed, and `disable` / `jam` on a rover that is already immobile. The hazard entry is `{"id", "type": "damage", "t", "rover", "x", "y", "severity", "part", "active": false}`.
- A rockfall or spire fall placed on or within 2 m of a rover is aimed at it (instead of the old 400); the lander is still refused. `lifeform` accepts `"charge": true` (demo): the herd's territorial member charges the rover a few seconds after it appears.
- Rescue phases: hitching up is reported as `"towing"`, waiting for a free helper as `"en_route"` with `helper` null, waiting for the decision (or the operator) as `"diagnosing"`. Finished rescues stay in `rescues` with phase `"done"` for 60 s.
- A `rescue` request's `rover` is the helper (it waits by the disabled rover); `about` is the disabled rover. Option ids are `tow`, `salvage`, `leave`, the safe ones first; `default` is the rules' first safe option. Answering an option with `safe: false` gets HTTP 400 and the request stays open.
- Towing is physical (a MuJoCo `connect` constraint between the helper's rear hitch and the towed rover's front hitch, the towed rover's drives passive). If it gets stuck the towed rover follows the hitch kinematically instead; the event log says which ("on the tow bar" or "on the rigid bar").
- Repair at the lander: a spare is fitted and tested in about 20 s; a rover whose broken part has no spare waits at the lander (state `"repairing"`, charging) until one is delivered. Salvaged parts count once the helper docks, or at once when it fits one to itself.
- Radio range is about 220 m times `comms` (the whole 120 m map at 100%); a rover heard by neither the lander nor another rover for 15 s goes silent.
- Chat: question topic `"health"`; the command parser adds `{"type": "rescue", "rover": helper or null, "target": rover or null, "plan": "help" | "tow" | "salvage" | "leave"}` (null target: the rover currently in trouble).

### `GET /api/camera/{rover_id}.jpg`

Live image rendered by MuJoCo from the rover's point of view, at the current sim state.
Query: `view=fpv` (onboard camera on the mast, looking forward, default) or `view=chase` (behind and above the rover, showing the rover itself), `w` and `h` (pixels, default 320 x 240, max 640 x 480).
Returns `image/jpeg`. 404 for an unknown rover, 503 with a JSON error if rendering is unavailable on this machine. The UI polls it a few times a second for the selected rover only.

## Version 7b additions (launch settings and the setup phase)

Built in parallel with Versions 7 to 9 on another branch. In code comments, "v7" next to launch settings, the setup phase, biomes, world features (mesas, canyon, dunes, boulder fields, pits, geysers, ice sheets, flora) and lifeform kinds refers to this section.

All optional for clients. A client that knows none of this still works: an older client sees a sim that
starts when someone launches it (or at once with `AUTO_LAUNCH=1`), and `POST /api/reset {"seed": n}` behaves
as before.

### Server phase

With `AUTO_LAUNCH=0` (the default) the sim builds the world from its env vars at start and waits in the
**setup phase**: the world is frozen at t = 0 (the fleet stands on the lander pad), `POST /api/hazard`,
`POST /api/chat` and `POST /api/answer` answer 409 `{"ok": false, "error": "The mission has not launched yet."}`,
while `GET /api/truth`, `GET /api/config` and the camera work. `POST /api/launch` or `POST /api/reset` starts
the mission. `AUTO_LAUNCH=1`, or `AUTO_RESET_S` above 0 (the unattended demo loop), starts at once as before.

### `init` additions

- `phase`: `"setup"` or `"running"`. Missing (older sims) means running.
- `config`: the launch settings this world was built with, keyed like the table below, for example
  `{"gravity_g": 2.0, "seed": 42, "num_rovers": 3, "map_size": 120, "water": true, "spires": true, "lifeforms": true, "mesas": true, ..., "wear": "normal", "sim_speed": 3.0}`.

A launch and a reset send every client a fresh `init` with phase `"running"`.

### Launch settings

| Key | Type | Range | Env var (default) | Meaning |
|---|---|---|---|---|
| `gravity_g` | float | 0.5 to 3, step 0.1 | `GRAVITY_G` (2.0) | gravity in Earth g; the climb and tow limits and the slope thresholds follow it (Sim notes below) |
| `seed` | int | 0 to 2147483647 | `SEED` (42) | same seed, same terrain, targets and life (`SEED` and `POST /api/reset` also take larger seeds, as before) |
| `num_rovers` | int | 1 to 8 | `NUM_ROVERS` (3) | rovers that land |
| `map_size` | int (m) | 60 to 200, step 10 | `MAP_SIZE` (120) | width of the square sector |
| `water`, `spires`, `lifeforms` | bool | | none (true) | the version 4 features |
| `mesas`, `canyon`, `dunes`, `boulder_fields`, `pits`, `geysers`, `ice_sheets`, `flora`, `ground_detail` | bool | | `MESAS`, `CANYON`, `DUNES`, `BOULDER_FIELDS`, `PITS`, `GEYSERS`, `ICE_SHEETS`, `FLORA`, `GROUND_DETAIL` (1 each) | the version 7 world features; each is built in its own v7 step, until then its switch is only stored and echoed |
| `biomes`, `life_variety` | bool | | `BIOMES`, `LIFE_VARIETY` (1 each) | version 7: each world's own ground, colours and terrain mix; more lifeform kinds (see Biomes below) |
| `wear` | choice | `off`, `normal`, `harsh` | `WEAR` (normal) | how fast wheels and motors wear out (harsh: about 4 times sooner) |
| `sim_speed` | float | 0.5 to 10, step 0.5 | `SIM_SPEED` (3.0) | sim seconds per real second |

### `GET /api/config`

```json
{"ok": true, "phase": "setup",
 "current": {"gravity_g": 2.0, "seed": 42, "num_rovers": 3, "map_size": 120, "water": true, "wear": "normal", "sim_speed": 3.0},
 "defaults": {"gravity_g": 2.0, "seed": 42, "num_rovers": 3, "map_size": 120, "water": true, "wear": "normal", "sim_speed": 3.0},
 "fields": [{"key": "gravity_g", "type": "float", "label": "Gravity", "group": "planet",
             "help": "Heavier gravity makes every slope harder: the rovers climb and tow less.",
             "min": 0.5, "max": 3.0, "step": 0.1, "unit": "g"},
            {"key": "wear", "type": "choice", "label": "Wear", "group": "wear",
             "help": "How fast wheels and motors wear out. A worn-out rover breaks down and needs a tow.",
             "choices": [{"value": "off", "label": "Off", "help": "Rovers never wear out."}]}],
 "presets": {"gravity_g": [{"label": "Earth", "value": 1.0}, {"label": "Redshift-b", "value": 2.0}, {"label": "Heavy", "value": 3.0}]},
 "physics": [{"gravity_g": 0.5, "climb_deg": 30.0, "tow_deg": 24.0, "grip_limited": true},
             {"gravity_g": 2.0, "climb_deg": 21.3, "tow_deg": 8.4, "grip_limited": false}]}
```

(Shortened: `current` and `defaults` carry every key of the table, `fields` every row, `physics` 26 rows.
Values have the table's type: `int` settings are JSON integers (`"map_size": 120`), except that an env
value off the 10 m grid, e.g. `MAP_SIZE=95.5`, is reported as it is (`95.5`). Such a value, or any value
outside the table that `current` holds, may be sent back to `POST /api/launch` unchanged: a value equal to
the current world's is no change and always passes.)

- `current`: the settings of the world the sim holds now (built and waiting, or running). `defaults`: what
  the env vars give.
- `fields`: the table above as data, in order, so a client can draw the form from it (a setting added later
  shows up without a client change): `key`, `type` (`int`, `float`, `bool`, `choice`), `label`, `group`
  (`planet`, `fleet`, `sector`, `features`, `wear`, `simulation`), `help` (one line), plus `min`, `max`,
  `step`, `unit` for numbers and `choices` (`value`, `label`, `help`) for a choice.
- `presets.gravity_g`: named gravities for quick picks.
- `physics`: the predicted steepest climb and towing limit (deg) at each gravity step from 0.5 to 3 g for the
  rover this sim builds; `grip_limited` where the motors could climb more than the 30 deg grip cap.

### `POST /api/launch`

Body: any launch settings, for example `{"gravity_g": 1.0, "num_rovers": 4}`. Keys left out keep the values
of the world the sim holds when the launch is carried out (launches and resets go one at a time, so one
that waits behind another builds on the settings that one left); `{}` launches it as built. Numbers keep to
their step: `gravity_g` 2.04 is refused (`gravity_g must be in steps of 0.1 g`), float noise such as
1.7000000000000002 counts as 1.7.

- From the setup phase with nothing changed, the built world starts. Otherwise (changed settings, or a
  launch while running, which starts a new mission) a new world is built first, then the old one stops.
- A launch sent while the sim was in the setup phase starts the mission once. If another launch (or a
  reset) started it while this one waited, nothing is rebuilt: `200` as below when this launch asked for
  the same settings as the running world, else `409 {"ok": false, "error": "Another launch started the
  mission first. Use New mission to change its settings."}`. So two setup screens, or a script and a
  person, launching together land one mission.
- `200 {"ok": true, "phase": "running", "config": {...}}`; every client gets a fresh `init`.
- `400 {"ok": false, "error": "gravity_g must be between 0.5 and 3 g; water must be true or false", "errors": ["gravity_g must be between 0.5 and 3 g", "water must be true or false"]}`:
  every problem found, and nothing changes. The texts: `unknown setting 'warp' (use one of gravity_g, seed, ...)`,
  `<key> must be true or false`, `<key> must be one of off, normal, harsh`, `<key> must be a number`,
  `<key> must be a finite number`, `<key> must be a whole number`, `<key> must be between <min> and <max> <unit>`,
  `<key> must be in steps of <step> <unit>`, `Body must be a JSON object like {"gravity_g": 1.5, "num_rovers": 4}`.
  The texts name settings by key; the dashboard's setup screen shows them with the field labels.
- `500 {"ok": false, "error": "Could not build the new world: ..."}`: the world the sim held is untouched
  (and keeps running if it was). `503` before the sim has a world.

### `POST /api/reset` additions

- Takes any launch setting next to `seed`, for example `{"seed": 7, "gravity_g": 1.5}`. Keys left out keep
  the current world's values (so "Reset sim" keeps gravity, fleet and features and only changes the seed),
  read when the reset is carried out: a reset queued behind a launch keeps that launch's settings.
- Keeps its old leniency: keys that are not launch settings are ignored (and listed in `ignored`), and
  `seed` works as before (`"8"` and `7.9` work, an empty body keeps the seed, a body that is not an object
  is ignored, any seed of 0 or more works, also above the setup screen's 2147483647). A negative seed gives
  400 `seed must be 0 or more` (it never built a world), one that is not a number `seed must be an integer`.
  Bad launch settings give 400 as for `POST /api/launch`; a failed build gives 500 and the old world keeps
  running.
- Works from the setup phase too: the mission starts at once.
- Returns `{"ok": true, "seed": 7, "config": {...}}`, plus `"ignored": ["rovers"]` when the body had keys
  that are not settings.

### Sim notes

- The slope thresholds tuned for 2 g follow the launch gravity. The predicted climb limit is
  `asin(4 T / r / (m g))` with T = 7.2 N m per wheel motor, r = 0.18 m and m = 22.5 kg, capped at 30 deg
  where grip and tipping take over: 21.3 deg at 2 g, 14.0 at 3 g, 30 at 1 g. From it: target paths at most
  `min(15, climb - 2)` deg, the lake and river no-trap flood `min(18, climb - 1)`, hazard placement
  `min(20, climb - 1)`, the floor of the learned slope limit `min(12, climb - 2)`, the sand wheel load
  `0.2 * min(1, g / 2)`, and the towing limit `0.8 * min(30, climb for twice the mass)`. At 2 g every one is
  exactly its old value.
- The towing limit is `Config.tow_limit_deg`, the one place it is computed: the formula of the wear and
  towing design, `0.8 * asin(4 T / r / ((m1 + m2) g))`, with the asin capped at the same 30 deg grip limit as
  a single rover (a tow train never climbs steeper than one rover alone). The two agree from about 0.73 g up
  (8.4 deg at 2 g, 17.0 at 1 g); below that the cap holds it at 24 deg. Later steps (canyon ramps, tow paths
  and their announcements) read `cfg.tow_limit_deg` rather than computing their own, so the setup screen,
  `physics` and the rovers always quote the same number.
- Gemma's prompts state each request's gravity: at exactly 2 g the text is byte for byte as before
  ("super-Earth (gravity 2 g, ..."); any other gravity is written as it is ("gravity 2.04 g" for
  `GRAVITY_G=2.04`); at 1 g and below the world is "an unknown planet (gravity 1 g, ...". Operator commands
  and chat answers name the world without a number ("a super-Earth", at 1 g and below "a planet"), since
  they may only quote their own inputs.
- `Config.from_env` clamps `NUM_ROVERS`, `SIM_SPEED`, `MAP_SIZE` and `GRAVITY_G` to the table's ranges
  (values inside the range are kept as given, also off the step grid). `SEED` is only kept from going
  negative: as before version 7, `SEED=3000000000` builds that very world. The feature env vars are on when
  unset; set, they are on only for `1`, `true`, `yes` or `on`. An unknown `WEAR` gives normal.

### Geysers (`geysers` setting)

Only when the feature is on. `features.geysers` lists the vents: `[{id, x, y, radius, active, period_s,
next_s, height}]`. `radius` is the steam column radius (m), `height` its height (m), `active` is true
while the vent erupts and `next_s` is the sim time of the next start or end. The tick after an eruption
starts or ends carries `features` again. `fleet.geyser_zones: [[x, y, r]]` holds the keep-out zones the
fleet learned from plume returns; the key is present only when it is not empty. While a plume scatters
the LiDAR, the rover `scan` also holds the short steam returns. They never enter `cells` or `tracks`.

### Alien flora (`FLORA`)

Only when flora is on. All optional.

```json
"features": {"flora": [{"id": 0, "kind": "tree", "x": 42.6, "y": 54.5, "height": 3.1, "radius": 1.8, "hue": 0.42},
                       {"id": 20, "kind": "bush", "x": 40.1, "y": 50.2, "height": 0.5, "radius": 0.4, "hue": 0.1, "bulbs": 8}]},
"fleet": {"flora_zones": [[42.51, 54.37, 6.2]], "flora_seen": 1}
```

- `features.flora` is ground truth (init, and with any later `features`). Trees have a stalk and a domed
  cap; `hue` goes from 0 (violet) to 1 (teal). Bushes are lime bulbs; `bulbs` is how many.
- `fleet.flora_zones` are keep-out circles `[x, y, radius]` of plants the fleet identified. The radius is
  the assumed plant size (2.2 m tree, 0.6 m bush) plus 4 m. Planning treats them as impassable.
  `flora_seen` counts identified plants. Both keys appear only once a plant is known.
- A rover identifies a plant from its own sensors. LiDAR sees the stalk as an obstacle within 10 m, then
  the mast camera takes a small segmentation frame (at most every 2 s per rover) and labels it. The first
  sighting of each plant logs an event with `at`, for example
  "r1 sees possible plant life: fungal tree (camera 0.90, 6.6 m). Keeping 4 m away."

### Biomes (`biomes` setting, `BIOMES`)

Only when biomes are on. All optional for clients.

```json
"world": {"name": "Redshift-b", "gravity": 19.62, "gravity_g": 2.0, "seed": 7,
          "biome": {"id": "verdigris", "name": "Verdigris lowlands", "secondary": "rust"}},
"features": {"biome_map": {"nx": 31, "ny": 31, "origin": [-60.0, -60.0], "res": 4.0, "w": [0.0, 0.0, 0.12, ...]}}
```

- `world.biome` (init only): the primary biome `id` and `name`, and the `secondary` biome's id. Ids: `basalt`,
  `verdigris`, `frost`, `ash`, `salt`, `obsidian`, `rust`. `rust` (today's look) is never the primary.
- `features.biome_map`: the secondary biome's share of the ground, 0 to 1 rounded to 2 decimals, on a coarse
  grid of about 4 m cells covering the map (`w` row major by j, `nx * ny` values, sample (i, j) at
  `origin + (i, j) * res`). Sent with every `features`. A client blends the two biomes' colours with it.
- The biome scales how many of each v7 feature are made (the feature settings stay the upper limits) and sets
  the colours of the ground, rocks, sky and light in the camera. Rovers never read it.
- `lifeforms` setting `life_variety` (`LIFE_VARIETY`): more lifeform kinds; `lifeforms[].kind` is then one of
  `grazer`, `skitterer`, `drifter`, `crawler`, `burrower`.
- Launch settings: `biomes` and `life_variety` are bools in group `features` (labels "Biomes" and "More
  lifeforms"), on from the environment, off in a bare `Config()`.

## HTTP

- `GET /api/health` -> `{"ok": true}`
- `GET /api/truth` -> ground truth for the "reality" toggle (not what the rovers know):
  `{"nx": 161, "ny": 161, "res": 0.5, "origin": [-40, -40], "heights": [...nx*ny floats, row major by j...], "targets": [{"id": "t1", "x": 1.0, "y": 2.0, "kind": "mineral"}]}`
- `POST /api/reset` with optional JSON body `{"seed": 7}` -> restarts the sim (new terrain if a new seed), then broadcasts a fresh `init`. Returns `{"ok": true, "seed": 7}` (version 7: plus `config`, and it takes launch settings too).
- `GET /api/config` -> the launch settings, their table, presets and predicted limits (version 7).
- `POST /api/launch` with optional JSON body of launch settings -> starts the mission (version 7).

## Client -> server

Nothing required. The server must ignore any text the client sends.

## Sim environment variables

| Var | Default | Meaning |
|---|---|---|
| `SEED` | 42 | terrain and target layout (0 or more; the setup screen offers 0 to 2147483647) |
| `NUM_ROVERS` | 3 | rovers in the fleet (1 to 8) |
| `SIM_SPEED` | 3.0 | sim seconds per wall second (0.5 to 10) |
| `MAP_SIZE` | 120 | map width in metres (60 to 200) |
| `GRAVITY_G` | 2.0 | gravity in Earth g (0.5 to 3); out of range values are clamped |
| `VIEWER` | 0 | 1 = open the native MuJoCo viewer (only when running outside Docker) |
| `BRAIN_BUDGET_MS` | 40 | stated compute budget per rover per decision cycle |
| `GEMMA_ENABLED` | 1 | 0 = rules only |
| `GEMMA_URL` | http://host.docker.internal:11434 | Ollama server (falls back to localhost:11434) |
| `GEMMA_MODEL` | gemma4:e2b | preferred model; falls back to gemma4:e4b, then gemma3:4b if not installed |
| `GEMMA_FAKE` | 0 | 1 = deterministic fake model (demo and tests without Ollama) |
| `OPERATOR_TIMEOUT_S` | 0 | 0 = operator requests never expire (the rover waits); above 0 = wall seconds before the rover takes the safe default |
| `AUTONOMY_MODE` | autonomous | `autonomous` (Gemma decides, logs every decision) or `supervised` (the operator is asked every decision) |
| `AUTO_RESET_S` | 0 | off by default (the sim only resets on `POST /api/reset`); if set, after exploration completes and all rovers park, wait this long, then reset with a random seed and broadcast `init` |
| `AUTO_LAUNCH` | 0 | 0 = build the world and wait in the setup phase for `POST /api/launch`; 1 = start at once (`AUTO_RESET_S` above 0 also starts at once) |
| `MESAS`, `CANYON`, `DUNES`, `BOULDER_FIELDS`, `PITS`, `GEYSERS`, `ICE_SHEETS`, `FLORA`, `GROUND_DETAIL` | 1 | version 7 world features, 0 = leave it out (each is built in its own v7 step) |
| `BIOMES`, `LIFE_VARIETY` | 1 | version 7 biomes (own ground, colours and terrain mix per seed) and more lifeform kinds, 0 = off |

| `LIFE_VARIETY` | 1 | version 7 "More lifeforms": skitterers, drifters, crawlers and burrowers join the grazers, chosen by biome. 0 = grazers only, exactly as before |
| `WEAR` | normal | `off`, `normal` or `harsh`: how fast wheels and motors wear out |

### Landforms (`MESAS`, `CANYON`, `PITS`, `BOULDER_FIELDS`)

Optional `features` keys, sent only when the flag is on:

- `mesas`: `[{id, kind, x, y, radius, height, level}]`. `kind` is `mesa` or `butte`.
- `canyon`: `{id, path: [[x, y]...], width, depth, ramps: [{x, y, heading, length}]}` or null.
- `pits`: `[{id, kind, x, y, radius, depth}]`. `kind` is `sinkhole` or `skylight`.
- `boulder_fields`: `[{id, x, y, radius, count, rocks}]`. The boulders are geoms `bfield_<i>` and are in the truth heights like rocks; `rocks` lists that field's boulders as `[x, y, size]` (size: mean horizontal half-size, m), in geom order.

### More lifeforms (`LIFE_VARIETY`)

On by default from the environment, off in `Config()`. Launch setting key `life_variety` ("More lifeforms",
group `features`). With it off, `lifeforms` are grazers only and every message is exactly as before.

- `lifeforms[].kind` is one of `grazer`, `skitterer`, `drifter`, `crawler`, `burrower`. The other fields are as
  in version 4 (`id, kind, x, y, z, yaw, speed, size`). For a drifter `z` is the height of its bell (1.5 to
  3.5 m above the ground); for a burrower `z` goes below the ground while it is down.
- A burrower also has `up`: 0 (down, under the ground) to 1 (all the way up). Clients draw its mound always
  and its head by `up`. No other kind sends `up`.
- Each group of creatures gets a kind from the biome weights at its home spot (the primary biome when the
  terrain has no regions; every kind weighs the same when biomes are off), and a couple of extra groups are
  added. The lifeform hazard brings a group whose kind follows the weights at the spot it picked.
- Kinds: grazer (herds of 3 to 4), skitterer (swarms of 6 to 14, 0.4 to 0.55 m, run up to 1.4 m/s and scatter
  from a rover within 8 m), drifter (2 to 5, drift with their group's wind at about 0.2 m/s), crawler (1 or 2,
  1.6 to 2 m long, at most 0.15 m/s), burrower (3 to 6; up for 3 to 6 s, down for 6 to 14 s, and never up
  within 5 m of a rover).
- Rovers only see LiDAR points and camera frames. They keep at least 4 m from any track, and 6 m (following at
  8 m) from a track their LiDAR measures at 1.5 m or more or whose camera label names a crawler or a shell. They
  also stop for a fresh unconfirmed cluster within that distance ahead. With Gemma, the describe prompt lists
  what the kinds look like, so a track `label` can name one ("possible skitterer swarm, small and fast").
- Event text for the lifeform hazard names the group ("A swarm of skitterers wanders into view of Rover 1.").

### Dunes and ice sheets

Both are on by default from the environment (`DUNES`, `ICE_SHEETS`) and off in `Config()`.

- `features.dunes: [{id, x, y, radius, wind_deg}]` is sent only when dunes are on. Dune fields are shapes in the
  height field: long ripples across the wind plus a few larger dunes. Wheel friction there is 0.5 (firm ground
  0.8) with a little extra rolling resistance, so wheels slip more but rovers can still cross.
- `features.ice: [{id, x, y, radius}]` is sent only when ice sheets are on. An ice sheet is level ground with
  wheel friction 0.05.
- A rover on ice sees high slip at low motor load on level ground. It records an anomaly with
  `kind: "low_traction"` (fields `slip`, `load`, `at`), slows, backs out along its tracks and marks an ice zone.
  This is not the soft ground rule and makes no soft zone.
- `fleet.ice_zones: [[x, y, r]]` appears in `state.fleet` only when ice sheets are on and the fleet has learned
  at least one zone. Ice zones are closed to planning (like an avoid area).
