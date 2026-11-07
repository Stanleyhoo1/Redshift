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
- Default: map size 80 m, `res = 0.5`, so `nx = ny = 161`, `origin = [-40, -40]`.
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
| `MAP_SIZE` | 80 | map width in metres |
| `GRAVITY_G` | 2.0 | gravity in Earth g |
| `VIEWER` | 0 | 1 = open the native MuJoCo viewer (only when running outside Docker) |
| `BRAIN_BUDGET_MS` | 40 | stated compute budget per rover per decision cycle |
| `AUTO_RESET_S` | 60 | after exploration completes and all rovers park, wait this long, then reset with a random seed and broadcast `init` (0 = off) |
