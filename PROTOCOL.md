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
| `AUTO_RESET_S` | 60 | after exploration completes and all rovers park, wait this long, then reset with a random seed and broadcast `init` (0 = off) |
