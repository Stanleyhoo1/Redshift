# Redshift mission control (web UI)

Live 3D view of the rover fleet exploring the super-Earth: the map fills in as the
rovers sense it, with each rover's state, battery and goal, and the fleet's decisions
in the event log. It talks to the sim over the protocol in [`../PROTOCOL.md`](../PROTOCOL.md).

Vite + React + three.js (@react-three/fiber, drei). Node 22.

## Run with the mock sim (no MuJoCo needed)

```sh
npm install
npm run mock      # fake sim on port 8000 (env: SEED, NUM_ROVERS, SIM_SPEED, MAP_SIZE, GRAVITY_G)
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

- Toggles: Reality (ground truth overlay), Frontiers, Paths, Slope colours.
- Follow a rover with the buttons or by clicking its card; Off returns to the overview.
- Keys: `R` `F` `P` `S` toggle layers, `1` to `9` follow a rover, `0` or `Esc` for the overview.
- Reset sim asks for confirmation, then restarts the sim with a random seed.
- The overview keeps the explored area in frame. Dragging or zooming takes over the
  camera until you press Off (or the sim resets).
