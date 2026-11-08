# Redshift v7: richer terrain, alien flora, globe view, launch settings, wear and towing, camera detector

Date: 2026-11-08. Branch: `ui-improvements` (based on `main` e9de11b, the MuJoCo `sim/` + React `web/` line).
Approved by the user in the brainstorming session ("approve all"). The user asked for incremental commits
with no Claude co-author trailer, and for nothing existing to break.

Background facts with file:line references for every subsystem below live in
`.superpowers/codebase-map.json` (git-ignored; produced by a read-only mapping pass). Read the section for
your subsystem before changing it.

## Goals

1. More terrain: mesas and buttes, a winding canyon with two ramps, dune fields, boulder fields, sinkholes
   and lava-tube skylights, geysers, ice sheets, and look-only ground detail.
2. Alien flora: fungal "trees" (tall stalks with wide violet or teal caps) and "bushes" (low clusters of
   softly glowing lime bulbs). Flora is life: planetary protection applies.
3. A view toggle in the dashboard between the current plane (sector) view and a globe view of the whole
   planet (option A: a procedural planet you can spin, the landing sector as a glowing marker with live
   fleet stats; toggling flies from orbit down into the plane view and back).
4. A setup screen before the simulation starts: gravity, seed, rover count, map size, sim speed, which
   features exist, wear rate. Nothing runs until Launch. `AUTO_LAUNCH=1` keeps today's behaviour.
5. Wear and tear: each rover wears out; when its drivetrain fails, the nearest able rover tows it to the
   lander with a real MuJoCo tow link, where it is repaired.
6. A toggleable computer-vision overlay on the rover camera feeds: bounding boxes, labels, confidence and
   distance from a simulated object detector, labelled as such in the UI.

## Non-negotiable rules (apply to every section)

- **Nothing existing breaks.** The 134 existing sim tests (`cd sim && env -u PYTHONPATH
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/ -p no:cacheprovider`, about 3 min)
  keep passing with zero edits where at all possible. If an edit is unavoidable it must be purely additive
  (for example a whitelist gaining a new optional key) and explained in the commit message. `npm run build`
  in `web/` keeps passing. Existing features (hazards, click-to-place, chat, place commands, questions,
  samples, Reality, lifeforms, water, spires, camera viewer, sensor detail, mock server) keep working with
  the new features both off and on.
- **Feature flags.** Every new world-changing feature has a `Config` field that defaults to OFF in the
  dataclass, so `Config(seed=42)` (what the tests build) produces a bit-identical world and run.
  `Config.from_env()` (what the server uses) turns them ON by default, each overridable by an env var.
- **Own random streams.** Each new feature draws only from its own stream,
  `np.random.default_rng([cfg.seed, K])` with a distinct constant K per feature. Never draw from the
  existing `rng`, `frng`, `rover.rng`, hazard or lifeform streams. Schedules (geyser eruptions, wear
  thresholds) are fixed at build time from those streams.
- **Rovers only know what they sense.** Rover logic never reads ground truth (terrain heights, feature
  lists, hazard lists, wear truth). They learn through LiDAR, IMU, wheel encoders, motor feedback, camera
  and the new camera detector. Where the sim routes raw hits by geom id (as it already does for creatures
  and water), say so in the code as a stated simplification, like the existing tracker note.
- **Rules keep authority; every decision quotes what was sensed** ("slip 82% at load 0.12", "camera: fungal
  tree 0.87 at 6.1 m"). New events start with the rover id and carry `at` when they have a place.
- **Determinism per seed** for everything that is not wall-clock (the existing `plan_clock` caveat stays).
- **Budgets.** Brain p95 stays well under 40 ms. Keep new MuJoCo geoms at 120 m under about 350 in total
  (ray cost is linear in geom count whatever the group). Camera renders on the sim thread are throttled and
  go through `_hw_call` (sensor hardware time, not brain time).
- **Protocol.** All new fields are optional and documented in a `## Version 7 additions` section of
  `PROTOCOL.md`. Fields that exist only when a feature is on are emitted only then (precedent: rover
  `sampling`). Server-level init fields (`phase`, `config`) are added in `server.build_init_text`, not in
  `Simulation.init_message`, so test_smoke's exact key sets hold. Never use the em dash character anywhere
  (test_v6 scans PROTOCOL.md for it).
- **Gravity.** Launch settings allow 0.5 to 3.0 g. Thresholds that assume 2 g are derived from the climb
  limit so they are identical at 2 g: `asin(4*motor_torque/wheel_radius / (m*g))` gives 21.3 deg at 2 g
  (rover mass 22.5 kg). Specifically TARGET_PATH_SLOPE `min(15, climb-2)`, REACH_SLOPE `min(18, climb-1)`,
  hazards `_passable_base` `min(20, climb-1)`, slope floor `min(12, climb-2)`, the sand spin-load test
  scaled by `min(1, g/2)`. The Gemma prompts state the configured gravity per request (the planner is shared
  across resets) and must keep the 2 g prompt the same length (test_gemma budgets: interpretation prompt is
  397 of 400 tokens).
- **Commits** after each finished piece, authored by the user only: no `Co-Authored-By` trailer, no
  "Generated with" footer.

## 1. Launch settings and the setup screen

### Server

- `config.py`: a `SETTINGS` schema, one entry per exposed setting: key, type, min, max, step, unit, label,
  group, help. Exposed: `seed` (0 to 2147483647), `num_rovers` (1 to 8), `map_size` (60 to 200 m, step 10),
  `gravity_g` (0.5 to 3.0, step 0.1; presets Earth 1.0, Redshift-b 2.0, Heavy 3.0), `sim_speed` (0.5 to 10),
  booleans `water`, `spires`, `lifeforms`, `mesas`, `canyon`, `dunes`, `boulder_fields`, `pits`, `geysers`,
  `ice_sheets`, `flora`, `ground_detail`, and `wear` as a choice `off | normal | harsh` (stored as
  `wear: bool` plus `wear_rate: float`, 1.0 normal, 4.0 harsh).
- `Config.from_settings(base, data) -> (cfg, errors)`: rejects unknown keys, wrong types (bool where a
  number is expected), non-finite and out-of-range values, with readable messages; then
  `dataclasses.replace`. `from_env` reads its clamps from the same table where they overlap.
- Server phase: env `AUTO_LAUNCH` (default 0). With 0, lifespan builds `SimRunner(Config.from_env())` but
  does not start it (`state.phase = "setup"`), so every endpoint, the WebSocket init, the camera and the
  native viewer keep working on a frozen t=0 world. `AUTO_LAUNCH=1`, or `AUTO_RESET_S > 0`, starts at once
  as today. While in setup, `POST /api/hazard`, `/api/chat` and `/api/answer` return 409
  `{"ok": false, "error": "The mission has not launched yet."}`.
- `GET /api/config` -> `{ok, phase, current, defaults, fields, presets, physics}`: `current` is the
  effective settings of the runner's cfg, `defaults` those of `Config.from_env()`, `physics` a small table of
  climb limit and tow limit per gravity step for the UI readout.
- `POST /api/launch` with a settings body (may be empty): validate, then if the cfg equals the built
  runner's and it never started, start it; otherwise rebuild with `restart_sim(cfg)`. Set phase running and
  re-init every client. 400 with the error text on bad input.
- `POST /api/reset` keeps `{"seed": 7}` working exactly as today (restart and run at once) and additionally
  accepts the same settings keys (omitted keys keep the current cfg). Build the new runner before stopping
  the old one, so a failed build leaves the old sim running. `restart_sim(cfg)` replaces `restart_sim(seed)`
  (auto_reset passes `old.cfg.with_seed(random)`).
- Init gains optional `phase` ("setup" | "running") and `config` (the effective settings), added in
  `build_init_text`.
- `docker-compose.yml` forwards `AUTO_LAUNCH` and every new feature env var.

### Web

- `simState.applyInit` stores `phase` (default "running") and `config` (sanitised; derived from
  world/map/rovers/fleet when absent).
- `components/SetupScreen.jsx`: full-screen overlay shown while `phase === "setup"` (cannot be dismissed),
  and opened from a new "New mission" button next to Reset sim (dismissible, prefilled with the current
  settings). Built from the server's `fields` (falls back to a seed-only form if `GET /api/config` is 404,
  i.e. an older sim or the mock). Sections: Planet (gravity slider with presets, m/s2 readout and a small
  slope diagram showing the max climbable slope and the tow limit at that gravity; seed with a random
  button), Fleet (rover count stepper), Sector (map size), Features (toggle grid with one-line
  descriptions), Wear (off / normal / harsh), Simulation (speed). A Launch button posts `POST /api/launch`;
  server validation errors show inline. Visual style matches the existing dashboard (dark, Archivo, the
  theme tokens in `lib/theme.js`).
- Reset sim stays one click (random seed, same settings).
- The mock server: `GET /api/config`, `POST /api/launch`, init `phase`/`config` behind `MOCK_PROTOCOL >= 7`
  (default 7), `MOCK_PRELAUNCH=1` to start in setup; its MAP_SIZE, NUM_ROVERS, GRAVITY_G and SIM_SPEED
  consts become a mutable settings object.

### Tests

`tests/test_v7_settings.py`: the settings parser table (valid, unknown key, out of range, wrong type,
non-finite, negative seed), `from_settings` round trip, `GET /api/config` shape, launch from setup starts the
built runner, launch with changed settings rebuilds (gravity and rover count appear in init), reset with
`{"seed": n}` still restarts and runs, 409s during setup, the derived gravity thresholds are identical at
2 g and lower at 3 g, and the Gemma prompt length at 2 g is unchanged.

## 2. Camera object detector (simulated) and the CV overlay

### Sim

- `world.py` builds a registry mapping every geom id to `(cls, object_id)`: `rock`, `boulder`, `spire`,
  `target:<kind>` (object id = target id), `creature`, `tree`, `bush`, `geyser` (vent and plume), `rover`
  (object id = rover id), `lander`, `water`, `mesa`/terrain stay unlabelled (ground). Decoration geoms
  (group 4) map to their owner so a box covers the whole visible object.
- `camera.py`: `detect_objects(sim, rover_id, view, w, h)` renders the look frame plus MuJoCo's
  segmentation buffer (`mjRND_SEGMENT` + `mjRND_IDCOLOR`) and depth on the existing render thread. Per
  object: pixel mask -> box (normalised `[x0, y0, x1, y1]`), distance (median depth), pixel count, visible
  fraction. A detector model turns that into what a real detector would report: confidence falls with
  distance beyond 6 m, with small pixel area and in a dust storm (visibility from the hazard manager's
  storm state, which models the optics); objects under a minimum size or below a confidence floor are
  missed; labels are human-readable ("fungal tree", "glow bush", "ice deposit", "grazer", "Rover 2").
  Deterministic per (frame time, rover) so tests are stable.
- Endpoint `GET /api/camera/{rover_id}/detect?view=fpv&w=&h=` -> JSON `{ok, t, w, h, view, image:
  <base64 jpeg of the same render>, detections: [{cls, label, id, box, conf, dist_m}], detector:
  "simulated"}`. Image and boxes come from the same render so they always line up. Same size limits and
  error codes as the jpg endpoint (404 unknown rover, 503 no rendering).
- The rovers may use `detect_objects` as a sensor (section 4 uses it to identify flora). Calls from the
  rules are throttled and go through `_hw_call`.

### Web

- A "CV" toggle on the sensor panel's camera and in the large camera viewer (key `C` inside the viewer).
  When on, the feed polls the detect endpoint instead of the jpg endpoint and draws boxes and labels on a
  canvas or SVG overlay sized to the image, coloured by class (dashboard colours: targets by kind, life in
  lime, rovers in their colour, hazards red), label text "fungal tree 0.87 · 6.1 m", and a small
  "simulated detector" tag. The choice persists per browser (visualPrefs pattern).

### Tests

`tests/test_v7_cv.py`: registry covers every geom group used; a rover facing a known target, spire,
creature and another rover detects each with the right class and a box that contains the projected centre;
distance within 0.5 m of truth at 3 to 8 m; no detections of hidden or parked bodies (parked boulders,
spare spires below the map); confidence lower in a dust storm; endpoint shape and 404. Skip cleanly when no
OpenGL is available (as test_camera does).

## 3. Terrain features

Flags: `mesas`, `canyon`, `dunes`, `boulder_fields`, `pits`, `geysers`, `ice_sheets`, `ground_detail`.
With all flags off the world is bit-identical to today.

### Generation (terrain.py)

A landform pass runs right after `_make_heights` (after its rescale) and before water, in a fixed order:
canyon, mesas and buttes, pits, geyser mounds, dunes, ice sheets. Everything after it (water, slope, rocks,
spires, flora, reach, herds, targets) then sits on the final ground. Every landform keeps at least about
20 m from the lander and clear of the lander flat zone, and passes a generalised no-trap check (the
`_no_trap` flood from the lander at the derived REACH_SLOPE may lose at most a small area outside the
feature itself; all targets stay reachable). Counts scale with map area like the existing features.

- **Mesas and buttes**: 1 to 2 mesas (12 to 20 m across) and 1 to 3 buttes (5 to 9 m), flat tops 3 to 6 m
  high, cliff walls steeper than 45 deg (impassable, LiDAR sees them as walls), slightly irregular outlines
  (noise on the radius), a talus apron at the foot. `features.mesas: [{id, x, y, radius, height,
  outline}]`.
- **Canyon**: one winding trench (a smoothed random polyline across a good part of the map, missing the
  lander zone), floor 3 to 4 m below the rim, 6 to 9 m wide, walls steeper than the climb limit, and two
  ramps cutting the walls at no more than the tow limit (about 8 deg at 2 g, derived from gravity) and at
  least 4 m wide, so rovers can cross and a towed rover can be brought home. The flood fill must reach both
  rims and the floor through the ramps. `features.canyon: {id, path: [[x, y]...], width, depth, ramps:
  [{x, y, heading}]}` (or null).
- **Pits**: 2 to 4 sinkholes or skylights, radius 1.5 to 3.5 m, 4 to 6 m deep, sheer walls. The LiDAR sees
  the ground drop away; the existing hole and drop-off logic and the slope limit keep rovers out; the rules
  camera must not call a pit liquid (dark, not blue). `features.pits: [{id, x, y, radius, depth}]`.
- **Dunes**: 1 to 2 dune fields (15 to 25 m across) of ripples (0.3 to 0.6 m amplitude, 3 to 5 m
  wavelength, aligned with one wind direction) with loose sand physics: lower wheel friction plus mild
  rolling resistance and body drag (milder than the sand hazard), so wheels slip on dune flanks and the
  existing slip detector marks soft ground the fleet then avoids or crosses slowly. `features.dunes: [{id,
  x, y, radius, wind_deg}]`.
- **Ice sheets**: 1 to 3 smooth frozen patches (radius 3 to 7 m) on near-flat ground (local slope under 3
  deg) with very low wheel friction (mu about 0.05). Ice must NOT trip the sand detector: on ice the wheels
  spin freely with low motor load (spin_load about 2.76*mu, about 0.14) where sand makes them work hard. A
  new rule detects low traction: slip above 0.5 while spin_load is low, on near-level ground. Reaction:
  slow down, stop driving on, mark an ice zone (`fleet.ice_zones: [[x, y, r]]`, high cost so the planner
  goes around), log the numbers. Anomaly kind `low_traction` (never a name containing "slip" or "sand":
  the fake Gemma maps those to back_out). Ice is not the existing `ice` target kind; in text call it "ice
  sheet" or "frozen ground". `features.ice: [{id, x, y, radius}]`.
- **Geysers**: 1 to 3 vents on low mounds with a fixed eruption schedule from their stream (period 60 to
  120 sim s, eruption 8 to 15 s, phase offset). During an eruption a plume column (about 1.5 m radius, 6 to
  10 m tall) stands over the vent: LiDAR rays into it are mostly lost and the rest return noisy short ranges
  (steam scatters the beam; plume hits are routed by geom id, a stated simplification like the creature
  tracker), the camera shows the plume (visual decor) and the detector labels it "geyser plume". The rover
  that sees an active plume logs it, slows if close, and adds a keep-out zone (`fleet.geyser_zones:
  [[x, y, r]]`, impassable) so the fleet stays clear for good; when the same rover or a teammate sees a
  second eruption the fleet estimates the period and logs it. Plume hits must never become map obstacles,
  motion tracks or new-rock detections. `features.geysers: [{id, x, y, radius, active, period_s,
  next_s}]`, resent when an eruption starts or ends (`features` is already resent on change and allowed
  in ticks).
- **Boulder fields**: 2 to 4 dense patches of rocks (8 to 25 boulders each, 0.3 to 0.9 m) at mesa feet,
  crater rims or the canyon rim, as solid group-1 geoms; total at 120 m about 80 to 120. They go into
  `truth_nospire` like rocks. `features.boulder_fields: [{id, x, y, radius, count}]`.
- **Ground detail** (look only): pebbles, hoodoos (only on ground that is already impassable: mesa feet,
  canyon walls), rock colour bands on cliffs, crater ejecta streaks, wind streaks. Web rendering only plus
  MuJoCo texture variety; no new MuJoCo geoms. No protocol field: the client derives the placement
  deterministically from `world.seed` and the other features, and only when init `config.ground_detail`
  is true (or `config` is absent and any v7 feature list is present).

### Physics

One composer per array writes runtime model parameters each tick, combining every active cause: per wheel
friction (normal, sand hazard, dune sand, ice), wheel dof damping (sand, dunes, wear), body drag (sand,
dunes), and per rover actuator mode and gains (drive, brake, neutral for towing). Today `Rover._apply` and
`HazardManager.update_physics` each overwrite parts of these; refactor so they no longer clobber each other,
keeping today's values exactly when only today's causes are active.

### Perception

Mesas, canyon walls, pits and boulder fields are seen by the LiDAR through the existing map, slope and
change logic (no ground truth). Make sure: pits are drop-offs, not lakes; canyon ramps are mapped as
passable; boulder fields become blocked cells; plume returns do not enter the map or the tracker; the
`blocked_ahead`, `cost_grid`, frontier, `places._cell_ok` and hazard placement checks (`_check_point`,
`_spot_ok`, `_sand_spot_ok`, `_fall_ok`, web `Placement.jsx` assess) all respect pits, canyon walls,
mesas, geyser and ice zones, so hazards and place goals never land in a pit or on a vent. Gemma: register
`low_traction` and `geyser` anomaly kinds in the advisor tables (RULE_REACTION, ANOMALY_NAME,
QUESTION_OPTIONS, `_anomaly_payload`) and pass a per-kind class subset (`classes=`) so the interpretation
prompt shrinks instead of growing; teach the fake backend the new kinds. Place words: "geyser", "ice
sheet", "dunes", "pit", "canyon", "ramp", "mesa" resolve to the new feature places only from fleet
knowledge (what the fleet has sensed), without breaking the existing parser tests (today "geyser" maps to
vent targets and "sinkhole" and "sand" to hazards: keep those tests passing).

### Web

Render each feature from `features` plus the fleet's knowledge, following the existing patterns (spires
appear when the ground under them is mapped; Reality shows everything): mesas and canyon walls with layered
colour bands and triplanar-style cliff shading in the terrain shader; dune ripples as a normal or colour
modulation in the dune areas; pits dark; ice glossy pale blue; geysers with an animated plume when active
(particles on High graphics, a billboard on Low) and a vent marker; boulder fields with instancing; ground
detail (pebbles, hoodoos, bands, streaks) with instancing on High, reduced on Low. Fleet beliefs
(`ice_zones`, `geyser_zones`) drawn like `soft_zones` and `water_zones`. Legend entries appear only when the
feature exists. Keep draw calls modest (about 12 extra).

### Tests

`tests/test_v7_terrain.py`: all flags off is bit-identical to today (heights, rocks, targets, spires, water
for seeds 42 and 7); each feature on alone generates its features within bounds and keeps every target
reachable and the no-trap property (seeds 42, 7, 123); the canyon has two crossable ramps; pits are never
classified as liquid by the rules; a rover driven onto ice reports `low_traction` and does not create a
soft zone; a rover driven onto dunes slips and the fleet marks soft ground; a geyser eruption seen by a
rover creates a geyser zone and never a map obstacle, track or new-rock event; boulder fields block cells;
geom count and LiDAR scan time stay within budget.

## 4. Alien flora

Flag `flora`. Own stream. Placed after spires, clear of water, targets, herds, spires, the lander zone (at
least 12 m) and the new landforms, on gentle ground, in a few groves (counts at 120 m: about 20 to 30 trees,
30 to 45 bushes; scale with area).

- **Trees**: a stalk 2.5 to 4.5 m tall with a wide domed cap (radius 1.2 to 2.2 m), violet to teal. MuJoCo:
  a group-2 stalk proxy capsule (seen by LiDAR and the rules camera, no bumper hits, solid contact with
  rovers via `contype 1 conaffinity 0`) and the cap and gills as group-4 visual decor (fungal colours,
  emissive look). Proxy colours must stay neutral so `spot_targets` never mistakes them for mineral (violet)
  or trace (lime) markers.
- **Bushes**: 5 to 12 lime bulbs on short stems in a 0.5 to 1.0 m cluster, about 0.4 to 0.7 m tall. MuJoCo: a
  low group-2 proxy (no contact), bulbs as group-4 decor.
- **Planetary protection**: rovers must never touch flora and keep at least `observe_min_dist` (4 m) from
  any flora they have identified. Identification is perception-driven: when the LiDAR maps a tall or new
  static object within about 10 m that the rover has not classified, the rover takes a detector frame
  (section 2, throttled, `_hw_call`) and, if the detector reports `tree` or `bush` with confidence at least
  0.6, the fleet records it (`fleet.flora_zones: [[x, y, r]]`, keep-out radius = plant radius + 4 m,
  impassable for planning and for place goals; `fleet.flora_seen` count). First sighting event:
  "r2 sees possible plant life: fungal tree (camera 0.87, 6.1 m). Keeping 4 m away." Static flora must
  never become motion tracks. A truth-side counter of rover-flora contacts and of closest approach is kept
  for the UI and tests.
- Chat and memory: flora appears in fact sheets and rule answers ("Seen any life?" mentions plants), the
  place words "plant", "tree", "fungus", "bush", "flora", "grove" resolve to known flora (approach goal at
  the keep-out distance; "investigate" surveys from there). Careful with existing parser pins ("glowing
  thing" maps to spire, "look for life" to lifeform).
- `features.flora: [{id, kind: "tree" | "bush", x, y, height, radius, hue}]`.

### Web

Instanced rendering (style B, alien fungal): trees as a tapered stalk, a domed cap curling under, an
emissive gill disc, violet to teal by `hue`, gentle sway on High; bushes as clusters of glowing lime bulbs
with a soft ground halo on High. Shown where the ground under them is mapped (like spires) or with Reality.
Identified flora gets a dashed lime keep-out ring and a "plant life" tag when the Life layer is on.

### Tests

`tests/test_v7_flora.py`: generation bounds and clearances; flora off leaves the world identical; static
flora never becomes a track; a rover that sees a tree identifies it with the detector, the fleet keep-out
appears, the rover keeps 4 m and makes zero flora contacts over a run near a grove; place command "go to
the trees" resolves only after identification; detector boxes label trees and bushes.

## 5. Globe view (web only)

- One Canvas. The plane scene stays mounted but hidden in globe mode; a globe scene with its own camera is
  mounted with r3f `createPortal`, and a render switch (`useFrame(..., 1)`) renders whichever is active.
  Every drei `Html` label in the plane scene is gated on the mode (they do not hide with their parent).
- Planet: a sphere with a procedural shader (3D fbm on the sphere direction seeded by `world.seed`):
  regolith colours from the theme, darker maria, craters, bright polar caps, an atmosphere rim glow, a soft
  day/night terminator with the landing site in daylight under a low sun, slow spin when idle. The landing
  site latitude and longitude come from the seed. A planet radius label derived from gravity (same density as
  Earth: radius = g x 6371 km) is shown with the planet name.
- Landing sector: a glowing marker ring at the site with a small live card: sector size, explored %, rovers
  and their states, gravity, sim time, "not to scale" note. Clicking the marker or toggling flies back down.
- Toggle: a "View: Sector | Planet" control in ViewControls (copy of the Graphics control) and the `G` key.
  Transition: plane camera rises to a top-down pose, a short fade, the globe camera starts close over the
  site and eases out to the home framing; the reverse dives in. `prefers-reduced-motion` switches instantly.
- In globe mode: hazard placement is cancelled; selecting a rover, clicking an event, pressing a hazard
  button or a `choose_place` request switches back to the plane first. Orbit controls (drag to spin, scroll
  to zoom, limits), `0` re-frames home. The Legend swaps to a short globe legend. Graphics Low uses a cheaper
  shader path.
- Debug hook: `window.__redshift.setView("globe" | "plane")` and `state().view` for scripts.

## 6. Wear and tear, breakdown, towing and repair

Flags `wear` (bool) and `wear_rate` (1.0 normal, 4.0 harsh). Default off in the dataclass; `from_env` on at
normal.

- **Truth (world side)**: the sim accumulates each rover's true wear from motor stress (per wheel I^2t from
  actuator force, as the energy bookkeeping already does), distance, hard jolts (a new IMU accelerometer
  sensor, ignoring the first 2 s), slip while working hard, and time on steep tilt, scaled by `wear_rate`.
  Calibrate so a normal-rate rover reaches its limit after roughly 1500 to 2500 m of typical driving, and a
  harsh one about 4 times sooner. Each rover has a hidden failure threshold drawn from its own stream
  (U(0.85, 1.15)), redrawn after each repair. Wear raises copper losses a little (higher energy per metre),
  never the torque cap (so worn rovers do not lower the fleet slope limit).
- **Estimate (rover side)**: the rover estimates its own wear only from its own channels (motor feedback,
  IMU, odometer, its learned energy per metre) and reports it as `wear` (0 to 1). When the estimate plus the
  wear expected on the way home passes a service threshold it returns for service (new return reason
  `service`, like the battery return), logging the numbers. Breakdowns still happen because the threshold
  is hidden and jolts add wear in jumps.
- **Breakdown**: when true wear reaches the threshold the drivetrain fails (hardware flag, like
  `motor_fault`). The rover notices from motor feedback (commanded wheel speed, zero encoder speed, torque
  at the cap), enters new state `broken`, releases its claims (targets, goals), logs the diagnosis with
  numbers, and the fleet requests a tow. Its electronics stay alive (LiDAR keeps mapping, battery drains at
  the base rate). New hazard type `breakdown` (demo trigger like `rover_fault`, `POST /api/hazard
  {"type": "breakdown", "rover": "r2"}`) sets that rover's true wear to its threshold so the same detection
  path runs. The existing `rover_fault` hazard is unchanged (it still reboots after `fault_s`).
- **Rescuer choice (rules)**: the nearest rover that is not parked, not returning on low battery, not
  already towing, and passes: A* to a staging pose about 2 m in front of the casualty on mapped passable
  ground; A* from the casualty home under the tow slope limit `0.8 * asin(4*T/r / ((m_self+m_casualty)*g))`
  (about 8 deg at 2 g; as built in step A, `Config.tow_limit_deg` caps the asin at the 30 deg grip limit of
  a single rover, since a tow train never climbs steeper than one rover alone, so below about 0.73 g the
  limit stays at 24 deg where the plain formula would give up to 37; read `cfg.tow_limit_deg`, never a copy
  of the formula); battery for the trip there plus twice the trip home plus the margin. The
  announcement quotes these numbers. If no rover qualifies, a `tow` request asks mission control (options:
  "Leave it parked", "Try anyway") and the casualty stops counting toward exploration completion so the run
  can still finish.
- **Tow (physics)**: one inactive MuJoCo `<connect>` equality per unordered rover pair (at most 28),
  declared in `world.py` only when `wear` is on (so the default model is unchanged). Hitch: rescuer drives to
  the staging pose, turns to face away from the casualty, reverses slowly until the hitch gap from the
  casualty's shared pose is under 0.15 m; the world checks the true gap, rewrites the constraint anchors so
  violation is zero, and activates it. The casualty's wheels go neutral (gain and bias 0), not braked. The
  rescuer tows at up to 0.4 m/s on a path planned with the tow slope limit, then releases inside 4 m of the
  lander centre. Any teleport (self-righting, physics recovery, out-of-world reset) releases the hitch first
  and sends the rescuer back to hitching. Reversing while hitched is never allowed (a Gemma back_out becomes
  wait). Stalls while towing never call `report_slope_stall`.
- **Repair**: a released casualty within 4 m of the lander enters state `repairing` (parked, braked, also
  charges). Repair time about 60 sim s scaled by wear and slowed by `charge_factor` in a dust storm. Done:
  wear 0, new hidden threshold, drive restored, the breakdown hazard goes inactive, memory notes "towed home
  by r1 and repaired after 340 s", then charging or exploring.
- **New states**: `broken`, `towing`, `repairing` (towing rovers are driving; broken and repairing are
  parked). Rover fields (only when wear is on): `wear` (estimate 0 to 1); `tow: {with, role:
  "towing" | "towed", phase}` while hitched or on the way; `repair: {progress}` while repairing. Hazard type
  `breakdown {id, type, t, rover, active}`. Chat `_busy_reason` texts for the new states; operator commands
  to a broken rover are refused with the reason; an optional chat command "tow r3 home" / "recover rover 3"
  asks the rules to start a tow. Memory and answers mention breakdowns and tows (wear appears in the status
  answer).
- **Web**: rover cards show a wear bar and the new state badges; the 3D view draws a tow bar between hitched
  rovers and a warning marker on a broken rover; the hazard bar gets "Breakdown" (pick a rover, like Motor
  fault); the sensor panel shows wear and the diagnosis.

### Tests

`tests/test_v7_wear.py`: wear off leaves the model and run identical (no equalities, no new sensors in the
default model); wear accumulates monotonically and faster at harsh rate; a forced breakdown is detected from
motor feedback with numbers in the event; the nearest able rover is chosen and the announcement quotes path,
tow slope and battery; hitch activates only when the true gap is small; the pair reaches the lander; repair
resets wear and the rover returns to work; a teleport while hitched releases cleanly; no-tow-possible opens
a `tow` request; the service return happens when the estimate is high; physics stays stable (no glitches,
no flips) during tows on flat ground and ramps up to the tow limit.

## 7. Cross-feature validation (before calling it done)

- Full existing suite green, plus all new test files.
- `tests/test_v7_integration.py`: everything on (from_env-like defaults), seeds 42, 7 and 123, 400 to 600
  sim s each with scripted hazards (rockfall, sand, dust storm, spire fall, lifeform, breakdown): no physics
  glitches, exploration progresses (at least 70% of the all-off baseline explored area at the same time),
  every hazard is triggered and handled, place commands still resolve, zero flora contacts, lifeform
  distance kept, brain p95 under 40 ms, real-time factor above 3x with 3 rovers.
- Web: `npm run build`; a `web/scripts/v7-check.mjs` Playwright run against the real sim (Chromium from
  `~/.cache/ms-playwright`): setup screen appears, change gravity and launch, plane view renders the new
  features, globe toggle both ways, CV overlay draws boxes in the camera viewer, no console errors.
  Screenshots reviewed by eye.
- Docs: `PROTOCOL.md` Version 7 section and env table, `README.md` (judging map, quick start mentions the
  setup screen, env table, demo script additions), `sim/README.md` module table and env table, `sim/GEMMA.md`
  new anomaly kinds, `web/README.md`.

## Out of scope

Real (pixel-only) CV for every class; a Gemma role in tow decisions; changing the 2 g default world when
features are off; Docker GPU rendering.
