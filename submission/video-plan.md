# Simulation video plan (60 seconds, one take)

What the brief asks for: up to 1 minute, captured from the stream, perception on screen next to the
actions, and one unscripted change, in a single continuous take.

The change is **soft sand dropped on a rover's path**. It is also the critical failure mode from the
technical summary (how NASA's Spirit was lost), so one event covers both. The rover finds it only
through wheel slip; the sensor panel shows the slip jump right next to the rover backing out.

## Before recording

- Start with a fixed seed so you can rehearse the exact layout:
  `SEED=42 SIM_SPEED=2 OPERATOR_TIMEOUT_S=20 GEMMA_ENABLED=1 docker compose up --build`
  (`SIM_SPEED=2` keeps the slip and back-out readable; `OPERATOR_TIMEOUT_S=20` so no question can
  stall the take waiting for an answer).
- Launch from the setup screen and let the fleet run for about 60 to 90 s before you hit record, so
  the map has some shape and the rovers have spread out. Avoid the first lifeform sighting during
  the take (it asks the operator a question): if it pops up before recording, answer it.
- Gemma pill must read `ready`. Browser full screen, 1920 x 1080, OS notifications off.
- Record with OBS (Display or Window capture), 30 or 60 fps. No cuts, no speed-ups in editing.
  Captions can be burned in afterwards as long as the footage stays one take.
- Rehearse 2 or 3 times: pick a rover heading over open, gentle ground (sand needs gentle ground; the
  bar says why if a spot is rejected).

## Shot list

| Time | On screen | Caption (optional, bottom third) |
|---|---|---|
| 0:00 to 0:06 | Overview of the sector: map filling in, three rovers on their paths, Gemma pill `ready`. | "3 rovers, no map, no link home. Everything they know comes from their own sensors." |
| 0:06 to 0:14 | Click Rover 2 (or press 2). Sensor panel opens: live camera, LiDAR radar, slip, motor load, brain time against the 40 ms budget. Let the camera follow it. | "Live perception: camera, LiDAR, wheel slip, motor load. Brain: about 3 ms of a 40 ms budget." |
| 0:14 to 0:20 | Hazards bar: Soft sand. Ghost marker. Click a few metres ahead on Rover 2's path. | "Unscripted: we hide soft sand on its path. The rover is not told." |
| 0:20 to 0:32 | Rover drives in. Keep the sensor panel in frame: slip climbs past 60% then to about 85% while ground speed drops. Event appears: "r2 ... wheels spinning but not moving (slip 85%). Soft ground". Rover stops. | "Wheels turning, rover not moving: slip 85%. This is how Spirit was lost." |
| 0:32 to 0:44 | Rover backs out along its own ruts. The soft zone appears on the shared map. Its new path bends around the patch. If another rover's path crossed the area, it reroutes too. | "It backs out, marks the zone for the whole fleet, and replans." |
| 0:44 to 0:54 | Click the event in the log: camera flies to the spot. Pull back toward the overview so all three rovers are visible again, still exploring. | "Every decision quotes the reading that caused it." |
| 0:54 to 1:00 | Hold on the overview with the event log in frame. | "Redshift. Local Gemma on the laptop GPU, rules in control, no cloud." |

## Fallbacks during the take

- If the sand spot is rejected, the hazard bar says why; click a flatter spot. Still one take.
- If the rover misses the patch, right-click instead: the sim then chooses a spot on that rover's
  route. Rehearse this first so it is not needed live.
- If a question card appears (rockfall or lifeform from earlier), answer it in one click and carry on.

## Things to check before the final take

- [ ] Brain time in the sensor panel matches what the summary says.
- [ ] The slip value in the event and in the gauge are both visible at the same time.
- [ ] Captions do not cover the sensor panel or the event log.
- [ ] Video is 60 s or shorter.
