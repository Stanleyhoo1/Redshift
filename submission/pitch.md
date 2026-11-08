# Pitch (5 minutes, live)

The brief: what problem the robot solves, how it works, what it can and cannot do, and why it matters
to its user. Judged on perception-driven action (25%), adaptation to change (25%), relevance and
impact (20%), operational autonomy (15%), ambition and quality (15%).

Plan: about 1 minute of story, 3 minutes of live demo with the judges choosing the change, 1 minute
on limits and why it matters. Keep the dashboard on screen the whole time; the demo is the argument.

Setup before you walk on: sim running for a couple of minutes (`SEED=42 SIM_SPEED=3
OPERATOR_TIMEOUT_S=20`), Gemma pill `ready`, Rover 2's sensor panel ready to open. Have the
1-page summary printed for the judges.

---

## 0:00 to 1:00 The problem

> In 2009 NASA's Spirit rover drove into soft soil it could not see. Its wheels dug in, and it never
> moved again. Help was 20 minutes away by radio and it still was not enough.
>
> Now picture a planet around another star. A message home takes years. Twice Earth's gravity, so
> every slope is harder. No map. The rover has to notice trouble, back off and decide what to do on
> its own, using only what its sensors tell it.
>
> Redshift is that rover team: three rovers that land with nothing but their sensors and explore a
> super-Earth by themselves. Our user is the mission team that cannot drive them.

## 1:00 to 1:40 How it works (point at the screen)

> Each rover has a LiDAR, a camera, an IMU, wheel encoders and motor feedback. That is all it knows.
> [Open Rover 2's sensor panel.]
>
> Two brains. A fast rule brain runs five times a second inside a 40 millisecond budget: it builds
> the shared map, picks frontiers, plans paths and handles every safety reaction. Here it is using
> about 3 milliseconds. Beside it, Gemma runs locally on this laptop's GPU and gives a second opinion:
> which goal is worth it, what the camera is seeing, what the operator meant. Gemma only ever picks
> from options the rules already checked. Pull the network cable and nothing changes.

## 1:40 to 4:00 Live demo: you choose the change

> You pick what goes wrong.

Offer the hazards bar to a judge (or ask them to name one). Narrate whichever they choose:

- **Soft sand** (lead with this if they have no preference: it is the Spirit failure).
  "Watch slip in the panel. Wheels turning, rover not moving, slip 85%. It stops, backs out along its
  own tracks, and marks the patch so the whole fleet avoids it."
- **Rockfall** ahead of a rover. "LiDAR sees ground that was flat is now 1 m higher in three sweeps
  in a row. It slows, Gemma looks at the camera, and the rover asks us. If nobody answers it takes
  the safe option." Answer the card.
- **Falling spire.** "One long, narrow strip of changed cells, so it calls it a long low obstacle,
  not a rock pile, and routes around it."
- **Dust storm.** "LiDAR range collapses from 8 to 3 m on the radar. Rovers slow down, stay on known
  ground and raise their battery reserve because the lander's panels are dusty."
- **Lifeform.** "Points that move between scans become a track. It stops, asks whether to follow,
  and never gets closer than 4 m: planetary protection."

Then, whatever happened, close the demo with two quick beats:

1. Click the event in the log: "Every event quotes the reading behind it, and clicking it flies to
   where it happened."
2. In Comms, type `everyone come home`. "Gemma turns typed instructions into commands. The rules
   check them and each rover answers, or refuses with its reasons."

If there is time: turn on **Reality** to compare what the fleet believes with the real planet,
including lakes it has not reached yet.

## 4:00 to 4:40 What it cannot do (say it plainly, judges reward honesty)

> It is a simulation. Positions come from the simulator, not from SLAM. The rovers share one map
> instantly, where real radio links would be patchy. The liquid check is a colour test, and a small
> model can misread a scene, which is exactly why Gemma only advises and the rules decide. Wear and
> towing, and a camera object detector, are built but not in today's build.

## 4:40 to 5:00 Why it matters

> Every rover we send further out has less help from home. The thing that matters most is not
> clever driving, it is noticing you are in trouble and getting out before it is too late, and
> being able to say why. Redshift does that on a laptop, with no cloud and no one to phone.
> Thank you.

---

## Likely judge questions (short answers)

- **Is anything scripted?** No paths, no timeline. You placed the hazard; the rover found it with its
  own sensors. The hazard list exists only for the dashboard; rover code never reads it.
- **What runs where?** Rule brain on the CPU, Gemma 4 E2B on the laptop GPU through Ollama, physics in
  MuJoCo on the same machine. Budgets 40 ms and 3 s, measured live in the sensor panel and the Gemma pill.
- **What if Gemma is wrong or slow?** Its answer is checked against the map, slope limit and battery;
  if it fails a check or takes over 3 s, the rules decide. The rovers run fine with Gemma off.
- **Why 2 g?** Harder climbs make the slope learning matter: the fleet starts believing 30 deg and
  learns about 20 deg from a real stall. Gravity is a launch setting from 0.5 to 3 g.
- **Why not Isaac Sim?** MuJoCo gives us fast, deterministic contact physics on a laptop CPU, which
  lets three rovers and their sensors run several times faster than real time with the GPU free for
  Gemma.
- **What would you do next?** SLAM instead of simulator position, patchy inter-rover radio, and merge
  the wear, towing and camera detector branches.
