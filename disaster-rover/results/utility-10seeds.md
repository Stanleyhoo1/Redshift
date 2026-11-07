# Survival: utility brain only (local TS brain), 10 seeds

`npm run bench -- --scenarios <s> --seeds 1-10` · duration 180 s · intensity 0.6 · fixed timing (first disaster at 12 s, then every 22 s).

| scenario | survived | probes placed (avg) | area surveyed (avg %) | hazard coverage by probes (avg %) | probe early warnings (avg) | failures |
|---|---|---|---|---|---|---|
| flood | 10/10 | 3.8 | 24.7 | 23.6 | 8.3 | - |
| wildfire | 5/10 | 3.5 | 24.9 | 11.0 | 12.2 | s2, s5, s6, s10 overheated (trapped by the front); s9 drowned (fled heat down a bank into the river) |
| earthquake | 10/10 | 4.5 | 25.3 | 6.4 | 5.8 | - |
| tornado | 10/10 | 4.8 | 32.0 | 30.7 | 4.2 | - |
| meteor shower | 7/10 | 3.8 | 25.2 | 17.1 | 9.5 | s1 direct meteor hit; s2 drowned; s10 overheated (impact-started fire) |
| all five at once | 7/10 | 2.9 | 26.6 | 10.1 | 8.1 | s1 flipped by tornado; s4 overheated; s8 tornado debris |
| **total** | **49/60** | | | | | |

Raw per-run JSON: `utility-10seeds-a.json`, `utility-10seeds-b.json` (wildfire/meteor rows re-run after the last heat-reflex change; same seeds).
