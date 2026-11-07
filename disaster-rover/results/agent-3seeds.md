# Survival: utility brain vs utility brain + Gemma 4 commander (seeds 1-3)

Local TS brain, 180 s, intensity 0.6. Agent runs are latency-faithful: sim time never runs ahead of wall time while a Gemma request is in flight, and runs are sequential (one Ollama, CPU only).

| scenario | utility only | utility + Gemma 4 | agent calls | accepted / rejected by rule set | failures with agent |
|---|---|---|---|---|---|
| flood | 3/3 | 3/3 | 33 | 23 / 9 | - |
| wildfire | 2/3 | 2/3 | 42 | 28 / 11 | s2 overheated |
| earthquake | 3/3 | 3/3 | 40 | 29 / 10 | - |
| tornado | 3/3 | 3/3 | 33 | 22 / 10 | - |
| meteor shower | 1/3 | 1/3 | 23 | 12 / 10 | s1 meteor hit; s3 overheated |
| all five | 2/3 | 3/3 | 42 | 31 / 10 | - |
| **total** | **14/18** | **15/18** | **213** | **145 / 60** (8 still in flight at run end) | |

- 0 agent errors / unparsable outputs. Per-run median latency 4.8-9.2 s (median of medians 6.0 s).
- Rejections by rule: probe_spacing 28, predicted_hazard 15, target_in_water 10, out_of_bounds 3, no_probes_left 2, unreachable 2.
- Caveats: 3 seeds is a small sample (one extra survival is not significant). The run was paused midway for UI work and resumed; the seed 1 and part of seed 2 runs predate the last heat-escape tweak (which made no measured difference on fire). The agent mainly changes WHERE probes go and what gets surveyed; survival is owned by the utility brain and reflexes by design.
