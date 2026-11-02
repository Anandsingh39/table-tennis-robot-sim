# e2_rail: two-robot cooperative rally

Config: config/rally_rail.yaml (two OpenArms, pre-tilted paddle mounts, cooperative lob shots, MPC swings, true ball state). Serve: launcher ball to robot A. Legal hit = after exactly one bounce on the receiver's half.

| metric | value |
|---|---|
| rallies | 20 |
| hit cap per rally | 30 |
| rally length (hits): mean / median / min / max | 30.0 / 30 / 30 / 30 |
| rallies reaching the cap | 20 / 20 |
| how rallies ended | max hits: 20 |
| total hits simulated | 600 |
| bounce distance past the net: mean ± std [m] | 0.410 ± 0.022 |
| ball height at impact: mean ± std [m] | 0.918 ± 0.008 |
| lateral hit position y: min / max / std [m] | -0.50 / 0.50 / 0.28 |
| rail travel used: min / max [m] | -0.45 / 0.58 |
| paddle normal vs nominal: median / max [deg] | 3.0 / 18.5 |
| paddle normal speed: median [m/s] | 1.24 |
| MPC solves converged with the exact impact / total | 30528 / 30528 (100.0%) |
| failed solves: best-effort swing executed / best effort not used (exact plan running) / best effort also failed (current plan kept) | 0 / 0 / 0 |
| MPC solve time: median / p95 / max [ms] | 21 / 30 / 144 |
| peak torque, worst rally [% of limit] per joint | [75 27 29 25 32 10 52 15] |
| rallies with torque saturation | 0 |
| min joint-limit margin [rad] | 0.050 |
| simulated time / wall time per rally (mean) [s] | 40.3 / 212 |
