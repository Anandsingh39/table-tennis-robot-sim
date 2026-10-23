# e2_spin v2 (backspin 50 + reachable-spin gate): two-robot cooperative rally

Config: config/rally_spin.yaml (two OpenArms, pre-tilted paddle mounts, cooperative lob shots, MPC swings, true ball state). Serve: launcher ball to robot A. Legal hit = after exactly one bounce on the receiver's half.

| metric | value |
|---|---|
| rallies | 20 |
| hit cap per rally | 30 |
| rally length (hits): mean / median / min / max | 10.9 / 6 / 1 / 30 |
| rallies reaching the cap | 2 / 20 |
| how rallies ended | max hits: 2, missed (A): 5, missed (B): 13 |
| total hits simulated | 219 |
| bounce distance past the net: mean ± std [m] | 0.726 ± 0.110 |
| ball height at impact: mean ± std [m] | 0.884 ± 0.026 |
| lateral hit position y: min / max / std [m] | -0.49 / 0.55 / 0.28 |
| backspin shots (commanded -50 rad/s): count, achieved mean ± std [rad/s] | 11, -49.6 ± 0.3 |
| flat shots (commanded +0 rad/s): count, achieved mean ± std [rad/s] | 121, -0.2 ± 0.7 |
| topspin shots (commanded +60 rad/s): count, achieved mean ± std [rad/s] | 87, +60.0 ± 0.7 |
| spin shots played flat by the reachable-spin gate | 730 |
| rail travel used: min / max [m] | -0.65 / 0.65 |
| paddle normal vs nominal: median / max [deg] | 14.4 / 19.9 |
| paddle normal speed: median [m/s] | 1.96 |
| MPC solve time: median / p95 / max [ms] | 42 / 424 / 1295 |
| peak torque, worst rally [% of limit] per joint | [ 95  47  44  45  63  82  67 100] |
| rallies with torque saturation | 1 |
| min joint-limit margin [rad] | 0.048 |
| simulated time / wall time per rally (mean) [s] | 21.9 / 3374 |
