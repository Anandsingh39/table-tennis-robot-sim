# e2_spin v1 (backspin 80, no reachable-spin gate): two-robot cooperative rally

Config: config/rally_spin.yaml (two OpenArms, pre-tilted paddle mounts, cooperative lob shots, MPC swings, true ball state). Serve: launcher ball to robot A. Legal hit = after exactly one bounce on the receiver's half.

| metric | value |
|---|---|
| rallies | 20 |
| hit cap per rally | 30 |
| rally length (hits): mean / median / min / max | 23.8 / 30 / 2 / 30 |
| rallies reaching the cap | 11 / 20 |
| how rallies ended | double bounce (B): 2, max hits: 11, missed (A): 2, missed (B): 5 |
| total hits simulated | 475 |
| bounce distance past the net: mean ± std [m] | 0.706 ± 0.129 |
| ball height at impact: mean ± std [m] | 0.890 ± 0.037 |
| lateral hit position y: min / max / std [m] | -0.51 / 0.49 / 0.26 |
| backspin shots (commanded -80 rad/s): count, achieved mean ± std [rad/s] | 50, -72.7 ± 14.7 |
| flat shots (commanded +0 rad/s): count, achieved mean ± std [rad/s] | 170, -0.1 ± 0.4 |
| topspin shots (commanded +60 rad/s): count, achieved mean ± std [rad/s] | 255, +59.9 ± 0.8 |
| rail travel used: min / max [m] | -0.65 / 0.65 |
| paddle normal vs nominal: median / max [deg] | 11.3 / 41.3 |
| paddle normal speed: median [m/s] | 1.95 |
| MPC solve time: median / p95 / max [ms] | 28 / 61 / 1248 |
| peak torque, worst rally [% of limit] per joint | [ 96  72 100  51 100 100 100 100] |
| rallies with torque saturation | 15 |
| min joint-limit margin [rad] | 0.042 |
| simulated time / wall time per rally (mean) [s] | 40.8 / 1517 |
