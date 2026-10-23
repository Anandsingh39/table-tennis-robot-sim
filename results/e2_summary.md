# E2: two-robot cooperative rally

Config: config/rally2.yaml (two OpenArms, pre-tilted paddle mounts, cooperative lob shots, MPC swings, true ball state). Serve: launcher ball to robot A. Legal hit = after exactly one bounce on the receiver's half.

| metric | value |
|---|---|
| rallies | 20 |
| hit cap per rally | 30 |
| rally length (hits): mean / median / min / max | 30.0 / 30 / 30 / 30 |
| rallies reaching the cap | 20 / 20 |
| how rallies ended | max hits: 20 |
| total hits simulated | 600 |
| bounce distance past the net: mean ± std [m] | 0.412 ± 0.023 |
| ball height at impact: mean ± std [m] | 0.922 ± 0.007 |
| paddle normal vs nominal: median / max [deg] | 1.9 / 17.9 |
| paddle normal speed: median [m/s] | 1.20 |
| MPC solve time: median / p95 / max [ms] | 18 / 22 / 97 |
| peak torque, worst rally [% of limit] per joint | [31  5  9 39 10 57  3] |
| rallies with torque saturation | 0 |
| min joint-limit margin [rad] | 0.104 |
| simulated time / wall time per rally (mean) [s] | 40.8 / 88 |
