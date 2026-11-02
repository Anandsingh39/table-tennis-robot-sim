# E1: single-arm return with MPC swing

Config: config/rally.yaml (pre-tilted paddle mount, hit zone [0.4, 0.0, 0.9]); true ball state; launcher = stand-in opponent (config launcher:). Target: [3.048 0.   ] on the opponent half.

| metric | value |
|---|---|
| balls | 100 |
| outcomes | in: 100 |
| hit rate (paddle contact) | 100% |
| return rate (lands on opponent half) | 100% |
| landing error vs target: mean / RMS / p95 / max [mm] | 5.4 / 6.8 / 10.7 / 23.5 |
| impact: paddle centre vs plan, RMS [mm] | 1.13 |
| impact: normal error, RMS [deg] | 0.114 |
| impact: time vs plan, RMS [ms] | 0.04 |
| outgoing ball velocity vs plan, RMS [mm/s] | 20.2 |
| shot: flight time T [s] (min / median / max) | 0.7 / 1.0 / 1.1 |
| shot: paddle tilt [deg], normal speed [m/s] (median) | 53, 1.57 |
| ball height over the net, median [m] | 2.04 |
| incoming ball launch speed, median [m/s] | 6.77 |
| MPC solves per ball (mean) | 18.9 (0 failed in total) |
| MPC solves converged with the exact impact / total | 1890 / 1890 (100.0%) |
| failed solves: best-effort swing executed / best effort not used (exact plan running) / best effort also failed (current plan kept) | 0 / 0 / 0 |
| MPC solve time: median / p95 / max [ms] | 19 / 56 / 114 |
| peak torque, worst ball [% of limit] per joint | [38  8 15 43 22 60  5] |
| balls with torque saturation | 0 |
| max joint speed, worst ball [% of rated] | [44  7 18 34 14 31  1] |
| min joint-limit margin over all balls [rad] | 0.050 |
