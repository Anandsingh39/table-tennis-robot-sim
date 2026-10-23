# E1: single-arm return with MPC swing

Config: config/rally.yaml (pre-tilted paddle mount, hit zone [0.4, 0.0, 0.9]); true ball state; launcher = stand-in opponent (config launcher:). Target: [3.048 0.   ] on the opponent half.

| metric | value |
|---|---|
| balls | 100 |
| outcomes | in: 100 |
| hit rate (paddle contact) | 100% |
| return rate (lands on opponent half) | 100% |
| landing error vs target: mean / RMS / p95 / max [mm] | 5.3 / 6.8 / 11.4 / 22.7 |
| impact: paddle centre vs plan, RMS [mm] | 1.13 |
| impact: normal error, RMS [deg] | 0.113 |
| impact: time vs plan, RMS [ms] | 0.04 |
| outgoing ball velocity vs plan, RMS [mm/s] | 20.0 |
| shot: flight time T [s] (min / median / max) | 0.6 / 1.0 / 1.1 |
| shot: paddle tilt [deg], normal speed [m/s] (median) | 53, 1.57 |
| ball height over the net, median [m] | 2.04 |
| incoming ball launch speed, median [m/s] | 6.71 |
| MPC solves per ball (mean) | 18.9 (0 failed in total) |
| MPC solve time: median / p95 / max [ms] | 19 / 53 / 108 |
| peak torque, worst ball [% of limit] per joint | [38  9 16 46 25 61  6] |
| balls with torque saturation | 0 |
| max joint speed, worst ball [% of rated] | [46  8 19 33 16 31  2] |
| min joint-limit margin over all balls [rad] | 0.050 |
