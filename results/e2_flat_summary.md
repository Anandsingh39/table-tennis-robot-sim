# e2_flat: two-robot rally with flat drives (paddle pre-tilted 80 deg)

Config: config/rally_flat.yaml (two OpenArms, paddles pre-tilted 80 deg, cooperative FLAT DRIVES (5.8-6.5 m/s, apex ~1.27 m), MPC swings, true ball state). Serve: launcher ball to robot A. Legal hit = after exactly one bounce on the receiver's half.

| metric | value |
|---|---|
| rallies | 20 |
| hit cap per rally | 30 |
| rally length (hits): mean / median / min / max | 30.0 / 30 / 30 / 30 |
| rallies reaching the cap | 20 / 20 |
| how rallies ended | max hits: 20 |
| total hits simulated | 600 |
| bounce distance past the net: mean ± std [m] | 0.917 ± 0.142 |
| ball height at impact: mean ± std [m] | 1.052 ± 0.007 |
| lateral hit position y: min / max / std [m] | -0.52 / 0.52 / 0.28 |
| rail travel used: min / max [m] | -0.27 / 0.65 |
| paddle normal vs nominal: median / max [deg] | 4.5 / 15.3 |
| paddle normal speed: median [m/s] | 1.85 |
| MPC solve time: median / p95 / max [ms] | 30 / 48 / 637 |
| peak torque, worst rally [% of limit] per joint | [78 62 68 56 64 86 51 75] |
| rallies with torque saturation | 0 |
| min joint-limit margin [rad] | 0.049 |
| simulated time / wall time per rally (mean) [s] | 21.6 / 84 |
