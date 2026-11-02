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
| bounce distance past the net: mean ± std [m] | 0.903 ± 0.148 |
| ball height at impact: mean ± std [m] | 1.052 ± 0.007 |
| lateral hit position y: min / max / std [m] | -0.52 / 0.52 / 0.28 |
| rail travel used: min / max [m] | -0.23 / 0.65 |
| paddle normal vs nominal: median / max [deg] | 4.6 / 15.7 |
| paddle normal speed: median [m/s] | 1.85 |
| MPC solves converged with the exact impact / total | 12822 / 12858 (99.7%) |
| failed solves: best-effort swing executed / best effort not used (exact plan running) / best effort also failed (current plan kept) | 0 / 5 / 31 |
| failed-solve IPOPT status | Infeasible_Problem_Detected: 32, Restoration_Failed: 4 |
| MPC solve time: median / p95 / max [ms] | 27 / 50 / 1042 |
| peak torque, worst rally [% of limit] per joint | [68 62 64 55 54 86 54 57] |
| rallies with torque saturation | 0 |
| min joint-limit margin [rad] | 0.049 |
| simulated time / wall time per rally (mean) [s] | 21.6 / 84 |
