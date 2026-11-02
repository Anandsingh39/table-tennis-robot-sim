# RL add-on: self-play shot strategy (CEM / evolution strategies) on top of the MPC rally

Config: config/selfplay_best_effort.yaml (E2 + rail, no spin). 60 points per match, sides alternate, point cap 10 hits (draw). The MPC executes every shot; only the shot choice differs.

| match (X vs Y) | X wins | X losses | draws | win rate of decided points | mean hits/point |
|---|---|---|---|---|---|
| cooperative vs cooperative (baseline) | 0 | 0 | 60 | 0% | 10.0 |
| RL strategy vs cooperative | 2 | 0 | 58 | 100% | 9.7 |
| RL strategy vs random | 7 | 23 | 30 | 23% | 7.6 |
| random vs cooperative | 8 | 0 | 52 | 100% | 9.3 |

Learned action at a neutral observation (ball centred, rails centred): y_c, z_c, d_b = [0.533, 0.8, 0.38] (cooperative: [0, 0.918, 0.42]; ranges [-0.8, 0.8, 0.35] .. [0.8, 1.2, 1.0])

MPC solves per match (both robots):

- cooperative vs cooperative (baseline): MPC solves converged with the exact impact / total: 28679 / 28679 (100.0%); failed solves: best-effort swing executed / best effort not used (exact plan running) / best effort also failed (current plan kept): 0 / 0 / 0
- RL strategy vs cooperative: MPC solves converged with the exact impact / total: 14979 / 27641 (54.2%); failed solves: best-effort swing executed / best effort not used (exact plan running) / best effort also failed (current plan kept): 12662 / 0 / 0; failed-solve IPOPT status: Infeasible_Problem_Detected: 12598, Maximum_Iterations_Exceeded: 64; executed best-effort swings, planned miss median / p95: paddle centre [mm], normal [deg], speed [m/s]: 44.0 / 81.5, 0.00 / 0.00, 0.000 / 0.000
- RL strategy vs random: MPC solves converged with the exact impact / total: 11537 / 22192 (52.0%); failed solves: best-effort swing executed / best effort not used (exact plan running) / best effort also failed (current plan kept): 10644 / 0 / 11; failed-solve IPOPT status: Infeasible_Problem_Detected: 10585, Maximum_Iterations_Exceeded: 70; executed best-effort swings, planned miss median / p95: paddle centre [mm], normal [deg], speed [m/s]: 54.4 / 220.5, 0.00 / 0.00, 0.000 / 0.000
- random vs cooperative: MPC solves converged with the exact impact / total: 25553 / 26180 (97.6%); failed solves: best-effort swing executed / best effort not used (exact plan running) / best effort also failed (current plan kept): 626 / 0 / 1; failed-solve IPOPT status: Infeasible_Problem_Detected: 618, Maximum_Iterations_Exceeded: 8, Restoration_Failed: 1; executed best-effort swings, planned miss median / p95: paddle centre [mm], normal [deg], speed [m/s]: 55.7 / 152.2, 0.00 / 0.00, 0.000 / 0.000
