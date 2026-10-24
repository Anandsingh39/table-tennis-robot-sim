# RL add-on: self-play shot strategy (CEM / evolution strategies) on top of the MPC rally

Config: config/selfplay.yaml (E2 + rail, no spin). 60 points per match, sides alternate, point cap 10 hits (draw). The MPC executes every shot; only the shot choice differs.

| match (X vs Y) | X wins | X losses | draws | win rate of decided points | mean hits/point |
|---|---|---|---|---|---|
| cooperative vs cooperative (baseline) | 0 | 0 | 60 | 0% | 10.0 |
| RL strategy vs cooperative | 60 | 0 | 0 | 100% | 1.5 |
| RL strategy vs random | 60 | 0 | 0 | 100% | 1.6 |
| random vs cooperative | 19 | 0 | 41 | 100% | 8.4 |

Learned action at a neutral observation (ball centred, rails centred): y_c, z_c, d_b = [0.533, 0.8, 0.38] (cooperative: [0, 0.918, 0.42]; ranges [-0.8, 0.8, 0.35] .. [0.8, 1.2, 1.0])
