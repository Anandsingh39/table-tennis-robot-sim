# How this project relates to the references on the task sheet

| Ref | What they did | How our project relates |
|---|---|---|
| [9] Büchler et al. 2022 (T-RO) | Model-free RL on a pneumatic-muscle arm; hybrid sim/real training; reward = landing-point distance. 96% hit, 77% return on 107 real balls. | Same task objective (send the ball to a desired landing point), but we use model-based control (prediction + rebound model + MPC), not RL. Our E1 return rate 100% (sim, exact ball state) vs their 77% (real robot). |
| [10] Ji et al. 2021 (IROS) | Model-based trajectory prediction; quintic-polynomial joint trajectories; **hitting velocity control** (choose the racket velocity for the desired return); >95% return for 3-7 m/s balls. | Closest to our juggling law: our M6 is a hitting-velocity law (paddle normal + speed from the desired outgoing ball velocity through the rebound model), our M7 uses quintic joint trajectories, M5 is model-based prediction. |
| [11] Gao et al. 2020 (IROS) | Simulated 6-DOF arm **+ 2 linear axes**; model-free ES/PPO policies; 80% return of random throws, no spin. | Their robot also uses linear axes, which supports our sideways rail. We solve the same simulated return task with MPC instead of RL. |
| [4] Nguyen et al. 2025 (ICRA, MPC) | Joint-space OCP (CasADi/IPOPT), terminal paddle position/velocity/orientation, MPC re-solved during the swing; 87.5-89.2% hit rate for loop/chop/drive. | Our extra-credit swing replicates this formulation (min joint-acceleration OCP, terminal paddle pose and velocity via FK, CasADi/IPOPT, re-solve until impact) on OpenArm, with topspin/backspin shots like their loop/chop. Our median warm solve ~20 ms (Python, desktop CPU) vs their ~3 ms. |
| [2],[3] Nakashima et al. | Paddle juggling with a 5-DOF manipulator, 120 Hz cameras; mean hitting error 1.4/3.6 cm over 100 s. | Juggling replication (results/replication_table.md). |
| [7] Ji et al. 2024 | Juggling robot (3-DOF parallel), 250 FPS vision, PD tilt control, each juggle within 10% of the set height. | Juggling replication; our PD-tilt comparison law follows their controller. |

Notes: the task sheet lists [9]-[11] for "a heuristic control law"; [9] and [11] are RL papers and [10] is model-based. Our juggling controller is a heuristic (closed-form) hitting-velocity law in the spirit of [10], with the juggling targets of [2],[3],[7].
