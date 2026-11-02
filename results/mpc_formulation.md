# MPC swing formulation: alignment with Nguyen et al. 2025 [4] (November 2026)

Code: `tt/mpc.py`; settings: `config/rally.yaml` (`mpc:`); paper notes: `config/literature.yaml` (`nguyen2025`).

## What [4] does (arXiv:2505.01617, Sec. IV, eq. 3, Table I)

- Cost: Σ_n (w_a ‖q̈[n]‖² + w_v ‖q̇[n]‖²) (eq. 3a); the weights are not published.
- Euler double integrator in joint space, joint limits.
- Terminal conditions as tolerances: ‖FK(q_f) − p_des‖² ≤ ε_p, ‖J q̇_f − v_des‖² ≤ ε_v, orientation ‖·‖² ≤ ε_o (eq. 3g–3i); ε not published.
- Two variants: shrinking horizon (SHMPC: current state, Δt shrinks) and fixed horizon (FHMPC: ready state, constant Δt). Table I (warm start, i9-14900K): FHMPC 99.6% / 2.8 ms and 99.5% / 3.2 ms; SHMPC 89.6% / 6.2 ms and 91.2% / 6.7 ms; FHMPC was chosen.

## What changed in our MPC

| | before | now |
|---|---|---|
| Cost | Σ h ‖q̈‖² + w_end ‖q_N − q_ready‖² | Σ h (w_a ‖q̈‖² + w_v ‖q̇‖²) + w_end ‖q_N − q_ready‖², w_a = 1, w_v = 5 ([4] eq. 3a as a time integral) |
| Impact conditions | exact | exact (default) or [4]'s tolerance form (`impact_constraints: tolerance`) |
| Ball out of reach | solve fails, the arm keeps its plan or does not move | best-effort fallback: the impact conditions as L1 penalties (slack variables), the arm swings to the closest reachable paddle state (our addition) |
| Reporting | solve times | convergence rate, failure status, best-effort use and planned miss (as Table I of [4]) |

Unchanged: shrinking horizon (= [4]'s SHMPC), exact constant-acceleration integrator, joint position/speed/acceleration limits,
follow-through to rest, warm start, 50 Hz re-solve with a 30 ms freeze. Setting `w_vel: 0` and `best_effort: false` gives the
previous MPC; `config/rally_spin.yaml` and `config/selfplay.yaml` keep those settings so their published results stay reproducible.

## Checks

1. **Old behaviour reproduced.** With the old settings the new code returns bit-identical plans (19 solves incl. warm starts
   and an infeasible target; max difference 0.0), and E1 seeds 0–7 land exactly where the committed results did.
2. **Penalty weights from the multipliers.** Finite differences of the optimal cost on 75 recorded swings (E1, rail, flat):
   |λ| ≤ 801 per m (position), 411 (normal), 272 per m/s (speed). The L1 weights are 10⁴, more than 12× larger.
3. **Fallback is exact when an exact swing exists.** On 360 recorded feasible swings the soft problem met the impact with
   residuals below 0.0001 mm, 0.00001° and 0.0002 mm/s and matched the exact joint path (≤ 5×10⁻⁴ rad), except one flat-rally swing where it found a
   different swing that also hits exactly (the problem is non-convex).
4. **Tolerance form ([4] eq. 3g–3i), 300 recorded swings, tolerances 1 mm / 0.05° / 2 mm/s.** Written as ‖r‖²/ε² ≤ 1 (unscaled,
   IPOPT's feasibility tolerance loosened ε by 0.5%). Result: no better convergence than exact constraints (E1 99% vs 100%,
   rail 92% vs 100%, flat 100% vs 100%), 2–3× slower (median 35–44 ms vs 11–20 ms, p95 up to 449 ms; 18 processes in
   parallel, so relative), and the optimum sits on the boundary: the speed error equals ε in every solve, position and normal in
   most (median error = ε). That is a systematic error, so exact constraints stay the default.
5. **Velocity weight sweep, same swings.** w_v = 0 / 1 / 5 / 20: flat-rally mean joint speed 1.065 / 1.053 / 1.008 / 0.909 rad/s;
   rail-rally peak acceleration 0.85 / 0.86 / 0.90 / 1.00 of the limit. w_v = 5 lowers speeds without reaching the limit.
6. **Tests:** `tests/test_rally.py` (tolerance mode inside ε, soft problem exact when feasible, best effort for an unreachable
   ball within all limits, fallback off keeps the failure, velocity cost lowers joint speeds, adoption rule, summary rows).

## Same serves, old vs new MPC

| | old MPC | new MPC |
|---|---|---|
| E1 (100 balls) | 100/100, landing RMS 6.7 mm, 1890/1890 converged | 100/100, landing RMS 6.8 mm, 1890/1890 converged |
| Lob rally (20) | 20/20 at the cap, 9714/9714 converged, peak torque 57% | 20/20, 9715/9715, 56% |
| Rail rally (20) | 20/20, 30523/30530 converged, peak 77% | 20/20, 30528/30528, 75% |
| Flat rally (20) | 20/20, 12724/12756 (99.7%), peak per joint [78 62 68 56 64 86 51 75]% | 20/20, 12822/12858 (99.7%), [68 62 64 55 54 86 54 57]% |

Overall the new MPC converged in 54,955 of 54,991 solves (99.93%). The best-effort swing was never executed in these
cooperative runs; in the flat rally 5 failed solves had a best-effort swing that was not used (an exact plan was running) and
31 failed in both problems while a plan was running. No torque saturation anywhere.

## Launcher fix found on the way

E1 seed 61: the launcher's unbounded least-squares diverged to a 1410 m/s serve (the ball moves metres per predictor step, so
the event detection was meaningless and the residual passed). Launches faster than `launcher.max_speed` = 15 m/s are now
rejected; all other launches over 240 scanned seeds are ≤ 9.9 m/s and unchanged. The committed E1 results predate the launcher
change that exposed this (spin commit 1d0b3a2); the old/new comparison above uses the fixed launcher for both.

## Self-play with the best-effort receiver

`config/selfplay_best_effort.yaml` = `config/selfplay.yaml` + `mpc.best_effort: true` (same learned policy `results/rl/theta.npy`,
same 60 evaluation seeds per match). Results: `results/rl_best_effort_summary.md`.

| match (60 points) | receiver stands still (rl_summary.md) | best-effort receiver |
|---|---|---|
| learned vs cooperative | 60 – 0 | 2 – 0, 58 draws |
| learned vs random | 60 – 0 | 7 – 23, 30 draws |
| random vs cooperative | 19 – 0, 41 draws | 8 – 0, 52 draws |
| cooperative vs cooperative | 60 draws | 60 draws |

Against the learned strategy only 54% of solves meet the impact exactly (the shots are aimed out of reach); the executed
best-effort swings meet the paddle normal and speed exactly and miss the planned paddle centre by 44 mm median, 82 mm p95
(blade radius 75 mm), so most still meet the ball on the blade. The learned strategy's 60–0 was an exploit of a receiver that
stood still; retraining against the best-effort receiver is the next step.
