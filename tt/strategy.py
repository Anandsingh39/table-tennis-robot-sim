"""Self-play shot strategy (RL add-on to the E2 rally): WHAT shot to play; the MPC still decides HOW to swing.

Decision (once per stroke, at the first plan): in the hitter's own frame (x toward the opponent, y to its left;
the far robot's frame is the near robot's rotated 180 deg, so one policy plays from either end = self-play)
    action a = (y_c, z_c, d_b):  where the ball crosses the opponent's hitting plane (lateral y_c, height z_c) and
                                 how far past the net it bounces (d_b); the cooperative law then solves the flight.
Observation o (hitter frame, normalised): incoming impact point (y, z - z_hit), incoming velocity (3),
own rail position, opponent rail position.
Policy: a = lo + (hi - lo) (tanh(W o + b) + 1) / 2  (deterministic; 3 x 8 = 24 parameters).
Training (scripts/selfplay_es.py): cross-entropy method in parameter space with self-play (candidates play
points against the current mean policy), i.e. evolution strategies as used for table tennis by Gao et al. [11].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_OBS = 7
N_ACT = 3


def action_bounds(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    sc = cfg["strategy"]
    lo = np.array([-sc["y_max"], sc["z_range"][0], sc["d_range"][0]])
    hi = np.array([sc["y_max"], sc["z_range"][1], sc["d_range"][1]])
    return lo, hi


def features(pred, sign: float, cfg: dict, ctx: dict) -> np.ndarray:
    """Observation in the hitter's frame (see module docstring), roughly normalised to O(1)."""
    z_hit = cfg["hit"]["center"][2]
    return np.array([
        sign * pred.pos[1] / 0.5,
        (pred.pos[2] - z_hit) / 0.1,
        sign * pred.vel[0] / 4.0,
        sign * pred.vel[1] / 1.0,
        pred.vel[2] / 4.0,
        ctx["own_rail"] / 0.5,
        -ctx["opp_rail"] / 0.5,           # opponent's rail coordinate expressed in the hitter's lateral axis
    ])


@dataclass
class LinearPolicy:
    theta: np.ndarray                    # (N_ACT * (N_OBS + 1),)
    lo: np.ndarray
    hi: np.ndarray

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        W = self.theta.reshape(N_ACT, N_OBS + 1)
        u = np.tanh(W[:, :N_OBS] @ obs + W[:, N_OBS])
        return self.lo + (self.hi - self.lo) * (u + 1.0) / 2.0


def cooperative_theta(cfg: dict) -> np.ndarray:
    """Parameters that reproduce the hand-designed cooperative shot: aim at the opponent's nominal hit zone
    (y = 0, z = its nominal crossing height, the configured default bounce distance), ignoring the observation."""
    lo, hi = action_bounds(cfg)
    sc = cfg["strategy"]
    target = np.array([0.0, sc["coop_z"], sc["coop_d"]])
    u = np.clip(2 * (target - lo) / (hi - lo) - 1, -0.999, 0.999)
    theta = np.zeros((N_ACT, N_OBS + 1))
    theta[:, N_OBS] = np.arctanh(u)
    return theta.ravel()


@dataclass
class RandomPolicy:
    lo: np.ndarray
    hi: np.ndarray
    rng: np.random.Generator

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(self.lo, self.hi)


def make_policy(kind: str, cfg: dict, theta=None, seed: int = 0):
    lo, hi = action_bounds(cfg)
    if kind == "linear":
        return LinearPolicy(np.asarray(theta, float), lo, hi)
    if kind == "cooperative":
        return LinearPolicy(cooperative_theta(cfg), lo, hi)
    if kind == "random":
        return RandomPolicy(lo, hi, np.random.default_rng(seed))
    raise ValueError(kind)


def play_point(cfg: dict, seed: int, pol_a, pol_b, max_hits: int) -> dict:
    """One point (serve = launcher ball to A). Winner = the player who did not fault; 'max hits' = draw."""
    from tt.rally2 import run_rally

    r = run_rally(cfg, seed, max_hits, strategies={"A": pol_a, "B": pol_b})
    if r.fault_by in ("A", "B"):
        winner = "B" if r.fault_by == "A" else "A"
    else:
        winner = None
    return dict(seed=seed, winner=winner, fault=r.fault, by=r.fault_by, hits=len(r.hits),
                fallbacks={k: v for k, v in (r.fallbacks or {}).items()},
                decisions={k: [(o.tolist(), a.tolist()) for o, a in v] for k, v in (r.decisions or {}).items()})
