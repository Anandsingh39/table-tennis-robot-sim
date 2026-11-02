"""E1 ball launcher: the stand-in opponent for single-arm return tests.

Each ball is launched from behind the far end of the table and must (i) bounce on the robot's
half at x = x_bounce and (ii) cross the robot's hitting plane x = x_plane at (y_c, z_c). The three
targets are sampled (seeded) from the config ranges; the launch velocity (3 unknowns) is found by
least squares on predictor rollouts with the table bounce (3 equations). Samples whose solve does
not converge are redrawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from tt.predictor import BallPredictor, TableBounce


@dataclass
class Launch:
    pos: np.ndarray
    vel: np.ndarray
    x_bounce: float
    cross_yz: np.ndarray      # where it was aimed at on the hitting plane


class Launcher:
    def __init__(self, cfg: dict, predictor: BallPredictor, table: TableBounce, x_plane: float):
        self.lc = cfg["launcher"]
        self.pred, self.table, self.x_plane = predictor, table, x_plane

    def _events(self, pos, vel):
        """(x at the first table bounce, (y, z) where the ball then crosses the hitting plane), or None.

        Stops as soon as the outcome is known (crossing, or crossing / falling without a bounce)."""
        y = np.concatenate([pos, vel]).astype(float)
        w, x_b, t = np.zeros(3), None, 0.0
        while t < self.lc["max_flight"]:
            y_next, contact, w = self.pred._step_with_table(y, w, self.table)
            if contact is not None:
                if x_b is not None:
                    return None                     # second bounce before the hitting plane: not a valid serve
                x_b = contact[0]                    # exact bounce x (smooth in the launch velocity)
            if y[0] > self.x_plane >= y_next[0]:
                if x_b is None:
                    return None
                a = (y[0] - self.x_plane) / (y[0] - y_next[0])
                return x_b, y[1:3] + a * (y_next[1:3] - y[1:3])
            if y_next[2] < self.table.z_contact - 0.05 and x_b is not None and y_next[5] < 0:
                return None                         # fell below the table top after the bounce
            if y_next[2] < self.table.z_contact - 0.05 and x_b is None:
                return None                         # missed the table
            y, t = y_next, t + self.pred.dt
        return None

    def _residual(self, vel, pos, x_b, y_c, z_c):
        ev = self._events(pos, vel)
        if ev is None:
            return np.array([10.0, 10.0, 10.0])
        xb, yz = ev
        return np.array([xb - x_b, yz[0] - y_c, yz[1] - z_c])

    def _initial_guesses(self, pos, x_b, y_c, z_c) -> list[np.ndarray]:
        """Coarse grid over (v_x, v_z): the residual is flat where the ball misses the table, so the
        local solver needs a start inside the feasible region. Feasible launches come in a fast/flat
        and a slow/lob family; candidates no faster than the slowest + speed_slack are kept (cooperative
        ball) and returned best-residual first."""
        lc = self.lc
        cands = []
        for vx in np.arange(*lc["grid_vx"]):
            for vz in np.arange(*lc["grid_vz"]):
                T = (self.x_plane - pos[0]) / vx
                v = np.array([vx, (y_c - pos[1]) / T, vz])
                r = float(np.linalg.norm(self._residual(v, pos, x_b, y_c, z_c)))
                if r < lc["grid_accept"]:
                    cands.append((r, float(np.linalg.norm(v)), v))
        if not cands:
            return []
        v_min = min(c[1] for c in cands)
        return [c[2] for c in sorted(c for c in cands if c[1] <= v_min + lc["speed_slack"])]

    def sample(self, rng: np.random.Generator) -> Launch:
        lc = self.lc
        for _ in range(lc["max_tries"]):
            pos = np.array([lc["x"], rng.uniform(*lc["y_range"]), lc["z"]])
            x_b = rng.uniform(*lc["bounce_x_range"])
            y_c = rng.uniform(*lc["cross_y_range"])
            z_c = rng.uniform(*lc["cross_z_range"])
            for v0 in self._initial_guesses(pos, x_b, y_c, z_c)[:lc["starts"]]:
                sol = least_squares(self._residual, v0, args=(pos, x_b, y_c, z_c), diff_step=lc["diff_step"])
                # the solve is unbounded: a diverged start can "converge" at absurd speeds, where a ball moves metres
                # per predictor step and the event detection is meaningless -> reject and try the next start
                if np.linalg.norm(sol.fun) < lc["tol"] and np.linalg.norm(sol.x) <= lc["max_speed"]:
                    return Launch(pos, sol.x, x_b, np.array([y_c, z_c]))
        raise RuntimeError("launcher: no feasible launch found")
