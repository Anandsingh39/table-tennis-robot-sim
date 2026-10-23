"""M5 — Ball predictor: when and where the ball next falls through the impact height.

RK4 rollout of the nominal flight model (tt.aero: gravity + drag + Magnus) from the current
ball state until the centre crosses z = z_cross while descending; inside the last step the
motion is taken as constant-acceleration (a = dv/dt over the step) and the quadratic is solved
for the crossing time (exact in vacuum; with drag the error is O(dt^3), far below a micrometre).
Linear interpolation would be off by up to a dt^2 / 8 ~ 30 um (~9 us) at dt = 5 ms.
The predictor uses its own nominal parameters, so model mismatch (Phase 7) is possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tt.aero import AeroParams, rk4_step


@dataclass(frozen=True)
class Prediction:
    t_rel: float          # time from now to the crossing [s]
    pos: np.ndarray       # ball centre at the crossing
    vel: np.ndarray       # ball velocity at the crossing (v_in)


class BallPredictor:
    def __init__(self, params: AeroParams, dt: float, horizon: float):
        self.p, self.dt, self.horizon = params, dt, horizon

    def crossing(self, pos, vel, spin, z_cross: float) -> Prediction | None:
        """Next descending crossing of z_cross within the horizon, or None (e.g. already below and falling)."""
        y = np.concatenate([pos, vel]).astype(float)
        w = np.asarray(spin, float)
        if y[2] <= z_cross and y[5] <= 0:
            return None
        t = 0.0
        while t < self.horizon:
            y_next = rk4_step(y, w, self.p, self.dt)
            if y_next[5] < 0 and y[2] > z_cross >= y_next[2]:
                # constant acceleration over the step: z0 + vz tau + a_z tau^2 / 2 = z_cross
                acc = (y_next[3:] - y[3:]) / self.dt
                a_z, v_z, dz = acc[2], y[5], y[2] - z_cross
                if abs(a_z) > 1e-12:
                    disc = np.sqrt(max(v_z * v_z - 2 * a_z * dz, 0.0))
                    roots = [r for r in ((-v_z + disc) / a_z, (-v_z - disc) / a_z) if r >= 0.0]
                    tau = min(roots) if roots else self.dt
                else:
                    tau = dz / -v_z
                tau = min(max(tau, 0.0), self.dt)
                pos_c = y[:3] + y[3:] * tau + 0.5 * acc * tau ** 2
                return Prediction(t + tau, pos_c, y[3:] + acc * tau)
            y, t = y_next, t + self.dt
        return None
