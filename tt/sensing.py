"""M4 — Sensing and estimation.

Sensor (the "clearly specified measurement noise"):
  * ball centre POSITION sampled at `rate` Hz: z_k = p(t_k) + n_k,  n_k ~ N(0, sigma^2 I3), i.i.d.;
  * each sample is delivered `latency` s after it was taken; each is lost with probability
    `dropout_prob`; all randomness from a seeded generator. Velocity is NOT measured.

Estimator: extended Kalman filter on x = [p, v] with the nominal flight model (gravity + drag):
  predict  x <- RK4 flight over dt,  P <- F P F^T + Q
           F = [[I, I dt], [0, I + J_a dt]],  J_a = d(-k|v|v)/dv = -k (|v| I + v v^T / |v|)
           Q = q^2 [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]  (white acceleration noise, unmodelled aero)
  update   H = [I 0], R = sigma^2 I  (standard KF equations; Joseph form for P)
The filter state lives at the time of the latest measurement it has processed; state_at(t)
propagates it (mean only) to the current time, which is how latency is handled.

Bounce handling (no access to the simulator's contact flag): a paddle impact changes the velocity
by ~5 m/s upward, so the first measurements after it land far ABOVE the prediction. If the
normalised innovation squared  NIS = y^T S^-1 y  exceeds a chi^2(3) threshold, the innovation
points up (y_z > 0) and the ball is near the impact height, the filter declares a bounce and
re-initialises: p = measurement, v = the velocity the juggler commanded (if known, else 0),
P = diag(sigma^2, reset_vel_std^2). Detection takes 1–2 samples (<= 10 ms at 200 Hz).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tt.aero import AeroParams, rk4_step


@dataclass
class Measurement:
    t: float               # time the sample was taken
    z: np.ndarray          # measured position
    t_deliver: float


class BallSensor:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        sc = cfg["sensing"]
        self.period = 1.0 / sc["rate"]
        self.sigma, self.latency, self.dropout = sc["pos_noise_std"], sc["latency"], sc["dropout_prob"]
        self.rng = rng
        self.t_next = 0.0
        self.queue: deque[Measurement] = deque()

    def sample(self, t: float, p_true: np.ndarray) -> Measurement | None:
        """Call every physics step with the true position; returns the new sample (if taken now)."""
        if t + 1e-9 < self.t_next:
            return None
        self.t_next += self.period
        noise = self.rng.normal(0.0, self.sigma, 3)       # drawn even if dropped: dropouts don't shift the stream
        if self.rng.random() < self.dropout:
            return None
        m = Measurement(t, p_true + noise, t + self.latency)
        self.queue.append(m)
        return m

    def delivered(self, t: float) -> list[Measurement]:
        out = []
        while self.queue and self.queue[0].t_deliver <= t + 1e-9:
            out.append(self.queue.popleft())
        return out


class BallEKF:
    def __init__(self, cfg: dict, params: AeroParams, z_impact: float,
                 expected_v_out: Callable[[], np.ndarray | None] | None = None):
        ec = cfg["estimator"]
        self.p = params
        self.sigma = cfg["sensing"]["pos_noise_std"]
        self.q = ec["accel_noise_std"]
        self.init_vel_std = ec["init_vel_std"]
        self.reset_vel_std = ec["reset_vel_std"]
        self.nis_threshold = ec["bounce_nis_threshold"]
        self.gate_z = z_impact + ec["bounce_gate_height"]
        self.min_sigma = ec["min_pos_std"]
        self.expected_v_out = expected_v_out
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.t: float | None = None
        self.bounces: list[float] = []         # times of detected bounces (measurement time)
        self.last_nis = np.nan

    @property
    def initialised(self) -> bool:
        return self.x is not None

    def _R(self) -> np.ndarray:
        s = max(self.sigma, self.min_sigma)      # sigma = 0 would make S singular after a reset
        return s * s * np.eye(3)

    def _reset(self, t: float, z: np.ndarray, v: np.ndarray, vel_std: float) -> None:
        s = max(self.sigma, self.min_sigma)
        self.x = np.concatenate([z, v])
        self.P = np.diag([s * s] * 3 + [vel_std ** 2] * 3)
        self.t = t

    def _jac_accel(self, v: np.ndarray) -> np.ndarray:
        sp = np.linalg.norm(v)
        if sp < 1e-9:
            return np.zeros((3, 3))
        return -self.p.k_d * (sp * np.eye(3) + np.outer(v, v) / sp)

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        F[3:, 3:] += dt * self._jac_accel(self.x[3:])
        I3 = np.eye(3)
        Q = self.q ** 2 * np.block([[dt ** 3 / 3 * I3, dt ** 2 / 2 * I3], [dt ** 2 / 2 * I3, dt * I3]])
        self.x = rk4_step(self.x, np.zeros(3), self.p, dt)
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, m: Measurement) -> bool:
        """Process one measurement; returns True if a bounce was detected (and the filter reset)."""
        if not self.initialised:
            self._reset(m.t, m.z, np.zeros(3), self.init_vel_std)
            return False
        self.predict(m.t)
        y = m.z - self.x[:3]
        S = self.P[:3, :3] + self._R()
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis
        if nis > self.nis_threshold and y[2] > 0 and min(self.x[2], m.z[2]) < self.gate_z:
            v_prior = self.expected_v_out() if self.expected_v_out else None
            self._reset(m.t, m.z, np.zeros(3) if v_prior is None else np.asarray(v_prior, float),
                        self.init_vel_std if v_prior is None else self.reset_vel_std)
            self.bounces.append(m.t)
            return True
        K = np.linalg.solve(S, self.P[:3, :]).T          # P H^T S^-1  (S symmetric)
        self.x = self.x + K @ y
        IKH = np.eye(6)
        IKH[:, :3] -= K
        self.P = IKH @ self.P @ IKH.T + K @ self._R() @ K.T
        return False

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Estimated (position, velocity) at time t >= filter time (mean propagated by the flight model)."""
        x = self.x
        dt = t - self.t
        if dt > 1e-12:
            x = rk4_step(x, np.zeros(3), self.p, dt)
        return x[:3].copy(), x[3:].copy()
