"""M6 — Juggling law (Section 6 of the plan): impact-targeting, mirror-law style.

Inverse of the frictionless impulse (M3). The ball's velocity changes only along the paddle normal n:
    v_out = v_in - (1 + e) ((v_in - v_p) . n) n
Hence (v_out - v_in) is parallel to n, which fixes the paddle normal, and projecting on n:
    v_out . n = -e v_in . n + (1 + e) v_p . n
    =>  n       = (v_out - v_in) / |v_out - v_in|
        v_p . n = (v_out . n + e v_in . n) / (1 + e)
Only the paddle's normal velocity matters; its tangential velocity is free (we use v_p = (v_p . n) n).

Choice of v_out (the heuristic part):
  vertical:   launch speed that reaches apex height h with quadratic drag k (1-D, closed form)
                  v_z = sqrt((g/k)(exp(2 k h) - 1))          (-> sqrt(2 g h) as k -> 0)
  horizontal: send the ball back toward the centre c in one flight of duration T
                  v_xy = k_c (c - p*_xy) / T,   0 < k_c <= 1 (1 = deadbeat, ~0.5 = smoother)
              The incoming horizontal velocity is cancelled automatically by the choice of n.
  tilt limit: the angle between n and vertical is capped at max_tilt by shrinking the horizontal
              part of (v_out - v_in).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tt.aero import AeroParams, rk4_step


def paddle_command(v_in: np.ndarray, v_out: np.ndarray, e: float) -> tuple[np.ndarray, float]:
    """Paddle normal n and normal speed v_p . n that turn v_in into v_out."""
    dv = np.asarray(v_out, float) - np.asarray(v_in, float)
    n = dv / np.linalg.norm(dv)
    return n, float((v_out @ n + e * (v_in @ n)) / (1.0 + e))


def frictionless_rebound(v_in: np.ndarray, n: np.ndarray, v_p: np.ndarray, e: float) -> np.ndarray:
    """Forward formula (M3 with mu = 0) for checks."""
    return v_in - (1.0 + e) * ((v_in - v_p) @ n) * n


def launch_speed_for_apex(h: float, k: float, g: float) -> float:
    """Vertical launch speed reaching height h under gravity g and quadratic drag k (k = 0: sqrt(2 g h))."""
    return float(np.sqrt(2 * g * h)) if k == 0 else float(np.sqrt(g / k * np.expm1(2 * k * h)))


def flight_time(v_z: float, params: AeroParams, dt: float) -> float:
    """Time for a vertical launch at v_z to come back down to its launch height (RK4, with drag)."""
    y, t = np.array([0, 0, 0, 0, 0, v_z], float), 0.0
    while True:
        y_next = rk4_step(y, np.zeros(3), params, dt)
        if y_next[5] < 0 and y_next[2] <= 0:
            return t + dt * y[2] / (y[2] - y_next[2])
        y, t = y_next, t + dt


@dataclass
class ImpactTargetingLaw:
    e: float                 # nominal paddle restitution
    center_xy: np.ndarray    # juggling centre c
    v_z: float               # outgoing vertical speed for the apex target
    T: float                 # flight time back to the impact height
    k_c: float               # centring gain
    max_tilt: float          # rad

    @classmethod
    def from_config(cls, cfg: dict, params: AeroParams, dt: float) -> "ImpactTargetingLaw":
        jc = cfg["juggle"]
        v_z = launch_speed_for_apex(jc["apex_height"], params.k_d, -params.gravity[2])
        return cls(e=cfg["contact"]["paddle_restitution"], center_xy=np.asarray(cfg["hit"]["center"][:2], float),
                   v_z=v_z, T=flight_time(v_z, params, dt), k_c=jc["centering_gain"],
                   max_tilt=np.deg2rad(jc["max_tilt_deg"]))

    def v_out(self, p_star: np.ndarray, v_in: np.ndarray) -> np.ndarray:
        v_xy = self.k_c * (self.center_xy - p_star[:2]) / self.T
        dv = np.array([v_xy[0] - v_in[0], v_xy[1] - v_in[1], self.v_z - v_in[2]])
        dv_h_max = np.tan(self.max_tilt) * dv[2]
        dv_h = np.linalg.norm(dv[:2])
        if dv_h > dv_h_max:                         # saturate the tilt
            dv[:2] *= dv_h_max / dv_h
        return v_in + dv

    def __call__(self, p_star: np.ndarray, v_in: np.ndarray):
        """(n, paddle normal speed, v_out) for a ball arriving at p_star with velocity v_in."""
        v_out = self.v_out(p_star, v_in)
        n, s = paddle_command(v_in, v_out, self.e)
        return n, s, v_out
