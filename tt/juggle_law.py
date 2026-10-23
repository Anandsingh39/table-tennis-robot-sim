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


@dataclass
class PDTiltLaw:
    """Comparison law (paddle-juggling papers, e.g. Ji et al. 2024 Sec. 5): PD on the paddle tilt.

        tilt t = Kp (c - p*_xy) - Kd v_in,xy        (rad; capped at max_tilt),   n = (t_x, t_y, 1) / |.|
        paddle speed along n chosen so the vertical outgoing speed is v_z (apex target):
            v_out,z = v_in,z - (1 + e) (v_in . n - s) n_z = v_z   =>   s = v_in . n + (v_z - v_in,z) / ((1 + e) n_z)
    For small tilts v_out,xy ~ v_in,xy + dv_z n_xy (dv_z = v_z - v_in,z), so the impact-targeting law is
    this PD law with Kp = k_c / (T dv_z) and Kd = 1 / dv_z: the default gains (config null) are exactly
    those, i.e. the PD law is the linearised mirror law with fixed gains.
    """

    e: float
    center_xy: np.ndarray
    v_z: float
    kp: float               # rad / m
    kd: float               # rad / (m/s)
    max_tilt: float

    @classmethod
    def from_config(cls, cfg: dict, params: AeroParams, dt: float) -> "PDTiltLaw":
        it = ImpactTargetingLaw.from_config(cfg, params, dt)
        jc = cfg["juggle"]
        g, k, h = -params.gravity[2], params.k_d, jc["apex_height"]
        v_fall = np.sqrt(2 * g * h) if k == 0 else np.sqrt(g / k * -np.expm1(-2 * k * h))  # nominal |v_in,z|
        dv_z = it.v_z + v_fall
        kp = jc["pd_kp"] if jc.get("pd_kp") is not None else it.k_c / (it.T * dv_z)
        kd = jc["pd_kd"] if jc.get("pd_kd") is not None else 1.0 / dv_z
        return cls(e=it.e, center_xy=it.center_xy, v_z=it.v_z, kp=kp, kd=kd, max_tilt=it.max_tilt)

    def __call__(self, p_star: np.ndarray, v_in: np.ndarray):
        t = self.kp * (self.center_xy - p_star[:2]) - self.kd * v_in[:2]
        mag = np.linalg.norm(t)
        if mag > np.tan(self.max_tilt):
            t *= np.tan(self.max_tilt) / mag
        n = np.array([t[0], t[1], 1.0]) / np.sqrt(1.0 + t @ t)
        s = float(v_in @ n + (self.v_z - v_in[2]) / ((1.0 + self.e) * n[2]))
        v_out = frictionless_rebound(v_in, n, s * n, self.e)
        return n, s, v_out


def make_law(cfg: dict, params: AeroParams, dt: float):
    """juggle.law: impact_targeting (default) | pd_tilt."""
    law = cfg["juggle"].get("law", "impact_targeting")
    return {"impact_targeting": ImpactTargetingLaw, "pd_tilt": PDTiltLaw}[law].from_config(cfg, params, dt)


def grip_paddle_command(v_in, w_in, v_out, w_des, e_n, e_t, e_s, mass, radius, inertia, n0=None):
    """Inverse of the racket-rubber contact (tt.contact.impact_grip): paddle normal n and contact-point
    velocity v_c that turn (v_in, w_in) into v_out with the in-plane spin w_des.

    With grip the tangential impulse couples velocity and spin: Δω = -(κ / r) n × Δv_t, so for a given
    velocity change the spin is set by how far n is tilted away from (v_out - v_in). Unknowns: n (two
    angles) and v_c (3); equations: v_out (3) and the spin components perpendicular to n (2; the spin
    about n is only damped, e_s, not controllable). Solved by least squares from the frictionless solution.
    Returns (n, v_c, w_out, residual).
    """
    from scipy.optimize import least_squares

    from tt.contact import impact_grip

    v_in, w_in, v_out, w_des = (np.asarray(x, float) for x in (v_in, w_in, v_out, w_des))
    if n0 is None:
        n0, s0 = paddle_command(v_in, v_out, e_n)
    else:
        s0 = float((v_out @ n0 + e_n * (v_in @ n0)) / (1.0 + e_n))

    def nvec(a):
        th, ph = a
        return np.array([np.sin(th) * np.cos(ph), -np.sin(ph), np.cos(th) * np.cos(ph)])

    def res(x):
        n = nvec(x[:2])
        vo, wo = impact_grip(v_in, w_in, n, x[2:], e_n, e_t, e_s, mass, radius, inertia)
        dw = wo - w_des
        dw = dw - (dw @ n) * n
        return np.concatenate([vo - v_out, radius * dw])          # spin residual scaled to m/s

    x0 = np.concatenate([[np.arctan2(n0[0], n0[2]), -np.arcsin(np.clip(n0[1], -1, 1))], s0 * n0])
    sol = least_squares(res, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    n = nvec(sol.x[:2])
    _, w_out = impact_grip(v_in, w_in, n, sol.x[2:], e_n, e_t, e_s, mass, radius, inertia)
    return n, sol.x[2:].copy(), w_out, float(np.linalg.norm(sol.fun))
