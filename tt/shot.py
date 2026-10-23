"""M6 (rally) — Return-shot law: which outgoing ball velocity, and so which paddle state, at impact.

Shooting problem. For the ball centre at p* (impact) and a landing target (x_t, y_t) on the
opponent's half, find v_out such that the nominal flight (gravity + drag, RK4) reaches
(x_t, y_t, z_table + r) after a flight time T:  F(v) = x(T; p*, v) - target = 0.
Solved by Newton's method with a finite-difference Jacobian (3 extra rollouts per iteration),
starting from the vacuum solution v0 = (target - p* - g T^2 / 2) / T (or a warm start).
The flight time T is the "arc" choice: shorter T = flatter, faster ball, larger paddle tilt
(E0: T 0.8 -> ~50 deg, 1.0 -> ~33 deg). On the first plan of a stroke T is the shortest of
candidate_times (flattest shot) that clears the net with the paddle normal within max_tilt of the
mount's nominal normal (reachable); if none is, the one needing the least tilt.
Paddle state then follows from the frictionless impact law (tt.juggle_law.paddle_command).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from tt.aero import AeroParams, rk4_step
from tt.juggle_law import paddle_command


def fly_for(params: AeroParams, pos, vel, T: float, dt: float) -> np.ndarray:
    """Ball state [x, v] after exactly T seconds of free flight (no table)."""
    y = np.concatenate([pos, vel]).astype(float)
    t, w = 0.0, np.zeros(3)
    while t < T - 1e-12:
        h = min(dt, T - t)
        y = rk4_step(y, w, params, h)
        t += h
    return y


def height_at_x(params: AeroParams, pos, vel, x: float, dt: float, t_max: float) -> float:
    """Ball centre height when it crosses the plane at x (moving +x or -x), or -inf if it never does."""
    y = np.concatenate([pos, vel]).astype(float)
    s0, t, w = np.sign(x - y[0]), 0.0, np.zeros(3)
    while t < t_max:
        y_next = rk4_step(y, w, params, dt)
        if np.sign(x - y_next[0]) != s0:
            a = (x - y[0]) / (y_next[0] - y[0])
            return float(y[2] + a * (y_next[2] - y[2]))
        y, t = y_next, t + dt
    return -np.inf


@dataclass
class ShotPlan:
    v_out: np.ndarray
    T: float
    n: np.ndarray            # paddle normal
    speed: float             # paddle normal speed
    tilt_deg: float          # angle between n and the mount's nominal normal
    net_clearance: float     # ball centre height above (net top + r) at the net [m]
    miss: float              # |F(v_out)| of the shooting solve [m]


class ReturnShot:
    def __init__(self, cfg: dict, params: AeroParams, sign: float = 1.0):
        """sign = +1 for the robot at the near end (hits toward +x), -1 for the far-end robot."""
        sc, tc = cfg["shot"], cfg["table"]
        self.p, self.sign = params, sign
        self.e = cfg["contact"]["paddle_restitution"]
        self.dt, self.iters, self.tol, self.fd = sc["dt"], sc["newton_iters"], sc["newton_tol"], sc["fd_step"]
        self.times = sc["candidate_times"]
        self.margin = sc["net_margin"]
        self.z_land = tc["top_height"] + cfg["ball"]["radius"]
        self.x_net = tc["near_edge_x"] + tc["length"] / 2
        self.z_net = tc["top_height"] + tc["net_height"] + cfg["ball"]["radius"]
        tilt = np.deg2rad(cfg["workspace_scan"].get("nominal_tilt_deg", 0.0))
        self.n_nom = np.array([sign * np.sin(tilt), 0.0, np.cos(tilt)])
        self.max_tilt = cfg["workspace_scan"]["max_tilt_deg"]

    def solve_velocity(self, p_hit, target_xy, T: float, v0=None) -> tuple[np.ndarray, float]:
        target = np.array([target_xy[0], target_xy[1], self.z_land])
        g = self.p.gravity
        v = (target - p_hit - 0.5 * g * T ** 2) / T if v0 is None else np.asarray(v0, float).copy()
        miss = np.inf
        for _ in range(self.iters):
            F = fly_for(self.p, p_hit, v, T, self.dt)[:3] - target
            miss = float(np.linalg.norm(F))
            if miss < self.tol:
                break
            J = np.empty((3, 3))
            for k in range(3):
                dv = np.zeros(3)
                dv[k] = self.fd
                J[:, k] = (fly_for(self.p, p_hit, v + dv, T, self.dt)[:3] - target - F) / self.fd
            v = v - np.linalg.solve(J, F)
        return v, miss

    def plan(self, p_hit, v_in, target_xy, T: float | None = None, v0=None) -> ShotPlan | None:
        """Shot for a fixed T (re-plans) or the best T from candidate_times (first plan)."""
        best = None
        for Tc in ([T] if T is not None else self.times):
            v_out, miss = self.solve_velocity(p_hit, target_xy, Tc, v0 if T is not None else None)
            n, s = paddle_command(v_in, v_out, self.e)
            tilt = float(np.degrees(np.arccos(np.clip(n @ self.n_nom, -1, 1))))
            clear = height_at_x(self.p, p_hit, v_out, self.x_net, self.dt, Tc) - self.z_net
            plan = ShotPlan(v_out, Tc, n, s, tilt, clear, miss)
            ok = clear >= self.margin and miss < 10 * self.tol
            reachable = tilt <= self.max_tilt
            key = (ok, reachable, -Tc if reachable else -tilt)   # flattest reachable shot, else least tilt
            if best is None or key > best[0]:
                best = (key, plan)
        return best[1] if best else None


@dataclass
class CoopPlan:
    v_out: np.ndarray
    n: np.ndarray
    speed: float
    tilt_deg: float          # angle between n and the mount's nominal normal
    bounce_dist: float       # planned bounce distance past the net on the partner's half [m]
    flight_time: float       # time from the hit to the partner's hitting plane [s]
    miss: float              # residual of the aim solve [m]


class CoopShot:
    """M6 (E2) — cooperative rally shot: the partner must be able to return the ball.

    v_out is chosen so that the ball (nominal flight with the frictionless table bounce, as M5)
      (i)  bounces on the partner's half at a distance d_b past the net, and
      (ii) then crosses the partner's hitting plane (ball-centre x) at (y_c, z_c) = their hit zone.
    Three equations, three unknowns (least squares on predictor rollouts). Two solution families exist
    (flat/fast and lob); the lob family is selected by requiring the flight time to the partner's plane to
    lie in coop.flight_time_range, which (E2 analysis) needs a reachable paddle tilt of ~30-40 deg, whereas
    the flat family needs 70-90 deg. d_b is picked from coop.bounce_dists so that the paddle tilt is
    closest to the mount's nominal tilt (first plan: candidates nearest the previous stroke's d_b first, stopping
at the first within good_tilt_deg), then kept for the re-plans (warm-started from the last solution).
    sign = +1: the hitter is the near robot (ball goes toward +x); -1: the far robot.
    """

    def __init__(self, cfg: dict, params: AeroParams, predictor, table, sign: float,
                 partner_plane_x: float, partner_cross_z: float):
        cc, tc = cfg["coop"], cfg["table"]
        self.p, self.pred, self.table, self.sign = params, predictor, table, sign
        self.e = cfg["contact"]["paddle_restitution"]
        self.x_mid = tc["near_edge_x"] + tc["length"] / 2
        self.x_plane, self.z_c = partner_plane_x, partner_cross_z
        self.dists, self.t_range = cc["bounce_dists"], cc["flight_time_range"]
        self.grid_vx, self.grid_vz, self.accept = cc["grid_vx"], cc["grid_vz"], cc["grid_accept"]
        self.tol, self.diff_step, self.max_flight = cc["tol"], cc["diff_step"], cc["max_flight"]
        tilt = np.deg2rad(cfg["workspace_scan"].get("nominal_tilt_deg", 0.0))
        self.n_nom = np.array([sign * np.sin(tilt), 0.0, np.cos(tilt)])
        self.max_tilt = cfg["workspace_scan"]["max_tilt_deg"]
        self.good_tilt = cc["good_tilt_deg"]
        self.d_b, self.d_last, self.v_prev = None, None, None

    def _events(self, start, vel):
        """(bounce x, (y, z) at the partner's plane, time to the plane) or None; stops at the outcome."""
        y = np.concatenate([start, vel]).astype(float)
        w, x_b, t, s = np.zeros(3), None, 0.0, self.sign
        while t < self.max_flight:
            y_next, contact = self.pred._step_with_table(y, w, self.table)
            if contact is not None:
                if x_b is not None or (contact[0] - self.x_mid) * s <= 0:
                    return None                                 # second bounce, or bounce on the hitter's half
                x_b = contact[0]
            if (y[0] - self.x_plane) * s < 0 <= (y_next[0] - self.x_plane) * s:
                if x_b is None:
                    return None
                a = (self.x_plane - y[0]) / (y_next[0] - y[0])
                return x_b, y[1:3] + a * (y_next[1:3] - y[1:3]), t + a * self.pred.dt
            if y_next[2] < self.table.z_contact - 0.05:
                return None
            y, t = y_next, t + self.pred.dt
        return None

    def _residual(self, v, start, x_b, y_c):
        ev = self._events(start, v)
        if ev is None:
            return np.array([10.0, 10.0, 10.0])
        xb, yz, _ = ev
        return np.array([xb - x_b, yz[0] - y_c, yz[1] - self.z_c])

    def _guesses(self, start, x_b, y_c):
        cands = []
        for vx in np.arange(*self.grid_vx):
            for vz in np.arange(*self.grid_vz):
                v = np.array([self.sign * vx, 0.0, vz])
                ev = self._events(start, v)
                if ev is None or not (self.t_range[0] <= ev[2] <= self.t_range[1]):
                    continue
                r = np.linalg.norm(self._residual(v, start, x_b, y_c))
                if r < self.accept:
                    cands.append((r, v))
        return [v for _, v in sorted(cands, key=lambda c: c[0])]

    def _aim(self, start, x_b, y_c, v0_list):
        for v0 in v0_list:
            sol = least_squares(self._residual, v0, args=(start, x_b, y_c), diff_step=self.diff_step)
            ev = self._events(start, sol.x)
            if np.linalg.norm(sol.fun) < self.tol and ev and self.t_range[0] <= ev[2] <= self.t_range[1]:
                return sol.x, float(np.linalg.norm(sol.fun)), ev[2]
        return None

    def plan(self, p_hit, v_in, y_c: float = 0.0, replan: bool = False) -> CoopPlan | None:
        best = None
        if replan and self.d_b is not None:
            dists = [self.d_b]
        else:   # nearest to the last stroke's choice first; stop at the first good-enough tilt
            ref = self.d_last if self.d_last is not None else float(np.median(self.dists))
            dists = sorted(self.dists, key=lambda d: abs(d - ref))
        for d in dists:
            x_b = self.x_mid + self.sign * d
            # warm start from the last solution (previous re-plan or previous stroke); grid only as a fallback
            res = self._aim(p_hit, x_b, y_c, [self.v_prev]) if self.v_prev is not None else None
            if res is None:
                res = self._aim(p_hit, x_b, y_c, self._guesses(p_hit, x_b, y_c)[:3])
            if res is None:
                continue
            v_out, miss, T = res
            n, s = paddle_command(v_in, v_out, self.e)
            tilt = float(np.degrees(np.arccos(np.clip(n @ self.n_nom, -1, 1))))
            plan = CoopPlan(v_out, n, s, tilt, d, T, miss)
            key = (tilt <= self.max_tilt, -tilt)
            if best is None or key > best[0]:
                best = (key, plan)
            if tilt <= self.good_tilt:
                break
        if best is None:
            return None
        self.d_b, self.d_last, self.v_prev = best[1].bounce_dist, best[1].bounce_dist, best[1].v_out
        return best[1]

    def new_stroke(self):
        """Next stroke: choose d_b again (v_prev is kept as a warm start)."""
        self.d_b = None
