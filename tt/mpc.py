"""M7 (rally) — MPC swing: joint-space optimal control through the impact (after Nguyen, Cancio, Kim 2025 [4]).

Decision variables over N = N1 + N2 intervals (N1 before the impact, N2 for the follow-through):
    joint states x_k = (q_k, q̇_k), k = 0..N,   piecewise-constant joint accelerations a_k, k = 0..N-1.
Dynamics (exact for a double integrator with constant acceleration over the interval h_k):
    q_{k+1} = q_k + h_k q̇_k + h_k² a_k / 2,     q̇_{k+1} = q̇_k + h_k a_k
    h_k = (t_hit - t_now) / N1 before the impact, T_ft / N2 after it (the impact is node N1).
Constraints:
    x_0 = current desired joint state (re-plans start where the last plan is, so q_d stays continuous)
    impact (node N1):  p(q) = c*,   t1·n(q) = 0, t2·n(q) = 0, n·n* >= cos(tol)   (paddle centre and normal)
                       n*ᵀ J_p(q) q̇ = s*                                            (paddle normal speed)
    end (node N):      q̇_N = 0                                                      (stopped after the follow-through)
    every node:        joint limits (minus a margin), |q̇| <= speed limits, |a| <= acceleration limits
Cost:  Σ_k h_k ‖a_k‖²  (smooth, low effort)  +  w_end ‖q_N - q_ready‖²  (finish near the ready pose).
p(q), n(q), J_p q̇ are the symbolic MuJoCo kinematics (tt.fk_casadi). Solved with IPOPT; re-solves are
warm-started from the previous solution. The acceleration limits come from the torque limits:
a_max,i = safety · (τ_max,i - |g_i(q_ready)|) / M_ii(q_ready) (diagonal approximation, hence the safety factor).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import mujoco
import numpy as np

from tt.fk_casadi import PaddleFK
from tt.kinematics import tangent_basis
from tt.model import SceneIds


@dataclass
class SwingPlan:
    t0: float
    h: np.ndarray            # (N,) interval lengths
    q: np.ndarray            # (N+1, n)
    qd: np.ndarray           # (N+1, n)
    a: np.ndarray            # (N, n)
    n_pre: int               # impact node index
    solve_time: float
    success: bool
    status: str

    @property
    def t_hit(self) -> float:
        return self.t0 + float(self.h[:self.n_pre].sum())

    @property
    def t_end(self) -> float:
        return self.t0 + float(self.h.sum())

    def __call__(self, t: float):
        """Desired (q, q̇, q̈) at time t (holds the final state after the end)."""
        s = t - self.t0
        if s <= 0:
            return self.q[0].copy(), self.qd[0].copy(), self.a[0].copy()
        edges = np.cumsum(self.h)
        k = int(np.searchsorted(edges, s, side="right"))
        if k >= len(self.h):
            return self.q[-1].copy(), np.zeros_like(self.q[-1]), np.zeros_like(self.q[-1])
        tau = s - (edges[k - 1] if k > 0 else 0.0)
        return (self.q[k] + self.qd[k] * tau + 0.5 * self.a[k] * tau ** 2,
                self.qd[k] + self.a[k] * tau, self.a[k].copy())


class SwingMPC:
    def __init__(self, m: mujoco.MjModel, ids: SceneIds, cfg: dict, q_ready: np.ndarray):
        mc = cfg["mpc"]
        self.N1, self.N2 = mc["n_pre"], mc["n_post"]
        self.T_ft = mc["follow_through_time"]
        self.q_ready = np.asarray(q_ready, float)
        nq = len(ids.arm_qpos)
        self.nq = nq
        # limits
        d = mujoco.MjData(m)
        d.qpos[ids.arm_qpos] = q_ready
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        Mii = np.diag(M)[ids.arm_dof]
        g = np.abs(d.qfrc_bias[ids.arm_dof])
        self.a_max = mc["acc_safety"] * (ids.tau_max - g) / Mii
        self.qd_max = mc["speed_factor"] * np.asarray(cfg["robot"]["joint_speed_rated"], float)
        self.q_lo = ids.q_lo + mc["joint_margin"]
        self.q_hi = ids.q_hi - mc["joint_margin"]
        fk = PaddleFK(m, ids.paddle_site, ids.arm_qpos)
        self._build(fk, mc)
        self.last: SwingPlan | None = None

    def _build(self, fk: PaddleFK, mc: dict) -> None:
        N, N1, nq = self.N1 + self.N2, self.N1, self.nq
        opti = ca.Opti()
        Q = opti.variable(nq, N + 1)
        QD = opti.variable(nq, N + 1)
        A = opti.variable(nq, N)
        self.P_x0 = opti.parameter(2 * nq)
        self.P_h1 = opti.parameter()          # interval length before the impact
        self.P_c = opti.parameter(3)          # paddle centre at impact
        self.P_n = opti.parameter(3)          # paddle normal at impact
        self.P_t = opti.parameter(2, 3)       # tangent basis of the normal
        self.P_s = opti.parameter()           # paddle normal speed at impact
        h2 = self.T_ft / self.N2
        cost = 0
        for k in range(N):
            h = self.P_h1 if k < N1 else h2
            opti.subject_to(Q[:, k + 1] == Q[:, k] + h * QD[:, k] + 0.5 * h ** 2 * A[:, k])
            opti.subject_to(QD[:, k + 1] == QD[:, k] + h * A[:, k])
            cost += h * ca.sumsqr(A[:, k])
        opti.subject_to(Q[:, 0] == self.P_x0[:nq])
        opti.subject_to(QD[:, 0] == self.P_x0[nq:])
        q_i, qd_i = Q[:, N1], QD[:, N1]
        n_i = fk.rot(q_i)[:, 2]
        opti.subject_to(fk.pos(q_i) == self.P_c)
        opti.subject_to(self.P_t @ n_i == 0)
        opti.subject_to(ca.dot(self.P_n, n_i) >= np.cos(np.deg2rad(mc["normal_tol_deg"])))
        opti.subject_to(ca.dot(self.P_n, fk.vel(q_i, qd_i)) == self.P_s)
        opti.subject_to(QD[:, N] == 0)
        for k in range(N + 1):
            opti.subject_to(opti.bounded(self.q_lo, Q[:, k], self.q_hi))
            opti.subject_to(opti.bounded(-self.qd_max, QD[:, k], self.qd_max))
        for k in range(N):
            opti.subject_to(opti.bounded(-self.a_max, A[:, k], self.a_max))
        cost += mc["w_end"] * ca.sumsqr(Q[:, N] - self.q_ready)
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
                              "ipopt.max_iter": mc["max_iter"], "ipopt.tol": mc["tol"],
                              "ipopt.warm_start_init_point": "yes"})
        self.opti, self.Q, self.QD, self.A = opti, Q, QD, A

    def solve(self, t_now: float, q0, qd0, t_hit: float, c_star, n_star, speed) -> SwingPlan:
        N1, N = self.N1, self.N1 + self.N2
        h1 = (t_hit - t_now) / N1
        n_star = np.asarray(n_star, float) / np.linalg.norm(n_star)
        o = self.opti
        o.set_value(self.P_x0, np.concatenate([q0, qd0]))
        o.set_value(self.P_h1, h1)
        o.set_value(self.P_c, c_star)
        o.set_value(self.P_n, n_star)
        o.set_value(self.P_t, tangent_basis(n_star))
        o.set_value(self.P_s, speed)
        if self.last is not None and self.last.success:             # warm start: previous plan resampled
            ts = t_now + np.concatenate([[0], np.cumsum([h1] * N1 + [self.T_ft / self.N2] * self.N2)])
            prev = [self.last(t) for t in ts]
            o.set_initial(self.Q, np.array([p[0] for p in prev]).T)
            o.set_initial(self.QD, np.array([p[1] for p in prev]).T)
            o.set_initial(self.A, np.array([p[2] for p in prev[:-1]]).T)
        else:                                                        # cold start: hold q0
            o.set_initial(self.Q, np.tile(np.asarray(q0, float)[:, None], (1, N + 1)))
            o.set_initial(self.QD, 0)
            o.set_initial(self.A, 0)
        t0 = time.perf_counter()
        try:
            sol = o.solve()
            ok, val = True, sol.value
            status = "ok"
        except RuntimeError:
            ok, val = False, o.debug.value
            status = o.stats().get("return_status", "failed")
        h = np.array([h1] * N1 + [self.T_ft / self.N2] * self.N2)
        plan = SwingPlan(t_now, h, np.array(val(self.Q)).T, np.array(val(self.QD)).T, np.array(val(self.A)).T,
                         N1, time.perf_counter() - t0, ok, status)
        if ok:
            self.last = plan
        return plan
