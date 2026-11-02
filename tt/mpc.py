"""M7 (rally) — MPC swing: joint-space optimal control through the impact (after Nguyen, Cancio, Kim 2025 [4]).

Decision variables over N = N1 + N2 intervals (N1 before the impact, N2 for the follow-through):
    joint states x_k = (q_k, q̇_k), k = 0..N,   piecewise-constant joint accelerations a_k, k = 0..N-1.
Dynamics (exact for a double integrator with constant acceleration over the interval h_k):
    q_{k+1} = q_k + h_k q̇_k + h_k² a_k / 2,     q̇_{k+1} = q̇_k + h_k a_k
    h_k = (t_hit - t_now) / N1 before the impact, T_ft / N2 after it (the impact is node N1).
    (Shrinking horizon: x_0 is the current state and h_k shrinks as the impact approaches = the SHMPC variant of [4].)
Constraints:
    x_0 = current desired joint state (re-plans start where the last plan is, so q_d stays continuous)
    impact (node N1), paddle centre p, normal n, paddle velocity v = J_p(q) q̇ — mpc.impact_constraints:
        "equality":   p = c*,  t1·n = t2·n = 0,  n·n* >= cos(tol),  n*·v = s*         (exact)
        "tolerance":  ‖p - c*‖² <= ε_p²,  ‖n - n*‖² <= ε_n²,  (n*·v - s*)² <= ε_v²   ([4] eq. 3g-3i)
    (with mpc.full_velocity the velocity condition is on the full vector v - v*: brushing for spin shots)
    end (node N):      q̇_N = 0                                                      (stopped after the follow-through)
    every node:        joint limits (minus a margin), |q̇| <= speed limits, |a| <= acceleration limits
Cost:  Σ_k h_k (w_a ‖a_k‖² + w_v ‖q̇_k‖²)  ([4] eq. 3a, as a time integral)  +  w_end ‖q_N - q_ready‖²
       (finish near the ready pose).
Best effort (mpc.best_effort, our addition, not in [4]): if the impact problem has no solution (e.g. the ball is out
of reach), a second problem with the impact conditions as exact L1 penalties (slack variables s >= |residual|) is
solved: it returns the swing that comes closest. SwingPlan.success stays False for it; .best_effort is True.
With penalty weights above the impact multipliers the soft problem meets the impact exactly whenever the hard one
can (checked on recorded swings); the problem is non-convex, so it may then return a different, equally exact swing.
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

IMPACT_MODES = ("equality", "tolerance", "soft")


@dataclass
class SwingPlan:
    t0: float
    h: np.ndarray            # (N,) interval lengths
    q: np.ndarray            # (N+1, n)
    qd: np.ndarray           # (N+1, n)
    a: np.ndarray            # (N, n)
    n_pre: int               # impact node index
    solve_time: float        # all solves of this call (incl. a best-effort fallback)
    success: bool            # the configured impact problem converged
    status: str
    best_effort: bool = False                      # success is False, the soft fallback converged instead
    impact_err: tuple = (np.nan, np.nan, np.nan)   # planned (|p - c*| [m], angle(n, n*) [deg], velocity error [m/s])

    @property
    def usable(self) -> bool:
        return self.success or self.best_effort

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


def solve_stat_rows(log) -> list[tuple[str, str]]:
    """Summary rows from [(success, best_effort, adopted, status, impact_err)] per solve (convergence as in [4])."""
    n = len(log)
    if not n:
        return [("MPC solves", "0")]
    ok = sum(1 for s, *_ in log if s)
    failed = [(b, a, st, err) for s, b, a, st, err in log if not s]
    rows = [("MPC solves converged with the exact impact / total", f"{ok} / {n} ({100 * ok / n:.1f}%)"),
            ("failed solves: best-effort swing executed / best effort not used (exact plan running) / "
             "best effort also failed (current plan kept)",
             f"{sum(1 for b, a, *_ in failed if b and a)} / {sum(1 for b, a, *_ in failed if b and not a)} / "
             f"{sum(1 for b, *_ in failed if not b)}")]
    if failed:
        counts: dict[str, int] = {}
        for _, _, st, _ in failed:
            st = st.replace("best effort after ", "")
            counts[st] = counts.get(st, 0) + 1
        rows.append(("failed-solve IPOPT status", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))))
        be = np.array([err for b, a, _, err in failed if b and a], float).reshape(-1, 3)
        if len(be):
            rows.append(("executed best-effort swings, planned miss median / p95: paddle centre [mm], normal [deg], "
                         "speed [m/s]", "{:.1f} / {:.1f}, {:.2f} / {:.2f}, {:.3f} / {:.3f}".format(
                             *(f(be[:, i]) * (1e3 if i == 0 else 1) for i in range(3)
                               for f in (np.median, lambda x: np.percentile(x, 95))))))
    return rows


@dataclass
class _Problem:
    opti: ca.Opti
    Q: ca.MX
    QD: ca.MX
    A: ca.MX
    P: dict                  # parameters


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
        self.full_velocity = mc.get("full_velocity", False)
        self.mode = mc.get("impact_constraints", "equality")
        if self.mode not in ("equality", "tolerance"):
            raise ValueError(f"mpc.impact_constraints must be 'equality' or 'tolerance', not {self.mode!r}")
        self.fk = PaddleFK(m, ids.paddle_site, ids.arm_qpos)
        self.prob = self._build(self.fk, mc, self.mode)
        self.soft = self._build(self.fk, mc, "soft") if mc.get("best_effort", False) else None
        self.last: SwingPlan | None = None          # last converged plan (warm start)

    def _build(self, fk: PaddleFK, mc: dict, mode: str) -> _Problem:
        assert mode in IMPACT_MODES
        N, N1, nq = self.N1 + self.N2, self.N1, self.nq
        opti = ca.Opti()
        Q = opti.variable(nq, N + 1)
        QD = opti.variable(nq, N + 1)
        A = opti.variable(nq, N)
        P = dict(x0=opti.parameter(2 * nq),     # current desired joint state
                 h1=opti.parameter(),           # interval length before the impact
                 c=opti.parameter(3),           # paddle centre at impact
                 n=opti.parameter(3),           # paddle normal at impact
                 t=opti.parameter(2, 3),        # tangent basis of the normal
                 s=opti.parameter())            # paddle normal speed at impact
        if self.full_velocity:
            P["v"] = opti.parameter(3)          # full paddle-centre velocity at impact
        h2 = self.T_ft / self.N2
        w_a, w_v = mc.get("w_acc", 1.0), mc.get("w_vel", 0.0)
        cost = 0
        for k in range(N):
            h = P["h1"] if k < N1 else h2
            opti.subject_to(Q[:, k + 1] == Q[:, k] + h * QD[:, k] + 0.5 * h ** 2 * A[:, k])
            opti.subject_to(QD[:, k + 1] == QD[:, k] + h * A[:, k])
            cost += h * w_a * ca.sumsqr(A[:, k])
            if w_v:
                cost += h * w_v * ca.sumsqr(QD[:, k])
        opti.subject_to(Q[:, 0] == P["x0"][:nq])
        opti.subject_to(QD[:, 0] == P["x0"][nq:])

        q_i, qd_i = Q[:, N1], QD[:, N1]
        n_i = fk.rot(q_i)[:, 2]
        r_p = fk.pos(q_i) - P["c"]                                         # (3,)  paddle centre error
        v_i = fk.vel(q_i, qd_i)
        r_v = v_i - P["v"] if self.full_velocity else ca.dot(P["n"], v_i) - P["s"]   # (3,) or (1,)
        if mode == "equality":
            opti.subject_to(r_p == 0)
            opti.subject_to(P["t"] @ n_i == 0)
            opti.subject_to(ca.dot(P["n"], n_i) >= np.cos(np.deg2rad(mc["normal_tol_deg"])))
            opti.subject_to(r_v == 0)
        elif mode == "tolerance":                                         # [4] eq. 3g-3i
            # scaled to ‖r‖²/ε² <= 1: unscaled (ε² ~ 1e-6) IPOPT's feasibility tolerance would loosen ε by ~0.5%
            eps_n = 2 * np.sin(np.deg2rad(mc["tol_normal_deg"]) / 2)      # chord length of the angle tolerance
            opti.subject_to(ca.sumsqr(r_p) / mc["tol_pos"] ** 2 <= 1)
            opti.subject_to(ca.sumsqr(n_i - P["n"]) / eps_n ** 2 <= 1)
            opti.subject_to(ca.sumsqr(r_v) / mc["tol_vel"] ** 2 <= 1)
        else:                                                             # soft: exact L1 penalties via slacks
            r_n = P["t"] @ n_i                                            # (2,) sin of the normal error
            for r, w in ((r_p, mc["soft_w_pos"]), (r_n, mc["soft_w_normal"]), (r_v, mc["soft_w_vel"])):
                s = opti.variable(r.shape[0])
                opti.subject_to(r <= s)
                opti.subject_to(-s <= r)
                cost += w * ca.sum1(s)
            opti.subject_to(ca.dot(P["n"], n_i) >= 0)                     # tangent residuals alone allow -n*
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
        return _Problem(opti, Q, QD, A, P)

    def impact_error(self, q, qd, c_star, n_star, speed, v_paddle=None) -> tuple:
        """(|p - c*| [m], angle(n, n*) [deg], velocity error [m/s]) of a joint state at the impact."""
        p = np.asarray(self.fk.pos(q)).ravel()
        n = np.asarray(self.fk.rot(q))[:, 2]
        v = np.asarray(self.fk.vel(q, qd)).ravel()
        if self.full_velocity:
            ev = float(np.linalg.norm(v - (speed * n_star if v_paddle is None else v_paddle)))
        else:
            ev = abs(float(n_star @ v) - speed)
        return (float(np.linalg.norm(p - c_star)), float(np.degrees(np.arccos(np.clip(n @ n_star, -1.0, 1.0)))), ev)

    def _solve(self, pb: _Problem, t_now, q0, qd0, h1, c_star, n_star, speed, v_paddle) -> SwingPlan:
        N1, N = self.N1, self.N1 + self.N2
        o, P = pb.opti, pb.P
        o.set_value(P["x0"], np.concatenate([q0, qd0]))
        o.set_value(P["h1"], h1)
        o.set_value(P["c"], c_star)
        o.set_value(P["n"], n_star)
        o.set_value(P["t"], tangent_basis(n_star))
        o.set_value(P["s"], speed)
        if self.full_velocity:
            o.set_value(P["v"], np.asarray(speed * n_star if v_paddle is None else v_paddle, float))
        if self.last is not None:                                    # warm start: previous plan resampled
            ts = t_now + np.concatenate([[0], np.cumsum([h1] * N1 + [self.T_ft / self.N2] * self.N2)])
            prev = [self.last(t) for t in ts]
            o.set_initial(pb.Q, np.array([p[0] for p in prev]).T)
            o.set_initial(pb.QD, np.array([p[1] for p in prev]).T)
            o.set_initial(pb.A, np.array([p[2] for p in prev[:-1]]).T)
        else:                                                        # cold start: hold q0
            o.set_initial(pb.Q, np.tile(np.asarray(q0, float)[:, None], (1, N + 1)))
            o.set_initial(pb.QD, 0)
            o.set_initial(pb.A, 0)
        t0 = time.perf_counter()
        try:
            sol = o.solve()
            ok, val = True, sol.value
            status = "ok"
        except RuntimeError:
            ok, val = False, o.debug.value
            status = o.stats().get("return_status", "failed")
        h = np.array([h1] * N1 + [self.T_ft / self.N2] * self.N2)
        plan = SwingPlan(t_now, h, np.array(val(pb.Q)).T, np.array(val(pb.QD)).T, np.array(val(pb.A)).T,
                         N1, time.perf_counter() - t0, ok, status)
        plan.impact_err = self.impact_error(plan.q[N1], plan.qd[N1], c_star, n_star, speed, v_paddle)
        return plan

    def solve(self, t_now: float, q0, qd0, t_hit: float, c_star, n_star, speed, v_paddle=None) -> SwingPlan:
        """speed: paddle normal speed; v_paddle: full paddle-centre velocity (used when mpc.full_velocity)."""
        h1 = (t_hit - t_now) / self.N1
        n_star = np.asarray(n_star, float) / np.linalg.norm(n_star)
        c_star = np.asarray(c_star, float)
        plan = self._solve(self.prob, t_now, q0, qd0, h1, c_star, n_star, speed, v_paddle)
        if not plan.success and self.soft is not None:
            soft = self._solve(self.soft, t_now, q0, qd0, h1, c_star, n_star, speed, v_paddle)
            soft.solve_time += plan.solve_time
            if soft.success:
                soft.success, soft.best_effort, soft.status = False, True, f"best effort after {plan.status}"
                plan = soft
            else:
                plan.solve_time = soft.solve_time
        if plan.usable:
            self.last = plan
        return plan
