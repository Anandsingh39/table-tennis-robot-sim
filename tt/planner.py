"""M7 — Motion planner: impact joint state (IK + velocity) and quintic joint trajectories.

Impact state. For a desired paddle centre p*, normal n* and velocity v* at time t*:
  q*  = IK(p*, n*)                         (tt.kinematics; spin about n is free, posture prior picks it)
  q̇*  = J⁺ [v*; 0; 0]                      J = 5x7 task Jacobian (centre velocity + tilt rates);
                                           the two zeros ask for no tilting at impact; J⁺ is the
                                           (damped) minimum-norm pseudo-inverse.
Trajectory. A quintic per joint matches position, velocity and acceleration at both ends:
  approach:  (q, q̇, q̈)(t_now) -> (q*, q̇*, 0) at t*      (re-planned from the current desired
                                                       state, so q_d stays C2-continuous)
  return:    (q*, q̇*, 0) at t* -> (q_ready, 0, 0) at t* + T_return   (follow-through + return)
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tt.kinematics import refine_paddle_pose, solve_paddle_pose_cfg, task_jacobian
from tt.model import SceneIds


@dataclass(frozen=True)
class Quintic:
    """q(t) = sum_k c_k (t - t0)^k on [t0, t0 + T], vectorised over joints (c has shape (6, n))."""

    t0: float
    T: float
    c: np.ndarray

    @classmethod
    def fit(cls, t0, T, q0, v0, a0, q1, v1, a1) -> "Quintic":
        q0, v0, a0, q1, v1, a1 = (np.asarray(x, float) for x in (q0, v0, a0, q1, v1, a1))
        dq = q1 - q0
        c3 = (20 * dq - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3)
        c4 = (-30 * dq + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4)
        c5 = (12 * dq - 6 * (v1 + v0) * T - (a0 - a1) * T ** 2) / (2 * T ** 5)
        return cls(t0, T, np.vstack([q0, v0, a0 / 2, c3, c4, c5]))

    def __call__(self, t: float):
        s = min(max(t - self.t0, 0.0), self.T)
        c = self.c
        q = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))))
        qd = c[1] + s * (2 * c[2] + s * (3 * c[3] + s * (4 * c[4] + s * 5 * c[5])))
        qdd = 2 * c[2] + s * (6 * c[3] + s * (12 * c[4] + s * 20 * c[5]))
        return q, qd, qdd

    @property
    def t1(self) -> float:
        return self.t0 + self.T


class JointTrajectory:
    """Piecewise quintic; before the first segment / after the last it holds the end state at rest."""

    def __init__(self, segments: list[Quintic]):
        self.segments = segments

    def __call__(self, t: float):
        for seg in self.segments:
            if t <= seg.t1:
                return seg(t)
        q, _, _ = self.segments[-1](self.segments[-1].t1)
        return q, np.zeros_like(q), np.zeros_like(q)

    @property
    def t_end(self) -> float:
        return self.segments[-1].t1


def impact_joint_state(m: mujoco.MjModel, d_scratch: mujoco.MjData, ids: SceneIds, cfg: dict,
                       p, n, v, q_seed, warm: bool = False):
    """(q*, q̇*, info) achieving paddle centre p, normal n and centre velocity v (no tilt rate).

    warm=True: q_seed is a nearby impact pose (previous plan) -> fast damped Gauss-Newton refinement,
    falling back to the full two-stage IK if it does not converge.
    """
    pl = cfg["planner"]
    ok = False
    if warm:
        q, p_ik, n_ik, ok = refine_paddle_pose(m, d_scratch, ids, p, n, q_seed, pl["ik_refine_damping"],
                                               pl["ik_refine_tol"], pl["ik_refine_max_iter"])
    if not ok:
        q, p_ik, n_ik = solve_paddle_pose_cfg(m, ids, cfg, p, n, q0=q_seed)
    J = task_jacobian(m, d_scratch, ids, q)
    lam = cfg["planner"]["pinv_damping"]
    xdot = np.concatenate([np.asarray(v, float), np.zeros(2)])
    qd = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(J.shape[0]), xdot)
    info = dict(ik_pos_err=float(np.linalg.norm(p_ik - p)),
                ik_normal_err_deg=float(np.rad2deg(np.arccos(np.clip(n_ik @ n, -1, 1)))),
                task_vel_err=float(np.linalg.norm(J @ qd - xdot)))
    return q, qd, info


def plan_stroke(t_now, q0, qd0, qdd0, t_hit, q_hit, qd_hit, t_return, q_ready) -> JointTrajectory:
    approach = Quintic.fit(t_now, t_hit - t_now, q0, qd0, qdd0, q_hit, qd_hit, np.zeros_like(q_hit))
    ret = Quintic.fit(t_hit, t_return, q_hit, qd_hit, np.zeros_like(q_hit), q_ready, np.zeros_like(q_hit),
                      np.zeros_like(q_hit))
    return JointTrajectory([approach, ret])


def trajectory_stats(traj: JointTrajectory, ids: SceneIds, t0: float, dt: float) -> dict:
    ts = np.arange(t0, traj.t_end + dt, dt)
    Q = np.array([traj(t) for t in ts])          # (N, 3, 7)
    q, qd, qdd = Q[:, 0], Q[:, 1], Q[:, 2]
    return dict(min_margin=float(np.min(np.minimum(q - ids.q_lo, ids.q_hi - q))),
                max_qd=np.abs(qd).max(axis=0), max_qdd=np.abs(qdd).max(axis=0))
