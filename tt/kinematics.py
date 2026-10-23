"""Paddle forward kinematics and a static pose IK (used for the ready pose; M7 builds on it).

Paddle pose = position p of site `paddle_center` and unit normal n (the site's z-axis).
Rotation about n is left free, so the IK has 5 task constraints on a 7-DOF arm;
the remaining redundancy is resolved by a small posture term pulling q toward q_nominal.

IK = bounded nonlinear least squares over q in the joint limits:
    r(q) = [ w_p (p(q) - p*) ;  w_n (n(q) - n*) ;  w_q (q - q_nom) ]
with analytic Jacobian
    dp/dq = J_p,   dn/dq = -[n]x J_r   (since dn/dt = w x n = -n x w),
where J_p, J_r are MuJoCo's site translational/rotational Jacobians (mj_jacSite).
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

from tt.config import ready_paddle_pos
from tt.model import SceneIds


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def paddle_pose(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds) -> tuple[np.ndarray, np.ndarray]:
    """(position, unit normal) of the paddle center in world; needs kinematics up to date."""
    return d.site_xpos[ids.paddle_site].copy(), d.site_xmat[ids.paddle_site].reshape(3, 3)[:, 2].copy()


def paddle_fk(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds, q: np.ndarray):
    """Set arm joints to q in scratch data d and return (p, n, J_p, J_n) restricted to the arm dofs."""
    d.qpos[ids.arm_qpos] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, ids.paddle_site)
    p, n = paddle_pose(m, d, ids)
    return p, n, jp[:, ids.arm_dof], -skew(n) @ jr[:, ids.arm_dof]


def solve_paddle_pose(m: mujoco.MjModel, ids: SceneIds, p_target, n_target, q_nominal, w_pos: float,
                      w_normal: float, w_posture: float, tol: float, q0=None, refine_ratio: float = 1e-4):
    """Joint angles placing the paddle center at p_target with normal n_target. Returns (q, p, n)."""
    d = mujoco.MjData(m)
    p_t = np.asarray(p_target, float)
    n_t = np.asarray(n_target, float) / np.linalg.norm(n_target)
    q_nom = np.asarray(q_nominal, float)
    n = len(ids.arm_qpos)

    w_q = w_posture

    def residual(q):
        p, nn, _, _ = paddle_fk(m, d, ids, q)
        return np.concatenate([w_pos * (p - p_t), w_normal * (nn - n_t), w_q * (q - q_nom)])

    def jac(q):
        _, _, jp, jn = paddle_fk(m, d, ids, q)
        return np.vstack([w_pos * jp, w_normal * jn, w_q * np.eye(n)])

    x0 = np.clip(q_nom if q0 is None else q0, ids.q_lo, ids.q_hi)
    # Stage 1 picks the posture (redundancy) near q_nominal; stage 2 re-solves from there with a
    # negligible posture weight so the task (position + normal) is met exactly, not traded off.
    for w in (w_posture, w_posture * refine_ratio):
        w_q = w
        x0 = least_squares(residual, x0, jac=jac, bounds=(ids.q_lo, ids.q_hi), method="trf",
                           ftol=tol, xtol=tol, gtol=tol).x
    p, nn, _, _ = paddle_fk(m, d, ids, x0)
    return x0, p, nn


def refine_paddle_pose(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds, p_target, n_target, q0,
                       damping: float, tol: float, max_iter: int):
    """Fast warm-started IK: damped Gauss-Newton on r = [p - p*, n - n*] from q0, joint limits by clipping.

    q <- clip(q - J^T (J J^T + lambda^2 I)^-1 r). From a nearby solution (the previous impact pose)
    it converges in a few iterations. Returns (q, p, n, converged).
    """
    p_t = np.asarray(p_target, float)
    n_t = np.asarray(n_target, float) / np.linalg.norm(n_target)
    q = np.clip(np.asarray(q0, float), ids.q_lo, ids.q_hi)
    for _ in range(max_iter):
        p, n, jp, jn = paddle_fk(m, d, ids, q)
        r = np.concatenate([p - p_t, n - n_t])
        if np.linalg.norm(r) < tol:
            return q, p, n, True
        J = np.vstack([jp, jn])
        q = np.clip(q - J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), r), ids.q_lo, ids.q_hi)
    p, n, _, _ = paddle_fk(m, d, ids, q)
    return q, p, n, np.linalg.norm(np.concatenate([p - p_t, n - n_t])) < tol


def solve_paddle_pose_cfg(m: mujoco.MjModel, ids: SceneIds, cfg: dict, p_target, n_target, q0=None):
    """solve_paddle_pose with the IK weights/tolerance/posture prior from the config."""
    rp = cfg["ready_pose"]
    return solve_paddle_pose(m, ids, p_target, n_target, rp["q_nominal"], rp["ik_weight_pos"],
                             rp["ik_weight_normal"], rp["ik_weight_posture"], rp["ik_tol"], q0=q0,
                             refine_ratio=rp["ik_refine_posture_ratio"])


def ready_pose(m: mujoco.MjModel, ids: SceneIds, cfg: dict):
    return solve_paddle_pose_cfg(m, ids, cfg, ready_paddle_pos(cfg), cfg["ready_pose"]["paddle_normal"])


def tangent_basis(n: np.ndarray) -> np.ndarray:
    """(2, 3) rows t1, t2 with (t1, t2, n) orthonormal."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    return np.vstack([t1, np.cross(n, t1)])


def task_jacobian(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds, q: np.ndarray):
    """5x7 Jacobian of the paddle task: centre velocity (3) and tilt rates about t1, t2 (2).

    Spin about the normal is not part of the task (it is free), so this is the Jacobian the
    planner inverts for q_dot at impact and the one whose manipulability we maximise.
    """
    d.qpos[ids.arm_qpos] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, ids.paddle_site)
    n = d.site_xmat[ids.paddle_site].reshape(3, 3)[:, 2]
    return np.vstack([jp[:, ids.arm_dof], tangent_basis(n) @ jr[:, ids.arm_dof]])


def manipulability(J: np.ndarray) -> float:
    """Yoshikawa manipulability sqrt(det(J J^T))."""
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
