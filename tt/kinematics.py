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
                      w_normal: float, w_posture: float, tol: float, q0=None):
    """Joint angles placing the paddle center at p_target with normal n_target. Returns (q, p, n)."""
    d = mujoco.MjData(m)
    p_t = np.asarray(p_target, float)
    n_t = np.asarray(n_target, float) / np.linalg.norm(n_target)
    q_nom = np.asarray(q_nominal, float)
    n = len(ids.arm_qpos)

    def residual(q):
        p, nn, _, _ = paddle_fk(m, d, ids, q)
        return np.concatenate([w_pos * (p - p_t), w_normal * (nn - n_t), w_posture * (q - q_nom)])

    def jac(q):
        _, _, jp, jn = paddle_fk(m, d, ids, q)
        return np.vstack([w_pos * jp, w_normal * jn, w_posture * np.eye(n)])

    x0 = np.clip(q_nom if q0 is None else q0, ids.q_lo, ids.q_hi)
    sol = least_squares(residual, x0, jac=jac, bounds=(ids.q_lo, ids.q_hi), method="trf",
                        ftol=tol, xtol=tol, gtol=tol)
    p, nn, _, _ = paddle_fk(m, d, ids, sol.x)
    return sol.x, p, nn


def ready_pose(m: mujoco.MjModel, ids: SceneIds, cfg: dict):
    rp = cfg["ready_pose"]
    return solve_paddle_pose(m, ids, rp["paddle_pos"], rp["paddle_normal"], rp["q_nominal"],
                             rp["ik_weight_pos"], rp["ik_weight_normal"], rp["ik_weight_posture"], rp["ik_tol"])
