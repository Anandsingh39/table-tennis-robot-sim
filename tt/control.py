"""M8 — Low-level joint torque control.

gravity_comp_pd (Phase 1, hold a pose):  tau = h(q, qdot) + Kp (q_d - q) - Kd qdot
computed_torque (Phase 4, track q_d(t)): tau = M(q)(qdd_d + Kp e + Kd e_dot) + h - f_passive + friction ff
h = MuJoCo's qfrc_bias (gravity + Coriolis/centrifugal). Both need the current state's
dynamics quantities, so call them between mj_step1 and mj_step2.
The trajectory fixes all 7 joints (redundancy is resolved in the IK posture prior), so no
null-space term is needed here.
"""

from __future__ import annotations

import mujoco
import numpy as np

from tt.model import SceneIds


def computed_torque(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds, q_d: np.ndarray, qd_d: np.ndarray,
                    qdd_d: np.ndarray, kp: np.ndarray, kd: np.ndarray, friction_ff: bool,
                    friction_smoothing: float) -> tuple[np.ndarray, bool]:
    """Computed torque (inverse dynamics + PD) for the arm. Call between mj_step1 and mj_step2.

        tau = M(q) (qdd_d + Kp e + Kd e_dot) + h(q, qdot) - f_passive(qdot) + f_c tanh(qd_d / v_s)
    M     : arm block of the joint-space inertia matrix (incl. motor armature), from mj_fullM;
    h     : qfrc_bias = gravity + Coriolis/centrifugal;
    f_passive : qfrc_passive = joint viscous damping (-b qdot), cancelled;
    f_c   : the model's dry joint friction (frictionloss), fed forward along the desired motion.
    With an exact model the closed loop is e_dd + Kd e_d + Kp e = 0 for every joint.
    Returns (tau, saturated).
    """
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)          # MuJoCo >= 3.17 signature (m, d, dst)
    dof = ids.arm_dof
    e = q_d - d.qpos[ids.arm_qpos]
    ed = qd_d - d.qvel[dof]
    tau = M[np.ix_(dof, dof)] @ (qdd_d + kp * e + kd * ed) + d.qfrc_bias[dof] - d.qfrc_passive[dof]
    if friction_ff:
        tau = tau + m.dof_frictionloss[dof] * np.tanh(qd_d / friction_smoothing)
    sat = bool(np.any(np.abs(tau) > ids.tau_max))
    return np.clip(tau, -ids.tau_max, ids.tau_max), sat


def gravity_comp_pd(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds, q_des: np.ndarray,
                    kp: np.ndarray, kd: np.ndarray) -> np.ndarray:
    q = d.qpos[ids.arm_qpos]
    qd = d.qvel[ids.arm_dof]
    tau = d.qfrc_bias[ids.arm_dof] + kp * (q_des - q) - kd * qd
    return np.clip(tau, -ids.tau_max, ids.tau_max)
