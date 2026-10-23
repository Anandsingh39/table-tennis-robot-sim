"""M8 — Low-level joint torque control.

Phase 1: gravity/Coriolis compensation + joint PD (hold a pose)
    tau = h(q, qdot) + Kp (q_d - q) - Kd qdot,
where h = MuJoCo's qfrc_bias (gravity + Coriolis/centrifugal). qfrc_bias must be
computed for the current state, so call this between mj_step1 and mj_step2.
Computed torque (Phase 4) extends this with M(q) * (qdd_d + PD).
"""

from __future__ import annotations

import mujoco
import numpy as np

from tt.model import SceneIds


def gravity_comp_pd(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds, q_des: np.ndarray,
                    kp: np.ndarray, kd: np.ndarray) -> np.ndarray:
    q = d.qpos[ids.arm_qpos]
    qd = d.qvel[ids.arm_dof]
    tau = d.qfrc_bias[ids.arm_dof] + kp * (q_des - q) - kd * qd
    return np.clip(tau, -ids.tau_max, ids.tau_max)
