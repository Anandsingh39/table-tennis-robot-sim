"""Simulation setup shared by the scripts (the full juggling loop is added in Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tt.control import gravity_comp_pd
from tt.kinematics import ready_pose
from tt.model import SceneIds, build_model


@dataclass
class Sim:
    cfg: dict
    m: mujoco.MjModel
    d: mujoco.MjData
    ids: SceneIds
    q_ready: np.ndarray
    p_ready: np.ndarray
    n_ready: np.ndarray

    def hold_arm(self) -> None:
        """Gravity comp + PD to the ready pose (call between mj_step1 and mj_step2)."""
        cc = self.cfg["control"]
        self.d.ctrl[self.ids.arm_act] = gravity_comp_pd(
            self.m, self.d, self.ids, self.q_ready, np.asarray(cc["hold_kp"]), np.asarray(cc["hold_kd"]))


def make_sim(cfg: dict) -> Sim:
    """Build the scene and put the arm at rest in the IK ready pose; ball at its model default."""
    m, ids = build_model(cfg)
    d = mujoco.MjData(m)
    q, p, n = ready_pose(m, ids, cfg)
    d.qpos[ids.arm_qpos] = q
    mujoco.mj_forward(m, d)
    return Sim(cfg, m, d, ids, q, p, n)
