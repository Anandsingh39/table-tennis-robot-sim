"""Symbolic paddle forward kinematics in CasADi, built from the compiled MuJoCo model (for the MPC, E1).

MuJoCo's kinematics for a chain of bodies with hinge joints:
    R_b = R_parent · R(quat_b) · Rot(axis_b, q_b),
    p_b = p_parent + R_parent · pos_b + (R_parent R(quat_b)) · a_b - R_b · a_b    (a_b = joint anchor in the body frame)
a slide joint (the rally rail) translates the body along its axis instead: p_b += (R_parent R(quat_b)) · axis_b q_b,
and the site: p_s = p_b + R_b · pos_s, R_s = R_b · R(quat_s). Rot(axis, q) is Rodrigues' formula.
All constants (pos_b, quat_b, axis_b, a_b, site offsets) are read from the MjModel, so the
symbolic model is exactly MuJoCo's (unit-tested against mj_kinematics).
"""

from __future__ import annotations

import casadi as ca
import mujoco
import numpy as np


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _axis_rot(axis: np.ndarray, q):
    """Rodrigues: rotation by angle q (symbolic) about the unit vector axis (numeric)."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = ca.DM([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return ca.DM.eye(3) + ca.sin(q) * K + (1 - ca.cos(q)) * (K @ K)


class PaddleFK:
    """CasADi functions of the arm joint vector q (7):
         pos(q)  -> paddle centre position (3)
         rot(q)  -> paddle frame rotation (3x3); column 2 = blade normal
         vel(q, qd) -> paddle centre linear velocity (3) = J_p(q) qd
    """

    def __init__(self, m: mujoco.MjModel, site_id: int, arm_qpos: np.ndarray):
        chain = []
        b = int(m.site_bodyid[site_id])
        while b != 0:
            chain.append(b)
            b = int(m.body_parentid[b])
        chain.reverse()
        jnt_of_qpos = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
        arm_joints = [jnt_of_qpos[int(a)] for a in arm_qpos]
        q = ca.SX.sym("q", len(arm_joints))
        R = ca.DM.eye(3)
        p = ca.DM.zeros(3)
        for b in chain:
            R_fixed = R @ ca.DM(_quat_to_mat(m.body_quat[b]))
            p = p + R @ ca.DM(m.body_pos[b])
            R = R_fixed
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                qi = q[arm_joints.index(j)]
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:      # rail: translate along the axis
                    p = p + R_fixed @ ca.DM(m.jnt_axis[j]) * qi
                    continue
                if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("only hinge and slide joints are supported")
                a = ca.DM(m.jnt_pos[j])
                R_new = R_fixed @ _axis_rot(m.jnt_axis[j], qi)
                p = p + R_fixed @ a - R_new @ a          # rotate about the anchor
                R = R_new
        p_site = p + R @ ca.DM(m.site_pos[site_id])
        R_site = R @ ca.DM(_quat_to_mat(m.site_quat[site_id]))
        qd = ca.SX.sym("qd", q.shape[0])
        self.n = q.shape[0]
        self.pos = ca.Function("paddle_pos", [q], [p_site])
        self.rot = ca.Function("paddle_rot", [q], [R_site])
        self.vel = ca.Function("paddle_vel", [q, qd], [ca.jacobian(p_site, q) @ qd])
