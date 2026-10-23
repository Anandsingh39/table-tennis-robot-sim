"""Phase 1 tests: scene structure (M1), ready-pose IK, and the gravity-comp + PD hold."""

import mujoco
import numpy as np
import pytest

from tt.config import load_config
from tt.control import gravity_comp_pd
from tt.kinematics import paddle_fk, paddle_pose, ready_pose
from tt.model import build_model


@pytest.fixture(scope="module")
def scene():
    cfg = load_config()
    m, ids = build_model(cfg)
    return cfg, m, ids


def test_single_arm_with_torque_motors(scene):
    cfg, m, ids = scene
    assert m.nu == cfg["robot"]["n_joints"] == 7
    assert all("left" not in m.joint(i).name and "finger" not in m.joint(i).name for i in range(m.njnt))
    # motor: f = ctrl (gain 1, no bias), torque-limited to the stock actuatorfrcrange
    assert np.all(m.actuator_biastype[ids.arm_act] == mujoco.mjtBias.mjBIAS_NONE)
    assert np.allclose(m.actuator_gainprm[ids.arm_act, 0], 1.0)
    jid = [m.joint(n).id for n in ids.joint_names]
    assert np.allclose(m.actuator_ctrlrange[ids.arm_act], m.jnt_actfrcrange[jid])
    assert np.allclose(ids.tau_max, [40, 40, 27, 27, 7, 7, 7])


def test_ball_and_paddle_parameters(scene):
    cfg, m, ids = scene
    bc, pc = cfg["ball"], cfg["paddle"]
    assert m.body_mass[ids.ball_body] == pytest.approx(bc["mass"])
    assert m.body_inertia[ids.ball_body] == pytest.approx([2 / 3 * bc["mass"] * bc["radius"] ** 2] * 3)
    assert m.geom_size[ids.ball_geom, 0] == pytest.approx(bc["radius"])
    assert m.body_subtreemass[ids.paddle_body] == pytest.approx(pc["blade_mass"] + pc["handle_mass"])
    # ball collides with nothing but the floor (analytic paddle/table contact in M3)
    floor = m.geom("floor").id
    for g in range(m.ngeom):
        if g in (ids.ball_geom, floor):
            continue
        hit = (m.geom_contype[g] & m.geom_conaffinity[ids.ball_geom]) or (
            m.geom_contype[ids.ball_geom] & m.geom_conaffinity[g])
        assert not hit, m.geom(g).name


def test_paddle_jacobian_matches_finite_difference(scene):
    _, m, ids = scene
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    q = rng.uniform(ids.q_lo, ids.q_hi)
    p, n, jp, jn = paddle_fk(m, d, ids, q)
    eps = 1e-6
    for k in range(len(q)):
        dq = np.zeros_like(q)
        dq[k] = eps
        p2, n2, _, _ = paddle_fk(m, d, ids, q + dq)
        assert np.allclose((p2 - p) / eps, jp[:, k], atol=1e-5)
        assert np.allclose((n2 - n) / eps, jn[:, k], atol=1e-5)


def test_ready_pose_reaches_target(scene):
    cfg, m, ids = scene
    q, p, n = ready_pose(m, ids, cfg)
    ck, rp = cfg["checks"], cfg["ready_pose"]
    assert np.all(q > ids.q_lo) and np.all(q < ids.q_hi)
    assert np.linalg.norm(p - rp["paddle_pos"]) < ck["ready_pose_pos_tol"]
    assert np.rad2deg(np.arccos(np.clip(n @ rp["paddle_normal"], -1, 1))) < ck["ready_pose_normal_tol_deg"]


def test_hold_ready_pose(scene):
    cfg, m, ids = scene
    d = mujoco.MjData(m)
    q_ready, _, _ = ready_pose(m, ids, cfg)
    d.qpos[ids.arm_qpos] = q_ready
    mujoco.mj_forward(m, d)
    p0, _ = paddle_pose(m, d, ids)
    kp, kd = np.array(cfg["control"]["hold_kp"]), np.array(cfg["control"]["hold_kd"])
    drift = 0.0
    for _ in range(int(cfg["checks"]["hold_duration"] / m.opt.timestep)):
        mujoco.mj_step1(m, d)
        d.ctrl[ids.arm_act] = gravity_comp_pd(m, d, ids, q_ready, kp, kd)
        drift = max(drift, np.linalg.norm(d.site_xpos[ids.paddle_site] - p0))
        mujoco.mj_step2(m, d)
    assert drift < cfg["checks"]["hold_max_drift"]
