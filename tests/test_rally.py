"""E1 tests: predictor with the table bounce, return-shot solver, MPC swing constraints."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tt.aero import AeroParams
from tt.config import RALLY_CONFIG, load_config
from tt.kinematics import paddle_fk
from tt.mpc import SwingMPC
from tt.predictor import BallPredictor, TableBounce
from tt.shot import ReturnShot, fly_for
from tt.sim import make_sim


@pytest.fixture(scope="module")
def cfg():
    return load_config(RALLY_CONFIG)


def test_rally_config_extends_default(cfg):
    base = load_config()
    assert cfg["paddle"]["flange_quat"] != base["paddle"]["flange_quat"]
    assert cfg["ball"] == base["ball"] and cfg["table"] == base["table"]


def test_plane_crossing_after_table_bounce_vacuum(cfg):
    p = AeroParams.from_config(cfg, drag_coefficient=0.0)
    g, e = 9.81, 0.9
    table = TableBounce((0.0, 3.0), (-1.0, 1.0), 0.78, e)
    pos, vel = np.array([2.5, 0.0, 1.0]), np.array([-5.0, 0.0, 0.0])      # fast enough for exactly one bounce
    pred = BallPredictor(p, 0.005, 3.0).plane_crossing(pos, vel, np.zeros(3), 0, 0.4, -1.0, table, require_bounce=True)
    # analytic: fall 0.22 m to the bounce, then rebound with e * v_z until x = 0.4
    t_b = np.sqrt(2 * 0.22 / g)
    vz_b = e * g * t_b
    t_c = (2.5 - 0.4) / 5.0
    z_c = 0.78 + vz_b * (t_c - t_b) - 0.5 * g * (t_c - t_b) ** 2
    assert pred.bounced and pred.t_rel == pytest.approx(t_c, abs=1e-6)
    assert pred.pos[2] == pytest.approx(z_c, abs=1e-6)
    assert pred.vel[2] == pytest.approx(vz_b - g * (t_c - t_b), abs=1e-5)


def test_return_shot_lands_on_target(cfg):
    p = AeroParams.from_config(cfg)
    shot = ReturnShot(cfg, p)
    p_hit, target = np.array([0.41, 0.01, 0.93]), np.array([3.05, 0.1])
    plan = shot.plan(p_hit, np.array([-3.5, 0.0, -1.0]), target)
    end = fly_for(p, p_hit, plan.v_out, plan.T, 1e-4)        # independent finer integration
    assert np.linalg.norm(end[:2] - target) < 2e-3
    assert end[2] == pytest.approx(shot.z_land, abs=2e-3)
    assert plan.net_clearance >= cfg["shot"]["net_margin"]


def test_mpc_swing_meets_impact_and_limits(cfg):
    sim = make_sim(cfg)
    m, ids = sim.m, sim.ids
    mpc = SwingMPC(m, ids, cfg, sim.q_ready)
    c = np.array([0.40, 0.02, 0.92])
    n = Rotation.from_euler("y", 50, degrees=True).apply([0, 0, 1.0])
    plan = mpc.solve(0.0, sim.q_ready, np.zeros(7), 0.35, c, n, 1.8)
    assert plan.success
    d = mujoco.MjData(m)
    p, nn, jp, _ = paddle_fk(m, d, ids, plan.q[plan.n_pre])
    assert np.linalg.norm(p - c) < 1e-6 and np.degrees(np.arccos(np.clip(nn @ n, -1, 1))) < 1e-3
    assert n @ (jp @ plan.qd[plan.n_pre]) == pytest.approx(1.8, abs=1e-6)
    tol = 1e-6
    assert np.all(plan.q >= mpc.q_lo - tol) and np.all(plan.q <= mpc.q_hi + tol)
    assert np.all(np.abs(plan.qd) <= mpc.qd_max + tol) and np.all(np.abs(plan.a) <= mpc.a_max + tol)
    assert np.allclose(plan.qd[-1], 0, atol=tol)
    q, qd, _ = plan(plan.t_hit)                                   # the trajectory passes the impact node
    assert np.allclose(q, plan.q[plan.n_pre], atol=1e-9)


def _swing_case(cfg, **mpc_overrides):
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["mpc"].update(mpc_overrides)
    sim = make_sim(cfg)
    return sim, SwingMPC(sim.m, sim.ids, cfg, sim.q_ready)


def _within_limits(mpc, plan, tol=1e-6):
    return (np.all(plan.q >= mpc.q_lo - tol) and np.all(plan.q <= mpc.q_hi + tol) and
            np.all(np.abs(plan.qd) <= mpc.qd_max + tol) and np.all(np.abs(plan.a) <= mpc.a_max + tol) and
            np.allclose(plan.qd[-1], 0, atol=tol))


C_OK = np.array([0.40, 0.02, 0.92])
N_OK = Rotation.from_euler("y", 50, degrees=True).apply([0, 0, 1.0])


def test_mpc_tolerance_mode_meets_tolerances(cfg):
    """[4] eq. 3g-3i: the impact state is inside the tolerance balls (and the optimum sits on their boundary)."""
    sim, mpc = _swing_case(cfg, impact_constraints="tolerance", best_effort=False)
    plan = mpc.solve(0.0, sim.q_ready, np.zeros(7), 0.35, C_OK, N_OK, 1.8)
    assert plan.success
    mc = cfg["mpc"]
    e_p, e_n, e_v = plan.impact_err
    assert e_p <= mc["tol_pos"] * (1 + 1e-6) and e_n <= mc["tol_normal_deg"] * (1 + 1e-6)
    assert e_v <= mc["tol_vel"] * (1 + 1e-6)
    assert _within_limits(mpc, plan)


def test_mpc_soft_problem_is_exact_when_feasible(cfg):
    """Exact penalty: with weights above the multipliers the L1 problem hits the target exactly."""
    sim, mpc = _swing_case(cfg, best_effort=True)
    n = N_OK / np.linalg.norm(N_OK)
    hard = mpc._solve(mpc.prob, 0.0, sim.q_ready, np.zeros(7), 0.35 / mpc.N1, C_OK, n, 1.8, None)
    soft = mpc._solve(mpc.soft, 0.0, sim.q_ready, np.zeros(7), 0.35 / mpc.N1, C_OK, n, 1.8, None)
    assert hard.success and soft.success
    assert soft.impact_err[0] < 1e-6 and soft.impact_err[1] < 1e-3 and soft.impact_err[2] < 1e-5


def test_mpc_best_effort_for_unreachable_ball(cfg):
    """Out of reach (0.5 m too low): the exact problem fails, the fallback still returns a swing within all limits,
    and the paddle state it reaches is itself exactly reachable (a consistency check of the soft solution)."""
    sim, mpc = _swing_case(cfg, best_effort=True)
    c_low = np.array([0.40, 0.0, 0.40])
    plan = mpc.solve(0.0, sim.q_ready, np.zeros(7), 0.35, c_low, N_OK, 1.8)
    assert not plan.success and plan.best_effort and plan.usable
    assert plan.impact_err[0] > 0.1
    assert _within_limits(mpc, plan)
    d = mujoco.MjData(sim.m)
    p, nn, jp, _ = paddle_fk(sim.m, d, sim.ids, plan.q[plan.n_pre])
    s_reached = float(nn @ (jp @ plan.qd[plan.n_pre]))
    mpc.last = None
    again = mpc.solve(0.0, sim.q_ready, np.zeros(7), 0.35, p, nn, s_reached)
    assert again.success


def test_mpc_best_effort_off_keeps_failure(cfg):
    sim, mpc = _swing_case(cfg, best_effort=False)
    plan = mpc.solve(0.0, sim.q_ready, np.zeros(7), 0.35, np.array([0.40, 0.0, 0.40]), N_OK, 1.8)
    assert not plan.success and not plan.best_effort and not plan.usable


def test_mpc_velocity_cost_lowers_joint_speeds(cfg):
    """[4] eq. 3a: adding w_v |q̇|² lowers the integrated squared joint speed of the same swing."""
    def speed_integral(w_v):
        sim, mpc = _swing_case(cfg, w_vel=w_v, best_effort=False)
        plan = mpc.solve(0.0, sim.q_ready, np.zeros(7), 0.35, C_OK, N_OK, 1.8)
        assert plan.success
        return float(np.sum(plan.h[:, None] * plan.qd[:-1] ** 2))
    assert speed_integral(5.0) < speed_integral(0.0)


def test_adopt_plan_rule():
    from types import SimpleNamespace as NS

    from tt.rally import adopt_plan
    exact, best, failed = NS(success=True, best_effort=False), NS(success=False, best_effort=True), \
        NS(success=False, best_effort=False)
    assert adopt_plan(exact, None) and adopt_plan(exact, best) and adopt_plan(exact, exact)
    assert adopt_plan(best, None) and adopt_plan(best, best)      # no exact plan running: take the closest swing
    assert not adopt_plan(best, exact)                              # never replace an exact plan by a best effort
    assert not adopt_plan(failed, None)


def test_launcher_rejects_diverged_solve(cfg):
    """E1 seed 61: an unbounded least-squares start diverged to ~1400 m/s; it must be rejected (speed cap)."""
    from tt.launcher import Launcher
    from tt.rally import nominal_geometry
    p = AeroParams.from_config(cfg)
    table, _, _, x_plane, _ = nominal_geometry(cfg, p)
    pred = BallPredictor(p, cfg["juggle"]["predictor_dt"], cfg["rally"]["predictor_horizon"])
    launch = Launcher(cfg, pred, table, x_plane).sample(np.random.default_rng(61))
    assert np.linalg.norm(launch.vel) <= cfg["launcher"]["max_speed"]


def test_solve_stat_rows_counts():
    from tt.mpc import solve_stat_rows
    log = [(True, False, True, "ok", (0.0, 0.0, 0.0)),
           (False, True, True, "best effort after Infeasible_Problem_Detected", (0.01, 2.0, 0.1)),
           (False, True, False, "best effort after Infeasible_Problem_Detected", (0.02, 1.0, 0.2)),
           (False, False, False, "Maximum_Iterations_Exceeded", (np.nan, np.nan, np.nan))]
    rows = dict(solve_stat_rows(log))
    assert rows["MPC solves converged with the exact impact / total"] == "1 / 4 (25.0%)"
    counts = [v for k, v in rows.items() if k.startswith("failed solves:")][0]
    assert counts == "1 / 1 / 1"
    assert rows["failed-solve IPOPT status"] == "Infeasible_Problem_Detected: 2, Maximum_Iterations_Exceeded: 1"
    miss = [v for k, v in rows.items() if k.startswith("executed best-effort")][0]
    assert miss == "10.0 / 10.0, 2.00 / 2.00, 0.100 / 0.100"          # only the executed best-effort swing


def test_two_robot_rally_short():
    """E2: two mirrored arms, cooperative lobs; a short rally completes without a fault."""
    from tt.config import REPO
    from tt.rally2 import run_rally

    cfg2 = load_config(REPO / "config" / "rally2.yaml")
    r = run_rally(cfg2, seed=1, max_hits=4)
    assert r.fault == "max hits" and len(r.hits) == 4
    assert [who for who, _ in r.hits] == ["A", "B", "A", "B"]
    assert not (r.logs["sat_A"].any() or r.logs["sat_B"].any())


def test_rail_kinematics_and_short_rally():
    """E2 + rail: slide joint in the symbolic FK; a short rally with lateral targets completes."""
    from tt.config import REPO
    from tt.fk_casadi import PaddleFK
    from tt.model import build_model, scene_ids
    from tt.rally2 import run_rally

    cfg3 = load_config(REPO / "config" / "rally_rail.yaml")
    m, ids = build_model(cfg3)
    ido = scene_ids(m, cfg3, cfg3["opponent"]["prefix"])
    assert ids.joint_names[0] == "rail_slide" and len(ids.arm_qpos) == 8
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for idx in (ids, ido):
        fk = PaddleFK(m, idx.paddle_site, idx.arm_qpos)
        q = rng.uniform(idx.q_lo, idx.q_hi)
        p, _, _, _ = paddle_fk(m, d, idx, q)
        assert np.allclose(np.array(fk.pos(q)).ravel(), p, atol=1e-9)
    r = run_rally(cfg3, seed=2, max_hits=3)
    assert r.fault == "max hits" and len(r.hits) == 3


def test_grip_contact_inverse_round_trip():
    """Spin: the grip paddle inverse reproduces the requested outgoing velocity and in-plane spin."""
    from tt.contact import impact_grip
    from tt.juggle_law import grip_paddle_command

    m, r = 0.0027, 0.02
    inertia = 2 / 3 * m * r ** 2
    v_in, w_in = np.array([-3.2, 0.1, -2.8]), np.array([0.0, -40.0, 5.0])
    v_out = np.array([2.9, 0.05, 4.1])
    for spin in (-80.0, 0.0, 60.0):
        w_des = spin * np.array([0.0, 1.0, 0.0])          # topspin axis for a ball going +x
        n, v_c, w_out, res = grip_paddle_command(v_in, w_in, v_out, w_des, 0.8, 0.819, 0.805, m, r, inertia)
        vo, wo = impact_grip(v_in, w_in, n, v_c, 0.8, 0.819, 0.805, m, r, inertia)
        assert res < 1e-8 and np.allclose(vo, v_out, atol=1e-8)
        dw = wo - w_des
        assert np.linalg.norm(dw - (dw @ n) * n) < 1e-5


def test_predictor_spinning_table_bounce_matches_contact_law():
    """Spin: the predictor's table bounce (with friction) is the simulator's impulse law."""
    from tt.contact import impact

    tb = TableBounce((0.0, 3.0), (-1.0, 1.0), 0.78, 0.9, 0.25, 0.0027, 0.02, 2 / 3 * 0.0027 * 0.02 ** 2)
    v, w = np.array([3.0, 0.2, -4.0]), np.array([0.0, 100.0, 10.0])
    v1, w1 = tb.bounce(v, w)
    v2, w2, _ = impact(v, w, np.array([0, 0, 1.0]), np.zeros(3), 0.9, 0.25, 0.0027, 0.02, 2 / 3 * 0.0027 * 0.02 ** 2)
    assert np.allclose(v1, v2) and np.allclose(w1, w2)
    v0, _ = tb.bounce(v, np.zeros(3))
    vb, _ = tb.bounce(v, -w)
    assert v1[0] > v0[0] > vb[0]              # topspin keeps more forward speed off the table than no spin; backspin checks
