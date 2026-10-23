"""Simulation setup and the main juggling loop.

Rates: physics + torque control 1 kHz (every step), planner 100 Hz (juggle.planner_period).
Per physics step (between mj_step1 and mj_step2, so all quantities refer to the same state):
  ball state -> [planner tick] -> desired joint state -> computed torque -> contact (M3) -> aero (M2) -> log
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tt.aero import AeroParams, BallAero
from tt.contact import ContactModel, arm_paddle_state, paddle_surface, table_surface
from tt.control import computed_torque, gravity_comp_pd
from tt.kinematics import paddle_pose, ready_pose
from tt.log_utils import Logger
from tt.model import SceneIds, build_model
from tt.render import simulate


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


def make_sim(cfg: dict, prescribed_paddle: bool = False) -> Sim:
    """Build the scene and put the arm at rest in the IK ready pose; ball at its model default."""
    m, ids = build_model(cfg, prescribed_paddle)
    d = mujoco.MjData(m)
    q, p, n = ready_pose(m, ids, cfg)
    d.qpos[ids.arm_qpos] = q
    mujoco.mj_forward(m, d)
    return Sim(cfg, m, d, ids, q, p, n)


@dataclass
class JuggleRun:
    logs: dict
    events: list            # paddle ContactEvents
    plans: list             # juggler PlanRecords
    failure: str | None     # None = reached max_bounces
    z_impact: float
    seed: int
    tau_max: np.ndarray
    predictions: list = None        # juggler predictions (t, t_hit_pred, p*, v_in)
    detected_bounces: list = None   # estimator bounce detections (measurement times); None with true state


def run_juggle(cfg: dict, seed: int, max_bounces: int | None = None, video=None, video_duration=None,
               viewer: bool = False, slowmo: float = 1.0) -> JuggleRun:
    """Drop the ball over the paddle and juggle it until max_bounces or failure.

    sensing.ball_source = "true": the juggler sees the simulator's ball state (Phase 5);
    "estimate": it sees the EKF estimate built from noisy, possibly delayed/dropped measurements (Phase 6).
    Randomness: initial offset from default_rng(seed); sensor noise from default_rng([seed, 1]).
    """
    from tt.juggler import Juggler        # local import: juggler imports Sim from this module
    from tt.sensing import BallEKF, BallSensor

    jc, cc = cfg["juggle"], cfg["control"]
    use_estimate = cfg["sensing"]["ball_source"] == "estimate"
    max_bounces = max_bounces or jc["max_bounces"]
    sim = make_sim(cfg)
    m, d, ids = sim.m, sim.d, sim.ids
    mm = cfg["mismatch"]
    params = AeroParams.from_config(cfg)                                                # nominal (robot's model)
    params_true = AeroParams.from_config(cfg, drag_coefficient=mm["true_drag_coefficient"])   # simulated world
    aero = BallAero(m, ids, params_true)
    contact = ContactModel(m, ids, aero, [paddle_surface(cfg, arm_paddle_state(ids), mm["true_paddle_restitution"]),
                                          table_surface(cfg, params_true)], cfg["contact"]["rewind_to_impact_time"])
    juggler = Juggler(sim, cfg, params)
    center = np.asarray(cfg["hit"]["center"], float)

    rng = np.random.default_rng(seed)
    xy0 = center[:2] + rng.uniform(-jc["drop_xy_range"], jc["drop_xy_range"], 2)
    aero.set_ball_state(m, d, [xy0[0], xy0[1], juggler.z_impact + jc["drop_height"]], np.zeros(3))
    mujoco.mj_forward(m, d)

    sensor = ekf = None
    if use_estimate:
        sensor = BallSensor(cfg, np.random.default_rng([seed, 1]))
        ekf = BallEKF(cfg, params, juggler.z_impact,
                      expected_bounce=lambda: (juggler.plans[-1].t_hit, juggler.plans[-1].v_out) if juggler.plans else None)

    log = Logger()
    paddle_events: list = []
    status = {"failure": None, "next_tick": 0.0}
    state_code = {"POST": 0, "APPROACH": 1}
    nan3 = np.full(3, np.nan)

    def control(m, d):
        t = d.time
        p, v, w = aero.ball_state(d)
        meas = nan3
        if use_estimate:
            new = sensor.sample(t, p)
            if new is not None:
                meas = new.z
            for msg in sensor.delivered(t):
                ekf.update(msg)
            p_hat, v_hat = ekf.state_at(t) if ekf.initialised else (nan3, nan3)
            w_hat = np.zeros(3)                      # spin is not observed
        else:
            p_hat, v_hat, w_hat = p, v, w
        if t + 1e-9 >= status["next_tick"]:
            if not np.isnan(p_hat[0]):
                juggler.tick(t, p_hat, v_hat, w_hat)
            status["next_tick"] += jc["planner_period"]
        q_d, qd_d, qdd_d = juggler.desired(t)
        tau, sat = computed_torque(m, d, ids, q_d, qd_d, qdd_d, cc["ct_kp"], cc["ct_kd"], cc["friction_ff"],
                                   cc["friction_smoothing"])
        d.ctrl[ids.arm_act] = tau
        pp, pn = paddle_pose(m, d, ids)
        log.log(t=t, ball_p=p, ball_v=v, meas=meas, est_p=p_hat, est_v=v_hat, q=d.qpos[ids.arm_qpos],
                qd=d.qvel[ids.arm_dof], q_d=q_d, qd_d=qd_d, tau=tau, sat=sat, paddle_p=pp, paddle_n=pn,
                state=state_code[juggler.state])
        for ev in contact.step(m, d):
            if ev.surface == "paddle":
                paddle_events.append(ev)
                if np.linalg.norm(ev.pos[:2] - center[:2]) > jc["fail_radius"]:
                    status["failure"] = f"impact {len(paddle_events)} too far from centre"
            else:
                status["failure"] = f"ball hit the {ev.surface}"
        aero.apply(d)
        if p[2] < juggler.z_impact - jc["fail_drop"]:
            status["failure"] = "ball dropped"
        return status["failure"] is not None or len(paddle_events) > max_bounces

    max_time = jc["drop_height"] + (max_bounces + 2) * (juggler.flight_time + 0.1)
    simulate(m, d, max_time, control, cfg=cfg, video=video, viewer=viewer, slowmo=slowmo,
             video_duration=video_duration)
    failure = status["failure"]
    if failure is None and len(paddle_events) <= max_bounces:
        failure = "time limit"
    return JuggleRun(log.arrays(), paddle_events, juggler.plans, failure, juggler.z_impact, seed, ids.tau_max.copy(),
                     juggler.predictions, ekf.bounces if ekf else None)
