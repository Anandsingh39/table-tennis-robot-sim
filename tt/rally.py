"""E1: one arm returns a launched ball over the net (MPC swing). Shared by E2 (two arms).

Striker state machine (ticked at juggle.planner_period):
  WAIT    hold the ready pose; predict when the ball, after bouncing on our half, crosses the hitting
          plane x = x_plane (M5 with the table bounce). When that is closer than rally.plan_horizon:
          shot law (M6, picks the flight time T) -> paddle centre c* = p* - h n, normal n, speed s ->
          MPC swing (M7) -> SWING.
  SWING   re-predict, re-solve the shot (same T, warm start) and the MPC (warm start, from the current
          desired state) every mpc.period until mpc.freeze_time before the impact; then the plan is fixed.
          After the plan's end (follow-through finished, arm at rest) -> RETURN.
  RETURN  quintic back to the ready pose, then WAIT.
The low level is the same computed-torque controller (M8) as for juggling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from tt.aero import AeroParams, BallAero
from tt.contact import ContactModel, arm_paddle_state, paddle_surface, table_restitution_from_rule, table_surface
from tt.control import computed_torque
from tt.kinematics import paddle_pose
from tt.launcher import Launch, Launcher
from tt.log_utils import Logger
from tt.mpc import SwingMPC, SwingPlan
from tt.planner import JointTrajectory, Quintic
from tt.predictor import BallPredictor, TableBounce
from tt.render import simulate
from tt.shot import ReturnShot
from tt.sim import Sim, make_sim


@dataclass
class StrikeRecord:
    t_plan: float
    t_hit: float
    p_star: np.ndarray
    v_in: np.ndarray
    v_out: np.ndarray
    n: np.ndarray
    speed: float
    c_star: np.ndarray
    T: float
    net_clearance: float
    solve_time: float
    success: bool
    best_effort: bool = False         # the soft fallback converged (success False)
    adopted: bool = False             # this plan became the executed swing
    status: str = ""
    impact_err: tuple = ()            # planned (|p - c*| [m], normal [deg], velocity [m/s]) at the impact node


def adopt_plan(plan, current) -> bool:
    """Execute a converged plan; a best-effort plan only when no exact plan is already being executed."""
    return plan.success or (plan.best_effort and not (current is not None and current.success))


def nominal_geometry(cfg: dict, params: AeroParams):
    """Table bounce model, shot law, hitting plane (ball centre x at impact) and landing target."""
    cc, tc = cfg["contact"], cfg["table"]
    e_t = table_restitution_from_rule(cc["table_rule_drop_height"], cc["table_rule_rebound_height"], params.k_d)
    table = TableBounce.from_config(cfg, e_t)
    shot = ReturnShot(cfg, params)
    h = cfg["paddle"]["blade_thickness"] / 2 + cfg["ball"]["radius"]
    x_plane = cfg["hit"]["center"][0] + h * shot.n_nom[0]
    x_net = tc["near_edge_x"] + tc["length"] / 2
    fx, ty = cfg["shot"]["target"]
    target = np.array([x_net + fx * tc["length"] / 2, ty])
    return table, shot, h, x_plane, target


@dataclass
class Striker:
    sim: Sim
    cfg: dict
    params: AeroParams
    state: str = "WAIT"
    plan: SwingPlan | None = None
    strikes: list[StrikeRecord] = field(default_factory=list)
    solve_times: list[float] = field(default_factory=list)

    def __post_init__(self):
        rc, jc = self.cfg["rally"], self.cfg["juggle"]
        self.table, self.shot, self.h, self.x_plane, self.target = nominal_geometry(self.cfg, self.params)
        self.predictor = BallPredictor(self.params, jc["predictor_dt"], rc["predictor_horizon"])
        self.mpc = SwingMPC(self.sim.m, self.sim.ids, self.cfg, self.sim.q_ready)
        q, z = self.sim.q_ready, np.zeros_like(self.sim.q_ready)
        self.traj = JointTrajectory([Quintic.fit(0.0, 1.0, q, z, z, q, z, z)])
        self.t_hit, self.T, self.v_out, self.next_solve = np.inf, None, None, -np.inf

    def desired(self, t: float):
        if self.state == "SWING":
            return self.plan(t)
        return self.traj(t)

    def tick(self, t: float, pos, vel, spin, bounced: bool) -> None:
        rc, mc = self.cfg["rally"], self.cfg["mpc"]
        if self.state == "SWING" and t >= self.plan.t_end:
            q_end = self.plan.q[-1]
            z = np.zeros_like(q_end)
            self.traj = JointTrajectory([Quintic.fit(t, rc["return_time"], q_end, z, z, self.sim.q_ready, z, z)])
            self.state, self.T, self.v_out = "RETURN", None, None
        if self.state == "RETURN" and t >= self.traj.t_end:
            self.state = "WAIT"
        if self.state == "RETURN":
            return
        if self.state == "SWING" and (t >= self.t_hit - mc["freeze_time"] or t < self.next_solve):
            return
        pred = self.predictor.plane_crossing(pos, vel, spin, 0, self.x_plane, -1.0, self.table,
                                             require_bounce=True, already_bounced=bounced)
        if pred is None or (self.state == "WAIT" and pred.t_rel > rc["plan_horizon"]):
            return
        sp = self.shot.plan(pred.pos, pred.vel, self.target, T=self.T, v0=self.v_out)
        if sp is None:
            return
        c_star = pred.pos - self.h * sp.n
        q0, qd0, _ = self.desired(t)
        t_hit = t + pred.t_rel
        plan = self.mpc.solve(t, q0, qd0, t_hit, c_star, sp.n, sp.speed)
        adopt = adopt_plan(plan, self.plan if self.state == "SWING" else None)
        self.solve_times.append(plan.solve_time)
        self.strikes.append(StrikeRecord(t, t_hit, pred.pos, pred.vel, sp.v_out, sp.n, sp.speed, c_star, sp.T,
                                         sp.net_clearance, plan.solve_time, plan.success, plan.best_effort, adopt,
                                         plan.status, plan.impact_err))
        self.next_solve = t + mc["period"]
        if adopt:
            self.plan, self.t_hit, self.T, self.v_out, self.state = plan, plan.t_hit, sp.T, sp.v_out, "SWING"


@dataclass
class ReturnRun:
    logs: dict
    launch: Launch
    paddle_events: list
    table_events: list
    strikes: list
    outcome: str            # in | net | long | wide | no_hit | ...
    landing: np.ndarray | None
    target: np.ndarray
    z_at_net: float
    seed: int
    tau_max: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray


def run_return(cfg: dict, seed: int, launch: Launch | None = None, video=None, viewer=False, slowmo=1.0) -> ReturnRun:
    """One episode: launch a ball (seeded) at the robot and let it return it (true ball state)."""
    rc, jc, cc = cfg["rally"], cfg["juggle"], cfg["control"]
    sim = make_sim(cfg)
    m, d, ids = sim.m, sim.d, sim.ids
    mm = cfg["mismatch"]
    params = AeroParams.from_config(cfg)
    params_true = AeroParams.from_config(cfg, drag_coefficient=mm["true_drag_coefficient"])
    aero = BallAero(m, ids, params_true)
    contact = ContactModel(m, ids, aero, [paddle_surface(cfg, arm_paddle_state(ids), mm["true_paddle_restitution"]),
                                          table_surface(cfg, params_true)], cfg["contact"]["rewind_to_impact_time"])
    striker = Striker(sim, cfg, params)
    tc = cfg["table"]
    x_net = tc["near_edge_x"] + tc["length"] / 2
    z_net = tc["top_height"] + tc["net_height"] + cfg["ball"]["radius"]
    if launch is None:
        launch = Launcher(cfg, striker.predictor, striker.table, striker.x_plane).sample(np.random.default_rng(seed))
    aero.set_ball_state(m, d, launch.pos, launch.vel)
    mujoco.mj_forward(m, d)

    log = Logger()
    paddle_ev, table_ev = [], []
    st = {"outcome": None, "landing": None, "next_tick": 0.0, "z_net": np.nan, "prev_x": launch.pos[0],
          "prev_z": launch.pos[2]}
    code = {"WAIT": 0, "SWING": 1, "RETURN": 2}

    def control(m, d):
        t = d.time
        p, v, w = aero.ball_state(d)
        bounced = any(e.pos[0] < x_net for e in table_ev) and not paddle_ev
        if t + 1e-9 >= st["next_tick"]:
            if not paddle_ev:
                striker.tick(t, p, v, w, bounced)
            else:
                striker.tick(t, p, v, w, False)        # still advances SWING -> RETURN -> WAIT
            st["next_tick"] += jc["planner_period"]
        q_d, qd_d, qdd_d = striker.desired(t)
        tau, sat = computed_torque(m, d, ids, q_d, qd_d, qdd_d, cc["ct_kp"], cc["ct_kd"], cc["friction_ff"],
                                   cc["friction_smoothing"])
        d.ctrl[ids.arm_act] = tau
        pp, pn = paddle_pose(m, d, ids)
        log.log(t=t, ball_p=p, ball_v=v, q=d.qpos[ids.arm_qpos], qd=d.qvel[ids.arm_dof], q_d=q_d, qd_d=qd_d,
                tau=tau, sat=sat, paddle_p=pp, paddle_n=pn, state=code[striker.state])
        if paddle_ev and np.isnan(st["z_net"]) and (st["prev_x"] - x_net) * (p[0] - x_net) <= 0:
            a = (x_net - st["prev_x"]) / (p[0] - st["prev_x"]) if p[0] != st["prev_x"] else 0.0
            st["z_net"] = st["prev_z"] + a * (p[2] - st["prev_z"])
        st["prev_x"], st["prev_z"] = p[0], p[2]
        for ev in contact.step(m, d):
            (paddle_ev if ev.surface == "paddle" else table_ev).append(ev)
            if ev.surface == "table" and paddle_ev and st["outcome"] is None:
                st["landing"] = ev.pos[:2].copy()
                st["outcome"] = "in" if ev.pos[0] > x_net else "short"      # landed on our own half
        aero.apply(d)
        if st["outcome"] is None:
            if paddle_ev and not np.isnan(st["z_net"]) and st["z_net"] < z_net:
                st["outcome"] = "net"
            elif p[2] < tc["top_height"] - rc["fall_margin"]:
                st["outcome"] = "no_hit" if not paddle_ev else ("long" if p[0] > x_net else "short")
        done = st["outcome"] is not None
        if done and st["outcome"] == "in" and abs(st["landing"][1]) > tc["width"] / 2:
            st["outcome"] = "wide"
        return done and striker.state != "SWING"

    simulate(m, d, rc["max_episode_time"], control, cfg=cfg, video=video, viewer=viewer, slowmo=slowmo)
    if st["outcome"] is None:
        st["outcome"] = "timeout" if paddle_ev else "no_hit"
    return ReturnRun(log.arrays(), launch, paddle_ev, table_ev, striker.strikes, st["outcome"], st["landing"],
                     striker.target, st["z_net"], seed, ids.tau_max.copy(), ids.q_lo.copy(), ids.q_hi.copy())
