"""E2: two OpenArms rally against each other (cooperative lob rally, MPC swings).

Players: "A" at the near end (hits toward +x) and "B" at the far end (the same arm rotated 180 deg about
the net centre, so its joint-space solutions mirror A's). Each player runs the E1 pipeline with the
cooperative shot law (tt.shot.CoopShot): predict the ball after it bounces on its own half (M5),
aim it to bounce just past the net on the partner's half and then pass through the partner's hitting
zone (M6), swing with the MPC (M7), computed-torque control (M8). The serve is a launcher ball to A.

Rules tracked (true contact events, used only for scoring and turn-taking):
  a hit is legal if the receiver hits it after exactly one bounce on its own half;
  faults: net (ball centre crosses the net plane below net top + r), short (lands on the hitter's
  own half), double bounce on the receiver's half, ball falls off the table / is missed, wrong hitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from tt.aero import AeroParams, BallAero
from tt.contact import ContactModel, arm_paddle_state, paddle_surface, table_surface
from tt.control import computed_torque
from tt.kinematics import paddle_pose
from tt.launcher import Launcher
from tt.log_utils import Logger
from tt.model import SceneIds, scene_ids
from tt.mpc import SwingMPC, SwingPlan
from tt.planner import JointTrajectory, Quintic
from tt.predictor import BallPredictor
from tt.rally import nominal_geometry
from tt.render import simulate
from tt.shot import CoopShot
from tt.sim import make_sim


@dataclass
class Shot:
    player: str
    t_plan: float
    t_hit: float
    c_star: np.ndarray
    n: np.ndarray
    speed: float
    v_out: np.ndarray
    bounce_dist: float
    tilt_off_deg: float


@dataclass
class RallyPlayer:
    name: str
    sign: float                   # +1 near end (hits toward +x), -1 far end
    m: mujoco.MjModel
    ids: SceneIds
    q_ready: np.ndarray
    cfg: dict
    params: AeroParams
    predictor: BallPredictor
    table: object
    x_plane: float                # own hitting plane (ball-centre x at impact)
    coop: CoopShot
    h: float
    rng: np.random.Generator
    state: str = "WAIT"
    plan: SwingPlan | None = None
    shots: list[Shot] = field(default_factory=list)
    solve_times: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.mpc = SwingMPC(self.m, self.ids, self.cfg, self.q_ready)
        q, z = self.q_ready, np.zeros_like(self.q_ready)
        self.traj = JointTrajectory([Quintic.fit(0.0, 1.0, q, z, z, q, z, z)])
        self.t_hit, self.next_solve, self.y_target = np.inf, -np.inf, 0.0

    def desired(self, t: float):
        return self.plan(t) if self.state == "SWING" else self.traj(t)

    def tick(self, t, pos, vel, spin, my_turn: bool, bounced: bool) -> None:
        rc, mc = self.cfg["rally"], self.cfg["mpc"]
        if self.state == "SWING" and t >= self.plan.t_end:
            q_end, z = self.plan.q[-1], np.zeros_like(self.q_ready)
            self.traj = JointTrajectory([Quintic.fit(t, rc["return_time"], q_end, z, z, self.q_ready, z, z)])
            self.state = "RETURN"
        if self.state == "RETURN" and t >= self.traj.t_end:
            self.state = "WAIT"
        if self.state == "RETURN" or (self.state == "WAIT" and not my_turn):
            return
        if self.state == "SWING" and (t >= self.t_hit - mc["freeze_time"] or t < self.next_solve):
            return
        pred = self.predictor.plane_crossing(pos, vel, spin, 0, self.x_plane, -self.sign, self.table,
                                             require_bounce=True, already_bounced=bounced)
        if pred is None or (self.state == "WAIT" and pred.t_rel > rc["plan_horizon"]):
            return
        replan = self.state == "SWING"
        if not replan:
            self.coop.new_stroke()
            self.y_target = self.rng.uniform(*self.cfg["coop"].get("target_y_range", [0.0, 0.0]))
        cp = self.coop.plan(pred.pos, pred.vel, y_c=self.y_target, replan=replan)
        if cp is None:
            return
        c_star = pred.pos - self.h * cp.n
        q0, qd0, _ = self.desired(t)
        plan = self.mpc.solve(t, q0, qd0, t + pred.t_rel, c_star, cp.n, cp.speed)
        self.solve_times.append(plan.solve_time)
        self.next_solve = t + mc["period"]
        if plan.success:
            self.plan, self.t_hit, self.state = plan, plan.t_hit, "SWING"
            rec = Shot(self.name, t, plan.t_hit, c_star, cp.n, cp.speed, cp.v_out, cp.bounce_dist, cp.tilt_deg)
            if replan and self.shots and self.shots[-1].t_hit > t - 1.0:
                self.shots[-1] = rec
            else:
                self.shots.append(rec)


@dataclass
class RallyRun:
    logs: dict
    hits: list                    # (player, ContactEvent)
    bounces: list                 # ContactEvents on the table
    shots: dict                   # player -> list[Shot]
    solve_times: dict
    fault: str
    fault_by: str | None
    seed: int
    tau_max: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray


def run_rally(cfg: dict, seed: int, max_hits: int | None = None, video=None, viewer=False, slowmo=1.0) -> RallyRun:
    rc, jc, cc = cfg["rally"], cfg["juggle"], cfg["control"]
    max_hits = max_hits or cfg["e2"]["max_hits"]
    sim = make_sim(cfg)
    m, d = sim.m, sim.d
    ids_a, ids_b = sim.ids, scene_ids(m, cfg, cfg["opponent"]["prefix"])
    q_ready = sim.q_ready                         # the far arm is A rotated about the net: same joint angles
    d.qpos[ids_b.arm_qpos] = q_ready
    mujoco.mj_forward(m, d)

    params = AeroParams.from_config(cfg)
    mm = cfg["mismatch"]
    params_true = AeroParams.from_config(cfg, drag_coefficient=mm["true_drag_coefficient"])
    aero = BallAero(m, ids_a, params_true)
    contact = ContactModel(m, ids_a, aero, [
        paddle_surface(cfg, arm_paddle_state(ids_a), mm["true_paddle_restitution"], name="A"),
        paddle_surface(cfg, arm_paddle_state(ids_b), mm["true_paddle_restitution"], name="B"),
        table_surface(cfg, params_true)], cfg["contact"]["rewind_to_impact_time"])

    table, shot_nom, h, x_plane_a, _ = nominal_geometry(cfg, params)
    tc = cfg["table"]
    x_mid = tc["near_edge_x"] + tc["length"] / 2
    x_plane_b = 2 * x_mid - x_plane_a
    z_cross = cfg["hit"]["center"][2] + h * shot_nom.n_nom[2]
    z_net = tc["top_height"] + tc["net_height"] + cfg["ball"]["radius"]
    pred = BallPredictor(params, jc["predictor_dt"], rc["predictor_horizon"])
    players = {
        "A": RallyPlayer("A", +1.0, m, ids_a, q_ready, cfg, params, pred, table, x_plane_a,
                         CoopShot(cfg, params, pred, table, +1.0, x_plane_b, z_cross), h,
                         np.random.default_rng([seed, 2])),
        "B": RallyPlayer("B", -1.0, m, ids_b, q_ready, cfg, params, pred, table, x_plane_b,
                         CoopShot(cfg, params, pred, table, -1.0, x_plane_a, z_cross), h,
                         np.random.default_rng([seed, 3])),
    }
    serve = Launcher(cfg, pred, table, x_plane_a).sample(np.random.default_rng(seed))
    aero.set_ball_state(m, d, serve.pos, serve.vel)
    mujoco.mj_forward(m, d)

    log = Logger()
    hits, bounces = [], []
    st = {"receiver": "A", "hitter": "B", "rx_bounces": 0, "fault": None, "by": None, "next_tick": 0.0,
          "prev_x": serve.pos[0], "prev_z": serve.pos[2], "end_t": None}
    code = {"WAIT": 0, "SWING": 1, "RETURN": 2}

    def own_half(name, x):
        return (x - x_mid) * players[name].sign < 0

    def control(m, d):
        t = d.time
        p, v, w = aero.ball_state(d)
        if t + 1e-9 >= st["next_tick"]:
            for name, pl in players.items():
                pl.tick(t, p, v, w, my_turn=(st["fault"] is None and st["receiver"] == name),
                        bounced=(st["receiver"] == name and st["rx_bounces"] == 1))
            st["next_tick"] += jc["planner_period"]
        row = dict(t=t, ball_p=p, ball_v=v)
        for name, pl in players.items():
            q_d, qd_d, qdd_d = pl.desired(t)
            tau, sat = computed_torque(m, d, pl.ids, q_d, qd_d, qdd_d, cc["ct_kp"], cc["ct_kd"], cc["friction_ff"],
                                       cc["friction_smoothing"])
            d.ctrl[pl.ids.arm_act] = tau
            pp, _ = paddle_pose(m, d, pl.ids)
            row.update({f"q_{name}": d.qpos[pl.ids.arm_qpos], f"tau_{name}": tau, f"sat_{name}": sat,
                        f"paddle_{name}": pp, f"state_{name}": code[pl.state]})
        log.log(**row)
        # net check for the ball in flight after a hit
        if hits and st["fault"] is None and (st["prev_x"] - x_mid) * (p[0] - x_mid) <= 0 and p[0] != st["prev_x"]:
            a = (x_mid - st["prev_x"]) / (p[0] - st["prev_x"])
            if st["prev_z"] + a * (p[2] - st["prev_z"]) < z_net:
                st["fault"], st["by"] = "net", st["hitter"]
        st["prev_x"], st["prev_z"] = p[0], p[2]
        for ev in contact.step(m, d):
            if st["fault"] is not None:
                continue
            if ev.surface in players:
                if ev.surface != st["receiver"]:
                    st["fault"], st["by"] = "wrong hitter", ev.surface
                elif st["rx_bounces"] != 1:
                    st["fault"], st["by"] = "volley" if st["rx_bounces"] == 0 else "double bounce", ev.surface
                else:
                    hits.append((ev.surface, ev))
                    st["hitter"], st["receiver"], st["rx_bounces"] = ev.surface, ("B" if ev.surface == "A" else "A"), 0
            else:
                bounces.append(ev)
                if own_half(st["hitter"], ev.pos[0]):
                    st["fault"], st["by"] = "short", st["hitter"]
                else:
                    st["rx_bounces"] += 1
                    if st["rx_bounces"] > 1:
                        st["fault"], st["by"] = "double bounce", st["receiver"]
        aero.apply(d)
        if st["fault"] is None and p[2] < tc["top_height"] - rc["fall_margin"]:
            st["fault"], st["by"] = ("missed", st["receiver"]) if st["rx_bounces"] == 1 else ("out", st["hitter"])
        if st["fault"] is None and len(hits) >= max_hits and st["end_t"] is None:
            st["fault"], st["by"] = "max hits", None
        if st["fault"] is not None and st["end_t"] is None:
            st["end_t"] = t + cfg["e2"]["tail_time"]
        return st["end_t"] is not None and t >= st["end_t"]

    t_max = cfg["e2"]["time_per_hit"] * (max_hits + 1) + cfg["e2"]["tail_time"]
    simulate(m, d, t_max, control, cfg=cfg, video=video, viewer=viewer, slowmo=slowmo)
    return RallyRun(log.arrays(), hits, bounces, {k: p.shots for k, p in players.items()},
                    {k: p.solve_times for k, p in players.items()}, st["fault"] or "time limit", st["by"], seed,
                    ids_a.tau_max.copy(), ids_a.q_lo.copy(), ids_a.q_hi.copy())
