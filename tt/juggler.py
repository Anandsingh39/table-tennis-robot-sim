"""High-level juggling controller: chains M5 (predictor) -> M6 (juggling law) -> M7 (planner).

State machine, ticked at the planner rate (100 Hz):
  POST      after an impact (or at start): the arm follows the current plan (follow-through,
            then return toward the ready pose). Each tick predicts the next impact; once it is
            closer than plan_horizon -> plan a stroke, go to APPROACH.
  APPROACH  re-plan every tick from the current desired state (q_d stays C2-continuous) to the
            impact state for the latest prediction, until freeze_time before impact (then the
            plan is frozen). When the planned impact time has passed -> POST.
Impact state: ball centre p* and v_in from M5; n, v_p.n from M6; paddle centre c* = p* - h n
(h = blade half-thickness + ball radius, so the paddle face touches the ball at p*); joint
state (q*, q̇*) from IK + task-Jacobian pseudo-inverse (M7) with v_p = (v_p.n) n.
The juggler only sees the ball state it is given (true state in Phase 5, estimate in Phase 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from tt.aero import AeroParams
from tt.juggle_law import ImpactTargetingLaw
from tt.planner import JointTrajectory, Quintic, impact_joint_state, plan_stroke
from tt.predictor import BallPredictor
from tt.sim import Sim


@dataclass
class PlanRecord:
    t: float            # when the plan was made
    t_hit: float        # planned impact time
    p_star: np.ndarray  # predicted ball centre at impact
    v_in: np.ndarray
    v_out: np.ndarray   # desired outgoing velocity
    n: np.ndarray
    speed: float        # paddle normal speed


@dataclass
class Juggler:
    sim: Sim
    cfg: dict
    params: AeroParams              # nominal ball model used by predictor and law
    state: str = "POST"
    t_hit: float = np.inf
    q_star: np.ndarray | None = None
    plans: list[PlanRecord] = field(default_factory=list)

    def __post_init__(self):
        jc, pc = self.cfg["juggle"], self.cfg["paddle"]
        self.h_slab = pc["blade_thickness"] / 2 + self.cfg["ball"]["radius"]
        self.z_impact = self.cfg["hit"]["center"][2] + self.h_slab     # ball centre height at impact
        self.predictor = BallPredictor(self.params, jc["predictor_dt"], jc["predictor_horizon"])
        self.law = ImpactTargetingLaw.from_config(self.cfg, self.params, jc["predictor_dt"])
        q = self.sim.q_ready
        z = np.zeros_like(q)
        self.traj = JointTrajectory([Quintic.fit(0.0, 1.0, q, z, z, q, z, z)])    # hold the ready pose
        self.scratch = mujoco.MjData(self.sim.m)

    def desired(self, t: float):
        return self.traj(t)

    def tick(self, t: float, pos, vel, spin) -> None:
        jc, pl = self.cfg["juggle"], self.cfg["planner"]
        if self.state == "APPROACH" and t >= self.t_hit:
            self.state = "POST"
        if self.state == "APPROACH" and self.t_hit - t <= jc["freeze_time"]:
            return
        pred = self.predictor.crossing(pos, vel, spin, self.z_impact)
        if pred is None or (self.state == "POST" and pred.t_rel > jc["plan_horizon"]):
            return
        n, s, v_out = self.law(pred.pos, pred.vel)
        c_star = pred.pos - self.h_slab * n
        seed = self.sim.q_ready if self.q_star is None else self.q_star
        q_star, qd_star, _ = impact_joint_state(self.sim.m, self.scratch, self.sim.ids, self.cfg, c_star, n, s * n, seed)
        t_hit = t + pred.t_rel
        q0, qd0, qdd0 = self.traj(t)
        self.traj = plan_stroke(t, q0, qd0, qdd0, t_hit, q_star, qd_star, pl["return_time"], self.sim.q_ready)
        self.q_star, self.t_hit, self.state = q_star, t_hit, "APPROACH"
        self.plans.append(PlanRecord(t, t_hit, pred.pos, pred.vel, v_out, n, s))
