"""M1 — Scene/robot model.

Builds the simulation scene with MjSpec from the stock OpenArm v2 MJCF:
  1. load `openarm_bimanual.xml` (openarm-mujoco), delete the left arm and the
     gripper fingers, delete the stock position-servo actuators;
  2. attach the single arm into assets/scene.xml at `robot.base_pos`;
  3. add one torque motor per arm joint (ctrlrange = the joint's stock
     actuatorfrcrange), a paddle welded to the wrist, a free ball and a table.

Collision filtering (contype/conaffinity bit masks):
  arm, table, floor : contype 1   -> arm collides with table/floor
  ball              : contype 2, conaffinity 2 -> collides ONLY with the floor
                      (floor conaffinity 3); ball-paddle and ball-table contact
                      are analytic (M3), never MuJoCo's soft contact.
  paddle, stand     : contype 0, conaffinity 0 (no MuJoCo contacts)
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from openarm_mujoco import v2 as openarm

from tt.config import REPO

SCENE_XML = REPO / "assets" / "scene.xml"

PADDLE_BODY = "paddle"
PADDLE_SITE = "paddle_center"
BALL_BODY = "ball"
BALL_JOINT = "ball_free"
BALL_GEOM = "ball_geom"
TABLE_TOP_GEOM = "table_top"
NO_CONTACT = dict(contype=0, conaffinity=0)


@dataclass(frozen=True)
class SceneIds:
    """Index lookups into MjModel/MjData for the arm, paddle and ball."""

    joint_names: tuple[str, ...]
    arm_qpos: np.ndarray        # (7,) qpos addresses of joint1..7
    arm_dof: np.ndarray         # (7,) dof addresses of joint1..7
    arm_act: np.ndarray         # (7,) actuator ids (torque motors) for joint1..7
    q_lo: np.ndarray            # (7,) joint lower limits [rad]
    q_hi: np.ndarray            # (7,) joint upper limits [rad]
    tau_max: np.ndarray         # (7,) motor torque limits [N·m]
    ee_body: int
    paddle_body: int
    paddle_site: int
    ball_body: int
    ball_qpos: int              # first of 7 qpos (pos xyz, quat wxyz)
    ball_dof: int               # first of 6 dofs (lin vel world, ang vel body-local)
    ball_geom: int
    table_top_geom: int


def _arm_spec(cfg: dict) -> mujoco.MjSpec:
    """Stock bimanual MJCF reduced to one arm with no gripper and no actuators."""
    rc = cfg["robot"]
    spec = mujoco.MjSpec.from_file(openarm.openarm_bimanual_xml())
    spec.delete(spec.body(rc["removed_arm_body"]))
    for name in rc["gripper_bodies"]:
        spec.delete(spec.body(name))
    for eq in list(spec.equalities):          # finger mimic constraints
        spec.delete(eq)
    for act in list(spec.actuators):          # stock position servos -> replaced by motors
        spec.delete(act)
    # The stock file sets cone=elliptic, impratio=10 for grasping; no grasping here, so use
    # MuJoCo defaults (and avoid the option conflict warning on attach).
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    keep = rc["arm_prefix"].removeprefix("openarm_").rstrip("_")   # "right"
    for mesh in list(spec.meshes):            # drop assets only the deleted bodies used
        if keep not in mesh.name or "finger" in mesh.name:
            spec.delete(mesh)
    return spec


def _add_motors(spec: mujoco.MjSpec, cfg: dict) -> None:
    rc = cfg["robot"]
    for i in range(1, rc["n_joints"] + 1):
        jname = f"{rc['arm_prefix']}joint{i}"
        lo, hi = spec.joint(jname).actfrcrange      # stock per-joint torque limit
        act = spec.add_actuator(name=f"{jname}_motor", target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_motor()
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [lo, hi]


def _add_paddle(spec: mujoco.MjSpec, cfg: dict) -> None:
    """Paddle frame: origin at the flange, +x along the handle, +z = blade normal."""
    pc = cfg["paddle"]
    ee = spec.body(cfg["robot"]["ee_body"])
    L, R, t = pc["handle_length"], pc["blade_radius"], pc["blade_thickness"]
    paddle = ee.add_body(name=PADDLE_BODY, pos=pc["flange_pos"], quat=pc["flange_quat"])
    paddle.add_geom(
        name="paddle_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[L / 2, pc["handle_half_width"], pc["handle_half_thickness"]], pos=[L / 2, 0, 0],
        mass=pc["handle_mass"], material="handle_wood", **NO_CONTACT,
    )
    paddle.add_geom(
        name="paddle_blade", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[R, t / 2, 0], pos=[L + R, 0, 0], mass=pc["blade_mass"], material="blade_red", **NO_CONTACT,
    )
    paddle.add_site(name=PADDLE_SITE, pos=[L + R, 0, 0], size=[0.005, 0, 0], rgba=[0, 1, 0, 1])


def _add_ball(spec: mujoco.MjSpec, cfg: dict) -> None:
    bc = cfg["ball"]
    pos = np.array(cfg["ready_pose"]["paddle_pos"]) + [0, 0, bc["init_offset_above_paddle"]]
    ball = spec.worldbody.add_body(name=BALL_BODY, pos=pos)
    ball.add_freejoint(name=BALL_JOINT)
    ball.add_geom(
        name=BALL_GEOM, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[bc["radius"], 0, 0],
        mass=bc["mass"], material="ball_orange", contype=2, conaffinity=2,
    )
    if bc["hollow_shell"]:
        inertia = 2.0 / 3.0 * bc["mass"] * bc["radius"] ** 2
        ball.explicitinertial = True
        ball.mass = bc["mass"]
        ball.ipos = [0, 0, 0]
        ball.inertia = [inertia] * 3


def _add_table(spec: mujoco.MjSpec, cfg: dict) -> None:
    tc = cfg["table"]
    L, W, H, th = tc["length"], tc["width"], tc["top_height"], tc["thickness"]
    cx = tc["near_edge_x"] + L / 2
    table = spec.worldbody.add_body(name="table", pos=[cx, 0, 0])
    table.add_geom(name=TABLE_TOP_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, W / 2, th / 2],
                   pos=[0, 0, H - th / 2], material="table_top")
    # visual markings: side lines, end lines, center line (2 cm wide; ITTF)
    lw, eps = 0.01, 0.0005
    for i, (x, y, hx, hy) in enumerate([
        (0, W / 2 - lw, L / 2, lw), (0, -W / 2 + lw, L / 2, lw),
        (L / 2 - lw, 0, lw, W / 2), (-L / 2 + lw, 0, lw, W / 2), (0, 0, L / 2, lw / 2),
    ]):
        table.add_geom(name=f"table_line{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, eps],
                       pos=[x, y, H + eps], material="table_line", **NO_CONTACT)
    hw = W / 2 + tc["net_overhang"]
    table.add_geom(name="net", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.001, hw, tc["net_height"] / 2],
                   pos=[0, 0, H + tc["net_height"] / 2], material="net", **NO_CONTACT)
    lh = tc["leg_half_width"]
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        table.add_geom(name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[lh, lh, (H - th) / 2],
                       pos=[sx * (L / 2 - 2 * lh), sy * (W / 2 - 2 * lh), (H - th) / 2],
                       material="stand", **NO_CONTACT)


def build_spec(cfg: dict) -> mujoco.MjSpec:
    pc = cfg["physics"]
    scene = mujoco.MjSpec.from_file(str(SCENE_XML))
    scene.option.timestep = pc["timestep"]
    scene.option.gravity = pc["gravity"]
    scene.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{pc['integrator'].upper()}")

    base = np.array(cfg["robot"]["base_pos"], dtype=float)
    hw = cfg["robot"]["stand_half_width"]
    scene.worldbody.add_geom(name="stand", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hw, hw, base[2] / 2],
                             pos=[base[0], base[1], base[2] / 2], material="stand", **NO_CONTACT)
    frame = scene.worldbody.add_frame(pos=base)
    scene.attach(_arm_spec(cfg), prefix="", frame=frame)

    _add_motors(scene, cfg)
    _add_paddle(scene, cfg)
    _add_ball(scene, cfg)
    _add_table(scene, cfg)
    return scene


def scene_ids(m: mujoco.MjModel, cfg: dict) -> SceneIds:
    rc = cfg["robot"]
    names = tuple(f"{rc['arm_prefix']}joint{i}" for i in range(1, rc["n_joints"] + 1))
    jid = [m.joint(n).id for n in names]
    act = np.array([m.actuator(f"{n}_motor").id for n in names])
    ball_j = m.joint(BALL_JOINT).id
    return SceneIds(
        joint_names=names,
        arm_qpos=np.array([m.jnt_qposadr[j] for j in jid]),
        arm_dof=np.array([m.jnt_dofadr[j] for j in jid]),
        arm_act=act,
        q_lo=m.jnt_range[jid, 0].copy(),
        q_hi=m.jnt_range[jid, 1].copy(),
        tau_max=m.actuator_ctrlrange[act, 1].copy(),
        ee_body=m.body(rc["ee_body"]).id,
        paddle_body=m.body(PADDLE_BODY).id,
        paddle_site=m.site(PADDLE_SITE).id,
        ball_body=m.body(BALL_BODY).id,
        ball_qpos=int(m.jnt_qposadr[ball_j]),
        ball_dof=int(m.jnt_dofadr[ball_j]),
        ball_geom=m.geom(BALL_GEOM).id,
        table_top_geom=m.geom(TABLE_TOP_GEOM).id,
    )


def build_model(cfg: dict) -> tuple[mujoco.MjModel, SceneIds]:
    m = build_spec(cfg).compile()
    return m, scene_ids(m, cfg)
