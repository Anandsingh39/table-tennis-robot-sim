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

from tt.config import REPO, ready_paddle_pos

SCENE_XML = REPO / "assets" / "scene.xml"

PADDLE_BODY = "paddle"
PADDLE_SITE = "paddle_center"
BALL_BODY = "ball"
BALL_JOINT = "ball_free"
BALL_GEOM = "ball_geom"
TABLE_TOP_GEOM = "table_top"
PRESCRIBED_BODY = "paddle_prescribed"
PRESCRIBED_SITE = "paddle_prescribed_center"
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
    prescribed_mocap: int = -1  # mocap id of the test-only prescribed paddle (-1 if absent)
    prescribed_site: int = -1


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


def _add_motors(spec: mujoco.MjSpec, cfg: dict, prefix: str = "") -> None:
    rc = cfg["robot"]
    for i in range(1, rc["n_joints"] + 1):
        jname = f"{prefix}{rc['arm_prefix']}joint{i}"
        lo, hi = spec.joint(jname).actfrcrange      # stock per-joint torque limit
        act = spec.add_actuator(name=f"{jname}_motor", target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_motor()
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [lo, hi]


def _add_paddle(spec: mujoco.MjSpec, cfg: dict, prefix: str = "") -> None:
    """Paddle frame: origin at the flange, +x along the handle, +z = blade normal."""
    pc = cfg["paddle"]
    ee = spec.body(prefix + cfg["robot"]["ee_body"])
    L, R, t = pc["handle_length"], pc["blade_radius"], pc["blade_thickness"]
    paddle = ee.add_body(name=prefix + PADDLE_BODY, pos=pc["flange_pos"], quat=pc["flange_quat"])
    paddle.add_geom(
        name=prefix + "paddle_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[L / 2, pc["handle_half_width"], pc["handle_half_thickness"]], pos=[L / 2, 0, 0],
        mass=pc["handle_mass"], material="handle_wood", **NO_CONTACT,
    )
    paddle.add_geom(
        name=prefix + "paddle_blade", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[R, t / 2, 0], pos=[L + R, 0, 0], mass=pc["blade_mass"], material="blade_red", **NO_CONTACT,
    )
    paddle.add_site(name=prefix + PADDLE_SITE, pos=[L + R, 0, 0], size=[0.005, 0, 0], rgba=[0, 1, 0, 1])


def _add_prescribed_paddle(spec: mujoco.MjSpec, cfg: dict) -> None:
    """Test-only paddle blade on a mocap body (pose set by the test each step); body frame = paddle_center."""
    pc = cfg["paddle"]
    body = spec.worldbody.add_body(name=PRESCRIBED_BODY, mocap=True, pos=cfg["checks"]["prescribed_paddle_pos"])
    body.add_geom(name="prescribed_blade", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[pc["blade_radius"], pc["blade_thickness"] / 2, 0], material="blade_red", **NO_CONTACT)
    body.add_site(name=PRESCRIBED_SITE, size=[0.005, 0, 0], rgba=[0, 1, 0, 1])


def _add_ball(spec: mujoco.MjSpec, cfg: dict) -> None:
    bc = cfg["ball"]
    pos = ready_paddle_pos(cfg) + [0, 0, bc["init_offset_above_paddle"]]
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


def build_spec(cfg: dict, prescribed_paddle: bool = False) -> mujoco.MjSpec:
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

    oc = cfg.get("opponent", {})
    if oc.get("enabled", False):
        # second arm at the far end: the first arm rotated 180 deg about the vertical axis through the net centre
        x_mid = cfg["table"]["near_edge_x"] + cfg["table"]["length"] / 2
        base_o = mirror_point(base, x_mid)
        scene.worldbody.add_geom(name=oc["prefix"] + "stand", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hw, hw, base[2] / 2],
                                 pos=[base_o[0], base_o[1], base[2] / 2], material="stand", **NO_CONTACT)
        frame_o = scene.worldbody.add_frame(pos=base_o, quat=[0.0, 0.0, 0.0, 1.0])
        scene.attach(_arm_spec(cfg), prefix=oc["prefix"], frame=frame_o)
        _add_motors(scene, cfg, oc["prefix"])
        _add_paddle(scene, cfg, oc["prefix"])
    _add_ball(scene, cfg)
    _add_table(scene, cfg)
    if prescribed_paddle:
        _add_prescribed_paddle(scene, cfg)
    return scene


def mirror_point(p, x_mid: float) -> np.ndarray:
    """Rotation by 180 deg about the vertical axis through (x_mid, 0): the far robot's frame from the near one's."""
    p = np.asarray(p, float)
    return np.array([2 * x_mid - p[0], -p[1], p[2]])


def mirror_vector(v) -> np.ndarray:
    v = np.asarray(v, float)
    return np.array([-v[0], -v[1], v[2]])


def scene_ids(m: mujoco.MjModel, cfg: dict, prefix: str = "") -> SceneIds:
    """Ids for the arm whose element names start with prefix ("" = near arm, opponent.prefix = far arm)."""
    rc = cfg["robot"]
    names = tuple(f"{prefix}{rc['arm_prefix']}joint{i}" for i in range(1, rc["n_joints"] + 1))
    jid = [m.joint(n).id for n in names]
    act = np.array([m.actuator(f"{n}_motor").id for n in names])
    ball_j = m.joint(BALL_JOINT).id
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PRESCRIBED_BODY)
    return SceneIds(
        prescribed_mocap=int(m.body_mocapid[pid]) if pid >= 0 else -1,
        prescribed_site=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PRESCRIBED_SITE),
        joint_names=names,
        arm_qpos=np.array([m.jnt_qposadr[j] for j in jid]),
        arm_dof=np.array([m.jnt_dofadr[j] for j in jid]),
        arm_act=act,
        q_lo=m.jnt_range[jid, 0].copy(),
        q_hi=m.jnt_range[jid, 1].copy(),
        tau_max=m.actuator_ctrlrange[act, 1].copy(),
        ee_body=m.body(prefix + rc["ee_body"]).id,
        paddle_body=m.body(prefix + PADDLE_BODY).id,
        paddle_site=m.site(prefix + PADDLE_SITE).id,
        ball_body=m.body(BALL_BODY).id,
        ball_qpos=int(m.jnt_qposadr[ball_j]),
        ball_dof=int(m.jnt_dofadr[ball_j]),
        ball_geom=m.geom(BALL_GEOM).id,
        table_top_geom=m.geom(TABLE_TOP_GEOM).id,
    )


def build_model(cfg: dict, prescribed_paddle: bool = False) -> tuple[mujoco.MjModel, SceneIds]:
    m = build_spec(cfg, prescribed_paddle).compile()
    return m, scene_ids(m, cfg)
