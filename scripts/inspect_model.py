"""Phase 0: inspect the OpenArm v2 MJCF shipped with `openarm-mujoco`.

Writes results/model_report.md with joint names/ranges, actuator types and
ctrl/force ranges, end-effector bodies/sites, link masses, total DOF, and the
gravity torque at the pedestal "home" keyframe. Optionally renders one PNG
(--snapshot) as a headless-rendering smoke test.

Usage:
    python scripts/inspect_model.py [--snapshot results/phase0_home.png]
"""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
from openarm_mujoco import v2 as openarm

REPO = Path(__file__).resolve().parents[1]

GAIN_NAMES = {int(v): k for k, v in mujoco.mjtGain.__members__.items()}
BIAS_NAMES = {int(v): k for k, v in mujoco.mjtBias.__members__.items()}
DYN_NAMES = {int(v): k for k, v in mujoco.mjtDyn.__members__.items()}
JNT_NAMES = {int(v): k for k, v in mujoco.mjtJoint.__members__.items()}


def name(m: mujoco.MjModel, obj: mujoco.mjtObj, i: int) -> str:
    return mujoco.mj_id2name(m, obj, i) or f"<{i}>"


def classify_actuator(m: mujoco.MjModel, i: int) -> str:
    """Classify an actuator from its gain/bias parameters.

    MuJoCo actuator force: f = gain * ctrl + bias, with
      position: gain = kp, bias = -kp*q - kv*qdot  (biastype AFFINE)
      motor:    gain = gear-scaled 1, bias = 0      (biastype NONE)
    """
    gaintype = int(m.actuator_gaintype[i])
    biastype = int(m.actuator_biastype[i])
    gain = m.actuator_gainprm[i]
    bias = m.actuator_biasprm[i]
    if biastype == int(mujoco.mjtBias.mjBIAS_NONE) and gaintype == int(mujoco.mjtGain.mjGAIN_FIXED):
        return "motor (torque)"
    if biastype == int(mujoco.mjtBias.mjBIAS_AFFINE) and np.isclose(bias[1], -gain[0]):
        return f"position servo (kp={gain[0]:g}, kv={-bias[2]:g})"
    return "other"


def joint_table(m: mujoco.MjModel) -> list[str]:
    rows = [
        "| id | joint | type | range [rad] | actuatorfrcrange [N·m] | damping | frictionloss | armature | qpos adr | dof adr |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for j in range(m.njnt):
        d = m.jnt_dofadr[j]
        lo, hi = m.jnt_range[j]
        rng = f"[{lo:.4f}, {hi:.4f}]" if m.jnt_limited[j] else "unlimited"
        frc = m.jnt_actfrcrange[j]
        frc_s = f"[{frc[0]:g}, {frc[1]:g}]" if m.jnt_actfrclimited[j] else "—"
        rows.append(
            f"| {j} | {name(m, mujoco.mjtObj.mjOBJ_JOINT, j)} | {JNT_NAMES[int(m.jnt_type[j])]} | {rng} | {frc_s} | "
            f"{m.dof_damping[d]:g} | {m.dof_frictionloss[d]:g} | {m.dof_armature[d]:g} | {m.jnt_qposadr[j]} | {d} |"
        )
    return rows


def actuator_table(m: mujoco.MjModel) -> list[str]:
    rows = [
        "| id | actuator | joint | class | dyn/gain/bias type | ctrlrange | forcerange [N·m] | gear |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        cr = m.actuator_ctrlrange[a]
        fr = m.actuator_forcerange[a]
        types = (
            f"{DYN_NAMES[int(m.actuator_dyntype[a])]}/{GAIN_NAMES[int(m.actuator_gaintype[a])]}/"
            f"{BIAS_NAMES[int(m.actuator_biastype[a])]}"
        )
        rows.append(
            f"| {a} | {name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)} | {name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)} | "
            f"{classify_actuator(m, a)} | {types} | [{cr[0]:.4f}, {cr[1]:.4f}] | "
            f"{'[%g, %g]' % tuple(fr) if m.actuator_forcelimited[a] else '—'} | {m.actuator_gear[a, 0]:g} |"
        )
    return rows


def body_table(m: mujoco.MjModel) -> list[str]:
    rows = ["| id | body | parent | mass [kg] | subtree mass [kg] |", "|---|---|---|---|---|"]
    for b in range(m.nbody):
        rows.append(
            f"| {b} | {name(m, mujoco.mjtObj.mjOBJ_BODY, b)} | {name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b])} | "
            f"{m.body_mass[b]:.4f} | {m.body_subtreemass[b]:.4f} |"
        )
    return rows


def site_table(m: mujoco.MjModel) -> list[str]:
    rows = ["| site | body | pos in body |", "|---|---|---|"]
    for s in range(m.nsite):
        rows.append(
            f"| {name(m, mujoco.mjtObj.mjOBJ_SITE, s)} | {name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[s])} | "
            f"{np.array2string(m.site_pos[s], precision=4)} |"
        )
    return rows


def gravity_torques(m: mujoco.MjModel, key: str) -> list[str]:
    """Static gravity torque per joint at a keyframe: qfrc_bias with qvel = 0."""
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    mujoco.mj_resetDataKeyframe(m, d, kid)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    rows = ["| joint | q [rad] | gravity torque [N·m] | limit [N·m] | usage |", "|---|---|---|---|---|"]
    for j in range(m.njnt):
        jname = name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if "finger" in jname:
            continue
        g = d.qfrc_bias[m.jnt_dofadr[j]]
        lim = m.jnt_actfrcrange[j, 1]
        rows.append(
            f"| {jname} | {d.qpos[m.jnt_qposadr[j]]:.4f} | {g:+.3f} | {lim:g} | {abs(g) / lim * 100:.1f}% |"
        )
    return rows


def ee_pose(m: mujoco.MjModel, key: str) -> list[str]:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key))
    mujoco.mj_forward(m, d)
    rows = []
    for s in ("left_ee_control_point", "right_ee_control_point", "arm_origin"):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
        if sid >= 0:
            z_axis = d.site_xmat[sid].reshape(3, 3)[:, 2]
            rows.append(
                f"- `{s}`: world pos {np.array2string(d.site_xpos[sid], precision=4)}, "
                f"site z-axis {np.array2string(z_axis, precision=3)}"
            )
    return rows


def render_snapshot(m: mujoco.MjModel, key: str, path: Path) -> str:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key))
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.6]
    cam.distance, cam.azimuth, cam.elevation = 2.0, 150.0, -20.0
    with mujoco.Renderer(m, height=480, width=640) as r:
        r.update_scene(d, camera=cam)
        img = r.render()
    import imageio.v3 as iio

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, img)
    return f"{path.relative_to(REPO)} ({img.shape[1]}x{img.shape[0]})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "model_report.md")
    ap.add_argument("--snapshot", type=Path, default=None, help="render the pedestal home pose to this PNG")
    args = ap.parse_args()

    bimanual_path = openarm.openarm_bimanual_xml()
    pedestal_path = openarm.openarm_pedestal_xml()
    m = mujoco.MjModel.from_xml_path(bimanual_path)
    mp = mujoco.MjModel.from_xml_path(pedestal_path)
    share_root = Path(bimanual_path).parent

    xmls = sorted(str(p.relative_to(share_root)) for p in share_root.rglob("*.xml"))
    act_kinds = {classify_actuator(m, a).split(" (")[0] for a in range(m.nu)}

    lines = [
        "# OpenArm MuJoCo model report (Phase 0)",
        "",
        f"- mujoco {mujoco.__version__}, openarm-mujoco {version('openarm-mujoco')}",
        f"- MJCF root: `{share_root}`",
        f"- Available MJCF files (v2 only; no v1, no single-arm file): {', '.join(f'`{x}`' for x in xmls)}",
        "",
        "## Summary (`openarm_bimanual.xml`)",
        "",
        f"- nq = {m.nq}, nv = {m.nv} (total DOF), nu = {m.nu} actuators, nbody = {m.nbody}, njnt = {m.njnt}",
        "- Per arm: 7 revolute arm joints (joint1–7) + 2 finger joints (finger_joint2 mirrors finger_joint1 via a joint equality)",
        f"- Timestep in bimanual file: {m.opt.timestep} s; pedestal scene: {mp.opt.timestep} s",
        f"- **Actuator type(s): {', '.join(sorted(act_kinds))}** — see table; these are NOT torque motors.",
        "- Torque limits are also declared per joint via `actuatorfrcrange` (DM8009: ±40, DM4340: ±27, DM4310: ±7 N·m).",
        "- Joint velocity limits are not specified in the MJCF.",
        f"- Total model mass (bimanual, no pedestal): {m.body_subtreemass[0]:.3f} kg",
        "",
        "## Joints",
        "",
        *joint_table(m),
        "",
        "## Actuators",
        "",
        *actuator_table(m),
        "",
        "## Bodies",
        "",
        *body_table(m),
        "",
        "## Sites (end-effector candidates)",
        "",
        *site_table(m),
        "",
        "End-effector body per arm: `openarm_{left,right}_ee_base_link` (child of link6, rotated by joint7), "
        "which carries the gripper (`ee_inner_finger`, `ee_outer_finger`) and site `{left,right}_ee_control_point` at its origin.",
        "",
        "## Pedestal scene, `home` keyframe",
        "",
        "Arms are mounted at z = 0.698 m on a static pedestal.",
        "",
        *ee_pose(mp, "home"),
        "",
        "Static gravity torque (qfrc_bias at qvel = 0) vs joint torque limit:",
        "",
        *gravity_torques(mp, "home"),
        "",
    ]

    if args.snapshot is not None:
        try:
            lines += ["## Render smoke test", "", f"- MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}: wrote "
                      + render_snapshot(mp, "home", args.snapshot), ""]
        except Exception as exc:  # report rather than crash: rendering is optional here
            lines += ["## Render smoke test", "", f"- FAILED (MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}): {exc}", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    print(f"actuator types: {sorted(act_kinds)}")


if __name__ == "__main__":
    main()
