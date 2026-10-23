# OpenArm MuJoCo model report (Phase 0)

- mujoco 3.17.0, openarm-mujoco 2.3.0
- MJCF root: `/home/anandchauhan/Documents/CourseProjects/.venv/share/openarm_mujoco/v2`
- Available MJCF files (v2 only; no v1, no single-arm file): `cell/articulated_cell_scene.xml`, `cell/bottle_cell_scene.xml`, `cell/cell.xml`, `cell/demo.xml`, `cell/move_puck_cell_scene.xml`, `cell/peg_socket_cell_scene.xml`, `cell/tool_cell_scene.xml`, `cell/valve_cell_scene.xml`, `openarm_bimanual.xml`, `pedestal/articulated_scene.xml`, `pedestal/balance_scene.xml`, `pedestal/bottle_scene.xml`, `pedestal/catch_scene.xml`, `pedestal/move_puck_scene.xml`, `pedestal/openarm_pedestal.xml`, `pedestal/pedestal.xml`, `pedestal/peg_socket_scene.xml`, `pedestal/throw_multi_scene.xml`, `pedestal/throw_scene.xml`, `pedestal/tool_scene.xml`, `pedestal/valve_scene.xml`

## Summary (`openarm_bimanual.xml`)

- nq = 18, nv = 18 (total DOF), nu = 16 actuators, nbody = 21, njnt = 18
- Per arm: 7 revolute arm joints (joint1–7) + 2 finger joints (finger_joint2 mirrors finger_joint1 via a joint equality)
- Timestep in bimanual file: 0.002 s; pedestal scene: 0.001 s
- **Actuator type(s): position servo** — see table; these are NOT torque motors.
- Torque limits are also declared per joint via `actuatorfrcrange` (DM8009: ±40, DM4340: ±27, DM4310: ±7 N·m).
- Joint velocity limits are not specified in the MJCF.
- Total model mass (bimanual, no pedestal): 12.260 kg

## Joints

| id | joint | type | range [rad] | actuatorfrcrange [N·m] | damping | frictionloss | armature | qpos adr | dof adr |
|---|---|---|---|---|---|---|---|---|---|
| 0 | openarm_left_joint1 | mjJNT_HINGE | [-3.4907, 1.3963] | [-40, 40] | 1 | 0.2 | 0.0081 | 0 | 0 |
| 1 | openarm_left_joint2 | mjJNT_HINGE | [-3.3161, 0.1745] | [-40, 40] | 1 | 0.2 | 0.0081 | 1 | 1 |
| 2 | openarm_left_joint3 | mjJNT_HINGE | [-1.5708, 1.5708] | [-27, 27] | 0.9 | 0.1 | 0.16 | 2 | 2 |
| 3 | openarm_left_joint4 | mjJNT_HINGE | [0.0000, 2.4435] | [-27, 27] | 0.9 | 0.1 | 0.16 | 3 | 3 |
| 4 | openarm_left_joint5 | mjJNT_HINGE | [-1.5708, 1.5708] | [-7, 7] | 0.9 | 0.04 | 0.01 | 4 | 4 |
| 5 | openarm_left_joint6 | mjJNT_HINGE | [-0.7854, 0.7854] | [-7, 7] | 0.9 | 0.04 | 0.01 | 5 | 5 |
| 6 | openarm_left_joint7 | mjJNT_HINGE | [-1.5708, 1.5708] | [-7, 7] | 0.9 | 0.04 | 0.01 | 6 | 6 |
| 7 | openarm_left_finger_joint1 | mjJNT_HINGE | [0.0000, 0.7854] | [-7, 7] | 2 | 0 | 0.01 | 7 | 7 |
| 8 | openarm_left_finger_joint2 | mjJNT_HINGE | [0.0000, 0.7854] | [-7, 7] | 2 | 0 | 0.01 | 8 | 8 |
| 9 | openarm_right_joint1 | mjJNT_HINGE | [-1.3963, 3.4907] | [-40, 40] | 1 | 0.2 | 0.0081 | 9 | 9 |
| 10 | openarm_right_joint2 | mjJNT_HINGE | [-0.1745, 3.3161] | [-40, 40] | 1 | 0.2 | 0.0081 | 10 | 10 |
| 11 | openarm_right_joint3 | mjJNT_HINGE | [-1.5708, 1.5708] | [-27, 27] | 0.9 | 0.1 | 0.16 | 11 | 11 |
| 12 | openarm_right_joint4 | mjJNT_HINGE | [0.0000, 2.4435] | [-27, 27] | 0.9 | 0.1 | 0.16 | 12 | 12 |
| 13 | openarm_right_joint5 | mjJNT_HINGE | [-1.5708, 1.5708] | [-7, 7] | 0.9 | 0.04 | 0.01 | 13 | 13 |
| 14 | openarm_right_joint6 | mjJNT_HINGE | [-0.7854, 0.7854] | [-7, 7] | 0.9 | 0.04 | 0.01 | 14 | 14 |
| 15 | openarm_right_joint7 | mjJNT_HINGE | [-1.5708, 1.5708] | [-7, 7] | 0.9 | 0.04 | 0.01 | 15 | 15 |
| 16 | openarm_right_finger_joint1 | mjJNT_HINGE | [-0.7854, 0.0000] | [-7, 7] | 2 | 0 | 0.01 | 16 | 16 |
| 17 | openarm_right_finger_joint2 | mjJNT_HINGE | [-0.7854, 0.0000] | [-7, 7] | 2 | 0 | 0.01 | 17 | 17 |

## Actuators

| id | actuator | joint | class | dyn/gain/bias type | ctrlrange | forcerange [N·m] | gear |
|---|---|---|---|---|---|---|---|
| 0 | left_joint1_ctrl | openarm_left_joint1 | position servo (kp=230, kv=2.7) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-3.4907, 1.3963] | [-40, 40] | 1 |
| 1 | left_joint2_ctrl | openarm_left_joint2 | position servo (kp=230, kv=2.7) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-3.3161, 0.1745] | [-40, 40] | 1 |
| 2 | left_joint3_ctrl | openarm_left_joint3 | position servo (kp=190, kv=2.2) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-1.5708, 1.5708] | [-27, 27] | 1 |
| 3 | left_joint4_ctrl | openarm_left_joint4 | position servo (kp=190, kv=2.2) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [0.0000, 2.4435] | [-27, 27] | 1 |
| 4 | left_joint5_ctrl | openarm_left_joint5 | position servo (kp=30, kv=1.5) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-1.5708, 1.5708] | [-7, 7] | 1 |
| 5 | left_joint6_ctrl | openarm_left_joint6 | position servo (kp=30, kv=1.5) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-0.7854, 0.7854] | [-7, 7] | 1 |
| 6 | left_joint7_ctrl | openarm_left_joint7 | position servo (kp=30, kv=1.5) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-1.5708, 1.5708] | [-7, 7] | 1 |
| 7 | left_finger1_ctrl | openarm_left_finger_joint1 | position servo (kp=30, kv=0.2) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [0.0000, 0.7854] | [-7, 7] | 1 |
| 8 | right_joint1_ctrl | openarm_right_joint1 | position servo (kp=230, kv=2.7) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-1.3963, 3.4907] | [-40, 40] | 1 |
| 9 | right_joint2_ctrl | openarm_right_joint2 | position servo (kp=230, kv=2.7) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-0.1745, 3.3161] | [-40, 40] | 1 |
| 10 | right_joint3_ctrl | openarm_right_joint3 | position servo (kp=190, kv=2.2) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-1.5708, 1.5708] | [-27, 27] | 1 |
| 11 | right_joint4_ctrl | openarm_right_joint4 | position servo (kp=190, kv=2.2) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [0.0000, 2.4435] | [-27, 27] | 1 |
| 12 | right_joint5_ctrl | openarm_right_joint5 | position servo (kp=30, kv=1.5) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-1.5708, 1.5708] | [-7, 7] | 1 |
| 13 | right_joint6_ctrl | openarm_right_joint6 | position servo (kp=30, kv=1.5) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-0.7854, 0.7854] | [-7, 7] | 1 |
| 14 | right_joint7_ctrl | openarm_right_joint7 | position servo (kp=30, kv=1.5) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-1.5708, 1.5708] | [-7, 7] | 1 |
| 15 | right_finger1_ctrl | openarm_right_finger_joint1 | position servo (kp=30, kv=0.2) | mjDYN_NONE/mjGAIN_FIXED/mjBIAS_AFFINE | [-0.7854, 0.0000] | [-7, 7] | 1 |

## Bodies

| id | body | parent | mass [kg] | subtree mass [kg] |
|---|---|---|---|---|
| 0 | world | world | 0.0000 | 12.2600 |
| 1 | openarm_left_base_link | world | 1.1432 | 6.1300 |
| 2 | openarm_left_link1 | openarm_left_base_link | 1.1417 | 4.9868 |
| 3 | openarm_left_link2 | openarm_left_link1 | 0.2775 | 3.8451 |
| 4 | openarm_left_link3 | openarm_left_link2 | 1.0739 | 3.5676 |
| 5 | openarm_left_link4 | openarm_left_link3 | 0.6349 | 2.4937 |
| 6 | openarm_left_link5 | openarm_left_link4 | 0.6157 | 1.8589 |
| 7 | openarm_left_link6 | openarm_left_link5 | 0.4752 | 1.2432 |
| 8 | openarm_left_ee_base_link | openarm_left_link6 | 0.5283 | 0.7680 |
| 9 | openarm_left_ee_inner_finger | openarm_left_ee_base_link | 0.1260 | 0.1260 |
| 10 | openarm_left_ee_outer_finger | openarm_left_ee_base_link | 0.1137 | 0.1137 |
| 11 | openarm_right_base_link | world | 1.1432 | 6.1300 |
| 12 | openarm_right_link1 | openarm_right_base_link | 1.1417 | 4.9868 |
| 13 | openarm_right_link2 | openarm_right_link1 | 0.2775 | 3.8451 |
| 14 | openarm_right_link3 | openarm_right_link2 | 1.0739 | 3.5676 |
| 15 | openarm_right_link4 | openarm_right_link3 | 0.6349 | 2.4937 |
| 16 | openarm_right_link5 | openarm_right_link4 | 0.6157 | 1.8589 |
| 17 | openarm_right_link6 | openarm_right_link5 | 0.4752 | 1.2432 |
| 18 | openarm_right_ee_base_link | openarm_right_link6 | 0.5283 | 0.7680 |
| 19 | openarm_right_ee_inner_finger | openarm_right_ee_base_link | 0.1260 | 0.1260 |
| 20 | openarm_right_ee_outer_finger | openarm_right_ee_base_link | 0.1137 | 0.1137 |

## Sites (end-effector candidates)

| site | body | pos in body |
|---|---|---|
| arm_origin | world | [0. 0. 0.] |
| left_ee_control_point | openarm_left_ee_base_link | [0. 0. 0.] |
| right_ee_control_point | openarm_right_ee_base_link | [0. 0. 0.] |

End-effector body per arm: `openarm_{left,right}_ee_base_link` (child of link6, rotated by joint7), which carries the gripper (`ee_inner_finger`, `ee_outer_finger`) and site `{left,right}_ee_control_point` at its origin.

## Pedestal scene, `home` keyframe

Arms are mounted at z = 0.698 m on a static pedestal.

- `left_ee_control_point`: world pos [0.216  0.1535 0.478 ], site z-axis [-1.000e+00 -0.000e+00  3.268e-07]
- `right_ee_control_point`: world pos [ 0.216  -0.1535  0.478 ], site z-axis [-1.000e+00 -0.000e+00  3.268e-07]
- `arm_origin`: world pos [0.    0.    0.698], site z-axis [0. 0. 1.]

Static gravity torque (qfrc_bias at qvel = 0) vs joint torque limit:

| joint | q [rad] | gravity torque [N·m] | limit [N·m] | usage |
|---|---|---|---|---|
| openarm_left_joint1 | 0.0000 | -4.292 | 40 | 10.7% |
| openarm_left_joint2 | 0.0000 | +0.060 | 40 | 0.2% |
| openarm_left_joint3 | 0.0000 | +0.000 | 27 | 0.0% |
| openarm_left_joint4 | 1.5708 | +4.381 | 27 | 16.2% |
| openarm_left_joint5 | 0.0000 | -0.046 | 7 | 0.7% |
| openarm_left_joint6 | 0.0000 | +0.527 | 7 | 7.5% |
| openarm_left_joint7 | 0.0000 | -0.000 | 7 | 0.0% |
| openarm_right_joint1 | 0.0000 | +4.292 | 40 | 10.7% |
| openarm_right_joint2 | 0.0000 | -0.032 | 40 | 0.1% |
| openarm_right_joint3 | 0.0000 | +0.000 | 27 | 0.0% |
| openarm_right_joint4 | 1.5708 | +4.381 | 27 | 16.2% |
| openarm_right_joint5 | 0.0000 | +0.046 | 7 | 0.7% |
| openarm_right_joint6 | 0.0000 | -0.527 | 7 | 7.5% |
| openarm_right_joint7 | 0.0000 | +0.000 | 7 | 0.0% |

## Render smoke test

- MUJOCO_GL=egl: wrote results/phase0_home.png (640x480)
