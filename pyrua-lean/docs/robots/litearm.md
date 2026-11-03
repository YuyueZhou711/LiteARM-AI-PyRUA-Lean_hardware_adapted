# LiteArm

The [RoboDojo](robodojo.md) benchmark with two **LiteArm SE** arms (NEXFORM ROBOTICS), each
with a **LiteGrip** parallel gripper: seven joints per arm, driven as joint actions through
RoboDojo's own evaluation client. The policy side is the same PyRUA-Lean adapter; only the
embodiment differs.

## Overview

| | |
|---|---|
| **Robot** | Two LiteArm SE arms, 7 DoF each, with LiteGrip two-finger grippers |
| **Benchmark** | [RoboDojo](robodojo.md), the same 54 tasks and layouts |
| **Env cfg** | `litearm` (`env_cfg/litearm.yml`; robot config `dual_litearm`) |
| **Action** | Joint targets: `left_arm_joint_state` / `right_arm_joint_state` (7 each), normalized gripper opening (1 each) |
| **Observation** | Three colour cameras (`cam_head`, `cam_left_wrist`, `cam_right_wrist`) and both arms' state; no depth image |
| **VLA** | none: the primitive set is `no-vla` (there is no `pi0_pick`) |

The robot class is `pyrualean.litearm.LitearmRobot`; the host is
`pyrualean.hosts.xpolicylab_litearm`, with the primitives in
`pyrualean.hosts.litearm_tools` and the seven-joint kinematics in
`pyrualean.hosts.litearm_kinematics`.

## What differs from the X5 station

- Seven arm joints (`joint1` .. `joint7`), not six. `joint4` has an asymmetric limit
  (`-3.089 .. +0.035 rad`); `joint6` and `joint7` stop at `+-1.571 rad`.
- The reported end-effector pose is `tcp_link`, the grasp point itself (the middle of the
  LiteGrip finger pads): there is no TCP offset.
- The LiteGrip fingers are long: they span 4.6 cm either side of the grasp point along the
  pointing direction, so a top grasp puts the grasp point about 1 cm above the object's top
  and never lower than `table + 4.6 cm` (0.812 m). For a **hollow bowl** aim at the **rim
  wall** (`width_m/2 - 1.5 mm` from the measured centre) at the rim top; a close width of
  0.3–1.5 cm is the wall, 2–5 cm means the inner slope; lift and `locate` to confirm.
- The arms start folded forward over the table, close to the X5 station's start (grasp
  point ≈ `[±0.30, -0.20, 0.95]`, `pitch ≈ 10°`, `yaw ≈ -10°`); the wrist cameras see the
  table at reset. The URDF zero pose is still straight up; the robot's `init_state` is folded.
- Gripper width at a measured opening: `0.0020 + 0.0850 x opening` metres across the bare
  finger faces of the official finger meshes (no added pad geometry; the soft pad is
  represented by their friction). The faces open to 8.7 cm (the real gripper closes
  to 0 and opens to 8.7 cm).

## Real hardware (litearm-python / litegrip-python)

The simulator's TCP is the **pad middle** of the LiteGrip.  The two official SDKs control
different frames, so a real station needs these settings:

```python
arm.set_tcp_offset((0.0, 0.0, 0.1306), (-1.5707963, -1.5707963, 0.0))
```

That is the pad middle in the **flange** frame (0.0641 m gripper mount + 0.0665 m pad
middle, rpy of the `tcp_link`); measure it with `examples/08_tcp_offset.py` if the real
mount differs.  The SDK otherwise drives the flange (`ee_link`) and every pose is 13 cm /
90 degrees away from the simulator's grasp point.

- The SDK's `home()` goes to the fully extended zero pose; this station starts folded.
  After `reset()`/`enable()`, use `movej(start_q)` with the station's start joints, not
  `arm.home()`.
- Soft limits are per unit: `arm.params.get_joint_param(i).value.q_min/q_max`; check them
  against the URDF's limits.
- Grasp with `gripper.grasp(force_n=20.0, hold_s=...)` (`force_n x 0.1 N.m`, **not**
  calibrated); `close()` takes no force and uses `torque_limit_nm=3.5` (35 N).

## Installation

The assets live in RoboDojo's `Assets/Robots/litearm/` (`litearm.urdf`, `litearm.usd` plus a
`configuration/` directory, `robot_config.yml`, `curobo.yml`). They are built from the official
NEXFORM ROBOTICS descriptions by the scripts under `tools/litearm/` of the PyRUA-Lean
workspace; see `HANDOFF.md` §8 for the exact commands. The PyRUA-Lean side needs nothing
beyond the normal RoboDojo installation.

`env_cfg/litearm.yml` reuses RoboDojo's evaluation layouts through
`Assets/Eval_Layout/RoboDojo/litearm -> arx_x5` (object placements do not depend on the robot).

## Run a Task

The official entry point, exactly as for the X5 station but with `--env-cfg litearm`:

```bash
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/PyRUA_Lean --task stack_bowls \
    --ckpt no-checkpoint --policy-env <policy env> --eval-env <simulator env> \
    --env-cfg litearm --action-type joint --seed 0 --eval-num 1
```

Split evaluation uses `robodojo.sh server` on the policy machine and `robodojo.sh client` on
the simulator machine, both with `--env-cfg litearm`. The adapter chooses the LiteArm robot and
host from `env_cfg_type`; the `robot:` key in `deploy.yml` stays at the X5 default for every
other env cfg.

## Wiring Check

No simulator, no model call:

```bash
cd XPolicyLab/policy/PyRUA_Lean
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint litearm joint 0 0 0 pyrualean RoboDojo
```

## Primitives

- `move_to(xyz, arm, gripper, pitch, yaw, roll, path)`: the grasp point to a world position and
  orientation, solved by the host's seven-joint IK. `pitch` 90 points straight down, -90
  straight up; an angle left out keeps its current value.
- `set_gripper(arm, gripper, opening=None)`: close or open the LiteGrip until the fingers
  stop; the result says whether something is held and how wide it is.
- `home(arm="both")`: return an arm to the joints it started with; most tasks count as done
  only with both arms back and both grippers open.
- `locate(text_prompt, camera)`: measure an object in 3D from its masks in the colour cameras
  (centre, top, footprint, heading).
- `segment` / `back_project`: one mask and one world point.
- `camera_meta`, `show`, `image`, `state`, `task`: as for the X5 station.

??? note "API reference: what the agent reads, generated from `pyrualean.litearm` when these docs are built"

    <!-- api-reference: litearm primitives=no-vla -->
