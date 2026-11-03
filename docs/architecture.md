# Architecture

This document describes the components, data flow and boundaries of the LiteArm
stack in this repository.

## Components

| Component | Responsibility | Runs in |
| --- | --- | --- |
| PyRUA-Lean policy server | Writes robot programs, calls the chat model, drives the tools over XPolicyLab | `pyrualean` environment |
| LiteArm adapter | Robot API, kinematics, tools and knowledge files for the LiteArm station | `pyrualean` environment |
| XPolicyLab | WebSocket policy transport and adapter framework | Both hosts |
| RoboDojo evaluation client | Isaac Sim scenes, tasks, robot and observation management | `RoboDojo` environment |
| LiteArm robot asset | URDF, USD, collision and drive configuration, cuRobo template | Isaac Sim and cuRobo |
| `litearm-ai-pyrua-lean` CLI | Repository health checks | Any Python 3.10+ environment |

## Data flow

1. `scripts/run_eval.sh` calls `robodojo/scripts/robodojo.sh eval`, which starts
   the policy server and the simulator client on localhost.
2. The policy server loads the PyRUA-Lean LiteArm adapter, the SAM3 checkpoint
   and `deploy.yml`. It asks the configured chat endpoint for a program.
3. The program calls LiteArm tools. Each call travels over the XPolicyLab
   WebSocket to the evaluation client.
4. The client applies the action in Isaac Sim and returns the new observation.
5. When the episode ends, the client writes `_result.json` and the episode
   artifacts under `robodojo/eval_result/`.

In a split setup, steps 2 and 3 run on the policy host and steps 4 and 5 run on
the simulator host. The WebSocket connection is the only link between them.

## Repository layout

| Path | Contents |
| --- | --- |
| `pyrua-lean/` | Modified copy of PyRUA-Lean, including `integrations/xpolicylab/PyRUA_Lean` |
| `robodojo/` | Modified copy of RoboDojo; first-party source only, submodules are cloned by the install scripts |
| `integrations/xpolicylab/` | XPolicyLab overlay: robot dimensions, short-budget config and pose-check adapter |
| `assets/litearm/` | Built LiteArm USD asset, URDF, meshes, `robot_config.yml` and `curobo_tmp.yml` |
| `tools/litearm/` | Asset conversion, collision/drive normalization, FK checks and grasp validation |
| `scripts/` | Install, source, asset, evaluation and split-host scripts |
| `docs/assets/videos/` | Demo recordings |

## Upstream revisions

The copies in this repository are snapshots, not submodules. Their base commits
are:

| Project | Repository | Base commit |
| --- | --- | --- |
| PyRUA-Lean | <https://github.com/DAGroup-PKU/PyRUA-Lean> | `ffad6d4` |
| RoboDojo | <https://github.com/RoboDojo-Benchmark/RoboDojo> | `f57e163` |
| XPolicyLab | <https://github.com/XPolicyLab/XPolicyLab> | `5d0be14` |
| IsaacLab (submodule) | <https://github.com/yuechen0614/IsaacLab> | `afca7b0` |
| cuRobo (submodule) | <https://github.com/yuechen0614/curobo> | `d17b54c` |

`scripts/fetch_sources.sh` clones the three submodules at these revisions. It
applies the files in `integrations/xpolicylab/` on top of XPolicyLab and creates
the `policy/PyRUA_Lean` link to the PyRUA-Lean adapter. The vendored
`robodojo/scripts/install.sh` follows the same path when the repository has no
`.git` directory.

## LiteArm asset pipeline

The asset under `assets/litearm/` is the result of this pipeline:

1. `tools/litearm/build_litearm_assets.py` composes the official LiteArm SE and
   LiteGrip descriptions into `litearm.urdf` and copies the meshes. It needs the
   NEXFORM description repositories, which are not part of this repository.
2. `tools/litearm/convert_litearm_usd.py` imports the URDF into Isaac Sim and
   writes `litearm.usd` plus its configuration layers.
3. `tools/litearm/collision_normalization.py` moves the `CollisionAPI` from the
   STL wrapper Xforms onto the meshes. Without it, `UsdPhysics` finds zero
   collision shapes.
4. `tools/litearm/drive_normalization.py` changes the two gripper prismatic
   drives from acceleration to force, so the configured gains act as N/m and
   N s/m. The arm's angular drives keep their original mode.
5. The two finger joints are coupled with a PhysX mimic joint, so the fingers
   stay symmetric under load.
6. `tools/litearm/build_curobo_config.py` fits collision spheres to the meshes
   and writes `curobo_tmp.yml`. `scripts/fetch_assets.sh` replaces
   `${ASSETS_PATH}` and installs `curobo.yml` in the RoboDojo asset tree.

The prebuilt asset in `assets/litearm/` is the validated output of this pipeline.
The URDF and meshes are included so the asset can be rebuilt and so the
kinematics tests can run.

## Policy adapter

`pyrua-lean/integrations/xpolicylab/PyRUA_Lean/` implements the XPolicyLab
policy interface. `deploy.yml` selects the OpenAI-compatible runtime, a 100-turn
budget and a 5400-second episode budget. The adapter is linked into
`robodojo/XPolicyLab/policy/PyRUA_Lean` so the RoboDojo client can launch it as
an ordinary policy directory.

The knowledge files `pyrua-lean/src/pyrualean/knowledge/litearm.md` and
`litearm-novla.md` describe the station geometry, tool behaviour and grasp
procedure to the model. They are part of the policy, not documentation for the
operator.

## Boundaries and limitations

- This repository owns the LiteArm assets, adapter, environment configuration,
  evaluation scripts and the modified upstream snapshots. It does not own
  IsaacLab or cuRobo; both are cloned at pinned revisions.
- The RoboDojo asset dataset and the SAM3 checkpoint are downloaded, not stored
  here.
- The reference result is `stack_blocks` seed 0. `cover_blocks` was recorded but
  is not part of the verified result; other tasks are unverified on LiteArm.
- The simulator uses 640x480 colour cameras without depth. The reference 8 GB GPU
  runs the simulator only with reduced cameras; SAM3 needs a separate GPU.
- The LiteGrip opens to 8.7 cm, which excludes grasps on the 12 cm bowls.

## Security

The repository contains no API key. `config/env.sh` holds the endpoint
credentials and is git-ignored. `scripts/_lib.sh` and `scripts/run_*.sh` verify
that the variables are set without printing their values. The doctor command
reports only whether a variable is present.
