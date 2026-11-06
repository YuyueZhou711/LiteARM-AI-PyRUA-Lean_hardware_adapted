# LiteArm / LiteGrip robot description provenance

`litearm.urdf` is a composed description packaged for the real-hardware path.
It is not read from a RoboDojo asset directory at runtime.

## Arm geometry

- Source: `https://github.com/nexform-tech/litearm-description`
- Path: `SE/litearm_description/urdf/litearm_description.urdf`
- Commit used for verification: `b8cc7d9` (local checkout at
  `/home/hzh/Code/NEXTFORM/IsaacSim/litearm-description`)
- Package version: `1.0.0`
- License: BSD (declared in `SE/litearm_description/package.xml`)

## Gripper geometry

- Source: `https://github.com/nexform-tech/litegrip-description`
- Path: `litegrip_description/urdf/litegrip_description.urdf.xacro`
- Tag used for verification: `v2.0.0`
- License: Apache-2.0
- The measured per-finger stroke default is `0.042726 m`; the measured closed
  clearance is about `1.508 mm` and the measured open opening about
  `86.960 mm`.  Those are the official package's measured values, not a
  station calibration for every unit.

## Station mount

The arm and gripper descriptions do not define a robot-specific flange-to-gripper
mount.  The composed file uses the nominal integration dimensions
`gripper_mount = (0, 0, 0.0641) m` from `ee_link` and
`tcp_link = (0, 0, 0.0665) m, rpy = (-pi/2, -pi/2, 0)` from the gripper base.
These values are a description/collision default only.

**Execution does not use this URDF mount as measured truth.**  It uses
`calibration.arms.<arm>.flange_to_tcp` and refuses live motion when the
calibration is not marked `executable=true`.  A station must measure its own
mount and TCP; see `docs/robots/litearm-real.md`.

## Meshes

Mesh files are referenced by `package://` URIs and are not bundled.  Collision
checks in `pyrualean.hardware.kinematics` use conservative capsules and finger
boxes, not mesh-exact collision.
