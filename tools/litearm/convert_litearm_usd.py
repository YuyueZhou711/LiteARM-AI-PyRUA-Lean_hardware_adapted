#!/usr/bin/env python3
"""Convert the composed LiteArm URDF to a self-contained USD with Isaac Sim.

Run inside Isaac Sim (from the RoboDojo conda env):

  OMNI_KIT_ACCEPT_EULA=YES python tools/litearm/convert_litearm_usd.py

The in-memory ``URDFParseAndImportFile`` route puts link visuals in top-level
``/visuals`` scopes, which do not resolve when only the USD's default prim is
referenced the way RoboDojo does. ``URDFImportRobot`` with a ``dest_path``
writes a proper asset: the robot as the default prim, meshes and materials
inside it.

IsaacLab's ``UrdfConverter`` would do the same but pins the 2.4.31 importer,
whose API is not in this Isaac Sim 5.1 installation, so the command is issued
directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 640, "height": 480})

import omni.kit.commands  # noqa: E402


def normalize_robot_collisions(usd: str) -> None:
    """Repair collision schemas and slider drive units, then verify physics shapes."""
    from pxr import Usd
    from collision_normalization import normalize_collisions, validate_collision_shapes
    from drive_normalization import normalize_gripper_drives

    stage = Usd.Stage.Open(usd)
    root = stage.GetDefaultPrim().GetPath()
    migrated = normalize_collisions(stage, root)
    drives = normalize_gripper_drives(stage, root)
    count = validate_collision_shapes(stage, root)
    stage.GetRootLayer().Save()
    print(f"[convert] repaired {migrated} collision wrappers and {drives} slider drives; {count} valid shapes", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets" / "litearm" / "litearm.urdf",
    )
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets" / "litearm" / "litearm.usd",
    )
    args = parser.parse_args()
    urdf = str(args.urdf.resolve())
    usd = str(args.usd.resolve())
    Path(usd).unlink(missing_ok=True)

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    # metres, not centimetres
    import_config.set_distance_scale(1.0)
    # the robot becomes the default prim of the exported file
    import_config.set_make_default_prim(True)
    # the scene supplies its own physics scene
    import_config.set_create_physics_scene(False)
    import_config.set_density(0.0)
    import_config.set_convex_decomp(False)
    import_config.set_collision_from_visuals(False)
    # keep every fixed link: gripper mount, tcp and camera frames are bodies
    import_config.set_merge_fixed_joints(False)
    # RoboDojo's ArticulationRootPropertiesCfg(fix_root_link=True) fixes the base
    import_config.set_fix_base(False)
    import_config.set_self_collision(True)
    # litearm.urdf declares q_left = q_right for the slider pair (the real
    # LiteGrip has one motor driving both fingers).  The importer converts the
    # URDF multiplier to PhysX's additive mimic relation and writes the API into
    # the exported USD; normalize_gripper_drives then disables the mimic joint's
    # own drive so the constraint alone moves it.
    import_config.set_parse_mimic(True)
    import_config.set_replace_cylinders_with_capsules(False)

    ok, robot = omni.kit.commands.execute("URDFParseFile", urdf_path=urdf, import_config=import_config)
    if not ok:
        print(f"[convert] URDFParseFile failed for {urdf}", flush=True)
        return 1
    ok, err = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_path=urdf,
        urdf_robot=robot,
        import_config=import_config,
        dest_path=usd,
    )
    print(f"[convert] URDFImportRobot ok={ok} err={err} -> {usd}", flush=True)
    if ok:
        normalize_robot_collisions(usd)
    return 0 if ok else 1


try:
    exit_code = main()
except BaseException:  # app.close() below otherwise hides the traceback
    traceback.print_exc()
    sys.stdout.flush()
    exit_code = 1
app.close()
raise SystemExit(exit_code)
