#!/usr/bin/env python3
"""Generate the cuRobo planner config for RoboDojo's LiteArm asset.

Runs cuRobo's ``RobotBuilder`` on the composed URDF: fits collision spheres to
every link mesh, computes the self-collision ignore matrix, and writes
``curobo_tmp.yml`` in the layout ``env/planner_manager/curobo_planner.py``
expects (``robot_cfg.kinematics`` + ``planner.frame_bias``).  RoboDojo's
``utils/update_embodiment_config_path.py`` turns the ``${ASSETS_PATH}``
placeholder into ``curobo.yml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from curobo.robot_builder import RobotBuilder

#: Arm joints, in chain order. The gripper joints are left out of the planning
#: cspace (they stay at their URDF default, open).
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
DEFAULT_JOINT_POSITION = [0.0] * len(ARM_JOINTS)
#: Links kept out of the planning collision model. The finger prismatic joints
#: are not part of the planning cspace, and cuRobo's kinematics loader requires
#: every joint in the tree to be in the cspace; X5 likewise keeps its gripper
#: out (its collision links stop at link6).
EXCLUDED_LINKS = frozenset({"gripper_slider_link1", "gripper_slider_link2"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets" / "litearm",
    )
    parser.add_argument("--sphere-density", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()
    asset_dir = args.asset_dir.resolve()

    builder = RobotBuilder(
        urdf_path=str(asset_dir / "litearm.urdf"),
        asset_path=str(asset_dir),
        tool_frames=["tcp_link"],
    )
    print(f"[curobo] links: {builder._link_names}")
    print(f"[curobo] joints: {builder._joint_names}")

    builder.fit_collision_spheres(
        sphere_density=args.sphere_density,
        compute_metrics=True,
        # keep the base spheres on the tabletop
        clip_links={"base_link": ("z", 0.0)},
    )
    builder.compute_collision_matrix(num_samples=args.num_samples)
    config = builder.build()

    kinematics = {
        key: value
        for key, value in vars(config).items()
        if key not in ("device_cfg", "load_collision_spheres", "num_envs")
    }
    kinematics["format_version"] = 2.0
    kinematics["base_link"] = "base_link"
    kinematics["tool_frames"] = ["tcp_link"]
    kinematics["asset_root_path"] = "${ASSETS_PATH}/Assets/Robots/litearm"
    kinematics["urdf_path"] = "${ASSETS_PATH}/Assets/Robots/litearm/litearm.urdf"
    kinematics["grasp_contact_link_names"] = None
    kinematics["cspace"] = {
        "joint_names": list(ARM_JOINTS),
        "default_joint_position": list(DEFAULT_JOINT_POSITION),
        "cspace_distance_weight": [1.0] * len(ARM_JOINTS),
        "null_space_weight": [1.0] * len(ARM_JOINTS),
        "max_acceleration": 15.0,
        "max_jerk": 500.0,
        "velocity_scale": [1.0] * len(ARM_JOINTS),
        "position_limit_clip": 0.0,
    }

    # Keep the fingers out of the collision model (see EXCLUDED_LINKS).
    kinematics["collision_link_names"] = [
        name for name in kinematics["collision_link_names"] if name not in EXCLUDED_LINKS
    ]
    kinematics["collision_spheres"] = {
        name: spheres for name, spheres in kinematics["collision_spheres"].items() if name not in EXCLUDED_LINKS
    }
    kinematics["mesh_link_names"] = [
        name for name in kinematics["mesh_link_names"] if name not in EXCLUDED_LINKS
    ]
    kinematics["self_collision_ignore"] = {
        name: [other for other in others if other not in EXCLUDED_LINKS]
        for name, others in kinematics["self_collision_ignore"].items()
        if name not in EXCLUDED_LINKS
    }
    kinematics["self_collision_buffer"] = {
        name: value for name, value in kinematics["self_collision_buffer"].items() if name not in EXCLUDED_LINKS
    }

    data = {
        "robot_cfg": {"kinematics": kinematics},
        "planner": {"frame_bias": [0.0, 0.0, 0.0]},
    }
    out = asset_dir / "curobo_tmp.yml"
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    print(f"[curobo] wrote {out}")
    print(f"[curobo] {len(kinematics['collision_spheres'])} links have collision spheres")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
