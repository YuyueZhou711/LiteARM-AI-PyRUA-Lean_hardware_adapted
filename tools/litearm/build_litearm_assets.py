#!/usr/bin/env python3
"""Compose the LiteArm SE + LiteGrip robot description used by RoboDojo.

Inputs (all official NEXFORM ROBOTICS repositories, already checked out):
  * LiteArm SE URDF + meshes      : litearm-description/SE/litearm_description
  * LiteGrip URDF + meshes        : litegrip-description/litegrip_description
    (or litegrip-isaacsim/gripper2, whose meshes are byte-identical)

Output (RoboDojo asset directory):
  <out>/litearm.urdf
  <out>/meshes/arm/*.STL
  <out>/meshes/gripper/*.STL

The composition is:
  LiteArm base_link -> joint1..joint7 -> ee_link
  ee_link --fixed--> gripper_base (LiteGrip base_link)
  gripper_base --prismatic--> gripper_slider_link1 / gripper_slider_link2
  gripper_base --fixed--> tcp_link   (x = approach, y = finger closing axis,
                                      origin at the middle of the finger pads)
  ee_link --fixed--> camera_link     (wrist camera mount; x = approach, same
                                      orientation as tcp_link)

Mount geometry (from the URDF / mesh bounding boxes):
  * LiteArm ee_link +z is the tool axis; its flange face is the ee_link origin.
  * LiteGrip body spans z = -0.0641 .. +0.026 in its base frame, fingers
    0.0205 .. 0.1125; so the rear mounting face is z = -0.0641.
  * Hence gripper_base sits 0.0641 m along ee_link +z; the TCP is a further
    0.0665 m (middle of the finger-pad span).
  * LiteGrip x is the finger axis, so it is mounted with its x on ee_link x:
    at zero pose the fingers then close along the arm's x axis, like the X5.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

#: Distance from the LiteArm ee_link origin to the LiteGrip base frame.
GRIPPER_MOUNT_Z = 0.0641
#: Middle of the finger pads above the LiteGrip base frame (mesh z 0.0205..0.1125).
GRIPPER_PAD_Z = 0.0665
#: Wrist camera position in the ee_link frame and its orientation in degrees.
CAMERA_POS = (0.0, -0.06, 0.0)
CAMERA_RPY = (0.0, -70.0, 0.0)
#: Per-finger stroke of the LiteGrip: the official product manual gives 85.0 mm
#: total / 42.5 mm per finger.  (The earlier 0.042726 m was the CAD mesh stroke;
#: the cabinet's controller uses 42.5 mm, and this is the value the sim uses.)
GRIPPER_STROKE = 0.0425


def parse_urdf(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def mesh_name(link: ET.Element) -> str | None:
    node = link.find("visual/geometry/mesh")
    if node is None:
        return None
    return node.get("filename")


def rewrite_meshes(link: ET.Element, prefix: str) -> None:
    """Point every mesh filename at ``meshes/<prefix>/<basename>``."""
    for tag in ("visual", "collision"):
        node = link.find(f"{tag}/geometry/mesh")
        if node is not None:
            name = Path(node.get("filename", "")).name
            node.set("filename", f"meshes/{prefix}/{name}")


def add_fixed_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": fmt3(xyz), "rpy": fmt3(rpy)})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def fmt3(values: tuple[float, float, float]) -> str:
    return " ".join(f"{float(v):.10g}" for v in values)


def frame_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy (Rz(yaw) Ry(pitch) Rx(roll)) of a rotation matrix, **radians**."""
    # URDF <origin rpy> is radians.  Its fixed-axis rpy is scipy's extrinsic "xyz":
    # from_euler("xyz", [roll, pitch, yaw]) computes Rz(yaw) Ry(pitch) Rx(roll).  (Uppercase
    # "XYZ" is intrinsic and gives the reverse product; degrees would be read as radians.)
    rpy = Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)
    check = Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()
    assert np.allclose(check, matrix, atol=1e-9), (rpy, check, matrix)
    return tuple(float(v) for v in rpy)


def tcp_rotation() -> np.ndarray:
    """TCP axes expressed in the LiteGrip base frame.

    x = approach (+z of the gripper), y = finger closing axis (+x of the
    gripper), z = x cross y (+y of the gripper).
    """
    return np.column_stack(
        [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
    )


def empty_link(name: str, mass: float = 0.001) -> ET.Element:
    """A massless-in-practice link so the importer keeps it as a rigid body."""
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass:g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "1e-7",
            "ixy": "0",
            "ixz": "0",
            "iyy": "1e-7",
            "iyz": "0",
            "izz": "1e-7",
        },
    )
    return link


def build(arm_urdf: Path, arm_meshes: Path, gripper_urdf: Path, gripper_meshes: Path, out: Path) -> Path:
    arm = parse_urdf(arm_urdf)
    grip = parse_urdf(gripper_urdf)

    robot = ET.Element("robot", {"name": "litearm_description"})

    # The official URDF names both the arm links and the joints joint1..joint7.
    # cuRobo's parser drops a joint whose name matches a link name, so rename the
    # links to link1..link7 (X5's convention) and keep the joint names.
    link_rename = {f"joint{i}": f"link{i}" for i in range(1, 8)}

    # --- arm links and joints, unchanged except for the mesh paths ----------
    for link in arm.findall("link"):
        name = link.get("name")
        if name in link_rename:
            link.set("name", link_rename[name])
        rewrite_meshes(link, "arm")
        robot.append(link)
    for joint in arm.findall("joint"):
        for tag in ("parent", "child"):
            node = joint.find(tag)
            name = node.get("link")
            if name in link_rename:
                node.set("link", link_rename[name])
        robot.append(joint)

    # --- gripper: rename base_link -> gripper_base --------------------------
    rename = {"base_link": "gripper_base"}
    for link in grip.findall("link"):
        original = link.get("name")
        link.set("name", rename.get(original, original))
        rewrite_meshes(link, "gripper")
        robot.append(link)
    for joint in grip.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        parent.set("link", rename.get(parent.get("link"), parent.get("link")))
        child.set("link", rename.get(child.get("link"), child.get("link")))
        # The LiteGrip URDF ships the CAD stroke (0.067) that makes the fingers
        # overlap; use the measured per-finger stroke instead.
        limit = joint.find("limit")
        if joint.get("type") == "prismatic" and limit is not None:
            limit.set("upper", f"{GRIPPER_STROKE:.6g}")
        robot.append(joint)

    # --- mount the gripper on the arm --------------------------------------
    add_fixed_joint(
        robot,
        "gripper_mount",
        "ee_link",
        "gripper_base",
        (0.0, 0.0, GRIPPER_MOUNT_Z),
        (0.0, 0.0, 0.0),
    )

    # --- TCP frame ----------------------------------------------------------
    robot.append(empty_link("tcp_link"))
    add_fixed_joint(
        robot,
        "tcp_link_joint",
        "gripper_base",
        "tcp_link",
        (0.0, 0.0, GRIPPER_PAD_Z),
        frame_rpy(tcp_rotation()),
    )

    # --- wrist camera mount -------------------------------------------------
    # Oriented like tcp_link (x = approach, y = finger axis) so the camera's
    # ori in robot_config is a plain forward/down tilt.
    robot.append(empty_link("camera_link", mass=0.005))
    add_fixed_joint(
        robot,
        "camera_link_joint",
        "ee_link",
        "camera_link",
        CAMERA_POS,
        frame_rpy(tcp_rotation()),
    )

    # --- write --------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    meshes = out / "meshes"
    for prefix, source in (("arm", arm_meshes), ("gripper", gripper_meshes)):
        target = meshes / prefix
        target.mkdir(parents=True, exist_ok=True)
        for stl in sorted(source.glob("*.STL")):
            shutil.copy2(stl, target / stl.name)

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    urdf_out = out / "litearm.urdf"
    tree.write(urdf_out, encoding="utf-8", xml_declaration=True)
    print(f"[build_litearm_assets] wrote {urdf_out}")
    print(f"[build_litearm_assets]   tcp_link rpy (rad) = {frame_rpy(tcp_rotation())}")
    print(f"[build_litearm_assets]   camera pos = {CAMERA_POS} rpy = {CAMERA_RPY}")
    return urdf_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--gripper-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets" / "litearm",
    )
    args = parser.parse_args()
    if args.arm_dir is None or args.gripper_dir is None:
        parser.error("--arm-dir and --gripper-dir are required to rebuild the description")
    build(
        args.arm_dir / "urdf" / "litearm_description.urdf",
        args.arm_dir / "meshes",
        args.gripper_dir / "gripper2.urdf",
        args.gripper_dir / "meshes",
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
