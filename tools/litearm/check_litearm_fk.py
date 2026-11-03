#!/usr/bin/env python3
"""Check that the spawned LiteArm articulation matches the URDF kinematics.

Spawns one LiteArm from the RoboDojo robot config, writes a few joint
configurations through IsaacLab, and compares the world pose of ``tcp_link``
with forward kinematics computed from the composed URDF.  This validates the
joint order, the joint signs (joint6 is mounted reversed) and the TCP frame
that the policy-side IK will assume.

Run (headless, RoboDojo env):

  OMNI_KIT_ACCEPT_EULA=YES python tools/litearm/check_litearm_fk.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 640, "height": 480})

import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402


# ---------------------------------------------------------------------------
# URDF forward kinematics
# ---------------------------------------------------------------------------


def _rpy(rpy: str) -> np.ndarray:
    r, p, y = (float(v) for v in rpy.split())
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _transform(rot: np.ndarray, pos) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


class UrdfChain:
    """FK for a serial chain of the composed URDF, root -> leaf."""

    def __init__(self, urdf: Path, root: str, leaf: str):
        robot = ET.parse(urdf).getroot()
        parent_of = {j.find("child").get("link"): j for j in robot.findall("joint")}
        chain = []
        link = leaf
        while link != root:
            joint = parent_of[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain = list(reversed(chain))
        self.joint_names = [
            j.get("name") for j in self.chain if j.get("type") in ("revolute", "prismatic")
        ]

    def fk(self, q: dict[str, float]) -> np.ndarray:
        pose = np.eye(4)
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
            rot = _rpy(origin.get("rpy", "0 0 0"))
            pose = pose @ _transform(rot, xyz)
            if joint.get("type") == "revolute":
                axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
                pose = pose @ _transform(_axis_angle(axis, q.get(joint.get("name"), 0.0)), np.zeros(3))
            elif joint.get("type") == "prismatic":
                axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
                pose = pose @ _transform(np.eye(3), axis * q.get(joint.get("name"), 0.0))
        return pose


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets" / "litearm",
    )
    args = parser.parse_args()
    asset_dir = args.asset_dir.resolve()

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device="cuda:0"))
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Litearm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_dir / "litearm.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, fix_root_link=True
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                stiffness=4000.0,
                damping=100.0,
            )
        },
    )
    robot = Articulation(articulation_cfg)
    sim.reset()

    names = list(robot.data.joint_names)
    print(f"[fk] joint names: {names}")
    tcp_idx = robot.data.body_names.index("tcp_link")
    chain = UrdfChain(asset_dir / "litearm.urdf", "base_link", "tcp_link")
    print(f"[fk] URDF chain: {chain.joint_names}")

    rng = np.random.default_rng(0)
    for trial in range(4):
        q = {name: float(rng.uniform(-0.4, 0.4)) for name in chain.joint_names}
        # joint4 has an asymmetric limit: -3.089 .. +0.035 rad
        q["joint4"] = float(rng.uniform(-1.2, 0.0))
        q_ctrl = torch.zeros(len(names), device=robot.device)
        for name, value in q.items():
            q_ctrl[names.index(name)] = value
        robot.write_joint_state_to_sim(q_ctrl, torch.zeros_like(q_ctrl))
        robot.set_joint_position_target(q_ctrl)
        robot.write_data_to_sim()
        for _ in range(120):
            sim.step(render=False)
            robot.update(sim.get_physics_dt())
        pose = robot.data.body_pose_w[0, tcp_idx].cpu().numpy()
        expected = chain.fk(q)
        pos_err = np.linalg.norm(pose[:3] - expected[:3, 3])
        w, x, y, z = pose[3:]
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        angle = np.degrees(np.arccos(np.clip((np.trace(rot.T @ expected[:3, :3]) - 1) / 2, -1, 1)))
        print(f"[fk] trial {trial}: q={ {k: round(v, 3) for k, v in q.items()} }")
        print(f"[fk]   sim xyz={np.round(pose[:3], 4)} urdf xyz={np.round(expected[:3, 3], 4)}")
        print(f"[fk]   pos_err={pos_err * 1000:.2f} mm  rot_err={angle:.3f} deg")
    return 0


import os
import sys

try:
    code = main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
# app.close() can hang with a bare Articulation; this is a diagnostic script.
os._exit(code)
