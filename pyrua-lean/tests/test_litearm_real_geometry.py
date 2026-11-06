"""Independent FK/IK, joint mapping, TCP, camera and collision checks."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from pyrualean.hardware.config import load_station
from pyrualean.hardware.kinematics import (
    CollisionModel,
    LitearmKinematics,
    angles,
    axis_angle,
    homogeneous,
    orientation,
    rotation_error,
    rpy_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "litearm_real.example.json"


def _rpy(text: str) -> np.ndarray:
    values = [float(v) for v in text.split()]
    return rpy_matrix(values)


def _urdf_chain(path: Path):
    robot = ET.parse(path).getroot()
    parent_of = {j.find("child").get("link"): j for j in robot.findall("joint")}
    chain, link = [], "ee_link"
    while link != "base_link":
        joint = parent_of[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    return list(reversed(chain))


def _independent_fk(chain, q_model, base):
    pose = base.copy()
    moving = 0
    for joint in chain:
        origin = joint.find("origin")
        pose = pose @ homogeneous(
            _rpy(origin.get("rpy", "0 0 0")), [float(v) for v in origin.get("xyz", "0 0 0").split()]
        )
        if joint.get("type") == "revolute":
            axis = [float(v) for v in joint.find("axis").get("xyz").split()]
            pose = pose @ homogeneous(axis_angle(axis, q_model[moving]), (0, 0, 0))
            moving += 1
    return pose


@pytest.fixture(scope="module")
def station():
    return load_station(CONFIG)


@pytest.fixture(scope="module")
def kin(station):
    return LitearmKinematics(station.calibration)


def test_urdf_chain_and_fk_match_an_independent_reader(station, kin):
    chain = _urdf_chain(ROOT / "src" / "pyrualean" / "hardware" / "assets" / "litearm.urdf")
    rng = np.random.default_rng(17)
    for arm in ("left", "right"):
        cal = station.calibration.arm(arm)
        lo = np.array([pair[0] for pair in cal.joint_limits_rad])
        hi = np.array([pair[1] for pair in cal.joint_limits_rad])
        for _ in range(30):
            q = rng.uniform(lo, hi)
            expected = _independent_fk(chain, q, np.asarray(cal.base_pose)) @ np.asarray(
                cal.flange_to_tcp
            )
            got = kin.fk(arm, q)
            assert np.linalg.norm(got[:3, 3] - expected[:3, 3]) < 1e-6
            assert rotation_error(got[:3, :3], expected[:3, :3]) < 1e-4


def test_zero_pose_and_tcp_are_above_the_base(station, kin):
    cal = station.calibration.arm("left")
    zero = kin.fk("left", [0.0] * 7)
    assert zero[:3, 3] == pytest.approx(
        cal.base_pose[:3, 3] + np.array([0.0, 0.0, 0.9136]), abs=1e-4
    )
    assert kin.tcp_point("left", [0.0] * 7) == pytest.approx(zero[:3, 3])


def test_joint_map_round_trips(station):
    rng = np.random.default_rng(3)
    for arm in ("left", "right"):
        cal = station.calibration.arm(arm)
        q = rng.uniform(-1.0, 1.0, 7)
        sdk = cal.sdk_from_model(q)
        back = cal.model_from_sdk(sdk)
        assert back == pytest.approx(q, abs=1e-12)
        # The example map is the identity; the test still exercises both directions.
        assert sdk == pytest.approx(q)


def test_orientation_angle_round_trip():
    rng = np.random.default_rng(5)
    for _ in range(100):
        pitch = rng.uniform(-80, 85)
        yaw = rng.uniform(-179, 179)
        roll = rng.uniform(-179, 179)
        rot = orientation(pitch, yaw, roll)
        got = angles(rot)
        rebuilt = orientation(*got)
        assert rotation_error(rot, rebuilt) < 1e-4
    down = orientation(90, 33, 0)
    pitch, yaw, roll = angles(down)
    assert pitch == pytest.approx(90, abs=1e-4)
    assert yaw == pytest.approx(33, abs=1e-3)
    assert roll == pytest.approx(0, abs=1e-3)


def test_tcp_and_wrist_camera_transforms_compose(station, kin):
    q = np.array(station.calibration.arm("left").safe_home_rad)
    tcp = kin.fk("left", q)
    cam = station.calibration.camera("cam_left_wrist")
    expected = tcp @ np.asarray(cam.transform)
    got = kin.camera_pose("cam_left_wrist", "left", q)
    assert got == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError):
        kin.camera_pose("cam_left_wrist", "right", q)
    head = kin.camera_pose("cam_head", "left", q)
    assert head == pytest.approx(np.asarray(station.calibration.camera("cam_head").transform))


def test_ik_reaches_a_feasible_pose_and_checks_limits(station, kin):
    home = station.calibration.arm("left").safe_home_rad
    target = kin.fk("left", home)
    res = kin.best_solution("left", target, [0.0] * 7, extra_seeds=[home])
    assert res.ok and not res.collides
    assert res.pos_err_m < 0.002 and res.rot_err_deg < 1.0
    q = np.asarray(res.q)
    for value, (lo, hi) in zip(q, station.calibration.arm("left").joint_limits_rad, strict=True):
        assert lo - 1e-9 <= value <= hi + 1e-9
    with pytest.raises(ValueError, match="limits"):
        kin.fk("left", [9.0] * 7)


def test_collision_model_refuses_the_table_and_accepts_home(station, kin):
    collision = CollisionModel(station.calibration, kin)
    home = station.calibration.arm("left").safe_home_rad
    assert collision.self_violations("left", home) == []
    # A wrist-down pose with the grasp point below the table is a violation.
    target = np.eye(4)
    target[:3, :3] = orientation(90, 0, 0)
    target[:3, 3] = [-0.30, 0.30, -0.05]
    res = kin.solve("left", target, home, collision_check=True)
    assert res.collides or not res.ok


def test_line_path_is_collision_checked(station, kin):
    q0 = np.array(station.calibration.arm("left").safe_home_rad)
    pose = kin.fk("left", q0)
    target = pose.copy()
    target[2, 3] -= 0.10
    ways, why = kin.line_path("left", q0, target)
    assert why is None and ways
    assert np.linalg.norm(kin.tcp_point("left", ways[-1]) - target[:3, 3]) < 0.002
    below = pose.copy()
    below[2, 3] = -0.2
    _, why2 = kin.line_path("left", q0, below)
    assert why2 is not None
