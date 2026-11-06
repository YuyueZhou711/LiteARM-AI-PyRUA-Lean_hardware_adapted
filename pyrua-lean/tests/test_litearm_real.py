"""The real LiteArm adapter, result conversion and host contract (offline)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from pyrualean.litearm_real import (
    ADAPTER,
    LitearmRealRobot,
    RealGripper,
    RealMove,
    RealObjectSight,
    RealPlace,
    RealSeenPoint,
    RealSegment,
    RealState,
)
from pyrualean.prompt import _public_members, api_reference, knowledge_text
from pyrualean.robots import ROBOTS, get_adapter

ROOT = Path(__file__).resolve().parents[1]


class _StubBackend:
    TASK = "put the red block in the tray"

    def __init__(self) -> None:
        self.artifacts: dict[str, bytes] = {}

    def _envelope(self) -> dict[str, Any]:
        state = {
            "left_tcp_pos": [-0.30, 0.20, 0.80],
            "left_tcp_quat_wxyz": [0.0, 1.0, 0.0, 0.0],
            "left_pitch_yaw_roll": [90.0, 0.0, 0.0],
            "left_arm_joint_state": [0.1, 0.2, -0.3, -1.2, 0.0, 0.4, 0.0],
            "left_ee_joint_state": [0.5],
            "left_ee_width_m": 0.043,
            "left_data_age_s": 0.01,
            "left_fault": False,
            "right_tcp_pos": [0.30, 0.20, 0.80],
            "right_tcp_quat_wxyz": [0.0, 1.0, 0.0, 0.0],
            "right_pitch_yaw_roll": [90.0, 0.0, 0.0],
            "right_arm_joint_state": [0.0] * 7,
            "right_ee_joint_state": [1.0],
            "right_ee_width_m": 0.08696,
            "right_data_age_s": 0.01,
            "right_fault": False,
        }
        return {
            "step": 7,
            "task_language": self.TASK,
            "terminated": False,
            "truncated": False,
            "success_known": False,
            "state": {
                "state": state,
                "instruction": self.TASK,
                "cameras": {name: {"shape": [480, 640, 3]} for name in ADAPTER.cameras},
                "hardware": {"execution_state": "ready", "success_known": False},
            },
            "log": {"tool": "view_env_state", "result": {}},
        }

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        if name == "view_env_state":
            return self._envelope()
        if name == "view_camera_meta":
            return {
                "shape": [480, 640, 3],
                "intrinsic_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                "extrinsic_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "step": 7,
            }
        if name == "move_pose":
            payload = {
                "arm": "left",
                "target_xyz": [-0.3, 0.2, 0.8],
                "start_xyz": [-0.3, 0.2, 0.8],
                "final_xyz": [-0.3, 0.2, 0.8],
                "dist_to_target_m": 0.0,
                "orientation_error_deg": 0.0,
                "final_angles": {"pitch": 90.0, "yaw": 0.0, "roll": 0.0},
                "reached": True,
                "path": "line",
                "steps_used": 3,
            }
            env = self._envelope()
            env["log"] = {"tool": name, "result": payload}
            return env
        if name == "grip":
            env = self._envelope()
            env["log"] = {
                "tool": name,
                "result": {
                    "arm": "left",
                    "gripper": "closed",
                    "opening": 0.5,
                    "holding": True,
                    "width_m": 0.043,
                    "settled": True,
                    "steps_used": 1,
                    "step": 7,
                },
            }
            return env
        if name == "home":
            env = self._envelope()
            env["log"] = {
                "tool": name,
                "result": {
                    "arm": "both",
                    "reached": True,
                    "max_joint_error_rad": 0.0,
                    "steps_used": 2,
                },
            }
            return env
        if name == "place_on":
            env = self._envelope()
            env["log"] = {
                "tool": name,
                "result": {
                    "arm": "left",
                    "placed": True,
                    "released": True,
                    "safe_to_home": True,
                    "target_xyz": [0.0, 0.3, 0.8],
                    "place_z": 0.8,
                    "descent_reached": True,
                    "open_opening": 1.0,
                    "lift_reached": True,
                    "retreat_reached": True,
                    "steps_used": 4,
                },
            }
            return env
        if name == "locate":
            return {
                "found": True,
                "center_xyz": [0.0, 0.3, 0.05],
                "top_z": 0.1,
                "height_m": 0.1,
                "length_m": 0.04,
                "width_m": 0.03,
                "long_axis_deg": 0.0,
                "grasp_yaw": 90.0,
                "score": 0.9,
                "centroid_rc": [200, 300],
                "area_px": 100,
                "camera": "cam_head",
                "views": 2,
                "source": "multiview",
                "measurement_failed": False,
            }
        if name == "segment":
            return {
                "found": True,
                "score": 0.9,
                "centroid_rc": [200, 300],
                "area_px": 100,
                "camera": "cam_head",
            }
        if name == "back_project":
            return {
                "pixel": [200, 300],
                "world_xyz": [0.0, 0.3, 0.05],
                "depth_m": 1.2,
                "camera": "cam_head",
                "surface": "object",
            }
        return {"error": f"unexpected {name}"}

    def solved(self) -> bool:
        return False

    def artifact(self, name: str, step: int = -1) -> bytes:
        return self.artifacts.get(name, b"png")

    def env_steps(self) -> int:
        return 0

    def cancel(self) -> None:
        return None


def test_adapter_is_registered_and_light():
    assert "litearm_real" in ROBOTS
    adapter = get_adapter("litearm_real")
    assert adapter is ADAPTER
    assert adapter.robot_cls is LitearmRealRobot
    assert adapter.host == "pyrualean.hosts.litearm_hardware"
    assert adapter.knowledge == "litearm-real"
    assert adapter.name == "litearm_real"
    assert adapter.cameras == ("cam_head", "cam_left_wrist", "cam_right_wrist")
    assert adapter.live_success is False


def test_no_vendor_sdk_is_imported_at_package_import():
    code = (
        "import sys; import pyrualean.hardware; import pyrualean.hosts.litearm_hardware; "
        "import pyrualean.litearm_real; "
        "print(any(m == 'litearm' or m.startswith('litearm.') for m in sys.modules), "
        "any(m == 'litegrip' or m.startswith('litegrip.') for m in sys.modules))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False False"


def test_api_reference_and_knowledge_reflect_real_limits():
    reference = api_reference(LitearmRealRobot, primitives="no-vla")
    assert "pi0_pick" not in reference
    assert "measurement_failed" in reference
    assert "measured" in reference
    assert "success_known" in reference
    assert "device-action" in reference or "device action" in reference
    members = _public_members(LitearmRealRobot)
    assert "done" not in members and "world_map" not in members
    assert {"move_to", "set_gripper", "home", "place_on", "locate"} <= set(members)
    knowledge = knowledge_text("litearm-real", primitives="no-vla")
    assert "measurement_failed" in knowledge
    assert "environment_success_known = false" in knowledge
    assert "no independent task-success evaluator" in knowledge
    assert "2 mm + 85 mm" in knowledge  # explicitly rejected
    assert "z = 0" in knowledge


def test_state_and_result_conversion_from_measured_envelope():
    robo = LitearmRealRobot(_StubBackend())
    state = robo.state()
    assert isinstance(state, RealState)
    assert state.step == 7 and state.success_known is False
    assert state.left.tcp_pos == (-0.30, 0.20, 0.80)
    assert state.left.eef_pos == state.left.tcp_pos
    assert state.left.gripper == 0.5 and state.left.width_m == 0.043
    assert state.left.eef_quat == (0.0, 1.0, 0.0, 0.0)
    assert len(state.left.joints) == 7

    move = robo.move_to([-0.3, 0.2, 0.8], arm="left")
    assert isinstance(move, RealMove) and move.reached and move.steps_used == 3
    grip = robo.set_gripper("left", robo.CLOSE)
    assert isinstance(grip, RealGripper) and grip.holding and grip.settled
    home = robo.home("both")
    assert home.reached and home.note is None
    place = robo.place_on(0.0, 0.3, 0.75, 0.05, arm="left")
    assert isinstance(place, RealPlace) and place.placed and place.safe_to_home
    sight = robo.locate("the red block")
    assert isinstance(sight, RealObjectSight) and sight.views == 2 and sight.source == "multiview"
    assert sight.camera == "cam_head"  # no station toolkit: the documented default
    segment = robo.segment("the red block")
    assert isinstance(segment, RealSegment) and segment.found
    point = robo.back_project(200, 300)
    assert isinstance(point, RealSeenPoint) and point.surface == "object"
    meta = robo.camera_meta("cam_head")
    assert meta["shape"] == [480, 640, 3] and meta["step"] == 7


class _PrimaryCameraBackend(_StubBackend):
    def __init__(self, primary: str = "cam_left_wrist") -> None:
        super().__init__()
        import io
        from types import SimpleNamespace

        from PIL import Image

        self.toolkit = SimpleNamespace(config=SimpleNamespace(primary_camera=primary))
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.artifact_names: list[str] = []
        buffer = io.BytesIO()
        Image.new("RGB", (2, 3), (1, 2, 3)).save(buffer, "PNG")
        self.artifacts[f"{primary}.png"] = buffer.getvalue()

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        self.calls.append((name, dict(input_dict)))
        payload = super().execute_tool(name, input_dict)
        if isinstance(payload, dict) and "camera" in payload:
            payload = dict(payload)
            payload["camera"] = input_dict.get("camera", payload["camera"])
        return payload

    def artifact(self, name: str, step: int = -1) -> bytes:
        self.artifact_names.append(name)
        return super().artifact(name, step)


def test_omitted_camera_uses_the_station_primary_camera():
    backend = _PrimaryCameraBackend()
    robo = LitearmRealRobot(backend)
    sight = robo.locate("the red block")
    assert sight.camera == "cam_left_wrist"
    assert backend.calls[-1] == (
        "locate",
        {"text_prompt": "the red block", "camera": "cam_left_wrist", "min_score": 0.2},
    )
    segment = robo.segment("the red block")
    assert segment.camera == "cam_left_wrist"
    assert backend.calls[-1] == (
        "segment",
        {"text_prompt": "the red block", "camera": "cam_left_wrist", "min_score": 0.2},
    )
    point = robo.back_project(200, 300)
    assert point.camera == "cam_left_wrist"
    assert backend.calls[-1] == (
        "back_project",
        {"row": 200, "col": 300, "camera": "cam_left_wrist"},
    )
    meta = robo.camera_meta()
    assert meta["shape"] == [480, 640, 3]
    assert backend.calls[-1] == ("view_camera_meta", {"camera": "cam_left_wrist", "step": -1})
    image = robo.image()
    assert image.shape == (3, 2, 3)
    assert backend.artifact_names[-1] == "cam_left_wrist.png"

    # An explicit camera still wins over the station default.
    overridden = robo.segment("the red block", camera="cam_right_wrist")
    assert overridden.camera == "cam_right_wrist"
    assert backend.calls[-1][1]["camera"] == "cam_right_wrist"


class _NoGripperBackend(_StubBackend):
    def _envelope(self) -> dict[str, Any]:
        envelope = super()._envelope()
        envelope["state"]["state"]["left_ee_joint_state"] = None
        envelope["state"]["state"]["left_ee_width_m"] = None
        return envelope


def test_state_reports_a_missing_gripper_as_none():
    robo = LitearmRealRobot(_NoGripperBackend())
    state = robo.state()
    assert state.left.gripper is None
    assert state.left.width_m is None
    summary = robo._summary_dict()
    assert summary["left_gripper"] is None
    assert summary["left_width_m"] is None


def test_done_raises_and_finish_is_not_success():
    robo = LitearmRealRobot(_StubBackend())
    with pytest.raises(AttributeError):
        _ = robo.done
    assert robo._reported_success() is False


def test_host_contract_without_booting():
    from pyrualean.hosts import litearm_hardware as host

    assert host.ROBOT == "litearm_real"
    for name in (
        "Cell",
        "add_cell_args",
        "cell_from_args",
        "add_boot_args",
        "boot_options",
        "boot",
        "make_backend",
        "dump_card",
    ):
        assert hasattr(host, name)
    cell = host.Cell(
        hardware_config="/tmp/station.json",
        instruction="put the red block in the tray",
        task="pick_place",
        suite="cells",
        seed=3,
        max_episode_steps=12,
    )
    assert cell.tag == "pick_place-3"
