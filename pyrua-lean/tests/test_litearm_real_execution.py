"""End-to-end execution tests over a real localhost socket and the fake station."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from pyrualean.hardware.cameras import CameraManager
from pyrualean.hardware.config import load_station
from pyrualean.hardware.daemon import HardwareDaemon
from pyrualean.hardware.fake import FakeStation
from pyrualean.hardware.motion import grip, move_pose, place_on
from pyrualean.hardware.rpc import ConnectionLost, HardwareClient, RpcError
from pyrualean.hardware.toolkit import HardwareToolkit

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ROOT / "configs" / "litearm_real.offline.example.json"
CALIBRATION = ROOT / "configs" / "calibration" / "litearm_real.example.json"


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _config(
    tmp_path: Path,
    *,
    mode: str = "offline-fake",
    active_arms: tuple[str, ...] | None = None,
    devices: tuple[str, ...] | None = None,
    cameras: tuple[str, ...] | None = None,
    primary_camera: str | None = None,
    gripper: bool = True,
    **fake: Any,
):
    raw = json.loads(OFFLINE.read_text())
    raw["execution"]["mode"] = mode
    if active_arms is not None:
        raw["execution"]["active_arms"] = list(active_arms)
    if devices is not None:
        raw["devices"] = {name: raw["devices"][name] for name in devices}
    if not gripper:
        for item in raw["devices"].values():
            item["gripper"] = None
    if cameras is not None:
        raw["cameras"] = {name: raw["cameras"][name] for name in cameras}
    if primary_camera is not None:
        raw["perception"]["primary_camera"] = primary_camera
    calibration = json.loads(CALIBRATION.read_text())
    calibration_path = CALIBRATION
    if active_arms is not None or cameras is not None:
        # The loader requires the calibration to describe exactly the station's
        # active arms and configured cameras; write the matching subset.
        if active_arms is not None:
            calibration["arms"] = {name: calibration["arms"][name] for name in active_arms}
        if cameras is not None:
            calibration["cameras"] = {name: calibration["cameras"][name] for name in cameras}
        calibration_path = tmp_path / "calibration.json"
    raw["calibration"] = str(calibration_path)
    raw["service"]["port"] = _free_port()
    raw["service"]["heartbeat_s"] = 0.15
    raw["service"]["heartbeat_timeout_s"] = 0.45
    raw["snapshot"]["refresh_min_interval_s"] = 0.0
    raw["snapshot"]["camera_joint_sync_s"] = 0.20
    raw["fake"].update({"time_scale": 0.02, "object_present": True, "object_width_m": 0.03})
    raw["fake"].update(fake)
    tmp_path.mkdir(parents=True, exist_ok=True)
    if calibration_path != CALIBRATION:
        calibration_path.write_text(json.dumps(calibration))
    path = tmp_path / "station.json"
    path.write_text(json.dumps(raw))
    return load_station(path)


def _daemon(config, *, force_fake: bool = False) -> HardwareDaemon:
    station = None
    if force_fake:
        station = FakeStation(config, config.calibration)
    return HardwareDaemon(
        config,
        station=station,
        cameras=CameraManager(config, config.calibration),
    )


def _start(config, *, force_fake: bool = False):
    daemon = _daemon(config, force_fake=force_fake)
    daemon.start_service()
    daemon.listen("127.0.0.1", config.service.port)
    client = HardwareClient(config)
    client.connect()
    client.open_session(client_id="test")
    return daemon, client


def _stop(daemon, client):
    try:
        client.close()
    except Exception:
        client.close(force=True)
    daemon.close()


def test_daemon_close_before_start_does_not_mask_the_boot_error(tmp_path):
    config = _config(tmp_path, mode="offline-fake")
    daemon = _daemon(config)
    # start_service() never ran, so the worker threads do not exist yet.  A
    # bootstrap failure calls close(); it must not raise from join().
    daemon.close()


def _toolkit(config, client, tmp_path) -> HardwareToolkit:
    toolkit = HardwareToolkit(
        config,
        client,
        task="test",
        instruction="move a small distance",
        out_dir=tmp_path / "out",
    )
    toolkit.refresh_observation(tool="boot")
    return toolkit


def _trajectory_payload(config, toolkit, waypoints, *, timeout_s=5.0, carry=None):
    observation = toolkit.current_observation(refresh=True)
    return {
        "arm": "left",
        "q_start": [float(v) for v in observation.arm("left").q_model],
        "waypoints": [list(map(float, q)) for q in waypoints],
        "speed": 0.5,
        "timeout_s": timeout_s,
        "carry_baseline": carry,
        "slip_drop": 0.02,
        "path": "joint",
    }


def test_session_is_exclusive_and_read_only_refuses_motion(tmp_path):
    config = _config(tmp_path, mode="offline-fake")
    daemon, client = _start(config)
    try:
        other = HardwareClient(config)
        other.connect()
        with pytest.raises(RpcError, match="busy"):
            other.open_session(client_id="second")
        other.close(force=True)
    finally:
        _stop(daemon, client)

    # Test-only mode override: the fake station and cameras are explicit, and
    # the service is forced to read-only after construction (the station config
    # itself remains offline-fake because its fake camera drivers are legal
    # only there).
    from pyrualean.hardware.config import ExecutionMode

    read_only = _config(tmp_path / "ro", mode="offline-fake")
    daemon2 = _daemon(read_only, force_fake=True)
    daemon2.mode = ExecutionMode.READ_ONLY
    daemon2.start_service()
    daemon2.listen("127.0.0.1", read_only.service.port)
    client2 = HardwareClient(read_only)
    client2.connect()
    opened = client2.open_session(client_id="readonly")
    assert opened["read_only"] is True
    try:
        with pytest.raises(RpcError, match="read_only"):
            client2.request(
                "command",
                {
                    "command_id": "x",
                    "kind": "grip",
                    "payload": {"arm": "left", "command": "open"},
                    "digest": __import__(
                        "pyrualean.hardware.protocol", fromlist=["command_digest"]
                    ).command_digest("grip", {"arm": "left", "command": "open"}),
                },
            )
    finally:
        client2.close(force=True)
        daemon2.close()


def test_measured_move_reaches_and_counts_segments(tmp_path):
    config = _config(tmp_path)
    daemon, client = _start(config)
    try:
        toolkit = _toolkit(config, client, tmp_path)
        start = toolkit.current_observation(refresh=False).arm("left")
        import numpy as np

        xyz = np.asarray(start.tcp_pose).reshape(4, 4)[:3, 3]
        target = [float(xyz[0]), float(xyz[1]), float(xyz[2] - 0.05)]
        result = move_pose(toolkit, target, arm="left", path="line")
        assert result["reached"] is True, result
        assert result["steps_used"] > 0
        assert result["measured"] is True
        assert result["dist_to_target_m"] <= 0.01
        assert toolkit.env_steps() == result["steps_used"]
        state = toolkit.current_observation(refresh=True)
        assert state.step > 0
        assert state.arm("left").tcp_pose != start.tcp_pose
    finally:
        _stop(daemon, client)


def test_duplicate_command_is_deduplicated(tmp_path):
    config = _config(tmp_path)
    daemon, client = _start(config)
    try:
        toolkit = _toolkit(config, client, tmp_path)
        observation = toolkit.current_observation(refresh=True)
        payload = _trajectory_payload(config, toolkit, [observation.arm("left").q_model])
        from pyrualean.hardware.protocol import command_digest

        digest = command_digest("trajectory", payload)
        request = {
            "command_id": "dedupe-1",
            "kind": "trajectory",
            "payload": payload,
            "digest": digest,
        }
        first = client.request("command", request)
        second = client.request("command", request)
        assert first["command_id"] == second["command_id"]
        final = client.request(
            "wait_command", {"command_id": "dedupe-1", "timeout_s": 5.0}, timeout=7.0
        )
        assert final["state"] == "succeeded"
        assert final["steps_used"] == 1
        status = client.status()
        assert status["state"] == "ready"
    finally:
        _stop(daemon, client)


def test_cancel_stops_between_segments_and_refuses_more_motion(tmp_path):
    config = _config(tmp_path, time_scale=0.03)
    daemon, client = _start(config)
    try:
        toolkit = _toolkit(config, client, tmp_path)
        observation = toolkit.current_observation(refresh=True)
        home = list(observation.arm("left").q_model)
        waypoints = []
        for i in range(80):
            q = list(home)
            q[0] += 0.01 if i % 2 == 0 else 0.0
            waypoints.append(q)
        payload = _trajectory_payload(config, toolkit, waypoints, timeout_s=10.0)
        from pyrualean.hardware.protocol import command_digest

        command_id = "cancel-1"
        client.request(
            "command",
            {
                "command_id": command_id,
                "kind": "trajectory",
                "payload": payload,
                "digest": command_digest("trajectory", payload),
            },
        )
        time.sleep(0.15)
        reply = client.send_cancel()
        assert reply["accepted"] is True
        final = client.request(
            "wait_command", {"command_id": command_id, "timeout_s": 5.0}, timeout=7.0
        )
        assert final["state"] == "cancelled"
        assert final["stop"]["confirmed"] is True
        status = client.status()
        assert status["state"] == "stopped"
        with pytest.raises(RpcError, match="stopped"):
            client.request(
                "command",
                {
                    "command_id": "after-cancel",
                    "kind": "grip",
                    "payload": {"arm": "left", "command": "open"},
                    "digest": command_digest("grip", {"arm": "left", "command": "open"}),
                },
            )
    finally:
        _stop(daemon, client)


def test_timeout_faults_and_refuses_new_trajectory(tmp_path):
    config = _config(tmp_path, time_scale=0.04)
    daemon, client = _start(config)
    try:
        toolkit = _toolkit(config, client, tmp_path)
        observation = toolkit.current_observation(refresh=True)
        home = list(observation.arm("left").q_model)
        waypoints = []
        for i in range(40):
            q = list(home)
            q[0] += 0.005 if i % 2 == 0 else 0.0
            waypoints.append(q)
        payload = _trajectory_payload(config, toolkit, waypoints, timeout_s=0.10)
        from pyrualean.hardware.protocol import command_digest

        client.request(
            "command",
            {
                "command_id": "timeout-1",
                "kind": "trajectory",
                "payload": payload,
                "digest": command_digest("trajectory", payload),
            },
        )
        final = client.request(
            "wait_command", {"command_id": "timeout-1", "timeout_s": 6.0}, timeout=8.0
        )
        assert final["state"] in ("timed_out", "faulted")
        status = daemon._status({})
        assert status["state"] == "fault"
        assert status["stop"] is not None
        # The old connection may be marked lost by its heartbeat; a new client
        # must be refused at session open and must not resume motion.
        fresh = HardwareClient(config)
        fresh.connect()
        with pytest.raises(RpcError, match="faulted"):
            fresh.open_session(client_id="after-fault")
        fresh.close(force=True)
        with pytest.raises((RpcError, ConnectionLost)):
            client.request(
                "command",
                {
                    "command_id": "after-fault",
                    "kind": "grip",
                    "payload": {"arm": "left", "command": "open"},
                    "digest": command_digest("grip", {"arm": "left", "command": "open"}),
                },
            )
    finally:
        _stop(daemon, client)


def test_disconnect_faults_and_does_not_auto_resume(tmp_path):
    config = _config(tmp_path)
    daemon, client = _start(config)
    try:
        client.close(force=True)
        time.sleep(0.2)
        fresh = HardwareClient(config)
        fresh.connect()
        with pytest.raises(RpcError, match="faulted"):
            fresh.open_session(client_id="reconnect")
        fresh.close(force=True)
        status = daemon._status({})
        assert status["state"] == "fault"
        assert status["stop"] is not None
    finally:
        daemon.close()


def test_carry_slip_is_reported_without_claiming_reached(tmp_path):
    config = _config(tmp_path, slip_after_segments=1)
    daemon, client = _start(config)
    try:
        toolkit = _toolkit(config, client, tmp_path)
        closed = grip(toolkit, arm="left", gripper=1)
        assert closed["holding"] is True
        observation = toolkit.current_observation(refresh=True)
        baseline = observation.arm("left").gripper_opening
        home = list(observation.arm("left").q_model)
        target = list(home)
        target[0] += 0.02
        # Two small waypoints: the fake drops the object after the first.
        payload = _trajectory_payload(
            config,
            toolkit,
            [home, target],
            carry=baseline,
        )
        from pyrualean.hardware.protocol import command_digest

        client.request(
            "command",
            {
                "command_id": "slip-1",
                "kind": "trajectory",
                "payload": payload,
                "digest": command_digest("trajectory", payload),
            },
        )
        final = client.request(
            "wait_command", {"command_id": "slip-1", "timeout_s": 5.0}, timeout=7.0
        )
        assert final["state"] == "succeeded"
        assert final["result"]["slip"] is True
        assert final["result"]["success"] is False
    finally:
        _stop(daemon, client)


def test_heartbeat_loss_triggers_stop_and_fault(tmp_path):
    config = _config(tmp_path)
    daemon, client = _start(config)
    try:
        client._stop_heartbeat()
        time.sleep(0.8)
        status = daemon._status({})
        assert status["state"] == "fault"
        assert status["stop"] is not None
        with pytest.raises(RpcError, match="faulted"):
            client.request(
                "command",
                {
                    "command_id": "after-heartbeat",
                    "kind": "grip",
                    "payload": {"arm": "left", "command": "open"},
                    "digest": __import__(
                        "pyrualean.hardware.protocol", fromlist=["command_digest"]
                    ).command_digest("grip", {"arm": "left", "command": "open"}),
                },
            )
    finally:
        _stop(daemon, client)


def test_host_boot_robot_and_arm_toolkit_offline_chain(tmp_path):
    """The runner -> robo API -> toolkit -> localhost RPC -> fake station chain."""

    from pyrualean.arms import ArmToolkit
    from pyrualean.hosts import litearm_hardware as host
    from pyrualean.litearm_real import LitearmRealRobot

    config = _config(tmp_path)
    config_path = tmp_path / "station.json"
    daemon = _daemon(config)
    daemon.start_service()
    daemon.listen("127.0.0.1", config.service.port)
    cell = host.Cell(
        hardware_config=str(config_path),
        instruction="put the red block in the tray",
        task="pick_place",
        suite="cells",
        seed=0,
        max_episode_steps=50,
        execution_mode="offline-fake",
    )
    toolkit = None
    try:
        toolkit, daemons = host.boot(cell, tmp_path / "run", hardware_config=str(config_path))
        assert daemons == []
        backend = host.make_backend(toolkit)
        robot = LitearmRealRobot(backend)
        assert robot.task == cell.instruction
        arm = ArmToolkit(robot, arm="cells", workspace=tmp_path / "run" / "workspace")
        result = arm.execute_tool(
            "python",
            {
                "code": (
                    "s = robo.state()\n"
                    "print('measured', s.left.width_m)\n"
                    "r = robo.move_to([s.left.tcp_pos[0], s.left.tcp_pos[1], "
                    "s.left.tcp_pos[2]-0.03], arm='left', path='line')\n"
                    "print('reached', r.reached, r.steps_used)\n"
                    "assert r.reached\n"
                    "from robo_image_check import nothing\n"
                )
            },
        )
        # The last line is an intentional import error; the move still happened.
        assert "exception" in result.result
        assert "reached True" in result.result["stdout"]
        assert robot.state().success_known is False
        assert robot.image("cam_head").shape == (480, 640, 3)
        first = robot.state()
        first_image = robot.image("cam_head", step=first.step)
        first_meta = robot.camera_meta("cam_head", step=first.step)
        robot.move_to(
            [first.left.tcp_pos[0], first.left.tcp_pos[1], first.left.tcp_pos[2] - 0.02],
            arm="left",
            path="line",
        )
        assert (robot.image("cam_head", step=first.step) == first_image).all()
        assert robot.camera_meta("cam_head", step=first.step) == first_meta
        card = host.dump_card(toolkit, cell, tmp_path / "run")
        assert "not measured" in json.dumps(card.facts)
        toolkit.close(finish={"status": "success", "summary": "model claim"})
        stop = host.stop_metadata(toolkit)
        assert stop["requested"] is True
        assert stop["recorded_after_shutdown"] is True
        assert stop["environment_success_known"] is False
        assert (tmp_path / "run" / "episode.json").is_file()
        recording = json.loads((tmp_path / "run" / "episode.json").read_text())
        assert recording["task_success"] == "not_measured"
        assert recording["finish"]["status"] == "success"
        assert recording["stop"]["requested"] is True
    finally:
        if toolkit is not None:
            toolkit.close()
        daemon.close()


def test_stale_start_state_is_rejected(tmp_path):
    config = _config(tmp_path)
    daemon, client = _start(config)
    try:
        toolkit = _toolkit(config, client, tmp_path)
        observation = toolkit.current_observation(refresh=True)
        q = list(observation.arm("left").q_model)
        q[0] += 0.2
        payload = {
            "arm": "left",
            "q_start": q,
            "waypoints": [q],
            "speed": 0.2,
            "timeout_s": 2.0,
            "carry_baseline": None,
            "slip_drop": 0.02,
            "path": "joint",
        }
        from pyrualean.hardware.protocol import command_digest

        client.request(
            "command",
            {
                "command_id": "stale-1",
                "kind": "trajectory",
                "payload": payload,
                "digest": command_digest("trajectory", payload),
            },
        )
        final = client.request(
            "wait_command", {"command_id": "stale-1", "timeout_s": 3.0}, timeout=5.0
        )
        assert final["state"] == "rejected"
        assert final["error"]["code"] == "stale_state"
        assert final["steps_used"] == 0
    finally:
        _stop(daemon, client)


def _single_arm_config(tmp_path: Path, arm: str = "left"):
    return _config(
        tmp_path,
        active_arms=(arm,),
        devices=(arm,),
        cameras=("cam_head", f"cam_{arm}_wrist"),
    )


def test_single_arm_station_observes_and_moves_only_the_active_arm(tmp_path):
    config = _single_arm_config(tmp_path)
    daemon, client = _start(config)
    try:
        hello = client.hello or {}
        assert hello["active_arms"] == ["left"]
        assert hello["cameras"] == ["cam_head", "cam_left_wrist"]
        capabilities = hello["capabilities"]
        assert capabilities["active_arms"] == ["left"]
        assert set(capabilities["arms"]) == {"left"}
        assert set(capabilities["grippers"]) == {"left"}

        toolkit = _toolkit(config, client, tmp_path)
        observation = toolkit.current_observation(refresh=True)
        assert set(observation.arms) == {"left"}
        assert set(observation.camera_state) == {"cam_head", "cam_left_wrist"}
        assert "left_ee_pose" in observation.state
        assert "right_ee_pose" not in observation.state

        import numpy as np

        start = observation.arm("left")
        xyz = np.asarray(start.tcp_pose).reshape(4, 4)[:3, 3]
        target = [float(xyz[0]), float(xyz[1]), float(xyz[2] - 0.05)]
        moved = move_pose(toolkit, target, arm="left", path="line")
        assert moved["reached"] is True, moved
        assert moved["steps_used"] > 0
        after = toolkit.current_observation(refresh=True)
        assert after.arm("left").tcp_pose != start.tcp_pose

        from pyrualean.hardware.motion import home

        # ``home("both")`` means the active arms only: it plans for left and
        # never asks for the inactive arm's measured state.
        calls: list[dict[str, Any]] = []
        original_plan_home = toolkit.planner.plan_home

        def plan_home_spy(planned_arm: str, q_start: Any, **kwargs: Any):
            calls.append({"arm": planned_arm, **kwargs})
            return original_plan_home(planned_arm, q_start, **kwargs)

        toolkit.planner.plan_home = plan_home_spy
        homed = home(toolkit, arm="both")
        assert [call["arm"] for call in calls] == ["left"]
        assert calls[0]["other_arm"] is None
        assert calls[0]["other_q"] is None
        assert homed["arm"] == "both"
    finally:
        _stop(daemon, client)


def test_single_arm_station_rejects_inactive_arm_commands(tmp_path):
    config = _single_arm_config(tmp_path)
    daemon, client = _start(config)
    try:
        toolkit = _toolkit(config, client, tmp_path)
        from pyrualean.hardware.motion import grip, home, move_pose

        with pytest.raises(ValueError, match="not active"):
            move_pose(toolkit, [0.0, 0.0, 0.0], arm="right")
        with pytest.raises(ValueError, match="not active"):
            grip(toolkit, arm="right", gripper=1)
        with pytest.raises(ValueError, match="not active"):
            home(toolkit, arm="right")
        payload = toolkit.execute_tool("grip", {"arm": "right", "gripper": 1})
        assert "not active" in payload["error"]

        from pyrualean.hardware.protocol import command_digest

        grip_payload = {"arm": "right", "command": "open"}
        client.request(
            "command",
            {
                "command_id": "inactive-grip",
                "kind": "grip",
                "payload": grip_payload,
                "digest": command_digest("grip", grip_payload),
            },
        )
        final = client.request(
            "wait_command", {"command_id": "inactive-grip", "timeout_s": 3.0}, timeout=5.0
        )
        assert final["state"] == "rejected"
        assert final["error"]["code"] == "bad_request"
        assert "not active" in final["error"]["message"]
    finally:
        _stop(daemon, client)


def test_camera_manager_skips_the_inactive_wrist_camera(tmp_path):
    # The station file still configures the right wrist camera, but only the
    # left arm is present; the manager must not open or capture it.
    config = _config(tmp_path, active_arms=("left",), devices=("left",))
    manager = CameraManager(config, config.calibration)
    assert manager.names == ("cam_head", "cam_left_wrist")
    manager.open()
    try:
        assert set(manager.cameras) == {"cam_head", "cam_left_wrist"}
        assert set(manager.capture()) == {"cam_head", "cam_left_wrist"}
    finally:
        manager.close()


def test_single_arm_robot_state_and_inactive_arm_error(tmp_path):
    config = _single_arm_config(tmp_path)
    config_path = tmp_path / "station.json"
    daemon = _daemon(config)
    daemon.start_service()
    daemon.listen("127.0.0.1", config.service.port)
    from pyrualean import ToolError
    from pyrualean.hosts import litearm_hardware as host
    from pyrualean.litearm_real import LitearmRealRobot

    cell = host.Cell(
        hardware_config=str(config_path),
        instruction="single-arm smoke",
        task="single_arm",
        suite="cells",
        seed=0,
        max_episode_steps=20,
        execution_mode="offline-fake",
    )
    toolkit = None
    try:
        toolkit, daemons = host.boot(cell, tmp_path / "run", hardware_config=str(config_path))
        assert daemons == []
        robot = LitearmRealRobot(host.make_backend(toolkit))
        state = robot.state()
        assert state.active_arms == ("left",)
        assert state.left is not None and state.right is None
        with pytest.raises(ValueError, match="not active"):
            state.arm("right")
        moved = robot.move_to(
            [state.left.tcp_pos[0], state.left.tcp_pos[1], state.left.tcp_pos[2] - 0.03],
            arm="left",
            path="line",
        )
        assert moved.reached is True, moved
        with pytest.raises(ToolError, match="not active"):
            robot.set_gripper("right", 1)
        card = host.dump_card(toolkit, cell, tmp_path / "run")
        assert "left measured TCP [x, y, z]" in card.facts
        assert "right measured TCP [x, y, z]" not in card.facts
    finally:
        if toolkit is not None:
            toolkit.close()
        daemon.close()


def _arm_only_config(tmp_path: Path, arm: str = "left"):
    return _config(
        tmp_path,
        active_arms=(arm,),
        devices=(arm,),
        cameras=(f"cam_{arm}_wrist",),
        primary_camera=f"cam_{arm}_wrist",
        gripper=False,
    )


def test_arm_only_station_observes_and_refuses_gripper_actions(tmp_path):
    config = _arm_only_config(tmp_path)
    assert config.arms["left"].gripper is None
    assert config.gripper_arms == ()
    daemon, client = _start(config)
    try:
        hello = client.hello or {}
        assert hello["cameras"] == ["cam_left_wrist"]
        assert hello["capabilities"]["grippers"] == {}

        toolkit = _toolkit(config, client, tmp_path)
        observation = toolkit.current_observation(refresh=True)
        left = observation.arm("left")
        assert left.gripper_opening is None
        assert left.gripper_width_m is None
        assert observation.state["left_ee_joint_state"] is None
        assert observation.state["left_ee_width_m"] is None

        import numpy as np

        xyz = np.asarray(left.tcp_pose).reshape(4, 4)[:3, 3]
        target = [float(xyz[0]), float(xyz[1]), float(xyz[2] - 0.03)]
        moved = move_pose(toolkit, target, arm="left", path="line")
        assert moved["reached"] is True, moved

        with pytest.raises(ValueError, match="no gripper configured"):
            grip(toolkit, arm="left", gripper=1)
        with pytest.raises(ValueError, match="no gripper configured"):
            grip(toolkit, arm="left", gripper=-1)
        with pytest.raises(ValueError, match="no gripper configured"):
            move_pose(toolkit, target, arm="left", gripper=1)
        with pytest.raises(ValueError, match="no gripper configured"):
            place_on(toolkit, 0.0, 0.3, 0.75, 0.05, arm="left")
        payload = toolkit.execute_tool("grip", {"arm": "left", "gripper": 1})
        assert "no gripper configured" in payload["error"]

        from pyrualean.hardware.protocol import command_digest

        grip_payload = {"arm": "left", "command": "close"}
        client.request(
            "command",
            {
                "command_id": "arm-only-grip",
                "kind": "grip",
                "payload": grip_payload,
                "digest": command_digest("grip", grip_payload),
            },
        )
        final = client.request(
            "wait_command", {"command_id": "arm-only-grip", "timeout_s": 3.0}, timeout=5.0
        )
        assert final["state"] == "rejected"
        assert final["error"]["code"] == "bad_request"
        assert "no gripper configured" in final["error"]["message"]
        assert final["error"]["detail"]["has_gripper"] is False

        from pyrualean.hosts import litearm_hardware as host

        cell = host.Cell(
            hardware_config=str(tmp_path / "station.json"),
            instruction="arm-only task",
            task="arm_only",
            suite="cells",
            seed=0,
            execution_mode="offline-fake",
        )
        card = host.dump_card(toolkit, cell, tmp_path / "run")
        assert card.facts["primary camera"] == "cam_left_wrist"
        assert (
            card.facts["left measured gripper opening / width"]
            == "not configured (arm-only station)"
        )

        observation = toolkit.current_observation(refresh=True)
        carry_payload = _trajectory_payload(
            config, toolkit, [observation.arm("left").q_model], carry=0.5
        )
        client.request(
            "command",
            {
                "command_id": "arm-only-carry",
                "kind": "trajectory",
                "payload": carry_payload,
                "digest": command_digest("trajectory", carry_payload),
            },
        )
        carried = client.request(
            "wait_command", {"command_id": "arm-only-carry", "timeout_s": 3.0}, timeout=5.0
        )
        assert carried["state"] == "rejected"
        assert carried["error"]["code"] == "bad_request"
        assert "no gripper configured" in carried["error"]["message"]
    finally:
        _stop(daemon, client)


def test_lite_arm_station_opens_reads_and_stops_without_a_gripper(tmp_path, monkeypatch):
    from pyrualean.hardware import devices
    from pyrualean.hardware.devices import DeviceError, LiteArmStation
    from pyrualean.hardware.protocol import StopKind

    config = _arm_only_config(tmp_path)
    handles: list[Any] = []

    class _FakeArmHandle:
        def __init__(
            self,
            arm_config,
            calibration,
            *,
            device_name: str,
            transport_factory=None,
        ) -> None:
            self.config = arm_config
            self.calibration = calibration
            self.device_name = device_name
            self.transport_factory = transport_factory
            self.firmware = "Litearm.2.1.0"
            self.opened = False
            self.closed = False

        def open(self) -> None:
            self.opened = True

        def read(self) -> dict[str, Any]:
            q = list(self.calibration.safe_home_rad)
            return {
                "q_model": q,
                "q_sdk": q,
                "dq": [0.0] * 7,
                "enabled": True,
                "fault": False,
                "data_age_s": 0.01,
            }

        def verify_enabled(self) -> list[str]:
            return []

        def emergency_stop(self) -> tuple[bool, str | None, str | None]:
            return True, "emergency_stop", None

        def close(self) -> None:
            self.closed = True

    def make_arm_handle(arm_config, calibration, *, device_name: str, transport_factory=None):
        handle = _FakeArmHandle(
            arm_config,
            calibration,
            device_name=device_name,
            transport_factory=transport_factory,
        )
        handles.append(handle)
        return handle

    def refuse_gripper(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an arm-only station must not open a gripper handle")

    monkeypatch.setattr(devices, "_ArmHandle", make_arm_handle)
    monkeypatch.setattr(devices, "_GripperHandle", refuse_gripper)
    monkeypatch.setattr(LiteArmStation, "_verify_firmware_pose", lambda self, arm: None)

    station = LiteArmStation(config, config.calibration)
    station.open()
    try:
        assert len(handles) == 1 and handles[0].opened is True
        assert set(station.arms) == {"left"}
        assert station.grippers == {}
        assert station.capabilities()["grippers"] == {}
        reading = station.read()
        assert set(reading["arms"]) == {"left"}
        assert reading["grippers"] == {}
        assert station.verify_ready() == []

        import threading

        with pytest.raises(DeviceError, match="no gripper configured"):
            station.set_gripper(
                "left",
                "open",
                width_m=None,
                force_n=None,
                timeout_s=1.0,
                cancel=threading.Event(),
            )
        cooperative = station.stop(StopKind.CANCEL)
        assert cooperative.confirmed is True
        assert cooperative.state == "stopped_between_segments"
        emergency = station.stop(StopKind.EMERGENCY)
        assert emergency.confirmed is True
        assert emergency.state == "stopped"
    finally:
        station.close()
    assert handles[0].closed is True
