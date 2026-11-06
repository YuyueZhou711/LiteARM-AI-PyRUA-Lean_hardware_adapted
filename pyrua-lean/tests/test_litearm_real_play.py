"""play_episode records hardware runtime/stop metadata after shutdown."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

from pyrualean.hardware.config import load_station
from pyrualean.hardware.daemon import HardwareDaemon
from pyrualean.play import play_episode

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ROOT / "configs" / "litearm_real.offline.example.json"
CALIBRATION = ROOT / "configs" / "calibration" / "litearm_real.example.json"


def _port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_play_records_stop_after_shutdown_and_no_task_success(tmp_path, monkeypatch):
    raw = json.loads(OFFLINE.read_text())
    raw["calibration"] = str(CALIBRATION)
    raw["service"]["port"] = _port()
    raw["service"]["heartbeat_s"] = 0.2
    raw["service"]["heartbeat_timeout_s"] = 0.8
    raw["snapshot"]["camera_joint_sync_s"] = 0.20
    raw["fake"]["time_scale"] = 0.005
    config_path = tmp_path / "station.json"
    config_path.write_text(json.dumps(raw))
    config = load_station(config_path)
    daemon = HardwareDaemon(config)
    daemon.start_service()
    daemon.listen("127.0.0.1", config.service.port)

    import pyrualean.mcp_server as mcp_server
    import pyrualean.play as play

    class _Server:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return "http://127.0.0.1:1/mcp"

        def stop(self):
            return None

    def fake_agent(
        prompt,
        images,
        toolkit,
        *,
        model,
        base_url,
        api_key,
        max_turns,
        budget_s,
        out_dir,
        arm,
        on_timeout,
        should_stop=None,
        **kwargs,
    ):
        first = toolkit.execute_tool(
            "python", {"code": "print('measured', robo.state().left.width_m)"}
        )
        assert "measured" in first.result["stdout"]
        toolkit.finish = {"status": "success", "summary": "the model says so"}
        return {
            "status": "completed",
            "returncode": 0,
            "errors": [],
            "tool_calls": {"python": 1},
            "max_turns_hit": False,
        }

    monkeypatch.setattr(mcp_server, "ArmMcpServer", _Server)
    monkeypatch.setattr(play, "run_openai_agent", fake_agent)
    old_cwd = os.getcwd()
    try:
        from pyrualean.hosts import litearm_hardware as host

        cell = host.Cell(
            hardware_config=str(config_path),
            instruction="put the red block in the tray",
            task="pick_place",
            suite="cells",
            seed=0,
            max_episode_steps=20,
            execution_mode="offline-fake",
        )
        out = tmp_path / "run"
        result = play_episode(
            cell,
            out,
            arm="cells",
            model="test-model",
            budget_s=30.0,
            runtime="openai",
            base_url="http://127.0.0.1:1/v1",
            robot="litearm_real",
            boot_options={
                "hardware_config": str(config_path),
                "execution_mode": "offline-fake",
                "instruction": cell.instruction,
            },
            primitives="no-vla",
        )
        assert result["status"] == "completed", result
        assert result["environment_success"] is False
        assert result["environment_success_known"] is False
        assert result["task_success"] is None
        assert result["host"]["execution_mode"] == "offline-fake"
        assert result["host"]["config_sha256"] == config.sha256
        assert result["host"]["calibration_sha256"] == config.calibration_sha256
        assert result["host"]["success_known"] is False
        stop = result["host"]["stop"]
        assert stop["requested"] is True
        assert stop["recorded_after_shutdown"] is True
        assert stop["environment_success_known"] is False
        episode = json.loads((out / "episode.json").read_text())
        assert episode["task_success"] == "not_measured"
        assert episode["finish"]["status"] == "success"
        assert episode["stop"]["requested"] is True
        assert "environment_success_known = false" in (out / "prompt.txt").read_text()
        assert "Task success is NOT measured" in (out / "prompt.txt").read_text()
    finally:
        os.chdir(old_cwd)
        daemon.close()


def test_play_single_arm_records_only_the_active_arm(tmp_path, monkeypatch):
    raw = json.loads(OFFLINE.read_text())
    raw["execution"]["active_arms"] = ["left"]
    raw["devices"] = {"left": raw["devices"]["left"]}
    raw["cameras"] = {
        name: cfg for name, cfg in raw["cameras"].items() if name in ("cam_head", "cam_left_wrist")
    }
    calibration = json.loads(CALIBRATION.read_text())
    calibration["arms"] = {"left": calibration["arms"]["left"]}
    calibration["cameras"] = {
        name: cfg
        for name, cfg in calibration["cameras"].items()
        if name in ("cam_head", "cam_left_wrist")
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    raw["calibration"] = str(calibration_path)
    raw["service"]["port"] = _port()
    raw["service"]["heartbeat_s"] = 0.2
    raw["service"]["heartbeat_timeout_s"] = 0.8
    raw["snapshot"]["camera_joint_sync_s"] = 0.20
    raw["fake"]["time_scale"] = 0.005
    config_path = tmp_path / "station.json"
    config_path.write_text(json.dumps(raw))
    config = load_station(config_path)
    daemon = HardwareDaemon(config)
    daemon.start_service()
    daemon.listen("127.0.0.1", config.service.port)

    import pyrualean.mcp_server as mcp_server
    import pyrualean.play as play

    class _Server:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return "http://127.0.0.1:1/mcp"

        def stop(self):
            return None

    def fake_agent(
        prompt,
        images,
        toolkit,
        *,
        model,
        base_url,
        api_key,
        max_turns,
        budget_s,
        out_dir,
        arm,
        on_timeout,
        should_stop=None,
        **kwargs,
    ):
        first = toolkit.execute_tool(
            "python",
            {
                "code": (
                    "s = robo.state()\n"
                    "print('active', s.active_arms)\n"
                    "print('right', s.right)\n"
                    "print('width', s.left.width_m)\n"
                )
            },
        )
        assert "('left',)" in first.result["stdout"]
        assert "right None" in first.result["stdout"]
        assert "width" in first.result["stdout"]
        toolkit.finish = {"status": "success", "summary": "the model says so"}
        return {
            "status": "completed",
            "returncode": 0,
            "errors": [],
            "tool_calls": {"python": 1},
            "max_turns_hit": False,
        }

    monkeypatch.setattr(mcp_server, "ArmMcpServer", _Server)
    monkeypatch.setattr(play, "run_openai_agent", fake_agent)
    old_cwd = os.getcwd()
    try:
        from pyrualean.hosts import litearm_hardware as host

        cell = host.Cell(
            hardware_config=str(config_path),
            instruction="single-arm task",
            task="single_arm",
            suite="cells",
            seed=0,
            max_episode_steps=20,
            execution_mode="offline-fake",
        )
        out = tmp_path / "run"
        result = play_episode(
            cell,
            out,
            arm="cells",
            model="test-model",
            budget_s=30.0,
            runtime="openai",
            base_url="http://127.0.0.1:1/v1",
            robot="litearm_real",
            boot_options={
                "hardware_config": str(config_path),
                "execution_mode": "offline-fake",
                "instruction": cell.instruction,
            },
            primitives="no-vla",
        )
        assert result["status"] == "completed", result
        assert result["environment_success_known"] is False
        assert result["host"]["execution_mode"] == "offline-fake"
        assert result["host"]["stop"]["requested"] is True
        episode = json.loads((out / "episode.json").read_text())
        assert episode["task_success"] == "not_measured"
        assert episode["stop"]["requested"] is True
    finally:
        os.chdir(old_cwd)
        daemon.close()
