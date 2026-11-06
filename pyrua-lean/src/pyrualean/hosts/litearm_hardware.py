# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The host of ``--robot litearm_real``: the hardware service as the backing store.

This host does not start RoboDojo, RPent, a simulator or XPolicyLab.  It
connects to the already-running hardware service over localhost (or a remote
address in the same protocol), opens an exclusive session, verifies the
configuration/calibration hashes and builds a
:class:`pyrualean.hardware.toolkit.HardwareToolkit`.

The ``Cell`` fields ``suite``, ``task`` and ``seed`` are run identifiers only.
They are never used to look up a simulator task or layout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..hardware.config import ExecutionMode, load_station
from ..hardware.perception import Perception
from ..hardware.rpc import HardwareClient
from ..hardware.toolkit import HardwareBackend, HardwareToolkit
from ..prompt import TaskCard

ROBOT = "litearm_real"


@dataclass(frozen=True)
class Cell:
    """One hardware episode request.

    suite/task/seed are recorded; only ``instruction`` is shown to the model as
    the task.  ``max_episode_steps`` is the device-action budget (trajectory
    segments / gripper actions), not simulator frames.
    """

    hardware_config: str
    instruction: str
    station_id: str = ""
    task: str = "hardware_task"
    suite: str = "hardware"
    seed: int = 0
    max_episode_steps: int = 300
    execution_mode: str | None = None
    cuda_device: int | None = None

    @property
    def tag(self) -> str:
        return f"{self.task}-{self.seed}"


def add_cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task",
        default="hardware_task",
        help="run identifier recorded with the episode (not a simulator task lookup)",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="natural-language task instruction shown to the model (required for a real run)",
    )
    parser.add_argument(
        "--hardware-config",
        required=True,
        help="absolute path to the station JSON configuration",
    )
    parser.add_argument(
        "--execution-mode",
        choices=[m.value for m in ExecutionMode],
        default=None,
        help="override execution.mode; live requires an executable calibration",
    )
    parser.add_argument("--suite", default="hardware", help="run label (default: hardware)")


def cell_from_args(args: argparse.Namespace) -> Cell:
    config_path = Path(str(args.hardware_config)).expanduser()
    if not config_path.is_absolute():
        config_path = config_path.resolve()
    instruction = getattr(args, "instruction", None)
    if instruction is None:
        instruction = str(getattr(args, "task", "hardware_task"))
    # Load once here so the cell records the real station id and a live mode
    # with an unverified calibration fails before any session is opened.
    config = load_station(config_path, mode=getattr(args, "execution_mode", None))
    return Cell(
        hardware_config=str(config_path),
        instruction=str(instruction),
        station_id=config.station_id,
        task=str(getattr(args, "task", "hardware_task")),
        suite=str(getattr(args, "suite", "hardware") or "hardware"),
        seed=int(getattr(args, "seed", 0) or 0),
        max_episode_steps=int(getattr(args, "max_episode_steps", 300) or 300),
        execution_mode=getattr(args, "execution_mode", None),
        cuda_device=getattr(args, "cuda_device", None),
    )


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    # The station config owns service, cameras, SAM3 and results settings.  No
    # RoboDojo/RPent path is accepted here.
    return None


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hardware_config": str(args.hardware_config),
        "execution_mode": getattr(args, "execution_mode", None),
        "instruction": getattr(args, "instruction", None),
        "env_only": bool(getattr(args, "env_only", False)),
    }


def _perception(config: Any, cell: Cell) -> Perception:
    sam3 = config.sam3
    if cell.cuda_device is not None and sam3.mode == "local":
        from dataclasses import replace

        # The station file remains the source of truth for the checkpoint; the
        # CLI may only move the GPU of a local SAM3.  An HTTP SAM3 on Thor
        # ignores the agent-side GPU setting.
        sam3 = replace(sam3, cuda_device=int(cell.cuda_device))
    return Perception(config.calibration, sam3)


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    rpent_root: str | None = None,  # noqa: ARG001 - host interface; nothing of RPent's runs
    env_only: bool = False,
    hardware_config: str | None = None,
    execution_mode: str | None = None,
    instruction: str | None = None,
    **_: Any,
) -> tuple[HardwareToolkit, list[Any]]:
    """Connect to the service, open an exclusive session and return the toolkit.

    No device is opened here: the service already owns it.  On a boot failure
    the session is closed and no motion command is left queued.
    """

    config = load_station(
        hardware_config or cell.hardware_config,
        mode=execution_mode or cell.execution_mode,
    )
    if cell.station_id and cell.station_id != config.station_id:
        raise ValueError(
            f"cell station {cell.station_id!r} does not match config {config.station_id!r}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = HardwareClient(config)
    toolkit: HardwareToolkit | None = None
    try:
        client.connect()
        client.open_session(client_id=f"pyrualean-{cell.tag}")
        perception = None if env_only else _perception(config, cell)
        toolkit = HardwareToolkit(
            config,
            client,
            task=cell.task,
            instruction=str(instruction or cell.instruction),
            out_dir=out_dir,
            perception=perception,
        )
        toolkit.max_device_actions = min(
            int(cell.max_episode_steps), int(config.results.max_episode_actions)
        )
        toolkit.refresh_observation(tool="boot")
        (out_dir / "hardware_host.json").write_text(
            json.dumps(
                {
                    "station_id": config.station_id,
                    "execution_mode": config.execution.mode.value,
                    "config_sha256": config.sha256,
                    "calibration_sha256": config.calibration_sha256,
                    "calibration_version": config.calibration.version,
                    "success_known": False,
                    "task": cell.task,
                    "instruction": toolkit.instruction,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return toolkit, []
    except BaseException:
        if toolkit is not None:
            toolkit.close()
        else:
            try:
                if client.session is not None:
                    client.close_session()
            except Exception:  # noqa: BLE001 - already failing
                pass
            client.close(force=True)
        raise


def make_backend(toolkit: HardwareToolkit) -> HardwareBackend:
    return HardwareBackend(toolkit)


def runtime_metadata(toolkit: HardwareToolkit) -> dict[str, Any]:
    return toolkit.runtime_metadata()


def stop_metadata(toolkit: HardwareToolkit) -> dict[str, Any]:
    return toolkit.stop_metadata()


def dump_card(toolkit: HardwareToolkit, cell: Cell, out_dir: Path) -> TaskCard:
    """Write a task card from the measured initial state.  No object pose is invented."""

    observation = toolkit.current_observation(refresh=False)
    card_dir = Path(out_dir) / "card"
    card_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, str] = {}
    for camera in observation.camera_state:
        target = card_dir / f"{camera}.png"
        try:
            target.write_bytes(toolkit.artifact(f"{camera}.png", observation.step))
        except Exception as exc:  # noqa: BLE001 - report rather than fake an image
            (card_dir / f"{camera}.error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            continue
        images[camera] = str(target)
    table = toolkit.calibration.table
    effective_budget = toolkit.max_device_actions or cell.max_episode_steps
    facts: dict[str, Any] = {
        "station": toolkit.config.station_id,
        "execution mode": toolkit.config.execution.mode.value,
        "config sha256": toolkit.config.sha256,
        "calibration version / sha256": (
            f"{toolkit.calibration.version} / {toolkit.calibration.sha256}"
        ),
        "world origin": "centre of the table edge nearest the robots",
        "axes": "+x right, +y into the table, +z up; table surface z = 0; metres, radians",
        "quaternion order": "[w, x, y, z]",
        "table extent": (
            f"x [{table.x_min_m}, {table.x_max_m}] m, y [{table.y_min_m}, {table.y_max_m}] m"
        ),
        "available cameras": ", ".join(sorted(observation.camera_state)),
        "primary camera": toolkit.config.primary_camera,
        "device action budget": effective_budget,
        "configured joint speed ceiling": (
            f"{toolkit.config.execution.max_joint_speed_rad_s} rad/s at speed "
            f"fraction {toolkit.config.execution.max_speed}"
        ),
        "configured max segment / command timeout": (
            f"{toolkit.config.execution.max_segment_duration_s} s / "
            f"{toolkit.config.execution.command_timeout_s} s"
        ),
        "state.step semantics": "observation snapshot number, not a simulator step",
        "steps_used semantics": "executed synchronous trajectory segments / gripper actions",
        "task success": "not measured; environment_success_known=false",
    }
    for name in observation.arms:
        arm = observation.arms[name]
        facts[f"{name} measured TCP [x, y, z]"] = (
            f"[{arm.tcp_pose[0]:.4f}, {arm.tcp_pose[1]:.4f}, {arm.tcp_pose[2]:.4f}] m"
        )
        facts[f"{name} measured gripper opening / width"] = (
            "not configured (arm-only station)"
            if arm.gripper_opening is None or arm.gripper_width_m is None
            else f"{arm.gripper_opening:.3f} / {arm.gripper_width_m:.5f} m"
        )
        facts[f"{name} fault"] = arm.fault_detail or "none"
    card = TaskCard(
        robot=ROBOT,
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        task_language=cell.instruction,
        object_names=[],
        images=images,
        max_env_steps=effective_budget,
        facts=facts,
    )
    (card_dir / "card.json").write_text(card.to_json() + "\n", encoding="utf-8")
    return card


__all__ = [
    "Cell",
    "ROBOT",
    "add_boot_args",
    "add_cell_args",
    "boot",
    "boot_options",
    "cell_from_args",
    "dump_card",
    "make_backend",
    "runtime_metadata",
    "stop_metadata",
]
