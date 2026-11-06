# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Build the PyRUA-Lean observation from measured hardware state.

The observation keeps the shape the existing LiteArm robot expects, but every
pose and gripper width in it is derived from a measured encoder value and a
calibrated transform.  A commanded target is never copied into a measured
field:

* ``<arm>_ee_pose`` is the TCP (grasp point) pose computed by FK from the
  encoder angles, the calibrated base and the calibrated flange-to-TCP
  transform;
* ``<arm>_ee_joint_state`` is the normalised opening derived from the LiteGrip
  measured position and its per-unit closed/open calibration;
* each arm carries its measured joint vector, velocities, torques, firmware
  flags and a data age; each gripper carries measured width, force, freshness
  and its last command separately.

The world frame is the one the station calibration defines: origin at the
centre of the table edge nearest the robots, +x right, +y into the table, +z
up, table surface ``z = 0``.  Quaternions are ``[w, x, y, z]``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Calibration
from .kinematics import LitearmKinematics, angles, quat_wxyz

ARMS = ("left", "right")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class StateError(ValueError):
    """A measured state is missing, malformed or too old to use."""


@dataclass(frozen=True)
class ArmMeasurement:
    arm: str
    q_model: tuple[float, ...]
    q_sdk: tuple[float, ...]
    dq: tuple[float, ...]
    tau: tuple[float, ...]
    tcp_pose: tuple[float, ...]
    flange_pose: tuple[float, ...]
    gripper_width_m: float | None
    gripper_opening: float | None
    gripper_force_n: float | None
    gripper_enabled: bool | None
    gripper_error: int | None
    gripper_fresh: bool | None
    gripper_age_s: float | None
    gripper_command: str | None
    enabled: bool
    mode: int | None
    flags: int | None
    fault: bool
    fault_detail: str | None
    age_s: float | None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def as_state_entry(self) -> dict[str, Any]:
        tcp = np.asarray(self.tcp_pose, dtype=float).reshape(4, 4)
        flange = np.asarray(self.flange_pose, dtype=float).reshape(4, 4)
        pitch, yaw, roll = angles(tcp[:3, :3])
        quat = quat_wxyz(tcp[:3, :3])
        return {
            "tcp_pos": [float(v) for v in tcp[:3, 3]],
            "tcp_quat_wxyz": list(quat),
            "eef_pos": [float(v) for v in tcp[:3, 3]],
            "eef_quat_wxyz": list(quat),
            "flange_pos": [float(v) for v in flange[:3, 3]],
            "flange_quat_wxyz": list(quat_wxyz(flange[:3, :3])),
            "pitch": round(float(pitch), 4),
            "yaw": round(float(yaw), 4),
            "roll": round(float(roll), 4),
            "joints_model_rad": [float(v) for v in self.q_model],
            "joints_sdk_rad": [float(v) for v in self.q_sdk],
            "joint_velocity_rad_s": [float(v) for v in self.dq],
            "joint_torque_nm": [float(v) for v in self.tau],
            "gripper_opening": (
                None if self.gripper_opening is None else float(self.gripper_opening)
            ),
            "gripper_width_m": (
                None if self.gripper_width_m is None else float(self.gripper_width_m)
            ),
            "gripper_force_n": self.gripper_force_n,
            "gripper_enabled": self.gripper_enabled,
            "gripper_error_code": self.gripper_error,
            "gripper_fresh": self.gripper_fresh,
            "gripper_data_age_s": self.gripper_age_s,
            "gripper_command": self.gripper_command,
            "arm_enabled": self.enabled,
            "arm_mode": self.mode,
            "arm_flags": self.flags,
            "fault": bool(self.fault),
            "fault_detail": self.fault_detail,
            "data_age_s": self.age_s,
            "measured": True,
        }


@dataclass(frozen=True)
class CameraMeasurement:
    name: str
    camera_to_world: tuple[float, ...]
    intrinsic: tuple[float, ...]
    distortion: tuple[float, ...]
    shape: tuple[int, int, int]
    captured_at_wall_ns: int
    captured_at_monotonic_ns: int
    color_artifact: str | None
    depth_artifact: str | None
    has_depth: bool
    arm: str | None
    q_model_at_capture: tuple[float, ...] | None
    age_s: float | None
    timestamp_source: str

    def as_vision_entry(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "intrinsic_matrix": _matrix_rows(self.intrinsic, 3, 3),
            "extrinsic_matrix": _matrix_rows(self.camera_to_world, 4, 4),
            "camera_to_world": _matrix_rows(self.camera_to_world, 4, 4),
            "distortion_coefficients": list(self.distortion),
            "color_artifact": self.color_artifact,
            "depth_artifact": self.depth_artifact,
            "has_depth": self.has_depth,
            "arm": self.arm,
            "q_model_at_capture": (
                [float(v) for v in self.q_model_at_capture]
                if self.q_model_at_capture is not None
                else None
            ),
            "captured_at_wall_ns": self.captured_at_wall_ns,
            "captured_at_monotonic_ns": self.captured_at_monotonic_ns,
            "timestamp_source": self.timestamp_source,
            "data_age_s": self.age_s,
        }


@dataclass(frozen=True)
class Observation:
    step: int
    task: str
    instruction: str
    arms: Mapping[str, ArmMeasurement]
    cameras: Mapping[str, CameraMeasurement]
    state: Mapping[str, Any]
    camera_state: Mapping[str, Any]
    execution_state: str
    execution_mode: str
    station_id: str
    config_sha256: str
    calibration_sha256: str
    calibration_version: str
    captured_at_wall_ns: int
    captured_at_monotonic_ns: int
    age_s: float | None
    success_known: bool = False
    fault: bool = False
    stop: Mapping[str, Any] | None = None
    raw_snapshot: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def arm(self, name: str) -> ArmMeasurement:
        try:
            return self.arms[name]
        except KeyError:
            raise StateError(f"arm {name!r} is not in the observation") from None

    def camera(self, name: str) -> CameraMeasurement:
        try:
            return self.cameras[name]
        except KeyError:
            raise StateError(f"camera {name!r} is not in the observation") from None

    def envelope(self, *, truncated: bool = False) -> dict[str, Any]:
        return {
            "step": int(self.step),
            # ``robo.task`` is the natural-language instruction.  The run id is
            # kept separately as task_id; it is not a simulator task lookup.
            "task_language": self.instruction,
            "task_id": self.task,
            "instruction": self.instruction,
            "terminated": False,
            "truncated": bool(truncated),
            "success_known": bool(self.success_known),
            "state": {
                "state": dict(self.state),
                "instruction": self.instruction,
                "cameras": dict(self.camera_state),
                "hardware": {
                    "execution_state": self.execution_state,
                    "execution_mode": self.execution_mode,
                    "station_id": self.station_id,
                    "config_sha256": self.config_sha256,
                    "calibration_sha256": self.calibration_sha256,
                    "calibration_version": self.calibration_version,
                    "captured_at_wall_ns": self.captured_at_wall_ns,
                    "captured_at_monotonic_ns": self.captured_at_monotonic_ns,
                    "data_age_s": self.age_s,
                    "measured": True,
                    "success_known": False,
                    "stop": dict(self.stop) if self.stop else None,
                },
            },
            "log": {},
        }


def _matrix_rows(values: Sequence[float] | np.ndarray, rows: int, cols: int) -> list[list[float]]:
    array = np.asarray(values, dtype=float).reshape(rows, cols)
    return [[float(v) for v in row] for row in array]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StateError(f"{name} is not a number")
    out = float(value)
    if not math.isfinite(out):
        raise StateError(f"{name} is not finite")
    return out


def _vec(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise StateError(f"{name} must have {length} values")
    return tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(value))


def build_arms(
    calibration: Calibration,
    kinematics: LitearmKinematics,
    reading: Mapping[str, Any],
    *,
    max_age_s: float | None = None,
    active_arms: Sequence[str] | None = None,
) -> dict[str, ArmMeasurement]:
    """Turn the device service's measured arm/gripper dict into observations.

    ``active_arms`` defaults to every arm for backward compatibility; when a
    subset is given, the reading is only required to contain those arms and
    inactive arms are not represented in the observation.
    """

    raw_arms = reading.get("arms")
    raw_grippers = reading.get("grippers")
    if not isinstance(raw_arms, dict) or not isinstance(raw_grippers, dict):
        raise StateError("device reading is missing arms/grippers")
    arms = tuple(ARMS if active_arms is None else active_arms)
    out: dict[str, ArmMeasurement] = {}
    now = time.monotonic()
    for arm in arms:
        if arm not in raw_arms:
            raise StateError(f"device reading is missing {arm}")
        raw = raw_arms[arm]
        # A missing gripper reading means this arm has no gripper configured.
        # LiteArmStation/FakeStation only emit grippers they actually own, so a
        # configured gripper that failed to read raises before this point rather
        # than being mistaken for an absent one.
        grip = raw_grippers.get(arm)
        if grip is not None and not isinstance(grip, dict):
            raise StateError(f"grippers.{arm} is not an object")
        q_model = _vec(raw.get("q_model"), f"arms.{arm}.q_model", 7)
        q_sdk = _vec(raw.get("q_sdk"), f"arms.{arm}.q_sdk", 7)
        dq = _vec(raw.get("dq"), f"arms.{arm}.dq", 7)
        tau = _vec(raw.get("tau"), f"arms.{arm}.tau", 7)
        # Cross-check the map both ways; a bad calibration must not silently
        # produce a plausible-looking pose.
        mapped = calibration.arm(arm).sdk_from_model(q_model)
        if max(abs(float(a) - float(b)) for a, b in zip(mapped, q_sdk, strict=True)) > 1e-5:
            raise StateError(f"arms.{arm}: SDK/model joint maps disagree with the reading")
        fl = kinematics.flange_pose(arm, q_model)
        tcp = kinematics.fk(arm, q_model)
        raw_tcp = raw.get("flange_pose_raw")
        if raw_tcp is not None:
            # The firmware flange pose is evidence, not the control state.  We
            # compare positions only; the firmware's rpy convention is checked
            # by the smoke test, and a mismatch is reported rather than hidden.
            raw_tcp_v = _vec(raw_tcp, f"arms.{arm}.flange_pose_raw", 6)
            base = np.asarray(calibration.arm(arm).base_pose, dtype=float)
            raw_world = base[:3, :3] @ np.asarray(raw_tcp_v[:3]) + base[:3, 3]
            delta = float(np.linalg.norm(raw_world - fl[:3, 3]))
            if delta > 0.02:
                raise StateError(
                    f"arms.{arm}: firmware flange position disagrees with the calibrated FK "
                    f"by {delta:.4f} m"
                )
        width: float | None = None
        opening: float | None = None
        force_n: float | None = None
        grip_enabled: bool | None = None
        grip_error: int | None = None
        grip_fresh: bool | None = None
        grip_age: float | None = None
        grip_command: str | None = None
        if grip is not None:
            width = _finite(grip.get("width_m"), f"grippers.{arm}.width_m")
            grip_cal = calibration.arm(arm).gripper
            if width < grip_cal.closed_m - 0.01 or width > grip_cal.open_m + 0.01:
                raise StateError(
                    f"grippers.{arm}.width_m={width:.4f} is outside the calibrated "
                    f"[{grip_cal.closed_m:.4f}, {grip_cal.open_m:.4f}] m"
                )
            opening = float(np.clip(grip_cal.width_to_opening(width), -0.05, 1.05))
            force_n = (
                _finite(grip["force_n"], f"grippers.{arm}.force_n")
                if grip.get("force_n") is not None
                else None
            )
            grip_enabled = bool(grip.get("enabled")) if grip.get("enabled") is not None else None
            grip_error = int(grip["error_code"]) if grip.get("error_code") is not None else None
            grip_fresh = bool(grip.get("fresh")) if grip.get("fresh") is not None else None
            grip_age = (
                _finite(grip["data_age_s"], f"grippers.{arm}.data_age_s")
                if grip.get("data_age_s") is not None
                else None
            )
            grip_command = grip.get("command")
        age = raw.get("data_age_s")
        age_s = None if age is None else _finite(age, f"arms.{arm}.data_age_s")
        if max_age_s is not None:
            if age_s is None:
                raise StateError(f"arms.{arm}: no measurable data age")
            if age_s > max_age_s:
                raise StateError(
                    f"arms.{arm}: state is {age_s:.2f}s old, over the {max_age_s:.2f}s limit"
                )
        out[arm] = ArmMeasurement(
            arm=arm,
            q_model=q_model,
            q_sdk=q_sdk,
            dq=dq,
            tau=tau,
            tcp_pose=tuple(float(v) for v in tcp.reshape(-1)),
            flange_pose=tuple(float(v) for v in fl.reshape(-1)),
            gripper_width_m=width,
            gripper_opening=opening,
            gripper_force_n=force_n,
            gripper_enabled=grip_enabled,
            gripper_error=grip_error,
            gripper_fresh=grip_fresh,
            gripper_age_s=grip_age,
            gripper_command=grip_command,
            enabled=bool(raw.get("enabled")),
            mode=int(raw["mode"]) if raw.get("mode") is not None else None,
            flags=int(raw["flags"]) if raw.get("flags") is not None else None,
            fault=bool(raw.get("fault")),
            fault_detail=raw.get("fault_detail"),
            age_s=age_s,
            raw=raw,
        )
    _ = now
    return out


def build_observation(
    calibration: Calibration,
    kinematics: LitearmKinematics,
    snapshot: Mapping[str, Any],
    *,
    task: str,
    instruction: str,
    execution_mode: str,
    max_age_s: float | None,
    now_wall_ns: int | None = None,
    active_arms: Sequence[str] | None = None,
) -> Observation:
    reading = snapshot.get("reading")
    if not isinstance(reading, dict):
        raise StateError("snapshot has no device reading")
    arms = tuple(ARMS if active_arms is None else active_arms)
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise StateError(f"active_arms contains unknown arms: {unknown}")
    measurements = build_arms(
        calibration, kinematics, reading, max_age_s=max_age_s, active_arms=arms
    )
    cameras_raw = reading.get("cameras")
    if not isinstance(cameras_raw, dict):
        raise StateError("device reading has no cameras")
    cameras: dict[str, CameraMeasurement] = {}
    now_wall = time.time_ns() if now_wall_ns is None else int(now_wall_ns)
    for name, raw in cameras_raw.items():
        if name not in CAMERAS:
            raise StateError(f"unknown camera {name!r}")
        if not isinstance(raw, dict):
            raise StateError(f"camera {name} reading is not an object")
        cap_wall = int(raw.get("captured_at_wall_ns") or 0)
        cap_mono = int(raw.get("captured_at_monotonic_ns") or 0)
        if cap_wall <= 0:
            raise StateError(f"camera {name} has no capture timestamp")
        age = max(0.0, (now_wall - cap_wall) / 1e9)
        if max_age_s is not None and age > max_age_s:
            raise StateError(
                f"camera {name} frame is {age:.2f}s old, over the {max_age_s:.2f}s limit"
            )
        arm = raw.get("arm")
        q_at = raw.get("q_model_at_capture")
        c2w = raw.get("camera_to_world")
        if c2w is None and arm in ARMS and q_at is not None:
            c2w = kinematics.camera_pose(name, arm, _vec(q_at, f"cameras.{name}.q", 7))
        if c2w is None:
            raise StateError(f"camera {name} has no camera-to-world transform")
        cameras[name] = CameraMeasurement(
            name=name,
            camera_to_world=tuple(float(v) for v in np.asarray(c2w, dtype=float).reshape(-1)),
            intrinsic=tuple(
                float(v)
                for v in np.asarray(
                    raw.get("intrinsic_matrix", calibration.camera(name).intrinsic),
                    dtype=float,
                ).reshape(-1)
            ),
            distortion=tuple(
                float(v)
                for v in raw.get("distortion_coefficients", calibration.camera(name).distortion)
            ),
            shape=tuple(
                int(v) for v in raw.get("shape", calibration.camera(name).resolution + (3,))
            ),
            captured_at_wall_ns=cap_wall,
            captured_at_monotonic_ns=cap_mono,
            color_artifact=raw.get("color_artifact"),
            depth_artifact=raw.get("depth_artifact"),
            has_depth=bool(raw.get("depth_artifact")),
            arm=arm,
            q_model_at_capture=(tuple(float(v) for v in q_at) if q_at is not None else None),
            age_s=age,
            timestamp_source=str(raw.get("timestamp_source", "host")),
        )
    # Only cameras mounted on an active arm (or a world camera) must appear in
    # the snapshot.  A wrist camera whose arm is inactive is neither opened by
    # the service nor required in the observation.
    required = {
        name for name, cam in calibration.cameras.items() if cam.arm is None or cam.arm in arms
    }
    missing = sorted(required - set(cameras))
    if missing:
        raise StateError(f"snapshot is missing calibrated cameras {missing}")
    state: dict[str, Any] = {}
    for arm in arms:
        entry = measurements[arm].as_state_entry()
        state[f"{arm}_ee_pose"] = [*entry["tcp_pos"], *entry["tcp_quat_wxyz"]]
        state[f"{arm}_arm_joint_state"] = entry["joints_model_rad"]
        state[f"{arm}_joint_velocity_rad_s"] = entry["joint_velocity_rad_s"]
        opening = entry["gripper_opening"]
        state[f"{arm}_ee_joint_state"] = None if opening is None else [opening]
        state[f"{arm}_ee_width_m"] = entry["gripper_width_m"]
        state[f"{arm}_gripper_force_n"] = entry["gripper_force_n"]
        state[f"{arm}_tcp_pos"] = entry["tcp_pos"]
        state[f"{arm}_tcp_quat_wxyz"] = entry["tcp_quat_wxyz"]
        state[f"{arm}_pitch_yaw_roll"] = [entry["pitch"], entry["yaw"], entry["roll"]]
        state[f"{arm}_flange_pos"] = entry["flange_pos"]
        state[f"{arm}_fault"] = entry["fault"]
        state[f"{arm}_fault_detail"] = entry["fault_detail"]
        state[f"{arm}_data_age_s"] = entry["data_age_s"]
        state[f"{arm}_gripper_command"] = entry["gripper_command"]
        state[f"{arm}_measured"] = True
    state["measured"] = True
    state["command_targets_are_measured"] = False
    camera_state = {name: cam.as_vision_entry() for name, cam in cameras.items()}
    hardware = snapshot.get("hardware") or {}
    captured_wall = int(
        hardware.get("captured_at_wall_ns") or snapshot.get("captured_at_wall_ns") or 0
    )
    captured_mono = int(
        hardware.get("captured_at_monotonic_ns") or snapshot.get("captured_at_monotonic_ns") or 0
    )
    age = max(0.0, (now_wall - captured_wall) / 1e9) if captured_wall else None
    if max_age_s is not None and age is not None and age > max_age_s:
        raise StateError(
            f"snapshot is {age:.2f}s old, over the {max_age_s:.2f}s limit; "
            "refusing to plan or report from it"
        )
    fault = any(a.fault for a in measurements.values()) or bool(hardware.get("fault"))
    return Observation(
        step=int(snapshot.get("index", snapshot.get("step", -1))),
        task=task,
        instruction=instruction,
        arms=measurements,
        cameras=cameras,
        state=state,
        camera_state=camera_state,
        execution_state=str(hardware.get("execution_state", "unknown")),
        execution_mode=execution_mode,
        station_id=str(snapshot.get("station_id", calibration.station_id)),
        config_sha256=str(snapshot.get("config_sha256", "")),
        calibration_sha256=str(snapshot.get("calibration_sha256", calibration.sha256)),
        calibration_version=calibration.version,
        captured_at_wall_ns=captured_wall,
        captured_at_monotonic_ns=captured_mono,
        age_s=age,
        success_known=False,
        fault=fault,
        stop=hardware.get("stop"),
        raw_snapshot=snapshot,
    )


__all__ = [
    "ARMS",
    "ArmMeasurement",
    "CAMERAS",
    "CameraMeasurement",
    "Observation",
    "StateError",
    "build_arms",
    "build_observation",
]
