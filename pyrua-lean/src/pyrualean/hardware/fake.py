# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Explicit synthetic station for the offline end-to-end path.

This module is selected **only** by ``execution.mode = "offline-fake"``.  A
real driver failure never falls back here.  The fake has deterministic simple
dynamics and a single virtual object between the jaws; it exists to exercise
the runner -> robo API -> toolkit -> localhost RPC -> service chain, not to
claim anything about physical grasping.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from .config import Calibration, StationConfig
from .devices import (
    DeviceCancelled,
    DeviceError,
    GripperOutcome,
    MotionOutcome,
    NotReady,
    StationBase,
)
from .kinematics import LitearmKinematics, matrix_to_rpy
from .protocol import StopKind, StopResult, now_monotonic_ns, now_wall_ns


class FakeStation(StationBase):
    fake = True

    def __init__(self, config: StationConfig, calibration: Calibration) -> None:
        self.config = config
        self.calibration = calibration
        self.active_arms = tuple(config.execution.active_arms)
        self._active = frozenset(self.active_arms)
        self.gripper_arms = tuple(
            arm for arm in self.active_arms if config.arms[arm].gripper is not None
        )
        self._gripper_set = frozenset(self.gripper_arms)
        self.kinematics = LitearmKinematics(calibration)
        settings = config.raw.get("fake", {}) if isinstance(config.raw, dict) else {}
        self.time_scale = float(settings.get("time_scale", 0.02))
        self.object_present = bool(settings.get("object_present", True))
        self.object_width_m = float(settings.get("object_width_m", 0.030))
        self.slip_after_segments = int(settings.get("slip_after_segments", -1))
        self._segments = 0
        self._q = {
            arm: np.asarray(calibration.arm(arm).safe_home_rad, dtype=float)
            for arm in self.active_arms
        }
        self._dq = {arm: np.zeros(7) for arm in self.active_arms}
        self._width = {arm: calibration.arm(arm).gripper.open_m for arm in self.gripper_arms}
        self._command = {arm: "open" for arm in self.gripper_arms}
        self._holding = {arm: False for arm in self.gripper_arms}
        self._enabled = True
        self._fault = False
        self._fault_detail: str | None = None
        self._cancel = threading.Event()
        self._lock = threading.RLock()

    def _require_active(self, arm: str, *, device: str) -> None:
        if arm not in self._active:
            raise NotReady(
                f"arm {arm!r} is not active on this station; active arms: {list(self.active_arms)}",
                device=device,
            )

    def open(self) -> None:
        return

    def close(self) -> None:
        self._cancel.set()

    def capabilities(self) -> dict[str, Any]:
        return {
            "fake": True,
            "sdk": "synthetic (offline-fake mode only)",
            "active_arms": list(self.active_arms),
            "arms": {
                arm: {"firmware": "FakeLitearm2.1.0-7J", "joints": 7, "port": None}
                for arm in self.active_arms
            },
            "grippers": {
                arm: {"channel": "fake", "can_id": 8, "force_n": 15.0} for arm in self.gripper_arms
            },
        }

    def _arm_reading(self, arm: str) -> dict[str, Any]:
        cal = self.calibration.arm(arm)
        q = self._q[arm]
        q_sdk = [float(v) for v in cal.sdk_from_model(q)]
        flange = self.kinematics.flange_pose(arm, q)
        base = np.asarray(cal.base_pose)
        local = np.linalg.inv(base) @ flange
        rpy = matrix_to_rpy(local[:3, :3])
        tool = self.kinematics.fk(arm, q)
        tool_local = np.linalg.inv(base) @ tool
        tool_rpy = matrix_to_rpy(tool_local[:3, :3])
        return {
            "q_sdk": q_sdk,
            "q_model": [float(v) for v in q],
            "dq": [float(v) for v in self._dq[arm]],
            "tau": [0.0] * 7,
            "enabled": self._enabled and not self._fault,
            "mode": 0,
            "flags": 0,
            "fault": self._fault,
            "fault_detail": self._fault_detail,
            "joint_fault": 0,
            "flange_pose_raw": [*[float(v) for v in local[:3, 3]], *[float(v) for v in rpy]],
            "tool_pose_raw": [
                *[float(v) for v in tool_local[:3, 3]],
                *[float(v) for v in tool_rpy],
            ],
            "firmware": "FakeLitearm2.1.0-7J",
            "data_age_s": 0.0,
        }

    def read(self) -> dict[str, Any]:
        with self._lock:
            grippers = {}
            for arm in self.gripper_arms:
                cal = self.calibration.arm(arm).gripper
                width = self._width[arm]
                grippers[arm] = {
                    "width_m": width,
                    "opening": cal.width_to_opening(width),
                    "force_n": 12.0 if self._holding[arm] else 0.0,
                    "enabled": self._enabled and not self._fault,
                    "error_code": 1 if self._enabled and not self._fault else 0,
                    "fresh": True,
                    "data_age_s": 0.0,
                    "command": self._command[arm],
                    "force_calibrated": False,
                }
            return {
                "arms": {arm: self._arm_reading(arm) for arm in self.active_arms},
                "grippers": grippers,
                "captured_at_wall_ns": now_wall_ns(),
                "captured_at_monotonic_ns": now_monotonic_ns(),
            }

    def verify_ready(self) -> list[str]:
        if self._fault:
            return [f"fake station fault: {self._fault_detail}"]
        if not self._enabled:
            return ["fake station is disabled"]
        return []

    def move_joints(
        self,
        arm: str,
        q_sdk: list[float],
        speed: float,
        *,
        cancel: threading.Event,
        timeout_s: float,
    ) -> MotionOutcome:
        self._require_active(arm, device=f"fake_{arm}")
        with self._lock:
            if self._fault:
                raise NotReady(f"station fault: {self._fault_detail}", device=f"fake_{arm}")
            start = now_wall_ns()
            t0 = time.monotonic()
            target_model = self.calibration.arm(arm).model_from_sdk(q_sdk)
            start_q = self._q[arm].copy()
            delta = target_model - start_q
            steps = max(
                1, int(math.ceil(float(np.max(np.abs(delta))) / max(0.02, 0.25 * float(speed))))
            )
            steps = min(steps, 400)
            for i in range(1, steps + 1):
                if self._cancel.is_set() or cancel.is_set():
                    self._dq[arm] = np.zeros(7)
                    raise DeviceCancelled(
                        "stop requested during the fake segment", device=f"fake_{arm}"
                    )
                self._q[arm] = start_q + delta * (i / steps)
                self._dq[arm] = delta / steps / max(self.time_scale, 1e-4)
                time.sleep(max(0.0001, self.time_scale))
            self._q[arm] = target_model.copy()
            self._dq[arm] = np.zeros(7)
            self._segments += 1
            if (
                self.slip_after_segments >= 0
                and self._segments > self.slip_after_segments
                and self._holding.get(arm, False)
            ):
                self._holding[arm] = False
                self._width[arm] = self.calibration.arm(arm).gripper.closed_m
            q_now = [float(v) for v in self._q[arm]]
            if time.monotonic() - t0 > timeout_s + 0.5:
                raise NotReady(
                    f"fake segment exceeded the timeout ({time.monotonic() - t0:.2f}s)",
                    device=f"fake_{arm}",
                )
            return MotionOutcome(
                arm=arm,
                q_model=tuple(q_now),
                q_sdk=tuple(float(v) for v in self.calibration.arm(arm).sdk_from_model(q_now)),
                q_target_model=tuple(float(v) for v in target_model),
                reached=True,
                max_joint_error_rad=0.0,
                started_at_wall_ns=start,
                finished_at_wall_ns=now_wall_ns(),
                elapsed_s=time.monotonic() - t0,
                raw={"fake": True},
            )

    def set_gripper(
        self,
        arm: str,
        command: str,
        *,
        width_m: float | None,
        force_n: float | None,
        timeout_s: float,
        cancel: threading.Event,
    ) -> GripperOutcome:
        self._require_active(arm, device=f"fake_gripper_{arm}")
        if arm not in self._gripper_set:
            raise DeviceError(
                f"arm {arm!r} has no gripper configured on this station",
                device=f"fake_gripper_{arm}",
            )
        with self._lock:
            if self._fault:
                raise NotReady(f"station fault: {self._fault_detail}", device=f"fake_gripper_{arm}")
            started = now_wall_ns()
            t0 = time.monotonic()
            cal = self.calibration.arm(arm).gripper
            if command == "close":
                if self._command[arm] == "closed" and self._holding[arm]:
                    pass
                else:
                    self._command[arm] = "closed"
                    if self.object_present and width_m is None:
                        self._width[arm] = min(cal.open_m, max(cal.closed_m, self.object_width_m))
                        self._holding[arm] = True
                    else:
                        self._width[arm] = cal.closed_m
                        self._holding[arm] = False
                    if cancel.is_set():
                        self._command[arm] = "open"
                        self._holding[arm] = False
                        raise DeviceCancelled(
                            "stop requested during the fake grasp", device=f"fake_gripper_{arm}"
                        )
            elif command == "open":
                self._command[arm] = "open"
                self._holding[arm] = False
                target = (
                    cal.open_m if width_m is None else min(cal.open_m, max(cal.closed_m, width_m))
                )
                self._width[arm] = target
            elif command == "keep":
                pass
            else:
                raise NotReady(f"unknown gripper command {command!r}", device=f"fake_gripper_{arm}")
            time.sleep(max(0.0001, self.time_scale))
            width = self._width[arm]
            return GripperOutcome(
                arm=arm,
                command=command,
                opening=cal.width_to_opening(width),
                width_m=width,
                holding=bool(self._holding[arm]),
                settled=True,
                force_n=12.0 if self._holding[arm] else 0.0,
                error_code=1,
                started_at_wall_ns=started,
                finished_at_wall_ns=now_wall_ns(),
                elapsed_s=time.monotonic() - t0,
                released=command == "open",
                raw={"fake": True},
            )

    def request_gripper_abort(self) -> None:
        for arm in self.gripper_arms:
            self._holding[arm] = False
            self._command[arm] = "open"

    def stop(self, kind: StopKind, arms: tuple[str, ...] | None = None) -> StopResult:
        targets = tuple(self.active_arms if arms is None else arms)
        with self._lock:
            requested = now_wall_ns()
            if kind is StopKind.CANCEL:
                self._cancel.set()
                max_dq = max(float(np.max(np.abs(self._dq[a]))) for a in targets)
                confirmed = max_dq <= float(self.config.safety.joint_speed_confirm_rad_s)
                return StopResult(
                    requested=True,
                    kind=kind.value,
                    confirmed=confirmed,
                    state="stopped_between_segments" if confirmed else "stop_unconfirmed",
                    method="fake_cooperative",
                    message=None if confirmed else f"max |dq|={max_dq:.3f} rad/s",
                    requested_at_wall_ns=requested,
                    confirmed_at_wall_ns=now_wall_ns(),
                )
            self._cancel.set()
            self.request_gripper_abort()
            for arm in targets:
                self._dq[arm] = np.zeros(7)
            self._fault = True
            self._fault_detail = f"fake {kind.value} stop"
            return StopResult(
                requested=True,
                kind=kind.value,
                confirmed=True,
                state="stopped",
                method="fake_emergency_stop",
                message=(
                    "synthetic station records a fault; physical hold behaviour is not modelled"
                ),
                requested_at_wall_ns=requested,
                confirmed_at_wall_ns=now_wall_ns(),
            )


__all__ = ["FakeStation"]
