# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Thin wrappers around the verified LiteArm and LiteGrip Python SDKs.

The vendor SDKs are imported **inside the device-creation functions**, never at
module import time, so the offline agent environment does not need either SDK.
The wrappers do not invent a fallback: if a driver cannot be imported or a
device does not answer, the error is explicit and the service refuses motion.

Verified against:
* ``litearm-python`` 2.1.0 at ``/home/hzh/Code/litearm-python``: seven joints,
  ``get_state(refresh=True)`` returning a ``Msg`` envelope, blocking
  ``movej_sync(q, speed)`` with ``speed`` in 0..1, ``get_tcp`` returning the
  flange pose, ``get_tool_tcp`` the tool pose, ``emergency_stop`` as a firmware
  command, and ``disable``/``park`` as separate operations.
* ``litegrip-python`` 2.2.0, local `/home/hzh/Code/NEXTFORM/litegrip-python`,
  source commit 6b5fdce: ``grasp(force_n,
  hold_s=0)`` blocks while holding and is ended by ``release()`` from another
  thread; ``get_state`` reports measured ``position_mm`` and freshness;
  ``move_at_speed`` is the bounded positional ramp; ``stop`` sends one
  zero-torque frame and is not a safety-rated stop.

What is *not* verified here: real firmware emergency-stop hold/disable
behaviour, whether an emergency stop preempts a blocking ``movej_sync`` on the
same serial session, CAN timing, gripper force calibration, and all physical
grasp behaviour.  These are field-acceptance items in
``docs/robots/litearm-real.md``.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ArmCalibration, Calibration, ExecutionMode, StationConfig
from .kinematics import LitearmKinematics, matrix_to_rpy
from .protocol import StopKind, StopResult, now_monotonic_ns, now_wall_ns


class DeviceError(RuntimeError):
    """A vendor SDK call failed; the message names the device and keeps the cause."""

    def __init__(
        self,
        message: str,
        *,
        device: str,
        fault_detail: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(f"{device}: {message}")
        self.device = device
        self.fault_detail = fault_detail
        self.cause = cause


class DeviceCancelled(DeviceError):
    """The caller asked to stop before a synchronous segment was sent."""


class NotReady(DeviceError):
    """The device is connected but cannot be commanded (disabled, faulted, ...)."""


@dataclass
class MotionOutcome:
    arm: str
    q_model: tuple[float, ...]
    q_sdk: tuple[float, ...]
    q_target_model: tuple[float, ...]
    reached: bool
    max_joint_error_rad: float
    started_at_wall_ns: int
    finished_at_wall_ns: int
    elapsed_s: float
    fault: bool = False
    fault_detail: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class GripperOutcome:
    arm: str
    command: str
    opening: float
    width_m: float
    holding: bool
    settled: bool
    force_n: float | None
    error_code: int | None
    started_at_wall_ns: int
    finished_at_wall_ns: int
    elapsed_s: float
    released: bool = False
    note: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class StationBase:
    """Interface the daemon uses; the fake station implements the same one."""

    fake = False

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Any]:
        raise NotImplementedError

    def read(self) -> dict[str, Any]:
        """Return ``{"arms": ..., "grippers": ..., "timestamp...": ...}``."""

        raise NotImplementedError

    def verify_ready(self) -> list[str]:
        """Reasons the station cannot execute; empty means ready."""

        raise NotImplementedError

    def move_joints(
        self,
        arm: str,
        q_sdk: list[float],
        speed: float,
        *,
        cancel: threading.Event,
        timeout_s: float,
    ) -> MotionOutcome:
        raise NotImplementedError

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
        raise NotImplementedError

    def stop(self, kind: StopKind, arms: tuple[str, ...] | None = None) -> StopResult:
        """Stop the station.  ``None`` means the station's active arms."""

        raise NotImplementedError

    def request_gripper_abort(self) -> None:
        """Best-effort non-blocking release for emergency paths."""

        raise NotImplementedError


# ---------------------------------------------------------------------------
# real SDK wrappers
# ---------------------------------------------------------------------------


def _import_sdk(module_name: str, device: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise DeviceError(
            f"{module_name} is not installed in the device environment; install the "
            "pinned wheel built from the verified SDK source",
            device=device,
            cause=exc,
        ) from None


def _parse_firmware(text: str) -> tuple[str, tuple[int, ...]]:
    value = str(text or "").strip()
    prefix, _, rest = value.partition("-")
    numbers = prefix.replace("Litearm", "").split(".")
    try:
        version = tuple(int(v) for v in numbers[:3])
    except ValueError:
        version = ()
    return value, version


class _GripperHandle:
    """One LiteGrip, with a persistent grasp thread for ``KEEP`` semantics.

    ``LiteGrip.grasp(hold_s=0)`` blocks while it holds.  Calling it again on
    every motion would restart the squeeze; instead one thread owns the hold
    until an open command calls ``release()``.
    """

    def __init__(
        self,
        config: Any,
        calibration: ArmCalibration,
        *,
        device_name: str,
        require_enabled: bool = True,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.gripper_cal = calibration.gripper
        self.device_name = device_name
        self.require_enabled = require_enabled
        self._sdk: Any = None
        self._thread: threading.Thread | None = None
        self._result: Any = None
        self._error: BaseException | None = None
        self._command = "unknown"
        self._lock = threading.Lock()

    @property
    def command(self) -> str:
        return self._command

    def open(self) -> None:
        sdk = _import_sdk("litegrip", self.device_name)
        try:
            grip = sdk.LiteGrip(channel=self.config.channel, can_id=self.config.can_id)
            grip.connect()
            path = self.config.calibration_path
            loaded = grip.load_calibration(path=path) if path else grip.load_calibration()
        except Exception as exc:  # noqa: BLE001 - vendor exception kept as evidence
            raise DeviceError(
                f"could not connect/load calibration: {exc}", device=self.device_name, cause=exc
            ) from None
        config = getattr(grip, "config", None)
        calibrated = config is None or bool(getattr(config, "calibrated", False))
        if not loaded or not calibrated:
            grip.disconnect()
            source = str(path) if path else "the SDK factory calibration"
            raise DeviceError(
                f"no usable LiteGrip calibration from {source} for channel "
                f"{self.config.channel!r}. The shipped factory file declares "
                "channel 'can0', so a can1 gripper has no implicit calibration. "
                "Set devices.<arm>.gripper.calibration_path to a per-channel file "
                "(LiteGrip.default_calib_path(channel)) or run the vendor zero()/"
                "calibrate_guided()/calibrate_manual() first; do not enable or "
                "move an uncalibrated gripper.",
                device=self.device_name,
            )
        self._sdk = grip
        state = grip.get_state(wait=False)
        if self.require_enabled and not getattr(state, "is_enabled", False):
            raise NotReady(
                "the gripper is not enabled; the service never enables or clears a "
                "gripper fault automatically.  Enable it outside the service after "
                "checking the hardware, then restart the service.",
                device=self.device_name,
                fault_detail=f"error_code={getattr(state, 'error_code', None)}",
            )
        self._command = "open"

    def _start_grasp(self, force_n: float | None = None) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._result = None
            self._error = None
            force = float(self.config.force_n if force_n is None else force_n)

            def hold() -> None:
                try:
                    self._result = self._sdk.grasp(force_n=force, hold_s=0.0)
                except BaseException as exc:  # noqa: BLE001 - read by the caller
                    self._error = exc

            self._thread = threading.Thread(
                target=hold, name=f"litegrip-{self.device_name}", daemon=True
            )
            self._thread.start()
            self._command = "closed"

    def _join_if_finished(self) -> None:
        thread = self._thread
        if thread is not None and not thread.is_alive():
            # The grasp ended (fault, no bite, release).  Keep the evidence.
            thread.join(timeout=0)

    def state(self) -> Any:
        return self._sdk.get_state(wait=False)

    def _measured(self) -> tuple[float, float, Any]:
        state = self.state()
        width = float(getattr(state, "position_mm", 0.0)) / 1000.0
        opening = self.gripper_cal.width_to_opening(width)
        return width, opening, state

    def command_open(
        self, *, width_m: float | None, timeout_s: float, cancel: threading.Event
    ) -> GripperOutcome:
        started = now_wall_ns()
        t0 = time.monotonic()
        if self._thread is not None and self._thread.is_alive():
            try:
                self._sdk.release(
                    clearance_mm=float(self.config.hold_clearance_mm),
                    timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                raise DeviceError(
                    f"release failed: {exc}", device=self.device_name, cause=exc
                ) from None
            self._thread.join(timeout=max(0.1, timeout_s))
            if self._thread.is_alive():
                raise DeviceError(
                    "the hold thread did not finish after release(); the gripper state is "
                    "not trusted for a new open",
                    device=self.device_name,
                )
            self._thread = None
        target = (
            self.gripper_cal.open_m
            if width_m is None
            else max(self.gripper_cal.closed_m, min(width_m, self.gripper_cal.open_m))
        )
        # Use the positional ramp, not ``open()``: the SDK's ``open()`` stops at
        # its optional work-stroke (80 mm in the factory file), while robo's
        # normalised opening 1.0 means this station's measured open limit.
        try:
            result = self._sdk.move_at_speed(
                target * 1000.0, speed_mm_s=float(self.config.open_speed_mm_s)
            )
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                f"positional open failed: {exc}", device=self.device_name, cause=exc
            ) from None
        if cancel.is_set():
            raise DeviceCancelled("stop requested during gripper open", device=self.device_name)
        self._command = "open"
        width, opening, state = self._measured()
        ok = bool(getattr(result, "ok", False)) or bool(getattr(result, "reached", False))
        return GripperOutcome(
            arm=self.calibration.arm,
            command="open",
            opening=opening,
            width_m=width,
            holding=False,
            settled=ok,
            force_n=float(getattr(state, "force_n", 0.0)),
            error_code=int(getattr(state, "error_code", 0)),
            started_at_wall_ns=started,
            finished_at_wall_ns=now_wall_ns(),
            elapsed_s=time.monotonic() - t0,
            released=True,
            note=None if ok else "the open ramp did not report a settled arrival",
            raw={"result": repr(result)},
        )

    def command_close(
        self,
        *,
        force_n: float | None = None,
        timeout_s: float,
        cancel: threading.Event,
    ) -> GripperOutcome:
        started = now_wall_ns()
        t0 = time.monotonic()
        self._start_grasp(force_n)
        width0, opening0, state0 = self._measured()
        stable = 0
        last = width0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if cancel.is_set():
                self._sdk.request_release(abort=True)
                raise DeviceCancelled("stop requested during grasp", device=self.device_name)
            self._join_if_finished()
            if self._error is not None:
                raise DeviceError(
                    f"grasp failed: {self._error}",
                    device=self.device_name,
                    cause=self._error,
                )
            if (
                self._thread is not None
                and not self._thread.is_alive()
                and self._result is not None
            ):
                result = self._result
                if not bool(getattr(result, "ok", False)):
                    raise DeviceError(
                        f"grasp ended without a normal hold: {result!r}",
                        device=self.device_name,
                        fault_detail=repr(result),
                    )
                if getattr(result, "no_bite", False):
                    raise DeviceError(
                        "grasp refused the hand-over: the approach never moved from its "
                        "start and the force budget could not break static friction",
                        device=self.device_name,
                    )
            width, opening, state = self._measured()
            if abs(width - last) < 0.0005:
                stable += 1
            else:
                stable = 0
            last = width
            if stable >= 3:
                break
            time.sleep(0.02)
        width, opening, state = self._measured()
        enabled = bool(getattr(state, "is_enabled", False))
        holding = bool(
            width > self.gripper_cal.closed_m + self.gripper_cal.min_holding_m
            and width < self.gripper_cal.open_m - 0.001
        )
        settled = stable >= 3
        if not enabled:
            self._thread = None
            raise DeviceError(
                "gripper disabled itself during the grasp (communication loss or fault)",
                device=self.device_name,
                fault_detail=f"error_code={getattr(state, 'error_code', None)}",
            )
        return GripperOutcome(
            arm=self.calibration.arm,
            command="closed",
            opening=opening,
            width_m=width,
            holding=holding,
            settled=settled,
            force_n=float(getattr(state, "force_n", 0.0)),
            error_code=int(getattr(state, "error_code", 0)),
            started_at_wall_ns=started,
            finished_at_wall_ns=now_wall_ns(),
            elapsed_s=time.monotonic() - t0,
            note=(
                None
                if settled
                else "the fingers did not settle before the grasp timeout; the hold thread "
                "stays active and the opening is a candidate, not proof of a grasp"
            ),
            raw={"width_before_m": width0, "opening_before": opening0},
        )

    def abort(self) -> None:
        if self._sdk is None:
            return
        try:
            released = self._sdk.request_release(abort=True)
            if not released:
                self._sdk.stop()
        except Exception:  # noqa: BLE001 - emergency path, evidence is in the device state
            pass

    def close(self) -> None:
        if self._sdk is None:
            return
        self.abort()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._sdk.disconnect()
        except Exception:  # noqa: BLE001 - closing must not mask the original failure
            pass
        self._sdk = None
        self._thread = None


class _ArmHandle:
    def __init__(
        self,
        config: Any,
        calibration: ArmCalibration,
        *,
        device_name: str,
        transport_factory: Any = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.device_name = device_name
        self.transport_factory = transport_factory
        self._sdk: Any = None
        self._litearm: Any = None
        self.firmware = ""

    def open(self) -> None:
        sdk = _import_sdk("litearm", self.device_name)
        self._litearm = sdk
        port = self.config.port
        try:
            if self.transport_factory is not None:
                # Offline SDK test: FakeTransport requires a non-empty placeholder port.
                arm = sdk.Arm(
                    port=port or "fake",
                    transport_factory=self.transport_factory,
                )
            elif port:
                arm = sdk.Arm(port=port)
            else:
                arm = sdk.Arm()
            arm.move_timeout = float(self.config.move_timeout_s)
            arm.connect()
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                f"could not connect: {exc}", device=self.device_name, cause=exc
            ) from None
        self._sdk = arm
        self.firmware = str(getattr(arm, "firmware", ""))
        if not self.firmware:
            self.close()
            raise DeviceError("the SDK did not report a firmware version", device=self.device_name)
        _, version = _parse_firmware(self.firmware)
        if not version:
            self.close()
            raise DeviceError(
                f"cannot parse firmware version {self.firmware!r}", device=self.device_name
            )
        wanted_text, wanted = _parse_firmware(str(self.config.min_firmware))
        if wanted and version and version < wanted:
            self.close()
            raise DeviceError(
                f"firmware {self.firmware!r} is older than the required {wanted_text!r}; "
                "the SDK's motion and stop semantics are not guaranteed on it",
                device=self.device_name,
            )
        if int(getattr(arm, "n", 0)) != 7:
            self.close()
            raise DeviceError(
                f"expected a 7-joint arm, SDK reports n={getattr(arm, 'n', None)}",
                device=self.device_name,
            )
        xyz = self.calibration.flange_to_tcp[:3, 3]
        rpy = matrix_to_rpy(self.calibration.flange_to_tcp[:3, :3])
        try:
            arm.set_tcp_offset(
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                (float(rpy[0]), float(rpy[1]), float(rpy[2])),
            )
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                f"could not set the calibrated TCP offset: {exc}",
                device=self.device_name,
                cause=exc,
            ) from None

    def _state(self, *, refresh: bool = True, timeout: float | None = None) -> Any:
        msg = self._sdk.get_state(
            refresh=refresh,
            timeout=float(self.config.read_timeout_s if timeout is None else timeout),
        )
        state = getattr(msg, "value", None)
        if state is None:
            raise DeviceError("no state frame", device=self.device_name)
        return state

    def read(self) -> dict[str, Any]:
        # The firmware emits a 100 Hz passive state stream.  Read the cache
        # instead of issuing a refresh + two TCP queries on every 20 Hz monitor
        # tick; the latter would exceed the tick budget on a real serial link.
        msg = self._sdk.get_state(refresh=False, timeout=float(self.config.read_timeout_s))
        state = getattr(msg, "value", None)
        if state is None:
            raise DeviceError("no cached state frame", device=self.device_name)
        q_sdk = [float(j.q) for j in state.joints]
        q_model = [float(v) for v in self.calibration.model_from_sdk(q_sdk)]
        fault_detail = state.fault_detail if getattr(state, "faulted", False) else None
        timestamp = float(getattr(msg, "timestamp", 0.0) or 0.0)
        age = max(0.0, time.monotonic() - timestamp) if timestamp > 0 else None
        return {
            "q_sdk": q_sdk,
            "q_model": q_model,
            "dq": [float(j.dq) for j in state.joints],
            "tau": [float(j.tau) for j in state.joints],
            "enabled": bool(getattr(state, "enabled", False)),
            "mode": int(getattr(state, "mode", 0)),
            "flags": int(getattr(state, "flags", 0)),
            "fault": bool(getattr(state, "faulted", False)),
            "fault_detail": fault_detail,
            "joint_fault": int(getattr(state, "joint_fault", 0)),
            "flange_pose_raw": None,
            "tool_pose_raw": None,
            "firmware": self.firmware,
            "data_age_s": age,
        }

    def read_pose(self) -> dict[str, Any]:
        """Read the firmware flange and tool poses (diagnostics, not the 20 Hz loop)."""

        tcp = self._sdk.get_tcp(timeout=float(self.config.read_timeout_s)).value
        tool = self._sdk.get_tool_tcp(timeout=float(self.config.read_timeout_s)).value
        return {
            "flange_pose_raw": [float(v) for v in tcp] if tcp is not None else None,
            "tool_pose_raw": [float(v) for v in tool] if tool is not None else None,
        }

    def verify_enabled(self) -> list[str]:
        state = self._state(refresh=True)
        problems = []
        if not bool(getattr(state, "enabled", False)):
            problems.append(f"{self.device_name} is not enabled")
        if bool(getattr(state, "faulted", False)):
            problems.append(f"{self.device_name} fault: {state.fault_detail}")
        return problems

    def move_sync(
        self,
        q_sdk: list[float],
        speed: float,
        *,
        cancel: threading.Event,
        timeout_s: float,
    ) -> MotionOutcome:
        if cancel.is_set():
            raise DeviceCancelled(
                "stop requested before the segment was sent", device=self.device_name
            )
        if not 0.0 < float(speed) <= 1.0:
            raise DeviceError(
                f"speed fraction must be in (0, 1], got {speed!r}", device=self.device_name
            )
        target_model = self.calibration.model_from_sdk(q_sdk)
        limits = self.calibration.joint_limits_rad
        for index, (value, (lo, hi)) in enumerate(zip(target_model, limits, strict=True)):
            if not (lo - 1e-9 <= float(value) <= hi + 1e-9):
                raise DeviceError(
                    f"target joint {index} ({float(value):.4f} rad) is outside the calibrated "
                    f"limit [{lo:.4f}, {hi:.4f}]",
                    device=self.device_name,
                )
        state = self._state(refresh=True)
        if not bool(getattr(state, "enabled", False)):
            raise NotReady("arm is not enabled", device=self.device_name)
        if bool(getattr(state, "faulted", False)):
            raise NotReady(
                f"arm fault: {state.fault_detail}",
                device=self.device_name,
                fault_detail=state.fault_detail,
            )
        started = now_wall_ns()
        t0 = time.monotonic()
        try:
            self._sdk.movej_sync(q_sdk, speed=float(speed))
        except Exception as exc:  # noqa: BLE001
            # Read the state for evidence; do not hide the SDK exception.
            try:
                after = self._state(refresh=True, timeout=1.0)
                detail = after.fault_detail
            except Exception:  # noqa: BLE001
                detail = None
            raise DeviceError(
                f"synchronous segment failed: {exc}",
                device=self.device_name,
                fault_detail=detail,
                cause=exc,
            ) from None
        after = self._state(refresh=True)
        q_actual = [float(j.q) for j in after.joints]
        q_model = self.calibration.model_from_sdk(q_actual)
        target_model = self.calibration.model_from_sdk(q_sdk)
        err = max(abs(float(a) - float(b)) for a, b in zip(q_model, target_model, strict=True))
        if time.monotonic() - t0 > timeout_s + 0.5:
            raise DeviceError(
                f"segment took {time.monotonic() - t0:.2f}s, over {timeout_s:.2f}s",
                device=self.device_name,
            )
        return MotionOutcome(
            arm=self.calibration.arm,
            q_model=tuple(float(v) for v in q_model),
            q_sdk=tuple(float(v) for v in q_actual),
            q_target_model=tuple(float(v) for v in target_model),
            reached=err <= 0.02,
            max_joint_error_rad=err,
            started_at_wall_ns=started,
            finished_at_wall_ns=now_wall_ns(),
            elapsed_s=time.monotonic() - t0,
            fault=bool(getattr(after, "faulted", False)),
            fault_detail=after.fault_detail if getattr(after, "faulted", False) else None,
            raw={
                "enabled": bool(getattr(after, "enabled", False)),
                "mode": int(getattr(after, "mode", 0)),
                "flags": int(getattr(after, "flags", 0)),
            },
        )

    def emergency_stop(self) -> tuple[bool, str | None, str | None]:
        try:
            self._sdk.emergency_stop()
        except Exception as exc:  # noqa: BLE001
            return False, "emergency_stop", f"emergency_stop raised {type(exc).__name__}: {exc}"
        try:
            state = self._state(refresh=True, timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            return False, "emergency_stop", f"stop acknowledged but no state could be read: {exc}"
        max_dq = max((abs(float(j.dq)) for j in state.joints), default=math.inf)
        stopped = (
            max_dq <= 0.05
            or bool(getattr(state, "faulted", False))
            or not bool(getattr(state, "enabled", False))
        )
        detail = (
            f"max |dq|={max_dq:.3f} rad/s, enabled={bool(getattr(state, 'enabled', False))}, "
            f"fault={bool(getattr(state, 'faulted', False))}"
        )
        return stopped, "emergency_stop", None if stopped else detail

    def close(self) -> None:
        if self._sdk is not None:
            try:
                self._sdk.close()
            except Exception:  # noqa: BLE001
                pass
        self._sdk = None


class LiteArmStation(StationBase):
    """The active LiteArm arms and LiteGrips, owned by one process.

    ``execution.active_arms`` decides which arms exist for this service: only
    those are opened, read, verified, commanded and stopped.  Inactive arms are
    never opened, so their devices entry (and their wrist camera) is optional.
    """

    fake = False

    def __init__(
        self,
        config: StationConfig,
        calibration: Calibration,
        *,
        arm_transport_factory: Any = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.active_arms = tuple(config.execution.active_arms)
        self._active = frozenset(self.active_arms)
        self.kinematics = LitearmKinematics(calibration)
        self.arm_transport_factory = arm_transport_factory
        self.arms: dict[str, _ArmHandle] = {}
        self.grippers: dict[str, _GripperHandle] = {}
        self._stop_lock = threading.Lock()
        self._stop_result: StopResult | None = None

    def _require_active(self, arm: str, *, device: str | None = None) -> None:
        if arm not in self._active:
            raise DeviceError(
                f"arm {arm!r} is not active on this station; active arms: {list(self.active_arms)}",
                device=device or f"arm_{arm}",
            )

    def open(self) -> None:
        if len(self.active_arms) > 1 and all(
            not self.config.arms[arm].arm.port for arm in self.active_arms
        ):
            raise DeviceError(
                "multiple arms are configured but no serial port is named; automatic CDC "
                "discovery cannot tell the arms apart",
                device="station",
            )
        try:
            for arm in self.active_arms:
                handle = _ArmHandle(
                    self.config.arms[arm].arm,
                    self.calibration.arm(arm),
                    device_name=f"arm_{arm}",
                    transport_factory=self.arm_transport_factory,
                )
                handle.open()
                self.arms[arm] = handle
            for arm in self.active_arms:
                self._verify_firmware_pose(arm)
            for arm in self.active_arms:
                grip_config = self.config.arms[arm].gripper
                if grip_config is None:
                    # Arm-only station: no gripper exists to open, read or stop.
                    continue
                grip = _GripperHandle(
                    grip_config,
                    self.calibration.arm(arm),
                    device_name=f"gripper_{arm}",
                    require_enabled=self.config.execution.mode is not ExecutionMode.READ_ONLY,
                )
                grip.open()
                self.grippers[arm] = grip
        except BaseException:
            self.close()
            raise

    def _verify_firmware_pose(self, arm: str) -> None:
        handle = self.arms[arm]
        reading = handle.read()
        poses = handle.read_pose()
        flange_raw = poses.get("flange_pose_raw")
        tool_raw = poses.get("tool_pose_raw")
        if flange_raw is None:
            raise DeviceError("firmware returned no flange pose", device=f"arm_{arm}")
        base = np.asarray(self.calibration.arm(arm).base_pose, dtype=float)
        q_model = reading["q_model"]
        expected_flange = self.kinematics.flange_pose(arm, q_model)
        expected_world = flange_raw[:3]
        actual_world = base[:3, :3] @ np.asarray(expected_world) + base[:3, 3]
        delta = float(np.linalg.norm(actual_world - expected_flange[:3, 3]))
        if delta > 0.02:
            raise DeviceError(
                f"firmware flange pose differs from the calibrated FK by {delta:.4f} m; "
                "check the base transform and joint map",
                device=f"arm_{arm}",
            )
        if tool_raw is None:
            raise DeviceError("firmware returned no tool pose", device=f"arm_{arm}")
        tool_local = np.linalg.inv(base) @ self.kinematics.fk(arm, q_model)
        tool_delta = float(np.linalg.norm(np.asarray(tool_raw[:3]) - tool_local[:3, 3]))
        if tool_delta > 0.02:
            raise DeviceError(
                f"firmware tool pose differs from calibration by {tool_delta:.4f} m; "
                "check flange_to_tcp",
                device=f"arm_{arm}",
            )

    def capabilities(self) -> dict[str, Any]:
        return {
            "fake": False,
            "sdk": "litearm-python + litegrip-python",
            "active_arms": list(self.active_arms),
            "arms": {
                arm: {
                    "firmware": self.arms[arm].firmware,
                    "joints": 7,
                    "port": self.config.arms[arm].arm.port,
                }
                for arm in self.arms
            },
            "grippers": {
                arm: {
                    "channel": self.config.arms[arm].gripper.channel,
                    "can_id": self.config.arms[arm].gripper.can_id,
                    "force_n": self.config.arms[arm].gripper.force_n,
                }
                for arm in self.grippers
            },
        }

    def read(self) -> dict[str, Any]:
        arms = {arm: handle.read() for arm, handle in self.arms.items()}
        grippers = {}
        for arm, handle in self.grippers.items():
            state = handle.state()
            width = float(getattr(state, "position_mm", 0.0)) / 1000.0
            grippers[arm] = {
                "width_m": width,
                "opening": self.calibration.arm(arm).gripper.width_to_opening(width),
                "force_n": float(getattr(state, "force_n", 0.0)),
                "enabled": bool(getattr(state, "is_enabled", False)),
                "error_code": int(getattr(state, "error_code", 0)),
                "fresh": bool(getattr(state, "fresh", False)),
                "data_age_s": getattr(state, "data_age_s", None),
                "command": handle.command,
                "force_calibrated": False,
            }
        return {
            "arms": arms,
            "grippers": grippers,
            "captured_at_wall_ns": now_wall_ns(),
            "captured_at_monotonic_ns": now_monotonic_ns(),
        }

    def verify_ready(self) -> list[str]:
        problems: list[str] = []
        for arm, handle in self.arms.items():
            problems.extend(handle.verify_enabled())
            grip = self.grippers.get(arm)
            if grip is None:
                continue
            state = grip.state()
            if not bool(getattr(state, "is_enabled", False)):
                problems.append(
                    f"gripper_{arm} is not enabled "
                    f"(error_code={getattr(state, 'error_code', None)})"
                )
            if getattr(state, "is_error", False):
                problems.append(
                    f"gripper_{arm} fault (error_code={getattr(state, 'error_code', None)})"
                )
        return problems

    def move_joints(
        self,
        arm: str,
        q_sdk: list[float],
        speed: float,
        *,
        cancel: threading.Event,
        timeout_s: float,
    ) -> MotionOutcome:
        self._require_active(arm, device=f"arm_{arm}")
        return self.arms[arm].move_sync(q_sdk, speed, cancel=cancel, timeout_s=timeout_s)

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
        self._require_active(arm, device=f"gripper_{arm}")
        handle = self.grippers.get(arm)
        if handle is None:
            raise DeviceError(
                f"arm {arm!r} has no gripper configured on this station",
                device=f"gripper_{arm}",
            )
        if command == "open":
            return handle.command_open(width_m=width_m, timeout_s=timeout_s, cancel=cancel)
        if command == "close":
            # The SDK's force_n is a nominal 0.1 Nm/N scaling, not a calibrated
            # force; the station config is the default and the payload may lower it.
            if force_n is not None and force_n > handle.config.force_n + 1e-9:
                raise DeviceError(
                    f"requested force {force_n:.1f} N exceeds the station limit "
                    f"{handle.config.force_n:.1f} N",
                    device=f"gripper_{arm}",
                )
            return handle.command_close(force_n=force_n, timeout_s=timeout_s, cancel=cancel)
        if command == "keep":
            width, opening, state = handle._measured()
            return GripperOutcome(
                arm=arm,
                command="keep",
                opening=opening,
                width_m=width,
                holding=width
                > self.calibration.arm(arm).gripper.closed_m
                + self.calibration.arm(arm).gripper.min_holding_m,
                settled=True,
                force_n=float(getattr(state, "force_n", 0.0)),
                error_code=int(getattr(state, "error_code", 0)),
                started_at_wall_ns=now_wall_ns(),
                finished_at_wall_ns=now_wall_ns(),
                elapsed_s=0.0,
            )
        raise DeviceError(f"unknown gripper command {command!r}", device=f"gripper_{arm}")

    def request_gripper_abort(self) -> None:
        for handle in self.grippers.values():
            handle.abort()

    def _stop_arms(self) -> tuple[bool, str | None]:
        confirmed = True
        detail: list[str] = []
        for arm, handle in self.arms.items():
            ok, method, why = handle.emergency_stop()
            confirmed &= ok
            if not ok:
                detail.append(f"{arm}: {method}: {why}")
        return confirmed, ("; ".join(detail) if detail else None)

    def stop(self, kind: StopKind, arms: tuple[str, ...] | None = None) -> StopResult:
        # Only opened (active) arms exist in ``self.arms``; ``arms`` is accepted
        # for the shared interface but the active set is authoritative here.
        _ = arms
        # A normal cancel is handled by the executor between synchronous
        # segments; calling emergency_stop there would be a heavier action than
        # requested.  Timeout and emergency both call the firmware emergency stop.
        with self._stop_lock:
            if kind is StopKind.CANCEL:
                # No segment is interrupted here (the executor checks the cancel
                # flag between synchronous segments).  Confirm with a fresh state
                # read instead of treating the flag as evidence.
                try:
                    reading = self.read()
                    ages = [a.get("data_age_s") for a in reading["arms"].values()]
                    fresh = all(age is not None and float(age) <= 0.5 for age in ages)
                    max_dq = max(
                        (abs(float(v)) for a in reading["arms"].values() for v in a["dq"]),
                        default=math.inf,
                    )
                    confirmed = fresh and max_dq <= float(
                        self.config.safety.joint_speed_confirm_rad_s
                    )
                    why = (
                        None
                        if confirmed
                        else (f"max |dq|={max_dq:.3f} rad/s after the boundary; fresh={fresh}")
                    )
                except Exception as exc:  # noqa: BLE001
                    confirmed = False
                    why = f"no state readback: {exc}"
                result = StopResult(
                    requested=True,
                    kind=kind.value,
                    confirmed=confirmed,
                    state="stopped_between_segments" if confirmed else "stop_unconfirmed",
                    method="cooperative_segment_boundary+state_readback",
                    message=why
                    or "no in-flight segment was interrupted; the arm holds its measured pose",
                    requested_at_wall_ns=now_wall_ns(),
                    confirmed_at_wall_ns=now_wall_ns(),
                )
                self._stop_result = result
                return result
            self.request_gripper_abort()
            confirmed, why = self._stop_arms()
            state = "stopped" if confirmed else "stop_unconfirmed"
            result = StopResult(
                requested=True,
                kind=kind.value,
                confirmed=confirmed,
                state=state,
                method="firmware_emergency_stop",
                message=why,
                requested_at_wall_ns=now_wall_ns(),
                confirmed_at_wall_ns=now_wall_ns(),
            )
            self._stop_result = result
            return result

    def close(self) -> None:
        for grip in getattr(self, "grippers", {}).values():
            grip.close()
        for arm in getattr(self, "arms", {}).values():
            arm.close()


def make_station(config: StationConfig) -> StationBase:
    """Create the station selected by the execution mode.

    ``offline-fake`` is an explicit mode and instantiates the synthetic station;
    it is never selected because a real driver failed.  ``read-only`` and
    ``live`` always construct the real SDK wrapper.
    """

    if config.execution.mode.value == "offline-fake":
        from .fake import FakeStation

        return FakeStation(config, config.calibration)
    if config.execution.mode.value == "live":
        config.calibration.require_live_execution(
            arms=config.execution.active_arms, cameras=tuple(config.cameras)
        )
    return LiteArmStation(config, config.calibration)


__all__ = [
    "DeviceCancelled",
    "DeviceError",
    "GripperOutcome",
    "LiteArmStation",
    "MotionOutcome",
    "NotReady",
    "StationBase",
    "make_station",
]
