# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The hardware toolkit and backend the PyRUA-Lean robot talks to.

This replaces RPent's toolkit only for the real station.  It records every
snapshot, serves the same envelope shape, and keeps three counters separate:

* ``State.step`` - the observation snapshot number;
* ``steps_used`` - trajectory segments / device actions executed (reported per
  primitive and by ``env_steps``);
* episode budgets - tool calls and wall-clock time, owned by ``ArmToolkit`` in
  :mod:`pyrualean.arms`.

There is no independent task-success evaluator; ``solved()`` is always False
and the result metadata says ``environment_success_known = false``.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import StationConfig
from .kinematics import LitearmKinematics
from .motion import grip as motion_grip
from .motion import home as motion_home
from .motion import move_pose as motion_move_pose
from .motion import place_on as motion_place_on
from .perception import FrameData, Perception, frame_from_observation
from .planning import Planner
from .recording import Recorder
from .rpc import HardwareClient, RpcError
from .state import Observation, build_observation


class HardwareToolkit:
    def __init__(
        self,
        config: StationConfig,
        client: HardwareClient,
        *,
        task: str,
        instruction: str,
        out_dir: str | Path,
        perception: Perception | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        self.config = config
        self.calibration = config.calibration
        self.client = client
        self.task = task
        self.instruction = instruction
        self.kinematics = LitearmKinematics(self.calibration)
        self.planner = Planner(self.calibration, config.execution, config.safety, self.kinematics)
        self.perception = perception
        self.recorder = recorder or Recorder(out_dir, config)
        self.held_openings: dict[str, float] = {}
        self.device_steps = 0
        self.max_device_actions: int | None = None
        self._current: Observation | None = None
        self._artifacts: dict[str, bytes] = {}
        self._lock = threading.RLock()
        self._closed = False
        self.stop_result: dict[str, Any] | None = None
        self.stop_error: str | None = None
        self.finish: dict[str, Any] | None = None

    # -- observation --------------------------------------------------------

    def refresh_observation(self, *, tool: str | None = None) -> Observation:
        with self._lock:
            snapshot = self.client.snapshot()
            observation = build_observation(
                self.calibration,
                self.kinematics,
                snapshot.payload,
                task=self.task,
                instruction=self.instruction,
                execution_mode=self.config.execution.mode.value,
                max_age_s=self.config.snapshot.max_age_s,
                now_wall_ns=self.client.server_wall_equivalent(),
                active_arms=self.config.execution.active_arms,
            )
            self.recorder.record(
                snapshot.payload,
                snapshot.artifacts,
                observation=observation.envelope(),
                tool=tool,
            )
            self._current = observation
            self._artifacts = dict(snapshot.artifacts)
            return observation

    def current_observation(self, *, refresh: bool = False) -> Observation:
        with self._lock:
            current = self._current
            if current is None:
                return self.refresh_observation(tool="snapshot")
            if refresh:
                age = self.client.age_s(current.captured_at_wall_ns)
                if age > self.config.snapshot.refresh_min_interval_s:
                    return self.refresh_observation(tool="snapshot")
            return current

    def observation(self, step: int = -1) -> Observation:
        if step == -1:
            return self.current_observation(refresh=False)
        payload = self.recorder.payload(step)
        # A historical observation is frozen evidence: age is evaluated at its
        # own capture time, so a later arm pose or wall clock cannot rewrite it.
        return build_observation(
            self.calibration,
            self.kinematics,
            payload,
            task=self.task,
            instruction=self.instruction,
            execution_mode=self.config.execution.mode.value,
            max_age_s=None,
            now_wall_ns=int(payload.get("captured_at_wall_ns") or 0),
            active_arms=self.config.execution.active_arms,
        )

    def frames(self, observation: Observation) -> dict[str, FrameData]:
        artifacts: dict[str, bytes] = {}
        for _name, meta in observation.camera_state.items():
            for key in ("color_artifact", "depth_artifact"):
                artifact = meta.get(key)
                if artifact and artifact not in artifacts:
                    if observation.step == (self._current.step if self._current else None):
                        if artifact in self._artifacts:
                            artifacts[artifact] = self._artifacts[artifact]
                            continue
                    artifacts[artifact] = self.recorder.artifact(artifact, observation.step)
        return {
            name: frame_from_observation(observation, artifacts, name)
            for name in observation.camera_state
        }

    # -- tool dispatch ------------------------------------------------------

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(input_dict or {})
        try:
            if name == "view_env_state":
                return self._view_env_state(kwargs)
            if name == "view_camera_meta":
                return self._view_camera_meta(kwargs)
            if name == "locate":
                return self._locate(kwargs)
            if name == "segment":
                return self._segment(kwargs)
            if name == "back_project":
                return self._back_project(kwargs)
            if name == "move_pose":
                result = motion_move_pose(
                    self,
                    kwargs.get("xyz"),
                    arm=str(kwargs.get("arm", "right")),
                    gripper=kwargs.get("gripper", 0),
                    pitch=kwargs.get("pitch"),
                    yaw=kwargs.get("yaw"),
                    roll=kwargs.get("roll"),
                    path=str(kwargs.get("path", "auto")),
                )
            elif name == "grip":
                result = motion_grip(
                    self,
                    arm=str(kwargs.get("arm", "right")),
                    gripper=kwargs.get("gripper", 1),
                    opening=kwargs.get("opening"),
                )
            elif name == "home":
                result = motion_home(self, arm=str(kwargs.get("arm", "both")))
            elif name == "place_on":
                result = motion_place_on(
                    self,
                    kwargs.get("x"),
                    kwargs.get("y"),
                    kwargs.get("support_top"),
                    kwargs.get("tcp_to_bottom"),
                    arm=str(kwargs.get("arm", "right")),
                    clearance=kwargs.get("clearance", 0.002),
                    lift_m=kwargs.get("lift_m", 0.12),
                    retreat_xy=tuple(kwargs.get("retreat_xy", (0.0, -0.10))),
                    opening=kwargs.get("opening"),
                    path=str(kwargs.get("path", "line")),
                )
            else:
                return {"error": f"this hardware host has no tool {name!r}"}
            observation = self.refresh_observation(tool=name)
            return self._envelope(observation, name, result)
        except RpcError as exc:
            return {"error": f"{exc.code}: {exc.message}", "detail": exc.detail}
        except Exception as exc:  # noqa: BLE001 - the policy sees the failure
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _envelope(
        self, observation: Observation, tool: str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        envelope = observation.envelope()
        envelope["log"] = {"tool": tool, "result": dict(result)}
        return envelope

    def _view_env_state(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        step = int(kwargs.get("step", -1))
        if step == -1:
            observation = self.refresh_observation(tool="view_env_state")
        else:
            observation = self.observation(step)
        envelope = observation.envelope()
        envelope["log"] = {"tool": "view_env_state", "result": {}}
        return envelope

    def _view_camera_meta(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        camera = str(kwargs.get("camera", "cam_head"))
        step = int(kwargs.get("step", -1))
        observation = (
            self.current_observation(refresh=False) if step == -1 else self.observation(step)
        )
        if camera not in observation.camera_state:
            return {"error": f"camera {camera!r} is not in this snapshot"}
        meta = dict(observation.camera_state[camera])
        meta["step"] = observation.step
        meta["measured"] = True
        return meta

    def _segment(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.perception is None:
            return {"error": "SAM3 perception is not configured (host booted env_only)"}
        observation = self.current_observation(refresh=True)
        frames = self.frames(observation)
        return self.perception.segment(
            frames,
            str(kwargs.get("text_prompt", "")),
            str(kwargs.get("camera", "cam_head")),
            min_score=kwargs.get("min_score"),
        )

    def _locate(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.perception is None:
            return {"error": "SAM3 perception is not configured (host booted env_only)"}
        observation = self.current_observation(refresh=True)
        frames = self.frames(observation)
        return self.perception.locate(
            frames,
            str(kwargs.get("text_prompt", "")),
            str(kwargs.get("camera", "cam_head")),
            min_score=kwargs.get("min_score"),
        )

    def _back_project(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.perception is None:
            return {"error": "SAM3 perception is not configured (host booted env_only)"}
        observation = self.current_observation(refresh=True)
        frames = self.frames(observation)
        return self.perception.back_project(
            frames,
            int(kwargs["row"]),
            int(kwargs["col"]),
            str(kwargs.get("camera", "cam_head")),
        )

    # -- command submission ------------------------------------------------

    def _check_action_budget(self, additional: int) -> None:
        if self.max_device_actions is None:
            return
        if self.device_steps + additional > self.max_device_actions:
            raise RuntimeError(
                f"device-action budget exhausted: {self.device_steps} used, "
                f"{additional} requested, limit {self.max_device_actions}"
            )

    def submit_trajectory(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        self._check_action_budget(len(payload.get("waypoints") or []))
        result = self.client.execute_command("trajectory", payload, timeout_s=timeout_s)
        self.device_steps += int(result.get("steps_used", 0))
        return result

    def submit_grip(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        self._check_action_budget(1)
        result = self.client.execute_command("grip", payload, timeout_s=timeout_s)
        self.device_steps += int(result.get("steps_used", 0))
        return result

    # -- backend contract --------------------------------------------------

    def env_steps(self) -> int:
        return self.device_steps

    def solved(self) -> bool:
        return False

    def cancel_active_and_wait(self) -> None:
        with self._lock:
            try:
                self.client.send_cancel()
            except Exception as exc:  # noqa: BLE001 - recorded; stop evidence is what matters
                self.stop_error = f"cancel request failed: {exc}"

    def conclude(self) -> None:
        """No independent success evaluator exists; finishing is not a verdict."""

        return None

    def artifact(self, name: str, step: int = -1) -> bytes:
        return self.recorder.artifact(name, step)

    def close(self, *, finish: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.finish = dict(finish) if finish else self.finish
            try:
                result = self.client.close_session()
                self.stop_result = result.get("stop")
            except Exception as exc:  # noqa: BLE001 - the absence of evidence is recorded
                self.stop_error = f"{type(exc).__name__}: {exc}"
                self.stop_result = None
            finally:
                self.client.close(stop=False)
                self.recorder.finalize(
                    stop=self.stop_result,
                    finish=self.finish,
                    execution={
                        "device_actions": self.device_steps,
                        "snapshot_step": self._current.step if self._current else None,
                        "execution_mode": self.config.execution.mode.value,
                    },
                    environment_success_known=False,
                )

    def stop_metadata(self) -> dict[str, Any]:
        stop = self.stop_result
        return {
            "requested": bool(stop and stop.get("requested")),
            "confirmed": bool(stop and stop.get("confirmed")),
            "method": (stop or {}).get("method"),
            "state": (stop or {}).get("state", "not_recorded"),
            "message": (stop or {}).get("message") or self.stop_error,
            "recorded_after_shutdown": self._closed,
            "environment_success_known": False,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "execution_mode": self.config.execution.mode.value,
            "config_sha256": self.config.sha256,
            "calibration_sha256": self.config.calibration_sha256,
            "calibration_version": self.calibration.version,
            "station_id": self.config.station_id,
            "devices": self.client.hello.get("capabilities") if self.client.hello else None,
            "success_known": False,
            "task_success": "not_measured",
            "step_semantics": {
                "state_step": "observation snapshot number",
                "steps_used": "trajectory segments or device actions",
                "env_steps": "same as steps_used for this backend",
            },
        }

    def camera_meta(self, camera: str = "cam_head", *, step: int = -1) -> dict[str, Any]:
        observation = (
            self.current_observation(refresh=False) if step == -1 else self.observation(step)
        )
        if camera not in observation.camera_state:
            raise ValueError(f"camera must be one of {sorted(observation.camera_state)}")
        return self._view_camera_meta({"camera": camera, "step": observation.step})


class HardwareBackend:
    """The host's backend: a hardware toolkit behind the ordinary Backend shape."""

    def __init__(self, toolkit: HardwareToolkit) -> None:
        self._toolkit = toolkit

    @property
    def toolkit(self) -> HardwareToolkit:
        return self._toolkit

    @property
    def environment_success_known(self) -> bool:
        return False

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self._toolkit.execute_tool(name, input_dict)

    def solved(self) -> bool:
        return False

    def artifact(self, name: str, step: int = -1) -> bytes:
        return self._toolkit.artifact(name, step)

    def env_steps(self) -> int | None:
        return self._toolkit.env_steps()

    def cancel(self) -> None:
        self._toolkit.cancel_active_and_wait()

    def conclude(self) -> None:
        # There is no success evaluator to call; finishing the program does not
        # establish task success.
        self._toolkit.conclude()


__all__ = ["HardwareBackend", "HardwareToolkit"]
