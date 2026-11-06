# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The real LiteArm / LiteGrip station, behind the same ``robo`` API.

The robot class inherits :class:`pyrualean.litearm.LitearmRobot`'s policy
surface and replaces its observation and result semantics where the simulator
is not true:

* ``State.step`` is an observation snapshot number, not a simulator action;
* ``steps_used`` counts trajectory segments/device actions;
* the table height and arm bases come from the calibration, not the
  RoboDojo scene;
* the gripper width comes from the measured LiteGrip position and the
  per-unit closed/open calibration, not a fixed 2 mm + 85 mm formula;
* there is no task-success evaluator.  ``robo.done`` raises, every result says
  ``measured=True``, and the host records ``environment_success_known=false``.

The class does not import a vendor SDK or open a device.  It talks to a
:class:`pyrualean.hardware.toolkit.HardwareBackend` that the host creates only
after the hardware service handshake succeeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from .litearm import LitearmRobot
from .robodojo import _arm, _camera, _command, _floats, _sequence, _tuple3
from .robots import RobotAdapter

#: The real station's cameras (calibration names).
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

#: Arm names, in the canonical order used by the configuration.
ARMS = ("left", "right")


@dataclass(frozen=True)
class RealArmState:
    """One measured arm/gripper in a hardware observation.

    tcp_pos: measured grasp point [x, y, z], metres, in the calibrated world
        frame (origin at the centre of the near table edge, +x right, +y into
        the table, +z up, table z = 0).
    pitch / yaw / roll: orientation of the measured TCP, degrees; the existing
        ``robo`` convention is preserved (pitch 90 = pointing down).
    eef_pos / eef_quat: the same TCP pose (the station defines the TCP as the
        grasp point; the flange pose is available in the observation under
        ``flange_pos`` but is not the policy-facing end effector).
    gripper: measured normalised opening (1.0 = measured open limit,
        0.0 = measured closed limit); ``None`` when this arm has no gripper.
    width_m: measured opening across the finger pads, metres; ``None`` when
        this arm has no gripper.
    joints: seven measured model-side joint angles, radians.
    data_age_s: age of the encoder state when the snapshot was made.
    fault / fault_detail: firmware fault evidence, not a command target.
    """

    tcp_pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    gripper: float | None
    joints: tuple[float, ...]
    width_m: float | None
    data_age_s: float | None
    fault: bool
    fault_detail: str | None
    measured: bool = True


@dataclass(frozen=True)
class RealState:
    """A measured snapshot of the station's active arms.

    ``left`` and ``right`` are ``None`` when that arm is not part of the
    station (``execution.active_arms``); :meth:`active_arms` names the arms
    that are present.  Asking for an inactive arm raises instead of returning
    a fabricated zero pose.

    step: observation snapshot number; a new snapshot is recorded after every
        state-changing primitive and when the current state is refreshed.
    task / instruction: the natural-language instruction supplied at boot.
    truncated: always False; the hardware path has a wall-clock/call budget,
        not a simulator action budget.  The service state and faults are in
        the host metadata.
    success_known: False.  There is no independent task-success evaluator.
    execution_state: the service's current state (ready, executing, stopped,
        fault, ...).
    fault / fault_detail: service-level fault evidence, if any.
    """

    step: int
    task: str
    left: RealArmState | None
    right: RealArmState | None
    truncated: bool
    execution_state: str
    success_known: bool = False
    fault: bool = False
    fault_detail: str | None = None

    @property
    def active_arms(self) -> tuple[str, ...]:
        """Arms present in this snapshot, in canonical order."""

        return tuple(name for name in ARMS if getattr(self, name) is not None)

    def arm(self, name: str) -> RealArmState:
        if name not in ARMS:
            raise ValueError("arm must be 'left' or 'right'")
        value = self.left if name == "left" else self.right
        if value is None:
            raise ValueError(
                f"arm {name!r} is not active on this station; active arms: {list(self.active_arms)}"
            )
        return value


@dataclass(frozen=True)
class RealMove:
    """Outcome of ``move_to``, from measured state and executed segments."""

    arm: str
    target_xyz: tuple[float, float, float]
    start_xyz: tuple[float, float, float]
    final_xyz: tuple[float, float, float]
    dist_to_target_m: float
    orientation_error_deg: float
    pitch: float
    yaw: float
    roll: float
    reached: bool
    path: str | None
    ik_error: str | None
    note: str | None
    blocked: bool
    steps_used: int
    truncated: bool
    slip: bool = False
    slip_opening: float | None = None
    slip_baseline: float | None = None
    measured: bool = True


@dataclass(frozen=True)
class RealGripper:
    """Outcome of ``set_gripper``; ``holding`` is a candidate, not proof of a grasp."""

    arm: str
    commanded: str
    opening: float
    holding: bool
    width_m: float | None
    steps_used: int
    step: int | None
    step_limit: int | None
    truncated: bool
    settled: bool = False
    measured: bool = True
    note: str | None = None


@dataclass(frozen=True)
class RealHome:
    """Outcome of ``home``; the configured safe pose is not the SDK zero/home pose."""

    arm: str
    reached: bool
    max_joint_error_rad: float | None
    steps_used: int
    truncated: bool
    note: str | None = None
    measured: bool = True


@dataclass(frozen=True)
class RealPlace:
    """Outcome of ``place_on``: descent, release, lift and retreat are separate evidence."""

    arm: str
    placed: bool
    released: bool
    safe_to_home: bool
    target_xyz: tuple[float, float, float]
    place_z: float
    descent_reached: bool
    descent_blocked: bool
    descent_slip: bool
    open_opening: float | None
    lift_reached: bool
    lift_blocked: bool
    retreat_reached: bool
    retreat_blocked: bool
    steps_used: int
    truncated: bool
    note: str | None
    measured: bool = True


@dataclass(frozen=True)
class RealSegment:
    """Outcome of ``segment``: SAM3 mask statistics in the current snapshot."""

    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    score: float | None
    box: Any
    camera: str
    measured: bool = True


@dataclass(frozen=True)
class RealObjectSight:
    """Outcome of ``locate`` from measured depth or calibrated multi-view RGB.

    measurement_failed: True when no reliable 3-D extent could be measured
        (missing depth and fewer than two calibrated views, too few points,
        invalid calibration).  ``center_xyz`` and the extents are None then;
        no table intersection or fixed height is substituted.
    source: ``"depth"`` or ``"multiview"``.
    views: number of calibrated views that produced a mask and took part.
    measurement_quality: point/voxel counts, valid-depth ratio, etc.
    """

    found: bool
    center_xyz: tuple[float, float, float] | None
    top_z: float | None
    height_m: float | None
    length_m: float | None
    width_m: float | None
    long_axis_deg: float | None
    grasp_yaw: float | None
    score: float | None
    centroid: tuple[int, int] | None
    area_px: int
    camera: str
    note: str | None
    views: int = 1
    source: str | None = None
    measurement_failed: bool = False
    measurement_quality: dict[str, Any] | None = None


@dataclass(frozen=True)
class RealSeenPoint:
    """Outcome of ``back_project``: measured surface point at a pixel."""

    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    depth_m: float
    camera: str
    surface: str
    measured: bool = True


def _arm_present(raw: dict[str, Any], name: str) -> bool:
    """Whether a measured arm appears in the observation state dict."""

    return f"{name}_arm_joint_state" in raw or f"{name}_ee_pose" in raw


def _real_arm(raw: dict[str, Any], name: str) -> RealArmState:
    tcp = _sequence(raw.get(f"{name}_tcp_pos", raw.get(f"{name}_ee_pose")))
    if len(tcp) < 3:
        pose = _sequence(raw.get(f"{name}_ee_pose"))
        tcp = pose if len(pose) >= 7 else (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    quat = _sequence(raw.get(f"{name}_tcp_quat_wxyz"))
    if len(quat) != 4:
        quat = tcp[3:7] if len(tcp) >= 7 else (1.0, 0.0, 0.0, 0.0)
    pitch, yaw, roll = _sequence(raw.get(f"{name}_pitch_yaw_roll"))
    joints = _sequence(raw.get(f"{name}_arm_joint_state"))
    grip = _sequence(raw.get(f"{name}_ee_joint_state"))
    width = raw.get(f"{name}_ee_width_m")
    return RealArmState(
        tcp_pos=(float(tcp[0]), float(tcp[1]), float(tcp[2])),
        pitch=float(pitch),
        yaw=float(yaw),
        roll=float(roll),
        eef_pos=(float(tcp[0]), float(tcp[1]), float(tcp[2])),
        eef_quat=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        gripper=float(grip[0]) if grip else None,
        joints=tuple(float(v) for v in joints),
        width_m=float(width) if width is not None else None,
        data_age_s=(
            float(raw[f"{name}_data_age_s"]) if raw.get(f"{name}_data_age_s") is not None else None
        ),
        fault=bool(raw.get(f"{name}_fault")),
        fault_detail=str(raw.get(f"{name}_fault_detail") or "") or None,
    )


def _angle_value(angles: dict[str, Any], name: str) -> float:
    try:
        return float(angles.get(name, math.nan))
    except (TypeError, ValueError):
        return math.nan


class LitearmRealRobot(LitearmRobot):
    """Calibrated LiteArm arms with LiteGrip grippers at a real station.

    World frame: origin at the centre of the table edge nearest the robots,
    +x right, +y into the table, +z up, table surface z = 0.  Metres and
    degrees.  The arms are ``"left"`` and ``"right"``; orientations and the
    ``pitch`` / ``yaw`` / ``roll`` convention are the same as the LiteArm
    adapter: pitch 90 points straight down, yaw is the heading on the table,
    roll turns the fingers about the pointing direction.

    The station config's ``execution.active_arms`` decides which arms exist.
    With one arm, state carries that arm only and commands for the other arm
    fail without touching hardware; ``home("both")`` homes the active arms.

    ``move_to`` and the other primitives use the latest measured encoder
    state.  A command is planned on the runner and validated again by the
    device service against the measured start, calibrated joint limits, the
    table/obstacles and the other arm's measured pose.  A move executes
    synchronous joint segments; ``reached`` is decided after reading the
    measured joints/TCP back, not from an acknowledgement.  The first version
    does not expose continuous servoing, raw motor commands or SDK handles.

    There is no task-success evaluator: ``robo.done`` raises, results never
    claim task success, and the host records ``environment_success_known =
    false``.  ``finish(status="success")`` is a model statement only.
    ``State.step`` is an observation snapshot number; ``steps_used`` counts
    executed trajectory segments / device actions, not simulator frames.

    Perception uses the calibrated cameras.  With depth, ``locate`` measures
    from valid depth pixels.  Without depth it needs at least two calibrated
    views; one view cannot determine height and returns
    ``measurement_failed``.  Object positions are never read from a task
    configuration.
    """

    NAME: ClassVar[str] = "litearm_real"
    CAMERAS: ClassVar[tuple[str, ...]] = CAMERAS
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        RealState,
        RealArmState,
        RealMove,
        RealGripper,
        RealHome,
        RealPlace,
        RealObjectSight,
        RealSegment,
        RealSeenPoint,
    )
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = frozenset({"move_pose", "grip", "home", "place_on"})
    ABSENT_MEMBERS: ClassVar[frozenset[str]] = frozenset(
        {"done", "world_map", "pi0_pick", "region_center"}
    )
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ()
    VLA_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = ()
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        (camera, "high"): f"{camera}.png" for camera in CAMERAS
    }
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = tuple(f"{camera}.png" for camera in CAMERAS)
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        *LitearmRobot.SUMMARY_KEYS,
        "measured",
        "settled",
        "source",
        "views",
        "surface",
        "holding",
        "width_m",
        "safe_to_home",
    )

    def pi0_pick(self, *args: Any, **kwargs: Any) -> Any:
        raise AttributeError("the real LiteArm station has no VLA: plan with the primitives")

    def _reported_success(self) -> bool:
        return False

    # -- observation --------------------------------------------------------

    @property
    def task(self) -> str:
        """The natural-language instruction supplied at boot (authoritative)."""
        return super().task

    @property
    def done(self) -> bool:
        """Absent: the real path has no task-success evaluator."""
        raise AttributeError(
            "the real LiteArm path has no task-success evaluator: there is no robo.done; "
            "judge progress from measured state and images"
        )

    def world_map(
        self, camera: str = "cam_head", *, resolution: str = "high", step: int = -1
    ) -> Any:
        """Absent: no per-pixel world maps are recorded; use ``back_project``."""
        raise ValueError(
            "the real LiteArm path records no world maps; use back_project(row, col, camera)"
        )

    def state(self, step: int = -1) -> RealState:
        """Read a measured snapshot of the station's active arms.

        Args:
            step: observation snapshot number; ``-1`` refreshes the current
                snapshot (subject to the service's minimum refresh interval)
                and returns it.  Historical snapshots never change with the
                current arm pose.
        Returns:
            RealState with measured TCP poses, joints, gripper openings and
            fault evidence.  An arm that is not configured as active is
            ``None``; ``success_known`` is False.
        """
        return super().state(step)

    def _state_from_envelope(self, envelope: dict[str, Any]) -> RealState:
        summary = envelope.get("state") or {}
        raw = summary.get("state") or {}
        hardware = summary.get("hardware") or {}
        return RealState(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or summary.get("instruction") or ""),
            left=_real_arm(raw, "left") if _arm_present(raw, "left") else None,
            right=_real_arm(raw, "right") if _arm_present(raw, "right") else None,
            truncated=bool(envelope.get("truncated")),
            execution_state=str(hardware.get("execution_state", "unknown")),
            success_known=bool(hardware.get("success_known", False)),
            fault=bool(hardware.get("fault", False)),
            fault_detail=(str(hardware["fault_reason"]) if hardware.get("fault_reason") else None),
        )

    def _summary_dict(self) -> dict[str, Any]:
        state = self.state()
        out: dict[str, Any] = {
            "step": state.step,
            "execution_state": state.execution_state,
            "success_known": False,
        }
        for name in state.active_arms:
            arm = state.arm(name)
            out[f"{name}_tcp"] = [round(v, 4) for v in arm.tcp_pos]
            out[f"{name}_pitch_yaw_roll"] = [
                round(arm.pitch, 1),
                round(arm.yaw, 1),
                round(arm.roll, 1),
            ]
            out[f"{name}_gripper"] = None if arm.gripper is None else round(arm.gripper, 4)
            out[f"{name}_width_m"] = None if arm.width_m is None else round(arm.width_m, 5)
            out[f"{name}_fault"] = arm.fault
        out["truncated"] = False
        return out

    # -- camera -------------------------------------------------------------

    def _primary_camera(self) -> str:
        """The station's ``perception.primary_camera`` (default ``cam_head``)."""

        toolkit = getattr(self._backend, "toolkit", None)
        config = getattr(toolkit, "config", None)
        primary = getattr(config, "primary_camera", None)
        return str(primary) if primary else "cam_head"

    def _resolve_camera(self, camera: str | None) -> str:
        chosen = self._primary_camera() if camera is None else camera
        return _camera(chosen, self.CAMERAS)

    def image(self, camera: str | None = None, *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a uint8 ``(H, W, 3)`` RGB array.

        Omitting ``camera`` selects ``perception.primary_camera`` (``cam_head``
        by default).  ``step=-1`` reads the pinned current snapshot.  The
        camera's resolution is the calibrated one; it is not assumed to be
        480x640.
        """
        return super().image(self._resolve_camera(camera), resolution=resolution, step=step)

    def camera_meta(self, camera: str | None = None, *, step: int = -1) -> dict[str, Any]:
        """Return the calibration of one camera from the same snapshot as the image.

        Omitting ``camera`` selects ``perception.primary_camera``.
        Keys: ``shape``, ``intrinsic_matrix``, ``extrinsic_matrix`` (camera-to-
        world), ``distortion_coefficients``, ``has_depth``, capture timestamps
        and the snapshot ``step``.  A wrist camera's pose is the one computed
        at capture time from the interpolated joint state; it does not follow
        the arm afterwards.
        """
        camera = self._resolve_camera(camera)
        payload = self._readonly("view_camera_meta", {"camera": camera, "step": int(step)})
        return dict(payload)

    # -- manipulation -------------------------------------------------------

    def move_to(
        self,
        xyz: Any,
        arm: str = "right",
        gripper: int = 0,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
    ) -> RealMove:
        """Move one arm's measured grasp point to a world position and orientation.

        The target is planned from the latest measured state.  A straight TCP
        line is generated and checked waypoint by waypoint; synchronous joint
        segments then execute it.  ``reached`` comes from the measured TCP and
        joints after the last segment (0.01 m, 5 degrees), not from an ACK.
        ``blocked`` means the measured arm stopped short of a feasible plan.

        Args:
            xyz: measured grasp-point target [x, y, z], metres.
            arm: ``"left"`` or ``"right"``.
            gripper: ``robo.CLOSE`` (1), ``robo.OPEN`` (-1) or ``robo.KEEP``
                (0).  An open is applied before the move.  A close is applied
                only when the gripper is already holding; closing on an empty
                gripper during a move is refused with a note (call
                ``set_gripper`` after arriving).  ``KEEP`` changes nothing.
            pitch / yaw / roll: desired orientation, degrees; None keeps the
                measured value.  The empty-finger half-turn rule is the same
                as the simulated LiteArm.
            path: ``"auto"``, ``"line"`` or ``"joint"``.
        Returns:
            RealMove.  ``steps_used`` counts executed synchronous segments.
        """
        arm = _arm(arm)
        target = _floats(xyz, 3, "xyz")
        command = _command(gripper, allow_keep=True)
        if path not in ("auto", "line", "joint"):
            raise ValueError(f"path must be 'auto', 'line' or 'joint', got {path!r}")
        kwargs: dict[str, Any] = {
            "xyz": list(target),
            "arm": arm,
            "gripper": command,
            "path": path,
        }
        for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
            if value is not None:
                try:
                    kwargs[name] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be a number or None") from exc

        def build(p: dict[str, Any]) -> RealMove:
            angles = p.get("final_angles") or {}
            final = _tuple3(p.get("final_xyz"))
            target_xyz = _tuple3(p.get("target_xyz", target))
            return RealMove(
                arm=str(p.get("arm", arm)),
                target_xyz=target_xyz,
                start_xyz=_tuple3(p.get("start_xyz")),
                final_xyz=final,
                dist_to_target_m=float(p.get("dist_to_target_m", math.inf)),
                orientation_error_deg=float(p.get("orientation_error_deg", math.nan)),
                pitch=_angle_value(angles, "pitch"),
                yaw=_angle_value(angles, "yaw"),
                roll=_angle_value(angles, "roll"),
                reached=bool(p.get("reached")),
                path=str(p["path"]) if p.get("path") else None,
                ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
                note=str(p["note"]) if p.get("note") else None,
                blocked=bool(p.get("blocked")),
                steps_used=int(p.get("steps_used", 0)),
                truncated=False,
                slip=bool(p.get("slip")),
                slip_opening=(
                    float(p["slip_opening"]) if p.get("slip_opening") is not None else None
                ),
                slip_baseline=(
                    float(p["slip_baseline"]) if p.get("slip_baseline") is not None else None
                ),
            )

        return self._stateful("move_pose", kwargs, build)

    def set_gripper(self, arm: str, gripper: int, opening: float | None = None) -> RealGripper:
        """Open or close one LiteGrip, without moving the arms.

        A close starts a persistent calibrated-force squeeze.  Repeating the
        close while the hold is active does not restart it.  ``holding`` means
        the measured fingers stopped apart; it does NOT prove the intended
        object is held.  ``width_m`` is the measured finger opening, not the
        simulator's fixed 2 mm + 85 mm formula.

        Args:
            arm: ``"left"`` or ``"right"``.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1).
            opening: with OPEN only, target normalised opening in (0, 1].
        Returns:
            RealGripper with ``settled`` and measured opening/width.
        """
        arm = _arm(arm)
        command = _command(gripper, allow_keep=False)
        kwargs: dict[str, Any] = {"arm": arm, "gripper": command}
        if opening is not None:
            try:
                fraction = float(opening)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"opening must be a number in (0, 1], got {opening!r}") from exc
            if command != -1 or not 0.0 < fraction <= 1.0:
                raise ValueError("opening goes with robo.OPEN and must be in (0, 1]")
            kwargs["opening"] = fraction

        def build(p: dict[str, Any]) -> RealGripper:
            state = self._latest_state()
            width = p.get("width_m")
            opening = p.get("opening")
            if opening is None:
                opening = state.arm(arm).gripper
            return RealGripper(
                arm=arm,
                commanded=str(p.get("gripper") or ("closed" if command > 0 else "open")),
                opening=float(opening) if opening is not None else 0.0,
                holding=bool(p.get("holding")),
                width_m=float(width) if width is not None else None,
                steps_used=int(p.get("steps_used", 0)),
                step=int(p["step"]) if p.get("step") is not None else None,
                step_limit=None,
                truncated=False,
                settled=bool(p.get("settled")),
                note=str(p["note"]) if p.get("note") else None,
            )

        return self._stateful("grip", kwargs, build)

    def home(self, arm: str = "both") -> RealHome:
        """Return one or all active arms to the calibrated safe pose.

        ``"both"`` means every arm in ``execution.active_arms``; naming an arm
        that is not active fails before anything is commanded.

        This is NOT the SDK's ``home()`` (which drives the firmware zero pose):
        the service plans a measured vertical lift, checks the return path
        against the table/obstacles/other arm and executes synchronous joint
        segments.  ``reached`` is measured.  The gripper keeps its command.
        """
        if arm not in (*self.ARMS, "both"):
            raise ValueError(f"arm must be 'left', 'right' or 'both', got {arm!r}")
        which = str(arm)

        def build(p: dict[str, Any]) -> RealHome:
            error = p.get("max_joint_error_rad")
            return RealHome(
                arm=str(p.get("arm", which)),
                reached=bool(p.get("reached")),
                max_joint_error_rad=float(error) if error is not None else None,
                steps_used=int(p.get("steps_used", 0)),
                truncated=False,
                note=str(p["note"]) if p.get("note") else None,
            )

        return self._stateful("home", {"arm": which}, build)

    def place_on(
        self,
        x: float,
        y: float,
        support_top: float,
        tcp_to_bottom: float,
        *,
        arm: str = "right",
        clearance: float = 0.002,
        lift_m: float = 0.12,
        retreat_xy: tuple[float, float] = (0.0, -0.10),
        opening: float | None = None,
        path: str = "line",
    ) -> RealPlace:
        """Put the held object on a measured support, release, lift and retreat.

        ``support_top`` and ``tcp_to_bottom`` must come from measurement or the
        current grasp geometry; no station constant is assumed.  The gripper is
        opened only after the measured descent arrives and is not blocked or
        slipping.  ``safe_to_home`` is True only when the lift and retreat both
        arrived without being blocked.  This call never homes.
        """
        arm = _arm(arm)
        values: dict[str, float] = {}
        for name, value in (
            ("x", x),
            ("y", y),
            ("support_top", support_top),
            ("tcp_to_bottom", tcp_to_bottom),
            ("clearance", clearance),
            ("lift_m", lift_m),
        ):
            try:
                values[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number, got {value!r}") from exc
        if values["tcp_to_bottom"] <= 0.0:
            raise ValueError("tcp_to_bottom must be positive")
        if values["clearance"] < 0.0 or values["lift_m"] < 0.0:
            raise ValueError("clearance and lift_m must not be negative")
        try:
            retreat = [float(v) for v in retreat_xy]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"retreat_xy must be two numbers, got {retreat_xy!r}") from exc
        if len(retreat) != 2:
            raise ValueError("retreat_xy must be two numbers")
        if path not in ("auto", "line", "joint"):
            raise ValueError("path must be 'auto', 'line' or 'joint'")
        kwargs: dict[str, Any] = {
            "x": values["x"],
            "y": values["y"],
            "support_top": values["support_top"],
            "tcp_to_bottom": values["tcp_to_bottom"],
            "arm": arm,
            "clearance": values["clearance"],
            "lift_m": values["lift_m"],
            "retreat_xy": retreat,
            "path": path,
        }
        if opening is not None:
            fraction = float(opening)
            if not 0.0 < fraction <= 1.0:
                raise ValueError("opening must be in (0, 1]")
            kwargs["opening"] = fraction

        def build(p: dict[str, Any]) -> RealPlace:
            target = p.get("target_xyz") or (0.0, 0.0, 0.0)
            return RealPlace(
                arm=str(p.get("arm", arm)),
                placed=bool(p.get("placed")),
                released=bool(p.get("released")),
                safe_to_home=bool(p.get("safe_to_home")),
                target_xyz=tuple(float(v) for v in target[:3]),
                place_z=float(p.get("place_z", math.nan)),
                descent_reached=bool(p.get("descent_reached")),
                descent_blocked=bool(p.get("descent_blocked")),
                descent_slip=bool(p.get("descent_slip")),
                open_opening=(
                    float(p["open_opening"]) if p.get("open_opening") is not None else None
                ),
                lift_reached=bool(p.get("lift_reached")),
                lift_blocked=bool(p.get("lift_blocked")),
                retreat_reached=bool(p.get("retreat_reached")),
                retreat_blocked=bool(p.get("retreat_blocked")),
                steps_used=int(p.get("steps_used", 0)),
                truncated=False,
                note=str(p["note"]) if p.get("note") else None,
            )

        return self._stateful("place_on", kwargs, build)

    # -- perception ---------------------------------------------------------

    def segment(
        self, text_prompt: str, camera: str | None = None, min_score: float = 0.2
    ) -> RealSegment:
        """Segment an object by text in the current snapshot.

        Omitting ``camera`` selects ``perception.primary_camera``.  The mask is
        SAM3's best match, not proof of identity.  Pass its centroid to
        ``back_project`` with the same camera to get a measured world point.
        """
        camera = self._resolve_camera(camera)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        p = self._readonly(
            "segment",
            {"text_prompt": prompt, "camera": camera, "min_score": float(min_score)},
        )
        centroid = p.get("centroid_rc") or ()
        return RealSegment(
            found=bool(p.get("found")),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            score=float(p["score"]) if p.get("score") is not None else None,
            box=p.get("box_px"),
            camera=str(p.get("camera", camera)),
        )

    def back_project(self, row: int, col: int, camera: str | None = None) -> RealSeenPoint:
        """Return the measured world point seen at a pixel.

        Omitting ``camera`` selects ``perception.primary_camera``.  With a valid
        depth pixel this is the deprojected surface point.  Without depth, the
        object at the pixel is segmented and measured from calibrated views; if
        that fails the table point is returned only when the pixel's calibrated
        ray meets the table.  A pixel in mid-air over no calibrated surface
        raises rather than returning a made-up point.
        """
        camera = self._resolve_camera(camera)
        p = self._readonly("back_project", {"row": int(row), "col": int(col), "camera": camera})
        pixel = p.get("pixel") or (int(row), int(col))
        return RealSeenPoint(
            pixel=(int(pixel[0]), int(pixel[1])),
            xyz=_tuple3(p.get("world_xyz")),
            depth_m=float(p.get("depth_m", math.nan)),
            camera=str(p.get("camera", camera)),
            surface=str(p.get("surface") or "table"),
        )

    def locate(
        self, text_prompt: str, camera: str | None = None, min_score: float = 0.2
    ) -> RealObjectSight:
        """Find an object and measure it in 3D from the calibrated cameras.

        Omitting ``camera`` selects ``perception.primary_camera``, so a
        wrist-only station can call ``locate(prompt)`` without a head camera.
        With depth, the mask's valid depth pixels give the top, height and
        footprint.  Without depth, at least two calibrated views are required;
        the visual hull of their masks gives the extent.  One RGB view cannot
        determine height and returns ``measurement_failed=True`` with a note;
        no table intersection or fixed table height is used as a substitute.

        Args:
            text_prompt: what to find, e.g. ``"the red block"``.
            camera: a configured camera name, or None for the primary camera.
            min_score: SAM3 masks below this score are not found.
        Returns:
            RealObjectSight.  Check ``measurement_failed`` and ``views``.  With
            ``views >= 2`` (or depth) the top/centre/footprint are measured;
            a ``note`` still describes occlusion or a rough fit.  The mask is
            not proof of identity.
        """
        camera = self._resolve_camera(camera)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        p = self._readonly(
            "locate",
            {"text_prompt": prompt, "camera": camera, "min_score": float(min_score)},
        )
        centroid = p.get("centroid_rc") or ()
        failed = bool(p.get("measurement_failed", p.get("center_xyz") is None))
        note = p.get("geometry_error") or p.get("note")
        return RealObjectSight(
            found=bool(p.get("found")),
            center_xyz=_tuple3(p.get("center_xyz")) if p.get("center_xyz") is not None else None,
            top_z=float(p["top_z"]) if p.get("top_z") is not None else None,
            height_m=float(p["height_m"]) if p.get("height_m") is not None else None,
            length_m=float(p["length_m"]) if p.get("length_m") is not None else None,
            width_m=float(p["width_m"]) if p.get("width_m") is not None else None,
            long_axis_deg=(
                float(p["long_axis_deg"]) if p.get("long_axis_deg") is not None else None
            ),
            grasp_yaw=float(p["grasp_yaw"]) if p.get("grasp_yaw") is not None else None,
            score=float(p["score"]) if p.get("score") is not None else None,
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            camera=str(p.get("camera", camera)),
            note=str(note) if note else None,
            views=int(p.get("views") or 1),
            source=str(p["source"]) if p.get("source") else None,
            measurement_failed=failed,
            measurement_quality=p.get("measurement_quality"),
        )


ADAPTER = RobotAdapter(
    name="litearm_real",
    robot_cls=LitearmRealRobot,
    host="pyrualean.hosts.litearm_hardware",
    knowledge="litearm-real",
    blurb="a calibrated dual-arm LiteArm workstation with LiteGrip grippers",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('cam_head')`, `robo.show('cam_left_wrist')` or `robo.show('cam_right_wrist')`"
    ),
    services="",
    services_novla="",
)


__all__ = [
    "ADAPTER",
    "CAMERAS",
    "LitearmRealRobot",
    "RealArmState",
    "RealGripper",
    "RealHome",
    "RealMove",
    "RealObjectSight",
    "RealPlace",
    "RealSeenPoint",
    "RealSegment",
    "RealState",
]
