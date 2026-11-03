# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LiteArm under RoboDojo's own evaluation client, as that client is configured.

The robot is the dual LiteArm SE station: two 7-DoF arms with LiteGrip parallel grippers at a
table, driven by joint actions through RoboDojo's evaluation client.  It is the same benchmark
and the same host as :mod:`pyrualean.robodojo_xpl_rgb`; only the embodiment differs:

*   seven arm joints (``joint1`` .. ``joint7``) instead of six, and a LiteGrip with two
    prismatic fingers (0 = open, 0.042726 m = closed) instead of the X5's two-finger gripper;
*   the reported end-effector pose is ``tcp_link``, the grasp point itself, so there is no
    TCP offset: ``ArmState.tcp_pos`` and ``ArmState.eef_pos`` are the same point;
*   the wrist camera is mounted on ``camera_link`` (13.06 cm behind and 6 cm to the side of the
    grasp point, looking 20 degrees down), not on the X5's forearm.

The host is :mod:`pyrualean.hosts.xpolicylab_litearm` and the primitives live in
:mod:`pyrualean.hosts.litearm_tools` / :mod:`pyrualean.hosts.litearm_kinematics`.  There is no
VLA in this adapter: run it with ``--primitives no-vla``.

The docstrings in this module are the model-facing documentation: the prompt is generated from
them (see :mod:`pyrualean.prompt`), so keep them accurate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from .robodojo import (
    Home,
    Move,
    Segment,
    _arm,
    _camera,
    _command,
    _scalar,
    _sequence,
    gripper_angles,
)
from .robodojo_xpl_rgb import ObjectSight, SeenPoint, RobodojoOfficialColourRobot
from .robots import RobotAdapter

#: The grasp point is ``tcp_link`` itself; the LiteGrip pads span 4.6 cm either side of it.
TCP_OFFSET = 0.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmState:
    """One LiteArm arm at a recorded step.

    tcp_pos: the grasp point [x, y, z] in the world frame, metres: the middle
        of the LiteGrip finger pads.  ``move_to`` drives this point.  It is
        also the end-effector pose RoboDojo reports (``eef_pos`` is the same
        point): the pads reach 4.6 cm to either side of it.
    pitch / yaw / roll: the gripper's orientation, degrees (see the class
        docstring): ``pitch`` 0 points level and 90 straight down (-90
        straight up, the start pose), ``yaw`` is the heading of the pointing
        direction on the table (0 away from the robot, +90 towards the
        robot's left), ``roll`` the turn of the fingers about the pointing
        direction (0 as at the start).
    eef_pos: position of the end-effector link (``tcp_link``), metres; the
        same point as ``tcp_pos``.
    eef_quat: end-effector orientation as a unit quaternion [w, x, y, z].
    gripper: measured gripper opening, normalised: 1.0 = open, 0.0 = closed on
        nothing; closed on an object 0.0015 + 0.0855 x this reading is the
        object's width in metres.
        (Gripper COMMANDS use the opposite sign convention: see ``set_gripper``.)
    joints: the seven arm joint angles, radians (the last turns the fingers
        about the pointing direction).
    """

    tcp_pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    gripper: float
    joints: tuple[float, ...]


@dataclass(frozen=True)
class State:
    """Snapshot of both arms at one recorded step.

    step: index of the recorded step (0 = initial scene).  A new step is
        recorded after every motion call.
    task: the task instruction given by the benchmark (authoritative).
    left / right: the two arms (``state.arm("left")`` picks one by name).
    truncated: the native-action budget of the episode is exhausted; no
        further motion is possible.
    """

    step: int
    task: str
    left: ArmState
    right: ArmState
    truncated: bool

    def arm(self, name: str) -> ArmState:
        if name == "left":
            return self.left
        if name == "right":
            return self.right
        raise ValueError("arm must be 'left' or 'right'")


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper``.

    commanded: ``"closed"`` or ``"open"``.
    opening: the opening once the fingers stopped (1.0 = open, 0.0 = closed on
        nothing).
    holding: after a close, the fingers stopped apart (an opening above 0.01,
        about 2.4 mm at the pads): something is between them.  It does not say
        what: check that the intended object moves with the arm.
    width_m: after a close, the width of what is held, across the fingers
        (0.0 when nothing is), else None.
    steps_used: native actions consumed.
    step / step_limit: native actions executed so far and the episode budget
        (None if the runtime did not report them).
    truncated: the native-action budget ran out.
    """

    arm: str
    commanded: str
    opening: float
    holding: bool
    width_m: float | None
    steps_used: int
    step: int | None
    step_limit: int | None
    truncated: bool


@dataclass(frozen=True)
class Place:
    """Outcome of ``place_on``.

    placed: the object was released at the place height: the descent arrived and
        the gripper opened.
    released: the open finished (the fingers settled), not cut off by the
        budget.
    safe_to_home: the lift and the sideways retreat both finished without
        ``blocked``; only then may the caller ``home``.
    target_xyz / place_z: the grasp point the descent aimed at.
    descent_reached / descent_blocked / descent_slip: what the descent reported.
    open_opening: the measured opening after the open.
    lift_reached / lift_blocked, retreat_reached / retreat_blocked: the two
        movements after the release.
    steps_used / truncated / note: as on ``Move``.
    """

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


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def _arm_state(raw: dict[str, Any], name: str) -> ArmState:
    pose = _sequence(raw.get(f"{name}_ee_pose"))
    if len(pose) != 7:
        pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    pitch, yaw, roll = gripper_angles(pose[3:])
    return ArmState(
        tcp_pos=pose[:3],  # type: ignore[arg-type]
        pitch=round(pitch, 2),
        yaw=round(yaw, 2),
        roll=round(roll, 2),
        eef_pos=pose[:3],  # type: ignore[arg-type]
        eef_quat=pose[3:],  # type: ignore[arg-type]
        gripper=_scalar(raw.get(f"{name}_ee_joint_state")),
        joints=_sequence(raw.get(f"{name}_arm_joint_state")),
    )


# ---------------------------------------------------------------------------
# The robot
# ---------------------------------------------------------------------------


class LitearmRobot(RobodojoOfficialColourRobot):
    """The RoboDojo dual-arm LiteArm station: two 7-DoF LiteArm SE arms with LiteGrip grippers.

    World frame: x runs across the table (the LEFT arm is at negative x and
    appears on the LEFT of the head image), y points away from the robot
    towards the far edge of the table, z is up; metres.  Orientations are unit
    quaternions ordered [w, x, y, z].  The arms are ``"left"`` and ``"right"``
    (``robo.LEFT`` / ``robo.RIGHT``); a motion call names one arm and holds
    the other where it is.  The arms stand on the table at x = -0.30 and
    +0.30, y = -0.45; the table surface is at z = 0.765.

    ``move_to`` drives an arm's GRASP POINT, the middle of the LiteGrip finger
    pads (``state().left.tcp_pos``), to a world position, with an orientation
    given in degrees: ``pitch`` is how far the gripper points below the
    horizontal (0 level, 90 straight down, -90 straight up), ``yaw`` the
    heading of the pointing direction on the table (0 away from the robot,
    +90 towards the robot's left, -90 towards its right) and ``roll`` the
    turn of the fingers about the pointing direction (0: side by side, as at
    the start).  Pointing straight down, the fingers close along the line at
    ``yaw`` degrees from +x (counter-clockwise seen from above):
    ``locate``'s ``grasp_yaw`` is that line across an object's narrow side.
    The arm starts folded forward over the table at about pitch 10 and yaw
    -10 (grasp point near [0.30, -0.20, 0.95]); a move to pitch=90 folds the
    gripper down to the table.

    The LiteGrip's pads are long: they span 4.6 cm either side of the grasp
    point along the pointing direction, so pointing straight down the grasp
    point cannot go lower than about 4.6 cm above the surface under it.  The
    gripper opens to 8.7 cm between the inner faces and closes to 1.5 mm.

    Two gripper conventions, do not mix them up.  COMMANDS are signed:
    ``robo.CLOSE`` (1) closes, ``robo.OPEN`` (-1) opens, ``robo.KEEP`` (0)
    leaves the gripper as it is.  MEASUREMENTS (``state().left.gripper``,
    ``Gripper.opening``) are openings: 1.0 = open, 0.0 = closed on nothing.

    Every native action (one step of ``move_to``, ``set_gripper`` or
    ``home``) counts against the episode's native-action budget; when it is
    spent the state reports ``truncated`` and motion calls raise
    ``EpisodeFinished``.  Motion calls block until the primitive finishes and
    return a typed result; nothing is queued.

    Perception is from colour images alone; there is no depth image.  A fixed
    head camera (``cam_head``, the whole table) and one wrist camera per arm
    (``cam_left_wrist``, ``cam_right_wrist``), 640x480 each.  ``locate`` finds
    an object by a text prompt and measures it in 3D (centre, top, footprint,
    heading) from the cameras that see it; ``segment`` / ``back_project`` give
    a mask centroid and the world point at one pixel.  Object poses are never
    given.

    There is no VLA on this robot: plan with the geometric primitives.
    """

    NAME: ClassVar[str] = "litearm"
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        Home,
        Place,
        ObjectSight,
        Segment,
        SeenPoint,
    )
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = (
        RobodojoOfficialColourRobot.STATEFUL_TOOLS | frozenset({"place_on"})
    )
    #: no success flag while the episode runs, no recorded world maps, and no VLA.
    ABSENT_MEMBERS: ClassVar[frozenset[str]] = frozenset({"done", "world_map", "pi0_pick"})
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ()
    VLA_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = ()

    def pi0_pick(self, *args: Any, **kwargs: Any) -> Any:
        raise AttributeError("the LiteArm adapter has no VLA: plan with the geometric primitives")

    # -- observation --------------------------------------------------------

    def state(self, step: int = -1) -> State:
        """Read the recorded state of both arms.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene.  A new step is recorded after every motion call.
        Returns:
            State with ``left`` / ``right`` end-effector poses (the grasp
            point itself), measured gripper openings and the seven joint
            angles, and the ``truncated`` flag.
        """
        return super().state(step)

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        summary = envelope.get("state") or {}
        raw = summary.get("state") or {}
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or summary.get("instruction") or ""),
            left=_arm_state(raw, "left"),
            right=_arm_state(raw, "right"),
            truncated=bool(envelope.get("truncated")),
        )

    def camera_meta(self, camera: str = "cam_head", *, step: int = -1) -> dict[str, Any]:
        """Return the calibration of one camera at a recorded step as a dict.

        Keys: ``shape`` (image shape), ``intrinsic_matrix`` (3x3) and
        ``extrinsic_matrix`` (4x4 camera-to-world; the camera looks down its
        own -z axis with +y up).  The head camera is fixed; a wrist camera
        moves with its arm, and its pose is worked out from that arm's state.
        """
        from .hosts import litearm_sight as sight

        camera = _camera(camera, self.CAMERAS)
        if camera == "cam_head":
            view = sight.head_view()
        else:
            arm = self.state(step).arm("left" if camera == "cam_left_wrist" else "right")
            view = sight.wrist_view([*arm.eef_pos, *arm.eef_quat])
        k = [[view.fx, 0.0, view.cx], [0.0, view.fy, view.cy], [0.0, 0.0, 1.0]]
        return {
            "shape": [view.shape[0], view.shape[1], 3],
            "intrinsic_matrix": k,
            "extrinsic_matrix": [[float(v) for v in row] for row in view.pose],
        }

    # -- gripper ------------------------------------------------------------

    def set_gripper(self, arm: str, gripper: int, opening: float | None = None) -> Gripper:
        """Close or open one gripper, without moving the arms.

        The command is repeated until the opening has not changed for three
        actions (about 32 for a full close from open at the LiteGrip's 50 mm/s).
        After a close, ``holding`` tells whether something stopped
        the fingers apart and ``width_m`` how wide it is.  Later motions with
        ``robo.KEEP`` keep commanding what was commanded last, so a held
        object stays squeezed and a partial opening stays as it is.

        Args:
            arm: ``"left"`` or ``"right"``.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1).
            opening: with ``robo.OPEN`` only: open to this reading
                (0 < opening <= 1) instead of all the way, e.g. to reach
                between close neighbours.  Something 0.0015 + 0.0855 x
                opening metres wide then just fits between the fingers.
        Returns:
            Gripper.  ``holding`` is not proof that the INTENDED object is
            held: confirm it moves with the arm (wrist camera, or ``locate``
            it again after a lift).
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

        def build(p: dict[str, Any]) -> Gripper:
            status = p.get("status") or {}
            state = self._latest_state()
            width = p.get("width_m")
            return Gripper(
                arm=arm,
                commanded=str(p.get("gripper") or ("closed" if command > 0 else "open")),
                opening=float(p.get("opening", state.arm(arm).gripper)),
                holding=bool(p.get("holding")),
                width_m=float(width) if width is not None else None,
                steps_used=int(p.get("steps_used", 0)),
                step=int(status["step"]) if status.get("step") is not None else None,
                step_limit=(
                    int(status["step_limit"]) if status.get("step_limit") is not None else None
                ),
                truncated=state.truncated,
            )

        return self._stateful("grip", kwargs, build)

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
    ) -> Place:
        """Put the held object on a support and back the arm clear.

        ``support_top`` is the support's top z, measured before the pick or
        from the head camera (a holding arm's wrist camera sees the held
        object, not the support).  ``tcp_to_bottom`` is the distance from the
        grasp point to the held object's BOTTOM for this grasp, not a
        universal constant: the current stack block hangs 0.050 m below the
        grasp point.  The descent targets ``support_top + tcp_to_bottom +
        clearance``.

        The gripper is opened only when the descent arrived and was not
        blocked or slipping; the arm then rises ``lift_m`` and moves
        ``retreat_xy`` away with the fingers kept open.  This call never
        homes: test ``safe_to_home`` and only then ``home``.

        Args:
            x / y: the support centre (the place point), metres.
            support_top: the support's top z, metres.
            tcp_to_bottom: grasp point to the held object's bottom, metres.
            arm: ``"left"`` or ``"right"`` (default ``"right"``).
            clearance: extra height above the computed bottom contact, metres.
            lift_m: straight-up distance after the release, metres.
            retreat_xy: sideways move after the lift, metres.
            opening: width to leave between the fingers after the open
                (0 < opening <= 1); None opens fully.
            path: ``"line"`` (default), ``"auto"`` or ``"joint"``.
        Returns:
            Place.  ``placed`` and ``released`` say the object was put down;
            ``safe_to_home`` says the arm got clear.
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
            raise ValueError(f"path must be 'auto', 'line' or 'joint', got {path!r}")
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
            try:
                fraction = float(opening)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"opening must be a number in (0, 1], got {opening!r}") from exc
            if not 0.0 < fraction <= 1.0:
                raise ValueError("opening must be in (0, 1]")
            kwargs["opening"] = fraction

        def build(p: dict[str, Any]) -> Place:
            target = p.get("target_xyz") or (0.0, 0.0, 0.0)
            return Place(
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
                truncated=self._latest_state().truncated,
                note=str(p["note"]) if p.get("note") else None,
            )

        return self._stateful("place_on", kwargs, build)


ADAPTER = RobotAdapter(
    name="litearm",
    robot_cls=LitearmRobot,
    host="pyrualean.hosts.xpolicylab_litearm",
    knowledge="litearm",
    blurb="a dual-arm LiteArm SE robot with LiteGrip grippers in the RoboDojo benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('cam_head')`, `robo.show('cam_left_wrist')` or `robo.show('cam_right_wrist')`"
    ),
    services="a SAM3 segmentation service",
    services_novla="a SAM3 segmentation service",
)


__all__ = ["ADAPTER", "ArmState", "Gripper", "LitearmRobot", "Place", "State"]
