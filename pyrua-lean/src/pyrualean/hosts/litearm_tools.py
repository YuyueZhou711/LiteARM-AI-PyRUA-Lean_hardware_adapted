# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LiteArm tools of this library's own, registered on RPent's toolkit at boot.

RPent's RoboDojo toolset moves an arm only by position (``move_to``) and grips one native
action at a time (``set_gripper``); its perception returns one mask centroid and one surface
point.  The tools here are what a program needs on top of that, computed in the host from the
same observations and commanded through the same action path:

``move_pose``    the grasp point to a position AND an orientation, along a straight line,
                 with the host's own IK (:mod:`.litearm_kinematics`);
``grip``         a full gripper stroke, repeated until the fingers stop, with what they hold;
``locate``       an object's 3D extent, heading and top from its mask and the depth image;
``home``         the joints the episode started with (:func:`home`, kept from before).

Every tool is registered as a stateful (not read-only) tool, so RPent's toolkit records a step
after it like after its own motions.  Each native action is one ``env.step`` and counts against
the episode's budget; a tool stops at the step limit.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

from ..robodojo import HOME_MAX_ACTIONS, HOME_TOL_RAD
from . import litearm_kinematics as kin
from . import litearm_sight as sight

ARMS = kin.ARMS
#: Extra native actions at the end of a motion while the joints settle on the target.
SETTLE_ACTIONS = 32
#: The motion has arrived when the grasp point is this close (metres) and turned this little.
ARRIVE_M = 0.01
ARRIVE_DEG = 5.0
#: A gripper stroke stops when the opening changes less than this in one action.
#: The SDK's open/close/grasp speed is 50 mm/s of jaw travel (litegrip-python,
#: MotionConfig.speed_mm_s), i.e. 25 mm/s per finger.  One native action moves the
#: opening ~1.4 mm; the normalized still threshold below is 0.0425 mm of jaw
#: travel, and a full 87 mm stroke needs
#: ~60 actions.  A 100 mm/s trial carried the block worse (it slipped out during
#: the lift), so the sim keeps the SDK's 50 mm/s.
GRIP_STILL = 0.0005
GRIP_STILL_ACTIONS = 3
GRIP_MAX_ACTIONS = 80
#: Largest joint change commanded in one action of a joint-space swing (radians).
JOINT_STEP_RAD = 0.15
#: A pose the IK misses by no more than this is taken (the closest one), with a note: in the
#: first sweep 11 of 17 refused targets were within 2 cm and 7 degrees of a reachable pose.
NEAR_M = 0.02
NEAR_DEG = 8.0
#: ``home`` first lifts a grasp point lower than this straight up (by up to HOME_LIFT_M) so the
#: swing back does not sweep across the table: one swing knocked a placed board off.
HOME_LIFT_BELOW = 1.0
HOME_LIFT_M = 0.06
#: ``locate`` measures a footprint on a grid this fine (metres), one point per occupied cell:
#: an oblique camera sees an object's sides edge-on, and counted pixel by pixel they outweigh
#: its top (a cube's sides pulled its long axis onto the diagonal).
FOOTPRINT_CELL_M = 0.002
#: A footprint spread at least this many times further along its principal axis than across it
#: (variances) is measured along its principal axes; one nearer to a square or a circle has no
#: axis of its own and takes the sides of the smallest rectangle around it.
LONG_FOOTPRINT = 1.4
#: The extents of a footprint leave this percentage of its cells outside at either end: a few
#: stray cells at the mask's edge do not stretch it, and a thin end (a brush's head) stays in.
FOOTPRINT_TRIM = 0.5
#: The lowest points of a mask are as often what the object stands on as the object (a pad
#: under a cube, 5 mm high, added 1.5 cm to its footprint): for an object taller than
#: FOOT_MIN_HEIGHT_M the footprint is taken from the points more than FOOT_M above its lowest.
FOOT_M = 0.003
FOOT_MIN_HEIGHT_M = 0.012
#: ``locate`` completes a round object's hidden far side: a slice this thick (metres) whose
#: points lie on an arc of a circle of a graspable size, within this fraction of the radius
#: (the two sides of a box corner miss their best circle by a tenth of it) and over at least
#: this much of the circle, counts as round.
SLICE_M = 0.01
ROUND_MIN_RADIUS_M = 0.008
ROUND_MAX_RADIUS_M = 0.12
ROUND_FIT = 0.04
ROUND_MIN_ARC_DEG = 100.0
#: ... and, of three or more such slices, one whose centre is further than this (metres) from
#: the median of their centres is dropped: a sparse slice of a doll once fitted a circle twice
#: too large.  More than half of them have to be that close, or none is used.
ROUND_AXIS_M = 0.006
#: The LiteGrip beyond the arm's links, in the tcp_link frame (x the pointing direction, y the
#: closing axis; metres): the finger rail 4 cm behind the grasp point, 4.35 cm to either side,
#: and the two fingers from 4.6 cm behind to 4.6 cm ahead of it, 0.8 mm off the axis closed and
#: 4.35 cm open.  ``locate`` drops mask pixels on them.
RAIL_AT_M = -0.040
RAIL_HALF_M = 0.0435  # the bare finger's inner face at full opening
RAIL_RADIUS_M = 0.035
FINGER_FROM_M = -0.046
FINGER_CLOSED_M = 0.0008
FINGER_TRAVEL_M = 0.0425
FINGER_RADIUS_M = 0.030
#: Without a depth image an object is looked for in the other cameras at pixels at least this
#: far apart (pixels), and a pixel asked about lies on a measured object when one of that
#: object's measured pixels is this near.  A mask over more than PROBE_TABLE_SHARE of the image
#: is the table.
SEED_APART_PX = 6
#: Without depth, a mask is the robot's own arm when the segmenter was this unsure of it and
#: this share of it lies on the robot's outline in the picture (a block under an open gripper
#: lies on that outline too, but is found with a high score).
ROBOT_MASK_SCORE = 0.5
ROBOT_MASK_SHARE = 0.5
#: ... and a measured centre this far outside the table top is not an object on the table.
OFF_TABLE_M = 0.02
#: The robot is in front of part of an object when this share of the rim just outside its mask
#: (RIM_PX wide) lies on the robot's outline.
HIDDEN_RIM_SHARE = 0.15
RIM_PX = 3
#: Two cameras whose masks fit the measured solid less well than this (F1) agree only roughly:
#: over 260 objects the height was then off by 7 mm at the median and 35 mm at the worst tenth,
#: against 3 and 15 above it.
ROUGH_MATCH = 0.9
PROBE_REACH_PX = 2.5
PROBE_TABLE_SHARE = 0.25
#: ``move_pose`` may take the half-turned pose (fingers swapped) only with a gripper this open.
OPEN_FOR_HALF_TURN = 0.9
#: A close on nothing reaches the joint stop (opening 0.0).  The model uses the official
#: finger meshes (no added pad geometry): the held width is the bare finger inner-face gap.  Above an opening of 0.010 that gap is 2.85 mm, more than the empty
#: pose's 2.0 mm, so an empty close is not reported as holding and a 3 mm rim still is.
HOLD_MIN_OPENING = 0.010
#: The width of what is held at a measured opening, metres: GRIP_ZERO_M + GRIP_SPAN_M x
#: opening.  The LiteGrip's official stroke is 85.0 mm (42.5 mm per finger); at q =
#: 0.0425 the URDF's bare finger inner faces are 2.0 mm apart, and at full opening
#: 2.0 + 85.0 = 87.0 mm.  The real product's closed gap is 0, within the 2.708 mm
#: position accuracy, so the same formula is used for the real gripper (0 .. 87 mm).
GRIP_ZERO_M = 0.0020
GRIP_SPAN_M = 0.0850
#: A carried object has slipped when its opening falls this far below the one a
#: settled close left.  0.02 is about 1.7 mm of jaw travel (the width formula
#: moves 85 mm per opening unit).  Initial value; `diagnose_carry.py` records the
#: per-action readings so the threshold can be calibrated against real slip data.
CARRY_SLIP_DROP = 0.02
#: A held object is carried at this finer joint step (more actions per waypoint,
#: so a slower motion) than free-space ``JOINT_STEP_RAD``.  The 0.15/0.025
#: carry diagnostic showed the yellow block tilting and sliding out at 0.15
#: while holding at 0.025; carrying uses the slower value.
CARRY_JOINT_STEP_RAD = 0.025


# ---------------------------------------------------------------------------
# observation and action plumbing
# ---------------------------------------------------------------------------


def _obs(primitives: Any) -> dict[str, Any]:
    obs = primitives.env.get_obs()
    primitives._last_obs = obs
    return obs


def _joints(state: dict[str, Any], arm: str) -> list[float]:
    return [float(v) for v in state[f"{arm}_arm_joint_state"]]


def _opening(state: dict[str, Any], arm: str) -> float:
    value = state.get(f"{arm}_ee_joint_state")
    for _ in range(4):
        if value is None or isinstance(value, (int, float)):
            break
        try:
            value = next(iter(value))
        except (TypeError, StopIteration):
            break
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _held_openings(primitives: Any) -> dict[str, float] | None:
    """The opening each arm's last settled close left, kept across motion calls.

    ``grip`` records it on a close that holds and clears it on an open.  ``move_pose``
    measures a carry against it, so a block that slipped between two segments is still
    compared with the close, not with its already-slipped reading.
    """

    held = getattr(primitives, "_held_openings", None)
    if held is None:
        held = {}
        try:
            primitives._held_openings = held
        except AttributeError:  # a read-only primitives stub: no cross-call baseline
            return None
    return held


def _command(gripper: Any) -> float | None:
    """1 (close) -> 0.0, -1 (open) -> 1.0, 0 / None (keep) -> None."""

    if gripper is None:
        return None
    g = float(gripper)
    if g > 0:
        return 0.0
    if g < 0:
        return 1.0
    return None


def _act(
    primitives: Any,
    state: dict[str, Any],
    joints: dict[str, Sequence[float]],
    grips: dict[str, float | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One native action: the arms named in ``joints`` to those angles, the others held."""

    action: dict[str, list[float]] = {}
    for arm in ARMS:
        q = joints.get(arm)
        action[f"{arm}_arm_joint_state"] = [
            float(v) for v in (q if q is not None else _joints(state, arm))
        ]
        g = grips.get(arm)
        if g is None:
            g = _keep(primitives, state, arm)
        _remember(primitives, arm, g)
        action[f"{arm}_ee_joint_state"] = [float(g)]
    _obs_step, _reward, _done, info = primitives.env.step(action)
    obs = _obs(primitives)
    return obs["state"], (info.get("status") or {}) if isinstance(info, dict) else {}


def _remember(primitives: Any, arm: str, command: float) -> None:
    commands = getattr(primitives, "_grip_commands", None)
    if commands is None:
        commands = {}
        try:
            primitives._grip_commands = commands
        except AttributeError:  # an object that takes no attributes: KEEP measures, as before
            return
    commands[arm] = float(command)


def _keep(primitives: Any, state: dict[str, Any], arm: str) -> float:
    """The opening to command for a gripper that keeps what it has.

    The last command again, not the measured opening: commanding the opening a held object
    leaves takes the squeeze off (a held object was lost after the OTHER arm's motions 9.6 % of
    the time, after its own 6.1 %).  Only a gripper last told to CLOSE is held closed.  After an
    OPEN command a smaller reading means the fingers are being pushed shut by something the arm
    is moving past (a home sweep through a stack did exactly this); that must keep commanding
    the open target, never turn into a close that grabs what the fingers touched.  A reading
    wider than the last open command is followed.
    """

    measured = _opening(state, arm)
    last = (getattr(primitives, "_grip_commands", None) or {}).get(arm)
    if last is None:
        return measured
    if last <= 0.05:  # last told to close
        return 0.0 if measured < 0.95 else measured
    if measured > last + 0.08:  # opened further in between: follow it
        return measured
    return last


def _spent(status: dict[str, Any]) -> bool:
    try:
        return int(status.get("step", 0)) >= int(status.get("step_limit", 1 << 30))
    except (TypeError, ValueError):
        return False


def _tcp(arm: str, q: Sequence[float]) -> list[float]:
    return [round(float(v), 4) for v in kin.tcp(kin.fk(arm, q))]


def _pose_summary(arm: str, q: Sequence[float]) -> dict[str, Any]:
    pose = kin.fk(arm, q)
    pitch, yaw, roll = kin.angles(pose[:3, :3])
    return {
        "xyz": [round(float(v), 4) for v in kin.tcp(pose)],
        "pitch": round(pitch, 1),
        "yaw": round(yaw, 1),
        "roll": round(roll, 1),
    }


# ---------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------


def _target(arm: str, q: Sequence[float], xyz, pitch, yaw, roll) -> tuple[np.ndarray, tuple]:
    cur = kin.angles(kin.fk(arm, q)[:3, :3])
    want = tuple(
        float(v) if v is not None else c for v, c in zip((pitch, yaw, roll), cur, strict=True)
    )
    t = np.eye(4)
    t[:3, :3] = kin.orientation(*want)
    t[:3, 3] = [float(v) for v in xyz]
    return t, want


def _joint_path_clear(arm: str, q0: Sequence[float], q1: Sequence[float]) -> bool:
    a, b = np.asarray(q0, float), np.asarray(q1, float)
    return all(
        not kin.clearance_violations(arm, a + (b - a) * t) for t in np.linspace(0, 1, 11)[1:]
    )


def _pose_of(arm: str, q: Sequence[float]) -> np.ndarray:
    """The grasp-point pose (rotation of link6, position of the grasp point) of joints ``q``."""

    t = kin.fk(arm, q)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3]
    out[:3, 3] = kin.tcp(t)
    return out


def _plan(
    arm: str,
    q: Sequence[float],
    target: np.ndarray,
    path: str,
    start_joints,
    step_rad: float | None = None,
) -> tuple[list, str | None, str | None, str | None]:
    """(joint waypoints, the path taken, why nothing can be done, a note) for one target pose.

    The pose itself is solved first, so a refusal says whether the TARGET is out of reach or
    only the straight line to it.  A target missed by at most NEAR_M / NEAR_DEG is replaced by
    the closest reachable pose, and the note says so.
    """

    res = kin.best_solution(arm, target, q, extra_seeds=[start_joints[arm]])
    near = res.pos_err_m <= NEAR_M and res.rot_err_deg <= NEAR_DEG
    if not res.ok and not near:
        return [], None, _unreachable(res), None
    if res.collides:
        return [], None, "every solution puts part of the arm into the table", None
    note = None
    goal = target
    if not res.ok:
        goal = _pose_of(arm, res.q)
        note = (
            f"the exact pose is out of reach; went to the closest one, {res.pos_err_m:.3f} m "
            f"and {res.rot_err_deg:.0f} deg from it"
        )
    if path in ("line", "auto"):
        ways, why = kin.line_path(arm, q, goal)
        if why is None:
            return _followable(q, ways, step_rad), "line", None, note
        if path == "line":
            return [], "line", why + " (the target itself is reachable: path='auto' swings)", None
    if not _joint_path_clear(arm, q, res.q):
        return [], "joint", "the joint-space swing to this pose passes through the table", None
    return _followable(q, [res.q], step_rad), "joint", None, note


def _unreachable(res: kin.IKResult) -> str:
    return (
        f"out of reach: the closest pose the arm can take is {res.pos_err_m:.3f} m and "
        f"{res.rot_err_deg:.0f} deg away"
    )


def _alternatives(
    arm: str, q: Sequence[float], xyz: Sequence[float], want: tuple, start_joints
) -> list[str]:
    """Nearby poses that ARE reachable: a less steep pitch here, or this pose higher / lower."""

    def ok(p: tuple[float, float, float], angles: tuple[float, float, float]) -> bool:
        t = np.eye(4)
        t[:3, :3] = kin.orientation(*angles)
        t[:3, 3] = p
        res = kin.best_solution(arm, t, q, extra_seeds=[start_joints[arm]])
        return res.ok and not res.collides

    out = []
    pitch, yaw, roll = want
    for p in (60.0, 45.0, 30.0, 0.0):
        if p < pitch - 5.0 and ok(tuple(xyz), (p, yaw, roll)):
            out.append(f"pitch={p:.0f} at this position")
            break
    for dz in (-0.03, -0.06, -0.10, 0.03, 0.06):
        z = xyz[2] + dz
        if z > kin.TABLE_Z + 0.015 and ok((xyz[0], xyz[1], z), want):
            out.append(f"z={z:.3f} with this orientation")
            break
    return out


def _smooth_ends(ways: Sequence[Sequence[float]], q: Sequence[float]) -> list[list[float]]:
    """Split the first and last joint step of ``ways`` (ease in / out for a carry)."""

    out: list[list[float]] = []
    prev = np.asarray(q, float)
    last = len(ways) - 1
    for i, way in enumerate(ways):
        b = np.asarray(way, float)
        if i == 0:
            out.extend((prev + (b - prev) * 0.35, prev + (b - prev) * 0.7, b))
        elif i == last:
            out.extend((prev + (b - prev) * 0.5, prev + (b - prev) * 0.8, b))
        else:
            out.append(b)
        prev = b
    return out


def _followable(
    q: Sequence[float], ways: Sequence[Sequence[float]], step_rad: float | None = None
) -> list[list[float]]:
    """Waypoints split into steps the simulator can follow in one action each.

    It turns a joint about 0.2 rad per action; an arm that then falls behind its waypoints is
    one that something stopped.  ``step_rad`` overrides ``JOINT_STEP_RAD`` (carries use the
    finer ``CARRY_JOINT_STEP_RAD``).
    """

    step = float(step_rad) if step_rad is not None else JOINT_STEP_RAD
    out: list[list[float]] = []
    a = np.asarray(q, float)
    for way in ways:
        b = np.asarray(way, float)
        n = max(1, math.ceil(float(np.max(np.abs(b - a))) / step))
        out.extend(list(a + (b - a) * (i / n)) for i in range(1, n + 1))
        a = b
    return out


def move_pose(
    primitives: Any,
    start_joints: dict[str, list[float]],
    xyz: Sequence[float],
    arm: str = "right",
    gripper: float = 0,
    pitch: float | None = None,
    yaw: float | None = None,
    roll: float | None = None,
    path: str = "auto",
    half_turn: bool = True,
    smooth: bool = False,
) -> dict[str, Any]:
    """Move one arm's grasp point to ``xyz`` and an orientation: the tool behind ``move_to``.

    ``pitch`` / ``yaw`` / ``roll`` in degrees (:mod:`.litearm_kinematics`); an angle left None
    keeps its current value, so a plain position move no longer turns the wrist.  The two
    fingers are interchangeable, so the target turned half a turn about the pointing axis is
    tried too and the one needing less joint travel is taken.  ``path``: ``"line"`` moves the
    grasp point along a straight line (waypoints every 2.5 cm or 8 degrees, each solved from the
    last), ``"joint"`` swings the joints straight to the solution, ``"auto"`` takes the line and
    falls back to the swing when the line cannot be followed.  Nothing moves when the target is
    out of reach or the motion would take the arm into the table: the result says why.
    ``half_turn=False`` pins the pose exactly (``place_on`` uses it after the release, so an
    empty lift does not turn the wrist over the block just put down).  ``smooth=True`` splits
    the first and last joint steps into smaller ones, so a carried object sees a gentler
    start and stop than the uniform ``CARRY_JOINT_STEP_RAD`` steps give.
    """

    if arm not in ARMS:
        return {"error": "arm must be 'left' or 'right'"}
    if path not in ("auto", "line", "joint"):
        return {"error": "path must be 'auto', 'line' or 'joint'"}
    try:
        xyz = [float(v) for v in xyz]
    except (TypeError, ValueError):
        return {"error": "xyz must be three numbers"}
    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
        return {"error": "xyz must be three finite numbers"}
    grip = _command(gripper)
    state = _obs(primitives)["state"]
    q = _joints(state, arm)
    start = _tcp(arm, q)
    target, want = _target(arm, q, xyz, pitch, yaw, roll)
    flipped, _ = _target(arm, q, xyz, want[0], want[1], want[2] + 180.0)
    # The fingers are interchangeable only while they hold nothing: with an object in the hand
    # the half-turned pose turns the object (172 of 3373 moves made holding something arrived
    # turned 180 degrees: pours tipped the wrong way, a laptop stood up upside down).
    empty = _opening(state, arm) >= OPEN_FOR_HALF_TURN and grip != 0.0
    # A carried object is measured against the opening its settled close left, not
    # against the reading at the start of this segment: the close baseline is kept
    # across every move_to (CLOSE or KEEP).  An open motion releases it.
    held = _held_openings(primitives)
    stored = (held or {}).get(arm)
    # The guard is for a carry after a settled close.  Without that close (a
    # move that commands CLOSE while still empty, as a probe does) there is no
    # held object to watch, so the close itself is not read as a slip.
    carrying = (
        grip != 1.0
        and stored is not None
        and stored >= HOLD_MIN_OPENING
    )
    baseline: float | None = stored if carrying else None
    if grip == 1.0 and held is not None:
        held.pop(arm, None)
    options = []
    first_why: str | None = None
    for t in (target, flipped) if empty and half_turn else (target,):
        ways, taken, why, note = _plan(
            arm,
            q,
            t,
            path,
            start_joints,
            step_rad=CARRY_JOINT_STEP_RAD if carrying else None,
        )
        if why is None:
            travel = float(np.sum(np.abs(np.asarray(ways[-1]) - np.asarray(q))))
            # an exact solution beats a near miss
            options.append((note is not None, travel, ways, taken, t, note))
        elif first_why is None or first_why.startswith("out of reach"):
            first_why = why
    result: dict[str, Any] = {
        "arm": arm,
        "target_xyz": [round(v, 4) for v in xyz],
        "target_angles": {
            "pitch": round(want[0], 1),
            "yaw": round(want[1], 1),
            "roll": round(want[2], 1),
        },
        "start_xyz": start,
        "steps_used": 0,
        "path": None,
    }
    if not options:
        if first_why and first_why.startswith("out of reach"):
            alts = _alternatives(arm, q, xyz, want, start_joints)
            first_why += (
                "; reachable instead: " + ", or ".join(alts)
                if alts
                else "; nothing close is reachable either: use the other arm or move the object"
            )
        now = _pose_summary(arm, q)
        result.update(
            final_xyz=now["xyz"],
            final_angles={k: now[k] for k in ("pitch", "yaw", "roll")},
            dist_to_target_m=round(float(np.linalg.norm(np.asarray(now["xyz"]) - xyz)), 4),
            orientation_error_deg=round(
                kin.rotation_error(kin.fk(arm, q)[:3, :3], target[:3, :3]), 1
            ),
            reached=False,
            ik_error=first_why,
        )
        return result
    slip = False
    slip_opening: float | None = None
    if carrying and baseline is not None and _opening(state, arm) < baseline - CARRY_SLIP_DROP:
        slip_opening = _opening(state, arm)
        now = _pose_summary(arm, q)
        result.update(
            final_xyz=now["xyz"],
            final_angles={k: now[k] for k in ("pitch", "yaw", "roll")},
            dist_to_target_m=round(float(np.linalg.norm(np.asarray(now["xyz"]) - xyz)), 4),
            orientation_error_deg=round(
                kin.rotation_error(kin.fk(arm, q)[:3, :3], target[:3, :3]), 1
            ),
            reached=False,
            settled=False,
            steps_used=0,
            ik_error=None,
            slip=True,
            slip_opening=round(slip_opening, 4),
            slip_baseline=round(baseline, 4),
        )
        result["note"] = (
            "the held object has already slipped: opening fell from "
            f"{baseline:.3f} to {slip_opening:.3f}; nothing was moved"
        )
        return result
    _near, _travel, ways, taken, goal, note = min(options, key=lambda o: (o[0], o[1]))
    result["path"] = taken
    if goal is flipped:
        half = (
            "took the pose turned half a turn about the pointing direction (the empty "
            "fingers are interchangeable): yaw / roll read 180 off"
        )
        note = f"{note}; {half}" if note else half
    if note:
        result["note"] = note
    if smooth and len(ways) >= 2:
        ways = _smooth_ends(ways, q)
    steps = 0
    status: dict[str, Any] = {}
    for way in ways:
        primitives._check_cancelled()
        state, status = _act(primitives, state, {arm: way}, {arm: grip})
        steps += 1
        if carrying and baseline is not None:
            now_open = _opening(state, arm)
            if now_open < baseline - CARRY_SLIP_DROP:
                slip = True
                slip_opening = now_open
                break
        if _spent(status):
            break
    final_q = ways[-1]
    planned = np.asarray(kin.tcp(kin.fk(arm, final_q)))
    last_xyz = np.asarray(kin.tcp(kin.fk(arm, _joints(state, arm))))
    still_actions = 0
    settled = False
    for _ in range(SETTLE_ACTIONS):
        # A slip in the waypoint loop must not send another target command.
        if slip or _spent(status):
            break
        primitives._check_cancelled()
        state, status = _act(primitives, state, {arm: final_q}, {arm: grip})
        steps += 1
        if carrying and baseline is not None:
            now_open = _opening(state, arm)
            if now_open < baseline - CARRY_SLIP_DROP:
                slip = True
                slip_opening = now_open
                break
        measured_q = _joints(state, arm)
        measured_xyz = np.asarray(kin.tcp(kin.fk(arm, measured_q)))
        err = max(abs(a - b) for a, b in zip(measured_q, final_q, strict=True))
        at_target = err < 0.01 and np.linalg.norm(measured_xyz - planned) <= 0.005
        still = np.linalg.norm(measured_xyz - last_xyz) < 0.001
        still_actions = still_actions + 1 if at_target and still else 0
        last_xyz = measured_xyz
        if still_actions >= 2:
            settled = True
            break
    q_now = _joints(state, arm)
    now = _pose_summary(arm, q_now)
    dist = float(np.linalg.norm(np.asarray(kin.tcp(kin.fk(arm, q_now))) - np.asarray(xyz)))
    turn = kin.rotation_error(kin.fk(arm, q_now)[:3, :3], goal[:3, :3])
    result.update(
        final_xyz=now["xyz"],
        final_angles={k: now[k] for k in ("pitch", "yaw", "roll")},
        dist_to_target_m=round(dist, 4),
        orientation_error_deg=round(turn, 1),
        reached=bool(settled and dist <= ARRIVE_M and turn <= ARRIVE_DEG and not slip),
        settled=bool(settled and not slip),
        steps_used=steps,
        ik_error=None,
        slip=bool(slip),
        slip_opening=round(slip_opening, 4) if slip_opening is not None else None,
        slip_baseline=round(baseline, 4) if baseline is not None else None,
    )
    if slip and slip_opening is not None and baseline is not None:
        slip_note = (
            "the held object slipped during the move: opening fell from "
            f"{baseline:.3f} to {slip_opening:.3f}; the arm stopped and the gripper "
            "keeps its last command"
        )
        result["note"] = f"{result['note']}; {slip_note}" if result.get("note") else slip_note
    # behind the planned pose by more than a few mm: a joint lag of 0.03 rad is already ~1 cm
    # at the grasp point (a press stalled 1.5 cm short read as no lag at all)
    lag_m = float(np.linalg.norm(np.asarray(kin.tcp(kin.fk(arm, q_now))) - planned))
    if lag_m > 0.005 and not _spent(status) and not slip:
        result["blocked"] = "the arm stopped short of the planned pose: something is in the way"
    return result


def grip(
    primitives: Any, arm: str = "right", gripper: float = 1, opening: float | None = None
) -> dict[str, Any]:
    """Close (1) or open (-1) one gripper: repeat the command until the fingers stop.

    ``opening`` (0..1, with an open command only) opens to that reading instead of all the
    way, e.g. to reach between close neighbours.
    """

    if arm not in ARMS:
        return {"error": "arm must be 'left' or 'right'"}
    command = _command(gripper)
    if command is None:
        return {"error": "gripper must be 1 (close) or -1 (open)"}
    if opening is not None:
        try:
            opening = float(opening)
        except (TypeError, ValueError):
            return {"error": "opening must be a number between 0 and 1"}
        if command < 0.5 or not 0.0 < opening <= 1.0:
            return {"error": "opening is for an open command, between 0 (excluded) and 1"}
        command = opening
    state = _obs(primitives)["state"]
    hold = {a: _joints(state, a) for a in ARMS}  # both arms keep the joints they had
    before = _opening(state, arm)
    steps = 0
    status: dict[str, Any] = {}
    last = before
    still_actions = 0
    settled = False
    for _ in range(GRIP_MAX_ACTIONS):
        primitives._check_cancelled()
        state, status = _act(primitives, state, hold, {arm: command})
        steps += 1
        now = _opening(state, arm)
        still_actions = still_actions + 1 if abs(now - last) < GRIP_STILL else 0
        if still_actions >= GRIP_STILL_ACTIONS:
            settled = True
            break
        last = now
        if _spent(status):
            break
    measured = _opening(state, arm)
    closing = command == 0.0
    # A budget/iteration limit can interrupt the fingers mid-stroke. An open
    # reading alone then says nothing about whether an object stopped closure.
    holding = bool(closing and settled and measured > HOLD_MIN_OPENING)
    width = max(0.0, GRIP_ZERO_M + GRIP_SPAN_M * measured) if holding else 0.0
    # The close's stable opening is the baseline every later carry segment is measured
    # against; an open (including a partial one) means nothing is held any more.
    held = _held_openings(primitives)
    if held is not None:
        if closing and holding:
            held[arm] = measured
        else:
            held.pop(arm, None)
    return {
        "arm": arm,
        "gripper": "closed" if closing else "open",
        "opening": round(measured, 3),
        "opening_before": round(before, 3),
        "holding": holding,
        "settled": settled,
        "width_m": round(width, 4) if closing else None,
        "steps_used": steps,
        "status": {k: status[k] for k in ("step", "step_limit") if k in status},
    }


def place_on(
    primitives: Any,
    start_joints: dict[str, list[float]],
    x: float,
    y: float,
    support_top: float,
    tcp_to_bottom: float,
    arm: str = "right",
    clearance: float = 0.002,
    lift_m: float = 0.12,
    retreat_xy: Sequence[float] = (0.0, -0.10),
    opening: float | None = None,
    path: str = "line",
) -> dict[str, Any]:
    """Put a held object on a support and back the arm clear before any home.

    ``support_top`` is the support's top z (metres), measured before the pick or
    from the head camera; ``tcp_to_bottom`` is the distance from the grasp point
    to the held object's BOTTOM for this grasp.  It is not a universal constant:
    the current stack block hangs 0.050 m below the grasp point, another object
    or grasp differs.  The descent target is ``support_top + tcp_to_bottom +
    clearance``.

    The gripper is opened only after that descent reached the target and was not
    blocked or slipping; the arm then rises ``lift_m`` and moves ``retreat_xy``
    away with the fingers kept open.  The tool never calls ``home``: the result's
    ``safe_to_home`` says whether the retreat finished, and the caller decides
    when to home.  A blocked descent leaves the gripper closed and the object
    held.
    """

    if arm not in ARMS:
        return {"error": "arm must be 'left' or 'right'"}
    try:
        x = float(x)
        y = float(y)
        support_top = float(support_top)
        tcp_to_bottom = float(tcp_to_bottom)
        clearance = float(clearance)
        lift_m = float(lift_m)
        retreat = [float(v) for v in retreat_xy]
    except (TypeError, ValueError):
        return {"error": "x, y, support_top, tcp_to_bottom, clearance, lift_m and retreat_xy must be numbers"}
    if not all(math.isfinite(v) for v in (x, y, support_top, tcp_to_bottom, clearance, lift_m)):
        return {"error": "place_on arguments must be finite"}
    if len(retreat) != 2 or not all(math.isfinite(v) for v in retreat):
        return {"error": "retreat_xy must be two finite numbers"}
    if tcp_to_bottom <= 0.0:
        return {"error": "tcp_to_bottom must be positive (the grasp point is above the object's bottom)"}
    if clearance < 0.0 or lift_m < 0.0:
        return {"error": "clearance and lift_m must not be negative"}
    if path not in ("auto", "line", "joint"):
        return {"error": "path must be 'auto', 'line' or 'joint'"}

    place_z = support_top + tcp_to_bottom + clearance
    target = [round(x, 4), round(y, 4), round(place_z, 4)]
    # Keep the carry orientation explicitly for all three moves: after the open
    # the fingers are empty, and an unqualified move may take the half turn.
    state0 = _obs(primitives)["state"]
    angles = kin.angles(kin.fk(arm, _joints(state0, arm))[:3, :3])
    orientation = dict(pitch=angles[0], yaw=angles[1], roll=angles[2])
    descent = move_pose(
        primitives,
        start_joints,
        [x, y, place_z],
        arm=arm,
        gripper=0,
        path=path,
        half_turn=False,
        **orientation,
    )
    result: dict[str, Any] = {
        "arm": arm,
        "placed": False,
        "released": False,
        "safe_to_home": False,
        "target_xyz": target,
        "place_z": round(place_z, 4),
        "descent_reached": bool(descent.get("reached")),
        "descent_blocked": bool(descent.get("blocked")),
        "descent_slip": bool(descent.get("slip")),
        "open_opening": None,
        "lift_reached": False,
        "lift_blocked": False,
        "retreat_reached": False,
        "retreat_blocked": False,
        "steps_used": int(descent.get("steps_used", 0)),
        "status": descent.get("status") or {},
        "truncated": bool(descent.get("truncated")),
    }
    if not descent.get("reached") or descent.get("blocked") or descent.get("slip"):
        result["note"] = (
            descent.get("note")
            or descent.get("ik_error")
            or "the descent did not reach the place pose; nothing was opened"
        )
        return result

    opened = grip(primitives, arm=arm, gripper=-1, opening=opening)
    result["steps_used"] += int(opened.get("steps_used", 0))
    result["open_opening"] = opened.get("opening")
    released = bool(opened.get("settled")) and not opened.get("error")
    result["released"] = released
    if not released:
        result["note"] = (
            opened.get("error") or "the gripper did not finish opening; the object was not released"
        )
        return result
    result["placed"] = True

    lift = move_pose(
        primitives,
        start_joints,
        [x, y, place_z + lift_m],
        arm=arm,
        gripper=0,
        path=path,
        half_turn=False,
        **orientation,
    )
    result["steps_used"] += int(lift.get("steps_used", 0))
    result["lift_reached"] = bool(lift.get("reached"))
    result["lift_blocked"] = bool(lift.get("blocked"))
    if not lift.get("reached") or lift.get("blocked") or lift.get("slip"):
        result["note"] = (
            lift.get("note")
            or lift.get("ik_error")
            or "the lift after opening did not finish; do not home"
        )
        return result

    retreat = move_pose(
        primitives,
        start_joints,
        [x + retreat[0], y + retreat[1], place_z + lift_m],
        arm=arm,
        gripper=0,
        path=path,
        half_turn=False,
        **orientation,
    )
    result["steps_used"] += int(retreat.get("steps_used", 0))
    result["retreat_reached"] = bool(retreat.get("reached"))
    result["retreat_blocked"] = bool(retreat.get("blocked"))
    result["safe_to_home"] = bool(
        retreat.get("reached") and not retreat.get("blocked") and not retreat.get("slip")
    )
    if not result["safe_to_home"]:
        result["note"] = (
            retreat.get("note")
            or retreat.get("ik_error")
            or "the retreat did not finish; do not home"
        )
    return result


def home(
    primitives: Any,
    start_joints: dict[str, list[float]],
    arm: str = "both",
    tol: float = HOME_TOL_RAD,
    max_steps: int = HOME_MAX_ACTIONS,
    lift: bool = True,
) -> dict[str, Any]:
    """Drive one arm or both back to ``start_joints``: the tool behind ``robo.home``.

    With ``lift``, an arm away from its start whose grasp point is below HOME_LIFT_BELOW first
    goes straight up by HOME_LIFT_M (or less, where the line ends sooner), orientation kept.

    The same joint action as ``move_pose``, the other arm held where it is and each gripper
    commanded its measured opening, with the joint target given instead of solved for.  The
    simulator moves a joint about 0.2 rad in one action: a return from a short reach took 5,
    and of 135 returns in one sweep half were still under way after 10 (the limit then).
    """

    if arm not in (*ARMS, "both"):
        return {"error": "arm must be 'left', 'right' or 'both'"}
    arms = ARMS if arm == "both" else (arm,)

    def error_of(state: dict[str, Any]) -> float:
        gaps = [0.0]
        for name in arms:
            gaps += [
                abs(a - b) for a, b in zip(_joints(state, name), start_joints[name], strict=True)
            ]
        return max(gaps)

    state = _obs(primitives)["state"]
    error = error_of(state)
    steps_used = 0
    status: dict[str, Any] = {}
    lifted: dict[str, float] = {}
    for name in arms if lift and error > tol else ():
        q = _joints(state, name)
        if max(abs(a - b) for a, b in zip(q, start_joints[name], strict=True)) <= tol:
            continue
        pose = _pose_of(name, q)
        if pose[2, 3] >= HOME_LIFT_BELOW:
            continue
        for dz in (HOME_LIFT_M, HOME_LIFT_M * 2 / 3, HOME_LIFT_M / 3):
            up = pose.copy()
            up[2, 3] += dz
            ways, why = kin.line_path(name, q, up)
            if why is None:
                for way in _followable(q, ways):
                    primitives._check_cancelled()
                    state, status = _act(primitives, state, {name: way}, {})
                    steps_used += 1
                    if _spent(status):
                        break
                lifted[name] = round(dz, 3)
                break
        if _spent(status):
            break
    error = error_of(state)
    swing = 0
    while error > tol and swing < max_steps and not _spent(status):
        primitives._check_cancelled()
        state, status = _act(primitives, state, {name: start_joints[name] for name in arms}, {})
        steps_used += 1
        swing += 1
        error = error_of(state)
    out = {
        "arm": arm,
        "reached": error <= tol,
        "max_joint_error_rad": round(error, 4),
        "steps_used": steps_used,
    }
    if lifted:
        out["lifted_m"] = lifted
    return out


# ---------------------------------------------------------------------------
# perception
# ---------------------------------------------------------------------------


def _on_robot(state: dict[str, Any], points: np.ndarray, radius: float = 0.045) -> np.ndarray:
    """Which world points lie on either arm: within ``radius`` of its links (base to the tips),
    on the rail of its gripper or on one of its fingers, wherever those stand open."""

    hit = np.zeros(len(points), dtype=bool)

    def near(a: np.ndarray, b: np.ndarray, within: float) -> np.ndarray:
        ab = b - a
        n = float(ab @ ab)
        t = np.clip(((points - a) @ ab) / n, 0.0, 1.0) if n > 1e-12 else np.zeros(len(points))
        return np.linalg.norm(points - (a + t[:, None] * ab), axis=1) < within

    for arm in ARMS:
        try:
            q = _joints(state, arm)
        except (KeyError, TypeError, ValueError):
            continue
        if len(q) != 6:
            continue
        fr = kin.frames(arm, q)
        chain = [f[:3, 3] for f in fr]
        chain.append(fr[-1][:3, 3] + fr[-1][:3, 0] * kin.TIP_OFFSET)
        for a, b in zip(chain, chain[1:], strict=False):
            hit |= near(a, b, radius)
        origin, forward, side = fr[-1][:3, 3], fr[-1][:3, 0], fr[-1][:3, 1]
        rail = origin + forward * RAIL_AT_M
        hit |= near(rail - side * RAIL_HALF_M, rail + side * RAIL_HALF_M, RAIL_RADIUS_M)
        opening = min(max(_opening(state, arm), 0.0), 1.0)
        for sign in (1.0, -1.0):
            off = side * sign * (FINGER_CLOSED_M + FINGER_TRAVEL_M * opening)
            hit |= near(
                origin + forward * FINGER_FROM_M + off,
                origin + forward * kin.TIP_OFFSET + off,
                FINGER_RADIUS_M,
            )
    return hit


def _camera(obs: dict[str, Any], camera: str) -> dict[str, Any] | None:
    return (obs.get("vision") or {}).get(camera)


def depth_image(cam: dict[str, Any]) -> np.ndarray | None:
    """A camera's depth along its optical axis, metres (H x W), or None when it has none.

    RoboDojo's simulator gives it as ``depth`` (float metres; RPent's bridge calls the same
    array ``distance_to_image_plane``) or as ``approximate_depth`` (whole millimetres, the
    form its released data has), whichever its configuration switches on.
    """

    for key, scale in (
        ("distance_to_image_plane", 1.0),
        ("depth", 1.0),
        ("approximate_depth", 1e-3),
    ):
        raw = cam.get(key)
        if raw is None:
            continue
        depth = np.asarray(raw)
        scale = 1e-3 if depth.dtype.kind in "iu" else scale  # integers are millimetres
        depth = depth.astype(np.float64) * scale
        return depth[..., 0] if depth.ndim == 3 else depth
    return None


def extrinsic_of(cam: dict[str, Any]) -> Any:
    """A camera's pose in the world (4 x 4): RoboDojo names it ``extrinsic_matrix``,
    XPolicyLab's observation format ``extrinsics_matrix``."""

    pose = cam.get("extrinsic_matrix")
    return cam.get("extrinsics_matrix") if pose is None else pose


def world_points(
    cam: dict[str, Any], rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World points of pixels through the depth image, and which pixels had a valid depth.

    The same projection as RPent's ``back_project``: Isaac cameras look along -Z and the
    depth (``distance_to_image_plane``) is along the optical axis.
    """

    depth = depth_image(cam)
    if depth is None:
        raise KeyError("no depth image in the observation")
    k = np.asarray(cam["intrinsic_matrix"], dtype=np.float64)
    t = np.asarray(extrinsic_of(cam), dtype=np.float64)
    d = depth[rows, cols]
    valid = np.isfinite(d) & (d > 0)
    r, c, d = rows[valid], cols[valid], d[valid]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    p = np.stack([(c - cx) / fx * d, -(r - cy) / fy * d, -d, np.ones_like(d)])
    return (t @ p)[:3].T, valid


def _cells(xy: np.ndarray, cell: float = FOOTPRINT_CELL_M) -> np.ndarray:
    """One point per occupied grid cell (the mean of the points in it): the footprint with
    the same weight everywhere."""

    _, cell_of, count = np.unique(
        np.floor(xy / cell).astype(np.int64), axis=0, return_inverse=True, return_counts=True
    )
    cell_of = cell_of.reshape(-1)
    sums = np.stack([np.bincount(cell_of, weights=xy[:, i]) for i in (0, 1)], axis=1)
    return sums / count[:, None]


def _footprint(xy: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """A footprint's rectangle: the heading of its long side (degrees from +x, -90 .. 90), its
    length and width, and its centre.

    Every occupied cell counts once, so the answer does not depend on how densely each part
    of the object was seen.  A footprint clearly longer one way lies along its principal
    axes (a T, a hammer, a shoe: their smallest rectangle can sit askew).  A square or a
    circle has no principal axis: there the rectangle of least area over all headings is
    taken, which finds the sides of a square.
    """

    cells = _cells(xy)
    trim = [FOOTPRINT_TRIM, 100.0 - FOOTPRINT_TRIM]

    def extents(deg: float) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
        a = math.radians(deg)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        (ul, uh), (vl, vh) = np.percentile(cells @ u, trim), np.percentile(cells @ v, trim)
        return float((uh - ul) * (vh - vl)), ul, uh, vl, vh, u, v

    spread, axes = np.linalg.eigh(np.cov((cells - cells.mean(axis=0)).T))
    if spread[1] >= LONG_FOOTPRINT * max(float(spread[0]), 1e-12):
        heading = math.degrees(math.atan2(axes[1, 1], axes[0, 1]))
    else:
        coarse = min(np.arange(0.0, 90.0, 2.0), key=lambda d: extents(d)[0])
        heading = min(np.arange(coarse - 2.0, coarse + 2.01, 0.25), key=lambda d: extents(d)[0])
    _, ul, uh, vl, vh, u, v = extents(heading)
    centre = u * (ul + uh) / 2 + v * (vl + vh) / 2
    length, width = uh - ul, vh - vl
    if width > length:
        length, width, heading = width, length, heading + 90.0
    return (heading + 90.0) % 180.0 - 90.0, float(length), float(width), centre


def _far_side(pts: np.ndarray) -> np.ndarray:
    """Footprint points (N x 2) of what the camera cannot see of a round object.

    A camera sees the near half of an upright bottle, cup or doll, so its visible points end
    at the middle and their footprint sits towards the camera: against the scenes' true poses
    the centre of an upright bottle was 6 to 14 mm off and its far edge up to 2 cm short.
    Each horizontal slice whose visible points lie on a circular arc is completed to the full
    circle, provided the circles of the object share one upright axis.  Flat-sided objects are
    left alone: their slices are not arcs, and their tops, which the camera does see, cover
    them.  So is a rounded body that is not round about an upright axis.
    """

    circles = []
    for low in np.arange(kin.TABLE_Z + 0.004, float(pts[:, 2].max()), SLICE_M):
        xy = pts[(pts[:, 2] >= low) & (pts[:, 2] < low + SLICE_M), :2]
        if len(xy) < 15:
            continue
        # algebraic circle fit: |p|^2 = 2 c.p + (r^2 - |c|^2)
        sol, *_ = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy**2).sum(axis=1), rcond=None)
        centre = sol[:2]
        radius = math.sqrt(max(float(sol[2] + centre @ centre), 0.0))
        if not ROUND_MIN_RADIUS_M <= radius <= ROUND_MAX_RADIUS_M:
            continue
        rms = float(np.sqrt(((np.linalg.norm(xy - centre, axis=1) - radius) ** 2).mean()))
        angles = np.sort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
        arc = 360.0 - math.degrees(float(gaps.max()))  # what the points cover of the circle
        if rms <= ROUND_FIT * radius and arc >= ROUND_MIN_ARC_DEG:
            circles.append((centre, radius))
    if len(circles) >= 3:
        # The slices of a round object share an axis: a circle off it is a misfit.  And when no
        # more than half of them share one, the object is not round about an upright axis: the
        # body of a piggy bank lying on its feet gave four circles, two of them alike, and the
        # two put 2 cm on its width.
        axis = np.median(np.array([c for c, _ in circles]), axis=0)
        alike = [(c, r) for c, r in circles if np.linalg.norm(c - axis) <= ROUND_AXIS_M]
        circles = alike if 2 * len(alike) > len(circles) else []
    turn = np.linspace(0.0, 2 * math.pi, 90, endpoint=False)
    ring = np.c_[np.cos(turn), np.sin(turn)]
    return np.vstack([c + r * ring for c, r in circles]) if circles else np.zeros((0, 2))


def object_geometry(points: np.ndarray) -> dict[str, Any] | None:
    """Extent, heading and heights of an object from its visible surface points (N x 3)."""

    pts = points[points[:, 2] > kin.TABLE_Z + 0.004]  # the table seen through gaps in the mask
    if len(pts) < 12:
        return None
    med = np.median(pts[:, :2], axis=0)
    spread = np.median(np.linalg.norm(pts[:, :2] - med, axis=1))
    keep = np.linalg.norm(pts[:, :2] - med, axis=1) <= max(4.0 * spread, 0.03)
    pts = pts[keep]  # pixels of other things at the mask's edge
    top = float(np.percentile(pts[:, 2], 97))
    low = float(pts[:, 2].min())
    standing = pts[pts[:, 2] > low + FOOT_M] if top - low > FOOT_MIN_HEIGHT_M else pts
    heading, length, width, mid = _footprint(np.vstack([standing[:, :2], _far_side(pts)]))
    across_heading = (heading + 180.0) % 180.0 - 90.0
    return {
        "center_xyz": [
            round(float(mid[0]), 4),
            round(float(mid[1]), 4),
            round((top + kin.TABLE_Z) / 2, 4),
        ],
        "top_z": round(top, 4),
        "height_m": round(top - kin.TABLE_Z, 4),
        "length_m": round(length, 4),
        "width_m": round(width, 4),
        "long_axis_deg": round(heading, 1),
        # pointing straight down, this yaw closes the fingers across the narrow side
        "grasp_yaw": round(across_heading, 1),
        "points": int(len(pts)),
    }


# ---------------------------------------------------------------------------
# perception without a depth image (RoboDojo's evaluation client sends none by default)
# ---------------------------------------------------------------------------

_WRISTS = {"cam_left_wrist": "left", "cam_right_wrist": "right"}


def colour_only(primitives: Any, cam: dict[str, Any] | None = None) -> bool:
    """Whether perception works from colour images alone.

    Always when the host says so (``primitives.colour_only``, which the host for RoboDojo's
    own evaluation client sets), and whenever the observation simply carries no depth image.
    """

    if getattr(primitives, "colour_only", False):
        return True
    if cam is None:
        return False
    return depth_image(cam) is None


def _shown(obs: dict[str, Any]) -> dict[str, Any]:
    """The robot state an observation's pictures show.

    RoboDojo's evaluation client before its commit 6e3190f hands out each picture one
    observation late: the state is the robot's now, the pictures are of the state before the
    last action (measured in that client: a wrist camera's picture fits the previous
    end-effector pose to a millimetre and the current one not at all while the arm moves).  A
    host that knows this puts the state the pictures belong to under ``image_state``; without
    it they are taken to be current.
    """

    shown = obs.get("image_state")
    return shown if isinstance(shown, dict) and shown else (obs.get("state") or {})


def _view(camera: str, state: dict[str, Any]) -> sight.View | None:
    """A camera's pose from the state alone: the head camera is fixed, a wrist camera rides on
    its arm's end-effector link."""

    if camera == "cam_head":
        return sight.head_view()
    arm = _WRISTS.get(camera)
    pose = state.get(f"{arm}_ee_pose") if arm else None
    if pose is None:
        return None
    return sight.wrist_view([float(v) for v in np.ravel(pose)[:7]])


def _scene(primitives: Any, obs: dict[str, Any]) -> dict[str, Any]:
    """What has been measured in the images of this observation (kept until they change)."""

    marks = []
    for name in sorted(obs.get("vision") or {}):
        color = (obs["vision"][name] or {}).get("color")
        if color is not None:
            marks.append(zlib.crc32(np.ascontiguousarray(np.asarray(color)[::8, ::8]).tobytes()))
    key = tuple(marks)
    scene = getattr(primitives, "_scene", None)
    if not isinstance(scene, dict) or scene.get("key") != key:
        scene = {"key": key, "regions": []}
        try:
            primitives._scene = scene
        except AttributeError:
            pass
    return scene


def _heights(primitives: Any) -> dict[str, tuple[float, str, float | None]]:
    known = getattr(primitives, "_measured_heights", None)
    if known is None:
        known = {}
        try:
            primitives._measured_heights = known
        except AttributeError:
            pass
    return known


def _behind_robot(mask: np.ndarray, outline: np.ndarray) -> bool:
    """Whether the robot's outline runs along the mask's edge: it then hides part of the object."""

    grown = mask.copy()
    for _ in range(RIM_PX):
        wider = grown.copy()
        wider[1:] |= grown[:-1]
        wider[:-1] |= grown[1:]
        wider[:, 1:] |= grown[:, :-1]
        wider[:, :-1] |= grown[:, 1:]
        grown = wider
    rim = grown & ~mask
    return bool(rim.any()) and float(outline[rim].mean()) >= HIDDEN_RIM_SHARE


def _measure_in_colour(
    primitives: Any,
    obs: dict[str, Any],
    camera: str,
    mask: np.ndarray,
    text_prompt: str | None,
    min_score: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """An object's geometry from its mask in ``camera`` and whatever the other cameras show of it.

    The other cameras are asked for the object twice: by the same words, and by the pixels
    where points of the first mask's rays fall in their image (:func:`.litearm_sight.seeds`);
    :func:`.litearm_sight.measure` takes of those masks the one that is the same object.
    Returns the geometry (``object_geometry`` of the surface points) and what was seen.
    """

    state = _shown(obs)
    vision = obs.get("vision") or {}
    seen: dict[str, Any] = {"views": 1}
    view = _view(camera, state)
    if view is None:
        return None, seen
    first = sight.Sight(view, mask)
    sam3 = getattr(primitives, "sam3_client", None)
    others = []
    spots = sight.seeds(first)
    for name in vision:
        if name == camera or sam3 is None:
            continue
        other = _view(name, state)
        color = (vision[name] or {}).get("color")
        if other is None or color is None:
            continue
        robot = sight.robot_pixels(other, state, own=_WRISTS.get(name))
        masks = []
        if text_prompt:
            res = sam3.segment(color, text_prompt=text_prompt, min_score=min_score)
            if res.found:
                masks.append(np.asarray(res.mask, dtype=bool))
        tried: list[tuple[int, int]] = []
        if len(spots):
            r, c, depth = other.project(spots)
            for ri, ci, di in zip(r, c, depth, strict=True):
                at = (int(round(float(ri))), int(round(float(ci))))
                inside = 0 <= at[0] < other.shape[0] and 0 <= at[1] < other.shape[1]
                if di <= 0.02 or not inside or robot[at]:
                    continue
                if any(abs(at[0] - a) + abs(at[1] - b) < SEED_APART_PX for a, b in tried):
                    continue
                if any(m[at] for m in masks):
                    continue  # the mask there is already a candidate
                tried.append(at)
                res = sam3.segment(color, point=[at[0], at[1]], min_score=min_score)
                if res.found:
                    masks.append(np.asarray(res.mask, dtype=bool))
        if masks:
            others.append([sight.Sight(other, m, robot) for m in masks])
    key = " ".join(str(text_prompt or "").lower().split())
    known = _heights(primitives).get(key) if key else None
    res = sight.measure(first, others, known=known)
    if res is None:
        return None, seen
    seen["views"] = int(res["views"])
    if res["views"] == 1 and not res.get("remembered"):
        # one camera, nothing to check it against: has the robot got in front of the object?
        outline = sight.robot_pixels(view, state, own=_WRISTS.get(camera), scale=1.0, between=False)
        if _behind_robot(np.asarray(mask, dtype=bool), outline):
            if res.get("outgrown"):
                # two cameras measured it taller than what shows now: the rest is behind the arm
                seen["hidden"] = (
                    "the robot's arm hides part of it now (move the arm away and locate again)"
                )
                return None, seen
            seen["guess"] = (
                "one camera sees it and the robot's arm is in front of part of it: its size and "
                "place may be off (move the arm away and locate again)"
            )
            points = np.asarray(res["points"], dtype=np.float64)
            return object_geometry(points), seen
    if res["views"] >= 2:
        rough = res["match"] is not None and res["match"] < ROUGH_MATCH
        if key and not rough:
            # only what the cameras agreed on well is kept: a rough reading (an object in the
            # gripper read as a column down to the table) would spoil every later one
            _heights(primitives)[key] = (float(res["height_m"]), str(res["kind"]), res["match"])
        if rough:
            seen["guess"] = (
                "the cameras agree only roughly: its height may be centimetres off "
                "(locate it again from above before a grasp)"
            )
    elif res.get("remembered"):
        seen["guess"] = "its height as measured before"
    else:
        seen["guess"] = (
            "one camera sees it: its height is a guess (locate it again from above before a grasp)"
        )
    points = np.asarray(res["points"], dtype=np.float64)
    _scene(primitives, obs)["regions"].append(
        {"camera": camera, "rows": res["rows"], "cols": res["cols"], "points": points}
    )
    return object_geometry(points), seen


def probe(primitives: Any, row: int, col: int, camera: str = "cam_head") -> dict[str, Any]:
    """The world point a pixel shows, without a depth image.

    The object under the pixel is segmented from that pixel and measured as ``locate`` measures
    one; the answer is where the pixel's ray meets it.  A pixel on nothing that stands out
    (the table, or something too flat to measure) is where its ray meets the table.
    """

    obs = _obs(primitives)
    cam = _camera(obs, camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    color = cam.get("color")
    if color is None:
        return {"error": "camera color image missing"}
    state = _shown(obs)
    view = _view(camera, state)
    if view is None:
        return {"error": f"no pose for camera {camera!r} in the state"}
    row, col = int(row), int(col)
    if not (0 <= row < view.shape[0] and 0 <= col < view.shape[1]):
        return {
            "error": f"pixel [{row}, {col}] is outside the {view.shape[0]} x {view.shape[1]} image"
        }

    def answer(xyz: np.ndarray, surface: str) -> dict[str, Any]:
        depth = float(-((xyz - view.pos) @ view.rot)[2])
        return {
            "camera": camera,
            "pixel": [row, col],
            "world_xyz": [round(float(v), 4) for v in xyz],
            "depth_m": round(depth, 4),
            "surface": surface,
        }

    def on_table() -> dict[str, Any]:
        xy, ok = view.on_plane(np.array([row]), np.array([col]), kin.TABLE_Z)
        x, y = float(xy[0, 0]), float(xy[0, 1])
        on = kin.TABLE_X[0] - 0.01 <= x <= kin.TABLE_X[1] + 0.01
        on = on and kin.TABLE_Y[0] - 0.01 <= y <= kin.TABLE_Y[1] + 0.01
        if not ok[0] or not on:
            return {"error": "the pixel looks past the table: there is no depth image to say where"}
        return answer(np.array([x, y, kin.TABLE_Z]), "table")

    def in_region(region: dict[str, Any]) -> dict[str, Any] | None:
        if region["camera"] != camera or len(region["rows"]) == 0:
            return None
        gap = (region["rows"] - row) ** 2 + (region["cols"] - col) ** 2
        i = int(np.argmin(gap))
        if gap[i] > PROBE_REACH_PX**2:
            return None
        if gap[i] == 0:
            return answer(region["points"][i], "object")
        # between measured pixels: this pixel's own ray, as far out as its neighbour's point
        d = view.rays(np.array([row]), np.array([col]))[0]
        depth = float(-((region["points"][i] - view.pos) @ view.rot)[2])
        return answer(view.pos + d * depth, "object")

    scene = _scene(primitives, obs)
    for region in scene["regions"]:
        hit = in_region(region)
        if hit is not None:
            return hit
    for flat in scene.get("flat", ()):
        if flat["camera"] == camera and flat["mask"][row, col]:
            return on_table()
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        return on_table()
    res = sam3.segment(color, point=[row, col], min_score=0.0)
    mask = np.asarray(res.mask, dtype=bool) if res.found else None
    if mask is None or not mask[row, col] or mask.sum() > PROBE_TABLE_SHARE * mask.size:
        if mask is not None and mask[row, col]:
            scene.setdefault("flat", []).append({"camera": camera, "mask": mask})
        return on_table()
    geo, _seen = _measure_in_colour(primitives, obs, camera, mask, None, 0.0)
    if geo is not None:
        hit = in_region(scene["regions"][-1])
        if hit is not None:
            return hit
    scene.setdefault("flat", []).append({"camera": camera, "mask": mask})
    return on_table()


probe._readonly = True  # type: ignore[attr-defined]


def back_project(primitives: Any, row: int, col: int, camera: str = "cam_head") -> dict[str, Any]:
    """The world point a pixel shows: through the depth image where there is one.

    With depth it is RPent's ``back_project`` (the same projection, the same answer and the
    same errors), for a host that runs without RPent; without depth it is :func:`probe`.
    """

    obs = _obs(primitives)
    cam = _camera(obs, camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    if colour_only(primitives, cam):
        return probe(primitives, row, col, camera)
    depth = depth_image(cam)
    k = np.asarray(cam.get("intrinsic_matrix"), dtype=np.float64)
    t = np.asarray(extrinsic_of(cam), dtype=np.float64)
    if k.shape != (3, 3) or t.shape != (4, 4):
        return {"error": f"calibration missing/invalid: K={k.shape} T={t.shape}"}
    h, w = int(depth.shape[0]), int(depth.shape[1])
    row, col = int(row), int(col)
    if not (0 <= row < h and 0 <= col < w):
        return {"error": f"pixel out of range: row 0..{h - 1}, col 0..{w - 1}"}
    d = float(depth[row, col])
    if not math.isfinite(d) or d <= 0:
        return {"error": f"invalid depth at pixel: {d}"}
    points, _valid = world_points(cam, np.array([row]), np.array([col]))
    return {
        "camera": camera,
        "pixel": [row, col],
        "depth_m": round(d, 4),
        "world_xyz": [round(float(v), 4) for v in points[0]],
    }


back_project._readonly = True  # type: ignore[attr-defined]


def locate(
    primitives: Any,
    text_prompt: str,
    camera: str = "cam_head",
    min_score: float = 0.2,
) -> dict[str, Any]:
    """Segment an object by text and measure it in 3D from the depth image."""

    if not isinstance(text_prompt, str) or not text_prompt.strip():
        return {"error": "text_prompt must be a non-empty string"}
    obs = _obs(primitives)
    cam = _camera(obs, camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    color = cam.get("color")
    if color is None:
        return {"error": "camera color image missing"}
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        return {"error": "sam3_client not configured"}
    res = sam3.segment(color, text_prompt=text_prompt, min_score=float(min_score))
    if not res.found:
        return {"camera": camera, "found": False, "text_prompt": text_prompt}
    mask = np.asarray(res.mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    out: dict[str, Any] = {
        "camera": camera,
        "found": True,
        "text_prompt": text_prompt,
        "score": round(float(res.score), 3),
        "area_px": int(mask.sum()),
        "centroid_rc": [int(rows.mean()), int(cols.mean())] if len(rows) else None,
    }
    if colour_only(primitives, cam):
        view = _view(camera, _shown(obs))
        if view is not None:
            outline = sight.robot_pixels(
                view, _shown(obs), own=_WRISTS.get(camera), scale=1.0, between=False
            )
            share = float(outline[mask].mean()) if mask.any() else 0.0
            if share > 0:
                out["robot_px_fraction"] = round(share, 3)
            if share >= ROBOT_MASK_SHARE and float(res.score) < ROBOT_MASK_SCORE:
                out["views"] = 1
                out["geometry_error"] = "the mask lies on the robot's own arm or gripper"
                return out
        geo, seen = _measure_in_colour(primitives, obs, camera, mask, text_prompt, float(min_score))
        out["views"] = seen["views"]
        if seen.get("guess"):
            out["height_guess"] = seen["guess"]
        if geo is None:
            out["geometry_error"] = (
                seen.get("hidden") or "too little of it above the table (flat, far or hidden)"
            )
            return out
        x, y = float(geo["center_xyz"][0]), float(geo["center_xyz"][1])
        if not (
            kin.TABLE_X[0] - OFF_TABLE_M <= x <= kin.TABLE_X[1] + OFF_TABLE_M
            and kin.TABLE_Y[0] - OFF_TABLE_M <= y <= kin.TABLE_Y[1] + OFF_TABLE_M
        ):
            out["geometry_error"] = "what the mask shows is not on the table"
            return out
        out.update(geo)
        return out
    try:
        points, _valid = world_points(cam, rows, cols)
    except Exception as exc:  # noqa: BLE001 - a missing depth image is a result, not a crash
        out["geometry_error"] = f"no depth: {exc}"
        return out
    on_robot = _on_robot(_shown(obs), points)
    if on_robot.any():
        out["robot_px_fraction"] = round(float(on_robot.mean()), 3)
        points = points[~on_robot]
    geo = object_geometry(points)
    if geo is None:
        out["geometry_error"] = (
            "the mask lies on the robot's own arm or gripper"
            if out.get("robot_px_fraction", 0) > 0.5
            else "too few points above the table (flat, far or hidden)"
        )
        return out
    out.update(geo)
    return out


#: Changes nothing in the scene: RPent's toolkit records no step after it (its ``readonly``).
locate._readonly = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_ARM = {"type": "string", "enum": list(ARMS)}
SPECS: dict[str, dict[str, Any]] = {
    "move_pose": {
        "name": "move_pose",
        "description": "Move an arm's grasp point to a position and orientation (degrees) with "
        "the host's IK; angles left out keep their current value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {"type": "array", "items": {"type": "number"}},
                "arm": _ARM,
                "gripper": {"type": "number"},
                "pitch": {"type": "number"},
                "yaw": {"type": "number"},
                "roll": {"type": "number"},
                "path": {"type": "string", "enum": ["auto", "line", "joint"]},
            },
            "required": ["xyz"],
        },
    },
    "grip": {
        "name": "grip",
        "description": "Close (1) or open (-1) a gripper fully and report what it holds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM,
                "gripper": {"type": "number"},
                "opening": {"type": "number"},
            },
            "required": ["gripper"],
        },
    },
    "locate": {
        "name": "locate",
        "description": "Segment an object by text and measure its extent, heading and height.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {"type": "string"},
                "camera": {"type": "string"},
                "min_score": {"type": "number"},
            },
            "required": ["text_prompt"],
        },
    },
    "back_project": {
        "name": "back_project",
        "description": "The world point a pixel shows: on the object under it, else on the table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer"},
                "col": {"type": "integer"},
                "camera": {"type": "string"},
            },
            "required": ["row", "col"],
        },
    },
    "home": {
        "name": "home",
        "description": "Return one arm or both to the joint angles the episode started with; "
        "gripper openings are kept.",
        "input_schema": {
            "type": "object",
            "properties": {"arm": {"type": "string", "enum": [*ARMS, "both"]}},
            "required": [],
        },
    },
    "place_on": {
        "name": "place_on",
        "description": "Put the held object on a support and back the arm clear: descend to "
        "support_top + tcp_to_bottom + clearance, open only when the descent arrived, then "
        "lift and retreat; the caller still decides when to home.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "support_top": {"type": "number"},
                "tcp_to_bottom": {"type": "number"},
                "arm": _ARM,
                "clearance": {"type": "number"},
                "lift_m": {"type": "number"},
                "retreat_xy": {"type": "array", "items": {"type": "number"}},
                "opening": {"type": "number"},
                "path": {"type": "string", "enum": ["auto", "line", "joint"]},
            },
            "required": ["x", "y", "support_top", "tcp_to_bottom"],
        },
    },
}


def register(toolkit: Any) -> None:
    """Add this module's tools to a freshly booted RoboDojo toolkit.

    The joint angles of ``home`` are read here: boot leaves both arms at their origin.  RPent's
    toolkit records a step after every handler except ``locate`` (marked read-only; RPent
    looks through the ``partial``).
    """

    primitives = toolkit._primitives
    state = primitives.env.get_obs().get("state", {})
    start = {arm: _joints(state, arm) for arm in ARMS}
    handlers = {
        "move_pose": partial(move_pose, primitives, start),
        "grip": partial(grip, primitives),
        "locate": partial(locate, primitives),
        "home": partial(home, primitives, start),
        "place_on": partial(place_on, primitives, start),
    }
    for name, handler in handlers.items():
        toolkit.add_tool(name, SPECS[name], handler)


__all__ = [
    "CARRY_JOINT_STEP_RAD",
    "CARRY_SLIP_DROP",
    "GRIP_SPAN_M",
    "GRIP_ZERO_M",
    "HOLD_MIN_OPENING",
    "HOME_MAX_ACTIONS",
    "HOME_TOL_RAD",
    "SPECS",
    "back_project",
    "colour_only",
    "depth_image",
    "extrinsic_of",
    "grip",
    "home",
    "locate",
    "move_pose",
    "object_geometry",
    "place_on",
    "probe",
    "register",
    "world_points",
]
