# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Kinematics of RoboDojo's two LiteArm SE arms, computed in the host: full pose, no RPC.

The chain is the composed URDF (``Assets/Robots/litearm/litearm.urdf``): ``base_link`` ->
``joint1`` .. ``joint7`` -> ``ee_link`` -> ``gripper_base`` -> ``tcp_link``.  The seven arm
joints are revolute; the LiteGrip's two prismatic finger joints hang off ``gripper_base`` and
are not part of the arm chain.  ``tcp_link`` is the grasp point between the finger pads, so
:data:`TCP_OFFSET` is 0: the reported ``<arm>_ee_pose`` is already the point ``move_to`` drives.

Frames.  ``tcp_link``'s x axis is the pointing direction (wrist to fingertips) and its y axis
the line the two fingers close along.  At the start the arm stands straight up, so the TCP
points up (pitch -90); the gripper can be brought down to the table by the IK.

Orientation for people: ``pitch`` is how far the pointing direction is below the horizontal
(0 level, 90 straight down, -90 straight up), ``yaw`` its heading on the table (0 away from the
robot, +90 to the robot's left, i.e. world -x), ``roll`` the turn of the fingers about the
pointing direction.  The rotation is ``Rz(90 + yaw) @ Ry(pitch) @ Rx(roll)``.

The base pose of each arm is the RoboDojo robot origin: x = -0.30 / +0.30, y = -0.45,
z = 0.765, turned 90 degrees about z (``env_cfg/robot/dual_litearm.yml``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .robodojo_kinematics import (
    _axis_angle,
    _homogeneous,
    _rot_x,
    _rot_y,
    _rot_z,
    _rotvec,
    _wrap,
    quat_wxyz,
    rot_from_quat,
    rotation_error,
    slerp,
)

ARMS = ("left", "right")

#: Joint origins (xyz, rpy) and axes of joint1 .. joint7 of litearm.urdf.  joint6's axis is
#: (0, 0, -1) on purpose: the real motor is mounted with stator and rotor reversed.
_CHAIN = (
    ((0.0, 0.0, 0.06975), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0635), (1.5708, 0.0, -1.5708), (0.0, 0.0, 1.0)),
    ((0.0, 0.1135, 0.0), (1.5708, 0.0, 3.1416), (0.0, 0.0, 1.0)),
    ((0.038, 0.0, 0.186), (1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((-0.038, 0.08545, 0.0), (1.5708, 0.0, 3.1416), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.13705), (1.5708, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ((0.0, 0.074, 0.0), (1.5708, 0.0, 1.5708), (0.0, 0.0, 1.0)),
)
#: Joint limits from the URDF, radians.  joint4 is asymmetric: -176.99 .. +2.01 degrees.
JOINT_LIMITS = (
    (-2.8274, 2.8274),
    (-1.7453, 1.7453),
    (-2.8274, 2.8274),
    (-3.0892, 0.0350),
    (-2.8274, 2.8274),
    (-1.5710, 1.5710),
    (-1.5710, 1.5710),
)
#: Fitted base of each arm: (x, y, z) and yaw about the world z axis (the robot origin).
BASES = {
    "left": ((-0.3, -0.45, 0.765), 1.5707963267948966),
    "right": ((0.3, -0.45, 0.765), 1.5707963267948966),
}
#: Table surface height (the arm bases stand on it).
TABLE_Z = 0.765
#: The table top's extent [x], [y] (RoboDojo's ``env_cfg/scene/default.yml``).
TABLE_X = (-0.70, 0.70)
TABLE_Y = (-0.60, 0.50)
_EDGE_MARGIN = 0.03
#: ``tcp_link`` is the grasp point: no offset along the tool axis.
TCP_OFFSET = 0.0
#: The fingertips, for the table check: the finger pads reach 4.6 cm beyond the TCP.
TIP_OFFSET = 0.046


def _rot(rpy: Sequence[float]) -> np.ndarray:
    return _rot_z(rpy[2]) @ _rot_y(rpy[1]) @ _rot_x(rpy[0])


#: From joint7's frame to ``tcp_link``: ee_link (fixed), gripper_mount (fixed) and tcp_link
#: (fixed).  This is the URDF's fixed tail of the chain.
_TCP_TRANSFORM = (
    _homogeneous(_rot((0.0, 1.5708, 0.0)), (0.05375, 0.0, 0.0))
    @ _homogeneous(_rot((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0641))
    @ _homogeneous(_rot((math.radians(-90.0), math.radians(-90.0), 0.0)), (0.0, 0.0, 0.0665))
)

_ORIGINS = tuple(_homogeneous(_rot(rpy), xyz) for xyz, rpy, _ in _CHAIN)
_AXES = tuple(np.asarray(axis) for _, _, axis in _CHAIN)
_BASE_T = {arm: _homogeneous(_rot_z(yaw), xyz) for arm, (xyz, yaw) in BASES.items()}


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")


def frames(arm: str, q: Sequence[float]) -> list[np.ndarray]:
    """World poses of each joint frame (after its rotation), in chain order."""

    _check_arm(arm)
    if len(q) != 7:
        raise ValueError("a LiteArm arm has 7 joints")
    t = _BASE_T[arm]
    out = []
    for origin, axis, angle in zip(_ORIGINS, _AXES, q, strict=True):
        t = t @ origin @ _homogeneous(_axis_angle(axis, float(angle)), (0.0, 0.0, 0.0))
        out.append(t)
    return out


def fk(arm: str, q: Sequence[float]) -> np.ndarray:
    """World pose (4x4) of ``tcp_link``, the frame RoboDojo reports as ``<arm>_ee_pose``."""

    return frames(arm, q)[-1] @ _TCP_TRANSFORM


def tcp(pose: np.ndarray, offset: float = TCP_OFFSET) -> np.ndarray:
    """The grasp point of a tcp_link pose (the pose itself: offset 0)."""

    return pose[:3, 3] + pose[:3, 0] * offset


def orientation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """The gripper rotation for (pitch, yaw, roll) in degrees (module docstring)."""

    return (
        _rot_z(math.radians(90.0 + yaw)) @ _rot_y(math.radians(pitch)) @ _rot_x(math.radians(roll))
    )


_VERTICAL = math.sin(math.radians(1.0))


def angles(rot: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of a gripper rotation; roll 0 when pointing straight down."""

    point = rot[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    horizontal = math.hypot(point[0], point[1])
    if horizontal < _VERTICAL:
        jaw = rot[:, 1]
        yaw = math.degrees(math.atan2(-jaw[1], -jaw[0]))
        return pitch, _wrap(yaw), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    base = orientation(pitch, yaw, 0.0)
    rest = base.T @ rot
    roll = math.degrees(math.atan2(rest[2, 1], rest[1, 1]))
    return pitch, _wrap(yaw), _wrap(roll)


def jacobian(arm: str, q: Sequence[float], offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(6x7 world-frame geometric Jacobian of the TCP, its pose)."""

    fr = frames(arm, q)
    end = fr[-1] @ _TCP_TRANSFORM
    point = end[:3, 3] + end[:3, 0] * offset
    jac = np.zeros((6, 7))
    for i, (t, axis) in enumerate(zip(fr, _AXES, strict=True)):
        a = t[:3, :3] @ axis
        jac[:3, i] = np.cross(a, point - t[:3, 3])
        jac[3:, i] = a
    return jac, end


@dataclass(frozen=True)
class IKResult:
    q: tuple[float, ...]
    pos_err_m: float
    rot_err_deg: float
    ok: bool
    collides: bool


def clearance_violations(arm: str, q: Sequence[float], offset: float = TCP_OFFSET) -> list[str]:
    """Parts of the arm in the table: below its top, over its extent."""

    def over(p: np.ndarray) -> bool:
        return (
            TABLE_X[0] - _EDGE_MARGIN <= p[0] <= TABLE_X[1] + _EDGE_MARGIN
            and TABLE_Y[0] - _EDGE_MARGIN <= p[1] <= TABLE_Y[1] + _EDGE_MARGIN
        )

    fr = frames(arm, q)
    bad = []
    for i, t in enumerate(fr[1:], start=2):
        if t[2, 3] < TABLE_Z + 0.015 and over(t[:3, 3]):
            bad.append(f"joint{i}")
    end = fr[-1] @ _TCP_TRANSFORM
    tip = end[:3, 3] + end[:3, 0] * max(offset, TIP_OFFSET)
    if tip[2] < TABLE_Z + 0.002 and over(tip):
        bad.append("fingertips")
    return bad


def solve(
    arm: str,
    target: np.ndarray,
    seed: Sequence[float],
    *,
    offset: float = TCP_OFFSET,
    rot_weight: float = 0.2,
    pos_tol: float = 0.002,
    rot_tol_deg: float = 1.0,
    iterations: int = 200,
) -> IKResult:
    """Damped least squares from ``seed`` for a 4x4 target of the TCP.

    Seven joints for a six-dimensional target: the extra degree of freedom is settled by the
    damped least-squares step and the seeds, which keep the solution near ``current``.
    """

    q = np.array(seed, dtype=float)
    lo = np.array([lim[0] for lim in JOINT_LIMITS])
    hi = np.array([lim[1] for lim in JOINT_LIMITS])
    damping = 0.02
    for _ in range(iterations):
        jac, end = jacobian(arm, q, offset)
        point = end[:3, 3] + end[:3, 0] * offset
        dp = target[:3, 3] - point
        dr = _rotvec(target[:3, :3] @ end[:3, :3].T)
        pos_err, rot_err = float(np.linalg.norm(dp)), math.degrees(float(np.linalg.norm(dr)))
        if pos_err < pos_tol * 0.25 and rot_err < rot_tol_deg * 0.25:
            break
        err = np.concatenate([dp, rot_weight * dr])
        j = jac.copy()
        j[3:] *= rot_weight
        step = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(6), err)
        norm = float(np.linalg.norm(step))
        if norm > 0.25:
            step *= 0.25 / norm
        q = np.clip(q + step, lo, hi)
    end = fk(arm, q)
    point = end[:3, 3] + end[:3, 0] * offset
    pos_err = float(np.linalg.norm(target[:3, 3] - point))
    rot_err = rotation_error(target[:3, :3], end[:3, :3])
    ok = pos_err <= pos_tol and rot_err <= rot_tol_deg
    return IKResult(
        tuple(float(v) for v in q), pos_err, rot_err, ok, bool(clearance_violations(arm, q, offset))
    )


def best_solution(
    arm: str,
    target: np.ndarray,
    current: Sequence[float],
    *,
    offset: float = TCP_OFFSET,
    extra_seeds: Sequence[Sequence[float]] = (),
    **kwargs,
) -> IKResult:
    """The solution nearest ``current`` that reaches the target and clears the table.

    Seeds: the current joints, the same with the tool rotation turned half a turn either way,
    the extra seeds given (the starting joints), and a few fixed perturbations of the current
    joints.  When no seed reaches the target, the closest miss is returned (``ok`` False).
    """

    cur = np.array(current, dtype=float)
    seeds = [cur]
    for d in (math.pi, -math.pi):
        s = cur.copy()
        s[6] = s[6] + d
        if JOINT_LIMITS[6][0] <= s[6] <= JOINT_LIMITS[6][1]:
            seeds.append(s)
    seeds.extend(np.array(s, dtype=float) for s in extra_seeds)
    rng = np.random.default_rng(0)
    seeds.extend(cur + rng.normal(0.0, 0.6, size=7) for _ in range(8))
    good, misses = [], []
    for s in seeds:
        res = solve(arm, target, s, offset=offset, **kwargs)
        (good if res.ok and not res.collides else misses).append(res)
    if good:
        weights = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.8, 0.5])
        return min(good, key=lambda r: float(np.sum(weights * np.abs(np.array(r.q) - cur))))
    return min(misses, key=lambda r: (r.collides, r.pos_err_m + 0.1 * math.radians(r.rot_err_deg)))


def line_path(
    arm: str,
    start_q: Sequence[float],
    target: np.ndarray,
    *,
    offset: float = TCP_OFFSET,
    step_m: float = 0.02,
    step_deg: float = 8.0,
) -> tuple[list[tuple[float, ...]], str | None]:
    """Joint waypoints that move the grasp point along a straight line to ``target``."""

    start = fk(arm, start_q)
    p0 = start[:3, 3] + start[:3, 0] * offset
    dist = float(np.linalg.norm(target[:3, 3] - p0))
    turn = rotation_error(start[:3, :3], target[:3, :3])
    n = max(1, math.ceil(dist / step_m), math.ceil(turn / step_deg))
    q = tuple(float(v) for v in start_q)
    out = []
    for k in range(1, n + 1):
        t = k / n
        way = np.eye(4)
        way[:3, :3] = slerp(start[:3, :3], target[:3, :3], t)
        way[:3, 3] = p0 + (target[:3, 3] - p0) * t
        res = solve(arm, way, q, offset=offset)
        if not res.ok:
            return out, f"the line leaves the arm's reach {t * dist:.3f} m along it"
        if res.collides:
            return out, f"the line takes the arm into the table {t * dist:.3f} m along it"
        if max(abs(a - b) for a, b in zip(res.q, q, strict=True)) > 0.6:
            return out, "the line passes a singular pose (the wrist would flip)"
        q = res.q
        out.append(q)
    return out, None


__all__ = [
    "ARMS",
    "BASES",
    "IKResult",
    "JOINT_LIMITS",
    "TABLE_X",
    "TABLE_Y",
    "TABLE_Z",
    "TCP_OFFSET",
    "TIP_OFFSET",
    "angles",
    "best_solution",
    "clearance_violations",
    "fk",
    "frames",
    "jacobian",
    "line_path",
    "orientation",
    "quat_wxyz",
    "rot_from_quat",
    "rotation_error",
    "slerp",
    "solve",
    "tcp",
]
