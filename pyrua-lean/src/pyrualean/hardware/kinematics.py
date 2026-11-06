# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Instantiated LiteArm kinematics and conservative collision geometry.

The chain comes from the packaged URDF description (``assets/litearm.urdf``),
not from RoboDojo's asset directory and not from the simulator's global
constants.  The base pose, joint map, joint limits, flange-to-TCP transform,
camera mounts, table and obstacles come from the station calibration.

The implementation deliberately mirrors the existing angle convention used by
``robo``::

    R = Rz(90 deg + yaw) @ Ry(pitch) @ Rx(roll)

with ``x`` across the table, ``y`` into the table and ``z`` up.  The URDF's
``tcp_link`` frame is the grasp point: its ``x`` axis points from the wrist
towards the fingertips and its ``y`` axis is the closing direction.

Collision geometry is conservative and primitive-based (capsules, a tool
body and two finger boxes).  It is **not** a mesh-accurate collision checker;
the documentation states this limit and the field acceptance procedure.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArmCalibration, Calibration, ConfigError

ARM_NAMES = ("left", "right")
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
DEFAULT_URDF = Path(__file__).resolve().parent / "assets" / "litearm.urdf"


# ---------------------------------------------------------------------------
# small rotation helpers (the same conventions as the existing package)
# ---------------------------------------------------------------------------


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def homogeneous(rot: np.ndarray, pos: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = np.asarray(rot, dtype=float)
    out[:3, 3] = np.asarray(pos, dtype=float)
    return out


def axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n == 0.0:
        return np.eye(3)
    x, y, z = a / n
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def matrix_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """Generic URDF/SDK roll-pitch-yaw for ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""

    r = np.asarray(rot, dtype=float)
    sy = math.hypot(float(r[0, 0]), float(r[1, 0]))
    if sy > 1e-9:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        pitch = math.atan2(-float(r[2, 0]), sy)
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        pitch = math.atan2(-float(r[2, 0]), sy)
        yaw = 0.0
    return (roll, pitch, yaw)


def orientation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Gripper rotation for ``(pitch, yaw, roll)`` in degrees (existing convention)."""

    return rot_z(math.radians(90.0 + yaw)) @ rot_y(math.radians(pitch)) @ rot_x(math.radians(roll))


_VERTICAL = math.sin(math.radians(1.0))


def wrap_deg(angle: float) -> float:
    out = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if out == -180.0 else out


def angles(rot: np.ndarray) -> tuple[float, float, float]:
    """``(pitch, yaw, roll)`` in degrees of a gripper rotation."""

    point = np.asarray(rot, dtype=float)[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(point[2])))))
    horizontal = math.hypot(float(point[0]), float(point[1]))
    if horizontal < _VERTICAL:
        jaw = np.asarray(rot, dtype=float)[:, 1]
        yaw = math.degrees(math.atan2(-float(jaw[1]), -float(jaw[0])))
        return pitch, wrap_deg(yaw), 0.0
    yaw = math.degrees(math.atan2(float(point[1]), float(point[0]))) - 90.0
    base = orientation(pitch, yaw, 0.0)
    rest = base.T @ np.asarray(rot, dtype=float)
    roll = math.degrees(math.atan2(float(rest[2, 1]), float(rest[1, 1])))
    return pitch, wrap_deg(yaw), wrap_deg(roll)


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two rotations, degrees."""

    c = (float(np.trace(np.asarray(a).T @ np.asarray(b))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rotvec(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float)
    c = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))
    angle = math.acos(c)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return axis * (angle / (2.0 * math.sin(angle)))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = max(-1.0, min(1.0, float(np.trace(a.T @ b) - 1.0) / 2.0))
    angle = math.acos(c)
    if angle < 1e-9:
        return a.copy()
    out = (math.sin((1.0 - t) * angle) * a + math.sin(t * angle) * b) / math.sin(angle)
    u, _, vh = np.linalg.svd(out)
    return u @ vh


def quat_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(rot, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


# ---------------------------------------------------------------------------
# URDF chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if origin is not None:
        if origin.get("xyz"):
            xyz = tuple(float(v) for v in origin.get("xyz", "").split())
        if origin.get("rpy"):
            rpy = tuple(float(v) for v in origin.get("rpy", "").split())
    return homogeneous(rpy_matrix(rpy), xyz)


def load_urdf_chain(path: str | Path = DEFAULT_URDF) -> tuple[_Joint, ...]:
    """Read ``base_link`` -> ``ee_link`` from an official LiteArm URDF."""

    location = Path(path)
    if not location.is_file():
        raise ConfigError(f"robot description not found: {location}")
    try:
        robot = ET.parse(location).getroot()
    except ET.ParseError as exc:
        raise ConfigError(f"{location} is not a valid URDF: {exc}") from None
    if robot.tag != "robot":
        raise ConfigError(f"{location}: root element is not <robot>")
    parent_to_joint = {j.find("child").get("link"): j for j in robot.findall("joint")}
    if "ee_link" not in parent_to_joint:
        raise ConfigError(f"{location} has no ee_link")
    chain: list[ET.Element] = []
    link = "ee_link"
    seen = set()
    while link != "base_link":
        if link in seen:
            raise ConfigError(f"{location}: joint tree contains a cycle at {link}")
        seen.add(link)
        joint = parent_to_joint.get(link)
        if joint is None:
            raise ConfigError(f"{location}: no chain from base_link to ee_link")
        chain.append(joint)
        link = joint.find("parent").get("link")
    chain.reverse()
    joints: list[_Joint] = []
    for joint in chain:
        name = joint.get("name", "")
        kind = joint.get("type", "")
        if name not in JOINT_NAMES and kind != "fixed":
            raise ConfigError(f"{location}: unexpected moving joint {name!r}")
        axis_el = joint.find("axis")
        axis = (0.0, 0.0, 1.0)
        if axis_el is not None and axis_el.get("xyz"):
            axis = tuple(float(v) for v in axis_el.get("xyz", "").split())
        limit = joint.find("limit")
        lower = float(limit.get("lower", -math.pi)) if limit is not None else 0.0
        upper = float(limit.get("upper", math.pi)) if limit is not None else 0.0
        joints.append(
            _Joint(name, kind, _origin(joint), np.asarray(axis, dtype=float), lower, upper)
        )
    moving = [j.name for j in joints if j.kind == "revolute"]
    if moving != list(JOINT_NAMES):
        raise ConfigError(f"{location}: expected joints {JOINT_NAMES}, got {moving}")
    return tuple(joints)


@dataclass(frozen=True)
class IKResult:
    q: tuple[float, ...]
    pos_err_m: float
    rot_err_deg: float
    ok: bool
    collides: bool
    reason: str | None = None


class LitearmKinematics:
    """FK/IK for one calibrated station."""

    def __init__(self, calibration: Calibration, *, urdf: str | Path = DEFAULT_URDF) -> None:
        self.calibration = calibration
        self.urdf = Path(urdf)
        self.joints = load_urdf_chain(self.urdf)
        self._origins = tuple(j.origin for j in self.joints)
        self._axes = tuple(j.axis for j in self.joints)
        self._moving = tuple(j for j in self.joints if j.kind == "revolute")
        for arm_name, arm in calibration.arms.items():
            for i, (lo, hi) in enumerate(arm.joint_limits_rad):
                urdf_joint = self._moving[i]
                if lo < urdf_joint.lower - 1e-6 or hi > urdf_joint.upper + 1e-6:
                    raise ConfigError(
                        f"arms.{arm_name}.joint_limits_rad[{i}] ({lo:.6f}, {hi:.6f}) "
                        f"exceeds the URDF limit ({urdf_joint.lower:.6f}, "
                        f"{urdf_joint.upper:.6f})"
                    )
        self._base = {name: arm.base_pose for name, arm in calibration.arms.items()}
        self._tool = {name: arm.flange_to_tcp for name, arm in calibration.arms.items()}
        self._limits = {name: arm.joint_limits_rad for name, arm in calibration.arms.items()}

    def _check_arm(self, arm: str) -> ArmCalibration:
        try:
            return self.calibration.arms[arm]
        except KeyError:
            raise ValueError(f"arm must be one of {ARM_NAMES}, got {arm!r}") from None

    def frames(self, arm: str, q_model: Sequence[float]) -> list[np.ndarray]:
        """World poses of each URDF joint frame after its rotation (7 entries)."""

        cal = self._check_arm(arm)
        q = np.asarray(q_model, dtype=float)
        if q.shape != (7,):
            raise ValueError("a LiteArm arm has 7 joints")
        if not np.all(np.isfinite(q)):
            raise ValueError("joint angles must be finite")
        limits = np.asarray(cal.joint_limits_rad)
        if np.any(q < limits[:, 0] - 1e-9) or np.any(q > limits[:, 1] + 1e-9):
            raise ValueError(f"joint angles outside the calibrated limits: {q.tolist()}")
        t = np.asarray(cal.base_pose, dtype=float).copy()
        out = []
        moving_index = 0
        for joint in self.joints:
            t = t @ joint.origin
            if joint.kind == "revolute":
                t = t @ homogeneous(axis_angle(joint.axis, float(q[moving_index])), (0.0, 0.0, 0.0))
                out.append(t.copy())
                moving_index += 1
        return out

    def flange_pose(self, arm: str, q_model: Sequence[float]) -> np.ndarray:
        cal = self._check_arm(arm)
        q = np.asarray(q_model, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("a LiteArm arm has 7 finite joints")
        limits = np.asarray(cal.joint_limits_rad)
        if np.any(q < limits[:, 0] - 1e-9) or np.any(q > limits[:, 1] + 1e-9):
            raise ValueError(f"joint angles outside the calibrated limits: {q.tolist()}")
        t = np.asarray(cal.base_pose, dtype=float).copy()
        moving_index = 0
        for joint in self.joints:
            t = t @ joint.origin
            if joint.kind == "revolute":
                t = t @ homogeneous(axis_angle(joint.axis, float(q[moving_index])), (0.0, 0.0, 0.0))
                moving_index += 1
        return t

    def fk(self, arm: str, q_model: Sequence[float]) -> np.ndarray:
        """World pose of the calibrated grasp point (TCP)."""

        return self.flange_pose(arm, q_model) @ self._check_arm(arm).flange_to_tcp

    def tcp_point(self, arm: str, q_model: Sequence[float]) -> np.ndarray:
        return self.fk(arm, q_model)[:3, 3]

    def sdk_from_model(self, arm: str, q_model: Sequence[float]) -> np.ndarray:
        return self._check_arm(arm).sdk_from_model(q_model)

    def model_from_sdk(self, arm: str, q_sdk: Sequence[float]) -> np.ndarray:
        return self._check_arm(arm).model_from_sdk(q_sdk)

    def jacobian(self, arm: str, q_model: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """6x7 world Jacobian of the TCP and its world pose."""

        frames = self.frames(arm, q_model)
        end = self.fk(arm, q_model)
        point = end[:3, 3]
        jac = np.zeros((6, 7))
        for i, (frame, joint) in enumerate(zip(frames, self._moving, strict=True)):
            axis = frame[:3, :3] @ joint.axis
            jac[:3, i] = np.cross(axis, point - frame[:3, 3])
            jac[3:, i] = axis
        return jac, end

    def solve(
        self,
        arm: str,
        target: np.ndarray,
        seed: Sequence[float],
        *,
        rot_weight: float = 0.2,
        pos_tol: float = 0.002,
        rot_tol_deg: float = 1.0,
        iterations: int = 180,
        collision_check: bool = True,
    ) -> IKResult:
        cal = self._check_arm(arm)
        lim = np.asarray(cal.joint_limits_rad)
        q = np.asarray(seed, dtype=float).copy()
        q = np.clip(q, lim[:, 0], lim[:, 1])
        target = np.asarray(target, dtype=float)
        damping = 0.02
        for _ in range(iterations):
            jac, end = self.jacobian(arm, q)
            dp = target[:3, 3] - end[:3, 3]
            dr = rotvec(target[:3, :3] @ end[:3, :3].T)
            pos_err = float(np.linalg.norm(dp))
            rot_err = math.degrees(float(np.linalg.norm(dr)))
            if pos_err < pos_tol * 0.25 and rot_err < rot_tol_deg * 0.25:
                break
            err = np.concatenate([dp, rot_weight * dr])
            j = jac.copy()
            j[3:] *= rot_weight
            try:
                step = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(6), err)
            except np.linalg.LinAlgError:
                break
            norm = float(np.linalg.norm(step))
            if norm > 0.20:
                step *= 0.20 / norm
            q = np.clip(q + step, lim[:, 0], lim[:, 1])
        end = self.fk(arm, q)
        pos_err = float(np.linalg.norm(target[:3, 3] - end[:3, 3]))
        rot_err = rotation_error(target[:3, :3], end[:3, :3])
        ok = pos_err <= pos_tol and rot_err <= rot_tol_deg
        collides = collision_check and bool(
            CollisionModel(self.calibration, self).self_violations(arm, q)
        )
        return IKResult(tuple(float(v) for v in q), pos_err, rot_err, ok, collides)

    def best_solution(
        self,
        arm: str,
        target: np.ndarray,
        current: Sequence[float],
        *,
        extra_seeds: Sequence[Sequence[float]] = (),
        **kwargs: Any,
    ) -> IKResult:
        cal = self._check_arm(arm)
        cur = np.asarray(current, dtype=float)
        seeds = [cur]
        for delta in (math.pi, -math.pi):
            other = cur.copy()
            other[6] = other[6] + delta
            lo, hi = cal.joint_limits_rad[6]
            if lo <= other[6] <= hi:
                seeds.append(other)
        seeds.extend(np.asarray(s, dtype=float) for s in extra_seeds)
        rng = np.random.default_rng(11)
        seeds.extend(cur + rng.normal(0.0, 0.55, size=7) for _ in range(6))
        good: list[IKResult] = []
        misses: list[IKResult] = []
        for seed in seeds:
            res = self.solve(arm, target, seed, **kwargs)
            (good if res.ok and not res.collides else misses).append(res)
        if good:
            weights = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.8, 0.5])
            return min(
                good,
                key=lambda r: float(np.sum(weights * np.abs(np.asarray(r.q) - cur))),
            )
        return min(
            misses,
            key=lambda r: (r.collides, r.pos_err_m + 0.1 * math.radians(r.rot_err_deg)),
        )

    def line_path(
        self,
        arm: str,
        start_q: Sequence[float],
        target: np.ndarray,
        *,
        step_m: float = 0.02,
        step_deg: float = 8.0,
        max_joint_step_rad: float = 0.30,
    ) -> tuple[list[tuple[float, ...]], str | None]:
        start = self.fk(arm, start_q)
        p0 = start[:3, 3]
        dist = float(np.linalg.norm(target[:3, 3] - p0))
        turn = rotation_error(start[:3, :3], target[:3, :3])
        count = max(1, math.ceil(dist / step_m), math.ceil(turn / step_deg))
        q = tuple(float(v) for v in start_q)
        out: list[tuple[float, ...]] = []
        for k in range(1, count + 1):
            t = k / count
            way = np.eye(4)
            way[:3, :3] = slerp(start[:3, :3], target[:3, :3], t)
            way[:3, 3] = p0 + (target[:3, 3] - p0) * t
            res = self.solve(arm, way, q)
            if not res.ok:
                return out, f"the line leaves the arm's reach {t * dist:.3f} m along it"
            if res.collides:
                return out, "the line takes the arm into the table or an obstacle"
            if max(abs(a - b) for a, b in zip(res.q, q, strict=True)) > max_joint_step_rad:
                return out, "the line passes a singular pose (the wrist would flip)"
            q = res.q
            out.append(q)
        return out, None

    def camera_pose(self, camera: str, arm: str, q_model: Sequence[float]) -> np.ndarray:
        """World camera-to-world transform at the given arm configuration."""

        cam = self.calibration.camera(camera)
        if cam.mount == "world":
            return np.asarray(cam.transform, dtype=float).copy()
        if cam.arm != arm:
            raise ValueError(
                f"{camera} is mounted on the {cam.arm} arm, not {arm}: a wrist camera "
                "pose must be evaluated with its own arm's joints"
            )
        return self.fk(arm, q_model) @ np.asarray(cam.transform, dtype=float)

    def camera_extrinsic(
        self, camera: str, arm: str, q_model: Sequence[float]
    ) -> list[list[float]]:
        return [[float(v) for v in row] for row in self.camera_pose(camera, arm, q_model)]

    def check_safe_home(self, arm: str) -> list[str]:
        cal = self._check_arm(arm)
        return CollisionModel(self.calibration, self).self_violations(arm, cal.safe_home_rad)


# ---------------------------------------------------------------------------
# approximate collision geometry
# ---------------------------------------------------------------------------


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = max(0.0, min(1.0, float((point - a) @ ab) / denom))
    return float(np.linalg.norm(point - (a + t * ab)))


def _segment_segment_distance(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> float:
    """Shortest distance between two 3-D segments (Ericson, clamped)."""

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return float(np.linalg.norm(c1 - c2))


class CollisionModel:
    """Conservative collision checks for one calibrated station.

    Each arm link is a capsule between consecutive joint frames; the tool is a
    capsule from the flange to the grasp point plus two finger boxes sampled at
    their corners.  The table, configured axis-aligned obstacles and the other
    arm's *measured* configuration are checked.  Mesh-exact contact is out of
    scope; see ``docs/robots/litearm-real.md``.
    """

    def __init__(self, calibration: Calibration, kinematics: LitearmKinematics) -> None:
        self.calibration = calibration
        self.kinematics = kinematics

    def _link_segments(
        self, arm: str, q: Sequence[float]
    ) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
        cal = self.calibration.arm(arm)
        frames = self.kinematics.frames(arm, q)
        flange = self.kinematics.flange_pose(arm, q)
        points = [np.asarray(cal.base_pose)[:3, 3]] + [f[:3, 3] for f in frames] + [flange[:3, 3]]
        radii = list(cal.link_radii_m) + [cal.link_radii_m[-1]]
        out = []
        for i in range(7):
            out.append(
                (
                    points[i],
                    points[i + 1],
                    float(radii[i]),
                    f"{arm}_link{i + 1}",
                )
            )
        tcp = self.kinematics.fk(arm, q)
        tool = cal.tool_collision
        out.append(
            (
                flange[:3, 3],
                tcp[:3, 3],
                float(tool.body_radius_m),
                f"{arm}_tool_body",
            )
        )
        # Fingers: two boxes around the grasp point, sampled at their corners.
        for side in (-1.0, 1.0):
            for along in (-tool.body_length_m, tool.tip_offset_m):
                for across in (-tool.finger_half_width_m, tool.finger_half_width_m):
                    for thick in (-tool.finger_half_thickness_m, tool.finger_half_thickness_m):
                        p = tcp[:3, 3] + tcp[:3, :3] @ np.array([along, side * across, thick])
                        out.append((p.copy(), p.copy(), 0.004, f"{arm}_finger"))
        return out

    def _point_in_table(self, point: np.ndarray, margin: float = 0.0) -> bool:
        table = self.calibration.table
        return (
            table.x_min_m - table.edge_margin_m - margin
            <= float(point[0])
            <= table.x_max_m + table.edge_margin_m + margin
            and table.y_min_m - table.edge_margin_m - margin
            <= float(point[1])
            <= table.y_max_m + table.edge_margin_m + margin
        )

    def table_violations(self, arm: str, q: Sequence[float]) -> list[str]:
        table = self.calibration.table
        bad = []
        # The first link is the base column bolted to the table; its lower end is
        # supposed to touch the table, so it is not a collision.
        for a, b, radius, name in self._link_segments(arm, q)[1:]:
            # sample the segment; the radius is added below the table plane
            for t in np.linspace(0.0, 1.0, 7):
                p = a + (b - a) * t
                if self._point_in_table(p) and float(p[2]) - radius < table.z_m + 0.002:
                    bad.append(name)
                    break
        return bad

    def obstacle_violations(self, arm: str, q: Sequence[float]) -> list[str]:
        bad = []
        for a, b, radius, name in self._link_segments(arm, q):
            for obstacle in self.calibration.obstacles:
                lo = np.asarray(obstacle.minimum) - radius
                hi = np.asarray(obstacle.maximum) + radius
                hit = False
                for t in np.linspace(0.0, 1.0, 9):
                    p = a + (b - a) * t
                    if np.all(p >= lo) and np.all(p <= hi):
                        hit = True
                        break
                if hit:
                    bad.append(f"{name}->{obstacle.name}")
        return bad

    def self_violations(self, arm: str, q: Sequence[float]) -> list[str]:
        segments = self._link_segments(arm, q)
        bad = self.table_violations(arm, q) + self.obstacle_violations(arm, q)
        # Consecutive and next-consecutive links share joints and their
        # conservative capsules overlap by construction; skip |i-j| <= 2.
        for i in range(len(segments)):
            for j in range(i + 3, len(segments)):
                if segments[i][3] == segments[j][3]:
                    continue
                # The tool body and its own fingers are one rigid assembly; their
                # primitive boxes overlap at the mount by construction.
                if "tool_body" in segments[i][3] and "finger" in segments[j][3]:
                    continue
                if "finger" in segments[i][3] and "tool_body" in segments[j][3]:
                    continue
                d = _segment_segment_distance(
                    segments[i][0], segments[i][1], segments[j][0], segments[j][1]
                )
                if d < segments[i][2] + segments[j][2] - 1e-6:
                    bad.append(f"{segments[i][3]}<->{segments[j][3]}")
        return sorted(set(bad))

    def pair_violations(
        self,
        arm: str,
        q: Sequence[float],
        other_arm: str,
        other_q: Sequence[float],
    ) -> list[str]:
        if arm == other_arm:
            return []
        a = self._link_segments(arm, q)
        b = self._link_segments(other_arm, other_q)
        bad = []
        for sa in a:
            for sb in b:
                d = _segment_segment_distance(sa[0], sa[1], sb[0], sb[1])
                if d < sa[2] + sb[2] + 0.005:
                    bad.append(f"{sa[3]}<->{sb[3]}")
                    break
            if bad:
                break
        return bad

    def all_violations(
        self,
        arm: str,
        q: Sequence[float],
        *,
        other_arm: str | None = None,
        other_q: Sequence[float] | None = None,
    ) -> list[str]:
        out = self.self_violations(arm, q)
        if other_arm is not None and other_q is not None:
            out.extend(self.pair_violations(arm, q, other_arm, other_q))
        return out


__all__ = [
    "ARM_NAMES",
    "CollisionModel",
    "DEFAULT_URDF",
    "IKResult",
    "JOINT_NAMES",
    "LitearmKinematics",
    "angles",
    "axis_angle",
    "homogeneous",
    "load_urdf_chain",
    "matrix_to_rpy",
    "orientation",
    "quat_wxyz",
    "rotvec",
    "rotation_error",
    "rpy_matrix",
    "slerp",
    "wrap_deg",
]
