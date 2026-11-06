# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Real-hardware primitives behind ``robo.move_to`` / ``grip`` / ``home`` / ``place_on``.

Every primitive starts from a fresh **measured** observation, plans on the
runner (the service validates again), and executes synchronous ``movej_sync``
segments.  Arrival is decided from the measured joints/TCP after the segment,
not from an ACK.  The first version deliberately has no ``move_js`` servo
path: continuous servoing needs a fixed-rate scheduler, keep-alive and a stop
mechanism that this version does not claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .kinematics import angles, orientation, rotation_error
from .planning import PlanningError
from .state import Observation


def _arm(value: Any) -> str:
    if value not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    return str(value)


def _active_arms(toolkit: Any) -> tuple[str, ...]:
    return tuple(toolkit.config.execution.active_arms)


def _require_active(toolkit: Any, arm: str) -> str:
    active = _active_arms(toolkit)
    if arm not in active:
        raise ValueError(f"arm {arm!r} is not active on this station; active arms: {list(active)}")
    return arm


def _other_arm(toolkit: Any, arm: str) -> str | None:
    return next((name for name in _active_arms(toolkit) if name != arm), None)


def _require_gripper(toolkit: Any, arm: str) -> None:
    """Refuse a gripper command on an arm-only station.

    There is deliberately no synthetic gripper: an arm without a configured
    gripper can move, but OPEN/CLOSE have no device to act on.
    """

    cfg = toolkit.config.arms.get(arm)
    if cfg is None or cfg.gripper is None:
        raise ValueError(
            f"arm {arm!r} has no gripper configured on this station; OPEN/CLOSE are unavailable"
        )


def _command(value: Any, *, allow_keep: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("gripper must be 1 (close), -1 (open) or 0 (keep)")
    number = int(value)
    if number not in (-1, 0, 1) or (number == 0 and not allow_keep):
        allowed = "1, -1 or 0" if allow_keep else "1 or -1"
        raise ValueError(f"gripper must be {allowed}")
    return number


def _xyz(value: Sequence[float]) -> tuple[float, float, float]:
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError("xyz must be three numbers") from None
    if len(out) != 3 or not all(math.isfinite(v) for v in out):
        raise ValueError("xyz must be three finite numbers")
    return out


def _angle(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number or None") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _pose_summary(obs: Observation, arm: str) -> dict[str, Any]:
    measurement = obs.arm(arm)
    pose = np.asarray(measurement.tcp_pose).reshape(4, 4)
    pitch, yaw, roll = angles(pose[:3, :3])
    return {
        "xyz": [round(float(v), 4) for v in pose[:3, 3]],
        "pitch": round(float(pitch), 1),
        "yaw": round(float(yaw), 1),
        "roll": round(float(roll), 1),
    }


def _blank_move(obs: Observation, arm: str, target: tuple[float, float, float]) -> dict[str, Any]:
    now = _pose_summary(obs, arm)
    return {
        "arm": arm,
        "target_xyz": list(target),
        "start_xyz": now["xyz"],
        "final_xyz": now["xyz"],
        "dist_to_target_m": round(float(np.linalg.norm(np.asarray(now["xyz"]) - target)), 4),
        "orientation_error_deg": float("nan"),
        "target_angles": None,
        "final_angles": {k: now[k] for k in ("pitch", "yaw", "roll")},
        "reached": False,
        "settled": False,
        "path": None,
        "ik_error": None,
        "note": None,
        "blocked": False,
        "steps_used": 0,
        "truncated": False,
        "slip": False,
        "slip_opening": None,
        "slip_baseline": None,
        "measured": True,
    }


def _command_error(result: dict[str, Any]) -> str | None:
    err = result.get("error")
    if not err:
        return None
    return f"{err.get('code', 'error')}: {err.get('message', '')}"


def move_pose(
    toolkit: Any,
    xyz: Sequence[float],
    arm: str = "right",
    gripper: int = 0,
    pitch: float | None = None,
    yaw: float | None = None,
    roll: float | None = None,
    path: str = "auto",
) -> dict[str, Any]:
    arm = _arm(arm)
    _require_active(toolkit, arm)
    target = _xyz(xyz)
    command = _command(gripper, allow_keep=True)
    if command != 0:
        _require_gripper(toolkit, arm)
    if path not in ("auto", "line", "joint"):
        raise ValueError("path must be 'auto', 'line' or 'joint'")
    obs = toolkit.current_observation(refresh=True)
    result = _blank_move(obs, arm, target)
    if obs.fault:
        result["ik_error"] = "hardware fault: " + "; ".join(
            f"{a}: {obs.arm(a).fault_detail or 'fault'}"
            for a in _active_arms(toolkit)
            if obs.arm(a).fault
        )
        return result
    extra_steps = 0
    pre_note: str | None = None
    if command == -1:
        # An open is safe and local; apply it before the move so the carry
        # guard and the half-turn rule see the real open state.
        opened = grip(toolkit, arm=arm, gripper=-1)
        extra_steps += int(opened.get("steps_used", 0))
        if not opened.get("settled"):
            result["steps_used"] = extra_steps
            result["ik_error"] = (
                "the requested open before the move did not settle: "
                f"{opened.get('note') or 'unknown gripper state'}"
            )
            return result
        obs = toolkit.current_observation(refresh=True)
    held = toolkit.held_openings.get(arm)
    if command == 1 and held is None:
        pre_note = (
            "a close command is not applied during a real move; call set_gripper "
            "after the arm arrives"
        )
    holding = bool(held is not None and command != -1)
    result["steps_used"] = extra_steps
    # A close command while empty is allowed as a probe; the plan then treats
    # the fingers as holding (no half-turn) because the caller asked to close.
    holding_for_plan = bool(holding or command == 1)
    angles_arg = (_angle(pitch, "pitch"), _angle(yaw, "yaw"), _angle(roll, "roll"))
    other = _other_arm(toolkit, arm)
    other_q = obs.arm(other).q_model if other is not None else None
    try:
        planned = toolkit.planner.plan_move(
            arm,
            obs.arm(arm).q_model,
            target,
            pitch=angles_arg[0],
            yaw=angles_arg[1],
            roll=angles_arg[2],
            path=path,
            half_turn=not holding_for_plan,
            other_arm=other,
            other_q=other_q,
            holding=holding_for_plan,
        )
    except PlanningError as exc:
        result["ik_error"] = str(exc)
        result["note"] = (exc.detail or {}).get("why")
        return result
    payload = {
        "arm": arm,
        "q_start": [float(v) for v in obs.arm(arm).q_model],
        "waypoints": [list(w) for w in planned.waypoints],
        "speed": float(toolkit.config.execution.max_speed),
        "timeout_s": float(toolkit.config.execution.command_timeout_s),
        "carry_baseline": (float(held) if holding else None),
        "slip_drop": 0.02,
        "path": planned.path,
    }
    try:
        command_result = toolkit.submit_trajectory(payload, timeout_s=payload["timeout_s"])
    except Exception as exc:  # noqa: BLE001 - reported as a primitive result
        result["ik_error"] = _command_error({"error": {"code": "rpc_error", "message": str(exc)}})
        return result
    after = toolkit.refresh_observation()
    result.update(_pose_after(after, arm))
    result["path"] = planned.path
    result["target_angles"] = {
        "pitch": round(float(planned.target_angles[0]), 1),
        "yaw": round(float(planned.target_angles[1]), 1),
        "roll": round(float(planned.target_angles[2]), 1),
    }
    notes = [note for note in (planned.note, pre_note) if note]
    result["note"] = "; ".join(notes) if notes else None
    result["steps_used"] = extra_steps + int(command_result.get("steps_used", 0))
    result["truncated"] = False
    apply = command_result.get("result") or {}
    result["slip"] = bool(apply.get("slip"))
    result["slip_opening"] = apply.get("slip_opening")
    result["slip_baseline"] = apply.get("slip_baseline")
    result["dist_to_target_m"] = round(
        float(np.linalg.norm(np.asarray(result["final_xyz"]) - np.asarray(target))), 4
    )
    goal_rot = orientation(*planned.target_angles)
    final_rot = np.asarray(after.arm(arm).tcp_pose).reshape(4, 4)[:3, :3]
    result["orientation_error_deg"] = round(rotation_error(goal_rot, final_rot), 1)
    success = command_result.get("state") == "succeeded" and apply.get("success", False)
    result["settled"] = bool(success and not result["slip"])
    result["reached"] = bool(
        result["settled"]
        and result["dist_to_target_m"] <= 0.01
        and result["orientation_error_deg"] <= 5.0
    )
    err = _command_error(command_result)
    if err:
        result["ik_error"] = err
    if not result["reached"] and not result["slip"] and not err:
        result["blocked"] = bool(
            not apply.get("reached", True) or float(apply.get("max_joint_error_rad", 0.0)) > 0.02
        )
    if (
        command == -1
        or result["slip"]
        or command_result.get("state")
        in (
            "faulted",
            "timed_out",
            "cancelled",
            "rejected",
        )
    ):
        toolkit.held_openings.pop(arm, None)
    return result


def _pose_after(obs: Observation, arm: str) -> dict[str, Any]:
    now = _pose_summary(obs, arm)
    return {
        "final_xyz": now["xyz"],
        "final_angles": {k: now[k] for k in ("pitch", "yaw", "roll")},
    }


def grip(
    toolkit: Any,
    arm: str = "right",
    gripper: int = 1,
    opening: float | None = None,
) -> dict[str, Any]:
    arm = _arm(arm)
    _require_active(toolkit, arm)
    _require_gripper(toolkit, arm)
    command = _command(gripper, allow_keep=False)
    if opening is not None:
        try:
            opening = float(opening)
        except (TypeError, ValueError):
            raise ValueError("opening must be a number in (0, 1]") from None
        if command != -1 or not 0.0 < opening <= 1.0:
            raise ValueError("opening goes with robo.OPEN and must be in (0, 1]")
    obs = toolkit.current_observation(refresh=True)
    before = obs.arm(arm).gripper_opening
    base = {
        "arm": arm,
        "gripper": "closed" if command == 1 else "open",
        "opening": round(float(before), 4),
        "opening_before": round(float(before), 4),
        "holding": bool(toolkit.held_openings.get(arm) is not None),
        "settled": False,
        "width_m": None,
        "steps_used": 0,
        "step": obs.step,
        "step_limit": None,
        "truncated": False,
        "measured": True,
        "note": None,
    }
    held_baseline = toolkit.held_openings.get(arm)
    if command == 1 and held_baseline is not None and before < held_baseline - 0.02:
        # The measured opening fell below the settled close baseline: the
        # object may have slipped.  Start a fresh close instead of trusting the
        # old hold state.
        toolkit.held_openings.pop(arm, None)
        held_baseline = None
    if command == 1 and held_baseline is not None:
        # Keep the existing squeeze: do not restart grasp().
        base.update(
            holding=True,
            settled=True,
            opening=round(float(before), 4),
            width_m=round(float(obs.arm(arm).gripper_width_m), 5),
            steps_used=0,
            note="the gripper is already holding; the hold command was kept, not restarted",
        )
        return base
    width = None
    if command == -1 and opening is not None:
        width = toolkit.calibration.arm(arm).gripper.opening_to_width(opening)
    result = toolkit.submit_grip(
        {
            "arm": arm,
            "command": "close" if command == 1 else "open",
            "width_m": width,
            "force_n": toolkit.calibration.arm(arm).gripper.force_n,
        },
        timeout_s=toolkit.config.execution.command_timeout_s,
    )
    apply = result.get("result") or {}
    after = toolkit.refresh_observation()
    measured = after.arm(arm)
    base["opening"] = round(float(measured.gripper_opening), 4)
    base["holding"] = bool(apply.get("holding", False))
    base["settled"] = bool(apply.get("settled", False))
    base["steps_used"] = int(result.get("steps_used", 0))
    base["step"] = after.step
    base["width_m"] = (
        round(float(measured.gripper_width_m), 5)
        if command == 1 and base["holding"]
        else (None if command == -1 else 0.0)
    )
    base["note"] = apply.get("note") or _command_error(result)
    if command == 1 and base["holding"]:
        toolkit.held_openings[arm] = float(measured.gripper_opening)
    else:
        toolkit.held_openings.pop(arm, None)
    if after.fault:
        base["note"] = (base["note"] or "") + f"; hardware fault: {after.fault}"
    return base


def home(toolkit: Any, arm: str = "both") -> dict[str, Any]:
    if arm not in ("left", "right", "both"):
        raise ValueError("arm must be 'left', 'right' or 'both'")
    if arm != "both":
        _require_active(toolkit, arm)
    obs = toolkit.current_observation(refresh=True)
    arms = _active_arms(toolkit) if arm == "both" else (arm,)
    total_steps = 0
    max_error = 0.0
    notes: list[str] = []
    for name in arms:
        if obs.fault:
            notes.append(f"hardware fault; no motion: {obs.arm(name).fault_detail}")
            max_error = math.inf
            continue
        other = _other_arm(toolkit, name)
        safe_home = toolkit.calibration.arm(name).safe_home_rad
        error_now = max(
            abs(float(a) - float(b)) for a, b in zip(obs.arm(name).q_model, safe_home, strict=True)
        )
        if error_now <= toolkit.config.execution.settle_tolerance_rad:
            continue
        try:
            planned = toolkit.planner.plan_home(
                name,
                obs.arm(name).q_model,
                other_arm=other,
                other_q=obs.arm(other).q_model if other is not None else None,
            )
        except PlanningError as exc:
            notes.append(str(exc))
            max_error = math.inf
            continue
        payload = {
            "arm": name,
            "q_start": [float(v) for v in obs.arm(name).q_model],
            "waypoints": [list(w) for w in planned.waypoints],
            "speed": float(toolkit.config.execution.max_speed),
            "timeout_s": float(toolkit.config.execution.command_timeout_s),
            "carry_baseline": None,
            "slip_drop": 0.02,
            "path": planned.path,
        }
        try:
            result = toolkit.submit_trajectory(payload, timeout_s=payload["timeout_s"])
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{name}: {exc}")
            max_error = math.inf
            continue
        total_steps += int(result.get("steps_used", 0))
        apply = result.get("result") or {}
        error = float(apply.get("max_joint_error_rad", math.inf))
        max_error = max(max_error, error)
        if not apply.get("reached", False):
            notes.append(f"{name}: did not reach the configured home joint tolerance")
        obs = toolkit.refresh_observation()
    return {
        "arm": arm,
        "reached": max_error <= toolkit.config.execution.settle_tolerance_rad,
        "max_joint_error_rad": (float(max_error) if math.isfinite(max_error) else None),
        "steps_used": total_steps,
        "truncated": False,
        "note": "; ".join(notes) if notes else None,
        "measured": True,
    }


def place_on(
    toolkit: Any,
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
) -> dict[str, Any]:
    arm = _arm(arm)
    _require_active(toolkit, arm)
    _require_gripper(toolkit, arm)
    values = {}
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
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number") from None
        if not math.isfinite(values[name]):
            raise ValueError(f"{name} must be finite")
    if values["tcp_to_bottom"] <= 0.0:
        raise ValueError("tcp_to_bottom must be positive")
    if values["clearance"] < 0.0 or values["lift_m"] < 0.0:
        raise ValueError("clearance and lift_m must not be negative")
    try:
        retreat = (float(retreat_xy[0]), float(retreat_xy[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError("retreat_xy must be two numbers") from None
    if path not in ("auto", "line", "joint"):
        raise ValueError("path must be 'auto', 'line' or 'joint'")
    obs = toolkit.current_observation(refresh=True)
    _pose_summary(obs, arm)
    place_z = values["support_top"] + values["tcp_to_bottom"] + values["clearance"]
    target = (values["x"], values["y"], place_z)
    descent = move_pose(
        toolkit,
        target,
        arm=arm,
        gripper=0,
        pitch=None,
        yaw=None,
        roll=None,
        path=path,
    )
    out: dict[str, Any] = {
        "arm": arm,
        "placed": False,
        "released": False,
        "safe_to_home": False,
        "target_xyz": [round(v, 4) for v in target],
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
        "truncated": False,
        "note": None,
        "measured": True,
    }
    if not descent.get("reached") or descent.get("blocked") or descent.get("slip"):
        out["note"] = "descent did not arrive safely; the gripper was not opened"
        return out
    opened = grip(toolkit, arm=arm, gripper=-1, opening=opening)
    out["steps_used"] += int(opened.get("steps_used", 0))
    out["open_opening"] = opened.get("opening")
    out["released"] = bool(opened.get("settled"))
    if not out["released"]:
        out["note"] = "the descend arrived but the open did not settle; do not home"
        return out
    out["placed"] = True
    after = toolkit.current_observation(refresh=True)
    current = _pose_summary(after, arm)
    lift_target = (
        float(current["xyz"][0]),
        float(current["xyz"][1]),
        float(current["xyz"][2]) + values["lift_m"],
    )
    lift = move_pose(toolkit, lift_target, arm=arm, gripper=0, path="line")
    out["steps_used"] += int(lift.get("steps_used", 0))
    out["lift_reached"] = bool(lift.get("reached"))
    out["lift_blocked"] = bool(lift.get("blocked"))
    if not out["lift_reached"]:
        out["note"] = "the release succeeded but the lift did not arrive; do not home"
        return out
    retreat_target = (lift_target[0] + retreat[0], lift_target[1] + retreat[1], lift_target[2])
    retreat = move_pose(toolkit, retreat_target, arm=arm, gripper=0, path="line")
    out["steps_used"] += int(retreat.get("steps_used", 0))
    out["retreat_reached"] = bool(retreat.get("reached"))
    out["retreat_blocked"] = bool(retreat.get("blocked"))
    out["safe_to_home"] = bool(
        out["lift_reached"]
        and out["retreat_reached"]
        and not out["lift_blocked"]
        and not out["retreat_blocked"]
    )
    if not out["safe_to_home"]:
        out["note"] = "the retreat did not arrive safely; do not home"
    return out


__all__ = ["grip", "home", "move_pose", "place_on"]
