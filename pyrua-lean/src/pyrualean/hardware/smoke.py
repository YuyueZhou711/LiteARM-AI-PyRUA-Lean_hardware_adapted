# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Hardware smoke test without a language model.

Modes:
* read-only: connect, handshake, snapshot, verify hashes, clock, camera frames,
  measured TCP and stop evidence.  It never commands motion.
* offline-fake / live with ``--allow-motion``: additionally exercise a small
  joint move, the gripper, and (in offline-fake) a single-object pick/place.
  Live motion requires ``--allow-motion`` explicitly; this repository's
  validation did not run that path on real hardware.

The output directory holds the JSON evidence and every snapshot image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ConfigError, load_station
from .motion import grip, move_pose, place_on
from .perception import Perception
from .rpc import HardwareClient, RpcError
from .toolkit import HardwareToolkit


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_station(args.config, mode=args.execution_mode)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    client = HardwareClient(config)
    result: dict[str, Any] = {"mode": config.execution.mode.value, "checks": {}, "errors": []}
    toolkit: HardwareToolkit | None = None
    try:
        hello = client.connect()
        result["hello"] = hello
        result["checks"]["protocol"] = hello.get("protocol_version")
        result["checks"]["station"] = hello.get("station_id")
        result["checks"]["config_hash_match"] = hello.get("config_sha256") == config.sha256
        result["checks"]["calibration_hash_match"] = (
            hello.get("calibration_sha256") == config.calibration_sha256
        )
        client.open_session(client_id="litearm-smoke")
        toolkit = HardwareToolkit(
            config,
            client,
            task="smoke",
            instruction="hardware smoke test",
            out_dir=out / "episode",
        )
        observation = toolkit.refresh_observation(tool="smoke_readonly")
        result["checks"]["snapshot_step"] = observation.step
        result["checks"]["data_age_s"] = observation.age_s
        result["checks"]["execution_state"] = observation.execution_state
        result["checks"]["primary_camera"] = config.primary_camera
        result["checks"]["faults"] = {
            arm: observation.arms[arm].fault_detail for arm in observation.arms
        }
        result["state"] = {
            arm: {
                "tcp": observation.arms[arm].tcp_pose,
                "opening": observation.arms[arm].gripper_opening,
                "width_m": observation.arms[arm].gripper_width_m,
                "joints": observation.arms[arm].q_model,
                "data_age_s": observation.arms[arm].age_s,
            }
            for arm in observation.arms
        }
        result["cameras"] = {
            name: {
                "shape": observation.camera_state[name].get("shape"),
                "has_depth": observation.camera_state[name].get("has_depth"),
            }
            for name in observation.camera_state
        }
        for camera in observation.camera_state:
            (out / f"{camera}.png").write_bytes(toolkit.artifact(f"{camera}.png", observation.step))
        motion_allowed = config.execution.mode.value == "offline-fake" or (
            config.execution.mode.value == "live" and args.allow_motion
        )
        if not motion_allowed:
            result["checks"]["motion"] = "not requested; read-only evidence only"
        else:
            if config.execution.mode.value == "live" and not args.allow_motion:
                raise RuntimeError("live motion requires --allow-motion")
            result["motion"] = _motion_checks(toolkit, args)
        return result
    finally:
        if toolkit is not None:
            try:
                toolkit.close()
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"close: {type(exc).__name__}: {exc}")
            result["stop"] = toolkit.stop_metadata()
        else:
            try:
                client.close(force=True)
            except Exception:  # noqa: BLE001
                pass
        _write(out / "smoke.json", result)


def _motion_checks(toolkit: HardwareToolkit, args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    import numpy as np

    observation = toolkit.current_observation(refresh=True)
    if args.arm not in observation.arms:
        raise RuntimeError(
            f"arm {args.arm!r} is not active on this station; active arms: "
            f"{sorted(observation.arms)}"
        )
    start = observation.arm(args.arm)
    start_xyz = np.asarray(start.tcp_pose, dtype=float).reshape(4, 4)[:3, 3]
    target = [float(start_xyz[0]), float(start_xyz[1] + 0.03), float(start_xyz[2])]
    move = move_pose(toolkit, target, arm=args.arm, path="joint")
    checks["small_move"] = {
        "target": target,
        "reached": move["reached"],
        "error": move["ik_error"],
        "steps_used": move["steps_used"],
    }
    back = move_pose(
        toolkit,
        [float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])],
        arm=args.arm,
        path="joint",
    )
    checks["return_move"] = {
        "reached": back["reached"],
        "error": back["ik_error"],
        "steps_used": back["steps_used"],
    }
    has_gripper = toolkit.config.arms[args.arm].gripper is not None
    if has_gripper:
        opened = grip(toolkit, arm=args.arm, gripper=-1)
        closed = grip(toolkit, arm=args.arm, gripper=1)
        checks["gripper"] = {
            "open": {"settled": opened["settled"], "opening": opened["opening"]},
            "close": {
                "settled": closed["settled"],
                "holding": closed["holding"],
                "width_m": closed["width_m"],
            },
        }
    else:
        checks["gripper"] = {"skipped": f"arm {args.arm!r} has no gripper configured"}
    if args.object_prompt:
        perception = toolkit.perception or Perception(toolkit.calibration, toolkit.config.sam3)
        frames = toolkit.frames(toolkit.current_observation(refresh=True))
        sight = perception.locate(frames, args.object_prompt, args.camera)
        checks["locate"] = sight
        if (
            has_gripper
            and not sight.get("measurement_failed")
            and args.place_xyz
            and args.support_top is not None
        ):
            x, y = sight["center_xyz"][0], sight["center_xyz"][1]
            above = [x, y, sight["top_z"] + args.approach_m]
            approach = move_pose(toolkit, above, arm=args.arm, path="line")
            checks["approach"] = {"reached": approach["reached"], "error": approach["ik_error"]}
            grasp = move_pose(
                toolkit,
                [x, y, sight["top_z"] + args.approach_m / 2.0],
                arm=args.arm,
                path="line",
            )
            checks["descend"] = {"reached": grasp["reached"], "error": grasp["ik_error"]}
            held = grip(toolkit, arm=args.arm, gripper=1)
            checks["grasp"] = {
                "holding": held["holding"],
                "width_m": held["width_m"],
                "settled": held["settled"],
            }
            lift = move_pose(toolkit, above, arm=args.arm, path="line")
            checks["lift"] = {"reached": lift["reached"], "error": lift["ik_error"]}
            place = place_on(
                toolkit,
                float(args.place_xyz[0]),
                float(args.place_xyz[1]),
                float(args.support_top),
                float(args.tcp_to_bottom),
                arm=args.arm,
                lift_m=0.08,
                retreat_xy=(0.0, -0.05),
            )
            checks["place"] = place
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="diagnostic evidence directory")
    parser.add_argument(
        "--execution-mode",
        choices=("read-only", "offline-fake", "live"),
        default=None,
    )
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help="allow the small motion checks in live mode (explicit; not run in this task)",
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--camera", default="cam_head")
    parser.add_argument("--object-prompt", default=None)
    parser.add_argument("--place-xyz", type=float, nargs=2, default=None)
    parser.add_argument("--support-top", type=float, default=None)
    parser.add_argument("--tcp-to-bottom", type=float, default=None)
    parser.add_argument("--approach-m", type=float, default=0.06)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ConfigError, RpcError, RuntimeError) as exc:
        print(f"smoke failed: {exc}")
        return 2
    print(json.dumps({"checks": result.get("checks"), "stop": result.get("stop")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run"]
