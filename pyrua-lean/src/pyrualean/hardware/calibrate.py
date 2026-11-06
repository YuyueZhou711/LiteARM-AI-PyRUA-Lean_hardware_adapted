# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Create a station calibration file from human measurements or an import.

This tool never invents a base pose, TCP offset, gripper width, camera matrix or
home pose.  Measurements must be supplied in a JSON document (``--from``); the
tool validates the geometry, stamps provenance/quality and writes a new file.

Motion-assisted capture is deliberately **not** implemented in this version.
Passing ``--allow-motion`` only makes that refusal explicit; it does not move an
arm.  The documented field procedure in ``docs/robots/litearm-real.md``
describes how to measure each section and how to verify it before setting
``executable=true``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import ARMS, CAMERAS, ConfigError, load_calibration


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the top level must be an object")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="source",
        required=True,
        help="JSON with measured values (start from configs/calibration/litearm_real.example.json)",
    )
    parser.add_argument("--out", required=True, help="calibration file to write")
    parser.add_argument(
        "--station-id", default=None, help="station id; must match the station file"
    )
    parser.add_argument("--version", default=None, help="calibration version (default: timestamp)")
    parser.add_argument(
        "--measurement-quality",
        default="manual-unverified",
        help="quality status recorded in the file (default: manual-unverified)",
    )
    parser.add_argument(
        "--mark-executable",
        action="store_true",
        help="mark executable=true (requires quality status 'measured')",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help=(
            "acknowledge that motion-assisted capture is explicitly requested "
            "(not implemented here)"
        ),
    )
    parser.add_argument(
        "--steps",
        default="base,tcp,gripper,camera,home",
        help="sections to stamp/validate (base,tcp,gripper,camera,home)",
    )
    args = parser.parse_args(argv)

    if args.allow_motion:
        print(
            "motion-assisted capture is not implemented in this version; no device was opened and "
            "no arm was moved.  Supply the measurements measured by hand in --from.",
        )
        return 2
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    known = {"base", "tcp", "gripper", "camera", "home"}
    unknown = steps - known
    if unknown:
        print(f"unknown calibration steps: {sorted(unknown)}")
        return 2
    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    try:
        doc = _read(source)
        doc = copy.deepcopy(doc)
        if args.station_id:
            doc["station_id"] = args.station_id
        version = args.version or time.strftime("manual-%Y%m%d-%H%M%S")
        doc["calibration_version"] = version
        doc["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        doc["source"] = f"manual/import from {source}; sections {sorted(steps)}"
        quality = doc.setdefault("quality", {})
        original_status = str(quality.get("status", "")).lower()
        original_source = str(doc.get("source", "")).lower()
        if args.mark_executable and (
            "example" in original_status
            or "unverified" in original_status
            or "template" in original_source
        ):
            raise ConfigError(
                "the input is still marked as an example/template; replace every measured "
                "value and set its quality.status to a measured status before --mark-executable"
            )
        quality["status"] = args.measurement_quality
        quality["notes"] = (
            "Values came from an operator-supplied file. Re-check each mirror and the TCP "
            "after assembly; set quality.status to 'measured' only after the field checklist."
        )
        quality["steps"] = sorted(steps)
        doc["executable"] = False
        if args.mark_executable:
            if args.measurement_quality != "measured":
                raise ConfigError(
                    "--mark-executable requires --measurement-quality measured; "
                    "an unverified file must not drive live motion"
                )
            doc["executable"] = True
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".tmp")
        temporary.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Validate exactly the entries the document carries: a single-arm,
        # wrist-only operator may legitimately stamp a trimmed calibration.
        # Malformed or empty sections stay ``None`` so the loader reports the
        # ordinary exact-set error against the full schema.
        raw_arms = doc.get("arms")
        arms = (
            tuple(name for name in ARMS if name in raw_arms) if isinstance(raw_arms, dict) else ()
        )
        raw_cameras = doc.get("cameras")
        cameras = (
            tuple(name for name in CAMERAS if name in raw_cameras)
            if isinstance(raw_cameras, dict)
            else ()
        )
        try:
            calibration = load_calibration(temporary, arms=arms or None, cameras=cameras or None)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        if out.exists() and not args.force:
            temporary.unlink(missing_ok=True)
            raise ConfigError(f"{out} exists; pass --force to replace it")
        os.replace(temporary, out)
        print(
            json.dumps(
                {
                    "out": str(out),
                    "station_id": calibration.station_id,
                    "version": calibration.version,
                    "executable": calibration.executable,
                    "sha256": calibration.sha256,
                    "quality": calibration.quality,
                },
                indent=2,
            )
        )
        return 0
    except ConfigError as exc:
        print(f"calibration refused: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
