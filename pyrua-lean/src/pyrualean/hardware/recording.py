# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""On-disk evidence for a hardware episode.

A snapshot is kept as one immutable directory: the observation envelope, the
raw device reading, camera calibration, PNG/NumPy artifacts, the executor
state and the stop evidence.  Historical ``step`` values read these files, so
a later arm pose cannot rewrite what an earlier image showed.

No connection credentials or vendor SDK objects are written.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .config import StationConfig


class RecordingError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class Recorder:
    def __init__(self, root: str | Path, config: StationConfig) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "steps").mkdir(exist_ok=True)
        self.config = config
        self.records: dict[int, dict[str, Any]] = {}
        self._by_basename: dict[tuple[int, str], str] = {}
        self.finalized = False
        self._write_json(
            self.root / "recording.json",
            {
                "schema": "pyrualean-hardware-recording-v1",
                "config": config.summary(),
                "config_sha256": config.sha256,
                "calibration_sha256": config.calibration_sha256,
                "success_semantics": {
                    "environment_success_known": False,
                    "task_success": "unknown",
                    "note": (
                        "The hardware path has no independent task-success evaluator. A model "
                        "finish(status=success) is a statement, not a measured task outcome; "
                        "environment_success stays false and the verdict is not_measured."
                    ),
                },
            },
        )

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_jsonable(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record(
        self,
        snapshot_payload: Mapping[str, Any],
        artifacts: Mapping[str, bytes],
        *,
        observation: Mapping[str, Any],
        tool: str | None,
    ) -> int:
        index = int(snapshot_payload.get("index"))
        folder = self.root / "steps" / f"{index:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in artifacts.items():
            target = folder / "artifacts" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            self._by_basename[(index, Path(name).name)] = name
        self._write_json(folder / "snapshot.json", snapshot_payload)
        self._write_json(folder / "observation.json", observation)
        self._write_json(
            folder / "camera_meta.json",
            (snapshot_payload.get("reading") or {}).get("cameras") or {},
        )
        self._write_json(
            folder / "device.json",
            (snapshot_payload.get("reading") or {}).get("arms") or {},
        )
        record = {
            "index": index,
            "tool": tool,
            "snapshot": snapshot_payload,
            "observation": observation,
            "artifacts": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for name, data in artifacts.items()
            },
        }
        self.records[index] = record
        self._write_json(
            folder / "index.json",
            {"index": index, "tool": tool, "artifacts": sorted(artifacts)},
        )
        return index

    def latest_index(self) -> int | None:
        return max(self.records) if self.records else None

    def index(self, step: int) -> int:
        if not self.records:
            raise RecordingError("no snapshot recorded")
        if step < 0:
            return max(self.records)
        if step not in self.records:
            raise RecordingError(
                f"step {step} is not recorded (available {min(self.records)}..{max(self.records)})"
            )
        return step

    def payload(self, step: int = -1) -> dict[str, Any]:
        return self.records[self.index(step)]["snapshot"]

    def observation(self, step: int = -1) -> dict[str, Any]:
        return self.records[self.index(step)]["observation"]

    def artifact(self, name: str, step: int = -1) -> bytes:
        index = self.index(step)
        if "/" in name:
            path = self.root / "steps" / f"{index:06d}" / "artifacts" / name
            if path.is_file():
                return path.read_bytes()
        actual = self._by_basename.get((index, Path(name).name))
        if actual is None:
            payload = self.records[index]["snapshot"]
            cameras = (payload.get("reading") or {}).get("cameras") or {}
            for cam in cameras.values():
                for key in ("color_artifact", "depth_artifact"):
                    candidate = cam.get(key)
                    if candidate and Path(candidate).name == name:
                        actual = candidate
                        break
                if actual:
                    break
        if actual is None:
            raise RecordingError(f"step {index} has no artifact named {name!r}")
        path = self.root / "steps" / f"{index:06d}" / "artifacts" / actual
        if not path.is_file():
            raise RecordingError(f"recorded artifact is missing: {path}")
        return path.read_bytes()

    def copy_artifact(self, name: str, target: Path, step: int = -1) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.artifact(name, step))
        return target

    def finalize(
        self,
        *,
        stop: Mapping[str, Any] | None,
        finish: Mapping[str, Any] | None,
        execution: Mapping[str, Any],
        environment_success_known: bool = False,
    ) -> None:
        if self.finalized:
            return
        self.finalized = True
        self._write_json(
            self.root / "episode.json",
            {
                "schema": "pyrualean-hardware-episode-v1",
                "config_sha256": self.config.sha256,
                "calibration_sha256": self.config.calibration_sha256,
                "steps_recorded": sorted(self.records),
                "finish": finish,
                "stop": stop,
                "execution": execution,
                "environment_success_known": bool(environment_success_known),
                "task_success": "not_measured" if not environment_success_known else "unknown",
                "note": (
                    "finish(success) is the model's claim; stop evidence is what the service "
                    "confirmed after shutdown; neither is an independent task-success verdict."
                ),
            },
        )

    def close(self) -> None:
        return


__all__ = ["Recorder", "RecordingError"]
