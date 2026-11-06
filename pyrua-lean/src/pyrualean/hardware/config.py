# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Site configuration and robot calibration for the real LiteArm station.

A station file is a small JSON document.  Every path inside it is resolved
relative to the file's own directory, so a site directory can be copied as a
unit.  The loader is strict on purpose:

* no missing key is filled with a plausible example value;
* numbers must be finite (JSON ``NaN``/``Infinity`` are refused);
* matrices must have the right shape and rotations must be orthonormal;
* joint maps must be permutations with signs ``+1``/``-1``;
* a calibration marked ``"executable": false`` (every example shipped here)
  is refused for ``live`` execution.

The result is immutable enough for the service: the parsed dataclasses are
frozen and the raw JSON is retained only for hashing and diagnostics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import ExecutionMode, content_hash, require_finite

CONFIG_SCHEMA = 1
CALIBRATION_SCHEMA = 1
ARMS = ("left", "right")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

_SECRET_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private_key",
    "credential",
    "credentials",
}


class ConfigError(ValueError):
    """A station file or calibration file is missing or invalid."""


def _path(value: Any, name: str, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ConfigError(f"{name} is required")
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path string")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from None
    try:
        raw = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ConfigError(f"{path}: JSON constant {value!r} is not allowed")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the top level must be a JSON object")
    return raw


def _reject_secret_fields(value: Any, where: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEYS:
                raise ConfigError(
                    f"{where}: credential-like key {key!r} must not be stored in a "
                    "station file; use an operator-managed secret outside this format"
                )
            _reject_secret_fields(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_fields(item, f"{where}[{index}]")


def _obj(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _str(value: Any, name: str, *, default: str | None = None, allow_empty: bool = False) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _arms(value: Any, name: str) -> tuple[str, ...]:
    """Validate an optional arm-name list and return it in canonical order.

    ``None`` means the field was absent; the caller decides the default.  A
    present value must be a non-empty JSON array of distinct names from
    :data:`ARMS`.
    """

    if value is None:
        return ARMS
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a JSON array, got {type(value).__name__}")
    if not value:
        raise ConfigError(f"{name} must name at least one arm")
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{name} entries must be strings")
    unknown = sorted({item for item in value if item not in ARMS})
    if unknown:
        raise ConfigError(f"{name} must contain only {ARMS}, got {unknown}")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in value:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    if duplicates:
        raise ConfigError(f"{name} must not repeat an arm: {sorted(duplicates)}")
    # Canonical order keeps downstream iteration and recorded metadata stable
    # regardless of the order in the JSON array.
    return tuple(arm for arm in ARMS if arm in value)


def _names(
    value: Any,
    name: str,
    allowed: tuple[str, ...],
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate a calibration subset selector and return canonical order.

    ``None`` selects ``default``.  A present selector must name at least one
    known entry, without duplicates.  The result is ordered like ``allowed`` so
    parsing and recorded metadata do not depend on JSON object order.
    """

    if value is None:
        return default
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{name} must be a sequence of names")
    items = tuple(value)
    if not items:
        raise ConfigError(f"{name} must name at least one entry")
    for item in items:
        if not isinstance(item, str):
            raise ConfigError(f"{name} entries must be strings")
    unknown = sorted(set(items) - set(allowed))
    if unknown:
        raise ConfigError(f"{name} must contain only {allowed}, got {unknown}")
    duplicates = sorted({item for item in items if items.count(item) > 1})
    if duplicates:
        raise ConfigError(f"{name} must not repeat an entry: {duplicates}")
    return tuple(item for item in allowed if item in items)


def _num(value: Any, name: str, *, low: float | None = None, high: float | None = None) -> float:
    try:
        out = require_finite(value, name)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if low is not None and out < low:
        raise ConfigError(f"{name} must be >= {low}, got {out}")
    if high is not None and out > high:
        raise ConfigError(f"{name} must be <= {high}, got {out}")
    return out


def _int(value: Any, name: str, *, low: int | None = None, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if low is not None and value < low:
        raise ConfigError(f"{name} must be >= {low}, got {value}")
    if high is not None and value > high:
        raise ConfigError(f"{name} must be <= {high}, got {value}")
    return value


def _vec(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ConfigError(f"{name} must be a list of {length} numbers")
    return tuple(_num(v, f"{name}[{i}]") for i, v in enumerate(value))


def _matrix(value: Any, name: str, rows: int, cols: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise ConfigError(f"{name} must be a {rows}x{cols} matrix")
    out = np.empty((rows, cols), dtype=float)
    for r, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            raise ConfigError(f"{name}[{r}] must be a list of {cols} numbers")
        for c, entry in enumerate(row):
            out[r, c] = _num(entry, f"{name}[{r}][{c}]")
    return out


def _transform(value: Any, name: str) -> np.ndarray:
    t = _matrix(value, name, 4, 4)
    if not np.allclose(t[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ConfigError(f"{name}: the bottom row must be [0, 0, 0, 1]")
    r = t[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-5):
        raise ConfigError(f"{name}: the rotation block is not orthonormal")
    if abs(float(np.linalg.det(r)) - 1.0) > 1e-5:
        raise ConfigError(f"{name}: the rotation determinant must be +1")
    return t


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _require_file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise ConfigError(f"{name} does not exist: {path}")
    return path


@dataclass(frozen=True)
class ServiceConfig:
    host: str
    port: int
    connect_timeout_s: float
    request_timeout_s: float
    heartbeat_s: float
    heartbeat_timeout_s: float
    max_clock_skew_s: float


@dataclass(frozen=True)
class ExecutionConfig:
    mode: ExecutionMode
    #: Arms that are physically present and may be opened, read and commanded.
    #: ``execution.active_arms`` is optional and defaults to both arms.
    active_arms: tuple[str, ...]
    max_speed: float
    max_joint_speed_rad_s: float
    max_tcp_speed_m_s: float
    max_joint_step_rad: float
    max_segment_duration_s: float
    settle_tolerance_rad: float
    settle_timeout_s: float
    command_timeout_s: float
    start_tolerance_rad: float


@dataclass(frozen=True)
class GripperDeviceConfig:
    driver: str
    channel: str
    can_id: int
    calibration_path: str | None
    force_n: float
    grasp_timeout_s: float
    hold_clearance_mm: float
    open_speed_mm_s: float
    close_speed_mm_s: float


@dataclass(frozen=True)
class ArmDeviceConfig:
    sdk: str
    port: str | None
    min_firmware: str
    connect_timeout_s: float
    move_timeout_s: float
    read_timeout_s: float


@dataclass(frozen=True)
class ArmConfig:
    arm: ArmDeviceConfig
    #: ``None`` means this arm is present without a gripper.  Arm-only stations
    #: never fabricate a gripper: read/status report the absence and gripper
    #: commands are refused.
    gripper: GripperDeviceConfig | None


@dataclass(frozen=True)
class CameraConfig:
    name: str
    driver: str
    index: int | None
    serial: str | None
    width: int
    height: int
    fps: int
    depth: bool
    timeout_s: float
    #: Explicit OpenCV device path (for example ``/dev/video4``).  When set,
    #: the opencv driver opens it with ``cv2.CAP_V4L2``; ``index`` is then
    #: ignored.  Integer indices cannot address every V4L2 node.
    device: str | None = None
    #: Optional four-character V4L2 pixel format (for example ``"YUYV"``),
    #: opencv driver only.
    fourcc: str | None = None


@dataclass(frozen=True)
class Sam3Config:
    mode: str
    endpoint: str | None
    timeout_s: float
    checkpoint: str | None
    cuda_device: int | None
    min_score: float
    max_object_height_m: float
    voxel_m: float
    max_voxels: int
    depth_min_valid_ratio: float


@dataclass(frozen=True)
class SnapshotConfig:
    max_age_s: float
    camera_joint_sync_s: float
    refresh_min_interval_s: float
    history: int


@dataclass(frozen=True)
class ResultsConfig:
    directory: str
    max_episode_actions: int
    max_wall_s: float


@dataclass(frozen=True)
class SafetyConfig:
    stop_timeout_s: float
    joint_speed_confirm_rad_s: float
    home_lift_m: float
    require_operator_reset_after_fault: bool


@dataclass(frozen=True)
class StationConfig:
    path: Path
    raw: Mapping[str, Any]
    sha256: str
    schema_version: int
    station_id: str
    calibration_path: Path
    calibration_sha256: str
    service: ServiceConfig
    execution: ExecutionConfig
    arms: Mapping[str, ArmConfig]
    cameras: Mapping[str, CameraConfig]
    primary_camera: str
    sam3: Sam3Config
    snapshot: SnapshotConfig
    results: ResultsConfig
    safety: SafetyConfig
    calibration: Calibration

    @property
    def active_arms(self) -> tuple[str, ...]:
        """Arms this station may open and command, in canonical order."""

        return self.execution.active_arms

    @property
    def gripper_arms(self) -> tuple[str, ...]:
        """Active arms that actually have a gripper configured."""

        return tuple(
            arm
            for arm in self.execution.active_arms
            if arm in self.arms and self.arms[arm].gripper is not None
        )

    def summary(self) -> dict[str, Any]:
        """Non-secret, JSON-ready summary recorded with every episode."""

        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "station_id": self.station_id,
            "calibration": {
                "path": str(self.calibration_path),
                "sha256": self.calibration_sha256,
                "version": self.calibration.version,
                "executable": self.calibration.executable,
                "quality": self.calibration.quality,
            },
            "execution": {
                "mode": self.execution.mode.value,
                "active_arms": list(self.execution.active_arms),
                "max_speed": self.execution.max_speed,
                "max_joint_speed_rad_s": self.execution.max_joint_speed_rad_s,
                "max_tcp_speed_m_s": self.execution.max_tcp_speed_m_s,
                "max_joint_step_rad": self.execution.max_joint_step_rad,
                "max_segment_duration_s": self.execution.max_segment_duration_s,
                "command_timeout_s": self.execution.command_timeout_s,
            },
            "service": {"host": self.service.host, "port": self.service.port},
            "arms": {
                name: {
                    "sdk": cfg.arm.sdk,
                    "port": cfg.arm.port,
                    "min_firmware": cfg.arm.min_firmware,
                    "has_gripper": cfg.gripper is not None,
                    "gripper": (
                        None
                        if cfg.gripper is None
                        else {
                            "driver": cfg.gripper.driver,
                            "channel": cfg.gripper.channel,
                            "can_id": cfg.gripper.can_id,
                            "force_n": cfg.gripper.force_n,
                            "calibration_path": cfg.gripper.calibration_path,
                        }
                    ),
                }
                for name, cfg in self.arms.items()
            },
            "cameras": {
                name: {
                    "driver": cfg.driver,
                    "index": cfg.index,
                    "device": cfg.device,
                    "fourcc": cfg.fourcc,
                    "serial": cfg.serial,
                    "resolution": [cfg.height, cfg.width],
                    "fps": cfg.fps,
                    "depth": cfg.depth,
                }
                for name, cfg in self.cameras.items()
            },
            "perception": {
                "primary_camera": self.primary_camera,
                "sam3_mode": self.sam3.mode,
                "sam3_endpoint": self.sam3.endpoint,
                "sam3_checkpoint": self.sam3.checkpoint,
            },
            "results_directory": self.results.directory,
        }


@dataclass(frozen=True)
class JointMapEntry:
    model_joint: int
    sdk_index: int
    sign: float
    offset_rad: float


@dataclass(frozen=True)
class GripperCalibration:
    closed_m: float
    open_m: float
    force_n: float
    min_holding_m: float
    max_closing_speed_m_s: float

    @property
    def span_m(self) -> float:
        return self.open_m - self.closed_m

    def width_to_opening(self, width_m: float) -> float:
        return (float(width_m) - self.closed_m) / self.span_m

    def opening_to_width(self, opening: float) -> float:
        return self.closed_m + self.span_m * float(opening)


@dataclass(frozen=True)
class ToolCollision:
    body_length_m: float
    body_radius_m: float
    tip_offset_m: float
    finger_half_width_m: float
    finger_half_thickness_m: float


@dataclass(frozen=True)
class ArmCalibration:
    arm: str
    base_pose: np.ndarray
    flange_to_tcp: np.ndarray
    joint_map: tuple[JointMapEntry, ...]
    joint_limits_rad: tuple[tuple[float, float], ...]
    safe_home_rad: tuple[float, ...]
    gripper: GripperCalibration
    tool_collision: ToolCollision
    link_radii_m: tuple[float, ...]

    def sdk_from_model(self, q_model: Any) -> np.ndarray:
        q = np.asarray(q_model, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"arm {self.arm}: expected 7 joint angles, got {q.shape}")
        out = np.empty(7, dtype=float)
        for entry in self.joint_map:
            out[entry.sdk_index] = entry.sign * q[entry.model_joint] + entry.offset_rad
        return out

    def model_from_sdk(self, q_sdk: Any) -> np.ndarray:
        q = np.asarray(q_sdk, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"arm {self.arm}: expected 7 SDK joint angles, got {q.shape}")
        out = np.empty(7, dtype=float)
        for entry in self.joint_map:
            out[entry.model_joint] = (q[entry.sdk_index] - entry.offset_rad) / entry.sign
        return out


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    arm: str | None
    mount: str
    transform: np.ndarray
    intrinsic: np.ndarray
    distortion: tuple[float, ...]
    resolution: tuple[int, int]
    has_depth: bool

    @property
    def width(self) -> int:
        return int(self.resolution[1])

    @property
    def height(self) -> int:
        return int(self.resolution[0])


@dataclass(frozen=True)
class TableCalibration:
    z_m: float
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    edge_margin_m: float


@dataclass(frozen=True)
class ObstacleCalibration:
    name: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


@dataclass(frozen=True)
class Calibration:
    path: Path
    raw: Mapping[str, Any]
    sha256: str
    schema_version: int
    station_id: str
    version: str
    executable: bool
    source: str
    created_at: str
    quality: Mapping[str, Any]
    arms: Mapping[str, ArmCalibration]
    cameras: Mapping[str, CameraCalibration]
    table: TableCalibration
    obstacles: tuple[ObstacleCalibration, ...]

    def camera(self, name: str) -> CameraCalibration:
        try:
            return self.cameras[name]
        except KeyError:
            raise ConfigError(f"camera {name!r} is not calibrated") from None

    def arm(self, name: str) -> ArmCalibration:
        try:
            return self.arms[name]
        except KeyError:
            raise ConfigError(f"arm {name!r} is not calibrated") from None

    def require_live_execution(
        self,
        *,
        arms: Sequence[str] | None = None,
        cameras: Sequence[str] | None = None,
    ) -> None:
        """Refuse live motion unless this calibration covers the station.

        ``arms``/``cameras`` default to every arm/camera for backward
        compatibility.  A station that uses only some arms or cameras passes
        exactly those; entries for physically absent devices are not required.
        """

        if not self.executable:
            raise ConfigError(
                f"calibration {self.path} is marked executable=false "
                f"({self.quality.get('status', 'unverified')}); live motion is refused"
            )
        wanted_arms = _names(arms, "arms", ARMS, default=ARMS)
        wanted_cameras = _names(cameras, "cameras", CAMERAS, default=CAMERAS)
        missing_arms = [a for a in wanted_arms if a not in self.arms]
        if missing_arms:
            raise ConfigError(f"calibration is missing arms: {missing_arms}")
        missing = [c for c in wanted_cameras if c not in self.cameras]
        if missing:
            raise ConfigError(f"calibration is missing cameras: {missing}")


def _joint_map(value: Any, arm: str) -> tuple[JointMapEntry, ...]:
    if not isinstance(value, list) or len(value) != 7:
        raise ConfigError(f"arms.{arm}.joint_map must list 7 entries")
    entries: list[JointMapEntry] = []
    for i, raw in enumerate(value):
        item = _obj(raw, f"arms.{arm}.joint_map[{i}]")
        entry = JointMapEntry(
            model_joint=_int(
                item.get("model_joint"), f"arms.{arm}.joint_map[{i}].model_joint", low=0, high=6
            ),
            sdk_index=_int(
                item.get("sdk_index"), f"arms.{arm}.joint_map[{i}].sdk_index", low=0, high=6
            ),
            sign=_num(item.get("sign"), f"arms.{arm}.joint_map[{i}].sign"),
            offset_rad=_num(item.get("offset_rad", 0.0), f"arms.{arm}.joint_map[{i}].offset_rad"),
        )
        if entry.sign not in (-1.0, 1.0):
            raise ConfigError(f"arms.{arm}.joint_map[{i}].sign must be +1 or -1")
        entries.append(entry)
    if sorted(e.model_joint for e in entries) != list(range(7)):
        raise ConfigError(f"arms.{arm}.joint_map must use model joints 0..6 exactly once")
    if sorted(e.sdk_index for e in entries) != list(range(7)):
        raise ConfigError(f"arms.{arm}.joint_map must use SDK indices 0..6 exactly once")
    return tuple(entries)


def _limits(value: Any, arm: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) != 7:
        raise ConfigError(f"arms.{arm}.joint_limits_rad must list 7 [lower, upper] pairs")
    out = []
    for i, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ConfigError(f"arms.{arm}.joint_limits_rad[{i}] must be [lower, upper]")
        lower = _num(pair[0], f"arms.{arm}.joint_limits_rad[{i}][0]")
        upper = _num(pair[1], f"arms.{arm}.joint_limits_rad[{i}][1]")
        if not lower < upper:
            raise ConfigError(f"arms.{arm}.joint_limits_rad[{i}] has lower >= upper")
        out.append((lower, upper))
    return tuple(out)


def load_calibration(
    path: str | Path,
    *,
    station_id: str | None = None,
    arms: Sequence[str] | None = None,
    cameras: Sequence[str] | None = None,
) -> Calibration:
    """Load a calibration file, optionally for a station subset.

    ``arms``/``cameras`` are the arms and cameras the station actually uses.
    They default to all arms/all cameras.  In either case the file must contain
    exactly the selected entries: this loader never synthesises a placeholder
    for an inactive arm or an unconfigured camera, and a stale extra entry is
    rejected rather than silently ignored.
    """

    location = Path(path).expanduser().resolve()
    raw = _read_json(_require_file(location, "calibration"))
    _reject_secret_fields(raw, "calibration")
    if _int(raw.get("schema_version"), "calibration.schema_version") != CALIBRATION_SCHEMA:
        raise ConfigError(
            f"calibration.schema_version must be {CALIBRATION_SCHEMA}, "
            f"got {raw.get('schema_version')!r}"
        )
    identity = _str(raw.get("station_id"), "calibration.station_id")
    if station_id is not None and identity != station_id:
        raise ConfigError(
            f"calibration station_id {identity!r} does not match station {station_id!r}"
        )
    executable = _bool(raw.get("executable"), "calibration.executable")
    quality = _obj(raw.get("quality", {}), "calibration.quality")
    if executable and quality.get("status") in (None, "unverified", "example"):
        raise ConfigError(
            "calibration.executable=true needs a measured quality status, not "
            f"{quality.get('status')!r}"
        )
    selected_arms = _names(arms, "calibration.arms", ARMS, default=ARMS)
    selected_cameras = _names(cameras, "calibration.cameras", CAMERAS, default=CAMERAS)
    arms_raw = _obj(raw.get("arms"), "calibration.arms")
    if set(arms_raw) != set(selected_arms):
        raise ConfigError(
            f"calibration.arms must have exactly {selected_arms}, got {sorted(arms_raw)}"
        )
    parsed_arms: dict[str, ArmCalibration] = {}
    for arm in selected_arms:
        item = _obj(arms_raw[arm], f"calibration.arms.{arm}")
        gripper_raw = _obj(item.get("gripper"), f"calibration.arms.{arm}.gripper")
        closed = _num(
            gripper_raw.get("closed_m"), f"arms.{arm}.gripper.closed_m", low=0.0, high=0.2
        )
        opened = _num(gripper_raw.get("open_m"), f"arms.{arm}.gripper.open_m", low=0.0, high=0.2)
        if not closed < opened:
            raise ConfigError(f"arms.{arm}.gripper.closed_m must be < open_m")
        gripper = GripperCalibration(
            closed_m=closed,
            open_m=opened,
            force_n=_num(
                gripper_raw.get("force_n", 15.0), f"arms.{arm}.gripper.force_n", low=0.0, high=100.0
            ),
            min_holding_m=_num(
                gripper_raw.get("min_holding_m", 0.002),
                f"arms.{arm}.gripper.min_holding_m",
                low=0.0,
                high=0.05,
            ),
            max_closing_speed_m_s=_num(
                gripper_raw.get("max_closing_speed_m_s", 0.05),
                f"arms.{arm}.gripper.max_closing_speed_m_s",
                low=0.0,
                high=0.2,
            ),
        )
        collision_raw = _obj(
            item.get("tool_collision", {}), f"calibration.arms.{arm}.tool_collision"
        )
        tool = ToolCollision(
            body_length_m=_num(
                collision_raw.get("body_length_m", 0.13),
                f"arms.{arm}.tool_collision.body_length_m",
                low=0.0,
                high=0.5,
            ),
            body_radius_m=_num(
                collision_raw.get("body_radius_m", 0.035),
                f"arms.{arm}.tool_collision.body_radius_m",
                low=0.0,
                high=0.2,
            ),
            tip_offset_m=_num(
                collision_raw.get("tip_offset_m", 0.046),
                f"arms.{arm}.tool_collision.tip_offset_m",
                low=0.0,
                high=0.3,
            ),
            finger_half_width_m=_num(
                collision_raw.get("finger_half_width_m", 0.003),
                f"arms.{arm}.tool_collision.finger_half_width_m",
                low=0.0,
                high=0.1,
            ),
            finger_half_thickness_m=_num(
                collision_raw.get("finger_half_thickness_m", 0.025),
                f"arms.{arm}.tool_collision.finger_half_thickness_m",
                low=0.0,
                high=0.1,
            ),
        )
        radii_raw = item.get("link_radii_m", [0.045] * 7)
        if not isinstance(radii_raw, list) or len(radii_raw) != 7:
            raise ConfigError(f"arms.{arm}.link_radii_m must list 7 radii")
        radii = tuple(
            _num(v, f"arms.{arm}.link_radii_m[{i}]", low=0.0, high=0.3)
            for i, v in enumerate(radii_raw)
        )
        parsed_arms[arm] = ArmCalibration(
            arm=arm,
            base_pose=_transform(item.get("base_pose"), f"arms.{arm}.base_pose"),
            flange_to_tcp=_transform(item.get("flange_to_tcp"), f"arms.{arm}.flange_to_tcp"),
            joint_map=_joint_map(item.get("joint_map"), arm),
            joint_limits_rad=_limits(item.get("joint_limits_rad"), arm),
            safe_home_rad=_vec(item.get("safe_home_rad"), f"arms.{arm}.safe_home_rad", 7),
            gripper=gripper,
            tool_collision=tool,
            link_radii_m=radii,
        )
        home = np.asarray(parsed_arms[arm].safe_home_rad)
        for i, (lo, hi) in enumerate(parsed_arms[arm].joint_limits_rad):
            if not lo <= home[i] <= hi:
                raise ConfigError(
                    f"arms.{arm}.safe_home_rad[{i}]={home[i]:.4f} is outside [{lo:.4f}, {hi:.4f}]"
                )

    cams_raw = _obj(raw.get("cameras"), "calibration.cameras")
    if set(cams_raw) != set(selected_cameras):
        raise ConfigError(
            f"calibration.cameras must have exactly {selected_cameras}, got {sorted(cams_raw)}"
        )
    parsed_cameras: dict[str, CameraCalibration] = {}
    for name in selected_cameras:
        item = _obj(cams_raw[name], f"calibration.cameras.{name}")
        mount = _str(item.get("mount"), f"calibration.cameras.{name}.mount")
        if mount not in ("world", "tool"):
            raise ConfigError(f"calibration.cameras.{name}.mount must be 'world' or 'tool'")
        arm = item.get("arm")
        if mount == "tool" and arm not in ARMS:
            raise ConfigError(f"calibration.cameras.{name}.arm must be left/right for a tool mount")
        if mount == "world" and arm is not None:
            raise ConfigError(f"calibration.cameras.{name}.arm is only valid for a tool mount")
        transform_key = "camera_to_world" if mount == "world" else "camera_to_tool"
        intrinsic = _matrix(
            item.get("intrinsic_matrix"), f"calibration.cameras.{name}.intrinsic_matrix", 3, 3
        )
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ConfigError(f"calibration.cameras.{name}: fx and fy must be positive")
        if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-9):
            raise ConfigError(f"calibration.cameras.{name}: intrinsic bottom row must be [0,0,1]")
        resolution = item.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ConfigError(f"calibration.cameras.{name}.resolution must be [height, width]")
        height = _int(resolution[0], f"calibration.cameras.{name}.resolution[0]", low=1)
        width = _int(resolution[1], f"calibration.cameras.{name}.resolution[1]", low=1)
        distortion = item.get("distortion_coefficients", [0.0] * 5)
        if not isinstance(distortion, list) or len(distortion) not in (4, 5, 8, 12, 14):
            raise ConfigError(
                f"calibration.cameras.{name}.distortion_coefficients must be a 4/5/8/12/14 list"
            )
        parsed_cameras[name] = CameraCalibration(
            name=name,
            arm=arm,
            mount=mount,
            transform=_transform(item.get(transform_key), f"cameras.{name}.{transform_key}"),
            intrinsic=intrinsic,
            distortion=tuple(
                _num(v, f"cameras.{name}.distortion_coefficients[{i}]")
                for i, v in enumerate(distortion)
            ),
            resolution=(height, width),
            has_depth=_bool(item.get("has_depth", False), f"cameras.{name}.has_depth"),
        )

    table_raw = _obj(raw.get("table"), "calibration.table")
    table = TableCalibration(
        z_m=_num(table_raw.get("z_m"), "table.z_m"),
        x_min_m=_num(table_raw.get("x_min_m"), "table.x_min_m"),
        x_max_m=_num(table_raw.get("x_max_m"), "table.x_max_m"),
        y_min_m=_num(table_raw.get("y_min_m"), "table.y_min_m"),
        y_max_m=_num(table_raw.get("y_max_m"), "table.y_max_m"),
        edge_margin_m=_num(table_raw.get("edge_margin_m", 0.02), "table.edge_margin_m", low=0.0),
    )
    if not table.x_min_m < table.x_max_m or not table.y_min_m < table.y_max_m:
        raise ConfigError("table bounds must have min < max on both axes")
    obstacles_raw = raw.get("obstacles", [])
    if not isinstance(obstacles_raw, list):
        raise ConfigError("calibration.obstacles must be a list")
    obstacles = []
    for i, item in enumerate(obstacles_raw):
        obj = _obj(item, f"obstacles[{i}]")
        minimum = _vec(obj.get("min"), f"obstacles[{i}].min", 3)
        maximum = _vec(obj.get("max"), f"obstacles[{i}].max", 3)
        if any(a >= b for a, b in zip(minimum, maximum, strict=True)):
            raise ConfigError(f"obstacles[{i}] min must be below max on every axis")
        obstacles.append(
            ObstacleCalibration(
                name=_str(obj.get("name", f"obstacle_{i}"), f"obstacles[{i}].name"),
                minimum=minimum,
                maximum=maximum,
            )
        )

    return Calibration(
        path=location,
        raw=raw,
        sha256=content_hash(raw),
        schema_version=CALIBRATION_SCHEMA,
        station_id=identity,
        version=_str(raw.get("calibration_version"), "calibration.calibration_version"),
        executable=executable,
        source=_str(raw.get("source", "unspecified"), "calibration.source"),
        created_at=_str(
            raw.get("created_at", ""), "calibration.created_at", default="", allow_empty=True
        ),
        quality=quality,
        arms=parsed_arms,
        cameras=parsed_cameras,
        table=table,
        obstacles=tuple(obstacles),
    )


def _gripper_device(value: Any, name: str, base: Path) -> GripperDeviceConfig | None:
    """Parse ``devices.<arm>.gripper``.

    An absent key or an explicit ``null`` means this arm has no gripper.  Any
    present object is parsed strictly; nothing is invented for an arm-only
    station.
    """

    if value is None:
        return None
    grip_raw = _obj(value, name)
    grip_calib = grip_raw.get("calibration_path")
    grip_path = (
        str(_resolve(base, _path(grip_calib, f"{name}.calibration_path")))
        if grip_calib is not None
        else None
    )
    # The path is optional: the LiteGrip SDK ships a factory calibration and
    # ``load_calibration()`` without a path uses it.  A path is needed only for
    # a measured/per-channel file (notably a gripper on can1, because the
    # shipped factory file declares channel can0).  The device wrapper checks
    # that the loaded configuration is actually calibrated.
    if grip_path:
        _require_file(Path(grip_path), f"{name}.calibration_path")
    driver = _str(grip_raw.get("driver", "litegrip"), f"{name}.driver", default="litegrip")
    if driver != "litegrip":
        raise ConfigError(f"{name}.driver must be 'litegrip' in this version, got {driver!r}")
    return GripperDeviceConfig(
        driver=driver,
        channel=_str(grip_raw.get("channel"), f"{name}.channel"),
        can_id=_int(grip_raw.get("can_id", 8), f"{name}.can_id", low=0, high=127),
        calibration_path=grip_path,
        force_n=_num(grip_raw.get("force_n", 15.0), f"{name}.force_n", low=0.0, high=100.0),
        grasp_timeout_s=_num(
            grip_raw.get("grasp_timeout_s", 5.0), f"{name}.grasp_timeout_s", low=0.1
        ),
        hold_clearance_mm=_num(
            grip_raw.get("hold_clearance_mm", 8.0), f"{name}.hold_clearance_mm", low=0.0, high=30.0
        ),
        open_speed_mm_s=_num(
            grip_raw.get("open_speed_mm_s", 40.0), f"{name}.open_speed_mm_s", low=1.0, high=80.0
        ),
        close_speed_mm_s=_num(
            grip_raw.get("close_speed_mm_s", 40.0), f"{name}.close_speed_mm_s", low=1.0, high=80.0
        ),
    )


def load_station(path: str | Path, *, mode: ExecutionMode | str | None = None) -> StationConfig:
    location = Path(path).expanduser().resolve()
    raw = _read_json(_require_file(location, "station config"))
    _reject_secret_fields(raw)
    if _int(raw.get("schema_version"), "schema_version") != CONFIG_SCHEMA:
        raise ConfigError(
            f"schema_version must be {CONFIG_SCHEMA}, got {raw.get('schema_version')!r}"
        )
    station_id = _str(raw.get("station_id"), "station_id")
    calib_value = _path(raw.get("calibration"), "calibration")
    calib_path = _resolve(location.parent, calib_value)

    svc = _obj(raw.get("service"), "service")
    service = ServiceConfig(
        host=_str(svc.get("host", "127.0.0.1"), "service.host", default="127.0.0.1"),
        port=_int(svc.get("port"), "service.port", low=1, high=65535),
        connect_timeout_s=_num(
            svc.get("connect_timeout_s", 5.0), "service.connect_timeout_s", low=0.05
        ),
        request_timeout_s=_num(
            svc.get("request_timeout_s", 15.0), "service.request_timeout_s", low=0.05
        ),
        heartbeat_s=_num(svc.get("heartbeat_s", 2.0), "service.heartbeat_s", low=0.05),
        heartbeat_timeout_s=_num(
            svc.get("heartbeat_timeout_s", 6.0), "service.heartbeat_timeout_s", low=0.1
        ),
        max_clock_skew_s=_num(
            svc.get("max_clock_skew_s", 0.5), "service.max_clock_skew_s", low=0.0, high=60.0
        ),
    )
    if service.heartbeat_timeout_s <= service.heartbeat_s:
        raise ConfigError("service.heartbeat_timeout_s must exceed service.heartbeat_s")

    exec_raw = _obj(raw.get("execution"), "execution")
    mode_value = mode.value if isinstance(mode, ExecutionMode) else (mode or exec_raw.get("mode"))
    try:
        execution_mode = ExecutionMode(mode_value)
    except ValueError:
        raise ConfigError(
            f"execution.mode must be one of {[m.value for m in ExecutionMode]}, got {mode_value!r}"
        ) from None
    execution = ExecutionConfig(
        mode=execution_mode,
        active_arms=_arms(exec_raw.get("active_arms"), "execution.active_arms"),
        max_speed=_num(exec_raw.get("max_speed", 0.5), "execution.max_speed", low=0.0, high=1.0),
        max_joint_speed_rad_s=_num(
            exec_raw.get("max_joint_speed_rad_s", 0.5),
            "execution.max_joint_speed_rad_s",
            low=0.001,
            high=10.0,
        ),
        max_tcp_speed_m_s=_num(
            exec_raw.get("max_tcp_speed_m_s", 0.10),
            "execution.max_tcp_speed_m_s",
            low=0.001,
            high=2.0,
        ),
        max_joint_step_rad=_num(
            exec_raw.get("max_joint_step_rad", 0.20),
            "execution.max_joint_step_rad",
            low=0.001,
            high=1.0,
        ),
        max_segment_duration_s=_num(
            exec_raw.get("max_segment_duration_s", 4.0),
            "execution.max_segment_duration_s",
            low=0.05,
        ),
        settle_tolerance_rad=_num(
            exec_raw.get("settle_tolerance_rad", 0.01),
            "execution.settle_tolerance_rad",
            low=0.0,
            high=0.2,
        ),
        settle_timeout_s=_num(
            exec_raw.get("settle_timeout_s", 3.0), "execution.settle_timeout_s", low=0.0
        ),
        command_timeout_s=_num(
            exec_raw.get("command_timeout_s", 60.0), "execution.command_timeout_s", low=0.5
        ),
        start_tolerance_rad=_num(
            exec_raw.get("start_tolerance_rad", 0.02),
            "execution.start_tolerance_rad",
            low=0.0,
            high=0.5,
        ),
    )

    # The calibration must describe exactly the devices this station uses: the
    # active arms and the cameras it configures.  Parse those selectors before
    # loading so an arm-only / wrist-only site does not need fabricated entries
    # for absent hardware.
    cams_raw = _obj(raw.get("cameras"), "cameras")
    if not cams_raw:
        raise ConfigError("cameras must configure at least one camera")
    unknown_cameras = sorted(set(cams_raw) - set(CAMERAS))
    if unknown_cameras:
        raise ConfigError(f"camera names must be among {CAMERAS}, got unknown {unknown_cameras}")
    configured_cameras = tuple(name for name in CAMERAS if name in cams_raw)
    calibration = load_calibration(
        calib_path,
        station_id=station_id,
        arms=execution.active_arms,
        cameras=configured_cameras,
    )

    if execution_mode is ExecutionMode.LIVE:
        calibration.require_live_execution(arms=execution.active_arms, cameras=configured_cameras)
        for arm_name in execution.active_arms:
            arm_raw = calibration.raw["arms"][arm_name]
            missing = [
                key
                for key in ("tool_collision", "link_radii_m", "safe_home_rad", "joint_map")
                if key not in arm_raw
            ]
            if missing:
                raise ConfigError(
                    f"calibration.arms.{arm_name} is missing safety-critical fields "
                    f"for live execution: {missing}"
                )
    if execution_mode is not ExecutionMode.OFFLINE_FAKE:
        missing = [
            name
            for name, cam in calibration.cameras.items()
            if cam.mount == "tool" and cam.arm is None
        ]
        if missing:
            raise ConfigError(f"tool-mounted cameras lack an arm: {missing}")

    devices_raw = _obj(raw.get("devices"), "devices")
    if not devices_raw:
        raise ConfigError("devices must configure at least one arm")
    unknown_devices = sorted(set(devices_raw) - set(ARMS))
    if unknown_devices:
        raise ConfigError(f"devices may only name {ARMS}, got unknown {unknown_devices}")
    missing_active = [arm for arm in execution.active_arms if arm not in devices_raw]
    if missing_active:
        raise ConfigError(
            "execution.active_arms names arms with no devices entry: "
            f"{missing_active}; add devices.{missing_active[0]} or remove it from active_arms"
        )
    arms: dict[str, ArmConfig] = {}
    for arm in ARMS:
        if arm not in devices_raw:
            # An inactive arm may be absent from the station entirely.
            continue
        item = _obj(devices_raw[arm], f"devices.{arm}")
        arm_raw = _obj(item.get("arm"), f"devices.{arm}.arm")
        gripper = _gripper_device(item.get("gripper"), f"devices.{arm}.gripper", location.parent)
        arms[arm] = ArmConfig(
            arm=ArmDeviceConfig(
                sdk=_str(
                    arm_raw.get("sdk", "litearm"), f"devices.{arm}.arm.sdk", default="litearm"
                ),
                port=arm_raw.get("port"),
                min_firmware=_str(
                    arm_raw.get("min_firmware", "Litearm1.5.0"),
                    f"devices.{arm}.arm.min_firmware",
                    default="Litearm1.5.0",
                ),
                connect_timeout_s=_num(
                    arm_raw.get("connect_timeout_s", 5.0),
                    f"devices.{arm}.arm.connect_timeout_s",
                    low=0.1,
                ),
                move_timeout_s=_num(
                    arm_raw.get("move_timeout_s", 20.0),
                    f"devices.{arm}.arm.move_timeout_s",
                    low=0.1,
                ),
                read_timeout_s=_num(
                    arm_raw.get("read_timeout_s", 1.0),
                    f"devices.{arm}.arm.read_timeout_s",
                    low=0.05,
                ),
            ),
            gripper=gripper,
        )
        if arms[arm].arm.port is not None and not isinstance(arms[arm].arm.port, str):
            raise ConfigError(f"devices.{arm}.arm.port must be a string or null")
    if execution_mode is ExecutionMode.LIVE:
        unset = [arm for arm in execution.active_arms if not arms[arm].arm.port]
        if unset:
            raise ConfigError(
                f"live execution needs an explicit serial port for every arm; missing: {unset}"
            )

    cameras: dict[str, CameraConfig] = {}
    for name in configured_cameras:
        item = cams_raw[name]
        obj = _obj(item, f"cameras.{name}")
        driver = _str(obj.get("driver"), f"cameras.{name}.driver")
        if driver not in ("opencv", "realsense", "fake"):
            raise ConfigError(f"cameras.{name}.driver must be opencv, realsense or fake")
        if driver == "fake" and execution_mode is not ExecutionMode.OFFLINE_FAKE:
            raise ConfigError(f"cameras.{name}: fake driver is only allowed in offline-fake mode")
        device = obj.get("device")
        if device is not None:
            device = _str(device, f"cameras.{name}.device")
        fourcc = obj.get("fourcc")
        if fourcc is not None:
            fourcc = _str(fourcc, f"cameras.{name}.fourcc")
            invalid_fourcc = (
                len(fourcc) != 4
                or not fourcc.isascii()
                or not fourcc.isprintable()
                or " " in fourcc
            )
            if invalid_fourcc:
                raise ConfigError(
                    f"cameras.{name}.fourcc must be exactly four printable ASCII characters"
                )
        if driver != "opencv":
            if device is not None:
                raise ConfigError(f"cameras.{name}.device is only valid for the opencv driver")
            if fourcc is not None:
                raise ConfigError(f"cameras.{name}.fourcc is only valid for the opencv driver")
        elif device is None and obj.get("index") is None:
            raise ConfigError(f"cameras.{name}: opencv needs an index or a device path")
        cal = calibration.camera(name)
        cameras[name] = CameraConfig(
            name=name,
            driver=driver,
            index=(
                _int(obj.get("index"), f"cameras.{name}.index", low=0)
                if obj.get("index") is not None
                else None
            ),
            serial=obj.get("serial"),
            width=_int(obj.get("width", cal.width), f"cameras.{name}.width", low=1),
            height=_int(obj.get("height", cal.height), f"cameras.{name}.height", low=1),
            fps=_int(obj.get("fps", 30), f"cameras.{name}.fps", low=1, high=240),
            depth=_bool(obj.get("depth", cal.has_depth), f"cameras.{name}.depth"),
            timeout_s=_num(obj.get("timeout_s", 2.0), f"cameras.{name}.timeout_s", low=0.05),
            device=device,
            fourcc=fourcc,
        )
        if cameras[name].width != cal.width or cameras[name].height != cal.height:
            raise ConfigError(
                f"cameras.{name}: configured resolution {cameras[name].height}x"
                f"{cameras[name].width} does not match calibration "
                f"{cal.height}x{cal.width}; recalibrate or fix the configuration"
            )
        if cameras[name].depth and not cal.has_depth:
            raise ConfigError(
                f"cameras.{name}: depth=true but the calibration says has_depth=false"
            )

    sam_raw = _obj(raw.get("perception", {}), "perception")
    sam3_raw = _obj(sam_raw.get("sam3", {}), "perception.sam3")
    checkpoint = sam3_raw.get("checkpoint")
    if checkpoint is not None:
        checkpoint = str(_resolve(location.parent, _path(checkpoint, "perception.sam3.checkpoint")))
    mode = _str(sam3_raw.get("mode", "local"), "perception.sam3.mode", default="local")
    if mode not in ("http", "local"):
        raise ConfigError(f"perception.sam3.mode must be 'http' or 'local', got {mode!r}")
    endpoint = sam3_raw.get("endpoint")
    if mode == "http":
        endpoint = _str(endpoint, "perception.sam3.endpoint")
        if not endpoint.startswith(("http://", "https://")):
            endpoint = "http://" + endpoint
    else:
        endpoint = None
    sam3 = Sam3Config(
        mode=mode,
        endpoint=endpoint,
        timeout_s=_num(sam3_raw.get("timeout_s", 60.0), "perception.sam3.timeout_s", low=0.1),
        checkpoint=checkpoint,
        cuda_device=(
            _int(sam3_raw.get("cuda_device"), "perception.sam3.cuda_device", low=0)
            if sam3_raw.get("cuda_device") is not None
            else None
        ),
        min_score=_num(
            sam3_raw.get("min_score", 0.2), "perception.sam3.min_score", low=0.0, high=1.0
        ),
        max_object_height_m=_num(
            sam_raw.get("max_object_height_m", 0.35),
            "perception.max_object_height_m",
            low=0.01,
            high=1.0,
        ),
        voxel_m=_num(sam_raw.get("voxel_m", 0.005), "perception.voxel_m", low=0.002, high=0.03),
        max_voxels=_int(sam_raw.get("max_voxels", 250_000), "perception.max_voxels", low=1_000),
        depth_min_valid_ratio=_num(
            sam_raw.get("depth_min_valid_ratio", 0.35),
            "perception.depth_min_valid_ratio",
            low=0.0,
            high=1.0,
        ),
    )
    # The robot adapter's camera-optional methods (image, locate, segment,
    # back_project, camera_meta) use this camera.  It must exist on this
    # station, so a wrist-only site names its wrist camera explicitly.
    primary_camera = _str(
        sam_raw.get("primary_camera", "cam_head"),
        "perception.primary_camera",
        default="cam_head",
    )
    if primary_camera not in cameras:
        raise ConfigError(
            f"perception.primary_camera={primary_camera!r} is not a configured camera; "
            f"configured: {sorted(cameras)}"
        )

    snap_raw = _obj(raw.get("snapshot", {}), "snapshot")
    snapshot = SnapshotConfig(
        max_age_s=_num(snap_raw.get("max_age_s", 1.0), "snapshot.max_age_s", low=0.05),
        camera_joint_sync_s=_num(
            snap_raw.get("camera_joint_sync_s", 0.05), "snapshot.camera_joint_sync_s", low=0.001
        ),
        refresh_min_interval_s=_num(
            snap_raw.get("refresh_min_interval_s", 0.15),
            "snapshot.refresh_min_interval_s",
            low=0.0,
        ),
        history=_int(snap_raw.get("history", 512), "snapshot.history", low=10),
    )

    res_raw = _obj(raw.get("results", {}), "results")
    if snapshot.refresh_min_interval_s > snapshot.max_age_s:
        raise ConfigError("snapshot.refresh_min_interval_s must be <= snapshot.max_age_s")
    results = ResultsConfig(
        directory=str(
            _resolve(
                location.parent,
                _str(res_raw.get("directory", "runs/hardware"), "results.directory"),
            )
        ),
        max_episode_actions=_int(
            res_raw.get("max_episode_actions", 300), "results.max_episode_actions", low=1
        ),
        max_wall_s=_num(res_raw.get("max_wall_s", 1200.0), "results.max_wall_s", low=1.0),
    )

    safe_raw = _obj(raw.get("safety", {}), "safety")
    safety = SafetyConfig(
        stop_timeout_s=_num(safe_raw.get("stop_timeout_s", 2.0), "safety.stop_timeout_s", low=0.1),
        joint_speed_confirm_rad_s=_num(
            safe_raw.get("joint_speed_confirm_rad_s", 0.05),
            "safety.joint_speed_confirm_rad_s",
            low=0.001,
        ),
        home_lift_m=_num(
            safe_raw.get("home_lift_m", 0.08), "safety.home_lift_m", low=0.0, high=0.3
        ),
        require_operator_reset_after_fault=_bool(
            safe_raw.get("require_operator_reset_after_fault", True),
            "safety.require_operator_reset_after_fault",
        ),
    )

    return StationConfig(
        path=location,
        raw=raw,
        sha256=content_hash(raw),
        schema_version=CONFIG_SCHEMA,
        station_id=station_id,
        calibration_path=calib_path,
        calibration_sha256=calibration.sha256,
        service=service,
        execution=execution,
        arms=arms,
        cameras=cameras,
        primary_camera=primary_camera,
        sam3=sam3,
        snapshot=snapshot,
        results=results,
        safety=safety,
        calibration=calibration,
    )


def load_station_and_compare(
    path: str | Path, *, expected_sha256: str | None = None
) -> StationConfig:
    """Load a station file and refuse a content hash other than ``expected_sha256``."""

    config = load_station(path)
    if expected_sha256 is not None and config.sha256 != expected_sha256:
        raise ConfigError(
            f"station config hash changed: expected {expected_sha256}, got {config.sha256}"
        )
    return config


__all__ = [
    "ARMS",
    "CAMERAS",
    "CALIBRATION_SCHEMA",
    "CONFIG_SCHEMA",
    "ArmCalibration",
    "ArmConfig",
    "ArmDeviceConfig",
    "Calibration",
    "CameraCalibration",
    "CameraConfig",
    "ConfigError",
    "ExecutionConfig",
    "ExecutionMode",
    "GripperCalibration",
    "GripperDeviceConfig",
    "JointMapEntry",
    "ObstacleCalibration",
    "ResultsConfig",
    "SafetyConfig",
    "Sam3Config",
    "ServiceConfig",
    "SnapshotConfig",
    "StationConfig",
    "TableCalibration",
    "ToolCollision",
    "load_calibration",
    "load_station",
    "load_station_and_compare",
]
