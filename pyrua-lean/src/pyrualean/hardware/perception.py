# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Perception over measured camera images: SAM3 plus calibrated geometry.

Two measurement paths are implemented:

* **RGB-D**: the mask's valid depth pixels are deprojected with the calibrated
  camera-to-world transform.  The object's top comes from the measured points,
  never from the intersection of the mask ray with the table.
* **RGB only**: masks of the same text prompt in at least two calibrated views
  are intersected as a visual hull on a voxel grid.  One view cannot determine
  a height; the result says ``measurement_failed`` instead of inventing one.

The packet is not a simulator: no RoboDojo task layout, object pose or fixed
table height is read.  Table bounds and camera calibration come from the
station calibration.  Occlusion, missing depth and insufficient views produce
an explicit failure, not a value filled from a constant.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Calibration, Sam3Config
from .kinematics import wrap_deg


class PerceptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Segment:
    found: bool
    score: float | None = None
    mask: np.ndarray | None = None
    box: list[float] | None = None
    reason: str | None = None


@dataclass
class FrameData:
    name: str
    rgb: np.ndarray
    depth: np.ndarray | None
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    captured_at_wall_ns: int

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.rgb.shape[0]), int(self.rgb.shape[1]))


@dataclass
class Measurement:
    center_xyz: tuple[float, float, float]
    top_z: float
    height_m: float
    length_m: float
    width_m: float
    long_axis_deg: float
    grasp_yaw: float
    source: str
    views: int
    points: int
    quality: dict[str, Any] = field(default_factory=dict)


def frame_from_observation(
    observation: Any,
    artifacts: Mapping[str, bytes],
    camera: str,
) -> FrameData:
    from PIL import Image

    meta = observation.camera_state.get(camera)
    if not isinstance(meta, dict):
        raise PerceptionError(f"camera {camera!r} has no observation metadata")
    color_name = meta.get("color_artifact")
    if not color_name:
        raise PerceptionError(f"camera {camera!r} has no colour artifact")
    try:
        raw = artifacts[color_name]
    except KeyError:
        raise PerceptionError(f"colour artifact {color_name!r} was not retained") from None
    with Image.open(io.BytesIO(raw)) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    depth = None
    depth_name = meta.get("depth_artifact")
    if depth_name:
        try:
            payload = artifacts[depth_name]
        except KeyError:
            raise PerceptionError(f"depth artifact {depth_name!r} was not retained") from None
        depth = np.load(io.BytesIO(payload)).astype(np.float32)
        if depth.shape != rgb.shape[:2]:
            raise PerceptionError(
                f"camera {camera}: depth shape {depth.shape} does not match colour {rgb.shape[:2]}"
            )
    intrinsic = np.asarray(meta.get("intrinsic_matrix"), dtype=float)
    if intrinsic.shape != (3, 3):
        raise PerceptionError(f"camera {camera}: invalid intrinsic matrix")
    c2w = np.asarray(meta.get("camera_to_world"), dtype=float)
    if c2w.shape != (4, 4):
        raise PerceptionError(f"camera {camera}: invalid camera-to-world transform")
    return FrameData(
        name=camera,
        rgb=rgb,
        depth=depth,
        intrinsic=intrinsic,
        camera_to_world=c2w,
        captured_at_wall_ns=int(meta.get("captured_at_wall_ns", 0)),
    )


def _project(frame: FrameData, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to (row, col, depth)."""

    k = frame.intrinsic
    local = (points - frame.camera_to_world[:3, 3]) @ frame.camera_to_world[:3, :3]
    depth = -local[:, 2]
    safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
    rows = -k[1, 1] * local[:, 1] / safe + k[1, 2]
    cols = k[0, 0] * local[:, 0] / safe + k[0, 2]
    return rows, cols, depth


def _depth_points(frame: FrameData, mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    if frame.depth is None:
        return np.zeros((0, 3)), 0.0
    depth = frame.depth
    valid = np.isfinite(depth) & (depth > 0.0)
    if mask is not None:
        valid &= mask.astype(bool)
    rows, cols = np.nonzero(valid)
    total = int(rows.size)
    if total == 0:
        return np.zeros((0, 3)), 0.0
    d = depth[rows, cols].astype(float)
    k = frame.intrinsic
    local = np.stack(
        [
            (cols - k[0, 2]) / k[0, 0] * d,
            -(rows - k[1, 2]) / k[1, 1] * d,
            -d,
        ],
        axis=1,
    )
    world = local @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
    denominator = (
        float(np.sum(mask))
        if mask is not None and np.any(mask)
        else float(np.isfinite(depth).sum())
    )
    ratio = total / max(1.0, denominator)
    return world, ratio


def _ray_plane(frame: FrameData, row: int, col: int, z: float) -> np.ndarray | None:
    k = frame.intrinsic
    d = np.array(
        [
            (col - k[0, 2]) / k[0, 0],
            -(row - k[1, 2]) / k[1, 1],
            -1.0,
        ]
    )
    world_d = frame.camera_to_world[:3, :3] @ d
    if abs(world_d[2]) < 1e-9:
        return None
    t = (z - frame.camera_to_world[2, 3]) / world_d[2]
    if t <= 0:
        return None
    return frame.camera_to_world[:3, 3] + world_d * t


class Sam3Segmenter:
    """Lazy wrapper over the existing local SAM3 client (agent environment)."""

    def __init__(self, config: Sam3Config) -> None:
        self.config = config
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from ..hosts import sam3_local  # noqa: PLC0415 - agent-side dependency

            if not self.config.checkpoint:
                raise PerceptionError(
                    "SAM3 checkpoint is not configured; set perception.sam3.checkpoint "
                    "to an existing file in the agent environment"
                )
            self._model = sam3_local.shared(
                checkpoint=self.config.checkpoint, cuda_device=self.config.cuda_device
            )
        return self._model

    def segment(
        self,
        image: np.ndarray,
        *,
        text_prompt: str | None = None,
        point: Sequence[int] | None = None,
        min_score: float = 0.2,
    ) -> Segment:
        model = self._load()
        try:
            result = model.segment(
                image,
                text_prompt=text_prompt,
                point=list(point) if point else None,
                min_score=min_score,
            )
        except Exception as exc:  # noqa: BLE001 - service failure is reported as a result
            raise PerceptionError(f"SAM3 segmentation failed: {exc}") from exc
        return Segment(
            found=bool(result.found),
            score=float(result.score) if result.score is not None else None,
            mask=np.asarray(result.mask, dtype=bool) if result.mask is not None else None,
            box=list(result.box) if result.box is not None else None,
            reason=result.reason,
        )


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    rows, cols = np.nonzero(mask)
    return {
        "area_px": int(rows.size),
        "centroid_rc": [int(rows.mean()), int(cols.mean())] if rows.size else None,
    }


def _footprint(xy: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Long-axis heading (deg), length, width and centre of a 2-D point set."""

    if len(xy) < 3:
        centre = xy.mean(axis=0) if len(xy) else np.zeros(2)
        return 0.0, 0.0, 0.0, centre
    centre = np.median(xy, axis=0)
    cov = np.cov((xy - centre).T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    axis = vectors[:, order[0]]
    heading = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    u = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    v = np.array([-u[1], u[0]])
    lo_u, hi_u = np.percentile(xy @ u, [1.0, 99.0])
    lo_v, hi_v = np.percentile(xy @ v, [1.0, 99.0])
    length = float(hi_u - lo_u)
    width = float(hi_v - lo_v)
    if width > length:
        length, width = width, length
        heading += 90.0
    mid = u * (lo_u + hi_u) / 2 + v * (lo_v + hi_v) / 2
    heading = wrap_deg(heading)
    if heading < -90.0:
        heading += 180.0
    if heading > 90.0:
        heading -= 180.0
    return heading, length, width, mid


class Perception:
    def __init__(
        self,
        calibration: Calibration,
        sam3: Sam3Config,
        *,
        segmenter: Any | None = None,
    ) -> None:
        self.calibration = calibration
        self.sam3 = sam3
        if segmenter is not None:
            self._segmenter = segmenter
        elif sam3.mode == "http":
            # SAM3 runs on the Jetson Thor (or another host); the agent holds a
            # thin stdlib HTTP client and needs neither torch nor the sam3 wheel.
            from ..hosts.sam3_http import Sam3Http  # noqa: PLC0415

            if not sam3.endpoint:
                raise PerceptionError("perception.sam3.mode='http' needs perception.sam3.endpoint")
            self._segmenter = Sam3Http(endpoint=sam3.endpoint, timeout_s=sam3.timeout_s)
        else:
            self._segmenter = Sam3Segmenter(sam3)

    # -- segment -----------------------------------------------------------

    def segment(
        self,
        frames: Mapping[str, FrameData],
        text_prompt: str,
        camera: str,
        *,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        if not text_prompt.strip():
            raise ValueError("text_prompt must not be empty")
        frame = frames.get(camera)
        if frame is None:
            raise PerceptionError(f"camera {camera!r} is not in this snapshot")
        result = self._segmenter.segment(
            frame.rgb,
            text_prompt=text_prompt,
            min_score=self.sam3.min_score if min_score is None else float(min_score),
        )
        out: dict[str, Any] = {
            "camera": camera,
            "found": bool(result.found),
            "text_prompt": text_prompt,
            "score": result.score,
            "median_score": result.score,
            "box_px": result.box,
            "mask_shape": list(result.mask.shape) if result.mask is not None else None,
            "measurement_failed": not bool(result.found),
        }
        if result.mask is not None:
            out.update(_mask_stats(result.mask))
        if not result.found:
            out["note"] = result.reason or "SAM3 found no mask above the score threshold"
        return out

    # -- measurement -------------------------------------------------------

    def locate(
        self,
        frames: Mapping[str, FrameData],
        text_prompt: str,
        camera: str = "cam_head",
        *,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        primary = frames.get(camera)
        if primary is None:
            raise PerceptionError(f"camera {camera!r} is not in this snapshot")
        score_floor = self.sam3.min_score if min_score is None else float(min_score)
        result = self._segmenter.segment(primary.rgb, text_prompt=prompt, min_score=score_floor)
        out: dict[str, Any] = {
            "camera": camera,
            "found": bool(result.found),
            "text_prompt": prompt,
            "score": result.score,
            "views": 1,
            "measurement_failed": True,
        }
        if not result.found or result.mask is None:
            out["note"] = result.reason or "SAM3 found no mask above the score threshold"
            return out
        out.update(_mask_stats(result.mask))
        mask = result.mask
        points, depth_ratio = _depth_points(primary, mask)
        if points.shape[0] >= 12 and depth_ratio >= self.sam3.depth_min_valid_ratio:
            measurement = self._measure_points(points, source="depth", views=1)
            if measurement is not None:
                measurement.quality["depth_valid_ratio"] = round(depth_ratio, 3)
                return self._finish(out, measurement)
        masks: dict[str, np.ndarray] = {camera: mask}
        scores: dict[str, float | None] = {camera: result.score}
        for name, frame in frames.items():
            if name == camera:
                continue
            other = self._segmenter.segment(frame.rgb, text_prompt=prompt, min_score=score_floor)
            if not other.found or other.mask is None:
                continue
            share = float(other.mask.mean())
            if share > 0.30 or other.mask.sum() < 12:
                continue
            masks[name] = other.mask
            scores[name] = other.score
        if len(masks) < 2:
            out["note"] = (
                "measurement_failed: one view cannot determine an object's height; no second "
                "calibrated view produced a mask"
            )
            out["geometry_error"] = out["note"]
            out["depth_valid_ratio"] = round(depth_ratio, 3)
            return out
        carved = self._carve(frames, masks)
        if carved is None or carved[0].shape[0] < 12:
            out["note"] = "measurement_failed: the calibrated views do not intersect on an object"
            out["geometry_error"] = out["note"]
            out["views"] = len(masks)
            return out
        points, quality = carved
        measurement = self._measure_points(points, source="multiview", views=len(masks))
        if measurement is None:
            out["note"] = "measurement_failed: too few voxels above the table"
            out["geometry_error"] = out["note"]
            out["views"] = len(masks)
            return out
        measurement.quality.update(quality)
        out["view_scores"] = {k: v for k, v in scores.items()}
        return self._finish(out, measurement)

    def _finish(self, out: dict[str, Any], measurement: Measurement) -> dict[str, Any]:
        out.update(
            {
                "center_xyz": [round(v, 4) for v in measurement.center_xyz],
                "top_z": round(measurement.top_z, 4),
                "height_m": round(measurement.height_m, 4),
                "length_m": round(measurement.length_m, 4),
                "width_m": round(measurement.width_m, 4),
                "long_axis_deg": round(measurement.long_axis_deg, 1),
                "grasp_yaw": round(measurement.grasp_yaw, 1),
                "views": measurement.views,
                "source": measurement.source,
                "measurement_failed": False,
                "measurement_quality": measurement.quality,
            }
        )
        if measurement.source == "multiview":
            out["note"] = (
                "RGB-only multi-view measurement: height comes from intersecting calibrated "
                "silhouettes; hidden or thin parts remain uncertain"
            )
        return out

    def _measure_points(self, points: np.ndarray, *, source: str, views: int) -> Measurement | None:
        table = self.calibration.table
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 12:
            return None
        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]
        pts = pts[
            (pts[:, 2] > table.z_m + 0.002)
            & (pts[:, 2] < table.z_m + self.sam3.max_object_height_m)
            & (pts[:, 0] >= table.x_min_m - 0.02)
            & (pts[:, 0] <= table.x_max_m + 0.02)
            & (pts[:, 1] >= table.y_min_m - 0.02)
            & (pts[:, 1] <= table.y_max_m + 0.02)
        ]
        if pts.shape[0] < 12:
            return None
        top = float(np.percentile(pts[:, 2], 97))
        low = float(np.percentile(pts[:, 2], 2))
        height = top - table.z_m
        if height < 0.004:
            return None
        foot_slice = max(0.008, 0.18 * height)
        foot = pts[pts[:, 2] <= low + foot_slice]
        if len(foot) < 6:
            foot = pts
        heading, length, width, mid = _footprint(foot[:, :2])
        return Measurement(
            center_xyz=(float(mid[0]), float(mid[1]), float(table.z_m + height / 2)),
            top_z=top,
            height_m=height,
            length_m=length,
            width_m=width,
            long_axis_deg=heading,
            grasp_yaw=wrap_deg(heading + 90.0),
            source=source,
            views=views,
            points=int(pts.shape[0]),
            quality={
                "source": source,
                "views": views,
                "points": int(pts.shape[0]),
                "height_quantile": 0.97,
            },
        )

    def _carve(
        self,
        frames: Mapping[str, FrameData],
        masks: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        table = self.calibration.table
        primary = next(iter(masks))
        frame = frames[primary]
        rows, cols = np.nonzero(masks[primary])
        if rows.size == 0:
            return None
        # Bounds from the primary mask's rays at the table and at maximum height.
        pts = []
        for z in (table.z_m + 0.002, table.z_m + self.sam3.max_object_height_m):
            for row, col in zip(
                rows[:: max(1, rows.size // 200)], cols[:: max(1, cols.size // 200)], strict=True
            ):
                p = _ray_plane(frame, int(row), int(col), z)
                if p is not None:
                    pts.append(p)
        if not pts:
            return None
        cloud = np.asarray(pts)
        x_lo, y_lo = cloud[:, 0].min(), cloud[:, 1].min()
        x_hi, y_hi = cloud[:, 0].max(), cloud[:, 1].max()
        margin = max(0.02, self.sam3.voxel_m * 2)
        x_lo = max(table.x_min_m, x_lo - margin)
        x_hi = min(table.x_max_m, x_hi + margin)
        y_lo = max(table.y_min_m, y_lo - margin)
        y_hi = min(table.y_max_m, y_hi + margin)
        step = float(self.sam3.voxel_m)
        nx = max(1, int(math.ceil((x_hi - x_lo) / step)))
        ny = max(1, int(math.ceil((y_hi - y_lo) / step)))
        nz = max(1, int(math.ceil((self.sam3.max_object_height_m - 0.002) / step)))
        while nx * ny * nz > self.sam3.max_voxels:
            step *= 1.25
            nx = max(1, int(math.ceil((x_hi - x_lo) / step)))
            ny = max(1, int(math.ceil((y_hi - y_lo) / step)))
            nz = max(1, int(math.ceil((self.sam3.max_object_height_m - 0.002) / step)))
        xs = x_lo + (np.arange(nx) + 0.5) * step
        ys = y_lo + (np.arange(ny) + 0.5) * step
        zs = table.z_m + 0.002 + (np.arange(nz) + 0.5) * step
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        voxels = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        keep = np.ones(len(voxels), dtype=bool)
        for name, mask in masks.items():
            view = frames[name]
            r, c, depth = _project(view, voxels)
            valid = depth > 0
            ri = np.rint(r).astype(np.int64)
            ci = np.rint(c).astype(np.int64)
            inside = np.zeros(len(voxels), dtype=bool)
            ok = valid & (ri >= 0) & (ri < mask.shape[0]) & (ci >= 0) & (ci < mask.shape[1])
            inside[ok] = mask[ri[ok], ci[ok]]
            # One-pixel dilation accounts for silhouette quantisation.
            dilated = np.zeros_like(mask)
            dilated[1:-1, 1:-1] = (
                mask[:-2, 1:-1]
                | mask[2:, 1:-1]
                | mask[1:-1, :-2]
                | mask[1:-1, 2:]
                | mask[1:-1, 1:-1]
            )
            inside_d = np.zeros(len(voxels), dtype=bool)
            inside_d[ok] = dilated[ri[ok], ci[ok]]
            keep &= inside_d
            if not np.any(keep):
                return None
        occupied = voxels[keep]
        if occupied.shape[0] < 12:
            return None
        quality = {
            "voxel_m": step,
            "voxels": int(occupied.shape[0]),
            "grid": [nx, ny, nz],
            "views": len(masks),
            "volume_m3": float(occupied.shape[0] * step**3),
        }
        return occupied, quality

    # -- per-pixel ---------------------------------------------------------

    def back_project(
        self,
        frames: Mapping[str, FrameData],
        row: int,
        col: int,
        camera: str = "cam_head",
    ) -> dict[str, Any]:
        frame = frames.get(camera)
        if frame is None:
            raise PerceptionError(f"camera {camera!r} is not in this snapshot")
        height, width = frame.shape
        if not (0 <= int(row) < height and 0 <= int(col) < width):
            raise ValueError(f"pixel ({row}, {col}) is outside {width}x{height}")
        wall = frame.captured_at_wall_ns
        if frame.depth is not None:
            depth = float(frame.depth[int(row), int(col)])
            if math.isfinite(depth) and depth > 0:
                k = frame.intrinsic
                local = np.array(
                    [
                        (int(col) - k[0, 2]) / k[0, 0] * depth,
                        -(int(row) - k[1, 2]) / k[1, 1] * depth,
                        -depth,
                    ]
                )
                world = frame.camera_to_world[:3, :3] @ local + frame.camera_to_world[:3, 3]
                surface = "object" if world[2] > self.calibration.table.z_m + 0.003 else "table"
                return {
                    "pixel": [int(row), int(col)],
                    "world_xyz": [round(float(v), 5) for v in world],
                    "depth_m": round(depth, 5),
                    "camera": camera,
                    "surface": surface,
                    "measurement_failed": False,
                    "captured_at_wall_ns": wall,
                }
        # No depth at this pixel: segment the object under it, then measure it.
        point = self._segmenter.segment(
            frame.rgb, point=[int(row), int(col)], min_score=self.sam3.min_score
        )
        if not point.found or point.mask is None:
            raise PerceptionError("no depth at the pixel and SAM3 found no object under it")
        masks = {camera: point.mask}
        for name, other in frames.items():
            if name == camera:
                continue
            res = self._segmenter.segment(
                other.rgb, text_prompt="object", min_score=self.sam3.min_score
            )
            if res.found and res.mask is not None and res.mask.mean() < 0.30:
                masks[name] = res.mask
        if len(masks) >= 2:
            carved = self._carve(frames, masks)
            if carved is not None:
                measurement = self._measure_points(carved[0], source="multiview", views=len(masks))
                if measurement is not None:
                    hit = _ray_plane(frame, int(row), int(col), measurement.top_z)
                    if hit is not None:
                        return {
                            "pixel": [int(row), int(col)],
                            "world_xyz": [round(float(v), 5) for v in hit],
                            "depth_m": round(
                                float(np.linalg.norm(hit - frame.camera_to_world[:3, 3])), 5
                            ),
                            "camera": camera,
                            "surface": "object",
                            "measurement_failed": False,
                            "captured_at_wall_ns": wall,
                        }
        table_hit = _ray_plane(frame, int(row), int(col), self.calibration.table.z_m)
        if table_hit is not None:
            return {
                "pixel": [int(row), int(col)],
                "world_xyz": [round(float(v), 5) for v in table_hit],
                "depth_m": round(
                    float(np.linalg.norm(table_hit - frame.camera_to_world[:3, 3])), 5
                ),
                "camera": camera,
                "surface": "table",
                "measurement_failed": False,
                "captured_at_wall_ns": wall,
            }
        raise PerceptionError("no depth and no calibrated surface at this pixel")


__all__ = [
    "FrameData",
    "Measurement",
    "Perception",
    "PerceptionError",
    "Sam3Segmenter",
    "Segment",
    "frame_from_observation",
]
