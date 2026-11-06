# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Camera drivers and RGB-D acquisition for the hardware service.

Two real drivers are implemented:

* ``opencv`` - a V4L2/UVC colour camera through OpenCV (colour only);
* ``realsense`` - an Intel RealSense colour+depth camera through
  ``pyrealsense2`` with depth aligned to colour.

Both are imported lazily inside the driver's ``open``.  The module never opens
a camera at import time.  A configured driver that is not installed fails with
an actionable error; it never falls back to a synthetic camera.

Calibration is authoritative for projection.  For RealSense the driver reads
the device intrinsics once and refuses to run if they differ from the station
calibration by more than a small tolerance: a stale calibration must be visible
before it corrupts a 3-D measurement, not after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Calibration, CameraCalibration, CameraConfig
from .protocol import now_monotonic_ns, now_wall_ns


class CameraError(RuntimeError):
    def __init__(self, camera: str, message: str) -> None:
        super().__init__(f"{camera}: {message}")
        self.camera = camera


@dataclass
class CameraFrame:
    name: str
    rgb: np.ndarray
    depth: np.ndarray | None
    captured_at_wall_ns: int
    captured_at_monotonic_ns: int
    timestamp_source: str = "host_capture"
    depth_scale_m: float | None = None
    undistorted: bool = False
    intrinsic: np.ndarray | None = None
    distortion: tuple[float, ...] = ()
    notes: list[str] = field(default_factory=list)


class CameraBase:
    name = ""

    def open(self) -> None:
        raise NotImplementedError

    def capture(self) -> CameraFrame:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVCamera(CameraBase):
    def __init__(self, config: CameraConfig, calibration: CameraCalibration) -> None:
        self.config = config
        self.calibration = calibration
        self.name = config.name
        self._cap: Any = None

    def open(self) -> None:
        if self.config.depth:
            raise CameraError(
                self.name,
                "driver 'opencv' provides RGB only; configure a depth-capable driver "
                "for a camera whose calibration has has_depth=true",
            )
        if self.config.index is None and not self.config.device:
            raise CameraError(self.name, "opencv driver needs a device index or device path")
        try:
            import cv2  # noqa: PLC0415 - lazy driver import
        except ImportError as exc:
            raise CameraError(
                self.name,
                "opencv-python-headless is not installed in the device environment",
            ) from exc
        if self.config.device:
            # An integer index cannot address every V4L2 node (for example a
            # RealSense colour node exposed as /dev/video4).  An explicit path
            # with the V4L2 backend is required there.
            cap = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
            target = self.config.device
        else:
            cap = cv2.VideoCapture(int(self.config.index))
            target = f"/dev/video{self.config.index}"
        if not cap.isOpened():
            cap.release()
            raise CameraError(self.name, f"cannot open {target}")
        if self.config.fourcc:
            # FourCC is a V4L2 pixel format (for example YUYV or MJPG).  Set it
            # before the resolution/FPS negotiation so the driver can pick a
            # mode that matches the calibration.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def capture(self) -> CameraFrame:
        import cv2  # noqa: PLC0415

        assert self._cap is not None
        wall = now_wall_ns()
        mono = now_monotonic_ns()
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise CameraError(self.name, "camera returned no frame")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.config.height, self.config.width):
            raise CameraError(
                self.name,
                f"camera produced {rgb.shape[1]}x{rgb.shape[0]}, configured "
                f"{self.config.width}x{self.config.height}",
            )
        return CameraFrame(
            name=self.name,
            rgb=rgb,
            depth=None,
            captured_at_wall_ns=wall,
            captured_at_monotonic_ns=mono,
            timestamp_source="host_capture",
        )

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._cap = None


class RealSenseCamera(CameraBase):
    """Intel RealSense colour + aligned depth through ``pyrealsense2``."""

    def __init__(self, config: CameraConfig, calibration: CameraCalibration) -> None:
        self.config = config
        self.calibration = calibration
        self.name = config.name
        self._pipeline: Any = None
        self._align: Any = None
        self._np: Any = None

    def open(self) -> None:
        try:
            import numpy as np  # noqa: PLC0415
            import pyrealsense2 as rs  # noqa: PLC0415
        except ImportError as exc:
            raise CameraError(
                self.name,
                "pyrealsense2 is not installed in the device environment; install the "
                "wheel matching the camera firmware and librealsense",
            ) from exc
        self._np = np
        pipeline = rs.pipeline()
        cfg = rs.config()
        if self.config.serial:
            cfg.enable_device(str(self.config.serial))
        # D435i firmware 5.17.0.9 timed out when this stream requested rgb8 at
        # 640x480@30; bgr8 at the same mode delivers color+depth reliably.
        # ``capture`` converts BGR to RGB before returning the frame, preserving
        # the adapter's RGB colour convention.
        cfg.enable_stream(
            rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps
        )
        if self.config.depth:
            cfg.enable_stream(
                rs.stream.depth,
                self.config.width,
                self.config.height,
                rs.format.z16,
                self.config.fps,
            )
        try:
            profile = pipeline.start(cfg)
        except Exception as exc:  # noqa: BLE001
            raise CameraError(self.name, f"RealSense pipeline failed to start: {exc}") from exc
        self._pipeline = pipeline
        if self.config.depth:
            self._align = rs.align(rs.stream.color)
        try:
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color.get_intrinsics()
            expected = self.calibration.intrinsic
            diffs = {
                "fx": abs(float(intr.fx) - float(expected[0, 0])),
                "fy": abs(float(intr.fy) - float(expected[1, 1])),
                "cx": abs(float(intr.ppx) - float(expected[0, 2])),
                "cy": abs(float(intr.ppy) - float(expected[1, 2])),
            }
            tolerance = {
                "fx": max(2.0, 0.02 * abs(float(expected[0, 0]))),
                "fy": max(2.0, 0.02 * abs(float(expected[1, 1]))),
                "cx": 4.0,
                "cy": 4.0,
            }
            bad = {k: (v, tolerance[k]) for k, v in diffs.items() if v > tolerance[k]}
            if bad:
                raise CameraError(
                    self.name,
                    "device colour intrinsics differ from calibration "
                    f"({bad}); rerun calibration, do not use a stale matrix",
                )
        except CameraError:
            pipeline.stop()
            self._pipeline = None
            raise
        except Exception as exc:  # noqa: BLE001
            pipeline.stop()
            self._pipeline = None
            raise CameraError(self.name, f"could not read/verify intrinsics: {exc}") from exc

    def capture(self) -> CameraFrame:
        assert self._pipeline is not None
        wall = now_wall_ns()
        mono = now_monotonic_ns()
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=int(self.config.timeout_s * 1000))
            if self._align is not None:
                frames = self._align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame() if self.config.depth else None
            if not color:
                raise CameraError(self.name, "no colour frame")
            bgr = self._np.asanyarray(color.get_data())
            # The stream is bgr8 (see open); every other camera path returns RGB.
            rgb = bgr[..., ::-1].copy()
            depth_array = None
            if depth is not None:
                raw = self._np.asanyarray(depth.get_data()).copy()
                scale = self._np.asanyarray(depth.get_units())
                depth_array = raw.astype(self._np.float32) * float(scale)
                depth_array[~self._np.isfinite(depth_array)] = self._np.nan
                depth_array[depth_array <= 0.0] = self._np.nan
        except CameraError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CameraError(self.name, f"capture failed: {exc}") from exc
        if rgb.shape[:2] != (self.config.height, self.config.width):
            raise CameraError(
                self.name,
                f"camera produced {rgb.shape[1]}x{rgb.shape[0]}, configured "
                f"{self.config.width}x{self.config.height}",
            )
        return CameraFrame(
            name=self.name,
            rgb=rgb,
            depth=depth_array,
            captured_at_wall_ns=wall,
            captured_at_monotonic_ns=mono,
            timestamp_source="host_capture",
        )

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
        self._pipeline = None


class SyntheticCamera(CameraBase):
    """Explicit offline camera; only the fake station may create it."""

    def __init__(self, config: CameraConfig, calibration: CameraCalibration) -> None:
        self.config = config
        self.calibration = calibration
        self.name = config.name
        # Deterministic simple scene; this is test data, not a physical claim.
        self._t = 0

    def open(self) -> None:
        return

    def capture(self) -> CameraFrame:
        wall = now_wall_ns()
        mono = now_monotonic_ns()
        height, width = self.config.height, self.config.width
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 2] = 40  # dark blue background
        rgb[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = (200, 40, 40)
        depth = None
        if self.config.depth:
            depth = np.full((height, width), np.nan, dtype=np.float32)
            depth[height // 2 :] = 1.0
        self._t += 1
        return CameraFrame(
            name=self.name,
            rgb=rgb,
            depth=depth,
            captured_at_wall_ns=wall,
            captured_at_monotonic_ns=mono,
            timestamp_source="synthetic",
        )

    def close(self) -> None:
        return


def _undistort(
    frame: CameraFrame, calibration: CameraCalibration, *, cv2: Any, np: Any
) -> CameraFrame:
    coefficients = np.asarray(calibration.distortion, dtype=np.float64)
    if coefficients.size == 0 or np.allclose(coefficients, 0.0):
        frame.intrinsic = np.asarray(calibration.intrinsic, dtype=float).copy()
        frame.distortion = ()
        return frame
    h, w = frame.rgb.shape[:2]
    k = np.asarray(calibration.intrinsic, dtype=np.float64)
    new_k, roi = cv2.getOptimalNewCameraMatrix(k, coefficients, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(k, coefficients, None, new_k, (w, h), cv2.CV_32FC1)
    rgb = cv2.remap(frame.rgb, map1, map2, interpolation=cv2.INTER_LINEAR)
    depth = (
        cv2.remap(
            frame.depth,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        if frame.depth is not None
        else None
    )
    _ = roi
    # The projection must use the new camera matrix after undistortion.
    out = CameraFrame(
        name=frame.name,
        rgb=rgb,
        depth=depth,
        captured_at_wall_ns=frame.captured_at_wall_ns,
        captured_at_monotonic_ns=frame.captured_at_monotonic_ns,
        timestamp_source=frame.timestamp_source,
        depth_scale_m=frame.depth_scale_m,
        undistorted=True,
        intrinsic=new_k,
        distortion=(),
        notes=list(frame.notes),
    )
    out.notes.append("undistorted with cv2.getOptimalNewCameraMatrix")
    return out


class CameraManager:
    """Owns every configured camera; captures RGB and optional aligned depth.

    Only cameras that belong to this station are opened: a world-mounted
    camera always belongs, and a tool-mounted camera belongs when its arm is
    listed in ``execution.active_arms``.  A tool camera on an arm that is not
    physically present is neither required nor touched.
    """

    def __init__(self, config: Any, calibration: Calibration) -> None:
        self.config = config
        self.calibration = calibration
        self.cameras: dict[str, CameraBase] = {}
        self._cv2: Any = None
        self._np: Any = None

    def _owned(self, name: str) -> bool:
        cal = self.calibration.camera(name)
        return cal.arm is None or cal.arm in self.config.execution.active_arms

    @property
    def names(self) -> tuple[str, ...]:
        """Cameras this station owns, in calibration order."""

        return tuple(name for name in self.calibration.cameras if self._owned(name))

    def open(self) -> None:
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError as exc:
            raise CameraError(
                "station", f"numpy/opencv are required for image processing: {exc}"
            ) from exc
        self._cv2, self._np = cv2, np
        try:
            for name in self.calibration.cameras:
                if not self._owned(name):
                    continue
                cfg = self.config.cameras.get(name)
                if cfg is None:
                    raise CameraError(name, "configured camera is missing")
                cal = self.calibration.camera(name)
                if cfg.driver == "opencv":
                    driver: CameraBase = OpenCVCamera(cfg, cal)
                elif cfg.driver == "realsense":
                    driver = RealSenseCamera(cfg, cal)
                elif cfg.driver == "fake":
                    driver = SyntheticCamera(cfg, cal)
                else:
                    raise CameraError(name, f"unknown driver {cfg.driver!r}")
                driver.open()
                self.cameras[name] = driver
        except BaseException:
            self.close()
            raise

    def capture(self) -> dict[str, CameraFrame]:
        frames: dict[str, CameraFrame] = {}
        for name, driver in self.cameras.items():
            frame = driver.capture()
            if frame.rgb.shape[:2] != (
                self.calibration.camera(name).height,
                self.calibration.camera(name).width,
            ):
                raise CameraError(
                    name,
                    f"image {frame.rgb.shape[1]}x{frame.rgb.shape[0]} does not match "
                    f"calibration {self.calibration.camera(name).width}x"
                    f"{self.calibration.camera(name).height}",
                )
            frame = _undistort(frame, self.calibration.camera(name), cv2=self._cv2, np=self._np)
            if frame.intrinsic is None:
                frame.intrinsic = np.asarray(
                    self.calibration.camera(name).intrinsic, dtype=float
                ).copy()
            frames[name] = frame
        return frames

    def close(self) -> None:
        for driver in self.cameras.values():
            try:
                driver.close()
            except Exception:  # noqa: BLE001
                pass
        self.cameras = {}


__all__ = [
    "CameraBase",
    "CameraError",
    "CameraFrame",
    "CameraManager",
    "OpenCVCamera",
    "RealSenseCamera",
    "SyntheticCamera",
]
