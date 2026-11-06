"""Camera driver selection and colour handling (offline, fake SDKs)."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrualean.hardware.cameras import CameraError, OpenCVCamera, RealSenseCamera
from pyrualean.hardware.config import load_station

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "litearm_real.example.json"


class _FakeCapture:
    def __init__(self) -> None:
        self.sets: list[tuple[int, int]] = []
        self.released = False
        self.open = True

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV method name
        return self.open

    def set(self, prop: int, value: int) -> bool:
        self.sets.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


def _fake_cv2() -> tuple[types.SimpleNamespace, _FakeCapture, dict]:
    cap = _FakeCapture()
    calls: dict = {}

    def VideoCapture(source, api=None):  # noqa: N802, ANN001 - fake OpenCV API
        calls["capture"] = (source, api)
        return cap

    def VideoWriter_fourcc(*chars):  # noqa: N802 - fake OpenCV API
        calls["fourcc"] = chars
        return 0x56595559

    cv2 = types.SimpleNamespace(
        VideoCapture=VideoCapture,
        VideoWriter_fourcc=VideoWriter_fourcc,
        CAP_V4L2=200,
        CAP_PROP_FOURCC=6,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_PROP_BUFFERSIZE=38,
    )
    return cv2, cap, calls


def test_opencv_uses_the_v4l2_device_path_and_fourcc(monkeypatch):
    config = load_station(EXAMPLE)
    cam_config = replace(
        config.cameras["cam_left_wrist"],
        index=None,
        device="/dev/video4",
        fourcc="YUYV",
    )
    cv2, cap, calls = _fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    camera = OpenCVCamera(cam_config, config.calibration.camera("cam_left_wrist"))
    camera.open()
    assert calls["capture"] == ("/dev/video4", cv2.CAP_V4L2)
    assert calls["fourcc"] == ("Y", "U", "Y", "V")
    assert cap.sets[0] == (cv2.CAP_PROP_FOURCC, 0x56595559)
    assert (cv2.CAP_PROP_FRAME_WIDTH, cam_config.width) in cap.sets
    assert (cv2.CAP_PROP_FRAME_HEIGHT, cam_config.height) in cap.sets
    assert (cv2.CAP_PROP_FPS, cam_config.fps) in cap.sets
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in cap.sets


def test_opencv_index_is_used_when_no_device_path_is_set(monkeypatch):
    config = load_station(EXAMPLE)
    cam_config = replace(config.cameras["cam_left_wrist"], index=4, device=None, fourcc=None)
    cv2, cap, calls = _fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    camera = OpenCVCamera(cam_config, config.calibration.camera("cam_left_wrist"))
    camera.open()
    assert calls["capture"] == (4, None)
    assert "fourcc" not in calls
    assert cap.sets[0] == (cv2.CAP_PROP_FRAME_WIDTH, cam_config.width)


def _fake_realsense(
    bgr: np.ndarray, *, intrinsics: tuple[float, float, float, float]
) -> tuple[types.SimpleNamespace, list, dict]:
    requested: list[tuple] = []
    state: dict = {}

    class _Config:
        def __init__(self) -> None:
            self.serial = None

        def enable_device(self, serial: str) -> None:
            self.serial = serial

        def enable_stream(self, stream, width, height, fmt, fps) -> None:  # noqa: ANN001
            requested.append((stream, width, height, fmt, fps))

    class _Intrinsics:
        def __init__(self) -> None:
            self.fx, self.fy, self.ppx, self.ppy = intrinsics

    class _VideoProfile:
        def get_intrinsics(self) -> _Intrinsics:
            return _Intrinsics()

    class _StreamProfile:
        def as_video_stream_profile(self) -> _VideoProfile:
            return _VideoProfile()

    class _Profile:
        def get_stream(self, stream) -> _StreamProfile:  # noqa: ANN001
            assert stream == "color"
            return _StreamProfile()

    class _ColorFrame:
        def get_data(self) -> np.ndarray:
            return bgr

    class _Frames:
        def get_color_frame(self) -> _ColorFrame:
            return _ColorFrame()

    class _Pipeline:
        def __init__(self) -> None:
            self.stopped = False

        def start(self, cfg) -> _Profile:  # noqa: ANN001
            state["config"] = cfg
            return _Profile()

        def wait_for_frames(self, timeout_ms: int = 0) -> _Frames:
            state["timeout_ms"] = timeout_ms
            return _Frames()

        def stop(self) -> None:
            self.stopped = True

    pipeline = _Pipeline()
    state["pipeline"] = pipeline
    rs = types.SimpleNamespace(
        pipeline=lambda: pipeline,
        config=_Config,
        stream=types.SimpleNamespace(color="color", depth="depth"),
        format=types.SimpleNamespace(bgr8="bgr8", rgb8="rgb8", z16="z16"),
        align=lambda stream: None,
    )
    return rs, requested, state


def test_realsense_requests_bgr8_and_returns_rgb(monkeypatch):
    config = load_station(EXAMPLE)
    cam_config = replace(config.cameras["cam_head"], depth=False)
    bgr = np.zeros((cam_config.height, cam_config.width, 3), dtype=np.uint8)
    bgr[0, 0] = (10, 20, 30)
    bgr[1, 1] = (1, 2, 3)
    rs, requested, state = _fake_realsense(bgr, intrinsics=(500.0, 500.0, 320.0, 240.0))
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    camera = RealSenseCamera(cam_config, config.calibration.camera("cam_head"))
    camera.open()
    assert requested == [("color", cam_config.width, cam_config.height, "bgr8", cam_config.fps)]
    frame = camera.capture()
    assert frame.rgb.shape == (cam_config.height, cam_config.width, 3)
    assert tuple(frame.rgb[0, 0]) == (30, 20, 10)
    assert tuple(frame.rgb[1, 1]) == (3, 2, 1)
    assert frame.depth is None
    assert state["timeout_ms"] == int(cam_config.timeout_s * 1000)
    camera.close()
    assert state["pipeline"].stopped is True


def test_realsense_still_refuses_mismatched_intrinsics(monkeypatch):
    config = load_station(EXAMPLE)
    cam_config = replace(config.cameras["cam_head"], depth=False)
    bgr = np.zeros((cam_config.height, cam_config.width, 3), dtype=np.uint8)
    rs, requested, state = _fake_realsense(bgr, intrinsics=(600.0, 500.0, 320.0, 240.0))
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    camera = RealSenseCamera(cam_config, config.calibration.camera("cam_head"))
    with pytest.raises(CameraError, match="intrinsics differ"):
        camera.open()
    assert requested[0][3] == "bgr8"
    assert state["pipeline"].stopped is True


def test_realsense_still_refuses_a_wrong_frame_size(monkeypatch):
    config = load_station(EXAMPLE)
    cam_config = replace(config.cameras["cam_head"], depth=False)
    bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    rs, _, _ = _fake_realsense(bgr, intrinsics=(500.0, 500.0, 320.0, 240.0))
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    camera = RealSenseCamera(cam_config, config.calibration.camera("cam_head"))
    camera.open()
    with pytest.raises(CameraError, match="configured"):
        camera.capture()
