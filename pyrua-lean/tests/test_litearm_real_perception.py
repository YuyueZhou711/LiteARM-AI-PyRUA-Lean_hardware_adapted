"""Calibrated RGB-D and multi-view measurement tests with known geometry."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pytest

from pyrualean.hardware.perception import (
    FrameData,
    Perception,
    PerceptionError,
    Segment,
    frame_from_observation,
)

H, W = 48, 64
K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
BOX = {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (0.0, 0.10)}


def _look_at(position, target) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    z = position - target
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    norm = np.linalg.norm(x)
    if norm < 1e-9:
        # Looking straight down/up: any horizontal x is valid; pick world +x.
        x = np.array([1.0, 0.0, 0.0])
        x = x - z * np.dot(x, z)
        norm = np.linalg.norm(x)
    x /= norm
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2], t[:3, 3] = x, y, z, position
    return t


def _project(c2w, points):
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    depth = -local[:, 2]
    safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
    rows = -K[1, 1] * local[:, 1] / safe + K[1, 2]
    cols = K[0, 0] * local[:, 0] / safe + K[0, 2]
    return rows, cols, depth


def _box_mask(c2w) -> np.ndarray:
    import cv2

    corners = np.array([[x, y, z] for x in BOX["x"] for y in BOX["y"] for z in BOX["z"]])
    rows, cols, depth = _project(c2w, corners)
    pts = np.stack([cols, rows], axis=1)
    pts = np.round(pts).astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    return mask.astype(bool)


def _rgb(mask):
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[mask] = (220, 30, 30)
    return image


def _top_depth():
    c2w = _look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    depth = np.full((H, W), np.nan, dtype=np.float32)
    rows, cols = np.mgrid[0:H, 0:W]
    local = np.stack(
        [
            (cols - K[0, 2]) / K[0, 0],
            -(rows - K[1, 2]) / K[1, 1],
            -np.ones_like(rows),
        ],
        axis=-1,
    ).astype(float)
    # Each ray meets the box top (z=0.10) at t=0.90 and the table (z=0) at
    # t=1.0.  A pixel inside the box footprint reports the top's depth; every
    # other pixel reports the table.
    for r in range(H):
        for c in range(W):
            direction = local[r, c]
            p = c2w[:3, 3] + direction * 0.90
            if BOX["x"][0] <= p[0] <= BOX["x"][1] and BOX["y"][0] <= p[1] <= BOX["y"][1]:
                depth[r, c] = 0.90
            else:
                depth[r, c] = 1.0
    return depth, c2w


@dataclass
class _Scripted:
    masks: dict[str, np.ndarray]

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        raise AssertionError("the scripted segmenter is called through the camera map")


class _PerCameraSegmenter:
    def __init__(self, masks):
        self.masks = masks

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        # The test encodes the camera in the red channel intensity of the image's
        # top-left pixel: 1 -> head, 2 -> side, 3 -> other.
        key = int(image[0, 0, 1])
        mask = self.masks[key]
        if point is not None:
            row, col = int(point[0]), int(point[1])
            if 0 <= row < H and 0 <= col < W and mask[row, col]:
                return Segment(True, 0.9, mask, [0, 0, W, H])
            return Segment(False, reason="no object at point")
        return Segment(True, 0.9, mask, [0, 0, W, H])


def _frame(name, key, c2w, mask, depth=None):
    rgb = _rgb(mask)
    if key:
        rgb[0, 0, 1] = key
    return FrameData(name, rgb, depth, K.copy(), c2w, 1)


def test_depth_measurement_uses_measured_top_and_footprint():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    segmenter = _PerCameraSegmenter({1: mask})
    perception = Perception(
        _calibration(),
        _sam3(),
        segmenter=segmenter,
    )
    frame = _frame("cam_head", 1, c2w, mask, depth)
    result = perception.locate({"cam_head": frame}, "the red block")
    assert result["found"] and not result["measurement_failed"]
    assert result["source"] == "depth"
    assert result["top_z"] == pytest.approx(0.10, abs=0.005)
    assert result["height_m"] == pytest.approx(0.10, abs=0.005)
    assert result["center_xyz"][0] == pytest.approx(0.0, abs=0.005)
    assert result["center_xyz"][1] == pytest.approx(0.0, abs=0.005)
    assert result["width_m"] == pytest.approx(0.04, abs=0.01)
    assert result["length_m"] == pytest.approx(0.06, abs=0.01)
    assert result["views"] >= 1
    assert result["measurement_quality"]["depth_valid_ratio"] > 0.9


def test_one_rgb_view_cannot_measure_height():
    c2w = _look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    mask = _box_mask(c2w)
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate({"cam_head": _frame("cam_head", 1, c2w, mask)}, "the red block")
    assert result["found"] is True
    assert result["measurement_failed"] is True
    assert result.get("center_xyz") is None
    assert "one view" in result["geometry_error"]


def test_two_views_measure_a_visual_hull():
    top = _look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    side = _look_at([0.0, -0.55, 0.35], [0.0, 0.0, 0.05])
    masks = {1: _box_mask(top), 2: _box_mask(side)}
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter(masks))
    result = perception.locate(
        {
            "cam_head": _frame("cam_head", 1, top, masks[1]),
            "cam_l": _frame("cam_l", 2, side, masks[2]),
        },
        "the red block",
    )
    assert result["found"] and not result["measurement_failed"]
    assert result["source"] == "multiview"
    assert result["views"] == 2
    assert result["top_z"] == pytest.approx(0.10, abs=0.035)
    assert result["center_xyz"][0] == pytest.approx(0.0, abs=0.02)


def test_invalid_depth_falls_back_or_fails_explicitly():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    depth[:] = np.nan
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate(
        {"cam_head": _frame("cam_head", 1, c2w, mask, depth)}, "the red block"
    )
    assert result["measurement_failed"] is True
    assert result.get("center_xyz") is None


def test_back_project_uses_valid_depth_and_marks_table():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    frame = _frame("cam_head", 1, c2w, mask, depth)
    row, col = np.argwhere(mask)[0]
    point = perception.back_project({"cam_head": frame}, int(row), int(col))
    assert point["surface"] == "object"
    assert point["world_xyz"][2] == pytest.approx(0.10, abs=0.005)
    table_pixel = (0, 0)
    table = perception.back_project({"cam_head": frame}, *table_pixel)
    assert table["surface"] == "table"
    assert table["world_xyz"][2] == pytest.approx(0.0, abs=0.005)


def test_frame_loader_refuses_mismatched_depth_and_missing_artifacts():
    class Obs:
        camera_state = {
            "cam_head": {
                "color_artifact": "a.png",
                "depth_artifact": "a_depth.npy",
                "intrinsic_matrix": K.tolist(),
                "camera_to_world": np.eye(4).tolist(),
                "captured_at_wall_ns": 1,
            }
        }

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (W, H), (1, 2, 3)).save(buf, format="PNG")
    depth = np.zeros((H + 1, W), dtype=np.float32)
    dbuf = io.BytesIO()
    np.save(dbuf, depth)
    with pytest.raises(PerceptionError, match="depth shape"):
        frame_from_observation(
            Obs(), {"a.png": buf.getvalue(), "a_depth.npy": dbuf.getvalue()}, "cam_head"
        )
    with pytest.raises(PerceptionError, match="not retained"):
        frame_from_observation(Obs(), {"a_depth.npy": dbuf.getvalue()}, "cam_head")


def _calibration():
    from pathlib import Path

    from pyrualean.hardware.config import load_station

    root = Path(__file__).resolve().parents[1]
    return load_station(root / "configs" / "litearm_real.example.json").calibration


def _sam3():
    from pathlib import Path

    from pyrualean.hardware.config import load_station

    root = Path(__file__).resolve().parents[1]
    return load_station(root / "configs" / "litearm_real.example.json").sam3


def test_http_sam3_endpoint_is_used_and_a_thor_style_mask_is_honoured():
    """SAM3 runs on Thor; the agent only sends the RGB image over HTTP."""

    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from pyrualean.hosts.sam3_http import _pack_mask

    depth, c2w = _top_depth()
    mask = _box_mask(c2w)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            if self.path != "/segment":
                self.send_response(404)
                self.end_headers()
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["text_prompt"] == "the red block"
            payload = {
                "found": True,
                "score": 0.9,
                "box": [0, 0, 64, 48],
                "mask": _pack_mask(mask),
                "reason": None,
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        from dataclasses import replace

        sam3 = replace(_sam3(), mode="http", endpoint=endpoint, checkpoint=None)
        perception = Perception(_calibration(), sam3)
        frame = _frame("cam_head", 1, c2w, mask, depth)
        result = perception.locate({"cam_head": frame}, "the red block")
        assert result["found"] is True
        assert result["measurement_failed"] is False
        assert result["top_z"] == pytest.approx(0.10, abs=0.005)
    finally:
        server.shutdown()
        server.server_close()
