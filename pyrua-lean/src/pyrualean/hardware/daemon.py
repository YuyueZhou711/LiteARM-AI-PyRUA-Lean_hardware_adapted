# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Hardware service: owns the devices, the exclusive session and the stop state machine.

The service is deliberately the only process that opens an arm, gripper or
camera.  It accepts one runner session at a time, executes submitted
synchronous segments one by one, and keeps snapshots and evidence.  Requests
are handled on independent threads, so ``cancel`` and ``emergency_stop`` are
answered even while a command is running.  The actual mechanical stop is a
separate action and the service records whether it could be confirmed.

States: ``read_only`` -> ``ready`` -> ``executing`` -> ``stopping`` ->
``stopped`` or ``fault``.  A fault never clears itself on reconnect; the
process must be restarted by an operator after the hardware is checked.
"""

from __future__ import annotations

import argparse
import base64
import collections
import contextlib
import io
import logging
import signal
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cameras import CameraManager
from .config import ExecutionMode, StationConfig, load_station
from .devices import DeviceCancelled, DeviceError, NotReady, StationBase, make_station
from .kinematics import LitearmKinematics
from .planning import Planner, PlanningError
from .protocol import (
    COMMAND_KINDS,
    MAX_LINE_BYTES,
    PROTOCOL_VERSION,
    CommandState,
    ErrorCode,
    SessionState,
    StopKind,
    StopResult,
    command_digest,
    decode_message,
    encode_message,
    error,
    new_id,
    now_monotonic_ns,
    now_wall_ns,
    require_dict,
    require_finite,
    require_str,
    response,
)

log = logging.getLogger("pyrualean.hardware")


@dataclass
class _Artifact:
    name: str
    media_type: str
    data: bytes
    sha256: str


@dataclass
class _Snapshot:
    index: int
    payload: dict[str, Any]
    artifacts: dict[str, _Artifact]


@dataclass
class _Command:
    command_id: str
    kind: str
    digest: str
    payload: dict[str, Any]
    state: str = CommandState.QUEUED.value
    created_at_wall_ns: int = field(default_factory=now_wall_ns)
    started_at_wall_ns: int | None = None
    finished_at_wall_ns: int | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    stop: dict[str, Any] | None = None
    steps_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "kind": self.kind,
            "state": self.state,
            "success": self.state == CommandState.SUCCEEDED.value
            and (self.result or {}).get("success", True),
            "created_at_wall_ns": self.created_at_wall_ns,
            "started_at_wall_ns": self.started_at_wall_ns,
            "finished_at_wall_ns": self.finished_at_wall_ns,
            "steps_used": self.steps_used,
            "result": self.result,
            "error": self.error,
            "stop": self.stop,
        }


class _Connection:
    def __init__(self, sock: Any) -> None:
        self.sock = sock
        self.lock = threading.Lock()
        self.closed = False

    def send(self, message: dict[str, Any]) -> None:
        data = encode_message(message)
        with self.lock:
            if self.closed:
                return
            try:
                self.sock.sendall(data)
            except OSError:
                self.closed = True

    def close(self) -> None:
        with self.lock:
            self.closed = True
            with contextlib.suppress(OSError):
                self.sock.shutdown(2)


class _ThreadedServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], daemon: HardwareDaemon) -> None:
        self.daemon_ref = daemon
        super().__init__(server_address, _Handler)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        conn = _Connection(self.request)
        self.server.daemon_ref.connection_opened(conn)  # type: ignore[attr-defined]
        try:
            while True:
                line = self.rfile.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    conn.send(
                        response("", error_obj=error(ErrorCode.BAD_REQUEST, "message too large"))
                    )
                    break
                try:
                    message = decode_message(line)
                except ValueError as exc:
                    conn.send(response("", error_obj=error(ErrorCode.BAD_REQUEST, str(exc))))
                    continue
                thread = threading.Thread(
                    target=self.server.daemon_ref.dispatch,  # type: ignore[attr-defined]
                    args=(conn, message),
                    daemon=True,
                )
                thread.start()
        finally:
            conn.close()
            self.server.daemon_ref.connection_closed(conn)  # type: ignore[attr-defined]


class HardwareDaemon:
    def __init__(
        self,
        config: StationConfig,
        *,
        station: StationBase | None = None,
        cameras: CameraManager | None = None,
    ) -> None:
        self.config = config
        self.calibration = config.calibration
        self.mode = config.execution.mode
        self.active_arms = tuple(config.execution.active_arms)
        self._active = frozenset(self.active_arms)
        self.station = station or make_station(config)
        self.cameras = cameras or CameraManager(config, self.calibration)
        self.kinematics = LitearmKinematics(self.calibration)
        self.planner = Planner(
            self.config.calibration, self.config.execution, self.config.safety, self.kinematics
        )
        self.state = SessionState.READ_ONLY
        self.session_id: str | None = None
        self.session_client: str | None = None
        self.session_conn: _Connection | None = None
        self.last_heartbeat_mono: int = 0
        self.stop_result: StopResult | None = None
        self.fault_reason: str | None = None
        self.current: _Command | None = None
        self.commands: dict[str, _Command] = {}
        self.snapshots: collections.OrderedDict[int, _Snapshot] = collections.OrderedDict()
        self.snapshot_index = 0
        self.latest_reading: dict[str, Any] | None = None
        self.latest_reading_mono: int = 0
        self.history: collections.deque[tuple[int, dict[str, list[float]]]] = collections.deque(
            maxlen=max(20, self.config.snapshot.history)
        )
        self._lock = threading.RLock()
        self._snapshot_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._queue: collections.deque[_Command] = collections.deque()
        self._wake = threading.Event()
        self._stop_monitor = threading.Event()
        self._executor = threading.Thread(
            target=self._executor_loop, name="hw-executor", daemon=True
        )
        self._monitor = threading.Thread(target=self._monitor_loop, name="hw-monitor", daemon=True)
        self._server: _ThreadedServer | None = None
        self._server_thread: threading.Thread | None = None
        self._closing = False

    # -- lifecycle ----------------------------------------------------------

    def start_service(self) -> None:
        self.station.open()
        try:
            self.cameras.open()
            self._sample_state()
        except BaseException:
            self.cameras.close()
            self.station.close()
            raise
        self._executor.start()
        self._monitor.start()

    def listen(self, host: str | None = None, port: int | None = None) -> tuple[str, int]:
        address = (
            host or self.config.service.host,
            int(self.config.service.port if port is None else port),
        )
        server = _ThreadedServer(address, self)
        self._server = server
        self._server_thread = threading.Thread(
            target=server.serve_forever, name="hw-server", daemon=True
        )
        self._server_thread.start()
        bound = server.server_address
        return str(bound[0]), int(bound[1])

    def serve_forever(self, host: str | None = None, port: int | None = None) -> None:
        self.start_service()
        host, port = self.listen(host, port)
        log.info("hardware service listening on %s:%s mode=%s", host, port, self.mode.value)
        try:
            while self._server_thread is not None and self._server_thread.is_alive():
                self._server_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            log.info("interrupt received")
        finally:
            self.close()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_monitor.set()
        self._wake.set()
        with self._lock:
            running = self.current is not None and not self.current.done.is_set()
        if running:
            self._begin_fault(
                "service shutting down with a command in flight", kind=StopKind.EMERGENCY
            )
        if self._server is not None:
            with contextlib.suppress(Exception):
                self._server.shutdown()
            with contextlib.suppress(Exception):
                self._server.server_close()
        # ``start_service`` may fail before either worker is started (for
        # example the arm port is already held by another process).  Joining an
        # unstarted thread raises RuntimeError and would hide the real boot
        # error, so only join workers that actually started.
        if self._executor.is_alive() or self._executor.ident is not None:
            self._executor.join(timeout=2.0)
        if self._monitor.is_alive() or self._monitor.ident is not None:
            self._monitor.join(timeout=2.0)
        with contextlib.suppress(Exception):
            self.cameras.close()
        with contextlib.suppress(Exception):
            self.station.close()

    def connection_opened(self, conn: _Connection) -> None:
        _ = conn

    def connection_closed(self, conn: _Connection) -> None:
        with self._lock:
            if self.session_conn is conn and self.session_id is not None:
                # A dropped runner is a lost heartbeat: stop and fault, do not
                # silently resume on the next connection.
                self._begin_fault("runner connection closed")

    # -- request dispatch ---------------------------------------------------

    def dispatch(self, conn: _Connection, message: dict[str, Any]) -> None:
        request_id = str(message.get("id") or "")
        try:
            if message.get("v") != PROTOCOL_VERSION:
                raise _RequestError(
                    ErrorCode.VERSION_MISMATCH,
                    f"protocol version {message.get('v')!r} != {PROTOCOL_VERSION}",
                )
            kind = require_str(message.get("type"), "type")
            if kind == "hello":
                result = self._hello(conn, message)
            elif kind == "open_session":
                result = self._open_session(conn, message)
            elif kind == "heartbeat":
                result = self._heartbeat(message)
            elif kind == "snapshot":
                result = self._snapshot_request(message)
            elif kind == "get_artifact":
                result = self._get_artifact(message)
            elif kind == "command":
                result = self._submit_command(message)
            elif kind == "command_status":
                result = self._command_status(message)
            elif kind == "wait_command":
                result = self._wait_command(message)
            elif kind == "cancel":
                result = self._cancel(message)
            elif kind == "emergency_stop":
                result = self._emergency_stop(message)
            elif kind == "disconnect":
                result = self._disconnect(message)
            elif kind == "close_session":
                result = self._close_session(message)
            elif kind == "status":
                result = self._status(message)
            else:
                raise _RequestError(ErrorCode.BAD_REQUEST, f"unknown request type {kind!r}")
            conn.send(response(request_id, result=result))
        except _RequestError as exc:
            conn.send(response(request_id, error_obj=error(exc.code, str(exc), detail=exc.detail)))
        except Exception as exc:  # noqa: BLE001 - never crash a connection thread silently
            log.exception("request %s failed", request_id)
            conn.send(
                response(
                    request_id, error_obj=error(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")
                )
            )

    # -- session / requests -------------------------------------------------

    def _hello(self, conn: _Connection, message: dict[str, Any]) -> dict[str, Any]:
        client_wall = message.get("client_wall_ns")
        server_wall = now_wall_ns()
        offset = None
        if client_wall is not None:
            try:
                offset = server_wall - int(client_wall)
            except (TypeError, ValueError):
                offset = None
        _ = conn
        return {
            "protocol_version": PROTOCOL_VERSION,
            "station_id": self.config.station_id,
            "config_sha256": self.config.sha256,
            "calibration_sha256": self.config.calibration_sha256,
            "execution_mode": self.mode.value,
            "active_arms": list(self.active_arms),
            "capabilities": self.station.capabilities(),
            "cameras": sorted(self.cameras.names),
            "state": self.state.value,
            "session_active": self.session_id is not None,
            "clock": {
                "server_wall_ns": server_wall,
                "server_monotonic_ns": now_monotonic_ns(),
                "offset_server_minus_client_ns": offset,
                "source": "wall_clock_requires_ntp_across_hosts",
            },
            "limits": {
                "max_speed": self.config.execution.max_speed,
                "max_joint_step_rad": self.config.execution.max_joint_step_rad,
                "max_joint_speed_rad_s": self.config.execution.max_joint_speed_rad_s,
                "command_timeout_s": self.config.execution.command_timeout_s,
            },
            "success_evaluator": False,
        }

    def _check_session(
        self,
        message: dict[str, Any],
        *,
        allow_read_only: bool = False,
        allow_fault: bool = False,
        allow_stopped: bool = False,
    ) -> None:
        session = message.get("session")
        if self.session_id is None:
            raise _RequestError(ErrorCode.NO_SESSION, "open a session first")
        if session != self.session_id:
            raise _RequestError(ErrorCode.SESSION_MISMATCH, "wrong or expired session id")
        if self.state is SessionState.FAULT and not allow_fault:
            raise _RequestError(ErrorCode.FAULTED, self.fault_reason or "service is faulted")
        if self.state is SessionState.STOPPED and not allow_stopped:
            raise _RequestError(ErrorCode.STOPPED, "service is stopped; restart it")
        if self.state is SessionState.READ_ONLY and not allow_read_only:
            raise _RequestError(ErrorCode.READ_ONLY, "service is in read-only mode")

    def _open_session(self, conn: _Connection, message: dict[str, Any]) -> dict[str, Any]:
        protocol = message.get("protocol_version", message.get("v"))
        if protocol != PROTOCOL_VERSION:
            raise _RequestError(ErrorCode.VERSION_MISMATCH, "protocol mismatch")
        config_hash = require_str(message.get("config_sha256"), "config_sha256")
        calibration_hash = require_str(message.get("calibration_sha256"), "calibration_sha256")
        if config_hash != self.config.sha256:
            raise _RequestError(
                ErrorCode.CONFIG_MISMATCH,
                "runner and service loaded different station configs",
                detail={"service": self.config.sha256, "runner": config_hash},
            )
        if calibration_hash != self.config.calibration_sha256:
            raise _RequestError(
                ErrorCode.CALIBRATION_MISMATCH,
                "runner and service loaded different calibrations",
                detail={"service": self.config.calibration_sha256, "runner": calibration_hash},
            )
        with self._lock:
            if self.state is SessionState.FAULT:
                raise _RequestError(ErrorCode.FAULTED, self.fault_reason or "service is faulted")
            if self.state is SessionState.STOPPED:
                raise _RequestError(ErrorCode.STOPPED, "service was stopped; restart it")
            if self.session_id is not None:
                raise _RequestError(ErrorCode.BUSY, "another runner owns the station")
            problems = self.station.verify_ready()
            if self.mode is not ExecutionMode.READ_ONLY and problems:
                self._enter_fault("station not ready: " + "; ".join(problems))
                raise _RequestError(ErrorCode.NOT_READY, "station is not ready", detail=problems)
            self.session_id = new_id("session")
            self.session_client = str(message.get("client_id") or "runner")
            self.session_conn = conn
            self.last_heartbeat_mono = now_monotonic_ns()
            self.state = (
                SessionState.READ_ONLY
                if self.mode is ExecutionMode.READ_ONLY
                else SessionState.READY
            )
            return {
                "session": self.session_id,
                "state": self.state.value,
                "execution_mode": self.mode.value,
                "config_sha256": self.config.sha256,
                "calibration_sha256": self.config.calibration_sha256,
                "success_evaluator": False,
                "read_only": self.mode is ExecutionMode.READ_ONLY,
            }

    def _heartbeat(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True)
        self.last_heartbeat_mono = now_monotonic_ns()
        return {"state": self.state.value, "time": now_wall_ns()}

    def _status(self, message: dict[str, Any]) -> dict[str, Any]:
        session = message.get("session")
        if self.session_id is not None and session not in (None, self.session_id):
            raise _RequestError(ErrorCode.SESSION_MISMATCH, "wrong session id")
        return {
            "state": self.state.value,
            "execution_mode": self.mode.value,
            "session_active": self.session_id is not None,
            "current_command": self.current.to_dict() if self.current else None,
            "stop": self.stop_result.to_dict() if self.stop_result else None,
            "fault": self.fault_reason,
            "success_evaluator": False,
        }

    # -- snapshots ----------------------------------------------------------

    def _sample_state(self) -> dict[str, Any]:
        with self._read_lock:
            reading = self.station.read()
            self.latest_reading = reading
            self.latest_reading_mono = now_monotonic_ns()
            mono = reading.get("captured_at_monotonic_ns", now_monotonic_ns())
            q = {arm: list(reading["arms"][arm]["q_model"]) for arm in self.active_arms}
            self.history.append((int(mono), q))
            return reading

    def _monitor_loop(self) -> None:
        period = 1.0 / 20.0
        next_beat = time.monotonic()
        while not self._stop_monitor.is_set():
            now = time.monotonic()
            if now >= next_beat:
                next_beat = now + period
                try:
                    self._sample_state()
                except Exception as exc:  # noqa: BLE001 - a lost read is a fault while executing
                    log.warning("state sample failed: %s", exc)
                    if self.state in (SessionState.READY, SessionState.EXECUTING):
                        self._begin_fault(f"state read failed: {exc}")
                if self.session_id is not None and self.state in (
                    SessionState.READY,
                    SessionState.EXECUTING,
                ):
                    age_s = (now_monotonic_ns() - self.last_heartbeat_mono) / 1e9
                    if age_s > self.config.service.heartbeat_timeout_s:
                        self._begin_fault(f"heartbeat lost for {age_s:.2f}s")
            self._stop_monitor.wait(0.02)

    def _q_at(self, wall_ns: int, mono_ns: int) -> dict[str, list[float]]:
        target = int(mono_ns)
        samples = list(self.history)
        if not samples:
            raise _RequestError(
                ErrorCode.STALE_STATE, "no joint history for camera synchronization"
            )
        before = None
        after = None
        for ts, q in samples:
            if ts <= target:
                before = (ts, q)
            if ts >= target and after is None:
                after = (ts, q)
        if before is None:
            before = samples[0]
        if after is None:
            after = samples[-1]
        gap = max(abs(target - before[0]), abs(after[0] - target)) / 1e9
        if gap > self.config.snapshot.camera_joint_sync_s:
            raise _RequestError(
                ErrorCode.STALE_STATE,
                f"camera frame is {gap:.3f}s from the nearest joint sample, over "
                f"{self.config.snapshot.camera_joint_sync_s:.3f}s",
            )
        if before[0] == after[0]:
            return {arm: list(before[1][arm]) for arm in self.active_arms}
        t = (target - before[0]) / (after[0] - before[0])
        return {
            arm: [
                float(a) + (float(b) - float(a)) * t
                for a, b in zip(before[1][arm], after[1][arm], strict=True)
            ]
            for arm in self.active_arms
        }

    def _make_artifact(self, name: str, media_type: str, data: bytes) -> _Artifact:
        import hashlib

        return _Artifact(
            name=name,
            media_type=media_type,
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def _capture_snapshot(self) -> _Snapshot:
        with self._snapshot_lock:
            # Use the latest monitor sample so a snapshot does not block behind a
            # synchronous move command; require it to be fresh.
            with self._lock:
                reading = self.latest_reading
                reading_mono = self.latest_reading_mono
            if (
                reading is None
                or (now_monotonic_ns() - reading_mono) / 1e9 > self.config.snapshot.max_age_s
            ):
                reading = self._sample_state()
                reading_mono = self.latest_reading_mono
            # Bracket the camera capture with joint samples; if the arm moves
            # during capture, _q_at interpolates between them.
            self._sample_state()
            frames = self.cameras.capture()
            reading = self._sample_state()
            reading_mono = self.latest_reading_mono
            artifacts: dict[str, _Artifact] = {}
            cameras: dict[str, Any] = {}
            for name, frame in frames.items():
                cam_cal = self.calibration.camera(name)
                arm = cam_cal.arm
                try:
                    q = self._q_at(frame.captured_at_wall_ns, frame.captured_at_monotonic_ns)
                except _RequestError:
                    raise
                if cam_cal.mount == "world":
                    c2w = np.asarray(cam_cal.transform, dtype=float)
                else:
                    if arm is None:
                        raise _RequestError(ErrorCode.INTERNAL, "tool camera has no arm")
                    c2w = self.kinematics.camera_pose(name, arm, q[arm])
                color_name = f"snap_{self.snapshot_index + 1:06d}/{name}.png"
                color = _png(frame.rgb)
                artifacts[color_name] = self._make_artifact(color_name, "image/png", color)
                depth_name = None
                if frame.depth is not None:
                    depth_name = f"snap_{self.snapshot_index + 1:06d}/{name}_depth.npy"
                    buf = io.BytesIO()
                    np.save(buf, frame.depth.astype(np.float32))
                    data = buf.getvalue()
                    artifacts[depth_name] = self._make_artifact(
                        depth_name, "application/x-npy", data
                    )
                intrinsic = frame.intrinsic if frame.intrinsic is not None else cam_cal.intrinsic
                cameras[name] = {
                    "shape": [int(frame.rgb.shape[0]), int(frame.rgb.shape[1]), 3],
                    "intrinsic_matrix": [[float(v) for v in row] for row in np.asarray(intrinsic)],
                    "distortion_coefficients": list(frame.distortion),
                    "camera_to_world": [[float(v) for v in row] for row in c2w],
                    "color_artifact": color_name,
                    "depth_artifact": depth_name,
                    "has_depth": depth_name is not None,
                    "arm": arm,
                    "q_model_at_capture": q[arm] if arm else None,
                    "captured_at_wall_ns": int(frame.captured_at_wall_ns),
                    "captured_at_monotonic_ns": int(frame.captured_at_monotonic_ns),
                    "timestamp_source": frame.timestamp_source,
                }
            reading = dict(reading)
            reading["cameras"] = cameras
            index = self.snapshot_index + 1
            self.snapshot_index = index
            payload = {
                "index": index,
                "station_id": self.config.station_id,
                "config_sha256": self.config.sha256,
                "calibration_sha256": self.config.calibration_sha256,
                "captured_at_wall_ns": int(reading.get("captured_at_wall_ns", now_wall_ns())),
                "captured_at_monotonic_ns": int(
                    reading.get("captured_at_monotonic_ns", now_monotonic_ns())
                ),
                "reading": reading,
                "hardware": {
                    "execution_state": self.state.value,
                    "execution_mode": self.mode.value,
                    "fault": self.state is SessionState.FAULT,
                    "fault_reason": self.fault_reason,
                    "stop": self.stop_result.to_dict() if self.stop_result else None,
                    "station_id": self.config.station_id,
                    "calibration_version": self.calibration.version,
                    "captured_at_wall_ns": int(reading.get("captured_at_wall_ns", now_wall_ns())),
                    "captured_at_monotonic_ns": int(
                        reading.get("captured_at_monotonic_ns", now_monotonic_ns())
                    ),
                },
                "artifacts": {
                    name: {
                        "media_type": item.media_type,
                        "bytes": len(item.data),
                        "sha256": item.sha256,
                    }
                    for name, item in artifacts.items()
                },
            }
            snapshot = _Snapshot(index=index, payload=payload, artifacts=artifacts)
            self.snapshots[index] = snapshot
            while len(self.snapshots) > self.config.snapshot.history:
                self.snapshots.popitem(last=False)
            return snapshot

    def _snapshot_request(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True, allow_fault=True, allow_stopped=True)
        snapshot = self._capture_snapshot()
        return snapshot.payload

    def _get_artifact(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True, allow_fault=True, allow_stopped=True)
        name = require_str(message.get("name"), "name")
        for snapshot in reversed(self.snapshots.values()):
            item = snapshot.artifacts.get(name)
            if item is not None:
                return {
                    "name": name,
                    "media_type": item.media_type,
                    "sha256": item.sha256,
                    "data_b64": base64.b64encode(item.data).decode("ascii"),
                }
        raise _RequestError(ErrorCode.BAD_REQUEST, f"artifact {name!r} is not retained")

    # -- commands -----------------------------------------------------------

    def _validate_command_envelope(
        self, message: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        command_id = require_str(message.get("command_id"), "command_id")
        kind = require_str(message.get("kind"), "kind")
        if kind not in COMMAND_KINDS:
            raise _RequestError(ErrorCode.UNSUPPORTED, f"unsupported command kind {kind!r}")
        payload = require_dict(message.get("payload"), "payload")
        digest = require_str(message.get("digest"), "digest")
        if digest != command_digest(kind, payload):
            raise _RequestError(ErrorCode.BAD_REQUEST, "command digest does not match the payload")
        return command_id, kind, payload

    def _submit_command(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message)
        if self.mode is ExecutionMode.READ_ONLY:
            raise _RequestError(ErrorCode.READ_ONLY, "read-only service refuses motion")
        command_id, kind, payload = self._validate_command_envelope(message)
        with self._lock:
            # A repeated request is answered from the record even while that
            # command is still running.  It must never be enqueued twice.
            existing = self.commands.get(command_id)
            if existing is not None:
                if existing.digest != require_str(message.get("digest"), "digest"):
                    raise _RequestError(
                        ErrorCode.COMMAND_CONFLICT,
                        "the same command id was used with a different payload",
                    )
                return existing.to_dict()
            if self.current is not None and not self.current.done.is_set():
                raise _RequestError(ErrorCode.BUSY, "a command is already running")
            command = _Command(
                command_id=command_id, kind=kind, digest=message["digest"], payload=payload
            )
            self.commands[command_id] = command
            self._queue.append(command)
            self._wake.set()
            return command.to_dict()

    def _command_status(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True)
        command_id = require_str(message.get("command_id"), "command_id")
        command = self.commands.get(command_id)
        if command is None:
            raise _RequestError(ErrorCode.BAD_REQUEST, f"unknown command {command_id!r}")
        return command.to_dict()

    def _wait_command(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True)
        command_id = require_str(message.get("command_id"), "command_id")
        command = self.commands.get(command_id)
        if command is None:
            raise _RequestError(ErrorCode.BAD_REQUEST, f"unknown command {command_id!r}")
        timeout = require_finite(message.get("timeout_s", 60.0), "timeout_s")
        command.done.wait(max(0.0, float(timeout)))
        return command.to_dict()

    def _cancel(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True)
        with self._lock:
            command = self.current
            if command is not None and not command.done.is_set():
                command.cancel.set()
                return {"accepted": True, "command_id": command.command_id, "state": command.state}
            if self.state is SessionState.READY:
                stop = self._apply_stop(StopKind.CANCEL)
                return {
                    "accepted": True,
                    "command_id": None,
                    "state": self.state.value,
                    "stop": stop.to_dict(),
                }
            return {"accepted": False, "command_id": None, "state": self.state.value}

    def _emergency_stop(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True)
        with self._lock:
            if self.current is not None:
                self.current.cancel.set()
        self._begin_fault("explicit emergency stop", kind=StopKind.EMERGENCY)
        return {
            "accepted": True,
            "state": self.state.value,
            "stop": self.stop_result.to_dict() if self.stop_result else None,
        }

    def _disconnect(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True)
        with self._lock:
            if self.current is not None:
                self.current.cancel.set()
        self._begin_fault("runner disconnected")
        return {"accepted": True, "state": self.state.value}

    def _close_session(self, message: dict[str, Any]) -> dict[str, Any]:
        self._check_session(message, allow_read_only=True, allow_fault=True, allow_stopped=True)
        with self._lock:
            command = self.current
            if command is not None and not command.done.is_set():
                command.cancel.set()
        if command is not None and not command.done.wait(
            self.config.execution.command_timeout_s + 2.0
        ):
            self._begin_fault(
                "command did not stop before close_session timeout", kind=StopKind.TIMEOUT
            )
        with self._lock:
            faulted = self.state is SessionState.FAULT
        # A fault's emergency-stop evidence must not be overwritten by the
        # cooperative boundary stop that follows a normal shutdown.
        stop = (
            self.stop_result
            if faulted and self.stop_result is not None
            else self._apply_stop(StopKind.CANCEL)
        )
        with self._lock:
            self.session_id = None
            self.session_client = None
            self.session_conn = None
            self.state = (
                SessionState.FAULT if self.state is SessionState.FAULT else SessionState.STOPPED
            )
        return {
            "state": self.state.value,
            "stop": stop.to_dict(),
            "success_evaluator": False,
        }

    # -- stop / fault state machine ----------------------------------------

    def _station_stop(self, kind: StopKind) -> StopResult:
        """Call the device stop with a hard return bound.

        A stop call that does not return is not evidence of a stop; report it
        as unconfirmed instead of blocking the service forever.
        """

        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["result"] = self.station.stop(kind, self.active_arms)
            except BaseException as exc:  # noqa: BLE001 - recorded as unconfirmed
                box["error"] = exc

        thread = threading.Thread(target=run, name=f"hw-stop-{kind.value}", daemon=True)
        thread.start()
        thread.join(timeout=float(self.config.safety.stop_timeout_s))
        if thread.is_alive():
            return StopResult(
                requested=True,
                kind=kind.value,
                confirmed=False,
                state="stop_unconfirmed",
                method=None,
                message=(
                    f"station stop did not return within {self.config.safety.stop_timeout_s:.2f}s"
                ),
                requested_at_wall_ns=now_wall_ns(),
            )
        if "error" in box:
            exc = box["error"]
            return StopResult(
                requested=True,
                kind=kind.value,
                confirmed=False,
                state="stop_unconfirmed",
                method=None,
                message=f"stop raised {type(exc).__name__}: {exc}",
                requested_at_wall_ns=now_wall_ns(),
            )
        result = box.get("result")
        if not isinstance(result, StopResult):
            return StopResult(
                requested=True,
                kind=kind.value,
                confirmed=False,
                state="stop_unconfirmed",
                method=None,
                message="station stop returned no result",
                requested_at_wall_ns=now_wall_ns(),
            )
        return result

    def _apply_stop(self, kind: StopKind) -> StopResult:
        stop = self._station_stop(kind)
        self.stop_result = stop
        if kind is StopKind.CANCEL:
            with self._lock:
                if self.state is not SessionState.FAULT:
                    self.state = SessionState.STOPPED if stop.confirmed else SessionState.FAULT
                    if not stop.confirmed:
                        self.fault_reason = stop.message or "cancel stop could not be confirmed"
        return stop

    def _begin_fault(self, reason: str, *, kind: StopKind = StopKind.TIMEOUT) -> None:
        with self._lock:
            if self.state is SessionState.FAULT:
                return
            self.state = SessionState.STOPPING
            self.fault_reason = reason
        # Do not hold the request lock while the vendor SDK stop call runs.
        stop = self._station_stop(kind)
        with self._lock:
            self.stop_result = stop
            self.state = SessionState.FAULT
            if not stop.confirmed:
                self.fault_reason = f"{reason}; stop unconfirmed: {stop.message}"
            command = self.current
            if command is not None and not command.done.is_set():
                command.cancel.set()
        log.error("hardware service fault: %s", self.fault_reason)

    def _enter_fault(self, reason: str) -> None:
        with self._lock:
            self.state = SessionState.FAULT
            self.fault_reason = reason

    # -- executor -----------------------------------------------------------

    def _executor_loop(self) -> None:
        while not self._stop_monitor.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    command = self._queue.popleft()
                    if self.state is not SessionState.READY:
                        command.state = CommandState.REJECTED.value
                        command.error = error(
                            ErrorCode.NOT_READY, f"service state {self.state.value}"
                        )
                        command.done.set()
                        continue
                    self.current = command
                    command.state = CommandState.RUNNING.value
                    command.started_at_wall_ns = now_wall_ns()
                    self.state = SessionState.EXECUTING
                try:
                    self._run_command(command)
                except BaseException as exc:  # noqa: BLE001 - recorded as a fault
                    log.exception("command %s crashed", command.command_id)
                    command.state = CommandState.FAULTED.value
                    command.error = error(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")
                    self._begin_fault(f"command {command.command_id} crashed: {exc}")
                finally:
                    command.finished_at_wall_ns = now_wall_ns()
                    command.done.set()
                    with self._lock:
                        if self.state is SessionState.EXECUTING:
                            self.state = SessionState.READY
                        self.current = None

    def _run_command(self, command: _Command) -> None:
        started = time.monotonic()
        try:
            problems = self.station.verify_ready()
            if problems:
                self._enter_fault("station not ready before command: " + "; ".join(problems))
                command.state = CommandState.REJECTED.value
                command.error = error(
                    ErrorCode.FAULTED,
                    "station is not ready; command refused",
                    detail=problems,
                )
                return
            if command.kind == "trajectory":
                self._run_trajectory(command, started)
            elif command.kind == "grip":
                self._run_grip(command)
            else:
                command.state = CommandState.REJECTED.value
                command.error = error(ErrorCode.UNSUPPORTED, f"unknown command {command.kind!r}")
        except DeviceCancelled as exc:
            command.state = CommandState.CANCELLED.value
            command.error = error(ErrorCode.CANCELLED, str(exc))
            command.stop = self._apply_stop(StopKind.CANCEL).to_dict()
        except (DeviceError, NotReady) as exc:
            command.state = CommandState.FAULTED.value
            command.error = error(ErrorCode.DEVICE_FAULT, str(exc), detail=exc.fault_detail)
            self._begin_fault(f"device error: {exc}", kind=StopKind.EMERGENCY)
            command.stop = self.stop_result.to_dict() if self.stop_result else None
        except _RequestError as exc:
            command.state = CommandState.REJECTED.value
            command.error = error(exc.code, str(exc), detail=exc.detail)
        except PlanningError as exc:
            command.state = CommandState.REJECTED.value
            command.error = error(ErrorCode.COLLISION, str(exc), detail=exc.detail)
        except Exception as exc:  # noqa: BLE001
            command.state = CommandState.FAULTED.value
            command.error = error(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")
            self._begin_fault(f"command failed: {exc}", kind=StopKind.EMERGENCY)
            command.stop = self.stop_result.to_dict() if self.stop_result else None

    def _validate_start(self, arm: str, q_start: list[float], reading: dict[str, Any]) -> None:
        actual = list(reading["arms"][arm]["q_model"])
        err = max(abs(float(a) - float(b)) for a, b in zip(actual, q_start, strict=True))
        if err > self.config.execution.start_tolerance_rad:
            raise _RequestError(
                ErrorCode.STALE_STATE,
                f"planned start differs from measured {arm} joints by {err:.4f} rad, over "
                f"{self.config.execution.start_tolerance_rad:.4f}; plan from a fresh state",
            )

    def _check_command_arm(self, payload: dict[str, Any]) -> str:
        arm = require_str(payload.get("arm"), "payload.arm")
        if arm not in self._active:
            raise _RequestError(
                ErrorCode.BAD_REQUEST,
                f"arm {arm!r} is not active on this station; active arms: {list(self.active_arms)}",
                detail={"active_arms": list(self.active_arms)},
            )
        return arm

    def _has_gripper(self, arm: str) -> bool:
        cfg = self.config.arms.get(arm)
        return cfg is not None and cfg.gripper is not None

    def _require_gripper(self, arm: str) -> None:
        if not self._has_gripper(arm):
            raise _RequestError(
                ErrorCode.BAD_REQUEST,
                f"arm {arm!r} has no gripper configured on this station",
                detail={"arm": arm, "has_gripper": False},
            )

    def _run_trajectory(self, command: _Command, started: float) -> None:
        arm = self._check_command_arm(command.payload)
        waypoints = command.payload.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise _RequestError(ErrorCode.BAD_REQUEST, "payload.waypoints must be a non-empty list")
        q_start = command.payload.get("q_start")
        if not isinstance(q_start, list) or len(q_start) != 7:
            raise _RequestError(ErrorCode.BAD_REQUEST, "payload.q_start must be 7 joint angles")
        reading = self._sample_state()
        ages = [reading["arms"][a].get("data_age_s") for a in self.active_arms]
        if any(age is None or float(age) > self.config.snapshot.max_age_s for age in ages):
            raise _RequestError(
                ErrorCode.STALE_STATE,
                "measured arm state is missing or older than snapshot.max_age_s; "
                "refresh before planning",
                detail={"data_age_s": ages},
            )
        self._validate_start(arm, [float(v) for v in q_start], reading)
        other = next((name for name in self.active_arms if name != arm), None)
        other_q = reading["arms"][other]["q_model"] if other is not None else None
        errors = self.planner.validate_waypoints(
            arm,
            [float(v) for v in q_start],
            waypoints,
            other_arm=other,
            other_q=other_q,
        )
        if errors:
            raise PlanningError("; ".join(errors), detail={"violations": errors})
        # Re-check the configured per-segment duration on the device side too;
        # the runner's plan is advisory, not the authority.
        self.planner._validate_duration([float(v) for v in q_start], waypoints)
        self.planner.validate_tcp_speed(arm, [float(v) for v in q_start], waypoints)
        speed = min(
            self.config.execution.max_speed,
            max(0.01, float(command.payload.get("speed", self.config.execution.max_speed))),
        )
        timeout = min(
            self.config.execution.command_timeout_s,
            max(
                0.1,
                float(command.payload.get("timeout_s", self.config.execution.command_timeout_s)),
            ),
        )
        baseline = command.payload.get("carry_baseline")
        if baseline is not None:
            self._require_gripper(arm)
        slip_drop = float(command.payload.get("slip_drop", 0.02))
        actual_q = [float(v) for v in q_start]
        slip = False
        for index, way in enumerate(waypoints):
            if command.cancel.is_set():
                raise DeviceCancelled("cancel requested between segments", device=f"arm_{arm}")
            if time.monotonic() - started > timeout:
                self._begin_fault(f"trajectory timeout after {time.monotonic() - started:.1f}s")
                command.state = CommandState.TIMED_OUT.value
                command.error = error(ErrorCode.TIMEOUT, "trajectory command timeout")
                command.stop = self.stop_result.to_dict() if self.stop_result else None
                return
            q_model = [float(v) for v in way]
            q_sdk = [float(v) for v in self.kinematics.sdk_from_model(arm, q_model)]
            outcome = self.station.move_joints(
                arm, q_sdk, speed, cancel=command.cancel, timeout_s=timeout
            )
            command.steps_used += 1
            actual_q = list(outcome.q_model)
            self._sample_state()
            if baseline is not None:
                grip = self._sample_state()["grippers"][arm]
                opening = float(grip["opening"])
                if opening < float(baseline) - slip_drop:
                    slip = True
                    command.result = {
                        "arm": arm,
                        "q_model": actual_q,
                        "steps_used": command.steps_used,
                        "reached": False,
                        "slip": True,
                        "slip_opening": opening,
                        "slip_baseline": float(baseline),
                        "success": False,
                        "segments_completed": index,
                    }
                    command.state = CommandState.SUCCEEDED.value
                    return
        final = self._sample_state()["arms"][arm]
        target = [float(v) for v in waypoints[-1]]
        err = max(abs(a - b) for a, b in zip(final["q_model"], target, strict=True))
        command.result = {
            "arm": arm,
            "q_model": [float(v) for v in final["q_model"]],
            "q_sdk": [float(v) for v in final["q_sdk"]],
            "steps_used": command.steps_used,
            "reached": err <= self.config.execution.settle_tolerance_rad,
            "max_joint_error_rad": err,
            "slip": slip,
            "success": err <= self.config.execution.settle_tolerance_rad and not slip,
            "segments_completed": len(waypoints),
        }
        command.state = CommandState.SUCCEEDED.value

    def _run_grip(self, command: _Command) -> None:
        arm = self._check_command_arm(command.payload)
        self._require_gripper(arm)
        action = require_str(command.payload.get("command"), "payload.command")
        if action not in ("open", "close", "keep"):
            raise _RequestError(ErrorCode.BAD_REQUEST, "payload.command must be open/close/keep")
        width = command.payload.get("width_m")
        if width is not None:
            width = require_finite(width, "payload.width_m")
            if width <= 0:
                raise _RequestError(ErrorCode.BAD_REQUEST, "payload.width_m must be positive")
        force = command.payload.get("force_n")
        if force is not None:
            force = require_finite(force, "payload.force_n")
            if force <= 0:
                raise _RequestError(ErrorCode.BAD_REQUEST, "payload.force_n must be positive")
        command.steps_used = 1
        outcome = self.station.set_gripper(
            arm,
            action,
            width_m=width,
            force_n=force,
            timeout_s=self.config.execution.command_timeout_s,
            cancel=command.cancel,
        )
        command.result = {
            "arm": arm,
            "gripper": "closed" if action == "close" else ("open" if action == "open" else "keep"),
            "opening": outcome.opening,
            "width_m": outcome.width_m,
            "holding": outcome.holding,
            "settled": outcome.settled,
            "force_n": outcome.force_n,
            "error_code": outcome.error_code,
            "released": outcome.released,
            "note": outcome.note,
            "steps_used": 1,
            "success": bool(outcome.settled),
        }
        command.state = CommandState.SUCCEEDED.value
        self._sample_state()


class _RequestError(Exception):
    def __init__(self, code: ErrorCode, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


def _png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _setup_logging(path: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="absolute path to the station JSON")
    parser.add_argument(
        "--execution-mode",
        choices=[m.value for m in ExecutionMode],
        default=None,
        help="override execution.mode; live still requires an executable calibration",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log", default=None, help="diagnostic log path")
    args = parser.parse_args(argv)
    _setup_logging(args.log)
    try:
        config = load_station(args.config, mode=args.execution_mode)
    except Exception as exc:  # noqa: BLE001
        log.error("configuration refused: %s", exc)
        return 2
    daemon = HardwareDaemon(config)
    stop_requested = {"value": False}

    def on_signal(signum: int, frame: Any) -> None:
        _ = signum, frame
        if stop_requested["value"]:
            raise KeyboardInterrupt
        stop_requested["value"] = True
        log.warning("shutdown requested; stopping the listener")
        server = daemon._server
        if server is not None:
            with contextlib.suppress(Exception):
                server.shutdown()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        daemon.serve_forever(args.host, args.port)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001 - the process must explain a boot failure
        log.exception("hardware service failed")
        print(f"hardware service failed: {exc}", file=sys.stderr)
        return 2
    finally:
        daemon.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HardwareDaemon", "main"]
