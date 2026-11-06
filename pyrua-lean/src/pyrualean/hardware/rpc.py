# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""TCP client for the hardware service.

The client keeps at most one outstanding request per generated id and never
retries a motion command automatically.  If the connection dies while a
trajectory is running, the caller receives :class:`ConnectionLost` with the
command id; resubmitting it is a decision for the operator/policy, not a
hidden retry.
"""

from __future__ import annotations

import base64
import hashlib
import socket
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from .config import StationConfig
from .protocol import (
    MAX_LINE_BYTES,
    PROTOCOL_VERSION,
    ErrorCode,
    command_digest,
    decode_message,
    encode_message,
    new_id,
    now_wall_ns,
)


class RpcError(RuntimeError):
    def __init__(self, code: str, message: str, *, detail: Any = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.detail = detail


class ConnectionLost(RpcError):
    def __init__(self, message: str, *, command_id: str | None = None) -> None:
        super().__init__(ErrorCode.SESSION_LOST.value, message)
        self.command_id = command_id


@dataclass
class Snapshot:
    index: int
    payload: dict[str, Any]
    artifacts: dict[str, bytes]

    @property
    def reading(self) -> dict[str, Any]:
        return self.payload["reading"]

    def artifact(self, name: str) -> bytes:
        try:
            return self.artifacts[name]
        except KeyError:
            raise KeyError(f"snapshot {self.index} has no artifact {name!r}") from None


class HardwareClient:
    def __init__(self, config: StationConfig) -> None:
        self.config = config
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._heartbeat: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._pending: dict[str, Future] = {}
        self._pending_lock = threading.Lock()
        self._closed = threading.Event()
        self._lost = threading.Event()
        self.session: str | None = None
        self.clock_offset_ns: int | None = None
        self.hello: dict[str, Any] | None = None
        self.stop_result: dict[str, Any] | None = None
        self.close_error: str | None = None
        self._last_heartbeat = 0.0
        self._heartbeat_stop = threading.Event()

    # -- connection ---------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        if self._sock is not None:
            return self.hello or {}
        self._lost.clear()
        self._closed.clear()
        self.close_error = None
        try:
            sock = socket.create_connection(
                (self.config.service.host, self.config.service.port),
                timeout=self.config.service.connect_timeout_s,
            )
            sock.settimeout(None)
        except OSError as exc:
            raise ConnectionLost(f"cannot connect to hardware service: {exc}") from None
        self._sock = sock
        self._reader = threading.Thread(target=self._reader_loop, name="hw-rpc-reader", daemon=True)
        self._reader.start()
        try:
            hello = self.request(
                "hello",
                {"client_wall_ns": now_wall_ns()},
                timeout=self.config.service.connect_timeout_s,
            )
        except BaseException:
            self.close(force=True)
            raise
        if hello.get("protocol_version") != PROTOCOL_VERSION:
            self.close(force=True)
            raise RpcError(
                ErrorCode.VERSION_MISMATCH.value,
                f"service speaks {hello.get('protocol_version')!r}, client {PROTOCOL_VERSION}",
            )
        if hello.get("station_id") != self.config.station_id:
            self.close(force=True)
            raise RpcError(
                ErrorCode.CONFIG_MISMATCH.value,
                f"service station is {hello.get('station_id')!r}, "
                f"expected {self.config.station_id!r}",
            )
        if hello.get("config_sha256") != self.config.sha256:
            self.close(force=True)
            raise RpcError(ErrorCode.CONFIG_MISMATCH.value, "station config hashes differ")
        if hello.get("calibration_sha256") != self.config.calibration_sha256:
            self.close(force=True)
            raise RpcError(ErrorCode.CALIBRATION_MISMATCH.value, "calibration hashes differ")
        clock = hello.get("clock") or {}
        offset = clock.get("offset_server_minus_client_ns")
        if offset is None:
            self.close(force=True)
            raise RpcError(ErrorCode.CLOCK_SKEW.value, "service did not return a clock offset")
        self.clock_offset_ns = int(offset)
        max_skew = self.config.service.max_clock_skew_s
        # The handshake estimate includes one-way latency; allow it in the bound.
        if abs(self.clock_offset_ns) > int(max_skew * 1e9):
            self.close(force=True)
            raise RpcError(
                ErrorCode.CLOCK_SKEW.value,
                f"clock offset {self.clock_offset_ns / 1e9:.3f}s exceeds max_clock_skew_s; "
                "synchronise the hosts (NTP/PTP) before using hardware snapshots",
            )
        self.hello = hello
        self._heartbeat_stop.clear()
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="hw-rpc-heartbeat", daemon=True
        )
        self._heartbeat.start()
        return hello

    def open_session(self, client_id: str = "pyrualean") -> dict[str, Any]:
        self._require_connected()
        result = self.request(
            "open_session",
            {
                "client_id": client_id,
                "protocol_version": PROTOCOL_VERSION,
                "config_sha256": self.config.sha256,
                "calibration_sha256": self.config.calibration_sha256,
            },
            timeout=self.config.service.request_timeout_s,
        )
        self.session = str(result["session"])
        self._last_heartbeat = time.monotonic()
        return result

    # -- requests -----------------------------------------------------------

    def request(self, kind: str, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        self._require_connected()
        request_id = new_id("req")
        future: Future = Future()
        with self._pending_lock:
            self._pending[request_id] = future
        message = {
            "v": PROTOCOL_VERSION,
            "id": request_id,
            "type": kind,
            **payload,
        }
        if self.session and "session" not in message and kind not in ("hello", "open_session"):
            message["session"] = self.session
        try:
            with self._send_lock:
                assert self._sock is not None
                self._sock.sendall(encode_message(message))
        except OSError as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._mark_lost(f"send failed: {exc}")
            raise ConnectionLost(f"send failed: {exc}") from None
        wait = self.config.service.request_timeout_s if timeout is None else float(timeout)
        try:
            reply = future.result(timeout=wait)
        except FutureTimeout:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise RpcError(
                ErrorCode.TIMEOUT.value,
                f"service did not answer {kind!r} within {wait:.2f}s; "
                "no motion request is retried automatically",
            ) from None
        if not isinstance(reply, dict):
            raise RpcError(ErrorCode.INTERNAL.value, "service returned a non-object reply")
        if not reply.get("ok", False):
            err = reply.get("error") or {}
            raise RpcError(
                str(err.get("code", "internal")),
                str(err.get("message", "request failed")),
                detail=err.get("detail"),
            )
        return reply.get("result")

    def _reader_loop(self) -> None:
        assert self._sock is not None
        buffer = b""
        try:
            while not self._closed.is_set():
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > MAX_LINE_BYTES + 1:
                    self._mark_lost("service sent an oversized line")
                    break
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue
                    try:
                        message = decode_message(line)
                    except ValueError:
                        continue
                    request_id = str(message.get("id", ""))
                    with self._pending_lock:
                        future = self._pending.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_result(message)
        except OSError as exc:
            self._mark_lost(f"connection lost: {exc}")
        finally:
            if not self._closed.is_set():
                self._mark_lost("connection closed by service")

    def _mark_lost(self, message: str) -> None:
        self._lost.set()
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(ConnectionLost(message))

    def send_cancel(self) -> dict[str, Any]:
        self._require_connected()
        return self.request("cancel", {}, timeout=self.config.service.request_timeout_s)

    def emergency_stop(self) -> dict[str, Any]:
        self._require_connected()
        return self.request("emergency_stop", {}, timeout=self.config.service.request_timeout_s)

    def status(self) -> dict[str, Any]:
        self._require_connected()
        return self.request("status", {}, timeout=self.config.service.request_timeout_s)

    def snapshot(self, *, timeout_s: float | None = None) -> Snapshot:
        self._require_connected()
        payload = self.request(
            "snapshot",
            {},
            timeout=self.config.service.request_timeout_s if timeout_s is None else timeout_s,
        )
        artifacts: dict[str, bytes] = {}
        for name in payload.get("artifacts") or {}:
            item = self.request(
                "get_artifact",
                {"name": name},
                timeout=self.config.service.request_timeout_s,
            )
            try:
                data = base64.b64decode(item["data_b64"], validate=True)
            except (KeyError, ValueError) as exc:
                raise RpcError(ErrorCode.INTERNAL.value, f"bad artifact {name!r}: {exc}") from None
            expected = (payload.get("artifacts") or {}).get(name, {}).get("sha256")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise RpcError(ErrorCode.INTERNAL.value, f"artifact {name!r} failed its checksum")
            artifacts[name] = data
        return Snapshot(index=int(payload["index"]), payload=payload, artifacts=artifacts)

    def execute_command(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self._require_connected()
        if kind not in ("trajectory", "grip"):
            raise ValueError(f"unsupported command kind {kind!r}")
        command_id = new_id("cmd")
        digest = command_digest(kind, payload)
        self.request(
            "command",
            {"command_id": command_id, "kind": kind, "payload": payload, "digest": digest},
            timeout=self.config.service.request_timeout_s,
        )
        wait = (
            self.config.execution.command_timeout_s + 5.0
            if timeout_s is None
            else float(timeout_s) + 5.0
        )
        try:
            final = self.request(
                "wait_command", {"command_id": command_id, "timeout_s": wait}, timeout=wait + 2.0
            )
        except ConnectionLost as exc:
            # Do not resend.  A motion may have executed; the only safe next
            # step is an operator check.
            exc.command_id = command_id
            raise
        if final.get("state") not in ("succeeded", "cancelled", "timed_out", "stopped", "rejected"):
            raise RpcError(
                ErrorCode.INTERNAL.value,
                f"command {command_id} is still {final.get('state')!r} after its wait",
            )
        return final

    def close_session(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        if self.session is None:
            return {"state": "no_session", "stop": None}
        try:
            result = self.request(
                "close_session",
                {},
                timeout=self.config.service.request_timeout_s if timeout_s is None else timeout_s,
            )
            self.stop_result = result.get("stop")
            return result
        except (RpcError, ConnectionLost) as exc:
            self.close_error = f"{type(exc).__name__}: {exc}"
            self.stop_result = None
            raise
        finally:
            self.session = None

    def close(self, *, force: bool = False, stop: bool = True) -> dict[str, Any] | None:
        """Close the session and socket.  Record what the service actually confirmed."""

        result: dict[str, Any] | None = None
        if self.session is not None and not force and stop:
            try:
                result = self.close_session()
            except Exception:  # noqa: BLE001 - recorded in close_error
                result = None
        self._stop_heartbeat()
        self._closed.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self._mark_lost("client closed")
        self._reader = None
        self.session = None
        return result

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            interval = max(0.05, float(self.config.service.heartbeat_s))
            if self._heartbeat_stop.wait(interval):
                break
            if self.session is None or self._closed.is_set():
                continue
            try:
                self.request(
                    "heartbeat", {}, timeout=max(0.5, float(self.config.service.request_timeout_s))
                )
                self._last_heartbeat = time.monotonic()
            except (RpcError, ConnectionLost) as exc:
                if not self._heartbeat_stop.is_set():
                    self.close_error = f"heartbeat failed: {exc}"
                    self._mark_lost(f"heartbeat failed: {exc}")
                break

    def _require_connected(self) -> None:
        if self._sock is None:
            raise ConnectionLost("client is not connected")
        if self._lost.is_set():
            raise ConnectionLost(self.close_error or "connection lost")

    # -- clock-aware ages ---------------------------------------------------

    def server_wall_equivalent(self) -> int:
        return now_wall_ns() + int(self.clock_offset_ns or 0)

    def age_s(self, captured_wall_ns: int) -> float:
        return max(0.0, (self.server_wall_equivalent() - int(captured_wall_ns)) / 1e9)


__all__ = ["ConnectionLost", "HardwareClient", "RpcError", "Snapshot"]
