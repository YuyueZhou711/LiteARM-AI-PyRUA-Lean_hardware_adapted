# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Wire protocol shared by the hardware runner and the hardware service.

The link is newline-delimited JSON over TCP.  One connection carries
independent requests (each with its own ``id``), so a ``cancel`` can be
answered while a long trajectory is still running.  The protocol is
deliberately small and explicit:

* a hello/hello reply negotiates the protocol version, the station identity
  and the configuration/calibration hashes;
* a session is exclusive (one runner at a time);
* commands carry a caller-generated ``command_id`` and a content digest, so a
  retried request can be de-duplicated instead of moving the robot twice;
* a snapshot names its artifacts (PNG / NumPy) instead of embedding them in
  JSON; ``get_artifact`` fetches bytes separately;
* all timestamps are wall-clock nanoseconds plus the daemon's monotonic
  nanoseconds.  Wall clocks of different machines are not compared without a
  handshake offset and a configured skew bound.

Nothing in this module imports a vendor SDK, opens a socket or starts a
thread.  It is safe to import in the offline agent environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Bumped when a field changes meaning; the daemon and the client must agree.
PROTOCOL_VERSION = "1.0"

#: A single wire line may not exceed this size.  Artifacts are transferred by
#: their own request, so ordinary replies stay far below it.
MAX_LINE_BYTES = 32 * 1024 * 1024

#: Request kinds.
REQUEST_TYPES = (
    "hello",
    "open_session",
    "heartbeat",
    "snapshot",
    "get_artifact",
    "command",
    "command_status",
    "wait_command",
    "cancel",
    "emergency_stop",
    "disconnect",
    "close_session",
    "status",
)

#: Command kinds a runner may submit.  ``move_js`` continuous servoing is
#: deliberately absent: the first hardware version uses synchronous ``movej``
#: segments only.
COMMAND_KINDS = ("trajectory", "grip")


class ErrorCode(str, Enum):
    """Machine-readable refusal/failure reasons (the message is for people)."""

    BAD_REQUEST = "bad_request"
    UNSUPPORTED = "unsupported"
    VERSION_MISMATCH = "version_mismatch"
    CONFIG_MISMATCH = "config_mismatch"
    CALIBRATION_MISMATCH = "calibration_mismatch"
    BUSY = "busy"
    NO_SESSION = "no_session"
    SESSION_MISMATCH = "session_mismatch"
    SESSION_LOST = "session_lost"
    READ_ONLY = "read_only"
    NOT_READY = "not_ready"
    FAULTED = "faulted"
    STOPPED = "stopped"
    COMMAND_CONFLICT = "command_conflict"
    STALE_STATE = "stale_state"
    CLOCK_SKEW = "clock_skew"
    LIMIT_VIOLATION = "limit_violation"
    COLLISION = "collision"
    UNREACHABLE = "unreachable"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    DEVICE_FAULT = "device_fault"
    STOP_UNCONFIRMED = "stop_unconfirmed"
    INTERNAL = "internal"


class SessionState(str, Enum):
    READ_ONLY = "read_only"
    READY = "ready"
    EXECUTING = "executing"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULT = "fault"


class CommandState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAULTED = "faulted"
    STOPPED = "stopped"
    REJECTED = "rejected"


class StopKind(str, Enum):
    """The three stop semantics are intentionally distinct."""

    #: Stop after the current synchronous segment.  The arm stays enabled and
    #: holds its measured pose; the gripper keeps its command.  The session
    #: enters ``STOPPED``; a new trajectory needs an operator restart.
    CANCEL = "cancel"
    #: The command missed its deadline or the runner's heartbeat was lost.
    #: The device service calls the firmware emergency stop and the harness
    #: reports FAULT.  It does not call ``disable`` or ``park``.
    TIMEOUT = "timeout"
    #: An explicit emergency stop.  Same device action as ``TIMEOUT``; the
    #: difference is the reason recorded in the stop evidence.
    EMERGENCY = "emergency"


class ExecutionMode(str, Enum):
    READ_ONLY = "read-only"
    OFFLINE_FAKE = "offline-fake"
    LIVE = "live"


@dataclass(frozen=True)
class StopResult:
    """What actually happened when the service was asked to stop.

    ``confirmed`` is False unless the service read back a state that supports
    it.  A returned stop command, a Python flag, ``park()`` or ``disable()`` is
    not evidence of a controlled stop.
    """

    requested: bool
    kind: str
    confirmed: bool
    state: str
    method: str | None
    message: str | None
    requested_at_wall_ns: int | None = None
    confirmed_at_wall_ns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "kind": self.kind,
            "confirmed": self.confirmed,
            "state": self.state,
            "method": self.method,
            "message": self.message,
            "requested_at_wall_ns": self.requested_at_wall_ns,
            "confirmed_at_wall_ns": self.confirmed_at_wall_ns,
        }


@dataclass(frozen=True)
class CommandResult:
    """Terminal state of one command, as the service recorded it."""

    command_id: str
    kind: str
    state: str
    success: bool
    started_at_wall_ns: int | None = None
    finished_at_wall_ns: int | None = None
    elapsed_s: float | None = None
    steps_used: int = 0
    result: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    stop: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "kind": self.kind,
            "state": self.state,
            "success": self.success,
            "started_at_wall_ns": self.started_at_wall_ns,
            "finished_at_wall_ns": self.finished_at_wall_ns,
            "elapsed_s": self.elapsed_s,
            "steps_used": self.steps_used,
            "result": self.result,
            "error": self.error,
            "stop": self.stop,
        }


@dataclass(frozen=True)
class Clock:
    """The clocks on which a snapshot's age is judged."""

    wall_ns: int
    monotonic_ns: int
    clock_source: str = "daemon_wall_ns"

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_ns": self.wall_ns,
            "monotonic_ns": self.monotonic_ns,
            "clock_source": self.clock_source,
        }


def now_wall_ns() -> int:
    return time.time_ns()


def now_monotonic_ns() -> int:
    return time.monotonic_ns()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def canonical_json(value: Any) -> str:
    """Stable JSON for hashing (sorted keys, compact separators, UTF-8)."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def command_digest(kind: str, payload: Mapping[str, Any]) -> str:
    if kind not in COMMAND_KINDS:
        raise ValueError(f"unknown command kind {kind!r}; known: {COMMAND_KINDS}")
    return content_hash({"kind": kind, "payload": dict(payload)})


def encode_message(message: Mapping[str, Any]) -> bytes:
    data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_LINE_BYTES:
        raise ValueError(f"message is {len(data)} bytes, over the {MAX_LINE_BYTES} byte limit")
    return data + b"\n"


def decode_message(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_LINE_BYTES:
        raise ValueError("message exceeds the wire size limit")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON message: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError("a protocol message must be a JSON object")
    return value


def error(code: ErrorCode | str, message: str, *, detail: Any = None) -> dict[str, Any]:
    return {
        "code": str(getattr(code, "value", code)),
        "message": str(message),
        "detail": detail,
    }


def response(request_id: str, *, result: Any = None, error_obj: Mapping[str, Any] | None = None):
    out: dict[str, Any] = {"v": PROTOCOL_VERSION, "id": request_id, "ok": error_obj is None}
    if error_obj is not None:
        out["error"] = dict(error_obj)
    else:
        out["result"] = result
    return out


def require_finite(value: Any, name: str) -> float:
    """A finite float; booleans are not numbers here (JSON ``true`` is not 1.0)."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object, got {type(value).__name__}")
    return value


def require_list(value: Any, name: str, *, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} must have {length} items, got {len(value)}")
    return value


def require_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


__all__ = [
    "COMMAND_KINDS",
    "Clock",
    "CommandResult",
    "CommandState",
    "ErrorCode",
    "ExecutionMode",
    "MAX_LINE_BYTES",
    "PROTOCOL_VERSION",
    "REQUEST_TYPES",
    "SessionState",
    "StopKind",
    "StopResult",
    "canonical_json",
    "command_digest",
    "content_hash",
    "decode_message",
    "encode_message",
    "error",
    "new_id",
    "now_monotonic_ns",
    "now_wall_ns",
    "require_dict",
    "require_finite",
    "require_int",
    "require_list",
    "require_str",
    "response",
]
