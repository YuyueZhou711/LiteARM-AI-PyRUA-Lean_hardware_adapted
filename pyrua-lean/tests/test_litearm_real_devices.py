"""Vendor-SDK wrapper tests with litearm.testing.FakeTransport and a mock gripper.

This file runs only in the device environment.  The fake transport is not
physical evidence: it verifies the wrapper's mapping, freshness checks, stop
reporting and repeated-grasp bookkeeping, not real dynamics.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

litearm = pytest.importorskip("litearm")
from litearm import _protocol as P  # noqa: E402
from litearm.testing import FakeTransport, _status  # noqa: E402

from pyrualean.hardware.config import load_station  # noqa: E402
from pyrualean.hardware.devices import (  # noqa: E402
    DeviceCancelled,
    DeviceError,
    _ArmHandle,
    _GripperHandle,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "litearm_real.example.json"


@pytest.fixture(scope="module")
def station():
    return load_station(CONFIG)


class _EnabledTransport(FakeTransport):
    """FakeTransport with the firmware's enabled status bit set.

    The stock stub leaves flags=0, so the SDK's ``RobotState.enabled`` (bit 9)
    never becomes true.  This subclass only fills that bit; motion, stop and
    state frames still come from the verified FakeTransport.
    """

    def _status_frame(self, q, dq=None, mode=1):
        return _status(
            q,
            dq=self.dq if dq is None else dq,
            mode=mode,
            flags=1 << P.FLAG_ENABLED_BIT,
            n=self.n,
            tau=self.tau,
        )


def test_arm_wrapper_maps_measured_state_and_never_hides_a_fault(station):
    transport = _EnabledTransport(
        port="fake",
        fw="Litearm2.1.0-7J",
        n=7,
    )
    transport.q = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    handle = _ArmHandle(
        station.arms["left"].arm,
        station.calibration.arm("left"),
        device_name="test_arm",
        transport_factory=lambda _port: transport,
    )
    handle.open()
    try:
        state = handle.read()
        assert len(state["q_sdk"]) == 7
        assert state["q_model"] == pytest.approx(transport.q)
        assert state["fault"] is False
        # The wrapper does not enable automatically.
        assert "not enabled" in " ".join(handle.verify_enabled())
        handle._sdk.enable()
        assert handle.verify_enabled() == []
        same = handle.move_sync(list(transport.q), 0.2, cancel=threading.Event(), timeout_s=5.0)
        assert same.reached is True
        ok, method, why = handle.emergency_stop()
        assert method == "emergency_stop"
        assert ok is True and why is None
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(DeviceCancelled):
            handle.move_sync(list(transport.q), 0.2, cancel=cancel, timeout_s=5.0)
    finally:
        handle.close()


class _State:
    def __init__(self, position_mm: float, enabled: bool = True) -> None:
        self.position_mm = position_mm
        self.force_n = 12.0
        self.is_enabled = enabled
        self.error_code = 1 if enabled else 0
        self.fresh = True
        self.data_age_s = 0.01
        self.is_error = False


class _Move:
    ok = True
    reached = True
    stalled = False
    no_bite = False


class _FakeLiteGrip:
    grasp_calls = 0
    release_calls = 0

    def __init__(self, channel, can_id, **kwargs):
        self.channel = channel
        self.can_id = can_id
        self.width = 86.96
        self.enabled = True
        self._release = threading.Event()

    def connect(self):
        return True

    def load_calibration(self, path=None):
        return True

    def get_state(self, wait=False):
        return _State(self.width, self.enabled)

    def move_at_speed(self, target_mm, speed_mm_s=30.0, **kwargs):
        self.width = float(target_mm)
        return _Move()

    def grasp(self, force_n=None, hold_s=0.0, **kwargs):
        type(self).grasp_calls += 1
        self.width = 30.0
        self._release.wait(5.0)
        self.width = 86.96
        result = _Move()
        result.released = True
        return result

    def release(self, clearance_mm=10.0, abort=False, timeout_s=None, **kwargs):
        type(self).release_calls += 1
        self.width = 86.96
        self._release.set()
        return _Move()

    def request_release(self, clearance_mm=10.0, abort=False):
        if not self._release.is_set():
            self._release.set()
            return True
        return False

    def stop(self):
        return None

    def disconnect(self):
        self._release.set()


def test_gripper_wrapper_keeps_a_repeated_close_and_uses_measured_width(station, monkeypatch):
    module = types.ModuleType("litegrip")
    module.LiteGrip = _FakeLiteGrip
    monkeypatch.setitem(sys.modules, "litegrip", module)
    _FakeLiteGrip.grasp_calls = 0
    _FakeLiteGrip.release_calls = 0
    handle = _GripperHandle(
        station.arms["left"].gripper,
        station.calibration.arm("left"),
        device_name="test_gripper",
    )
    handle.open()
    try:
        cancel = threading.Event()
        first = handle.command_close(timeout_s=2.0, cancel=cancel)
        assert first.holding is True
        assert first.width_m == pytest.approx(0.030)
        second = handle.command_close(timeout_s=2.0, cancel=cancel)
        assert second.holding is True
        assert _FakeLiteGrip.grasp_calls == 1  # no restart
        opened = handle.command_open(width_m=None, timeout_s=2.0, cancel=cancel)
        assert opened.released is True
        assert opened.width_m == pytest.approx(0.08696)
        assert _FakeLiteGrip.release_calls == 1
        partial = handle.command_open(width_m=0.040, timeout_s=2.0, cancel=cancel)
        assert partial.width_m == pytest.approx(0.040)
    finally:
        handle.close()


class _UncalibratedLiteGrip(_FakeLiteGrip):
    def load_calibration(self, path=None):
        return False


def test_gripper_without_a_usable_calibration_is_refused(station, monkeypatch):
    module = types.ModuleType("litegrip")
    module.LiteGrip = _UncalibratedLiteGrip
    monkeypatch.setitem(sys.modules, "litegrip", module)
    handle = _GripperHandle(
        station.arms["left"].gripper,
        station.calibration.arm("left"),
        device_name="test_gripper",
    )
    with pytest.raises(DeviceError, match="no usable LiteGrip calibration"):
        handle.open()
