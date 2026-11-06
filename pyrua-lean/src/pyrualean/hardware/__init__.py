# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Hardware path for the real LiteArm station.

This package is intentionally light: importing it does not import a vendor
SDK, open a device/socket, load a GPU model or start a thread.  The robot
adapter (:mod:`pyrualean.litearm_real`) and the service entry point import
their heavy dependencies inside functions.
"""

from .config import Calibration, ConfigError, StationConfig, load_calibration, load_station
from .protocol import PROTOCOL_VERSION, ExecutionMode

__all__ = [
    "Calibration",
    "ConfigError",
    "ExecutionMode",
    "PROTOCOL_VERSION",
    "StationConfig",
    "load_calibration",
    "load_station",
]
