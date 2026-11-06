#!/usr/bin/env python3
"""Station calibration helper; see docs/robots/litearm-real.md.

Run this in the device or agent environment with PyRUA-Lean installed.  This
version validates and stamps human measurements; it does not move an arm.
"""

from pyrualean.hardware.calibrate import main

if __name__ == "__main__":
    raise SystemExit(main())
