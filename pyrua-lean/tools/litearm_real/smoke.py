#!/usr/bin/env python3
"""Hardware smoke test; see docs/robots/litearm-real.md.

Read-only by default.  Motion requires --allow-motion (live) or an explicit
offline-fake service; this repository did not run the live-motion path.
"""

from pyrualean.hardware.smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
