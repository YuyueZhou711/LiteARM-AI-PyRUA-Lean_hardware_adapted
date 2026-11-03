"""Command-line entry point: ``python -m litearm_ai_pyrua_lean doctor``."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from . import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]


def _check(path: Path, label: str) -> tuple[str, str]:
    if path.exists():
        return "ok", label
    return "missing", f"{label} ({path.relative_to(REPO_ROOT)})"


def _env_check(name: str) -> tuple[str, str]:
    if os.environ.get(name):
        return "ok", f"environment variable {name}"
    return "missing", f"environment variable {name}"


def doctor() -> int:
    print(f"repository {REPO_ROOT}")
    print(f"version    {__version__}")
    print()

    install_checks = [
        _check(REPO_ROOT / "pyrua-lean" / "src" / "pyrualean" / "__init__.py", "PyRUA-Lean source"),
        _check(REPO_ROOT / "robodojo" / "env", "RoboDojo source"),
        _check(
            REPO_ROOT / "robodojo" / "XPolicyLab" / "client_server" / "ws" / "model_client.py",
            "XPolicyLab checkout",
        ),
        _check(
            REPO_ROOT / "robodojo" / "XPolicyLab" / "policy" / "PyRUA_Lean" / "setup_eval_policy_server.sh",
            "PyRUA-Lean policy adapter",
        ),
        _check(REPO_ROOT / "assets" / "litearm" / "litearm.usd", "LiteArm source asset"),
        _check(
            REPO_ROOT / "robodojo" / "Assets" / "Robots" / "litearm" / "litearm.usd",
            "LiteArm installed asset",
        ),
        _check(
            REPO_ROOT / "robodojo" / "Assets" / "Robots" / "litearm" / "curobo.yml",
            "cuRobo robot config",
        ),
    ]

    runtime_checks = [
        _check(
            REPO_ROOT / "robodojo" / "XPolicyLab" / "policy" / "PyRUA_Lean" / "checkpoints" / "sam3.pt",
            "SAM3 checkpoint",
        ),
        _check(REPO_ROOT / "config" / "env.sh", "model endpoint config"),
        _env_check("OPENAI_BASE_URL"),
        _env_check("OPENAI_API_KEY"),
        _env_check("PYRUALEAN_MODEL"),
    ]

    for status, label in install_checks + runtime_checks:
        print(f"[{status:7}] {label}")

    if shutil.which("conda"):
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=False
        )
        envs = {line.split()[0] for line in result.stdout.splitlines() if line.strip()}
        for name in ("pyrualean", "RoboDojo"):
            print(f"[{'ok' if name in envs else 'missing':7}] conda environment {name}")

    failed = [label for status, label in install_checks if status == "missing"]
    pending = [label for status, label in runtime_checks if status == "missing"]
    print()
    if failed:
        print(f"{len(failed)} install checks failed; see docs/setup.md")
        return 1
    if pending:
        print(f"{len(pending)} runtime items are not configured yet; see docs/usage.md")
    else:
        print("all checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="litearm-ai-pyrua-lean")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", choices=("doctor",), help="check the local installation")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
