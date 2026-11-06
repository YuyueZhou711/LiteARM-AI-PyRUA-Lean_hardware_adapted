"""Offline CLI commands."""

from __future__ import annotations

import json

from pyrualean.cli import main


def test_api_and_prompt_render_offline(capsys):
    assert main(["api"]) == 0
    assert "robo.move_to(" in capsys.readouterr().out
    assert main(["prompt", "--system-only"]) == 0
    out = capsys.readouterr().out
    assert "# Operating knowledge" in out and "def run(robo)" in out


def test_summarize_reads_result_dirs(tmp_path, capsys):
    directory = tmp_path / "ep"
    directory.mkdir()
    (directory / "result.json").write_text(
        json.dumps(
            {
                "schema": "pyrualean-episode-v2",
                "suite": "libero_10",
                "task": 0,
                "seed": 1,
                "environment_success": False,
                "status": "error",
                "calls": 3,
                "env_steps": 40,
                "policy_wall_s": 12.5,
                "error": {"type": "ValueError", "message": "bad"},
            }
        )
    )
    assert main(["summarize", str(directory), "--json"]) == 0
    out = capsys.readouterr().out
    assert "libero_10:0" in out and '"native_sr": 0.0' in out


def test_provider_overrides_shape():
    from pyrualean.generate import provider_overrides

    assert provider_overrides(None, None) == []
    overrides = provider_overrides("https://gw.example.com", "MY_KEY")
    joined = " ".join(overrides)
    assert 'model_provider="pyrualean"' in joined
    assert 'model_providers.pyrualean.base_url="https://gw.example.com/v1"' in joined
    assert 'model_providers.pyrualean.wire_api="responses"' in joined
    assert 'model_providers.pyrualean.env_key="MY_KEY"' in joined


def _fake_play_host(monkeypatch, called):
    from pathlib import Path

    import pyrualean.cli as cli

    def run_host(argv, *, timeout_s, log_path=None, **kwargs):
        called["argv"] = list(argv)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "environment_success": False,
                    "environment_success_known": False,
                    "task_success": None,
                    "tool_turns": 0,
                    "env_steps": 0,
                    "policy_wall_s": 0.1,
                }
            )
        )
        return 0

    monkeypatch.setattr(cli, "run_host", run_host)


def test_play_default_arguments_are_unchanged(tmp_path, monkeypatch):
    called = {}
    _fake_play_host(monkeypatch, called)
    assert (
        main(
            ["play", "--robot", "litearm", "--arm", "cells", "--task", "0", "--out", str(tmp_path)]
        )
        == 0
    )
    argv = called["argv"]
    assert argv[argv.index("--robot") + 1] == "litearm"
    assert argv[argv.index("--arm") + 1] == "cells"
    assert "--hardware-config" not in argv
    assert "--instruction" not in argv
    assert "--execution-mode" not in argv
    # The old default (full) is still the implicit primitive set.
    assert "--primitives" not in argv


def test_play_forwards_hardware_arguments(tmp_path, monkeypatch):
    called = {}
    _fake_play_host(monkeypatch, called)
    config = tmp_path / "station.json"
    config.write_text("{}")
    code = main(
        [
            "play",
            "--robot",
            "litearm_real",
            "--arm",
            "cells",
            "--primitives",
            "no-vla",
            "--task",
            "pick_place",
            "--instruction",
            "put the red block in the tray",
            "--hardware-config",
            str(config),
            "--execution-mode",
            "offline-fake",
            "--budget-s",
            "42",
            "--max-episode-steps",
            "17",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == 0
    argv = called["argv"]
    pairs = {token: argv[i + 1] for i, token in enumerate(argv[:-1]) if token.startswith("--")}
    assert pairs["--robot"] == "litearm_real"
    assert pairs["--primitives"] == "no-vla"
    assert pairs["--hardware-config"] == str(config)
    assert pairs["--instruction"] == "put the red block in the tray"
    assert pairs["--execution-mode"] == "offline-fake"
    assert pairs["--task"] == "pick_place"
    assert pairs["--budget-s"] == "42.0"
    assert pairs["--max-episode-steps"] == "17"
