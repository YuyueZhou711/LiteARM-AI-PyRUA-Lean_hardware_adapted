from litearm_ai_pyrua_lean.__main__ import main


def test_doctor_runs(capsys) -> None:
    result = main(["doctor"])
    assert result in (0, 1)
    assert "repository" in capsys.readouterr().out
