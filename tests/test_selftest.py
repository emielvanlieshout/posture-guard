from __future__ import annotations

import pytest

from posture_guard.cli import build_parser, main
from posture_guard.selftest import run_selftest, selftest_passed


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTURE_GUARD_HOME", str(tmp_path / "home"))


@pytest.mark.parametrize("view", ["side", "frontal"])
def test_selftest_passes_for_both_views(view):
    checks = run_selftest(view)
    failures = [str(c) for c in checks if not c.passed]
    assert selftest_passed(checks), "\n".join(failures)


def test_selftest_covers_the_pipeline():
    names = {c.name for c in run_selftest("side")}
    assert names >= {
        "calibration",
        "score anchors",
        "monotonic in protraction",
        "distance invariance",
        "alert fires after the dwell window",
        "alert clears on correction",
        "history and report",
    }


class TestCli:
    def test_selftest_exits_zero(self, capsys):
        assert main(["selftest", "--view", "side"]) == 0
        assert "all checks passed" in capsys.readouterr().out

    def test_run_without_calibration_says_so(self):
        with pytest.raises(SystemExit, match="calibrate"):
            main(["run"])

    def test_report_on_an_empty_history_works(self, capsys):
        assert main(["report", "--days", "3"]) == 0
        assert "posture-guard history" in capsys.readouterr().out

    def test_config_shows_and_sets(self, capsys):
        assert main(["config", "--set", "fps=8"]) == 0
        assert "fps                     8.0" in capsys.readouterr().out
        assert main(["config"]) == 0
        assert "8.0" in capsys.readouterr().out

    def test_config_rejects_a_bad_value(self):
        with pytest.raises(SystemExit, match="max_dim"):
            main(["config", "--set", "max_dim=0.99"])

    def test_pause_and_resume_round_trip(self, capsys):
        from posture_guard.app import _read_pause_flag

        assert main(["pause", "5"]) == 0
        assert _read_pause_flag() is not None
        assert main(["resume"]) == 0
        assert _read_pause_flag() is None

    def test_every_subcommand_is_wired_up(self):
        parser = build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        commands = set(actions[0].choices)
        assert commands == {
            "setup", "doctor", "calibrate", "run", "preview",
            "report", "pause", "resume", "selftest", "config",
        }
        for name, sub in actions[0].choices.items():
            assert sub.get_default("func") is not None, f"{name} has no handler"
