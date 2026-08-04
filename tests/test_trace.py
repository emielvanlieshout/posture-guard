"""Startup tracing: the evidence that separates "died" from "hid itself"."""

from __future__ import annotations

import io

from posture_guard.trace import Trace, looks_bundled


class TestTrace:
    def test_lines_are_timestamped(self):
        buf = io.StringIO()
        Trace(stream=buf).step("opening camera 0")
        line = buf.getvalue()
        assert "opening camera 0" in line
        assert line[0] == "[" and line[9] == "]", "HH:MM:SS in brackets"

    def test_every_line_is_flushed_immediately(self):
        """A crash on the next line must not swallow the line explaining it."""
        flushes = []

        class Counting(io.StringIO):
            def flush(self):
                flushes.append(len(self.getvalue()))

        trace = Trace(stream=Counting())
        trace.step("one")
        trace.step("two")
        assert len(flushes) == 2

    def test_detail_is_quiet_unless_debugging(self):
        buf = io.StringIO()
        trace = Trace(stream=buf)
        trace.detail("score 0.42")
        assert buf.getvalue() == ""

        buf = io.StringIO()
        trace = Trace(stream=buf, verbose=True)
        trace.detail("score 0.42")
        assert "score 0.42" in buf.getvalue()

    def test_it_can_be_switched_off_entirely(self):
        buf = io.StringIO()
        Trace(enabled=False, stream=buf).step("nothing")
        assert buf.getvalue() == ""

    def test_it_is_callable_so_it_can_be_passed_as_a_log(self):
        buf = io.StringIO()
        trace = Trace(stream=buf)
        trace("alerter disabled")
        assert "alerter disabled" in buf.getvalue()


class TestBundleDetection:
    def test_a_terminal_is_not_a_bundle(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("sys.stdout", io.StringIO())
        monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
        monkeypatch.setattr("os.getcwd", lambda: "/Users/someone/project")
        assert not looks_bundled()

    def test_no_terminal_means_bundled(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("sys.stdout", io.StringIO())
        monkeypatch.setattr("os.getcwd", lambda: "/Users/someone/project")
        assert looks_bundled(), "StringIO is not a tty"

    def test_launched_from_root_means_bundled(self, monkeypatch):
        """Launch Services starts an app with the working directory set to /."""
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("os.getcwd", lambda: "/")
        assert looks_bundled()

    def test_never_on_other_platforms(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr("os.getcwd", lambda: "/")
        assert not looks_bundled()


class TestTheStartupPathIsActuallyTraced:
    """These exist because a patch to cmd_run silently did not apply.

    The log went dark exactly where the missing lines would have been, and four
    rounds of guessing followed. Asserting the wiring is present is cheap; not
    being able to see the startup is not.
    """

    def source(self):
        from pathlib import Path

        from posture_guard import cli

        text = Path(cli.__file__).read_text()
        start = text.index("def cmd_run(")
        return text[start : text.index("\ndef ", start + 1)]

    def test_the_runner_is_given_the_trace(self):
        source = self.source()
        assert "trace=trace" in source, "the Runner would report nothing at all"
        assert "log=trace.step" in source, "its messages would go to a buffered stdout"

    def test_every_stage_of_startup_announces_itself(self):
        source = self.source()
        for step in (
            "starting  pid=",
            "config:",
            "calibration:",
            "model:",
            "camera permission:",
            "alerters ready:",
            "starting the menu bar",
        ):
            assert step in source, f"nothing logs {step!r}, so a hang there is invisible"

    def test_the_last_line_before_the_menu_bar_is_the_menu_bar(self):
        """Whatever hangs after that line, the log names the stage it hung in."""
        source = self.source()
        assert source.index("alerters ready:") < source.index("starting the menu bar")
        assert source.index("starting the menu bar") < source.index("run_tray(runner")

    def test_the_bundle_announces_itself(self):
        assert "announce_running()" in self.source()


class TestHelpersCannotHangTheApp:
    def test_the_notification_has_a_timeout(self):
        """osascript can sit on an Automation prompt for ever."""
        from pathlib import Path

        from posture_guard import trace as module

        assert "timeout=" in Path(module.__file__).read_text()

    def test_the_failure_alert_has_a_timeout(self):
        from pathlib import Path

        from posture_guard import cli

        source = Path(cli.__file__).read_text()
        surface = source[source.index("def _surface(") :]
        assert "timeout=" in surface[: surface.index("\ndef ")]
