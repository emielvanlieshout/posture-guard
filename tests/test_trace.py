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
