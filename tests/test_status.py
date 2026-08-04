"""Running or not, and capturing or not.

Those are different questions with unrelated fixes, and four rounds of "nothing
happens" went by with neither of them answerable.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from posture_guard.status import Instance, Status, collect, find_instances, report


def make(running=(), sample_age=None, log_line="", now=1_000_000.0):
    return Status(
        instances=[Instance(pid=p, command=c) for p, c in running],
        last_log_line=log_line,
        last_sample_ts=None if sample_age is None else now - sample_age,
        now=now,
    )


class TestInstances:
    def test_a_bundle_launch_is_recognised(self):
        assert Instance(1, "/usr/bin/arch -arm64 /x/.venv/bin/python -m posture_guard run").from_bundle
        assert Instance(1, "/Users/x/Applications/P.app/Contents/MacOS/PostureGuard").from_bundle

    def test_a_terminal_launch_is_not(self):
        assert not Instance(1, "/x/.venv/bin/python -m posture_guard run --verbose").from_bundle

    def test_this_process_is_not_counted_as_a_monitor(self):
        """Otherwise `status` would always report itself as running."""
        assert all(i.pid != __import__("os").getpid() for i in find_instances())

    def test_it_survives_without_ps(self, monkeypatch):
        monkeypatch.setattr("subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
        assert find_instances() == []


class TestVerdict:
    def test_nothing_running_says_how_to_start(self):
        lines = report(make())
        assert lines[0] == "not running"
        assert "run --verbose" in lines[-1]

    def test_running_and_capturing_points_at_the_notch(self):
        """The likeliest reason a working menu bar app looks broken."""
        lines = report(make(running=[(42, "python -m posture_guard run")], sample_age=20))
        assert "running, pid 42" in lines[0]
        assert "notch" in lines[-1]

    def test_running_without_data_points_at_the_log(self):
        lines = report(make(running=[(42, "python -m posture_guard run")], log_line="[09:00] opening camera 0"))
        assert "not recording" in lines[-1]
        assert "opening camera 0" in "\n".join(lines)

    def test_stale_data_does_not_count_as_capturing(self):
        assert not make(running=[(1, "x")], sample_age=3600).capturing

    def test_two_copies_are_called_out(self):
        lines = report(make(running=[(1, "a"), (2, "b")], sample_age=10))
        assert "2 copies" in "\n".join(lines)
        assert "fight" in "\n".join(lines)

    def test_it_says_where_it_was_started_from(self):
        bundle = report(make(running=[(1, "/x/P.app/Contents/MacOS/PostureGuard")], sample_age=5))
        terminal = report(make(running=[(1, "python -m posture_guard run")], sample_age=5))
        assert "the app bundle" in bundle[0]
        assert "a terminal" in terminal[0]


class TestCollect:
    def test_it_reads_the_freshest_measurement(self, tmp_path):
        db = tmp_path / "posture.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE buckets (bucket_ts INTEGER PRIMARY KEY)")
            conn.executemany("INSERT INTO buckets VALUES (?)", [(100,), (900,), (500,)])
        status = collect(db, tmp_path / "missing.log", now=1000.0)
        assert status.last_sample_ts == 900

    def test_a_missing_database_is_not_an_error(self, tmp_path):
        status = collect(tmp_path / "nope.db", tmp_path / "nope.log", now=1000.0)
        assert status.last_sample_ts is None
        assert status.last_log_line == ""

    def test_a_corrupt_database_is_not_an_error(self, tmp_path):
        db = tmp_path / "posture.db"
        db.write_bytes(b"this is not a database")
        assert collect(db, tmp_path / "nope.log", now=1000.0).last_sample_ts is None

    def test_the_last_meaningful_log_line_is_taken(self, tmp_path):
        log = tmp_path / "p.log"
        log.write_text("[09:00] first\n[09:01] opening camera 0\n\n   \n")
        assert collect(tmp_path / "n.db", log, now=1000.0).last_log_line == "[09:01] opening camera 0"

    def test_reading_the_database_does_not_disturb_a_running_monitor(self, tmp_path):
        """Opened read-only, so status can never block or corrupt a live writer."""
        import inspect

        from posture_guard import status as module

        assert "mode=ro" in inspect.getsource(module.collect)
