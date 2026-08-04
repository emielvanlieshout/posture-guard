"""Is it running, and is it actually seeing anything.

Four rounds of "nothing happens" went by without that question being answerable.
A menu bar app has no window, its item is one character wide and can sit behind
a notch, and its log lives somewhere nobody opens -- so "started and invisible"
and "died on launch" look identical from the outside, and the fixes for them
have nothing in common.

This separates them: whether a process exists, whether it has written anything,
and whether the database has grown in the last minute, which is the only proof
that the camera is actually being read.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "posture_guard"


@dataclass
class Instance:
    pid: int
    command: str

    @property
    def from_bundle(self) -> bool:
        return ".app/Contents/MacOS" in self.command or "/usr/bin/arch" in self.command


@dataclass
class Status:
    instances: list[Instance] = field(default_factory=list)
    last_log_line: str = ""
    last_sample_ts: float | None = None
    now: float = 0.0

    @property
    def running(self) -> bool:
        return bool(self.instances)

    @property
    def capturing(self) -> bool:
        """Fresh data in the database is the only proof the camera is being read."""
        if self.last_sample_ts is None:
            return False
        return (self.now - self.last_sample_ts) < 180


def find_instances(exclude_self: bool = True) -> list[Instance]:
    """Running monitors, found without a third-party dependency."""
    try:
        result = subprocess.run(  # noqa: S603, S607 - fixed system binary
            ["/bin/ps", "-Ao", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    mine = os.getpid()
    found = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        if MARKER not in command or " run" not in command:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if exclude_self and pid == mine:
            continue
        found.append(Instance(pid=pid, command=command.strip()))
    return found


def collect(database: Path, log: Path, now: float | None = None) -> Status:
    now = time.time() if now is None else now
    status = Status(instances=find_instances(), now=now)

    if log.exists():
        lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
        status.last_log_line = lines[-1] if lines else ""

    if database.exists():
        import sqlite3

        try:
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
                row = conn.execute("SELECT MAX(bucket_ts) FROM buckets").fetchone()
            if row and row[0]:
                status.last_sample_ts = float(row[0])
        except sqlite3.Error:
            pass
    return status


def report(status: Status) -> list[str]:
    """Lines for the terminal, ending in what to do about it."""
    lines = []

    if not status.running:
        lines.append("not running")
    else:
        for instance in status.instances:
            where = "the app bundle" if instance.from_bundle else "a terminal"
            lines.append(f"running, pid {instance.pid}, started from {where}")
        if len(status.instances) > 1:
            lines.append(f"{len(status.instances)} copies are running; expect them to fight")

    if status.last_sample_ts is None:
        lines.append("no posture data has ever been recorded")
    else:
        age = status.now - status.last_sample_ts
        when = f"{age / 60:.0f} min ago" if age > 90 else f"{age:.0f}s ago"
        lines.append(f"last measurement {when}")

    if status.last_log_line:
        lines.append(f"log ends with: {status.last_log_line}")

    lines.append("")
    if status.running and status.capturing:
        lines.append(
            "It is running and watching you. If you cannot see it, the menu bar item is "
            "one character wide and may be hidden behind the notch — try `posture-guard "
            "run --verbose` in a terminal instead."
        )
    elif status.running:
        lines.append(
            "It is running but not recording. The log line above is how far it got; "
            "`posture-guard app-log` has the rest."
        )
    else:
        lines.append("Start it with `posture-guard run --verbose`, or open the app.")
    return lines
