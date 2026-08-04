"""Startup tracing.

Started from Finder there is no terminal and, until now, nothing of ours in the
log either -- only mediapipe's chatter. An app that died during startup and an
app that started and hid itself in the menu bar produced identical evidence.

So every startup step announces itself, timestamped and flushed immediately, and
the last line in the log is where it got to. The steps are few and only run once,
so this is always on rather than something to remember to switch on when
something has already gone wrong.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime


class Trace:
    """Timestamped step logging, flushed line by line."""

    def __init__(self, enabled: bool = True, stream=None, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self.stream = stream or sys.stdout

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.stream.write(f"[{stamp}] {message}\n")
        # Unbuffered on purpose: a crash one line later must not take the line
        # that would have explained it.
        self.stream.flush()

    def detail(self, message: str) -> None:
        """Only in --debug. Anything that repeats belongs here."""
        if self.verbose:
            self.step(message)

    def __call__(self, message: str) -> None:
        self.step(message)


def looks_bundled() -> bool:
    """True when this is the .app rather than a shell.

    Two independent signals, because either alone is wrong somewhere: a bundle
    has no terminal attached, and Launch Services starts an app from /.
    """
    if sys.platform != "darwin":
        return False
    no_terminal = not sys.stdout.isatty()
    launched_by_finder = os.getcwd() == "/"
    return no_terminal or launched_by_finder


def announce_running() -> None:
    """A notification saying it started, for when there is no terminal to say so.

    The menu bar item is a single character wide. On a laptop with a notch and a
    dozen other menu bar items it can be genuinely invisible, at which point a
    working app and a dead one look the same.
    """
    if sys.platform != "darwin":
        return
    import shutil
    import subprocess

    osascript = shutil.which("osascript")
    if not osascript:
        return
    script = (
        'display notification "Watching your posture. Look for the bar in your menu bar." '
        'with title "posture-guard is running"'
    )
    subprocess.run(  # noqa: S603 - fixed binary, literal script
        [osascript, "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
