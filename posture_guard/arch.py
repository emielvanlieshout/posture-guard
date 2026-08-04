"""Which architecture this process is actually running as.

On Apple Silicon a universal Python can be launched either natively or under
Rosetta, and which one you get depends on the process that started it. Packages
with compiled extensions are built for one architecture only, so a venv that
works from one shell can fail from another with nothing but a wall of text about
numpy's C extensions -- text that never mentions Rosetta.

The app bundle makes this worse: launchd may start it translated even when the
terminal runs it natively, at which case the first import dies and the app is
gone before it draws anything. So `install-app` records the architecture that
was working and pins the launcher to it.
"""

from __future__ import annotations

import platform
import subprocess
import sys

ROSETTA_HINT = "incompatible architecture"


def running_arch() -> str:
    """The architecture of *this* process: arm64, or x86_64 under Rosetta."""
    return platform.machine()


def under_rosetta() -> bool:
    """True when an Apple Silicon Mac is running this process translated."""
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(  # noqa: S603, S607 - fixed system binary
            ["/usr/sbin/sysctl", "-n", "sysctl.proc_translated"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == "1"


def describe() -> str:
    """One line for `posture-guard doctor`."""
    arch = running_arch()
    if under_rosetta():
        return f"{arch} (translated by Rosetta — packages built for arm64 will not load)"
    return arch


def explain_import_error(exc: BaseException) -> str | None:
    """Turn an architecture mismatch into a sentence about the actual problem.

    numpy's own message is a page long, mentions Rosetta nowhere, and suggests
    the install is corrupt. It is not: the packages are fine, the interpreter is
    running as the wrong architecture.
    """
    text = str(exc)
    if ROSETTA_HINT not in text:
        return None

    arch = running_arch()
    wanted = "arm64" if arch == "x86_64" else "x86_64"
    return (
        f"This Python is running as {arch}, but its packages are built for {wanted}.\n"
        "Nothing is corrupt — the interpreter was started under the wrong architecture, "
        "which on Apple Silicon usually means Rosetta.\n\n"
        "  arch                       # what your shell is running as\n"
        f"  arch -{wanted} python3 -m posture_guard doctor    # force the right one\n\n"
        "If your shell is translated, the virtualenv was probably created by a "
        f"translated Python. Recreating it under {wanted} fixes it for good:\n"
        f"  arch -{wanted} python3 -m venv --clear .venv\n"
        "  source .venv/bin/activate && pip install -e '.[macos]'"
    )
