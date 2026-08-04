"""macOS notification plus a short sound, rate limited.

Notifications are fired only when an alert starts, never while it lasts, and a
cooldown keeps a bad afternoon from turning into fifty banners. The dim overlay
is the continuous channel; this one is punctuation.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time

from ..state import EventKind, Tick

DEFAULT_SOUND = "/System/Library/Sounds/Tink.aiff"
_MESSAGES = (
    "Shoulders forward. Roll them back and down.",
    "Chest open, shoulder blades toward each other.",
    "Sit back into the chair, shoulders down.",
)


class NotificationAlerter:
    name = "notify"

    def __init__(self, cooldown_s: float = 300.0, sound: str | None = DEFAULT_SOUND):
        self.cooldown_s = max(0.0, cooldown_s)
        self.sound = sound
        self._last_fire = 0.0
        self._count = 0
        self._osascript = None
        self._afplay = None

    def start(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("the notification alerter needs macOS")
        self._osascript = shutil.which("osascript")
        self._afplay = shutil.which("afplay")
        if not self._osascript:
            raise RuntimeError("osascript not found")

    def stop(self) -> None:
        pass

    def apply(self, tick: Tick) -> None:
        if not any(e.kind is EventKind.ALERT_STARTED for e in tick.events):
            return
        now = time.monotonic()
        if now - self._last_fire < self.cooldown_s:
            return
        self._last_fire = now
        # Rotate the wording; an identical banner stops registering after a week.
        message = _MESSAGES[self._count % len(_MESSAGES)]
        self._count += 1
        self._fire(message)

    def _fire(self, message: str) -> None:
        script = (
            f'display notification "{_escape(message)}" '
            f'with title "posture-guard"'
        )
        subprocess.Popen(  # noqa: S603 - fixed binary, escaped literal
            [self._osascript, "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self.sound and self._afplay:
            subprocess.Popen(  # noqa: S603
                [self._afplay, self.sound],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')
