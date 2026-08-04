"""Prints state changes to stdout. The fallback that works everywhere."""

from __future__ import annotations

import sys
from datetime import datetime

from ..state import EventKind, State, Tick


class ConsoleAlerter:
    name = "console"

    def __init__(self, stream=None, bell: bool = True):
        self.stream = stream or sys.stdout
        self.bell = bell
        self._last_state: State | None = None

    def start(self) -> None:
        self._last_state = None

    def apply(self, tick: Tick) -> None:
        for event in tick.events:
            if event.kind is EventKind.ALERT_STARTED:
                self._write("shoulders forward - sit back", bell=self.bell)
            elif event.kind is EventKind.ALERT_ENDED:
                seconds = event.detail.get("duration", 0.0)
                if event.detail.get("reason") == "corrected":
                    self._write(f"corrected after {seconds:.0f}s")
        if tick.state is not self._last_state:
            if tick.state in (State.ABSENT, State.PAUSED) or self._last_state in (
                State.ABSENT,
                State.PAUSED,
            ):
                self._write(f"[{tick.state.value}]")
            self._last_state = tick.state

    def stop(self) -> None:
        pass

    def _write(self, message: str, bell: bool = False) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.stream.write(f"\a{stamp}  {message}\n" if bell else f"{stamp}  {message}\n")
        self.stream.flush()
