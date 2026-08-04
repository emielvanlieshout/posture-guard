"""A live one-line readout, for answering "is this actually working".

The menu bar glyph tells you the state at a glance, but on a first run you want
to watch the number move while you deliberately slouch, and confirm the thing is
reacting to you rather than to nothing. This prints that, in place, and gets out
of the way afterwards.

It is an alerter only in the mechanical sense: it is driven by the same tick and
never alerts. Combining it with the dim overlay is the point.
"""

from __future__ import annotations

import sys
import time

from ..state import State, Tick

BLOCKS = "░▁▂▃▄▅▆▇█"
WIDTH = 16


def bar(score: float | None, width: int = WIDTH) -> str:
    if score is None:
        return "·" * width
    filled = max(0.0, min(1.0, score)) * width
    full = int(filled)
    out = "█" * full
    if full < width:
        out += BLOCKS[int((filled - full) * (len(BLOCKS) - 1))]
    return out.ljust(width, "░")[:width]


class StatusLine:
    name = "status"

    def __init__(self, stream=None, interval: float = 0.4):
        self.stream = stream or sys.stdout
        self.interval = interval
        self._last = 0.0
        self._wrote = False

    def start(self) -> None:
        self._last = 0.0

    def apply(self, tick: Tick) -> None:
        now = time.monotonic()
        if now - self._last < self.interval:
            return
        self._last = now

        score = "  --" if tick.score is None else f"{tick.score:5.2f}"
        note = {
            State.CALM: "sitting well",
            State.WARN: "slipping, not alerting yet",
            State.ALERT: "slouching",
            State.ABSENT: "nobody in view",
            State.PAUSED: "paused",
        }[tick.state]
        line = (
            f"\r{bar(tick.score)}  score {score}  "
            f"dim {tick.intensity * 100:3.0f}%  {note:<28}"
        )
        self.stream.write(line)
        self.stream.flush()
        self._wrote = True

    def stop(self) -> None:
        if self._wrote:
            self.stream.write("\n")
            self.stream.flush()
