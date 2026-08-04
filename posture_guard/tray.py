"""Menu bar front end.

The title is a single bar glyph that grows as your score climbs, so the current
state is readable at a glance without opening anything. The menu carries the
controls that must stay reachable while the screen is dimmed -- which they are,
because the overlay ignores mouse events.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from .app import Runner
from .config import Config
from .state import State

# Eight levels of "how far into your slouch are you".
GLYPHS = "▁▂▃▄▅▆▇█"
PAUSED_GLYPH = "‖"
ABSENT_GLYPH = "·"


def title_for(state: State, score: float | None, paused_for: float = 0.0) -> str:
    if paused_for > 0 or state is State.PAUSED:
        minutes = int(paused_for // 60) + 1 if paused_for > 0 else 0
        return f"{PAUSED_GLYPH}{minutes}m" if minutes else PAUSED_GLYPH
    if state is State.ABSENT or score is None:
        return ABSENT_GLYPH
    index = int(max(0.0, min(1.0, score)) * (len(GLYPHS) - 1))
    return GLYPHS[index]


def run_tray(runner: Runner, cfg: Config, report_path: Path) -> None:
    """Start the menu bar app. Blocks until the user quits."""
    import rumps  # noqa: PLC0415 - macOS only

    class PostureApp(rumps.App):
        def __init__(self) -> None:
            super().__init__("posture-guard", title=ABSENT_GLYPH, quit_button=None)
            self.status_item = rumps.MenuItem("Starting…")
            self.menu = [
                self.status_item,
                None,
                rumps.MenuItem("Pause 15 minutes", callback=lambda _: self._pause(15)),
                rumps.MenuItem("Pause 60 minutes", callback=lambda _: self._pause(60)),
                rumps.MenuItem("Resume now", callback=lambda _: runner.resume()),
                None,
                rumps.MenuItem("Open report", callback=lambda _: self._report()),
                rumps.MenuItem("Quit", callback=self._quit),
            ]
            # 20 Hz keeps the dim smooth; the alerters do nothing when idle.
            self._pump = rumps.Timer(lambda _: runner.pump(), 0.05)
            self._refresh = rumps.Timer(lambda _: self._update_title(), 1.0)

        def _pause(self, minutes: int) -> None:
            runner.pause(minutes * 60)

        def _report(self) -> None:
            from .report import write_html  # noqa: PLC0415
            from .storage import Store  # noqa: PLC0415

            with Store(runner.store.path) as store:
                write_html(store, cfg, report_path)
            webbrowser.open(report_path.as_uri())

        def _quit(self, _) -> None:
            runner.stop()
            rumps.quit_application()

        def _update_title(self) -> None:
            state, score = runner.state, runner.score
            paused = runner.paused_for
            self.title = title_for(state, score, paused)
            if paused > 0:
                self.status_item.title = f"Paused for {int(paused // 60) + 1} more minutes"
            elif state is State.ABSENT:
                self.status_item.title = "Nobody in view"
            elif score is None:
                self.status_item.title = "Waiting for a clear view"
            else:
                self.status_item.title = f"Score {score:.2f}  ({state.value})"

        def run(self, **kwargs):  # type: ignore[override]
            runner.start()
            self._pump.start()
            self._refresh.start()
            return super().run(**kwargs)

    try:
        PostureApp().run()
    finally:
        runner.stop()
