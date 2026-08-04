"""The escalating dim: a click-through black window whose opacity tracks the alert.

Why a window rather than the display's backlight: changing screen brightness is a
global, persistent setting that survives a crash and affects other users of the
machine. An overlay window belongs to this process and disappears with it.

Three safeguards, because a screen you cannot un-dim is worse than bad posture:

* opacity is capped by ``max_dim`` and the config refuses anything above 0.85, so
  the screen always stays readable;
* the window ignores mouse events, so nothing underneath becomes unclickable --
  the menu bar included;
* it is driven from a main-thread timer, so if the capture thread stalls or dies
  the runner feeds a zero intensity and the dim lifts on its own.

macOS only. Every AppKit import is deferred so the module can be imported, and
the rest of the suite tested, on any platform.
"""

from __future__ import annotations

import sys

from ..state import Tick

# NSWindowCollectionBehavior bits, spelled out so the module imports on Linux.
_CAN_JOIN_ALL_SPACES = 1 << 0
_STATIONARY = 1 << 4
_IGNORES_CYCLE = 1 << 6
_FULLSCREEN_AUXILIARY = 1 << 8
_SCREEN_SAVER_LEVEL = 1000

# Below this the overlay is invisible anyway; hide the window entirely so it
# stays out of screenshots and screen sharing while you are sitting properly.
_HIDE_BELOW = 0.004
_SCREEN_RECHECK_S = 2.0


class UnsupportedPlatform(RuntimeError):
    pass


class DimOverlay:
    name = "dim"

    def __init__(self, max_alpha: float = 0.55):
        self.max_alpha = max(0.0, min(0.85, max_alpha))
        self._windows: list = []
        self._visible = False
        self._alpha = 0.0
        self._last_screen_check = 0.0
        self._screen_count = 0

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if sys.platform != "darwin":
            raise UnsupportedPlatform(
                "the dim overlay needs macOS; use `alerters` = notify or console instead"
            )
        self._build_windows()

    def stop(self) -> None:
        for window in self._windows:
            try:
                window.setAlphaValue_(0.0)
                window.orderOut_(None)
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
        self._windows = []
        self._visible = False
        self._alpha = 0.0

    # -- driving -------------------------------------------------------------

    def apply(self, tick: Tick) -> None:
        self._maybe_rebuild()
        self.set_alpha(tick.intensity * self.max_alpha)

    def set_alpha(self, alpha: float) -> None:
        alpha = max(0.0, min(self.max_alpha, float(alpha)))
        if abs(alpha - self._alpha) < 1e-3 and (alpha > _HIDE_BELOW) == self._visible:
            return
        self._alpha = alpha

        should_show = alpha > _HIDE_BELOW
        for window in self._windows:
            window.setAlphaValue_(alpha)
            if should_show and not self._visible:
                window.orderFrontRegardless()
            elif not should_show and self._visible:
                window.orderOut_(None)
        self._visible = should_show

    # -- internals -----------------------------------------------------------

    def _build_windows(self) -> None:
        from AppKit import (  # noqa: PLC0415 - macOS only, deferred on purpose
            NSBackingStoreBuffered,
            NSColor,
            NSScreen,
            NSWindow,
        )

        self.stop()
        behaviour = _CAN_JOIN_ALL_SPACES | _STATIONARY | _IGNORES_CYCLE | _FULLSCREEN_AUXILIARY
        screens = list(NSScreen.screens())
        windows = []
        for screen in screens:
            window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_screen_(
                screen.frame(), 0, NSBackingStoreBuffered, False, screen
            )
            window.setOpaque_(False)
            window.setBackgroundColor_(NSColor.blackColor())
            window.setAlphaValue_(0.0)
            window.setHasShadow_(False)
            window.setIgnoresMouseEvents_(True)
            window.setLevel_(_SCREEN_SAVER_LEVEL)
            window.setCollectionBehavior_(behaviour)
            window.setReleasedWhenClosed_(False)
            windows.append(window)

        self._windows = windows
        self._screen_count = len(screens)
        self._visible = False
        self._alpha = 0.0

    def _maybe_rebuild(self) -> None:
        """Rebuild when displays are plugged in or unplugged."""
        import time  # noqa: PLC0415

        from AppKit import NSScreen  # noqa: PLC0415

        now = time.monotonic()
        if now - self._last_screen_check < _SCREEN_RECHECK_S:
            return
        self._last_screen_check = now
        if len(NSScreen.screens()) != self._screen_count:
            alpha = self._alpha
            self._build_windows()
            self.set_alpha(alpha)
