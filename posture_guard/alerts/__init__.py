"""Alerters: the things that actually get your attention.

An alerter is driven from the main thread and told the current :class:`Tick`
several times a second. It decides what, if anything, that should look like.

Everything is fail-open. If an alerter raises -- a missing framework, a
permission that was revoked, a display that vanished -- it is disabled and the
monitoring carries on. Losing the dim is a nuisance; a stuck dark screen you
cannot clear is not acceptable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..state import Tick


@runtime_checkable
class Alerter(Protocol):
    name: str

    def start(self) -> None: ...

    def apply(self, tick: Tick) -> None: ...

    def stop(self) -> None: ...


class SafeAlerter:
    """Wraps an alerter so one broken backend cannot take the app down."""

    def __init__(self, inner: Alerter, on_error=None):
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.failed = False
        self._on_error = on_error

    def _guard(self, action: str, fn, *args) -> None:
        if self.failed:
            return
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            self.failed = True
            if self._on_error:
                self._on_error(f"alerter {self.name!r} disabled after failing to {action}: {exc}")
            try:
                self.inner.stop()
            except Exception:  # noqa: BLE001
                pass

    def start(self) -> None:
        self._guard("start", self.inner.start)

    def apply(self, tick: Tick) -> None:
        self._guard("update", self.inner.apply, tick)

    def stop(self) -> None:
        if not self.failed:
            try:
                self.inner.stop()
            except Exception:  # noqa: BLE001
                pass


def build_alerters(cfg, on_error=None) -> list[SafeAlerter]:
    """Instantiate the alerters named in the config, skipping unavailable ones."""
    from .console import ConsoleAlerter
    from .dim import DimOverlay
    from .notify import NotificationAlerter

    built: list[SafeAlerter] = []
    for name in cfg.alerters:
        key = name.strip().lower()
        if key in ("none", ""):
            continue
        if key == "dim":
            built.append(SafeAlerter(DimOverlay(max_alpha=cfg.max_dim), on_error))
        elif key in ("notify", "notification"):
            built.append(SafeAlerter(NotificationAlerter(cooldown_s=cfg.notify_cooldown_s), on_error))
        elif key == "console":
            built.append(SafeAlerter(ConsoleAlerter(), on_error))
        elif on_error:
            on_error(f"unknown alerter {name!r} in config, ignoring")
    return built
