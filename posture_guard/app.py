"""Wiring: a capture thread producing scores, a main thread driving the alerters.

The split is forced by AppKit -- windows may only be touched from the main
thread -- but it buys a safety property worth having anyway. The main thread
watches how old the last published tick is, and once it goes stale it feeds the
alerters a zero. So if the camera hangs, mediapipe deadlocks or the worker dies
outright, the dim lifts by itself within a couple of seconds instead of leaving
a dark screen behind.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from datetime import datetime
from pathlib import Path

from .alerts import SafeAlerter, build_alerters
from .config import Config, pause_flag_path
from .features import QualityLimits, get_feature_set
from .ratchet import current_enter, evaluate
from .scoring import Profile, Scorer
from .state import AlertPolicy, EventKind, Monitor, State, Tick
from .storage import BucketAggregator, Store

STALE_AFTER_S = 3.0
RATCHET_EVERY_S = 3600.0


class Shared:
    """One tick, published by the worker and read by the main thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tick: Tick | None = None
        self._updated_at = 0.0
        self.error: str | None = None

    def publish(self, tick: Tick) -> None:
        with self._lock:
            self._tick = tick
            self._updated_at = time.monotonic()

    def read(self) -> tuple[Tick | None, float]:
        with self._lock:
            age = time.monotonic() - self._updated_at if self._tick else float("inf")
            return self._tick, age


class Worker(threading.Thread):
    """Camera -> features -> score -> state, at the configured frame rate."""

    daemon = True

    def __init__(self, cfg: Config, profile: Profile, store: Store, shared: Shared, model_path: Path):
        super().__init__(name="posture-capture")
        self.cfg = cfg
        self.profile = profile
        self.store = store
        self.shared = shared
        self.model_path = model_path
        self.stop_event = threading.Event()

        enter = current_enter(store, cfg)
        policy_kwargs = cfg.policy_kwargs() | {"enter": enter}
        if policy_kwargs["exit"] >= enter:
            # The ratchet can tighten `enter` past a hand-set `exit`; keep the
            # hysteresis gap rather than crashing on an invalid policy.
            policy_kwargs["exit"] = round(enter * 0.6, 4)
        self.policy = AlertPolicy(**policy_kwargs)

        self.feature_set = get_feature_set(profile.view)
        self.limits = QualityLimits()
        self.scorer = Scorer(profile, tau=cfg.ema_tau)
        self.monitor = Monitor(self.policy)
        self.aggregator = BucketAggregator(store, cfg.bucket_seconds, good_below=enter)
        self._last_ratchet = time.monotonic()

    def run(self) -> None:
        from .capture import PoseSource  # noqa: PLC0415 - keeps cv2 out of the import path

        source = PoseSource(
            self.model_path,
            camera_index=self.cfg.camera_index,
            width=self.cfg.camera_width,
            height=self.cfg.camera_height,
            fps=self.cfg.fps,
        )
        try:
            with source:
                self._loop(source)
        except Exception as exc:  # noqa: BLE001 - surfaced to the main thread
            self.shared.error = str(exc)
        finally:
            try:
                self.aggregator.flush()
            except Exception:  # noqa: BLE001
                pass

    def _loop(self, source) -> None:
        previous = None
        for frame in source.frames():
            if self.stop_event.is_set():
                return

            now = time.time()
            dt = 0.0 if previous is None else now - previous
            previous = now

            score = None
            if frame is not None:
                sample = self.feature_set.extract(frame, self.limits)
                if sample.ok:
                    score = self.scorer.update(now, sample.values)

            self._sync_pause(now)
            tick = self.monitor.update(now, score)

            if _is_quiet(self.cfg, now) and tick.intensity > 0:
                tick = dataclasses.replace(tick, intensity=0.0)

            for event in tick.events:
                self.store.log_event(event.ts, event.kind.value, event.detail)
            self.aggregator.add(now, score, dt, alerting=tick.state is State.ALERT)
            self.shared.publish(tick)

            if time.monotonic() - self._last_ratchet > RATCHET_EVERY_S:
                self._last_ratchet = time.monotonic()
                self.aggregator.flush()
                self._apply_ratchet()

    def _sync_pause(self, now: float) -> None:
        """Honour the pause flag file, so a terminal can always call off the dim."""
        until = _read_pause_flag()
        if until and until > now:
            if not self.monitor.paused:
                event = self.monitor.pause(now, until - now)
                self.store.log_event(event.ts, event.kind.value, event.detail)
        elif self.monitor.paused and not until:
            event = self.monitor.resume(now)
            self.store.log_event(event.ts, event.kind.value, event.detail)

    def _apply_ratchet(self) -> None:
        decision = evaluate(self.store, self.cfg)
        if not decision.changed:
            return
        self.policy = dataclasses.replace(
            self.policy,
            enter=decision.new_enter,
            exit=min(self.policy.exit, round(decision.new_enter * 0.6, 4)),
        )
        self.monitor.policy = self.policy
        self.aggregator.good_below = decision.new_enter

    def stop(self) -> None:
        self.stop_event.set()


class Runner:
    """Owns the worker and the alerters. ``pump`` must be called on the main thread."""

    def __init__(self, cfg: Config, profile: Profile, store: Store, model_path: Path, log=print):
        self.cfg = cfg
        self.profile = profile
        self.store = store
        self.log = log
        self.shared = Shared()
        self.worker = Worker(cfg, profile, store, self.shared, model_path)
        self.alerters: list[SafeAlerter] = build_alerters(cfg, on_error=log)
        self._started = False
        self._reported_error = False

    @property
    def score(self) -> float | None:
        tick, _ = self.shared.read()
        return tick.score if tick else None

    @property
    def state(self) -> State:
        tick, age = self.shared.read()
        if tick is None or age > STALE_AFTER_S:
            return State.ABSENT
        return tick.state

    def start(self) -> None:
        for alerter in self.alerters:
            alerter.start()
        self.worker.start()
        self._started = True

    def pump(self) -> None:
        """Push the latest state into the alerters. Safe to call at any rate."""
        tick, age = self.shared.read()
        if tick is None or age > STALE_AFTER_S:
            # Nothing fresh: clear everything rather than leave a stale dim.
            tick = Tick(ts=time.time(), state=State.ABSENT, intensity=0.0, score=None)
        for alerter in self.alerters:
            alerter.apply(tick)

        if self.shared.error and not self._reported_error:
            self._reported_error = True
            self.log(f"capture stopped: {self.shared.error}")

    def stop(self) -> None:
        if not self._started:
            return
        self.worker.stop()
        self.worker.join(timeout=3.0)
        for alerter in self.alerters:
            alerter.stop()
        self._started = False

    def pause(self, seconds: float) -> None:
        _write_pause_flag(time.time() + seconds)

    def resume(self) -> None:
        _clear_pause_flag()

    @property
    def paused_for(self) -> float:
        until = _read_pause_flag()
        return max(0.0, until - time.time()) if until else 0.0


def _is_quiet(cfg: Config, ts: float) -> bool:
    from .config import in_quiet_hours  # noqa: PLC0415 - avoids a cycle at import time

    local = datetime.fromtimestamp(ts)
    return in_quiet_hours(cfg, local.hour, local.minute)


def _read_pause_flag() -> float | None:
    path = pause_flag_path()
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _write_pause_flag(until: float) -> None:
    path = pause_flag_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(until))


def _clear_pause_flag() -> None:
    pause_flag_path().unlink(missing_ok=True)


def run_headless(runner: Runner, poll: float = 0.05) -> None:
    """Main loop without a menu bar. Ctrl-C stops it."""
    runner.start()
    try:
        while True:
            runner.pump()
            time.sleep(poll)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()


__all__ = ["Runner", "Shared", "Worker", "run_headless", "EventKind"]
